//! Core QdrantVectorStore implementation.
//!
//! This module contains the main QdrantVectorStore struct and its VectorStore trait
//! implementation, providing the primary interface for vector storage operations.

use async_trait::async_trait;
use cheungfun_core::{
    Result,
    traits::{StorageStats, VectorStore},
    types::{Node, Query, ScoredNode},
};
use qdrant_client::qdrant::{
    DeletePointsBuilder, GetPointsBuilder, PointId, SearchPointsBuilder, UpsertPointsBuilder,
    point_id::PointIdOptions,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, error, info};
use uuid::Uuid;

use super::{
    advanced::QdrantAdvanced,
    client::QdrantClient,
    config::QdrantConfig,
    conversion::{node_to_point, retrieved_point_to_node, scored_point_to_node},
    error::map_qdrant_error,
};

/// Qdrant vector store implementation.
///
/// This store provides a production-grade vector storage solution using Qdrant
/// as the backend. It supports all VectorStore operations with high performance,
/// scalability, and reliability.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_integrations::vector_stores::qdrant::{QdrantVectorStore, QdrantConfig};
/// use cheungfun_core::traits::{VectorStore, DistanceMetric};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = QdrantConfig::new("http://localhost:6334", "my_collection", 384)
///     .with_distance_metric(DistanceMetric::Cosine);
///
/// let store = QdrantVectorStore::new(config).await?;
/// # Ok(())
/// # }
/// ```
pub struct QdrantVectorStore {
    /// Qdrant client wrapper
    client: QdrantClient,
    /// Statistics tracking
    stats: Arc<RwLock<StorageStats>>,
}

impl std::fmt::Debug for QdrantVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantVectorStore")
            .field("client", &self.client)
            .field("stats", &self.stats)
            .finish()
    }
}

impl QdrantVectorStore {
    /// Create a new QdrantVectorStore.
    ///
    /// This will establish a connection to the Qdrant server and optionally
    /// create the collection if it doesn't exist.
    ///
    /// # Arguments
    ///
    /// * `config` - Qdrant configuration
    ///
    /// # Returns
    ///
    /// A Result containing the QdrantVectorStore or an error
    ///
    /// # Errors
    ///
    /// Returns an error if connection fails or collection creation fails.
    pub async fn new(config: QdrantConfig) -> Result<Self> {
        let client = QdrantClient::new(config).await?;
        let stats = Arc::new(RwLock::new(StorageStats::new()));

        let store = Self { client, stats };

        // Initialize collection if needed
        if store.client.config().create_collection_if_missing {
            store.client.ensure_collection_exists().await?;
        }

        info!("QdrantVectorStore created successfully");
        Ok(store)
    }

    /// Get the configuration.
    pub fn config(&self) -> &QdrantConfig {
        self.client.config()
    }

    /// Get the Qdrant client.
    pub fn client(&self) -> &QdrantClient {
        &self.client
    }

    /// Get access to advanced operations.
    ///
    /// This provides access to advanced Qdrant operations that go beyond
    /// the basic VectorStore interface.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use cheungfun_integrations::vector_stores::qdrant::{QdrantVectorStore, QdrantConfig};
    /// # async fn example(store: QdrantVectorStore) -> Result<(), Box<dyn std::error::Error>> {
    /// let advanced = store.advanced();
    /// let stats = advanced.collection_stats().await?;
    /// println!("Collection stats: {:?}", stats);
    /// # Ok(())
    /// # }
    /// ```
    pub fn advanced(&self) -> QdrantAdvanced {
        QdrantAdvanced::new(&self.client)
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn add(&self, nodes: Vec<Node>) -> Result<Vec<Uuid>> {
        debug!(
            "Adding {} nodes to Qdrant collection '{}'",
            nodes.len(),
            self.config().collection_name
        );

        let mut points = Vec::new();
        let mut node_ids = Vec::new();

        for node in nodes {
            let point = node_to_point(&node, self.config())?;
            node_ids.push(node.id);
            points.push(point);
        }

        let upsert_request = UpsertPointsBuilder::new(&self.config().collection_name, points);

        self.client
            .client()
            .upsert_points(upsert_request)
            .await
            .map_err(|e| {
                error!("Failed to add nodes to Qdrant: {}", e);
                map_qdrant_error(e)
            })?;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.insert_operations += 1;
            stats.total_nodes += node_ids.len();
        }

        info!("Successfully added {} nodes to Qdrant", node_ids.len());
        Ok(node_ids)
    }

    async fn update(&self, nodes: Vec<Node>) -> Result<()> {
        let node_count = nodes.len();
        debug!(
            "Updating {} nodes in Qdrant collection '{}'",
            node_count,
            self.config().collection_name
        );

        let mut points = Vec::new();

        for node in &nodes {
            let point = node_to_point(node, self.config())?;
            points.push(point);
        }

        let upsert_request = UpsertPointsBuilder::new(&self.config().collection_name, points);

        self.client
            .client()
            .upsert_points(upsert_request)
            .await
            .map_err(|e| {
                error!("Failed to update nodes in Qdrant: {}", e);
                map_qdrant_error(e)
            })?;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.update_operations += 1;
        }

        info!("Successfully updated {} nodes in Qdrant", node_count);
        Ok(())
    }

    async fn delete(&self, node_ids: Vec<Uuid>) -> Result<()> {
        debug!(
            "Deleting {} nodes from Qdrant collection '{}'",
            node_ids.len(),
            self.config().collection_name
        );

        let point_ids: Vec<PointIdOptions> = node_ids
            .iter()
            .map(|id| PointIdOptions::Uuid(id.to_string()))
            .collect();

        let delete_request =
            DeletePointsBuilder::new(&self.config().collection_name).points(point_ids);

        self.client
            .client()
            .delete_points(delete_request)
            .await
            .map_err(|e| {
                error!("Failed to delete nodes from Qdrant: {}", e);
                map_qdrant_error(e)
            })?;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.delete_operations += 1;
            stats.total_nodes = stats.total_nodes.saturating_sub(node_ids.len());
        }

        info!("Successfully deleted {} nodes from Qdrant", node_ids.len());
        Ok(())
    }

    async fn search(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        debug!(
            "Searching Qdrant collection '{}' with query: '{}', top_k: {}",
            self.config().collection_name,
            query.text,
            query.top_k
        );

        let query_embedding =
            query
                .embedding
                .as_ref()
                .ok_or_else(|| cheungfun_core::CheungfunError::Validation {
                    message: "Query must have an embedding for vector search".to_string(),
                })?;

        // Validate vector dimension
        if query_embedding.len() != self.config().dimension {
            return Err(cheungfun_core::CheungfunError::Validation {
                message: format!(
                    "Query vector dimension {} does not match collection dimension {}",
                    query_embedding.len(),
                    self.config().dimension
                ),
            });
        }

        let mut search_request = SearchPointsBuilder::new(
            &self.config().collection_name,
            query_embedding.clone(),
            query.top_k as u64,
        )
        .with_payload(true)
        .with_vectors(true);

        // Apply similarity threshold if specified
        if let Some(threshold) = query.similarity_threshold {
            search_request = search_request.score_threshold(threshold);
        }

        // TODO: Add metadata filters support
        // This would require converting Query filters to Qdrant Filter format

        let search_result = self
            .client
            .client()
            .search_points(search_request)
            .await
            .map_err(|e| {
                error!("Failed to search in Qdrant: {}", e);
                map_qdrant_error(e)
            })?;

        let mut scored_nodes = Vec::new();

        for scored_point in search_result.result {
            if let Some(node) = scored_point_to_node(scored_point.clone())? {
                scored_nodes.push(ScoredNode::new(node, scored_point.score));
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.search_operations += 1;
        }

        info!(
            "Search completed, returning {} results from Qdrant",
            scored_nodes.len()
        );

        Ok(scored_nodes)
    }

    async fn get(&self, node_ids: Vec<Uuid>) -> Result<Vec<Option<Node>>> {
        debug!(
            "Getting {} nodes from Qdrant collection '{}'",
            node_ids.len(),
            self.config().collection_name
        );

        let point_ids: Vec<PointId> = node_ids
            .iter()
            .map(|id| PointId {
                point_id_options: Some(PointIdOptions::Uuid(id.to_string())),
            })
            .collect();

        let get_request = GetPointsBuilder::new(&self.config().collection_name, point_ids)
            .with_payload(true)
            .with_vectors(true);

        let get_result = self
            .client
            .client()
            .get_points(get_request)
            .await
            .map_err(|e| {
                error!("Failed to get nodes from Qdrant: {}", e);
                map_qdrant_error(e)
            })?;

        let mut results = Vec::new();

        for (i, _) in node_ids.iter().enumerate() {
            let node = get_result
                .result
                .get(i)
                .and_then(|point| retrieved_point_to_node(point.clone()).transpose())
                .transpose()?;

            results.push(node);
        }

        Ok(results)
    }

    async fn health_check(&self) -> Result<()> {
        self.client.health_check().await
    }

    fn name(&self) -> &'static str {
        "QdrantVectorStore"
    }

    async fn count(&self) -> Result<usize> {
        let collection_info = self.client.collection_info().await?;
        Ok(collection_info.points_count.unwrap_or(0) as usize)
    }

    async fn metadata(&self) -> Result<HashMap<String, Value>> {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "qdrant".into());
        metadata.insert("url".to_string(), self.config().url.clone().into());
        metadata.insert(
            "collection_name".to_string(),
            self.config().collection_name.clone().into(),
        );
        metadata.insert("dimension".to_string(), self.config().dimension.into());
        metadata.insert(
            "distance_metric".to_string(),
            format!("{:?}", self.config().distance_metric).into(),
        );

        let node_count = self.count().await?;
        metadata.insert("node_count".to_string(), node_count.into());

        Ok(metadata)
    }

    async fn clear(&self) -> Result<()> {
        debug!(
            "Clearing all nodes from Qdrant collection '{}'",
            self.config().collection_name
        );

        // Delete the collection and recreate it
        self.client
            .client()
            .delete_collection(&self.config().collection_name)
            .await
            .map_err(map_qdrant_error)?;
        self.client.create_collection().await?;

        // Reset statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_nodes = 0;
        }

        info!("Successfully cleared all nodes from Qdrant collection");
        Ok(())
    }

    async fn stats(&self) -> Result<StorageStats> {
        let stats = self.stats.read().unwrap();
        Ok(stats.clone())
    }
}
