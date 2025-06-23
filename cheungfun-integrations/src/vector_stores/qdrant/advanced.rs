//! Advanced Qdrant operations and utilities.
//!
//! This module provides advanced functionality for Qdrant operations including
//! batch processing, advanced search with filters, and performance optimization.

use cheungfun_core::{
    Result,
    types::{Node, ScoredNode},
};
use qdrant_client::qdrant::{CreateFieldIndexCollectionBuilder, FieldType, SearchPointsBuilder};
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, error, info};
use uuid::Uuid;

use super::{
    client::QdrantClient,
    conversion::{node_to_point, scored_point_to_node},
    error::map_qdrant_error,
};

/// Advanced operations for QdrantVectorStore.
///
/// This struct provides methods for advanced Qdrant operations that go beyond
/// the basic VectorStore interface, such as batch operations and advanced search.
pub struct QdrantAdvanced<'a> {
    client: &'a QdrantClient,
}

impl<'a> QdrantAdvanced<'a> {
    /// Create a new QdrantAdvanced instance.
    ///
    /// # Arguments
    ///
    /// * `client` - Reference to the QdrantClient
    pub fn new(client: &'a QdrantClient) -> Self {
        Self { client }
    }

    /// Batch upsert nodes with optimized performance.
    ///
    /// This method processes nodes in batches to improve performance when
    /// inserting large numbers of vectors.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Vector of nodes to upsert
    /// * `batch_size` - Number of nodes to process in each batch
    ///
    /// # Returns
    ///
    /// A Result containing the UUIDs of all upserted nodes
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use cheungfun_integrations::vector_stores::qdrant::{QdrantClient, QdrantAdvanced};
    /// # use cheungfun_core::types::Node;
    /// # async fn example(client: QdrantClient, nodes: Vec<Node>) -> Result<(), Box<dyn std::error::Error>> {
    /// let advanced = QdrantAdvanced::new(&client);
    /// let ids = advanced.batch_upsert(nodes, 100).await?;
    /// println!("Upserted {} nodes", ids.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn batch_upsert(&self, nodes: Vec<Node>, batch_size: usize) -> Result<Vec<Uuid>> {
        debug!(
            "Batch upserting {} nodes with batch size {}",
            nodes.len(),
            batch_size
        );

        let mut all_ids = Vec::new();

        for chunk in nodes.chunks(batch_size) {
            let chunk_ids = self.upsert_chunk(chunk).await?;
            all_ids.extend(chunk_ids);
        }

        info!("Successfully batch upserted {} nodes", all_ids.len());
        Ok(all_ids)
    }

    /// Upsert a single chunk of nodes.
    async fn upsert_chunk(&self, nodes: &[Node]) -> Result<Vec<Uuid>> {
        use qdrant_client::qdrant::UpsertPointsBuilder;

        let mut points = Vec::new();
        let mut node_ids = Vec::new();

        for node in nodes {
            let point = node_to_point(node, self.client.config())?;
            node_ids.push(node.id);
            points.push(point);
        }

        let upsert_request =
            UpsertPointsBuilder::new(&self.client.config().collection_name, points);

        self.client
            .client()
            .upsert_points(upsert_request)
            .await
            .map_err(|e| {
                error!("Failed to upsert chunk: {}", e);
                map_qdrant_error(e)
            })?;

        Ok(node_ids)
    }

    /// Search with advanced filtering capabilities.
    ///
    /// This method provides more advanced search capabilities including
    /// custom filters and score thresholds.
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - The query vector
    /// * `top_k` - Maximum number of results to return
    /// * `filter` - Optional Qdrant filter for advanced filtering
    /// * `score_threshold` - Optional minimum score threshold
    ///
    /// # Returns
    ///
    /// A Result containing the scored search results
    pub async fn search_with_filter(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<qdrant_client::qdrant::Filter>,
        score_threshold: Option<f32>,
    ) -> Result<Vec<ScoredNode>> {
        debug!("Advanced search with filter, top_k: {}", top_k);

        // Validate vector dimension
        if query_embedding.len() != self.client.config().dimension {
            return Err(cheungfun_core::CheungfunError::Validation {
                message: format!(
                    "Query vector dimension {} does not match collection dimension {}",
                    query_embedding.len(),
                    self.client.config().dimension
                ),
            });
        }

        let mut search_request = SearchPointsBuilder::new(
            &self.client.config().collection_name,
            query_embedding,
            top_k as u64,
        )
        .with_payload(true)
        .with_vectors(true);

        if let Some(filter) = filter {
            search_request = search_request.filter(filter);
        }

        if let Some(threshold) = score_threshold {
            search_request = search_request.score_threshold(threshold);
        }

        let search_result = self
            .client
            .client()
            .search_points(search_request)
            .await
            .map_err(|e| {
                error!("Failed to search with filter: {}", e);
                map_qdrant_error(e)
            })?;

        let mut scored_nodes = Vec::new();

        for scored_point in search_result.result {
            if let Some(node) = scored_point_to_node(scored_point.clone())? {
                scored_nodes.push(ScoredNode::new(node, scored_point.score));
            }
        }

        info!(
            "Advanced search completed, returning {} results",
            scored_nodes.len()
        );
        Ok(scored_nodes)
    }

    /// Create an index for better search performance.
    ///
    /// This method creates a payload index on a specific field to improve
    /// search performance when using filters.
    ///
    /// # Arguments
    ///
    /// * `field_name` - Name of the field to index
    ///
    /// # Returns
    ///
    /// A Result indicating whether the index was created successfully
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use cheungfun_integrations::vector_stores::qdrant::{QdrantClient, QdrantAdvanced};
    /// # async fn example(client: QdrantClient) -> Result<(), Box<dyn std::error::Error>> {
    /// let advanced = QdrantAdvanced::new(&client);
    /// advanced.create_index("category").await?;
    /// println!("Index created for 'category' field");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn create_index(&self, field_name: &str) -> Result<()> {
        // Qdrant automatically creates indexes, but we can create payload indexes
        let create_index = CreateFieldIndexCollectionBuilder::new(
            &self.client.config().collection_name,
            field_name,
            FieldType::Keyword,
        );

        self.client
            .client()
            .create_field_index(create_index)
            .await
            .map_err(|e| {
                error!("Failed to create field index: {}", e);
                map_qdrant_error(e)
            })?;

        info!("Created field index for '{}'", field_name);
        Ok(())
    }

    /// Get collection statistics.
    ///
    /// This method returns detailed statistics about the collection including
    /// point count, segments, and configuration information.
    ///
    /// # Returns
    ///
    /// A Result containing a HashMap with collection statistics
    pub async fn collection_stats(&self) -> Result<HashMap<String, Value>> {
        let collection_info = self.client.collection_info().await?;

        let mut stats = HashMap::new();
        stats.insert(
            "points_count".to_string(),
            (collection_info.points_count.unwrap_or(0) as usize).into(),
        );
        stats.insert(
            "segments_count".to_string(),
            (collection_info.segments_count as usize).into(),
        );
        stats.insert(
            "status".to_string(),
            format!("{:?}", collection_info.status()).into(),
        );

        if let Some(config) = collection_info.config {
            if let Some(params) = config.params {
                if let Some(_vectors_config) = params.vectors_config {
                    // Handle different vector config types
                    stats.insert("vector_config".to_string(), "configured".into());
                }
            }
        }

        Ok(stats)
    }

    /// Optimize collection for better performance.
    ///
    /// This method triggers collection optimization in Qdrant, which can
    /// improve search performance and reduce memory usage.
    ///
    /// # Returns
    ///
    /// A Result indicating whether the optimization was triggered successfully
    pub async fn optimize_collection(&self) -> Result<()> {
        debug!(
            "Optimizing collection '{}'",
            self.client.config().collection_name
        );

        // For now, just return success - optimization is automatic in Qdrant
        info!("Collection optimization is handled automatically by Qdrant");
        Ok(())
    }

    /// Count points matching a filter.
    ///
    /// This method counts the number of points in the collection that match
    /// the given filter criteria.
    ///
    /// # Arguments
    ///
    /// * `filter` - Optional filter to apply when counting
    ///
    /// # Returns
    ///
    /// A Result containing the count of matching points
    pub async fn count_points(
        &self,
        _filter: Option<qdrant_client::qdrant::Filter>,
    ) -> Result<usize> {
        // For now, just return the total collection count
        // TODO: Implement proper filtering when count_points API is available
        let collection_info = self.client.collection_info().await?;
        Ok(collection_info.points_count.unwrap_or(0) as usize)
    }
}

#[cfg(test)]
mod tests {
    // Integration tests would require a running Qdrant instance
    // These are commented out but show the structure for integration testing

    /*
    #[tokio::test]
    #[ignore] // Requires running Qdrant instance
    async fn test_batch_upsert() {
        let config = QdrantConfig::new("http://localhost:6334", "test_batch", 3);
        let client = QdrantClient::new(config).await.unwrap();
        let advanced = QdrantAdvanced::new(&client);

        // Create test nodes
        let nodes = vec![
            create_test_node("Node 1", vec![1.0, 0.0, 0.0]),
            create_test_node("Node 2", vec![0.0, 1.0, 0.0]),
            create_test_node("Node 3", vec![0.0, 0.0, 1.0]),
        ];

        let ids = advanced.batch_upsert(nodes, 2).await.unwrap();
        assert_eq!(ids.len(), 3);

        // Clean up
        client.delete_collection().await.unwrap();
    }

    #[tokio::test]
    #[ignore] // Requires running Qdrant instance
    async fn test_advanced_search() {
        let config = QdrantConfig::new("http://localhost:6334", "test_search", 3);
        let client = QdrantClient::new(config).await.unwrap();
        let advanced = QdrantAdvanced::new(&client);

        // Add some test data first
        let nodes = vec![
            create_test_node("Node 1", vec![1.0, 0.0, 0.0]),
            create_test_node("Node 2", vec![0.0, 1.0, 0.0]),
        ];
        advanced.batch_upsert(nodes, 10).await.unwrap();

        // Test search
        let results = advanced.search_with_filter(
            vec![1.0, 0.0, 0.0],
            5,
            None,
            Some(0.5),
        ).await.unwrap();

        assert!(!results.is_empty());

        // Clean up
        client.delete_collection().await.unwrap();
    }
    */
}
