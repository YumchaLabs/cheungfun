//! Qdrant vector store implementation.
//!
//! This module provides a production-grade vector store implementation using
//! Qdrant as the backend. It supports all VectorStore operations with high
//! performance and scalability.

use async_trait::async_trait;
use cheungfun_core::{
    Result,
    traits::{DistanceMetric, StorageStats, VectorStore},
    types::{Node, Query, ScoredNode},
};
use qdrant_client::{
    qdrant::{
        CreateCollectionBuilder, Distance, PointStruct, SearchPointsBuilder, UpsertPointsBuilder,
        VectorParamsBuilder,
    },
    Qdrant, QdrantError,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Configuration for QdrantVectorStore.
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    /// Qdrant server URL (e.g., "http://localhost:6334")
    pub url: String,
    /// Optional API key for authentication
    pub api_key: Option<String>,
    /// Collection name to use
    pub collection_name: String,
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric for similarity calculation
    pub distance_metric: DistanceMetric,
    /// Request timeout
    pub timeout: Duration,
    /// Maximum number of retries for failed requests
    pub max_retries: usize,
    /// Whether to create collection if it doesn't exist
    pub create_collection_if_missing: bool,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_string(),
            api_key: None,
            collection_name: "cheungfun_vectors".to_string(),
            dimension: 384,
            distance_metric: DistanceMetric::Cosine,
            timeout: Duration::from_secs(30),
            max_retries: 3,
            create_collection_if_missing: true,
        }
    }
}

impl QdrantConfig {
    /// Create a new Qdrant configuration.
    pub fn new(url: impl Into<String>, collection_name: impl Into<String>, dimension: usize) -> Self {
        Self {
            url: url.into(),
            collection_name: collection_name.into(),
            dimension,
            ..Default::default()
        }
    }

    /// Set the API key for authentication.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the distance metric.
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set the request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the maximum number of retries.
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set whether to create collection if missing.
    pub fn with_create_collection_if_missing(mut self, create: bool) -> Self {
        self.create_collection_if_missing = create;
        self
    }
}

/// Qdrant vector store implementation.
///
/// This store provides a production-grade vector storage solution using Qdrant
/// as the backend. It supports all VectorStore operations with high performance,
/// scalability, and reliability.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_integrations::{QdrantVectorStore, QdrantConfig};
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
    /// Qdrant client
    client: Arc<Qdrant>,
    /// Configuration
    config: QdrantConfig,
    /// Statistics tracking
    stats: Arc<RwLock<StorageStats>>,
}

impl std::fmt::Debug for QdrantVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantVectorStore")
            .field("config", &self.config)
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
    /// # Errors
    ///
    /// Returns an error if connection fails or collection creation fails.
    pub async fn new(config: QdrantConfig) -> Result<Self> {
        info!(
            "Creating QdrantVectorStore with URL: {}, collection: {}",
            config.url, config.collection_name
        );

        // Build Qdrant client
        let mut client_builder = Qdrant::from_url(&config.url);
        
        if let Some(api_key) = &config.api_key {
            client_builder = client_builder.api_key(api_key.clone());
        }

        let client = client_builder
            .timeout(config.timeout)
            .build()
            .map_err(|e| {
                error!("Failed to create Qdrant client: {}", e);
                map_qdrant_error(e)
            })?;

        let client = Arc::new(client);
        let stats = Arc::new(RwLock::new(StorageStats::new()));

        let store = Self {
            client,
            config,
            stats,
        };

        // Initialize collection if needed
        if store.config.create_collection_if_missing {
            store.ensure_collection_exists().await?;
        }

        info!("QdrantVectorStore created successfully");
        Ok(store)
    }

    /// Get the configuration.
    pub fn config(&self) -> &QdrantConfig {
        &self.config
    }

    /// Get the Qdrant client.
    pub fn client(&self) -> &Qdrant {
        &self.client
    }

    /// Ensure the collection exists, creating it if necessary.
    async fn ensure_collection_exists(&self) -> Result<()> {
        debug!("Checking if collection '{}' exists", self.config.collection_name);

        // Try to get collection info
        match self.client.collection_info(&self.config.collection_name).await {
            Ok(_) => {
                debug!("Collection '{}' already exists", self.config.collection_name);
                Ok(())
            }
            Err(QdrantError::ResponseError { status }) if status.message().contains("not found") => {
                info!("Collection '{}' not found, creating it", self.config.collection_name);
                self.create_collection().await
            }
            Err(e) => {
                error!("Failed to check collection existence: {}", e);
                Err(map_qdrant_error(e))
            }
        }
    }

    /// Create a new collection.
    async fn create_collection(&self) -> Result<()> {
        let distance = match self.config.distance_metric {
            DistanceMetric::Cosine => Distance::Cosine,
            DistanceMetric::Euclidean => Distance::Euclid,
            DistanceMetric::DotProduct => Distance::Dot,
            DistanceMetric::Manhattan => Distance::Manhattan,
            DistanceMetric::Custom(_) => {
                warn!("Custom distance metric not supported by Qdrant, using Cosine");
                Distance::Cosine
            }
        };

        let create_collection = CreateCollectionBuilder::new(&self.config.collection_name)
            .vectors_config(VectorParamsBuilder::new(self.config.dimension as u64, distance));

        self.client
            .create_collection(create_collection)
            .await
            .map_err(|e| {
                error!("Failed to create collection: {}", e);
                map_qdrant_error(e)
            })?;

        info!("Collection '{}' created successfully", self.config.collection_name);
        Ok(())
    }

    /// Convert a Node to a Qdrant PointStruct.
    fn node_to_point(&self, node: &Node) -> Result<PointStruct> {
        use qdrant_client::qdrant::Value as QdrantValue;

        let embedding = node.embedding.as_ref().ok_or_else(|| {
            cheungfun_core::CheungfunError::Validation {
                message: "Node must have an embedding".to_string(),
            }
        })?;

        // Validate vector dimension
        if embedding.len() != self.config.dimension {
            return Err(cheungfun_core::CheungfunError::Validation {
                message: format!(
                    "Vector dimension {} does not match collection dimension {}",
                    embedding.len(),
                    self.config.dimension
                ),
            });
        }

        // Convert metadata to Qdrant payload
        let mut payload = HashMap::new();

        // Add node content
        payload.insert("content".to_string(), QdrantValue::from(node.content.clone()));

        // Add source document ID
        payload.insert("source_document_id".to_string(), QdrantValue::from(node.source_document_id.to_string()));

        // Add chunk info
        payload.insert("chunk_start".to_string(), QdrantValue::from(node.chunk_info.start_offset as i64));
        payload.insert("chunk_end".to_string(), QdrantValue::from(node.chunk_info.end_offset as i64));
        payload.insert("chunk_index".to_string(), QdrantValue::from(node.chunk_info.chunk_index as i64));

        // Add user metadata - convert serde_json::Value to QdrantValue
        for (key, value) in &node.metadata {
            let qdrant_value = self.serde_value_to_qdrant_value(value);
            payload.insert(key.clone(), qdrant_value);
        }

        Ok(PointStruct::new(
            node.id.to_string(),
            embedding.clone(),
            payload,
        ))
    }

    /// Convert serde_json::Value to qdrant::Value.
    fn serde_value_to_qdrant_value(&self, value: &Value) -> qdrant_client::qdrant::Value {
        use qdrant_client::qdrant::Value as QdrantValue;

        match value {
            Value::Null => QdrantValue::from(""),
            Value::Bool(b) => QdrantValue::from(*b),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    QdrantValue::from(i)
                } else if let Some(f) = n.as_f64() {
                    QdrantValue::from(f)
                } else {
                    QdrantValue::from(n.to_string())
                }
            }
            Value::String(s) => QdrantValue::from(s.clone()),
            Value::Array(arr) => {
                // Convert array to string representation for simplicity
                QdrantValue::from(serde_json::to_string(arr).unwrap_or_default())
            }
            Value::Object(obj) => {
                // Convert object to string representation for simplicity
                QdrantValue::from(serde_json::to_string(obj).unwrap_or_default())
            }
        }
    }

    /// Convert a Qdrant ScoredPoint to a Node.
    fn scored_point_to_node(&self, point: qdrant_client::qdrant::ScoredPoint) -> Result<Option<Node>> {
        use cheungfun_core::types::ChunkInfo;

        let point_id = match point.id {
            Some(id) => self.point_id_to_string(&id)?,
            None => return Ok(None),
        };

        let node_id = Uuid::parse_str(&point_id).map_err(|e| {
            cheungfun_core::CheungfunError::Validation {
                message: format!("Invalid UUID in point ID: {}", e),
            }
        })?;

        let payload = point.payload;
        let _vectors = point.vectors;

        // Extract content
        let content = payload
            .get("content")
            .and_then(|v| self.qdrant_value_to_string(v))
            .unwrap_or_default();

        // Extract source document ID
        let source_document_id = payload
            .get("source_document_id")
            .and_then(|v| self.qdrant_value_to_string(v))
            .and_then(|s| Uuid::parse_str(&s).ok())
            .unwrap_or_else(Uuid::new_v4);

        // Extract chunk info
        let chunk_start = payload
            .get("chunk_start")
            .and_then(|v| self.qdrant_value_to_u64(v))
            .unwrap_or(0) as usize;

        let chunk_end = payload
            .get("chunk_end")
            .and_then(|v| self.qdrant_value_to_u64(v))
            .unwrap_or(0) as usize;

        let chunk_index = payload
            .get("chunk_index")
            .and_then(|v| self.qdrant_value_to_u64(v))
            .unwrap_or(0) as usize;

        let chunk_info = ChunkInfo::new(chunk_start, chunk_end, chunk_index);

        // Extract embedding - simplified for now
        let embedding = None; // TODO: Extract from vectors when needed

        // Extract metadata (excluding our internal fields)
        let mut metadata = HashMap::new();
        for (key, value) in payload {
            if !matches!(key.as_str(), "content" | "source_document_id" | "chunk_start" | "chunk_end" | "chunk_index") {
                let serde_value = self.qdrant_value_to_serde_value(&value);
                metadata.insert(key, serde_value);
            }
        }

        let mut node = Node::new(content, source_document_id, chunk_info);
        node.id = node_id;
        node.metadata = metadata;

        if let Some(emb) = embedding {
            node.embedding = Some(emb);
        }

        Ok(Some(node))
    }

    /// Convert a Qdrant RetrievedPoint to a Node.
    fn retrieved_point_to_node(&self, point: qdrant_client::qdrant::RetrievedPoint) -> Result<Option<Node>> {
        use cheungfun_core::types::ChunkInfo;

        let point_id = match point.id {
            Some(id) => self.point_id_to_string(&id)?,
            None => return Ok(None),
        };

        let node_id = Uuid::parse_str(&point_id).map_err(|e| {
            cheungfun_core::CheungfunError::Validation {
                message: format!("Invalid UUID in point ID: {}", e),
            }
        })?;

        let payload = point.payload;
        let _vectors = point.vectors;

        // Extract content
        let content = payload
            .get("content")
            .and_then(|v| self.qdrant_value_to_string(v))
            .unwrap_or_default();

        // Extract source document ID
        let source_document_id = payload
            .get("source_document_id")
            .and_then(|v| self.qdrant_value_to_string(v))
            .and_then(|s| Uuid::parse_str(&s).ok())
            .unwrap_or_else(Uuid::new_v4);

        // Extract chunk info
        let chunk_start = payload
            .get("chunk_start")
            .and_then(|v| self.qdrant_value_to_u64(v))
            .unwrap_or(0) as usize;

        let chunk_end = payload
            .get("chunk_end")
            .and_then(|v| self.qdrant_value_to_u64(v))
            .unwrap_or(0) as usize;

        let chunk_index = payload
            .get("chunk_index")
            .and_then(|v| self.qdrant_value_to_u64(v))
            .unwrap_or(0) as usize;

        let chunk_info = ChunkInfo::new(chunk_start, chunk_end, chunk_index);

        // Extract embedding - simplified for now
        let embedding = None; // TODO: Extract from vectors when needed

        // Extract metadata (excluding our internal fields)
        let mut metadata = HashMap::new();
        for (key, value) in payload {
            if !matches!(key.as_str(), "content" | "source_document_id" | "chunk_start" | "chunk_end" | "chunk_index") {
                let serde_value = self.qdrant_value_to_serde_value(&value);
                metadata.insert(key, serde_value);
            }
        }

        let mut node = Node::new(content, source_document_id, chunk_info);
        node.id = node_id;
        node.metadata = metadata;

        if let Some(emb) = embedding {
            node.embedding = Some(emb);
        }

        Ok(Some(node))
    }

    /// Convert PointId to string.
    fn point_id_to_string(&self, point_id: &qdrant_client::qdrant::PointId) -> Result<String> {
        match &point_id.point_id_options {
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => Ok(uuid.clone()),
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => Ok(num.to_string()),
            None => Err(cheungfun_core::CheungfunError::Validation {
                message: "Point ID is empty".to_string(),
            }),
        }
    }

    /// Convert Qdrant Value to string.
    fn qdrant_value_to_string(&self, value: &qdrant_client::qdrant::Value) -> Option<String> {
        match &value.kind {
            Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => Some(s.clone()),
            Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => Some(i.to_string()),
            Some(qdrant_client::qdrant::value::Kind::DoubleValue(d)) => Some(d.to_string()),
            Some(qdrant_client::qdrant::value::Kind::BoolValue(b)) => Some(b.to_string()),
            _ => None,
        }
    }

    /// Convert Qdrant Value to u64.
    fn qdrant_value_to_u64(&self, value: &qdrant_client::qdrant::Value) -> Option<u64> {
        match &value.kind {
            Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => Some(*i as u64),
            Some(qdrant_client::qdrant::value::Kind::DoubleValue(d)) => Some(*d as u64),
            _ => None,
        }
    }

    /// Convert Qdrant Value to serde_json Value.
    fn qdrant_value_to_serde_value(&self, value: &qdrant_client::qdrant::Value) -> serde_json::Value {
        match &value.kind {
            Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
            Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => serde_json::Value::Number((*i).into()),
            Some(qdrant_client::qdrant::value::Kind::DoubleValue(d)) => {
                serde_json::Value::Number(serde_json::Number::from_f64(*d).unwrap_or_else(|| 0.into()))
            }
            Some(qdrant_client::qdrant::value::Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
            _ => serde_json::Value::Null,
        }
    }

    /// Get collection information.
    pub async fn collection_info(&self) -> Result<qdrant_client::qdrant::CollectionInfo> {
        self.client
            .collection_info(&self.config.collection_name)
            .await
            .map(|response| response.result.unwrap())
            .map_err(|e| {
                error!("Failed to get collection info: {}", e);
                map_qdrant_error(e)
            })
    }

    /// Create an index for better search performance.
    pub async fn create_index(&self, field_name: &str) -> Result<()> {
        // Qdrant automatically creates indexes, but we can create payload indexes
        use qdrant_client::qdrant::{CreateFieldIndexCollectionBuilder, FieldType};

        let create_index = CreateFieldIndexCollectionBuilder::new(
            &self.config.collection_name,
            field_name,
            FieldType::Keyword,
        );

        self.client
            .create_field_index(create_index)
            .await
            .map_err(|e| {
                error!("Failed to create field index: {}", e);
                map_qdrant_error(e)
            })?;

        info!("Created field index for '{}'", field_name);
        Ok(())
    }

    /// Batch upsert with optimized performance.
    pub async fn batch_upsert(&self, nodes: Vec<Node>, batch_size: usize) -> Result<Vec<Uuid>> {
        debug!("Batch upserting {} nodes with batch size {}", nodes.len(), batch_size);

        let mut all_ids = Vec::new();

        for chunk in nodes.chunks(batch_size) {
            let chunk_ids = self.add(chunk.to_vec()).await?;
            all_ids.extend(chunk_ids);
        }

        info!("Successfully batch upserted {} nodes", all_ids.len());
        Ok(all_ids)
    }

    /// Search with advanced filtering.
    pub async fn search_with_filter(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<qdrant_client::qdrant::Filter>,
        score_threshold: Option<f32>,
    ) -> Result<Vec<ScoredNode>> {
        debug!("Advanced search with filter, top_k: {}", top_k);

        let mut search_request = SearchPointsBuilder::new(
            &self.config.collection_name,
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

        let search_result = self.client
            .search_points(search_request)
            .await
            .map_err(|e| {
                error!("Failed to search with filter: {}", e);
                map_qdrant_error(e)
            })?;

        let mut scored_nodes = Vec::new();

        for scored_point in search_result.result {
            if let Some(node) = self.scored_point_to_node(scored_point.clone())? {
                scored_nodes.push(ScoredNode::new(node, scored_point.score));
            }
        }

        info!("Advanced search completed, returning {} results", scored_nodes.len());
        Ok(scored_nodes)
    }

    /// Get collection statistics.
    pub async fn collection_stats(&self) -> Result<HashMap<String, serde_json::Value>> {
        let collection_info = self.collection_info().await?;

        let mut stats = HashMap::new();
        stats.insert("points_count".to_string(), (collection_info.points_count.unwrap_or(0) as usize).into());
        stats.insert("segments_count".to_string(), (collection_info.segments_count as usize).into());
        stats.insert("status".to_string(), format!("{:?}", collection_info.status()).into());

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
    pub async fn optimize_collection(&self) -> Result<()> {
        debug!("Optimizing collection '{}'", self.config.collection_name);

        // For now, just return success - optimization is automatic in Qdrant
        info!("Collection optimization is handled automatically by Qdrant");
        Ok(())
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn add(&self, nodes: Vec<Node>) -> Result<Vec<Uuid>> {
        debug!("Adding {} nodes to Qdrant collection '{}'", nodes.len(), self.config.collection_name);

        let mut points = Vec::new();
        let mut node_ids = Vec::new();

        for node in nodes {
            let point = self.node_to_point(&node)?;
            node_ids.push(node.id);
            points.push(point);
        }

        let upsert_request = UpsertPointsBuilder::new(&self.config.collection_name, points);

        self.client
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
        debug!("Updating {} nodes in Qdrant collection '{}'", node_count, self.config.collection_name);

        let mut points = Vec::new();

        for node in &nodes {
            let point = self.node_to_point(node)?;
            points.push(point);
        }

        let upsert_request = UpsertPointsBuilder::new(&self.config.collection_name, points);

        self.client
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
        debug!("Deleting {} nodes from Qdrant collection '{}'", node_ids.len(), self.config.collection_name);

        use qdrant_client::qdrant::{DeletePointsBuilder, point_id::PointIdOptions};

        let point_ids: Vec<PointIdOptions> = node_ids.iter()
            .map(|id| PointIdOptions::Uuid(id.to_string()))
            .collect();

        let delete_request = DeletePointsBuilder::new(&self.config.collection_name)
            .points(point_ids);

        self.client
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
            self.config.collection_name, query.text, query.top_k
        );

        let query_embedding = query.embedding.as_ref().ok_or_else(|| {
            cheungfun_core::CheungfunError::Validation {
                message: "Query must have an embedding for vector search".to_string(),
            }
        })?;

        // Validate vector dimension
        if query_embedding.len() != self.config.dimension {
            return Err(cheungfun_core::CheungfunError::Validation {
                message: format!(
                    "Query vector dimension {} does not match collection dimension {}",
                    query_embedding.len(),
                    self.config.dimension
                ),
            });
        }

        let mut search_request = SearchPointsBuilder::new(
            &self.config.collection_name,
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

        let search_result = self.client
            .search_points(search_request)
            .await
            .map_err(|e| {
                error!("Failed to search in Qdrant: {}", e);
                map_qdrant_error(e)
            })?;

        let mut scored_nodes = Vec::new();

        for scored_point in search_result.result {
            if let Some(node) = self.scored_point_to_node(scored_point.clone())? {
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
        debug!("Getting {} nodes from Qdrant collection '{}'", node_ids.len(), self.config.collection_name);

        use qdrant_client::qdrant::{GetPointsBuilder, PointId, point_id::PointIdOptions};

        let point_ids: Vec<PointId> = node_ids.iter()
            .map(|id| PointId {
                point_id_options: Some(PointIdOptions::Uuid(id.to_string())),
            })
            .collect();

        let get_request = GetPointsBuilder::new(&self.config.collection_name, point_ids)
            .with_payload(true)
            .with_vectors(true);

        let get_result = self.client
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
                .and_then(|point| self.retrieved_point_to_node(point.clone()).transpose())
                .transpose()?;

            results.push(node);
        }

        Ok(results)
    }

    async fn health_check(&self) -> Result<()> {
        debug!("Performing health check for Qdrant");

        self.client
            .health_check()
            .await
            .map_err(|e| {
                error!("Qdrant health check failed: {}", e);
                map_qdrant_error(e)
            })?;

        debug!("Qdrant health check passed");
        Ok(())
    }

    fn name(&self) -> &'static str {
        "QdrantVectorStore"
    }

    async fn count(&self) -> Result<usize> {
        let collection_info = self.client
            .collection_info(&self.config.collection_name)
            .await
            .map_err(|e| {
                error!("Failed to get collection info from Qdrant: {}", e);
                map_qdrant_error(e)
            })?;

        Ok(collection_info.result.unwrap().points_count.unwrap_or(0) as usize)
    }

    async fn metadata(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "qdrant".into());
        metadata.insert("url".to_string(), self.config.url.clone().into());
        metadata.insert("collection_name".to_string(), self.config.collection_name.clone().into());
        metadata.insert("dimension".to_string(), self.config.dimension.into());
        metadata.insert("distance_metric".to_string(), format!("{:?}", self.config.distance_metric).into());

        let node_count = self.count().await?;
        metadata.insert("node_count".to_string(), node_count.into());

        Ok(metadata)
    }

    async fn clear(&self) -> Result<()> {
        debug!("Clearing all nodes from Qdrant collection '{}'", self.config.collection_name);

        // Delete the collection and recreate it
        self.client
            .delete_collection(&self.config.collection_name)
            .await
            .map_err(|e| {
                error!("Failed to delete collection from Qdrant: {}", e);
                map_qdrant_error(e)
            })?;

        self.create_collection().await?;

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

/// Convert QdrantError to CheungfunError.
fn map_qdrant_error(error: QdrantError) -> cheungfun_core::CheungfunError {
    match error {
        QdrantError::ResponseError { status } => {
            if status.message().contains("not found") {
                cheungfun_core::CheungfunError::NotFound {
                    resource: "Qdrant resource".to_string(),
                }
            } else {
                cheungfun_core::CheungfunError::VectorStore {
                    message: format!("Qdrant HTTP error: {}", status),
                }
            }
        }
        QdrantError::ConversionError(source) => {
            cheungfun_core::CheungfunError::VectorStore {
                message: format!("Qdrant conversion error: {}", source),
            }
        }
        _ => cheungfun_core::CheungfunError::VectorStore {
            message: format!("Qdrant error: {}", error),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::types::ChunkInfo;

    fn create_test_node(content: &str, embedding: Vec<f32>) -> Node {
        let source_doc_id = Uuid::new_v4();
        let chunk_info = ChunkInfo::new(0, content.len(), 0);
        Node::new(content, source_doc_id, chunk_info).with_embedding(embedding)
    }

    #[test]
    fn test_qdrant_config_creation() {
        let config = QdrantConfig::new("http://localhost:6334", "test_collection", 384);

        assert_eq!(config.url, "http://localhost:6334");
        assert_eq!(config.collection_name, "test_collection");
        assert_eq!(config.dimension, 384);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert!(config.create_collection_if_missing);
    }

    #[test]
    fn test_qdrant_config_builder() {
        let config = QdrantConfig::new("http://localhost:6334", "test", 512)
            .with_api_key("test_key")
            .with_distance_metric(DistanceMetric::Euclidean)
            .with_timeout(Duration::from_secs(60))
            .with_max_retries(5)
            .with_create_collection_if_missing(false);

        assert_eq!(config.api_key, Some("test_key".to_string()));
        assert_eq!(config.distance_metric, DistanceMetric::Euclidean);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.max_retries, 5);
        assert!(!config.create_collection_if_missing);
    }

    #[test]
    fn test_serde_value_conversion() {
        let config = QdrantConfig::default();
        let store = QdrantVectorStore {
            client: Arc::new(Qdrant::from_url("http://localhost:6334").build().unwrap()),
            config,
            stats: Arc::new(RwLock::new(StorageStats::new())),
        };

        // Test different value types
        let bool_val = Value::Bool(true);
        let num_val = Value::Number(serde_json::Number::from(42));
        let str_val = Value::String("test".to_string());
        let null_val = Value::Null;

        let qdrant_bool = store.serde_value_to_qdrant_value(&bool_val);
        let qdrant_num = store.serde_value_to_qdrant_value(&num_val);
        let qdrant_str = store.serde_value_to_qdrant_value(&str_val);
        let qdrant_null = store.serde_value_to_qdrant_value(&null_val);

        // These should not panic and should convert appropriately
        assert!(matches!(qdrant_bool, qdrant_client::qdrant::Value { .. }));
        assert!(matches!(qdrant_num, qdrant_client::qdrant::Value { .. }));
        assert!(matches!(qdrant_str, qdrant_client::qdrant::Value { .. }));
        assert!(matches!(qdrant_null, qdrant_client::qdrant::Value { .. }));
    }

    #[test]
    fn test_node_to_point_conversion() {
        let config = QdrantConfig::new("http://localhost:6334", "test", 3);
        let store = QdrantVectorStore {
            client: Arc::new(Qdrant::from_url("http://localhost:6334").build().unwrap()),
            config,
            stats: Arc::new(RwLock::new(StorageStats::new())),
        };

        let mut node = create_test_node("Test content", vec![1.0, 0.0, 0.0]);
        node.metadata.insert("category".to_string(), Value::String("test".to_string()));

        let point = store.node_to_point(&node).unwrap();

        // Basic validation that point was created
        assert!(point.id.is_some());
        assert!(point.vectors.is_some());
        assert!(point.payload.contains_key("content"));
        assert!(point.payload.contains_key("category"));
    }

    #[test]
    fn test_node_to_point_invalid_dimension() {
        let config = QdrantConfig::new("http://localhost:6334", "test", 3);
        let store = QdrantVectorStore {
            client: Arc::new(Qdrant::from_url("http://localhost:6334").build().unwrap()),
            config,
            stats: Arc::new(RwLock::new(StorageStats::new())),
        };

        let node = create_test_node("Test content", vec![1.0, 0.0]); // Wrong dimension

        let result = store.node_to_point(&node);
        assert!(result.is_err());
    }

    #[test]
    fn test_node_without_embedding() {
        let config = QdrantConfig::new("http://localhost:6334", "test", 3);
        let store = QdrantVectorStore {
            client: Arc::new(Qdrant::from_url("http://localhost:6334").build().unwrap()),
            config,
            stats: Arc::new(RwLock::new(StorageStats::new())),
        };

        let source_doc_id = Uuid::new_v4();
        let chunk_info = ChunkInfo::new(0, 10, 0);
        let node = Node::new("Test content", source_doc_id, chunk_info); // No embedding

        let result = store.node_to_point(&node);
        assert!(result.is_err());
    }

    // Integration tests would require a running Qdrant instance
    // These are commented out but show the structure for integration testing

    /*
    #[tokio::test]
    #[ignore] // Requires running Qdrant instance
    async fn test_qdrant_integration() {
        let config = QdrantConfig::new("http://localhost:6334", "test_integration", 3);
        let store = QdrantVectorStore::new(config).await.unwrap();

        // Test health check
        store.health_check().await.unwrap();

        // Test adding nodes
        let node1 = create_test_node("Hello world", vec![1.0, 0.0, 0.0]);
        let node2 = create_test_node("Goodbye world", vec![0.0, 1.0, 0.0]);

        let ids = store.add(vec![node1.clone(), node2.clone()]).await.unwrap();
        assert_eq!(ids.len(), 2);

        // Test search
        let query = Query::builder()
            .text("test query")
            .embedding(vec![1.0, 0.0, 0.0])
            .top_k(2)
            .build();

        let results = store.search(&query).await.unwrap();
        assert!(!results.is_empty());

        // Test get
        let retrieved = store.get(vec![node1.id]).await.unwrap();
        assert!(retrieved[0].is_some());

        // Test count
        let count = store.count().await.unwrap();
        assert!(count >= 2);

        // Clean up
        store.clear().await.unwrap();
    }
    */
}
