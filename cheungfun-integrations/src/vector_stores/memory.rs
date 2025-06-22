//! In-memory vector store implementation.
//!
//! This module provides a simple in-memory vector store that stores all
//! vectors and nodes in memory. It's suitable for development, testing,
//! and small-scale applications.

use async_trait::async_trait;
use cheungfun_core::{
    Result,
    traits::{DistanceMetric, StorageStats, VectorStore},
    types::{Node, Query, ScoredNode},
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// In-memory vector store implementation.
///
/// This store keeps all vectors and nodes in memory using HashMap for storage.
/// It supports basic CRUD operations and similarity search with different
/// distance metrics.
///
/// # Examples
///
/// ```rust
/// use cheungfun_integrations::InMemoryVectorStore;
/// use cheungfun_core::traits::{VectorStore, DistanceMetric};
///
/// let store = InMemoryVectorStore::new(384, DistanceMetric::Cosine);
/// ```
#[derive(Debug)]
pub struct InMemoryVectorStore {
    /// Vector dimension
    dimension: usize,
    /// Distance metric for similarity calculation
    distance_metric: DistanceMetric,
    /// Storage for nodes indexed by ID
    nodes: Arc<RwLock<HashMap<Uuid, Node>>>,
    /// Storage for vectors indexed by node ID
    vectors: Arc<RwLock<HashMap<Uuid, Vec<f32>>>>,
    /// Statistics tracking
    stats: Arc<RwLock<StorageStats>>,
}

impl InMemoryVectorStore {
    /// Create a new in-memory vector store.
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension of vectors to store
    /// * `distance_metric` - The distance metric to use for similarity search
    pub fn new(dimension: usize, distance_metric: DistanceMetric) -> Self {
        info!(
            "Creating InMemoryVectorStore with dimension {} and metric {:?}",
            dimension, distance_metric
        );

        Self {
            dimension,
            distance_metric,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            vectors: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(StorageStats::new())),
        }
    }

    /// Get the vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the distance metric.
    pub fn distance_metric(&self) -> &DistanceMetric {
        &self.distance_metric
    }

    /// Calculate similarity between two vectors based on the configured metric.
    fn calculate_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        match self.distance_metric {
            DistanceMetric::Cosine => cosine_similarity(vec1, vec2),
            DistanceMetric::Euclidean => {
                // Convert distance to similarity (higher is better)
                let distance = euclidean_distance(vec1, vec2);
                1.0 / (1.0 + distance)
            }
            DistanceMetric::DotProduct => dot_product(vec1, vec2),
            DistanceMetric::Manhattan => {
                // Convert distance to similarity (higher is better)
                let distance = manhattan_distance(vec1, vec2);
                1.0 / (1.0 + distance)
            }
            DistanceMetric::Custom(_) => {
                warn!("Custom distance metric not implemented, falling back to cosine");
                cosine_similarity(vec1, vec2)
            }
        }
    }

    /// Validate that a vector has the correct dimension.
    fn validate_vector(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(cheungfun_core::CheungfunError::Validation {
                message: format!(
                    "Vector dimension {} does not match store dimension {}",
                    vector.len(),
                    self.dimension
                ),
            });
        }
        Ok(())
    }

    /// Check if metadata filters match a node.
    fn matches_filters(&self, node: &Node, filters: &HashMap<String, serde_json::Value>) -> bool {
        if filters.is_empty() {
            return true;
        }

        for (key, expected_value) in filters {
            match node.metadata.get(key) {
                Some(actual_value) if actual_value == expected_value => continue,
                _ => return false,
            }
        }

        true
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn add(&self, nodes: Vec<Node>) -> Result<Vec<Uuid>> {
        debug!("Adding {} nodes to InMemoryVectorStore", nodes.len());

        let mut node_storage = self.nodes.write().unwrap();
        let mut vector_storage = self.vectors.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        let mut added_ids = Vec::new();

        for node in nodes {
            // Validate that the node has an embedding
            let embedding = node.embedding.as_ref().ok_or_else(|| {
                cheungfun_core::CheungfunError::Validation {
                    message: "Node must have an embedding to be stored".to_string(),
                }
            })?;

            // Validate vector dimension
            self.validate_vector(embedding)?;

            let node_id = node.id;
            vector_storage.insert(node_id, embedding.clone());
            node_storage.insert(node_id, node);
            added_ids.push(node_id);
        }

        stats.insert_operations += 1;
        stats.total_nodes = node_storage.len();

        info!("Successfully added {} nodes", added_ids.len());
        Ok(added_ids)
    }

    async fn update(&self, nodes: Vec<Node>) -> Result<()> {
        let node_count = nodes.len();
        debug!("Updating {} nodes in InMemoryVectorStore", node_count);

        let mut node_storage = self.nodes.write().unwrap();
        let mut vector_storage = self.vectors.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        for node in nodes {
            let node_id = node.id;

            // Check if node exists
            if !node_storage.contains_key(&node_id) {
                return Err(cheungfun_core::CheungfunError::NotFound {
                    resource: format!("Node with ID {}", node_id),
                });
            }

            // Update embedding if provided
            if let Some(embedding) = &node.embedding {
                self.validate_vector(embedding)?;
                vector_storage.insert(node_id, embedding.clone());
            }

            // Update node
            node_storage.insert(node_id, node);
        }

        stats.update_operations += 1;

        info!("Successfully updated {} nodes", node_count);
        Ok(())
    }

    async fn delete(&self, node_ids: Vec<Uuid>) -> Result<()> {
        debug!("Deleting {} nodes from InMemoryVectorStore", node_ids.len());

        let mut node_storage = self.nodes.write().unwrap();
        let mut vector_storage = self.vectors.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        for node_id in &node_ids {
            node_storage.remove(node_id);
            vector_storage.remove(node_id);
        }

        stats.delete_operations += 1;
        stats.total_nodes = node_storage.len();

        info!("Successfully deleted {} nodes", node_ids.len());
        Ok(())
    }

    async fn search(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        debug!(
            "Searching InMemoryVectorStore with query: '{}', top_k: {}",
            query.text, query.top_k
        );

        let node_storage = self.nodes.read().unwrap();
        let vector_storage = self.vectors.read().unwrap();
        let mut stats = self.stats.write().unwrap();

        // Get query embedding
        let query_embedding =
            query
                .embedding
                .as_ref()
                .ok_or_else(|| cheungfun_core::CheungfunError::Validation {
                    message: "Query must have an embedding for vector search".to_string(),
                })?;

        self.validate_vector(query_embedding)?;

        let mut scored_nodes = Vec::new();

        // Calculate similarity for each stored vector
        for (node_id, stored_vector) in vector_storage.iter() {
            if let Some(node) = node_storage.get(node_id) {
                // Apply metadata filters
                if !self.matches_filters(node, &query.filters) {
                    continue;
                }

                // Calculate similarity
                let similarity = self.calculate_similarity(query_embedding, stored_vector);

                // Apply similarity threshold if specified
                if let Some(threshold) = query.similarity_threshold {
                    if similarity < threshold {
                        continue;
                    }
                }

                scored_nodes.push(ScoredNode::new(node.clone(), similarity));
            }
        }

        // Sort by similarity (highest first)
        scored_nodes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to top_k results
        scored_nodes.truncate(query.top_k);

        stats.search_operations += 1;

        info!(
            "Search completed, returning {} results out of {} total nodes",
            scored_nodes.len(),
            node_storage.len()
        );

        Ok(scored_nodes)
    }

    async fn get(&self, node_ids: Vec<Uuid>) -> Result<Vec<Option<Node>>> {
        debug!("Getting {} nodes from InMemoryVectorStore", node_ids.len());

        let node_storage = self.nodes.read().unwrap();
        let results = node_ids
            .iter()
            .map(|id| node_storage.get(id).cloned())
            .collect();

        Ok(results)
    }

    async fn health_check(&self) -> Result<()> {
        // For in-memory store, we just check if we can acquire locks
        let _nodes = self.nodes.read().unwrap();
        let _vectors = self.vectors.read().unwrap();
        let _stats = self.stats.read().unwrap();

        debug!("InMemoryVectorStore health check passed");
        Ok(())
    }

    fn name(&self) -> &'static str {
        "InMemoryVectorStore"
    }

    async fn count(&self) -> Result<usize> {
        let node_storage = self.nodes.read().unwrap();
        Ok(node_storage.len())
    }

    async fn metadata(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "in_memory".into());
        metadata.insert("dimension".to_string(), self.dimension.into());
        metadata.insert(
            "distance_metric".to_string(),
            format!("{:?}", self.distance_metric).into(),
        );

        let node_count = self.count().await?;
        metadata.insert("node_count".to_string(), node_count.into());

        Ok(metadata)
    }

    async fn clear(&self) -> Result<()> {
        debug!("Clearing all nodes from InMemoryVectorStore");

        let mut node_storage = self.nodes.write().unwrap();
        let mut vector_storage = self.vectors.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        node_storage.clear();
        vector_storage.clear();
        stats.total_nodes = 0;

        info!("Successfully cleared all nodes");
        Ok(())
    }

    async fn stats(&self) -> Result<StorageStats> {
        let stats = self.stats.read().unwrap();
        Ok(stats.clone())
    }
}

/// Calculate cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product = dot_product(a, b);
    let norm_a = (a.iter().map(|x| x * x).sum::<f32>()).sqrt();
    let norm_b = (b.iter().map(|x| x * x).sum::<f32>()).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Calculate dot product between two vectors.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate Euclidean distance between two vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Calculate Manhattan distance between two vectors.
fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::types::{ChunkInfo, Query};
    use uuid::Uuid;

    fn create_test_node(content: &str, embedding: Vec<f32>) -> Node {
        let source_doc_id = Uuid::new_v4();
        let chunk_info = ChunkInfo::new(0, content.len(), 0);

        Node::new(content, source_doc_id, chunk_info).with_embedding(embedding)
    }

    #[tokio::test]
    async fn test_new_store() {
        let store = InMemoryVectorStore::new(3, DistanceMetric::Cosine);
        assert_eq!(store.dimension(), 3);
        assert_eq!(store.distance_metric(), &DistanceMetric::Cosine);

        let count = store.count().await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_add_and_get_nodes() {
        let store = InMemoryVectorStore::new(3, DistanceMetric::Cosine);

        let node1 = create_test_node("Hello world", vec![1.0, 0.0, 0.0]);
        let node2 = create_test_node("Goodbye world", vec![0.0, 1.0, 0.0]);

        let node1_id = node1.id;
        let node2_id = node2.id;

        // Add nodes
        let added_ids = store.add(vec![node1.clone(), node2.clone()]).await.unwrap();
        assert_eq!(added_ids.len(), 2);
        assert!(added_ids.contains(&node1_id));
        assert!(added_ids.contains(&node2_id));

        // Check count
        let count = store.count().await.unwrap();
        assert_eq!(count, 2);

        // Get nodes
        let retrieved = store.get(vec![node1_id, node2_id]).await.unwrap();
        assert_eq!(retrieved.len(), 2);
        assert!(retrieved[0].is_some());
        assert!(retrieved[1].is_some());

        let retrieved_node1 = retrieved[0].as_ref().unwrap();
        let retrieved_node2 = retrieved[1].as_ref().unwrap();

        assert_eq!(retrieved_node1.content, "Hello world");
        assert_eq!(retrieved_node2.content, "Goodbye world");
    }

    #[tokio::test]
    async fn test_search() {
        let store = InMemoryVectorStore::new(3, DistanceMetric::Cosine);

        let node1 = create_test_node("Hello world", vec![1.0, 0.0, 0.0]);
        let node2 = create_test_node("Goodbye world", vec![0.0, 1.0, 0.0]);
        let node3 = create_test_node("Hello again", vec![0.9, 0.1, 0.0]);

        store.add(vec![node1, node2, node3]).await.unwrap();

        // Search with query similar to first node
        let query = Query::builder()
            .text("test query")
            .embedding(vec![1.0, 0.0, 0.0])
            .top_k(2)
            .build();

        let results = store.search(&query).await.unwrap();
        assert_eq!(results.len(), 2);

        // Results should be sorted by similarity (highest first)
        assert!(results[0].score >= results[1].score);

        // First result should be most similar (exact match)
        assert_eq!(results[0].node.content, "Hello world");
        assert!((results[0].score - 1.0).abs() < 0.001); // Cosine similarity should be 1.0
    }

    #[tokio::test]
    async fn test_search_with_filters() {
        let store = InMemoryVectorStore::new(3, DistanceMetric::Cosine);

        let mut node1 = create_test_node("Hello world", vec![1.0, 0.0, 0.0]);
        node1
            .metadata
            .insert("category".to_string(), "greeting".into());

        let mut node2 = create_test_node("Goodbye world", vec![0.0, 1.0, 0.0]);
        node2
            .metadata
            .insert("category".to_string(), "farewell".into());

        store.add(vec![node1, node2]).await.unwrap();

        // Search with filter
        let query = Query::builder()
            .text("test query")
            .embedding(vec![1.0, 0.0, 0.0])
            .filter("category", "greeting")
            .top_k(10)
            .build();

        let results = store.search(&query).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node.content, "Hello world");
    }

    #[tokio::test]
    async fn test_update_node() {
        let store = InMemoryVectorStore::new(3, DistanceMetric::Cosine);

        let mut node = create_test_node("Original content", vec![1.0, 0.0, 0.0]);
        let node_id = node.id;

        store.add(vec![node.clone()]).await.unwrap();

        // Update the node
        node.content = "Updated content".to_string();
        node.embedding = Some(vec![0.0, 1.0, 0.0]);

        store.update(vec![node]).await.unwrap();

        // Verify update
        let retrieved = store.get(vec![node_id]).await.unwrap();
        let updated_node = retrieved[0].as_ref().unwrap();

        assert_eq!(updated_node.content, "Updated content");
        assert_eq!(updated_node.embedding, Some(vec![0.0, 1.0, 0.0]));
    }

    #[tokio::test]
    async fn test_delete_nodes() {
        let store = InMemoryVectorStore::new(3, DistanceMetric::Cosine);

        let node1 = create_test_node("Hello world", vec![1.0, 0.0, 0.0]);
        let node2 = create_test_node("Goodbye world", vec![0.0, 1.0, 0.0]);

        let node1_id = node1.id;
        let node2_id = node2.id;

        store.add(vec![node1, node2]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);

        // Delete one node
        store.delete(vec![node1_id]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);

        // Verify deletion
        let retrieved = store.get(vec![node1_id, node2_id]).await.unwrap();
        assert!(retrieved[0].is_none());
        assert!(retrieved[1].is_some());
    }

    #[tokio::test]
    async fn test_clear() {
        let store = InMemoryVectorStore::new(3, DistanceMetric::Cosine);

        let node1 = create_test_node("Hello world", vec![1.0, 0.0, 0.0]);
        let node2 = create_test_node("Goodbye world", vec![0.0, 1.0, 0.0]);

        store.add(vec![node1, node2]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);

        store.clear().await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_health_check() {
        let store = InMemoryVectorStore::new(3, DistanceMetric::Cosine);
        store.health_check().await.unwrap();
    }

    #[tokio::test]
    async fn test_metadata() {
        let store = InMemoryVectorStore::new(384, DistanceMetric::Euclidean);
        let metadata = store.metadata().await.unwrap();

        assert_eq!(metadata.get("type").unwrap(), "in_memory");
        assert_eq!(metadata.get("dimension").unwrap(), &384);
        assert_eq!(metadata.get("node_count").unwrap(), &0);
    }

    #[test]
    fn test_similarity_functions() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        let vec3 = vec![1.0, 0.0, 0.0];

        // Cosine similarity
        let cos_sim_orthogonal = cosine_similarity(&vec1, &vec2);
        let cos_sim_identical = cosine_similarity(&vec1, &vec3);

        assert!((cos_sim_orthogonal - 0.0).abs() < 0.001);
        assert!((cos_sim_identical - 1.0).abs() < 0.001);

        // Dot product
        let dot_orthogonal = dot_product(&vec1, &vec2);
        let dot_identical = dot_product(&vec1, &vec3);

        assert!((dot_orthogonal - 0.0).abs() < 0.001);
        assert!((dot_identical - 1.0).abs() < 0.001);

        // Euclidean distance
        let eucl_orthogonal = euclidean_distance(&vec1, &vec2);
        let eucl_identical = euclidean_distance(&vec1, &vec3);

        assert!((eucl_orthogonal - 1.414).abs() < 0.01); // sqrt(2)
        assert!((eucl_identical - 0.0).abs() < 0.001);

        // Manhattan distance
        let manh_orthogonal = manhattan_distance(&vec1, &vec2);
        let manh_identical = manhattan_distance(&vec1, &vec3);

        assert!((manh_orthogonal - 2.0).abs() < 0.001);
        assert!((manh_identical - 0.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_invalid_dimension() {
        let store = InMemoryVectorStore::new(3, DistanceMetric::Cosine);

        let node = create_test_node("Test", vec![1.0, 0.0]); // Wrong dimension

        let result = store.add(vec![node]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_without_embedding() {
        let store = InMemoryVectorStore::new(3, DistanceMetric::Cosine);

        let query = Query::builder().text("test query").top_k(5).build(); // No embedding

        let result = store.search(&query).await;
        assert!(result.is_err());
    }
}
