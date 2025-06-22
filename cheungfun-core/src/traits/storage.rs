//! Vector storage and retrieval traits.
//!
//! This module defines traits for storing and retrieving vector embeddings
//! along with their associated nodes. Vector stores are the core component
//! for similarity search in RAG applications.

use async_trait::async_trait;
use std::collections::HashMap;
use uuid::Uuid;

use crate::{Node, Query, Result, ScoredNode};

/// Stores and retrieves vector embeddings with associated nodes.
///
/// This trait provides a unified interface for different vector database
/// implementations, allowing the framework to work with various backends
/// like Qdrant, Pinecone, Weaviate, or in-memory stores.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::VectorStore;
/// use cheungfun_core::{Node, Query, ScoredNode, Result};
/// use async_trait::async_trait;
/// use uuid::Uuid;
///
/// struct InMemoryVectorStore {
///     nodes: std::collections::HashMap<Uuid, Node>,
/// }
///
/// #[async_trait]
/// impl VectorStore for InMemoryVectorStore {
///     async fn add(&self, nodes: Vec<Node>) -> Result<Vec<Uuid>> {
///         // Implementation would store nodes and return their IDs
///         Ok(nodes.iter().map(|n| n.id).collect())
///     }
///
///     async fn search(&self, query: &Query) -> Result<Vec<ScoredNode>> {
///         // Implementation would perform similarity search
///         Ok(vec![])
///     }
///
///     async fn get(&self, node_ids: Vec<Uuid>) -> Result<Vec<Option<Node>>> {
///         // Implementation would retrieve nodes by ID
///         Ok(vec![])
///     }
///
///     async fn delete(&self, node_ids: Vec<Uuid>) -> Result<()> {
///         // Implementation would delete nodes
///         Ok(())
///     }
///
///     async fn update(&self, nodes: Vec<Node>) -> Result<()> {
///         // Implementation would update existing nodes
///         Ok(())
///     }
///
///     async fn health_check(&self) -> Result<()> {
///         Ok(())
///     }
/// }
/// ```
#[async_trait]
pub trait VectorStore: Send + Sync + std::fmt::Debug {
    /// Add nodes to the vector store.
    ///
    /// This method stores the provided nodes along with their embeddings
    /// in the vector store. If a node doesn't have an embedding, it should
    /// be rejected or the implementation should generate one.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Vector of nodes to store
    ///
    /// # Returns
    ///
    /// A vector of UUIDs for the stored nodes. The order should match
    /// the input nodes.
    ///
    /// # Errors
    ///
    /// Returns an error if storage fails due to connection issues,
    /// invalid data, or storage capacity limits.
    async fn add(&self, nodes: Vec<Node>) -> Result<Vec<Uuid>>;

    /// Update existing nodes in the store.
    ///
    /// This method updates nodes that already exist in the store.
    /// If a node doesn't exist, it may be added or an error may be returned
    /// depending on the implementation.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Vector of nodes to update
    async fn update(&self, nodes: Vec<Node>) -> Result<()>;

    /// Delete nodes from the store by their IDs.
    ///
    /// # Arguments
    ///
    /// * `node_ids` - Vector of node IDs to delete
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails. Missing nodes may be
    /// ignored or cause an error depending on the implementation.
    async fn delete(&self, node_ids: Vec<Uuid>) -> Result<()>;

    /// Search for similar nodes using vector similarity.
    ///
    /// This is the core method for retrieval. It performs similarity
    /// search based on the query embedding and returns the most similar
    /// nodes with their similarity scores.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query containing embedding and filters
    ///
    /// # Returns
    ///
    /// A vector of scored nodes sorted by similarity (highest first).
    /// The number of results should respect the `top_k` parameter in the query.
    async fn search(&self, query: &Query) -> Result<Vec<ScoredNode>>;

    /// Retrieve nodes by their IDs.
    ///
    /// # Arguments
    ///
    /// * `node_ids` - Vector of node IDs to retrieve
    ///
    /// # Returns
    ///
    /// A vector of optional nodes. `None` indicates that a node
    /// with the corresponding ID was not found.
    async fn get(&self, node_ids: Vec<Uuid>) -> Result<Vec<Option<Node>>>;

    /// Check if the vector store is healthy and accessible.
    ///
    /// This method can be used to verify connectivity and basic
    /// functionality before performing operations.
    async fn health_check(&self) -> Result<()>;

    /// Get a human-readable name for this vector store.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Get the total number of nodes in the store.
    async fn count(&self) -> Result<usize> {
        // Default implementation returns 0
        Ok(0)
    }

    /// Get metadata about the vector store.
    ///
    /// This can include information like storage capacity, index type,
    /// distance metric, etc.
    async fn metadata(&self) -> Result<HashMap<String, serde_json::Value>> {
        // Default implementation returns empty metadata
        Ok(HashMap::new())
    }

    /// Clear all nodes from the store.
    ///
    /// # Warning
    ///
    /// This operation is destructive and cannot be undone.
    async fn clear(&self) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Create an index for better search performance.
    ///
    /// Some vector stores require explicit index creation or rebuilding
    /// for optimal performance.
    async fn create_index(&self, _config: &IndexConfig) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Get statistics about the vector store.
    async fn stats(&self) -> Result<StorageStats> {
        Ok(StorageStats::default())
    }
}

/// Configuration for vector store indexing.
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Type of index to create (e.g., "hnsw", "ivf", "flat").
    pub index_type: String,

    /// Distance metric to use (e.g., "cosine", "euclidean", "dot").
    pub distance_metric: DistanceMetric,

    /// Index-specific parameters.
    pub parameters: HashMap<String, serde_json::Value>,

    /// Whether to replace existing index.
    pub replace_existing: bool,
}

/// Distance metrics for vector similarity.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity (1 - cosine distance).
    Cosine,

    /// Euclidean distance (L2 norm).
    Euclidean,

    /// Dot product similarity.
    DotProduct,

    /// Manhattan distance (L1 norm).
    Manhattan,

    /// Custom distance metric.
    Custom(String),
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            index_type: "hnsw".to_string(),
            distance_metric: DistanceMetric::Cosine,
            parameters: HashMap::new(),
            replace_existing: false,
        }
    }
}

/// Statistics about vector store operations.
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total number of nodes stored.
    pub total_nodes: usize,

    /// Total storage size in bytes.
    pub storage_size_bytes: Option<u64>,

    /// Average search latency in milliseconds.
    pub avg_search_latency_ms: Option<f64>,

    /// Number of search operations performed.
    pub search_operations: usize,

    /// Number of insert operations performed.
    pub insert_operations: usize,

    /// Number of update operations performed.
    pub update_operations: usize,

    /// Number of delete operations performed.
    pub delete_operations: usize,

    /// Index information.
    pub index_info: Option<IndexInfo>,

    /// Additional store-specific statistics.
    pub additional_stats: HashMap<String, serde_json::Value>,
}

/// Information about vector store indexes.
#[derive(Debug, Clone)]
pub struct IndexInfo {
    /// Type of index.
    pub index_type: String,

    /// Distance metric used.
    pub distance_metric: DistanceMetric,

    /// Dimension of vectors.
    pub dimension: usize,

    /// Whether the index is ready for queries.
    pub is_ready: bool,

    /// Index build time.
    pub build_time: Option<std::time::Duration>,

    /// Index-specific metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl StorageStats {
    /// Create new storage statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate the average operations per second.
    pub fn operations_per_second(&self, duration: std::time::Duration) -> f64 {
        let total_ops = self.search_operations
            + self.insert_operations
            + self.update_operations
            + self.delete_operations;

        if duration.is_zero() {
            0.0
        } else {
            total_ops as f64 / duration.as_secs_f64()
        }
    }

    /// Get the storage efficiency (nodes per byte).
    pub fn storage_efficiency(&self) -> Option<f64> {
        self.storage_size_bytes.map(|size| {
            if size == 0 {
                0.0
            } else {
                self.total_nodes as f64 / size as f64
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_stats() {
        let mut stats = StorageStats::new();
        stats.total_nodes = 1000;
        stats.storage_size_bytes = Some(1_000_000);
        stats.search_operations = 500;
        stats.insert_operations = 100;

        let duration = std::time::Duration::from_secs(10);
        assert_eq!(stats.operations_per_second(duration), 60.0);
        assert_eq!(stats.storage_efficiency(), Some(0.001));
    }

    #[test]
    fn test_distance_metric() {
        assert_eq!(DistanceMetric::Cosine, DistanceMetric::Cosine);
        assert_ne!(DistanceMetric::Cosine, DistanceMetric::Euclidean);

        let custom = DistanceMetric::Custom("hamming".to_string());
        if let DistanceMetric::Custom(name) = custom {
            assert_eq!(name, "hamming");
        }
    }

    #[test]
    fn test_index_config_default() {
        let config = IndexConfig::default();
        assert_eq!(config.index_type, "hnsw");
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert!(!config.replace_existing);
    }
}
