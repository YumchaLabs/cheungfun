//! High-performance HNSW (Hierarchical Navigable Small World) vector store implementation
//!
//! This implementation provides approximate nearest neighbor search with excellent performance
//! characteristics for large-scale vector databases. HNSW offers:
//! - Sub-linear search complexity O(log N)
//! - High recall rates (>95% typical)
//! - Excellent performance scaling
//! - Memory efficient storage

use async_trait::async_trait;
use cheungfun_core::{
    CheungfunError, Result,
    traits::{DistanceMetric, VectorStore},
    types::{Node, Query, ScoredNode},
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, info, warn};
use uuid::Uuid;

#[cfg(feature = "hnsw")]
use hnsw_rs::prelude::*;

/// Configuration for HNSW vector store
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of connections per layer
    pub max_connections: usize,
    /// Maximum number of connections for layer 0
    pub max_connections_0: usize,
    /// Level generation factor
    pub ml: f32,
    /// Search parameter for construction
    pub ef_construction: usize,
    /// Search parameter for queries
    pub ef_search: usize,
    /// Enable parallel construction
    pub parallel: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_connections: 16,
            max_connections_0: 32,
            ml: 1.0 / (2.0_f32).ln(),
            ef_construction: 200,
            ef_search: 50,
            parallel: true,
        }
    }
}

/// Performance statistics for HNSW operations
#[derive(Debug, Default, Clone)]
pub struct HnswStats {
    /// Total number of vectors indexed
    pub vectors_indexed: usize,
    /// Total number of search operations
    pub searches_performed: usize,
    /// Average search time in microseconds
    pub avg_search_time_us: f64,
    /// Index construction time in milliseconds
    pub construction_time_ms: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Average recall rate
    pub avg_recall: f64,
    /// Number of layers in the HNSW graph
    pub num_layers: usize,
    /// Total number of connections in the graph
    pub total_connections: usize,
    /// Search efficiency (0.0 to 1.0)
    pub search_efficiency: f64,
}

/// High-performance HNSW vector store
pub struct HnswVectorStore {
    /// Vector dimension
    dimension: usize,
    /// Distance metric for similarity calculation
    distance_metric: DistanceMetric,
    /// HNSW configuration
    config: HnswConfig,
    /// Storage for nodes indexed by ID
    nodes: Arc<RwLock<HashMap<Uuid, Node>>>,
    /// HNSW index
    #[cfg(feature = "hnsw")]
    hnsw_index: Arc<RwLock<Option<Hnsw<'static, f32, DistCosine>>>>,
    /// Vector ID mapping (HNSW internal ID -> UUID)
    #[cfg(feature = "hnsw")]
    id_mapping: Arc<RwLock<HashMap<usize, Uuid>>>,
    /// Reverse ID mapping (UUID -> HNSW internal ID)
    #[cfg(feature = "hnsw")]
    reverse_id_mapping: Arc<RwLock<HashMap<Uuid, usize>>>,
    /// Performance statistics
    stats: Arc<RwLock<HnswStats>>,
    /// Next internal ID counter
    next_id: Arc<RwLock<usize>>,
}

impl HnswVectorStore {
    /// Create a new HNSW vector store
    pub fn new(dimension: usize, distance_metric: DistanceMetric) -> Self {
        Self::with_config(dimension, distance_metric, HnswConfig::default())
    }

    /// Create a new HNSW vector store with custom configuration
    pub fn with_config(
        dimension: usize,
        distance_metric: DistanceMetric,
        config: HnswConfig,
    ) -> Self {
        Self {
            dimension,
            distance_metric,
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "hnsw")]
            hnsw_index: Arc::new(RwLock::new(None)),
            #[cfg(feature = "hnsw")]
            id_mapping: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "hnsw")]
            reverse_id_mapping: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(HnswStats::default())),
            next_id: Arc::new(RwLock::new(0)),
        }
    }

    /// Initialize the HNSW index with estimated capacity
    #[cfg(feature = "hnsw")]
    pub fn initialize_index(&self, estimated_capacity: usize) -> Result<()> {
        let mut index_guard = self.hnsw_index.write().unwrap();

        if index_guard.is_some() {
            warn!("HNSW index already initialized");
            return Ok(());
        }

        let hnsw = Hnsw::<'static, f32, DistCosine>::new(
            self.config.max_connections,
            estimated_capacity,
            self.config.max_connections_0,
            self.config.ef_construction,
            DistCosine {},
        );

        *index_guard = Some(hnsw);
        info!(
            "HNSW index initialized with capacity: {}",
            estimated_capacity
        );

        Ok(())
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> HnswStats {
        self.stats.read().unwrap().clone()
    }

    /// Rebuild the index (useful after bulk insertions)
    #[cfg(feature = "hnsw")]
    pub fn rebuild_index(&self) -> Result<()> {
        let nodes = self.nodes.read().unwrap();
        let node_count = nodes.len();
        drop(nodes);

        if node_count == 0 {
            return Ok(());
        }

        info!("Rebuilding HNSW index for {} nodes", node_count);

        // Clear existing index
        {
            let mut index_guard = self.hnsw_index.write().unwrap();
            *index_guard = None;
        }

        // Reinitialize with current node count
        self.initialize_index(node_count)?;

        // Re-add all nodes
        let nodes = self.nodes.read().unwrap();
        let mut vectors_to_add = Vec::new();

        for (uuid, node) in nodes.iter() {
            if let Some(ref embedding) = node.embedding {
                vectors_to_add.push((*uuid, embedding.clone()));
            }
        }
        drop(nodes);

        // Add vectors to index
        for (uuid, embedding) in vectors_to_add {
            self.add_vector_to_index(uuid, &embedding)?;
        }

        info!("HNSW index rebuilt successfully");
        Ok(())
    }

    /// Add a vector to the HNSW index
    #[cfg(feature = "hnsw")]
    fn add_vector_to_index(&self, uuid: Uuid, embedding: &[f32]) -> Result<()> {
        let mut index_guard = self.hnsw_index.write().unwrap();

        if let Some(ref mut hnsw) = index_guard.as_mut() {
            let internal_id = {
                let mut next_id = self.next_id.write().unwrap();
                let id = *next_id;
                *next_id += 1;
                id
            };

            // Insert vector into HNSW
            hnsw.insert((embedding, internal_id));

            // Update ID mappings
            {
                let mut id_mapping = self.id_mapping.write().unwrap();
                let mut reverse_mapping = self.reverse_id_mapping.write().unwrap();

                id_mapping.insert(internal_id, uuid);
                reverse_mapping.insert(uuid, internal_id);
            }

            // Update stats
            {
                let mut stats = self.stats.write().unwrap();
                stats.vectors_indexed += 1;
            }
        } else {
            return Err(CheungfunError::VectorStore {
                message: "HNSW index not initialized".to_string(),
            });
        }

        Ok(())
    }

    /// Search the HNSW index
    #[cfg(feature = "hnsw")]
    fn search_hnsw(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<(Uuid, f32)>> {
        let start_time = std::time::Instant::now();

        let index_guard = self.hnsw_index.read().unwrap();

        if let Some(ref hnsw) = index_guard.as_ref() {
            let search_results = hnsw.search(query_embedding, top_k, self.config.ef_search);

            let mut results = Vec::new();
            let id_mapping = self.id_mapping.read().unwrap();

            for neighbor in search_results {
                if let Some(&uuid) = id_mapping.get(&neighbor.d_id) {
                    // Convert distance to similarity (HNSW returns distances)
                    let similarity = 1.0 - neighbor.distance;
                    results.push((uuid, similarity));
                }
            }

            // Update search statistics
            {
                let mut stats = self.stats.write().unwrap();
                stats.searches_performed += 1;
                let search_time_us = start_time.elapsed().as_micros() as f64;
                stats.avg_search_time_us = (stats.avg_search_time_us
                    * (stats.searches_performed - 1) as f64
                    + search_time_us)
                    / stats.searches_performed as f64;
            }

            Ok(results)
        } else {
            Err(CheungfunError::VectorStore {
                message: "HNSW index not initialized".to_string(),
            })
        }
    }

    /// Fallback to linear search when HNSW is not available
    #[cfg(not(feature = "hnsw"))]
    fn search_linear(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<(Uuid, f32)>> {
        warn!("HNSW feature not enabled, falling back to linear search");

        let nodes = self.nodes.read().unwrap();
        let mut scored_results = Vec::new();

        for (uuid, node) in nodes.iter() {
            if let Some(ref stored_embedding) = node.embedding {
                if stored_embedding.len() != query_embedding.len() {
                    continue;
                }

                let similarity = self.calculate_similarity(query_embedding, stored_embedding);
                scored_results.push((*uuid, similarity));
            }
        }

        // Sort by similarity (descending) and take top_k
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_results.truncate(top_k);

        Ok(scored_results)
    }

    /// Calculate similarity between two vectors
    fn calculate_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        match self.distance_metric {
            DistanceMetric::Cosine => {
                let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
                let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

                if norm1 == 0.0 || norm2 == 0.0 {
                    0.0
                } else {
                    dot_product / (norm1 * norm2)
                }
            }
            DistanceMetric::Euclidean => {
                let distance: f32 = vec1
                    .iter()
                    .zip(vec2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                1.0 / (1.0 + distance)
            }
            DistanceMetric::DotProduct => vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum(),
            DistanceMetric::Manhattan => {
                let distance: f32 = vec1
                    .iter()
                    .zip(vec2.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                1.0 / (1.0 + distance)
            }
            DistanceMetric::Custom(_) => {
                warn!("Custom distance metric not implemented, falling back to cosine");
                let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
                let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

                if norm1 == 0.0 || norm2 == 0.0 {
                    0.0
                } else {
                    dot_product / (norm1 * norm2)
                }
            }
        }
    }
}

impl std::fmt::Debug for HnswVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswVectorStore")
            .field("dimension", &self.dimension)
            .field("distance_metric", &self.distance_metric)
            .field("config", &self.config)
            .field("node_count", &self.nodes.read().unwrap().len())
            .finish()
    }
}

#[async_trait]
impl VectorStore for HnswVectorStore {
    async fn add(&self, nodes: Vec<Node>) -> Result<Vec<Uuid>> {
        debug!("Adding {} nodes to HnswVectorStore", nodes.len());

        let mut node_storage = self.nodes.write().unwrap();
        let mut node_ids = Vec::new();

        // Initialize index if not already done
        #[cfg(feature = "hnsw")]
        {
            let index_guard = self.hnsw_index.read().unwrap();
            if index_guard.is_none() {
                drop(index_guard);
                self.initialize_index(1000)?; // Default capacity
            }
        }

        for node in nodes {
            let node_id = node.id;

            // Validate embedding
            if let Some(ref embedding) = node.embedding {
                if embedding.len() != self.dimension {
                    return Err(CheungfunError::Validation {
                        message: format!(
                            "Embedding dimension mismatch: expected {}, got {}",
                            self.dimension,
                            embedding.len()
                        ),
                    });
                }

                // Add to HNSW index
                #[cfg(feature = "hnsw")]
                self.add_vector_to_index(node_id, embedding)?;
            }

            // Store node
            node_storage.insert(node_id, node);
            node_ids.push(node_id);
        }

        info!("Added {} nodes to HNSW vector store", node_ids.len());
        Ok(node_ids)
    }

    async fn search(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        debug!("Searching HNSW vector store with query: {:?}", query.text);

        let query_embedding =
            query
                .embedding
                .as_ref()
                .ok_or_else(|| CheungfunError::Validation {
                    message: "Query embedding is required for vector search".to_string(),
                })?;

        if query_embedding.len() != self.dimension {
            return Err(CheungfunError::Validation {
                message: format!(
                    "Query embedding dimension mismatch: expected {}, got {}",
                    self.dimension,
                    query_embedding.len()
                ),
            });
        }

        // Perform search
        #[cfg(feature = "hnsw")]
        let search_results = self.search_hnsw(&query_embedding, query.top_k)?;

        #[cfg(not(feature = "hnsw"))]
        let search_results = self.search_linear(&query_embedding, query.top_k)?;

        // Convert to ScoredNode
        let nodes = self.nodes.read().unwrap();
        let mut scored_nodes = Vec::new();

        for (uuid, similarity) in search_results {
            if let Some(node) = nodes.get(&uuid) {
                // Apply similarity threshold if specified
                if let Some(threshold) = query.similarity_threshold {
                    if similarity < threshold {
                        continue;
                    }
                }

                scored_nodes.push(ScoredNode::new(node.clone(), similarity));
            }
        }

        debug!("Found {} matching nodes", scored_nodes.len());
        Ok(scored_nodes)
    }

    async fn get(&self, node_ids: Vec<Uuid>) -> Result<Vec<Option<Node>>> {
        let nodes = self.nodes.read().unwrap();
        let mut results = Vec::new();
        for id in node_ids {
            results.push(nodes.get(&id).cloned());
        }
        Ok(results)
    }

    async fn delete(&self, node_ids: Vec<Uuid>) -> Result<()> {
        let mut nodes = self.nodes.write().unwrap();

        for id in node_ids {
            let removed = nodes.remove(&id).is_some();

            if removed {
                // Remove from HNSW index mappings
                #[cfg(feature = "hnsw")]
                {
                    let mut reverse_mapping = self.reverse_id_mapping.write().unwrap();
                    if let Some(internal_id) = reverse_mapping.remove(&id) {
                        let mut id_mapping = self.id_mapping.write().unwrap();
                        id_mapping.remove(&internal_id);
                    }
                }

                debug!("Deleted node with ID: {}", id);
            }
        }

        Ok(())
    }

    async fn update(&self, nodes: Vec<Node>) -> Result<()> {
        let mut node_storage = self.nodes.write().unwrap();

        for node in nodes {
            let node_id = node.id;

            if let Some(existing_node) = node_storage.get_mut(&node_id) {
                // Update the node
                *existing_node = node.clone();

                // If embedding changed, update HNSW index
                if let Some(ref new_embedding) = node.embedding {
                    if new_embedding.len() != self.dimension {
                        return Err(CheungfunError::Validation {
                            message: format!(
                                "Embedding dimension mismatch: expected {}, got {}",
                                self.dimension,
                                new_embedding.len()
                            ),
                        });
                    }

                    // For HNSW, we need to remove and re-add the vector
                    // This is a limitation of most HNSW implementations
                    #[cfg(feature = "hnsw")]
                    {
                        // Remove from mappings
                        let mut reverse_mapping = self.reverse_id_mapping.write().unwrap();
                        if let Some(internal_id) = reverse_mapping.remove(&node_id) {
                            let mut id_mapping = self.id_mapping.write().unwrap();
                            id_mapping.remove(&internal_id);
                        }
                        drop(reverse_mapping);

                        // Re-add with new embedding
                        self.add_vector_to_index(node_id, new_embedding)?;
                    }
                }

                debug!("Updated node with ID: {}", node_id);
            } else {
                return Err(CheungfunError::VectorStore {
                    message: format!("Node with ID {} not found", node_id),
                });
            }
        }

        Ok(())
    }

    async fn health_check(&self) -> Result<()> {
        // Check if the vector store is accessible and functional
        let nodes = self.nodes.read().unwrap();
        let node_count = nodes.len();
        drop(nodes);

        #[cfg(feature = "hnsw")]
        {
            let index_guard = self.hnsw_index.read().unwrap();
            if index_guard.is_none() && node_count > 0 {
                return Err(CheungfunError::VectorStore {
                    message: "HNSW index not initialized but nodes exist".to_string(),
                });
            }
        }

        debug!(
            "HNSW vector store health check passed: {} nodes",
            node_count
        );
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let mut nodes = self.nodes.write().unwrap();
        nodes.clear();

        #[cfg(feature = "hnsw")]
        {
            let mut index_guard = self.hnsw_index.write().unwrap();
            *index_guard = None;

            let mut id_mapping = self.id_mapping.write().unwrap();
            let mut reverse_mapping = self.reverse_id_mapping.write().unwrap();
            let mut next_id = self.next_id.write().unwrap();

            id_mapping.clear();
            reverse_mapping.clear();
            *next_id = 0;
        }

        let mut stats = self.stats.write().unwrap();
        *stats = HnswStats::default();

        info!("Cleared HNSW vector store");
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let nodes = self.nodes.read().unwrap();
        Ok(nodes.len())
    }
}
