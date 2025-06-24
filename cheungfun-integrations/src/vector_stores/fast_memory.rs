//! Fast in-memory vector store with real performance optimizations
//!
//! This implementation focuses on actual performance improvements:
//! - Reduced lock contention
//! - Optimized memory layout
//! - Manual vectorization
//! - Minimal overhead

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, info};
use uuid::Uuid;

use cheungfun_core::{
    CheungfunError, Result,
    traits::{DistanceMetric, VectorStore},
    types::{Node, Query, ScoredNode},
};

/// Fast in-memory vector store with real optimizations
#[derive(Debug)]
pub struct FastInMemoryVectorStore {
    dimension: usize,
    distance_metric: DistanceMetric,

    // Optimized storage: separate vectors for better cache locality
    nodes: Arc<RwLock<HashMap<Uuid, Node>>>,
    vectors: Arc<RwLock<Vec<(Uuid, Vec<f32>)>>>, // Flat vector storage

    // Performance counters (minimal overhead)
    search_count: Arc<RwLock<u64>>,
}

impl FastInMemoryVectorStore {
    /// Create a new fast in-memory vector store
    #[must_use]
    pub fn new(dimension: usize, distance_metric: DistanceMetric) -> Self {
        Self {
            dimension,
            distance_metric,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            vectors: Arc::new(RwLock::new(Vec::new())),
            search_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Get search statistics
    #[must_use]
    pub fn get_search_count(&self) -> u64 {
        *self.search_count.read().unwrap()
    }

    /// Validate vector dimensions
    fn validate_vector(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(CheungfunError::Validation {
                message: format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.dimension,
                    vector.len()
                ),
            });
        }
        Ok(())
    }

    /// Fast cosine similarity with manual vectorization
    fn cosine_similarity_fast(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dot_product = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        // Process 4 elements at a time for better CPU vectorization
        let chunks = a.len() / 4;
        let remainder = a.len() % 4;

        for i in 0..chunks {
            let base = i * 4;

            // Unrolled loop for better performance
            let a0 = a[base];
            let a1 = a[base + 1];
            let a2 = a[base + 2];
            let a3 = a[base + 3];

            let b0 = b[base];
            let b1 = b[base + 1];
            let b2 = b[base + 2];
            let b3 = b[base + 3];

            dot_product += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            norm_a += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
            norm_b += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
        }

        // Handle remaining elements
        for i in (chunks * 4)..(chunks * 4 + remainder) {
            let ai = a[i];
            let bi = b[i];
            dot_product += ai * bi;
            norm_a += ai * ai;
            norm_b += bi * bi;
        }

        let norm_a = norm_a.sqrt();
        let norm_b = norm_b.sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Fast dot product
    fn dot_product_fast(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        let chunks = a.len() / 4;
        let remainder = a.len() % 4;

        for i in 0..chunks {
            let base = i * 4;
            sum += a[base] * b[base]
                + a[base + 1] * b[base + 1]
                + a[base + 2] * b[base + 2]
                + a[base + 3] * b[base + 3];
        }

        for i in (chunks * 4)..(chunks * 4 + remainder) {
            sum += a[i] * b[i];
        }

        sum
    }

    /// Fast Euclidean distance
    fn euclidean_distance_fast(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        let chunks = a.len() / 4;
        let remainder = a.len() % 4;

        for i in 0..chunks {
            let base = i * 4;
            let d0 = a[base] - b[base];
            let d1 = a[base + 1] - b[base + 1];
            let d2 = a[base + 2] - b[base + 2];
            let d3 = a[base + 3] - b[base + 3];
            sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }

        for i in (chunks * 4)..(chunks * 4 + remainder) {
            let d = a[i] - b[i];
            sum += d * d;
        }

        sum.sqrt()
    }

    /// Calculate similarity based on distance metric
    fn calculate_similarity(&self, query_vec: &[f32], stored_vec: &[f32]) -> f32 {
        match self.distance_metric {
            DistanceMetric::Cosine => self.cosine_similarity_fast(query_vec, stored_vec),
            DistanceMetric::DotProduct => self.dot_product_fast(query_vec, stored_vec),
            DistanceMetric::Euclidean => {
                let distance = self.euclidean_distance_fast(query_vec, stored_vec);
                1.0 / (1.0 + distance)
            }
            DistanceMetric::Manhattan => {
                let distance: f32 = query_vec
                    .iter()
                    .zip(stored_vec.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                1.0 / (1.0 + distance)
            }
            DistanceMetric::Custom(_) => {
                // Fallback to cosine
                self.cosine_similarity_fast(query_vec, stored_vec)
            }
        }
    }

    /// Check if node matches filters
    fn matches_filters(&self, node: &Node, filters: &HashMap<String, serde_json::Value>) -> bool {
        if filters.is_empty() {
            return true;
        }

        for (key, expected_value) in filters {
            if let Some(actual_value) = node.metadata.get(key) {
                if actual_value != expected_value {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

#[async_trait]
impl VectorStore for FastInMemoryVectorStore {
    async fn add(&self, nodes: Vec<Node>) -> Result<Vec<Uuid>> {
        debug!("Adding {} nodes to FastInMemoryVectorStore", nodes.len());

        let mut node_storage = self.nodes.write().unwrap();
        let mut vector_storage = self.vectors.write().unwrap();
        let mut node_ids = Vec::new();

        for node in nodes {
            if let Some(embedding) = &node.embedding {
                self.validate_vector(embedding)?;

                let node_id = node.id;
                let embedding_clone = embedding.clone();

                node_storage.insert(node_id, node);
                vector_storage.push((node_id, embedding_clone));
                node_ids.push(node_id);
            } else {
                return Err(CheungfunError::Validation {
                    message: "Node must have an embedding".to_string(),
                });
            }
        }

        info!("Successfully added {} nodes", node_ids.len());
        Ok(node_ids)
    }

    async fn search(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        let start_time = std::time::Instant::now();

        debug!(
            "Searching FastInMemoryVectorStore with query: '{}', top_k: {}",
            query.text, query.top_k
        );

        let query_embedding =
            query
                .embedding
                .as_ref()
                .ok_or_else(|| CheungfunError::Validation {
                    message: "Query must have an embedding for vector search".to_string(),
                })?;

        self.validate_vector(query_embedding)?;

        // Read locks once and keep them
        let node_storage = self.nodes.read().unwrap();
        let vector_storage = self.vectors.read().unwrap();

        let mut scored_nodes = Vec::new();

        // Direct iteration without intermediate collections
        for (node_id, stored_vector) in vector_storage.iter() {
            if let Some(node) = node_storage.get(node_id) {
                if self.matches_filters(node, &query.filters) {
                    let similarity = self.calculate_similarity(query_embedding, stored_vector);

                    if let Some(threshold) = query.similarity_threshold {
                        if similarity < threshold {
                            continue;
                        }
                    }

                    scored_nodes.push(ScoredNode::new(node.clone(), similarity));
                }
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

        // Update search count (minimal overhead)
        {
            let mut count = self.search_count.write().unwrap();
            *count += 1;
        }

        let duration = start_time.elapsed();
        info!(
            "Fast search completed in {:?}, returning {} results out of {} total nodes",
            duration,
            scored_nodes.len(),
            vector_storage.len()
        );

        Ok(scored_nodes)
    }

    async fn get(&self, node_ids: Vec<Uuid>) -> Result<Vec<Option<Node>>> {
        debug!(
            "Getting {} nodes from FastInMemoryVectorStore",
            node_ids.len()
        );

        let node_storage = self.nodes.read().unwrap();
        let results = node_ids
            .iter()
            .map(|id| node_storage.get(id).cloned())
            .collect();

        Ok(results)
    }

    async fn update(&self, nodes: Vec<Node>) -> Result<()> {
        debug!("Updating {} nodes in FastInMemoryVectorStore", nodes.len());

        let mut node_storage = self.nodes.write().unwrap();
        let mut vector_storage = self.vectors.write().unwrap();

        for node in nodes {
            if let Some(embedding) = &node.embedding {
                self.validate_vector(embedding)?;

                let node_id = node.id;
                let embedding_clone = embedding.clone();

                node_storage.insert(node_id, node);

                // Update vector in the flat storage
                if let Some(pos) = vector_storage.iter().position(|(id, _)| *id == node_id) {
                    vector_storage[pos] = (node_id, embedding_clone);
                } else {
                    vector_storage.push((node_id, embedding_clone));
                }
            } else {
                return Err(CheungfunError::Validation {
                    message: "Node must have an embedding".to_string(),
                });
            }
        }

        info!("Successfully updated nodes");
        Ok(())
    }

    async fn delete(&self, node_ids: Vec<Uuid>) -> Result<()> {
        debug!(
            "Deleting {} nodes from FastInMemoryVectorStore",
            node_ids.len()
        );

        let mut node_storage = self.nodes.write().unwrap();
        let mut vector_storage = self.vectors.write().unwrap();

        for node_id in &node_ids {
            node_storage.remove(node_id);
            vector_storage.retain(|(id, _)| id != node_id);
        }

        info!("Successfully deleted {} nodes", node_ids.len());
        Ok(())
    }

    async fn health_check(&self) -> Result<()> {
        let node_count = self.nodes.read().unwrap().len();
        let vector_count = self.vectors.read().unwrap().len();

        if node_count != vector_count {
            return Err(CheungfunError::Internal {
                message: format!("Data inconsistency: nodes={node_count}, vectors={vector_count}"),
            });
        }

        debug!(
            "FastInMemoryVectorStore health check passed: {} nodes",
            node_count
        );
        Ok(())
    }
}
