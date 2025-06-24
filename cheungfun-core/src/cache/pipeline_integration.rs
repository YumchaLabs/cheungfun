//! Pipeline-level cache integration utilities.
//!
//! This module provides enhanced cache integration for indexing and query pipelines,
//! including intelligent cache key generation, batch operations, and cache warming.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info};

use crate::traits::{CacheKeyGenerator, PipelineCache};
use crate::{CheungfunError, Node};

/// Enhanced cache manager for pipeline operations.
///
/// This manager provides intelligent caching strategies for different pipeline stages,
/// including embedding generation, node processing, and query results.
#[derive(Debug)]
pub struct PipelineCacheManager {
    /// The underlying cache implementation
    cache: Arc<dyn PipelineCache<Error = CheungfunError>>,
    /// Cache configuration
    config: PipelineCacheConfig,
    /// Cache statistics
    stats: Arc<tokio::sync::RwLock<PipelineCacheStats>>,
}

/// Configuration for pipeline cache integration.
#[derive(Debug, Clone)]
pub struct PipelineCacheConfig {
    /// Default TTL for embedding cache entries
    pub embedding_ttl: Duration,
    /// Default TTL for node cache entries
    pub node_ttl: Duration,
    /// Default TTL for query cache entries
    pub query_ttl: Duration,
    /// Whether to enable batch operations
    pub enable_batch_operations: bool,
    /// Batch size for cache operations
    pub batch_size: usize,
    /// Whether to enable cache warming
    pub enable_cache_warming: bool,
    /// Model name for embedding cache keys
    pub model_name: String,
    /// Model version for cache key generation
    pub model_version: Option<String>,
}

impl Default for PipelineCacheConfig {
    fn default() -> Self {
        Self {
            embedding_ttl: Duration::from_secs(3600 * 24), // 24 hours
            node_ttl: Duration::from_secs(3600 * 12),      // 12 hours
            query_ttl: Duration::from_secs(3600),          // 1 hour
            enable_batch_operations: true,
            batch_size: 100,
            enable_cache_warming: false,
            model_name: "default".to_string(),
            model_version: None,
        }
    }
}

/// Statistics for pipeline cache operations.
#[derive(Debug, Default, Clone)]
pub struct PipelineCacheStats {
    /// Embedding cache operations
    pub embedding_hits: u64,
    /// Number of embedding cache misses
    pub embedding_misses: u64,
    /// Number of embedding cache stores
    pub embedding_stores: u64,
    /// Node cache operations
    pub node_hits: u64,
    /// Number of node cache misses
    pub node_misses: u64,
    /// Number of node cache stores
    pub node_stores: u64,
    /// Query cache operations
    pub query_hits: u64,
    /// Number of query cache misses
    pub query_misses: u64,
    /// Number of query cache stores
    pub query_stores: u64,
    /// Batch operation statistics
    pub batch_operations: u64,
    /// Number of items processed in batch operations
    pub batch_items_processed: u64,
}

impl PipelineCacheStats {
    /// Calculate overall hit rate across all cache types.
    #[must_use]
    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits = self.embedding_hits + self.node_hits + self.query_hits;
        let total_operations =
            total_hits + self.embedding_misses + self.node_misses + self.query_misses;

        if total_operations == 0 {
            0.0
        } else {
            (total_hits as f64 / total_operations as f64) * 100.0
        }
    }

    /// Calculate embedding cache hit rate.
    #[must_use]
    pub fn embedding_hit_rate(&self) -> f64 {
        let total = self.embedding_hits + self.embedding_misses;
        if total == 0 {
            0.0
        } else {
            (self.embedding_hits as f64 / total as f64) * 100.0
        }
    }

    /// Calculate node cache hit rate.
    #[must_use]
    pub fn node_hit_rate(&self) -> f64 {
        let total = self.node_hits + self.node_misses;
        if total == 0 {
            0.0
        } else {
            (self.node_hits as f64 / total as f64) * 100.0
        }
    }

    /// Calculate query cache hit rate.
    #[must_use]
    pub fn query_hit_rate(&self) -> f64 {
        let total = self.query_hits + self.query_misses;
        if total == 0 {
            0.0
        } else {
            (self.query_hits as f64 / total as f64) * 100.0
        }
    }
}

impl PipelineCacheManager {
    /// Create a new pipeline cache manager.
    pub fn new(
        cache: Arc<dyn PipelineCache<Error = CheungfunError>>,
        config: PipelineCacheConfig,
    ) -> Self {
        Self {
            cache,
            config,
            stats: Arc::new(tokio::sync::RwLock::new(PipelineCacheStats::default())),
        }
    }

    /// Create a new pipeline cache manager with default configuration.
    pub fn with_default_config(cache: Arc<dyn PipelineCache<Error = CheungfunError>>) -> Self {
        Self::new(cache, PipelineCacheConfig::default())
    }

    /// Get cache statistics.
    pub async fn stats(&self) -> PipelineCacheStats {
        self.stats.read().await.clone()
    }

    /// Clear cache statistics.
    pub async fn clear_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = PipelineCacheStats::default();
    }

    /// Generate cache key for embedding.
    fn embedding_cache_key(&self, content: &str) -> String {
        CacheKeyGenerator::embedding_key(
            content,
            &self.config.model_name,
            self.config.model_version.as_deref(),
        )
    }

    /// Generate cache key for nodes.
    fn nodes_cache_key(&self, document_id: &str, chunk_size: usize, overlap: usize) -> String {
        CacheKeyGenerator::nodes_key(document_id, chunk_size, overlap)
    }

    /// Generate cache key for query.
    fn query_cache_key(&self, query: &str, params: &HashMap<String, String>) -> String {
        CacheKeyGenerator::query_key(query, params)
    }
}

/// Trait for embedding cache operations with pipeline integration.
#[async_trait]
pub trait EmbeddingCacheOps {
    /// Get cached embeddings for multiple texts.
    async fn get_embeddings_cached(
        &self,
        texts: &[String],
    ) -> Result<Vec<Option<Vec<f32>>>, CheungfunError>;

    /// Store embeddings for multiple texts.
    async fn store_embeddings_cached(
        &self,
        text_embeddings: &[(String, Vec<f32>)],
    ) -> Result<(), CheungfunError>;

    /// Get embedding cache statistics.
    async fn embedding_cache_stats(&self) -> Result<(u64, u64), CheungfunError>;
}

#[async_trait]
impl EmbeddingCacheOps for PipelineCacheManager {
    async fn get_embeddings_cached(
        &self,
        texts: &[String],
    ) -> Result<Vec<Option<Vec<f32>>>, CheungfunError> {
        debug!("Getting cached embeddings for {} texts", texts.len());

        if self.config.enable_batch_operations && texts.len() > 1 {
            // Use batch operations for better performance
            let keys: Vec<String> = texts
                .iter()
                .map(|text| self.embedding_cache_key(text))
                .collect();

            let key_refs: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
            let results = self.cache.get_embeddings_batch(&key_refs).await?;

            // Update statistics
            let mut stats = self.stats.write().await;
            stats.batch_operations += 1;
            stats.batch_items_processed += texts.len() as u64;

            for result in &results {
                if result.is_some() {
                    stats.embedding_hits += 1;
                } else {
                    stats.embedding_misses += 1;
                }
            }

            debug!(
                "Batch embedding cache: {} hits, {} misses",
                results.iter().filter(|r| r.is_some()).count(),
                results.iter().filter(|r| r.is_none()).count()
            );

            Ok(results)
        } else {
            // Use individual operations
            let mut results = Vec::with_capacity(texts.len());
            let mut hits = 0;
            let mut misses = 0;

            for text in texts {
                let key = self.embedding_cache_key(text);
                let result = self.cache.get_embedding(&key).await?;

                if result.is_some() {
                    hits += 1;
                } else {
                    misses += 1;
                }

                results.push(result);
            }

            // Update statistics
            let mut stats = self.stats.write().await;
            stats.embedding_hits += hits;
            stats.embedding_misses += misses;

            debug!(
                "Individual embedding cache: {} hits, {} misses",
                hits, misses
            );

            Ok(results)
        }
    }

    async fn store_embeddings_cached(
        &self,
        text_embeddings: &[(String, Vec<f32>)],
    ) -> Result<(), CheungfunError> {
        debug!("Storing {} embeddings in cache", text_embeddings.len());

        if self.config.enable_batch_operations && text_embeddings.len() > 1 {
            // Use batch operations for better performance
            let items: Vec<(String, Vec<f32>, Duration)> = text_embeddings
                .iter()
                .map(|(text, embedding)| {
                    let key = self.embedding_cache_key(text);
                    (key, embedding.clone(), self.config.embedding_ttl)
                })
                .collect();

            // Convert to the expected format for the batch operation
            let batch_items: Vec<(&str, Vec<f32>, Duration)> = items
                .iter()
                .map(|(key, embedding, ttl)| (key.as_str(), embedding.clone(), *ttl))
                .collect();

            self.cache.put_embeddings_batch(&batch_items).await?;

            // Update statistics
            let mut stats = self.stats.write().await;
            stats.batch_operations += 1;
            stats.batch_items_processed += text_embeddings.len() as u64;
            stats.embedding_stores += text_embeddings.len() as u64;

            debug!("Stored {} embeddings in batch", text_embeddings.len());
        } else {
            // Use individual operations
            for (text, embedding) in text_embeddings {
                let key = self.embedding_cache_key(text);
                self.cache
                    .put_embedding(&key, embedding.clone(), self.config.embedding_ttl)
                    .await?;
            }

            // Update statistics
            let mut stats = self.stats.write().await;
            stats.embedding_stores += text_embeddings.len() as u64;

            debug!("Stored {} embeddings individually", text_embeddings.len());
        }

        Ok(())
    }

    async fn embedding_cache_stats(&self) -> Result<(u64, u64), CheungfunError> {
        let stats = self.stats.read().await;
        Ok((stats.embedding_hits, stats.embedding_misses))
    }
}

/// Trait for node cache operations with pipeline integration.
#[async_trait]
pub trait NodeCacheOps {
    /// Get cached nodes for a document.
    async fn get_nodes_cached(
        &self,
        document_id: &str,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<Option<Vec<Node>>, CheungfunError>;

    /// Store nodes for a document.
    async fn store_nodes_cached(
        &self,
        document_id: &str,
        chunk_size: usize,
        overlap: usize,
        nodes: Vec<Node>,
    ) -> Result<(), CheungfunError>;

    /// Get node cache statistics.
    async fn node_cache_stats(&self) -> Result<(u64, u64), CheungfunError>;
}

#[async_trait]
impl NodeCacheOps for PipelineCacheManager {
    async fn get_nodes_cached(
        &self,
        document_id: &str,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<Option<Vec<Node>>, CheungfunError> {
        let key = self.nodes_cache_key(document_id, chunk_size, overlap);
        let result = self.cache.get_nodes(&key).await?;

        // Update statistics
        let mut stats = self.stats.write().await;
        if result.is_some() {
            stats.node_hits += 1;
            debug!("Node cache hit for document: {}", document_id);
        } else {
            stats.node_misses += 1;
            debug!("Node cache miss for document: {}", document_id);
        }

        Ok(result)
    }

    async fn store_nodes_cached(
        &self,
        document_id: &str,
        chunk_size: usize,
        overlap: usize,
        nodes: Vec<Node>,
    ) -> Result<(), CheungfunError> {
        let key = self.nodes_cache_key(document_id, chunk_size, overlap);
        self.cache
            .put_nodes(&key, nodes, self.config.node_ttl)
            .await?;

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.node_stores += 1;

        debug!("Stored nodes for document: {}", document_id);
        Ok(())
    }

    async fn node_cache_stats(&self) -> Result<(u64, u64), CheungfunError> {
        let stats = self.stats.read().await;
        Ok((stats.node_hits, stats.node_misses))
    }
}

/// Cache warming utilities for pipeline optimization.
pub struct CacheWarmer {
    cache_manager: Arc<PipelineCacheManager>,
}

impl CacheWarmer {
    /// Create a new cache warmer.
    pub fn new(cache_manager: Arc<PipelineCacheManager>) -> Self {
        Self { cache_manager }
    }

    /// Warm up the cache with frequently used embeddings.
    pub async fn warm_embeddings(
        &self,
        texts: Vec<String>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<(), CheungfunError> {
        if texts.len() != embeddings.len() {
            return Err(CheungfunError::internal(
                "Texts and embeddings length mismatch",
            ));
        }

        info!("Warming cache with {} embeddings", texts.len());

        let text_embeddings: Vec<(String, Vec<f32>)> =
            texts.into_iter().zip(embeddings.into_iter()).collect();

        self.cache_manager
            .store_embeddings_cached(&text_embeddings)
            .await?;

        info!("Cache warming completed");
        Ok(())
    }

    /// Warm up the cache with frequently used nodes.
    pub async fn warm_nodes(
        &self,
        document_nodes: Vec<(String, usize, usize, Vec<Node>)>,
    ) -> Result<(), CheungfunError> {
        info!(
            "Warming cache with {} node collections",
            document_nodes.len()
        );

        for (document_id, chunk_size, overlap, nodes) in document_nodes {
            self.cache_manager
                .store_nodes_cached(&document_id, chunk_size, overlap, nodes)
                .await?;
        }

        info!("Node cache warming completed");
        Ok(())
    }
}
