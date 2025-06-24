//! Cache adapters for integrating existing cache implementations with the unified PipelineCache interface.
//!
//! This module provides adapters that allow existing cache implementations like EmbeddingCache
//! and QueryCache to work with the unified PipelineCache trait.

use async_trait::async_trait;
use std::time::Duration;
use tracing::{debug, warn};

use crate::traits::{CacheHealth, CacheStats, HealthStatus, PipelineCache};
use crate::{CheungfunError, Node};

/// Adapter for integrating EmbeddingCache with PipelineCache.
///
/// This adapter allows existing EmbeddingCache implementations to be used
/// through the unified PipelineCache interface.
#[derive(Debug)]
pub struct EmbeddingCacheAdapter<T> {
    /// The underlying embedding cache
    embedding_cache: T,
    /// Cache name for logging
    name: String,
}

impl<T> EmbeddingCacheAdapter<T> {
    /// Create a new embedding cache adapter.
    ///
    /// # Arguments
    /// * `embedding_cache` - The underlying embedding cache implementation
    /// * `name` - A name for this cache instance (used in logging)
    pub fn new(embedding_cache: T, name: String) -> Self {
        Self {
            embedding_cache,
            name,
        }
    }

    /// Get the underlying embedding cache.
    pub fn inner(&self) -> &T {
        &self.embedding_cache
    }

    /// Get a mutable reference to the underlying embedding cache.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.embedding_cache
    }
}

/// Trait for embedding caches that can be adapted to PipelineCache.
///
/// This trait defines the minimal interface that an embedding cache must implement
/// to be used with the EmbeddingCacheAdapter.
#[async_trait]
pub trait AdaptableEmbeddingCache: Send + Sync + std::fmt::Debug {
    /// Error type for cache operations.
    type Error: Send + Sync + std::error::Error + 'static;

    /// Get an embedding from the cache.
    async fn get_embedding(&self, key: &str) -> Result<Option<Vec<f32>>, Self::Error>;

    /// Store an embedding in the cache.
    async fn put_embedding(
        &self,
        key: &str,
        embedding: Vec<f32>,
        ttl: Duration,
    ) -> Result<(), Self::Error>;

    /// Get cache statistics (optional).
    async fn get_stats(&self) -> Result<Option<CacheStats>, Self::Error> {
        Ok(None)
    }

    /// Clear the cache (optional).
    async fn clear_cache(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[async_trait]
impl<T> PipelineCache for EmbeddingCacheAdapter<T>
where
    T: AdaptableEmbeddingCache,
    CheungfunError: From<T::Error>,
{
    type Error = CheungfunError;

    async fn get_embedding(&self, key: &str) -> std::result::Result<Option<Vec<f32>>, Self::Error> {
        debug!("Getting embedding from {} cache: {}", self.name, key);
        self.embedding_cache
            .get_embedding(key)
            .await
            .map_err(CheungfunError::from)
    }

    async fn put_embedding(
        &self,
        key: &str,
        embedding: Vec<f32>,
        ttl: Duration,
    ) -> std::result::Result<(), Self::Error> {
        debug!("Storing embedding in {} cache: {}", self.name, key);
        self.embedding_cache
            .put_embedding(key, embedding, ttl)
            .await
            .map_err(CheungfunError::from)
    }

    async fn get_nodes(&self, _key: &str) -> std::result::Result<Option<Vec<Node>>, Self::Error> {
        // EmbeddingCache doesn't support nodes
        Ok(None)
    }

    async fn put_nodes(
        &self,
        _key: &str,
        _nodes: Vec<Node>,
        _ttl: Duration,
    ) -> std::result::Result<(), Self::Error> {
        // EmbeddingCache doesn't support nodes - silently ignore
        warn!(
            "Attempted to store nodes in embedding-only cache: {}",
            self.name
        );
        Ok(())
    }

    async fn get_data_bytes(
        &self,
        _key: &str,
    ) -> std::result::Result<Option<Vec<u8>>, Self::Error> {
        // EmbeddingCache doesn't support arbitrary data
        Ok(None)
    }

    async fn put_data_bytes(
        &self,
        _key: &str,
        _data_bytes: Vec<u8>,
        _ttl: Duration,
    ) -> std::result::Result<(), Self::Error> {
        // EmbeddingCache doesn't support arbitrary data - silently ignore
        warn!(
            "Attempted to store data bytes in embedding-only cache: {}",
            self.name
        );
        Ok(())
    }

    async fn exists(&self, key: &str) -> std::result::Result<bool, Self::Error> {
        // Check if embedding exists
        Ok(self.get_embedding(key).await?.is_some())
    }

    async fn remove(&self, _key: &str) -> std::result::Result<(), Self::Error> {
        // Most embedding caches don't support removal - silently ignore
        warn!(
            "Remove operation not supported by embedding cache: {}",
            self.name
        );
        Ok(())
    }

    async fn clear(&self) -> std::result::Result<(), Self::Error> {
        debug!("Clearing {} cache", self.name);
        self.embedding_cache
            .clear_cache()
            .await
            .map_err(CheungfunError::from)
    }

    async fn cleanup(&self) -> std::result::Result<usize, Self::Error> {
        // Most embedding caches don't support cleanup - return 0
        debug!(
            "Cleanup operation not supported by embedding cache: {}",
            self.name
        );
        Ok(0)
    }

    async fn stats(&self) -> std::result::Result<CacheStats, Self::Error> {
        match self
            .embedding_cache
            .get_stats()
            .await
            .map_err(CheungfunError::from)?
        {
            Some(stats) => Ok(stats),
            None => Ok(CacheStats::default()),
        }
    }

    async fn health(&self) -> std::result::Result<CacheHealth, Self::Error> {
        let stats = self.stats().await?;

        // Simple health check based on hit rate
        let health_status = if stats.total_operations() > 100 && stats.hit_rate() < 30.0 {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        };

        let mut messages = Vec::new();
        if stats.hit_rate() < 50.0 && stats.total_operations() > 100 {
            messages.push(format!("Low cache hit rate: {:.1}%", stats.hit_rate()));
        }

        Ok(CacheHealth {
            status: health_status,
            usage_ratio: 0.0, // Unknown for adapted caches
            hit_rate: stats.hit_rate(),
            total_entries: stats.total_entries,
            estimated_size_mb: 0.0, // Unknown for adapted caches
            messages,
        })
    }
}

/// Configuration for cache adapters.
#[derive(Debug, Clone)]
pub struct CacheAdapterConfig {
    /// Name for the cache adapter (used in logging)
    pub name: String,
    /// Whether to enable detailed logging
    pub enable_logging: bool,
    /// Whether to enable statistics collection
    pub enable_stats: bool,
}

impl Default for CacheAdapterConfig {
    fn default() -> Self {
        Self {
            name: "adapter".to_string(),
            enable_logging: true,
            enable_stats: true,
        }
    }
}

impl CacheAdapterConfig {
    /// Create a new cache adapter configuration.
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            ..Default::default()
        }
    }

    /// Set whether to enable detailed logging.
    #[must_use]
    pub fn with_logging(mut self, enable: bool) -> Self {
        self.enable_logging = enable;
        self
    }

    /// Set whether to enable statistics collection.
    #[must_use]
    pub fn with_stats(mut self, enable: bool) -> Self {
        self.enable_stats = enable;
        self
    }
}
