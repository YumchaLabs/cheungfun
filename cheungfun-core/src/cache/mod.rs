//! Cache implementations for the Cheungfun framework.
//!
//! This module provides various cache implementations that can be used
//! across different components of the RAG pipeline.

pub mod file_cache;
pub mod memory_cache;

use async_trait::async_trait;
use std::time::Duration;

use crate::traits::{CacheHealth, CacheStats, PipelineCache};
use crate::{CheungfunError, Node};

// Re-export cache implementations
pub use file_cache::FileCache;
pub use memory_cache::MemoryCache;

/// Unified cache enum that can hold different cache implementations.
///
/// This enum allows using different cache types in a type-safe way without
/// requiring trait objects, which solves the dyn compatibility issue.
#[derive(Debug)]
pub enum UnifiedCache {
    /// Memory-based cache
    Memory(MemoryCache),
    /// File-based cache
    File(FileCache),
}

#[async_trait]
impl PipelineCache for UnifiedCache {
    type Error = CheungfunError;

    async fn get_embedding(&self, key: &str) -> std::result::Result<Option<Vec<f32>>, Self::Error> {
        match self {
            UnifiedCache::Memory(cache) => cache.get_embedding(key).await,
            UnifiedCache::File(cache) => cache.get_embedding(key).await,
        }
    }

    async fn put_embedding(
        &self,
        key: &str,
        embedding: Vec<f32>,
        ttl: Duration,
    ) -> std::result::Result<(), Self::Error> {
        match self {
            UnifiedCache::Memory(cache) => cache.put_embedding(key, embedding, ttl).await,
            UnifiedCache::File(cache) => cache.put_embedding(key, embedding, ttl).await,
        }
    }

    async fn get_nodes(&self, key: &str) -> std::result::Result<Option<Vec<Node>>, Self::Error> {
        match self {
            UnifiedCache::Memory(cache) => cache.get_nodes(key).await,
            UnifiedCache::File(cache) => cache.get_nodes(key).await,
        }
    }

    async fn put_nodes(
        &self,
        key: &str,
        nodes: Vec<Node>,
        ttl: Duration,
    ) -> std::result::Result<(), Self::Error> {
        match self {
            UnifiedCache::Memory(cache) => cache.put_nodes(key, nodes, ttl).await,
            UnifiedCache::File(cache) => cache.put_nodes(key, nodes, ttl).await,
        }
    }

    async fn get_data_bytes(&self, key: &str) -> std::result::Result<Option<Vec<u8>>, Self::Error> {
        match self {
            UnifiedCache::Memory(cache) => cache.get_data_bytes(key).await,
            UnifiedCache::File(cache) => cache.get_data_bytes(key).await,
        }
    }

    async fn put_data_bytes(
        &self,
        key: &str,
        data_bytes: Vec<u8>,
        ttl: Duration,
    ) -> std::result::Result<(), Self::Error> {
        match self {
            UnifiedCache::Memory(cache) => cache.put_data_bytes(key, data_bytes, ttl).await,
            UnifiedCache::File(cache) => cache.put_data_bytes(key, data_bytes, ttl).await,
        }
    }

    async fn exists(&self, key: &str) -> std::result::Result<bool, Self::Error> {
        match self {
            UnifiedCache::Memory(cache) => cache.exists(key).await,
            UnifiedCache::File(cache) => cache.exists(key).await,
        }
    }

    async fn remove(&self, key: &str) -> std::result::Result<(), Self::Error> {
        match self {
            UnifiedCache::Memory(cache) => cache.remove(key).await,
            UnifiedCache::File(cache) => cache.remove(key).await,
        }
    }

    async fn clear(&self) -> std::result::Result<(), Self::Error> {
        match self {
            UnifiedCache::Memory(cache) => cache.clear().await,
            UnifiedCache::File(cache) => cache.clear().await,
        }
    }

    async fn cleanup(&self) -> std::result::Result<usize, Self::Error> {
        match self {
            UnifiedCache::Memory(cache) => cache.cleanup().await,
            UnifiedCache::File(cache) => cache.cleanup().await,
        }
    }

    async fn stats(&self) -> std::result::Result<CacheStats, Self::Error> {
        match self {
            UnifiedCache::Memory(cache) => cache.stats().await,
            UnifiedCache::File(cache) => cache.stats().await,
        }
    }

    async fn health(&self) -> std::result::Result<CacheHealth, Self::Error> {
        match self {
            UnifiedCache::Memory(cache) => cache.health().await,
            UnifiedCache::File(cache) => cache.health().await,
        }
    }
}

impl UnifiedCache {
    /// Create a new memory cache.
    #[must_use]
    pub fn memory() -> Self {
        UnifiedCache::Memory(MemoryCache::new())
    }

    /// Create a new memory cache with custom configuration.
    #[must_use]
    pub fn memory_with_config(default_ttl: Duration, max_size: usize) -> Self {
        UnifiedCache::Memory(MemoryCache::with_config(default_ttl, max_size))
    }

    /// Create a new file cache.
    pub async fn file<P: AsRef<std::path::Path>>(cache_dir: P) -> Result<Self, CheungfunError> {
        Ok(UnifiedCache::File(FileCache::new(cache_dir).await?))
    }

    /// Create a new file cache with custom configuration.
    pub async fn file_with_config<P: AsRef<std::path::Path>>(
        cache_dir: P,
        default_ttl: Duration,
        max_size: usize,
    ) -> Result<Self, CheungfunError> {
        Ok(UnifiedCache::File(
            FileCache::with_config(cache_dir, default_ttl, max_size).await?,
        ))
    }
}
