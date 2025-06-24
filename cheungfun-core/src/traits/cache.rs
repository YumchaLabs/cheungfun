//! Cache traits for the Cheungfun framework.
//!
//! This module defines unified caching interfaces that can be used across
//! different components of the RAG pipeline, including embeddings, query results,
//! and processed nodes.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::Node;

/// Unified cache interface for pipeline components.
///
/// This trait provides a consistent caching interface that can be used
/// across different stages of the RAG pipeline, supporting both embeddings
/// and structured data caching with TTL and statistics.
#[async_trait]
pub trait PipelineCache: Send + Sync + std::fmt::Debug {
    /// Error type for cache operations.
    type Error: Send + Sync + std::error::Error + 'static;

    /// Get a cached embedding vector.
    ///
    /// # Arguments
    /// * `key` - The cache key for the embedding
    ///
    /// # Returns
    /// * `Ok(Some(embedding))` - If the embedding is found and not expired
    /// * `Ok(None)` - If the embedding is not found or expired
    /// * `Err(error)` - If there was an error accessing the cache
    async fn get_embedding(&self, key: &str) -> std::result::Result<Option<Vec<f32>>, Self::Error>;

    /// Store an embedding vector in the cache.
    ///
    /// # Arguments
    /// * `key` - The cache key for the embedding
    /// * `embedding` - The embedding vector to cache
    /// * `ttl` - Time-to-live for the cache entry
    async fn put_embedding(
        &self,
        key: &str,
        embedding: Vec<f32>,
        ttl: Duration,
    ) -> std::result::Result<(), Self::Error>;

    /// Get cached nodes (processed document chunks).
    ///
    /// # Arguments
    /// * `key` - The cache key for the nodes
    ///
    /// # Returns
    /// * `Ok(Some(nodes))` - If the nodes are found and not expired
    /// * `Ok(None)` - If the nodes are not found or expired
    /// * `Err(error)` - If there was an error accessing the cache
    async fn get_nodes(&self, key: &str) -> std::result::Result<Option<Vec<Node>>, Self::Error>;

    /// Store processed nodes in the cache.
    ///
    /// # Arguments
    /// * `key` - The cache key for the nodes
    /// * `nodes` - The nodes to cache
    /// * `ttl` - Time-to-live for the cache entry
    async fn put_nodes(
        &self,
        key: &str,
        nodes: Vec<Node>,
        ttl: Duration,
    ) -> std::result::Result<(), Self::Error>;

    /// Get arbitrary serializable data from the cache as bytes.
    ///
    /// This method returns raw bytes that can be deserialized by the caller.
    ///
    /// # Arguments
    /// * `key` - The cache key
    ///
    /// # Returns
    /// * `Ok(Some(bytes))` - If the data is found and not expired
    /// * `Ok(None)` - If the data is not found or expired
    /// * `Err(error)` - If there was an error accessing the cache
    async fn get_data_bytes(&self, key: &str) -> std::result::Result<Option<Vec<u8>>, Self::Error>;

    /// Store arbitrary serializable data in the cache as bytes.
    ///
    /// # Arguments
    /// * `key` - The cache key
    /// * `data_bytes` - The serialized data to cache
    /// * `ttl` - Time-to-live for the cache entry
    async fn put_data_bytes(
        &self,
        key: &str,
        data_bytes: Vec<u8>,
        ttl: Duration,
    ) -> std::result::Result<(), Self::Error>;

    /// Check if a cache entry exists and is not expired.
    ///
    /// # Arguments
    /// * `key` - The cache key to check
    ///
    /// # Returns
    /// * `Ok(true)` - If the entry exists and is not expired
    /// * `Ok(false)` - If the entry doesn't exist or is expired
    /// * `Err(error)` - If there was an error accessing the cache
    async fn exists(&self, key: &str) -> std::result::Result<bool, Self::Error>;

    /// Remove a specific cache entry.
    ///
    /// # Arguments
    /// * `key` - The cache key to remove
    async fn remove(&self, key: &str) -> std::result::Result<(), Self::Error>;

    /// Clear all cache entries.
    async fn clear(&self) -> std::result::Result<(), Self::Error>;

    /// Clean up expired entries and return the number of entries removed.
    ///
    /// # Returns
    /// * `Ok(count)` - Number of expired entries removed
    /// * `Err(error)` - If there was an error during cleanup
    async fn cleanup(&self) -> std::result::Result<usize, Self::Error>;

    /// Get cache statistics.
    ///
    /// # Returns
    /// * `Ok(stats)` - Current cache statistics
    /// * `Err(error)` - If there was an error getting statistics
    async fn stats(&self) -> std::result::Result<CacheStats, Self::Error>;

    /// Get cache health status.
    ///
    /// # Returns
    /// * `Ok(health)` - Current cache health status
    /// * `Err(error)` - If there was an error getting health status
    async fn health(&self) -> std::result::Result<CacheHealth, Self::Error>;
}

/// Extension trait for PipelineCache that provides convenient generic methods.
///
/// This trait is automatically implemented for all types that implement PipelineCache
/// and provides type-safe serialization/deserialization methods.
#[async_trait]
pub trait PipelineCacheExt: PipelineCache {
    /// Get arbitrary serializable data from the cache.
    ///
    /// This is a convenience method that handles serialization automatically.
    ///
    /// # Arguments
    /// * `key` - The cache key
    ///
    /// # Returns
    /// * `Ok(Some(data))` - If the data is found and not expired
    /// * `Ok(None)` - If the data is not found or expired
    /// * `Err(error)` - If there was an error accessing the cache or deserializing
    async fn get_data<T>(&self, key: &str) -> std::result::Result<Option<T>, Self::Error>
    where
        T: for<'de> Deserialize<'de> + Send,
        Self::Error: From<bincode::Error>,
    {
        match self.get_data_bytes(key).await? {
            Some(bytes) => match bincode::deserialize(&bytes) {
                Ok(data) => Ok(Some(data)),
                Err(e) => Err(Self::Error::from(e)),
            },
            None => Ok(None),
        }
    }

    /// Store arbitrary serializable data in the cache.
    ///
    /// This is a convenience method that handles serialization automatically.
    ///
    /// # Arguments
    /// * `key` - The cache key
    /// * `data` - The data to cache
    /// * `ttl` - Time-to-live for the cache entry
    async fn put_data<T>(
        &self,
        key: &str,
        data: &T,
        ttl: Duration,
    ) -> std::result::Result<(), Self::Error>
    where
        T: Serialize + Send + Sync,
        Self::Error: From<bincode::Error>,
    {
        let bytes = bincode::serialize(data).map_err(Self::Error::from)?;
        self.put_data_bytes(key, bytes, ttl).await
    }
}

/// Blanket implementation of PipelineCacheExt for all PipelineCache implementors.
impl<T: PipelineCache> PipelineCacheExt for T {}

/// Cache statistics for monitoring and debugging.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total number of cache entries
    pub total_entries: usize,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of expired entries
    pub expired_entries: usize,
    /// Number of evicted entries (due to size limits)
    pub evictions: u64,
    /// Estimated cache size in bytes
    pub estimated_size_bytes: u64,
}

impl CacheStats {
    /// Calculate the cache hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            (self.hits as f64 / (self.hits + self.misses) as f64) * 100.0
        }
    }

    /// Calculate the cache miss rate as a percentage.
    pub fn miss_rate(&self) -> f64 {
        100.0 - self.hit_rate()
    }

    /// Get the total number of cache operations.
    pub fn total_operations(&self) -> u64 {
        self.hits + self.misses
    }
}

/// Cache health status for monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheHealth {
    /// Overall health status
    pub status: HealthStatus,
    /// Cache usage ratio (0.0 to 1.0)
    pub usage_ratio: f64,
    /// Current hit rate percentage
    pub hit_rate: f64,
    /// Total number of entries
    pub total_entries: usize,
    /// Estimated size in megabytes
    pub estimated_size_mb: f64,
    /// Any warning or error messages
    pub messages: Vec<String>,
}

/// Health status levels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Cache is operating normally
    Healthy,
    /// Cache has some issues but is still functional
    Warning,
    /// Cache has critical issues
    Critical,
}

/// Cache key generation utilities.
pub struct CacheKeyGenerator;

impl CacheKeyGenerator {
    /// Generate a cache key for embeddings.
    ///
    /// # Arguments
    /// * `text` - The text content
    /// * `model_name` - The embedding model name
    /// * `model_version` - The model version (optional)
    ///
    /// # Returns
    /// A deterministic cache key based on the input parameters
    pub fn embedding_key(text: &str, model_name: &str, model_version: Option<&str>) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        model_name.hash(&mut hasher);
        if let Some(version) = model_version {
            version.hash(&mut hasher);
        }

        format!("embedding:{:x}", hasher.finish())
    }

    /// Generate a cache key for processed nodes.
    ///
    /// # Arguments
    /// * `document_id` - The document identifier
    /// * `chunk_size` - The chunk size used for splitting
    /// * `overlap` - The overlap size used for splitting
    ///
    /// # Returns
    /// A deterministic cache key based on the input parameters
    pub fn nodes_key(document_id: &str, chunk_size: usize, overlap: usize) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        document_id.hash(&mut hasher);
        chunk_size.hash(&mut hasher);
        overlap.hash(&mut hasher);

        format!("nodes:{:x}", hasher.finish())
    }

    /// Generate a cache key for query results.
    ///
    /// # Arguments
    /// * `query` - The query text
    /// * `parameters` - Additional query parameters
    ///
    /// # Returns
    /// A deterministic cache key based on the input parameters
    pub fn query_key(query: &str, parameters: &HashMap<String, String>) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);

        // Sort parameters for consistent hashing
        let mut sorted_params: Vec<_> = parameters.iter().collect();
        sorted_params.sort_by_key(|(k, _)| *k);
        for (key, value) in sorted_params {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }

        format!("query:{:x}", hasher.finish())
    }

    /// Generate a generic cache key for arbitrary data.
    ///
    /// # Arguments
    /// * `prefix` - A prefix to categorize the cache entry
    /// * `data` - The data to generate a key for (must be serializable)
    ///
    /// # Returns
    /// A deterministic cache key based on the serialized data
    pub fn generic_key<T: Serialize>(
        prefix: &str,
        data: &T,
    ) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let serialized = bincode::serialize(data)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
        let mut hasher = DefaultHasher::new();
        serialized.hash(&mut hasher);

        Ok(format!("{}:{:x}", prefix, hasher.finish()))
    }
}
