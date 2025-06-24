//! In-memory cache implementation.
//!
//! This module provides a simple in-memory cache that implements the PipelineCache trait.
//! It's useful for testing and scenarios where persistence is not required.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::traits::{CacheHealth, CacheStats, HealthStatus, PipelineCache};
use crate::{CheungfunError, Node};

/// In-memory cache implementation.
///
/// This cache stores data in memory and does not persist across application restarts.
/// It's useful for testing and scenarios where you need fast access but don't require
/// persistence.
///
/// # Features
/// - Fast in-memory access
/// - TTL (Time-To-Live) support for automatic expiration
/// - LRU eviction when cache size limits are reached
/// - Statistics tracking for monitoring
/// - Health monitoring
/// - Support for embeddings, nodes, and arbitrary serializable data
///
/// # Example
/// ```rust
/// use cheungfun_core::cache::MemoryCache;
/// use cheungfun_core::traits::PipelineCache;
/// use std::time::Duration;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a memory cache
/// let cache = MemoryCache::new();
///
/// // Cache an embedding
/// let embedding = vec![0.1, 0.2, 0.3, 0.4];
/// cache.put_embedding("test_key", embedding.clone(), Duration::from_secs(3600)).await?;
///
/// // Retrieve the embedding
/// let cached_embedding = cache.get_embedding("test_key").await?;
/// assert_eq!(cached_embedding, Some(embedding));
///
/// // Check cache statistics
/// let stats = cache.stats().await?;
/// println!("Cache hit rate: {:.2}%", stats.hit_rate());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct MemoryCache {
    /// Cache for embeddings
    embedding_cache: Arc<RwLock<HashMap<String, CacheEntry<Vec<f32>>>>>,
    /// Cache for nodes
    nodes_cache: Arc<RwLock<HashMap<String, CacheEntry<Vec<Node>>>>>,
    /// Cache for arbitrary data
    data_cache: Arc<RwLock<HashMap<String, CacheEntry<Vec<u8>>>>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Default TTL for cache entries
    default_ttl: Duration,
    /// Maximum cache size (number of entries)
    max_size: usize,
}

/// A cache entry with TTL information.
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    /// The cached data
    data: T,
    /// When the entry was created
    created_at: Instant,
    /// TTL for the entry
    ttl: Duration,
}

impl<T> CacheEntry<T> {
    /// Create a new cache entry with the given TTL.
    fn new(data: T, ttl: Duration) -> Self {
        Self {
            data,
            created_at: Instant::now(),
            ttl,
        }
    }

    /// Check if the cache entry has expired.
    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }

    /// Get the remaining TTL for this entry.
    fn remaining_ttl(&self) -> Duration {
        if self.is_expired() {
            Duration::ZERO
        } else {
            self.ttl - self.created_at.elapsed()
        }
    }
}

impl MemoryCache {
    /// Create a new memory cache with default configuration.
    pub fn new() -> Self {
        Self {
            embedding_cache: Arc::new(RwLock::new(HashMap::new())),
            nodes_cache: Arc::new(RwLock::new(HashMap::new())),
            data_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            default_ttl: Duration::from_secs(3600), // 1 hour default
            max_size: 10000,                        // Default max 10k entries
        }
    }

    /// Create a new memory cache with custom configuration.
    ///
    /// # Arguments
    /// * `default_ttl` - Default TTL for cache entries
    /// * `max_size` - Maximum number of entries per cache type
    pub fn with_config(default_ttl: Duration, max_size: usize) -> Self {
        Self {
            embedding_cache: Arc::new(RwLock::new(HashMap::new())),
            nodes_cache: Arc::new(RwLock::new(HashMap::new())),
            data_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            default_ttl,
            max_size,
        }
    }

    /// Update cache statistics for a hit.
    async fn record_hit(&self) {
        let mut stats = self.stats.write().await;
        stats.hits += 1;
    }

    /// Update cache statistics for a miss.
    async fn record_miss(&self) {
        let mut stats = self.stats.write().await;
        stats.misses += 1;
    }

    /// Update cache statistics for an eviction.
    async fn record_eviction(&self) {
        let mut stats = self.stats.write().await;
        stats.evictions += 1;
    }

    /// Clean up expired entries from a specific cache.
    async fn cleanup_cache<T>(&self, cache: &Arc<RwLock<HashMap<String, CacheEntry<T>>>>) -> usize {
        let mut cache_guard = cache.write().await;
        let mut expired_keys = Vec::new();

        // Find expired keys
        for (key, entry) in cache_guard.iter() {
            if entry.is_expired() {
                expired_keys.push(key.clone());
            }
        }

        // Remove expired entries
        for key in &expired_keys {
            cache_guard.remove(key);
        }

        let removed_count = expired_keys.len();
        if removed_count > 0 {
            debug!("Cleaned up {} expired entries", removed_count);
        }

        removed_count
    }

    /// Evict least recently used entries if cache is full.
    async fn evict_if_needed<T>(&self, cache: &Arc<RwLock<HashMap<String, CacheEntry<T>>>>) {
        let mut cache_guard = cache.write().await;

        if cache_guard.len() >= self.max_size {
            // Simple eviction: remove oldest entry
            if let Some(oldest_key) = cache_guard
                .iter()
                .min_by_key(|(_, entry)| entry.created_at)
                .map(|(key, _)| key.clone())
            {
                cache_guard.remove(&oldest_key);
                self.record_eviction().await;
                debug!("Evicted cache entry: {}", oldest_key);
            }
        }
    }
}

impl Default for MemoryCache {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineCache for MemoryCache {
    type Error = CheungfunError;

    async fn get_embedding(&self, key: &str) -> std::result::Result<Option<Vec<f32>>, Self::Error> {
        debug!("Getting embedding from memory cache: {}", key);

        let cache = self.embedding_cache.read().await;

        if let Some(entry) = cache.get(key) {
            if entry.is_expired() {
                self.record_miss().await;
                debug!("Embedding cache entry expired: {}", key);
                return Ok(None);
            }

            self.record_hit().await;
            debug!("Embedding cache hit: {}", key);
            return Ok(Some(entry.data.clone()));
        }

        self.record_miss().await;
        debug!("Embedding cache miss: {}", key);
        Ok(None)
    }

    async fn put_embedding(
        &self,
        key: &str,
        embedding: Vec<f32>,
        ttl: Duration,
    ) -> std::result::Result<(), Self::Error> {
        debug!("Storing embedding in memory cache: {}", key);

        self.evict_if_needed(&self.embedding_cache).await;

        let entry = CacheEntry::new(embedding, ttl);
        let mut cache = self.embedding_cache.write().await;

        cache.insert(key.to_string(), entry);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_entries += 1;

        Ok(())
    }

    async fn get_nodes(&self, key: &str) -> std::result::Result<Option<Vec<Node>>, Self::Error> {
        debug!("Getting nodes from memory cache: {}", key);

        let cache = self.nodes_cache.read().await;

        if let Some(entry) = cache.get(key) {
            if entry.is_expired() {
                self.record_miss().await;
                debug!("Nodes cache entry expired: {}", key);
                return Ok(None);
            }

            self.record_hit().await;
            debug!("Nodes cache hit: {}", key);
            return Ok(Some(entry.data.clone()));
        }

        self.record_miss().await;
        debug!("Nodes cache miss: {}", key);
        Ok(None)
    }

    async fn put_nodes(
        &self,
        key: &str,
        nodes: Vec<Node>,
        ttl: Duration,
    ) -> std::result::Result<(), Self::Error> {
        debug!("Storing nodes in memory cache: {}", key);

        self.evict_if_needed(&self.nodes_cache).await;

        let entry = CacheEntry::new(nodes, ttl);
        let mut cache = self.nodes_cache.write().await;

        cache.insert(key.to_string(), entry);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_entries += 1;

        Ok(())
    }

    async fn get_data_bytes(&self, key: &str) -> std::result::Result<Option<Vec<u8>>, Self::Error> {
        debug!("Getting data bytes from memory cache: {}", key);

        let cache = self.data_cache.read().await;

        if let Some(entry) = cache.get(key) {
            if entry.is_expired() {
                self.record_miss().await;
                debug!("Data cache entry expired: {}", key);
                return Ok(None);
            }

            self.record_hit().await;
            debug!("Data cache hit: {}", key);
            Ok(Some(entry.data.clone()))
        } else {
            self.record_miss().await;
            debug!("Data cache miss: {}", key);
            Ok(None)
        }
    }

    async fn put_data_bytes(
        &self,
        key: &str,
        data_bytes: Vec<u8>,
        ttl: Duration,
    ) -> std::result::Result<(), Self::Error> {
        debug!("Storing data bytes in memory cache: {}", key);

        self.evict_if_needed(&self.data_cache).await;

        let entry = CacheEntry::new(data_bytes, ttl);
        let mut cache = self.data_cache.write().await;

        cache.insert(key.to_string(), entry);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_entries += 1;

        Ok(())
    }

    async fn exists(&self, key: &str) -> std::result::Result<bool, Self::Error> {
        // Check all cache types
        let embedding_exists = {
            let cache = self.embedding_cache.read().await;
            cache.get(key).map_or(false, |entry| !entry.is_expired())
        };

        if embedding_exists {
            return Ok(true);
        }

        let nodes_exists = {
            let cache = self.nodes_cache.read().await;
            cache.get(key).map_or(false, |entry| !entry.is_expired())
        };

        if nodes_exists {
            return Ok(true);
        }

        let data_exists = {
            let cache = self.data_cache.read().await;
            cache.get(key).map_or(false, |entry| !entry.is_expired())
        };

        Ok(data_exists)
    }

    async fn remove(&self, key: &str) -> std::result::Result<(), Self::Error> {
        debug!("Removing cache entry: {}", key);

        // Remove from all cache types
        {
            let mut cache = self.embedding_cache.write().await;
            cache.remove(key);
        }

        {
            let mut cache = self.nodes_cache.write().await;
            cache.remove(key);
        }

        {
            let mut cache = self.data_cache.write().await;
            cache.remove(key);
        }

        Ok(())
    }

    async fn clear(&self) -> std::result::Result<(), Self::Error> {
        info!("Clearing all cache entries");

        // Clear all cache types
        {
            let mut cache = self.embedding_cache.write().await;
            cache.clear();
        }

        {
            let mut cache = self.nodes_cache.write().await;
            cache.clear();
        }

        {
            let mut cache = self.data_cache.write().await;
            cache.clear();
        }

        // Reset stats
        {
            let mut stats = self.stats.write().await;
            *stats = CacheStats::default();
        }

        Ok(())
    }

    async fn cleanup(&self) -> std::result::Result<usize, Self::Error> {
        debug!("Cleaning up expired cache entries");

        let embedding_removed = self.cleanup_cache(&self.embedding_cache).await;
        let nodes_removed = self.cleanup_cache(&self.nodes_cache).await;
        let data_removed = self.cleanup_cache(&self.data_cache).await;

        let total_removed = embedding_removed + nodes_removed + data_removed;

        if total_removed > 0 {
            info!("Cleaned up {} expired cache entries", total_removed);

            // Update stats
            let mut stats = self.stats.write().await;
            stats.expired_entries += total_removed;
        }

        Ok(total_removed)
    }

    async fn stats(&self) -> std::result::Result<CacheStats, Self::Error> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }

    async fn health(&self) -> std::result::Result<CacheHealth, Self::Error> {
        let stats = self.stats().await?;

        // Calculate usage ratio
        let total_entries = {
            let embedding_cache = self.embedding_cache.read().await;
            let nodes_cache = self.nodes_cache.read().await;
            let data_cache = self.data_cache.read().await;
            embedding_cache.len() + nodes_cache.len() + data_cache.len()
        };

        let usage_ratio = (total_entries as f64 / (self.max_size * 3) as f64).min(1.0);

        // Determine health status
        let health_status = if usage_ratio > 0.9 {
            HealthStatus::Warning
        } else if usage_ratio > 0.95 {
            HealthStatus::Critical
        } else {
            HealthStatus::Healthy
        };

        let mut messages = Vec::new();
        if usage_ratio > 0.8 {
            messages.push(format!("Cache usage is high: {:.1}%", usage_ratio * 100.0));
        }

        let total_ops = stats.total_operations();
        if stats.hit_rate() < 50.0 && total_ops > 100 {
            messages.push(format!("Low cache hit rate: {:.1}%", stats.hit_rate()));
        }

        Ok(CacheHealth {
            status: health_status,
            usage_ratio,
            hit_rate: stats.hit_rate(),
            total_entries,
            estimated_size_mb: 0.0, // Simplified for memory cache
            messages,
        })
    }
}
