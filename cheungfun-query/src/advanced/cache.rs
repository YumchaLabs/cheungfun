// Query Cache Implementation

use super::{Result, RetrievalResponse};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// A cache for query results.
#[derive(Debug)]
pub struct QueryCache {
    /// The cache storage.
    cache: Arc<RwLock<HashMap<String, CachedResult>>>,
    /// Cache Time-to-Live (TTL).
    ttl: Duration,
    /// Maximum cache size.
    max_size: usize,
    /// Cache statistics.
    stats: Arc<RwLock<CacheStats>>,
}

/// A cached entry.
#[derive(Debug, Clone)]
struct CachedResult {
    /// The cached response.
    response: RetrievalResponse,
    /// Timestamp of when the item was cached.
    timestamp: Instant,
    /// Access count.
    access_count: usize,
    /// Last access time.
    last_accessed: Instant,
}

/// Cache statistics information.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: usize,
    /// Number of cache misses.
    pub misses: usize,
    /// Number of entries in the cache.
    pub entries: usize,
    /// Number of evictions due to expiration or size limits.
    pub evictions: usize,
    /// Total estimated storage size in bytes.
    pub estimated_size_bytes: usize,
}

impl CacheStats {
    /// Calculates the hit rate.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl QueryCache {
    /// Creates a new query cache.
    #[must_use]
    pub fn new(ttl: Duration, max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            ttl,
            max_size,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Gets a cached result.
    pub async fn get(&self, query_hash: &str) -> Option<RetrievalResponse> {
        debug!("Cache lookup for query hash: {}", query_hash);

        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;

        if let Some(cached) = cache.get_mut(query_hash) {
            // Check if expired
            if cached.timestamp.elapsed() < self.ttl {
                // Update access stats
                cached.access_count += 1;
                cached.last_accessed = Instant::now();

                stats.hits += 1;
                info!("Cache hit for query hash: {}", query_hash);
                return Some(cached.response.clone());
            }
            // Expired, remove it
            cache.remove(query_hash);
            stats.evictions += 1;
            debug!("Cache entry expired for query hash: {}", query_hash);
        }

        stats.misses += 1;
        debug!("Cache miss for query hash: {}", query_hash);
        None
    }

    /// Stores a result in the cache.
    pub async fn put(&self, query_hash: String, response: RetrievalResponse) {
        debug!("Caching result for query hash: {}", query_hash);

        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;

        // Clean up expired entries
        self.cleanup_expired(&mut cache, &mut stats).await;

        // If the cache is full, evict the least used entry
        if cache.len() >= self.max_size {
            self.evict_lru(&mut cache, &mut stats).await;
        }

        // Add the new entry
        let cached_result = CachedResult {
            response,
            timestamp: Instant::now(),
            access_count: 0,
            last_accessed: Instant::now(),
        };

        cache.insert(query_hash.clone(), cached_result);
        stats.entries = cache.len();

        // Update estimated size
        stats.estimated_size_bytes = self.estimate_cache_size(&cache);

        info!("Cached result for query hash: {}", query_hash);
    }

    /// Cleans up expired entries.
    async fn cleanup_expired(
        &self,
        cache: &mut HashMap<String, CachedResult>,
        stats: &mut CacheStats,
    ) {
        let expired_keys: Vec<_> = cache
            .iter()
            .filter(|(_, cached)| cached.timestamp.elapsed() >= self.ttl)
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            cache.remove(&key);
            stats.evictions += 1;
        }

        if !cache.is_empty() {
            debug!("Cleaned up expired cache entries");
        }
    }

    /// Evicts the least used entry (LFU with LRU as tie-breaker).
    async fn evict_lru(&self, cache: &mut HashMap<String, CachedResult>, stats: &mut CacheStats) {
        if let Some(lru_key) = cache
            .iter()
            .min_by_key(|(_, cached)| (cached.access_count, cached.last_accessed))
            .map(|(key, _)| key.clone())
        {
            cache.remove(&lru_key);
            stats.evictions += 1;
            debug!("Evicted least used cache entry: {}", lru_key);
        }
    }

    /// Estimates the cache size.
    fn estimate_cache_size(&self, cache: &HashMap<String, CachedResult>) -> usize {
        cache
            .iter()
            .map(|(key, cached)| {
                key.len() + cached.response.nodes.len() * 1000 + // Estimate 1KB per node
            cached.response.query.original_text.len() +
            cached.response.query.transformed_queries.iter().map(std::string::String::len).sum::<usize>()
            })
            .sum()
    }

    /// Gets the cache statistics.
    pub async fn get_stats(&self) -> CacheStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Clears the cache.
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;

        cache.clear();
        stats.entries = 0;
        stats.estimated_size_bytes = 0;

        info!("Cache cleared");
    }

    /// Gets the cache size.
    pub async fn size(&self) -> usize {
        let cache = self.cache.read().await;
        cache.len()
    }

    /// Checks the cache health status.
    pub async fn health_check(&self) -> CacheHealthStatus {
        let cache = self.cache.read().await;
        let stats = self.stats.read().await;

        let usage_ratio = cache.len() as f64 / self.max_size as f64;
        let hit_rate = stats.hit_rate();

        let status = if usage_ratio > 0.9 {
            CacheHealth::Warning
        } else if hit_rate < 0.1 && stats.hits + stats.misses > 100 {
            CacheHealth::Warning
        } else {
            CacheHealth::Healthy
        };

        CacheHealthStatus {
            status,
            usage_ratio,
            hit_rate,
            total_entries: cache.len(),
            estimated_size_mb: stats.estimated_size_bytes as f64 / 1024.0 / 1024.0,
        }
    }
}

/// Cache health status.
#[derive(Debug, Clone)]
pub struct CacheHealthStatus {
    pub status: CacheHealth,
    pub usage_ratio: f64,
    pub hit_rate: f64,
    pub total_entries: usize,
    pub estimated_size_mb: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CacheHealth {
    Healthy,
    Warning,
    Critical,
}

/// Tiered Cache Implementation
///
/// Supports multiple cache levels, e.g., in-memory cache + Redis cache.
#[derive(Debug)]
pub struct TieredQueryCache {
    /// L1 cache (in-memory).
    l1_cache: QueryCache,
    /// L2 cache (optional, e.g., Redis).
    l2_cache: Option<Arc<dyn ExternalCache>>,
    /// Configuration.
    config: TieredCacheConfig,
}

/// Tiered cache configuration.
#[derive(Debug, Clone)]
pub struct TieredCacheConfig {
    /// L1 cache max size.
    pub l1_max_size: usize,
    /// L1 cache TTL.
    pub l1_ttl: Duration,
    /// L2 cache TTL.
    pub l2_ttl: Duration,
    /// Whether L2 cache is enabled.
    pub enable_l2: bool,
}

impl Default for TieredCacheConfig {
    fn default() -> Self {
        Self {
            l1_max_size: 1000,
            l1_ttl: Duration::from_secs(300),  // 5 minutes
            l2_ttl: Duration::from_secs(3600), // 1 hour
            enable_l2: false,
        }
    }
}

impl TieredQueryCache {
    /// Creates a new tiered cache.
    #[must_use]
    pub fn new(config: TieredCacheConfig) -> Self {
        Self {
            l1_cache: QueryCache::new(config.l1_ttl, config.l1_max_size),
            l2_cache: None,
            config,
        }
    }

    /// Sets the L2 cache.
    pub fn with_l2_cache(mut self, l2_cache: Arc<dyn ExternalCache>) -> Self {
        self.l2_cache = Some(l2_cache);
        self
    }

    /// Gets a cached result.
    pub async fn get(&self, query_hash: &str) -> Option<RetrievalResponse> {
        // First, check the L1 cache
        if let Some(result) = self.l1_cache.get(query_hash).await {
            return Some(result);
        }

        // Then, check the L2 cache
        if self.config.enable_l2 {
            if let Some(l2_cache) = &self.l2_cache {
                if let Ok(Some(result)) = l2_cache.get(query_hash).await {
                    // Put the result into the L1 cache
                    self.l1_cache
                        .put(query_hash.to_string(), result.clone())
                        .await;
                    return Some(result);
                }
            }
        }

        None
    }

    /// Stores a result in the cache.
    pub async fn put(&self, query_hash: String, response: RetrievalResponse) {
        // Store in L1 cache
        self.l1_cache
            .put(query_hash.clone(), response.clone())
            .await;

        // Store in L2 cache
        if self.config.enable_l2 {
            if let Some(l2_cache) = &self.l2_cache {
                if let Err(e) = l2_cache
                    .put(&query_hash, &response, self.config.l2_ttl)
                    .await
                {
                    warn!("Failed to store to L2 cache: {}", e);
                }
            }
        }
    }
}

/// Trait for an external cache (e.g., Redis).
#[async_trait::async_trait]
pub trait ExternalCache: Send + Sync + std::fmt::Debug {
    /// Gets a cached result.
    async fn get(&self, key: &str) -> Result<Option<RetrievalResponse>>;

    /// Stores a result in the cache.
    async fn put(&self, key: &str, value: &RetrievalResponse, ttl: Duration) -> Result<()>;

    /// Deletes a cache entry.
    async fn delete(&self, key: &str) -> Result<()>;

    /// Clears the cache.
    async fn clear(&self) -> Result<()>;

    /// Checks the connection status.
    async fn ping(&self) -> Result<()>;
}

/// Example implementation of a Redis cache.
#[derive(Debug)]
pub struct RedisCache {
    // Redis client connection
    // client: redis::Client,
    // This is just a sample struct; the actual implementation requires a Redis dependency.
}

#[async_trait::async_trait]
impl ExternalCache for RedisCache {
    async fn get(&self, _key: &str) -> Result<Option<RetrievalResponse>> {
        // TODO: Implement Redis GET operation
        // Requires serialization/deserialization of RetrievalResponse
        todo!("Implement Redis GET")
    }

    async fn put(&self, _key: &str, _value: &RetrievalResponse, _ttl: Duration) -> Result<()> {
        // TODO: Implement Redis SET operation
        todo!("Implement Redis SET")
    }

    async fn delete(&self, _key: &str) -> Result<()> {
        // TODO: Implement Redis DEL operation
        todo!("Implement Redis DEL")
    }

    async fn clear(&self) -> Result<()> {
        // TODO: Implement Redis FLUSHDB operation
        todo!("Implement Redis FLUSHDB")
    }

    async fn ping(&self) -> Result<()> {
        // TODO: Implement Redis PING operation
        todo!("Implement Redis PING")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_query_cache_basic() {
        let cache = QueryCache::new(Duration::from_secs(60), 10);

        // Create a test response
        let response = RetrievalResponse {
            nodes: vec![],
            query: crate::advanced::AdvancedQuery::from_text("test query"),
            metadata: std::collections::HashMap::new(),
            stats: crate::advanced::RetrievalStats::default(),
        };

        // Test storing and getting
        cache.put("test_key".to_string(), response.clone()).await;
        let retrieved = cache.get("test_key").await;

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().query.original_text, "test query");

        // Test statistics
        let stats = cache.get_stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let cache = QueryCache::new(Duration::from_millis(100), 10);

        let response = RetrievalResponse {
            nodes: vec![],
            query: crate::advanced::AdvancedQuery::from_text("test query"),
            metadata: std::collections::HashMap::new(),
            stats: crate::advanced::RetrievalStats::default(),
        };

        cache.put("test_key".to_string(), response).await;

        // Immediate get should succeed
        assert!(cache.get("test_key").await.is_some());

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should return None after expiration
        assert!(cache.get("test_key").await.is_none());
    }
}
