//! Caching mechanisms for API embedders.

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::error::Result;

/// Cache key for embeddings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Model name
    pub model: String,
    /// Text content hash
    pub text_hash: String,
}

impl CacheKey {
    /// Create a new cache key from model and text.
    pub fn new<S: AsRef<str>>(model: S, text: S) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(text.as_ref().as_bytes());
        let text_hash = format!("{:x}", hasher.finalize());

        Self {
            model: model.as_ref().to_string(),
            text_hash,
        }
    }
}

/// Cached embedding entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// When this entry was created
    pub created_at: Instant,
    /// Time to live for this entry
    pub ttl: Duration,
}

impl CacheEntry {
    /// Create a new cache entry.
    pub fn new(embedding: Vec<f32>, ttl: Duration) -> Self {
        Self {
            embedding,
            created_at: Instant::now(),
            ttl,
        }
    }

    /// Check if this entry has expired.
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

/// Trait for embedding caches.
#[async_trait]
pub trait EmbeddingCache: Send + Sync + std::fmt::Debug {
    /// Get an embedding from the cache.
    async fn get(&self, key: &CacheKey) -> Result<Option<Vec<f32>>>;

    /// Store an embedding in the cache.
    async fn put(&self, key: CacheKey, embedding: Vec<f32>, ttl: Duration) -> Result<()>;

    /// Get multiple embeddings from the cache.
    async fn get_batch(&self, keys: &[CacheKey]) -> Result<Vec<Option<Vec<f32>>>> {
        let mut results = Vec::with_capacity(keys.len());
        for key in keys {
            results.push(self.get(key).await?);
        }
        Ok(results)
    }

    /// Store multiple embeddings in the cache.
    async fn put_batch(&self, entries: Vec<(CacheKey, Vec<f32>)>, ttl: Duration) -> Result<()> {
        for (key, embedding) in entries {
            self.put(key, embedding, ttl).await?;
        }
        Ok(())
    }

    /// Clear expired entries from the cache.
    async fn cleanup(&self) -> Result<usize>;

    /// Get cache statistics.
    async fn stats(&self) -> CacheStats;

    /// Clear all entries from the cache.
    async fn clear(&self) -> Result<()>;
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of entries
    pub total_entries: usize,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of expired entries
    pub expired_entries: usize,
}

impl CacheStats {
    /// Calculate hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            (self.hits as f64) / ((self.hits + self.misses) as f64) * 100.0
        }
    }
}

/// In-memory embedding cache implementation.
#[derive(Debug)]
pub struct InMemoryCache {
    /// Cache storage
    cache: Arc<RwLock<HashMap<CacheKey, CacheEntry>>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
}

impl InMemoryCache {
    /// Create a new in-memory cache.
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Create a new in-memory cache with initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::with_capacity(capacity))),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }
}

impl Default for InMemoryCache {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EmbeddingCache for InMemoryCache {
    async fn get(&self, key: &CacheKey) -> Result<Option<Vec<f32>>> {
        let cache = self.cache.read().await;
        let mut stats = self.stats.write().await;

        if let Some(entry) = cache.get(key) {
            if entry.is_expired() {
                stats.misses += 1;
                Ok(None)
            } else {
                stats.hits += 1;
                Ok(Some(entry.embedding.clone()))
            }
        } else {
            stats.misses += 1;
            Ok(None)
        }
    }

    async fn put(&self, key: CacheKey, embedding: Vec<f32>, ttl: Duration) -> Result<()> {
        let mut cache = self.cache.write().await;
        let entry = CacheEntry::new(embedding, ttl);
        cache.insert(key, entry);
        Ok(())
    }

    async fn cleanup(&self) -> Result<usize> {
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;

        let initial_count = cache.len();
        cache.retain(|_, entry| !entry.is_expired());
        let final_count = cache.len();
        let removed_count = initial_count - final_count;

        stats.expired_entries += removed_count;
        stats.total_entries = final_count;

        Ok(removed_count)
    }

    async fn stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        let mut stats = self.stats.read().await.clone();
        stats.total_entries = cache.len();
        stats
    }

    async fn clear(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;

        cache.clear();
        stats.total_entries = 0;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{Duration, sleep};

    #[tokio::test]
    async fn test_cache_key_creation() {
        let key1 = CacheKey::new("model1", "hello world");
        let key2 = CacheKey::new("model1", "hello world");
        let key3 = CacheKey::new("model1", "different text");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[tokio::test]
    async fn test_cache_entry_expiration() {
        let entry = CacheEntry::new(vec![1.0, 2.0, 3.0], Duration::from_millis(10));
        assert!(!entry.is_expired());

        sleep(Duration::from_millis(20)).await;
        assert!(entry.is_expired());
    }

    #[tokio::test]
    async fn test_in_memory_cache() {
        let cache = InMemoryCache::new();
        let key = CacheKey::new("test-model", "test text");
        let embedding = vec![1.0, 2.0, 3.0];

        // Test miss
        assert!(cache.get(&key).await.unwrap().is_none());

        // Test put and hit
        cache
            .put(key.clone(), embedding.clone(), Duration::from_secs(60))
            .await
            .unwrap();
        let result = cache.get(&key).await.unwrap();
        assert_eq!(result, Some(embedding));

        // Test stats
        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total_entries, 1);
    }
}
