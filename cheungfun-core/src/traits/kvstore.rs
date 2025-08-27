//! Key-Value store traits for the Cheungfun framework.
//!
//! This module defines the core KVStore trait that provides a unified interface
//! for different key-value storage backends. The design follows LlamaIndex's
//! KVStore pattern, providing collection-based organization and async operations.

use crate::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

/// Default collection name for KV operations.
pub const DEFAULT_COLLECTION: &str = "default";

/// Default batch size for bulk operations.
pub const DEFAULT_BATCH_SIZE: usize = 100;

/// Key-Value store trait for unified storage operations.
///
/// This trait provides a unified interface for different key-value storage
/// backends, allowing the framework to work with various implementations
/// like SQLx, MongoDB, Redis, or in-memory stores.
///
/// The design follows LlamaIndex's KVStore pattern with collection-based
/// organization, where each collection acts as a namespace for keys.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::{KVStore, DEFAULT_COLLECTION};
/// use serde_json::json;
/// use async_trait::async_trait;
///
/// struct InMemoryKVStore {
///     data: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, std::collections::HashMap<String, serde_json::Value>>>>,
/// }
///
/// #[async_trait]
/// impl KVStore for InMemoryKVStore {
///     async fn put(&self, key: &str, value: serde_json::Value, collection: &str) -> cheungfun_core::Result<()> {
///         let mut data = self.data.write().await;
///         data.entry(collection.to_string())
///             .or_insert_with(std::collections::HashMap::new)
///             .insert(key.to_string(), value);
///         Ok(())
///     }
///
///     async fn get(&self, key: &str, collection: &str) -> cheungfun_core::Result<Option<serde_json::Value>> {
///         let data = self.data.read().await;
///         Ok(data.get(collection).and_then(|coll| coll.get(key)).cloned())
///     }
///
///     // ... other methods
/// #   async fn delete(&self, key: &str, collection: &str) -> cheungfun_core::Result<bool> { Ok(false) }
/// #   async fn get_all(&self, collection: &str) -> cheungfun_core::Result<std::collections::HashMap<String, serde_json::Value>> { Ok(std::collections::HashMap::new()) }
/// #   async fn list_collections(&self) -> cheungfun_core::Result<Vec<String>> { Ok(vec![]) }
/// #   async fn delete_collection(&self, collection: &str) -> cheungfun_core::Result<()> { Ok(()) }
/// }
/// ```
#[async_trait]
pub trait KVStore: Send + Sync + std::fmt::Debug {
    /// Put a key-value pair in the specified collection.
    ///
    /// This method stores the provided value under the given key in the
    /// specified collection. If the key already exists, it will be updated.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to store the value under
    /// * `value` - The JSON value to store
    /// * `collection` - The collection (namespace) to store in
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails due to connection
    /// issues, serialization problems, or storage capacity limits.
    async fn put(&self, key: &str, value: Value, collection: &str) -> Result<()>;

    /// Get a value by key from the specified collection.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to retrieve
    /// * `collection` - The collection to search in
    ///
    /// # Returns
    ///
    /// Returns `Some(value)` if the key exists, `None` otherwise.
    async fn get(&self, key: &str, collection: &str) -> Result<Option<Value>>;

    /// Delete a key from the specified collection.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to delete
    /// * `collection` - The collection to delete from
    ///
    /// # Returns
    ///
    /// Returns `true` if the key was deleted, `false` if it didn't exist.
    async fn delete(&self, key: &str, collection: &str) -> Result<bool>;

    /// Get all key-value pairs from a collection.
    ///
    /// # Arguments
    ///
    /// * `collection` - The collection to retrieve all data from
    ///
    /// # Returns
    ///
    /// A HashMap containing all key-value pairs in the collection.
    ///
    /// # Warning
    ///
    /// This method can be expensive for large collections. Use with caution
    /// in production environments.
    async fn get_all(&self, collection: &str) -> Result<HashMap<String, Value>>;

    /// Put multiple key-value pairs in a collection (batch operation).
    ///
    /// This method provides an optimized way to store multiple key-value
    /// pairs at once. Implementations should provide transaction-like
    /// behavior where possible.
    ///
    /// # Arguments
    ///
    /// * `kv_pairs` - Vector of (key, value) tuples to store
    /// * `collection` - The collection to store in
    ///
    /// # Default Implementation
    ///
    /// The default implementation calls `put` for each pair sequentially.
    /// Implementations should override this for better performance.
    async fn put_all(&self, kv_pairs: Vec<(String, Value)>, collection: &str) -> Result<()> {
        for (key, value) in kv_pairs {
            self.put(&key, value, collection).await?;
        }
        Ok(())
    }

    /// List all collections in the store.
    ///
    /// # Returns
    ///
    /// A vector of collection names.
    async fn list_collections(&self) -> Result<Vec<String>>;

    /// Delete an entire collection and all its data.
    ///
    /// # Arguments
    ///
    /// * `collection` - The collection to delete
    ///
    /// # Warning
    ///
    /// This operation is destructive and cannot be undone.
    async fn delete_collection(&self, collection: &str) -> Result<()>;

    /// Check if a key exists in a collection.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to check
    /// * `collection` - The collection to search in
    ///
    /// # Default Implementation
    ///
    /// The default implementation calls `get` and checks if the result is `Some`.
    /// Implementations may override this for better performance.
    async fn exists(&self, key: &str, collection: &str) -> Result<bool> {
        Ok(self.get(key, collection).await?.is_some())
    }

    /// Get the number of items in a collection.
    ///
    /// # Arguments
    ///
    /// * `collection` - The collection to count
    ///
    /// # Default Implementation
    ///
    /// The default implementation calls `get_all` and returns the length.
    /// Implementations should override this for better performance.
    async fn count(&self, collection: &str) -> Result<usize> {
        Ok(self.get_all(collection).await?.len())
    }

    /// Get a human-readable name for this KV store.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Batch operation helper for KVStore implementations.
///
/// This struct provides utilities for batching operations efficiently.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of items per batch.
    pub batch_size: usize,

    /// Whether to fail fast on first error or collect all errors.
    pub fail_fast: bool,

    /// Maximum number of concurrent batches.
    pub max_concurrency: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            fail_fast: true,
            max_concurrency: 4,
        }
    }
}

/// Statistics about KVStore operations.
#[derive(Debug, Clone, Default)]
pub struct KVStoreStats {
    /// Total number of put operations.
    pub put_operations: usize,

    /// Total number of get operations.
    pub get_operations: usize,

    /// Total number of delete operations.
    pub delete_operations: usize,

    /// Total number of batch operations.
    pub batch_operations: usize,

    /// Average operation latency in milliseconds.
    pub avg_latency_ms: Option<f64>,

    /// Total storage size in bytes (if available).
    pub storage_size_bytes: Option<u64>,

    /// Additional store-specific statistics.
    pub additional_stats: HashMap<String, Value>,
}

impl KVStoreStats {
    /// Create new KV store statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the total number of operations.
    #[must_use]
    pub fn total_operations(&self) -> usize {
        self.put_operations + self.get_operations + self.delete_operations + self.batch_operations
    }

    /// Calculate operations per second given a duration.
    #[must_use]
    pub fn operations_per_second(&self, duration: std::time::Duration) -> f64 {
        if duration.is_zero() {
            0.0
        } else {
            self.total_operations() as f64 / duration.as_secs_f64()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.batch_size, DEFAULT_BATCH_SIZE);
        assert!(config.fail_fast);
        assert_eq!(config.max_concurrency, 4);
    }

    #[test]
    fn test_kvstore_stats() {
        let mut stats = KVStoreStats::new();
        stats.put_operations = 100;
        stats.get_operations = 200;
        stats.delete_operations = 50;
        stats.batch_operations = 10;

        assert_eq!(stats.total_operations(), 360);

        let duration = std::time::Duration::from_secs(10);
        assert_eq!(stats.operations_per_second(duration), 36.0);
    }
}
