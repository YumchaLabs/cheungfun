//! In-memory KVStore implementation.

use async_trait::async_trait;
use cheungfun_core::{traits::KVStore, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

/// In-memory key-value store implementation.
///
/// This store keeps all data in memory using a HashMap structure.
/// It's useful for testing, development, and scenarios where persistence
/// is not required. The store is thread-safe and supports concurrent access.
///
/// # Examples
///
/// ```rust
/// use cheungfun_integrations::storage::kvstore::InMemoryKVStore;
/// use cheungfun_core::traits::{KVStore, DEFAULT_COLLECTION};
/// use serde_json::json;
///
/// # tokio_test::block_on(async {
/// let store = InMemoryKVStore::new();
/// 
/// // Store a value
/// store.put("key1", json!({"name": "test"}), DEFAULT_COLLECTION).await.unwrap();
/// 
/// // Retrieve the value
/// let value = store.get("key1", DEFAULT_COLLECTION).await.unwrap();
/// assert!(value.is_some());
/// # });
/// ```
#[derive(Debug)]
pub struct InMemoryKVStore {
    /// Collections storage: collection_name -> (key -> value)
    collections: Arc<RwLock<HashMap<String, HashMap<String, Value>>>>,
}

impl InMemoryKVStore {
    /// Create a new in-memory KV store.
    pub fn new() -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new in-memory KV store with initial data.
    ///
    /// # Arguments
    ///
    /// * `initial_data` - Initial collections and their data
    pub fn with_data(initial_data: HashMap<String, HashMap<String, Value>>) -> Self {
        Self {
            collections: Arc::new(RwLock::new(initial_data)),
        }
    }

    /// Get a snapshot of all data in the store.
    ///
    /// This method returns a clone of all data, which can be expensive
    /// for large datasets but is useful for debugging and testing.
    pub async fn snapshot(&self) -> HashMap<String, HashMap<String, Value>> {
        self.collections.read().await.clone()
    }

    /// Clear all data from the store.
    pub async fn clear(&self) {
        self.collections.write().await.clear();
        debug!("Cleared all data from in-memory KV store");
    }

    /// Get memory usage statistics.
    pub async fn memory_stats(&self) -> MemoryStats {
        let collections = self.collections.read().await;
        let mut stats = MemoryStats::default();
        
        stats.collection_count = collections.len();
        
        for (collection_name, collection_data) in collections.iter() {
            stats.total_keys += collection_data.len();
            
            // Estimate memory usage (rough approximation)
            let collection_size = collection_name.len() * std::mem::size_of::<char>()
                + collection_data.iter().map(|(k, v)| {
                    k.len() * std::mem::size_of::<char>() + estimate_value_size(v)
                }).sum::<usize>();
            
            stats.estimated_memory_bytes += collection_size;
        }
        
        stats
    }
}

impl Default for InMemoryKVStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl KVStore for InMemoryKVStore {
    async fn put(&self, key: &str, value: Value, collection: &str) -> Result<()> {
        let mut collections = self.collections.write().await;
        collections
            .entry(collection.to_string())
            .or_insert_with(HashMap::new)
            .insert(key.to_string(), value);
        
        debug!("Put key '{}' in collection '{}'", key, collection);
        Ok(())
    }

    async fn get(&self, key: &str, collection: &str) -> Result<Option<Value>> {
        let collections = self.collections.read().await;
        let result = collections
            .get(collection)
            .and_then(|coll| coll.get(key))
            .cloned();
        
        debug!(
            "Get key '{}' from collection '{}': {}",
            key,
            collection,
            if result.is_some() { "found" } else { "not found" }
        );
        
        Ok(result)
    }

    async fn delete(&self, key: &str, collection: &str) -> Result<bool> {
        let mut collections = self.collections.write().await;
        let deleted = collections
            .get_mut(collection)
            .map(|coll| coll.remove(key).is_some())
            .unwrap_or(false);
        
        debug!(
            "Delete key '{}' from collection '{}': {}",
            key,
            collection,
            if deleted { "deleted" } else { "not found" }
        );
        
        Ok(deleted)
    }

    async fn get_all(&self, collection: &str) -> Result<HashMap<String, Value>> {
        let collections = self.collections.read().await;
        let result = collections
            .get(collection)
            .cloned()
            .unwrap_or_default();
        
        debug!("Get all from collection '{}': {} items", collection, result.len());
        Ok(result)
    }

    async fn put_all(&self, kv_pairs: Vec<(String, Value)>, collection: &str) -> Result<()> {
        let mut collections = self.collections.write().await;
        let coll = collections
            .entry(collection.to_string())
            .or_insert_with(HashMap::new);
        
        for (key, value) in kv_pairs.iter() {
            coll.insert(key.clone(), value.clone());
        }
        
        debug!("Put {} items in collection '{}'", kv_pairs.len(), collection);
        Ok(())
    }

    async fn list_collections(&self) -> Result<Vec<String>> {
        let collections = self.collections.read().await;
        let result: Vec<String> = collections.keys().cloned().collect();
        
        debug!("List collections: {} found", result.len());
        Ok(result)
    }

    async fn delete_collection(&self, collection: &str) -> Result<()> {
        let mut collections = self.collections.write().await;
        collections.remove(collection);
        
        debug!("Deleted collection '{}'", collection);
        Ok(())
    }

    async fn count(&self, collection: &str) -> Result<usize> {
        let collections = self.collections.read().await;
        let count = collections
            .get(collection)
            .map(|coll| coll.len())
            .unwrap_or(0);
        
        debug!("Count collection '{}': {} items", collection, count);
        Ok(count)
    }

    fn name(&self) -> &'static str {
        "InMemoryKVStore"
    }
}

/// Memory usage statistics for the in-memory store.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Number of collections.
    pub collection_count: usize,
    
    /// Total number of keys across all collections.
    pub total_keys: usize,
    
    /// Estimated memory usage in bytes.
    pub estimated_memory_bytes: usize,
}

/// Rough estimation of JSON value memory usage.
fn estimate_value_size(value: &Value) -> usize {
    match value {
        Value::Null => 0,
        Value::Bool(_) => std::mem::size_of::<bool>(),
        Value::Number(_) => std::mem::size_of::<f64>(),
        Value::String(s) => s.len() * std::mem::size_of::<char>(),
        Value::Array(arr) => arr.iter().map(estimate_value_size).sum(),
        Value::Object(obj) => obj.iter().map(|(k, v)| {
            k.len() * std::mem::size_of::<char>() + estimate_value_size(v)
        }).sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_basic_operations() {
        let store = InMemoryKVStore::new();
        let collection = "test";
        
        // Test put and get
        store.put("key1", json!({"value": 42}), collection).await.unwrap();
        let result = store.get("key1", collection).await.unwrap();
        assert_eq!(result, Some(json!({"value": 42})));
        
        // Test non-existent key
        let result = store.get("nonexistent", collection).await.unwrap();
        assert_eq!(result, None);
        
        // Test delete
        let deleted = store.delete("key1", collection).await.unwrap();
        assert!(deleted);
        
        let result = store.get("key1", collection).await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_collections() {
        let store = InMemoryKVStore::new();
        
        // Add data to different collections
        store.put("key1", json!("value1"), "collection1").await.unwrap();
        store.put("key1", json!("value2"), "collection2").await.unwrap();
        
        // Verify isolation
        let val1 = store.get("key1", "collection1").await.unwrap();
        let val2 = store.get("key1", "collection2").await.unwrap();
        
        assert_eq!(val1, Some(json!("value1")));
        assert_eq!(val2, Some(json!("value2")));
        
        // Test list collections
        let collections = store.list_collections().await.unwrap();
        assert_eq!(collections.len(), 2);
        assert!(collections.contains(&"collection1".to_string()));
        assert!(collections.contains(&"collection2".to_string()));
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let store = InMemoryKVStore::new();
        let collection = "test";
        
        let kv_pairs = vec![
            ("key1".to_string(), json!("value1")),
            ("key2".to_string(), json!("value2")),
            ("key3".to_string(), json!("value3")),
        ];
        
        store.put_all(kv_pairs, collection).await.unwrap();
        
        let count = store.count(collection).await.unwrap();
        assert_eq!(count, 3);
        
        let all_data = store.get_all(collection).await.unwrap();
        assert_eq!(all_data.len(), 3);
        assert_eq!(all_data.get("key1"), Some(&json!("value1")));
    }
}
