//! Key-value based index store implementation.

use async_trait::async_trait;
use cheungfun_core::{
    traits::{IndexStore, IndexStoreStats, IndexStruct, KVStore},
    CheungfunError, Result,
};
use serde_json::Value;

use std::sync::Arc;
use tracing::{debug, error, info};

/// Index store implementation based on KVStore.
///
/// This implementation uses a KVStore backend to persist index metadata
/// and structures. Each index is stored as a JSON value with its ID as the key.
/// The store supports collection-based organization for different index types.
///
/// # Examples
///
/// ```rust
/// use cheungfun_integrations::storage::{kvstore::InMemoryKVStore, index_store::KVIndexStore};
/// use cheungfun_core::traits::{IndexStore, IndexStruct};
/// use std::sync::Arc;
/// use std::collections::HashMap;
///
/// # tokio_test::block_on(async {
/// let kv_store = Arc::new(InMemoryKVStore::new());
/// let index_store = KVIndexStore::new(kv_store, Some("indexes".to_string()));
///
/// let index = IndexStruct {
///     index_id: "test_index".to_string(),
///     summary: Some("Test index".to_string()),
///     nodes_dict: HashMap::new(),
///     doc_id_to_node_ids: HashMap::new(),
/// };
///
/// index_store.add_index_struct(&index).await.unwrap();
/// # });
/// ```
#[derive(Debug)]
pub struct KVIndexStore {
    /// Underlying KV store for persistence.
    kv_store: Arc<dyn KVStore>,
    /// Collection name for index storage.
    collection: String,
}

impl KVIndexStore {
    /// Create a new KV-based index store.
    ///
    /// # Arguments
    ///
    /// * `kv_store` - The underlying KV store implementation
    /// * `collection` - Optional collection name (defaults to "indexes")
    pub fn new(kv_store: Arc<dyn KVStore>, collection: Option<String>) -> Self {
        let collection = collection.unwrap_or_else(|| "indexes".to_string());
        info!("Created KV index store with collection '{}'", collection);

        Self {
            kv_store,
            collection,
        }
    }

    /// Get the collection name used by this store.
    pub fn collection(&self) -> &str {
        &self.collection
    }

    /// Get a reference to the underlying KV store.
    pub fn kv_store(&self) -> &Arc<dyn KVStore> {
        &self.kv_store
    }

    /// Convert an IndexStruct to a JSON value for storage.
    fn index_to_value(&self, index: &IndexStruct) -> Result<Value> {
        serde_json::to_value(index).map_err(|e| CheungfunError::Serialization(e))
    }

    /// Convert a JSON value back to an IndexStruct.
    fn value_to_index(&self, value: Value) -> Result<IndexStruct> {
        serde_json::from_value(value).map_err(|e| CheungfunError::Serialization(e))
    }

    /// Get storage statistics for this index store.
    pub async fn get_stats(&self) -> Result<IndexStoreStats> {
        let count = self.kv_store.count(&self.collection).await?;
        let collections = self.kv_store.list_collections().await?;

        Ok(IndexStoreStats {
            index_count: count,
            collection_name: self.collection.clone(),
            total_collections: collections.len(),
        })
    }

    /// Check if the index store is healthy.
    pub async fn health_check(&self) -> Result<bool> {
        match self.kv_store.list_collections().await {
            Ok(_) => Ok(true),
            Err(e) => {
                error!("Index store health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

#[async_trait]
impl IndexStore for KVIndexStore {
    async fn add_index_struct(&self, index: IndexStruct) -> Result<()> {
        let index_id = &index.index_id;
        let index_value = self.index_to_value(&index)?;

        self.kv_store
            .put(index_id, index_value, &self.collection)
            .await?;

        debug!(
            "Added index '{}' to collection '{}'",
            index_id, self.collection
        );
        Ok(())
    }

    async fn delete_index_struct(&self, index_id: &str) -> Result<()> {
        let deleted = self.kv_store.delete(index_id, &self.collection).await?;

        if deleted {
            debug!(
                "Deleted index '{}' from collection '{}'",
                index_id, self.collection
            );
        } else {
            debug!(
                "Index '{}' not found for deletion in collection '{}'",
                index_id, self.collection
            );
        }

        Ok(())
    }

    async fn get_index_struct(&self, index_id: &str) -> Result<Option<IndexStruct>> {
        match self.kv_store.get(index_id, &self.collection).await? {
            Some(value) => {
                let index = self.value_to_index(value)?;
                debug!(
                    "Retrieved index '{}' from collection '{}'",
                    index_id, self.collection
                );
                Ok(Some(index))
            }
            None => {
                debug!(
                    "Index '{}' not found in collection '{}'",
                    index_id, self.collection
                );
                Ok(None)
            }
        }
    }

    async fn clear(&self) -> Result<()> {
        self.kv_store.delete_collection(&self.collection).await?;
        info!("Cleared all indexes from collection '{}'", self.collection);
        Ok(())
    }

    async fn update_index_struct(&self, index: IndexStruct) -> Result<()> {
        let index_id = &index.index_id;
        let index_value = self.index_to_value(&index)?;

        self.kv_store
            .put(index_id, index_value, &self.collection)
            .await?;

        debug!(
            "Updated index '{}' in collection '{}'",
            index_id, self.collection
        );
        Ok(())
    }

    async fn list_index_structs(&self) -> Result<Vec<String>> {
        let all_data = self.kv_store.get_all(&self.collection).await?;
        let index_ids: Vec<String> = all_data.keys().cloned().collect();
        debug!(
            "Listed {} index structs from collection '{}'",
            index_ids.len(),
            self.collection
        );
        Ok(index_ids)
    }

    async fn get_stats(&self) -> Result<IndexStoreStats> {
        let index_ids = self.list_index_structs().await?;
        let index_count = index_ids.len();
        let collections = self.kv_store.list_collections().await?;

        Ok(IndexStoreStats {
            index_count,
            collection_name: self.collection.clone(),
            total_collections: collections.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::kvstore::InMemoryKVStore;

    fn create_test_index(id: &str, index_type: &str) -> IndexStruct {
        IndexStruct::new(id, index_type)
    }

    async fn create_test_store() -> KVIndexStore {
        let kv_store = Arc::new(InMemoryKVStore::new());
        KVIndexStore::new(kv_store, Some("test_indexes".to_string()))
    }

    #[tokio::test]
    async fn test_add_and_get_index() {
        let store = create_test_store().await;
        let index = create_test_index("index1", "Test index");

        // Add index
        store.add_index_struct(index).await.unwrap();

        // Get index
        let retrieved = store.get_index_struct("index1").await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.index_id, "index1");
        assert_eq!(retrieved.index_type, "Test index");
    }

    #[tokio::test]
    async fn test_index_operations() {
        let store = create_test_store().await;
        let index = create_test_index("index1", "Original summary");

        // Add index
        store.add_index_struct(index).await.unwrap();

        // Check existence
        assert!(store.index_struct_exists("index1").await.unwrap());
        assert!(!store.index_struct_exists("nonexistent").await.unwrap());

        // Update index
        let updated_index = create_test_index("index1", "Updated summary");
        store.update_index_struct(updated_index).await.unwrap();

        let retrieved = store.get_index_struct("index1").await.unwrap().unwrap();
        assert_eq!(retrieved.index_type, "Updated summary");

        // Delete index
        store.delete_index_struct("index1").await.unwrap();

        let retrieved = store.get_index_struct("index1").await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let store = create_test_store().await;

        let indexes = vec![
            create_test_index("index1", "Summary 1"),
            create_test_index("index2", "Summary 2"),
            create_test_index("index3", "Summary 3"),
        ];

        // Add multiple indexes
        for index in indexes {
            store.add_index_struct(index).await.unwrap();
        }

        // List indexes
        let listed_ids = store.list_index_structs().await.unwrap();
        assert_eq!(listed_ids.len(), 3);

        // Get specific indexes
        let index1 = store.get_index_struct("index1").await.unwrap();
        let index3 = store.get_index_struct("index3").await.unwrap();
        assert!(index1.is_some());
        assert!(index3.is_some());

        // Clear all
        store.clear().await.unwrap();
        let listed_ids_after_clear = store.list_index_structs().await.unwrap();
        assert_eq!(listed_ids_after_clear.len(), 0);
    }

    #[tokio::test]
    async fn test_complex_index_structure() {
        let store = create_test_store().await;

        let mut index = create_test_index("complex_index", "Complex test index");

        // Add some node IDs to the index
        index.node_ids.push(uuid::Uuid::new_v4());
        index.node_ids.push(uuid::Uuid::new_v4());

        // Add some metadata
        index.metadata.insert(
            "description".to_string(),
            serde_json::Value::String("Test index".to_string()),
        );

        // Store and retrieve
        store.add_index_struct(index).await.unwrap();
        let retrieved = store
            .get_index_struct("complex_index")
            .await
            .unwrap()
            .unwrap();

        assert_eq!(retrieved.node_ids.len(), 2);
        assert_eq!(retrieved.metadata.len(), 1);
        assert_eq!(retrieved.index_id, "complex_index");
        assert_eq!(retrieved.index_type, "Complex test index");
    }
}
