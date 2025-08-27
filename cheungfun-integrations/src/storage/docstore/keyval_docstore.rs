//! Key-value based document store implementation.

use async_trait::async_trait;
use cheungfun_core::{
    traits::{DocumentStore, KVStore, DocumentStoreStats},
    Document, Result, CheungfunError,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info};

/// Document store implementation based on KVStore.
///
/// This implementation uses a KVStore backend to persist Document objects.
/// Each document is stored as a JSON value with its ID as the key.
/// The store supports collection-based organization for multi-tenant scenarios.
///
/// # Examples
///
/// ```rust
/// use cheungfun_integrations::storage::{kvstore::InMemoryKVStore, docstore::KVDocumentStore};
/// use cheungfun_core::{traits::DocumentStore, Document};
/// use std::sync::Arc;
///
/// # tokio_test::block_on(async {
/// let kv_store = Arc::new(InMemoryKVStore::new());
/// let doc_store = KVDocumentStore::new(kv_store, Some("documents".to_string()));
/// 
/// let doc = Document::new("test content", None);
/// let doc_ids = doc_store.add_documents(vec![doc]).await.unwrap();
/// assert_eq!(doc_ids.len(), 1);
/// # });
/// ```
#[derive(Debug)]
pub struct KVDocumentStore {
    /// Underlying KV store for persistence.
    kv_store: Arc<dyn KVStore>,
    /// Collection name for document storage.
    collection: String,
}

impl KVDocumentStore {
    /// Create a new KV-based document store.
    ///
    /// # Arguments
    ///
    /// * `kv_store` - The underlying KV store implementation
    /// * `collection` - Optional collection name (defaults to "documents")
    pub fn new(kv_store: Arc<dyn KVStore>, collection: Option<String>) -> Self {
        let collection = collection.unwrap_or_else(|| "documents".to_string());
        info!("Created KV document store with collection '{}'", collection);
        
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

    /// Convert a Document to a JSON value for storage.
    fn document_to_value(&self, doc: &Document) -> Result<Value> {
        serde_json::to_value(doc)
            .map_err(|e| CheungfunError::Serialization(e))
    }

    /// Convert a JSON value back to a Document.
    fn value_to_document(&self, value: Value) -> Result<Document> {
        serde_json::from_value(value)
            .map_err(|e| CheungfunError::Serialization(e))
    }

    /// Get storage statistics for this document store.
    pub async fn get_stats(&self) -> Result<DocumentStoreStats> {
        let count = self.kv_store.count(&self.collection).await?;
        let collections = self.kv_store.list_collections().await?;

        Ok(DocumentStoreStats {
            document_count: count,
            collection_name: self.collection.clone(),
            total_collections: collections.len(),
        })
    }

    /// Check if the document store is healthy.
    pub async fn health_check(&self) -> Result<bool> {
        // Try to perform a basic operation to check health
        match self.kv_store.list_collections().await {
            Ok(_) => Ok(true),
            Err(e) => {
                error!("Document store health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

#[async_trait]
impl DocumentStore for KVDocumentStore {
    async fn add_documents(&self, docs: Vec<Document>) -> Result<Vec<String>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }

        let mut doc_ids = Vec::with_capacity(docs.len());
        let mut kv_pairs = Vec::with_capacity(docs.len());

        // Prepare all documents for batch insertion
        for doc in docs {
            let doc_id = doc.id.to_string();
            let doc_value = self.document_to_value(&doc)?;
            kv_pairs.push((doc_id.clone(), doc_value));
            doc_ids.push(doc_id);
        }

        // Use batch operation for better performance
        self.kv_store.put_all(kv_pairs, &self.collection).await?;
        
        debug!("Added {} documents to collection '{}'", doc_ids.len(), self.collection);
        Ok(doc_ids)
    }

    async fn get_document(&self, doc_id: &str) -> Result<Option<Document>> {
        match self.kv_store.get(doc_id, &self.collection).await? {
            Some(value) => {
                let doc = self.value_to_document(value)?;
                debug!("Retrieved document '{}' from collection '{}'", doc_id, self.collection);
                Ok(Some(doc))
            }
            None => {
                debug!("Document '{}' not found in collection '{}'", doc_id, self.collection);
                Ok(None)
            }
        }
    }

    async fn get_documents(&self, doc_ids: Vec<String>) -> Result<Vec<Document>> {
        let mut documents = Vec::with_capacity(doc_ids.len());
        
        for doc_id in doc_ids {
            if let Some(doc) = self.get_document(&doc_id).await? {
                documents.push(doc);
            }
        }
        
        debug!("Retrieved {} documents from collection '{}'", documents.len(), self.collection);
        Ok(documents)
    }

    async fn delete_document(&self, doc_id: &str) -> Result<()> {
        let deleted = self.kv_store.delete(doc_id, &self.collection).await?;

        if deleted {
            debug!("Deleted document '{}' from collection '{}'", doc_id, self.collection);
        } else {
            debug!("Document '{}' not found for deletion in collection '{}'", doc_id, self.collection);
        }

        Ok(())
    }



    async fn get_all_document_hashes(&self) -> Result<HashMap<String, String>> {
        // For simplicity, we'll use document ID as both key and hash
        // In a real implementation, you might want to compute actual content hashes
        let all_data = self.kv_store.get_all(&self.collection).await?;
        let mut hashes = HashMap::new();

        for (doc_id, _) in all_data {
            // Use a simple hash based on document ID for now
            // In production, you'd want to hash the actual content
            hashes.insert(doc_id.clone(), format!("hash_{}", doc_id));
        }

        debug!("Retrieved {} document hashes from collection '{}'", hashes.len(), self.collection);
        Ok(hashes)
    }





    async fn document_exists(&self, doc_id: &str) -> Result<bool> {
        let exists = self.kv_store.exists(doc_id, &self.collection).await?;
        debug!("Document '{}' exists in collection '{}': {}", doc_id, self.collection, exists);
        Ok(exists)
    }

    async fn count_documents(&self) -> Result<usize> {
        let count = self.kv_store.count(&self.collection).await?;
        debug!("Document count in collection '{}': {}", self.collection, count);
        Ok(count)
    }

    async fn clear(&self) -> Result<()> {
        self.kv_store.delete_collection(&self.collection).await?;
        info!("Cleared all documents from collection '{}'", self.collection);
        Ok(())
    }

    async fn get_all_documents(&self) -> Result<Vec<Document>> {
        let all_data = self.kv_store.get_all(&self.collection).await?;
        let mut documents = Vec::with_capacity(all_data.len());

        for (doc_id, value) in all_data {
            match self.value_to_document(value) {
                Ok(doc) => documents.push(doc),
                Err(e) => {
                    error!("Failed to deserialize document '{}': {}", doc_id, e);
                    // Continue processing other documents
                }
            }
        }

        debug!("Retrieved all {} documents from collection '{}'", documents.len(), self.collection);
        Ok(documents)
    }

    async fn get_documents_by_metadata(
        &self,
        metadata_filter: std::collections::HashMap<String, String>,
    ) -> Result<Vec<Document>> {
        if metadata_filter.is_empty() {
            return self.get_all_documents().await;
        }

        let all_documents = self.get_all_documents().await?;
        let filtered_documents: Vec<Document> = all_documents
            .into_iter()
            .filter(|doc| {
                // Check if document metadata matches all filter criteria
                metadata_filter.iter().all(|(key, value)| {
                    doc.metadata
                        .get(key)
                        .map(|v| v == value)
                        .unwrap_or(false)
                })
            })
            .collect();

        debug!(
            "Filtered {} documents by metadata from collection '{}'",
            filtered_documents.len(),
            self.collection
        );
        Ok(filtered_documents)
    }

    async fn get_stats(&self) -> Result<DocumentStoreStats> {
        let count = self.kv_store.count(&self.collection).await?;
        let collections = self.kv_store.list_collections().await?;

        Ok(DocumentStoreStats {
            document_count: count,
            collection_name: self.collection.clone(),
            total_collections: collections.len(),
        })
    }


}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::kvstore::InMemoryKVStore;

    fn create_test_document(id: &str, content: &str) -> Document {
        let mut doc = Document::new(content, None);
        doc.id = id.to_string();
        doc
    }

    async fn create_test_store() -> KVDocumentStore {
        let kv_store = Arc::new(InMemoryKVStore::new());
        KVDocumentStore::new(kv_store, Some("test_docs".to_string()))
    }

    #[tokio::test]
    async fn test_add_and_get_document() {
        let store = create_test_store().await;
        let doc = create_test_document("doc1", "Test content");
        
        // Add document
        let doc_ids = store.add_documents(vec![doc.clone()]).await.unwrap();
        assert_eq!(doc_ids, vec!["doc1"]);
        
        // Get document
        let retrieved = store.get_document("doc1").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Test content");
    }

    #[tokio::test]
    async fn test_document_operations() {
        let store = create_test_store().await;
        let doc = create_test_document("doc1", "Original content");
        
        // Add document
        store.add_documents(vec![doc]).await.unwrap();
        
        // Check existence
        assert!(store.document_exists("doc1").await.unwrap());
        assert!(!store.document_exists("nonexistent").await.unwrap());
        
        // Update document
        let updated_doc = create_test_document("doc1", "Updated content");
        store.update_document(updated_doc).await.unwrap();
        
        let retrieved = store.get_document("doc1").await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Updated content");
        
        // Delete document
        let deleted = store.delete_document("doc1").await.unwrap();
        assert!(deleted);
        
        let retrieved = store.get_document("doc1").await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let store = create_test_store().await;
        
        let docs = vec![
            create_test_document("doc1", "Content 1"),
            create_test_document("doc2", "Content 2"),
            create_test_document("doc3", "Content 3"),
        ];
        
        // Add multiple documents
        let doc_ids = store.add_documents(docs).await.unwrap();
        assert_eq!(doc_ids.len(), 3);
        
        // Count documents
        let count = store.count_documents().await.unwrap();
        assert_eq!(count, 3);
        
        // List documents
        let listed_ids = store.list_documents().await.unwrap();
        assert_eq!(listed_ids.len(), 3);
        
        // Get all documents
        let all_docs = store.get_all_documents().await.unwrap();
        assert_eq!(all_docs.len(), 3);
        
        // Clear all
        store.clear().await.unwrap();
        let count = store.count_documents().await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_metadata_filtering() {
        let store = create_test_store().await;
        
        let mut doc1 = create_test_document("doc1", "Content 1");
        doc1.metadata.insert("category".to_string(), "tech".to_string());
        doc1.metadata.insert("author".to_string(), "alice".to_string());
        
        let mut doc2 = create_test_document("doc2", "Content 2");
        doc2.metadata.insert("category".to_string(), "tech".to_string());
        doc2.metadata.insert("author".to_string(), "bob".to_string());
        
        let mut doc3 = create_test_document("doc3", "Content 3");
        doc3.metadata.insert("category".to_string(), "science".to_string());
        doc3.metadata.insert("author".to_string(), "alice".to_string());
        
        store.add_documents(vec![doc1, doc2, doc3]).await.unwrap();
        
        // Filter by category
        let mut filter = HashMap::new();
        filter.insert("category".to_string(), "tech".to_string());
        let filtered = store.get_documents_by_metadata(filter).await.unwrap();
        assert_eq!(filtered.len(), 2);
        
        // Filter by author
        let mut filter = HashMap::new();
        filter.insert("author".to_string(), "alice".to_string());
        let filtered = store.get_documents_by_metadata(filter).await.unwrap();
        assert_eq!(filtered.len(), 2);
        
        // Filter by both
        let mut filter = HashMap::new();
        filter.insert("category".to_string(), "tech".to_string());
        filter.insert("author".to_string(), "alice".to_string());
        let filtered = store.get_documents_by_metadata(filter).await.unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "doc1");
    }
}
