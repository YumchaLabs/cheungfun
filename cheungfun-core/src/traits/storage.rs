//! Storage traits for the Cheungfun framework.
//!
//! This module defines traits for storing and retrieving different types of data
//! in RAG applications, including vector embeddings, documents, indexes, and chat history.
//! The design follows LlamaIndex's StorageContext pattern for unified storage management.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use crate::{ChatMessage, Document, MessageRole, Node, Query, Result, ScoredNode};

/// Statistics for document store operations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DocumentStoreStats {
    /// Total number of documents in the store.
    pub document_count: usize,
    /// Name of the collection/namespace.
    pub collection_name: String,
    /// Total number of collections.
    pub total_collections: usize,
}

/// Statistics for index store operations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexStoreStats {
    /// Total number of indexes in the store.
    pub index_count: usize,
    /// Name of the collection/namespace.
    pub collection_name: String,
    /// Total number of collections.
    pub total_collections: usize,
}

/// Statistics for chat store operations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatStoreStats {
    /// Total number of conversations.
    pub conversation_count: usize,
    /// Total number of messages across all conversations.
    pub total_messages: usize,
    /// Name of the base collection/namespace.
    pub collection_name: String,
}

/// Stores and retrieves vector embeddings with associated nodes.
///
/// This trait provides a unified interface for different vector database
/// implementations, allowing the framework to work with various backends
/// like Qdrant, Pinecone, Weaviate, or in-memory stores.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::VectorStore;
/// use cheungfun_core::{Node, Query, ScoredNode, Result};
/// use async_trait::async_trait;
/// use uuid::Uuid;
///
/// struct InMemoryVectorStore {
///     nodes: std::collections::HashMap<Uuid, Node>,
/// }
///
/// #[async_trait]
/// impl VectorStore for InMemoryVectorStore {
///     async fn add(&self, nodes: Vec<Node>) -> Result<Vec<Uuid>> {
///         // Implementation would store nodes and return their IDs
///         Ok(nodes.iter().map(|n| n.id).collect())
///     }
///
///     async fn search(&self, query: &Query) -> Result<Vec<ScoredNode>> {
///         // Implementation would perform similarity search
///         Ok(vec![])
///     }
///
///     async fn get(&self, node_ids: Vec<Uuid>) -> Result<Vec<Option<Node>>> {
///         // Implementation would retrieve nodes by ID
///         Ok(vec![])
///     }
///
///     async fn delete(&self, node_ids: Vec<Uuid>) -> Result<()> {
///         // Implementation would delete nodes
///         Ok(())
///     }
///
///     async fn update(&self, nodes: Vec<Node>) -> Result<()> {
///         // Implementation would update existing nodes
///         Ok(())
///     }
///
///     async fn health_check(&self) -> Result<()> {
///         Ok(())
///     }
/// }
/// ```
#[async_trait]
pub trait VectorStore: Send + Sync + std::fmt::Debug {
    /// Add nodes to the vector store.
    ///
    /// This method stores the provided nodes along with their embeddings
    /// in the vector store. If a node doesn't have an embedding, it should
    /// be rejected or the implementation should generate one.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Vector of nodes to store
    ///
    /// # Returns
    ///
    /// A vector of UUIDs for the stored nodes. The order should match
    /// the input nodes.
    ///
    /// # Errors
    ///
    /// Returns an error if storage fails due to connection issues,
    /// invalid data, or storage capacity limits.
    async fn add(&self, nodes: Vec<Node>) -> Result<Vec<Uuid>>;

    /// Update existing nodes in the store.
    ///
    /// This method updates nodes that already exist in the store.
    /// If a node doesn't exist, it may be added or an error may be returned
    /// depending on the implementation.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Vector of nodes to update
    async fn update(&self, nodes: Vec<Node>) -> Result<()>;

    /// Delete nodes from the store by their IDs.
    ///
    /// # Arguments
    ///
    /// * `node_ids` - Vector of node IDs to delete
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails. Missing nodes may be
    /// ignored or cause an error depending on the implementation.
    async fn delete(&self, node_ids: Vec<Uuid>) -> Result<()>;

    /// Search for similar nodes using vector similarity.
    ///
    /// This is the core method for retrieval. It performs similarity
    /// search based on the query embedding and returns the most similar
    /// nodes with their similarity scores.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query containing embedding and filters
    ///
    /// # Returns
    ///
    /// A vector of scored nodes sorted by similarity (highest first).
    /// The number of results should respect the `top_k` parameter in the query.
    async fn search(&self, query: &Query) -> Result<Vec<ScoredNode>>;

    /// Retrieve nodes by their IDs.
    ///
    /// # Arguments
    ///
    /// * `node_ids` - Vector of node IDs to retrieve
    ///
    /// # Returns
    ///
    /// A vector of optional nodes. `None` indicates that a node
    /// with the corresponding ID was not found.
    async fn get(&self, node_ids: Vec<Uuid>) -> Result<Vec<Option<Node>>>;

    /// Check if the vector store is healthy and accessible.
    ///
    /// This method can be used to verify connectivity and basic
    /// functionality before performing operations.
    async fn health_check(&self) -> Result<()>;

    /// Get a human-readable name for this vector store.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Get the total number of nodes in the store.
    async fn count(&self) -> Result<usize> {
        // Default implementation returns 0
        Ok(0)
    }

    /// Get metadata about the vector store.
    ///
    /// This can include information like storage capacity, index type,
    /// distance metric, etc.
    async fn metadata(&self) -> Result<HashMap<String, serde_json::Value>> {
        // Default implementation returns empty metadata
        Ok(HashMap::new())
    }

    /// Clear all nodes from the store.
    ///
    /// # Warning
    ///
    /// This operation is destructive and cannot be undone.
    async fn clear(&self) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Create an index for better search performance.
    ///
    /// Some vector stores require explicit index creation or rebuilding
    /// for optimal performance.
    async fn create_index(&self, _config: &IndexConfig) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Get statistics about the vector store.
    async fn stats(&self) -> Result<StorageStats> {
        Ok(StorageStats::default())
    }
}

/// Configuration for vector store indexing.
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Type of index to create (e.g., "hnsw", "ivf", "flat").
    pub index_type: String,

    /// Distance metric to use (e.g., "cosine", "euclidean", "dot").
    pub distance_metric: DistanceMetric,

    /// Index-specific parameters.
    pub parameters: HashMap<String, serde_json::Value>,

    /// Whether to replace existing index.
    pub replace_existing: bool,
}

/// Distance metrics for vector similarity.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity (1 - cosine distance).
    Cosine,

    /// Euclidean distance (L2 norm).
    Euclidean,

    /// Dot product similarity.
    DotProduct,

    /// Manhattan distance (L1 norm).
    Manhattan,

    /// Custom distance metric.
    Custom(String),
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            index_type: "hnsw".to_string(),
            distance_metric: DistanceMetric::Cosine,
            parameters: HashMap::new(),
            replace_existing: false,
        }
    }
}

/// Statistics about vector store operations.
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total number of nodes stored.
    pub total_nodes: usize,

    /// Total storage size in bytes.
    pub storage_size_bytes: Option<u64>,

    /// Average search latency in milliseconds.
    pub avg_search_latency_ms: Option<f64>,

    /// Number of search operations performed.
    pub search_operations: usize,

    /// Number of insert operations performed.
    pub insert_operations: usize,

    /// Number of update operations performed.
    pub update_operations: usize,

    /// Number of delete operations performed.
    pub delete_operations: usize,

    /// Index information.
    pub index_info: Option<IndexInfo>,

    /// Additional store-specific statistics.
    pub additional_stats: HashMap<String, serde_json::Value>,
}

/// Information about vector store indexes.
#[derive(Debug, Clone)]
pub struct IndexInfo {
    /// Type of index.
    pub index_type: String,

    /// Distance metric used.
    pub distance_metric: DistanceMetric,

    /// Dimension of vectors.
    pub dimension: usize,

    /// Whether the index is ready for queries.
    pub is_ready: bool,

    /// Index build time.
    pub build_time: Option<std::time::Duration>,

    /// Index-specific metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl StorageStats {
    /// Create new storage statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate the average operations per second.
    #[must_use]
    pub fn operations_per_second(&self, duration: std::time::Duration) -> f64 {
        let total_ops = self.search_operations
            + self.insert_operations
            + self.update_operations
            + self.delete_operations;

        if duration.is_zero() {
            0.0
        } else {
            total_ops as f64 / duration.as_secs_f64()
        }
    }

    /// Get the storage efficiency (nodes per byte).
    #[must_use]
    pub fn storage_efficiency(&self) -> Option<f64> {
        self.storage_size_bytes.map(|size| {
            if size == 0 {
                0.0
            } else {
                self.total_nodes as f64 / size as f64
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_stats() {
        let mut stats = StorageStats::new();
        stats.total_nodes = 1000;
        stats.storage_size_bytes = Some(1_000_000);
        stats.search_operations = 500;
        stats.insert_operations = 100;

        let duration = std::time::Duration::from_secs(10);
        assert_eq!(stats.operations_per_second(duration), 60.0);
        assert_eq!(stats.storage_efficiency(), Some(0.001));
    }

    #[test]
    fn test_distance_metric() {
        assert_eq!(DistanceMetric::Cosine, DistanceMetric::Cosine);
        assert_ne!(DistanceMetric::Cosine, DistanceMetric::Euclidean);

        let custom = DistanceMetric::Custom("hamming".to_string());
        if let DistanceMetric::Custom(name) = custom {
            assert_eq!(name, "hamming");
        }
    }

    #[test]
    fn test_index_config_default() {
        let config = IndexConfig::default();
        assert_eq!(config.index_type, "hnsw");
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert!(!config.replace_existing);
    }
}

// ============================================================================
// Extended Storage Traits (LlamaIndex-style StorageContext)
// ============================================================================

/// Document storage trait for persisting and retrieving documents.
///
/// This trait provides a unified interface for storing documents with their
/// metadata, following LlamaIndex's DocumentStore pattern. Documents are
/// the raw input data before being processed into nodes.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::DocumentStore;
/// use cheungfun_core::{Document, Result};
/// use async_trait::async_trait;
/// use std::collections::HashMap;
///
/// struct SimpleDocumentStore {
///     documents: std::sync::RwLock<HashMap<String, Document>>,
/// }
///
/// #[async_trait]
/// impl DocumentStore for SimpleDocumentStore {
///     async fn add_documents(&self, docs: Vec<Document>) -> Result<Vec<String>> {
///         // Implementation would store documents and return their IDs
///         Ok(docs.iter().map(|d| d.id.clone()).collect())
///     }
///
///     async fn get_document(&self, doc_id: &str) -> Result<Option<Document>> {
///         // Implementation would retrieve document by ID
///         Ok(None)
///     }
///
///     async fn get_documents(&self, doc_ids: Vec<String>) -> Result<Vec<Document>> {
///         // Implementation would retrieve multiple documents
///         Ok(vec![])
///     }
///
///     async fn delete_document(&self, doc_id: &str) -> Result<()> {
///         // Implementation would delete document
///         Ok(())
///     }
///
///     async fn get_all_document_hashes(&self) -> Result<HashMap<String, String>> {
///         // Implementation would return document ID -> hash mapping
///         Ok(HashMap::new())
///     }
/// }
/// ```
#[async_trait]
pub trait DocumentStore: Send + Sync + std::fmt::Debug {
    /// Add documents to the store.
    ///
    /// # Arguments
    ///
    /// * `docs` - Vector of documents to store
    ///
    /// # Returns
    ///
    /// A vector of document IDs for the stored documents.
    async fn add_documents(&self, docs: Vec<Document>) -> Result<Vec<String>>;

    /// Get a single document by ID.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document ID to retrieve
    ///
    /// # Returns
    ///
    /// The document if found, None otherwise.
    async fn get_document(&self, doc_id: &str) -> Result<Option<Document>>;

    /// Get multiple documents by their IDs.
    ///
    /// # Arguments
    ///
    /// * `doc_ids` - Vector of document IDs to retrieve
    ///
    /// # Returns
    ///
    /// A vector of documents. Missing documents are omitted from the result.
    async fn get_documents(&self, doc_ids: Vec<String>) -> Result<Vec<Document>>;

    /// Delete a document from the store.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document ID to delete
    async fn delete_document(&self, doc_id: &str) -> Result<()>;

    /// Get all document hashes for change detection.
    ///
    /// # Returns
    ///
    /// A mapping of document ID to content hash.
    async fn get_all_document_hashes(&self) -> Result<HashMap<String, String>>;

    /// Check if a document exists in the store.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document ID to check
    async fn document_exists(&self, doc_id: &str) -> Result<bool> {
        Ok(self.get_document(doc_id).await?.is_some())
    }

    /// Get the total number of documents in the store.
    async fn count_documents(&self) -> Result<usize> {
        Ok(0) // Default implementation
    }

    /// Clear all documents from the store.
    ///
    /// # Warning
    ///
    /// This operation is destructive and cannot be undone.
    async fn clear(&self) -> Result<()> {
        Ok(()) // Default implementation does nothing
    }

    /// Get all documents from the store.
    ///
    /// # Warning
    ///
    /// This operation can be expensive for large document stores.
    /// Consider using pagination or filtering for production use.
    ///
    /// # Returns
    ///
    /// A vector of all documents in the store.
    async fn get_all_documents(&self) -> Result<Vec<Document>> {
        // Default implementation - subclasses should override for efficiency
        let hashes = self.get_all_document_hashes().await?;
        let doc_ids: Vec<String> = hashes.keys().cloned().collect();
        self.get_documents(doc_ids).await
    }

    /// Get documents filtered by metadata.
    ///
    /// # Arguments
    ///
    /// * `metadata_filter` - Key-value pairs that documents must match
    ///
    /// # Returns
    ///
    /// A vector of documents matching all filter criteria.
    async fn get_documents_by_metadata(
        &self,
        metadata_filter: HashMap<String, String>,
    ) -> Result<Vec<Document>> {
        if metadata_filter.is_empty() {
            return self.get_all_documents().await;
        }

        let all_documents = self.get_all_documents().await?;
        let filtered_documents: Vec<Document> = all_documents
            .into_iter()
            .filter(|doc| {
                // Check if document metadata matches all filter criteria
                metadata_filter
                    .iter()
                    .all(|(key, value)| doc.metadata.get(key).map(|v| v == value).unwrap_or(false))
            })
            .collect();

        Ok(filtered_documents)
    }

    /// Get storage statistics for this document store.
    ///
    /// # Returns
    ///
    /// Statistics about the document store including counts and collection info.
    async fn get_stats(&self) -> Result<DocumentStoreStats> {
        let document_count = self.count_documents().await?;
        Ok(DocumentStoreStats {
            document_count,
            collection_name: "default".to_string(), // Default implementation
            total_collections: 1,
        })
    }
}

/// Index structure for storing index metadata and relationships.
///
/// This represents the structure of an index, including its configuration,
/// node relationships, and metadata. It's used by IndexStore to persist
/// index information across sessions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexStruct {
    /// Unique identifier for the index.
    pub index_id: String,

    /// Type of index (e.g., "vector", "keyword", "graph").
    pub index_type: String,

    /// Index configuration parameters.
    pub config: HashMap<String, serde_json::Value>,

    /// Node IDs that belong to this index.
    pub node_ids: Vec<Uuid>,

    /// Index metadata.
    pub metadata: HashMap<String, serde_json::Value>,

    /// Timestamp when the index was created.
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Timestamp when the index was last updated.
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl IndexStruct {
    /// Create a new index structure.
    pub fn new<S: Into<String>>(index_id: S, index_type: S) -> Self {
        let now = chrono::Utc::now();
        Self {
            index_id: index_id.into(),
            index_type: index_type.into(),
            config: HashMap::new(),
            node_ids: Vec::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a node ID to the index.
    pub fn add_node(&mut self, node_id: Uuid) {
        if !self.node_ids.contains(&node_id) {
            self.node_ids.push(node_id);
            self.updated_at = chrono::Utc::now();
        }
    }

    /// Remove a node ID from the index.
    pub fn remove_node(&mut self, node_id: &Uuid) {
        if let Some(pos) = self.node_ids.iter().position(|id| id == node_id) {
            self.node_ids.remove(pos);
            self.updated_at = chrono::Utc::now();
        }
    }

    /// Update the index metadata.
    pub fn update_metadata(&mut self, key: String, value: serde_json::Value) {
        self.metadata.insert(key, value);
        self.updated_at = chrono::Utc::now();
    }
}

/// Index storage trait for persisting index structures and metadata.
///
/// This trait provides a unified interface for storing index metadata,
/// following LlamaIndex's IndexStore pattern. It stores information about
/// how data is organized and indexed, separate from the actual data storage.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::{IndexStore, IndexStruct};
/// use cheungfun_core::Result;
/// use async_trait::async_trait;
/// use std::collections::HashMap;
///
/// struct SimpleIndexStore {
///     indexes: std::sync::RwLock<HashMap<String, IndexStruct>>,
/// }
///
/// #[async_trait]
/// impl IndexStore for SimpleIndexStore {
///     async fn add_index_struct(&self, index_struct: IndexStruct) -> Result<()> {
///         // Implementation would store the index structure
///         Ok(())
///     }
///
///     async fn get_index_struct(&self, struct_id: &str) -> Result<Option<IndexStruct>> {
///         // Implementation would retrieve index structure by ID
///         Ok(None)
///     }
///
///     async fn delete_index_struct(&self, struct_id: &str) -> Result<()> {
///         // Implementation would delete index structure
///         Ok(())
///     }
/// }
/// ```
#[async_trait]
pub trait IndexStore: Send + Sync + std::fmt::Debug {
    /// Add an index structure to the store.
    ///
    /// # Arguments
    ///
    /// * `index_struct` - The index structure to store
    async fn add_index_struct(&self, index_struct: IndexStruct) -> Result<()>;

    /// Get an index structure by ID.
    ///
    /// # Arguments
    ///
    /// * `struct_id` - The index structure ID to retrieve
    ///
    /// # Returns
    ///
    /// The index structure if found, None otherwise.
    async fn get_index_struct(&self, struct_id: &str) -> Result<Option<IndexStruct>>;

    /// Delete an index structure from the store.
    ///
    /// # Arguments
    ///
    /// * `struct_id` - The index structure ID to delete
    async fn delete_index_struct(&self, struct_id: &str) -> Result<()>;

    /// List all index structure IDs.
    ///
    /// # Returns
    ///
    /// A vector of all index structure IDs in the store.
    async fn list_index_structs(&self) -> Result<Vec<String>> {
        Ok(Vec::new()) // Default implementation
    }

    /// Update an existing index structure.
    ///
    /// # Arguments
    ///
    /// * `index_struct` - The updated index structure
    async fn update_index_struct(&self, index_struct: IndexStruct) -> Result<()> {
        // Default implementation: delete and re-add
        self.delete_index_struct(&index_struct.index_id).await?;
        self.add_index_struct(index_struct).await
    }

    /// Check if an index structure exists.
    ///
    /// # Arguments
    ///
    /// * `struct_id` - The index structure ID to check
    async fn index_struct_exists(&self, struct_id: &str) -> Result<bool> {
        Ok(self.get_index_struct(struct_id).await?.is_some())
    }

    /// Clear all index structures from the store.
    ///
    /// # Warning
    ///
    /// This operation is destructive and cannot be undone.
    async fn clear(&self) -> Result<()> {
        Ok(()) // Default implementation does nothing
    }

    /// Get storage statistics for this index store.
    ///
    /// # Returns
    ///
    /// Statistics about the index store including counts and collection info.
    async fn get_stats(&self) -> Result<IndexStoreStats> {
        let index_ids = self.list_index_structs().await?;
        let index_count = index_ids.len();
        Ok(IndexStoreStats {
            index_count,
            collection_name: "default".to_string(), // Default implementation
            total_collections: 1,
        })
    }
}

/// Chat storage trait for persisting conversation history.
///
/// This trait provides a unified interface for storing and retrieving
/// chat messages, following LlamaIndex's ChatStore pattern. It enables
/// persistent conversation memory across sessions.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::ChatStore;
/// use cheungfun_core::{ChatMessage, Result};
/// use async_trait::async_trait;
/// use std::collections::HashMap;
///
/// struct SimpleChatStore {
///     conversations: std::sync::RwLock<HashMap<String, Vec<ChatMessage>>>,
/// }
///
/// #[async_trait]
/// impl ChatStore for SimpleChatStore {
///     async fn set_messages(&self, key: &str, messages: Vec<ChatMessage>) -> Result<()> {
///         // Implementation would store the conversation
///         Ok(())
///     }
///
///     async fn get_messages(&self, key: &str) -> Result<Vec<ChatMessage>> {
///         // Implementation would retrieve conversation by key
///         Ok(vec![])
///     }
///
///     async fn add_message(&self, key: &str, message: ChatMessage) -> Result<()> {
///         // Implementation would add message to conversation
///         Ok(())
///     }
///
///     async fn delete_messages(&self, key: &str) -> Result<()> {
///         // Implementation would delete conversation
///         Ok(())
///     }
///
///     async fn get_keys(&self) -> Result<Vec<String>> {
///         // Implementation would return all conversation keys
///         Ok(vec![])
///     }
/// }
/// ```
#[async_trait]
pub trait ChatStore: Send + Sync + std::fmt::Debug {
    /// Set messages for a conversation key (replacing existing ones).
    ///
    /// # Arguments
    ///
    /// * `key` - The conversation key (e.g., session ID, user ID)
    /// * `messages` - Vector of chat messages to store
    async fn set_messages(&self, key: &str, messages: Vec<ChatMessage>) -> Result<()>;

    /// Get all messages for a conversation key.
    ///
    /// # Arguments
    ///
    /// * `key` - The conversation key to retrieve
    ///
    /// # Returns
    ///
    /// A vector of chat messages for the conversation.
    async fn get_messages(&self, key: &str) -> Result<Vec<ChatMessage>>;

    /// Add a single message to a conversation.
    ///
    /// # Arguments
    ///
    /// * `key` - The conversation key
    /// * `message` - The chat message to add
    async fn add_message(&self, key: &str, message: ChatMessage) -> Result<()>;

    /// Delete all messages for a conversation key.
    ///
    /// # Arguments
    ///
    /// * `key` - The conversation key to delete
    async fn delete_messages(&self, key: &str) -> Result<()>;

    /// Delete a specific message from a conversation.
    ///
    /// # Arguments
    ///
    /// * `key` - The conversation key
    /// * `message_index` - The index of the message to delete
    async fn delete_message(&self, key: &str, message_index: usize) -> Result<()> {
        let mut messages = self.get_messages(key).await?;
        if message_index < messages.len() {
            messages.remove(message_index);
            self.set_messages(key, messages).await?;
        }
        Ok(())
    }

    /// Delete the last message from a conversation.
    ///
    /// # Arguments
    ///
    /// * `key` - The conversation key
    async fn delete_last_message(&self, key: &str) -> Result<Option<ChatMessage>> {
        let mut messages = self.get_messages(key).await?;
        let last_message = messages.pop();
        if last_message.is_some() {
            self.set_messages(key, messages).await?;
        }
        Ok(last_message)
    }

    /// Get all conversation keys.
    ///
    /// # Returns
    ///
    /// A vector of all conversation keys in the store.
    async fn get_keys(&self) -> Result<Vec<String>>;

    /// Check if a conversation exists.
    ///
    /// # Arguments
    ///
    /// * `key` - The conversation key to check
    async fn conversation_exists(&self, key: &str) -> Result<bool> {
        Ok(!self.get_messages(key).await?.is_empty())
    }

    /// Get the number of messages in a conversation.
    ///
    /// # Arguments
    ///
    /// * `key` - The conversation key
    async fn count_messages(&self, key: &str) -> Result<usize> {
        Ok(self.get_messages(key).await?.len())
    }

    /// Get recent messages from a conversation.
    ///
    /// # Arguments
    ///
    /// * `key` - The conversation key
    /// * `limit` - Maximum number of recent messages to retrieve
    async fn get_recent_messages(&self, key: &str, limit: usize) -> Result<Vec<ChatMessage>> {
        let messages = self.get_messages(key).await?;
        let start = messages.len().saturating_sub(limit);
        Ok(messages[start..].to_vec())
    }

    /// Clear all conversations from the store.
    ///
    /// # Warning
    ///
    /// This operation is destructive and cannot be undone.
    async fn clear(&self) -> Result<()> {
        Ok(()) // Default implementation does nothing
    }

    /// Get messages filtered by role.
    ///
    /// # Arguments
    ///
    /// * `key` - The conversation key
    /// * `role` - The message role to filter by
    ///
    /// # Returns
    ///
    /// A vector of messages matching the specified role.
    async fn get_messages_by_role(&self, key: &str, role: MessageRole) -> Result<Vec<ChatMessage>> {
        let all_messages = self.get_messages(key).await?;
        let filtered_messages: Vec<ChatMessage> = all_messages
            .into_iter()
            .filter(|message| message.role == role)
            .collect();
        Ok(filtered_messages)
    }

    /// Get the last N messages from a conversation.
    ///
    /// # Arguments
    ///
    /// * `key` - The conversation key
    /// * `limit` - Maximum number of messages to retrieve
    ///
    /// # Returns
    ///
    /// A vector of the last N messages.
    async fn get_last_messages(&self, key: &str, limit: usize) -> Result<Vec<ChatMessage>> {
        self.get_recent_messages(key, limit).await
    }

    /// List all conversation keys (alias for get_keys for consistency).
    ///
    /// # Returns
    ///
    /// A vector of all conversation keys in the store.
    async fn list_conversations(&self) -> Result<Vec<String>> {
        self.get_keys().await
    }

    /// Get storage statistics for this chat store.
    ///
    /// # Returns
    ///
    /// Statistics about the chat store including conversation and message counts.
    async fn get_stats(&self) -> Result<ChatStoreStats> {
        let conversation_keys = self.get_keys().await?;
        let conversation_count = conversation_keys.len();

        let mut total_messages = 0;
        for key in &conversation_keys {
            total_messages += self.count_messages(key).await?;
        }

        Ok(ChatStoreStats {
            conversation_count,
            total_messages,
            collection_name: "default".to_string(), // Default implementation
        })
    }
}

/// Unified storage context for managing all storage components.
///
/// This structure provides a centralized way to manage different storage
/// backends, following LlamaIndex's StorageContext pattern. It combines
/// document storage, index storage, vector storage, and chat storage
/// into a single, cohesive interface.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::{StorageContext, DocumentStore, IndexStore, VectorStore, ChatStore};
/// use std::sync::Arc;
///
/// // Create storage context with default implementations
/// let storage_context = StorageContext::from_defaults(None, None, None, None);
///
/// // Or with custom implementations
/// let storage_context = StorageContext::new(
///     Arc::new(my_doc_store),
///     Arc::new(my_index_store),
///     Arc::new(my_vector_store),
///     Some(Arc::new(my_chat_store)),
/// );
/// ```
#[derive(Debug)]
pub struct StorageContext {
    /// Document store for persisting raw documents.
    pub doc_store: Arc<dyn DocumentStore>,

    /// Index store for persisting index metadata.
    pub index_store: Arc<dyn IndexStore>,

    /// Vector stores for similarity search (can have multiple namespaces).
    pub vector_stores: HashMap<String, Arc<dyn VectorStore>>,

    /// Chat store for conversation history (optional).
    pub chat_store: Option<Arc<dyn ChatStore>>,
}

/// Default vector store namespace.
pub const DEFAULT_VECTOR_STORE_NAMESPACE: &str = "default";

impl StorageContext {
    /// Create a new storage context with the provided stores.
    ///
    /// # Arguments
    ///
    /// * `doc_store` - Document store implementation
    /// * `index_store` - Index store implementation
    /// * `vector_store` - Primary vector store implementation
    /// * `chat_store` - Optional chat store implementation
    pub fn new(
        doc_store: Arc<dyn DocumentStore>,
        index_store: Arc<dyn IndexStore>,
        vector_store: Arc<dyn VectorStore>,
        chat_store: Option<Arc<dyn ChatStore>>,
    ) -> Self {
        let mut vector_stores = HashMap::new();
        vector_stores.insert(DEFAULT_VECTOR_STORE_NAMESPACE.to_string(), vector_store);

        Self {
            doc_store,
            index_store,
            vector_stores,
            chat_store,
        }
    }

    /// Create a storage context with default implementations.
    ///
    /// This method will create default KV-based store implementations if none are provided.
    /// It uses an in-memory KV store by default, but can be configured with persistent stores.
    ///
    /// # Arguments
    ///
    /// * `doc_store` - Optional document store (will use KVDocumentStore if None)
    /// * `index_store` - Optional index store (will use KVIndexStore if None)
    /// * `vector_store` - Optional vector store (will use InMemoryVectorStore if None)
    /// * `chat_store` - Optional chat store (will use KVChatStore if None)
    pub async fn from_defaults(
        doc_store: Option<Arc<dyn DocumentStore>>,
        index_store: Option<Arc<dyn IndexStore>>,
        vector_store: Option<Arc<dyn VectorStore>>,
        chat_store: Option<Arc<dyn ChatStore>>,
    ) -> Result<Self> {
        // This will be implemented once we have the integrations module properly set up
        // For now, require explicit stores
        let doc_store = doc_store.ok_or_else(|| crate::CheungfunError::Configuration {
            message: "DocumentStore is required - use KVDocumentStore with InMemoryKVStore"
                .to_string(),
        })?;
        let index_store = index_store.ok_or_else(|| crate::CheungfunError::Configuration {
            message: "IndexStore is required - use KVIndexStore with InMemoryKVStore".to_string(),
        })?;
        let vector_store = vector_store.ok_or_else(|| crate::CheungfunError::Configuration {
            message: "VectorStore is required - use InMemoryVectorStore".to_string(),
        })?;

        Ok(Self::new(doc_store, index_store, vector_store, chat_store))
    }

    /// Persist all storage components to a directory.
    ///
    /// This method saves the storage context configuration and delegates
    /// persistence to individual stores that support it.
    ///
    /// # Arguments
    ///
    /// * `persist_dir` - Directory to save the storage context to
    pub async fn persist(&self, persist_dir: &Path) -> Result<()> {
        use std::fs;

        // Create persist directory
        fs::create_dir_all(persist_dir).map_err(|e| {
            crate::CheungfunError::Storage(format!("Failed to create persist directory: {e}"))
        })?;

        // Create storage context configuration
        let config = StorageContextConfig {
            doc_store_type: "KVDocumentStore".to_string(), // TODO: Get from trait method
            index_store_type: "KVIndexStore".to_string(),  // TODO: Get from trait method
            vector_store_type: "VectorStore".to_string(),  // TODO: Get from trait method
            chat_store_type: self.chat_store.as_ref().map(|_| "KVChatStore".to_string()),
            vector_store_namespaces: self.vector_stores.keys().cloned().collect(),
            created_at: chrono::Utc::now(),
        };

        // Save configuration
        let config_path = persist_dir.join("storage_context.json");
        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| crate::CheungfunError::Serialization(e))?;
        fs::write(config_path, config_json)
            .map_err(|e| crate::CheungfunError::Storage(format!("Failed to write config: {e}")))?;

        // TODO: Delegate persistence to individual stores if they support it
        // This would require extending the store traits with persistence methods

        Ok(())
    }

    /// Load storage context from a persisted directory.
    ///
    /// This method loads the storage context configuration and recreates
    /// the storage context with the appropriate store implementations.
    ///
    /// # Arguments
    ///
    /// * `persist_dir` - Directory to load the storage context from
    pub async fn from_persist_dir(persist_dir: &Path) -> Result<Self> {
        let config_path = persist_dir.join("storage_context.json");
        let config_json = std::fs::read_to_string(config_path)
            .map_err(|e| crate::CheungfunError::Storage(format!("Failed to read config: {e}")))?;
        let _config: StorageContextConfig = serde_json::from_str(&config_json)
            .map_err(|e| crate::CheungfunError::Serialization(e))?;

        // TODO: Recreate stores based on configuration
        // For now, return an error indicating this needs to be implemented
        Err(crate::CheungfunError::Configuration {
            message: "Loading from persist directory not yet implemented - use from_defaults with explicit stores".to_string(),
        })
    }

    /// Get the default vector store.
    pub fn vector_store(&self) -> &Arc<dyn VectorStore> {
        self.vector_stores
            .get(DEFAULT_VECTOR_STORE_NAMESPACE)
            .expect("Default vector store should always exist")
    }

    /// Add a vector store with a specific namespace.
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace for the vector store
    /// * `vector_store` - The vector store implementation
    pub fn add_vector_store(&mut self, namespace: String, vector_store: Arc<dyn VectorStore>) {
        self.vector_stores.insert(namespace, vector_store);
    }

    /// Get a vector store by namespace.
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace of the vector store
    pub fn get_vector_store(&self, namespace: &str) -> Option<&Arc<dyn VectorStore>> {
        self.vector_stores.get(namespace)
    }

    /// Check if all required stores are healthy.
    pub async fn health_check(&self) -> Result<()> {
        // Check vector store health
        self.vector_store().health_check().await?;

        // TODO: Add health checks for other stores once they implement it
        // self.doc_store.health_check().await?;
        // self.index_store.health_check().await?;
        // if let Some(chat_store) = &self.chat_store {
        //     chat_store.health_check().await?;
        // }

        Ok(())
    }

    /// Get storage statistics from all stores.
    pub async fn get_stats(&self) -> Result<StorageContextStats> {
        let vector_stats = self.vector_store().stats().await?;

        // TODO: Collect stats from other stores
        Ok(StorageContextStats {
            vector_stats,
            doc_count: 0,          // TODO: Get from doc_store
            index_count: 0,        // TODO: Get from index_store
            conversation_count: 0, // TODO: Get from chat_store
        })
    }
}

/// Configuration for persisting and loading storage context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageContextConfig {
    /// Type of document store.
    pub doc_store_type: String,

    /// Type of index store.
    pub index_store_type: String,

    /// Type of vector store.
    pub vector_store_type: String,

    /// Type of chat store (optional).
    pub chat_store_type: Option<String>,

    /// Vector store namespaces.
    pub vector_store_namespaces: Vec<String>,

    /// When the configuration was created.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Statistics for the entire storage context.
#[derive(Debug, Clone)]
pub struct StorageContextStats {
    /// Vector store statistics.
    pub vector_stats: StorageStats,

    /// Number of documents in document store.
    pub doc_count: usize,

    /// Number of indexes in index store.
    pub index_count: usize,

    /// Number of conversations in chat store.
    pub conversation_count: usize,
}
