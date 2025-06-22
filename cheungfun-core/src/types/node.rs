//! Node type and related structures.
//!
//! Nodes represent processed chunks of documents that are stored in vector databases
//! and used for retrieval during query processing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Represents a processed chunk of a document.
///
/// Nodes are created by transforming documents into smaller, searchable pieces.
/// Each node contains the text content, embeddings, metadata, and relationships
/// to other nodes or the original document.
///
/// # Examples
///
/// ```rust
/// use cheungfun_core::types::{Node, ChunkInfo};
/// use std::collections::HashMap;
/// use uuid::Uuid;
///
/// let source_doc_id = Uuid::new_v4();
/// let chunk_info = ChunkInfo {
///     start_offset: 0,
///     end_offset: 100,
///     chunk_index: 0,
/// };
///
/// let node = Node {
///     id: Uuid::new_v4(),
///     content: "This is a chunk of text.".to_string(),
///     metadata: HashMap::new(),
///     embedding: Some(vec![0.1, 0.2, 0.3]),
///     sparse_embedding: None,
///     relationships: HashMap::new(),
///     source_document_id: source_doc_id,
///     chunk_info,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Node {
    /// Unique identifier for the node.
    pub id: Uuid,

    /// Text content of the chunk.
    pub content: String,

    /// Node metadata (chunk info, extracted metadata, etc.).
    ///
    /// Common metadata keys include:
    /// - `chunk_type`: Type of chunk (text, table, image, etc.)
    /// - `section`: Document section this chunk belongs to
    /// - `page`: Page number (for documents with pages)
    /// - `extracted_entities`: Named entities found in the text
    /// - `summary`: Auto-generated summary of the chunk
    /// - `keywords`: Important keywords or phrases
    pub metadata: HashMap<String, serde_json::Value>,

    /// Dense vector embedding for semantic search.
    pub embedding: Option<Vec<f32>>,

    /// Sparse vector embedding for keyword/hybrid search.
    ///
    /// Maps token IDs to their weights for sparse retrieval methods
    /// like BM25 or SPLADE.
    pub sparse_embedding: Option<HashMap<u32, f32>>,

    /// Relationships to other nodes/documents.
    ///
    /// Common relationship types:
    /// - `parent`: Parent document or section
    /// - `next`: Next chunk in sequence
    /// - `previous`: Previous chunk in sequence
    /// - `similar`: Semantically similar nodes
    /// - `references`: Nodes this chunk references
    /// - `referenced_by`: Nodes that reference this chunk
    pub relationships: HashMap<String, Uuid>,

    /// Reference to the original document.
    pub source_document_id: Uuid,

    /// Information about the chunk's position in the original document.
    pub chunk_info: ChunkInfo,
}

/// Information about a chunk's position in the original document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkInfo {
    /// Start position in original document (character offset).
    pub start_offset: usize,

    /// End position in original document (character offset).
    pub end_offset: usize,

    /// Chunk index in the document (0-based).
    pub chunk_index: usize,
}

/// A node with an associated similarity score.
///
/// Used in search results to indicate how well a node matches a query.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoredNode {
    /// The node that was retrieved.
    pub node: Node,

    /// Similarity score (higher is more similar).
    ///
    /// The exact range and meaning depends on the similarity metric used:
    /// - Cosine similarity: [-1, 1] where 1 is identical
    /// - Euclidean distance: [0, ∞) where 0 is identical (inverted for scoring)
    /// - Dot product: (-∞, ∞) where higher is more similar
    pub score: f32,
}

impl Node {
    /// Create a new node with the given content and source document.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content of the node
    /// * `source_document_id` - ID of the source document
    /// * `chunk_info` - Information about the chunk's position
    pub fn new<S: Into<String>>(
        content: S,
        source_document_id: Uuid,
        chunk_info: ChunkInfo,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            metadata: HashMap::new(),
            embedding: None,
            sparse_embedding: None,
            relationships: HashMap::new(),
            source_document_id,
            chunk_info,
        }
    }

    /// Create a builder for constructing nodes with fluent API.
    pub fn builder() -> NodeBuilder {
        NodeBuilder::new()
    }

    /// Add or update metadata for this node.
    pub fn with_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the dense embedding for this node.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set the sparse embedding for this node.
    pub fn with_sparse_embedding(mut self, sparse_embedding: HashMap<u32, f32>) -> Self {
        self.sparse_embedding = Some(sparse_embedding);
        self
    }

    /// Add a relationship to another node.
    pub fn with_relationship<K: Into<String>>(
        mut self,
        relationship_type: K,
        target_id: Uuid,
    ) -> Self {
        self.relationships
            .insert(relationship_type.into(), target_id);
        self
    }

    /// Get metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }

    /// Get metadata value as a string.
    pub fn get_metadata_string(&self, key: &str) -> Option<String> {
        self.metadata.get(key)?.as_str().map(String::from)
    }

    /// Check if the node has a dense embedding.
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    /// Check if the node has a sparse embedding.
    pub fn has_sparse_embedding(&self) -> bool {
        self.sparse_embedding.is_some()
    }

    /// Get the size of the chunk in characters.
    pub fn size(&self) -> usize {
        self.content.len()
    }

    /// Check if the node content is empty.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Get the chunk size (end_offset - start_offset).
    pub fn chunk_size(&self) -> usize {
        self.chunk_info.end_offset - self.chunk_info.start_offset
    }
}

impl ScoredNode {
    /// Create a new scored node.
    pub fn new(node: Node, score: f32) -> Self {
        Self { node, score }
    }

    /// Get the node ID.
    pub fn id(&self) -> Uuid {
        self.node.id
    }

    /// Get the node content.
    pub fn content(&self) -> &str {
        &self.node.content
    }
}

impl ChunkInfo {
    /// Create new chunk information.
    pub fn new(start_offset: usize, end_offset: usize, chunk_index: usize) -> Self {
        Self {
            start_offset,
            end_offset,
            chunk_index,
        }
    }

    /// Get the size of the chunk.
    pub fn size(&self) -> usize {
        self.end_offset - self.start_offset
    }
}

/// Builder for creating nodes with a fluent API.
#[derive(Debug)]
pub struct NodeBuilder {
    id: Option<Uuid>,
    content: Option<String>,
    metadata: HashMap<String, serde_json::Value>,
    embedding: Option<Vec<f32>>,
    sparse_embedding: Option<HashMap<u32, f32>>,
    relationships: HashMap<String, Uuid>,
    source_document_id: Option<Uuid>,
    chunk_info: Option<ChunkInfo>,
}

impl NodeBuilder {
    /// Create a new node builder.
    pub fn new() -> Self {
        Self {
            id: None,
            content: None,
            metadata: HashMap::new(),
            embedding: None,
            sparse_embedding: None,
            relationships: HashMap::new(),
            source_document_id: None,
            chunk_info: None,
        }
    }

    /// Set the node ID.
    pub fn id(mut self, id: Uuid) -> Self {
        self.id = Some(id);
        self
    }

    /// Set the node content.
    pub fn content<S: Into<String>>(mut self, content: S) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Add metadata to the node.
    pub fn metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the node embedding.
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set the sparse embedding.
    pub fn sparse_embedding(mut self, sparse_embedding: HashMap<u32, f32>) -> Self {
        self.sparse_embedding = Some(sparse_embedding);
        self
    }

    /// Add a relationship.
    pub fn relationship<K: Into<String>>(mut self, relationship_type: K, target_id: Uuid) -> Self {
        self.relationships
            .insert(relationship_type.into(), target_id);
        self
    }

    /// Set the source document ID.
    pub fn source_document_id(mut self, id: Uuid) -> Self {
        self.source_document_id = Some(id);
        self
    }

    /// Set the chunk information.
    pub fn chunk_info(mut self, chunk_info: ChunkInfo) -> Self {
        self.chunk_info = Some(chunk_info);
        self
    }

    /// Build the node.
    ///
    /// # Panics
    ///
    /// Panics if required fields (content, source_document_id, chunk_info) are not set.
    pub fn build(self) -> Node {
        Node {
            id: self.id.unwrap_or_else(Uuid::new_v4),
            content: self.content.expect("Node content is required"),
            metadata: self.metadata,
            embedding: self.embedding,
            sparse_embedding: self.sparse_embedding,
            relationships: self.relationships,
            source_document_id: self
                .source_document_id
                .expect("Source document ID is required"),
            chunk_info: self.chunk_info.expect("Chunk info is required"),
        }
    }
}

impl Default for NodeBuilder {
    fn default() -> Self {
        Self::new()
    }
}
