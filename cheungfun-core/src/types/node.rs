//! Enhanced Node structure based on LlamaIndex design.
//!
//! This module provides a redesigned Node structure that closely follows
//! LlamaIndex's BaseNode and TextNode design patterns.
//!
//! **Reference**: LlamaIndex schema.py L260-L800

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::relationships::{NodeRelationships, RelatedNodeInfo};

/// Metadata mode for controlling which metadata is included.
///
/// **Reference**: LlamaIndex MetadataMode enum
/// - File: `llama-index-core/llama_index/core/schema.py`
/// - Lines: L241-L245
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetadataMode {
    /// Include all metadata.
    All,
    /// Include only metadata suitable for embeddings.
    Embed,
    /// Include only metadata suitable for LLM.
    Llm,
    /// Include no metadata.
    None,
}

/// Object type enumeration.
///
/// **Reference**: LlamaIndex ObjectType enum
/// - File: `llama-index-core/llama_index/core/schema.py`
/// - Lines: L226-L231
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectType {
    /// Text-based node.
    Text,
    /// Image-based node.
    Image,
    /// Index node.
    Index,
    /// Document node.
    Document,
    /// Multimodal node.
    Multimodal,
}

/// Enhanced Node structure based on LlamaIndex BaseNode.
///
/// **Reference**: LlamaIndex BaseNode class
/// - File: `llama-index-core/llama_index/core/schema.py`
/// - Lines: L260-L481
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Node {
    /// Unique identifier for the node.
    ///
    /// **Reference**: LlamaIndex BaseNode.id_
    pub id: Uuid,

    /// Text content of the node.
    ///
    /// **Reference**: LlamaIndex TextNode.text
    pub content: String,

    /// Node metadata.
    ///
    /// **Reference**: LlamaIndex BaseNode.metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// Dense vector embedding for semantic search.
    ///
    /// **Reference**: LlamaIndex BaseNode.embedding
    pub embedding: Option<Vec<f32>>,

    /// Sparse vector embedding for keyword/hybrid search.
    pub sparse_embedding: Option<HashMap<u32, f32>>,

    /// Structured relationships to other nodes.
    ///
    /// **Reference**: LlamaIndex BaseNode.relationships
    pub relationships: NodeRelationships,

    /// Reference to the original document.
    pub source_document_id: Uuid,

    /// Information about the chunk's position in the original document.
    pub chunk_info: ChunkInfo,

    /// Content hash for deduplication and caching.
    ///
    /// **Reference**: LlamaIndex BaseNode.hash
    pub hash: Option<String>,

    /// MIME type of the node content.
    ///
    /// **Reference**: LlamaIndex TextNode.mimetype
    pub mimetype: String,

    /// Metadata keys to exclude when creating embeddings.
    ///
    /// **Reference**: LlamaIndex BaseNode.excluded_embed_metadata_keys
    pub excluded_embed_metadata_keys: HashSet<String>,

    /// Metadata keys to exclude when sending to LLM.
    ///
    /// **Reference**: LlamaIndex BaseNode.excluded_llm_metadata_keys
    pub excluded_llm_metadata_keys: HashSet<String>,

    /// Template for formatting text with metadata.
    ///
    /// **Reference**: LlamaIndex TextNode.text_template
    pub text_template: String,

    /// Separator between metadata fields.
    ///
    /// **Reference**: LlamaIndex TextNode.metadata_separator
    pub metadata_separator: String,

    /// Template for formatting individual metadata entries.
    pub metadata_template: String,
}

/// Information about a chunk's position in the original document.
///
/// **Reference**: LlamaIndex TextNode start_char_idx/end_char_idx
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkInfo {
    /// Start position in original document (character offset).
    ///
    /// **Reference**: LlamaIndex TextNode.start_char_idx
    pub start_char_idx: Option<usize>,

    /// End position in original document (character offset).
    ///
    /// **Reference**: LlamaIndex TextNode.end_char_idx
    pub end_char_idx: Option<usize>,

    /// Chunk index in the document (0-based).
    pub chunk_index: usize,
}

/// A node with an associated similarity score.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoredNode {
    /// The node that was retrieved.
    pub node: Node,

    /// Similarity score (higher is more similar).
    pub score: f32,
}

impl Node {
    /// Create a new node with the given content and source document.
    pub fn new<S: Into<String>>(
        content: S,
        source_document_id: Uuid,
        chunk_info: ChunkInfo,
    ) -> Self {
        let content = content.into();
        let hash = Self::calculate_hash(&content, &HashMap::new());

        Self {
            id: Uuid::new_v4(),
            content,
            metadata: HashMap::new(),
            embedding: None,
            sparse_embedding: None,
            relationships: NodeRelationships::new(),
            source_document_id,
            chunk_info,
            hash: Some(hash),
            mimetype: "text/plain".to_string(),
            excluded_embed_metadata_keys: HashSet::new(),
            excluded_llm_metadata_keys: HashSet::new(),
            text_template: "{content}\n\n{metadata_str}".to_string(),
            metadata_separator: "\n".to_string(),
            metadata_template: "{key}: {value}".to_string(),
        }
    }

    /// Create a builder for constructing nodes with fluent API.
    #[must_use]
    pub fn builder() -> NodeBuilder {
        NodeBuilder::new()
    }

    /// Get the object type.
    ///
    /// **Reference**: LlamaIndex BaseNode.get_type()
    pub fn get_type(&self) -> ObjectType {
        ObjectType::Text
    }

    /// Get the content hash.
    ///
    /// **Reference**: LlamaIndex BaseNode.hash property
    pub fn hash(&self) -> String {
        self.hash
            .clone()
            .unwrap_or_else(|| Self::calculate_hash(&self.content, &self.metadata))
    }

    /// Calculate hash from content and metadata.
    pub fn calculate_hash(content: &str, metadata: &HashMap<String, serde_json::Value>) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hasher.update(
            serde_json::to_string(metadata)
                .unwrap_or_default()
                .as_bytes(),
        );
        format!("{:x}", hasher.finalize())
    }

    /// Get content with metadata formatting.
    ///
    /// **Reference**: LlamaIndex TextNode.get_content()
    pub fn get_content(&self, metadata_mode: MetadataMode) -> String {
        let metadata_str = self.get_metadata_str(metadata_mode);
        let metadata_str_trimmed = metadata_str.trim();
        if metadata_str_trimmed.is_empty() {
            return self.content.clone();
        }

        self.text_template
            .replace("{content}", &self.content)
            .replace("{metadata_str}", metadata_str_trimmed)
            .trim()
            .to_string()
    }

    /// Get formatted metadata string.
    ///
    /// **Reference**: LlamaIndex TextNode.get_metadata_str()
    pub fn get_metadata_str(&self, mode: MetadataMode) -> String {
        if mode == MetadataMode::None {
            return String::new();
        }

        let mut usable_keys: HashSet<String> = self.metadata.keys().cloned().collect();

        match mode {
            MetadataMode::Llm => {
                for key in &self.excluded_llm_metadata_keys {
                    usable_keys.remove(key);
                }
            }
            MetadataMode::Embed => {
                for key in &self.excluded_embed_metadata_keys {
                    usable_keys.remove(key);
                }
            }
            _ => {}
        }

        usable_keys
            .iter()
            .filter_map(|key| {
                self.metadata.get(key).map(|value| {
                    self.metadata_template
                        .replace("{key}", key)
                        .replace("{value}", &value.to_string())
                })
            })
            .collect::<Vec<_>>()
            .join(&self.metadata_separator)
    }

    /// Set the content of the node.
    ///
    /// **Reference**: LlamaIndex BaseNode.set_content()
    pub fn set_content<S: Into<String>>(&mut self, content: S) {
        self.content = content.into();
        self.hash = Some(Self::calculate_hash(&self.content, &self.metadata));
    }

    /// Get node info.
    ///
    /// **Reference**: LlamaIndex TextNode.get_node_info()
    pub fn get_node_info(&self) -> HashMap<String, serde_json::Value> {
        let mut info = HashMap::new();
        if let Some(start) = self.chunk_info.start_char_idx {
            info.insert("start".to_string(), serde_json::Value::Number(start.into()));
        }
        if let Some(end) = self.chunk_info.end_char_idx {
            info.insert("end".to_string(), serde_json::Value::Number(end.into()));
        }
        info
    }

    /// Convert node to RelatedNodeInfo.
    ///
    /// **Reference**: LlamaIndex BaseNode.as_related_node_info()
    pub fn as_related_node_info(&self) -> RelatedNodeInfo {
        RelatedNodeInfo::new(self.id)
            .set_type(self.get_type().to_string())
            .with_hash(self.hash())
    }

    // Convenience property methods following LlamaIndex pattern

    /// Get source node.
    ///
    /// **Reference**: LlamaIndex BaseNode.source_node property
    pub fn source_node(&self) -> Option<&RelatedNodeInfo> {
        self.relationships.source_node()
    }

    /// Get parent node.
    ///
    /// **Reference**: LlamaIndex BaseNode.parent_node property
    pub fn parent_node(&self) -> Option<&RelatedNodeInfo> {
        self.relationships.parent_node()
    }

    /// Get child nodes.
    ///
    /// **Reference**: LlamaIndex BaseNode.child_nodes property
    pub fn child_nodes(&self) -> Option<&Vec<RelatedNodeInfo>> {
        self.relationships.child_nodes()
    }

    /// Get previous node.
    ///
    /// **Reference**: LlamaIndex BaseNode.prev_node property
    pub fn prev_node(&self) -> Option<&RelatedNodeInfo> {
        self.relationships.prev_node()
    }

    /// Get next node.
    ///
    /// **Reference**: LlamaIndex BaseNode.next_node property
    pub fn next_node(&self) -> Option<&RelatedNodeInfo> {
        self.relationships.next_node()
    }

    // Additional convenience methods

    /// Add or update metadata for this node.
    pub fn with_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.metadata.insert(key.into(), value.into());
        self.hash = Some(Self::calculate_hash(&self.content, &self.metadata));
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

    /// Get the chunk size (end - start).
    pub fn chunk_size(&self) -> usize {
        match (self.chunk_info.start_char_idx, self.chunk_info.end_char_idx) {
            (Some(start), Some(end)) => end.saturating_sub(start),
            _ => self.content.len(),
        }
    }

    /// Get the text content (alias for get_content with MetadataMode::None).
    pub fn get_text(&self) -> String {
        self.get_content(MetadataMode::None)
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

    /// Get the score (alias for compatibility).
    pub fn get_score(&self) -> Option<f32> {
        Some(self.score)
    }
}

impl ChunkInfo {
    /// Create new chunk information.
    pub fn new(
        start_char_idx: Option<usize>,
        end_char_idx: Option<usize>,
        chunk_index: usize,
    ) -> Self {
        Self {
            start_char_idx,
            end_char_idx,
            chunk_index,
        }
    }

    /// Create chunk info with character indices.
    pub fn with_char_indices(start: usize, end: usize, chunk_index: usize) -> Self {
        Self {
            start_char_idx: Some(start),
            end_char_idx: Some(end),
            chunk_index,
        }
    }

    /// Get the size of the chunk.
    pub fn size(&self) -> usize {
        match (self.start_char_idx, self.end_char_idx) {
            (Some(start), Some(end)) => end.saturating_sub(start),
            _ => 0,
        }
    }
}

impl std::fmt::Display for ObjectType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjectType::Text => write!(f, "TEXT"),
            ObjectType::Image => write!(f, "IMAGE"),
            ObjectType::Index => write!(f, "INDEX"),
            ObjectType::Document => write!(f, "DOCUMENT"),
            ObjectType::Multimodal => write!(f, "MULTIMODAL"),
        }
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let content_preview = if self.content.len() > 100 {
            format!("{}...", &self.content[..97])
        } else {
            self.content.clone()
        };

        write!(f, "Node ID: {}\nText: {}\n", self.id, content_preview)
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
    relationships: NodeRelationships,
    source_document_id: Option<Uuid>,
    chunk_info: Option<ChunkInfo>,
    hash: Option<String>,
    mimetype: String,
    excluded_embed_metadata_keys: HashSet<String>,
    excluded_llm_metadata_keys: HashSet<String>,
    text_template: String,
    metadata_separator: String,
    metadata_template: String,
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
            relationships: NodeRelationships::new(),
            source_document_id: None,
            chunk_info: None,
            hash: None,
            mimetype: "text/plain".to_string(),
            excluded_embed_metadata_keys: HashSet::new(),
            excluded_llm_metadata_keys: HashSet::new(),
            text_template: "{content}\n\n{metadata_str}".to_string(),
            metadata_separator: "\n".to_string(),
            metadata_template: "{key}: {value}".to_string(),
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

    /// Add metadata.
    pub fn metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set embedding.
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set sparse embedding.
    pub fn sparse_embedding(mut self, sparse_embedding: HashMap<u32, f32>) -> Self {
        self.sparse_embedding = Some(sparse_embedding);
        self
    }

    /// Set source document ID.
    pub fn source_document_id(mut self, source_document_id: Uuid) -> Self {
        self.source_document_id = Some(source_document_id);
        self
    }

    /// Set chunk info.
    pub fn chunk_info(mut self, chunk_info: ChunkInfo) -> Self {
        self.chunk_info = Some(chunk_info);
        self
    }

    /// Set MIME type.
    pub fn mimetype<S: Into<String>>(mut self, mimetype: S) -> Self {
        self.mimetype = mimetype.into();
        self
    }

    /// Add excluded embed metadata key.
    pub fn exclude_embed_metadata<S: Into<String>>(mut self, key: S) -> Self {
        self.excluded_embed_metadata_keys.insert(key.into());
        self
    }

    /// Add excluded LLM metadata key.
    pub fn exclude_llm_metadata<S: Into<String>>(mut self, key: S) -> Self {
        self.excluded_llm_metadata_keys.insert(key.into());
        self
    }

    /// Set text template.
    pub fn text_template<S: Into<String>>(mut self, template: S) -> Self {
        self.text_template = template.into();
        self
    }

    /// Build the node.
    pub fn build(self) -> Result<Node, String> {
        let content = self.content.ok_or("Content is required")?;
        let source_document_id = self
            .source_document_id
            .ok_or("Source document ID is required")?;
        let chunk_info = self.chunk_info.ok_or("Chunk info is required")?;

        let hash = self
            .hash
            .unwrap_or_else(|| Node::calculate_hash(&content, &self.metadata));

        Ok(Node {
            id: self.id.unwrap_or_else(Uuid::new_v4),
            content,
            metadata: self.metadata,
            embedding: self.embedding,
            sparse_embedding: self.sparse_embedding,
            relationships: self.relationships,
            source_document_id,
            chunk_info,
            hash: Some(hash),
            mimetype: self.mimetype,
            excluded_embed_metadata_keys: self.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys: self.excluded_llm_metadata_keys,
            text_template: self.text_template,
            metadata_separator: self.metadata_separator,
            metadata_template: self.metadata_template,
        })
    }
}

impl Default for NodeBuilder {
    fn default() -> Self {
        Self::new()
    }
}
