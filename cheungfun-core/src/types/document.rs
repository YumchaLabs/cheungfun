//! Document type and related structures.
//!
//! Documents represent raw content from data sources before processing
//! into searchable nodes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Represents a raw document from data sources.
///
/// A document contains the original content along with metadata about its source,
/// creation time, and other relevant information. Documents are processed by
/// transformers to create searchable nodes.
///
/// # Examples
///
/// ```rust
/// use cheungfun_core::types::Document;
/// use std::collections::HashMap;
///
/// let mut metadata = HashMap::new();
/// metadata.insert("source".to_string(), serde_json::Value::String("example.txt".to_string()));
/// metadata.insert("author".to_string(), serde_json::Value::String("John Doe".to_string()));
///
/// let doc = Document {
///     id: uuid::Uuid::new_v4(),
///     content: "This is the document content.".to_string(),
///     metadata,
///     embedding: None,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Document {
    /// Unique identifier for the document.
    pub id: Uuid,

    /// Raw content of the document.
    pub content: String,

    /// Document metadata (source, creation time, etc.).
    ///
    /// Common metadata keys include:
    /// - `source`: Original file path or URL
    /// - `created_at`: Creation timestamp
    /// - `modified_at`: Last modification timestamp
    /// - `author`: Document author
    /// - `title`: Document title
    /// - `content_type`: MIME type or format
    /// - `size`: Document size in bytes
    pub metadata: HashMap<String, serde_json::Value>,

    /// Optional pre-computed embedding for the entire document.
    ///
    /// This is typically computed during indexing and used for
    /// document-level similarity searches.
    pub embedding: Option<Vec<f32>>,
}

impl Document {
    /// Create a new document with the given content.
    ///
    /// # Arguments
    ///
    /// * `content` - The raw text content of the document
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cheungfun_core::types::Document;
    ///
    /// let doc = Document::new("Hello, world!");
    /// assert_eq!(doc.content, "Hello, world!");
    /// assert!(doc.metadata.is_empty());
    /// assert!(doc.embedding.is_none());
    /// ```
    pub fn new<S: Into<String>>(content: S) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            metadata: HashMap::new(),
            embedding: None,
        }
    }

    /// Create a new document with a specific ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The unique identifier for the document
    /// * `content` - The raw text content of the document
    pub fn with_id<S: Into<String>>(id: Uuid, content: S) -> Self {
        Self {
            id,
            content: content.into(),
            metadata: HashMap::new(),
            embedding: None,
        }
    }

    /// Create a builder for constructing documents with fluent API.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cheungfun_core::types::Document;
    ///
    /// let doc = Document::builder()
    ///     .content("Document content")
    ///     .metadata("source", "example.txt")
    ///     .metadata("author", "John Doe")
    ///     .build();
    /// ```
    pub fn builder() -> DocumentBuilder {
        DocumentBuilder::new()
    }

    /// Add or update metadata for this document.
    ///
    /// # Arguments
    ///
    /// * `key` - The metadata key
    /// * `value` - The metadata value (must be JSON-serializable)
    pub fn with_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the embedding for this document.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
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

    /// Check if the document has an embedding.
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    /// Get the document size in characters.
    pub fn size(&self) -> usize {
        self.content.len()
    }

    /// Check if the document is empty.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}

/// Builder for creating documents with a fluent API.
#[derive(Debug, Default)]
pub struct DocumentBuilder {
    id: Option<Uuid>,
    content: Option<String>,
    metadata: HashMap<String, serde_json::Value>,
    embedding: Option<Vec<f32>>,
}

impl DocumentBuilder {
    /// Create a new document builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the document ID.
    pub fn id(mut self, id: Uuid) -> Self {
        self.id = Some(id);
        self
    }

    /// Set the document content.
    pub fn content<S: Into<String>>(mut self, content: S) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Add metadata to the document.
    pub fn metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the document embedding.
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Build the document.
    ///
    /// # Panics
    ///
    /// Panics if content is not set.
    pub fn build(self) -> Document {
        Document {
            id: self.id.unwrap_or_else(Uuid::new_v4),
            content: self.content.expect("Document content is required"),
            metadata: self.metadata,
            embedding: self.embedding,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let doc = Document::new("Test content");
        assert_eq!(doc.content, "Test content");
        assert!(doc.metadata.is_empty());
        assert!(doc.embedding.is_none());
        assert!(!doc.is_empty());
        assert_eq!(doc.size(), 12);
    }

    #[test]
    fn test_document_builder() {
        let doc = Document::builder()
            .content("Test content")
            .metadata("source", "test.txt")
            .metadata("author", "Test Author")
            .build();

        assert_eq!(doc.content, "Test content");
        assert_eq!(
            doc.get_metadata_string("source"),
            Some("test.txt".to_string())
        );
        assert_eq!(
            doc.get_metadata_string("author"),
            Some("Test Author".to_string())
        );
    }

    #[test]
    fn test_document_with_metadata() {
        let doc = Document::new("Test")
            .with_metadata("key1", "value1")
            .with_metadata("key2", 42);

        assert_eq!(doc.get_metadata_string("key1"), Some("value1".to_string()));
        assert_eq!(
            doc.get_metadata("key2"),
            Some(&serde_json::Value::Number(42.into()))
        );
    }

    #[test]
    fn test_document_with_embedding() {
        let embedding = vec![0.1, 0.2, 0.3];
        let doc = Document::new("Test").with_embedding(embedding.clone());

        assert!(doc.has_embedding());
        assert_eq!(doc.embedding, Some(embedding));
    }
}
