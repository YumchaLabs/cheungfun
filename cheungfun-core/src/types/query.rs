//! Query types and search modes.
//!
//! This module defines the structures used for querying the RAG system,
//! including different search modes and filtering options.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a search query with various parameters.
///
/// A query contains the search text, optional pre-computed embeddings,
/// metadata filters, and search configuration options.
///
/// # Examples
///
/// ```rust
/// use cheungfun_core::types::{Query, SearchMode};
/// use std::collections::HashMap;
///
/// let query = Query::builder()
///     .text("What is machine learning?")
///     .top_k(5)
///     .search_mode(SearchMode::Vector)
///     .filter("category", "technology")
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct Query {
    /// The query text to search for.
    pub text: String,

    /// Pre-computed query embedding (optional).
    ///
    /// If provided, this embedding will be used for vector search.
    /// If not provided, the embedder will generate it from the query text.
    pub embedding: Option<Vec<f32>>,

    /// Metadata filters to apply during search.
    ///
    /// Filters are applied as exact matches on node metadata.
    /// For example: `{"category": "technology", "language": "en"}`
    pub filters: HashMap<String, serde_json::Value>,

    /// Number of results to return.
    pub top_k: usize,

    /// Minimum similarity threshold for results.
    ///
    /// Results with similarity scores below this threshold will be filtered out.
    /// The exact meaning depends on the similarity metric used.
    pub similarity_threshold: Option<f32>,

    /// Search mode (vector, keyword, or hybrid).
    pub search_mode: SearchMode,
}

/// Different search modes supported by the query engine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SearchMode {
    /// Pure vector/semantic search using dense embeddings.
    Vector,

    /// Pure keyword search using sparse embeddings or traditional text search.
    Keyword,

    /// Hybrid search combining vector and keyword search.
    ///
    /// The `alpha` parameter controls the balance:
    /// - `alpha = 1.0`: Pure vector search
    /// - `alpha = 0.0`: Pure keyword search  
    /// - `alpha = 0.5`: Equal weight to both
    Hybrid {
        /// Weight for vector search (0.0 to 1.0).
        alpha: f32,
    },
}

impl Query {
    /// Create a new query with the given text.
    ///
    /// # Arguments
    ///
    /// * `text` - The query text
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cheungfun_core::types::Query;
    ///
    /// let query = Query::new("What is Rust?");
    /// assert_eq!(query.text, "What is Rust?");
    /// assert_eq!(query.top_k, 10); // default value
    /// ```
    pub fn new<S: Into<String>>(text: S) -> Self {
        Self {
            text: text.into(),
            embedding: None,
            filters: HashMap::new(),
            top_k: 10,
            similarity_threshold: None,
            search_mode: SearchMode::Vector,
        }
    }

    /// Create a builder for constructing queries with fluent API.
    pub fn builder() -> QueryBuilder {
        QueryBuilder::new()
    }

    /// Set the number of results to return.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the search mode.
    pub fn with_search_mode(mut self, search_mode: SearchMode) -> Self {
        self.search_mode = search_mode;
        self
    }

    /// Add a metadata filter.
    pub fn with_filter<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.filters.insert(key.into(), value.into());
        self
    }

    /// Set the similarity threshold.
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = Some(threshold);
        self
    }

    /// Set the pre-computed embedding.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Check if the query has filters.
    pub fn has_filters(&self) -> bool {
        !self.filters.is_empty()
    }

    /// Check if the query has a pre-computed embedding.
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    /// Check if the query uses hybrid search.
    pub fn is_hybrid(&self) -> bool {
        matches!(self.search_mode, SearchMode::Hybrid { .. })
    }

    /// Get the hybrid search alpha value, if applicable.
    pub fn hybrid_alpha(&self) -> Option<f32> {
        match self.search_mode {
            SearchMode::Hybrid { alpha } => Some(alpha),
            _ => None,
        }
    }
}

impl SearchMode {
    /// Create a hybrid search mode with the given alpha value.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Weight for vector search (0.0 to 1.0)
    ///
    /// # Panics
    ///
    /// Panics if alpha is not in the range [0.0, 1.0].
    pub fn hybrid(alpha: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "Alpha must be between 0.0 and 1.0"
        );
        Self::Hybrid { alpha }
    }

    /// Check if this is vector search mode.
    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Vector)
    }

    /// Check if this is keyword search mode.
    pub fn is_keyword(&self) -> bool {
        matches!(self, Self::Keyword)
    }

    /// Check if this is hybrid search mode.
    pub fn is_hybrid(&self) -> bool {
        matches!(self, Self::Hybrid { .. })
    }
}

/// Builder for creating queries with a fluent API.
#[derive(Debug, Default)]
pub struct QueryBuilder {
    text: Option<String>,
    embedding: Option<Vec<f32>>,
    filters: HashMap<String, serde_json::Value>,
    top_k: Option<usize>,
    similarity_threshold: Option<f32>,
    search_mode: Option<SearchMode>,
}

impl QueryBuilder {
    /// Create a new query builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the query text.
    pub fn text<S: Into<String>>(mut self, text: S) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Set the pre-computed embedding.
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Add a metadata filter.
    pub fn filter<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.filters.insert(key.into(), value.into());
        self
    }

    /// Set the number of results to return.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set the similarity threshold.
    pub fn similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = Some(threshold);
        self
    }

    /// Set the search mode.
    pub fn search_mode(mut self, search_mode: SearchMode) -> Self {
        self.search_mode = Some(search_mode);
        self
    }

    /// Build the query.
    ///
    /// # Panics
    ///
    /// Panics if text is not set.
    pub fn build(self) -> Query {
        Query {
            text: self.text.expect("Query text is required"),
            embedding: self.embedding,
            filters: self.filters,
            top_k: self.top_k.unwrap_or(10),
            similarity_threshold: self.similarity_threshold,
            search_mode: self.search_mode.unwrap_or(SearchMode::Vector),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_creation() {
        let query = Query::new("test query");
        assert_eq!(query.text, "test query");
        assert_eq!(query.top_k, 10);
        assert!(!query.has_filters());
        assert!(!query.has_embedding());
    }

    #[test]
    fn test_query_builder() {
        let query = Query::builder()
            .text("test query")
            .top_k(5)
            .filter("category", "test")
            .search_mode(SearchMode::hybrid(0.7))
            .build();

        assert_eq!(query.text, "test query");
        assert_eq!(query.top_k, 5);
        assert!(query.has_filters());
        assert!(query.is_hybrid());
        assert_eq!(query.hybrid_alpha(), Some(0.7));
    }

    #[test]
    fn test_search_mode() {
        assert!(SearchMode::Vector.is_vector());
        assert!(SearchMode::Keyword.is_keyword());

        let hybrid = SearchMode::hybrid(0.5);
        assert!(hybrid.is_hybrid());

        if let SearchMode::Hybrid { alpha } = hybrid {
            assert_eq!(alpha, 0.5);
        }
    }

    #[test]
    #[should_panic(expected = "Alpha must be between 0.0 and 1.0")]
    fn test_invalid_hybrid_alpha() {
        SearchMode::hybrid(1.5);
    }
}
