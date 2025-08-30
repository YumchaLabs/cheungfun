//! Keyword Table Index implementation.
//!
//! This module provides a KeywordTableIndex implementation that follows LlamaIndex's
//! design exactly. It uses keyword extraction and inverted indexing for efficient
//! keyword-based document retrieval.
//!
//! **Reference**: LlamaIndex KeywordTableIndex
//! - Inverted index mapping keywords to documents
//! - Efficient keyword-based retrieval
//! - Suitable for exact term matching and filtering

use async_trait::async_trait;
use cheungfun_core::{
    traits::{KeywordExtractor, KeywordStore, Retriever, SimpleKeywordExtractor},
    types::{Document, Node, Query, ScoredNode},
    Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, instrument};
use uuid::Uuid;

/// Configuration for KeywordTableIndex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordTableIndexConfig {
    /// Whether to show progress during indexing
    pub show_progress: bool,

    /// Maximum number of nodes to store
    pub max_nodes: Option<usize>,

    /// Whether to deduplicate nodes based on content hash
    pub deduplicate: bool,

    /// Minimum content length to include a node
    pub min_content_length: usize,

    /// Maximum number of results to return
    pub max_results: usize,

    /// Minimum relevance score threshold
    pub score_threshold: Option<f32>,

    /// Whether to use AND logic for multiple keywords (default: OR)
    pub use_and_logic: bool,
}

impl Default for KeywordTableIndexConfig {
    fn default() -> Self {
        Self {
            show_progress: false,
            max_nodes: None,
            deduplicate: true,
            min_content_length: 10,
            max_results: 100,
            score_threshold: Some(0.01),
            use_and_logic: false,
        }
    }
}

/// Keyword Table Index.
///
/// This index uses keyword extraction and inverted indexing for efficient
/// keyword-based document retrieval, similar to LlamaIndex's KeywordTableIndex.
///
/// # Features
///
/// - **Inverted Index**: Maps keywords to documents for fast lookup
/// - **Keyword Extraction**: Automatic keyword extraction from documents
/// - **Flexible Search**: Supports both AND and OR logic for multiple keywords
/// - **Relevance Scoring**: TF-IDF-like scoring for result ranking
/// - **LlamaIndex Compatibility**: Complete API compatibility
///
/// # Use Cases
///
/// - **Exact Term Matching**: When you need precise keyword matching
/// - **Technical Documentation**: For searching specific terms and concepts
/// - **Legal Documents**: For finding specific clauses and terminology
/// - **Fast Filtering**: Quick initial filtering before semantic search
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_query::indices::{KeywordTableIndex, KeywordTableIndexConfig};
/// use cheungfun_integrations::InMemoryKeywordStore;
/// use cheungfun_core::types::Document;
/// use std::sync::Arc;
///
/// # async fn example() -> cheungfun_core::Result<()> {
/// let keyword_store = Arc::new(InMemoryKeywordStore::new());
/// let config = KeywordTableIndexConfig::default();
///
/// let documents = vec![
///     Document::new("Rust is a systems programming language"),
///     Document::new("Python is great for data science"),
/// ];
///
/// let index = KeywordTableIndex::from_documents(
///     documents,
///     keyword_store,
///     None,
///     config
/// ).await?;
///
/// let retriever = index.as_retriever();
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct KeywordTableIndex {
    /// Keyword store for inverted index
    keyword_store: Arc<dyn KeywordStore>,

    /// All nodes stored in the index (for retrieval)
    nodes: HashMap<Uuid, Node>,

    /// Keyword extractor
    keyword_extractor: Box<dyn KeywordExtractor>,

    /// Configuration for the index
    config: KeywordTableIndexConfig,

    /// Statistics about the index
    stats: KeywordTableIndexStats,
}

/// Statistics for KeywordTableIndex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordTableIndexStats {
    /// Total number of nodes in the index
    pub total_nodes: usize,

    /// Total number of unique keywords
    pub total_keywords: usize,

    /// Total keyword-node mappings
    pub total_mappings: usize,

    /// Average keywords per node
    pub avg_keywords_per_node: f64,

    /// Average content length per node
    pub avg_content_length: f64,

    /// Number of unique documents
    pub unique_documents: usize,

    /// Index creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Default for KeywordTableIndexStats {
    fn default() -> Self {
        Self {
            total_nodes: 0,
            total_keywords: 0,
            total_mappings: 0,
            avg_keywords_per_node: 0.0,
            avg_content_length: 0.0,
            unique_documents: 0,
            created_at: chrono::Utc::now(),
        }
    }
}

impl KeywordTableIndex {
    /// Create a new KeywordTableIndex.
    ///
    /// # Arguments
    ///
    /// * `keyword_store` - Store for keyword-to-node mappings
    /// * `keyword_extractor` - Optional custom keyword extractor
    /// * `config` - Configuration for the index
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cheungfun_query::indices::{KeywordTableIndex, KeywordTableIndexConfig};
    /// use cheungfun_integrations::InMemoryKeywordStore;
    /// use std::sync::Arc;
    ///
    /// let keyword_store = Arc::new(InMemoryKeywordStore::new());
    /// let config = KeywordTableIndexConfig::default();
    /// let index = KeywordTableIndex::new(keyword_store, None, config);
    /// ```
    pub fn new(
        keyword_store: Arc<dyn KeywordStore>,
        keyword_extractor: Option<Box<dyn KeywordExtractor>>,
        config: KeywordTableIndexConfig,
    ) -> Self {
        let keyword_extractor =
            keyword_extractor.unwrap_or_else(|| Box::new(SimpleKeywordExtractor::with_defaults()));

        Self {
            keyword_store,
            nodes: HashMap::new(),
            keyword_extractor,
            config,
            stats: KeywordTableIndexStats::default(),
        }
    }

    /// Create a KeywordTableIndex with default configuration.
    pub fn with_defaults(keyword_store: Arc<dyn KeywordStore>) -> Self {
        Self::new(keyword_store, None, KeywordTableIndexConfig::default())
    }

    /// Create a KeywordTableIndex from documents.
    ///
    /// This is the primary constructor that follows LlamaIndex's `from_documents` pattern.
    ///
    /// # Arguments
    ///
    /// * `documents` - Documents to index
    /// * `keyword_store` - Store for keyword-to-node mappings
    /// * `keyword_extractor` - Optional custom keyword extractor
    /// * `config` - Optional configuration
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cheungfun_query::indices::KeywordTableIndex;
    /// use cheungfun_integrations::InMemoryKeywordStore;
    /// use cheungfun_core::types::Document;
    /// use std::sync::Arc;
    ///
    /// # async fn example() -> cheungfun_core::Result<()> {
    /// let keyword_store = Arc::new(InMemoryKeywordStore::new());
    /// let documents = vec![
    ///     Document::new("Document 1 content"),
    ///     Document::new("Document 2 content"),
    /// ];
    ///
    /// let index = KeywordTableIndex::from_documents(
    ///     documents,
    ///     keyword_store,
    ///     None,
    ///     None
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn from_documents(
        documents: Vec<Document>,
        keyword_store: Arc<dyn KeywordStore>,
        keyword_extractor: Option<Box<dyn KeywordExtractor>>,
        config: Option<KeywordTableIndexConfig>,
    ) -> Result<Self> {
        let mut index = Self::new(keyword_store, keyword_extractor, config.unwrap_or_default());
        index.insert_documents(documents).await?;
        Ok(index)
    }

    /// Create a KeywordTableIndex from nodes.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Nodes to index
    /// * `keyword_store` - Store for keyword-to-node mappings
    /// * `keyword_extractor` - Optional custom keyword extractor
    /// * `config` - Optional configuration
    pub async fn from_nodes(
        nodes: Vec<Node>,
        keyword_store: Arc<dyn KeywordStore>,
        keyword_extractor: Option<Box<dyn KeywordExtractor>>,
        config: Option<KeywordTableIndexConfig>,
    ) -> Result<Self> {
        let mut index = Self::new(keyword_store, keyword_extractor, config.unwrap_or_default());
        index.insert_nodes(nodes).await?;
        Ok(index)
    }

    /// Insert documents into the index.
    ///
    /// # Arguments
    ///
    /// * `documents` - Documents to insert
    pub async fn insert_documents(&mut self, documents: Vec<Document>) -> Result<()> {
        info!(
            "Inserting {} documents into KeywordTableIndex",
            documents.len()
        );

        // Convert documents to nodes (simple chunking)
        let mut nodes = Vec::new();
        for document in documents {
            // For KeywordTableIndex, we can use simple chunking or store documents as-is
            // Here we'll create one node per document for simplicity
            let chunk_info = cheungfun_core::ChunkInfo {
                start_char_idx: Some(0),
                end_char_idx: Some(document.content.len()),
                chunk_index: 0,
            };

            let mut node = Node::new(document.content.clone(), document.id, chunk_info);

            // Add document metadata to node
            for (key, value) in document.metadata {
                node = node.with_metadata(key, value);
            }

            nodes.push(node);
        }

        self.insert_nodes(nodes).await
    }

    /// Insert nodes into the index.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Nodes to insert
    pub async fn insert_nodes(&mut self, nodes: Vec<Node>) -> Result<()> {
        let initial_count = self.nodes.len();

        for node in nodes {
            // Filter by minimum content length
            if node.content.len() < self.config.min_content_length {
                debug!("Skipping node with content length {}", node.content.len());
                continue;
            }

            // Deduplicate if enabled
            if self.config.deduplicate {
                let content_hash = self.calculate_content_hash(&node.content);
                if self
                    .nodes
                    .values()
                    .any(|existing| self.calculate_content_hash(&existing.content) == content_hash)
                {
                    debug!("Skipping duplicate node");
                    continue;
                }
            }

            // Check max nodes limit
            if let Some(max_nodes) = self.config.max_nodes {
                if self.nodes.len() >= max_nodes {
                    debug!("Reached maximum nodes limit: {}", max_nodes);
                    break;
                }
            }

            // Extract keywords from the node
            let keywords = self.keyword_extractor.extract_keywords(&node.content)?;

            if !keywords.is_empty() {
                // Store the node
                let node_id = node.id;
                self.nodes.insert(node_id, node);

                // Add keywords to the keyword store
                self.keyword_store.add_keywords(node_id, keywords).await?;
            } else {
                debug!("Skipping node with no extractable keywords");
            }
        }

        let added_count = self.nodes.len() - initial_count;
        info!("Added {} nodes to KeywordTableIndex", added_count);

        // Update statistics
        self.update_stats().await?;

        Ok(())
    }

    /// Get all nodes in the index.
    pub fn nodes(&self) -> &HashMap<Uuid, Node> {
        &self.nodes
    }

    /// Get the number of nodes in the index.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get index statistics.
    pub fn stats(&self) -> &KeywordTableIndexStats {
        &self.stats
    }

    /// Get the keyword store.
    pub fn keyword_store(&self) -> &Arc<dyn KeywordStore> {
        &self.keyword_store
    }

    /// Create a retriever for this index.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cheungfun_query::indices::KeywordTableIndex;
    /// use cheungfun_integrations::InMemoryKeywordStore;
    /// use cheungfun_core::traits::Retriever;
    /// use std::sync::Arc;
    ///
    /// # async fn example() -> cheungfun_core::Result<()> {
    /// let keyword_store = Arc::new(InMemoryKeywordStore::new());
    /// let index = KeywordTableIndex::with_defaults(keyword_store);
    /// let retriever = index.as_retriever();
    ///
    /// // Use retriever for queries
    /// // let results = retriever.retrieve(&query).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn as_retriever(&self) -> KeywordTableRetriever {
        KeywordTableRetriever::new(
            self.keyword_store.clone(),
            self.nodes.clone(),
            self.keyword_extractor.config().clone(),
            self.config.clone(),
        )
    }

    /// Calculate a simple hash for content deduplication.
    fn calculate_content_hash(&self, content: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }

    /// Update index statistics.
    async fn update_stats(&mut self) -> Result<()> {
        let total_nodes = self.nodes.len();
        let total_content_length: usize = self.nodes.values().map(|node| node.content.len()).sum();

        let avg_content_length = if total_nodes > 0 {
            total_content_length as f64 / total_nodes as f64
        } else {
            0.0
        };

        // Get keyword store stats
        let keyword_stats = self.keyword_store.stats().await?;

        // Count unique documents (simplified - based on metadata doc_id if available)
        let unique_documents = self
            .nodes
            .values()
            .filter_map(|node| node.metadata.get("doc_id"))
            .collect::<std::collections::HashSet<_>>()
            .len()
            .max(1); // At least 1 if we have nodes

        self.stats = KeywordTableIndexStats {
            total_nodes,
            total_keywords: keyword_stats.total_keywords,
            total_mappings: keyword_stats.total_mappings,
            avg_keywords_per_node: keyword_stats.avg_keywords_per_node,
            avg_content_length,
            unique_documents,
            created_at: self.stats.created_at, // Preserve original creation time
        };

        Ok(())
    }
}

/// Retriever for KeywordTableIndex.
///
/// This retriever uses keyword extraction and inverted index lookup
/// to find relevant documents based on keyword matching.
#[derive(Debug)]
pub struct KeywordTableRetriever {
    /// Reference to the keyword store
    keyword_store: Arc<dyn KeywordStore>,

    /// All nodes in the index
    nodes: HashMap<Uuid, Node>,

    /// Keyword extractor for query processing
    keyword_extractor: Box<dyn KeywordExtractor>,

    /// Configuration
    config: KeywordTableIndexConfig,
}

impl KeywordTableRetriever {
    /// Create a new KeywordTableRetriever.
    pub fn new(
        keyword_store: Arc<dyn KeywordStore>,
        nodes: HashMap<Uuid, Node>,
        extractor_config: cheungfun_core::traits::KeywordExtractionConfig,
        config: KeywordTableIndexConfig,
    ) -> Self {
        let keyword_extractor = Box::new(SimpleKeywordExtractor::new(extractor_config));

        Self {
            keyword_store,
            nodes,
            keyword_extractor,
            config,
        }
    }

    /// Get the underlying keyword store.
    pub fn keyword_store(&self) -> &Arc<dyn KeywordStore> {
        &self.keyword_store
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[async_trait]
impl Retriever for KeywordTableRetriever {
    #[instrument(skip(self), fields(index_size = self.nodes.len()))]
    async fn retrieve(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        info!(
            "Starting KeywordTableIndex retrieval for query: {} (index size: {})",
            query.text,
            self.nodes.len()
        );

        if self.nodes.is_empty() {
            debug!("Index is empty, returning no results");
            return Ok(Vec::new());
        }

        // Extract keywords from the query
        let query_keywords = self.keyword_extractor.extract_keywords(&query.text)?;

        if query_keywords.is_empty() {
            debug!("No keywords extracted from query, returning no results");
            return Ok(Vec::new());
        }

        let keyword_list: Vec<String> = query_keywords.keys().cloned().collect();
        debug!("Extracted keywords from query: {:?}", keyword_list);

        // Search using keyword store
        let node_scores = if self.config.use_and_logic {
            // AND logic - all keywords must be present
            self.keyword_store
                .search_keywords_all(&keyword_list)
                .await?
        } else {
            // OR logic - any keyword can match
            self.keyword_store.search_keywords(&keyword_list).await?
        };

        debug!(
            "Found {} candidate nodes from keyword search",
            node_scores.len()
        );

        // Convert to ScoredNode and apply filters
        let mut scored_nodes = Vec::new();

        for (node_id, score) in node_scores {
            // Apply score threshold
            if let Some(threshold) = self.config.score_threshold {
                if score < threshold {
                    continue;
                }
            }

            // Get the actual node
            if let Some(node) = self.nodes.get(&node_id) {
                scored_nodes.push(ScoredNode::new(node.clone(), score));
            }
        }

        // Sort by score (descending)
        scored_nodes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply limits
        let max_results = query.top_k.min(self.config.max_results);
        if scored_nodes.len() > max_results {
            scored_nodes.truncate(max_results);
        }

        info!(
            "KeywordTableIndex retrieval completed: {} results returned",
            scored_nodes.len()
        );

        Ok(scored_nodes)
    }
}

/// Builder for KeywordTableIndex.
#[derive(Debug)]
pub struct KeywordTableIndexBuilder {
    keyword_store: Option<Arc<dyn KeywordStore>>,
    keyword_extractor: Option<Box<dyn KeywordExtractor>>,
    config: KeywordTableIndexConfig,
    documents: Vec<Document>,
    nodes: Vec<Node>,
}

impl Default for KeywordTableIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl KeywordTableIndexBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            keyword_store: None,
            keyword_extractor: None,
            config: KeywordTableIndexConfig::default(),
            documents: Vec::new(),
            nodes: Vec::new(),
        }
    }

    /// Set the keyword store.
    pub fn keyword_store(mut self, store: Arc<dyn KeywordStore>) -> Self {
        self.keyword_store = Some(store);
        self
    }

    /// Set the keyword extractor.
    pub fn keyword_extractor(mut self, extractor: Box<dyn KeywordExtractor>) -> Self {
        self.keyword_extractor = Some(extractor);
        self
    }

    /// Set whether to show progress.
    pub fn show_progress(mut self, show_progress: bool) -> Self {
        self.config.show_progress = show_progress;
        self
    }

    /// Set maximum number of nodes.
    pub fn max_nodes(mut self, max_nodes: usize) -> Self {
        self.config.max_nodes = Some(max_nodes);
        self
    }

    /// Set whether to deduplicate nodes.
    pub fn deduplicate(mut self, deduplicate: bool) -> Self {
        self.config.deduplicate = deduplicate;
        self
    }

    /// Set minimum content length.
    pub fn min_content_length(mut self, min_length: usize) -> Self {
        self.config.min_content_length = min_length;
        self
    }

    /// Set maximum results.
    pub fn max_results(mut self, max_results: usize) -> Self {
        self.config.max_results = max_results;
        self
    }

    /// Set score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.config.score_threshold = Some(threshold);
        self
    }

    /// Set whether to use AND logic for multiple keywords.
    pub fn use_and_logic(mut self, use_and: bool) -> Self {
        self.config.use_and_logic = use_and;
        self
    }

    /// Add documents to index.
    pub fn documents(mut self, documents: Vec<Document>) -> Self {
        self.documents = documents;
        self
    }

    /// Add nodes to index.
    pub fn nodes(mut self, nodes: Vec<Node>) -> Self {
        self.nodes = nodes;
        self
    }

    /// Build the KeywordTableIndex.
    pub async fn build(self) -> Result<KeywordTableIndex> {
        let keyword_store = self.keyword_store.ok_or_else(|| {
            cheungfun_core::CheungfunError::configuration("KeywordStore is required")
        })?;

        let mut index = KeywordTableIndex::new(keyword_store, self.keyword_extractor, self.config);

        if !self.documents.is_empty() {
            index.insert_documents(self.documents).await?;
        }

        if !self.nodes.is_empty() {
            index.insert_nodes(self.nodes).await?;
        }

        Ok(index)
    }
}
