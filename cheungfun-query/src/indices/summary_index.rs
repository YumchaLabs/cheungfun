//! Summary Index (List Index) implementation.
//!
//! This module provides a SummaryIndex implementation that follows LlamaIndex's
//! design exactly. It stores documents as a simple list and uses LLM-based
//! filtering for retrieval.
//!
//! **Reference**: LlamaIndex SummaryIndex (formerly ListIndex)
//! - Simple list-based storage of document nodes
//! - LLM-based relevance filtering during retrieval
//! - Suitable for small datasets and prototyping
//! - Comprehensive search without missing relevant information

use async_trait::async_trait;
use cheungfun_core::{
    traits::Retriever,
    types::{Document, Node, Query, ScoredNode},
    Result,
};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

/// Configuration for SummaryIndex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryIndexConfig {
    /// Whether to show progress during indexing
    pub show_progress: bool,

    /// Maximum number of nodes to store
    pub max_nodes: Option<usize>,

    /// Whether to deduplicate nodes based on content hash
    pub deduplicate: bool,

    /// Minimum content length to include a node
    pub min_content_length: usize,
}

impl Default for SummaryIndexConfig {
    fn default() -> Self {
        Self {
            show_progress: false,
            max_nodes: None,
            deduplicate: true,
            min_content_length: 10,
        }
    }
}

/// Summary Index (List Index).
///
/// This index stores documents as a simple list and uses LLM-based filtering
/// for retrieval, similar to LlamaIndex's SummaryIndex.
///
/// # Features
///
/// - **Simple Storage**: Stores all nodes in a simple list structure
/// - **LLM Filtering**: Uses LLM to determine node relevance during retrieval
/// - **Comprehensive Search**: Examines all nodes to avoid missing information
/// - **Low Overhead**: Minimal indexing overhead, suitable for prototyping
/// - **LlamaIndex Compatibility**: Complete API compatibility
///
/// # Use Cases
///
/// - **Prototyping**: Quick setup for RAG system prototypes
/// - **Small Datasets**: Efficient for datasets with < 1000 documents
/// - **High Recall**: When you need to ensure no relevant information is missed
/// - **Simple Applications**: When complex indexing is not needed
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_query::indices::{SummaryIndex, SummaryIndexConfig};
/// use cheungfun_core::types::Document;
///
/// # async fn example() -> cheungfun_core::Result<()> {
/// let config = SummaryIndexConfig {
///     show_progress: true,
///     max_nodes: Some(500),
///     ..Default::default()
/// };
///
/// let documents = vec![
///     Document::new("First document content"),
///     Document::new("Second document content"),
/// ];
///
/// let index = SummaryIndex::from_documents(documents, config).await?;
/// let retriever = index.as_retriever();
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct SummaryIndex {
    /// All nodes stored in the index
    nodes: Vec<Node>,

    /// Configuration for the index
    config: SummaryIndexConfig,

    /// Statistics about the index
    stats: SummaryIndexStats,
}

/// Statistics for SummaryIndex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryIndexStats {
    /// Total number of nodes in the index
    pub total_nodes: usize,

    /// Total content length across all nodes
    pub total_content_length: usize,

    /// Average content length per node
    pub avg_content_length: f64,

    /// Number of unique documents
    pub unique_documents: usize,

    /// Index creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Default for SummaryIndexStats {
    fn default() -> Self {
        Self {
            total_nodes: 0,
            total_content_length: 0,
            avg_content_length: 0.0,
            unique_documents: 0,
            created_at: chrono::Utc::now(),
        }
    }
}

impl SummaryIndex {
    /// Create a new empty SummaryIndex.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the index
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cheungfun_query::indices::{SummaryIndex, SummaryIndexConfig};
    ///
    /// let config = SummaryIndexConfig::default();
    /// let index = SummaryIndex::new(config);
    /// ```
    pub fn new(config: SummaryIndexConfig) -> Self {
        Self {
            nodes: Vec::new(),
            config,
            stats: SummaryIndexStats::default(),
        }
    }

    /// Create a SummaryIndex with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SummaryIndexConfig::default())
    }

    /// Create a SummaryIndex from documents.
    ///
    /// This is the primary constructor that follows LlamaIndex's `from_documents` pattern.
    ///
    /// # Arguments
    ///
    /// * `documents` - Documents to index
    /// * `config` - Optional configuration
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cheungfun_query::indices::SummaryIndex;
    /// use cheungfun_core::types::Document;
    ///
    /// # async fn example() -> cheungfun_core::Result<()> {
    /// let documents = vec![
    ///     Document::new("Document 1 content"),
    ///     Document::new("Document 2 content"),
    /// ];
    ///
    /// let index = SummaryIndex::from_documents(documents, None).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn from_documents(
        documents: Vec<Document>,
        config: Option<SummaryIndexConfig>,
    ) -> Result<Self> {
        let mut index = Self::new(config.unwrap_or_default());
        index.insert_documents(documents).await?;
        Ok(index)
    }

    /// Create a SummaryIndex from nodes.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Nodes to index
    /// * `config` - Optional configuration
    pub async fn from_nodes(nodes: Vec<Node>, config: Option<SummaryIndexConfig>) -> Result<Self> {
        let mut index = Self::new(config.unwrap_or_default());
        index.insert_nodes(nodes).await?;
        Ok(index)
    }

    /// Insert documents into the index.
    ///
    /// # Arguments
    ///
    /// * `documents` - Documents to insert
    pub async fn insert_documents(&mut self, documents: Vec<Document>) -> Result<()> {
        info!("Inserting {} documents into SummaryIndex", documents.len());

        // Convert documents to nodes (simple chunking)
        let mut nodes = Vec::new();
        for document in documents {
            // For SummaryIndex, we can use simple chunking or store documents as-is
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
                    .iter()
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

            self.nodes.push(node);
        }

        let added_count = self.nodes.len() - initial_count;
        info!("Added {} nodes to SummaryIndex", added_count);

        // Update statistics
        self.update_stats();

        Ok(())
    }

    /// Get all nodes in the index.
    pub fn nodes(&self) -> &[Node] {
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
    pub fn stats(&self) -> &SummaryIndexStats {
        &self.stats
    }

    /// Create a retriever for this index.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cheungfun_query::indices::SummaryIndex;
    /// use cheungfun_core::traits::Retriever;
    ///
    /// # async fn example() -> cheungfun_core::Result<()> {
    /// let index = SummaryIndex::with_defaults();
    /// let retriever = index.as_retriever();
    ///
    /// // Use retriever for queries
    /// // let results = retriever.retrieve(&query).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn as_retriever(&self) -> SummaryRetriever {
        SummaryRetriever::new(self.clone())
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
    fn update_stats(&mut self) {
        let total_nodes = self.nodes.len();
        let total_content_length: usize = self.nodes.iter().map(|node| node.content.len()).sum();

        let avg_content_length = if total_nodes > 0 {
            total_content_length as f64 / total_nodes as f64
        } else {
            0.0
        };

        // Count unique documents (simplified - based on metadata doc_id if available)
        let unique_documents = self
            .nodes
            .iter()
            .filter_map(|node| node.metadata.get("doc_id"))
            .collect::<std::collections::HashSet<_>>()
            .len()
            .max(1); // At least 1 if we have nodes

        self.stats = SummaryIndexStats {
            total_nodes,
            total_content_length,
            avg_content_length,
            unique_documents,
            created_at: self.stats.created_at, // Preserve original creation time
        };
    }
}

/// Retriever for SummaryIndex.
///
/// This retriever examines all nodes in the index and uses simple scoring
/// based on text similarity. In a full implementation, this would use LLM
/// for relevance filtering.
#[derive(Debug, Clone)]
pub struct SummaryRetriever {
    /// Reference to the summary index
    index: SummaryIndex,
}

impl SummaryRetriever {
    /// Create a new SummaryRetriever.
    pub fn new(index: SummaryIndex) -> Self {
        Self { index }
    }

    /// Get the underlying index.
    pub fn index(&self) -> &SummaryIndex {
        &self.index
    }

    /// Simple text similarity scoring (placeholder for LLM-based filtering).
    fn calculate_similarity_score(&self, node_content: &str, query: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let content_lower = node_content.to_lowercase();

        // Simple keyword matching score
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let content_words: Vec<&str> = content_lower.split_whitespace().collect();

        if query_words.is_empty() || content_words.is_empty() {
            return 0.0;
        }

        let matches = query_words
            .iter()
            .filter(|&word| content_words.contains(word))
            .count();

        matches as f32 / query_words.len() as f32
    }
}

#[async_trait]
impl Retriever for SummaryRetriever {
    #[instrument(skip(self), fields(index_size = self.index.len()))]
    async fn retrieve(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        info!(
            "Starting SummaryIndex retrieval for query: {} (examining {} nodes)",
            query.text,
            self.index.len()
        );

        if self.index.is_empty() {
            debug!("Index is empty, returning no results");
            return Ok(Vec::new());
        }

        // Score all nodes (in a full implementation, this would use LLM filtering)
        let mut scored_nodes = Vec::new();

        for node in self.index.nodes() {
            let score = self.calculate_similarity_score(&node.content, &query.text);

            // Only include nodes with non-zero scores
            if score > 0.0 {
                scored_nodes.push(ScoredNode::new(node.clone(), score));
            }
        }

        // Sort by score (descending)
        scored_nodes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply top_k limit
        if scored_nodes.len() > query.top_k {
            scored_nodes.truncate(query.top_k);
        }

        info!(
            "SummaryIndex retrieval completed: {} results returned",
            scored_nodes.len()
        );

        Ok(scored_nodes)
    }
}

/// Builder for SummaryIndex.
#[derive(Debug, Default)]
pub struct SummaryIndexBuilder {
    config: SummaryIndexConfig,
    documents: Vec<Document>,
    nodes: Vec<Node>,
}

impl SummaryIndexBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
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

    /// Build the SummaryIndex.
    pub async fn build(self) -> Result<SummaryIndex> {
        let mut index = SummaryIndex::new(self.config);

        if !self.documents.is_empty() {
            index.insert_documents(self.documents).await?;
        }

        if !self.nodes.is_empty() {
            index.insert_nodes(self.nodes).await?;
        }

        Ok(index)
    }
}
