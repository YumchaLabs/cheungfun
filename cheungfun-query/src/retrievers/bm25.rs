//! BM25 Retriever - Standalone BM25-based text retrieval.
//!
//! This module implements a dedicated BM25 retriever that provides keyword-based
//! search functionality, similar to LlamaIndex's BM25Retriever.
//!
//! **Reference**: LlamaIndex BM25Retriever
//! - Supports configurable BM25 parameters (k1, b)
//! - Handles document preprocessing and tokenization
//! - Provides similarity scoring based on BM25 algorithm

use async_trait::async_trait;
use cheungfun_core::{
    traits::Retriever,
    types::{Query, ScoredNode},
    Node, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, instrument};

/// BM25 algorithm parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Params {
    /// Term frequency saturation parameter (typically 1.2)
    pub k1: f32,
    
    /// Length normalization parameter (typically 0.75)
    pub b: f32,
}

impl Default for BM25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// Configuration for BM25Retriever.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Config {
    /// BM25 algorithm parameters
    pub params: BM25Params,
    
    /// Whether to perform case-sensitive matching
    pub case_sensitive: bool,
    
    /// Minimum word length to consider
    pub min_word_length: usize,
    
    /// Maximum number of results to return
    pub max_results: usize,
    
    /// Minimum BM25 score threshold
    pub score_threshold: Option<f32>,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self {
            params: BM25Params::default(),
            case_sensitive: false,
            min_word_length: 2,
            max_results: 100,
            score_threshold: None,
        }
    }
}

/// Document statistics for BM25 calculation.
#[derive(Debug, Clone)]
struct DocumentStats {
    /// Document length (number of terms)
    length: usize,
    
    /// Term frequencies in the document
    term_frequencies: HashMap<String, usize>,
}

/// BM25 index for efficient retrieval.
#[derive(Debug)]
struct BM25Index {
    /// Document statistics for each document
    documents: HashMap<String, DocumentStats>,
    
    /// Inverse document frequency for each term
    idf_scores: HashMap<String, f32>,
    
    /// Average document length
    avg_doc_length: f32,
    
    /// Total number of documents
    total_docs: usize,
}

impl BM25Index {
    /// Create a new empty BM25 index.
    fn new() -> Self {
        Self {
            documents: HashMap::new(),
            idf_scores: HashMap::new(),
            avg_doc_length: 0.0,
            total_docs: 0,
        }
    }

    /// Add a document to the index.
    fn add_document(&mut self, doc_id: String, content: &str, config: &BM25Config) {
        let terms = Self::tokenize(content, config);
        let term_frequencies = Self::calculate_term_frequencies(&terms);
        
        let doc_stats = DocumentStats {
            length: terms.len(),
            term_frequencies,
        };
        
        self.documents.insert(doc_id, doc_stats);
        self.total_docs += 1;
        
        // Update average document length
        let total_length: usize = self.documents.values().map(|doc| doc.length).sum();
        self.avg_doc_length = total_length as f32 / self.total_docs as f32;
        
        // Recalculate IDF scores
        self.calculate_idf_scores();
    }

    /// Tokenize text into terms.
    fn tokenize(text: &str, config: &BM25Config) -> Vec<String> {
        let text = if config.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };
        
        text.split_whitespace()
            .map(|word| {
                // Remove punctuation
                word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|word| word.len() >= config.min_word_length)
            .collect()
    }

    /// Calculate term frequencies for a list of terms.
    fn calculate_term_frequencies(terms: &[String]) -> HashMap<String, usize> {
        let mut frequencies = HashMap::new();
        for term in terms {
            *frequencies.entry(term.clone()).or_insert(0) += 1;
        }
        frequencies
    }

    /// Calculate IDF scores for all terms.
    fn calculate_idf_scores(&mut self) {
        let mut term_doc_counts: HashMap<String, usize> = HashMap::new();
        
        // Count documents containing each term
        for doc_stats in self.documents.values() {
            for term in doc_stats.term_frequencies.keys() {
                *term_doc_counts.entry(term.clone()).or_insert(0) += 1;
            }
        }
        
        // Calculate IDF scores
        self.idf_scores.clear();
        for (term, doc_count) in term_doc_counts {
            let idf = ((self.total_docs as f32 - doc_count as f32 + 0.5) / (doc_count as f32 + 0.5)).ln();
            self.idf_scores.insert(term, idf.max(0.0)); // Ensure non-negative IDF
        }
    }

    /// Calculate BM25 score for a query against a document.
    fn calculate_bm25_score(
        &self,
        query_terms: &[String],
        doc_id: &str,
        params: &BM25Params,
    ) -> f32 {
        let doc_stats = match self.documents.get(doc_id) {
            Some(stats) => stats,
            None => return 0.0,
        };
        
        let mut score = 0.0;
        
        for term in query_terms {
            let tf = *doc_stats.term_frequencies.get(term).unwrap_or(&0) as f32;
            let idf = *self.idf_scores.get(term).unwrap_or(&0.0);
            
            if tf > 0.0 {
                let normalized_tf = (tf * (params.k1 + 1.0)) / 
                    (tf + params.k1 * (1.0 - params.b + params.b * (doc_stats.length as f32 / self.avg_doc_length)));
                
                score += idf * normalized_tf;
            }
        }
        
        score
    }

    /// Search for documents matching the query.
    fn search(&self, query: &str, config: &BM25Config) -> Vec<(String, f32)> {
        let query_terms = Self::tokenize(query, config);
        
        if query_terms.is_empty() {
            return Vec::new();
        }
        
        let mut results: Vec<(String, f32)> = self.documents
            .keys()
            .map(|doc_id| {
                let score = self.calculate_bm25_score(&query_terms, doc_id, &config.params);
                (doc_id.clone(), score)
            })
            .filter(|(_, score)| {
                if let Some(threshold) = config.score_threshold {
                    *score >= threshold
                } else {
                    *score > 0.0
                }
            })
            .collect();
        
        // Sort by score (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit results
        if results.len() > config.max_results {
            results.truncate(config.max_results);
        }
        
        results
    }
}

/// A retriever that uses BM25 algorithm for keyword-based search.
///
/// This retriever provides standalone BM25 functionality, similar to
/// LlamaIndex's BM25Retriever. It builds an internal BM25 index from
/// the provided nodes and performs efficient keyword-based retrieval.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_query::retrievers::{BM25Retriever, BM25Config};
/// use cheungfun_core::Node;
///
/// # async fn example() -> cheungfun_core::Result<()> {
/// let config = BM25Config::default();
/// let mut retriever = BM25Retriever::new(config);
///
/// // Add nodes to the index
/// retriever.add_nodes(nodes).await?;
///
/// // Search
/// let results = retriever.retrieve(&query).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct BM25Retriever {
    /// BM25 index
    index: BM25Index,
    
    /// Node storage (for returning full nodes)
    nodes: HashMap<String, Node>,
    
    /// Configuration
    config: BM25Config,
}

impl BM25Retriever {
    /// Create a new BM25Retriever.
    pub fn new(config: BM25Config) -> Self {
        Self {
            index: BM25Index::new(),
            nodes: HashMap::new(),
            config,
        }
    }

    /// Create a BM25Retriever with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BM25Config::default())
    }

    /// Create a BM25Retriever from a list of nodes.
    pub async fn from_nodes(nodes: Vec<Node>, config: BM25Config) -> Result<Self> {
        let mut retriever = Self::new(config);
        retriever.add_nodes(nodes).await?;
        Ok(retriever)
    }

    /// Add nodes to the BM25 index.
    pub async fn add_nodes(&mut self, nodes: Vec<Node>) -> Result<()> {
        for node in nodes {
            let doc_id = node.id.to_string();
            self.index.add_document(doc_id.clone(), &node.content, &self.config);
            self.nodes.insert(doc_id, node);
        }
        
        info!("Added {} nodes to BM25 index", self.nodes.len());
        Ok(())
    }

    /// Add a single node to the index.
    pub async fn add_node(&mut self, node: Node) -> Result<()> {
        let doc_id = node.id.to_string();
        self.index.add_document(doc_id.clone(), &node.content, &self.config);
        self.nodes.insert(doc_id, node);
        Ok(())
    }

    /// Get the number of indexed documents.
    pub fn document_count(&self) -> usize {
        self.nodes.len()
    }

    /// Clear the index.
    pub fn clear(&mut self) {
        self.index = BM25Index::new();
        self.nodes.clear();
    }
}

#[async_trait]
impl Retriever for BM25Retriever {
    #[instrument(skip(self), fields(documents = self.nodes.len()))]
    async fn retrieve(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        info!("Starting BM25 retrieval for query: {}", query.text);

        if self.nodes.is_empty() {
            debug!("No documents in BM25 index");
            return Ok(Vec::new());
        }

        // Search using BM25 index
        let search_results = self.index.search(&query.text, &self.config);
        
        debug!("BM25 search found {} results", search_results.len());

        // Convert to ScoredNode results
        let mut scored_nodes = Vec::new();
        for (doc_id, score) in search_results {
            if let Some(node) = self.nodes.get(&doc_id) {
                scored_nodes.push(ScoredNode::new(node.clone(), score));
            }
        }

        // Apply top_k limit
        if scored_nodes.len() > query.top_k {
            scored_nodes.truncate(query.top_k);
        }

        info!(
            "BM25 retrieval completed: {} results returned",
            scored_nodes.len()
        );

        Ok(scored_nodes)
    }
}

/// Builder for BM25Retriever.
#[derive(Debug, Default)]
pub struct BM25RetrieverBuilder {
    config: BM25Config,
    nodes: Vec<Node>,
}

impl BM25RetrieverBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set BM25 parameters.
    pub fn bm25_params(mut self, k1: f32, b: f32) -> Self {
        self.config.params = BM25Params { k1, b };
        self
    }

    /// Set case sensitivity.
    pub fn case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.config.case_sensitive = case_sensitive;
        self
    }

    /// Set minimum word length.
    pub fn min_word_length(mut self, min_length: usize) -> Self {
        self.config.min_word_length = min_length;
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

    /// Add nodes to index.
    pub fn nodes(mut self, nodes: Vec<Node>) -> Self {
        self.nodes = nodes;
        self
    }

    /// Build the BM25Retriever.
    pub async fn build(self) -> Result<BM25Retriever> {
        BM25Retriever::from_nodes(self.nodes, self.config).await
    }
}
