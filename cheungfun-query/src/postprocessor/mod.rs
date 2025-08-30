//! Node postprocessing components for query results.
//!
//! This module provides various postprocessing capabilities for retrieved nodes,
//! including compression, filtering, reranking, and other transformations.
//!
//! Based on LlamaIndex's postprocessor design and LangChain's contextual compression.

use async_trait::async_trait;
use cheungfun_core::{Result, ScoredNode};
use serde::{Deserialize, Serialize};

pub mod config;
pub mod keyword_filter;
pub mod metadata_filter;
pub mod sentence_optimizer;
pub mod similarity_filter;

pub use config::*;
pub use keyword_filter::*;
pub use metadata_filter::*;
pub use sentence_optimizer::*;
pub use similarity_filter::*;

/// Core trait for document compression.
///
/// Document compressors take retrieved nodes and compress their content
/// while preserving information relevant to the query.
#[async_trait]
pub trait DocumentCompressor: Send + Sync + std::fmt::Debug {
    /// Compress a list of scored nodes based on the query.
    ///
    /// # Arguments
    ///
    /// * `nodes` - The nodes to compress
    /// * `query` - The original query for context
    ///
    /// # Returns
    ///
    /// Compressed nodes with potentially modified content and scores.
    async fn compress(&self, nodes: Vec<ScoredNode>, query: &str) -> Result<Vec<ScoredNode>>;

    /// Get the name of this compressor for logging/debugging.
    fn name(&self) -> &'static str;
}

/// Core trait for node postprocessing.
///
/// Node postprocessors can filter, rerank, transform, or otherwise
/// modify retrieved nodes before they are used for generation.
#[async_trait]
pub trait NodePostprocessor: Send + Sync + std::fmt::Debug {
    /// Postprocess a list of scored nodes.
    ///
    /// # Arguments
    ///
    /// * `nodes` - The nodes to postprocess
    /// * `query` - The original query for context
    ///
    /// # Returns
    ///
    /// Postprocessed nodes, potentially filtered, reranked, or transformed.
    async fn postprocess(&self, nodes: Vec<ScoredNode>, query: &str) -> Result<Vec<ScoredNode>>;

    /// Get the name of this postprocessor for logging/debugging.
    fn name(&self) -> &'static str;
}

/// Compression strategy enumeration.
///
/// Defines different approaches to compressing retrieved content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// Use LLM to intelligently extract relevant portions
    LlmBased {
        /// Target compression ratio (0.0-1.0)
        target_ratio: f32,
        /// Temperature for LLM generation
        temperature: f32,
        /// Maximum tokens for compressed output
        max_tokens: Option<usize>,
    },
    /// Extract sentences containing query keywords
    KeywordBased {
        /// Minimum keyword matches per sentence
        min_matches: usize,
        /// Case sensitive matching
        case_sensitive: bool,
    },
    /// Keep only high-similarity chunks
    SimilarityBased {
        /// Minimum similarity threshold
        threshold: f32,
        /// Maximum chunks to keep
        max_chunks: usize,
    },
    /// Combine multiple strategies
    Hybrid {
        /// Primary strategy
        primary: Box<CompressionStrategy>,
        /// Secondary strategy
        secondary: Box<CompressionStrategy>,
        /// Weight for primary strategy (0.0-1.0)
        primary_weight: f32,
    },
}

impl Default for CompressionStrategy {
    fn default() -> Self {
        Self::LlmBased {
            target_ratio: 0.7,
            temperature: 0.1,
            max_tokens: None,
        }
    }
}

// CompressionMetrics is defined in config.rs

/// Utility functions for postprocessing.
pub mod utils {
    use std::collections::HashSet;

    /// Extract keywords from a query string.
    pub fn extract_keywords(query: &str) -> Vec<String> {
        query
            .to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 2) // Filter out very short words
            .map(|word| {
                word.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|word| !word.is_empty())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }

    /// Calculate keyword overlap between query and content.
    pub fn calculate_keyword_overlap(query: &str, content: &str) -> f32 {
        let query_keywords: HashSet<String> = extract_keywords(query).into_iter().collect();
        let content_keywords: HashSet<String> = extract_keywords(content).into_iter().collect();

        if query_keywords.is_empty() {
            return 0.0;
        }

        let intersection = query_keywords.intersection(&content_keywords).count();
        intersection as f32 / query_keywords.len() as f32
    }

    /// Split content into sentences for sentence-level processing.
    pub fn split_into_sentences(content: &str) -> Vec<String> {
        content
            .split(&['.', '!', '?'])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() > 10) // Filter out very short sentences
            .collect()
    }

    /// Calculate relevance score between query and content.
    pub fn calculate_relevance_score(query: &str, content: &str) -> f32 {
        // Simple implementation based on keyword overlap
        // In a real implementation, you might use more sophisticated methods
        calculate_keyword_overlap(query, content)
    }
}

#[cfg(test)]
mod tests {
    use super::utils::*;

    #[test]
    fn test_extract_keywords() {
        let query = "machine learning algorithms for data science";
        let keywords = extract_keywords(query);

        assert!(keywords.contains(&"machine".to_string()));
        assert!(keywords.contains(&"learning".to_string()));
        assert!(keywords.contains(&"algorithms".to_string()));
        assert!(keywords.contains(&"data".to_string()));
        assert!(keywords.contains(&"science".to_string()));
    }

    #[test]
    fn test_keyword_overlap() {
        let query = "machine learning algorithms";
        let content = "This text discusses machine learning and various algorithms used in AI.";

        let overlap = calculate_keyword_overlap(query, content);
        assert!(overlap > 0.5); // Should have good overlap
    }

    #[test]
    fn test_compression_metrics() {
        let metrics = crate::postprocessor::CompressionMetrics::new(1000, 300, 0.8, 150, 800, 600);

        assert_eq!(metrics.original_length, 1000);
        assert_eq!(metrics.compressed_length, 300);
        assert_eq!(metrics.compression_ratio, 0.3);
        assert_eq!(metrics.relevance_score, 0.8);
        assert!(metrics.efficiency() > 2.0); // 0.8 / 0.3 ≈ 2.67
    }
}
