//! Configuration types for postprocessors.
//!
//! This module provides configuration structures for various postprocessing
//! strategies, based on LlamaIndex's postprocessor design patterns.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for sentence-level embedding optimization.
///
/// Based on LlamaIndex's SentenceEmbeddingOptimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceEmbeddingConfig {
    /// Percentile cutoff for the top k sentences to use (0.0-1.0).
    /// If set to 0.5, only the top 50% most relevant sentences are kept.
    pub percentile_cutoff: Option<f32>,

    /// Threshold cutoff for similarity score (0.0-1.0).
    /// Only sentences with similarity above this threshold are kept.
    pub threshold_cutoff: Option<f32>,

    /// Number of sentences before retrieved sentence for context.
    pub context_before: Option<usize>,

    /// Number of sentences after retrieved sentence for context.
    pub context_after: Option<usize>,

    /// Maximum number of sentences to process per node.
    pub max_sentences_per_node: Option<usize>,
}

impl Default for SentenceEmbeddingConfig {
    fn default() -> Self {
        Self {
            percentile_cutoff: Some(0.7),     // Keep top 70% of sentences
            threshold_cutoff: Some(0.5),      // Minimum similarity of 0.5
            context_before: Some(1),          // Include 1 sentence before
            context_after: Some(1),           // Include 1 sentence after
            max_sentences_per_node: Some(50), // Limit processing for performance
        }
    }
}

/// Configuration for LLM-based compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCompressionConfig {
    /// Target compression ratio (0.0-1.0).
    /// 0.5 means compress to 50% of original length.
    pub target_ratio: f32,

    /// Temperature for LLM generation (0.0-2.0).
    pub temperature: f32,

    /// Maximum tokens for compressed output.
    pub max_tokens: Option<usize>,

    /// Timeout for LLM calls.
    pub timeout: Duration,

    /// Whether to preserve important entities and keywords.
    pub preserve_entities: bool,

    /// Custom compression prompt template.
    pub custom_prompt: Option<String>,
}

impl Default for LlmCompressionConfig {
    fn default() -> Self {
        Self {
            target_ratio: 0.7,
            temperature: 0.1,
            max_tokens: None,
            timeout: Duration::from_secs(30),
            preserve_entities: true,
            custom_prompt: None,
        }
    }
}

/// Configuration for keyword-based filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordFilterConfig {
    /// Required keywords that must be present.
    pub required_keywords: Vec<String>,

    /// Keywords that should be excluded.
    pub exclude_keywords: Vec<String>,

    /// Whether keyword matching is case sensitive.
    pub case_sensitive: bool,

    /// Minimum number of required keywords that must match.
    pub min_required_matches: usize,
}

impl Default for KeywordFilterConfig {
    fn default() -> Self {
        Self {
            required_keywords: vec![],
            exclude_keywords: vec![],
            case_sensitive: false,
            min_required_matches: 1,
        }
    }
}

/// Configuration for similarity-based filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityFilterConfig {
    /// Minimum similarity threshold (0.0-1.0).
    pub similarity_cutoff: f32,

    /// Maximum number of nodes to keep.
    pub max_nodes: Option<usize>,

    /// Whether to use query embedding for similarity calculation.
    pub use_query_embedding: bool,
}

impl Default for SimilarityFilterConfig {
    fn default() -> Self {
        Self {
            similarity_cutoff: 0.7,
            max_nodes: None,
            use_query_embedding: true,
        }
    }
}

/// Configuration for metadata-based filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataFilterConfig {
    /// Required metadata key-value pairs.
    pub required_metadata: std::collections::HashMap<String, String>,

    /// Excluded metadata key-value pairs.
    pub excluded_metadata: std::collections::HashMap<String, String>,

    /// Whether to require all metadata conditions to match.
    pub require_all: bool,
}

impl Default for MetadataFilterConfig {
    fn default() -> Self {
        Self {
            required_metadata: std::collections::HashMap::new(),
            excluded_metadata: std::collections::HashMap::new(),
            require_all: true,
        }
    }
}

/// Configuration for recency-based filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecencyFilterConfig {
    /// Maximum age of documents to include (in days).
    pub max_age_days: Option<u32>,

    /// Weight factor for recency in scoring (0.0-1.0).
    pub recency_weight: f32,

    /// Metadata field containing the timestamp.
    pub timestamp_field: String,
}

impl Default for RecencyFilterConfig {
    fn default() -> Self {
        Self {
            max_age_days: None,
            recency_weight: 0.1,
            timestamp_field: "created_at".to_string(),
        }
    }
}

/// Unified postprocessor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostprocessorConfig {
    /// Sentence embedding optimization config.
    pub sentence_embedding: Option<SentenceEmbeddingConfig>,

    /// LLM compression config.
    pub llm_compression: Option<LlmCompressionConfig>,

    /// Keyword filtering config.
    pub keyword_filter: Option<KeywordFilterConfig>,

    /// Similarity filtering config.
    pub similarity_filter: Option<SimilarityFilterConfig>,

    /// Metadata filtering config.
    pub metadata_filter: Option<MetadataFilterConfig>,

    /// Recency filtering config.
    pub recency_filter: Option<RecencyFilterConfig>,

    /// Whether to enable parallel processing.
    pub enable_parallel: bool,

    /// Maximum number of concurrent operations.
    pub max_concurrency: usize,
}

impl Default for PostprocessorConfig {
    fn default() -> Self {
        Self {
            sentence_embedding: None,
            llm_compression: None,
            keyword_filter: None,
            similarity_filter: None,
            metadata_filter: None,
            recency_filter: None,
            enable_parallel: true,
            max_concurrency: 4,
        }
    }
}

/// Compression quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetrics {
    /// Original content length in characters.
    pub original_length: usize,

    /// Compressed content length in characters.
    pub compressed_length: usize,

    /// Actual compression ratio achieved.
    pub compression_ratio: f32,

    /// Estimated relevance score (0.0-1.0).
    pub relevance_score: f32,

    /// Processing time in milliseconds.
    pub processing_time_ms: u64,

    /// Number of sentences processed.
    pub sentences_processed: usize,

    /// Number of sentences kept.
    pub sentences_kept: usize,
}

impl CompressionMetrics {
    /// Create new compression metrics.
    pub fn new(
        original_length: usize,
        compressed_length: usize,
        relevance_score: f32,
        processing_time_ms: u64,
        sentences_processed: usize,
        sentences_kept: usize,
    ) -> Self {
        let compression_ratio = if original_length > 0 {
            compressed_length as f32 / original_length as f32
        } else {
            1.0
        };

        Self {
            original_length,
            compressed_length,
            compression_ratio,
            relevance_score,
            processing_time_ms,
            sentences_processed,
            sentences_kept,
        }
    }

    /// Calculate compression efficiency (relevance per compression ratio).
    pub fn efficiency(&self) -> f32 {
        if self.compression_ratio > 0.0 {
            self.relevance_score / self.compression_ratio
        } else {
            0.0
        }
    }

    /// Calculate sentence retention rate.
    pub fn sentence_retention_rate(&self) -> f32 {
        if self.sentences_processed > 0 {
            self.sentences_kept as f32 / self.sentences_processed as f32
        } else {
            0.0
        }
    }
}
