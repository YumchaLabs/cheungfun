//! Embedding generation traits.
//!
//! This module defines traits for generating dense and sparse embeddings
//! from text content. Embeddings are used for semantic search and similarity
//! calculations in the RAG pipeline.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::Result;

/// Generates dense embeddings for text content.
///
/// Dense embeddings are fixed-size vectors that capture semantic meaning
/// of text. They are used for vector similarity search and semantic retrieval.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::Embedder;
/// use cheungfun_core::Result;
/// use async_trait::async_trait;
///
/// struct SimpleEmbedder {
///     dimension: usize,
/// }
///
/// #[async_trait]
/// impl Embedder for SimpleEmbedder {
///     async fn embed(&self, text: &str) -> Result<Vec<f32>> {
///         // Simple implementation that returns random embeddings
///         Ok(vec![0.1; self.dimension])
///     }
///
///     async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
///         let mut embeddings = Vec::new();
///         for _text in texts {
///             embeddings.push(vec![0.1; self.dimension]);
///         }
///         Ok(embeddings)
///     }
///
///     fn dimension(&self) -> usize {
///         self.dimension
///     }
///
///     fn model_name(&self) -> &str {
///         "simple-embedder"
///     }
/// }
/// ```
#[async_trait]
pub trait Embedder: Send + Sync + std::fmt::Debug {
    /// Generate embedding for a single text.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed
    ///
    /// # Returns
    ///
    /// A vector of floating-point numbers representing the embedding.
    /// The length should match the value returned by `dimension()`.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding generation fails due to model
    /// issues, network problems, or invalid input.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple texts in batch.
    ///
    /// This method is more efficient than calling `embed()` multiple times
    /// as it can leverage batch processing capabilities of the underlying
    /// embedding model.
    ///
    /// # Arguments
    ///
    /// * `texts` - Vector of text strings to embed
    ///
    /// # Returns
    ///
    /// A vector of embeddings, where each embedding corresponds to the
    /// text at the same index in the input vector.
    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>>;

    /// Get the dimension of embeddings produced by this embedder.
    ///
    /// All embeddings from this embedder should have this many dimensions.
    fn dimension(&self) -> usize;

    /// Get the name/identifier of the embedding model.
    ///
    /// This is used for logging, caching, and compatibility checks.
    fn model_name(&self) -> &str;

    /// Get a human-readable name for this embedder.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Check if the embedder is healthy and ready to generate embeddings.
    async fn health_check(&self) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Get metadata about the embedding model.
    ///
    /// This can include information like model version, training data,
    /// supported languages, etc.
    fn metadata(&self) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();
        metadata.insert("model_name".to_string(), self.model_name().into());
        metadata.insert("dimension".to_string(), self.dimension().into());
        metadata
    }

    /// Get the maximum text length this embedder can handle.
    fn max_text_length(&self) -> Option<usize> {
        // Default implementation has no limit
        None
    }

    /// Preprocess text before embedding.
    ///
    /// This method can be overridden to implement custom text preprocessing
    /// such as normalization, truncation, or cleaning.
    fn preprocess_text(&self, text: &str) -> String {
        // Default implementation returns text as-is
        text.to_string()
    }
}

/// Generates sparse embeddings for keyword/hybrid search.
///
/// Sparse embeddings map tokens to weights and are used for keyword-based
/// retrieval methods like BM25 or SPLADE. They complement dense embeddings
/// in hybrid search scenarios.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::SparseEmbedder;
/// use cheungfun_core::Result;
/// use async_trait::async_trait;
/// use std::collections::HashMap;
///
/// struct SimpleSparseEmbedder;
///
/// #[async_trait]
/// impl SparseEmbedder for SimpleSparseEmbedder {
///     async fn embed_sparse(&self, text: &str) -> Result<HashMap<u32, f32>> {
///         let mut embedding = HashMap::new();
///         // Simple word-based sparse embedding
///         for (i, word) in text.split_whitespace().enumerate() {
///             embedding.insert(i as u32, 1.0);
///         }
///         Ok(embedding)
///     }
///
///     async fn embed_sparse_batch(&self, texts: Vec<&str>) -> Result<Vec<HashMap<u32, f32>>> {
///         let mut embeddings = Vec::new();
///         for text in texts {
///             embeddings.push(self.embed_sparse(text).await?);
///         }
///         Ok(embeddings)
///     }
/// }
/// ```
#[async_trait]
pub trait SparseEmbedder: Send + Sync + std::fmt::Debug {
    /// Generate sparse embedding for a single text.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed
    ///
    /// # Returns
    ///
    /// A HashMap mapping token IDs to their weights. Only non-zero
    /// weights should be included to maintain sparsity.
    async fn embed_sparse(&self, text: &str) -> Result<HashMap<u32, f32>>;

    /// Generate sparse embeddings for multiple texts in batch.
    ///
    /// # Arguments
    ///
    /// * `texts` - Vector of text strings to embed
    ///
    /// # Returns
    ///
    /// A vector of sparse embeddings corresponding to the input texts.
    async fn embed_sparse_batch(&self, texts: Vec<&str>) -> Result<Vec<HashMap<u32, f32>>>;

    /// Get a human-readable name for this embedder.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Get the vocabulary size of this sparse embedder.
    fn vocab_size(&self) -> Option<usize> {
        // Default implementation has no known vocabulary size
        None
    }

    /// Check if the embedder is healthy and ready to generate embeddings.
    async fn health_check(&self) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Get metadata about the sparse embedding model.
    fn metadata(&self) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();
        if let Some(vocab_size) = self.vocab_size() {
            metadata.insert("vocab_size".to_string(), vocab_size.into());
        }
        metadata
    }
}

/// Configuration for embedding operations.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Maximum batch size for embedding operations.
    pub batch_size: Option<usize>,

    /// Timeout for embedding operations in seconds.
    pub timeout_seconds: Option<u64>,

    /// Number of retry attempts for failed embeddings.
    pub max_retries: Option<usize>,

    /// Whether to normalize embeddings to unit length.
    pub normalize: bool,

    /// Additional model-specific configuration.
    pub model_config: HashMap<String, serde_json::Value>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            batch_size: Some(32),
            timeout_seconds: Some(30),
            max_retries: Some(3),
            normalize: false,
            model_config: HashMap::new(),
        }
    }
}

/// Statistics about embedding operations.
#[derive(Debug, Clone)]
pub struct EmbeddingStats {
    /// Number of texts embedded.
    pub texts_embedded: usize,

    /// Number of embedding operations that failed.
    pub embeddings_failed: usize,

    /// Total time taken for embedding operations.
    pub duration: std::time::Duration,

    /// Average time per embedding.
    pub avg_time_per_embedding: std::time::Duration,

    /// Total number of tokens processed.
    pub tokens_processed: Option<usize>,

    /// List of errors encountered.
    pub errors: Vec<String>,

    /// Additional statistics.
    pub additional_stats: HashMap<String, serde_json::Value>,
}

impl EmbeddingStats {
    /// Create new embedding statistics.
    pub fn new() -> Self {
        Self {
            texts_embedded: 0,
            embeddings_failed: 0,
            duration: std::time::Duration::ZERO,
            avg_time_per_embedding: std::time::Duration::ZERO,
            tokens_processed: None,
            errors: Vec::new(),
            additional_stats: HashMap::new(),
        }
    }

    /// Calculate the success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        let total = self.texts_embedded + self.embeddings_failed;
        if total == 0 {
            0.0
        } else {
            (self.texts_embedded as f64 / total as f64) * 100.0
        }
    }

    /// Update average time per embedding.
    pub fn update_avg_time(&mut self) {
        let total = self.texts_embedded + self.embeddings_failed;
        if total > 0 {
            self.avg_time_per_embedding = self.duration / total as u32;
        }
    }

    /// Calculate tokens per second if token count is available.
    pub fn tokens_per_second(&self) -> Option<f64> {
        if let Some(tokens) = self.tokens_processed {
            if !self.duration.is_zero() {
                Some(tokens as f64 / self.duration.as_secs_f64())
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl Default for EmbeddingStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_stats() {
        let mut stats = EmbeddingStats::new();
        stats.texts_embedded = 95;
        stats.embeddings_failed = 5;
        stats.duration = std::time::Duration::from_secs(10);
        stats.tokens_processed = Some(1000);
        stats.update_avg_time();

        assert_eq!(stats.success_rate(), 95.0);
        assert_eq!(
            stats.avg_time_per_embedding,
            std::time::Duration::from_millis(100)
        );
        assert_eq!(stats.tokens_per_second(), Some(100.0));
    }

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.batch_size, Some(32));
        assert_eq!(config.timeout_seconds, Some(30));
        assert_eq!(config.max_retries, Some(3));
        assert!(!config.normalize);
    }
}
