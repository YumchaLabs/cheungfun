//! Configuration for indexing and query pipelines.
//!
//! This module provides configuration structures for complete RAG pipelines,
//! combining embedder, storage, and LLM configurations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{EmbedderConfig, LlmConfig, VectorStoreConfig};
use crate::Result;

/// Configuration for indexing pipelines.
///
/// This configuration defines how documents are processed, embedded,
/// and stored in the vector database.
///
/// # Examples
///
/// ```rust
/// use cheungfun_core::config::{IndexingPipelineConfig, EmbedderConfig, VectorStoreConfig};
///
/// let config = IndexingPipelineConfig::new(
///     EmbedderConfig::candle("sentence-transformers/all-MiniLM-L6-v2", "cpu"),
///     VectorStoreConfig::memory(384)
/// )
/// .with_batch_size(50)
/// .with_chunk_size(512)
/// .with_chunk_overlap(50);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexingPipelineConfig {
    /// Embedder configuration.
    pub embedder: EmbedderConfig,

    /// Vector store configuration.
    pub vector_store: VectorStoreConfig,

    /// Batch size for processing documents.
    pub batch_size: usize,

    /// Chunk size for text splitting (in characters).
    pub chunk_size: usize,

    /// Overlap between chunks (in characters).
    pub chunk_overlap: usize,

    /// Maximum number of concurrent operations.
    pub concurrency: usize,

    /// Whether to continue processing if some documents fail.
    pub continue_on_error: bool,

    /// Additional pipeline-specific configuration.
    pub additional_config: HashMap<String, serde_json::Value>,
}

impl IndexingPipelineConfig {
    /// Create a new indexing pipeline configuration.
    #[must_use]
    pub fn new(embedder: EmbedderConfig, vector_store: VectorStoreConfig) -> Self {
        Self {
            embedder,
            vector_store,
            batch_size: 32,
            chunk_size: 1000,
            chunk_overlap: 200,
            concurrency: 4,
            continue_on_error: true,
            additional_config: HashMap::new(),
        }
    }

    /// Set the batch size.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the chunk size.
    #[must_use]
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set the chunk overlap.
    #[must_use]
    pub fn with_chunk_overlap(mut self, chunk_overlap: usize) -> Self {
        self.chunk_overlap = chunk_overlap;
        self
    }

    /// Set the concurrency level.
    #[must_use]
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Set whether to continue on error.
    #[must_use]
    pub fn with_continue_on_error(mut self, continue_on_error: bool) -> Self {
        self.continue_on_error = continue_on_error;
        self
    }

    /// Add additional configuration parameter.
    pub fn with_config<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.additional_config.insert(key.into(), value.into());
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        // Validate sub-configurations
        self.embedder.validate()?;
        self.vector_store.validate()?;

        // Validate pipeline-specific settings
        if self.batch_size == 0 {
            return Err(crate::CheungfunError::configuration(
                "Batch size must be greater than 0",
            ));
        }

        if self.chunk_size == 0 {
            return Err(crate::CheungfunError::configuration(
                "Chunk size must be greater than 0",
            ));
        }

        if self.chunk_overlap >= self.chunk_size {
            return Err(crate::CheungfunError::configuration(
                "Chunk overlap must be less than chunk size",
            ));
        }

        if self.concurrency == 0 {
            return Err(crate::CheungfunError::configuration(
                "Concurrency must be greater than 0",
            ));
        }

        // Check dimension compatibility
        let embedder_dim = match &self.embedder {
            EmbedderConfig::Candle {
                additional_config, ..
            } => additional_config
                .get("dimension")
                .and_then(serde_json::Value::as_u64)
                .map(|d| d as usize),
            _ => None,
        };

        if let Some(emb_dim) = embedder_dim {
            if emb_dim != self.vector_store.dimension() {
                return Err(crate::CheungfunError::configuration(format!(
                    "Embedder dimension ({}) does not match vector store dimension ({})",
                    emb_dim,
                    self.vector_store.dimension()
                )));
            }
        }

        Ok(())
    }
}

/// Configuration for query pipelines.
///
/// This configuration defines how queries are processed, how retrieval
/// is performed, and how responses are generated.
///
/// # Examples
///
/// ```rust
/// use cheungfun_core::config::{QueryPipelineConfig, EmbedderConfig, VectorStoreConfig, LlmConfig};
///
/// let config = QueryPipelineConfig::new(
///     EmbedderConfig::candle("sentence-transformers/all-MiniLM-L6-v2", "cpu"),
///     VectorStoreConfig::memory(384),
///     LlmConfig::openai("gpt-3.5-turbo", "your-api-key")
/// )
/// .with_top_k(10)
/// .with_similarity_threshold(0.7)
/// .with_max_tokens(1000);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryPipelineConfig {
    /// Embedder configuration for query embedding.
    pub embedder: EmbedderConfig,

    /// Vector store configuration for retrieval.
    pub vector_store: VectorStoreConfig,

    /// LLM configuration for response generation.
    pub llm: LlmConfig,

    /// Default number of results to retrieve.
    pub top_k: usize,

    /// Default similarity threshold for filtering results.
    pub similarity_threshold: Option<f32>,

    /// Maximum tokens for response generation.
    pub max_tokens: Option<usize>,

    /// Temperature for response generation.
    pub temperature: Option<f32>,

    /// System prompt template for response generation.
    pub system_prompt: Option<String>,

    /// Whether to include source citations in responses.
    pub include_citations: bool,

    /// Additional pipeline-specific configuration.
    pub additional_config: HashMap<String, serde_json::Value>,
}

impl QueryPipelineConfig {
    /// Create a new query pipeline configuration.
    #[must_use]
    pub fn new(embedder: EmbedderConfig, vector_store: VectorStoreConfig, llm: LlmConfig) -> Self {
        Self {
            embedder,
            vector_store,
            llm,
            top_k: 10,
            similarity_threshold: None,
            max_tokens: None,
            temperature: None,
            system_prompt: None,
            include_citations: false,
            additional_config: HashMap::new(),
        }
    }

    /// Set the default top-k value.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the similarity threshold.
    #[must_use]
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = Some(threshold);
        self
    }

    /// Set the maximum tokens for generation.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the temperature for generation.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the system prompt template.
    pub fn with_system_prompt<S: Into<String>>(mut self, system_prompt: S) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Set whether to include citations.
    #[must_use]
    pub fn with_citations(mut self, include_citations: bool) -> Self {
        self.include_citations = include_citations;
        self
    }

    /// Add additional configuration parameter.
    pub fn with_config<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.additional_config.insert(key.into(), value.into());
        self
    }

    /// Get the effective max tokens (from pipeline or LLM config).
    #[must_use]
    pub fn effective_max_tokens(&self) -> usize {
        self.max_tokens.or(self.llm.max_tokens).unwrap_or(1000)
    }

    /// Get the effective temperature (from pipeline or LLM config).
    #[must_use]
    pub fn effective_temperature(&self) -> f32 {
        self.temperature.or(self.llm.temperature).unwrap_or(0.7)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        // Validate sub-configurations
        self.embedder.validate()?;
        self.vector_store.validate()?;
        self.llm.validate()?;

        // Validate pipeline-specific settings
        if self.top_k == 0 {
            return Err(crate::CheungfunError::configuration(
                "Top-k must be greater than 0",
            ));
        }

        if let Some(threshold) = self.similarity_threshold {
            if !(-1.0..=1.0).contains(&threshold) {
                return Err(crate::CheungfunError::configuration(
                    "Similarity threshold must be between -1.0 and 1.0",
                ));
            }
        }

        if let Some(max_tokens) = self.max_tokens {
            if max_tokens == 0 {
                return Err(crate::CheungfunError::configuration(
                    "Max tokens must be greater than 0",
                ));
            }
        }

        if let Some(temp) = self.temperature {
            if !(0.0..=2.0).contains(&temp) {
                return Err(crate::CheungfunError::configuration(
                    "Temperature must be between 0.0 and 2.0",
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexing_pipeline_config() {
        let embedder = EmbedderConfig::candle("test-model", "cpu");
        let vector_store = VectorStoreConfig::memory(768);

        let config = IndexingPipelineConfig::new(embedder, vector_store)
            .with_batch_size(64)
            .with_chunk_size(2000)
            .with_chunk_overlap(100)
            .with_concurrency(8);

        assert_eq!(config.batch_size, 64);
        assert_eq!(config.chunk_size, 2000);
        assert_eq!(config.chunk_overlap, 100);
        assert_eq!(config.concurrency, 8);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_query_pipeline_config() {
        let embedder = EmbedderConfig::candle("test-model", "cpu");
        let vector_store = VectorStoreConfig::memory(768);
        let llm = LlmConfig::openai("gpt-3.5-turbo", "test-key");

        let config = QueryPipelineConfig::new(embedder, vector_store, llm)
            .with_top_k(15)
            .with_similarity_threshold(0.8)
            .with_max_tokens(1500)
            .with_temperature(0.9)
            .with_citations(true);

        assert_eq!(config.top_k, 15);
        assert_eq!(config.similarity_threshold, Some(0.8));
        assert_eq!(config.effective_max_tokens(), 1500);
        assert_eq!(config.effective_temperature(), 0.9);
        assert!(config.include_citations);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_errors() {
        let embedder = EmbedderConfig::candle("test-model", "cpu");
        let vector_store = VectorStoreConfig::memory(768);

        // Invalid batch size
        let invalid_batch =
            IndexingPipelineConfig::new(embedder.clone(), vector_store.clone()).with_batch_size(0);
        assert!(invalid_batch.validate().is_err());

        // Invalid chunk overlap
        let invalid_overlap = IndexingPipelineConfig::new(embedder.clone(), vector_store.clone())
            .with_chunk_size(100)
            .with_chunk_overlap(150);
        assert!(invalid_overlap.validate().is_err());

        // Query pipeline validation
        let llm = LlmConfig::openai("gpt-3.5-turbo", "test-key");
        let invalid_top_k = QueryPipelineConfig::new(embedder, vector_store, llm).with_top_k(0);
        assert!(invalid_top_k.validate().is_err());
    }

    #[test]
    fn test_effective_values() {
        let embedder = EmbedderConfig::candle("test-model", "cpu");
        let vector_store = VectorStoreConfig::memory(768);
        let llm = LlmConfig::openai("gpt-3.5-turbo", "test-key")
            .with_max_tokens(2000)
            .with_temperature(0.5);

        let config = QueryPipelineConfig::new(embedder, vector_store, llm);

        // Should use LLM config values when pipeline values are not set
        assert_eq!(config.effective_max_tokens(), 2000);
        assert_eq!(config.effective_temperature(), 0.5);

        // Pipeline values should override LLM values
        let config_with_overrides = config.with_max_tokens(1500).with_temperature(0.8);

        assert_eq!(config_with_overrides.effective_max_tokens(), 1500);
        assert_eq!(config_with_overrides.effective_temperature(), 0.8);
    }
}
