//! Configuration for FastEmbed embedder.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Predefined model configurations for common use cases.
#[derive(Debug, Clone)]
pub enum ModelPreset {
    /// Fast and lightweight model, good for most applications
    /// Model: BAAI/bge-small-en-v1.5 (384 dimensions)
    Default,

    /// High quality English model
    /// Model: BAAI/bge-large-en-v1.5 (1024 dimensions)
    HighQuality,

    /// Multilingual support
    /// Model: BAAI/bge-m3 (1024 dimensions)
    Multilingual,

    /// Ultra-fast model for high-throughput scenarios
    /// Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
    Fast,

    /// Code-specific embeddings
    /// Model: microsoft/codebert-base (768 dimensions)
    Code,
}

impl ModelPreset {
    /// Get the model name for this preset.
    pub fn model_name(&self) -> &'static str {
        match self {
            ModelPreset::Default => "BAAI/bge-small-en-v1.5",
            ModelPreset::HighQuality => "BAAI/bge-large-en-v1.5",
            ModelPreset::Multilingual => "intfloat/multilingual-e5-base",
            ModelPreset::Fast => "sentence-transformers/all-MiniLM-L6-v2",
            ModelPreset::Code => "jinaai/jina-embeddings-v2-base-code",
        }
    }

    /// Get the expected embedding dimension for this preset.
    pub fn dimension(&self) -> usize {
        match self {
            ModelPreset::Default | ModelPreset::Fast => 384,
            ModelPreset::HighQuality => 1024,
            ModelPreset::Multilingual => 768, // multilingual-e5-base has 768 dimensions
            ModelPreset::Code => 768,
        }
    }

    /// Get a description of this preset.
    pub fn description(&self) -> &'static str {
        match self {
            ModelPreset::Default => "Balanced performance and quality for English text",
            ModelPreset::HighQuality => "Best quality for English text, larger model",
            ModelPreset::Multilingual => "Supports multiple languages",
            ModelPreset::Fast => "Fastest inference, good for high-throughput",
            ModelPreset::Code => "Optimized for source code and technical text",
        }
    }
}

/// Configuration for FastEmbed embedder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastEmbedConfig {
    /// Model name or preset
    pub model_name: String,

    /// Maximum sequence length for tokenization
    pub max_length: usize,

    /// Batch size for processing
    pub batch_size: usize,

    /// Cache directory for models (None = default cache)
    pub cache_dir: Option<PathBuf>,

    /// Number of threads for ONNX runtime (None = auto)
    pub threads: Option<usize>,

    /// Whether to show download progress
    pub show_progress: bool,
}

impl Default for FastEmbedConfig {
    fn default() -> Self {
        Self {
            model_name: ModelPreset::Default.model_name().to_string(),
            max_length: 512,
            batch_size: 32,
            cache_dir: None,
            threads: None,
            show_progress: true,
        }
    }
}

impl FastEmbedConfig {
    /// Create a new configuration with the specified model.
    pub fn new<S: Into<String>>(model_name: S) -> Self {
        Self {
            model_name: model_name.into(),
            ..Default::default()
        }
    }

    /// Create configuration from a preset.
    pub fn from_preset(preset: ModelPreset) -> Self {
        Self {
            model_name: preset.model_name().to_string(),
            ..Default::default()
        }
    }

    /// Set the maximum sequence length.
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the cache directory.
    pub fn with_cache_dir<P: Into<PathBuf>>(mut self, cache_dir: P) -> Self {
        self.cache_dir = Some(cache_dir.into());
        self
    }

    /// Set the number of threads.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = Some(threads);
        self
    }

    /// Disable progress bar.
    pub fn without_progress(mut self) -> Self {
        self.show_progress = false;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), super::FastEmbedError> {
        if self.model_name.is_empty() {
            return Err(super::FastEmbedError::Config {
                reason: "Model name cannot be empty".to_string(),
            });
        }

        if self.max_length == 0 {
            return Err(super::FastEmbedError::Config {
                reason: "Max length must be greater than 0".to_string(),
            });
        }

        if self.batch_size == 0 {
            return Err(super::FastEmbedError::Config {
                reason: "Batch size must be greater than 0".to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_presets() {
        assert_eq!(ModelPreset::Default.model_name(), "BAAI/bge-small-en-v1.5");
        assert_eq!(ModelPreset::Default.dimension(), 384);
        assert!(!ModelPreset::Default.description().is_empty());
    }

    #[test]
    fn test_config_builder() {
        let config = FastEmbedConfig::new("test-model")
            .with_max_length(256)
            .with_batch_size(16)
            .without_progress();

        assert_eq!(config.model_name, "test-model");
        assert_eq!(config.max_length, 256);
        assert_eq!(config.batch_size, 16);
        assert!(!config.show_progress);
    }

    #[test]
    fn test_config_validation() {
        let valid_config = FastEmbedConfig::default();
        assert!(valid_config.validate().is_ok());

        let invalid_config = FastEmbedConfig {
            model_name: "".to_string(),
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }
}
