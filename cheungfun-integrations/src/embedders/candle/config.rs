//! Configuration for Candle embedder.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Candle embedder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleEmbedderConfig {
    /// Model name or path (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    pub model_name: String,

    /// Model revision/branch to use (default: "main")
    pub revision: String,

    /// Maximum sequence length for tokenization
    pub max_length: usize,

    /// Whether to normalize embeddings to unit length
    pub normalize: bool,

    /// Batch size for processing
    pub batch_size: usize,

    /// Embedding dimension (auto-detected if not specified)
    pub dimension: Option<usize>,

    /// Device preference ("auto", "cpu", "cuda", "metal")
    pub device: String,

    /// Whether to use half precision (f16) for inference
    pub use_half_precision: bool,

    /// Local cache directory for models
    pub cache_dir: Option<String>,

    /// Whether to trust remote code in model
    pub trust_remote_code: bool,

    /// Additional model-specific configuration
    pub model_config: HashMap<String, serde_json::Value>,
}

impl Default for CandleEmbedderConfig {
    fn default() -> Self {
        Self {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            revision: "main".to_string(),
            max_length: 512,
            normalize: true,
            batch_size: 32,
            dimension: None, // Auto-detect
            device: "auto".to_string(),
            use_half_precision: false,
            cache_dir: None, // Use default HF cache
            trust_remote_code: false,
            model_config: HashMap::new(),
        }
    }
}

impl CandleEmbedderConfig {
    /// Create a new configuration with the specified model name.
    pub fn new<S: Into<String>>(model_name: S) -> Self {
        Self {
            model_name: model_name.into(),
            ..Default::default()
        }
    }

    /// Set the model revision.
    pub fn with_revision<S: Into<String>>(mut self, revision: S) -> Self {
        self.revision = revision.into();
        self
    }

    /// Set the maximum sequence length.
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set whether to normalize embeddings.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the embedding dimension.
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }

    /// Set the device preference.
    pub fn with_device<S: Into<String>>(mut self, device: S) -> Self {
        self.device = device.into();
        self
    }

    /// Set whether to use half precision.
    pub fn with_half_precision(mut self, use_half_precision: bool) -> Self {
        self.use_half_precision = use_half_precision;
        self
    }

    /// Set the cache directory.
    pub fn with_cache_dir<S: Into<String>>(mut self, cache_dir: S) -> Self {
        self.cache_dir = Some(cache_dir.into());
        self
    }

    /// Set whether to trust remote code.
    pub fn with_trust_remote_code(mut self, trust_remote_code: bool) -> Self {
        self.trust_remote_code = trust_remote_code;
        self
    }

    /// Add model-specific configuration.
    pub fn with_model_config<K: Into<String>>(mut self, key: K, value: serde_json::Value) -> Self {
        self.model_config.insert(key.into(), value);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), super::error::CandleError> {
        if self.model_name.is_empty() {
            return Err(super::error::CandleError::Configuration {
                message: "Model name cannot be empty".to_string(),
            });
        }

        if self.max_length == 0 {
            return Err(super::error::CandleError::Configuration {
                message: "Max length must be greater than 0".to_string(),
            });
        }

        if self.batch_size == 0 {
            return Err(super::error::CandleError::Configuration {
                message: "Batch size must be greater than 0".to_string(),
            });
        }

        if let Some(dim) = self.dimension {
            if dim == 0 {
                return Err(super::error::CandleError::Configuration {
                    message: "Dimension must be greater than 0".to_string(),
                });
            }
        }

        Ok(())
    }
}
