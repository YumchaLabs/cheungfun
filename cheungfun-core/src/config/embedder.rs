//! Configuration for embedding models and services.
//!
//! This module provides configuration structures for different embedding
//! providers and models, including local models using Candle and remote
//! API services.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::Result;

/// Configuration for embedding models.
///
/// This enum supports different types of embedding providers including
/// local models (Candle), remote APIs, and custom implementations.
///
/// # Examples
///
/// ```rust
/// use cheungfun_core::config::EmbedderConfig;
/// use std::collections::HashMap;
///
/// // Local Candle model configuration
/// let candle_config = EmbedderConfig::Candle {
///     model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
///     device: "cuda".to_string(),
///     normalize: true,
///     batch_size: 32,
///     additional_config: HashMap::new(),
/// };
///
/// // Remote API configuration
/// let api_config = EmbedderConfig::Api {
///     provider: "openai".to_string(),
///     model_name: "text-embedding-ada-002".to_string(),
///     api_key: "your-api-key".to_string(),
///     base_url: None,
///     batch_size: 100,
///     additional_config: HashMap::new(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EmbedderConfig {
    /// Local embedding using Candle framework.
    Candle {
        /// Name or path of the model.
        model_name: String,

        /// Device to run on ("cpu", "cuda", "metal").
        device: String,

        /// Whether to normalize embeddings to unit length.
        normalize: bool,

        /// Batch size for processing.
        batch_size: usize,

        /// Additional model-specific configuration.
        additional_config: HashMap<String, serde_json::Value>,
    },

    /// Remote API embedding service.
    Api {
        /// Provider name (e.g., "openai", "cohere", "huggingface").
        provider: String,

        /// Model name or identifier.
        model_name: String,

        /// API key for authentication.
        api_key: String,

        /// Custom base URL (optional).
        base_url: Option<String>,

        /// Batch size for API requests.
        batch_size: usize,

        /// Additional provider-specific configuration.
        additional_config: HashMap<String, serde_json::Value>,
    },

    /// Custom embedding implementation.
    Custom {
        /// Implementation identifier.
        implementation: String,

        /// Custom configuration parameters.
        config: HashMap<String, serde_json::Value>,
    },
}

impl EmbedderConfig {
    /// Create a new Candle embedder configuration.
    pub fn candle<S: Into<String>>(model_name: S, device: S) -> Self {
        Self::Candle {
            model_name: model_name.into(),
            device: device.into(),
            normalize: true,
            batch_size: 32,
            additional_config: HashMap::new(),
        }
    }

    /// Create a new API embedder configuration.
    pub fn api<S: Into<String>>(provider: S, model_name: S, api_key: S) -> Self {
        Self::Api {
            provider: provider.into(),
            model_name: model_name.into(),
            api_key: api_key.into(),
            base_url: None,
            batch_size: 100,
            additional_config: HashMap::new(),
        }
    }

    /// Create a new custom embedder configuration.
    pub fn custom<S: Into<String>>(implementation: S) -> Self {
        Self::Custom {
            implementation: implementation.into(),
            config: HashMap::new(),
        }
    }

    /// Set batch size for the embedder.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        match &mut self {
            Self::Candle { batch_size: bs, .. } => *bs = batch_size,
            Self::Api { batch_size: bs, .. } => *bs = batch_size,
            Self::Custom { config, .. } => {
                config.insert("batch_size".to_string(), batch_size.into());
            }
        }
        self
    }

    /// Set normalization for Candle embedders.
    #[must_use]
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        if let Self::Candle {
            normalize: norm, ..
        } = &mut self
        {
            *norm = normalize;
        }
        self
    }

    /// Set base URL for API embedders.
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        if let Self::Api { base_url: url, .. } = &mut self {
            *url = Some(base_url.into());
        }
        self
    }

    /// Add additional configuration parameter.
    pub fn with_config<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        let config = match &mut self {
            Self::Candle {
                additional_config, ..
            } => additional_config,
            Self::Api {
                additional_config, ..
            } => additional_config,
            Self::Custom { config, .. } => config,
        };
        config.insert(key.into(), value.into());
        self
    }

    /// Get the model name.
    #[must_use]
    pub fn model_name(&self) -> &str {
        match self {
            Self::Candle { model_name, .. } => model_name,
            Self::Api { model_name, .. } => model_name,
            Self::Custom { implementation, .. } => implementation,
        }
    }

    /// Get the batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        match self {
            Self::Candle { batch_size, .. } => *batch_size,
            Self::Api { batch_size, .. } => *batch_size,
            Self::Custom { config, .. } => config
                .get("batch_size")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(32) as usize,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Candle {
                model_name,
                device,
                batch_size,
                ..
            } => {
                if model_name.is_empty() {
                    return Err(crate::CheungfunError::configuration(
                        "Model name cannot be empty",
                    ));
                }
                if device.is_empty() {
                    return Err(crate::CheungfunError::configuration(
                        "Device cannot be empty",
                    ));
                }
                if *batch_size == 0 {
                    return Err(crate::CheungfunError::configuration(
                        "Batch size must be greater than 0",
                    ));
                }
            }
            Self::Api {
                provider,
                model_name,
                api_key,
                batch_size,
                ..
            } => {
                if provider.is_empty() {
                    return Err(crate::CheungfunError::configuration(
                        "Provider cannot be empty",
                    ));
                }
                if model_name.is_empty() {
                    return Err(crate::CheungfunError::configuration(
                        "Model name cannot be empty",
                    ));
                }
                if api_key.is_empty() {
                    return Err(crate::CheungfunError::configuration(
                        "API key cannot be empty",
                    ));
                }
                if *batch_size == 0 {
                    return Err(crate::CheungfunError::configuration(
                        "Batch size must be greater than 0",
                    ));
                }
            }
            Self::Custom { implementation, .. } => {
                if implementation.is_empty() {
                    return Err(crate::CheungfunError::configuration(
                        "Implementation cannot be empty",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Check if this is a local embedder.
    #[must_use]
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Candle { .. } | Self::Custom { .. })
    }

    /// Check if this is a remote API embedder.
    #[must_use]
    pub fn is_api(&self) -> bool {
        matches!(self, Self::Api { .. })
    }

    /// Get the provider name.
    #[must_use]
    pub fn provider(&self) -> &str {
        match self {
            Self::Candle { .. } => "candle",
            Self::Api { provider, .. } => provider,
            Self::Custom { .. } => "custom",
        }
    }
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self::candle("sentence-transformers/all-MiniLM-L6-v2", "cpu")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_config() {
        let config = EmbedderConfig::candle("test-model", "cuda")
            .with_batch_size(64)
            .with_normalize(false)
            .with_config("temperature", 0.7);

        assert_eq!(config.model_name(), "test-model");
        assert_eq!(config.batch_size(), 64);
        assert!(config.is_local());
        assert!(!config.is_api());
        assert_eq!(config.provider(), "candle");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_api_config() {
        let config = EmbedderConfig::api("openai", "text-embedding-ada-002", "test-key")
            .with_batch_size(50)
            .with_base_url("https://api.openai.com/v1");

        assert_eq!(config.model_name(), "text-embedding-ada-002");
        assert_eq!(config.batch_size(), 50);
        assert!(!config.is_local());
        assert!(config.is_api());
        assert_eq!(config.provider(), "openai");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_custom_config() {
        let config = EmbedderConfig::custom("my-embedder")
            .with_config("dimension", 768)
            .with_config("batch_size", 16);

        assert_eq!(config.model_name(), "my-embedder");
        assert_eq!(config.batch_size(), 16);
        assert!(config.is_local());
        assert!(!config.is_api());
        assert_eq!(config.provider(), "custom");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_errors() {
        let invalid_config = EmbedderConfig::candle("", "cuda");
        assert!(invalid_config.validate().is_err());

        let invalid_api = EmbedderConfig::api("openai", "model", "");
        assert!(invalid_api.validate().is_err());

        let invalid_custom = EmbedderConfig::custom("");
        assert!(invalid_custom.validate().is_err());
    }

    #[test]
    fn test_serialization() {
        let config = EmbedderConfig::candle("test-model", "cpu");
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: EmbedderConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }
}
