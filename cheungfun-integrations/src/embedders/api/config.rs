//! Configuration for API-based embedders.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Supported API providers for embedding generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiProvider {
    /// OpenAI embedding API
    OpenAI,
    /// Anthropic embedding API (future support)
    Anthropic,
    /// Custom provider with base URL
    Custom {
        /// Provider name
        name: String,
        /// Base URL for the API
        base_url: String,
    },
}

impl ApiProvider {
    /// Get the provider name as a string.
    pub fn name(&self) -> &str {
        match self {
            Self::OpenAI => "openai",
            Self::Anthropic => "anthropic",
            Self::Custom { name, .. } => name,
        }
    }

    /// Check if this provider supports the given model.
    pub fn supports_model(&self, model: &str) -> bool {
        match self {
            Self::OpenAI => {
                model.starts_with("text-embedding-")
                    || model == "text-embedding-ada-002"
                    || model == "text-embedding-3-small"
                    || model == "text-embedding-3-large"
            }
            Self::Anthropic => {
                // Future support for Anthropic embedding models
                false
            }
            Self::Custom { .. } => {
                // Custom providers can support any model
                true
            }
        }
    }
}

/// Configuration for API-based embedders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEmbedderConfig {
    /// API provider
    pub provider: ApiProvider,

    /// Model name or identifier
    pub model: String,

    /// API key for authentication
    pub api_key: String,

    /// Custom base URL (overrides provider default)
    pub base_url: Option<String>,

    /// Batch size for API requests
    pub batch_size: usize,

    /// Maximum number of retries for failed requests
    pub max_retries: usize,

    /// Timeout for API requests
    pub timeout: Duration,

    /// Enable embedding caching
    pub enable_cache: bool,

    /// Cache TTL (time to live)
    pub cache_ttl: Duration,

    /// Additional provider-specific configuration
    pub additional_config: HashMap<String, serde_json::Value>,
}

impl ApiEmbedderConfig {
    /// Create a new configuration with OpenAI provider.
    pub fn openai<S: Into<String>>(api_key: S, model: S) -> Self {
        Self {
            provider: ApiProvider::OpenAI,
            model: model.into(),
            api_key: api_key.into(),
            base_url: None,
            batch_size: 100,
            max_retries: 3,
            timeout: Duration::from_secs(30),
            enable_cache: true,
            cache_ttl: Duration::from_secs(3600), // 1 hour
            additional_config: HashMap::new(),
        }
    }

    /// Create a new configuration with Anthropic provider.
    pub fn anthropic<S: Into<String>>(api_key: S, model: S) -> Self {
        Self {
            provider: ApiProvider::Anthropic,
            model: model.into(),
            api_key: api_key.into(),
            base_url: None,
            batch_size: 50, // Anthropic might have different limits
            max_retries: 3,
            timeout: Duration::from_secs(30),
            enable_cache: true,
            cache_ttl: Duration::from_secs(3600),
            additional_config: HashMap::new(),
        }
    }

    /// Create a new configuration with custom provider.
    pub fn custom<S: Into<String>>(name: S, base_url: S, api_key: S, model: S) -> Self {
        let base_url_string = base_url.into();
        Self {
            provider: ApiProvider::Custom {
                name: name.into(),
                base_url: base_url_string.clone(),
            },
            model: model.into(),
            api_key: api_key.into(),
            base_url: Some(base_url_string),
            batch_size: 100,
            max_retries: 3,
            timeout: Duration::from_secs(30),
            enable_cache: true,
            cache_ttl: Duration::from_secs(3600),
            additional_config: HashMap::new(),
        }
    }

    /// Set batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set maximum retries.
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable or disable caching.
    pub fn with_cache(mut self, enable: bool) -> Self {
        self.enable_cache = enable;
        self
    }

    /// Set cache TTL.
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Set base URL.
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Add additional configuration.
    pub fn with_config<S: Into<String>>(mut self, key: S, value: serde_json::Value) -> Self {
        self.additional_config.insert(key.into(), value);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.api_key.is_empty() {
            return Err("API key cannot be empty".to_string());
        }

        if self.model.is_empty() {
            return Err("Model name cannot be empty".to_string());
        }

        if !self.provider.supports_model(&self.model) {
            return Err(format!(
                "Model '{}' is not supported by provider '{}'",
                self.model,
                self.provider.name()
            ));
        }

        if self.batch_size == 0 {
            return Err("Batch size must be greater than 0".to_string());
        }

        if self.timeout.is_zero() {
            return Err("Timeout must be greater than 0".to_string());
        }

        Ok(())
    }
}

impl Default for ApiEmbedderConfig {
    fn default() -> Self {
        Self::openai("", "text-embedding-3-small")
    }
}
