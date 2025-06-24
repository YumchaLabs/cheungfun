//! Configuration for Large Language Models.
//!
//! This module provides configuration structures for different LLM
//! providers and models, with a focus on using the siumai crate
//! for LLM integration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::Result;

/// Configuration for Large Language Models.
///
/// This configuration is designed to work with the siumai crate
/// for LLM integration, supporting various providers and models.
///
/// # Examples
///
/// ```rust
/// use cheungfun_core::config::LlmConfig;
/// use std::collections::HashMap;
///
/// // OpenAI configuration
/// let openai_config = LlmConfig::new("openai", "gpt-4")
///     .with_api_key("your-api-key")
///     .with_temperature(0.7)
///     .with_max_tokens(1000);
///
/// // Anthropic configuration
/// let anthropic_config = LlmConfig::new("anthropic", "claude-3-sonnet")
///     .with_api_key("your-api-key")
///     .with_temperature(0.5);
///
/// // Local model configuration
/// let local_config = LlmConfig::new("local", "llama-2-7b")
///     .with_base_url("http://localhost:8080")
///     .with_temperature(0.8);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmConfig {
    /// Provider name (e.g., "openai", "anthropic", "local").
    pub provider: String,

    /// Model name or identifier.
    pub model: String,

    /// API key for authentication (optional for local models).
    pub api_key: Option<String>,

    /// Custom base URL (optional).
    pub base_url: Option<String>,

    /// Temperature for generation (0.0 to 2.0).
    pub temperature: Option<f32>,

    /// Maximum tokens to generate.
    pub max_tokens: Option<usize>,

    /// System prompt template.
    pub system_prompt: Option<String>,

    /// Request timeout in seconds.
    pub timeout_seconds: Option<u64>,

    /// Maximum number of retry attempts.
    pub max_retries: Option<usize>,

    /// Whether to stream responses.
    pub stream: bool,

    /// Additional provider-specific configuration.
    pub additional_config: HashMap<String, serde_json::Value>,
}

impl LlmConfig {
    /// Create a new LLM configuration.
    pub fn new<S1: Into<String>, S2: Into<String>>(provider: S1, model: S2) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            api_key: None,
            base_url: None,
            temperature: None,
            max_tokens: None,
            system_prompt: None,
            timeout_seconds: None,
            max_retries: None,
            stream: false,
            additional_config: HashMap::new(),
        }
    }

    /// Set the API key.
    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the base URL.
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the system prompt.
    pub fn with_system_prompt<S: Into<String>>(mut self, system_prompt: S) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Set the timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = Some(timeout_seconds);
        self
    }

    /// Set the maximum retries.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    /// Enable streaming.
    #[must_use]
    pub fn with_streaming(mut self, stream: bool) -> Self {
        self.stream = stream;
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

    /// Get the effective temperature (with default).
    #[must_use]
    pub fn effective_temperature(&self) -> f32 {
        self.temperature.unwrap_or(0.7)
    }

    /// Get the effective max tokens (with default).
    #[must_use]
    pub fn effective_max_tokens(&self) -> usize {
        self.max_tokens.unwrap_or(1000)
    }

    /// Get the effective timeout (with default).
    #[must_use]
    pub fn effective_timeout(&self) -> u64 {
        self.timeout_seconds.unwrap_or(60)
    }

    /// Get the effective max retries (with default).
    #[must_use]
    pub fn effective_max_retries(&self) -> usize {
        self.max_retries.unwrap_or(3)
    }

    /// Check if this is a local model.
    #[must_use]
    pub fn is_local(&self) -> bool {
        self.provider == "local" || self.base_url.is_some()
    }

    /// Check if this is a remote API model.
    #[must_use]
    pub fn is_remote(&self) -> bool {
        !self.is_local()
    }

    /// Check if API key is required.
    #[must_use]
    pub fn requires_api_key(&self) -> bool {
        !self.is_local() && !matches!(self.provider.as_str(), "local" | "ollama")
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.provider.is_empty() {
            return Err(crate::CheungfunError::configuration(
                "Provider cannot be empty",
            ));
        }

        if self.model.is_empty() {
            return Err(crate::CheungfunError::configuration(
                "Model cannot be empty",
            ));
        }

        if self.requires_api_key() && self.api_key.is_none() {
            return Err(crate::CheungfunError::configuration(format!(
                "API key is required for provider: {}",
                self.provider
            )));
        }

        if let Some(temp) = self.temperature {
            if !(0.0..=2.0).contains(&temp) {
                return Err(crate::CheungfunError::configuration(
                    "Temperature must be between 0.0 and 2.0",
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

        if let Some(timeout) = self.timeout_seconds {
            if timeout == 0 {
                return Err(crate::CheungfunError::configuration(
                    "Timeout must be greater than 0",
                ));
            }
        }

        if let Some(url) = &self.base_url {
            if !url.starts_with("http://") && !url.starts_with("https://") {
                return Err(crate::CheungfunError::configuration(
                    "Base URL must start with http:// or https://",
                ));
            }
        }

        Ok(())
    }

    /// Get connection information for logging (without sensitive data).
    #[must_use]
    pub fn connection_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("provider".to_string(), self.provider.clone());
        info.insert("model".to_string(), self.model.clone());

        if let Some(url) = &self.base_url {
            info.insert("base_url".to_string(), url.clone());
        }

        info.insert(
            "temperature".to_string(),
            self.effective_temperature().to_string(),
        );
        info.insert(
            "max_tokens".to_string(),
            self.effective_max_tokens().to_string(),
        );
        info.insert("stream".to_string(), self.stream.to_string());

        // Don't include API key in connection info
        info
    }

    /// Create a configuration for `OpenAI` models.
    pub fn openai<S: Into<String>>(model: S, api_key: S) -> Self {
        let model_str = model.into();
        let api_key_str = api_key.into();
        Self::new("openai", model_str).with_api_key(api_key_str)
    }

    /// Create a configuration for Anthropic models.
    pub fn anthropic<S: Into<String>>(model: S, api_key: S) -> Self {
        let model_str = model.into();
        let api_key_str = api_key.into();
        Self::new("anthropic", model_str).with_api_key(api_key_str)
    }

    /// Create a configuration for local models.
    pub fn local<S: Into<String>>(model: S, base_url: S) -> Self {
        let model_str = model.into();
        let base_url_str = base_url.into();
        Self::new("local", model_str).with_base_url(base_url_str)
    }

    /// Create a configuration for Ollama models.
    pub fn ollama<S: Into<String>>(model: S) -> Self {
        let model_str = model.into();
        Self::new("ollama", model_str).with_base_url("http://localhost:11434")
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self::new("openai", "gpt-3.5-turbo")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_openai_config() {
        let config = LlmConfig::openai("gpt-4", "test-key")
            .with_temperature(0.8)
            .with_max_tokens(2000)
            .with_streaming(true);

        assert_eq!(config.provider, "openai");
        assert_eq!(config.model, "gpt-4");
        assert_relative_eq!(config.effective_temperature(), 0.8);
        assert_eq!(config.effective_max_tokens(), 2000);
        assert!(config.stream);
        assert!(!config.is_local());
        assert!(config.is_remote());
        assert!(config.requires_api_key());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_local_config() {
        let config = LlmConfig::local("llama-2-7b", "http://localhost:8080")
            .with_temperature(0.5)
            .with_max_tokens(1500);

        assert_eq!(config.provider, "local");
        assert_eq!(config.model, "llama-2-7b");
        assert!(config.is_local());
        assert!(!config.is_remote());
        assert!(!config.requires_api_key());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_ollama_config() {
        let config = LlmConfig::ollama("llama2").with_temperature(0.9);

        assert_eq!(config.provider, "ollama");
        assert_eq!(config.model, "llama2");
        assert!(config.is_local());
        assert!(!config.requires_api_key());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_errors() {
        let invalid_provider = LlmConfig::new("", "model");
        assert!(invalid_provider.validate().is_err());

        let invalid_model = LlmConfig::new("openai", "");
        assert!(invalid_model.validate().is_err());

        let missing_api_key = LlmConfig::new("openai", "gpt-4");
        assert!(missing_api_key.validate().is_err());

        let invalid_temperature = LlmConfig::new("local", "model")
            .with_base_url("http://localhost:8080")
            .with_temperature(3.0);
        assert!(invalid_temperature.validate().is_err());

        let invalid_url = LlmConfig::new("local", "model").with_base_url("localhost:8080");
        assert!(invalid_url.validate().is_err());
    }

    #[test]
    fn test_connection_info() {
        let config = LlmConfig::openai("gpt-4", "secret-key").with_temperature(0.7);

        let info = config.connection_info();
        assert_eq!(info.get("provider"), Some(&"openai".to_string()));
        assert_eq!(info.get("model"), Some(&"gpt-4".to_string()));
        assert_eq!(info.get("temperature"), Some(&"0.7".to_string()));
        assert!(!info.contains_key("api_key")); // Should not include sensitive data
    }

    #[test]
    fn test_serialization() {
        let config = LlmConfig::openai("gpt-4", "test-key");
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: LlmConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_effective_values() {
        let config = LlmConfig::new("test", "model");

        assert_relative_eq!(config.effective_temperature(), 0.7);
        assert_eq!(config.effective_max_tokens(), 1000);
        assert_eq!(config.effective_timeout(), 60);
        assert_eq!(config.effective_max_retries(), 3);
    }
}
