//! Factory implementations for creating query components.
//!
//! This module provides concrete factory implementations for creating
//! query-related components like response generators using siumai.

use async_trait::async_trait;
use std::sync::Arc;

use cheungfun_core::{
    config::LlmConfig,
    factory::LlmFactory,
    traits::ResponseGenerator,
    Result,
};
use siumai::prelude::*;

use crate::generator::{SiumaiGenerator, SiumaiGeneratorConfig};

/// Concrete implementation of LlmFactory using siumai.
///
/// This factory creates SiumaiGenerator instances from LlmConfig,
/// providing a bridge between the core configuration system and
/// the siumai-based response generation.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_query::factory::SiumaiLlmFactory;
/// use cheungfun_core::{config::LlmConfig, factory::LlmFactory};
/// use std::sync::Arc;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let factory = SiumaiLlmFactory::new();
/// 
/// // Create OpenAI client
/// let openai_config = LlmConfig::openai("gpt-4", "your-api-key");
/// let openai_client = factory.create_llm(&openai_config).await?;
///
/// // Create Anthropic client
/// let anthropic_config = LlmConfig::anthropic("claude-3-sonnet", "your-api-key");
/// let anthropic_client = factory.create_llm(&anthropic_config).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct SiumaiLlmFactory;

impl SiumaiLlmFactory {
    /// Create a new Siumai LLM factory.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Create a siumai client based on the configuration.
    async fn create_siumai_client(&self, config: &LlmConfig) -> Result<Siumai> {
        let mut builder = match config.provider.as_str() {
            "openai" => {
                let mut builder = Siumai::builder().openai();
                if let Some(api_key) = &config.api_key {
                    builder = builder.api_key(api_key);
                }
                if let Some(base_url) = &config.base_url {
                    builder = builder.base_url(base_url);
                }
                builder.model(&config.model)
            }
            "anthropic" => {
                let mut builder = Siumai::builder().anthropic();
                if let Some(api_key) = &config.api_key {
                    builder = builder.api_key(api_key);
                }
                if let Some(base_url) = &config.base_url {
                    builder = builder.base_url(base_url);
                }
                builder.model(&config.model)
            }
            "ollama" => {
                let mut builder = Siumai::builder().ollama();
                if let Some(base_url) = &config.base_url {
                    builder = builder.base_url(base_url);
                } else {
                    builder = builder.base_url("http://localhost:11434");
                }
                builder.model(&config.model)
            }
            "local" => {
                // For local models, we'll use Ollama as the backend
                let mut builder = Siumai::builder().ollama();
                if let Some(base_url) = &config.base_url {
                    builder = builder.base_url(base_url);
                } else {
                    return Err(cheungfun_core::CheungfunError::configuration(
                        "Local provider requires base_url to be specified".to_string(),
                    ));
                }
                builder.model(&config.model)
            }
            _ => {
                return Err(cheungfun_core::CheungfunError::configuration(format!(
                    "Unsupported LLM provider: {}",
                    config.provider
                )));
            }
        };

        // Apply common parameters
        if let Some(temperature) = config.temperature {
            builder = builder.temperature(temperature);
        }
        if let Some(max_tokens) = config.max_tokens {
            builder = builder.max_tokens(max_tokens.try_into().unwrap_or(4096));
        }

        // Build the client
        builder.build().await.map_err(|e| {
            cheungfun_core::CheungfunError::configuration(format!(
                "Failed to create siumai client: {e}"
            ))
        })
    }

    /// Create generator configuration from LLM config.
    fn create_generator_config(&self, config: &LlmConfig) -> SiumaiGeneratorConfig {
        SiumaiGeneratorConfig {
            default_model: Some(config.model.clone()),
            default_temperature: config.effective_temperature(),
            default_max_tokens: config.effective_max_tokens(),
            default_system_prompt: config.system_prompt.clone().unwrap_or_else(|| {
                "You are a helpful AI assistant. Answer questions based on the provided context. If you cannot answer based on the context, say so clearly.".to_string()
            }),
            include_citations: true,
            max_context_length: 8000,
            timeout_seconds: config.timeout_seconds.unwrap_or(60),
        }
    }
}

#[async_trait]
impl LlmFactory for SiumaiLlmFactory {
    async fn create_llm(&self, config: &LlmConfig) -> Result<Arc<dyn ResponseGenerator>> {
        // Validate configuration first
        self.validate_config(config).await?;

        // Create siumai client
        let siumai_client = self.create_siumai_client(config).await?;

        // Create generator configuration
        let generator_config = self.create_generator_config(config);

        // Create and return the response generator
        let generator = SiumaiGenerator::with_config(siumai_client, generator_config);
        Ok(Arc::new(generator))
    }

    fn can_create(&self, config: &LlmConfig) -> bool {
        matches!(
            config.provider.as_str(),
            "openai" | "anthropic" | "ollama" | "local"
        )
    }

    fn supported_providers(&self) -> Vec<&'static str> {
        vec!["openai", "anthropic", "ollama", "local"]
    }

    async fn validate_config(&self, config: &LlmConfig) -> Result<()> {
        // First, use the config's built-in validation
        config.validate()?;

        // Additional siumai-specific validation
        if !self.can_create(config) {
            return Err(cheungfun_core::CheungfunError::configuration(format!(
                "Unsupported provider for SiumaiLlmFactory: {}",
                config.provider
            )));
        }

        // Check provider-specific requirements
        match config.provider.as_str() {
            "openai" | "anthropic" => {
                if config.api_key.is_none() {
                    return Err(cheungfun_core::CheungfunError::configuration(format!(
                        "API key is required for provider: {}",
                        config.provider
                    )));
                }
            }
            "local" => {
                if config.base_url.is_none() {
                    return Err(cheungfun_core::CheungfunError::configuration(
                        "base_url is required for local provider".to_string(),
                    ));
                }
            }
            "ollama" => {
                // Ollama is optional for base_url, defaults to localhost:11434
            }
            _ => {
                return Err(cheungfun_core::CheungfunError::configuration(format!(
                    "Unknown provider: {}",
                    config.provider
                )));
            }
        }

        Ok(())
    }

    fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("name".to_string(), "SiumaiLlmFactory".into());
        metadata.insert(
            "description".to_string(),
            "Factory for creating LLM clients using the siumai crate".into(),
        );
        metadata.insert(
            "supported_providers".to_string(),
            self.supported_providers().into(),
        );
        metadata.insert("version".to_string(), env!("CARGO_PKG_VERSION").into());
        metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::config::LlmConfig;

    #[test]
    fn test_factory_creation() {
        let factory = SiumaiLlmFactory::new();
        assert!(factory.supported_providers().contains(&"openai"));
        assert!(factory.supported_providers().contains(&"anthropic"));
        assert!(factory.supported_providers().contains(&"ollama"));
    }

    #[test]
    fn test_can_create() {
        let factory = SiumaiLlmFactory::new();
        
        let openai_config = LlmConfig::openai("gpt-4", "test-key");
        assert!(factory.can_create(&openai_config));

        let anthropic_config = LlmConfig::anthropic("claude-3", "test-key");
        assert!(factory.can_create(&anthropic_config));

        let unsupported_config = LlmConfig::new("unsupported", "model");
        assert!(!factory.can_create(&unsupported_config));
    }

    #[tokio::test]
    async fn test_validation() {
        let factory = SiumaiLlmFactory::new();

        // Valid OpenAI config
        let valid_config = LlmConfig::openai("gpt-4", "test-key");
        assert!(factory.validate_config(&valid_config).await.is_ok());

        // Invalid config - missing API key
        let invalid_config = LlmConfig::new("openai", "gpt-4");
        assert!(factory.validate_config(&invalid_config).await.is_err());

        // Invalid config - unsupported provider
        let unsupported_config = LlmConfig::new("unsupported", "model");
        assert!(factory.validate_config(&unsupported_config).await.is_err());
    }
}
