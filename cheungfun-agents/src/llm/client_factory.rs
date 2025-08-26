//! LLM client factory for creating siumai clients

use super::{LlmConfig, Result};
use crate::error::AgentError;
use siumai::prelude::*;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Factory for creating LLM clients using siumai
pub struct LlmClientFactory;

impl LlmClientFactory {
    /// Create a new factory instance
    pub fn new() -> Self {
        Self
    }

    /// Create a client based on the configuration
    pub async fn create_client(&self, config: &LlmConfig) -> Result<Arc<dyn ChatCapability>> {
        info!("Creating LLM client for provider: {}", config.provider);

        match config.provider.to_lowercase().as_str() {
            "openai" => self.create_openai_client(config).await,
            "anthropic" => self.create_anthropic_client(config).await,
            "ollama" => self.create_ollama_client(config).await,
            // "google" => self.create_google_client(config).await, // TODO: Enable when siumai supports Google
            provider => {
                warn!("Unsupported LLM provider: {}", provider);
                Err(AgentError::unsupported_provider(provider))
            }
        }
    }

    /// Create OpenAI client
    async fn create_openai_client(&self, config: &LlmConfig) -> Result<Arc<dyn ChatCapability>> {
        let api_key = config
            .api_key
            .as_ref()
            .ok_or_else(|| AgentError::missing_api_key("OpenAI"))?;

        debug!("Creating OpenAI client with model: {}", config.model);

        let mut builder = Siumai::builder()
            .openai()
            .api_key(api_key)
            .model(&config.model)
            .temperature(config.temperature);

        if let Some(max_tokens) = config.max_tokens {
            builder = builder.max_tokens(max_tokens);
        }

        if let Some(base_url) = &config.base_url {
            builder = builder.base_url(base_url);
        }

        let client = builder
            .build()
            .await
            .map_err(|e| AgentError::llm_error(format!("Failed to create OpenAI client: {}", e)))?;

        Ok(Arc::new(client))
    }

    /// Create Anthropic client
    async fn create_anthropic_client(&self, config: &LlmConfig) -> Result<Arc<dyn ChatCapability>> {
        let api_key = config
            .api_key
            .as_ref()
            .ok_or_else(|| AgentError::missing_api_key("Anthropic"))?;

        debug!("Creating Anthropic client with model: {}", config.model);

        let mut builder = Siumai::builder()
            .anthropic()
            .api_key(api_key)
            .model(&config.model)
            .temperature(config.temperature);

        if let Some(max_tokens) = config.max_tokens {
            builder = builder.max_tokens(max_tokens);
        }

        if let Some(base_url) = &config.base_url {
            builder = builder.base_url(base_url);
        }

        let client = builder.build().await.map_err(|e| {
            AgentError::llm_error(format!("Failed to create Anthropic client: {}", e))
        })?;

        Ok(Arc::new(client))
    }

    /// Create Ollama client
    async fn create_ollama_client(&self, config: &LlmConfig) -> Result<Arc<dyn ChatCapability>> {
        debug!("Creating Ollama client with model: {}", config.model);

        let base_url = config
            .base_url
            .as_deref()
            .unwrap_or("http://localhost:11434");

        let mut builder = Siumai::builder()
            .ollama()
            .base_url(base_url)
            .model(&config.model)
            .temperature(config.temperature);

        if let Some(max_tokens) = config.max_tokens {
            builder = builder.max_tokens(max_tokens);
        }

        let client = builder
            .build()
            .await
            .map_err(|e| AgentError::llm_error(format!("Failed to create Ollama client: {}", e)))?;

        Ok(Arc::new(client))
    }

    /// Create Google client (TODO: Enable when siumai supports Google)
    #[allow(dead_code)]
    async fn create_google_client(&self, _config: &LlmConfig) -> Result<Arc<dyn ChatCapability>> {
        Err(AgentError::unsupported_provider(
            "Google (not yet supported)",
        ))
    }

    /// Test client connectivity
    pub async fn test_client_connectivity(&self, client: &Arc<dyn ChatCapability>) -> Result<bool> {
        debug!("Testing LLM client connectivity");

        let test_messages =
            vec![siumai::types::ChatMessage::user("Hello, this is a connectivity test.").build()];

        match client.chat(test_messages).await {
            Ok(_) => {
                info!("LLM client connectivity test passed");
                Ok(true)
            }
            Err(e) => {
                warn!("LLM client connectivity test failed: {}", e);
                Ok(false)
            }
        }
    }

    /// Get recommended models for a provider
    pub fn get_recommended_models(&self, provider: &str) -> Vec<&'static str> {
        match provider.to_lowercase().as_str() {
            "openai" => vec!["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic" => vec![
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
            "ollama" => vec!["llama3.2", "llama3.1", "mistral", "codellama", "phi3"],
            "google" => vec!["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
            _ => vec![],
        }
    }

    /// Validate configuration before creating client
    pub fn validate_config(&self, config: &LlmConfig) -> Result<()> {
        // Check provider
        let supported_providers = ["openai", "anthropic", "ollama", "google"];
        if !supported_providers.contains(&config.provider.to_lowercase().as_str()) {
            return Err(AgentError::unsupported_provider(&config.provider));
        }

        // Check API key for providers that require it
        match config.provider.to_lowercase().as_str() {
            "openai" | "anthropic" | "google" => {
                if config.api_key.is_none() {
                    return Err(AgentError::missing_api_key(&config.provider));
                }
            }
            _ => {} // Ollama doesn't require API key
        }

        // Check temperature range
        if config.temperature < 0.0 || config.temperature > 2.0 {
            return Err(AgentError::invalid_configuration(
                "Temperature must be between 0.0 and 2.0",
            ));
        }

        // Check max tokens
        if let Some(max_tokens) = config.max_tokens {
            if max_tokens == 0 || max_tokens > 100_000 {
                return Err(AgentError::invalid_configuration(
                    "Max tokens must be between 1 and 100,000",
                ));
            }
        }

        Ok(())
    }
}

impl Default for LlmClientFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_config_valid() {
        let config = LlmConfig::ollama("llama3.2");
        let factory = LlmClientFactory::new();

        assert!(factory.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_invalid_temperature() {
        let config = LlmConfig::ollama("llama3.2").with_temperature(3.0);
        let factory = LlmClientFactory::new();

        assert!(factory.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_missing_api_key() {
        let mut config = LlmConfig::openai("", "gpt-4");
        config.api_key = None;
        let factory = LlmClientFactory::new();

        assert!(factory.validate_config(&config).is_err());
    }

    #[test]
    fn test_get_recommended_models() {
        let factory = LlmClientFactory::new();

        let openai_models = factory.get_recommended_models("openai");
        assert!(!openai_models.is_empty());
        assert!(openai_models.contains(&"gpt-4o"));

        let ollama_models = factory.get_recommended_models("ollama");
        assert!(!ollama_models.is_empty());
        assert!(ollama_models.contains(&"llama3.2"));
    }

    #[test]
    fn test_unsupported_provider() {
        let factory = LlmClientFactory::new();
        let models = factory.get_recommended_models("unsupported");
        assert!(models.is_empty());
    }
}
