//! LLM integration module for cheungfun-agents
//!
//! This module provides unified LLM client management using the siumai library.
//! It supports multiple providers (`OpenAI`, Anthropic, Ollama, etc.) with a
//! consistent interface.

use crate::error::{AgentError, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use siumai::prelude::*;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing::{debug, info, warn};

pub mod client_factory;
pub mod message_converter;
pub mod siumai_integration;

pub use client_factory::LlmClientFactory;
pub use message_converter::MessageConverter;
pub use siumai_integration::{LlmClientConfig, SiumaiLlmClient};

/// LLM provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Provider name (openai, anthropic, ollama, etc.)
    pub provider: String,
    /// API key (if required)
    pub api_key: Option<String>,
    /// Base URL (for custom endpoints)
    pub base_url: Option<String>,
    /// Model name
    pub model: String,
    /// Temperature for response generation
    pub temperature: f32,
    /// Maximum tokens in response
    pub max_tokens: Option<u32>,
    /// Request timeout
    pub timeout: Duration,
    /// Additional provider-specific settings
    pub extra_settings: HashMap<String, serde_json::Value>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            api_key: None,
            base_url: Some("http://localhost:11434".to_string()),
            model: "llama3.2".to_string(),
            temperature: 0.7,
            max_tokens: Some(1000),
            timeout: Duration::from_secs(30),
            extra_settings: HashMap::new(),
        }
    }
}

impl LlmConfig {
    /// Create `OpenAI` configuration
    #[must_use]
    pub fn openai(api_key: &str, model: &str) -> Self {
        Self {
            provider: "openai".to_string(),
            api_key: Some(api_key.to_string()),
            base_url: None,
            model: model.to_string(),
            temperature: 0.7,
            max_tokens: Some(1000),
            timeout: Duration::from_secs(30),
            extra_settings: HashMap::new(),
        }
    }

    /// Create Anthropic configuration
    #[must_use]
    pub fn anthropic(api_key: &str, model: &str) -> Self {
        Self {
            provider: "anthropic".to_string(),
            api_key: Some(api_key.to_string()),
            base_url: None,
            model: model.to_string(),
            temperature: 0.7,
            max_tokens: Some(1000),
            timeout: Duration::from_secs(30),
            extra_settings: HashMap::new(),
        }
    }

    /// Create Ollama configuration
    #[must_use]
    pub fn ollama(model: &str) -> Self {
        Self {
            provider: "ollama".to_string(),
            api_key: None,
            base_url: Some("http://localhost:11434".to_string()),
            model: model.to_string(),
            temperature: 0.7,
            max_tokens: Some(1000),
            timeout: Duration::from_secs(30),
            extra_settings: HashMap::new(),
        }
    }

    /// Set temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set max tokens
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set timeout
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set base URL
    #[must_use]
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = Some(base_url.to_string());
        self
    }
}

/// LLM client manager with connection pooling and caching
pub struct LlmClientManager {
    clients: tokio::sync::RwLock<HashMap<String, Arc<dyn ChatCapability>>>,
    factory: LlmClientFactory,
    max_clients: usize,
}

impl LlmClientManager {
    /// Create a new LLM client manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            clients: tokio::sync::RwLock::new(HashMap::new()),
            factory: LlmClientFactory::new(),
            max_clients: 10,
        }
    }

    /// Create with custom max clients
    #[must_use]
    pub fn with_max_clients(max_clients: usize) -> Self {
        Self {
            clients: tokio::sync::RwLock::new(HashMap::new()),
            factory: LlmClientFactory::new(),
            max_clients,
        }
    }

    /// Get or create a client for the given configuration
    pub async fn get_or_create_client(
        &self,
        config: &LlmConfig,
    ) -> Result<Arc<dyn ChatCapability>> {
        let key = self.generate_client_key(config);

        // Try to get existing client
        {
            let clients = self.clients.read().await;
            if let Some(client) = clients.get(&key) {
                debug!("Reusing existing LLM client for key: {}", key);
                return Ok(Arc::clone(client));
            }
        }

        // Create new client
        info!("Creating new LLM client for provider: {}", config.provider);
        let client = self.factory.create_client(config).await?;

        // Store in cache if under limit
        {
            let mut clients = self.clients.write().await;
            if clients.len() < self.max_clients {
                clients.insert(key.clone(), Arc::clone(&client));
                debug!("Cached LLM client with key: {}", key);
            } else {
                warn!("LLM client cache is full, not caching new client");
            }
        }

        Ok(client)
    }

    /// Clear all cached clients
    pub async fn clear_cache(&self) {
        let mut clients = self.clients.write().await;
        clients.clear();
        info!("Cleared LLM client cache");
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> LlmCacheStats {
        let clients = self.clients.read().await;
        LlmCacheStats {
            cached_clients: clients.len(),
            max_clients: self.max_clients,
            cache_keys: clients.keys().cloned().collect(),
        }
    }

    /// Generate a unique key for client caching
    fn generate_client_key(&self, config: &LlmConfig) -> String {
        format!(
            "{}:{}:{}:{}",
            config.provider,
            config.model,
            config.temperature,
            config.max_tokens.unwrap_or(0)
        )
    }
}

impl Default for LlmClientManager {
    fn default() -> Self {
        Self::new()
    }
}

/// LLM client cache statistics
#[derive(Debug, Clone)]
pub struct LlmCacheStats {
    /// Number of currently cached clients
    pub cached_clients: usize,
    /// Maximum number of clients that can be cached
    pub max_clients: usize,
    /// Keys of cached clients
    pub cache_keys: Vec<String>,
}

/// Trait for LLM-powered agents
#[async_trait::async_trait]
pub trait LlmAgent: Send + Sync {
    /// Get the LLM client
    fn llm_client(&self) -> &Arc<dyn ChatCapability>;

    /// Get the LLM configuration
    fn llm_config(&self) -> &LlmConfig;

    /// Generate a response using the LLM
    async fn generate_response(&self, messages: Vec<siumai::types::ChatMessage>) -> Result<String> {
        let client = self.llm_client();

        match client.chat(messages).await {
            Ok(response) => Ok(response.content.all_text()),
            Err(e) => {
                warn!("LLM generation failed: {}", e);
                Err(AgentError::llm_error(format!(
                    "Failed to generate response: {e}"
                )))
            }
        }
    }

    /// Generate a streaming response using the LLM
    async fn generate_streaming_response(
        &self,
        messages: Vec<siumai::types::ChatMessage>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String>> + Send + Unpin>> {
        let client = self.llm_client();

        match client.chat_stream(messages, None).await {
            Ok(stream) => {
                let mapped_stream = stream.map(|chunk| {
                    chunk
                        .map(|event| match event {
                            siumai::types::ChatStreamEvent::ContentDelta { delta, .. } => delta,
                            siumai::types::ChatStreamEvent::ThinkingDelta { delta } => delta,
                            siumai::types::ChatStreamEvent::StreamEnd { response } => {
                                response.content.all_text()
                            }
                            _ => String::new(), // Other events don't have text content
                        })
                        .map_err(|e| AgentError::llm_error(format!("Stream error: {e}")))
                });
                Ok(Box::new(mapped_stream))
            }
            Err(e) => {
                warn!("LLM streaming failed: {}", e);
                Err(AgentError::llm_error(format!(
                    "Failed to start streaming: {e}"
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_config_creation() {
        let config = LlmConfig::openai("test-key", "gpt-4");
        assert_eq!(config.provider, "openai");
        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.api_key, Some("test-key".to_string()));
    }

    #[test]
    fn test_llm_config_builder() {
        let config = LlmConfig::ollama("llama3.2")
            .with_temperature(0.5)
            .with_max_tokens(2000)
            .with_timeout(Duration::from_secs(60));

        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.max_tokens, Some(2000));
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_client_manager_cache() {
        let manager = LlmClientManager::new();
        let stats = manager.cache_stats().await;

        assert_eq!(stats.cached_clients, 0);
        assert_eq!(stats.max_clients, 10);
    }
}
