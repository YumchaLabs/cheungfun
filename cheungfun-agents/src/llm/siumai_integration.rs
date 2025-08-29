//! Siumai LLM Integration
//!
//! This module provides integration with the siumai library for LLM interactions,
//! supporting multiple providers like `OpenAI`, Anthropic, and Ollama.

use crate::{
    error::{AgentError, Result},
    types::{AgentMessage, MessageRole},
};
use serde::{Deserialize, Serialize};
use siumai::prelude::*;
use std::collections::HashMap;

/// LLM client wrapper for siumai integration
pub struct SiumaiLlmClient {
    /// The underlying siumai client
    client: Box<dyn LlmClient>,
    /// Client configuration
    config: LlmClientConfig,
}

impl Clone for SiumaiLlmClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone_box(),
            config: self.config.clone(),
        }
    }
}

impl std::fmt::Debug for SiumaiLlmClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SiumaiLlmClient")
            .field("config", &self.config)
            .field("client", &"<LlmClient>")
            .finish()
    }
}

/// Configuration for the LLM client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmClientConfig {
    /// Provider type (openai, anthropic, ollama, etc.)
    pub provider: String,
    /// Model name
    pub model: String,
    /// Temperature for response generation
    pub temperature: Option<f32>,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// System prompt
    pub system_prompt: Option<String>,
    /// Whether to include tool calls in responses
    pub enable_tools: bool,
    /// Custom provider-specific parameters
    pub custom_params: HashMap<String, serde_json::Value>,
}

impl Default for LlmClientConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            model: "gpt-4o-mini".to_string(),
            temperature: Some(0.7f32),
            max_tokens: Some(4096),
            system_prompt: None,
            enable_tools: true,
            custom_params: HashMap::new(),
        }
    }
}

impl LlmClientConfig {
    /// Create a new configuration
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            ..Default::default()
        }
    }

    /// Set temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set system prompt
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Enable or disable tools
    #[must_use]
    pub fn with_tools(mut self, enable: bool) -> Self {
        self.enable_tools = enable;
        self
    }

    /// Add custom parameter
    #[must_use]
    pub fn with_custom_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.custom_params.insert(key.into(), value);
        self
    }
}

impl SiumaiLlmClient {
    /// Create a new LLM client from configuration
    pub async fn from_config(config: LlmClientConfig) -> Result<Self> {
        let client = Self::build_client(&config).await?;

        Ok(Self { client, config })
    }

    /// Create an `OpenAI` client
    pub async fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let config = LlmClientConfig::new("openai", model);
        let client = LlmBuilder::new()
            .openai()
            .api_key(api_key.into())
            .model(&config.model)
            .temperature(config.temperature.unwrap_or(0.7f32))
            .build()
            .await
            .map_err(|e| AgentError::generic(format!("Failed to create OpenAI client: {e}")))?;

        Ok(Self {
            client: Box::new(client),
            config,
        })
    }

    /// Create an Anthropic client
    pub async fn anthropic(api_key: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let config = LlmClientConfig::new("anthropic", model);
        let client = LlmBuilder::new()
            .anthropic()
            .api_key(api_key.into())
            .model(&config.model)
            .temperature(config.temperature.unwrap_or(0.7f32))
            .build()
            .await
            .map_err(|e| AgentError::generic(format!("Failed to create Anthropic client: {e}")))?;

        Ok(Self {
            client: Box::new(client),
            config,
        })
    }

    /// Create an Ollama client
    pub async fn ollama(base_url: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let config = LlmClientConfig::new("ollama", model);
        let client = LlmBuilder::new()
            .ollama()
            .base_url(base_url.into())
            .model(&config.model)
            .temperature(config.temperature.unwrap_or(0.7f32))
            .build()
            .await
            .map_err(|e| AgentError::generic(format!("Failed to create Ollama client: {e}")))?;

        Ok(Self {
            client: Box::new(client),
            config,
        })
    }

    /// Build client from configuration
    async fn build_client(config: &LlmClientConfig) -> Result<Box<dyn LlmClient>> {
        // Build client based on provider
        let client: Box<dyn LlmClient> = match config.provider.as_str() {
            "openai" => {
                let api_key = std::env::var("OPENAI_API_KEY")
                    .map_err(|_| AgentError::invalid_configuration("OPENAI_API_KEY not set"))?;

                let mut builder = LlmBuilder::new()
                    .openai()
                    .api_key(api_key)
                    .model(&config.model);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                if let Some(max_tokens) = config.max_tokens {
                    builder = builder.max_tokens(max_tokens);
                }

                let client = builder.build().await.map_err(|e| {
                    AgentError::generic(format!("Failed to build OpenAI client: {e}"))
                })?;

                Box::new(client)
            }
            "anthropic" => {
                let api_key = std::env::var("ANTHROPIC_API_KEY")
                    .map_err(|_| AgentError::invalid_configuration("ANTHROPIC_API_KEY not set"))?;

                let mut builder = LlmBuilder::new()
                    .anthropic()
                    .api_key(api_key)
                    .model(&config.model);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                if let Some(max_tokens) = config.max_tokens {
                    builder = builder.max_tokens(max_tokens);
                }

                let client = builder.build().await.map_err(|e| {
                    AgentError::generic(format!("Failed to build Anthropic client: {e}"))
                })?;

                Box::new(client)
            }
            "ollama" => {
                let base_url = std::env::var("OLLAMA_BASE_URL")
                    .unwrap_or_else(|_| "http://localhost:11434".to_string());

                let mut builder = LlmBuilder::new()
                    .ollama()
                    .base_url(base_url)
                    .model(&config.model);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                if let Some(max_tokens) = config.max_tokens {
                    builder = builder.max_tokens(max_tokens);
                }

                let client = builder.build().await.map_err(|e| {
                    AgentError::generic(format!("Failed to build Ollama client: {e}"))
                })?;

                Box::new(client)
            }
            _ => {
                return Err(AgentError::invalid_configuration(format!(
                    "Unsupported provider: {}",
                    config.provider
                )));
            }
        };

        Ok(client)
    }

    /// Get the client configuration
    #[must_use]
    pub fn config(&self) -> &LlmClientConfig {
        &self.config
    }

    /// Send a chat request
    pub async fn chat(&self, messages: Vec<AgentMessage>) -> Result<String> {
        // Convert AgentMessage to siumai ChatMessage
        let siumai_messages = self.convert_messages(messages)?;

        // Send chat request
        let response = self
            .client
            .chat(siumai_messages)
            .await
            .map_err(|e| AgentError::generic(format!("Chat request failed: {e}")))?;

        // Extract text response
        if let Some(text) = response.content_text() {
            Ok(text.to_string())
        } else {
            Err(AgentError::generic("No text content in response"))
        }
    }

    /// Convert `AgentMessage` to siumai `ChatMessage`
    fn convert_messages(&self, messages: Vec<AgentMessage>) -> Result<Vec<ChatMessage>> {
        let mut siumai_messages = Vec::new();

        // Add system prompt if configured
        if let Some(system_prompt) = &self.config.system_prompt {
            siumai_messages.push(system!(system_prompt.clone()));
        }

        // Convert agent messages
        for msg in messages {
            let siumai_msg = match msg.role {
                MessageRole::User => user!(msg.content),
                MessageRole::Assistant => assistant!(msg.content),
                MessageRole::System => system!(msg.content),
                MessageRole::Tool => {
                    // For tool messages, we'll treat them as assistant messages for now
                    assistant!(format!("Tool result: {}", msg.content))
                }
            };
            siumai_messages.push(siumai_msg);
        }

        Ok(siumai_messages)
    }
}
