//! Response generation implementations using LLMs.
//!
//! This module provides concrete implementations of the `ResponseGenerator` trait
//! for different LLM providers and generation strategies.

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use std::collections::HashMap;
use std::pin::Pin;
use tracing::{debug, info, instrument};

use cheungfun_core::{
    traits::ResponseGenerator,
    types::{GeneratedResponse, GenerationOptions, ScoredNode, TokenUsage},
    Result,
};

use siumai::prelude::*;

/// A response generator that uses the Siumai crate for LLM integration.
///
/// This generator supports multiple LLM providers through the unified
/// Siumai interface and provides both streaming and non-streaming responses.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_query::generator::SiumaiGenerator;
/// use cheungfun_core::prelude::*;
/// use siumai::prelude::*;
///
/// # async fn example() -> Result<()> {
/// let siumai_client = Siumai::builder()
///     .openai()
///     .build()
///     .await?;
///
/// let generator = SiumaiGenerator::builder()
///     .client(siumai_client)
///     .build()?;
///
/// let options = GenerationOptions::default();
/// let response = generator.generate_response(
///     "What is machine learning?",
///     vec![],
///     &options
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub struct SiumaiGenerator {
    /// Siumai client for LLM communication.
    client: Siumai,

    /// Configuration for response generation.
    config: SiumaiGeneratorConfig,
}

impl std::fmt::Debug for SiumaiGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SiumaiGenerator")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Configuration for Siumai generator.
#[derive(Debug, Clone)]
pub struct SiumaiGeneratorConfig {
    /// Default model to use for generation.
    pub default_model: Option<String>,
    
    /// Default temperature for generation.
    pub default_temperature: f32,
    
    /// Default maximum tokens for responses.
    pub default_max_tokens: usize,
    
    /// Default system prompt.
    pub default_system_prompt: String,
    
    /// Whether to include source citations by default.
    pub include_citations: bool,
    
    /// Maximum context length to use.
    pub max_context_length: usize,
    
    /// Timeout for generation operations.
    pub timeout_seconds: u64,
}

impl Default for SiumaiGeneratorConfig {
    fn default() -> Self {
        Self {
            default_model: None,
            default_temperature: 0.7,
            default_max_tokens: 1000,
            default_system_prompt: "You are a helpful AI assistant. Answer questions based on the provided context. If you cannot answer based on the context, say so clearly.".to_string(),
            include_citations: true,
            max_context_length: 8000,
            timeout_seconds: 60,
        }
    }
}

impl SiumaiGenerator {
    /// Create a new Siumai generator.
    pub fn new(client: Siumai) -> Self {
        Self {
            client,
            config: SiumaiGeneratorConfig::default(),
        }
    }

    /// Create a new Siumai generator with custom configuration.
    pub fn with_config(client: Siumai, config: SiumaiGeneratorConfig) -> Self {
        Self { client, config }
    }

    /// Create a builder for constructing Siumai generators.
    pub fn builder() -> SiumaiGeneratorBuilder {
        SiumaiGeneratorBuilder::new()
    }

    /// Build the prompt from query and context nodes.
    fn build_prompt(&self, query: &str, context_nodes: &[ScoredNode], options: &GenerationOptions) -> String {
        let system_prompt = options
            .system_prompt
            .as_ref()
            .unwrap_or(&self.config.default_system_prompt);

        let mut prompt = format!("{}\n\n", system_prompt);

        if !context_nodes.is_empty() {
            prompt.push_str("Context:\n");
            for (i, scored_node) in context_nodes.iter().enumerate() {
                prompt.push_str(&format!("{}. {}\n", i + 1, scored_node.node.content));
                
                if self.config.include_citations || options.include_citations {
                    if let Some(source) = scored_node.node.metadata.get("source") {
                        prompt.push_str(&format!("   Source: {}\n", source));
                    }
                }
                prompt.push('\n');
            }
        }

        prompt.push_str(&format!("Question: {}\n\nAnswer:", query));
        prompt
    }

    /// Extract token usage from Siumai response.
    fn extract_token_usage(&self, response: &ChatResponse) -> Option<TokenUsage> {
        response.usage.as_ref().map(|usage| TokenUsage {
            prompt_tokens: usage.prompt_tokens as usize,
            completion_tokens: usage.completion_tokens as usize,
            total_tokens: usage.total_tokens as usize,
        })
    }

    /// Build Siumai chat messages.
    fn build_chat_messages(&self, prompt: &str, _options: &GenerationOptions) -> Vec<ChatMessage> {
        vec![
            ChatMessage::user(prompt).build()
        ]
    }
}

#[async_trait]
impl ResponseGenerator for SiumaiGenerator {
    #[instrument(skip(self, context_nodes), fields(generator = "SiumaiGenerator"))]
    async fn generate_response(
        &self,
        query: &str,
        context_nodes: Vec<ScoredNode>,
        options: &GenerationOptions,
    ) -> Result<GeneratedResponse> {
        info!("Generating response for query with {} context nodes", context_nodes.len());

        // Build prompt
        let prompt = self.build_prompt(query, &context_nodes, options);
        debug!("Built prompt with {} characters", prompt.len());

        // Build messages
        let messages = self.build_chat_messages(&prompt, options);

        // Generate response
        let response = self.client.chat(messages).await.map_err(|e| {
            cheungfun_core::CheungfunError::Llm {
                message: format!("Siumai generation failed: {}", e),
            }
        })?;

        // Extract content
        let content = match &response.content {
            siumai::MessageContent::Text(text) => text.clone(),
            _ => return Err(cheungfun_core::CheungfunError::Llm {
                message: "Unsupported content type in LLM response".to_string(),
            }),
        };

        // Extract source node IDs
        let source_nodes: Vec<uuid::Uuid> = context_nodes.iter().map(|node| node.node.id).collect();

        // Build metadata
        let mut metadata = HashMap::new();
        metadata.insert("model".to_string(), serde_json::Value::String(
            response.model.clone().unwrap_or_else(|| "unknown".to_string())
        ));
        metadata.insert("prompt_length".to_string(), serde_json::Value::Number(
            prompt.len().into()
        ));
        metadata.insert("context_nodes_count".to_string(), serde_json::Value::Number(
            context_nodes.len().into()
        ));

        // Extract token usage
        let usage = self.extract_token_usage(&response);

        info!("Generated response with {} characters", content.len());
        
        Ok(GeneratedResponse {
            content,
            source_nodes,
            metadata,
            usage,
        })
    }

    #[instrument(skip(self, context_nodes), fields(generator = "SiumaiGenerator"))]
    async fn generate_response_stream(
        &self,
        query: &str,
        context_nodes: Vec<ScoredNode>,
        options: &GenerationOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        info!("Generating streaming response for query with {} context nodes", context_nodes.len());

        // Build prompt
        let prompt = self.build_prompt(query, &context_nodes, options);
        debug!("Built prompt with {} characters", prompt.len());

        // Build messages
        let messages = self.build_chat_messages(&prompt, options);

        // Generate streaming response
        let stream = self.client.chat_stream(messages, None).await.map_err(|e| {
            cheungfun_core::CheungfunError::Llm {
                message: format!("Siumai streaming failed: {}", e),
            }
        })?;

        // Transform stream to extract content
        let content_stream = stream.map(|result| {
            result
                .map_err(|e| cheungfun_core::CheungfunError::Llm {
                    message: format!("Stream error: {}", e),
                })
                .and_then(|_chunk| {
                    // For now, return empty string for each chunk
                    // TODO: Implement proper streaming content extraction based on siumai API
                    Ok(String::new())
                })
        });

        Ok(Box::pin(content_stream))
    }

    fn name(&self) -> &'static str {
        "SiumaiGenerator"
    }

    async fn health_check(&self) -> Result<()> {
        // Try a simple generation to check if the client is working
        let test_messages = vec![ChatMessage::user("Hello").build()];

        self.client.chat(test_messages).await.map_err(|e| {
            cheungfun_core::CheungfunError::Llm {
                message: format!("Health check failed: {}", e),
            }
        })?;

        Ok(())
    }

    fn config(&self) -> HashMap<String, serde_json::Value> {
        let mut config = HashMap::new();
        config.insert("default_temperature".to_string(), self.config.default_temperature.into());
        config.insert("default_max_tokens".to_string(), self.config.default_max_tokens.into());
        config.insert("include_citations".to_string(), self.config.include_citations.into());
        config.insert("max_context_length".to_string(), self.config.max_context_length.into());
        config.insert("timeout_seconds".to_string(), self.config.timeout_seconds.into());
        config
    }
}

/// Builder for creating Siumai generators.
#[derive(Default)]
pub struct SiumaiGeneratorBuilder {
    client: Option<Siumai>,
    config: Option<SiumaiGeneratorConfig>,
}

impl std::fmt::Debug for SiumaiGeneratorBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SiumaiGeneratorBuilder")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl SiumaiGeneratorBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the Siumai client.
    pub fn client(mut self, client: Siumai) -> Self {
        self.client = Some(client);
        self
    }

    /// Set the configuration.
    pub fn config(mut self, config: SiumaiGeneratorConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the Siumai generator.
    pub fn build(self) -> Result<SiumaiGenerator> {
        let client = self.client.ok_or_else(|| {
            cheungfun_core::CheungfunError::Configuration {
                message: "Siumai client is required".to_string(),
            }
        })?;

        let config = self.config.unwrap_or_default();

        Ok(SiumaiGenerator::with_config(client, config))
    }
}
