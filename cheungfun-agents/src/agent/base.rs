//! Base agent trait and common functionality
//!
//! This module defines the core agent interface following LlamaIndex's architectural
//! patterns, providing a clean and extensible foundation for all agent implementations.

use crate::{
    error::{AgentError, Result},
    types::{AgentCapabilities, AgentConfig, AgentId, AgentMessage, AgentResponse},
};
use async_trait::async_trait;
use cheungfun_core::traits::BaseMemory;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

/// Agent execution context for maintaining state across interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContext {
    /// Context variables
    pub variables: HashMap<String, serde_json::Value>,
    /// Conversation history
    pub history: Vec<AgentMessage>,
    /// Current step or phase
    pub current_step: Option<String>,
    /// Context metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl AgentContext {
    /// Create a new agent context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            history: Vec::new(),
            current_step: None,
            metadata: HashMap::new(),
        }
    }

    /// Set a context variable
    pub fn set_variable(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.variables.insert(key.into(), value);
    }

    /// Get a context variable
    pub fn get_variable(&self, key: &str) -> Option<&serde_json::Value> {
        self.variables.get(key)
    }

    /// Add a message to the conversation history
    pub fn add_message(&mut self, message: AgentMessage) {
        self.history.push(message);
    }

    /// Get the conversation history
    pub fn get_history(&self) -> &[AgentMessage] {
        &self.history
    }

    /// Clear the conversation history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

impl Default for AgentContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Base agent trait that all agents must implement
///
/// This trait defines the core interface for all agents in the Cheungfun system,
/// following LlamaIndex's design patterns for consistency and extensibility.
#[async_trait]
pub trait BaseAgent: Send + Sync + std::fmt::Debug {
    /// Get the agent's unique identifier
    fn id(&self) -> AgentId;

    /// Get the agent's name
    fn name(&self) -> &str;

    /// Get the agent's description
    fn description(&self) -> Option<&str>;

    /// Get the agent's capabilities
    fn capabilities(&self) -> &AgentCapabilities;

    /// Get the agent's configuration
    fn config(&self) -> &AgentConfig;

    /// Process a single message and return a response
    ///
    /// This is the primary method for agent interaction. The agent processes
    /// the input message and returns a structured response.
    async fn chat(
        &self,
        message: AgentMessage,
        context: Option<&mut AgentContext>,
    ) -> Result<AgentResponse>;

    /// Process multiple messages in a conversation context
    async fn chat_with_history(
        &self,
        messages: Vec<AgentMessage>,
        context: Option<&mut AgentContext>,
    ) -> Result<AgentResponse>;

    /// Stream a response (if supported by the agent)
    async fn stream_chat(
        &self,
        message: AgentMessage,
        context: Option<&mut AgentContext>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String>> + Send + Unpin>> {
        // Default implementation for agents that don't support streaming
        if !self.capabilities().supports_streaming {
            return Err(AgentError::unsupported_operation(
                "Streaming not supported by this agent",
            ));
        }

        // This should be overridden by agents that support streaming
        Err(AgentError::not_implemented("Streaming not implemented"))
    }

    /// Reset the agent's internal state
    async fn reset(&self) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Get agent execution statistics
    fn stats(&self) -> HashMap<String, serde_json::Value> {
        // Default implementation returns empty stats
        HashMap::new()
    }

    /// Validate that the agent can handle a specific message
    fn can_handle(&self, message: &AgentMessage) -> bool {
        // Default implementation accepts all messages
        true
    }

    /// Get the agent's current status
    fn status(&self) -> AgentStatus {
        AgentStatus::Ready
    }
}

/// Agent status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Agent is ready to process requests
    Ready,
    /// Agent is currently processing a request
    Busy,
    /// Agent is temporarily unavailable
    Unavailable,
    /// Agent has encountered an error
    Error(String),
}

/// Agent builder for creating agents with a fluent API
///
/// This builder provides a convenient way to construct agent configurations
/// following the builder pattern for better ergonomics.
#[derive(Debug)]
pub struct AgentBuilder {
    config: AgentConfig,
}

impl AgentBuilder {
    /// Create a new agent builder with the specified name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            config: AgentConfig {
                name: name.into(),
                description: None,
                instructions: None,
                capabilities: AgentCapabilities::default(),
                max_execution_time_ms: Some(30_000), // 30 seconds default
                max_tool_calls: Some(10),
                verbose: false,
                custom_config: HashMap::new(),
            },
        }
    }

    /// Set the agent description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.config.description = Some(description.into());
        self
    }

    /// Set the agent instructions (system prompt)
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.config.instructions = Some(instructions.into());
        self
    }

    /// Set the agent capabilities
    pub fn capabilities(mut self, capabilities: AgentCapabilities) -> Self {
        self.config.capabilities = capabilities;
        self
    }

    /// Enable tool support
    pub fn with_tools(mut self) -> Self {
        self.config.capabilities.supports_tools = true;
        self
    }

    /// Enable streaming support
    pub fn with_streaming(mut self) -> Self {
        self.config.capabilities.supports_streaming = true;
        self
    }

    /// Enable conversation support
    pub fn with_conversation(mut self) -> Self {
        self.config.capabilities.supports_conversation = true;
        self
    }

    /// Set maximum execution time
    pub fn max_execution_time(mut self, ms: u64) -> Self {
        self.config.max_execution_time_ms = Some(ms);
        self
    }

    /// Set maximum tool calls
    pub fn max_tool_calls(mut self, count: usize) -> Self {
        self.config.max_tool_calls = Some(count);
        self
    }

    /// Enable verbose logging
    pub fn verbose(mut self) -> Self {
        self.config.verbose = true;
        self
    }

    /// Add custom configuration
    pub fn custom_config(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.config.custom_config.insert(key.into(), value);
        self
    }

    /// Get the built configuration
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Build the final configuration
    pub fn build(self) -> AgentConfig {
        self.config
    }
}

/// Utility functions for agent management
pub mod utils {
    use super::*;

    /// Generate a new agent ID
    pub fn generate_agent_id() -> AgentId {
        Uuid::new_v4()
    }

    /// Validate agent configuration
    pub fn validate_config(config: &AgentConfig) -> Result<()> {
        if config.name.is_empty() {
            return Err(AgentError::invalid_configuration(
                "Agent name cannot be empty",
            ));
        }

        if let Some(max_time) = config.max_execution_time_ms {
            if max_time == 0 {
                return Err(AgentError::invalid_configuration(
                    "Max execution time must be greater than 0",
                ));
            }
        }

        if let Some(max_calls) = config.max_tool_calls {
            if max_calls == 0 {
                return Err(AgentError::invalid_configuration(
                    "Max tool calls must be greater than 0",
                ));
            }
        }

        Ok(())
    }

    /// Create default capabilities for a specific agent type
    pub fn default_capabilities_for(agent_type: &str) -> AgentCapabilities {
        match agent_type {
            "react" => AgentCapabilities {
                supports_tools: true,
                supports_streaming: true,
                supports_conversation: true,
                supports_files: false,
                supports_web: false,
                supports_code_execution: false,
                max_context_length: Some(4096),
                supported_input_formats: vec!["text".to_string()],
                supported_output_formats: vec!["text".to_string()],
                custom_capabilities: HashMap::new(),
            },
            "function_calling" => AgentCapabilities {
                supports_tools: true,
                supports_streaming: false,
                supports_conversation: true,
                supports_files: false,
                supports_web: false,
                supports_code_execution: false,
                max_context_length: Some(4096),
                supported_input_formats: vec!["text".to_string()],
                supported_output_formats: vec!["text".to_string()],
                custom_capabilities: HashMap::new(),
            },
            "simple" => AgentCapabilities {
                supports_tools: false,
                supports_streaming: false,
                supports_conversation: true,
                supports_files: false,
                supports_web: false,
                supports_code_execution: false,
                max_context_length: Some(2048),
                supported_input_formats: vec!["text".to_string()],
                supported_output_formats: vec!["text".to_string()],
                custom_capabilities: HashMap::new(),
            },
            _ => AgentCapabilities::default(),
        }
    }

    /// Convert AgentMessage to a simple text representation
    pub fn message_to_text(message: &AgentMessage) -> String {
        format!("[{}] {}", message.role_to_string(), message.content)
    }

    /// Create a simple agent context with basic setup
    pub fn create_basic_context() -> AgentContext {
        let mut context = AgentContext::new();
        context.set_variable(
            "created_at".to_string(),
            serde_json::json!(chrono::Utc::now()),
        );
        context
    }
}

/// Extension trait for AgentMessage to add utility methods
pub trait AgentMessageExt {
    /// Convert message role to string
    fn role_to_string(&self) -> &'static str;

    /// Check if message has tool calls
    fn has_tool_calls(&self) -> bool;

    /// Get tool call count
    fn tool_call_count(&self) -> usize;
}

impl AgentMessageExt for AgentMessage {
    fn role_to_string(&self) -> &'static str {
        match self.role {
            crate::types::MessageRole::User => "User",
            crate::types::MessageRole::Assistant => "Assistant",
            crate::types::MessageRole::System => "System",
            crate::types::MessageRole::Tool => "Tool",
        }
    }

    fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    fn tool_call_count(&self) -> usize {
        self.tool_calls.len()
    }
}
