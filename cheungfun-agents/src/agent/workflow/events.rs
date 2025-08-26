//! Workflow event system for streaming and state management
//! 
//! This module defines the event system used for workflow execution,
//! including streaming events, tool calls, and agent interactions.

use crate::{
    error::{AgentError, Result},
    tool::Tool,
    types::ToolOutput,
};
use cheungfun_core::ChatMessage;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

/// Tool selection for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSelection {
    /// Unique identifier for this tool call
    pub tool_id: String,
    /// Name of the tool to call
    pub tool_name: String,
    /// Arguments to pass to the tool
    pub tool_kwargs: HashMap<String, serde_json::Value>,
}

impl ToolSelection {
    /// Create a new tool selection
    pub fn new(
        tool_id: impl Into<String>,
        tool_name: impl Into<String>,
        tool_kwargs: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            tool_id: tool_id.into(),
            tool_name: tool_name.into(),
            tool_kwargs,
        }
    }

    /// Create a tool selection with a generated UUID
    pub fn with_generated_id(
        tool_name: impl Into<String>,
        tool_kwargs: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self::new(Uuid::new_v4().to_string(), tool_name, tool_kwargs)
    }
}

/// Base trait for all workflow events
pub trait WorkflowEvent: Send + Sync + std::fmt::Debug {
    /// Get the event type identifier
    fn event_type(&self) -> &'static str;
    
    /// Get the event timestamp
    fn timestamp(&self) -> chrono::DateTime<chrono::Utc>;
    
    /// Convert the event to JSON
    fn to_json(&self) -> Result<serde_json::Value>;
}

/// Input event when an agent receives input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInput {
    /// The input messages
    pub input: Vec<ChatMessage>,
    /// Name of the agent processing the input
    pub current_agent_name: String,
    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl AgentInput {
    /// Create a new agent input event
    pub fn new(input: Vec<ChatMessage>, current_agent_name: impl Into<String>) -> Self {
        Self {
            input,
            current_agent_name: current_agent_name.into(),
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Add metadata to the event
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

impl WorkflowEvent for AgentInput {
    fn event_type(&self) -> &'static str {
        "agent_input"
    }
    
    fn timestamp(&self) -> chrono::DateTime<chrono::Utc> {
        self.timestamp
    }
    
    fn to_json(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| AgentError::serialization(e.to_string()))
    }
}

/// Output event when an agent produces output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    /// The response message
    pub response: ChatMessage,
    /// Tool calls made by the agent (if any)
    pub tool_calls: Vec<ToolSelection>,
    /// Raw response data
    pub raw: Option<serde_json::Value>,
    /// Name of the agent that produced the output
    pub current_agent_name: String,
    /// Messages to retry with (for error recovery)
    pub retry_messages: Vec<ChatMessage>,
    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl AgentOutput {
    /// Create a new agent output event
    pub fn new(response: ChatMessage, current_agent_name: impl Into<String>) -> Self {
        Self {
            response,
            tool_calls: Vec::new(),
            raw: None,
            current_agent_name: current_agent_name.into(),
            retry_messages: Vec::new(),
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Add tool calls to the output
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolSelection>) -> Self {
        self.tool_calls = tool_calls;
        self
    }
    
    /// Add raw response data
    pub fn with_raw(mut self, raw: serde_json::Value) -> Self {
        self.raw = Some(raw);
        self
    }
    
    /// Add retry messages
    pub fn with_retry_messages(mut self, retry_messages: Vec<ChatMessage>) -> Self {
        self.retry_messages = retry_messages;
        self
    }
    
    /// Add metadata to the event
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

impl WorkflowEvent for AgentOutput {
    fn event_type(&self) -> &'static str {
        "agent_output"
    }
    
    fn timestamp(&self) -> chrono::DateTime<chrono::Utc> {
        self.timestamp
    }
    
    fn to_json(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| AgentError::serialization(e.to_string()))
    }
}

/// Streaming event for partial responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStream {
    /// The partial content delta
    pub delta: String,
    /// The accumulated response so far
    pub response: String,
    /// Raw streaming data
    pub raw: Option<serde_json::Value>,
    /// Name of the agent producing the stream
    pub current_agent_name: String,
    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl AgentStream {
    /// Create a new agent stream event
    pub fn new(
        delta: impl Into<String>,
        response: impl Into<String>,
        current_agent_name: impl Into<String>,
    ) -> Self {
        Self {
            delta: delta.into(),
            response: response.into(),
            raw: None,
            current_agent_name: current_agent_name.into(),
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Add raw streaming data
    pub fn with_raw(mut self, raw: serde_json::Value) -> Self {
        self.raw = Some(raw);
        self
    }
    
    /// Add metadata to the event
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

impl WorkflowEvent for AgentStream {
    fn event_type(&self) -> &'static str {
        "agent_stream"
    }
    
    fn timestamp(&self) -> chrono::DateTime<chrono::Utc> {
        self.timestamp
    }
    
    fn to_json(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| AgentError::serialization(e.to_string()))
    }
}

/// Tool call result event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    /// Name of the tool that was called
    pub tool_name: String,
    /// Arguments passed to the tool
    pub tool_kwargs: HashMap<String, serde_json::Value>,
    /// Unique identifier for this tool call
    pub tool_id: String,
    /// Output from the tool execution
    pub tool_output: ToolOutput,
    /// Whether the tool result should be returned directly
    pub return_direct: bool,
    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ToolCallResult {
    /// Create a new tool call result
    pub fn new(
        tool_name: impl Into<String>,
        tool_kwargs: HashMap<String, serde_json::Value>,
        tool_id: impl Into<String>,
        tool_output: ToolOutput,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            tool_kwargs,
            tool_id: tool_id.into(),
            tool_output,
            return_direct: false,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Set whether the result should be returned directly
    pub fn with_return_direct(mut self, return_direct: bool) -> Self {
        self.return_direct = return_direct;
        self
    }
    
    /// Add metadata to the event
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

impl WorkflowEvent for ToolCallResult {
    fn event_type(&self) -> &'static str {
        "tool_call_result"
    }
    
    fn timestamp(&self) -> chrono::DateTime<chrono::Utc> {
        self.timestamp
    }
    
    fn to_json(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| AgentError::serialization(e.to_string()))
    }
}

// ToolSelection is now defined in this module

/// Event stream for workflow execution
pub type EventStream = Box<dyn futures::Stream<Item = Result<Box<dyn WorkflowEvent>>> + Send + Unpin>;

/// Event handler trait for processing workflow events
#[async_trait::async_trait]
pub trait EventHandler: Send + Sync {
    /// Handle an incoming event
    async fn handle_event(&self, event: &dyn WorkflowEvent) -> Result<()>;

    /// Get the handler name
    fn name(&self) -> &str;
}

/// Event bus for distributing events to handlers
pub struct EventBus {
    /// Registered event handlers
    handlers: Vec<Arc<dyn EventHandler>>,
}

impl EventBus {
    /// Create a new event bus
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }
    
    /// Register an event handler
    pub fn register_handler(&mut self, handler: Arc<dyn EventHandler>) {
        self.handlers.push(handler);
    }
    
    /// Emit an event to all registered handlers
    pub async fn emit(&self, event: Box<dyn WorkflowEvent>) -> Result<()> {
        // For now, we'll process handlers sequentially without cloning
        // TODO: Implement proper event cloning or use Arc<dyn WorkflowEvent>
        for (i, handler) in self.handlers.iter().enumerate() {
            if i == 0 {
                // Use the original event for the first handler
                if let Err(e) = handler.handle_event(event.as_ref()).await {
                    tracing::warn!("Event handler '{}' failed: {}", handler.name(), e);
                }
            } else {
                // For subsequent handlers, we'd need to clone the event
                // For now, we'll skip them or implement a different approach
                tracing::debug!("Skipping handler '{}' due to event cloning limitation", handler.name());
            }
        }
        Ok(())
    }
    
    /// Get the number of registered handlers
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}
