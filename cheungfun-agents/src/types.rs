//! Core types and data structures for the Cheungfun agents framework.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for agents
pub type AgentId = Uuid;

/// Unique identifier for tasks
pub type TaskId = Uuid;

/// Unique identifier for workflows
pub type WorkflowId = Uuid;

/// Unique identifier for tool calls
pub type ToolCallId = Uuid;

/// Agent message for communication between agents and users
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentMessage {
    /// Unique message identifier
    pub id: Uuid,
    /// Message content
    pub content: String,
    /// Message role (user, assistant, system, tool)
    pub role: MessageRole,
    /// Message metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp when the message was created
    pub timestamp: DateTime<Utc>,
    /// Optional tool calls in this message
    pub tool_calls: Vec<ToolCall>,
}

/// Message role in conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageRole {
    /// User message
    User,
    /// Assistant/agent message
    Assistant,
    /// System message
    System,
    /// Tool response message
    Tool,
}

/// Agent response containing the result of agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    /// Response content
    pub content: String,
    /// Response metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Tool calls made during execution
    pub tool_calls: Vec<ToolCall>,
    /// Tool outputs received
    pub tool_outputs: Vec<ToolOutput>,
    /// Execution statistics
    pub stats: ExecutionStats,
    /// Response timestamp
    pub timestamp: DateTime<Utc>,
}

/// Tool call information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Unique call identifier
    pub id: ToolCallId,
    /// Tool name
    pub tool_name: String,
    /// Tool arguments
    pub arguments: serde_json::Value,
    /// Call timestamp
    pub timestamp: DateTime<Utc>,
}

/// Tool execution output (for compatibility with existing code)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolOutput {
    /// Output content
    pub content: String,
    /// Whether the execution was an error
    pub is_error: bool,
    /// Output metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Execution statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutionStats {
    /// Total execution time in milliseconds
    pub execution_time_ms: u64,
    /// Number of tool calls made
    pub tool_calls_count: usize,
    /// Number of successful tool calls
    pub successful_tool_calls: usize,
    /// Number of failed tool calls
    pub failed_tool_calls: usize,
    /// Total tokens used (if applicable)
    pub tokens_used: Option<usize>,
    /// Additional custom metrics
    pub custom_metrics: HashMap<String, serde_json::Value>,
}

/// Agent capabilities definition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentCapabilities {
    /// Whether the agent supports tool calling
    pub supports_tools: bool,
    /// Whether the agent supports streaming responses
    pub supports_streaming: bool,
    /// Whether the agent supports multi-turn conversations
    pub supports_conversation: bool,
    /// Whether the agent supports file operations
    pub supports_files: bool,
    /// Whether the agent supports web access
    pub supports_web: bool,
    /// Whether the agent supports code execution
    pub supports_code_execution: bool,
    /// Maximum context length the agent can handle
    pub max_context_length: Option<usize>,
    /// Supported input formats
    pub supported_input_formats: Vec<String>,
    /// Supported output formats
    pub supported_output_formats: Vec<String>,
    /// Custom capabilities
    pub custom_capabilities: HashMap<String, serde_json::Value>,
}

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent name
    pub name: String,
    /// Agent description
    pub description: Option<String>,
    /// Agent instructions/system prompt
    pub instructions: Option<String>,
    /// Agent capabilities
    pub capabilities: AgentCapabilities,
    /// Maximum execution time in milliseconds
    pub max_execution_time_ms: Option<u64>,
    /// Maximum number of tool calls per task
    pub max_tool_calls: Option<usize>,
    /// Whether to enable verbose logging
    pub verbose: bool,
    /// Custom configuration parameters
    pub custom_config: HashMap<String, serde_json::Value>,
}

/// Tool schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Input schema (JSON Schema)
    pub input_schema: serde_json::Value,
    /// Output schema (JSON Schema)
    pub output_schema: Option<serde_json::Value>,
    /// Whether the tool is dangerous/requires confirmation
    pub dangerous: bool,
    /// Tool metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl AgentMessage {
    /// Create a new user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            role: MessageRole::User,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            tool_calls: Vec::new(),
        }
    }

    /// Create a new assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            role: MessageRole::Assistant,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            tool_calls: Vec::new(),
        }
    }

    /// Create a new system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            role: MessageRole::System,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            tool_calls: Vec::new(),
        }
    }

    /// Add a tool call to this message
    #[must_use]
    pub fn with_tool_call(mut self, tool_call: ToolCall) -> Self {
        self.tool_calls.push(tool_call);
        self
    }

    /// Add metadata to this message
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

impl ToolCall {
    /// Create a new tool call
    pub fn new(tool_name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            tool_name: tool_name.into(),
            arguments,
            timestamp: Utc::now(),
        }
    }
}

impl ToolOutput {
    /// Create a successful tool output
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            metadata: HashMap::new(),
        }
    }

    /// Create a failed tool output
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            metadata: HashMap::new(),
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "default_agent".to_string(),
            description: None,
            instructions: None,
            capabilities: AgentCapabilities::default(),
            max_execution_time_ms: Some(30_000), // 30 seconds
            max_tool_calls: Some(10),
            verbose: false,
            custom_config: HashMap::new(),
        }
    }
}
