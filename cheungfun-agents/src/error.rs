//! Error types for the Cheungfun agents framework.

use cheungfun_core::CheungfunError;
use thiserror::Error;

/// Result type alias for agent operations
pub type Result<T> = std::result::Result<T, AgentError>;

/// Comprehensive error types for agent operations
#[derive(Error, Debug)]
pub enum AgentError {
    /// Core Cheungfun errors
    #[error("Core error: {0}")]
    Core(#[from] CheungfunError),

    /// Agent configuration errors
    #[error("Agent configuration error: {message}")]
    Configuration {
        /// Error message
        message: String,
    },

    /// Agent execution errors
    #[error("Agent execution error: {message}")]
    Execution {
        /// Error message
        message: String,
    },

    /// Tool-related errors
    #[error("Tool error: {tool_name} - {message}")]
    Tool {
        /// Tool name
        tool_name: String,
        /// Error message
        message: String,
    },

    /// MCP protocol errors
    #[error("MCP protocol error: {message}")]
    Mcp {
        /// Error message
        message: String,
    },

    /// Task processing errors
    #[error("Task error: {task_id} - {message}")]
    Task {
        /// Task ID
        task_id: String,
        /// Error message
        message: String,
    },

    /// Orchestration errors
    #[error("Orchestration error: {message}")]
    Orchestration {
        /// Error message
        message: String,
    },

    /// Workflow errors
    #[error("Workflow error: {workflow_id} - {message}")]
    Workflow {
        /// Workflow ID
        workflow_id: String,
        /// Error message
        message: String,
    },

    /// Agent communication errors
    #[error("Communication error: {message}")]
    Communication {
        /// Error message
        message: String,
    },

    /// Resource access errors
    #[error("Resource access error: {resource} - {message}")]
    ResourceAccess {
        /// Resource name
        resource: String,
        /// Error message
        message: String,
    },

    /// Timeout errors
    #[error("Timeout error: {operation} took longer than {timeout_ms}ms")]
    Timeout {
        /// Operation name
        operation: String,
        /// Timeout in milliseconds
        timeout_ms: u64,
    },

    /// Authentication/authorization errors
    #[error("Authentication error: {message}")]
    Authentication {
        /// Error message
        message: String,
    },

    /// Validation errors
    #[error("Validation error: {field} - {message}")]
    Validation {
        /// Field name
        field: String,
        /// Error message
        message: String,
    },

    /// LLM-related errors
    #[error("LLM error: {message}")]
    LlmError {
        /// Error message
        message: String,
    },

    /// Unsupported provider error
    #[error("Unsupported provider: {provider}")]
    UnsupportedProvider {
        /// Provider name
        provider: String,
    },

    /// Missing API key error
    #[error("Missing API key for provider: {provider}")]
    MissingApiKey {
        /// Provider name
        provider: String,
    },

    /// Invalid configuration error
    #[error("Invalid configuration: {message}")]
    InvalidConfiguration {
        /// Error message
        message: String,
    },

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid input error
    #[error("Invalid input: {message}")]
    InvalidInput {
        /// Error message
        message: String,
    },

    /// Not implemented error
    #[error("Not implemented: {message}")]
    NotImplemented {
        /// Error message
        message: String,
    },

    /// No tools available error
    #[error("No tools available")]
    NoToolsAvailable,

    /// Unsupported operation error
    #[error("Unsupported operation: {message}")]
    UnsupportedOperation {
        /// Error message
        message: String,
    },

    /// Generic errors with context
    #[error("Agent error: {message}")]
    Generic {
        /// Error message
        message: String,
    },

    /// Resource limit error
    #[error("Resource limit error: {message}")]
    ResourceLimit {
        /// Error message
        message: String,
    },

    /// Memory error
    #[error("Memory error: {message}")]
    Memory {
        /// Error message
        message: String,
    },

    /// Parsing error
    #[error("Parsing error: {message}")]
    Parsing {
        /// Error message
        message: String,
    },
}

impl AgentError {
    /// Create a configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create an execution error
    pub fn execution(message: impl Into<String>) -> Self {
        Self::Execution {
            message: message.into(),
        }
    }

    /// Create a tool error
    pub fn tool(tool_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Tool {
            tool_name: tool_name.into(),
            message: message.into(),
        }
    }

    /// Create an MCP error
    pub fn mcp(message: impl Into<String>) -> Self {
        Self::Mcp {
            message: message.into(),
        }
    }

    /// Create a task error
    pub fn task(task_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Task {
            task_id: task_id.into(),
            message: message.into(),
        }
    }

    /// Create an orchestration error
    pub fn orchestration(message: impl Into<String>) -> Self {
        Self::Orchestration {
            message: message.into(),
        }
    }

    /// Create a workflow error
    pub fn workflow(workflow_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Workflow {
            workflow_id: workflow_id.into(),
            message: message.into(),
        }
    }

    /// Create a communication error
    pub fn communication(message: impl Into<String>) -> Self {
        Self::Communication {
            message: message.into(),
        }
    }

    /// Create a resource access error
    pub fn resource_access(resource: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ResourceAccess {
            resource: resource.into(),
            message: message.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout(operation: impl Into<String>, timeout_ms: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            timeout_ms,
        }
    }

    /// Create an authentication error
    pub fn authentication(message: impl Into<String>) -> Self {
        Self::Authentication {
            message: message.into(),
        }
    }

    /// Create a validation error
    pub fn validation(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Validation {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Create a serialization error
    pub fn serialization(message: impl Into<String>) -> Self {
        use serde::de::Error as DeError;
        let json_error = serde_json::Error::custom(message.into());
        Self::Serialization(json_error)
    }

    /// Create a resource limit error
    pub fn resource_limit(message: impl Into<String>) -> Self {
        Self::ResourceLimit {
            message: message.into(),
        }
    }

    /// Create a memory error
    pub fn memory(message: impl Into<String>) -> Self {
        Self::Memory {
            message: message.into(),
        }
    }

    /// Create a parsing error
    pub fn parsing(message: impl Into<String>) -> Self {
        Self::Parsing {
            message: message.into(),
        }
    }

    /// Create an LLM error
    pub fn llm_error<S: Into<String>>(message: S) -> Self {
        Self::LlmError {
            message: message.into(),
        }
    }

    /// Create an unsupported provider error
    pub fn unsupported_provider<S: Into<String>>(provider: S) -> Self {
        Self::UnsupportedProvider {
            provider: provider.into(),
        }
    }

    /// Create a missing API key error
    pub fn missing_api_key<S: Into<String>>(provider: S) -> Self {
        Self::MissingApiKey {
            provider: provider.into(),
        }
    }

    /// Create an invalid configuration error
    pub fn invalid_configuration<S: Into<String>>(message: S) -> Self {
        Self::InvalidConfiguration {
            message: message.into(),
        }
    }

    /// Create an invalid input error
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// Create a not implemented error
    pub fn not_implemented<S: Into<String>>(message: S) -> Self {
        Self::NotImplemented {
            message: message.into(),
        }
    }

    /// Create a no tools available error
    #[must_use]
    pub fn no_tools_available() -> Self {
        Self::NoToolsAvailable
    }

    /// Create an unsupported operation error
    pub fn unsupported_operation<S: Into<String>>(message: S) -> Self {
        Self::UnsupportedOperation {
            message: message.into(),
        }
    }

    /// Create a generic error
    pub fn generic(message: impl Into<String>) -> Self {
        Self::Generic {
            message: message.into(),
        }
    }

    /// Check if this is a retryable error
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Communication { .. }
                | Self::Timeout { .. }
                | Self::Io(_)
                | Self::Mcp { .. }
                | Self::ResourceAccess { .. }
        )
    }

    /// Get the error category for logging/metrics
    #[must_use]
    pub fn category(&self) -> &'static str {
        match self {
            Self::Core(_) => "core",
            Self::Configuration { .. } => "configuration",
            Self::Execution { .. } => "execution",
            Self::Tool { .. } => "tool",
            Self::Mcp { .. } => "mcp",
            Self::Task { .. } => "task",
            Self::Orchestration { .. } => "orchestration",
            Self::Workflow { .. } => "workflow",
            Self::Communication { .. } => "communication",
            Self::ResourceAccess { .. } => "resource_access",
            Self::Timeout { .. } => "timeout",
            Self::Authentication { .. } => "authentication",
            Self::Validation { .. } => "validation",
            Self::Serialization(_) => "serialization",
            Self::Io(_) => "io",
            Self::Generic { .. } => "generic",
            Self::LlmError { .. } => "llm_error",
            Self::UnsupportedProvider { .. } => "unsupported_provider",
            Self::MissingApiKey { .. } => "missing_api_key",
            Self::InvalidConfiguration { .. } => "invalid_configuration",
            Self::InvalidInput { .. } => "invalid_input",
            Self::NotImplemented { .. } => "not_implemented",
            Self::NoToolsAvailable => "no_tools_available",
            Self::UnsupportedOperation { .. } => "unsupported_operation",
            Self::ResourceLimit { .. } => "resource_limit",
            Self::Memory { .. } => "memory",
            Self::Parsing { .. } => "parsing",
        }
    }
}

// Convert from rmcp errors
impl From<rmcp::ErrorData> for AgentError {
    fn from(err: rmcp::ErrorData) -> Self {
        Self::mcp(format!("RMCP error: {err}"))
    }
}

// Convert from anyhow errors
impl From<anyhow::Error> for AgentError {
    fn from(err: anyhow::Error) -> Self {
        Self::generic(format!("Anyhow error: {err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = AgentError::configuration("Invalid config");
        assert!(matches!(err, AgentError::Configuration { .. }));
        assert_eq!(err.category(), "configuration");
    }

    #[test]
    fn test_error_retryable() {
        let timeout_err = AgentError::timeout("operation", 5000);
        assert!(timeout_err.is_retryable());

        let config_err = AgentError::configuration("bad config");
        assert!(!config_err.is_retryable());
    }

    #[test]
    fn test_error_display() {
        let err = AgentError::tool("calculator", "division by zero");
        let display = format!("{err}");
        assert!(display.contains("calculator"));
        assert!(display.contains("division by zero"));
    }
}
