//! Tool system for agent capabilities.

use crate::{
    error::{AgentError, Result},
    types::{ToolCallId, ToolSchema},
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod builtin;
pub mod registry;

pub use registry::ToolRegistry;

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Whether the tool execution was successful
    pub success: bool,
    /// Tool output content
    pub content: String,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Error message if execution failed
    pub error: Option<String>,
}

/// Tool execution context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolContext {
    /// Context variables available to the tool
    pub variables: HashMap<String, serde_json::Value>,
    /// Tool call ID for tracking
    pub call_id: Option<ToolCallId>,
    /// Additional context data
    pub data: HashMap<String, serde_json::Value>,
}

/// Core tool trait that all tools must implement
#[async_trait]
pub trait Tool: Send + Sync + std::fmt::Debug {
    /// Get the tool's schema definition
    fn schema(&self) -> ToolSchema;

    /// Execute the tool with given arguments and context
    async fn execute(
        &self,
        arguments: serde_json::Value,
        context: &ToolContext,
    ) -> Result<ToolResult>;

    /// Get the tool name
    fn name(&self) -> String {
        self.schema().name
    }

    /// Get the tool description
    fn description(&self) -> String {
        self.schema().description
    }

    /// Check if the tool is dangerous and requires confirmation
    fn is_dangerous(&self) -> bool {
        self.schema().dangerous
    }

    /// Validate tool arguments against the schema
    fn validate_arguments(&self, arguments: &serde_json::Value) -> Result<()> {
        // Basic validation - can be overridden for more sophisticated validation
        if arguments.is_null() {
            return Err(AgentError::validation(
                "arguments",
                "Tool arguments cannot be null",
            ));
        }
        Ok(())
    }

    /// Get tool capabilities/features
    fn capabilities(&self) -> Vec<String> {
        Vec::new()
    }

    /// Check if tool supports streaming output
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Execute tool with streaming output (if supported)
    async fn execute_streaming(
        &self,
        _arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String>> + Send + Unpin>> {
        Err(AgentError::tool(
            self.name(),
            "Streaming execution not supported",
        ))
    }
}

impl ToolResult {
    /// Create a successful tool result
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            success: true,
            content: content.into(),
            metadata: HashMap::new(),
            error: None,
        }
    }

    /// Create a successful tool result with metadata
    pub fn success_with_metadata(
        content: impl Into<String>,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            success: true,
            content: content.into(),
            metadata,
            error: None,
        }
    }

    /// Create a failed tool result
    pub fn error(error: impl Into<String>) -> Self {
        Self {
            success: false,
            content: String::new(),
            metadata: HashMap::new(),
            error: Some(error.into()),
        }
    }

    /// Create a failed tool result with partial content
    pub fn error_with_content(content: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            success: false,
            content: content.into(),
            metadata: HashMap::new(),
            error: Some(error.into()),
        }
    }

    /// Add metadata to the result
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Check if the result contains an error
    #[must_use]
    pub fn is_error(&self) -> bool {
        !self.success || self.error.is_some()
    }

    /// Get the error message if any
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

impl ToolContext {
    /// Create a new tool context
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a tool context with a call ID
    #[must_use]
    pub fn with_call_id(call_id: ToolCallId) -> Self {
        Self {
            call_id: Some(call_id),
            ..Default::default()
        }
    }

    /// Add a variable to the context
    pub fn with_variable(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.variables.insert(key.into(), value);
        self
    }

    /// Add data to the context
    pub fn with_data(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.data.insert(key.into(), value);
        self
    }

    /// Get a variable from the context
    #[must_use]
    pub fn get_variable(&self, key: &str) -> Option<&serde_json::Value> {
        self.variables.get(key)
    }

    /// Get data from the context
    #[must_use]
    pub fn get_data(&self, key: &str) -> Option<&serde_json::Value> {
        self.data.get(key)
    }

    /// Get a typed variable from the context
    pub fn get_typed_variable<T>(&self, key: &str) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let value = self.get_variable(key).ok_or_else(|| {
            AgentError::validation(key, format!("Variable '{key}' not found in context"))
        })?;

        serde_json::from_value(value.clone()).map_err(|e| {
            AgentError::validation(key, format!("Failed to deserialize variable '{key}': {e}"))
        })
    }
}

/// Helper macro for creating tool schemas
#[macro_export]
macro_rules! tool_schema {
    (
        name: $name:expr,
        description: $description:expr,
        input_schema: $input_schema:expr
    ) => {
        $crate::types::ToolSchema {
            name: $name.to_string(),
            description: $description.to_string(),
            input_schema: $input_schema,
            output_schema: None,
            dangerous: false,
            metadata: std::collections::HashMap::new(),
        }
    };
    (
        name: $name:expr,
        description: $description:expr,
        input_schema: $input_schema:expr,
        dangerous: $dangerous:expr
    ) => {
        $crate::types::ToolSchema {
            name: $name.to_string(),
            description: $description.to_string(),
            input_schema: $input_schema,
            output_schema: None,
            dangerous: $dangerous,
            metadata: std::collections::HashMap::new(),
        }
    };
}

/// Helper function to create a simple JSON schema for tool parameters
#[must_use]
pub fn create_simple_schema(
    properties: HashMap<String, serde_json::Value>,
    required: Vec<String>,
) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required
    })
}

/// Helper function to create a string parameter schema
#[must_use]
pub fn string_param(description: &str, required: bool) -> (serde_json::Value, bool) {
    (
        serde_json::json!({
            "type": "string",
            "description": description
        }),
        required,
    )
}

/// Helper function to create a number parameter schema
#[must_use]
pub fn number_param(description: &str, required: bool) -> (serde_json::Value, bool) {
    (
        serde_json::json!({
            "type": "number",
            "description": description
        }),
        required,
    )
}

/// Helper function to create a boolean parameter schema
#[must_use]
pub fn boolean_param(description: &str, required: bool) -> (serde_json::Value, bool) {
    (
        serde_json::json!({
            "type": "boolean",
            "description": description
        }),
        required,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_creation() {
        let result = ToolResult::success("Hello, world!");
        assert!(result.success);
        assert_eq!(result.content, "Hello, world!");
        assert!(!result.is_error());
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("Something went wrong");
        assert!(!result.success);
        assert!(result.is_error());
        assert_eq!(result.error_message(), Some("Something went wrong"));
    }

    #[test]
    fn test_tool_context() {
        let context = ToolContext::new()
            .with_variable("key1", serde_json::json!("value1"))
            .with_data("data1", serde_json::json!(42));

        assert_eq!(
            context.get_variable("key1"),
            Some(&serde_json::json!("value1"))
        );
        assert_eq!(context.get_data("data1"), Some(&serde_json::json!(42)));
    }

    #[test]
    fn test_schema_helpers() {
        let mut properties = HashMap::new();
        properties.insert("name".to_string(), serde_json::json!({"type": "string"}));

        let schema = create_simple_schema(properties, vec!["name".to_string()]);
        assert!(schema["properties"]["name"].is_object());
        assert_eq!(schema["required"], serde_json::json!(["name"]));
    }
}
