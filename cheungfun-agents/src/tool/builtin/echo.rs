//! Echo tool implementation - simple tool for testing and debugging.

use crate::{
    error::{AgentError, Result},
    tool::{Tool, ToolContext, ToolResult, create_simple_schema, string_param},
    types::ToolSchema,
};
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;

/// Echo tool for testing and debugging
#[derive(Debug, Clone)]
pub struct EchoTool {
    name: String,
}

impl EchoTool {
    /// Create a new echo tool
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "echo".to_string(),
        }
    }
}

impl Default for EchoTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for EchoTool {
    fn schema(&self) -> ToolSchema {
        let mut properties = HashMap::new();
        let (message_schema, _) = string_param("Message to echo back", true);
        properties.insert("message".to_string(), message_schema);

        ToolSchema {
            name: self.name.clone(),
            description: "Echo back the provided message. Useful for testing and debugging."
                .to_string(),
            input_schema: create_simple_schema(properties, vec!["message".to_string()]),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "echoed_message": {
                        "type": "string",
                        "description": "The echoed message"
                    }
                }
            })),
            dangerous: false,
            metadata: HashMap::new(),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult> {
        #[derive(Deserialize)]
        struct EchoArgs {
            message: String,
        }

        let args: EchoArgs = serde_json::from_value(arguments)
            .map_err(|e| AgentError::tool(&self.name, format!("Invalid arguments: {e}")))?;

        let result = ToolResult::success(&args.message).with_metadata(
            "echoed_message".to_string(),
            serde_json::json!(args.message),
        );

        Ok(result)
    }

    fn capabilities(&self) -> Vec<String> {
        vec!["testing".to_string(), "debugging".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_echo_tool() {
        let tool = EchoTool::new();
        let context = ToolContext::new();
        let args = serde_json::json!({"message": "Hello, World!"});

        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);
        assert_eq!(result.content, "Hello, World!");
        assert_eq!(
            result.metadata.get("echoed_message"),
            Some(&serde_json::json!("Hello, World!"))
        );
    }

    #[test]
    fn test_echo_tool_schema() {
        let tool = EchoTool::new();
        let schema = tool.schema();

        assert_eq!(schema.name, "echo");
        assert!(!schema.description.is_empty());
        assert!(!schema.dangerous);
        assert!(
            schema.input_schema["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("message"))
        );
    }

    #[tokio::test]
    async fn test_echo_tool_invalid_args() {
        let tool = EchoTool::new();
        let context = ToolContext::new();
        let args = serde_json::json!({"wrong_field": "test"});

        let result = tool.execute(args, &context).await;
        assert!(result.is_err());
    }
}
