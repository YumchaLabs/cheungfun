//! Built-in tools for common agent operations.

use crate::{
    error::{AgentError, Result},
    tool::{create_simple_schema, Tool, ToolContext, ToolResult},
    types::ToolSchema,
};
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;

pub mod echo;
pub mod file;
pub mod http;
pub mod search;

pub use echo::EchoTool;
pub use file::FileTool;
pub use http::HttpTool;
pub use search::SearchTool;

/// Math tool for basic calculations
#[derive(Debug, Clone)]
pub struct MathTool {
    name: String,
}

impl MathTool {
    /// Create a new math tool
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "math".to_string(),
        }
    }
}

impl Default for MathTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for MathTool {
    fn schema(&self) -> ToolSchema {
        let mut properties = HashMap::new();
        properties.insert(
            "expression".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4')"
            }),
        );

        ToolSchema {
            name: self.name.clone(),
            description: "Evaluate mathematical expressions. Supports basic arithmetic operations."
                .to_string(),
            input_schema: create_simple_schema(properties, vec!["expression".to_string()]),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "result": {
                        "type": "number",
                        "description": "The calculated result"
                    },
                    "expression": {
                        "type": "string",
                        "description": "The original expression"
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
        struct MathArgs {
            expression: String,
        }

        let args: MathArgs = serde_json::from_value(arguments)
            .map_err(|e| AgentError::tool(&self.name, format!("Invalid arguments: {e}")))?;

        // Simple expression evaluation (in a real implementation, you'd use a proper math parser)
        let result = match self.evaluate_expression(&args.expression) {
            Ok(value) => {
                let content = format!("Result: {value}");
                ToolResult::success(content)
                    .with_metadata("result".to_string(), serde_json::json!(value))
                    .with_metadata("expression".to_string(), serde_json::json!(args.expression))
            }
            Err(e) => ToolResult::error(format!("Math evaluation error: {e}")),
        };

        Ok(result)
    }

    fn capabilities(&self) -> Vec<String> {
        vec!["calculation".to_string(), "math".to_string()]
    }
}

impl MathTool {
    /// Simple expression evaluator (basic implementation)
    fn evaluate_expression(&self, expr: &str) -> Result<f64> {
        // This is a very basic implementation
        // In a real tool, you'd use a proper expression parser like `evalexpr` crate
        let cleaned = expr.replace(' ', "");

        // Handle simple cases
        if let Ok(num) = cleaned.parse::<f64>() {
            return Ok(num);
        }

        // Basic addition
        if let Some(pos) = cleaned.find('+') {
            let left = cleaned[..pos]
                .parse::<f64>()
                .map_err(|_| AgentError::tool("math", "Invalid left operand"))?;
            let right = cleaned[pos + 1..]
                .parse::<f64>()
                .map_err(|_| AgentError::tool("math", "Invalid right operand"))?;
            return Ok(left + right);
        }

        // Basic subtraction
        if let Some(pos) = cleaned.rfind('-') {
            if pos > 0 {
                // Not a negative number
                let left = cleaned[..pos]
                    .parse::<f64>()
                    .map_err(|_| AgentError::tool("math", "Invalid left operand"))?;
                let right = cleaned[pos + 1..]
                    .parse::<f64>()
                    .map_err(|_| AgentError::tool("math", "Invalid right operand"))?;
                return Ok(left - right);
            }
        }

        // Basic multiplication
        if let Some(pos) = cleaned.find('*') {
            let left = cleaned[..pos]
                .parse::<f64>()
                .map_err(|_| AgentError::tool("math", "Invalid left operand"))?;
            let right = cleaned[pos + 1..]
                .parse::<f64>()
                .map_err(|_| AgentError::tool("math", "Invalid right operand"))?;
            return Ok(left * right);
        }

        // Basic division
        if let Some(pos) = cleaned.find('/') {
            let left = cleaned[..pos]
                .parse::<f64>()
                .map_err(|_| AgentError::tool("math", "Invalid left operand"))?;
            let right = cleaned[pos + 1..]
                .parse::<f64>()
                .map_err(|_| AgentError::tool("math", "Invalid right operand"))?;
            if right == 0.0 {
                return Err(AgentError::tool("math", "Division by zero"));
            }
            return Ok(left / right);
        }

        Err(AgentError::tool("math", "Unsupported expression"))
    }
}

/// Time tool for getting current time and date information
#[derive(Debug, Clone)]
pub struct TimeTool {
    name: String,
}

impl TimeTool {
    /// Create a new time tool
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "time".to_string(),
        }
    }
}

impl Default for TimeTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for TimeTool {
    fn schema(&self) -> ToolSchema {
        let mut properties = HashMap::new();
        properties.insert(
            "format".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Time format: 'iso', 'unix', 'human'",
                "enum": ["iso", "unix", "human"],
                "default": "iso"
            }),
        );

        ToolSchema {
            name: self.name.clone(),
            description: "Get current date and time in various formats.".to_string(),
            input_schema: create_simple_schema(properties, vec![]),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "description": "The formatted timestamp"
                    },
                    "format": {
                        "type": "string",
                        "description": "The format used"
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
        struct TimeArgs {
            #[serde(default = "default_format")]
            format: String,
        }

        fn default_format() -> String {
            "iso".to_string()
        }

        let args: TimeArgs = serde_json::from_value(arguments)
            .map_err(|e| AgentError::tool(&self.name, format!("Invalid arguments: {e}")))?;

        let now = chrono::Utc::now();
        let timestamp = match args.format.as_str() {
            "iso" => now.to_rfc3339(),
            "unix" => now.timestamp().to_string(),
            "human" => now.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
            _ => {
                return Ok(ToolResult::error(
                    "Invalid format. Use 'iso', 'unix', or 'human'",
                ));
            }
        };

        let result = ToolResult::success(format!("Current time: {timestamp}"))
            .with_metadata("timestamp".to_string(), serde_json::json!(timestamp))
            .with_metadata("format".to_string(), serde_json::json!(args.format));

        Ok(result)
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "time".to_string(),
            "date".to_string(),
            "utility".to_string(),
        ]
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
    }

    #[tokio::test]
    async fn test_math_tool() {
        let tool = MathTool::new();
        let context = ToolContext::new();

        // Test addition
        let args = serde_json::json!({"expression": "2 + 3"});
        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);
        assert!(result.content.contains("5"));

        // Test division by zero
        let args = serde_json::json!({"expression": "5 / 0"});
        let result = tool.execute(args, &context).await.unwrap();
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_time_tool() {
        let tool = TimeTool::new();
        let context = ToolContext::new();

        // Test ISO format
        let args = serde_json::json!({"format": "iso"});
        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);
        assert!(result.content.contains("Current time:"));

        // Test unix format
        let args = serde_json::json!({"format": "unix"});
        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);
    }
}
