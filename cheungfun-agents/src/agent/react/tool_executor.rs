//! Tool execution system for ReAct agents.
//!
//! This module provides the `ToolExecutor` which handles the execution of tools
//! within the ReAct reasoning framework, including error handling, retries, and
//! result formatting.

use crate::{
    error::{AgentError, Result},
    tool::{Tool, ToolContext, ToolResult},
};
use std::{sync::Arc, time::Instant};
use tracing::{debug, error, info, warn};

/// Configuration for tool execution
#[derive(Debug, Clone)]
pub struct ToolExecutorConfig {
    /// Maximum execution time for a single tool call (in milliseconds)
    pub max_execution_time_ms: u64,
    /// Maximum number of retry attempts for failed tool calls
    pub max_retries: u32,
    /// Whether to include execution timing in results
    pub include_timing: bool,
    /// Whether to validate tool inputs before execution
    pub validate_inputs: bool,
}

impl Default for ToolExecutorConfig {
    fn default() -> Self {
        Self {
            max_execution_time_ms: 30_000, // 30 seconds
            max_retries: 2,
            include_timing: true,
            validate_inputs: true,
        }
    }
}

/// Tool executor for ReAct agents
#[derive(Debug)]
pub struct ToolExecutor {
    /// Configuration for tool execution
    config: ToolExecutorConfig,
    /// Available tools
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolExecutor {
    /// Create a new tool executor with default configuration
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        Self {
            config: ToolExecutorConfig::default(),
            tools,
        }
    }
    
    /// Create a tool executor with custom configuration
    pub fn with_config(tools: Vec<Arc<dyn Tool>>, config: ToolExecutorConfig) -> Self {
        Self { config, tools }
    }
    
    /// Execute a tool by name with the given arguments
    pub async fn execute_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
        context: &ToolContext,
    ) -> Result<ToolResult> {
        let start_time = Instant::now();
        
        debug!("Executing tool: {} with arguments: {}", tool_name, arguments);
        
        // Find the tool
        let tool = self.find_tool(tool_name)?;
        
        // Validate inputs if configured
        if self.config.validate_inputs {
            self.validate_tool_input(&tool, &arguments)?;
        }
        
        // Execute with retries
        let mut last_error = None;
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                warn!("Retrying tool execution (attempt {}/{})", attempt + 1, self.config.max_retries + 1);
            }
            
            match self.execute_with_timeout(&tool, &arguments, context).await {
                Ok(mut result) => {
                    let execution_time = start_time.elapsed();
                    
                    // Add timing information if configured
                    if self.config.include_timing {
                        result.metadata.insert(
                            "execution_time_ms".to_string(),
                            serde_json::json!(execution_time.as_millis()),
                        );
                        result.metadata.insert(
                            "attempts".to_string(),
                            serde_json::json!(attempt + 1),
                        );
                    }
                    
                    info!(
                        "Tool '{}' executed successfully in {}ms (attempt {})",
                        tool_name,
                        execution_time.as_millis(),
                        attempt + 1
                    );
                    
                    return Ok(result);
                }
                Err(e) => {
                    error!("Tool execution failed (attempt {}): {}", attempt + 1, e);
                    last_error = Some(e);
                    
                    // Don't retry on certain types of errors
                    if matches!(last_error.as_ref().unwrap(), AgentError::Tool { .. } | AgentError::Validation { .. }) {
                        break;
                    }
                }
            }
        }
        
        // All attempts failed
        Err(last_error.unwrap_or_else(|| {
            AgentError::Tool {
                tool_name: tool_name.to_string(),
                message: "Unknown execution error".to_string(),
            }
        }))
    }
    
    /// Find a tool by name
    fn find_tool(&self, tool_name: &str) -> Result<Arc<dyn Tool>> {
        self.tools
            .iter()
            .find(|tool| tool.name() == tool_name)
            .cloned()
            .ok_or_else(|| AgentError::Tool {
                tool_name: tool_name.to_string(),
                message: "Tool not found".to_string(),
            })
    }
    
    /// Validate tool input against its schema
    fn validate_tool_input(&self, tool: &Arc<dyn Tool>, arguments: &serde_json::Value) -> Result<()> {
        let schema = tool.schema();
        
        // Basic validation - check if required parameters are present
        if let Some(properties) = schema.input_schema.get("properties") {
            if let Some(required) = schema.input_schema.get("required") {
                if let Some(required_array) = required.as_array() {
                    for required_field in required_array {
                        if let Some(field_name) = required_field.as_str() {
                            if !arguments.get(field_name).is_some() {
                                return Err(AgentError::Validation {
                                    field: field_name.to_string(),
                                    message: format!(
                                        "Missing required parameter '{}' for tool '{}'",
                                        field_name,
                                        tool.name()
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Execute tool with timeout
    async fn execute_with_timeout(
        &self,
        tool: &Arc<dyn Tool>,
        arguments: &serde_json::Value,
        context: &ToolContext,
    ) -> Result<ToolResult> {
        // For now, we don't implement actual timeout - just execute directly
        // In a real implementation, you would use tokio::time::timeout
        tool.execute(arguments.clone(), context).await
    }
    
    /// Get list of available tool names
    pub fn available_tools(&self) -> Vec<String> {
        self.tools.iter().map(|tool| tool.name().to_string()).collect()
    }
    
    /// Get tool by name for inspection
    pub fn get_tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.iter().find(|tool| tool.name() == name).cloned()
    }
    
    /// Get execution statistics
    pub fn get_stats(&self) -> ToolExecutorStats {
        ToolExecutorStats {
            total_tools: self.tools.len(),
            config: self.config.clone(),
        }
    }
}

/// Statistics for tool executor
#[derive(Debug, Clone)]
pub struct ToolExecutorStats {
    /// Total number of available tools
    pub total_tools: usize,
    /// Current configuration
    pub config: ToolExecutorConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{ToolResult, ToolContext};
    use async_trait::async_trait;
    use std::collections::HashMap;

    #[derive(Debug)]
    struct MockTool {
        name: String,
        should_fail: bool,
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Mock tool for testing"
        }

        fn schema(&self) -> crate::types::ToolSchema {
            crate::types::ToolSchema {
                name: self.name.clone(),
                description: "Mock tool".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"}
                    },
                    "required": ["input"]
                }),
                output_schema: None,
                dangerous: false,
                metadata: HashMap::new(),
            }
        }

        async fn execute(
            &self,
            arguments: serde_json::Value,
            _context: &ToolContext,
        ) -> Result<ToolResult> {
            if self.should_fail {
                return Err(AgentError::ToolExecution {
                    tool_name: self.name.clone(),
                    message: "Mock failure".to_string(),
                });
            }

            Ok(ToolResult::success(format!(
                "Mock result for: {}",
                arguments.get("input").unwrap_or(&serde_json::json!("no input"))
            )))
        }
    }

    #[tokio::test]
    async fn test_tool_execution() {
        let tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(MockTool {
                name: "test_tool".to_string(),
                should_fail: false,
            }),
        ];

        let executor = ToolExecutor::new(tools);
        let context = ToolContext::new();
        let args = serde_json::json!({"input": "test"});

        let result = executor.execute_tool("test_tool", args, &context).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_tool_not_found() {
        let executor = ToolExecutor::new(vec![]);
        let context = ToolContext::new();
        let args = serde_json::json!({});

        let result = executor.execute_tool("nonexistent", args, &context).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AgentError::ToolNotFound(_)));
    }
}
