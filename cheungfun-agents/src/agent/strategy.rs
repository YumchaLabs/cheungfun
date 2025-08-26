//! Agent execution strategies
//!
//! This module defines different strategies for agent execution, including
//! direct response, function calling, and `ReAct` patterns.

use crate::{
    error::{AgentError, Result},
    tool::ToolRegistry,
    types::AgentResponse,
};
use async_trait::async_trait;
use cheungfun_core::traits::BaseMemory;
use std::sync::Arc;

/// Agent execution strategy trait
#[async_trait]
pub trait AgentStrategy: Send + Sync + std::fmt::Debug {
    /// Get the strategy name
    fn name(&self) -> &'static str;

    /// Execute the strategy with a message and optional memory
    async fn execute(
        &self,
        message: &str,
        memory: Option<&mut dyn BaseMemory>,
        tools: &Arc<ToolRegistry>,
    ) -> Result<AgentResponse>;

    /// Check if this strategy supports tools
    fn supports_tools(&self) -> bool;

    /// Check if this strategy supports memory
    fn supports_memory(&self) -> bool;
}

/// Direct strategy - simple response without tools
#[derive(Debug, Clone)]
pub struct DirectStrategy {
    /// Agent name for responses
    agent_name: String,
}

impl DirectStrategy {
    /// Create a new direct strategy
    #[must_use]
    pub fn new(agent_name: String) -> Self {
        Self { agent_name }
    }
}

#[async_trait]
impl AgentStrategy for DirectStrategy {
    fn name(&self) -> &'static str {
        "direct"
    }

    async fn execute(
        &self,
        message: &str,
        memory: Option<&mut dyn BaseMemory>,
        _tools: &Arc<ToolRegistry>,
    ) -> Result<AgentResponse> {
        let start_time = std::time::Instant::now();

        // Simple direct response
        let response_content = format!(
            "Hello! I'm {}, a direct response agent. You said: '{}'\n\n\
            I'm operating in direct mode, which means I can provide responses \
            but I don't have access to tools or complex reasoning capabilities. \
            If you need more advanced functionality, please use an agent with \
            function calling or ReAct strategy.",
            self.agent_name, message
        );

        // Add to memory if available
        if let Some(mem) = memory {
            // Add user message
            let user_msg = cheungfun_core::ChatMessage {
                role: cheungfun_core::MessageRole::User,
                content: message.to_string(),
                timestamp: chrono::Utc::now(),
                metadata: None,
            };
            mem.add_message(user_msg).await.map_err(|e| {
                AgentError::generic(format!("Failed to add user message to memory: {e}"))
            })?;

            // Add assistant message
            let assistant_msg = cheungfun_core::ChatMessage {
                role: cheungfun_core::MessageRole::Assistant,
                content: response_content.clone(),
                timestamp: chrono::Utc::now(),
                metadata: None,
            };
            mem.add_message(assistant_msg).await.map_err(|e| {
                AgentError::generic(format!("Failed to add assistant message to memory: {e}"))
            })?;
        }

        let execution_time = start_time.elapsed().as_millis() as u64;

        Ok(AgentResponse {
            content: response_content,
            tool_calls: Vec::new(),
            tool_outputs: Vec::new(),
            metadata: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
            stats: crate::types::ExecutionStats {
                execution_time_ms: execution_time,
                tool_calls_count: 0,
                successful_tool_calls: 0,
                failed_tool_calls: 0,
                tokens_used: None,
                custom_metrics: std::collections::HashMap::new(),
            },
        })
    }

    fn supports_tools(&self) -> bool {
        false
    }

    fn supports_memory(&self) -> bool {
        true
    }
}

/// Function calling strategy - uses tools to respond
#[derive(Debug, Clone)]
pub struct FunctionCallingStrategy {
    /// Agent name for responses
    agent_name: String,
    /// Maximum number of tool calls per request
    max_tool_calls: usize,
}

impl FunctionCallingStrategy {
    /// Create a new function calling strategy
    #[must_use]
    pub fn new(agent_name: String) -> Self {
        Self {
            agent_name,
            max_tool_calls: 5,
        }
    }

    /// Set maximum tool calls
    #[must_use]
    pub fn with_max_tool_calls(mut self, max_calls: usize) -> Self {
        self.max_tool_calls = max_calls;
        self
    }
}

#[async_trait]
impl AgentStrategy for FunctionCallingStrategy {
    fn name(&self) -> &'static str {
        "function_calling"
    }

    async fn execute(
        &self,
        message: &str,
        memory: Option<&mut dyn BaseMemory>,
        tools: &Arc<ToolRegistry>,
    ) -> Result<AgentResponse> {
        let start_time = std::time::Instant::now();
        let mut tool_calls_made = 0;

        // Simple function calling logic
        let mut response_content = format!(
            "Hello! I'm {}, a function calling agent. You said: '{}'\n\n",
            self.agent_name, message
        );

        // Check if we should use tools based on the message
        let should_use_tools = message.to_lowercase().contains("echo")
            || message.to_lowercase().contains("http")
            || message.to_lowercase().contains("search")
            || message.to_lowercase().contains("file");

        let mut tool_calls = Vec::new();

        if should_use_tools && !tools.tool_names().is_empty() {
            // Try to use echo tool if available and message contains "echo"
            if message.to_lowercase().contains("echo") && tools.contains("echo") {
                if let Some(_echo_tool) = tools.get("echo") {
                    // Extract text to echo (simple heuristic)
                    let echo_text = if let Some(start) = message.find("echo") {
                        let after_echo = &message[start + 4..].trim();
                        if after_echo.starts_with('\'') && after_echo.ends_with('\'') {
                            after_echo[1..after_echo.len() - 1].to_string()
                        } else if after_echo.starts_with('"') && after_echo.ends_with('"') {
                            after_echo[1..after_echo.len() - 1].to_string()
                        } else {
                            after_echo.split_whitespace().collect::<Vec<_>>().join(" ")
                        }
                    } else {
                        "Hello from function calling agent!".to_string()
                    };

                    let tool_input = serde_json::json!({
                        "message": echo_text
                    });

                    match tools
                        .execute("echo", tool_input.clone(), &crate::tool::ToolContext::new())
                        .await
                    {
                        Ok(result) => {
                            response_content.push_str(&format!(
                                "I used the echo tool and got: {}\n\n",
                                result.content
                            ));
                            tool_calls_made += 1;

                            let call_id = uuid::Uuid::new_v4();

                            tool_calls.push(crate::types::ToolCall {
                                id: call_id,
                                tool_name: "echo".to_string(),
                                arguments: tool_input,
                                timestamp: chrono::Utc::now(),
                            });
                        }
                        Err(e) => {
                            response_content.push_str(&format!(
                                "I tried to use the echo tool but got an error: {e}\n\n"
                            ));
                        }
                    }
                }
            }
        }

        if tool_calls_made == 0 {
            response_content.push_str(&format!(
                "Available tools: {:?}\n\
                I can use these tools to help you. Try asking me to 'echo something' \
                or mention other tool names in your message.",
                tools.tool_names()
            ));
        }

        // Add to memory if available
        if let Some(mem) = memory {
            // Add user message
            let user_msg = cheungfun_core::ChatMessage {
                role: cheungfun_core::MessageRole::User,
                content: message.to_string(),
                timestamp: chrono::Utc::now(),
                metadata: None,
            };
            mem.add_message(user_msg).await.map_err(|e| {
                AgentError::generic(format!("Failed to add user message to memory: {e}"))
            })?;

            // Add assistant message
            let assistant_msg = cheungfun_core::ChatMessage {
                role: cheungfun_core::MessageRole::Assistant,
                content: response_content.clone(),
                timestamp: chrono::Utc::now(),
                metadata: None,
            };
            mem.add_message(assistant_msg).await.map_err(|e| {
                AgentError::generic(format!("Failed to add assistant message to memory: {e}"))
            })?;
        }

        let execution_time = start_time.elapsed().as_millis() as u64;

        Ok(AgentResponse {
            content: response_content,
            tool_calls,
            tool_outputs: Vec::new(),
            metadata: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
            stats: crate::types::ExecutionStats {
                execution_time_ms: execution_time,
                tool_calls_count: tool_calls_made,
                successful_tool_calls: if tool_calls_made > 0 {
                    tool_calls_made
                } else {
                    0
                },
                failed_tool_calls: 0,
                tokens_used: None,
                custom_metrics: std::collections::HashMap::new(),
            },
        })
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_memory(&self) -> bool {
        true
    }
}
