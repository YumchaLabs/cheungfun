//! Core agent framework and implementations.

use crate::{
    error::{AgentError, Result},
    task::{Task, TaskContext},
    tool::{Tool, ToolContext, ToolRegistry},
    types::{
        AgentCapabilities, AgentConfig, AgentId, AgentMessage, AgentResponse, ExecutionStats,
        ToolCall, ToolOutput,
    },
};
use async_trait::async_trait;
use chrono::Utc;
use std::{collections::HashMap, sync::Arc, time::Instant};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

pub mod builder;
pub mod executor;
pub mod rag_agent;

pub use builder::AgentBuilder;
pub use executor::AgentExecutor;
pub use rag_agent::{RagAgent, RagAgentConfig, RagAgentStats};

/// Core agent trait that all agents must implement
#[async_trait]
pub trait Agent: Send + Sync + std::fmt::Debug {
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

    /// Execute a task and return the result
    async fn execute(&self, task: &Task) -> Result<AgentResponse>;

    /// Execute a task with streaming response
    async fn execute_streaming(
        &self,
        task: &Task,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String>> + Send + Unpin>> {
        // Default implementation: execute normally and return single item stream
        let response = self.execute(task).await?;
        let stream = futures::stream::once(async move { Ok(response.content) });
        Ok(Box::new(Box::pin(stream)))
    }

    /// Process a message and return a response
    async fn process_message(&self, message: &AgentMessage) -> Result<AgentResponse>;

    /// Get available tools
    fn tools(&self) -> Vec<String>;

    /// Check if the agent supports a specific capability
    fn supports_capability(&self, capability: &str) -> bool {
        match capability {
            "tools" => self.capabilities().supports_tools,
            "streaming" => self.capabilities().supports_streaming,
            "conversation" => self.capabilities().supports_conversation,
            "files" => self.capabilities().supports_files,
            "web" => self.capabilities().supports_web,
            "code_execution" => self.capabilities().supports_code_execution,
            _ => self
                .capabilities()
                .custom_capabilities
                .contains_key(capability),
        }
    }

    /// Validate that the agent can handle a task
    fn can_handle_task(&self, task: &Task) -> Result<()> {
        // Check if task requires capabilities the agent doesn't have
        if let Some(required_caps) = task.metadata.get("required_capabilities") {
            if let Some(caps_array) = required_caps.as_array() {
                for cap in caps_array {
                    if let Some(cap_str) = cap.as_str() {
                        if !self.supports_capability(cap_str) {
                            return Err(AgentError::execution(format!(
                                "Agent '{}' does not support required capability: {}",
                                self.name(),
                                cap_str
                            )));
                        }
                    }
                }
            }
        }

        // Check execution time limits
        if let Some(max_time) = self.config().max_execution_time_ms {
            if let Some(task_max_time) = task.max_execution_time_ms {
                if task_max_time > max_time {
                    return Err(AgentError::execution(format!(
                        "Task execution time limit ({task_max_time} ms) exceeds agent limit ({max_time} ms)"
                    )));
                }
            }
        }

        Ok(())
    }

    /// Get agent health status
    async fn health_check(&self) -> Result<AgentHealthStatus> {
        Ok(AgentHealthStatus {
            agent_id: self.id(),
            status: HealthStatus::Healthy,
            message: "Agent is operational".to_string(),
            last_check: Utc::now(),
            metrics: HashMap::new(),
        })
    }
}

/// Agent health status
#[derive(Debug, Clone)]
pub struct AgentHealthStatus {
    /// Agent ID
    pub agent_id: AgentId,
    /// Health status
    pub status: HealthStatus,
    /// Status message
    pub message: String,
    /// Last health check timestamp
    pub last_check: chrono::DateTime<chrono::Utc>,
    /// Additional metrics
    pub metrics: HashMap<String, serde_json::Value>,
}

/// Health status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// Agent is healthy and operational
    Healthy,
    /// Agent is degraded but functional
    Degraded,
    /// Agent is unhealthy and may not function properly
    Unhealthy,
    /// Agent status is unknown
    Unknown,
}

/// Basic agent implementation
#[derive(Debug)]
pub struct BasicAgent {
    /// Agent ID
    id: AgentId,
    /// Agent configuration
    config: AgentConfig,
    /// Tool registry
    tool_registry: Arc<ToolRegistry>,
    /// Agent statistics
    stats: Arc<tokio::sync::Mutex<AgentStats>>,
}

/// Agent execution statistics
#[derive(Debug, Default, Clone)]
pub struct AgentStats {
    /// Total tasks executed
    pub tasks_executed: usize,
    /// Total successful tasks
    pub tasks_successful: usize,
    /// Total failed tasks
    pub tasks_failed: usize,
    /// Total execution time in milliseconds
    pub total_execution_time_ms: u64,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Total tool calls made
    pub total_tool_calls: usize,
    /// Last execution timestamp
    pub last_execution: Option<chrono::DateTime<chrono::Utc>>,
}

impl BasicAgent {
    /// Create a new basic agent
    #[must_use]
    pub fn new(config: AgentConfig, tool_registry: Arc<ToolRegistry>) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            tool_registry,
            stats: Arc::new(tokio::sync::Mutex::new(AgentStats::default())),
        }
    }

    /// Get agent statistics
    pub async fn stats(&self) -> AgentStats {
        self.stats.lock().await.clone()
    }

    /// Execute a tool call
    async fn execute_tool_call(&self, tool_call: &ToolCall, context: &TaskContext) -> ToolOutput {
        debug!(
            "Executing tool call: {} ({})",
            tool_call.tool_name, tool_call.id
        );

        let tool_context = ToolContext::with_call_id(tool_call.id).with_data(
            "task_context",
            serde_json::to_value(context).unwrap_or_default(),
        );

        match self
            .tool_registry
            .execute(
                &tool_call.tool_name,
                tool_call.arguments.clone(),
                &tool_context,
            )
            .await
        {
            Ok(result) => {
                if result.success {
                    debug!("Tool call {} completed successfully", tool_call.id);
                    ToolOutput::success(tool_call.id, &tool_call.tool_name, result.content)
                } else {
                    warn!("Tool call {} failed: {:?}", tool_call.id, result.error);
                    ToolOutput::error(
                        tool_call.id,
                        &tool_call.tool_name,
                        result
                            .error
                            .unwrap_or_else(|| "Unknown tool error".to_string()),
                    )
                }
            }
            Err(e) => {
                error!("Tool call {} execution error: {}", tool_call.id, e);
                ToolOutput::error(tool_call.id, &tool_call.tool_name, e.to_string())
            }
        }
    }

    /// Update agent statistics
    async fn update_stats(&self, execution_time_ms: u64, success: bool, tool_calls: usize) {
        let mut stats = self.stats.lock().await;
        stats.tasks_executed += 1;
        if success {
            stats.tasks_successful += 1;
        } else {
            stats.tasks_failed += 1;
        }
        stats.total_execution_time_ms += execution_time_ms;
        stats.avg_execution_time_ms =
            stats.total_execution_time_ms as f64 / stats.tasks_executed as f64;
        stats.total_tool_calls += tool_calls;
        stats.last_execution = Some(Utc::now());
    }
}

#[async_trait]
impl Agent for BasicAgent {
    fn id(&self) -> AgentId {
        self.id
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn description(&self) -> Option<&str> {
        self.config.description.as_deref()
    }

    fn capabilities(&self) -> &AgentCapabilities {
        &self.config.capabilities
    }

    fn config(&self) -> &AgentConfig {
        &self.config
    }

    async fn execute(&self, task: &Task) -> Result<AgentResponse> {
        let start_time = Instant::now();
        info!("Agent '{}' executing task: {}", self.name(), task.id);

        // Validate task
        self.can_handle_task(task)?;

        // For this basic implementation, we'll simulate task execution
        // In a real implementation, this would integrate with an LLM
        let response_content = format!(
            "Task '{}' executed by agent '{}'. Input: {}",
            task.name,
            self.name(),
            task.input
        );

        // Simulate tool calls if the task requires them
        let mut tool_calls = Vec::new();
        let mut tool_outputs = Vec::new();

        // Check if task mentions any available tools
        let available_tools = self.tools();
        for tool_name in &available_tools {
            if task
                .input
                .to_lowercase()
                .contains(&tool_name.to_lowercase())
            {
                let tool_call = ToolCall::new(tool_name, serde_json::json!({"query": task.input}));
                let tool_output = self.execute_tool_call(&tool_call, &task.context).await;

                tool_calls.push(tool_call);
                tool_outputs.push(tool_output);
            }
        }

        let execution_time = start_time.elapsed();
        let execution_time_ms = execution_time.as_millis() as u64;

        // Update statistics
        self.update_stats(execution_time_ms, true, tool_calls.len())
            .await;

        let stats = ExecutionStats {
            execution_time_ms,
            tool_calls_count: tool_calls.len(),
            successful_tool_calls: tool_outputs.iter().filter(|o| o.success).count(),
            failed_tool_calls: tool_outputs.iter().filter(|o| !o.success).count(),
            tokens_used: None,
            custom_metrics: HashMap::new(),
        };

        info!(
            "Agent '{}' completed task {} in {}ms",
            self.name(),
            task.id,
            execution_time_ms
        );

        Ok(AgentResponse {
            content: response_content,
            metadata: HashMap::new(),
            tool_calls,
            tool_outputs,
            stats,
            timestamp: Utc::now(),
        })
    }

    async fn process_message(&self, message: &AgentMessage) -> Result<AgentResponse> {
        // Convert message to a task and execute
        let task = Task::builder()
            .name("Message Processing")
            .input(&message.content)
            .build()?;

        self.execute(&task).await
    }

    fn tools(&self) -> Vec<String> {
        self.tool_registry.tool_names()
    }

    async fn health_check(&self) -> Result<AgentHealthStatus> {
        let stats = self.stats().await;
        let mut metrics = HashMap::new();

        metrics.insert(
            "tasks_executed".to_string(),
            serde_json::json!(stats.tasks_executed),
        );
        metrics.insert(
            "success_rate".to_string(),
            serde_json::json!(if stats.tasks_executed > 0 {
                stats.tasks_successful as f64 / stats.tasks_executed as f64
            } else {
                0.0
            }),
        );
        metrics.insert(
            "avg_execution_time_ms".to_string(),
            serde_json::json!(stats.avg_execution_time_ms),
        );

        Ok(AgentHealthStatus {
            agent_id: self.id,
            status: HealthStatus::Healthy,
            message: format!(
                "Agent '{}' is operational. Executed {} tasks.",
                self.name(),
                stats.tasks_executed
            ),
            last_check: Utc::now(),
            metrics,
        })
    }
}
