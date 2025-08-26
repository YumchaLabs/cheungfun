//! Base workflow agent implementation
//! 
//! This module provides the base traits and implementations for workflow-based agents,
//! inspired by LlamaIndex's agent architecture.

use crate::{
    error::{AgentError, Result},
    tool::Tool,
    workflow::{
        context::WorkflowContext,
        events::{AgentOutput, ToolCallResult, WorkflowEvent},
    },
};
use async_trait::async_trait;
use cheungfun_core::{
    traits::BaseMemory,
    ChatMessage,
};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, pin::Pin, sync::Arc};
use uuid::Uuid;

/// Core trait for workflow-based agents
#[async_trait]
pub trait WorkflowAgent: Send + Sync + std::fmt::Debug {
    /// Get the agent's unique identifier
    fn id(&self) -> Uuid;
    
    /// Get the agent's name
    fn name(&self) -> &str;
    
    /// Get the agent's description
    fn description(&self) -> Option<&str> {
        None
    }
    
    /// Check if the agent supports streaming
    fn supports_streaming(&self) -> bool {
        false
    }
    
    /// Take a single step in the workflow
    async fn take_step(
        &self,
        ctx: &mut WorkflowContext,
        llm_input: Vec<ChatMessage>,
        tools: &[Arc<dyn Tool>],
        memory: &mut dyn BaseMemory,
    ) -> Result<AgentOutput>;
    
    /// Handle tool call results
    async fn handle_tool_call_results(
        &self,
        ctx: &mut WorkflowContext,
        results: Vec<ToolCallResult>,
        memory: &mut dyn BaseMemory,
    ) -> Result<()>;
    
    /// Finalize the agent's work (cleanup, memory updates, etc.)
    async fn finalize(
        &self,
        ctx: &mut WorkflowContext,
        output: AgentOutput,
        memory: &mut dyn BaseMemory,
    ) -> Result<AgentOutput>;
    
    /// Get available prompts for customization
    fn get_prompts(&self) -> HashMap<String, String> {
        HashMap::new()
    }
    
    /// Update prompts
    fn update_prompts(&mut self, _prompts: HashMap<String, String>) -> Result<()> {
        Ok(())
    }
}

/// Base implementation for workflow agents
#[derive(Debug, Clone)]
pub struct BaseWorkflowAgent {
    /// Agent identifier
    pub id: Uuid,
    /// Agent name
    pub name: String,
    /// Agent description
    pub description: Option<String>,
    /// System prompt for the agent
    pub system_prompt: Option<String>,
    /// Whether streaming is enabled
    pub streaming: bool,
    /// Available tools
    pub tools: Vec<Arc<dyn Tool>>,
    /// Agent configuration
    pub config: HashMap<String, serde_json::Value>,
}

impl BaseWorkflowAgent {
    /// Create a new base workflow agent
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            system_prompt: None,
            streaming: false,
            tools: Vec::new(),
            config: HashMap::new(),
        }
    }
    
    /// Set the agent description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    
    /// Set the system prompt
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }
    
    /// Enable or disable streaming
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }
    
    /// Add tools to the agent
    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }
    
    /// Add a single tool
    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }
    
    /// Set configuration
    pub fn with_config(mut self, config: HashMap<String, serde_json::Value>) -> Self {
        self.config = config;
        self
    }
    
    /// Add a configuration value
    pub fn with_config_value(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.config.insert(key.into(), value);
        self
    }
}

/// Agent execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStats {
    /// Total number of steps taken
    pub steps_taken: usize,
    /// Total number of tool calls made
    pub tool_calls_made: usize,
    /// Total execution time in milliseconds
    pub total_execution_time_ms: u64,
    /// Average step execution time in milliseconds
    pub avg_step_time_ms: f64,
    /// Number of successful executions
    pub successful_executions: usize,
    /// Number of failed executions
    pub failed_executions: usize,
    /// Last execution timestamp
    pub last_execution: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for AgentStats {
    fn default() -> Self {
        Self {
            steps_taken: 0,
            tool_calls_made: 0,
            total_execution_time_ms: 0,
            avg_step_time_ms: 0.0,
            successful_executions: 0,
            failed_executions: 0,
            last_execution: None,
        }
    }
}

impl AgentStats {
    /// Update statistics after a successful execution
    pub fn record_success(&mut self, execution_time_ms: u64, steps: usize, tool_calls: usize) {
        self.successful_executions += 1;
        self.steps_taken += steps;
        self.tool_calls_made += tool_calls;
        self.total_execution_time_ms += execution_time_ms;
        self.last_execution = Some(chrono::Utc::now());
        self.update_avg_step_time();
    }
    
    /// Update statistics after a failed execution
    pub fn record_failure(&mut self, execution_time_ms: u64) {
        self.failed_executions += 1;
        self.total_execution_time_ms += execution_time_ms;
        self.last_execution = Some(chrono::Utc::now());
    }
    
    /// Update the average step time
    fn update_avg_step_time(&mut self) {
        if self.steps_taken > 0 {
            self.avg_step_time_ms = self.total_execution_time_ms as f64 / self.steps_taken as f64;
        }
    }
    
    /// Get the success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        let total = self.successful_executions + self.failed_executions;
        if total > 0 {
            (self.successful_executions as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Agent builder for creating workflow agents
#[derive(Debug)]
pub struct AgentBuilder {
    name: Option<String>,
    description: Option<String>,
    system_prompt: Option<String>,
    streaming: bool,
    tools: Vec<Arc<dyn Tool>>,
    config: HashMap<String, serde_json::Value>,
}

impl AgentBuilder {
    /// Create a new agent builder
    pub fn new() -> Self {
        Self {
            name: None,
            description: None,
            system_prompt: None,
            streaming: false,
            tools: Vec::new(),
            config: HashMap::new(),
        }
    }
    
    /// Set the agent name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    
    /// Set the agent description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    
    /// Set the system prompt
    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }
    
    /// Enable streaming
    pub fn streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }
    
    /// Add tools
    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }
    
    /// Add a single tool
    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }
    
    /// Set configuration
    pub fn config(mut self, config: HashMap<String, serde_json::Value>) -> Self {
        self.config = config;
        self
    }
    
    /// Add a configuration value
    pub fn config_value(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.config.insert(key.into(), value);
        self
    }
    
    /// Build a base workflow agent
    pub fn build_base(self) -> Result<BaseWorkflowAgent> {
        let name = self.name.ok_or_else(|| AgentError::validation("name", "Agent name is required"))?;
        
        Ok(BaseWorkflowAgent {
            id: Uuid::new_v4(),
            name,
            description: self.description,
            system_prompt: self.system_prompt,
            streaming: self.streaming,
            tools: self.tools,
            config: self.config,
        })
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for agents that can be executed in a workflow
#[async_trait]
pub trait ExecutableAgent: WorkflowAgent {
    /// Execute the agent with the given input
    async fn execute(
        &self,
        input: &str,
        ctx: &mut WorkflowContext,
        memory: &mut dyn BaseMemory,
    ) -> Result<String>;
    
    /// Execute the agent with streaming output
    async fn execute_stream(
        &self,
        input: &str,
        ctx: &mut WorkflowContext,
        memory: &mut dyn BaseMemory,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Box<dyn WorkflowEvent>>> + Send>>>;
}

/// Utility functions for agent implementations
pub mod utils {
    use super::*;
    
    /// Create a simple chat message
    pub fn create_message(role: &str, content: impl Into<String>) -> ChatMessage {
        use cheungfun_core::MessageRole;

        let message_role = match role {
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            "system" => MessageRole::System,
            "tool" => MessageRole::Tool,
            _ => MessageRole::User, // Default fallback
        };

        ChatMessage {
            role: message_role,
            content: content.into(),
            metadata: None,
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Extract tool calls from agent output
    pub fn extract_tool_calls(output: &AgentOutput) -> Vec<&crate::workflow::events::ToolSelection> {
        output.tool_calls.iter().collect()
    }
    
    /// Check if output contains tool calls
    pub fn has_tool_calls(output: &AgentOutput) -> bool {
        !output.tool_calls.is_empty()
    }
}
