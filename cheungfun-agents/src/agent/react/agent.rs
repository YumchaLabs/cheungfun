//! ReAct Agent Implementation
//!
//! This module provides the main ReAct agent implementation that combines
//! reasoning and acting in a structured way, following the ReAct paper's methodology.

use super::reasoning::{
    ActionStep, FinalAnswerStep, ObservationStep, ReasoningStep, ReasoningTrace, ThoughtStep,
};
use crate::{
    agent::base::{AgentContext, AgentStatus, BaseAgent},
    error::{AgentError, Result},
    llm::{LlmClientConfig, SiumaiLlmClient},
    tool::{Tool, ToolRegistry},
    types::{
        AgentCapabilities, AgentConfig, AgentId, AgentMessage, AgentResponse, ExecutionStats,
        ToolCall, ToolOutput,
    },
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Instant};
use uuid::Uuid;

/// ReAct agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActConfig {
    /// Base agent configuration
    pub base_config: AgentConfig,
    /// Maximum number of reasoning iterations
    pub max_iterations: usize,
    /// Maximum thinking time per iteration in milliseconds
    pub max_thinking_time_ms: u64,
    /// Whether to include reasoning trace in response
    pub include_trace: bool,
    /// Custom ReAct-specific settings
    pub custom_settings: HashMap<String, serde_json::Value>,
}

impl ReActConfig {
    /// Create a new ReAct configuration
    pub fn new(name: impl Into<String>) -> Self {
        let mut capabilities = AgentCapabilities::default();
        capabilities.supports_tools = true;
        capabilities.supports_conversation = true;

        Self {
            base_config: AgentConfig {
                name: name.into(),
                description: Some("ReAct reasoning agent".to_string()),
                instructions: Some("You are a helpful assistant that uses reasoning and actions to solve problems. Think step by step and use tools when needed.".to_string()),
                capabilities,
                max_execution_time_ms: Some(60_000), // 1 minute
                max_tool_calls: Some(20),
                verbose: false,
                custom_config: HashMap::new(),
            },
            max_iterations: 10,
            max_thinking_time_ms: 5_000, // 5 seconds per thought
            include_trace: true,
            custom_settings: HashMap::new(),
        }
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set maximum thinking time
    pub fn with_max_thinking_time(mut self, ms: u64) -> Self {
        self.max_thinking_time_ms = ms;
        self
    }

    /// Enable or disable trace inclusion
    pub fn with_trace(mut self, include_trace: bool) -> Self {
        self.include_trace = include_trace;
        self
    }

    /// Set custom instructions
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.base_config.instructions = Some(instructions.into());
        self
    }
}

/// ReAct agent execution statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReActStats {
    /// Total number of executions
    pub total_executions: u64,
    /// Number of successful executions
    pub successful_executions: u64,
    /// Number of failed executions
    pub failed_executions: u64,
    /// Total reasoning steps taken
    pub total_reasoning_steps: u64,
    /// Total tool calls made
    pub total_tool_calls: u64,
    /// Average steps per execution
    pub avg_steps_per_execution: f64,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Last execution timestamp
    pub last_execution: Option<chrono::DateTime<chrono::Utc>>,
}

impl ReActStats {
    /// Update statistics after an execution
    pub fn update_execution(
        &mut self,
        success: bool,
        steps: usize,
        tool_calls: usize,
        execution_time_ms: u64,
    ) {
        self.total_executions += 1;
        if success {
            self.successful_executions += 1;
        } else {
            self.failed_executions += 1;
        }

        self.total_reasoning_steps += steps as u64;
        self.total_tool_calls += tool_calls as u64;

        // Update averages
        self.avg_steps_per_execution =
            self.total_reasoning_steps as f64 / self.total_executions as f64;
        self.success_rate = self.successful_executions as f64 / self.total_executions as f64;

        // Update average execution time
        let total_time = (self.avg_execution_time_ms * (self.total_executions - 1) as f64)
            + execution_time_ms as f64;
        self.avg_execution_time_ms = total_time / self.total_executions as f64;

        self.last_execution = Some(chrono::Utc::now());
    }
}

/// ReAct Agent - implements reasoning and acting pattern
#[derive(Debug)]
pub struct ReActAgent {
    /// Agent ID
    id: AgentId,
    /// Agent configuration
    config: ReActConfig,
    /// Tool registry for available tools
    tools: Arc<ToolRegistry>,
    /// LLM client for reasoning
    llm_client: Option<SiumaiLlmClient>,
    /// Agent statistics
    stats: ReActStats,
    /// Current status
    status: AgentStatus,
}

impl ReActAgent {
    /// Create a new ReAct agent
    pub fn new(config: ReActConfig, tools: Arc<ToolRegistry>) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            tools,
            llm_client: None,
            stats: ReActStats::default(),
            status: AgentStatus::Ready,
        }
    }

    /// Create a new ReAct agent with LLM client
    pub fn with_llm_client(
        config: ReActConfig,
        tools: Arc<ToolRegistry>,
        llm_client: SiumaiLlmClient,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            tools,
            llm_client: Some(llm_client),
            stats: ReActStats::default(),
            status: AgentStatus::Ready,
        }
    }

    /// Get the ReAct configuration
    pub fn react_config(&self) -> &ReActConfig {
        &self.config
    }

    /// Get the ReAct statistics
    pub fn react_stats(&self) -> &ReActStats {
        &self.stats
    }

    /// Update statistics
    pub fn update_stats(
        &mut self,
        success: bool,
        steps: usize,
        tool_calls: usize,
        execution_time_ms: u64,
    ) {
        self.stats
            .update_execution(success, steps, tool_calls, execution_time_ms);
    }

    /// Set LLM client
    pub fn set_llm_client(&mut self, llm_client: SiumaiLlmClient) {
        self.llm_client = Some(llm_client);
    }

    /// Get LLM client reference
    pub fn llm_client(&self) -> Option<&SiumaiLlmClient> {
        self.llm_client.as_ref()
    }

    /// Perform ReAct reasoning with LLM
    pub async fn reason_with_llm(&self, messages: Vec<AgentMessage>) -> Result<String> {
        if let Some(llm_client) = &self.llm_client {
            llm_client.chat(messages).await
        } else {
            Err(AgentError::invalid_configuration(
                "No LLM client configured",
            ))
        }
    }

    /// Create a ReAct prompt for the given message
    fn create_react_prompt(&self, message: &AgentMessage) -> AgentMessage {
        let system_prompt = format!(
            r#"You are a ReAct (Reasoning and Acting) agent. Your task is to think step by step and provide a structured response.

Follow this format:
1. Thought: [Your reasoning about the task]
2. Action: [If you need to use a tool, specify it here, otherwise skip]
3. Observation: [Results from the action, if any]
4. Final Answer: [Your final response to the user]

Available tools: {}

User's request: {}"#,
            self.get_available_tools_description(),
            message.content
        );

        AgentMessage {
            id: uuid::Uuid::new_v4(),
            content: system_prompt,
            role: crate::types::MessageRole::System,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
            tool_calls: Vec::new(),
        }
    }

    /// Get description of available tools
    fn get_available_tools_description(&self) -> String {
        let tools = self.tools.get_all_tools();
        if tools.is_empty() {
            "No tools available".to_string()
        } else {
            tools
                .iter()
                .map(|tool| format!("- {}: {}", tool.name(), tool.description()))
                .collect::<Vec<_>>()
                .join("\n")
        }
    }
}

#[async_trait]
impl BaseAgent for ReActAgent {
    fn id(&self) -> AgentId {
        self.id
    }

    fn name(&self) -> &str {
        &self.config.base_config.name
    }

    fn description(&self) -> Option<&str> {
        self.config.base_config.description.as_deref()
    }

    fn capabilities(&self) -> &AgentCapabilities {
        &self.config.base_config.capabilities
    }

    fn config(&self) -> &AgentConfig {
        &self.config.base_config
    }

    async fn chat(
        &self,
        message: AgentMessage,
        context: Option<&mut AgentContext>,
    ) -> Result<AgentResponse> {
        let start_time = Instant::now();

        // For now, return a simple mock response demonstrating ReAct pattern
        // TODO: Implement full ReAct reasoning loop with LLM integration

        // Use LLM for reasoning if available
        let response_content = if let Some(_llm_client) = &self.llm_client {
            // Create a ReAct prompt
            let react_prompt = self.create_react_prompt(&message);
            let messages = vec![react_prompt];

            match self.reason_with_llm(messages).await {
                Ok(llm_response) => llm_response,
                Err(e) => {
                    format!(
                        "Error during LLM reasoning: {}. Falling back to default response.",
                        e
                    )
                }
            }
        } else {
            format!(
                "ReAct Agent '{}' received message: '{}'\n\nThought: I need to analyze this request and determine if I need to use any tools.\n\nFinal Answer: This is a demonstration response from the ReAct agent. In a full implementation, I would use reasoning and tool calls to provide a comprehensive answer.",
                self.name(),
                message.content
            )
        };

        let execution_time = start_time.elapsed().as_millis() as u64;
        let stats = ExecutionStats {
            execution_time_ms: execution_time,
            tool_calls_count: 0,
            successful_tool_calls: 0,
            failed_tool_calls: 0,
            tokens_used: None,
            custom_metrics: HashMap::new(),
        };

        let mut response = AgentResponse {
            content: response_content,
            metadata: HashMap::new(),
            tool_calls: Vec::new(),
            tool_outputs: Vec::new(),
            stats,
            timestamp: chrono::Utc::now(),
        };

        // Add ReAct-specific metadata
        response.metadata.insert(
            "agent_type".to_string(),
            serde_json::Value::String("react".to_string()),
        );
        response.metadata.insert(
            "max_iterations".to_string(),
            serde_json::Value::Number(serde_json::Number::from(self.config.max_iterations)),
        );

        // Add message to context if provided
        if let Some(ctx) = context {
            ctx.add_message(message);
        }

        Ok(response)
    }

    async fn chat_with_history(
        &self,
        messages: Vec<AgentMessage>,
        mut context: Option<&mut AgentContext>,
    ) -> Result<AgentResponse> {
        // For ReAct, we process the last message with full history context
        if let Some(last_message) = messages.last() {
            // Add all previous messages to context
            if let Some(ref mut ctx) = context {
                for msg in &messages[..messages.len() - 1] {
                    ctx.add_message(msg.clone());
                }
            }

            self.chat(last_message.clone(), context).await
        } else {
            Err(AgentError::invalid_input("No messages provided"))
        }
    }

    fn status(&self) -> AgentStatus {
        self.status.clone()
    }

    fn stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        stats.insert(
            "react_stats".to_string(),
            serde_json::to_value(&self.stats).unwrap_or_default(),
        );
        stats
    }
}
