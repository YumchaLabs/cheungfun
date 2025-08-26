//! Multi-Agent Orchestration System
//!
//! This module implements a sophisticated multi-agent orchestration system
//! that enables specialized agents to collaborate on complex tasks.

use crate::{
    agent::base::{AgentContext, BaseAgent},
    error::{AgentError, Result},
    types::{AgentMessage, AgentResponse},
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Simple context for multi-agent coordination
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultiAgentContext {
    /// Shared state between agents
    pub shared_state: HashMap<String, serde_json::Value>,
    /// Message history
    pub message_history: Vec<AgentMessage>,
    /// Current active agent
    pub current_agent: Option<Uuid>,
}

impl MultiAgentContext {
    /// Set a variable in the shared state
    pub fn set_variable(&mut self, key: String, value: serde_json::Value) {
        self.shared_state.insert(key, value);
    }

    /// Get a variable from the shared state
    #[must_use]
    pub fn get_variable(&self, key: &str) -> Option<&serde_json::Value> {
        self.shared_state.get(key)
    }

    /// Add a message to the history
    pub fn add_message(&mut self, message: AgentMessage) {
        self.message_history.push(message);
    }
}

/// Agent handoff strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HandoffStrategy {
    /// Agents are called in a fixed sequence
    Sequential,
    /// Next agent is dynamically determined based on context
    Dynamic,
    /// Multiple agents work in parallel
    Parallel,
    /// A coordinator agent decides which agents to invoke
    Coordinator {
        /// ID of the coordinator agent
        coordinator_id: Uuid,
    },
}

/// Agent role definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRole {
    /// Agent identifier
    pub agent_id: Uuid,
    /// Role name
    pub role: String,
    /// Role description
    pub description: String,
    /// Capabilities of this role
    pub capabilities: Vec<String>,
    /// Preferred tasks for this role
    pub preferred_tasks: Vec<String>,
}

/// Multi-agent orchestration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentConfig {
    /// Orchestration name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Handoff strategy
    pub handoff_strategy: HandoffStrategy,
    /// Maximum handoffs allowed
    pub max_handoffs: usize,
    /// Enable parallel execution
    pub enable_parallel: bool,
    /// Timeout for each agent in milliseconds
    pub agent_timeout_ms: u64,
}

impl Default for MultiAgentConfig {
    fn default() -> Self {
        Self {
            name: "MultiAgentOrchestration".to_string(),
            description: None,
            handoff_strategy: HandoffStrategy::Dynamic,
            max_handoffs: 10,
            enable_parallel: false,
            agent_timeout_ms: 30_000,
        }
    }
}

/// Agent handoff information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHandoff {
    /// From agent ID
    pub from_agent: Uuid,
    /// To agent ID  
    pub to_agent: Uuid,
    /// Reason for handoff
    pub reason: String,
    /// Context to pass
    pub context: HashMap<String, serde_json::Value>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Multi-agent orchestration system
pub struct MultiAgentOrchestrator {
    /// Configuration
    config: MultiAgentConfig,
    /// Available agents with their roles
    agents: Arc<RwLock<HashMap<Uuid, (Arc<dyn BaseAgent>, AgentRole)>>>,
    /// Handoff history
    handoff_history: Arc<RwLock<Vec<AgentHandoff>>>,
    /// Shared context
    shared_context: Arc<RwLock<MultiAgentContext>>,
}

impl MultiAgentOrchestrator {
    /// Create new orchestrator
    #[must_use]
    pub fn new(config: MultiAgentConfig) -> Self {
        Self {
            config,
            agents: Arc::new(RwLock::new(HashMap::new())),
            handoff_history: Arc::new(RwLock::new(Vec::new())),
            shared_context: Arc::new(RwLock::new(MultiAgentContext::default())),
        }
    }

    /// Register an agent with its role
    pub async fn register_agent(&self, agent: Arc<dyn BaseAgent>, role: AgentRole) -> Result<()> {
        let mut agents = self.agents.write().await;
        if agents.contains_key(&role.agent_id) {
            return Err(AgentError::validation(
                "agent_id",
                format!("Agent {} already registered", role.agent_id),
            ));
        }
        agents.insert(role.agent_id, (agent, role));
        Ok(())
    }

    /// Execute orchestrated task
    pub async fn execute(&self, initial_message: AgentMessage) -> Result<MultiAgentResult> {
        let start_time = std::time::Instant::now();
        let mut handoff_count = 0;
        let mut agent_responses = Vec::new();

        // Initialize shared context
        {
            let mut ctx = self.shared_context.write().await;
            ctx.set_variable(
                "initial_message".to_string(),
                serde_json::to_value(&initial_message)?,
            );
            ctx.set_variable(
                "orchestration_start".to_string(),
                serde_json::json!(chrono::Utc::now()),
            );
        }

        // Execute based on strategy
        match &self.config.handoff_strategy {
            HandoffStrategy::Sequential => {
                self.execute_sequential(initial_message, &mut agent_responses, &mut handoff_count)
                    .await?;
            }
            HandoffStrategy::Dynamic => {
                self.execute_dynamic(initial_message, &mut agent_responses, &mut handoff_count)
                    .await?;
            }
            HandoffStrategy::Parallel => {
                self.execute_parallel(initial_message, &mut agent_responses)
                    .await?;
            }
            HandoffStrategy::Coordinator { coordinator_id } => {
                self.execute_with_coordinator(
                    *coordinator_id,
                    initial_message,
                    &mut agent_responses,
                    &mut handoff_count,
                )
                .await?;
            }
        }

        // Get final context
        let final_context = self.shared_context.read().await.clone();
        let handoff_history = self.handoff_history.read().await.clone();

        Ok(MultiAgentResult {
            responses: agent_responses,
            handoff_count,
            handoff_history,
            final_context,
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Execute agents sequentially
    async fn execute_sequential(
        &self,
        mut current_message: AgentMessage,
        responses: &mut Vec<AgentResponse>,
        handoff_count: &mut usize,
    ) -> Result<()> {
        let agents = self.agents.read().await;
        let agent_list: Vec<_> = agents.iter().collect();

        for (agent_id, (agent, role)) in agent_list {
            tracing::info!("Executing agent {} with role {}", agent_id, role.role);

            let mut agent_context = AgentContext::new();
            let response = agent
                .chat(current_message.clone(), Some(&mut agent_context))
                .await?;

            responses.push(response.clone());

            // Update message for next agent
            current_message = AgentMessage::assistant(response.content.clone());

            // Update shared context
            {
                let mut ctx = self.shared_context.write().await;
                ctx.set_variable(
                    format!("agent_{agent_id}_response"),
                    serde_json::to_value(&response)?,
                );
            }

            *handoff_count += 1;
            if *handoff_count >= self.config.max_handoffs {
                break;
            }
        }

        Ok(())
    }

    /// Execute agents dynamically based on context
    async fn execute_dynamic(
        &self,
        initial_message: AgentMessage,
        responses: &mut Vec<AgentResponse>,
        handoff_count: &mut usize,
    ) -> Result<()> {
        let mut current_message = initial_message;
        let mut current_agent_id = self.select_initial_agent().await?;

        loop {
            // Execute current agent
            let (response, should_handoff, next_agent) = self
                .execute_single_agent(current_agent_id, current_message.clone())
                .await?;

            responses.push(response.clone());

            if !should_handoff || *handoff_count >= self.config.max_handoffs {
                break;
            }

            // Record handoff
            if let Some(next_id) = next_agent {
                self.record_handoff(current_agent_id, next_id, "Dynamic handoff")
                    .await?;
                current_agent_id = next_id;
                current_message = AgentMessage::assistant(response.content);
                *handoff_count += 1;
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Execute agents in parallel
    async fn execute_parallel(
        &self,
        initial_message: AgentMessage,
        responses: &mut Vec<AgentResponse>,
    ) -> Result<()> {
        let agents = self.agents.read().await;
        let mut tasks = Vec::new();

        for (_agent_id, (agent, role)) in agents.iter() {
            let agent = agent.clone();
            let message = initial_message.clone();
            let role = role.clone();

            let task = tokio::spawn(async move {
                tracing::info!("Parallel execution of agent with role {}", role.role);
                let mut context = AgentContext::new();
                agent.chat(message, Some(&mut context)).await
            });

            tasks.push(task);
        }

        // Wait for all agents to complete
        for task in tasks {
            match task.await {
                Ok(Ok(response)) => responses.push(response),
                Ok(Err(e)) => tracing::error!("Agent execution failed: {}", e),
                Err(e) => tracing::error!("Task join error: {}", e),
            }
        }

        Ok(())
    }

    /// Execute with a coordinator agent
    async fn execute_with_coordinator(
        &self,
        coordinator_id: Uuid,
        initial_message: AgentMessage,
        responses: &mut Vec<AgentResponse>,
        handoff_count: &mut usize,
    ) -> Result<()> {
        // First, get coordinator's plan
        let coordinator_response = self
            .execute_single_agent(coordinator_id, initial_message.clone())
            .await?
            .0;
        responses.push(coordinator_response.clone());

        // Parse coordinator's decision (simplified - in production, use structured output)
        // For now, we'll execute all agents sequentially
        self.execute_sequential(initial_message, responses, handoff_count)
            .await
    }

    /// Execute a single agent
    async fn execute_single_agent(
        &self,
        agent_id: Uuid,
        message: AgentMessage,
    ) -> Result<(AgentResponse, bool, Option<Uuid>)> {
        let agents = self.agents.read().await;
        let (agent, role) = agents.get(&agent_id).ok_or_else(|| {
            AgentError::validation("agent_id", format!("Agent {agent_id} not found"))
        })?;

        let mut context = AgentContext::new();
        let response = agent.chat(message, Some(&mut context)).await?;

        // Check if handoff is needed (simplified logic)
        let should_handoff = response.content.contains("HANDOFF:")
            || response.metadata.get("needs_handoff").is_some();

        let next_agent = if should_handoff {
            self.determine_next_agent(&response, role).await.ok()
        } else {
            None
        };

        Ok((response, should_handoff, next_agent))
    }

    /// Select initial agent based on task
    async fn select_initial_agent(&self) -> Result<Uuid> {
        let agents = self.agents.read().await;
        agents
            .keys()
            .next()
            .copied()
            .ok_or_else(|| AgentError::validation("agents", "No agents registered"))
    }

    /// Determine next agent based on context
    async fn determine_next_agent(
        &self,
        _response: &AgentResponse,
        current_role: &AgentRole,
    ) -> Result<Uuid> {
        // Simplified logic - in production, use semantic matching or LLM
        let agents = self.agents.read().await;

        // Find an agent with different role
        for (id, (_, role)) in agents.iter() {
            if role.role != current_role.role {
                return Ok(*id);
            }
        }

        Err(AgentError::validation(
            "next_agent",
            "No suitable next agent found",
        ))
    }

    /// Record handoff event
    async fn record_handoff(&self, from: Uuid, to: Uuid, reason: &str) -> Result<()> {
        let handoff = AgentHandoff {
            from_agent: from,
            to_agent: to,
            reason: reason.to_string(),
            context: HashMap::new(),
            timestamp: chrono::Utc::now(),
        };

        let mut history = self.handoff_history.write().await;
        history.push(handoff);

        Ok(())
    }
}

/// Result of multi-agent orchestration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentResult {
    /// All agent responses
    pub responses: Vec<AgentResponse>,
    /// Number of handoffs
    pub handoff_count: usize,
    /// Handoff history
    pub handoff_history: Vec<AgentHandoff>,
    /// Final shared context
    pub final_context: MultiAgentContext,
    /// Total execution time
    pub execution_time_ms: u64,
}

/// Builder for multi-agent orchestrator
pub struct MultiAgentOrchestratorBuilder {
    config: MultiAgentConfig,
    agents: Vec<(Arc<dyn BaseAgent>, AgentRole)>,
}

impl MultiAgentOrchestratorBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: MultiAgentConfig::default(),
            agents: Vec::new(),
        }
    }

    /// Set orchestration name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    /// Set handoff strategy
    #[must_use]
    pub fn handoff_strategy(mut self, strategy: HandoffStrategy) -> Self {
        self.config.handoff_strategy = strategy;
        self
    }

    /// Set maximum number of handoffs
    #[must_use]
    pub fn max_handoffs(mut self, max: usize) -> Self {
        self.config.max_handoffs = max;
        self
    }

    /// Add an agent with its role
    pub fn add_agent(mut self, agent: Arc<dyn BaseAgent>, role: AgentRole) -> Self {
        self.agents.push((agent, role));
        self
    }

    /// Build the orchestrator with registered agents
    pub async fn build(self) -> Result<MultiAgentOrchestrator> {
        let orchestrator = MultiAgentOrchestrator::new(self.config);

        for (agent, role) in self.agents {
            orchestrator.register_agent(agent, role).await?;
        }

        Ok(orchestrator)
    }
}

/// Specialized agent roles (预定义的专业角色)
pub mod specialized_agents {
    use super::{AgentRole, Uuid};

    /// Create a research agent role
    #[must_use]
    pub fn research_agent_role(agent_id: Uuid) -> AgentRole {
        AgentRole {
            agent_id,
            role: "Researcher".to_string(),
            description: "Specializes in information gathering and research".to_string(),
            capabilities: vec![
                "web_search".to_string(),
                "document_analysis".to_string(),
                "fact_checking".to_string(),
            ],
            preferred_tasks: vec!["research".to_string(), "information_gathering".to_string()],
        }
    }

    /// Create an analyst agent role
    #[must_use]
    pub fn analyst_agent_role(agent_id: Uuid) -> AgentRole {
        AgentRole {
            agent_id,
            role: "Analyst".to_string(),
            description: "Specializes in data analysis and insights".to_string(),
            capabilities: vec![
                "data_analysis".to_string(),
                "pattern_recognition".to_string(),
                "statistical_analysis".to_string(),
            ],
            preferred_tasks: vec!["analysis".to_string(), "insights".to_string()],
        }
    }

    /// Create a writer agent role
    #[must_use]
    pub fn writer_agent_role(agent_id: Uuid) -> AgentRole {
        AgentRole {
            agent_id,
            role: "Writer".to_string(),
            description: "Specializes in content creation and documentation".to_string(),
            capabilities: vec![
                "content_creation".to_string(),
                "summarization".to_string(),
                "formatting".to_string(),
            ],
            preferred_tasks: vec!["writing".to_string(), "documentation".to_string()],
        }
    }
}

impl Default for MultiAgentOrchestratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
