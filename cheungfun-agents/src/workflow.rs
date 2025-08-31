//! Simple Workflow System
//!
//! This module provides a simplified workflow orchestration system for basic use cases.
//! For complex multi-agent coordination, use the `MultiAgentOrchestrator` instead.

use crate::{
    agent::base::BaseAgent,
    error::{AgentError, Result},
    types::{AgentId, AgentMessage, AgentResponse},
};
use cheungfun_core::traits::{BaseMemory, ChatMemoryConfig};
use cheungfun_query::memory::ChatMemoryBuffer;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

/// Simple workflow execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimpleWorkflowStatus {
    /// Workflow is ready to start
    Ready,
    /// Workflow is currently running
    Running,
    /// Workflow completed successfully
    Completed,
    /// Workflow failed with an error
    Failed(String),
    /// Workflow was cancelled
    Cancelled,
}

/// Simple workflow step definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleWorkflowStep {
    /// Step ID
    pub id: String,
    /// Step name
    pub name: String,
    /// Step description
    pub description: Option<String>,
    /// Agent ID to execute this step
    pub agent_id: AgentId,
    /// Dependencies (step IDs that must complete first)
    pub dependencies: Vec<String>,
    /// Step configuration
    pub config: HashMap<String, serde_json::Value>,
    /// Whether this step can be retried on failure
    pub retryable: bool,
    /// Maximum retry attempts
    pub max_retries: Option<usize>,
}

impl SimpleWorkflowStep {
    /// Create a new workflow step
    pub fn new(id: impl Into<String>, name: impl Into<String>, agent_id: AgentId) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            agent_id,
            dependencies: Vec::new(),
            config: HashMap::new(),
            retryable: false,
            max_retries: None,
        }
    }

    /// Set step description
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add dependencies
    #[must_use]
    pub fn with_dependencies(mut self, dependencies: Vec<String>) -> Self {
        self.dependencies = dependencies;
        self
    }

    /// Add a single dependency
    #[must_use]
    pub fn depends_on(mut self, step_id: impl Into<String>) -> Self {
        self.dependencies.push(step_id.into());
        self
    }

    /// Set as retryable
    #[must_use]
    pub fn retryable(mut self, retryable: bool, max_retries: Option<usize>) -> Self {
        self.retryable = retryable;
        self.max_retries = max_retries;
        self
    }

    /// Add configuration
    #[must_use]
    pub fn with_config(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.config.insert(key.into(), value);
        self
    }
}

/// Simple workflow definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleWorkflow {
    /// Workflow ID
    pub id: Uuid,
    /// Workflow name
    pub name: String,
    /// Workflow description
    pub description: Option<String>,
    /// Workflow steps
    pub steps: Vec<SimpleWorkflowStep>,
    /// Workflow metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl SimpleWorkflow {
    /// Create a new workflow
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            steps: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a step to the workflow
    #[must_use]
    pub fn add_step(mut self, step: SimpleWorkflowStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Set workflow description
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Validate the workflow structure
    pub fn validate(&self) -> Result<()> {
        if self.steps.is_empty() {
            return Err(AgentError::validation(
                "steps",
                "Workflow must have at least one step",
            ));
        }

        // Check for circular dependencies
        for step in &self.steps {
            if self.has_circular_dependency(&step.id, &step.dependencies)? {
                return Err(AgentError::validation(
                    "dependencies",
                    format!("Circular dependency detected for step: {}", step.id),
                ));
            }
        }

        Ok(())
    }

    /// Check for circular dependencies
    fn has_circular_dependency(&self, step_id: &str, deps: &[String]) -> Result<bool> {
        for dep in deps {
            if dep == step_id {
                return Ok(true);
            }

            if let Some(dep_step) = self.steps.iter().find(|s| s.id == *dep) {
                if self.has_circular_dependency(step_id, &dep_step.dependencies)? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

/// Simple workflow execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleWorkflowResult {
    /// Workflow ID
    pub workflow_id: Uuid,
    /// Execution status
    pub status: SimpleWorkflowStatus,
    /// Step results
    pub step_results: HashMap<String, AgentResponse>,
    /// Total execution time in milliseconds
    pub execution_time_ms: u64,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Detailed execution info
    pub execution_details: SimpleWorkflowExecutionDetails,
}

/// Detailed execution information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleWorkflowExecutionDetails {
    /// Number of steps completed
    pub steps_completed: usize,
    /// Number of steps failed
    pub steps_failed: usize,
    /// Number of steps skipped
    pub steps_skipped: usize,
    /// Step execution order
    pub execution_order: Vec<String>,
    /// Step execution times
    pub step_execution_times: HashMap<String, u64>,
}

/// Simple workflow executor (refactored to use the new workflow engine under the hood)
pub struct SimpleWorkflowExecutor {
    /// Available agents
    agents: HashMap<AgentId, Arc<dyn BaseAgent>>,
    /// Shared memory
    memory: Arc<dyn BaseMemory>,
    /// Execution configuration
    config: SimpleWorkflowExecutorConfig,
}

/// Configuration for simple workflow executor
#[derive(Debug, Clone)]
pub struct SimpleWorkflowExecutorConfig {
    /// Maximum execution time in milliseconds
    pub max_execution_time_ms: u64,
    /// Whether to stop on first error
    pub stop_on_error: bool,
    /// Whether to enable verbose logging
    pub verbose: bool,
    /// Maximum retries for retryable steps
    pub max_step_retries: usize,
}

impl Default for SimpleWorkflowExecutorConfig {
    fn default() -> Self {
        Self {
            max_execution_time_ms: 300_000, // 5 minutes
            stop_on_error: true,
            verbose: false,
            max_step_retries: 3,
        }
    }
}

impl SimpleWorkflowExecutor {
    /// Create a new workflow executor
    #[must_use]
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            memory: Arc::new(ChatMemoryBuffer::new(ChatMemoryConfig::default())),
            config: SimpleWorkflowExecutorConfig::default(),
        }
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: SimpleWorkflowExecutorConfig) -> Self {
        Self {
            agents: HashMap::new(),
            memory: Arc::new(ChatMemoryBuffer::new(ChatMemoryConfig::default())),
            config,
        }
    }

    /// Add an agent to the executor
    pub fn add_agent(&mut self, agent: Arc<dyn BaseAgent>) {
        self.agents.insert(agent.id(), agent);
    }

    /// Set memory provider
    pub fn with_memory(mut self, memory: Arc<dyn BaseMemory>) -> Self {
        self.memory = memory;
        self
    }

    /// Execute a workflow with enhanced orchestration
    pub async fn execute(
        &self,
        workflow: SimpleWorkflow,
        initial_message: AgentMessage,
    ) -> Result<SimpleWorkflowResult> {
        let start_time = std::time::Instant::now();

        // Validate workflow
        workflow.validate()?;

        let mut execution_details = SimpleWorkflowExecutionDetails {
            steps_completed: 0,
            steps_failed: 0,
            steps_skipped: 0,
            execution_order: Vec::new(),
            step_execution_times: HashMap::new(),
        };

        // For simple workflows, we execute steps sequentially according to dependencies
        let execution_order = self.compute_execution_order(&workflow)?;
        let mut step_results = HashMap::new();

        for step_id in &execution_order {
            let step = workflow
                .steps
                .iter()
                .find(|s| s.id == *step_id)
                .ok_or_else(|| {
                    AgentError::validation("step", format!("Step not found: {step_id}"))
                })?;

            // Check dependencies
            if !self.check_dependencies(&step.dependencies, &step_results) {
                execution_details.steps_skipped += 1;
                continue;
            }

            // Get the agent for this step
            let agent = if let Some(agent) = self.agents.get(&step.agent_id) {
                agent
            } else {
                let error = format!("Agent not found for step: {}", step.id);
                execution_details.steps_failed += 1;

                if self.config.stop_on_error {
                    return Ok(SimpleWorkflowResult {
                        workflow_id: workflow.id,
                        status: SimpleWorkflowStatus::Failed(error.clone()),
                        step_results,
                        execution_time_ms: start_time.elapsed().as_millis() as u64,
                        error: Some(error),
                        timestamp: chrono::Utc::now(),
                        execution_details,
                    });
                }
                continue;
            };

            // Execute the step with retries
            let step_start = std::time::Instant::now();
            let mut attempts = 0;
            let max_attempts = if step.retryable {
                step.max_retries.unwrap_or(self.config.max_step_retries)
            } else {
                1
            };

            let mut step_result = None;
            while attempts < max_attempts {
                attempts += 1;

                // For simple workflow, we just use the basic chat interface
                // More complex workflows would use the workflow engine
                let mut agent_context = crate::agent::base::AgentContext::new();

                match agent
                    .chat(initial_message.clone(), Some(&mut agent_context))
                    .await
                {
                    Ok(response) => {
                        step_result = Some(response);
                        break;
                    }
                    Err(e) => {
                        if attempts >= max_attempts {
                            execution_details.steps_failed += 1;

                            if self.config.stop_on_error {
                                let error = format!(
                                    "Step '{}' failed after {} attempts: {}",
                                    step.id, attempts, e
                                );
                                return Ok(SimpleWorkflowResult {
                                    workflow_id: workflow.id,
                                    status: SimpleWorkflowStatus::Failed(error.clone()),
                                    step_results,
                                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                                    error: Some(error),
                                    timestamp: chrono::Utc::now(),
                                    execution_details,
                                });
                            }
                        }

                        // Wait before retry (exponential backoff)
                        if attempts < max_attempts {
                            tokio::time::sleep(std::time::Duration::from_millis(
                                100 * (2_u64.pow((attempts - 1) as u32)),
                            ))
                            .await;
                        }
                    }
                }
            }

            let step_duration = step_start.elapsed().as_millis() as u64;
            execution_details
                .step_execution_times
                .insert(step.id.clone(), step_duration);
            execution_details.execution_order.push(step.id.clone());

            if let Some(result) = step_result {
                step_results.insert(step.id.clone(), result);
                execution_details.steps_completed += 1;
            }
        }

        Ok(SimpleWorkflowResult {
            workflow_id: workflow.id,
            status: SimpleWorkflowStatus::Completed,
            step_results,
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            error: None,
            timestamp: chrono::Utc::now(),
            execution_details,
        })
    }

    /// Compute execution order based on dependencies (topological sort)
    fn compute_execution_order(&self, workflow: &SimpleWorkflow) -> Result<Vec<String>> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();

        // Initialize
        for step in &workflow.steps {
            in_degree.insert(step.id.clone(), step.dependencies.len());
            adjacency.insert(step.id.clone(), Vec::new());
        }

        // Build adjacency list
        for step in &workflow.steps {
            for dep in &step.dependencies {
                if let Some(adj_list) = adjacency.get_mut(dep) {
                    adj_list.push(step.id.clone());
                }
            }
        }

        // Topological sort
        let mut queue = Vec::new();
        let mut result = Vec::new();

        // Find nodes with no dependencies
        for (step_id, degree) in &in_degree {
            if *degree == 0 {
                queue.push(step_id.clone());
            }
        }

        while let Some(current) = queue.pop() {
            result.push(current.clone());

            if let Some(neighbors) = adjacency.get(&current) {
                for neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push(neighbor.clone());
                        }
                    }
                }
            }
        }

        if result.len() != workflow.steps.len() {
            return Err(AgentError::validation(
                "dependencies",
                "Circular dependency detected in workflow",
            ));
        }

        Ok(result)
    }

    /// Check if step dependencies are satisfied
    fn check_dependencies(
        &self,
        dependencies: &[String],
        completed_steps: &HashMap<String, AgentResponse>,
    ) -> bool {
        dependencies
            .iter()
            .all(|dep| completed_steps.contains_key(dep))
    }
}

impl Default for SimpleWorkflowExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced workflow builder for creating workflows with a fluent API
pub struct SimpleWorkflowBuilder {
    workflow: SimpleWorkflow,
}

impl SimpleWorkflowBuilder {
    /// Create a new workflow builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            workflow: SimpleWorkflow::new(name),
        }
    }

    /// Set workflow description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.workflow.description = Some(description.into());
        self
    }

    /// Add a step to the workflow
    #[must_use]
    pub fn step(mut self, step: SimpleWorkflowStep) -> Self {
        self.workflow.steps.push(step);
        self
    }

    /// Add a simple step (convenience method)
    pub fn simple_step(
        mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        agent_id: AgentId,
    ) -> Self {
        let step = SimpleWorkflowStep::new(id, name, agent_id);
        self.workflow.steps.push(step);
        self
    }

    /// Add a sequential step (depends on the previous step)
    pub fn sequential_step(
        mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        agent_id: AgentId,
    ) -> Self {
        let step_id = id.into();
        let dependencies = if let Some(last_step) = self.workflow.steps.last() {
            vec![last_step.id.clone()]
        } else {
            Vec::new()
        };

        let step = SimpleWorkflowStep::new(step_id, name, agent_id).with_dependencies(dependencies);
        self.workflow.steps.push(step);
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.workflow.metadata.insert(key.into(), value);
        self
    }

    /// Build the workflow
    pub fn build(self) -> Result<SimpleWorkflow> {
        self.workflow.validate()?;
        Ok(self.workflow)
    }
}

/// Utility functions for workflow management
pub mod utils {
    use super::{AgentId, Result, SimpleWorkflow, SimpleWorkflowBuilder};

    /// Create a simple single-step workflow
    pub fn create_single_step_workflow(
        name: impl Into<String>,
        agent_id: AgentId,
        step_name: impl Into<String>,
    ) -> Result<SimpleWorkflow> {
        SimpleWorkflowBuilder::new(name)
            .simple_step("step_1", step_name, agent_id)
            .build()
    }

    /// Create a sequential workflow with multiple steps
    pub fn create_sequential_workflow(
        name: impl Into<String>,
        steps: Vec<(String, AgentId, String)>, // (step_id, agent_id, step_name)
    ) -> Result<SimpleWorkflow> {
        let mut builder = SimpleWorkflowBuilder::new(name);

        for (step_id, agent_id, step_name) in steps {
            builder = builder.sequential_step(step_id, step_name, agent_id);
        }

        builder.build()
    }

    /// Create a parallel workflow where steps can run independently
    pub fn create_parallel_workflow(
        name: impl Into<String>,
        steps: Vec<(String, AgentId, String)>, // (step_id, agent_id, step_name)
    ) -> Result<SimpleWorkflow> {
        let mut builder = SimpleWorkflowBuilder::new(name);

        for (step_id, agent_id, step_name) in steps {
            builder = builder.simple_step(step_id, step_name, agent_id);
        }

        builder.build()
    }
}
