//! Simple Workflow Engine
//!
//! This module provides a basic workflow engine for orchestrating agent interactions
//! and managing complex multi-step processes.

use crate::{
    agent::base::BaseAgent,
    error::{AgentError, Result},
    types::{AgentId, AgentMessage, AgentResponse, WorkflowContext, WorkflowStep},
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

/// Workflow execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowStatus {
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

/// Simple workflow definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// Workflow ID
    pub id: Uuid,
    /// Workflow name
    pub name: String,
    /// Workflow description
    pub description: Option<String>,
    /// Workflow steps
    pub steps: Vec<WorkflowStep>,
    /// Workflow metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Workflow {
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
    pub fn add_step(mut self, step: WorkflowStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Set workflow description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Workflow execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    /// Workflow ID
    pub workflow_id: Uuid,
    /// Execution status
    pub status: WorkflowStatus,
    /// Step results
    pub step_results: HashMap<String, AgentResponse>,
    /// Execution context
    pub context: WorkflowContext,
    /// Total execution time in milliseconds
    pub execution_time_ms: u64,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Simple workflow executor
pub struct WorkflowExecutor {
    /// Available agents
    agents: HashMap<AgentId, Arc<dyn BaseAgent>>,
}

impl WorkflowExecutor {
    /// Create a new workflow executor
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    /// Add an agent to the executor
    pub fn add_agent(&mut self, agent: Arc<dyn BaseAgent>) {
        self.agents.insert(agent.id(), agent);
    }

    /// Execute a workflow
    pub async fn execute(
        &self,
        workflow: Workflow,
        initial_message: AgentMessage,
    ) -> Result<WorkflowResult> {
        let start_time = std::time::Instant::now();
        let mut context = WorkflowContext::default();
        let mut step_results = HashMap::new();

        // Set initial context variables
        context.variables.insert(
            "initial_message".to_string(),
            serde_json::to_value(&initial_message)?,
        );
        context.variables.insert(
            "workflow_id".to_string(),
            serde_json::to_value(workflow.id)?,
        );

        // Execute steps in order
        for step in &workflow.steps {
            // Check dependencies
            if !self.check_dependencies(&step.dependencies, &step_results) {
                let error = format!("Dependencies not met for step: {}", step.id);
                return Ok(WorkflowResult {
                    workflow_id: workflow.id,
                    status: WorkflowStatus::Failed(error.clone()),
                    step_results,
                    context,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error: Some(error),
                    timestamp: chrono::Utc::now(),
                });
            }

            // Get the agent for this step
            let agent = match self.agents.get(&step.agent_id) {
                Some(agent) => agent,
                None => {
                    let error = format!("Agent not found for step: {}", step.id);
                    return Ok(WorkflowResult {
                        workflow_id: workflow.id,
                        status: WorkflowStatus::Failed(error.clone()),
                        step_results,
                        context,
                        execution_time_ms: start_time.elapsed().as_millis() as u64,
                        error: Some(error),
                        timestamp: chrono::Utc::now(),
                    });
                }
            };

            // Execute the step
            let mut agent_context = crate::agent::base::AgentContext::new();
            let response = agent
                .chat(initial_message.clone(), Some(&mut agent_context))
                .await?;

            // Store the result
            step_results.insert(step.id.clone(), response);

            // Update context with step result
            context.step_results.insert(
                step.id.clone(),
                serde_json::to_value(&step_results[&step.id])?,
            );
        }

        Ok(WorkflowResult {
            workflow_id: workflow.id,
            status: WorkflowStatus::Completed,
            step_results,
            context,
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            error: None,
            timestamp: chrono::Utc::now(),
        })
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

impl Default for WorkflowExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Workflow builder for creating workflows with a fluent API
pub struct WorkflowBuilder {
    workflow: Workflow,
}

impl WorkflowBuilder {
    /// Create a new workflow builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            workflow: Workflow::new(name),
        }
    }

    /// Set workflow description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.workflow.description = Some(description.into());
        self
    }

    /// Add a step to the workflow
    pub fn step(mut self, step: WorkflowStep) -> Self {
        self.workflow.steps.push(step);
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.workflow.metadata.insert(key.into(), value);
        self
    }

    /// Build the workflow
    pub fn build(self) -> Workflow {
        self.workflow
    }
}

/// Utility functions for workflow management
pub mod utils {
    use super::*;

    /// Create a simple single-step workflow
    pub fn create_simple_workflow(
        name: impl Into<String>,
        agent_id: AgentId,
        step_name: impl Into<String>,
    ) -> Workflow {
        let step = WorkflowStep {
            id: step_name.into(),
            name: "Single Step".to_string(),
            description: Some("A simple single-step workflow".to_string()),
            agent_id,
            dependencies: Vec::new(),
            config: HashMap::new(),
            retryable: false,
            max_retries: None,
        };

        Workflow::new(name).add_step(step)
    }

    /// Create a sequential workflow with multiple steps
    pub fn create_sequential_workflow(
        name: impl Into<String>,
        steps: Vec<(String, AgentId, String)>, // (step_id, agent_id, step_name)
    ) -> Workflow {
        let mut workflow = Workflow::new(name);
        let mut previous_step_id: Option<String> = None;

        for (step_id, agent_id, step_name) in steps {
            let dependencies = if let Some(prev_id) = &previous_step_id {
                vec![prev_id.clone()]
            } else {
                Vec::new()
            };

            let step = WorkflowStep {
                id: step_id.clone(),
                name: step_name,
                description: None,
                agent_id,
                dependencies,
                config: HashMap::new(),
                retryable: false,
                max_retries: None,
            };

            workflow = workflow.add_step(step);
            previous_step_id = Some(step_id);
        }

        workflow
    }
}
