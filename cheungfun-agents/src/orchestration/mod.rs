//! Agent orchestration and workflow management.

use crate::{
    agent::Agent,
    error::{AgentError, Result},
    task::{Task, TaskResult, TaskStatus},
    types::{AgentId, WorkflowContext, WorkflowId, WorkflowStep},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

pub mod workflow;

pub use workflow::{Workflow, WorkflowBuilder};

/// Agent orchestrator for managing multiple agents and workflows
#[derive(Debug)]
pub struct AgentOrchestrator {
    /// Registered agents by ID
    agents: HashMap<AgentId, Arc<dyn Agent>>,
    /// Active workflows by ID
    workflows: HashMap<WorkflowId, Workflow>,
    /// Orchestrator configuration
    config: OrchestratorConfig,
    /// Execution statistics
    stats: OrchestratorStats,
}

/// Orchestrator configuration
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Maximum number of concurrent workflows
    pub max_concurrent_workflows: usize,
    /// Maximum number of concurrent tasks per workflow
    pub max_concurrent_tasks_per_workflow: usize,
    /// Default workflow timeout in milliseconds
    pub default_workflow_timeout_ms: u64,
    /// Whether to enable detailed logging
    pub verbose_logging: bool,
    /// Whether to automatically retry failed tasks
    pub auto_retry_failed_tasks: bool,
    /// Maximum retry attempts for failed tasks
    pub max_retry_attempts: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_workflows: 10,
            max_concurrent_tasks_per_workflow: 5,
            default_workflow_timeout_ms: 300_000, // 5 minutes
            verbose_logging: false,
            auto_retry_failed_tasks: true,
            max_retry_attempts: 3,
        }
    }
}

/// Orchestrator execution statistics
#[derive(Debug, Default, Clone)]
pub struct OrchestratorStats {
    /// Total workflows executed
    pub total_workflows: usize,
    /// Successful workflows
    pub successful_workflows: usize,
    /// Failed workflows
    pub failed_workflows: usize,
    /// Total tasks executed
    pub total_tasks: usize,
    /// Successful tasks
    pub successful_tasks: usize,
    /// Failed tasks
    pub failed_tasks: usize,
    /// Average workflow execution time in milliseconds
    pub avg_workflow_time_ms: f64,
    /// Currently active workflows
    pub active_workflows: usize,
}

/// Workflow execution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorkflowStatus {
    /// Workflow is pending execution
    Pending,
    /// Workflow is currently running
    Running,
    /// Workflow completed successfully
    Completed,
    /// Workflow failed
    Failed,
    /// Workflow was cancelled
    Cancelled,
    /// Workflow is paused
    Paused,
}

/// Workflow execution result
#[derive(Debug, Clone)]
pub struct WorkflowResult {
    /// Workflow ID
    pub workflow_id: WorkflowId,
    /// Execution status
    pub status: WorkflowStatus,
    /// Execution start time
    pub started_at: DateTime<Utc>,
    /// Execution completion time
    pub completed_at: Option<DateTime<Utc>>,
    /// Total execution time in milliseconds
    pub duration_ms: Option<u64>,
    /// Task results by step ID
    pub task_results: HashMap<String, TaskResult>,
    /// Final workflow context
    pub context: WorkflowContext,
    /// Error message if failed
    pub error: Option<String>,
}

impl AgentOrchestrator {
    /// Create a new agent orchestrator
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            workflows: HashMap::new(),
            config: OrchestratorConfig::default(),
            stats: OrchestratorStats::default(),
        }
    }

    /// Create orchestrator with custom configuration
    pub fn with_config(config: OrchestratorConfig) -> Self {
        Self {
            agents: HashMap::new(),
            workflows: HashMap::new(),
            config,
            stats: OrchestratorStats::default(),
        }
    }

    /// Register an agent
    pub fn register_agent(&mut self, agent: Arc<dyn Agent>) -> Result<()> {
        let agent_id = agent.id();

        if self.agents.contains_key(&agent_id) {
            return Err(AgentError::orchestration(format!(
                "Agent with ID {} is already registered",
                agent_id
            )));
        }

        info!("Registering agent: {} ({})", agent.name(), agent_id);
        self.agents.insert(agent_id, agent);

        Ok(())
    }

    /// Unregister an agent
    pub fn unregister_agent(&mut self, agent_id: &AgentId) -> Result<()> {
        if self.agents.remove(agent_id).is_some() {
            info!("Unregistered agent: {}", agent_id);
            Ok(())
        } else {
            Err(AgentError::orchestration(format!(
                "Agent with ID {} not found",
                agent_id
            )))
        }
    }

    /// Get an agent by ID
    pub fn get_agent(&self, agent_id: &AgentId) -> Option<&Arc<dyn Agent>> {
        self.agents.get(agent_id)
    }

    /// List all registered agents
    pub fn list_agents(&self) -> Vec<&Arc<dyn Agent>> {
        self.agents.values().collect()
    }

    /// Execute a workflow
    pub async fn execute_workflow(&mut self, workflow: Workflow) -> Result<WorkflowResult> {
        let workflow_id = workflow.id();
        let start_time = Utc::now();

        info!(
            "Starting workflow execution: {} ({})",
            workflow.name(),
            workflow_id
        );

        // Check concurrent workflow limit
        if self.workflows.len() >= self.config.max_concurrent_workflows {
            return Err(AgentError::orchestration(
                "Maximum concurrent workflows limit reached",
            ));
        }

        // Validate workflow
        self.validate_workflow(&workflow)?;

        // Add to active workflows
        self.workflows.insert(workflow_id, workflow.clone());
        self.stats.active_workflows = self.workflows.len();

        let result = self.execute_workflow_internal(workflow).await;

        // Remove from active workflows
        self.workflows.remove(&workflow_id);
        self.stats.active_workflows = self.workflows.len();

        // Update statistics
        match &result {
            Ok(workflow_result) => self.update_workflow_stats(workflow_result, start_time),
            Err(_) => {
                // Handle error case for stats
                self.stats.total_workflows += 1;
                self.stats.failed_workflows += 1;
            }
        }

        result
    }

    /// Internal workflow execution logic
    async fn execute_workflow_internal(&self, mut workflow: Workflow) -> Result<WorkflowResult> {
        let workflow_id = workflow.id();
        let start_time = Utc::now();
        let mut task_results = HashMap::new();
        let mut completed_steps = Vec::new();

        workflow.set_status(WorkflowStatus::Running);

        if self.config.verbose_logging {
            debug!("Executing workflow steps for: {}", workflow_id);
        }

        // Execute workflow steps
        while let Some(ready_steps) = workflow.get_ready_steps(&completed_steps) {
            if ready_steps.is_empty() {
                break;
            }

            // Execute ready steps concurrently (up to the limit)
            let concurrent_limit = self.config.max_concurrent_tasks_per_workflow;
            let step_chunks: Vec<_> = ready_steps.chunks(concurrent_limit).collect();

            for chunk in step_chunks {
                let mut step_futures = Vec::new();

                for step in chunk {
                    let agent = self.agents.get(&step.agent_id).ok_or_else(|| {
                        AgentError::orchestration(format!(
                            "Agent {} not found for step {}",
                            step.agent_id, step.id
                        ))
                    })?;

                    let task = self.create_task_from_step(step, &workflow.context())?;
                    let agent_clone = Arc::clone(agent);

                    step_futures
                        .push(async move { (step.id.clone(), agent_clone.execute(&task).await) });
                }

                // Wait for all steps in this chunk to complete
                let step_results = futures::future::join_all(step_futures).await;

                for (step_id, result) in step_results {
                    match result {
                        Ok(response) => {
                            let task_result = TaskResult::success(
                                Uuid::new_v4(), // Task ID
                                response.content,
                                start_time,
                                Some(workflow.get_step(&step_id).unwrap().agent_id),
                            );

                            task_results.insert(step_id.clone(), task_result);

                            if self.config.verbose_logging {
                                debug!("Step '{}' completed successfully", step_id);
                            }

                            completed_steps.push(step_id);
                        }
                        Err(e) => {
                            let task_result = TaskResult::failure(
                                Uuid::new_v4(), // Task ID
                                e.to_string(),
                                start_time,
                                Some(workflow.get_step(&step_id).unwrap().agent_id),
                            );

                            task_results.insert(step_id.clone(), task_result);

                            error!("Step '{}' failed: {}", step_id, e);

                            // Check if step is retryable
                            if self.config.auto_retry_failed_tasks {
                                // TODO: Implement retry logic
                                warn!("Step '{}' failed, retry not implemented yet", step_id);
                            }

                            // For now, fail the entire workflow on any step failure
                            return Ok(WorkflowResult {
                                workflow_id,
                                status: WorkflowStatus::Failed,
                                started_at: start_time,
                                completed_at: Some(Utc::now()),
                                duration_ms: Some(
                                    (Utc::now() - start_time).num_milliseconds() as u64
                                ),
                                task_results,
                                context: workflow.context().clone(),
                                error: Some(format!("Step '{}' failed: {}", step_id, e)),
                            });
                        }
                    }
                }
            }
        }

        // Check if all steps completed
        let all_steps: Vec<_> = workflow.steps().iter().map(|s| s.id.clone()).collect();
        let workflow_completed = all_steps
            .iter()
            .all(|step_id| completed_steps.contains(step_id));

        let status = if workflow_completed {
            WorkflowStatus::Completed
        } else {
            WorkflowStatus::Failed
        };

        let completed_at = Utc::now();
        let duration_ms = (completed_at - start_time).num_milliseconds() as u64;

        info!(
            "Workflow {} {} in {}ms",
            workflow_id,
            if workflow_completed {
                "completed"
            } else {
                "failed"
            },
            duration_ms
        );

        Ok(WorkflowResult {
            workflow_id,
            status,
            started_at: start_time,
            completed_at: Some(completed_at),
            duration_ms: Some(duration_ms),
            task_results,
            context: workflow.context().clone(),
            error: if workflow_completed {
                None
            } else {
                Some("Not all workflow steps completed".to_string())
            },
        })
    }

    /// Validate a workflow before execution
    fn validate_workflow(&self, workflow: &Workflow) -> Result<()> {
        // Check that all required agents are registered
        for step in workflow.steps() {
            if !self.agents.contains_key(&step.agent_id) {
                return Err(AgentError::orchestration(format!(
                    "Agent {} required by step '{}' is not registered",
                    step.agent_id, step.id
                )));
            }
        }

        // Check for circular dependencies
        // TODO: Implement proper cycle detection

        Ok(())
    }

    /// Create a task from a workflow step
    fn create_task_from_step(
        &self,
        step: &WorkflowStep,
        context: &WorkflowContext,
    ) -> Result<Task> {
        let mut task = Task::builder()
            .name(&step.name)
            .input(step.description.as_deref().unwrap_or(""))
            .build()?;

        // Add workflow context variables to task context
        for (key, value) in &context.variables {
            task.add_context_variable(key, value.clone());
        }

        // Add step-specific configuration
        for (key, value) in &step.config {
            task.add_context_variable(format!("step_{}", key), value.clone());
        }

        Ok(task)
    }

    /// Update workflow statistics
    fn update_workflow_stats(&mut self, result: &WorkflowResult, start_time: DateTime<Utc>) {
        self.stats.total_workflows += 1;

        match result.status {
            WorkflowStatus::Completed => self.stats.successful_workflows += 1,
            WorkflowStatus::Failed => self.stats.failed_workflows += 1,
            _ => {}
        }

        let task_count = result.task_results.len();
        let successful_tasks = result
            .task_results
            .values()
            .filter(|r| r.status == TaskStatus::Completed)
            .count();

        self.stats.total_tasks += task_count;
        self.stats.successful_tasks += successful_tasks;
        self.stats.failed_tasks += task_count - successful_tasks;

        if let Some(duration_ms) = result.duration_ms {
            let total_time =
                self.stats.avg_workflow_time_ms * (self.stats.total_workflows - 1) as f64;
            self.stats.avg_workflow_time_ms =
                (total_time + duration_ms as f64) / self.stats.total_workflows as f64;
        }
    }

    /// Get orchestrator statistics
    pub fn stats(&self) -> &OrchestratorStats {
        &self.stats
    }

    /// Get orchestrator configuration
    pub fn config(&self) -> &OrchestratorConfig {
        &self.config
    }

    /// Update orchestrator configuration
    pub fn set_config(&mut self, config: OrchestratorConfig) {
        self.config = config;
    }

    /// Get active workflows
    pub fn active_workflows(&self) -> Vec<&Workflow> {
        self.workflows.values().collect()
    }

    /// Cancel a workflow
    pub fn cancel_workflow(&mut self, workflow_id: &WorkflowId) -> Result<()> {
        if let Some(workflow) = self.workflows.get_mut(workflow_id) {
            workflow.set_status(WorkflowStatus::Cancelled);
            info!("Cancelled workflow: {}", workflow_id);
            Ok(())
        } else {
            Err(AgentError::orchestration(format!(
                "Workflow {} not found or not active",
                workflow_id
            )))
        }
    }
}

impl Default for AgentOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}
