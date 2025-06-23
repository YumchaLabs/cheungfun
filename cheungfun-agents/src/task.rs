//! Task management and execution for agents.

use crate::{
    error::{AgentError, Result},
    types::{AgentId, AgentMessage, TaskId, WorkflowContext},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Task status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    /// Task is pending execution
    Pending,
    /// Task is currently running
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed with error
    Failed,
    /// Task was cancelled
    Cancelled,
    /// Task is paused/suspended
    Paused,
}

/// Task priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Low priority task
    Low,
    /// Normal priority task
    Normal,
    /// High priority task
    High,
    /// Critical priority task
    Critical,
}

/// Core task structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier
    pub id: TaskId,
    /// Task name/title
    pub name: String,
    /// Task description
    pub description: Option<String>,
    /// Task input/prompt
    pub input: String,
    /// Task priority
    pub priority: TaskPriority,
    /// Task status
    pub status: TaskStatus,
    /// Assigned agent ID
    pub agent_id: Option<AgentId>,
    /// Task context and variables
    pub context: TaskContext,
    /// Task metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Task creation timestamp
    pub created_at: DateTime<Utc>,
    /// Task last updated timestamp
    pub updated_at: DateTime<Utc>,
    /// Task deadline (optional)
    pub deadline: Option<DateTime<Utc>>,
    /// Maximum execution time in milliseconds
    pub max_execution_time_ms: Option<u64>,
    /// Task dependencies (must complete before this task)
    pub dependencies: Vec<TaskId>,
    /// Parent task ID (for subtasks)
    pub parent_task_id: Option<TaskId>,
}

/// Task execution context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TaskContext {
    /// Context variables
    pub variables: HashMap<String, serde_json::Value>,
    /// Conversation history
    pub messages: Vec<AgentMessage>,
    /// Workflow context (if part of a workflow)
    pub workflow_context: Option<WorkflowContext>,
    /// Additional context data
    pub data: HashMap<String, serde_json::Value>,
}

/// Task execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task ID
    pub task_id: TaskId,
    /// Execution status
    pub status: TaskStatus,
    /// Result content
    pub content: String,
    /// Result metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Execution start time
    pub started_at: DateTime<Utc>,
    /// Execution completion time
    pub completed_at: Option<DateTime<Utc>>,
    /// Execution duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Error message if failed
    pub error: Option<String>,
    /// Agent that executed the task
    pub agent_id: Option<AgentId>,
}

/// Task builder for convenient task creation
#[derive(Debug, Default)]
pub struct TaskBuilder {
    name: Option<String>,
    description: Option<String>,
    input: Option<String>,
    priority: TaskPriority,
    agent_id: Option<AgentId>,
    context: TaskContext,
    metadata: HashMap<String, serde_json::Value>,
    deadline: Option<DateTime<Utc>>,
    max_execution_time_ms: Option<u64>,
    dependencies: Vec<TaskId>,
    parent_task_id: Option<TaskId>,
}

impl Task {
    /// Create a new task with the given input
    pub fn new(input: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: "Untitled Task".to_string(),
            description: None,
            input: input.into(),
            priority: TaskPriority::Normal,
            status: TaskStatus::Pending,
            agent_id: None,
            context: TaskContext::default(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
            deadline: None,
            max_execution_time_ms: None,
            dependencies: Vec::new(),
            parent_task_id: None,
        }
    }

    /// Create a task builder
    pub fn builder() -> TaskBuilder {
        TaskBuilder::default()
    }

    /// Update task status
    pub fn set_status(&mut self, status: TaskStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }

    /// Assign agent to task
    pub fn assign_agent(&mut self, agent_id: AgentId) {
        self.agent_id = Some(agent_id);
        self.updated_at = Utc::now();
    }

    /// Add context variable
    pub fn add_context_variable(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.context.variables.insert(key.into(), value);
        self.updated_at = Utc::now();
    }

    /// Add message to context
    pub fn add_message(&mut self, message: AgentMessage) {
        self.context.messages.push(message);
        self.updated_at = Utc::now();
    }

    /// Check if task is ready to execute (all dependencies completed)
    pub fn is_ready(&self, completed_tasks: &[TaskId]) -> bool {
        self.dependencies
            .iter()
            .all(|dep| completed_tasks.contains(dep))
    }

    /// Check if task has expired (past deadline)
    pub fn is_expired(&self) -> bool {
        if let Some(deadline) = self.deadline {
            Utc::now() > deadline
        } else {
            false
        }
    }

    /// Get task age in milliseconds
    pub fn age_ms(&self) -> u64 {
        (Utc::now() - self.created_at).num_milliseconds() as u64
    }
}

impl TaskBuilder {
    /// Set task name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set task description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set task input
    pub fn input(mut self, input: impl Into<String>) -> Self {
        self.input = Some(input.into());
        self
    }

    /// Set task priority
    pub fn priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Assign agent to task
    pub fn agent_id(mut self, agent_id: AgentId) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    /// Add context variable
    pub fn context_variable(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.context.variables.insert(key.into(), value);
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set deadline
    pub fn deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Set maximum execution time
    pub fn max_execution_time_ms(mut self, ms: u64) -> Self {
        self.max_execution_time_ms = Some(ms);
        self
    }

    /// Add dependency
    pub fn dependency(mut self, task_id: TaskId) -> Self {
        self.dependencies.push(task_id);
        self
    }

    /// Set parent task
    pub fn parent_task(mut self, parent_id: TaskId) -> Self {
        self.parent_task_id = Some(parent_id);
        self
    }

    /// Build the task
    pub fn build(self) -> Result<Task> {
        let input = self
            .input
            .ok_or_else(|| AgentError::validation("input", "Task input is required"))?;

        let now = Utc::now();
        Ok(Task {
            id: Uuid::new_v4(),
            name: self.name.unwrap_or_else(|| "Untitled Task".to_string()),
            description: self.description,
            input,
            priority: self.priority,
            status: TaskStatus::Pending,
            agent_id: self.agent_id,
            context: self.context,
            metadata: self.metadata,
            created_at: now,
            updated_at: now,
            deadline: self.deadline,
            max_execution_time_ms: self.max_execution_time_ms,
            dependencies: self.dependencies,
            parent_task_id: self.parent_task_id,
        })
    }
}

impl TaskResult {
    /// Create a successful task result
    pub fn success(
        task_id: TaskId,
        content: impl Into<String>,
        started_at: DateTime<Utc>,
        agent_id: Option<AgentId>,
    ) -> Self {
        let completed_at = Utc::now();
        let duration_ms = (completed_at - started_at).num_milliseconds() as u64;

        Self {
            task_id,
            status: TaskStatus::Completed,
            content: content.into(),
            metadata: HashMap::new(),
            started_at,
            completed_at: Some(completed_at),
            duration_ms: Some(duration_ms),
            error: None,
            agent_id,
        }
    }

    /// Create a failed task result
    pub fn failure(
        task_id: TaskId,
        error: impl Into<String>,
        started_at: DateTime<Utc>,
        agent_id: Option<AgentId>,
    ) -> Self {
        let completed_at = Utc::now();
        let duration_ms = (completed_at - started_at).num_milliseconds() as u64;

        Self {
            task_id,
            status: TaskStatus::Failed,
            content: String::new(),
            metadata: HashMap::new(),
            started_at,
            completed_at: Some(completed_at),
            duration_ms: Some(duration_ms),
            error: Some(error.into()),
            agent_id,
        }
    }
}

impl Default for TaskPriority {
    fn default() -> Self {
        Self::Normal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_creation() {
        let task = Task::new("Test task");
        assert_eq!(task.input, "Test task");
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.priority, TaskPriority::Normal);
    }

    #[test]
    fn test_task_builder() {
        let task = Task::builder()
            .name("Test Task")
            .input("Do something")
            .priority(TaskPriority::High)
            .build()
            .unwrap();

        assert_eq!(task.name, "Test Task");
        assert_eq!(task.input, "Do something");
        assert_eq!(task.priority, TaskPriority::High);
    }

    #[test]
    fn test_task_ready_check() {
        let dep1 = Uuid::new_v4();
        let dep2 = Uuid::new_v4();

        let mut task = Task::new("Test");
        task.dependencies = vec![dep1, dep2];

        assert!(!task.is_ready(&[dep1]));
        assert!(task.is_ready(&[dep1, dep2]));
    }
}
