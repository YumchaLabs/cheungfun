//! Workflow definition and management.

use crate::{
    error::{AgentError, Result},
    orchestration::WorkflowStatus,
    types::{AgentId, WorkflowContext, WorkflowId, WorkflowStep},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Workflow definition for orchestrating multiple agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// Unique workflow identifier
    pub id: WorkflowId,
    /// Workflow name
    pub name: String,
    /// Workflow description
    pub description: Option<String>,
    /// Workflow steps
    pub steps: Vec<WorkflowStep>,
    /// Workflow context
    pub context: WorkflowContext,
    /// Workflow status
    pub status: WorkflowStatus,
    /// Workflow metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
    /// Workflow timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

impl Workflow {
    /// Create a new workflow
    pub fn new(name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            steps: Vec::new(),
            context: WorkflowContext::default(),
            status: WorkflowStatus::Pending,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
            timeout_ms: None,
        }
    }

    /// Create a workflow builder
    pub fn builder() -> WorkflowBuilder {
        WorkflowBuilder::new()
    }

    /// Get workflow ID
    pub fn id(&self) -> WorkflowId {
        self.id
    }

    /// Get workflow name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get workflow description
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Get workflow steps
    pub fn steps(&self) -> &[WorkflowStep] {
        &self.steps
    }

    /// Get workflow context
    pub fn context(&self) -> &WorkflowContext {
        &self.context
    }

    /// Get mutable workflow context
    pub fn context_mut(&mut self) -> &mut WorkflowContext {
        &mut self.context
    }

    /// Get workflow status
    pub fn status(&self) -> &WorkflowStatus {
        &self.status
    }

    /// Set workflow status
    pub fn set_status(&mut self, status: WorkflowStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }

    /// Add a step to the workflow
    pub fn add_step(&mut self, step: WorkflowStep) {
        self.steps.push(step);
        self.updated_at = Utc::now();
    }

    /// Get a step by ID
    pub fn get_step(&self, step_id: &str) -> Option<&WorkflowStep> {
        self.steps.iter().find(|step| step.id == step_id)
    }

    /// Get mutable step by ID
    pub fn get_step_mut(&mut self, step_id: &str) -> Option<&mut WorkflowStep> {
        self.steps.iter_mut().find(|step| step.id == step_id)
    }

    /// Remove a step by ID
    pub fn remove_step(&mut self, step_id: &str) -> Result<()> {
        let initial_len = self.steps.len();
        self.steps.retain(|step| step.id != step_id);

        if self.steps.len() < initial_len {
            self.updated_at = Utc::now();
            Ok(())
        } else {
            Err(AgentError::workflow(
                self.id.to_string(),
                format!("Step '{}' not found", step_id),
            ))
        }
    }

    /// Get steps that are ready to execute (all dependencies completed)
    pub fn get_ready_steps(&self, completed_steps: &[String]) -> Option<Vec<&WorkflowStep>> {
        let ready_steps: Vec<_> = self
            .steps
            .iter()
            .filter(|step| {
                // Step is ready if not already completed and all dependencies are completed
                !completed_steps.contains(&step.id)
                    && step
                        .dependencies
                        .iter()
                        .all(|dep| completed_steps.contains(dep))
            })
            .collect();

        if ready_steps.is_empty() {
            None
        } else {
            Some(ready_steps)
        }
    }

    /// Check if workflow has circular dependencies
    pub fn has_circular_dependencies(&self) -> bool {
        // Simple cycle detection using DFS
        let mut visited = HashMap::new();
        let mut rec_stack = HashMap::new();

        for step in &self.steps {
            if !visited.get(&step.id).unwrap_or(&false) {
                if self.has_cycle_util(&step.id, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }

        false
    }

    /// Utility function for cycle detection
    fn has_cycle_util(
        &self,
        step_id: &str,
        visited: &mut HashMap<String, bool>,
        rec_stack: &mut HashMap<String, bool>,
    ) -> bool {
        visited.insert(step_id.to_string(), true);
        rec_stack.insert(step_id.to_string(), true);

        if let Some(step) = self.get_step(step_id) {
            for dep in &step.dependencies {
                if !visited.get(dep).unwrap_or(&false) {
                    if self.has_cycle_util(dep, visited, rec_stack) {
                        return true;
                    }
                } else if *rec_stack.get(dep).unwrap_or(&false) {
                    return true;
                }
            }
        }

        rec_stack.insert(step_id.to_string(), false);
        false
    }

    /// Validate the workflow
    pub fn validate(&self) -> Result<()> {
        // Check for empty workflow
        if self.steps.is_empty() {
            return Err(AgentError::workflow(
                self.id.to_string(),
                "Workflow has no steps",
            ));
        }

        // Check for duplicate step IDs
        let mut step_ids = std::collections::HashSet::new();
        for step in &self.steps {
            if !step_ids.insert(&step.id) {
                return Err(AgentError::workflow(
                    self.id.to_string(),
                    format!("Duplicate step ID: {}", step.id),
                ));
            }
        }

        // Check for invalid dependencies
        for step in &self.steps {
            for dep in &step.dependencies {
                if !step_ids.contains(dep) {
                    return Err(AgentError::workflow(
                        self.id.to_string(),
                        format!("Step '{}' depends on non-existent step '{}'", step.id, dep),
                    ));
                }
            }
        }

        // Check for circular dependencies
        if self.has_circular_dependencies() {
            return Err(AgentError::workflow(
                self.id.to_string(),
                "Workflow has circular dependencies",
            ));
        }

        Ok(())
    }

    /// Get workflow execution order (topological sort)
    pub fn execution_order(&self) -> Result<Vec<String>> {
        self.validate()?;

        let mut in_degree = HashMap::new();
        let mut graph = HashMap::new();

        // Initialize in-degree and graph
        for step in &self.steps {
            in_degree.insert(step.id.clone(), step.dependencies.len());
            graph.insert(step.id.clone(), Vec::new());
        }

        // Build adjacency list
        for step in &self.steps {
            for dep in &step.dependencies {
                graph.get_mut(dep).unwrap().push(step.id.clone());
            }
        }

        // Topological sort using Kahn's algorithm
        let mut queue = Vec::new();
        let mut result = Vec::new();

        // Find all nodes with in-degree 0
        for (step_id, degree) in &in_degree {
            if *degree == 0 {
                queue.push(step_id.clone());
            }
        }

        while let Some(current) = queue.pop() {
            result.push(current.clone());

            // Reduce in-degree of neighbors
            if let Some(neighbors) = graph.get(&current) {
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

        if result.len() != self.steps.len() {
            Err(AgentError::workflow(
                self.id.to_string(),
                "Failed to determine execution order (circular dependency detected)",
            ))
        } else {
            Ok(result)
        }
    }

    /// Add context variable
    pub fn add_variable(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.context.variables.insert(key.into(), value);
        self.updated_at = Utc::now();
    }

    /// Get context variable
    pub fn get_variable(&self, key: &str) -> Option<&serde_json::Value> {
        self.context.variables.get(key)
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.metadata.insert(key.into(), value);
        self.updated_at = Utc::now();
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }
}

/// Workflow builder for convenient workflow construction
#[derive(Debug, Default)]
pub struct WorkflowBuilder {
    name: Option<String>,
    description: Option<String>,
    steps: Vec<WorkflowStep>,
    context: WorkflowContext,
    metadata: HashMap<String, serde_json::Value>,
    timeout_ms: Option<u64>,
}

impl WorkflowBuilder {
    /// Create a new workflow builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set workflow name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set workflow description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add a step to the workflow
    pub fn step(mut self, step: WorkflowStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Add multiple steps
    pub fn steps(mut self, steps: Vec<WorkflowStep>) -> Self {
        self.steps.extend(steps);
        self
    }

    /// Add context variable
    pub fn variable(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.context.variables.insert(key.into(), value);
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set workflow timeout
    pub fn timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Build the workflow
    pub fn build(self) -> Result<Workflow> {
        let name = self
            .name
            .ok_or_else(|| AgentError::validation("name", "Workflow name is required"))?;

        let now = Utc::now();
        let workflow = Workflow {
            id: Uuid::new_v4(),
            name,
            description: self.description,
            steps: self.steps,
            context: self.context,
            status: WorkflowStatus::Pending,
            metadata: self.metadata,
            created_at: now,
            updated_at: now,
            timeout_ms: self.timeout_ms,
        };

        // Validate the workflow
        workflow.validate()?;

        Ok(workflow)
    }
}

/// Helper function to create a workflow step
pub fn create_step(
    id: impl Into<String>,
    name: impl Into<String>,
    agent_id: AgentId,
) -> WorkflowStep {
    WorkflowStep {
        id: id.into(),
        name: name.into(),
        description: None,
        agent_id,
        dependencies: Vec::new(),
        config: HashMap::new(),
        retryable: true,
        max_retries: Some(3),
    }
}

/// Helper function to create a workflow step with dependencies
pub fn create_step_with_deps(
    id: impl Into<String>,
    name: impl Into<String>,
    agent_id: AgentId,
    dependencies: Vec<String>,
) -> WorkflowStep {
    WorkflowStep {
        id: id.into(),
        name: name.into(),
        description: None,
        agent_id,
        dependencies,
        config: HashMap::new(),
        retryable: true,
        max_retries: Some(3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_creation() {
        let workflow = Workflow::new("test_workflow");
        assert_eq!(workflow.name(), "test_workflow");
        assert_eq!(workflow.status(), &WorkflowStatus::Pending);
        assert!(workflow.steps().is_empty());
    }

    #[test]
    fn test_workflow_builder() {
        let agent_id = Uuid::new_v4();
        let step = create_step("step1", "Test Step", agent_id);

        let workflow = Workflow::builder()
            .name("Test Workflow")
            .description("A test workflow")
            .step(step)
            .variable("test_var", serde_json::json!("test_value"))
            .build()
            .unwrap();

        assert_eq!(workflow.name(), "Test Workflow");
        assert_eq!(workflow.steps().len(), 1);
        assert_eq!(
            workflow.get_variable("test_var"),
            Some(&serde_json::json!("test_value"))
        );
    }

    #[test]
    fn test_workflow_validation() {
        // Test empty workflow
        let workflow = Workflow::builder().name("Empty Workflow").build();
        assert!(workflow.is_err());

        // Test valid workflow
        let agent_id = Uuid::new_v4();
        let step = create_step("step1", "Test Step", agent_id);
        let workflow = Workflow::builder()
            .name("Valid Workflow")
            .step(step)
            .build();
        assert!(workflow.is_ok());
    }

    #[test]
    fn test_circular_dependency_detection() {
        let agent_id = Uuid::new_v4();

        // Create steps with circular dependency
        let step1 = create_step_with_deps("step1", "Step 1", agent_id, vec!["step2".to_string()]);
        let step2 = create_step_with_deps("step2", "Step 2", agent_id, vec!["step1".to_string()]);

        let workflow = Workflow::builder()
            .name("Circular Workflow")
            .step(step1)
            .step(step2)
            .build();

        assert!(workflow.is_err());
    }

    #[test]
    fn test_execution_order() {
        let agent_id = Uuid::new_v4();

        let step1 = create_step("step1", "Step 1", agent_id);
        let step2 = create_step_with_deps("step2", "Step 2", agent_id, vec!["step1".to_string()]);
        let step3 = create_step_with_deps(
            "step3",
            "Step 3",
            agent_id,
            vec!["step1".to_string(), "step2".to_string()],
        );

        let workflow = Workflow::builder()
            .name("Sequential Workflow")
            .step(step1)
            .step(step2)
            .step(step3)
            .build()
            .unwrap();

        let order = workflow.execution_order().unwrap();
        assert_eq!(order[0], "step1");
        assert!(
            order.iter().position(|x| x == "step2").unwrap()
                > order.iter().position(|x| x == "step1").unwrap()
        );
        assert!(
            order.iter().position(|x| x == "step3").unwrap()
                > order.iter().position(|x| x == "step2").unwrap()
        );
    }
}
