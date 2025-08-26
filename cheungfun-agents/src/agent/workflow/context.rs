//! Workflow context and state management
//! 
//! This module provides context management for workflows, including state storage,
//! variable management, and cross-step data sharing.

use crate::error::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Workflow context that maintains state across execution steps
#[derive(Debug, Clone)]
pub struct WorkflowContext {
    /// Unique context identifier
    pub id: Uuid,
    /// Key-value store for workflow state
    pub store: Arc<WorkflowStore>,
    /// Variables accessible throughout the workflow
    pub variables: HashMap<String, serde_json::Value>,
    /// Metadata for the current execution
    pub metadata: HashMap<String, serde_json::Value>,
    /// Current step information
    pub current_step: Option<String>,
    /// Execution history
    pub history: Vec<ExecutionStep>,
}

impl WorkflowContext {
    /// Create a new workflow context
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            store: Arc::new(WorkflowStore::new()),
            variables: HashMap::new(),
            metadata: HashMap::new(),
            current_step: None,
            history: Vec::new(),
        }
    }
    
    /// Create a context with initial variables
    pub fn with_variables(variables: HashMap<String, serde_json::Value>) -> Self {
        Self {
            id: Uuid::new_v4(),
            store: Arc::new(WorkflowStore::new()),
            variables,
            metadata: HashMap::new(),
            current_step: None,
            history: Vec::new(),
        }
    }
    
    /// Set a variable in the context
    pub fn set_variable(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.variables.insert(key.into(), value);
    }
    
    /// Get a variable from the context
    pub fn get_variable(&self, key: &str) -> Option<&serde_json::Value> {
        self.variables.get(key)
    }
    
    /// Set metadata
    pub fn set_metadata(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.metadata.insert(key.into(), value);
    }
    
    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }
    
    /// Set the current step
    pub fn set_current_step(&mut self, step: impl Into<String>) {
        self.current_step = Some(step.into());
    }
    
    /// Add an execution step to history
    pub fn add_step(&mut self, step: ExecutionStep) {
        self.history.push(step);
    }
    
    /// Get the execution history
    pub fn history(&self) -> &[ExecutionStep] {
        &self.history
    }
    
    /// Clear the context (useful for cleanup)
    pub async fn clear(&mut self) {
        self.variables.clear();
        self.metadata.clear();
        self.current_step = None;
        self.history.clear();
        self.store.clear().await;
    }
}

impl Default for WorkflowContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Persistent store for workflow state
#[derive(Debug)]
pub struct WorkflowStore {
    /// In-memory storage
    data: Arc<RwLock<HashMap<String, serde_json::Value>>>,
}

impl WorkflowStore {
    /// Create a new workflow store
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Store a value with the given key
    pub async fn set(&self, key: impl Into<String>, value: serde_json::Value) {
        let mut data = self.data.write().await;
        data.insert(key.into(), value);
    }
    
    /// Retrieve a value by key
    pub async fn get(&self, key: &str) -> Option<serde_json::Value> {
        let data = self.data.read().await;
        data.get(key).cloned()
    }
    
    /// Retrieve a value by key with a default
    pub async fn get_or_default<T>(&self, key: &str, default: T) -> T 
    where
        T: for<'de> Deserialize<'de> + Serialize + Clone,
    {
        match self.get(key).await {
            Some(value) => serde_json::from_value(value).unwrap_or(default),
            None => default,
        }
    }
    
    /// Check if a key exists
    pub async fn contains_key(&self, key: &str) -> bool {
        let data = self.data.read().await;
        data.contains_key(key)
    }
    
    /// Remove a value by key
    pub async fn remove(&self, key: &str) -> Option<serde_json::Value> {
        let mut data = self.data.write().await;
        data.remove(key)
    }
    
    /// Get all keys
    pub async fn keys(&self) -> Vec<String> {
        let data = self.data.read().await;
        data.keys().cloned().collect()
    }
    
    /// Clear all data
    pub async fn clear(&self) {
        let mut data = self.data.write().await;
        data.clear();
    }
    
    /// Get the number of stored items
    pub async fn len(&self) -> usize {
        let data = self.data.read().await;
        data.len()
    }
    
    /// Check if the store is empty
    pub async fn is_empty(&self) -> bool {
        let data = self.data.read().await;
        data.is_empty()
    }
}

impl Default for WorkflowStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a single execution step in the workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    /// Step identifier
    pub id: String,
    /// Step name
    pub name: String,
    /// Step type (e.g., "reasoning", "tool_call", "response")
    pub step_type: String,
    /// Input to the step
    pub input: Option<serde_json::Value>,
    /// Output from the step
    pub output: Option<serde_json::Value>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Whether the step was successful
    pub success: bool,
    /// Error message if the step failed
    pub error: Option<String>,
    /// Timestamp when the step started
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// Timestamp when the step completed
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Additional metadata for the step
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ExecutionStep {
    /// Create a new execution step
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        step_type: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            step_type: step_type.into(),
            input: None,
            output: None,
            execution_time_ms: 0,
            success: false,
            error: None,
            started_at: chrono::Utc::now(),
            completed_at: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Set the input for this step
    pub fn with_input(mut self, input: serde_json::Value) -> Self {
        self.input = Some(input);
        self
    }
    
    /// Set the output for this step
    pub fn with_output(mut self, output: serde_json::Value) -> Self {
        self.output = Some(output);
        self
    }
    
    /// Mark the step as completed successfully
    pub fn complete_success(mut self, output: Option<serde_json::Value>) -> Self {
        self.success = true;
        self.output = output;
        self.completed_at = Some(chrono::Utc::now());
        self.execution_time_ms = self.completed_at
            .unwrap()
            .signed_duration_since(self.started_at)
            .num_milliseconds() as u64;
        self
    }
    
    /// Mark the step as failed
    pub fn complete_error(mut self, error: impl Into<String>) -> Self {
        self.success = false;
        self.error = Some(error.into());
        self.completed_at = Some(chrono::Utc::now());
        self.execution_time_ms = self.completed_at
            .unwrap()
            .signed_duration_since(self.started_at)
            .num_milliseconds() as u64;
        self
    }
    
    /// Add metadata to the step
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Context manager for handling workflow execution context
#[derive(Debug)]
pub struct ContextManager {
    /// Active contexts
    contexts: Arc<RwLock<HashMap<Uuid, WorkflowContext>>>,
}

impl ContextManager {
    /// Create a new context manager
    pub fn new() -> Self {
        Self {
            contexts: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Create a new context
    pub async fn create_context(&self) -> WorkflowContext {
        let context = WorkflowContext::new();
        let mut contexts = self.contexts.write().await;
        contexts.insert(context.id, context.clone());
        context
    }
    
    /// Get a context by ID
    pub async fn get_context(&self, id: Uuid) -> Option<WorkflowContext> {
        let contexts = self.contexts.read().await;
        contexts.get(&id).cloned()
    }
    
    /// Update a context
    pub async fn update_context(&self, context: WorkflowContext) -> Result<()> {
        let mut contexts = self.contexts.write().await;
        contexts.insert(context.id, context);
        Ok(())
    }
    
    /// Remove a context
    pub async fn remove_context(&self, id: Uuid) -> Result<()> {
        let mut contexts = self.contexts.write().await;
        contexts.remove(&id);
        Ok(())
    }
    
    /// Get the number of active contexts
    pub async fn context_count(&self) -> usize {
        let contexts = self.contexts.read().await;
        contexts.len()
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new()
    }
}
