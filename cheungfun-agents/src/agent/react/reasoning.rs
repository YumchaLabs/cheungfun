//! `ReAct` reasoning step definitions and logic
//!
//! This module defines the core reasoning steps used in the `ReAct` pattern:
//! - Thought: Internal reasoning about the problem
//! - Action: Tool calls or actions to take
//! - Observation: Results from actions
//! - Final Answer: The final response to the user

use crate::types::{ToolCall, ToolOutput};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Type of reasoning step in the `ReAct` pattern
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningStepType {
    /// Internal reasoning/thinking step
    Thought,
    /// Action step (tool call)
    Action,
    /// Observation of action results
    Observation,
    /// Final answer to the user
    FinalAnswer,
}

/// A single step in the `ReAct` reasoning process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningStep {
    /// Thought step - internal reasoning
    Thought(ThoughtStep),
    /// Action step - tool call or action
    Action(ActionStep),
    /// Observation step - result of action
    Observation(ObservationStep),
    /// Final answer step - response to user
    FinalAnswer(FinalAnswerStep),
}

impl ReasoningStep {
    /// Get the type of this reasoning step
    #[must_use]
    pub fn step_type(&self) -> ReasoningStepType {
        match self {
            ReasoningStep::Thought(_) => ReasoningStepType::Thought,
            ReasoningStep::Action(_) => ReasoningStepType::Action,
            ReasoningStep::Observation(_) => ReasoningStepType::Observation,
            ReasoningStep::FinalAnswer(_) => ReasoningStepType::FinalAnswer,
        }
    }

    /// Get the content of this step as a string
    #[must_use]
    pub fn content(&self) -> &str {
        match self {
            ReasoningStep::Thought(step) => &step.content,
            ReasoningStep::Action(step) => &step.reasoning,
            ReasoningStep::Observation(step) => &step.content,
            ReasoningStep::FinalAnswer(step) => &step.content,
        }
    }

    /// Get the step ID
    #[must_use]
    pub fn id(&self) -> Uuid {
        match self {
            ReasoningStep::Thought(step) => step.id,
            ReasoningStep::Action(step) => step.id,
            ReasoningStep::Observation(step) => step.id,
            ReasoningStep::FinalAnswer(step) => step.id,
        }
    }
}

/// Thought step - internal reasoning about the problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtStep {
    /// Step ID
    pub id: Uuid,
    /// The thought content
    pub content: String,
    /// Metadata for this thought
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ThoughtStep {
    /// Create a new thought step
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Add metadata to this thought
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Action step - represents a tool call or action to take
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionStep {
    /// Step ID
    pub id: Uuid,
    /// Reasoning for taking this action
    pub reasoning: String,
    /// The tool call to make
    pub tool_call: ToolCall,
    /// Metadata for this action
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ActionStep {
    /// Create a new action step
    pub fn new(
        reasoning: impl Into<String>,
        tool_name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            reasoning: reasoning.into(),
            tool_call: ToolCall::new(tool_name, arguments),
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Create an action step with an existing tool call
    pub fn with_tool_call(reasoning: impl Into<String>, tool_call: ToolCall) -> Self {
        Self {
            id: Uuid::new_v4(),
            reasoning: reasoning.into(),
            tool_call,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Add metadata to this action
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Observation step - represents the result of an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationStep {
    /// Step ID
    pub id: Uuid,
    /// The observation content
    pub content: String,
    /// The tool output that generated this observation
    pub tool_output: Option<ToolOutput>,
    /// Whether the observation indicates success
    pub success: bool,
    /// Metadata for this observation
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ObservationStep {
    /// Create a new observation step
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            tool_output: None,
            success: true,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Create an observation step from a tool output
    #[must_use]
    pub fn from_tool_output(tool_output: ToolOutput) -> Self {
        let success = !tool_output.is_error;
        let content = tool_output.content.clone();

        Self {
            id: Uuid::new_v4(),
            content,
            tool_output: Some(tool_output),
            success,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Mark this observation as failed
    #[must_use]
    pub fn failed(mut self) -> Self {
        self.success = false;
        self
    }

    /// Add metadata to this observation
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Final answer step - represents the final response to the user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalAnswerStep {
    /// Step ID
    pub id: Uuid,
    /// The final answer content
    pub content: String,
    /// Confidence level (0.0 to 1.0)
    pub confidence: Option<f64>,
    /// Metadata for this final answer
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl FinalAnswerStep {
    /// Create a new final answer step
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            confidence: None,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Set the confidence level
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Add metadata to this final answer
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// A complete reasoning trace containing all steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrace {
    /// Trace ID
    pub id: Uuid,
    /// All reasoning steps in order
    pub steps: Vec<ReasoningStep>,
    /// Whether the reasoning was successful
    pub success: bool,
    /// Error message if unsuccessful
    pub error: Option<String>,
    /// Total reasoning time in milliseconds
    pub total_time_ms: u64,
    /// Metadata for this trace
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp when trace was created
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ReasoningTrace {
    /// Create a new empty reasoning trace
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            steps: Vec::new(),
            success: false,
            error: None,
            total_time_ms: 0,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Add a step to the trace
    pub fn add_step(&mut self, step: ReasoningStep) {
        self.steps.push(step);
    }

    /// Mark the trace as successful
    #[must_use]
    pub fn mark_success(mut self) -> Self {
        self.success = true;
        self
    }

    /// Mark the trace as failed with an error
    pub fn mark_failed(mut self, error: impl Into<String>) -> Self {
        self.success = false;
        self.error = Some(error.into());
        self
    }

    /// Set the total reasoning time
    #[must_use]
    pub fn with_total_time(mut self, ms: u64) -> Self {
        self.total_time_ms = ms;
        self
    }

    /// Get the final answer from the trace, if any
    #[must_use]
    pub fn final_answer(&self) -> Option<&FinalAnswerStep> {
        self.steps.iter().rev().find_map(|step| {
            if let ReasoningStep::FinalAnswer(answer) = step {
                Some(answer)
            } else {
                None
            }
        })
    }

    /// Get all thought steps
    #[must_use]
    pub fn thoughts(&self) -> Vec<&ThoughtStep> {
        self.steps
            .iter()
            .filter_map(|step| {
                if let ReasoningStep::Thought(thought) = step {
                    Some(thought)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all action steps
    #[must_use]
    pub fn actions(&self) -> Vec<&ActionStep> {
        self.steps
            .iter()
            .filter_map(|step| {
                if let ReasoningStep::Action(action) = step {
                    Some(action)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all observation steps
    #[must_use]
    pub fn observations(&self) -> Vec<&ObservationStep> {
        self.steps
            .iter()
            .filter_map(|step| {
                if let ReasoningStep::Observation(obs) = step {
                    Some(obs)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the number of steps
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Check if the trace is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

impl Default for ReasoningTrace {
    fn default() -> Self {
        Self::new()
    }
}
