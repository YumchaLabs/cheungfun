//! ReAct (Reasoning and Acting) agent implementation
//!
//! This module provides a complete ReAct agent implementation following the
//! ReAct paper's methodology for combining reasoning and action in language models.
//!
//! The ReAct pattern allows agents to:
//! 1. **Reason** about the problem and plan their approach
//! 2. **Act** by calling tools or taking actions
//! 3. **Observe** the results of their actions
//! 4. **Repeat** until the problem is solved

pub mod agent;
pub mod reasoning;

// Re-export main components
pub use agent::{ReActAgent, ReActConfig, ReActStats};
pub use reasoning::{
    ActionStep, FinalAnswerStep, ObservationStep, ReasoningStep, ReasoningStepType, ReasoningTrace,
    ThoughtStep,
};

/// ReAct agent prelude for convenient imports
pub mod prelude {
    pub use super::{
        ActionStep, FinalAnswerStep, ObservationStep, ReActAgent, ReActConfig, ReActStats,
        ReasoningStep, ReasoningStepType, ReasoningTrace, ThoughtStep,
    };
}
