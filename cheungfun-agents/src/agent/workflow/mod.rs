//! Workflow-based agent system
//! 
//! This module provides workflow-based agents that support streaming,
//! event-driven processing, and complex multi-step reasoning patterns.

pub mod base_agent;
pub mod context;
pub mod events;

// Re-export main components
pub use base_agent::{BaseWorkflowAgent, WorkflowAgent};
pub use context::{WorkflowContext, WorkflowStore};
pub use events::{
    WorkflowEvent, AgentInput, AgentOutput, AgentStream, 
    ToolCallResult, ToolSelection,
};

/// Workflow agent prelude for convenient imports
pub mod prelude {
    pub use super::{
        BaseWorkflowAgent, WorkflowAgent,
        WorkflowContext, WorkflowStore,
        AgentInput, AgentOutput, AgentStream,
        ToolCallResult, ToolSelection,
    };
}
