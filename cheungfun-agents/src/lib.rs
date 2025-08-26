//! Agent framework and MCP integration for Cheungfun.
//!
//! This crate provides intelligent agents and Model Context Protocol (MCP)
//! integration for tool calling and complex workflows.
//!
//! # Features
//!
//! - **Agent Framework**: Core agent traits and implementations
//! - **Tool System**: Extensible tool registry and built-in tools
//! - **MCP Integration**: Full Model Context Protocol support via rmcp
//! - **Agent Orchestration**: Multi-agent coordination and workflows
//! - **RAG Integration**: Deep integration with cheungfun-query for knowledge-enhanced agents
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use cheungfun_agents::prelude::*;
//! use cheungfun_core::Result;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create a simple agent
//!     let agent = Agent::builder()
//!         .name("assistant")
//!         .description("A helpful assistant")
//!         .tools(vec![
//!             Box::new(EchoTool::new()),
//!             Box::new(HttpTool::new()),
//!         ])
//!         .build()?;
//!
//!     // Execute a task
//!     let task = Task::new("Echo 'Hello, World!'");
//!     let response = agent.execute(&task).await?;
//!
//!     println!("Agent response: {}", response.content);
//!     Ok(())
//! }
//! ```

#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

// Core modules
pub mod agent;
pub mod error;
pub mod llm;
pub mod mcp;
pub mod tool;
pub mod types;
pub mod workflow;

// Re-exports for convenience
pub use error::{AgentError, Result};
pub use mcp::{McpClient, McpServer, McpService};
pub use tool::{Tool, ToolRegistry, ToolResult};
pub use types::*;
pub use workflow::{
    utils, Workflow, WorkflowBuilder, WorkflowExecutor, WorkflowResult, WorkflowStatus,
};

/// Prelude module for convenient imports
pub mod prelude {
    // Core exports
    pub use crate::{
        AgentError, McpClient, McpServer, McpService, Result, Tool, ToolRegistry, ToolResult,
    };

    // Built-in tools
    pub use crate::tool::builtin::{EchoTool, FileTool, HttpTool, SearchTool};

    // Memory management
    pub use cheungfun_core::traits::BaseMemory;
    pub use cheungfun_query::memory::{ChatMemoryBuffer, ChatMemoryConfig};

    // LLM integration
    pub use crate::llm::{LlmClientFactory, LlmClientManager, LlmConfig, MessageConverter};

    // MCP components
    pub use crate::mcp::{server::McpServerBuilder, McpToolRegistry};

    // Common types
    pub use crate::types::{
        AgentId, AgentMessage, AgentResponse, ToolCall, ToolCallId, ToolOutput, ToolSchema,
    };

    // Workflow system
    pub use crate::workflow::{
        utils as workflow_utils, Workflow, WorkflowBuilder, WorkflowExecutor, WorkflowResult,
        WorkflowStatus,
    };

    // Agent system
    pub use crate::agent::{
        ActionStep, AgentBuilder, BuiltAgent, AgentContext, AgentStatus, BaseAgent, FinalAnswerStep,
        ObservationStep, ReActAgent, ReActConfig, ReActStats, ReasoningStep, ReasoningStepType,
        ReasoningTrace, ThoughtStep, AgentStrategy, DirectStrategy, FunctionCallingStrategy,
    };
}
