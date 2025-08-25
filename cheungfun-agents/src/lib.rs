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
pub mod mcp;
pub mod orchestration;
pub mod task;
pub mod tool;
pub mod types;

// Re-exports for convenience
pub use agent::{Agent, AgentBuilder, AgentExecutor, RagAgent, RagAgentConfig};
pub use error::{AgentError, Result};
pub use mcp::{McpClient, McpServer, McpService};
pub use orchestration::{AgentOrchestrator, Workflow, WorkflowBuilder};
pub use task::{Task, TaskBuilder, TaskContext, TaskResult, TaskStatus};
pub use tool::{Tool, ToolRegistry, ToolResult};
pub use types::*;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        Agent, AgentBuilder, AgentError, AgentExecutor, AgentOrchestrator, McpClient, McpServer,
        McpService, RagAgent, RagAgentConfig, Result, Task, TaskBuilder, TaskContext, TaskResult,
        TaskStatus, Tool, ToolRegistry, ToolResult, Workflow, WorkflowBuilder,
    };

    // Re-export types
    pub use crate::types::AgentCapabilities;

    // Built-in tools
    pub use crate::tool::builtin::{EchoTool, FileTool, HttpTool, SearchTool};

    // Orchestration helpers
    pub use crate::orchestration::workflow::{create_step, create_step_with_deps};

    // MCP components
    pub use crate::mcp::{server::McpServerBuilder, McpToolRegistry};

    // Common types
    pub use crate::types::{
        AgentId, AgentMessage, AgentResponse, ToolCall, ToolOutput, WorkflowStep,
    };
}
