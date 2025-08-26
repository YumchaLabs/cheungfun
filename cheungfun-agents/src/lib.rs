//! Cheungfun Agents Framework
//!
//! A comprehensive agent framework for building AI-powered applications with support for
//! `ReAct` reasoning, workflow orchestration, and tool integration.
//!
//! ## Features
//!
//! - **Multiple Agent Types**: `ReAct`, Function Calling, and Workflow agents
//! - **Event-Driven Architecture**: Stream-based agent interactions with real-time feedback
//! - **Tool Integration**: Comprehensive tool system with async execution
//! - **Memory Management**: Persistent conversation and context management
//! - **Workflow Orchestration**: Both simple and advanced workflow execution engines
//! - **`LlamaIndex` Compatibility**: Architecture inspired by and compatible with `LlamaIndex`
//! - **MCP Integration**: Full Model Context Protocol support via rmcp
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use cheungfun_agents::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create a simple ReAct agent
//!     let agent = ReActAgent::builder()
//!         .name("Assistant")
//!         .description("A helpful assistant")
//!         .with_tools()
//!         .build()?;
//!         
//!     // Or create a workflow agent for complex interactions
//!     let workflow_agent = ReActWorkflowAgentBuilder::new()
//!         .name("Math Helper")
//!         .llm(llm)
//!         .tools(tools)
//!         .build()?;
//!     
//!     // Execute with workflow engine
//!     let mut engine = WorkflowEngineBuilder::new()
//!         .max_iterations(10)
//!         .build();
//!     
//!     let result = engine.run(
//!         &workflow_agent,
//!         Some("What is 2 + 2?".to_string()),
//!         None,
//!         None,
//!     ).await?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! The framework is built around several key concepts:
//!
//! ### Agents
//! - **`BaseAgent`**: Core agent trait for basic chat functionality
//! - **`WorkflowAgent`**: Advanced agent trait for event-driven workflows
//! - **`ReActAgent`**: Reasoning and acting agent with tool support
//! - **`ReActWorkflowAgent`**: `ReAct` agent integrated with workflow system
//!
//! ### Workflows
//! - **Simple Workflow**: Basic step-by-step execution with dependency management
//! - **Workflow Engine**: Event-driven execution with streaming and state management
//! - **Context Management**: Persistent state across workflow steps
//!
//! ### Tools
//! - **Tool Trait**: Async tool execution interface
//! - **`ToolRegistry`**: Centralized tool management
//! - **Built-in Tools**: Common utility tools (calculator, web search, etc.)
//!
//! ### Memory
//! - **`BaseMemory`**: Memory management interface from cheungfun-core
//! - **Integration**: Seamless integration with workflow contexts

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

// Re-export core functionality for convenience
pub use agent::{
    prelude as agent_prelude, AgentBuilder, AgentFactory, AgentType, BaseAgent, ReActAgent,
};
pub use error::{AgentError, Result};
pub use mcp::{McpClient, McpServer, McpService};
pub use tool::{Tool, ToolRegistry, ToolResult};
pub use types::{
    AgentCapabilities, AgentConfig, AgentId, AgentMessage, AgentResponse, ExecutionStats,
    MessageRole, ToolCall, ToolOutput, ToolSchema,
};

// Re-export workflow systems
pub use workflow::{
    utils as simple_workflow_utils, SimpleWorkflow, SimpleWorkflowBuilder, SimpleWorkflowExecutor,
    SimpleWorkflowResult, SimpleWorkflowStatus, SimpleWorkflowStep,
};

/// Prelude module for convenient imports
pub mod prelude {
    // Core types and agent system
    pub use crate::{
        AgentBuilder, AgentCapabilities, AgentConfig, AgentError, AgentFactory, AgentId,
        AgentMessage, AgentResponse, AgentType, BaseAgent, ReActAgent, Result, Tool, ToolRegistry,
        ToolResult,
    };

    // Simple workflow system
    pub use crate::{
        simple_workflow_utils, SimpleWorkflow, SimpleWorkflowBuilder, SimpleWorkflowExecutor,
        SimpleWorkflowResult, SimpleWorkflowStatus, SimpleWorkflowStep,
    };

    // Tool system
    pub use crate::tool::builtin::{EchoTool, FileTool, HttpTool, SearchTool};

    // MCP integration
    pub use crate::mcp::{server::McpServerBuilder, McpToolRegistry};
    pub use crate::{McpClient, McpServer, McpService};

    // LLM integration
    pub use crate::llm::{LlmClientFactory, LlmClientManager, LlmConfig, MessageConverter};

    // Memory management from cheungfun-core
    pub use cheungfun_core::traits::BaseMemory;
    // pub use cheungfun_query::memory::{ChatMemoryBuffer, ChatMemoryConfig};

    // Agent system components
    pub use crate::agent::{
        ActionStep, AgentContext, AgentStatus, AgentStrategy, BuiltAgent, DirectStrategy,
        FinalAnswerStep, FunctionCallingStrategy, ObservationStep, ReActConfig, ReActStats,
        ReasoningStep, ReasoningStepType, ReasoningTrace, ThoughtStep,
    };
}
