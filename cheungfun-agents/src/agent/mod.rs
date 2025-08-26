//! Agent system for Cheungfun
//!
//! This module provides a complete agent framework inspired by LlamaIndex,
//! with support for various reasoning patterns and tool integration.

pub mod base;
pub mod builder;
pub mod react;
pub mod strategy;

// Re-export core agent functionality
pub use base::{
    utils::{default_capabilities_for, generate_agent_id, validate_config},
    AgentBuilder as BaseAgentBuilder, AgentContext, AgentMessageExt, AgentStatus, BaseAgent,
};

// Re-export new builder system
pub use builder::{AgentBuilder, BuiltAgent};
pub use strategy::{AgentStrategy, DirectStrategy, FunctionCallingStrategy};

// Re-export ReAct agent
pub use react::{
    prelude as react_prelude, ActionStep, FinalAnswerStep, ObservationStep, ReActAgent,
    ReActConfig, ReActStats, ReasoningStep, ReasoningStepType, ReasoningTrace, ThoughtStep,
};

/// Agent types enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentType {
    /// ReAct reasoning agent
    ReAct,
    /// Function calling agent
    FunctionCalling,
    /// Custom workflow agent
    Workflow,
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentType::ReAct => write!(f, "react"),
            AgentType::FunctionCalling => write!(f, "function_calling"),
            AgentType::Workflow => write!(f, "workflow"),
        }
    }
}

impl std::str::FromStr for AgentType {
    type Err = crate::error::AgentError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "react" => Ok(AgentType::ReAct),
            "function_calling" => Ok(AgentType::FunctionCalling),
            "workflow" => Ok(AgentType::Workflow),
            _ => Err(crate::error::AgentError::invalid_configuration(format!(
                "Unknown agent type: {}",
                s
            ))),
        }
    }
}

/// Agent factory for creating different types of agents
pub struct AgentFactory;

impl AgentFactory {
    /// Create a new ReAct agent
    pub fn create_react_agent(
        config: crate::types::AgentConfig,
        tools: std::sync::Arc<crate::tool::ToolRegistry>,
    ) -> crate::error::Result<ReActAgent> {
        base::utils::validate_config(&config)?;

        let react_config = ReActConfig {
            base_config: config,
            max_iterations: 10,
            max_thinking_time_ms: 5_000,
            include_trace: true,
            custom_settings: std::collections::HashMap::new(),
        };

        Ok(ReActAgent::new(react_config, tools))
    }

    /// Create an agent from agent type and configuration
    pub fn create_agent(
        agent_type: AgentType,
        config: crate::types::AgentConfig,
        tools: std::sync::Arc<crate::tool::ToolRegistry>,
    ) -> crate::error::Result<Box<dyn BaseAgent>> {
        match agent_type {
            AgentType::ReAct => {
                let agent = Self::create_react_agent(config, tools)?;
                Ok(Box::new(agent))
            }
            AgentType::FunctionCalling => {
                // TODO: Implement function calling agent
                Err(crate::error::AgentError::not_implemented(
                    "Function calling agent not yet implemented",
                ))
            }
            AgentType::Workflow => {
                // TODO: Implement workflow agent
                Err(crate::error::AgentError::not_implemented(
                    "Workflow agent not yet implemented",
                ))
            }
        }
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use super::{
        default_capabilities_for, generate_agent_id, validate_config,
        AgentBuilder, BuiltAgent, AgentContext, AgentFactory, AgentStatus, AgentType,
        BaseAgent, ReActAgent, ReActConfig, ReActStats,
        AgentStrategy, DirectStrategy, FunctionCallingStrategy,
    };
    pub use crate::types::{AgentCapabilities, AgentConfig};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_type_parsing() {
        assert_eq!("react".parse::<AgentType>().unwrap(), AgentType::ReAct);
        assert_eq!(
            "function_calling".parse::<AgentType>().unwrap(),
            AgentType::FunctionCalling
        );
        assert_eq!(
            "workflow".parse::<AgentType>().unwrap(),
            AgentType::Workflow
        );

        assert!("invalid".parse::<AgentType>().is_err());
    }

    #[test]
    fn test_agent_config_builder() {
        let config = AgentBuilder::new("test_agent")
            .description("A test agent")
            .instructions("You are a helpful assistant")
            .with_tools()
            .with_streaming()
            .max_tool_calls(5)
            .build();

        assert_eq!(config.name, "test_agent");
        assert_eq!(config.description, Some("A test agent".to_string()));
        assert_eq!(
            config.instructions,
            Some("You are a helpful assistant".to_string())
        );
        assert!(config.capabilities.supports_tools);
        assert!(config.capabilities.supports_streaming);
        assert_eq!(config.max_tool_calls, Some(5));
    }

    #[test]
    fn test_config_validation() {
        use crate::types::AgentConfig;

        let valid_config = AgentConfig::default();
        assert!(base::utils::validate_config(&valid_config).is_ok());

        let mut invalid_config = AgentConfig::default();
        invalid_config.name = "".to_string();
        assert!(base::utils::validate_config(&invalid_config).is_err());
    }
}
