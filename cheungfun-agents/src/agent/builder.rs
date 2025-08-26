//! Agent builder for creating configured agents
//!
//! This module provides a fluent builder API for creating agents with
//! different strategies, tools, and configurations.

use super::strategy::{AgentStrategy, DirectStrategy, FunctionCallingStrategy};
use crate::{
    error::{AgentError, Result},
    tool::{Tool, ToolRegistry},
    types::{AgentMessage, AgentResponse},
};
use cheungfun_core::traits::BaseMemory;
use std::sync::Arc;
use tracing::{debug, info};

/// Agent builder for creating configured agents
#[derive(Debug)]
pub struct AgentBuilder {
    /// Agent name
    name: Option<String>,
    /// Agent description
    description: Option<String>,
    /// Tools to add to the agent
    tools: Vec<Arc<dyn Tool>>,
    /// Agent strategy
    strategy: Option<Box<dyn AgentStrategy>>,
    /// Whether to enable verbose logging
    verbose: bool,
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentBuilder {
    /// Create a new agent builder
    pub fn new() -> Self {
        Self {
            name: None,
            description: None,
            tools: Vec::new(),
            strategy: None,
            verbose: false,
        }
    }
    
    /// Set the agent name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    
    /// Set the agent description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    
    /// Add a tool to the agent
    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }
    
    /// Add multiple tools to the agent
    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }
    
    /// Set direct strategy (no tools, simple responses)
    pub fn direct_strategy(mut self) -> Self {
        let agent_name = self.name.clone().unwrap_or_else(|| "DirectAgent".to_string());
        self.strategy = Some(Box::new(DirectStrategy::new(agent_name)));
        self
    }
    
    /// Set function calling strategy (uses tools)
    pub fn function_calling_strategy(mut self) -> Self {
        let agent_name = self.name.clone().unwrap_or_else(|| "FunctionCallingAgent".to_string());
        self.strategy = Some(Box::new(FunctionCallingStrategy::new(agent_name)));
        self
    }
    
    /// Enable verbose logging
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
    
    /// Build the agent
    pub fn build(self) -> Result<BuiltAgent> {
        let name = self.name.unwrap_or_else(|| "Agent".to_string());
        let description = self.description.unwrap_or_else(|| "A Cheungfun agent".to_string());
        
        // Create tool registry
        let mut tool_registry = ToolRegistry::new();
        for tool in self.tools {
            tool_registry.register(tool)?;
        }
        let tool_registry = Arc::new(tool_registry);
        
        // Set default strategy if none provided
        let strategy = self.strategy.unwrap_or_else(|| {
            if tool_registry.tool_names().is_empty() {
                Box::new(DirectStrategy::new(name.clone()))
            } else {
                Box::new(FunctionCallingStrategy::new(name.clone()))
            }
        });
        
        if self.verbose {
            info!(
                "Built agent '{}' with strategy '{}' and {} tools",
                name,
                strategy.name(),
                tool_registry.tool_names().len()
            );
            debug!("Available tools: {:?}", tool_registry.tool_names());
        }
        
        Ok(BuiltAgent {
            name,
            description,
            strategy,
            tool_registry,
            verbose: self.verbose,
        })
    }
}

/// A built agent ready for use
#[derive(Debug)]
pub struct BuiltAgent {
    /// Agent name
    name: String,
    /// Agent description
    description: String,
    /// Execution strategy
    strategy: Box<dyn AgentStrategy>,
    /// Tool registry
    tool_registry: Arc<ToolRegistry>,
    /// Verbose logging
    verbose: bool,
}

impl BuiltAgent {
    /// Get the agent name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the agent description
    pub fn description(&self) -> &str {
        &self.description
    }
    
    /// Get the strategy
    pub fn strategy(&self) -> &dyn AgentStrategy {
        self.strategy.as_ref()
    }
    
    /// Get available tool names
    pub fn tools(&self) -> Vec<String> {
        self.tool_registry.tool_names()
    }
    
    /// Chat with the agent using memory
    pub async fn chat(
        &self,
        message: &str,
        memory: &mut dyn BaseMemory,
    ) -> Result<AgentResponse> {
        if self.verbose {
            debug!("Agent '{}' processing message: {}", self.name, message);
        }
        
        let response = self.strategy.execute(
            message,
            Some(memory),
            &self.tool_registry,
        ).await?;
        
        if self.verbose {
            debug!(
                "Agent '{}' response generated in {}ms with {} tool calls",
                self.name,
                response.stats.execution_time_ms,
                response.stats.tool_calls_count
            );
        }
        
        Ok(response)
    }
    
    /// Chat with the agent without memory
    pub async fn chat_simple(&self, message: &str) -> Result<AgentResponse> {
        if self.verbose {
            debug!("Agent '{}' processing simple message: {}", self.name, message);
        }
        
        let response = self.strategy.execute(
            message,
            None,
            &self.tool_registry,
        ).await?;
        
        if self.verbose {
            debug!(
                "Agent '{}' simple response generated in {}ms",
                self.name,
                response.stats.execution_time_ms
            );
        }
        
        Ok(response)
    }
    
    /// Check if the agent supports tools
    pub fn supports_tools(&self) -> bool {
        self.strategy.supports_tools()
    }
    
    /// Check if the agent supports memory
    pub fn supports_memory(&self) -> bool {
        self.strategy.supports_memory()
    }
    
    /// Get tool registry reference
    pub fn tool_registry(&self) -> &Arc<ToolRegistry> {
        &self.tool_registry
    }
}
