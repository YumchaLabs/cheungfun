//! Agent builder for convenient agent construction.

use crate::{
    agent::{Agent, BasicAgent},
    error::{AgentError, Result},
    tool::{Tool, ToolRegistry},
    types::{AgentCapabilities, AgentConfig},
};
use std::{collections::HashMap, sync::Arc};

/// Builder for creating agents with fluent API
#[derive(Debug)]
pub struct AgentBuilder {
    name: Option<String>,
    description: Option<String>,
    instructions: Option<String>,
    capabilities: AgentCapabilities,
    max_execution_time_ms: Option<u64>,
    max_tool_calls: Option<usize>,
    verbose: bool,
    custom_config: HashMap<String, serde_json::Value>,
    tools: Vec<Arc<dyn Tool>>,
    tool_registry: Option<Arc<ToolRegistry>>,
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
            instructions: None,
            capabilities: AgentCapabilities::default(),
            max_execution_time_ms: None,
            max_tool_calls: None,
            verbose: false,
            custom_config: HashMap::new(),
            tools: Vec::new(),
            tool_registry: None,
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

    /// Set the agent instructions/system prompt
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Set agent capabilities
    pub fn capabilities(mut self, capabilities: AgentCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Enable tool support
    pub fn with_tools(mut self) -> Self {
        self.capabilities.supports_tools = true;
        self
    }

    /// Enable streaming support
    pub fn with_streaming(mut self) -> Self {
        self.capabilities.supports_streaming = true;
        self
    }

    /// Enable conversation support
    pub fn with_conversation(mut self) -> Self {
        self.capabilities.supports_conversation = true;
        self
    }

    /// Enable file operations
    pub fn with_files(mut self) -> Self {
        self.capabilities.supports_files = true;
        self
    }

    /// Enable web access
    pub fn with_web(mut self) -> Self {
        self.capabilities.supports_web = true;
        self
    }

    /// Enable code execution
    pub fn with_code_execution(mut self) -> Self {
        self.capabilities.supports_code_execution = true;
        self
    }

    /// Set maximum context length
    pub fn max_context_length(mut self, length: usize) -> Self {
        self.capabilities.max_context_length = Some(length);
        self
    }

    /// Add supported input format
    pub fn input_format(mut self, format: impl Into<String>) -> Self {
        self.capabilities
            .supported_input_formats
            .push(format.into());
        self
    }

    /// Add supported output format
    pub fn output_format(mut self, format: impl Into<String>) -> Self {
        self.capabilities
            .supported_output_formats
            .push(format.into());
        self
    }

    /// Add custom capability
    pub fn custom_capability(mut self, name: impl Into<String>, value: serde_json::Value) -> Self {
        self.capabilities
            .custom_capabilities
            .insert(name.into(), value);
        self
    }

    /// Set maximum execution time
    pub fn max_execution_time_ms(mut self, ms: u64) -> Self {
        self.max_execution_time_ms = Some(ms);
        self
    }

    /// Set maximum tool calls per task
    pub fn max_tool_calls(mut self, count: usize) -> Self {
        self.max_tool_calls = Some(count);
        self
    }

    /// Enable verbose logging
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Add custom configuration
    pub fn config(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.custom_config.insert(key.into(), value);
        self
    }

    /// Add a single tool
    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self.capabilities.supports_tools = true;
        self
    }

    /// Add multiple tools
    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        if !self.tools.is_empty() {
            self.capabilities.supports_tools = true;
        }
        self
    }

    /// Use an existing tool registry
    pub fn tool_registry(mut self, registry: Arc<ToolRegistry>) -> Self {
        self.tool_registry = Some(registry);
        self.capabilities.supports_tools = true;
        self
    }

    /// Build the agent
    pub fn build(self) -> Result<Box<dyn Agent>> {
        let name = self.name.unwrap_or_else(|| "unnamed_agent".to_string());

        // Create or use tool registry
        let tool_registry = if let Some(registry) = self.tool_registry {
            registry
        } else {
            let mut registry = ToolRegistry::new();

            // Register provided tools
            for tool in self.tools {
                registry.register(tool).map_err(|e| {
                    AgentError::configuration(format!("Failed to register tool: {e}"))
                })?;
            }

            Arc::new(registry)
        };

        // Validate tool registry if tools are enabled
        if self.capabilities.supports_tools {
            tool_registry.validate_all().map_err(|e| {
                AgentError::configuration(format!("Tool registry validation failed: {e}"))
            })?;
        }

        let config = AgentConfig {
            name,
            description: self.description.clone(),
            instructions: self.instructions.clone(),
            capabilities: self.capabilities.clone(),
            max_execution_time_ms: self.max_execution_time_ms,
            max_tool_calls: self.max_tool_calls,
            verbose: self.verbose,
            custom_config: self.custom_config.clone(),
        };

        // Validate configuration before moving self
        if config.name.is_empty() {
            return Err(AgentError::validation("name", "Agent name cannot be empty"));
        }
        if let Some(max_time) = config.max_execution_time_ms {
            if max_time == 0 {
                return Err(AgentError::validation(
                    "max_execution_time_ms",
                    "Maximum execution time must be greater than 0",
                ));
            }
        }

        let agent = BasicAgent::new(config, tool_registry);
        Ok(Box::new(agent))
    }

    /// Validate the agent configuration
    fn validate_config(&self, config: &AgentConfig) -> Result<()> {
        // Validate name
        if config.name.is_empty() {
            return Err(AgentError::validation("name", "Agent name cannot be empty"));
        }

        // Validate execution time
        if let Some(max_time) = config.max_execution_time_ms {
            if max_time == 0 {
                return Err(AgentError::validation(
                    "max_execution_time_ms",
                    "Maximum execution time must be greater than 0",
                ));
            }
        }

        // Validate tool calls limit
        if let Some(max_calls) = config.max_tool_calls {
            if max_calls == 0 {
                return Err(AgentError::validation(
                    "max_tool_calls",
                    "Maximum tool calls must be greater than 0",
                ));
            }
        }

        // Validate context length
        if let Some(max_context) = config.capabilities.max_context_length {
            if max_context == 0 {
                return Err(AgentError::validation(
                    "max_context_length",
                    "Maximum context length must be greater than 0",
                ));
            }
        }

        Ok(())
    }
}

/// Convenience functions for common agent configurations
impl AgentBuilder {
    /// Create a basic assistant agent
    pub fn assistant() -> Self {
        Self::new()
            .name("assistant")
            .description("A helpful AI assistant")
            .with_conversation()
            .with_tools()
    }

    /// Create a research agent with web and search capabilities
    pub fn researcher() -> Self {
        Self::new()
            .name("researcher")
            .description("An AI agent specialized in research and information gathering")
            .with_conversation()
            .with_tools()
            .with_web()
            .max_context_length(8192)
    }

    /// Create a file management agent
    pub fn file_manager() -> Self {
        Self::new()
            .name("file_manager")
            .description("An AI agent for file operations and management")
            .with_tools()
            .with_files()
    }

    /// Create a code assistant agent
    pub fn code_assistant() -> Self {
        Self::new()
            .name("code_assistant")
            .description("An AI agent for code analysis and assistance")
            .with_conversation()
            .with_tools()
            .with_files()
            .with_code_execution()
            .max_context_length(16384)
    }

    /// Create a web agent for web scraping and API calls
    pub fn web_agent() -> Self {
        Self::new()
            .name("web_agent")
            .description("An AI agent for web interactions and API calls")
            .with_tools()
            .with_web()
            .max_execution_time_ms(60_000) // 1 minute for web requests
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::builtin::EchoTool;

    #[test]
    fn test_agent_builder_basic() {
        let agent = AgentBuilder::new()
            .name("test_agent")
            .description("A test agent")
            .build()
            .unwrap();

        assert_eq!(agent.name(), "test_agent");
        assert_eq!(agent.description(), Some("A test agent"));
    }

    #[test]
    fn test_agent_builder_with_tools() {
        let echo_tool = Arc::new(EchoTool::new());

        let agent = AgentBuilder::new()
            .name("tool_agent")
            .tool(echo_tool)
            .build()
            .unwrap();

        assert!(agent.supports_capability("tools"));
        assert!(agent.tools().contains(&"echo".to_string()));
    }

    #[test]
    fn test_agent_builder_presets() {
        let assistant = AgentBuilder::assistant().build().unwrap();
        assert_eq!(assistant.name(), "assistant");
        assert!(assistant.supports_capability("conversation"));
        assert!(assistant.supports_capability("tools"));

        let researcher = AgentBuilder::researcher().build().unwrap();
        assert_eq!(researcher.name(), "researcher");
        assert!(researcher.supports_capability("web"));
    }

    #[test]
    fn test_agent_builder_validation() {
        // Test empty name validation
        let result = AgentBuilder::new().name("").build();
        assert!(result.is_err());

        // Test zero execution time validation
        let result = AgentBuilder::new()
            .name("test")
            .max_execution_time_ms(0)
            .build();
        assert!(result.is_err());
    }
}
