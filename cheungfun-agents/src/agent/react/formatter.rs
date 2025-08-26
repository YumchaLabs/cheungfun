//! ReAct chat formatter for preparing LLM inputs
//! 
//! This module provides formatting capabilities for ReAct agents,
//! converting chat history and reasoning steps into proper LLM prompts.

use crate::tool::Tool;
use super::reasoning::ReasoningStep;
use cheungfun_core::ChatMessage;
use std::sync::Arc;

/// Default ReAct system header template
const DEFAULT_REACT_HEADER: &str = r#"You are a helpful AI assistant that can use tools to answer questions and solve problems. You have access to the following tools:

{tool_descriptions}

Use the following format for your responses:

Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action as a JSON object
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important guidelines:
- Always start with a Thought before taking an Action
- Action Input must be valid JSON
- If you don't need tools, go directly to Final Answer
- Be concise but thorough in your reasoning
- If a tool fails, try a different approach

Begin!"#;

/// ReAct chat formatter for preparing LLM inputs
#[derive(Debug, Clone)]
pub struct ReActFormatter {
    /// System header template
    system_header: String,
    /// Role for observation messages
    observation_role: String,
}

impl ReActFormatter {
    /// Create a new ReAct formatter
    pub fn new() -> Self {
        Self {
            system_header: DEFAULT_REACT_HEADER.to_string(),
            observation_role: "user".to_string(),
        }
    }
    
    /// Create a formatter with custom system header
    pub fn with_system_header(system_header: impl Into<String>) -> Self {
        Self {
            system_header: system_header.into(),
            observation_role: "user".to_string(),
        }
    }
    
    /// Set the system header template
    pub fn set_system_header(&mut self, system_header: impl Into<String>) {
        self.system_header = system_header.into();
    }
    
    /// Get the system header template
    pub fn system_header(&self) -> &str {
        &self.system_header
    }
    
    /// Set the observation role
    pub fn set_observation_role(&mut self, role: impl Into<String>) {
        self.observation_role = role.into();
    }
    
    /// Format chat history and reasoning steps for LLM input
    pub fn format(
        &self,
        tools: &[Arc<dyn Tool>],
        chat_history: &[ChatMessage],
        current_reasoning: &[Box<dyn ReasoningStep>],
    ) -> Vec<ChatMessage> {
        let mut messages = Vec::new();
        
        // Add system message with tool descriptions
        let system_message = self.format_system_message(tools);
        messages.push(system_message);
        
        // Add chat history
        messages.extend_from_slice(chat_history);
        
        // Add current reasoning steps as alternating messages
        for step in current_reasoning {
            let message = self.format_reasoning_step(step);
            messages.push(message);
        }
        
        messages
    }
    
    /// Format the system message with tool descriptions
    fn format_system_message(&self, tools: &[Arc<dyn Tool>]) -> ChatMessage {
        use cheungfun_core::MessageRole;

        let tool_descriptions = self.get_tool_descriptions(tools);
        let tool_names = tools.iter()
            .map(|tool| tool.name())
            .collect::<Vec<_>>()
            .join(", ");

        let formatted_header = self.system_header
            .replace("{tool_descriptions}", &tool_descriptions)
            .replace("{tool_names}", &tool_names);

        ChatMessage {
            role: MessageRole::System,
            content: formatted_header,
            metadata: None,
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Format a reasoning step as a chat message
    fn format_reasoning_step(&self, step: &Box<dyn ReasoningStep>) -> ChatMessage {
        use cheungfun_core::MessageRole;

        let role = if step.is_observation() {
            match self.observation_role.as_str() {
                "user" => MessageRole::User,
                "assistant" => MessageRole::Assistant,
                "system" => MessageRole::System,
                "tool" => MessageRole::Tool,
                _ => MessageRole::User,
            }
        } else {
            MessageRole::Assistant
        };

        ChatMessage {
            role,
            content: step.get_content(),
            metadata: None,
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Get formatted tool descriptions with enhanced formatting
    fn get_tool_descriptions(&self, tools: &[Arc<dyn Tool>]) -> String {
        if tools.is_empty() {
            return "No tools available.".to_string();
        }

        tools.iter()
            .map(|tool| {
                let schema = tool.schema();
                let mut description = format!("- **{}**: {}", tool.name(), tool.description());

                // Add input schema information if available
                if let Some(properties) = schema.input_schema.get("properties") {
                    if let Some(props_obj) = properties.as_object() {
                        let params: Vec<String> = props_obj.keys()
                            .map(|key| format!("`{}`", key))
                            .collect();
                        if !params.is_empty() {
                            description.push_str(&format!(" (Parameters: {})", params.join(", ")));
                        }
                    }
                }

                // Add danger warning if tool is marked as dangerous
                if schema.dangerous {
                    description.push_str(" ⚠️ **DANGEROUS** - Use with caution");
                }

                description
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for ReActFormatter {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for ReAct formatting
pub mod utils {
    use super::*;
    
    /// Extract tool names from a list of tools
    pub fn extract_tool_names(tools: &[Arc<dyn Tool>]) -> Vec<String> {
        tools.iter().map(|tool| tool.name()).collect()
    }
    
    /// Create a simple tool description
    pub fn create_tool_description(tool: &dyn Tool) -> String {
        format!("{}: {}", tool.name(), tool.description())
    }
    
    /// Format multiple tool descriptions
    pub fn format_tool_descriptions(tools: &[Arc<dyn Tool>]) -> String {
        tools.iter()
            .map(|tool| create_tool_description(tool.as_ref()))
            .collect::<Vec<_>>()
            .join("\n")
    }
    
    /// Create a chat message with the given role and content
    pub fn create_chat_message(role: impl Into<String>, content: impl Into<String>) -> ChatMessage {
        use cheungfun_core::MessageRole;

        let role_str = role.into();
        let message_role = match role_str.as_str() {
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            "system" => MessageRole::System,
            "tool" => MessageRole::Tool,
            _ => MessageRole::User,
        };

        ChatMessage {
            role: message_role,
            content: content.into(),
            metadata: None,
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Validate that a message has the required fields
    pub fn validate_message(message: &ChatMessage) -> bool {
        // MessageRole is an enum, so we just check if content is not empty
        !message.content.is_empty()
    }
}

/// Template manager for ReAct prompts
#[derive(Debug, Clone)]
pub struct ReActTemplateManager {
    /// Available templates
    templates: std::collections::HashMap<String, String>,
}

impl ReActTemplateManager {
    /// Create a new template manager
    pub fn new() -> Self {
        let mut templates = std::collections::HashMap::new();
        templates.insert("default".to_string(), DEFAULT_REACT_HEADER.to_string());
        
        Self { templates }
    }
    
    /// Add a custom template
    pub fn add_template(&mut self, name: impl Into<String>, template: impl Into<String>) {
        self.templates.insert(name.into(), template.into());
    }
    
    /// Get a template by name
    pub fn get_template(&self, name: &str) -> Option<&String> {
        self.templates.get(name)
    }
    
    /// List available template names
    pub fn list_templates(&self) -> Vec<String> {
        self.templates.keys().cloned().collect()
    }
    
    /// Create a formatter with the specified template
    pub fn create_formatter(&self, template_name: &str) -> Option<ReActFormatter> {
        self.get_template(template_name)
            .map(|template| ReActFormatter::with_system_header(template.clone()))
    }
}

impl Default for ReActTemplateManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolContext;
    use crate::types::{ToolSchema, ToolOutput};
    use async_trait::async_trait;
    
    #[derive(Debug)]
    struct MockTool {
        name: String,
        description: String,
    }
    
    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> String {
            self.name.clone()
        }
        
        fn description(&self) -> String {
            self.description.clone()
        }
        
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: self.name.clone(),
                description: self.description.clone(),
                input_schema: serde_json::json!({}),
                output_schema: None,
                dangerous: false,
                metadata: std::collections::HashMap::new(),
            }
        }
        
        async fn execute(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, crate::error::AgentError> {
            Ok(ToolOutput::success(
                uuid::Uuid::new_v4(),
                "mock_tool",
                "mock result"
            ))
        }
    }
    
    #[test]
    fn test_formatter_creation() {
        let formatter = ReActFormatter::new();
        assert!(formatter.system_header().contains("You are a helpful AI assistant"));
    }
    
    #[test]
    fn test_tool_descriptions() {
        let tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(MockTool {
                name: "calculator".to_string(),
                description: "Perform calculations".to_string(),
            }),
            Arc::new(MockTool {
                name: "search".to_string(),
                description: "Search the web".to_string(),
            }),
        ];
        
        let formatter = ReActFormatter::new();
        let descriptions = formatter.get_tool_descriptions(&tools);
        
        assert!(descriptions.contains("calculator: Perform calculations"));
        assert!(descriptions.contains("search: Search the web"));
    }
}
