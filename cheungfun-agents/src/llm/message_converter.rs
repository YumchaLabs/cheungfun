//! Message conversion utilities between cheungfun and siumai formats

use cheungfun_core::{ChatMessage as CheungfunMessage, MessageRole};
use siumai::types::ChatMessage as SiumaiMessage;

/// Utility for converting messages between different formats
pub struct MessageConverter;

impl MessageConverter {
    /// Convert cheungfun `ChatMessage` to siumai `ChatMessage`
    #[must_use]
    pub fn to_siumai_messages(messages: &[CheungfunMessage]) -> Vec<SiumaiMessage> {
        messages.iter().map(Self::convert_single_message).collect()
    }

    /// Convert a single cheungfun message to siumai format
    #[must_use]
    pub fn convert_single_message(message: &CheungfunMessage) -> SiumaiMessage {
        match message.role {
            MessageRole::User => SiumaiMessage::user(&message.content).build(),
            MessageRole::Assistant => SiumaiMessage::assistant(&message.content).build(),
            MessageRole::System => SiumaiMessage::system(&message.content).build(),
            MessageRole::Tool => {
                // Treat tool messages as assistant messages for now
                SiumaiMessage::assistant(&message.content).build()
            }
        }
    }

    /// Convert siumai `ChatMessage` to cheungfun `ChatMessage`
    #[must_use]
    pub fn from_siumai_messages(messages: &[SiumaiMessage]) -> Vec<CheungfunMessage> {
        messages.iter().map(Self::convert_from_siumai).collect()
    }

    /// Convert a single siumai message to cheungfun format
    #[must_use]
    pub fn convert_from_siumai(message: &SiumaiMessage) -> CheungfunMessage {
        let role = match message.role {
            siumai::types::MessageRole::User => MessageRole::User,
            siumai::types::MessageRole::Assistant => MessageRole::Assistant,
            siumai::types::MessageRole::System => MessageRole::System,
            siumai::types::MessageRole::Tool => MessageRole::Tool,
            siumai::types::MessageRole::Developer => MessageRole::System, // Treat developer as system
        };

        CheungfunMessage {
            role,
            content: message.content.all_text(),
            metadata: Some(std::collections::HashMap::new()),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Create a system message for agent instructions
    #[must_use]
    pub fn create_system_message(instructions: &str) -> SiumaiMessage {
        SiumaiMessage::system(instructions).build()
    }

    /// Create a user message
    #[must_use]
    pub fn create_user_message(content: &str) -> SiumaiMessage {
        SiumaiMessage::user(content).build()
    }

    /// Create an assistant message
    #[must_use]
    pub fn create_assistant_message(content: &str) -> SiumaiMessage {
        SiumaiMessage::assistant(content).build()
    }

    /// Build conversation context from memory
    #[must_use]
    pub fn build_conversation_context(
        system_prompt: Option<&str>,
        memory_messages: &[CheungfunMessage],
        current_input: &str,
    ) -> Vec<SiumaiMessage> {
        let mut messages = Vec::new();

        // Add system prompt if provided
        if let Some(prompt) = system_prompt {
            messages.push(Self::create_system_message(prompt));
        }

        // Add conversation history
        messages.extend(Self::to_siumai_messages(memory_messages));

        // Add current user input
        messages.push(Self::create_user_message(current_input));

        messages
    }

    /// Extract tool calls from assistant message content
    #[must_use]
    pub fn extract_tool_calls(content: &str) -> Vec<ToolCallExtraction> {
        let mut tool_calls = Vec::new();

        // Simple pattern matching for tool calls
        // This is a basic implementation - in practice, you might want more sophisticated parsing
        if content.contains("Action:") {
            let lines: Vec<&str> = content.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                if line.trim().starts_with("Action:") {
                    if let Some(tool_name) = Self::extract_tool_name(line) {
                        let args = if i + 1 < lines.len() {
                            Self::extract_tool_args(lines[i + 1])
                        } else {
                            serde_json::Value::Null
                        };

                        tool_calls.push(ToolCallExtraction {
                            tool_name,
                            arguments: args,
                            raw_text: (*line).to_string(),
                        });
                    }
                }
            }
        }

        tool_calls
    }

    /// Extract tool name from action line
    fn extract_tool_name(line: &str) -> Option<String> {
        // Pattern: "Action: tool_name"
        if let Some(action_part) = line.strip_prefix("Action:") {
            let tool_name = action_part.trim();
            if !tool_name.is_empty() {
                return Some(tool_name.to_string());
            }
        }
        None
    }

    /// Extract tool arguments from action input line
    fn extract_tool_args(line: &str) -> serde_json::Value {
        // Pattern: "Action Input: {json}" or "Action Input: simple_string"
        if let Some(input_part) = line.strip_prefix("Action Input:") {
            let input_str = input_part.trim();

            // Try to parse as JSON first
            if let Ok(json_value) = serde_json::from_str(input_str) {
                return json_value;
            }

            // If not JSON, treat as simple string
            return serde_json::json!({ "input": input_str });
        }

        serde_json::Value::Null
    }

    /// Format tool result for inclusion in conversation
    #[must_use]
    pub fn format_tool_result(tool_name: &str, result: &str) -> String {
        format!("Observation: Tool '{tool_name}' returned: {result}")
    }

    /// Create a ReAct-style thought message
    #[must_use]
    pub fn create_thought_message(thought: &str) -> String {
        format!("Thought: {thought}")
    }

    /// Create a ReAct-style action message
    #[must_use]
    pub fn create_action_message(tool_name: &str, args: &serde_json::Value) -> String {
        format!("Action: {tool_name}\nAction Input: {args}")
    }

    /// Parse ReAct-style response into components
    #[must_use]
    pub fn parse_react_response(content: &str) -> ReActParsedResponse {
        let mut thoughts = Vec::new();
        let mut actions = Vec::new();
        let mut observations = Vec::new();
        let mut final_answer = None;

        let lines: Vec<&str> = content.lines().collect();
        for line in lines {
            let trimmed = line.trim();

            if trimmed.starts_with("Thought:") {
                thoughts.push(
                    trimmed
                        .strip_prefix("Thought:")
                        .unwrap_or("")
                        .trim()
                        .to_string(),
                );
            } else if trimmed.starts_with("Action:") {
                actions.push(
                    trimmed
                        .strip_prefix("Action:")
                        .unwrap_or("")
                        .trim()
                        .to_string(),
                );
            } else if trimmed.starts_with("Observation:") {
                observations.push(
                    trimmed
                        .strip_prefix("Observation:")
                        .unwrap_or("")
                        .trim()
                        .to_string(),
                );
            } else if trimmed.starts_with("Final Answer:") || trimmed.starts_with("最终答案:") {
                final_answer = Some(trimmed.split(':').nth(1).unwrap_or("").trim().to_string());
            }
        }

        ReActParsedResponse {
            thoughts,
            actions,
            observations,
            final_answer,
            raw_content: content.to_string(),
        }
    }
}

/// Extracted tool call information
#[derive(Debug, Clone)]
pub struct ToolCallExtraction {
    /// Name of the tool to call
    pub tool_name: String,
    /// Arguments for the tool call
    pub arguments: serde_json::Value,
    /// Raw text that was parsed
    pub raw_text: String,
}

/// Parsed `ReAct` response components
#[derive(Debug, Clone)]
pub struct ReActParsedResponse {
    /// Extracted thought steps
    pub thoughts: Vec<String>,
    /// Extracted action steps
    pub actions: Vec<String>,
    /// Extracted observation steps
    pub observations: Vec<String>,
    /// Final answer if present
    pub final_answer: Option<String>,
    /// Original raw content
    pub raw_content: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_convert_to_siumai_messages() {
        let cheungfun_messages = vec![
            CheungfunMessage {
                role: MessageRole::System,
                content: "You are a helpful assistant.".to_string(),
                metadata: std::collections::HashMap::new(),
                timestamp: Utc::now(),
            },
            CheungfunMessage {
                role: MessageRole::User,
                content: "Hello!".to_string(),
                metadata: std::collections::HashMap::new(),
                timestamp: Utc::now(),
            },
        ];

        let siumai_messages = MessageConverter::to_siumai_messages(&cheungfun_messages);

        assert_eq!(siumai_messages.len(), 2);
        assert_eq!(siumai_messages[0].role, "system");
        assert_eq!(siumai_messages[0].content, "You are a helpful assistant.");
        assert_eq!(siumai_messages[1].role, "user");
        assert_eq!(siumai_messages[1].content, "Hello!");
    }

    #[test]
    fn test_extract_tool_calls() {
        let content = "Thought: I need to calculate something.\nAction: calculator\nAction Input: {\"expression\": \"2 + 2\"}";

        let tool_calls = MessageConverter::extract_tool_calls(content);

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].tool_name, "calculator");
        assert_eq!(tool_calls[0].arguments["expression"], "2 + 2");
    }

    #[test]
    fn test_parse_react_response() {
        let content = "Thought: I need to help the user.\nAction: search\nObservation: Found some results.\nFinal Answer: Here is the answer.";

        let parsed = MessageConverter::parse_react_response(content);

        assert_eq!(parsed.thoughts.len(), 1);
        assert_eq!(parsed.actions.len(), 1);
        assert_eq!(parsed.observations.len(), 1);
        assert_eq!(parsed.final_answer, Some("Here is the answer.".to_string()));
    }

    #[test]
    fn test_build_conversation_context() {
        let system_prompt = "You are a helpful assistant.";
        let memory_messages = vec![CheungfunMessage {
            role: MessageRole::User,
            content: "Previous question".to_string(),
            metadata: std::collections::HashMap::new(),
            timestamp: Utc::now(),
        }];
        let current_input = "Current question";

        let context = MessageConverter::build_conversation_context(
            Some(system_prompt),
            &memory_messages,
            current_input,
        );

        assert_eq!(context.len(), 3);
        assert_eq!(context[0].role, "system");
        assert_eq!(context[1].role, "user");
        assert_eq!(context[1].content, "Previous question");
        assert_eq!(context[2].role, "user");
        assert_eq!(context[2].content, "Current question");
    }
}
