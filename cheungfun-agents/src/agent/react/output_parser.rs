//! ReAct output parser for processing LLM responses
//! 
//! This module provides parsing capabilities for ReAct agent responses,
//! extracting reasoning steps from LLM outputs.

use crate::error::{AgentError, Result};
use super::reasoning::{ReasoningStep, ActionStep, ObservationStep, ResponseStep};
use regex::Regex;
use std::collections::HashMap;

/// ReAct output parser for processing LLM responses
#[derive(Debug)]
pub struct ReActOutputParser {
    /// Regex for extracting thought-action-input patterns
    action_regex: Regex,
    /// Regex for extracting final answers
    answer_regex: Regex,
    /// Regex for extracting observations
    observation_regex: Regex,
}

impl ReActOutputParser {
    /// Create a new ReAct output parser
    pub fn new() -> Self {
        // Enhanced regex patterns for parsing ReAct output with better flexibility
        let action_regex = Regex::new(
            r"(?s)Thought:\s*(.*?)\s*Action:\s*([^\n\r]+?)\s*Action Input:\s*(\{.*?\}|[^\n\r]+?)(?:\n|$)"
        ).expect("Invalid action regex");

        let answer_regex = Regex::new(
            r"(?s)Thought:\s*(.*?)\s*(?:Final Answer|Answer):\s*(.*?)(?:\n\n|$)"
        ).expect("Invalid answer regex");

        let observation_regex = Regex::new(
            r"(?s)Observation:\s*(.*?)(?:\n\n|$)"
        ).expect("Invalid observation regex");

        Self {
            action_regex,
            answer_regex,
            observation_regex,
        }
    }
    
    /// Parse LLM output into a reasoning step
    pub async fn parse(&self, output: &str) -> Result<Box<dyn ReasoningStep>> {
        let trimmed_output = output.trim();
        
        // Try to parse as action step first
        if let Some(action_step) = self.try_parse_action(trimmed_output)? {
            return Ok(action_step);
        }
        
        // Try to parse as final answer
        if let Some(response_step) = self.try_parse_response(trimmed_output)? {
            return Ok(response_step);
        }
        
        // Try to parse as observation
        if let Some(observation_step) = self.try_parse_observation(trimmed_output)? {
            return Ok(observation_step);
        }
        
        // If no specific pattern matches, treat as a general response
        Ok(Box::new(ResponseStep::new(
            "No specific reasoning pattern detected",
            trimmed_output,
            false,
        )))
    }
    
    /// Try to parse as an action step
    fn try_parse_action(&self, output: &str) -> Result<Option<Box<dyn ReasoningStep>>> {
        if let Some(captures) = self.action_regex.captures(output) {
            let thought = captures.get(1)
                .map(|m| m.as_str().trim())
                .unwrap_or("")
                .to_string();
            
            let action = captures.get(2)
                .map(|m| m.as_str().trim())
                .unwrap_or("")
                .to_string();
            
            let action_input_str = captures.get(3)
                .map(|m| m.as_str().trim())
                .unwrap_or("{}")
                .to_string();
            
            // Try to parse action input as JSON
            let action_input = self.parse_action_input(&action_input_str)?;
            
            let action_step = ActionStep::new(thought, action, action_input);
            return Ok(Some(Box::new(action_step)));
        }
        
        Ok(None)
    }
    
    /// Try to parse as a response step
    fn try_parse_response(&self, output: &str) -> Result<Option<Box<dyn ReasoningStep>>> {
        if let Some(captures) = self.answer_regex.captures(output) {
            let thought = captures.get(1)
                .map(|m| m.as_str().trim())
                .unwrap_or("")
                .to_string();
            
            let response = captures.get(2)
                .map(|m| m.as_str().trim())
                .unwrap_or("")
                .to_string();
            
            let response_step = ResponseStep::new(thought, response, false);
            return Ok(Some(Box::new(response_step)));
        }
        
        Ok(None)
    }
    
    /// Try to parse as an observation step
    fn try_parse_observation(&self, output: &str) -> Result<Option<Box<dyn ReasoningStep>>> {
        if let Some(captures) = self.observation_regex.captures(output) {
            let observation = captures.get(1)
                .map(|m| m.as_str().trim())
                .unwrap_or("")
                .to_string();
            
            let observation_step = ObservationStep::new(observation, false);
            return Ok(Some(Box::new(observation_step)));
        }
        
        Ok(None)
    }
    
    /// Parse action input string into a HashMap
    fn parse_action_input(&self, input_str: &str) -> Result<HashMap<String, serde_json::Value>> {
        // First try to parse as JSON
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(input_str) {
            if let serde_json::Value::Object(map) = json_value {
                return Ok(map.into_iter().collect());
            }
        }
        
        // If JSON parsing fails, try to parse as key-value pairs
        let mut result = HashMap::new();
        
        // Handle simple string input
        if !input_str.contains(':') && !input_str.contains('=') {
            result.insert("input".to_string(), serde_json::Value::String(input_str.to_string()));
            return Ok(result);
        }
        
        // Try to parse key-value pairs separated by commas
        for pair in input_str.split(',') {
            let pair = pair.trim();
            if let Some((key, value)) = pair.split_once(':').or_else(|| pair.split_once('=')) {
                let key = key.trim().trim_matches('"').trim_matches('\'');
                let value = value.trim().trim_matches('"').trim_matches('\'');
                
                // Try to parse value as different types
                let parsed_value = if let Ok(num) = value.parse::<f64>() {
                    serde_json::Value::Number(serde_json::Number::from_f64(num).unwrap_or_else(|| {
                        serde_json::Number::from_f64(0.0).unwrap()
                    }))
                } else if let Ok(bool_val) = value.parse::<bool>() {
                    serde_json::Value::Bool(bool_val)
                } else {
                    serde_json::Value::String(value.to_string())
                };
                
                result.insert(key.to_string(), parsed_value);
            }
        }
        
        // If no pairs were found, treat the whole string as input
        if result.is_empty() {
            result.insert("input".to_string(), serde_json::Value::String(input_str.to_string()));
        }
        
        Ok(result)
    }
}

impl Default for ReActOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for parsing ReAct outputs
pub mod utils {
    use super::*;
    
    /// Extract tool use information from text
    pub fn extract_tool_use(input_text: &str) -> Result<(String, String, String)> {
        let pattern = r"(?s)(?:\s*Thought:\s*(.*?)|(.+))\s*Action:\s*([^\n\(\)\s]+).*?\s*Action Input:\s*.*?(\{.*\}|\S+)";
        let regex = Regex::new(pattern).map_err(|e| {
            AgentError::parsing(format!("Invalid regex pattern: {}", e))
        })?;
        
        if let Some(captures) = regex.captures(input_text) {
            let thought = captures.get(1)
                .or_else(|| captures.get(2))
                .map(|m| m.as_str().trim())
                .unwrap_or("")
                .to_string();
            
            let action = captures.get(3)
                .map(|m| m.as_str().trim())
                .unwrap_or("")
                .to_string();
            
            let action_input = captures.get(4)
                .map(|m| m.as_str().trim())
                .unwrap_or("")
                .to_string();
            
            return Ok((thought, action, action_input));
        }
        
        Err(AgentError::parsing(format!(
            "Could not extract tool use from input text: {}",
            input_text
        )))
    }
    
    /// Check if text contains a final answer
    pub fn contains_final_answer(text: &str) -> bool {
        text.contains("Final Answer:") || text.contains("Answer:")
    }
    
    /// Check if text contains an action
    pub fn contains_action(text: &str) -> bool {
        text.contains("Action:") && text.contains("Action Input:")
    }
    
    /// Check if text contains an observation
    pub fn contains_observation(text: &str) -> bool {
        text.contains("Observation:")
    }
    
    /// Clean up text by removing extra whitespace and formatting
    pub fn clean_text(text: &str) -> String {
        text.lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_parse_action_step() {
        let parser = ReActOutputParser::new();
        let output = "Thought: I need to calculate something\nAction: calculator\nAction Input: {\"expression\": \"2 + 2\"}";
        
        let result = parser.parse(output).await.unwrap();
        assert!(result.is_action());
        
        if let Some(action_step) = result.as_action() {
            assert_eq!(action_step.thought(), "I need to calculate something");
            assert_eq!(action_step.action(), "calculator");
            assert!(action_step.action_input().contains_key("expression"));
        }
    }
    
    #[tokio::test]
    async fn test_parse_response_step() {
        let parser = ReActOutputParser::new();
        let output = "Thought: I now have the answer\nFinal Answer: The result is 4";
        
        let result = parser.parse(output).await.unwrap();
        assert!(result.is_response());
        
        if let Some(response_step) = result.as_response() {
            assert_eq!(response_step.thought(), "I now have the answer");
            assert_eq!(response_step.response(), "The result is 4");
        }
    }
    
    #[test]
    fn test_parse_action_input_json() {
        let parser = ReActOutputParser::new();
        let input = r#"{"query": "test", "limit": 10}"#;
        
        let result = parser.parse_action_input(input).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("query"));
        assert!(result.contains_key("limit"));
    }
    
    #[test]
    fn test_parse_action_input_simple() {
        let parser = ReActOutputParser::new();
        let input = "simple string input";
        
        let result = parser.parse_action_input(input).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains_key("input"));
    }
}
