//! Tool selection system for ReAct agents.
//!
//! This module provides the `ToolSelector` which analyzes reasoning text
//! and selects the most appropriate tool for execution.

use crate::{
    error::{AgentError, Result},
    tool::Tool,
};
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, info};

/// Strategy for tool selection
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionStrategy {
    /// Exact name matching
    ExactMatch,
    /// Fuzzy matching based on keywords
    FuzzyMatch,
    /// Semantic similarity (requires embeddings)
    Semantic,
    /// Hybrid approach combining multiple strategies
    Hybrid,
}

/// Configuration for tool selection
#[derive(Debug, Clone)]
pub struct ToolSelectorConfig {
    /// Selection strategy to use
    pub strategy: SelectionStrategy,
    /// Minimum confidence threshold for selection (0.0 to 1.0)
    pub confidence_threshold: f32,
    /// Whether to use tool descriptions in matching
    pub use_descriptions: bool,
    /// Maximum number of candidate tools to consider
    pub max_candidates: usize,
}

impl Default for ToolSelectorConfig {
    fn default() -> Self {
        Self {
            strategy: SelectionStrategy::Hybrid,
            confidence_threshold: 0.7,
            use_descriptions: true,
            max_candidates: 5,
        }
    }
}

/// Tool selection result with confidence score
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Selected tool
    pub tool: Arc<dyn Tool>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Reasoning for the selection
    pub reasoning: String,
}

/// Tool selector for ReAct agents
#[derive(Debug)]
pub struct ToolSelector {
    /// Configuration for selection
    config: ToolSelectorConfig,
    /// Available tools
    tools: Vec<Arc<dyn Tool>>,
    /// Keyword mappings for fuzzy matching
    keyword_mappings: HashMap<String, Vec<String>>,
}

impl ToolSelector {
    /// Create a new tool selector with default configuration
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        let mut selector = Self {
            config: ToolSelectorConfig::default(),
            tools: tools.clone(),
            keyword_mappings: HashMap::new(),
        };
        
        // Build keyword mappings
        selector.build_keyword_mappings();
        selector
    }
    
    /// Create a tool selector with custom configuration
    pub fn with_config(tools: Vec<Arc<dyn Tool>>, config: ToolSelectorConfig) -> Self {
        let mut selector = Self {
            config,
            tools: tools.clone(),
            keyword_mappings: HashMap::new(),
        };
        
        selector.build_keyword_mappings();
        selector
    }
    
    /// Select the best tool based on reasoning text
    pub fn select_tool(&self, reasoning: &str) -> Option<Arc<dyn Tool>> {
        debug!("Selecting tool for reasoning: {}", reasoning);
        
        let result = match self.config.strategy {
            SelectionStrategy::ExactMatch => self.exact_match_selection(reasoning),
            SelectionStrategy::FuzzyMatch => self.fuzzy_match_selection(reasoning),
            SelectionStrategy::Semantic => self.semantic_selection(reasoning),
            SelectionStrategy::Hybrid => self.hybrid_selection(reasoning),
        };
        
        match result {
            Ok(Some(selection)) => {
                if selection.confidence >= self.config.confidence_threshold {
                    info!(
                        "Selected tool '{}' with confidence {:.2} - {}",
                        selection.tool.name(),
                        selection.confidence,
                        selection.reasoning
                    );
                    Some(selection.tool)
                } else {
                    debug!(
                        "Tool selection confidence {:.2} below threshold {:.2}",
                        selection.confidence,
                        self.config.confidence_threshold
                    );
                    None
                }
            }
            Ok(None) => {
                debug!("No suitable tool found for reasoning");
                None
            }
            Err(e) => {
                debug!("Tool selection error: {}", e);
                None
            }
        }
    }
    
    /// Exact match selection - looks for exact tool names in reasoning
    fn exact_match_selection(&self, reasoning: &str) -> Result<Option<SelectionResult>> {
        let reasoning_lower = reasoning.to_lowercase();
        
        for tool in &self.tools {
            let tool_name_lower = tool.name().to_lowercase();
            if reasoning_lower.contains(&tool_name_lower) {
                return Ok(Some(SelectionResult {
                    tool: tool.clone(),
                    confidence: 1.0,
                    reasoning: format!("Exact match found for tool name '{}'", tool.name()),
                }));
            }
        }
        
        Ok(None)
    }
    
    /// Fuzzy match selection - uses keywords and partial matching
    fn fuzzy_match_selection(&self, reasoning: &str) -> Result<Option<SelectionResult>> {
        let reasoning_lower = reasoning.to_lowercase();
        let mut best_match: Option<SelectionResult> = None;
        
        for tool in &self.tools {
            let mut score = 0.0;
            let mut matched_keywords = Vec::new();
            
            // Check tool name
            if reasoning_lower.contains(&tool.name().to_lowercase()) {
                score += 0.5;
                matched_keywords.push(tool.name().to_string());
            }
            
            // Check keywords
            if let Some(keywords) = self.keyword_mappings.get(&tool.name().to_string()) {
                for keyword in keywords {
                    if reasoning_lower.contains(&keyword.to_lowercase()) {
                        score += 0.3 / keywords.len() as f32;
                        matched_keywords.push(keyword.clone());
                    }
                }
            }
            
            // Check description if enabled
            if self.config.use_descriptions {
                let description = tool.description();
                let description_words: Vec<&str> = description.split_whitespace().collect();
                let matching_words = description_words.iter()
                    .filter(|word| reasoning_lower.contains(&word.to_lowercase()))
                    .count();

                if matching_words > 0 {
                    score += 0.2 * (matching_words as f32 / description_words.len() as f32);
                }
            }
            
            if score > 0.0 {
                let confidence = score.min(1.0);
                let reasoning_text = if matched_keywords.is_empty() {
                    "Fuzzy match based on description".to_string()
                } else {
                    format!("Fuzzy match on keywords: {}", matched_keywords.join(", "))
                };
                
                let candidate = SelectionResult {
                    tool: tool.clone(),
                    confidence,
                    reasoning: reasoning_text,
                };
                
                if best_match.is_none() || candidate.confidence > best_match.as_ref().unwrap().confidence {
                    best_match = Some(candidate);
                }
            }
        }
        
        Ok(best_match)
    }
    
    /// Semantic selection - would use embeddings for similarity
    fn semantic_selection(&self, _reasoning: &str) -> Result<Option<SelectionResult>> {
        // TODO: Implement semantic selection using embeddings
        // For now, fall back to fuzzy matching
        self.fuzzy_match_selection(_reasoning)
    }
    
    /// Hybrid selection - combines multiple strategies
    fn hybrid_selection(&self, reasoning: &str) -> Result<Option<SelectionResult>> {
        // Try exact match first
        if let Some(exact_result) = self.exact_match_selection(reasoning)? {
            return Ok(Some(exact_result));
        }
        
        // Fall back to fuzzy matching
        self.fuzzy_match_selection(reasoning)
    }
    
    /// Build keyword mappings for tools
    fn build_keyword_mappings(&mut self) {
        for tool in &self.tools {
            let mut keywords = Vec::new();
            
            // Add variations of tool name
            let name = tool.name();
            keywords.push(name.to_string());
            keywords.push(name.replace('_', " "));
            keywords.push(name.replace('-', " "));
            
            // Extract keywords from description
            let description_words: Vec<String> = tool.description()
                .split_whitespace()
                .filter(|word| word.len() > 3) // Only meaningful words
                .map(|word| word.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                .filter(|word| !word.is_empty())
                .collect();
            
            keywords.extend(description_words);
            
            // Add tool-specific keywords based on common patterns
            match name.as_str() {
                name if name.contains("calculator") || name.contains("math") => {
                    keywords.extend(vec![
                        "calculate".to_string(),
                        "compute".to_string(),
                        "math".to_string(),
                        "arithmetic".to_string(),
                        "add".to_string(),
                        "subtract".to_string(),
                        "multiply".to_string(),
                        "divide".to_string(),
                    ]);
                }
                name if name.contains("search") => {
                    keywords.extend(vec![
                        "search".to_string(),
                        "find".to_string(),
                        "lookup".to_string(),
                        "query".to_string(),
                        "information".to_string(),
                    ]);
                }
                name if name.contains("file") => {
                    keywords.extend(vec![
                        "file".to_string(),
                        "read".to_string(),
                        "write".to_string(),
                        "save".to_string(),
                        "load".to_string(),
                    ]);
                }
                _ => {}
            }
            
            // Remove duplicates and store
            keywords.sort();
            keywords.dedup();
            self.keyword_mappings.insert(name.to_string(), keywords);
        }
    }
    
    /// Get available tools
    pub fn available_tools(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }
    
    /// Get selection statistics
    pub fn get_stats(&self) -> ToolSelectorStats {
        ToolSelectorStats {
            total_tools: self.tools.len(),
            total_keywords: self.keyword_mappings.values().map(|v| v.len()).sum(),
            config: self.config.clone(),
        }
    }
}

/// Statistics for tool selector
#[derive(Debug, Clone)]
pub struct ToolSelectorStats {
    /// Total number of available tools
    pub total_tools: usize,
    /// Total number of keywords across all tools
    pub total_keywords: usize,
    /// Current configuration
    pub config: ToolSelectorConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolContext;
    use async_trait::async_trait;
    use std::collections::HashMap;

    #[derive(Debug)]
    struct MockTool {
        name: String,
        description: String,
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            &self.description
        }

        fn schema(&self) -> crate::types::ToolSchema {
            crate::types::ToolSchema {
                name: self.name.clone(),
                description: self.description.clone(),
                input_schema: serde_json::json!({}),
                output_schema: None,
                dangerous: false,
                metadata: HashMap::new(),
            }
        }

        async fn execute(
            &self,
            _arguments: serde_json::Value,
            _context: &ToolContext,
        ) -> Result<crate::tool::ToolResult> {
            Ok(crate::tool::ToolResult::success("mock result"))
        }
    }

    #[test]
    fn test_exact_match_selection() {
        let tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(MockTool {
                name: "calculator".to_string(),
                description: "Performs calculations".to_string(),
            }),
        ];

        let selector = ToolSelector::new(tools);
        let result = selector.select_tool("I need to use the calculator tool");
        assert!(result.is_some());
        assert_eq!(result.unwrap().name(), "calculator");
    }

    #[test]
    fn test_fuzzy_match_selection() {
        let tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(MockTool {
                name: "search_engine".to_string(),
                description: "Searches for information online".to_string(),
            }),
        ];

        let selector = ToolSelector::new(tools);
        let result = selector.select_tool("I need to find some information");
        assert!(result.is_some());
        assert_eq!(result.unwrap().name(), "search_engine");
    }
}
