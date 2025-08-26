//! Tool Retrieval System
//!
//! This module implements a tool retrieval system that dynamically selects
//! relevant tools based on query context, similar to `LlamaIndex`'s tool retrieval.

use crate::{
    error::{AgentError, Result},
    tool::Tool,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

/// Tool retrieval strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetrievalStrategy {
    /// Retrieve all tools
    All,
    /// Retrieve based on similarity score
    Similarity {
        /// Minimum similarity threshold
        threshold: f64,
        /// Maximum number of tools to retrieve
        top_k: usize,
    },
    /// Retrieve based on tool tags/categories
    Category {
        /// Categories to filter by
        categories: Vec<String>,
    },
    /// Custom retrieval logic
    Custom,
}

impl Default for RetrievalStrategy {
    fn default() -> Self {
        Self::Similarity {
            threshold: 0.5,
            top_k: 5,
        }
    }
}

/// Tool metadata for retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMetadata {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Tool categories/tags
    pub categories: Vec<String>,
    /// Tool keywords
    pub keywords: Vec<String>,
    /// Usage examples
    pub examples: Vec<String>,
}

/// Tool with retrieval metadata
#[derive(Clone)]
pub struct RetrievableTool {
    /// The actual tool
    pub tool: Arc<dyn Tool>,
    /// Metadata for retrieval
    pub metadata: ToolMetadata,
}

/// Tool retrieval result
#[derive(Debug, Clone)]
pub struct ToolRetrievalResult {
    /// Retrieved tools
    pub tools: Vec<Arc<dyn Tool>>,
    /// Retrieval scores (if applicable)
    pub scores: Vec<f64>,
    /// Retrieval metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Base trait for tool retrieval
#[async_trait]
pub trait ToolRetriever: Send + Sync {
    /// Retrieve tools based on query
    async fn retrieve_tools(
        &self,
        query: &str,
        strategy: Option<RetrievalStrategy>,
    ) -> Result<ToolRetrievalResult>;

    /// Add tool to retriever
    async fn add_tool(&mut self, tool: RetrievableTool) -> Result<()>;

    /// Remove tool from retriever
    async fn remove_tool(&mut self, tool_name: &str) -> Result<bool>;

    /// List all available tools
    async fn list_tools(&self) -> Result<Vec<String>>;

    /// Update retrieval strategy
    async fn update_strategy(&mut self, strategy: RetrievalStrategy) -> Result<()>;
}

/// Simple keyword-based tool retriever
pub struct KeywordToolRetriever {
    /// Available tools
    tools: HashMap<String, RetrievableTool>,
    /// Default retrieval strategy
    strategy: RetrievalStrategy,
}

impl KeywordToolRetriever {
    /// Create new keyword tool retriever
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            strategy: RetrievalStrategy::default(),
        }
    }

    /// Create with strategy
    #[must_use]
    pub fn with_strategy(strategy: RetrievalStrategy) -> Self {
        Self {
            tools: HashMap::new(),
            strategy,
        }
    }

    /// Calculate keyword similarity score
    fn calculate_similarity(&self, query: &str, metadata: &ToolMetadata) -> f64 {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        // Check description (weight: 3.0)
        let desc_score =
            self.calculate_text_similarity(&query_words, &metadata.description.to_lowercase());
        total_score += desc_score * 3.0;
        total_weight += 3.0;

        // Check keywords (weight: 2.0)
        for keyword in &metadata.keywords {
            let keyword_score =
                self.calculate_text_similarity(&query_words, &keyword.to_lowercase());
            total_score += keyword_score * 2.0;
            total_weight += 2.0;
        }

        // Check categories (weight: 1.5)
        for category in &metadata.categories {
            let category_score =
                self.calculate_text_similarity(&query_words, &category.to_lowercase());
            total_score += category_score * 1.5;
            total_weight += 1.5;
        }

        // Check examples (weight: 1.0)
        for example in &metadata.examples {
            let example_score =
                self.calculate_text_similarity(&query_words, &example.to_lowercase());
            total_score += example_score * 1.0;
            total_weight += 1.0;
        }

        if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.0
        }
    }

    fn calculate_text_similarity(&self, query_words: &[&str], text: &str) -> f64 {
        let text_words: Vec<&str> = text.split_whitespace().collect();
        let mut matches = 0;

        for query_word in query_words {
            for text_word in &text_words {
                // Exact match
                if query_word == text_word {
                    matches += 2;
                    continue;
                }

                // Partial match
                if query_word.len() > 3 && text_word.contains(query_word) {
                    matches += 1;
                } else if text_word.len() > 3 && query_word.contains(text_word) {
                    matches += 1;
                }
            }
        }

        if query_words.is_empty() {
            0.0
        } else {
            f64::from(matches) / (query_words.len() * 2) as f64
        }
    }
}

impl Default for KeywordToolRetriever {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolRetriever for KeywordToolRetriever {
    async fn retrieve_tools(
        &self,
        query: &str,
        strategy: Option<RetrievalStrategy>,
    ) -> Result<ToolRetrievalResult> {
        let strategy = strategy.unwrap_or_else(|| self.strategy.clone());

        match strategy {
            RetrievalStrategy::All => {
                let tools: Vec<Arc<dyn Tool>> =
                    self.tools.values().map(|rt| rt.tool.clone()).collect();
                let scores = vec![1.0; tools.len()];

                Ok(ToolRetrievalResult {
                    tools,
                    scores,
                    metadata: HashMap::new(),
                })
            }

            RetrievalStrategy::Similarity { threshold, top_k } => {
                let mut scored_tools: Vec<(Arc<dyn Tool>, f64)> = Vec::new();

                for retrievable_tool in self.tools.values() {
                    let score = self.calculate_similarity(query, &retrievable_tool.metadata);
                    if score >= threshold {
                        scored_tools.push((retrievable_tool.tool.clone(), score));
                    }
                }

                // Sort by score descending
                scored_tools
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Take top k
                scored_tools.truncate(top_k);

                let tools: Vec<Arc<dyn Tool>> =
                    scored_tools.iter().map(|(tool, _)| tool.clone()).collect();
                let scores: Vec<f64> = scored_tools.iter().map(|(_, score)| *score).collect();

                let mut metadata = HashMap::new();
                metadata.insert("threshold".to_string(), serde_json::json!(threshold));
                metadata.insert("top_k".to_string(), serde_json::json!(top_k));
                metadata.insert("query".to_string(), serde_json::json!(query));

                Ok(ToolRetrievalResult {
                    tools,
                    scores,
                    metadata,
                })
            }

            RetrievalStrategy::Category { categories } => {
                let mut tools = Vec::new();
                let mut scores = Vec::new();

                for retrievable_tool in self.tools.values() {
                    let has_category = retrievable_tool
                        .metadata
                        .categories
                        .iter()
                        .any(|cat| categories.contains(cat));

                    if has_category {
                        tools.push(retrievable_tool.tool.clone());
                        scores.push(1.0);
                    }
                }

                let mut metadata = HashMap::new();
                metadata.insert("categories".to_string(), serde_json::json!(categories));

                Ok(ToolRetrievalResult {
                    tools,
                    scores,
                    metadata,
                })
            }

            RetrievalStrategy::Custom => {
                // Fallback to similarity-based retrieval
                self.retrieve_tools(
                    query,
                    Some(RetrievalStrategy::Similarity {
                        threshold: 0.3,
                        top_k: 10,
                    }),
                )
                .await
            }
        }
    }

    async fn add_tool(&mut self, tool: RetrievableTool) -> Result<()> {
        let tool_name = tool.metadata.name.clone();
        if self.tools.contains_key(&tool_name) {
            return Err(AgentError::validation(
                "tool_name",
                format!("Tool '{tool_name}' already exists"),
            ));
        }

        self.tools.insert(tool_name, tool);
        Ok(())
    }

    async fn remove_tool(&mut self, tool_name: &str) -> Result<bool> {
        Ok(self.tools.remove(tool_name).is_some())
    }

    async fn list_tools(&self) -> Result<Vec<String>> {
        Ok(self.tools.keys().cloned().collect())
    }

    async fn update_strategy(&mut self, strategy: RetrievalStrategy) -> Result<()> {
        self.strategy = strategy;
        Ok(())
    }
}

/// Tool retriever builder
pub struct ToolRetrieverBuilder {
    strategy: RetrievalStrategy,
    tools: Vec<RetrievableTool>,
}

impl ToolRetrieverBuilder {
    /// Create a new tool retriever builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            strategy: RetrievalStrategy::default(),
            tools: Vec::new(),
        }
    }

    /// Set retrieval strategy
    #[must_use]
    pub fn strategy(mut self, strategy: RetrievalStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Add a tool to the retriever
    #[must_use]
    pub fn add_tool(mut self, tool: RetrievableTool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Build the keyword tool retriever
    pub async fn build(self) -> Result<KeywordToolRetriever> {
        let mut retriever = KeywordToolRetriever::with_strategy(self.strategy);

        for tool in self.tools {
            retriever.add_tool(tool).await?;
        }

        Ok(retriever)
    }
}

impl Default for ToolRetrieverBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper functions for creating tool metadata
pub mod metadata_helpers {
    use super::ToolMetadata;

    /// Create tool metadata for web search tools
    #[must_use]
    pub fn web_search_metadata(tool_name: &str) -> ToolMetadata {
        ToolMetadata {
            name: tool_name.to_string(),
            description: "Search the web for information".to_string(),
            categories: vec!["search".to_string(), "web".to_string()],
            keywords: vec![
                "search".to_string(),
                "web".to_string(),
                "internet".to_string(),
                "query".to_string(),
                "find".to_string(),
            ],
            examples: vec![
                "search for latest news".to_string(),
                "find information about AI".to_string(),
            ],
        }
    }

    /// Create tool metadata for RAG tools
    #[must_use]
    pub fn rag_metadata(tool_name: &str) -> ToolMetadata {
        ToolMetadata {
            name: tool_name.to_string(),
            description: "Search through knowledge base using RAG".to_string(),
            categories: vec!["rag".to_string(), "knowledge".to_string()],
            keywords: vec![
                "knowledge".to_string(),
                "document".to_string(),
                "search".to_string(),
                "rag".to_string(),
                "retrieval".to_string(),
            ],
            examples: vec![
                "search company documents".to_string(),
                "find relevant information in knowledge base".to_string(),
            ],
        }
    }

    /// Create tool metadata for calculation tools
    #[must_use]
    pub fn calculation_metadata(tool_name: &str) -> ToolMetadata {
        ToolMetadata {
            name: tool_name.to_string(),
            description: "Perform mathematical calculations".to_string(),
            categories: vec!["math".to_string(), "calculation".to_string()],
            keywords: vec![
                "math".to_string(),
                "calculate".to_string(),
                "compute".to_string(),
                "number".to_string(),
                "arithmetic".to_string(),
            ],
            examples: vec![
                "calculate 2 + 2".to_string(),
                "compute the square root of 16".to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{ToolContext, ToolResult};
    use crate::types::ToolSchema;
    use async_trait::async_trait;

    // Mock tool for testing
    struct MockTool {
        name: String,
    }

    impl std::fmt::Debug for MockTool {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockTool")
                .field("name", &self.name)
                .finish()
        }
    }

    #[async_trait]
    impl Tool for MockTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: self.name.clone(),
                description: "Mock tool for testing".to_string(),
                input_schema: serde_json::json!({}),
                output_schema: None,
                dangerous: false,
                metadata: std::collections::HashMap::new(),
            }
        }

        async fn execute(
            &self,
            _arguments: serde_json::Value,
            _context: &ToolContext,
        ) -> Result<ToolResult> {
            Ok(ToolResult::success("mock result"))
        }
    }

    #[tokio::test]
    async fn test_keyword_tool_retriever() {
        let mut retriever = KeywordToolRetriever::new();

        // Add a search tool
        let search_tool = RetrievableTool {
            tool: Arc::new(MockTool {
                name: "web_search".to_string(),
            }),
            metadata: metadata_helpers::web_search_metadata("web_search"),
        };

        retriever.add_tool(search_tool).await.unwrap();

        // Test retrieval
        let result = retriever
            .retrieve_tools(
                "search for information online",
                Some(RetrievalStrategy::Similarity {
                    threshold: 0.1,
                    top_k: 5,
                }),
            )
            .await
            .unwrap();

        assert_eq!(result.tools.len(), 1);
        assert!(result.scores[0] > 0.1);
    }

    #[tokio::test]
    async fn test_category_retrieval() {
        let mut retriever = KeywordToolRetriever::new();

        // Add tools with different categories
        let search_tool = RetrievableTool {
            tool: Arc::new(MockTool {
                name: "web_search".to_string(),
            }),
            metadata: metadata_helpers::web_search_metadata("web_search"),
        };

        let calc_tool = RetrievableTool {
            tool: Arc::new(MockTool {
                name: "calculator".to_string(),
            }),
            metadata: metadata_helpers::calculation_metadata("calculator"),
        };

        retriever.add_tool(search_tool).await.unwrap();
        retriever.add_tool(calc_tool).await.unwrap();

        // Test category-based retrieval
        let result = retriever
            .retrieve_tools(
                "",
                Some(RetrievalStrategy::Category {
                    categories: vec!["math".to_string()],
                }),
            )
            .await
            .unwrap();

        assert_eq!(result.tools.len(), 1);
        assert_eq!(result.tools[0].name(), "calculator");
    }
}
