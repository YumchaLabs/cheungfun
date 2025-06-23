//! Search tool for integrating with Cheungfun's RAG system.

use crate::{
    error::{AgentError, Result},
    tool::{Tool, ToolContext, ToolResult, create_simple_schema, number_param, string_param},
    types::ToolSchema,
};
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;

/// Search tool that integrates with Cheungfun's RAG system
#[derive(Debug, Clone)]
pub struct SearchTool {
    name: String,
    /// Default number of results to return
    default_top_k: usize,
    /// Maximum number of results allowed
    max_top_k: usize,
}

impl SearchTool {
    /// Create a new search tool with default settings
    pub fn new() -> Self {
        Self {
            name: "search".to_string(),
            default_top_k: 5,
            max_top_k: 20,
        }
    }

    /// Create a search tool with custom limits
    pub fn with_limits(default_top_k: usize, max_top_k: usize) -> Self {
        Self {
            name: "search".to_string(),
            default_top_k,
            max_top_k,
        }
    }
}

impl Default for SearchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for SearchTool {
    fn schema(&self) -> ToolSchema {
        let mut properties = HashMap::new();

        let (query_schema, _) = string_param("Search query", true);
        properties.insert("query".to_string(), query_schema);

        let (top_k_schema, _) = number_param(
            &format!("Number of results to return (max: {})", self.max_top_k),
            false,
        );
        properties.insert("top_k".to_string(), top_k_schema);

        properties.insert(
            "search_type".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Type of search to perform",
                "enum": ["vector", "keyword", "hybrid"],
                "default": "hybrid"
            }),
        );

        properties.insert(
            "filters".to_string(),
            serde_json::json!({
                "type": "object",
                "description": "Metadata filters to apply",
                "additionalProperties": true
            }),
        );

        let (threshold_schema, _) = number_param("Similarity threshold (0.0-1.0)", false);
        properties.insert("similarity_threshold".to_string(), threshold_schema);

        ToolSchema {
            name: self.name.clone(),
            description: "Search through the knowledge base using vector similarity, keyword matching, or hybrid search. Returns relevant documents and passages.".to_string(),
            input_schema: create_simple_schema(properties, vec!["query".to_string()]),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "description": "Search results",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Document content"
                                },
                                "score": {
                                    "type": "number",
                                    "description": "Relevance score"
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Document metadata"
                                }
                            }
                        }
                    },
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "total_results": {
                        "type": "integer",
                        "description": "Total number of results found"
                    }
                }
            })),
            dangerous: false,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("default_top_k".to_string(), serde_json::json!(self.default_top_k));
                meta.insert("max_top_k".to_string(), serde_json::json!(self.max_top_k));
                meta
            },
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        context: &ToolContext,
    ) -> Result<ToolResult> {
        #[derive(Deserialize)]
        struct SearchArgs {
            query: String,
            top_k: Option<usize>,
            #[serde(default = "default_search_type")]
            search_type: String,
            #[serde(default)]
            filters: HashMap<String, serde_json::Value>,
            similarity_threshold: Option<f32>,
        }

        fn default_search_type() -> String {
            "hybrid".to_string()
        }

        let args: SearchArgs = serde_json::from_value(arguments)
            .map_err(|e| AgentError::tool(&self.name, format!("Invalid arguments: {e}")))?;

        // Validate and set top_k
        let top_k = args.top_k.unwrap_or(self.default_top_k);
        if top_k > self.max_top_k {
            return Ok(ToolResult::error(format!(
                "top_k ({}) exceeds maximum allowed ({})",
                top_k, self.max_top_k
            )));
        }

        // Validate similarity threshold
        if let Some(threshold) = args.similarity_threshold {
            if !(0.0..=1.0).contains(&threshold) {
                return Ok(ToolResult::error(
                    "similarity_threshold must be between 0.0 and 1.0",
                ));
            }
        }

        // Check if we have a retriever in the context
        // In a real implementation, this would integrate with cheungfun-query
        if let Some(retriever_data) = context.get_data("retriever") {
            // This is where we would integrate with the actual RAG system
            // For now, we'll simulate the search
            self.simulate_search(
                &args.query,
                top_k,
                &args.search_type,
                &args.filters,
                args.similarity_threshold,
            )
            .await
        } else {
            // No retriever available, return a helpful message
            Ok(ToolResult::error(
                "No search retriever available. Please ensure the agent is configured with a knowledge base.",
            ))
        }
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "search".to_string(),
            "retrieval".to_string(),
            "knowledge_base".to_string(),
            "vector_search".to_string(),
            "keyword_search".to_string(),
            "hybrid_search".to_string(),
        ]
    }
}

impl SearchTool {
    /// Simulate search results (placeholder for actual RAG integration)
    async fn simulate_search(
        &self,
        query: &str,
        top_k: usize,
        search_type: &str,
        _filters: &HashMap<String, serde_json::Value>,
        _similarity_threshold: Option<f32>,
    ) -> Result<ToolResult> {
        // This is a placeholder implementation
        // In the real implementation, this would:
        // 1. Use the retriever from context to perform the search
        // 2. Apply the specified search type (vector, keyword, hybrid)
        // 3. Apply filters and similarity threshold
        // 4. Return actual search results

        let mock_results = vec![
            serde_json::json!({
                "content": format!("This is a mock search result for query: '{}'", query),
                "score": 0.95,
                "metadata": {
                    "source": "mock_document_1.txt",
                    "chunk_id": "chunk_1",
                    "search_type": search_type
                }
            }),
            serde_json::json!({
                "content": format!("Another relevant result related to: '{}'", query),
                "score": 0.87,
                "metadata": {
                    "source": "mock_document_2.txt",
                    "chunk_id": "chunk_2",
                    "search_type": search_type
                }
            }),
        ];

        let limited_results: Vec<_> = mock_results.into_iter().take(top_k).collect();
        let total_results = limited_results.len();

        let content = format!(
            "Found {} results for query: '{}' using {} search",
            total_results, query, search_type
        );

        Ok(ToolResult::success(content)
            .with_metadata("results".to_string(), serde_json::json!(limited_results))
            .with_metadata("query".to_string(), serde_json::json!(query))
            .with_metadata(
                "total_results".to_string(),
                serde_json::json!(total_results),
            )
            .with_metadata("search_type".to_string(), serde_json::json!(search_type)))
    }

    /// Integration point for actual RAG system
    /// This method would be called by the agent when setting up the search tool
    pub fn with_retriever(self, _retriever: Box<dyn std::any::Any + Send + Sync>) -> Self {
        // In the real implementation, this would store a reference to the retriever
        // For now, we just return self
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_search_tool_basic() {
        let tool = SearchTool::new();
        let context = ToolContext::new();
        let args = serde_json::json!({
            "query": "test query",
            "top_k": 3
        });

        let result = tool.execute(args, &context).await.unwrap();
        // Should fail because no retriever is available
        assert!(!result.success);
        assert!(
            result
                .error_message()
                .unwrap()
                .contains("No search retriever")
        );
    }

    #[tokio::test]
    async fn test_search_tool_with_mock_retriever() {
        let tool = SearchTool::new();
        let context = ToolContext::new().with_data("retriever", serde_json::json!({"mock": true}));

        let args = serde_json::json!({
            "query": "artificial intelligence",
            "top_k": 2,
            "search_type": "vector"
        });

        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);

        // Check metadata
        assert_eq!(
            result.metadata.get("query"),
            Some(&serde_json::json!("artificial intelligence"))
        );
        assert_eq!(
            result.metadata.get("total_results"),
            Some(&serde_json::json!(2))
        );
    }

    #[tokio::test]
    async fn test_search_tool_validation() {
        let tool = SearchTool::with_limits(5, 10);
        let context = ToolContext::new().with_data("retriever", serde_json::json!({"mock": true}));

        // Test top_k limit
        let args = serde_json::json!({
            "query": "test",
            "top_k": 15  // Exceeds max_top_k of 10
        });

        let result = tool.execute(args, &context).await.unwrap();
        assert!(!result.success);
        assert!(result.error_message().unwrap().contains("exceeds maximum"));

        // Test invalid similarity threshold
        let args = serde_json::json!({
            "query": "test",
            "similarity_threshold": 1.5  // Invalid threshold
        });

        let result = tool.execute(args, &context).await.unwrap();
        assert!(!result.success);
        assert!(
            result
                .error_message()
                .unwrap()
                .contains("between 0.0 and 1.0")
        );
    }

    #[test]
    fn test_search_tool_schema() {
        let tool = SearchTool::new();
        let schema = tool.schema();

        assert_eq!(schema.name, "search");
        assert!(!schema.description.is_empty());
        assert!(!schema.dangerous);

        // Check that required fields are present
        let required = schema.input_schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("query")));
    }
}
