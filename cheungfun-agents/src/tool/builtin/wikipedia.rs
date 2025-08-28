//! Wikipedia Tool - Search and retrieve Wikipedia content
//!
//! This tool provides Wikipedia search and content retrieval functionality.

use crate::{
    error::{AgentError, Result},
    tool::{create_simple_schema, Tool, ToolContext, ToolResult},
    types::ToolSchema,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Wikipedia search and content retrieval tool
#[derive(Debug, Clone)]
pub struct WikipediaTool {
    name: String,
    language: String,
}

impl WikipediaTool {
    /// Create a new Wikipedia tool
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "wikipedia".to_string(),
            language: "en".to_string(),
        }
    }

    /// Create Wikipedia tool with specific language
    #[must_use]
    pub fn with_language(language: impl Into<String>) -> Self {
        Self {
            name: "wikipedia".to_string(),
            language: language.into(),
        }
    }
}

impl Default for WikipediaTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct WikipediaSearchResult {
    title: String,
    summary: String,
    url: String,
    content: String,
}

#[async_trait]
impl Tool for WikipediaTool {
    fn schema(&self) -> ToolSchema {
        let mut properties = HashMap::new();
        properties.insert(
            "query".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Search query for Wikipedia articles"
            }),
        );
        properties.insert(
            "max_results".to_string(),
            serde_json::json!({
                "type": "number",
                "description": "Maximum number of search results to return (1-10)",
                "minimum": 1,
                "maximum": 10,
                "default": 3
            }),
        );
        properties.insert(
            "include_content".to_string(),
            serde_json::json!({
                "type": "boolean",
                "description": "Whether to include full article content or just summary",
                "default": false
            }),
        );

        ToolSchema {
            name: self.name.clone(),
            description: "Search Wikipedia articles and retrieve content. Provides summaries and full article text for research purposes.".to_string(),
            input_schema: create_simple_schema(properties, vec!["query".to_string()]),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "Article title"
                                },
                                "summary": {
                                    "type": "string",
                                    "description": "Article summary"
                                },
                                "url": {
                                    "type": "string",
                                    "description": "Wikipedia URL"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Full article content (if requested)"
                                }
                            }
                        }
                    },
                    "query": {
                        "type": "string",
                        "description": "Original search query"
                    },
                    "total_results": {
                        "type": "number",
                        "description": "Number of results returned"
                    }
                }
            })),
            dangerous: false,
            metadata: HashMap::new(),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult> {
        #[derive(Deserialize)]
        struct WikipediaArgs {
            query: String,
            #[serde(default = "default_max_results")]
            max_results: usize,
            #[serde(default)]
            include_content: bool,
        }

        fn default_max_results() -> usize {
            3
        }

        let args: WikipediaArgs = serde_json::from_value(arguments)
            .map_err(|e| AgentError::tool(&self.name, format!("Invalid arguments: {e}")))?;

        if args.max_results > 10 || args.max_results == 0 {
            let error_msg = "max_results must be between 1 and 10";
            return Ok(ToolResult::error_with_content(error_msg, error_msg));
        }

        // In a real implementation, you would make HTTP requests to Wikipedia API
        // For demo purposes, we'll return mock data
        let results = self.search_wikipedia(&args.query, args.max_results, args.include_content)?;

        let content = if results.is_empty() {
            format!("No Wikipedia articles found for query: '{}'", args.query)
        } else {
            let mut output = format!(
                "Found {} Wikipedia article(s) for '{}':\n\n",
                results.len(),
                args.query
            );

            for (i, result) in results.iter().enumerate() {
                output.push_str(&format!("{}. **{}**\n", i + 1, result.title));
                output.push_str(&format!("   URL: {}\n", result.url));
                output.push_str(&format!("   Summary: {}\n", result.summary));

                if args.include_content && !result.content.is_empty() {
                    output.push_str(&format!(
                        "   Content: {}...\n",
                        if result.content.len() > 500 {
                            &result.content[..500]
                        } else {
                            &result.content
                        }
                    ));
                }
                output.push('\n');
            }

            output
        };

        let result = ToolResult::success(content)
            .with_metadata("results".to_string(), serde_json::to_value(&results)?)
            .with_metadata("query".to_string(), serde_json::json!(args.query))
            .with_metadata(
                "total_results".to_string(),
                serde_json::json!(results.len()),
            )
            .with_metadata(
                "include_content".to_string(),
                serde_json::json!(args.include_content),
            );

        Ok(result)
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "search".to_string(),
            "information".to_string(),
            "research".to_string(),
            "knowledge".to_string(),
            "external_api".to_string(),
        ]
    }
}

impl WikipediaTool {
    /// Search Wikipedia (mock implementation)
    /// In production, use Wikipedia API: <https://en.wikipedia.org/api/rest_v1>/
    fn search_wikipedia(
        &self,
        query: &str,
        max_results: usize,
        include_content: bool,
    ) -> Result<Vec<WikipediaSearchResult>> {
        // Mock data based on common search terms
        let mock_articles = match query.to_lowercase().as_str() {
            q if q.contains("rust") => vec![
                WikipediaSearchResult {
                    title: "Rust (programming language)".to_string(),
                    summary: "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.".to_string(),
                    url: "https://en.wikipedia.org/wiki/Rust_(programming_language)".to_string(),
                    content: if include_content { "Rust is a multi-paradigm, general-purpose programming language designed for performance and safety, especially safe concurrency. Rust is syntactically similar to C++, but can guarantee memory safety by using a borrow checker to validate references.".to_string() } else { String::new() }
                },
                WikipediaSearchResult {
                    title: "Mozilla".to_string(),
                    summary: "Mozilla Corporation is a wholly owned subsidiary of the Mozilla Foundation that coordinates and integrates the development of Internet-related applications.".to_string(),
                    url: "https://en.wikipedia.org/wiki/Mozilla".to_string(),
                    content: if include_content { "Mozilla Corporation was established in August 2005 as a wholly owned taxable subsidiary of the Mozilla Foundation to handle revenue-generating activities including development and distribution of Firefox and Thunderbird.".to_string() } else { String::new() }
                }
            ],
            q if q.contains("python") => vec![
                WikipediaSearchResult {
                    title: "Python (programming language)".to_string(),
                    summary: "Python is an interpreted high-level general-purpose programming language with dynamic semantics and automatic memory management.".to_string(),
                    url: "https://en.wikipedia.org/wiki/Python_(programming_language)".to_string(),
                    content: if include_content { "Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant whitespace.".to_string() } else { String::new() }
                }
            ],
            q if q.contains("artificial intelligence") || q.contains("ai") => vec![
                WikipediaSearchResult {
                    title: "Artificial intelligence".to_string(),
                    summary: "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.".to_string(),
                    url: "https://en.wikipedia.org/wiki/Artificial_intelligence".to_string(),
                    content: if include_content { "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents'.".to_string() } else { String::new() }
                },
                WikipediaSearchResult {
                    title: "Machine learning".to_string(),
                    summary: "Machine learning (ML) is a type of artificial intelligence (AI) that allows software applications to become more accurate at predicting outcomes.".to_string(),
                    url: "https://en.wikipedia.org/wiki/Machine_learning".to_string(),
                    content: if include_content { "Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks.".to_string() } else { String::new() }
                }
            ],
            _ => vec![
                WikipediaSearchResult {
                    title: format!("Search Results for '{query}'"),
                    summary: format!("This is a mock Wikipedia search result for the query: '{query}'."),
                    url: format!("https://en.wikipedia.org/wiki/{}", query.replace(' ', "_")),
                    content: if include_content { format!("This would be the full article content for '{query}'. In a real implementation, this would contain the actual Wikipedia article text retrieved via the Wikipedia API.") } else { String::new() }
                }
            ]
        };

        // Limit results to max_results
        let limited_results = mock_articles.into_iter().take(max_results).collect();
        Ok(limited_results)
    }

    /// Set language for Wikipedia searches
    pub fn set_language(&mut self, language: impl Into<String>) {
        self.language = language.into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wikipedia_search() {
        let tool = WikipediaTool::new();
        let context = ToolContext::new();

        // Test basic search
        let args = serde_json::json!({
            "query": "Rust programming language",
            "max_results": 2
        });

        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);
        assert!(result.content.contains("Rust"));

        // Check metadata
        assert!(result.metadata.contains_key("results"));
        assert!(result.metadata.contains_key("query"));
        assert!(result.metadata.contains_key("total_results"));
    }

    #[tokio::test]
    async fn test_wikipedia_with_content() {
        let tool = WikipediaTool::new();
        let context = ToolContext::new();

        // Test search with content
        let args = serde_json::json!({
            "query": "Python",
            "max_results": 1,
            "include_content": true
        });

        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);
        assert!(result.content.contains("Python"));

        // Verify content is included in metadata
        let results = result.metadata.get("results").unwrap();
        let results_array = results.as_array().unwrap();
        let first_result = &results_array[0];
        assert!(!first_result["content"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_wikipedia_invalid_max_results() {
        let tool = WikipediaTool::new();
        let context = ToolContext::new();

        // Test with invalid max_results
        let args = serde_json::json!({
            "query": "test",
            "max_results": 15
        });

        let result = tool.execute(args, &context).await.unwrap();
        assert!(!result.success);
        assert!(result
            .content
            .contains("max_results must be between 1 and 10"));
    }
}
