//! Query selectors for router query engines.
//!
//! This module provides implementations of query selectors that analyze queries
//! and select the most appropriate query engines.
//!
//! **Reference**: LlamaIndex Selectors
//! - File: `llama-index-core/llama_index/core/selectors/`
//! - Various selector implementations

use std::sync::Arc;

use async_trait::async_trait;
use serde_json;
use siumai::prelude::*;
use tracing::{debug, info, instrument};

use crate::engine::router::{
    QueryAnalysis, QueryEngineWrapper, QuerySelector, QueryType, SelectionResult,
};
use crate::generator::SiumaiGenerator;
use cheungfun_core::Result;

/// LLM-based query selector.
///
/// Uses an LLM to analyze queries and select appropriate engines.
///
/// **Reference**: LlamaIndex LLMSingleSelector
/// - File: `llama-index-core/llama_index/core/selectors/llm_selectors.py`
/// - Lines: Various LLM selector implementations
pub struct LLMQuerySelector {
    /// Siumai client for LLM analysis
    client: Siumai,
    /// Enable verbose logging
    verbose: bool,
}

impl std::fmt::Debug for LLMQuerySelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LLMQuerySelector")
            .field("verbose", &self.verbose)
            .finish()
    }
}

impl LLMQuerySelector {
    /// Create a new LLM query selector from a SiumaiGenerator.
    pub fn new(generator: Arc<SiumaiGenerator>) -> Self {
        Self {
            client: generator.client().clone(),
            verbose: false,
        }
    }

    /// Create a new LLM query selector from a Siumai client.
    pub fn from_client(client: Siumai) -> Self {
        Self {
            client,
            verbose: false,
        }
    }

    /// Create with verbose logging enabled.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Generate selection prompt for the LLM.
    ///
    /// **Reference**: LlamaIndex selector prompts
    fn create_selection_prompt(&self, engines: &[QueryEngineWrapper], query: &str) -> String {
        let mut prompt = String::new();

        prompt.push_str(
            "You are a query router that selects the best query engine for a given question.\n\n",
        );
        prompt.push_str("Available query engines:\n");

        for (i, engine) in engines.iter().enumerate() {
            prompt.push_str(&format!(
                "{}. {}: {}\n",
                i + 1,
                engine.metadata.name,
                engine.metadata.description
            ));
        }

        prompt.push_str("\nQuery: ");
        prompt.push_str(query);
        prompt.push_str("\n\n");

        prompt.push_str("Analyze the query and select the most appropriate engine(s).\n");
        prompt.push_str("Consider:\n");
        prompt.push_str("- Query complexity and scope\n");
        prompt.push_str("- Required level of detail\n");
        prompt.push_str("- Type of information needed\n\n");

        prompt.push_str("Respond with a JSON object containing:\n");
        prompt.push_str("{\n");
        prompt.push_str("  \"selected_engine\": <engine_number>,\n");
        prompt.push_str("  \"reason\": \"<explanation>\",\n");
        prompt.push_str("  \"query_type\": \"<Summary|Detailed|Hybrid|CodeSpecific>\",\n");
        prompt.push_str("  \"complexity\": <1-5>,\n");
        prompt.push_str("  \"confidence\": <0.0-1.0>\n");
        prompt.push_str("}\n");

        prompt
    }

    /// Parse LLM response to extract selection.
    fn parse_selection_response(
        &self,
        response: &str,
        engines: &[QueryEngineWrapper],
    ) -> Result<(SelectionResult, QueryAnalysis)> {
        // Try to extract JSON from response
        let json_start = response.find('{');
        let json_end = response.rfind('}');

        if let (Some(start), Some(end)) = (json_start, json_end) {
            let json_str = &response[start..=end];

            match serde_json::from_str::<serde_json::Value>(json_str) {
                Ok(parsed) => {
                    let selected_engine = parsed["selected_engine"].as_u64().unwrap_or(1) as usize;

                    let reason = parsed["reason"]
                        .as_str()
                        .unwrap_or("Selected by LLM")
                        .to_string();

                    let query_type_str = parsed["query_type"].as_str().unwrap_or("Summary");

                    let query_type = match query_type_str {
                        "Detailed" => QueryType::Detailed,
                        "Hybrid" => QueryType::Hybrid,
                        "CodeSpecific" => QueryType::CodeSpecific,
                        _ => QueryType::Summary,
                    };

                    let complexity = parsed["complexity"].as_u64().unwrap_or(3) as u8;

                    let confidence = parsed["confidence"].as_f64().unwrap_or(0.8) as f32;

                    // Convert to 0-based index
                    let engine_index = if selected_engine > 0 && selected_engine <= engines.len() {
                        selected_engine - 1
                    } else {
                        0 // Default to first engine
                    };

                    let selection_result = SelectionResult {
                        index: engine_index,
                        indices: vec![engine_index],
                        reason,
                        reasons: vec![],
                    };

                    let query_analysis = QueryAnalysis {
                        query_type,
                        complexity_level: complexity.clamp(1, 5),
                        context_depth: complexity.clamp(1, 5),
                        confidence: confidence.clamp(0.0, 1.0),
                    };

                    return Ok((selection_result, query_analysis));
                }
                Err(e) => {
                    debug!("Failed to parse JSON response: {}", e);
                }
            }
        }

        // Fallback: simple heuristic selection
        self.fallback_selection(engines, response)
    }

    /// Fallback selection when LLM response parsing fails.
    fn fallback_selection(
        &self,
        engines: &[QueryEngineWrapper],
        query: &str,
    ) -> Result<(SelectionResult, QueryAnalysis)> {
        // Simple heuristic: look for keywords
        let query_lower = query.to_lowercase();

        let (selected_index, query_type) = if query_lower.contains("summary")
            || query_lower.contains("overview")
            || query_lower.contains("what is")
        {
            (0, QueryType::Summary) // Assume first engine is for summaries
        } else if query_lower.contains("how")
            || query_lower.contains("implement")
            || query_lower.contains("code")
        {
            (engines.len().saturating_sub(1), QueryType::Detailed) // Last engine for details
        } else {
            (0, QueryType::Summary) // Default to summary
        };

        let selection_result = SelectionResult {
            index: selected_index,
            indices: vec![selected_index],
            reason: "Fallback heuristic selection".to_string(),
            reasons: vec![],
        };

        let query_analysis = QueryAnalysis {
            query_type,
            complexity_level: 3,
            context_depth: 3,
            confidence: 0.5,
        };

        Ok((selection_result, query_analysis))
    }
}

#[async_trait]
impl QuerySelector for LLMQuerySelector {
    /// Select the most appropriate query engine using LLM analysis.
    ///
    /// **Reference**: LlamaIndex LLMSingleSelector.select()
    #[instrument(skip(self, engines))]
    async fn select(&self, engines: &[QueryEngineWrapper], query: &str) -> Result<SelectionResult> {
        if engines.is_empty() {
            return Err(cheungfun_core::CheungfunError::Configuration {
                message: "No query engines available for selection".to_string(),
            });
        }

        if engines.len() == 1 {
            // Only one engine available
            return Ok(SelectionResult {
                index: 0,
                indices: vec![0],
                reason: "Only engine available".to_string(),
                reasons: vec![],
            });
        }

        // Generate selection prompt
        let prompt = self.create_selection_prompt(engines, query);

        if self.verbose {
            debug!("Selection prompt: {}", prompt);
        }

        // Get LLM response using Siumai client directly
        let messages = vec![ChatMessage::user(prompt).build()];
        let response =
            self.client
                .chat(messages)
                .await
                .map_err(|e| cheungfun_core::CheungfunError::Llm {
                    message: format!("LLM selection failed: {}", e),
                })?;

        let response_content = response.content.all_text();

        if self.verbose {
            debug!("LLM selection response: {}", response_content);
        }

        // Parse response
        let (selection_result, _analysis) =
            self.parse_selection_response(&response_content, engines)?;

        info!(
            "Selected engine '{}': {}",
            engines[selection_result.index].metadata.name, selection_result.reason
        );

        Ok(selection_result)
    }

    /// Analyze query characteristics using LLM.
    #[instrument(skip(self))]
    async fn analyze_query(&self, query: &str) -> Result<QueryAnalysis> {
        let prompt = format!(
            "Analyze the following query and classify it:\n\
             Query: {}\n\n\
             Respond with a JSON object:\n\
             {{\n\
               \"query_type\": \"<Summary|Detailed|Hybrid|CodeSpecific>\",\n\
               \"complexity\": <1-5>,\n\
               \"context_depth\": <1-5>,\n\
               \"confidence\": <0.0-1.0>\n\
             }}",
            query
        );

        let messages = vec![ChatMessage::user(prompt).build()];
        let response =
            self.client
                .chat(messages)
                .await
                .map_err(|e| cheungfun_core::CheungfunError::Llm {
                    message: format!("LLM analysis failed: {}", e),
                })?;

        let response_content = response.content.all_text();

        // Parse analysis response (similar to selection parsing)
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response_content) {
            let query_type_str = parsed["query_type"].as_str().unwrap_or("Summary");
            let query_type = match query_type_str {
                "Detailed" => QueryType::Detailed,
                "Hybrid" => QueryType::Hybrid,
                "CodeSpecific" => QueryType::CodeSpecific,
                _ => QueryType::Summary,
            };

            Ok(QueryAnalysis {
                query_type,
                complexity_level: parsed["complexity"].as_u64().unwrap_or(3) as u8,
                context_depth: parsed["context_depth"].as_u64().unwrap_or(3) as u8,
                confidence: parsed["confidence"].as_f64().unwrap_or(0.8) as f32,
            })
        } else {
            // Fallback analysis
            Ok(QueryAnalysis {
                query_type: QueryType::Summary,
                complexity_level: 3,
                context_depth: 3,
                confidence: 0.5,
            })
        }
    }
}

/// Simple rule-based query selector.
///
/// Uses predefined rules to select engines based on query patterns.
#[derive(Debug)]
pub struct RuleBasedQuerySelector {
    /// Enable verbose logging
    verbose: bool,
}

impl RuleBasedQuerySelector {
    pub fn new() -> Self {
        Self { verbose: false }
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

#[async_trait]
impl QuerySelector for RuleBasedQuerySelector {
    async fn select(&self, engines: &[QueryEngineWrapper], query: &str) -> Result<SelectionResult> {
        if engines.is_empty() {
            return Err(cheungfun_core::CheungfunError::Configuration {
                message: "No query engines available".to_string(),
            });
        }

        let query_lower = query.to_lowercase();

        // Simple rule-based selection
        let (selected_index, reason) = if query_lower.contains("summary")
            || query_lower.contains("overview")
            || query_lower.contains("what is")
        {
            (0, "Summary keywords detected")
        } else if query_lower.contains("how")
            || query_lower.contains("implement")
            || query_lower.contains("detail")
        {
            (engines.len().saturating_sub(1), "Detail keywords detected")
        } else {
            (0, "Default to first engine")
        };

        Ok(SelectionResult {
            index: selected_index,
            indices: vec![selected_index],
            reason: reason.to_string(),
            reasons: vec![],
        })
    }

    async fn analyze_query(&self, query: &str) -> Result<QueryAnalysis> {
        let query_lower = query.to_lowercase();

        let query_type = if query_lower.contains("summary") || query_lower.contains("overview") {
            QueryType::Summary
        } else if query_lower.contains("code") || query_lower.contains("implement") {
            QueryType::CodeSpecific
        } else if query_lower.contains("how") || query_lower.contains("detail") {
            QueryType::Detailed
        } else {
            QueryType::Summary
        };

        let complexity = if query_lower.len() > 100 {
            4
        } else if query_lower.len() > 50 {
            3
        } else {
            2
        };

        Ok(QueryAnalysis {
            query_type,
            complexity_level: complexity,
            context_depth: complexity,
            confidence: 0.7,
        })
    }
}
