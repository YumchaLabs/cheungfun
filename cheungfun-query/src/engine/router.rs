//! Router query engine for intelligent query routing.
//!
//! This module implements a router query engine that selects the most appropriate
//! query engine from a set of candidates based on query analysis.
//!
//! **Reference**: LlamaIndex RouterQueryEngine
//! - File: `llama-index-core/llama_index/core/query_engine/router_query_engine.py`
//! - Lines: L95-L203

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{info, instrument};

use crate::engine::QueryEngine;
use cheungfun_core::{types::QueryResponse, Result};

/// Query type classification for routing decisions.
///
/// **Reference**: router_query_engine.py query classification patterns
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    /// High-level overview and summary questions
    Summary,
    /// Detailed implementation and specific questions
    Detailed,
    /// Questions requiring both overview and details
    Hybrid,
    /// Code-specific pattern questions
    CodeSpecific,
}

/// Performance profile for query engines.
#[derive(Debug, Clone)]
pub enum PerformanceProfile {
    /// Fast but less comprehensive
    Fast,
    /// Thorough but potentially slower
    Thorough,
    /// Balanced approach
    Balanced,
}

/// Metadata for query engine tools.
///
/// **Reference**: router_query_engine.py L30-L45 QueryEngineTool concept
#[derive(Debug, Clone)]
pub struct QueryEngineMetadata {
    pub name: String,
    pub description: String,
    pub suitable_for: Vec<QueryType>,
    pub performance_profile: PerformanceProfile,
}

/// Wrapper for query engines with metadata.
///
/// **Reference**: router_query_engine.py QueryEngineTool pattern
#[derive(Debug)]
pub struct QueryEngineWrapper {
    pub engine: Arc<QueryEngine>,
    pub metadata: QueryEngineMetadata,
}

/// Selection result from query selector.
///
/// **Reference**: router_query_engine.py L164 selector result
#[derive(Debug)]
pub struct SelectionResult {
    /// Selected engine index
    pub index: usize,
    /// Multiple selected indices (for multi-engine queries)
    pub indices: Vec<usize>,
    /// Reason for selection
    pub reason: String,
    /// Reasons for multiple selections
    pub reasons: Vec<String>,
}

/// Query analysis result.
#[derive(Debug)]
pub struct QueryAnalysis {
    pub query_type: QueryType,
    pub complexity_level: u8, // 1-5 scale
    pub context_depth: u8,    // 1-5 scale
    pub confidence: f32,      // 0.0-1.0
}

/// Trait for query selection strategies.
///
/// **Reference**: router_query_engine.py L102 selector interface
#[async_trait]
pub trait QuerySelector: Send + Sync + std::fmt::Debug {
    /// Select the most appropriate query engine(s) for a query.
    async fn select(&self, engines: &[QueryEngineWrapper], query: &str) -> Result<SelectionResult>;

    /// Analyze query characteristics.
    async fn analyze_query(&self, query: &str) -> Result<QueryAnalysis>;
}

/// Response summarizer for combining multiple responses.
///
/// **Reference**: router_query_engine.py L107 summarizer
#[async_trait]
pub trait ResponseSummarizer: Send + Sync + std::fmt::Debug {
    /// Combine multiple responses into a single response.
    async fn combine_responses(
        &self,
        responses: Vec<QueryResponse>,
        original_query: &str,
    ) -> Result<QueryResponse>;
}

/// A router query engine that selects appropriate engines for queries.
///
/// This engine analyzes incoming queries and routes them to the most suitable
/// query engine from a set of candidates. It can also combine results from
/// multiple engines when beneficial.
///
/// **Reference**: LlamaIndex RouterQueryEngine
/// - File: `llama-index-core/llama_index/core/query_engine/router_query_engine.py`
/// - Lines: L95-L158
#[derive(Debug)]
pub struct RouterQueryEngine {
    /// Candidate query engines (Reference: L105)
    query_engines: Vec<QueryEngineWrapper>,
    /// Engine selector (Reference: L102)
    selector: Arc<dyn QuerySelector>,
    /// Response summarizer for multi-engine results (Reference: L107)
    summarizer: Option<Arc<dyn ResponseSummarizer>>,
    /// Enable verbose logging
    verbose: bool,
}

impl RouterQueryEngine {
    /// Create a new router query engine.
    ///
    /// **Reference**: router_query_engine.py L119-L158
    pub fn new(
        query_engines: Vec<QueryEngineWrapper>,
        selector: Arc<dyn QuerySelector>,
        summarizer: Option<Arc<dyn ResponseSummarizer>>,
    ) -> Self {
        Self {
            query_engines,
            selector,
            summarizer,
            verbose: false,
        }
    }

    /// Create a builder for configuring the router.
    pub fn builder() -> RouterQueryEngineBuilder {
        RouterQueryEngineBuilder::new()
    }

    /// Execute query with routing logic.
    ///
    /// **Reference**: router_query_engine.py L160-L203
    #[instrument(skip(self), fields(engine = "RouterQueryEngine"))]
    pub async fn query(&self, query: &str) -> Result<QueryResponse> {
        info!("Routing query: {}", query);

        // 1. Select appropriate engine(s) (Reference: L164)
        let selection_result = self.selector.select(&self.query_engines, query).await?;

        // 2. Execute based on selection (Reference: L166-L195)
        if selection_result.indices.len() > 1 {
            // Multi-engine execution (Reference: L167-L178)
            self.execute_multi_engine(query, &selection_result).await
        } else {
            // Single engine execution (Reference: L186-L195)
            self.execute_single_engine(query, &selection_result).await
        }
    }

    /// Execute query on multiple engines and combine results.
    ///
    /// **Reference**: router_query_engine.py L167-L184
    #[instrument(skip(self))]
    async fn execute_multi_engine(
        &self,
        query: &str,
        selection: &SelectionResult,
    ) -> Result<QueryResponse> {
        let mut responses = Vec::new();

        // Execute on each selected engine (Reference: L167-L178)
        for (i, &engine_idx) in selection.indices.iter().enumerate() {
            let engine = &self.query_engines[engine_idx];
            let default_reason = "Selected by router".to_string();
            let reason = selection.reasons.get(i).unwrap_or(&default_reason);

            if self.verbose {
                info!("Executing on engine '{}': {}", engine.metadata.name, reason);
            }

            let response = engine.engine.query(query).await?;
            responses.push(response);
        }

        // Combine responses if summarizer is available (Reference: L179-L184)
        if let Some(summarizer) = &self.summarizer {
            if responses.len() > 1 {
                summarizer.combine_responses(responses, query).await
            } else {
                Ok(responses.into_iter().next().unwrap())
            }
        } else {
            // Return first response if no summarizer
            Ok(responses.into_iter().next().unwrap())
        }
    }

    /// Execute query on a single selected engine.
    ///
    /// **Reference**: router_query_engine.py L186-L195
    #[instrument(skip(self))]
    async fn execute_single_engine(
        &self,
        query: &str,
        selection: &SelectionResult,
    ) -> Result<QueryResponse> {
        let selected_engine = &self.query_engines[selection.index];

        if self.verbose {
            info!(
                "Selected engine '{}': {}",
                selected_engine.metadata.name, selection.reason
            );
        }

        // Execute query on selected engine (Reference: L195)
        let mut response = selected_engine.engine.query(query).await?;

        // Add selection metadata to response (Reference: L198-L200)
        response.query_metadata.insert(
            "selected_engine".to_string(),
            serde_json::Value::String(selected_engine.metadata.name.clone()),
        );
        response.query_metadata.insert(
            "selection_reason".to_string(),
            serde_json::Value::String(selection.reason.clone()),
        );

        Ok(response)
    }

    /// Get available engine names and descriptions.
    pub fn get_engine_info(&self) -> Vec<(String, String)> {
        self.query_engines
            .iter()
            .map(|wrapper| {
                (
                    wrapper.metadata.name.clone(),
                    wrapper.metadata.description.clone(),
                )
            })
            .collect()
    }
}

/// Builder for creating router query engines.
///
/// **Reference**: LlamaIndex builder patterns
#[derive(Debug, Default)]
pub struct RouterQueryEngineBuilder {
    query_engines: Vec<QueryEngineWrapper>,
    selector: Option<Arc<dyn QuerySelector>>,
    summarizer: Option<Arc<dyn ResponseSummarizer>>,
    verbose: bool,
}

impl RouterQueryEngineBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a query engine with metadata.
    pub fn add_engine(mut self, engine: QueryEngine, name: &str, description: &str) -> Self {
        let wrapper = QueryEngineWrapper {
            engine: Arc::new(engine),
            metadata: QueryEngineMetadata {
                name: name.to_string(),
                description: description.to_string(),
                suitable_for: vec![QueryType::Summary, QueryType::Detailed], // Default
                performance_profile: PerformanceProfile::Balanced,
            },
        };
        self.query_engines.push(wrapper);
        self
    }

    /// Add a query engine with full metadata.
    pub fn add_engine_with_metadata(
        mut self,
        engine: QueryEngine,
        metadata: QueryEngineMetadata,
    ) -> Self {
        let wrapper = QueryEngineWrapper {
            engine: Arc::new(engine),
            metadata,
        };
        self.query_engines.push(wrapper);
        self
    }

    /// Set the query selector.
    pub fn selector(mut self, selector: Arc<dyn QuerySelector>) -> Self {
        self.selector = Some(selector);
        self
    }

    /// Set the response summarizer.
    pub fn summarizer(mut self, summarizer: Arc<dyn ResponseSummarizer>) -> Self {
        self.summarizer = Some(summarizer);
        self
    }

    /// Enable verbose logging.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build the router query engine.
    pub fn build(self) -> Result<RouterQueryEngine> {
        if self.query_engines.is_empty() {
            return Err(cheungfun_core::CheungfunError::Configuration {
                message: "At least one query engine is required".to_string(),
            });
        }

        let selector =
            self.selector
                .ok_or_else(|| cheungfun_core::CheungfunError::Configuration {
                    message: "Query selector is required".to_string(),
                })?;

        let mut router = RouterQueryEngine::new(self.query_engines, selector, self.summarizer);
        router.verbose = self.verbose;

        Ok(router)
    }
}
