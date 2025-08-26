//! RAG Query Engine Tool - Deep integration between cheungfun-query and agents
//!
//! This tool bridges our powerful RAG system with the agent framework,
//! enabling agents to perform sophisticated retrieval and generation.

use crate::{
    error::{AgentError, Result},
    tool::{Tool, ToolContext, ToolResult},
    types::ToolSchema,
};
use async_trait::async_trait;
use cheungfun_core::{traits::Retriever, Result as CoreResult};
use cheungfun_query::{
    engine::{QueryEngine, QueryEngineOptions, QueryRewriteStrategy},
    prelude::*,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::HashMap, sync::Arc};

/// Configuration for RAG Query Tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagQueryToolConfig {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Default `top_k` for retrieval
    pub default_top_k: usize,
    /// Enable deep research mode
    pub enable_deep_research: bool,
    /// Enable query rewriting
    pub enable_query_rewriting: bool,
    /// Default search mode
    pub default_search_mode: String,
    /// Maximum research depth
    pub max_research_depth: usize,
}

impl Default for RagQueryToolConfig {
    fn default() -> Self {
        Self {
            name: "rag_query".to_string(),
            description:
                "Advanced RAG query tool with retrieval, generation, and deep research capabilities"
                    .to_string(),
            default_top_k: 5,
            enable_deep_research: true,
            enable_query_rewriting: true,
            default_search_mode: "hybrid".to_string(),
            max_research_depth: 3,
        }
    }
}

/// Input parameters for RAG Query Tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagQueryInput {
    /// The query text
    pub query: String,
    /// Optional search mode override
    pub search_mode: Option<String>,
    /// Optional `top_k` override
    pub top_k: Option<usize>,
    /// Enable deep research for complex questions
    pub deep_research: Option<bool>,
    /// Query rewrite strategy
    pub rewrite_strategy: Option<String>,
    /// Additional metadata filters
    pub filters: Option<HashMap<String, serde_json::Value>>,
}

/// RAG Query Tool that integrates cheungfun-query with agents
#[derive(Debug)]
pub struct RagQueryTool {
    /// The underlying query engine
    query_engine: Arc<QueryEngine>,
    /// Tool configuration
    config: RagQueryToolConfig,
    /// Tool statistics
    stats: std::sync::Arc<std::sync::Mutex<RagQueryStats>>,
}

/// Statistics for RAG Query Tool usage
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RagQueryStats {
    /// Total queries processed
    pub total_queries: u64,
    /// Deep research queries
    pub deep_research_queries: u64,
    /// Rewritten queries
    pub rewritten_queries: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Total nodes retrieved
    pub total_nodes_retrieved: u64,
}

impl RagQueryTool {
    /// Create a new RAG Query Tool
    #[must_use]
    pub fn new(query_engine: Arc<QueryEngine>, config: RagQueryToolConfig) -> Self {
        Self {
            query_engine,
            config,
            stats: std::sync::Arc::new(std::sync::Mutex::new(RagQueryStats::default())),
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_defaults(query_engine: Arc<QueryEngine>) -> Self {
        Self::new(query_engine, RagQueryToolConfig::default())
    }

    /// Create from components
    pub fn from_components(
        retriever: Arc<dyn Retriever>,
        generator: Arc<dyn ResponseGenerator>,
        memory: Option<Arc<tokio::sync::Mutex<dyn BaseMemory>>>,
    ) -> CoreResult<Self> {
        let query_engine = if let Some(mem) = memory {
            Arc::new(
                QueryEngineBuilder::new()
                    .retriever(retriever)
                    .generator(generator)
                    .memory(mem)
                    .build()?,
            )
        } else {
            Arc::new(
                QueryEngineBuilder::new()
                    .retriever(retriever)
                    .generator(generator)
                    .build()?,
            )
        };

        Ok(Self::with_defaults(query_engine))
    }

    /// Parse tool input from arguments
    fn parse_input(&self, args: serde_json::Value) -> Result<RagQueryInput> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::validation("query", "Query text is required"))?
            .to_string();

        let search_mode = args
            .get("search_mode")
            .and_then(|v| v.as_str())
            .map(std::string::ToString::to_string);

        let top_k = args
            .get("top_k")
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as usize);

        let deep_research = args
            .get("deep_research")
            .and_then(serde_json::Value::as_bool);

        let rewrite_strategy = args
            .get("rewrite_strategy")
            .and_then(|v| v.as_str())
            .map(std::string::ToString::to_string);

        let filters = args.get("filters").and_then(|v| v.as_object()).map(|obj| {
            obj.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<HashMap<String, serde_json::Value>>()
        });

        Ok(RagQueryInput {
            query,
            search_mode,
            top_k,
            deep_research,
            rewrite_strategy,
            filters,
        })
    }

    /// Convert search mode string to enum
    fn parse_search_mode(&self, mode_str: &str) -> cheungfun_core::types::SearchMode {
        match mode_str.to_lowercase().as_str() {
            "vector" => cheungfun_core::types::SearchMode::Vector,
            "keyword" => cheungfun_core::types::SearchMode::Keyword,
            "hybrid" => cheungfun_core::types::SearchMode::hybrid(0.5), // Balanced hybrid
            _ => cheungfun_core::types::SearchMode::hybrid(0.5),        // Default fallback
        }
    }

    /// Convert rewrite strategy string to enum
    fn parse_rewrite_strategy(&self, strategy_str: &str) -> QueryRewriteStrategy {
        match strategy_str.to_lowercase().as_str() {
            "clarification" => QueryRewriteStrategy::Clarification,
            "expansion" => QueryRewriteStrategy::Expansion,
            "decomposition" => QueryRewriteStrategy::Decomposition,
            "hyde" => QueryRewriteStrategy::HyDE,
            _ => QueryRewriteStrategy::Expansion, // Default fallback
        }
    }

    /// Execute the RAG query
    async fn execute_query(
        &self,
        input: RagQueryInput,
    ) -> CoreResult<cheungfun_core::types::QueryResponse> {
        let start_time = std::time::Instant::now();

        // Build query options
        let mut options = QueryEngineOptions::new();

        if let Some(top_k) = input.top_k {
            options = options.with_top_k(top_k);
        } else {
            options = options.with_top_k(self.config.default_top_k);
        }

        if let Some(search_mode_str) = input.search_mode {
            let search_mode = self.parse_search_mode(&search_mode_str);
            options = options.with_search_mode(search_mode);
        } else {
            let default_search_mode = self.parse_search_mode(&self.config.default_search_mode);
            options = options.with_search_mode(default_search_mode);
        }

        if let Some(filters) = input.filters {
            for (key, value) in filters {
                options = options.with_filter(key, value);
            }
        }

        // Decide execution strategy
        let has_rewrite_strategy = input.rewrite_strategy.is_some();
        let response = if input.deep_research.unwrap_or(false) && self.config.enable_deep_research {
            // Use deep research for complex queries
            self.query_engine
                .deep_research(
                    &input.query,
                    Some(self.config.max_research_depth),
                    Some(&options),
                )
                .await?
        } else if let Some(rewrite_strategy_str) = input.rewrite_strategy {
            // Use query rewriting
            if self.config.enable_query_rewriting {
                let rewrite_strategy = self.parse_rewrite_strategy(&rewrite_strategy_str);
                self.query_engine
                    .query_with_rewrite(&input.query, rewrite_strategy, Some(&options))
                    .await?
            } else {
                // Fallback to normal query
                self.query_engine
                    .query_with_options(&input.query, &options)
                    .await?
            }
        } else {
            // Standard query
            self.query_engine
                .query_with_options(&input.query, &options)
                .await?
        };

        let execution_time = start_time.elapsed().as_millis() as u64;

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_queries += 1;
            if input.deep_research.unwrap_or(false) {
                stats.deep_research_queries += 1;
            }
            if has_rewrite_strategy {
                stats.rewritten_queries += 1;
            }
            stats.total_nodes_retrieved += response.retrieved_nodes.len() as u64;

            // Update average response time
            let total_time = stats.avg_response_time_ms * (stats.total_queries - 1) as f64
                + execution_time as f64;
            stats.avg_response_time_ms = total_time / stats.total_queries as f64;
        }

        tracing::info!(
            "RAG query completed in {}ms, retrieved {} nodes",
            execution_time,
            response.retrieved_nodes.len()
        );

        Ok(response)
    }

    /// Get current tool statistics
    pub fn get_stats(&self) -> Result<RagQueryStats> {
        self.stats
            .lock()
            .map(|stats| (*stats).clone())
            .map_err(|e| AgentError::execution(format!("Failed to get stats: {e}")))
    }

    /// Reset tool statistics
    pub fn reset_stats(&self) -> Result<()> {
        self.stats
            .lock()
            .map(|mut stats| *stats = RagQueryStats::default())
            .map_err(|e| AgentError::execution(format!("Failed to reset stats: {e}")))
    }
}

#[async_trait]
impl Tool for RagQueryTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.config.name.clone(),
            description: self.config.description.clone(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question to ask"
                    },
                    "search_mode": {
                        "type": "string",
                        "enum": ["vector", "keyword", "hybrid"],
                        "description": "Search mode to use (default: hybrid)"
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Number of results to retrieve (default: 5)"
                    },
                    "deep_research": {
                        "type": "boolean",
                        "description": "Enable deep research mode for complex questions (default: false)"
                    },
                    "rewrite_strategy": {
                        "type": "string",
                        "enum": ["clarification", "expansion", "decomposition", "hyde"],
                        "description": "Query rewriting strategy to improve results"
                    },
                    "filters": {
                        "type": "object",
                        "description": "Additional metadata filters to apply",
                        "additionalProperties": true
                    }
                },
                "required": ["query"]
            }),
            output_schema: None,
            dangerous: false,
            metadata: HashMap::new(),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult> {
        let input = self.parse_input(arguments)?;

        tracing::info!("Executing RAG query: {}", input.query);

        match self.execute_query(input).await {
            Ok(response) => {
                let mut metadata = HashMap::new();

                // Add response metadata
                metadata.insert(
                    "nodes_retrieved".to_string(),
                    json!(response.retrieved_nodes.len()),
                );
                metadata.insert("query_metadata".to_string(), json!(response.query_metadata));

                // Add node details
                let node_details: Vec<serde_json::Value> = response.retrieved_nodes
                    .iter()
                    .take(3) // Show top 3 nodes for context
                    .map(|scored_node| {
                        json!({
                            "score": scored_node.score,
                            "node_id": scored_node.node.id,
                            "content_preview": scored_node.node.content.chars().take(200).collect::<String>(),
                            "metadata": scored_node.node.metadata
                        })
                    })
                    .collect();

                metadata.insert("top_nodes".to_string(), json!(node_details));

                Ok(ToolResult::success_with_metadata(
                    response.response.content,
                    metadata,
                ))
            }
            Err(e) => {
                tracing::error!("RAG query failed: {}", e);
                Ok(ToolResult::error(format!("RAG query failed: {e}")))
            }
        }
    }
}

/// Builder for RAG Query Tool
pub struct RagQueryToolBuilder {
    query_engine: Option<Arc<QueryEngine>>,
    retriever: Option<Arc<dyn Retriever>>,
    generator: Option<Arc<dyn ResponseGenerator>>,
    memory: Option<Arc<tokio::sync::Mutex<dyn BaseMemory>>>,
    config: RagQueryToolConfig,
}

impl RagQueryToolBuilder {
    /// Create a new RAG query tool builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            query_engine: None,
            retriever: None,
            generator: None,
            memory: None,
            config: RagQueryToolConfig::default(),
        }
    }

    /// Set the query engine
    #[must_use]
    pub fn query_engine(mut self, engine: Arc<QueryEngine>) -> Self {
        self.query_engine = Some(engine);
        self
    }

    /// Set the retriever
    pub fn retriever(mut self, retriever: Arc<dyn Retriever>) -> Self {
        self.retriever = Some(retriever);
        self
    }

    /// Set the response generator
    pub fn generator(mut self, generator: Arc<dyn ResponseGenerator>) -> Self {
        self.generator = Some(generator);
        self
    }

    /// Set the memory
    pub fn memory(mut self, memory: Arc<tokio::sync::Mutex<dyn BaseMemory>>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Set the configuration
    #[must_use]
    pub fn config(mut self, config: RagQueryToolConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the tool name
    #[must_use]
    pub fn name(mut self, name: String) -> Self {
        self.config.name = name;
        self
    }

    /// Set the tool description
    #[must_use]
    pub fn description(mut self, description: String) -> Self {
        self.config.description = description;
        self
    }

    /// Enable or disable deep research mode
    #[must_use]
    pub fn enable_deep_research(mut self, enable: bool) -> Self {
        self.config.enable_deep_research = enable;
        self
    }

    /// Enable or disable query rewriting
    #[must_use]
    pub fn enable_query_rewriting(mut self, enable: bool) -> Self {
        self.config.enable_query_rewriting = enable;
        self
    }

    /// Build the RAG query tool
    pub fn build(self) -> Result<RagQueryTool> {
        if let Some(engine) = self.query_engine {
            Ok(RagQueryTool::new(engine, self.config))
        } else if let Some(retriever) = self.retriever {
            let generator = self.generator.ok_or_else(|| {
                AgentError::validation(
                    "generator",
                    "Generator is required when building from components",
                )
            })?;

            let tool =
                RagQueryTool::from_components(retriever, generator, self.memory).map_err(|e| {
                    AgentError::tool("rag_query_tool", format!("Failed to create RAG tool: {e}"))
                })?;

            Ok(tool)
        } else {
            Err(AgentError::validation(
                "query_engine_or_components",
                "Either query_engine or retriever+generator must be provided",
            ))
        }
    }
}

impl Default for RagQueryToolBuilder {
    fn default() -> Self {
        Self::new()
    }
}
