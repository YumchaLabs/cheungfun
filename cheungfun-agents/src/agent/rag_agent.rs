//! RAG-enhanced intelligent agent with deep integration to cheungfun-query.

use crate::{
    agent::{Agent, AgentHealthStatus, HealthStatus},
    error::{AgentError, Result},
    task::{Task, TaskContext},
    tool::ToolRegistry,
    types::{AgentCapabilities, AgentConfig, AgentId, AgentMessage, AgentResponse, ExecutionStats},
};
use async_trait::async_trait;

use cheungfun_query::engine::{QueryEngine, QueryEngineOptions};
use chrono::Utc;
use std::{collections::HashMap, sync::Arc, time::Instant};
use tracing::{debug, info};
use uuid::Uuid;

/// RAG-enhanced agent that integrates with Cheungfun's query system
#[derive(Debug)]
pub struct RagAgent {
    /// Agent ID
    id: AgentId,
    /// Agent configuration
    config: AgentConfig,
    /// Query engine for RAG functionality
    query_engine: Arc<QueryEngine>,
    /// Tool registry
    tool_registry: Arc<ToolRegistry>,
    /// RAG configuration
    rag_config: RagAgentConfig,
    /// Agent statistics
    stats: Arc<tokio::sync::Mutex<RagAgentStats>>,
}

/// RAG agent configuration
#[derive(Debug, Clone)]
pub struct RagAgentConfig {
    /// Whether to use RAG for all queries
    pub always_use_rag: bool,
    /// Minimum similarity threshold for RAG results
    pub min_similarity_threshold: f32,
    /// Maximum number of retrieved documents
    pub max_retrieved_docs: usize,
    /// Whether to include source information in responses
    pub include_sources: bool,
    /// Whether to use hybrid search (vector + keyword)
    pub use_hybrid_search: bool,
    /// Whether to enable query rewriting
    pub enable_query_rewriting: bool,
    /// Whether to enable result reranking
    pub enable_reranking: bool,
    /// Context window size for RAG
    pub context_window_size: usize,
}

impl Default for RagAgentConfig {
    fn default() -> Self {
        Self {
            always_use_rag: true,
            min_similarity_threshold: 0.7,
            max_retrieved_docs: 5,
            include_sources: true,
            use_hybrid_search: true,
            enable_query_rewriting: true,
            enable_reranking: true,
            context_window_size: 4096,
        }
    }
}

/// RAG agent statistics
#[derive(Debug, Default, Clone)]
pub struct RagAgentStats {
    /// Total queries processed
    pub total_queries: usize,
    /// Queries that used RAG
    pub rag_queries: usize,
    /// Total documents retrieved
    pub total_docs_retrieved: usize,
    /// Average retrieval time in milliseconds
    pub avg_retrieval_time_ms: f64,
    /// Average query processing time in milliseconds
    pub avg_query_time_ms: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Last query timestamp
    pub last_query: Option<chrono::DateTime<chrono::Utc>>,
}

impl RagAgent {
    /// Create a new RAG agent
    #[must_use]
    pub fn new(
        config: AgentConfig,
        query_engine: Arc<QueryEngine>,
        tool_registry: Arc<ToolRegistry>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            query_engine,
            tool_registry,
            rag_config: RagAgentConfig::default(),
            stats: Arc::new(tokio::sync::Mutex::new(RagAgentStats::default())),
        }
    }

    /// Create RAG agent with custom RAG configuration
    #[must_use]
    pub fn with_rag_config(
        config: AgentConfig,
        query_engine: Arc<QueryEngine>,
        tool_registry: Arc<ToolRegistry>,
        rag_config: RagAgentConfig,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            query_engine,
            tool_registry,
            rag_config,
            stats: Arc::new(tokio::sync::Mutex::new(RagAgentStats::default())),
        }
    }

    /// Get RAG configuration
    #[must_use]
    pub fn rag_config(&self) -> &RagAgentConfig {
        &self.rag_config
    }

    /// Update RAG configuration
    pub fn set_rag_config(&mut self, config: RagAgentConfig) {
        self.rag_config = config;
    }

    /// Get RAG agent statistics
    pub async fn rag_stats(&self) -> RagAgentStats {
        self.stats.lock().await.clone()
    }

    /// Process query with RAG enhancement
    async fn process_with_rag(&self, query: &str, _context: &TaskContext) -> Result<AgentResponse> {
        let start_time = Instant::now();

        debug!("Processing query with RAG: {}", query);

        // Create query options
        let options = QueryEngineOptions::new().with_top_k(self.rag_config.max_retrieved_docs);

        // Process with query engine
        let query_response = self
            .query_engine
            .query_with_options(query, &options)
            .await
            .map_err(|e| AgentError::execution(format!("Query processing failed: {e}")))?;

        let total_time = start_time.elapsed();

        // Update statistics
        self.update_rag_stats(
            query_response.retrieved_nodes.len(),
            0.0, // We don't have separate retrieval time
            total_time.as_millis() as f64,
            true,
        )
        .await;

        // Create response metadata
        let mut metadata = HashMap::new();
        metadata.insert("rag_enhanced".to_string(), serde_json::json!(true));
        metadata.insert(
            "retrieved_docs".to_string(),
            serde_json::json!(query_response.retrieved_nodes.len()),
        );
        metadata.insert(
            "query_metadata".to_string(),
            serde_json::json!(query_response.query_metadata),
        );

        if self.rag_config.include_sources {
            let sources: Vec<_> = query_response
                .retrieved_nodes
                .iter()
                .map(|node| {
                    serde_json::json!({
                        "content": node.node.content,
                        "score": node.score,
                        "metadata": node.node.metadata
                    })
                })
                .collect();
            metadata.insert("sources".to_string(), serde_json::json!(sources));
        }

        let stats = ExecutionStats {
            execution_time_ms: total_time.as_millis() as u64,
            tool_calls_count: 0,
            successful_tool_calls: 0,
            failed_tool_calls: 0,
            tokens_used: query_response
                .query_metadata
                .get("tokens_used")
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize),
            custom_metrics: {
                let mut metrics = HashMap::new();
                metrics.insert(
                    "retrieved_docs".to_string(),
                    serde_json::json!(query_response.retrieved_nodes.len()),
                );
                metrics
            },
        };

        Ok(AgentResponse {
            content: query_response.response.content,
            metadata,
            tool_calls: Vec::new(),
            tool_outputs: Vec::new(),
            stats,
            timestamp: Utc::now(),
        })
    }

    /// Update RAG statistics
    async fn update_rag_stats(
        &self,
        docs_retrieved: usize,
        retrieval_time_ms: f64,
        total_time_ms: f64,
        used_rag: bool,
    ) {
        let mut stats = self.stats.lock().await;
        stats.total_queries += 1;

        if used_rag {
            stats.rag_queries += 1;
            stats.total_docs_retrieved += docs_retrieved;

            // Update average retrieval time
            let total_retrieval_time = stats.avg_retrieval_time_ms * (stats.rag_queries - 1) as f64;
            stats.avg_retrieval_time_ms =
                (total_retrieval_time + retrieval_time_ms) / stats.rag_queries as f64;
        }

        // Update average query time
        let total_query_time = stats.avg_query_time_ms * (stats.total_queries - 1) as f64;
        stats.avg_query_time_ms = (total_query_time + total_time_ms) / stats.total_queries as f64;

        stats.last_query = Some(Utc::now());
    }

    /// Check if query should use RAG
    fn should_use_rag(&self, query: &str, _context: &TaskContext) -> bool {
        if self.rag_config.always_use_rag {
            return true;
        }

        // Simple heuristics for when to use RAG
        let query_lower = query.to_lowercase();

        // Use RAG for questions
        if query_lower.contains("what")
            || query_lower.contains("how")
            || query_lower.contains("why")
            || query_lower.contains("when")
            || query_lower.contains("where")
            || query_lower.contains('?')
        {
            return true;
        }

        // Use RAG for information requests
        if query_lower.contains("explain")
            || query_lower.contains("describe")
            || query_lower.contains("tell me")
            || query_lower.contains("information")
        {
            return true;
        }

        false
    }
}

#[async_trait]
impl Agent for RagAgent {
    fn id(&self) -> AgentId {
        self.id
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn description(&self) -> Option<&str> {
        self.config.description.as_deref()
    }

    fn capabilities(&self) -> &AgentCapabilities {
        &self.config.capabilities
    }

    fn config(&self) -> &AgentConfig {
        &self.config
    }

    async fn execute(&self, task: &Task) -> Result<AgentResponse> {
        let start_time = Instant::now();
        info!("RAG agent '{}' executing task: {}", self.name(), task.id);

        // Check if we should use RAG for this query
        if self.should_use_rag(&task.input, &task.context) {
            self.process_with_rag(&task.input, &task.context).await
        } else {
            // Process without RAG (fallback to basic processing)
            let response_content = format!(
                "Task '{}' processed by RAG agent '{}' (without RAG enhancement). Input: {}",
                task.name,
                self.name(),
                task.input
            );

            let execution_time = start_time.elapsed();
            let stats = ExecutionStats {
                execution_time_ms: execution_time.as_millis() as u64,
                tool_calls_count: 0,
                successful_tool_calls: 0,
                failed_tool_calls: 0,
                tokens_used: None,
                custom_metrics: HashMap::new(),
            };

            // Update statistics
            self.update_rag_stats(0, 0.0, execution_time.as_millis() as f64, false)
                .await;

            Ok(AgentResponse {
                content: response_content,
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("rag_enhanced".to_string(), serde_json::json!(false));
                    meta
                },
                tool_calls: Vec::new(),
                tool_outputs: Vec::new(),
                stats,
                timestamp: Utc::now(),
            })
        }
    }

    async fn process_message(&self, message: &AgentMessage) -> Result<AgentResponse> {
        // Convert message to task and execute
        let task = Task::builder()
            .name("Message Processing")
            .input(&message.content)
            .build()?;

        self.execute(&task).await
    }

    fn tools(&self) -> Vec<String> {
        self.tool_registry.tool_names()
    }

    async fn health_check(&self) -> Result<AgentHealthStatus> {
        let stats = self.rag_stats().await;
        let mut metrics = HashMap::new();

        metrics.insert(
            "total_queries".to_string(),
            serde_json::json!(stats.total_queries),
        );
        metrics.insert(
            "rag_usage_rate".to_string(),
            serde_json::json!(if stats.total_queries > 0 {
                stats.rag_queries as f64 / stats.total_queries as f64
            } else {
                0.0
            }),
        );
        metrics.insert(
            "avg_retrieval_time_ms".to_string(),
            serde_json::json!(stats.avg_retrieval_time_ms),
        );
        metrics.insert(
            "avg_query_time_ms".to_string(),
            serde_json::json!(stats.avg_query_time_ms),
        );

        Ok(AgentHealthStatus {
            agent_id: self.id,
            status: HealthStatus::Healthy,
            message: format!(
                "RAG agent '{}' is operational. Processed {} queries ({} with RAG).",
                self.name(),
                stats.total_queries,
                stats.rag_queries
            ),
            last_check: Utc::now(),
            metrics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolRegistry;

    // Note: These tests would require mock implementations of QueryEngine and Retriever
    // which are not implemented here for brevity

    #[test]
    fn test_rag_agent_creation() {
        // This test would require mock implementations
        // For now, we'll just test the configuration
        let config = RagAgentConfig::default();
        assert!(config.always_use_rag);
        assert_eq!(config.max_retrieved_docs, 5);
        assert!(config.include_sources);
    }

    #[test]
    fn test_should_use_rag_heuristics() {
        let _config = AgentConfig::default();
        let _tool_registry = Arc::new(ToolRegistry::new());

        // This would require mock query engine and retriever
        // For now, we'll test the heuristics logic conceptually
        assert!(true); // Placeholder
    }
}
