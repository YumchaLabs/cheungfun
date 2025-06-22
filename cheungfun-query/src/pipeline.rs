//! Query pipeline implementations for complete RAG workflows.
//!
//! This module provides high-level pipeline implementations that orchestrate
//! the complete query processing workflow with advanced features like
//! context management, conversation history, and streaming responses.

use async_trait::async_trait;
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tracing::{debug, info, instrument};

use cheungfun_core::{
    Result,
    traits::{QueryPipeline, ResponseGenerator, Retriever},
    types::{ChatMessage, GenerationOptions, Query, QueryResponse, RetrievalContext, SearchMode},
};

use crate::engine::{QueryEngine, QueryEngineConfig, QueryEngineOptions};

/// Default implementation of the QueryPipeline trait.
///
/// This pipeline provides a complete RAG workflow with support for:
/// - Context-aware retrieval
/// - Conversation history management
/// - Streaming responses
/// - Query preprocessing and response postprocessing
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_query::pipeline::DefaultQueryPipeline;
/// use cheungfun_core::prelude::*;
///
/// # async fn example() -> Result<()> {
/// let pipeline = DefaultQueryPipeline::builder()
///     .retriever(retriever)
///     .generator(generator)
///     .build()?;
///
/// let options = QueryOptions::default();
/// let response = pipeline.query("What is machine learning?", &options).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct DefaultQueryPipeline {
    /// Underlying query engine.
    engine: QueryEngine,

    /// Configuration for pipeline operations.
    config: QueryPipelineConfig,
}

/// Configuration for query pipeline operations.
#[derive(Debug, Clone)]
pub struct QueryPipelineConfig {
    /// Whether to enable conversation history tracking.
    pub enable_conversation_history: bool,

    /// Maximum number of conversation turns to keep in context.
    pub max_conversation_turns: usize,

    /// Whether to enable query rewriting based on conversation history.
    pub enable_query_rewriting: bool,

    /// Whether to enable context compression for long conversations.
    pub enable_context_compression: bool,

    /// Maximum total context length (in characters).
    pub max_total_context_length: usize,

    /// Whether to enable response caching.
    pub enable_response_caching: bool,

    /// Cache TTL in seconds.
    pub cache_ttl_seconds: u64,
}

impl Default for QueryPipelineConfig {
    fn default() -> Self {
        Self {
            enable_conversation_history: true,
            max_conversation_turns: 10,
            enable_query_rewriting: false,
            enable_context_compression: false,
            max_total_context_length: 16000,
            enable_response_caching: false,
            cache_ttl_seconds: 3600,
        }
    }
}

impl DefaultQueryPipeline {
    /// Create a new query pipeline.
    pub fn new(retriever: Arc<dyn Retriever>, generator: Arc<dyn ResponseGenerator>) -> Self {
        let engine = QueryEngine::new(retriever, generator);
        Self {
            engine,
            config: QueryPipelineConfig::default(),
        }
    }

    /// Create a new query pipeline with custom configuration.
    pub fn with_config(
        retriever: Arc<dyn Retriever>,
        generator: Arc<dyn ResponseGenerator>,
        engine_config: QueryEngineConfig,
        pipeline_config: QueryPipelineConfig,
    ) -> Self {
        let engine = QueryEngine::with_config(retriever, generator, engine_config);
        Self {
            engine,
            config: pipeline_config,
        }
    }

    /// Create a builder for constructing query pipelines.
    pub fn builder() -> QueryPipelineBuilder {
        QueryPipelineBuilder::new()
    }

    /// Process conversation history to extract relevant context.
    fn process_conversation_history(&self, history: &[ChatMessage]) -> String {
        if !self.config.enable_conversation_history || history.is_empty() {
            return String::new();
        }

        let mut context = String::new();
        let recent_messages = history
            .iter()
            .rev()
            .take(self.config.max_conversation_turns * 2) // User + Assistant pairs
            .collect::<Vec<_>>();

        for message in recent_messages.iter().rev() {
            let role = match message.role {
                cheungfun_core::types::MessageRole::User => "User",
                cheungfun_core::types::MessageRole::Assistant => "Assistant",
                cheungfun_core::types::MessageRole::System => "System",
                cheungfun_core::types::MessageRole::Tool => "Tool",
            };
            context.push_str(&format!("{}: {}\n", role, message.content));
        }

        context
    }

    /// Rewrite query based on conversation context.
    fn rewrite_query_with_context(&self, query: &str, context: &RetrievalContext) -> String {
        if !self.config.enable_query_rewriting || context.chat_history.is_empty() {
            return query.to_string();
        }

        // TODO: Implement sophisticated query rewriting
        // For now, just append recent context if the query seems incomplete
        if query.len() < 10 || !query.contains('?') {
            let recent_context = self.process_conversation_history(&context.chat_history);
            if !recent_context.is_empty() {
                return format!("{}\n\nPrevious conversation:\n{}", query, recent_context);
            }
        }

        query.to_string()
    }

    /// Compress context if it exceeds maximum length.
    fn compress_context_if_needed(&self, context: &str) -> String {
        if !self.config.enable_context_compression {
            return context.to_string();
        }

        if context.len() <= self.config.max_total_context_length {
            return context.to_string();
        }

        // Simple compression: take first and last parts
        let target_length = self.config.max_total_context_length;
        let half_length = target_length / 2;

        let start = &context[..half_length.min(context.len())];
        let end_start = context.len().saturating_sub(half_length);
        let end = &context[end_start..];

        format!("{}...[content truncated]...{}", start, end)
    }
}

#[async_trait]
impl QueryPipeline for DefaultQueryPipeline {
    #[instrument(skip(self), fields(pipeline = "DefaultQueryPipeline"))]
    async fn query(
        &self,
        query: &str,
        _options: &cheungfun_core::QueryOptions,
    ) -> Result<QueryResponse> {
        info!("Processing query through pipeline: {}", query);

        // For now, use default options since core QueryOptions is a placeholder
        let default_context = RetrievalContext::default();
        let conversation_context = self.process_conversation_history(&default_context.chat_history);
        debug!(
            "Processed conversation context: {} characters",
            conversation_context.len()
        );

        // Rewrite query if needed
        let processed_query = self.rewrite_query_with_context(query, &default_context);
        debug!("Query after rewriting: {}", processed_query);

        // Build engine options with defaults
        let engine_options = QueryEngineOptions::new();

        // Execute query through engine
        let response = self
            .engine
            .query_with_options(&processed_query, &engine_options)
            .await?;

        info!("Pipeline query processing completed");
        Ok(response)
    }

    #[instrument(skip(self), fields(pipeline = "DefaultQueryPipeline"))]
    async fn query_stream(
        &self,
        query: &str,
        _options: &cheungfun_core::QueryOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        info!("Processing streaming query through pipeline: {}", query);

        // For now, use default options since core QueryOptions is a placeholder
        let default_context = RetrievalContext::default();
        let conversation_context = self.process_conversation_history(&default_context.chat_history);
        debug!(
            "Processed conversation context: {} characters",
            conversation_context.len()
        );

        // Rewrite query if needed
        let processed_query = self.rewrite_query_with_context(query, &default_context);
        debug!("Query after rewriting: {}", processed_query);

        // Build query object for retrieval with defaults
        let retrieval_query = Query::new(&processed_query);

        // Retrieve context
        let retrieved_nodes = self.engine.retriever().retrieve(&retrieval_query).await?;
        debug!(
            "Retrieved {} context nodes for streaming",
            retrieved_nodes.len()
        );

        // Generate streaming response with default options
        let default_generation_options = GenerationOptions::default();
        let stream = self
            .engine
            .generator()
            .generate_response_stream(
                &processed_query,
                retrieved_nodes,
                &default_generation_options,
            )
            .await?;

        info!("Pipeline streaming query processing initiated");
        Ok(stream)
    }

    fn name(&self) -> &'static str {
        "DefaultQueryPipeline"
    }

    fn validate(&self) -> Result<()> {
        // Validate engine components
        // TODO: Add more sophisticated validation
        Ok(())
    }

    fn config(&self) -> HashMap<String, serde_json::Value> {
        let mut config = HashMap::new();
        config.insert(
            "enable_conversation_history".to_string(),
            self.config.enable_conversation_history.into(),
        );
        config.insert(
            "max_conversation_turns".to_string(),
            self.config.max_conversation_turns.into(),
        );
        config.insert(
            "enable_query_rewriting".to_string(),
            self.config.enable_query_rewriting.into(),
        );
        config.insert(
            "enable_context_compression".to_string(),
            self.config.enable_context_compression.into(),
        );
        config.insert(
            "max_total_context_length".to_string(),
            self.config.max_total_context_length.into(),
        );
        config.insert(
            "enable_response_caching".to_string(),
            self.config.enable_response_caching.into(),
        );
        config
    }

    async fn health_check(&self) -> Result<()> {
        self.engine.health_check().await
    }
}

/// Options for query execution in pipelines.
#[derive(Debug, Clone)]
pub struct QueryOptions {
    /// Retrieval options.
    pub retrieval_options: RetrievalOptions,

    /// Generation options.
    pub generation_options: GenerationOptions,

    /// Context for the query.
    pub context: RetrievalContext,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            retrieval_options: RetrievalOptions::default(),
            generation_options: GenerationOptions::default(),
            context: RetrievalContext::default(),
        }
    }
}

/// Options for retrieval operations.
#[derive(Debug, Clone)]
pub struct RetrievalOptions {
    /// Number of results to retrieve.
    pub top_k: Option<usize>,

    /// Search mode to use.
    pub search_mode: SearchMode,

    /// Metadata filters.
    pub filters: HashMap<String, serde_json::Value>,

    /// Similarity threshold.
    pub similarity_threshold: Option<f32>,
}

impl Default for RetrievalOptions {
    fn default() -> Self {
        Self {
            top_k: None,
            search_mode: SearchMode::Vector,
            filters: HashMap::new(),
            similarity_threshold: None,
        }
    }
}

/// Builder for creating query pipelines.
#[derive(Debug, Default)]
pub struct QueryPipelineBuilder {
    retriever: Option<Arc<dyn Retriever>>,
    generator: Option<Arc<dyn ResponseGenerator>>,
    engine_config: Option<QueryEngineConfig>,
    pipeline_config: Option<QueryPipelineConfig>,
}

impl QueryPipelineBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the retriever.
    pub fn retriever(mut self, retriever: Arc<dyn Retriever>) -> Self {
        self.retriever = Some(retriever);
        self
    }

    /// Set the generator.
    pub fn generator(mut self, generator: Arc<dyn ResponseGenerator>) -> Self {
        self.generator = Some(generator);
        self
    }

    /// Set the engine configuration.
    pub fn engine_config(mut self, config: QueryEngineConfig) -> Self {
        self.engine_config = Some(config);
        self
    }

    /// Set the pipeline configuration.
    pub fn pipeline_config(mut self, config: QueryPipelineConfig) -> Self {
        self.pipeline_config = Some(config);
        self
    }

    /// Build the query pipeline.
    pub fn build(self) -> Result<DefaultQueryPipeline> {
        let retriever =
            self.retriever
                .ok_or_else(|| cheungfun_core::CheungfunError::Configuration {
                    message: "Retriever is required".to_string(),
                })?;

        let generator =
            self.generator
                .ok_or_else(|| cheungfun_core::CheungfunError::Configuration {
                    message: "Generator is required".to_string(),
                })?;

        let engine_config = self.engine_config.unwrap_or_default();
        let pipeline_config = self.pipeline_config.unwrap_or_default();

        Ok(DefaultQueryPipeline::with_config(
            retriever,
            generator,
            engine_config,
            pipeline_config,
        ))
    }
}
