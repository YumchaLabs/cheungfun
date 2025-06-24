//! Query engine implementations for high-level query processing.
//!
//! This module provides the main query engine that combines retrievers
//! and generators to provide a unified interface for RAG operations.

use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, instrument};

use cheungfun_core::{
    Result,
    traits::{ResponseGenerator, Retriever},
    types::{GenerationOptions, Query, QueryResponse, ScoredNode},
};

/// A high-level query engine that combines retrieval and generation.
///
/// The query engine orchestrates the complete RAG process:
/// 1. Takes a user query
/// 2. Uses a retriever to find relevant context
/// 3. Uses a generator to create a response based on the context
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_query::engine::QueryEngine;
/// use cheungfun_core::prelude::*;
///
/// # async fn example() -> Result<()> {
/// let engine = QueryEngine::builder()
///     .retriever(retriever)
///     .generator(generator)
///     .build()?;
///
/// let response = engine.query("What is machine learning?").await?;
/// println!("Answer: {}", response.content);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct QueryEngine {
    /// Retriever for finding relevant context.
    retriever: Arc<dyn Retriever>,

    /// Generator for creating responses.
    generator: Arc<dyn ResponseGenerator>,

    /// Configuration for query processing.
    config: QueryEngineConfig,
}

/// Configuration for query engine operations.
#[derive(Debug, Clone)]
pub struct QueryEngineConfig {
    /// Default number of context nodes to retrieve.
    pub default_top_k: usize,

    /// Default generation options.
    pub default_generation_options: GenerationOptions,

    /// Whether to validate retrieved context before generation.
    pub validate_context: bool,

    /// Minimum number of context nodes required for generation.
    pub min_context_nodes: usize,

    /// Maximum number of context nodes to use for generation.
    pub max_context_nodes: usize,

    /// Whether to enable query preprocessing.
    pub enable_query_preprocessing: bool,

    /// Whether to enable response postprocessing.
    pub enable_response_postprocessing: bool,
}

impl Default for QueryEngineConfig {
    fn default() -> Self {
        Self {
            default_top_k: 5,
            default_generation_options: GenerationOptions::default(),
            validate_context: true,
            min_context_nodes: 1,
            max_context_nodes: 10,
            enable_query_preprocessing: true,
            enable_response_postprocessing: true,
        }
    }
}

impl QueryEngine {
    /// Create a new query engine.
    pub fn new(retriever: Arc<dyn Retriever>, generator: Arc<dyn ResponseGenerator>) -> Self {
        Self {
            retriever,
            generator,
            config: QueryEngineConfig::default(),
        }
    }

    /// Create a new query engine with custom configuration.
    pub fn with_config(
        retriever: Arc<dyn Retriever>,
        generator: Arc<dyn ResponseGenerator>,
        config: QueryEngineConfig,
    ) -> Self {
        Self {
            retriever,
            generator,
            config,
        }
    }

    /// Create a builder for constructing query engines.
    #[must_use]
    pub fn builder() -> QueryEngineBuilder {
        QueryEngineBuilder::new()
    }

    /// Execute a query and return a response.
    ///
    /// This is the main method that orchestrates the complete RAG process.
    #[instrument(skip(self), fields(engine = "QueryEngine"))]
    pub async fn query(&self, query_text: &str) -> Result<QueryResponse> {
        self.query_with_options(query_text, &QueryEngineOptions::default())
            .await
    }

    /// Execute a query with custom options.
    #[instrument(skip(self), fields(engine = "QueryEngine"))]
    pub async fn query_with_options(
        &self,
        query_text: &str,
        options: &QueryEngineOptions,
    ) -> Result<QueryResponse> {
        info!("Processing query: {}", query_text);

        // Build query object
        let mut query = Query::new(query_text);
        query.top_k = options.top_k.unwrap_or(self.config.default_top_k);

        // Apply any additional query options
        if let Some(search_mode) = &options.search_mode {
            query.search_mode = search_mode.clone();
        }

        for (key, value) in &options.filters {
            query.filters.insert(key.clone(), value.clone());
        }

        // Preprocess query if enabled
        if self.config.enable_query_preprocessing {
            // TODO: Implement query preprocessing (query expansion, spell correction, etc.)
            debug!("Query preprocessing enabled but not yet implemented");
        }

        // Retrieve relevant context
        debug!("Retrieving context for query");
        let mut retrieved_nodes = self.retriever.retrieve(&query).await?;
        info!("Retrieved {} context nodes", retrieved_nodes.len());

        // Validate context if enabled
        if self.config.validate_context {
            self.validate_retrieved_context(&retrieved_nodes)?;
        }

        // Limit context nodes
        if retrieved_nodes.len() > self.config.max_context_nodes {
            retrieved_nodes.truncate(self.config.max_context_nodes);
            debug!(
                "Truncated context to {} nodes",
                self.config.max_context_nodes
            );
        }

        // Prepare generation options
        let generation_options = options
            .generation_options
            .as_ref()
            .unwrap_or(&self.config.default_generation_options);

        // Generate response
        debug!("Generating response");
        let generated_response = self
            .generator
            .generate_response(query_text, retrieved_nodes.clone(), generation_options)
            .await?;

        // Postprocess response if enabled
        let final_response = if self.config.enable_response_postprocessing {
            // TODO: Implement response postprocessing (fact checking, formatting, etc.)
            debug!("Response postprocessing enabled but not yet implemented");
            generated_response
        } else {
            generated_response
        };

        // Build query metadata
        let mut query_metadata = HashMap::new();
        query_metadata.insert(
            "retriever".to_string(),
            serde_json::Value::String(self.retriever.name().to_string()),
        );
        query_metadata.insert(
            "generator".to_string(),
            serde_json::Value::String(self.generator.name().to_string()),
        );
        query_metadata.insert(
            "context_nodes_used".to_string(),
            serde_json::Value::Number(retrieved_nodes.len().into()),
        );

        info!("Query processing completed successfully");

        Ok(QueryResponse {
            response: final_response,
            retrieved_nodes,
            query_metadata,
        })
    }

    /// Validate that retrieved context meets minimum requirements.
    fn validate_retrieved_context(&self, nodes: &[ScoredNode]) -> Result<()> {
        if nodes.len() < self.config.min_context_nodes {
            return Err(cheungfun_core::CheungfunError::Validation {
                message: format!(
                    "Insufficient context: got {} nodes, minimum required: {}",
                    nodes.len(),
                    self.config.min_context_nodes
                ),
            });
        }
        Ok(())
    }

    /// Get the retriever used by this engine.
    #[must_use]
    pub fn retriever(&self) -> &Arc<dyn Retriever> {
        &self.retriever
    }

    /// Get the generator used by this engine.
    #[must_use]
    pub fn generator(&self) -> &Arc<dyn ResponseGenerator> {
        &self.generator
    }

    /// Get the configuration of this engine.
    #[must_use]
    pub fn config(&self) -> &QueryEngineConfig {
        &self.config
    }

    /// Perform a health check on all components.
    pub async fn health_check(&self) -> Result<()> {
        self.retriever.health_check().await?;
        self.generator.health_check().await?;
        Ok(())
    }
}

/// Options for query execution.
#[derive(Debug, Clone, Default)]
pub struct QueryEngineOptions {
    /// Number of context nodes to retrieve.
    pub top_k: Option<usize>,

    /// Search mode to use.
    pub search_mode: Option<cheungfun_core::types::SearchMode>,

    /// Metadata filters to apply.
    pub filters: HashMap<String, serde_json::Value>,

    /// Generation options.
    pub generation_options: Option<GenerationOptions>,
}

impl QueryEngineOptions {
    /// Create new query engine options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of context nodes to retrieve.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set the search mode.
    #[must_use]
    pub fn with_search_mode(mut self, search_mode: cheungfun_core::types::SearchMode) -> Self {
        self.search_mode = Some(search_mode);
        self
    }

    /// Add a metadata filter.
    pub fn with_filter<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.filters.insert(key.into(), value.into());
        self
    }

    /// Set generation options.
    #[must_use]
    pub fn with_generation_options(mut self, options: GenerationOptions) -> Self {
        self.generation_options = Some(options);
        self
    }
}

/// Builder for creating query engines.
#[derive(Debug, Default)]
pub struct QueryEngineBuilder {
    retriever: Option<Arc<dyn Retriever>>,
    generator: Option<Arc<dyn ResponseGenerator>>,
    config: Option<QueryEngineConfig>,
}

impl QueryEngineBuilder {
    /// Create a new builder.
    #[must_use]
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

    /// Set the configuration.
    #[must_use]
    pub fn config(mut self, config: QueryEngineConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the query engine.
    pub fn build(self) -> Result<QueryEngine> {
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

        let config = self.config.unwrap_or_default();

        Ok(QueryEngine::with_config(retriever, generator, config))
    }
}
