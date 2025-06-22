//! Retriever implementations for finding relevant nodes.
//!
//! This module provides concrete implementations of the `Retriever` trait
//! for different search strategies and data sources.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

use cheungfun_core::{
    traits::{Embedder, Retriever, VectorStore},
    types::{Query, SearchMode, ScoredNode},
    Result,
};

/// A retriever that uses vector stores for similarity search.
///
/// This retriever supports multiple search modes:
/// - Vector search using dense embeddings
/// - Keyword search using sparse embeddings (if supported)
/// - Hybrid search combining both approaches
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_query::retriever::VectorRetriever;
/// use cheungfun_core::prelude::*;
///
/// # async fn example() -> Result<()> {
/// let retriever = VectorRetriever::builder()
///     .vector_store(vector_store)
///     .embedder(embedder)
///     .build()?;
///
/// let query = Query::new("What is machine learning?");
/// let results = retriever.retrieve(&query).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct VectorRetriever {
    /// Vector store for similarity search.
    vector_store: Arc<dyn VectorStore>,
    
    /// Embedder for generating query embeddings.
    embedder: Arc<dyn Embedder>,
    
    /// Configuration for retrieval operations.
    config: VectorRetrieverConfig,
}

/// Configuration for vector retriever.
#[derive(Debug, Clone)]
pub struct VectorRetrieverConfig {
    /// Default number of results to return.
    pub default_top_k: usize,
    
    /// Maximum number of results allowed.
    pub max_top_k: usize,
    
    /// Default similarity threshold.
    pub default_similarity_threshold: Option<f32>,
    
    /// Whether to enable query expansion.
    pub enable_query_expansion: bool,
    
    /// Whether to enable result reranking.
    pub enable_reranking: bool,
    
    /// Timeout for retrieval operations.
    pub timeout_seconds: Option<u64>,
}

impl Default for VectorRetrieverConfig {
    fn default() -> Self {
        Self {
            default_top_k: 10,
            max_top_k: 100,
            default_similarity_threshold: None,
            enable_query_expansion: false,
            enable_reranking: false,
            timeout_seconds: Some(30),
        }
    }
}

impl VectorRetriever {
    /// Create a new vector retriever.
    pub fn new(
        vector_store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        Self {
            vector_store,
            embedder,
            config: VectorRetrieverConfig::default(),
        }
    }

    /// Create a new vector retriever with custom configuration.
    pub fn with_config(
        vector_store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedder>,
        config: VectorRetrieverConfig,
    ) -> Self {
        Self {
            vector_store,
            embedder,
            config,
        }
    }

    /// Create a builder for constructing vector retrievers.
    pub fn builder() -> VectorRetrieverBuilder {
        VectorRetrieverBuilder::new()
    }

    /// Validate and adjust query parameters.
    fn validate_query(&self, query: &Query) -> Query {
        let mut adjusted_query = query.clone();
        
        // Enforce max_top_k limit
        if adjusted_query.top_k > self.config.max_top_k {
            warn!(
                "Query top_k {} exceeds maximum {}, adjusting",
                adjusted_query.top_k, self.config.max_top_k
            );
            adjusted_query.top_k = self.config.max_top_k;
        }
        
        // Apply default similarity threshold if not set
        if adjusted_query.similarity_threshold.is_none() {
            adjusted_query.similarity_threshold = self.config.default_similarity_threshold;
        }
        
        adjusted_query
    }

    /// Generate embedding for query if not already present.
    async fn ensure_query_embedding(&self, query: &mut Query) -> Result<()> {
        if query.embedding.is_none() {
            debug!("Generating embedding for query: {}", query.text);
            let embedding = self.embedder.embed(&query.text).await?;
            query.embedding = Some(embedding);
        }
        Ok(())
    }

    /// Perform vector search.
    async fn vector_search(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        debug!("Performing vector search with top_k: {}", query.top_k);
        self.vector_store.search(query).await
    }

    /// Perform keyword search (placeholder for now).
    async fn keyword_search(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        debug!("Performing keyword search (placeholder)");
        // For now, fall back to vector search
        // TODO: Implement actual keyword search when sparse embeddings are available
        self.vector_search(query).await
    }

    /// Perform hybrid search combining vector and keyword results.
    async fn hybrid_search(&self, query: &Query, alpha: f32) -> Result<Vec<ScoredNode>> {
        debug!("Performing hybrid search with alpha: {}", alpha);
        
        // Get results from both search methods
        let vector_results = self.vector_search(query).await?;
        let keyword_results = self.keyword_search(query).await?;
        
        // Combine and rerank results
        self.combine_search_results(vector_results, keyword_results, alpha, query.top_k)
    }

    /// Combine results from vector and keyword search.
    fn combine_search_results(
        &self,
        vector_results: Vec<ScoredNode>,
        keyword_results: Vec<ScoredNode>,
        alpha: f32,
        top_k: usize,
    ) -> Result<Vec<ScoredNode>> {
        let mut combined_scores: HashMap<uuid::Uuid, f32> = HashMap::new();
        let mut node_map: HashMap<uuid::Uuid, ScoredNode> = HashMap::new();
        
        // Add vector search scores with alpha weight
        for scored_node in vector_results {
            let node_id = scored_node.node.id;
            combined_scores.insert(node_id, scored_node.score * alpha);
            node_map.insert(node_id, scored_node);
        }
        
        // Add keyword search scores with (1-alpha) weight
        for scored_node in keyword_results {
            let node_id = scored_node.node.id;
            let keyword_score = scored_node.score * (1.0 - alpha);
            
            if let Some(existing_score) = combined_scores.get_mut(&node_id) {
                *existing_score += keyword_score;
            } else {
                combined_scores.insert(node_id, keyword_score);
                node_map.insert(node_id, scored_node);
            }
        }
        
        // Sort by combined score and take top_k
        let mut combined_results: Vec<_> = combined_scores
            .into_iter()
            .filter_map(|(node_id, score)| {
                node_map.remove(&node_id).map(|mut scored_node| {
                    scored_node.score = score;
                    scored_node
                })
            })
            .collect();
        
        combined_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        combined_results.truncate(top_k);
        
        Ok(combined_results)
    }
}

#[async_trait]
impl Retriever for VectorRetriever {
    #[instrument(skip(self), fields(retriever = "VectorRetriever"))]
    async fn retrieve(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        info!("Starting retrieval for query: {}", query.text);
        
        // Validate and adjust query parameters
        let mut adjusted_query = self.validate_query(query);
        
        // Ensure query has embedding
        self.ensure_query_embedding(&mut adjusted_query).await?;
        
        // Perform search based on search mode
        let results = match &adjusted_query.search_mode {
            SearchMode::Vector => self.vector_search(&adjusted_query).await?,
            SearchMode::Keyword => self.keyword_search(&adjusted_query).await?,
            SearchMode::Hybrid { alpha } => self.hybrid_search(&adjusted_query, *alpha).await?,
        };
        
        info!("Retrieved {} results", results.len());
        Ok(results)
    }

    fn name(&self) -> &'static str {
        "VectorRetriever"
    }

    async fn health_check(&self) -> Result<()> {
        self.vector_store.health_check().await
    }

    fn config(&self) -> HashMap<String, serde_json::Value> {
        let mut config = HashMap::new();
        config.insert("default_top_k".to_string(), self.config.default_top_k.into());
        config.insert("max_top_k".to_string(), self.config.max_top_k.into());
        config.insert("enable_query_expansion".to_string(), self.config.enable_query_expansion.into());
        config.insert("enable_reranking".to_string(), self.config.enable_reranking.into());
        config
    }
}

/// Builder for creating vector retrievers.
#[derive(Debug, Default)]
pub struct VectorRetrieverBuilder {
    vector_store: Option<Arc<dyn VectorStore>>,
    embedder: Option<Arc<dyn Embedder>>,
    config: Option<VectorRetrieverConfig>,
}

impl VectorRetrieverBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the vector store.
    pub fn vector_store(mut self, vector_store: Arc<dyn VectorStore>) -> Self {
        self.vector_store = Some(vector_store);
        self
    }

    /// Set the embedder.
    pub fn embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Set the configuration.
    pub fn config(mut self, config: VectorRetrieverConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the vector retriever.
    pub fn build(self) -> Result<VectorRetriever> {
        let vector_store = self.vector_store.ok_or_else(|| {
            cheungfun_core::CheungfunError::Configuration {
                message: "Vector store is required".to_string(),
            }
        })?;

        let embedder = self.embedder.ok_or_else(|| {
            cheungfun_core::CheungfunError::Configuration {
                message: "Embedder is required".to_string(),
            }
        })?;

        let config = self.config.unwrap_or_default();

        Ok(VectorRetriever::with_config(vector_store, embedder, config))
    }
}
