// Cheungfun Advanced Retrieval Module
// Advanced Retrieval Module

/// Caching layer for improved performance.
pub mod cache;
/// Result fusion algorithms for combining multiple search results.
pub mod fusion;
/// Advanced retrieval pipeline orchestrating all components.
pub mod pipeline;
/// Query transformation algorithms including `HyDE` and sub-question generation.
pub mod query_transformers;
/// Result reranking algorithms for improving search relevance.
pub mod rerankers;
/// Response transformation and post-processing capabilities.
pub mod response_transformers;
/// Advanced search strategies including hybrid search and fusion methods.
pub mod search_strategies;

// Re-export core types
pub use cache::*;
pub use fusion::*;
pub use pipeline::*;
pub use query_transformers::*;
pub use rerankers::*;
pub use response_transformers::*;
pub use search_strategies::*;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

use cheungfun_core::{ScoredNode, VectorStore};

/// An extended query structure that supports various query forms.
#[derive(Debug, Clone)]
pub struct AdvancedQuery {
    /// Query ID.
    pub id: Uuid,
    /// Original query text.
    pub original_text: String,
    /// List of transformed query texts (supports multiple queries).
    pub transformed_queries: Vec<String>,
    /// Query embedding vector.
    pub embedding: Option<Vec<f32>>,
    /// Sparse embedding (for hybrid search).
    pub sparse_embedding: Option<HashMap<u32, f32>>,
    /// Query metadata.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Search parameters.
    pub search_params: SearchParams,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
}

impl AdvancedQuery {
    /// Creates a query from text.
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            original_text: text.into(),
            transformed_queries: Vec::new(),
            embedding: None,
            sparse_embedding: None,
            metadata: HashMap::new(),
            search_params: SearchParams::default(),
            created_at: Utc::now(),
        }
    }

    /// Adds a transformed query.
    pub fn add_transformed_query(&mut self, query: String) {
        self.transformed_queries.push(query);
    }

    /// Sets the embedding vector.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Sets the search parameters.
    #[must_use]
    pub fn with_search_params(mut self, params: SearchParams) -> Self {
        self.search_params = params;
        self
    }

    /// Gets all query texts (original + transformed).
    #[must_use]
    pub fn all_queries(&self) -> Vec<&str> {
        let mut queries = vec![self.original_text.as_str()];
        queries.extend(
            self.transformed_queries
                .iter()
                .map(std::string::String::as_str),
        );
        queries
    }
}

/// Search parameter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    /// Number of results to return.
    pub top_k: usize,
    /// Similarity threshold.
    pub similarity_threshold: Option<f32>,
    /// Search mode.
    pub search_mode: SearchMode,
    /// Filter conditions.
    pub filters: HashMap<String, serde_json::Value>,
    /// Timeout setting.
    pub timeout: Option<Duration>,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            top_k: 10,
            similarity_threshold: None,
            search_mode: SearchMode::Vector,
            filters: HashMap::new(),
            timeout: Some(Duration::from_secs(30)),
        }
    }
}

/// Enum for search modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchMode {
    /// Pure vector search.
    Vector,
    /// Pure keyword search.
    Keyword,
    /// Hybrid search.
    Hybrid {
        /// Vector search weight (0.0-1.0).
        vector_weight: f32,
        /// Keyword search weight (0.0-1.0).
        keyword_weight: f32,
        /// Fusion algorithm.
        fusion_method: FusionMethod,
    },
}

/// Enum for fusion methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion.
    ReciprocalRankFusion {
        /// RRF smoothing parameter, typically 60.0
        k: f32,
    },
    /// Weighted Average.
    WeightedAverage,
    /// Linear Combination.
    LinearCombination,
    /// Custom fusion function.
    Custom(String),
}

/// Structure for a retrieval response.
#[derive(Debug, Clone)]
pub struct RetrievalResponse {
    /// Retrieved nodes.
    pub nodes: Vec<ScoredNode>,
    /// Query information.
    pub query: AdvancedQuery,
    /// Retrieval metadata.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Retrieval statistics.
    pub stats: RetrievalStats,
}

/// Retrieval statistics.
#[derive(Debug, Clone, Default)]
pub struct RetrievalStats {
    /// Retrieval duration.
    pub retrieval_time: Duration,
    /// Total number of documents searched.
    pub total_searched: usize,
    /// Number of documents returned.
    pub returned_count: usize,
    /// Duration for each stage.
    pub stage_times: HashMap<String, Duration>,
    /// Number of query transformations.
    pub query_transformations: usize,
    /// Number of rerank operations.
    pub rerank_operations: usize,
    /// Number of response transformations.
    pub response_transformations: usize,
}

impl RetrievalStats {
    /// Adds duration for a stage.
    pub fn add_stage_time(&mut self, stage: String, duration: Duration) {
        self.stage_times.insert(stage, duration);
    }

    /// Gets the total duration.
    #[must_use]
    pub fn total_time(&self) -> Duration {
        self.stage_times.values().sum()
    }
}

/// Interface for a query transformer.
#[async_trait]
pub trait QueryTransformer: Send + Sync + std::fmt::Debug {
    /// Transforms a query.
    async fn transform(&self, query: &mut AdvancedQuery) -> Result<()>;

    /// Gets the name of the transformer.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Whether batch transformation is supported.
    fn supports_batch(&self) -> bool {
        false
    }

    /// Transforms a batch of queries (optional implementation).
    async fn transform_batch(&self, queries: &mut [AdvancedQuery]) -> Result<()> {
        for query in queries {
            self.transform(query).await?;
        }
        Ok(())
    }

    /// Validates if a query is applicable for this transformer.
    fn validate_query(&self, _query: &AdvancedQuery) -> Result<()> {
        Ok(())
    }
}

/// Interface for a search strategy.
#[async_trait]
pub trait SearchStrategy: Send + Sync + std::fmt::Debug {
    /// Executes a search.
    async fn search(
        &self,
        query: &AdvancedQuery,
        store: &dyn VectorStore,
    ) -> Result<Vec<ScoredNode>>;

    /// Gets the name of the strategy.
    fn name(&self) -> &'static str;

    /// Gets the supported search modes.
    fn supported_modes(&self) -> Vec<SearchMode>;

    /// Validates if a query is applicable for this strategy.
    fn validate_query(&self, query: &AdvancedQuery) -> Result<()>;

    /// Gets the estimated search time.
    fn estimated_search_time(&self, _query: &AdvancedQuery) -> Option<Duration> {
        None
    }
}

/// Interface for a reranker.
#[async_trait]
pub trait Reranker: Send + Sync + std::fmt::Debug {
    /// Reranks nodes.
    async fn rerank(
        &self,
        query: &AdvancedQuery,
        nodes: Vec<ScoredNode>,
    ) -> Result<Vec<ScoredNode>>;

    /// Gets the name of the reranker.
    fn name(&self) -> &'static str;

    /// Gets the maximum number of nodes supported.
    fn max_nodes(&self) -> Option<usize> {
        None
    }

    /// Whether batch reranking is supported.
    fn supports_batch(&self) -> bool {
        false
    }

    /// Gets the estimated reranking time.
    fn estimated_rerank_time(&self, _nodes_count: usize) -> Option<Duration> {
        None
    }
}

/// Interface for a response transformer.
#[async_trait]
pub trait ResponseTransformer: Send + Sync + std::fmt::Debug {
    /// Transforms a retrieval response.
    async fn transform(&self, response: &mut RetrievalResponse) -> Result<()>;

    /// Gets the name of the transformer.
    fn name(&self) -> &'static str;

    /// Whether it changes the number of nodes.
    fn changes_node_count(&self) -> bool {
        false
    }

    /// Gets the estimated transformation time.
    fn estimated_transform_time(&self, _nodes_count: usize) -> Option<Duration> {
        None
    }
}

// Re-export DistanceMetric from cheungfun-core for consistency
pub use cheungfun_core::traits::DistanceMetric;

/// Enum for normalization methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Min-Max normalization.
    MinMax,
    /// Z-score normalization.
    ZScore,
    /// Rank normalization.
    Rank,
    /// No normalization.
    None,
}

/// Retry configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries.
    pub max_retries: usize,
    /// Initial delay.
    pub initial_delay: Duration,
    /// Maximum delay.
    pub max_delay: Duration,
    /// Backoff multiplier.
    pub backoff_multiplier: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
        }
    }
}
