//! Query Fusion Retriever - Unified interface for multi-retriever fusion.
//!
//! This module implements a unified retriever that combines results from multiple
//! retrievers using various fusion algorithms, similar to LlamaIndex's QueryFusionRetriever.
//!
//! **Reference**: LlamaIndex QueryFusionRetriever
//! - Supports multiple fusion modes: RRF, weighted average, distance-based scoring
//! - Configurable retriever weights and fusion parameters
//! - Async-first design with comprehensive error handling

use async_trait::async_trait;
use cheungfun_core::{
    traits::Retriever,
    types::{Query, ScoredNode},
    Result,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

use crate::advanced::fusion::{ReciprocalRankFusion, WeightedAverageFusion};

/// Fusion mode for combining retriever results.
///
/// **Reference**: LlamaIndex QueryFusionRetriever fusion modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionMode {
    /// Reciprocal Rank Fusion with configurable k parameter
    ReciprocalRank { k: f32 },

    /// Weighted average fusion with retriever weights
    WeightedAverage { normalize_scores: bool },

    /// Distance-based score fusion (converts distances to similarities)
    DistBasedScore {
        /// Whether to normalize scores before fusion
        normalize: bool,
        /// Distance metric used by retrievers
        distance_metric: DistanceMetric,
    },

    /// Simple concatenation with deduplication
    SimpleConcat,
}

impl Default for FusionMode {
    fn default() -> Self {
        Self::ReciprocalRank { k: 60.0 }
    }
}

/// Distance metric for score conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

/// Configuration for QueryFusionRetriever.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFusionConfig {
    /// Fusion mode to use
    pub mode: FusionMode,

    /// Weights for each retriever (must match number of retrievers)
    pub retriever_weights: Vec<f32>,

    /// Whether to use async parallel retrieval
    pub use_async: bool,

    /// Number of queries to generate (for query expansion)
    pub num_queries: usize,

    /// Final number of results to return
    pub final_top_k: usize,

    /// Intermediate top_k multiplier for each retriever
    pub intermediate_multiplier: f32,
}

impl Default for QueryFusionConfig {
    fn default() -> Self {
        Self {
            mode: FusionMode::default(),
            retriever_weights: vec![1.0],
            use_async: true,
            num_queries: 1,
            final_top_k: 10,
            intermediate_multiplier: 2.0,
        }
    }
}

/// A retriever that fuses results from multiple retrievers.
///
/// This provides a unified interface for combining results from different
/// retrieval strategies, similar to LlamaIndex's QueryFusionRetriever.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_query::retrievers::{QueryFusionRetriever, QueryFusionConfig, FusionMode};
/// use std::sync::Arc;
///
/// # async fn example() -> cheungfun_core::Result<()> {
/// let config = QueryFusionConfig {
///     mode: FusionMode::ReciprocalRank { k: 60.0 },
///     retriever_weights: vec![0.6, 0.4],
///     use_async: true,
///     ..Default::default()
/// };
///
/// let fusion_retriever = QueryFusionRetriever::new(
///     vec![vector_retriever, bm25_retriever],
///     config,
/// );
///
/// let results = fusion_retriever.retrieve(&query).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct QueryFusionRetriever {
    /// The retrievers to fuse results from
    retrievers: Vec<Arc<dyn Retriever>>,

    /// Configuration for fusion
    config: QueryFusionConfig,

    /// RRF fuser (cached for performance)
    rrf_fuser: Option<ReciprocalRankFusion>,

    /// Weighted average fuser (cached for performance)
    weighted_fuser: Option<WeightedAverageFusion>,
}

impl QueryFusionRetriever {
    /// Create a new QueryFusionRetriever.
    ///
    /// # Arguments
    ///
    /// * `retrievers` - The retrievers to combine
    /// * `config` - Configuration for fusion behavior
    ///
    /// # Errors
    ///
    /// Returns an error if the number of retriever weights doesn't match the number of retrievers.
    pub fn new(retrievers: Vec<Arc<dyn Retriever>>, config: QueryFusionConfig) -> Result<Self> {
        // Validate configuration
        if config.retriever_weights.len() != retrievers.len() {
            return Err(cheungfun_core::CheungfunError::configuration(format!(
                "Number of retriever weights ({}) must match number of retrievers ({})",
                config.retriever_weights.len(),
                retrievers.len()
            )));
        }

        // Pre-create fusers based on mode
        let (rrf_fuser, weighted_fuser) = match &config.mode {
            FusionMode::ReciprocalRank { k } => (Some(ReciprocalRankFusion::new(*k)), None),
            FusionMode::WeightedAverage { normalize_scores } => (
                None,
                Some(
                    WeightedAverageFusion::new(config.retriever_weights.clone())
                        .with_normalize_scores(*normalize_scores),
                ),
            ),
            _ => (None, None),
        };

        Ok(Self {
            retrievers,
            config,
            rrf_fuser,
            weighted_fuser,
        })
    }

    /// Create a QueryFusionRetriever with default RRF fusion.
    pub fn with_rrf(retrievers: Vec<Arc<dyn Retriever>>, k: f32) -> Result<Self> {
        let config = QueryFusionConfig {
            mode: FusionMode::ReciprocalRank { k },
            retriever_weights: vec![1.0; retrievers.len()],
            ..Default::default()
        };
        Self::new(retrievers, config)
    }

    /// Create a QueryFusionRetriever with weighted average fusion.
    pub fn with_weighted_average(
        retrievers: Vec<Arc<dyn Retriever>>,
        weights: Vec<f32>,
        normalize_scores: bool,
    ) -> Result<Self> {
        let config = QueryFusionConfig {
            mode: FusionMode::WeightedAverage { normalize_scores },
            retriever_weights: weights,
            ..Default::default()
        };
        Self::new(retrievers, config)
    }

    /// Create a QueryFusionRetriever with distance-based score fusion.
    pub fn with_dist_based_score(
        retrievers: Vec<Arc<dyn Retriever>>,
        weights: Vec<f32>,
        distance_metric: DistanceMetric,
        normalize: bool,
    ) -> Result<Self> {
        let config = QueryFusionConfig {
            mode: FusionMode::DistBasedScore {
                normalize,
                distance_metric,
            },
            retriever_weights: weights,
            ..Default::default()
        };
        Self::new(retrievers, config)
    }

    /// Retrieve results from all retrievers.
    async fn retrieve_from_all(&self, query: &Query) -> Result<Vec<Vec<ScoredNode>>> {
        let intermediate_top_k =
            (self.config.final_top_k as f32 * self.config.intermediate_multiplier) as usize;

        // Adjust query for intermediate retrieval
        let mut adjusted_query = query.clone();
        adjusted_query.top_k = intermediate_top_k;

        if self.config.use_async {
            // Parallel retrieval
            let futures: Vec<_> = self
                .retrievers
                .iter()
                .map(|retriever| retriever.retrieve(&adjusted_query))
                .collect();

            let results = futures::future::try_join_all(futures).await?;
            Ok(results)
        } else {
            // Sequential retrieval
            let mut results = Vec::new();
            for retriever in &self.retrievers {
                let result = retriever.retrieve(&adjusted_query).await?;
                results.push(result);
            }
            Ok(results)
        }
    }

    /// Fuse results using the configured fusion mode.
    fn fuse_results(&self, results: Vec<Vec<ScoredNode>>) -> Vec<ScoredNode> {
        match &self.config.mode {
            FusionMode::ReciprocalRank { .. } => {
                if let Some(ref fuser) = self.rrf_fuser {
                    fuser.fuse_results(results)
                } else {
                    warn!("RRF fuser not initialized, falling back to simple concat");
                    self.simple_concat_fusion(results)
                }
            }
            FusionMode::WeightedAverage { .. } => {
                if let Some(ref fuser) = self.weighted_fuser {
                    fuser.fuse_results(results)
                } else {
                    warn!("Weighted fuser not initialized, falling back to simple concat");
                    self.simple_concat_fusion(results)
                }
            }
            FusionMode::DistBasedScore {
                normalize,
                distance_metric,
            } => self.dist_based_score_fusion(results, *normalize, distance_metric),
            FusionMode::SimpleConcat => self.simple_concat_fusion(results),
        }
    }

    /// Simple concatenation with deduplication.
    fn simple_concat_fusion(&self, results: Vec<Vec<ScoredNode>>) -> Vec<ScoredNode> {
        use std::collections::HashMap;

        let mut node_map: HashMap<String, ScoredNode> = HashMap::new();

        for result_list in results {
            for node in result_list {
                let node_id = node.node.id.to_string();
                node_map.entry(node_id).or_insert(node);
            }
        }

        let mut final_results: Vec<_> = node_map.into_values().collect();
        final_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        final_results
    }

    /// Distance-based score fusion.
    fn dist_based_score_fusion(
        &self,
        results: Vec<Vec<ScoredNode>>,
        _normalize: bool,
        distance_metric: &DistanceMetric,
    ) -> Vec<ScoredNode> {
        // Convert distances to similarities and apply weighted fusion
        let converted_results: Vec<Vec<ScoredNode>> = results
            .into_iter()
            .enumerate()
            .map(|(idx, mut result_list)| {
                let weight = self.config.retriever_weights[idx];

                for node in &mut result_list {
                    // Convert distance to similarity based on metric
                    let similarity = match distance_metric {
                        DistanceMetric::Cosine => 1.0 - node.score, // Cosine distance to similarity
                        DistanceMetric::Euclidean => 1.0 / (1.0 + node.score), // Euclidean distance to similarity
                        DistanceMetric::DotProduct => node.score, // Dot product is already similarity
                    };

                    node.score = similarity * weight;
                }

                result_list
            })
            .collect();

        // Use simple concatenation for now (could be enhanced with more sophisticated fusion)
        self.simple_concat_fusion(converted_results)
    }
}

#[async_trait]
impl Retriever for QueryFusionRetriever {
    #[instrument(skip(self), fields(retrievers = self.retrievers.len(), mode = ?self.config.mode))]
    async fn retrieve(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        info!(
            "Starting fusion retrieval with {} retrievers using {:?} mode",
            self.retrievers.len(),
            self.config.mode
        );

        // Retrieve from all retrievers
        let results = self.retrieve_from_all(query).await?;

        debug!(
            "Retrieved results from {} retrievers: {:?}",
            results.len(),
            results.iter().map(|r| r.len()).collect::<Vec<_>>()
        );

        // Fuse results
        let mut fused_results = self.fuse_results(results);

        // Limit to final top_k
        if fused_results.len() > self.config.final_top_k {
            fused_results.truncate(self.config.final_top_k);
        }

        info!(
            "Fusion retrieval completed: {} final results",
            fused_results.len()
        );

        Ok(fused_results)
    }
}

/// Builder for QueryFusionRetriever.
#[derive(Debug, Default)]
pub struct QueryFusionRetrieverBuilder {
    retrievers: Vec<Arc<dyn Retriever>>,
    config: QueryFusionConfig,
}

impl QueryFusionRetrieverBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a retriever.
    pub fn add_retriever(mut self, retriever: Arc<dyn Retriever>) -> Self {
        self.retrievers.push(retriever);
        self
    }

    /// Set retrievers.
    pub fn retrievers(mut self, retrievers: Vec<Arc<dyn Retriever>>) -> Self {
        self.retrievers = retrievers;
        self
    }

    /// Set fusion mode.
    pub fn fusion_mode(mut self, mode: FusionMode) -> Self {
        self.config.mode = mode;
        self
    }

    /// Set retriever weights.
    pub fn weights(mut self, weights: Vec<f32>) -> Self {
        self.config.retriever_weights = weights;
        self
    }

    /// Set whether to use async retrieval.
    pub fn use_async(mut self, use_async: bool) -> Self {
        self.config.use_async = use_async;
        self
    }

    /// Set final top_k.
    pub fn final_top_k(mut self, top_k: usize) -> Self {
        self.config.final_top_k = top_k;
        self
    }

    /// Build the QueryFusionRetriever.
    pub fn build(self) -> Result<QueryFusionRetriever> {
        QueryFusionRetriever::new(self.retrievers, self.config)
    }
}
