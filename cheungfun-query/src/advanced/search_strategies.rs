// Search Strategies Implementation

use super::{
    AdvancedQuery, DistanceMetric, FusionMethod, NormalizationMethod, SearchMode, SearchStrategy,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use cheungfun_core::{ScoredNode, VectorStore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Vector search configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchConfig {
    /// Vector field name.
    pub vector_field: String,
    /// Distance metric.
    pub distance_metric: DistanceMetric,
    /// Search parameters.
    pub search_params: HashMap<String, serde_json::Value>,
    /// Pre-filter conditions.
    pub pre_filter: Option<HashMap<String, serde_json::Value>>,
}

impl Default for VectorSearchConfig {
    fn default() -> Self {
        Self {
            vector_field: "embedding".to_string(),
            distance_metric: DistanceMetric::Cosine,
            search_params: HashMap::new(),
            pre_filter: None,
        }
    }
}

/// Keyword search configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordSearchConfig {
    /// Search fields.
    pub search_fields: Vec<String>,
    /// Analyzer configuration.
    pub analyzer: Option<String>,
    /// BM25 parameters.
    pub bm25_params: Option<BM25Params>,
    /// Minimum should match.
    pub minimum_should_match: Option<String>,
}

impl Default for KeywordSearchConfig {
    fn default() -> Self {
        Self {
            search_fields: vec!["content".to_string()],
            analyzer: Some("standard".to_string()),
            bm25_params: Some(BM25Params::default()),
            minimum_should_match: None,
        }
    }
}

/// BM25 parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Params {
    pub k1: f32,
    pub b: f32,
}

impl Default for BM25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// Fusion configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Fusion method.
    pub method: FusionMethod,
    /// Normalization method.
    pub normalization: NormalizationMethod,
    /// Final number of results.
    pub final_top_k: usize,
    /// Intermediate result multiplier.
    pub intermediate_multiplier: f32,
    /// Vector search weight (optional, used by weighted fusion methods).
    pub vector_weight: Option<f32>,
    /// Keyword search weight (optional, used by weighted fusion methods).
    pub keyword_weight: Option<f32>,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            method: FusionMethod::ReciprocalRankFusion { k: 60.0 },
            normalization: NormalizationMethod::Rank,
            final_top_k: 10,
            intermediate_multiplier: 2.0,
            vector_weight: None,
            keyword_weight: None,
        }
    }
}

/// Pure vector search strategy.
#[derive(Debug)]
pub struct VectorSearchStrategy {
    pub config: VectorSearchConfig,
}

impl VectorSearchStrategy {
    #[must_use]
    pub fn new(config: VectorSearchConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl SearchStrategy for VectorSearchStrategy {
    async fn search(
        &self,
        query: &AdvancedQuery,
        store: &dyn VectorStore,
    ) -> Result<Vec<ScoredNode>> {
        debug!(
            "Executing vector search with {} queries",
            query.all_queries().len()
        );

        // Ensure the query has an embedding vector.
        let embedding = query
            .embedding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Query embedding is required for vector search"))?;

        // Build the vector store query.
        let vector_query = cheungfun_core::Query {
            text: query.original_text.clone(),
            embedding: Some(embedding.clone()),
            filters: query.search_params.filters.clone(),
            top_k: query.search_params.top_k,
            similarity_threshold: query.search_params.similarity_threshold,
            search_mode: cheungfun_core::types::SearchMode::Vector,
        };

        let results = store
            .search(&vector_query)
            .await
            .context("Vector search failed")?;

        info!("Vector search returned {} results", results.len());
        Ok(results)
    }

    fn name(&self) -> &'static str {
        "VectorSearchStrategy"
    }

    fn supported_modes(&self) -> Vec<SearchMode> {
        vec![SearchMode::Vector]
    }

    fn validate_query(&self, query: &AdvancedQuery) -> Result<()> {
        if query.embedding.is_none() {
            anyhow::bail!("Vector search requires query embedding");
        }
        Ok(())
    }

    fn estimated_search_time(&self, _query: &AdvancedQuery) -> Option<Duration> {
        Some(Duration::from_millis(50)) // Estimate 50ms
    }
}

/// Keyword search strategy.
#[derive(Debug)]
pub struct KeywordSearchStrategy {
    pub config: KeywordSearchConfig,
}

impl KeywordSearchStrategy {
    #[must_use]
    pub fn new(config: KeywordSearchConfig) -> Self {
        Self { config }
    }

    /// Execute BM25 search.
    async fn bm25_search(
        &self,
        query_text: &str,
        store: &dyn VectorStore,
        top_k: usize,
    ) -> Result<Vec<ScoredNode>> {
        // Note: This requires the vector store to support keyword search.
        // In a real implementation, you would need to check if the store supports this feature.

        let keyword_query = cheungfun_core::Query {
            text: query_text.to_string(),
            embedding: None,
            filters: HashMap::new(),
            top_k,
            similarity_threshold: None,
            search_mode: cheungfun_core::types::SearchMode::Keyword,
        };

        store
            .search(&keyword_query)
            .await
            .context("Keyword search failed")
    }
}

#[async_trait]
impl SearchStrategy for KeywordSearchStrategy {
    async fn search(
        &self,
        query: &AdvancedQuery,
        store: &dyn VectorStore,
    ) -> Result<Vec<ScoredNode>> {
        debug!("Executing keyword search");

        // Search using all query texts.
        let mut all_results = Vec::new();

        for query_text in query.all_queries() {
            let results = self
                .bm25_search(query_text, store, query.search_params.top_k)
                .await?;
            all_results.extend(results);
        }

        // Deduplicate and sort by score.
        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // A simple deduplication by node ID
        let mut seen = std::collections::HashSet::new();
        all_results.retain(|node| seen.insert(node.node.id));
        all_results.truncate(query.search_params.top_k);

        info!("Keyword search returned {} results", all_results.len());
        Ok(all_results)
    }

    fn name(&self) -> &'static str {
        "KeywordSearchStrategy"
    }

    fn supported_modes(&self) -> Vec<SearchMode> {
        vec![SearchMode::Keyword]
    }

    fn validate_query(&self, query: &AdvancedQuery) -> Result<()> {
        if query.original_text.trim().is_empty() {
            anyhow::bail!("Keyword search requires non-empty query text");
        }
        Ok(())
    }

    fn estimated_search_time(&self, _query: &AdvancedQuery) -> Option<Duration> {
        Some(Duration::from_millis(100)) // Estimate 100ms
    }
}

/// Hybrid search strategy.
#[derive(Debug)]
pub struct HybridSearchStrategy {
    /// Vector search configuration.
    pub vector_config: VectorSearchConfig,
    /// Keyword search configuration.
    pub keyword_config: KeywordSearchConfig,
    /// Fusion configuration.
    pub fusion_config: FusionConfig,
}

impl HybridSearchStrategy {
    #[must_use]
    pub fn new(
        vector_config: VectorSearchConfig,
        keyword_config: KeywordSearchConfig,
        fusion_config: FusionConfig,
    ) -> Self {
        Self {
            vector_config,
            keyword_config,
            fusion_config,
        }
    }

    #[must_use]
    pub fn builder() -> HybridSearchStrategyBuilder {
        HybridSearchStrategyBuilder::new()
    }

    /// Create a hybrid search strategy optimized for general Q&A scenarios.
    ///
    /// Uses Reciprocal Rank Fusion with k=60 (as recommended in the original paper),
    /// balanced vector/keyword weights (0.7/0.3), and rank normalization.
    ///
    /// Reference: "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods"
    /// by Cormack et al.
    #[must_use]
    pub fn for_general_qa() -> HybridSearchStrategyBuilder {
        HybridSearchStrategyBuilder::new()
            .fusion_method(FusionMethod::ReciprocalRankFusion { k: 60.0 })
            .vector_weight(0.7)
            .keyword_weight(0.3)
            .normalization_method(NormalizationMethod::Rank)
            .top_k(10)
    }

    /// Create a hybrid search strategy optimized for code search scenarios.
    ///
    /// Uses Linear Combination fusion with higher vector weight (0.8/0.2)
    /// since code search benefits more from semantic similarity.
    #[must_use]
    pub fn for_code_search() -> HybridSearchStrategyBuilder {
        HybridSearchStrategyBuilder::new()
            .fusion_method(FusionMethod::LinearCombination)
            .vector_weight(0.8)
            .keyword_weight(0.2)
            .normalization_method(NormalizationMethod::MinMax)
            .top_k(15)
    }

    /// Create a hybrid search strategy optimized for academic paper search.
    ///
    /// Uses Weighted Average fusion with balanced weights and higher top_k
    /// for comprehensive academic research.
    #[must_use]
    pub fn for_academic_papers() -> HybridSearchStrategyBuilder {
        HybridSearchStrategyBuilder::new()
            .fusion_method(FusionMethod::WeightedAverage)
            .vector_weight(0.6)
            .keyword_weight(0.4)
            .normalization_method(NormalizationMethod::ZScore)
            .top_k(20)
    }

    /// Create a hybrid search strategy optimized for customer support/FAQ scenarios.
    ///
    /// Uses RRF with lower k value for more aggressive reranking,
    /// higher keyword weight for exact term matching.
    #[must_use]
    pub fn for_customer_support() -> HybridSearchStrategyBuilder {
        HybridSearchStrategyBuilder::new()
            .fusion_method(FusionMethod::ReciprocalRankFusion { k: 30.0 })
            .vector_weight(0.5)
            .keyword_weight(0.5)
            .normalization_method(NormalizationMethod::Rank)
            .top_k(8)
    }

    /// Create a hybrid search strategy with default settings.
    ///
    /// Equivalent to `for_general_qa()` but follows LlamaIndex naming convention.
    #[must_use]
    pub fn from_defaults() -> HybridSearchStrategyBuilder {
        Self::for_general_qa()
    }

    /// Execute vector search.
    async fn vector_search(
        &self,
        query: &AdvancedQuery,
        store: &dyn VectorStore,
    ) -> Result<Vec<ScoredNode>> {
        let vector_strategy = VectorSearchStrategy::new(self.vector_config.clone());
        vector_strategy.search(query, store).await
    }

    /// Execute keyword search.
    async fn keyword_search(
        &self,
        query: &AdvancedQuery,
        store: &dyn VectorStore,
    ) -> Result<Vec<ScoredNode>> {
        let keyword_strategy = KeywordSearchStrategy::new(self.keyword_config.clone());
        keyword_strategy.search(query, store).await
    }

    /// Fuse search results.
    fn fuse_results(
        &self,
        vector_results: Vec<ScoredNode>,
        keyword_results: Vec<ScoredNode>,
    ) -> Vec<ScoredNode> {
        use crate::advanced::fusion::ReciprocalRankFusion;

        match &self.fusion_config.method {
            FusionMethod::ReciprocalRankFusion { k } => {
                let fusion = ReciprocalRankFusion::new(*k);
                fusion.fuse_results(vec![vector_results, keyword_results])
            }
            FusionMethod::WeightedAverage => {
                // TODO: Implement proper weighted average fusion with score normalization.
                self.weighted_average_fusion(vector_results, keyword_results)
            }
            FusionMethod::LinearCombination => {
                // TODO: Implement proper linear combination fusion with score normalization.
                self.linear_combination_fusion(vector_results, keyword_results)
            }
            FusionMethod::Custom(_) => {
                // TODO: Implement custom fusion.
                warn!("Custom fusion method not implemented, falling back to RRF");
                let fusion = ReciprocalRankFusion::new(60.0);
                fusion.fuse_results(vec![vector_results, keyword_results])
            }
        }
    }

    /// Weighted average fusion (placeholder).
    fn weighted_average_fusion(
        &self,
        vector_results: Vec<ScoredNode>,
        keyword_results: Vec<ScoredNode>,
    ) -> Vec<ScoredNode> {
        // TODO: Implement a proper weighted average fusion algorithm.
        // For now, simply combine the results and take the top scores.
        let mut combined = vector_results;
        combined.extend(keyword_results);
        combined.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut seen = std::collections::HashSet::new();
        combined.retain(|node| seen.insert(node.node.id));
        combined.truncate(self.fusion_config.final_top_k);
        combined
    }

    /// Linear combination fusion (placeholder).
    fn linear_combination_fusion(
        &self,
        vector_results: Vec<ScoredNode>,
        keyword_results: Vec<ScoredNode>,
    ) -> Vec<ScoredNode> {
        // TODO: Implement a proper linear combination fusion algorithm.
        // For now, simply combine the results and take the top scores.
        let mut combined = vector_results;
        combined.extend(keyword_results);
        combined.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut seen = std::collections::HashSet::new();
        combined.retain(|node| seen.insert(node.node.id));
        combined.truncate(self.fusion_config.final_top_k);
        combined
    }
}

#[async_trait]
impl SearchStrategy for HybridSearchStrategy {
    async fn search(
        &self,
        query: &AdvancedQuery,
        store: &dyn VectorStore,
    ) -> Result<Vec<ScoredNode>> {
        debug!("Executing hybrid search");

        // Execute vector and keyword searches in parallel.
        let (vector_results, keyword_results) = tokio::try_join!(
            self.vector_search(query, store),
            self.keyword_search(query, store)
        )?;

        info!(
            "Vector search: {} results, Keyword search: {} results",
            vector_results.len(),
            keyword_results.len()
        );

        // Fuse the results.
        let fused_results = self.fuse_results(vector_results, keyword_results);

        info!(
            "Hybrid search returned {} fused results",
            fused_results.len()
        );
        Ok(fused_results)
    }

    fn name(&self) -> &'static str {
        "HybridSearchStrategy"
    }

    fn supported_modes(&self) -> Vec<SearchMode> {
        vec![SearchMode::Hybrid {
            vector_weight: 0.5, // Default/placeholder weights
            keyword_weight: 0.5,
            fusion_method: self.fusion_config.method.clone(),
        }]
    }

    fn validate_query(&self, query: &AdvancedQuery) -> Result<()> {
        if query.embedding.is_none() {
            anyhow::bail!("Hybrid search requires query embedding for vector component");
        }
        if query.original_text.trim().is_empty() {
            anyhow::bail!("Hybrid search requires non-empty query text for keyword component");
        }
        Ok(())
    }

    fn estimated_search_time(&self, _query: &AdvancedQuery) -> Option<Duration> {
        Some(Duration::from_millis(150)) // Estimate 150ms (for parallel execution)
    }
}

/// Hybrid search strategy builder.
pub struct HybridSearchStrategyBuilder {
    vector_config: VectorSearchConfig,
    keyword_config: KeywordSearchConfig,
    fusion_config: FusionConfig,
}

impl Default for HybridSearchStrategyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridSearchStrategyBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            vector_config: VectorSearchConfig::default(),
            keyword_config: KeywordSearchConfig::default(),
            fusion_config: FusionConfig::default(),
        }
    }

    #[must_use]
    pub fn with_vector_config(mut self, config: VectorSearchConfig) -> Self {
        self.vector_config = config;
        self
    }

    #[must_use]
    pub fn with_keyword_config(mut self, config: KeywordSearchConfig) -> Self {
        self.keyword_config = config;
        self
    }

    #[must_use]
    pub fn with_fusion_config(mut self, config: FusionConfig) -> Self {
        self.fusion_config = config;
        self
    }

    /// Set the vector search weight for fusion.
    #[must_use]
    pub fn vector_weight(mut self, weight: f32) -> Self {
        // Store weight in fusion config for later use
        self.fusion_config.vector_weight = Some(weight);
        self
    }

    /// Set the keyword search weight for fusion.
    #[must_use]
    pub fn keyword_weight(mut self, weight: f32) -> Self {
        // Store weight in fusion config for later use
        self.fusion_config.keyword_weight = Some(weight);
        self
    }

    /// Set the fusion method.
    #[must_use]
    pub fn fusion_method(mut self, method: FusionMethod) -> Self {
        self.fusion_config.method = method;
        self
    }

    /// Set the normalization method.
    #[must_use]
    pub fn normalization_method(mut self, method: NormalizationMethod) -> Self {
        self.fusion_config.normalization = method;
        self
    }

    /// Set the final top-k results to return.
    #[must_use]
    pub fn top_k(mut self, k: usize) -> Self {
        self.fusion_config.final_top_k = k;
        self
    }

    /// Set the distance metric for vector search.
    #[must_use]
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.vector_config.distance_metric = metric;
        self
    }

    /// Set the BM25 parameters for keyword search.
    #[must_use]
    pub fn bm25_params(mut self, k1: f32, b: f32) -> Self {
        self.keyword_config.bm25_params = Some(BM25Params { k1, b });
        self
    }

    #[must_use]
    pub fn with_keyword_fields(mut self, fields: Vec<&str>) -> Self {
        self.keyword_config.search_fields = fields
            .into_iter()
            .map(std::string::ToString::to_string)
            .collect();
        self
    }

    #[must_use]
    pub fn with_fusion_method(mut self, method: FusionMethod) -> Self {
        self.fusion_config.method = method;
        self
    }

    #[must_use]
    pub fn build(self) -> HybridSearchStrategy {
        HybridSearchStrategy::new(self.vector_config, self.keyword_config, self.fusion_config)
    }
}
