// Advanced Retrieval Pipeline Implementation

use super::{
    AdvancedQuery, QueryCache, QueryTransformer, Reranker,
    ResponseTransformer, RetrievalResponse, RetrievalStats, RetryConfig,
    ScoredNode, SearchStrategy,
};
use anyhow::{Context, Result};
use cheungfun_core::VectorStore;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// The Advanced Retrieval Pipeline.
#[derive(Debug)]
pub struct AdvancedRetrievalPipeline {
    /// Chain of query transformers.
    pub query_transformers: Vec<Arc<dyn QueryTransformer>>,
    /// The search strategy.
    pub search_strategy: Arc<dyn SearchStrategy>,
    /// Chain of rerankers.
    pub rerankers: Vec<Arc<dyn Reranker>>,
    /// Chain of response transformers.
    pub response_transformers: Vec<Arc<dyn ResponseTransformer>>,
    /// The vector store.
    pub vector_store: Arc<dyn VectorStore>,
    /// Pipeline configuration.
    pub config: PipelineConfig,
    /// Cache (optional).
    pub cache: Option<Arc<QueryCache>>,
}

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Concurrency level.
    pub concurrency: usize,
    /// Timeout setting.
    pub timeout: Duration,
    /// Whether to enable caching.
    pub enable_cache: bool,
    /// Whether to enable metrics collection.
    pub enable_metrics: bool,
    /// Retry configuration.
    pub retry_config: RetryConfig,
    /// Whether to enable concurrent query transformation.
    pub enable_concurrent_query_transform: bool,
    /// Whether to enable concurrent reranking.
    pub enable_concurrent_rerank: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            concurrency: 4,
            timeout: Duration::from_secs(30),
            enable_cache: true,
            enable_metrics: true,
            retry_config: RetryConfig::default(),
            enable_concurrent_query_transform: true,
            enable_concurrent_rerank: false, // Reranking is usually executed sequentially.
        }
    }
}

impl AdvancedRetrievalPipeline {
    /// Creates a pipeline using the builder pattern.
    #[must_use]
    pub fn builder() -> AdvancedRetrievalPipelineBuilder {
        AdvancedRetrievalPipelineBuilder::new()
    }

    /// Executes the complete retrieval process.
    pub async fn retrieve(&self, query_text: &str) -> Result<RetrievalResponse> {
        let start_time = Instant::now();
        let query = AdvancedQuery::from_text(query_text);

        info!("Starting advanced retrieval for query: {}", query_text);

        // Check the cache
        if let Some(cached_response) = self.check_cache(&query).await? {
            info!("Cache hit for query: {}", query_text);
            return Ok(cached_response);
        }

        // Execute the pipeline
        let response = timeout(self.config.timeout, self.execute_pipeline(query))
            .await
            .context("Pipeline execution timed out")?
            .context("Pipeline execution failed")?;

        // Update the cache
        if let Err(e) = self.update_cache(&response).await {
            warn!("Failed to update cache: {}", e);
        }

        let total_time = start_time.elapsed();
        info!("Advanced retrieval completed in {:?}", total_time);

        Ok(response)
    }

    /// Executes the core logic of the pipeline.
    async fn execute_pipeline(&self, mut query: AdvancedQuery) -> Result<RetrievalResponse> {
        let mut stats = RetrievalStats::default();
        let pipeline_start = Instant::now();

        // 1. Query transformation stage
        let transform_start = Instant::now();
        if self.config.enable_concurrent_query_transform && self.query_transformers.len() > 1 {
            self.transform_queries_concurrent(&mut query).await?;
        } else {
            self.transform_queries_sequential(&mut query).await?;
        }
        stats.add_stage_time("query_transform".to_string(), transform_start.elapsed());
        stats.query_transformations = self.query_transformers.len();

        // 2. Search stage
        let search_start = Instant::now();
        let mut nodes = self
            .search_strategy
            .search(&query, self.vector_store.as_ref())
            .await
            .context("Search stage failed")?;
        stats.add_stage_time("search".to_string(), search_start.elapsed());
        stats.total_searched = nodes.len();

        // 3. Reranking stage
        let rerank_start = Instant::now();
        if self.config.enable_concurrent_rerank && self.rerankers.len() > 1 {
            nodes = self.rerank_concurrent(&query, nodes).await?;
        } else {
            nodes = self.rerank_sequential(&query, nodes).await?;
        }
        stats.add_stage_time("rerank".to_string(), rerank_start.elapsed());
        stats.rerank_operations = self.rerankers.len();

        // 4. Response transformation stage
        let response_transform_start = Instant::now();
        let mut response = RetrievalResponse {
            nodes,
            query,
            metadata: std::collections::HashMap::new(),
            stats,
        };

        for transformer in &self.response_transformers {
            transformer
                .transform(&mut response)
                .await
                .with_context(|| format!("Response transformer '{}' failed", transformer.name()))?;
        }
        response.stats.add_stage_time(
            "response_transform".to_string(),
            response_transform_start.elapsed(),
        );
        response.stats.response_transformations = self.response_transformers.len();

        // Update final statistics
        response.stats.retrieval_time = pipeline_start.elapsed();
        response.stats.returned_count = response.nodes.len();

        Ok(response)
    }

    /// Executes query transformations sequentially.
    async fn transform_queries_sequential(&self, query: &mut AdvancedQuery) -> Result<()> {
        for transformer in &self.query_transformers {
            transformer
                .transform(query)
                .await
                .with_context(|| format!("Query transformer '{}' failed", transformer.name()))?;
        }
        Ok(())
    }

    /// Executes query transformations concurrently.
    async fn transform_queries_concurrent(&self, query: &mut AdvancedQuery) -> Result<()> {
        debug!(
            "Executing {} query transformers concurrently",
            self.query_transformers.len()
        );

        let futures: Vec<_> = self
            .query_transformers
            .iter()
            .map(|transformer| {
                let mut query_clone = query.clone();
                let transformer = transformer.clone();
                async move {
                    transformer.transform(&mut query_clone).await?;
                    Ok::<_, anyhow::Error>(query_clone.transformed_queries)
                }
            })
            .collect();

        let results = futures::future::try_join_all(futures).await?;

        // Merge all transformation results
        for transformed_queries in results {
            query.transformed_queries.extend(transformed_queries);
        }

        // Deduplicate
        query.transformed_queries.sort();
        query.transformed_queries.dedup();

        Ok(())
    }

    /// Executes reranking sequentially.
    async fn rerank_sequential(
        &self,
        query: &AdvancedQuery,
        mut nodes: Vec<ScoredNode>,
    ) -> Result<Vec<ScoredNode>> {
        for reranker in &self.rerankers {
            nodes = reranker
                .rerank(query, nodes)
                .await
                .with_context(|| format!("Reranker '{}' failed", reranker.name()))?;
        }
        Ok(nodes)
    }

    /// Executes reranking concurrently (experimental feature).
    async fn rerank_concurrent(
        &self,
        query: &AdvancedQuery,
        nodes: Vec<ScoredNode>,
    ) -> Result<Vec<ScoredNode>> {
        debug!("Executing {} rerankers concurrently", self.rerankers.len());

        // Note: Concurrent reranking might not always be suitable, as rerankers can have dependencies.
        // This provides a simple implementation; use with caution in practice.

        if self.rerankers.is_empty() {
            return Ok(nodes);
        }

        // This placeholder implementation simply executes them sequentially.
        // A true concurrent implementation would require more complex logic (e.g., scatter/gather).
        let first_reranker = &self.rerankers[0];
        let mut reranked_nodes = first_reranker.rerank(query, nodes).await?;

        // The remaining rerankers are executed sequentially.
        for reranker in &self.rerankers[1..] {
            reranked_nodes = reranker.rerank(query, reranked_nodes).await?;
        }

        Ok(reranked_nodes)
    }

    /// Checks the cache.
    async fn check_cache(&self, query: &AdvancedQuery) -> Result<Option<RetrievalResponse>> {
        if !self.config.enable_cache {
            return Ok(None);
        }

        if let Some(cache) = &self.cache {
            let query_hash = self.calculate_query_hash(query);
            return Ok(cache.get(&query_hash).await);
        }

        Ok(None)
    }

    /// Updates the cache.
    async fn update_cache(&self, response: &RetrievalResponse) -> Result<()> {
        if !self.config.enable_cache {
            return Ok(());
        }

        if let Some(cache) = &self.cache {
            let query_hash = self.calculate_query_hash(&response.query);
            cache.put(query_hash, response.clone()).await;
        }

        Ok(())
    }

    /// Calculates the query hash.
    fn calculate_query_hash(&self, query: &AdvancedQuery) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        query.original_text.hash(&mut hasher);
        query.search_params.top_k.hash(&mut hasher);
        // More parameters that affect caching can be added here.

        format!("{:x}", hasher.finish())
    }

    /// Gets pipeline statistics.
    #[must_use]
    pub fn get_stats(&self) -> PipelineStats {
        PipelineStats {
            query_transformers_count: self.query_transformers.len(),
            rerankers_count: self.rerankers.len(),
            response_transformers_count: self.response_transformers.len(),
            search_strategy: self.search_strategy.name().to_string(),
            cache_enabled: self.config.enable_cache,
            metrics_enabled: self.config.enable_metrics,
        }
    }
}

/// Pipeline statistics.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub query_transformers_count: usize,
    pub rerankers_count: usize,
    pub response_transformers_count: usize,
    pub search_strategy: String,
    pub cache_enabled: bool,
    pub metrics_enabled: bool,
}

/// Pipeline builder.
pub struct AdvancedRetrievalPipelineBuilder {
    query_transformers: Vec<Arc<dyn QueryTransformer>>,
    search_strategy: Option<Arc<dyn SearchStrategy>>,
    rerankers: Vec<Arc<dyn Reranker>>,
    response_transformers: Vec<Arc<dyn ResponseTransformer>>,
    vector_store: Option<Arc<dyn VectorStore>>,
    config: PipelineConfig,
    cache: Option<Arc<QueryCache>>,
}

impl Default for AdvancedRetrievalPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedRetrievalPipelineBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            query_transformers: Vec::new(),
            search_strategy: None,
            rerankers: Vec::new(),
            response_transformers: Vec::new(),
            vector_store: None,
            config: PipelineConfig::default(),
            cache: None,
        }
    }

    /// Adds a query transformer.
    pub fn add_query_transformer(mut self, transformer: impl QueryTransformer + 'static) -> Self {
        self.query_transformers.push(Arc::new(transformer));
        self
    }

    /// Adds a query transformer (Arc version).
    pub fn add_query_transformer_arc(mut self, transformer: Arc<dyn QueryTransformer>) -> Self {
        self.query_transformers.push(transformer);
        self
    }

    /// Sets the search strategy.
    pub fn with_search_strategy(mut self, strategy: impl SearchStrategy + 'static) -> Self {
        self.search_strategy = Some(Arc::new(strategy));
        self
    }

    /// Sets the search strategy (Arc version).
    pub fn with_search_strategy_arc(mut self, strategy: Arc<dyn SearchStrategy>) -> Self {
        self.search_strategy = Some(strategy);
        self
    }

    /// Adds a reranker.
    pub fn add_reranker(mut self, reranker: impl Reranker + 'static) -> Self {
        self.rerankers.push(Arc::new(reranker));
        self
    }

    /// Adds a reranker (Arc version).
    pub fn add_reranker_arc(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.rerankers.push(reranker);
        self
    }

    /// Adds a response transformer.
    pub fn add_response_transformer(
        mut self,
        transformer: impl ResponseTransformer + 'static,
    ) -> Self {
        self.response_transformers.push(Arc::new(transformer));
        self
    }

    /// Adds a response transformer (Arc version).
    pub fn add_response_transformer_arc(
        mut self,
        transformer: Arc<dyn ResponseTransformer>,
    ) -> Self {
        self.response_transformers.push(transformer);
        self
    }

    /// Sets the vector store.
    pub fn with_vector_store(mut self, store: Arc<dyn VectorStore>) -> Self {
        self.vector_store = Some(store);
        self
    }

    /// Sets the configuration.
    #[must_use]
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the cache.
    #[must_use]
    pub fn with_cache(mut self, cache: Arc<QueryCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Builds the pipeline.
    pub fn build(self) -> Result<AdvancedRetrievalPipeline> {
        // Validate required components
        let search_strategy = self
            .search_strategy
            .ok_or_else(|| anyhow::anyhow!("Search strategy is required"))?;

        let vector_store = self
            .vector_store
            .ok_or_else(|| anyhow::anyhow!("Vector store is required"))?;

        // Create cache if enabled and not provided
        let cache = if self.config.enable_cache && self.cache.is_none() {
            Some(Arc::new(QueryCache::new(
                Duration::from_secs(3600), // 1 hour TTL
                1000,                      // max 1000 entries
            )))
        } else {
            self.cache
        };

        Ok(AdvancedRetrievalPipeline {
            query_transformers: self.query_transformers,
            search_strategy,
            rerankers: self.rerankers,
            response_transformers: self.response_transformers,
            vector_store,
            config: self.config,
            cache,
        })
    }
}
