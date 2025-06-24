//! Retrieval traits for finding relevant nodes.
//!
//! This module defines traits for retrieving relevant nodes based on queries.
//! Retrievers combine vector stores with search logic to find the most
//! relevant content for a given query.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::{Query, Result, RetrievalContext, ScoredNode};

/// Retrieves relevant nodes for a query.
///
/// This trait provides the interface for finding relevant content based on
/// a query. Implementations can use different strategies like vector search,
/// keyword search, hybrid search, or more complex retrieval methods.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::Retriever;
/// use cheungfun_core::{Query, ScoredNode, RetrievalContext, Result};
/// use async_trait::async_trait;
///
/// struct SimpleRetriever;
///
/// #[async_trait]
/// impl Retriever for SimpleRetriever {
///     async fn retrieve(&self, query: &Query) -> Result<Vec<ScoredNode>> {
///         // Implementation would perform retrieval
///         Ok(vec![])
///     }
///
///     async fn retrieve_with_context(
///         &self,
///         query: &Query,
///         context: &RetrievalContext
///     ) -> Result<Vec<ScoredNode>> {
///         // Implementation would use context for better retrieval
///         self.retrieve(query).await
///     }
/// }
/// ```
#[async_trait]
pub trait Retriever: Send + Sync + std::fmt::Debug {
    /// Retrieve nodes for a query.
    ///
    /// This is the core retrieval method that finds relevant nodes
    /// based on the query parameters.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query containing text, filters, and parameters
    ///
    /// # Returns
    ///
    /// A vector of scored nodes sorted by relevance (highest score first).
    /// The number of results should respect the `top_k` parameter in the query.
    ///
    /// # Errors
    ///
    /// Returns an error if retrieval fails due to connection issues,
    /// invalid query parameters, or processing errors.
    async fn retrieve(&self, query: &Query) -> Result<Vec<ScoredNode>>;

    /// Retrieve with additional context for better results.
    ///
    /// This method allows retrievers to use additional context like
    /// conversation history, user preferences, or session information
    /// to improve retrieval quality.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query
    /// * `context` - Additional context for retrieval
    ///
    /// # Returns
    ///
    /// A vector of scored nodes with context-aware ranking.
    async fn retrieve_with_context(
        &self,
        query: &Query,
        _context: &RetrievalContext,
    ) -> Result<Vec<ScoredNode>> {
        // Default implementation ignores context
        self.retrieve(query).await
    }

    /// Get a human-readable name for this retriever.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Check if the retriever is healthy and ready to process queries.
    async fn health_check(&self) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Get configuration information about this retriever.
    fn config(&self) -> HashMap<String, serde_json::Value> {
        // Default implementation returns empty config
        HashMap::new()
    }

    /// Get statistics about retrieval operations.
    async fn stats(&self) -> Result<RetrievalStats> {
        Ok(RetrievalStats::default())
    }

    /// Explain how a query would be processed (for debugging).
    ///
    /// This method can be used to understand how the retriever would
    /// process a query without actually executing it.
    async fn explain(&self, _query: &Query) -> Result<RetrievalExplanation> {
        Ok(RetrievalExplanation::default())
    }
}

/// Statistics about retrieval operations.
#[derive(Debug, Clone, Default)]
pub struct RetrievalStats {
    /// Total number of queries processed.
    pub queries_processed: usize,

    /// Number of queries that failed.
    pub queries_failed: usize,

    /// Average retrieval time per query.
    pub avg_retrieval_time: std::time::Duration,

    /// Average number of results per query.
    pub avg_results_per_query: f64,

    /// Total retrieval time across all queries.
    pub total_retrieval_time: std::time::Duration,

    /// Cache hit rate (if caching is used).
    pub cache_hit_rate: Option<f64>,

    /// Additional retriever-specific statistics.
    pub additional_stats: HashMap<String, serde_json::Value>,
}

impl RetrievalStats {
    /// Create new retrieval statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate the success rate as a percentage.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.queries_processed == 0 {
            0.0
        } else {
            let successful = self.queries_processed - self.queries_failed;
            (successful as f64 / self.queries_processed as f64) * 100.0
        }
    }

    /// Calculate queries per second.
    #[must_use]
    pub fn queries_per_second(&self) -> f64 {
        if self.total_retrieval_time.is_zero() {
            0.0
        } else {
            self.queries_processed as f64 / self.total_retrieval_time.as_secs_f64()
        }
    }

    /// Update average retrieval time.
    pub fn update_avg_time(&mut self) {
        if self.queries_processed > 0 {
            self.avg_retrieval_time = self.total_retrieval_time / self.queries_processed as u32;
        }
    }
}

/// Explanation of how a query would be processed.
#[derive(Debug, Clone, Default)]
pub struct RetrievalExplanation {
    /// Steps that would be taken to process the query.
    pub steps: Vec<RetrievalStep>,

    /// Estimated processing time.
    pub estimated_time: Option<std::time::Duration>,

    /// Expected number of results.
    pub expected_results: Option<usize>,

    /// Additional explanation details.
    pub details: HashMap<String, serde_json::Value>,
}

/// A step in the retrieval process.
#[derive(Debug, Clone)]
pub struct RetrievalStep {
    /// Name of the step.
    pub name: String,

    /// Description of what this step does.
    pub description: String,

    /// Estimated time for this step.
    pub estimated_time: Option<std::time::Duration>,

    /// Parameters used in this step.
    pub parameters: HashMap<String, serde_json::Value>,
}

impl RetrievalStep {
    /// Create a new retrieval step.
    pub fn new<S: Into<String>>(name: S, description: S) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            estimated_time: None,
            parameters: HashMap::new(),
        }
    }

    /// Add a parameter to this step.
    pub fn with_parameter<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.parameters.insert(key.into(), value.into());
        self
    }

    /// Set the estimated time for this step.
    #[must_use]
    pub fn with_estimated_time(mut self, time: std::time::Duration) -> Self {
        self.estimated_time = Some(time);
        self
    }
}

/// Configuration for retrieval operations.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Default number of results to return.
    pub default_top_k: usize,

    /// Maximum number of results allowed.
    pub max_top_k: usize,

    /// Default similarity threshold.
    pub default_similarity_threshold: Option<f32>,

    /// Timeout for retrieval operations in seconds.
    pub timeout_seconds: Option<u64>,

    /// Whether to enable result caching.
    pub enable_caching: bool,

    /// Cache TTL in seconds.
    pub cache_ttl_seconds: Option<u64>,

    /// Additional retriever-specific configuration.
    pub additional_config: HashMap<String, serde_json::Value>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            default_top_k: 10,
            max_top_k: 100,
            default_similarity_threshold: None,
            timeout_seconds: Some(30),
            enable_caching: false,
            cache_ttl_seconds: Some(3600), // 1 hour
            additional_config: HashMap::new(),
        }
    }
}

/// A retriever that combines multiple retrieval strategies.
///
/// This trait allows for implementing ensemble retrievers that combine
/// results from multiple retrieval methods for better performance.
#[async_trait]
pub trait EnsembleRetriever: Retriever {
    /// Get the list of sub-retrievers used in this ensemble.
    fn sub_retrievers(&self) -> Vec<&dyn Retriever>;

    /// Combine results from multiple retrievers.
    ///
    /// This method defines how results from different retrievers
    /// should be combined and ranked.
    async fn combine_results(
        &self,
        results: Vec<Vec<ScoredNode>>,
        query: &Query,
    ) -> Result<Vec<ScoredNode>>;

    /// Get weights for each sub-retriever.
    fn retriever_weights(&self) -> Vec<f32> {
        // Default implementation gives equal weight to all retrievers
        let count = self.sub_retrievers().len();
        vec![1.0 / count as f32; count]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_stats() {
        let mut stats = RetrievalStats::new();
        stats.queries_processed = 100;
        stats.queries_failed = 5;
        stats.total_retrieval_time = std::time::Duration::from_secs(50);
        stats.avg_results_per_query = 8.5;
        stats.update_avg_time();

        assert_eq!(stats.success_rate(), 95.0);
        assert_eq!(stats.queries_per_second(), 2.0);
        assert_eq!(
            stats.avg_retrieval_time,
            std::time::Duration::from_millis(500)
        );
    }

    #[test]
    fn test_retrieval_step() {
        let step = RetrievalStep::new("vector_search", "Perform vector similarity search")
            .with_parameter("top_k", 10)
            .with_estimated_time(std::time::Duration::from_millis(100));

        assert_eq!(step.name, "vector_search");
        assert_eq!(
            step.estimated_time,
            Some(std::time::Duration::from_millis(100))
        );
        assert_eq!(
            step.parameters.get("top_k"),
            Some(&serde_json::Value::Number(10.into()))
        );
    }

    #[test]
    fn test_retrieval_config_default() {
        let config = RetrievalConfig::default();
        assert_eq!(config.default_top_k, 10);
        assert_eq!(config.max_top_k, 100);
        assert!(!config.enable_caching);
        assert_eq!(config.timeout_seconds, Some(30));
    }
}
