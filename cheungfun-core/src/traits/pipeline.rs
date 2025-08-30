//! Pipeline traits for orchestrating RAG operations.
//!
//! This module defines traits for high-level pipeline operations that
//! coordinate multiple components to perform indexing and querying.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::{QueryResponse, Result};

/// Indexing pipeline for processing documents into searchable nodes.
///
/// This trait orchestrates the complete indexing process from loading
/// documents to storing them in vector databases.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::IndexingPipeline;
/// use cheungfun_core::{Result, IndexingStats, IndexingProgress};
/// use async_trait::async_trait;
///
/// struct SimpleIndexingPipeline;
///
/// #[async_trait]
/// impl IndexingPipeline for SimpleIndexingPipeline {
///     async fn run(&self) -> Result<IndexingStats> {
///         // Implementation would run the complete indexing pipeline
///         Ok(IndexingStats::new())
///     }
///
///     async fn run_with_progress(
///         &self,
///         progress_callback: Box<dyn Fn(IndexingProgress) + Send + Sync>,
///     ) -> Result<IndexingStats> {
///         // Implementation would report progress during indexing
///         self.run().await
///     }
///
///     fn validate(&self) -> Result<()> {
///         // Implementation would validate pipeline configuration
///         Ok(())
///     }
/// }
/// ```
#[async_trait]
pub trait IndexingPipeline: Send + Sync {
    /// Run the complete indexing pipeline.
    ///
    /// This method executes all steps of the indexing process:
    /// 1. Load documents from sources
    /// 2. Transform documents into nodes
    /// 3. Generate embeddings for nodes
    /// 4. Store nodes in vector database
    ///
    /// # Returns
    ///
    /// Statistics about the indexing operation including number of
    /// documents processed, nodes created, and any errors encountered.
    ///
    /// # Errors
    ///
    /// Returns an error if any critical step in the pipeline fails.
    async fn run(&self) -> Result<IndexingStats>;

    /// Run pipeline with progress reporting.
    ///
    /// This method provides the same functionality as `run()` but calls
    /// the provided callback function to report progress during execution.
    ///
    /// # Arguments
    ///
    /// * `progress_callback` - Function called with progress updates
    async fn run_with_progress(
        &self,
        progress_callback: Box<dyn Fn(IndexingProgress) + Send + Sync>,
    ) -> Result<IndexingStats>;

    /// Validate pipeline configuration before running.
    ///
    /// This method checks that all required components are properly
    /// configured and can communicate with their dependencies.
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline configuration is invalid.
    fn validate(&self) -> Result<()>;

    /// Get a human-readable name for this pipeline.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Get configuration information about this pipeline.
    fn config(&self) -> HashMap<String, serde_json::Value> {
        // Default implementation returns empty config
        HashMap::new()
    }

    /// Estimate the time required to complete indexing.
    async fn estimate_duration(&self) -> Result<Option<std::time::Duration>> {
        // Default implementation returns no estimate
        Ok(None)
    }

    /// Get the current status of the pipeline.
    async fn status(&self) -> Result<PipelineStatus> {
        Ok(PipelineStatus::Ready)
    }

    /// Cancel a running indexing operation.
    async fn cancel(&self) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }
}

/// Query pipeline for answering questions using RAG.
///
/// This trait orchestrates the complete query process from retrieval
/// to response generation.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::QueryPipeline;
/// use cheungfun_core::{QueryResponse, QueryOptions, Result};
/// use async_trait::async_trait;
/// use futures::Stream;
/// use std::pin::Pin;
///
/// struct SimpleQueryPipeline;
///
/// #[async_trait]
/// impl QueryPipeline for SimpleQueryPipeline {
///     async fn query(&self, query: &str, options: &QueryOptions) -> Result<QueryResponse> {
///         // Implementation would execute the complete query pipeline
///         todo!("Implement query processing")
///     }
///
///     async fn query_stream(
///         &self,
///         query: &str,
///         options: &QueryOptions,
///     ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
///         // Implementation would return streaming response
///         Ok(Box::pin(futures::stream::empty()))
///     }
/// }
/// ```
#[async_trait]
pub trait QueryPipeline: Send + Sync {
    /// Execute a query and return a complete response.
    ///
    /// This method executes all steps of the query process:
    /// 1. Process and embed the query
    /// 2. Retrieve relevant nodes from vector store
    /// 3. Generate response using LLM
    ///
    /// # Arguments
    ///
    /// * `query` - The user's question or query
    /// * `options` - Query options and parameters
    ///
    /// # Returns
    ///
    /// A complete query response including the generated answer,
    /// source nodes, and metadata about the query execution.
    ///
    /// # Errors
    ///
    /// Returns an error if any step in the query pipeline fails.
    async fn query(&self, query: &str, options: &QueryOptions) -> Result<QueryResponse>;

    /// Execute query with streaming response.
    ///
    /// This method provides the same functionality as `query()` but
    /// returns a stream of text chunks for real-time display.
    ///
    /// # Arguments
    ///
    /// * `query` - The user's question or query
    /// * `options` - Query options and parameters
    ///
    /// # Returns
    ///
    /// A stream that yields response text chunks as they are generated.
    async fn query_stream(
        &self,
        query: &str,
        options: &QueryOptions,
    ) -> Result<std::pin::Pin<Box<dyn futures::Stream<Item = Result<String>> + Send>>>;

    /// Get a human-readable name for this pipeline.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Validate pipeline configuration.
    fn validate(&self) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Get configuration information about this pipeline.
    fn config(&self) -> HashMap<String, serde_json::Value> {
        // Default implementation returns empty config
        HashMap::new()
    }

    /// Check if the pipeline is healthy and ready to process queries.
    async fn health_check(&self) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Get statistics about query operations.
    async fn stats(&self) -> Result<QueryStats> {
        Ok(QueryStats::default())
    }
}

/// Statistics about indexing operations.
#[derive(Debug, Clone)]
pub struct IndexingStats {
    /// Number of documents processed.
    pub documents_processed: usize,

    /// Number of nodes created.
    pub nodes_created: usize,

    /// Number of nodes successfully stored.
    pub nodes_stored: usize,

    /// Total processing time.
    pub processing_time: std::time::Duration,

    /// List of errors encountered.
    pub errors: Vec<String>,

    /// Additional pipeline-specific statistics.
    pub additional_stats: HashMap<String, serde_json::Value>,
}

impl IndexingStats {
    /// Create new indexing statistics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            documents_processed: 0,
            nodes_created: 0,
            nodes_stored: 0,
            processing_time: std::time::Duration::ZERO,
            errors: Vec::new(),
            additional_stats: HashMap::new(),
        }
    }

    /// Calculate the success rate for document processing.
    #[must_use]
    pub fn document_success_rate(&self) -> f64 {
        if self.documents_processed == 0 {
            0.0
        } else {
            // Assume success if no errors for that document
            let failed_docs = self.errors.len().min(self.documents_processed);
            let successful_docs = self.documents_processed - failed_docs;
            (successful_docs as f64 / self.documents_processed as f64) * 100.0
        }
    }

    /// Calculate the storage success rate for nodes.
    #[must_use]
    pub fn storage_success_rate(&self) -> f64 {
        if self.nodes_created == 0 {
            0.0
        } else {
            (self.nodes_stored as f64 / self.nodes_created as f64) * 100.0
        }
    }

    /// Calculate average nodes per document.
    #[must_use]
    pub fn avg_nodes_per_document(&self) -> f64 {
        if self.documents_processed == 0 {
            0.0
        } else {
            self.nodes_created as f64 / self.documents_processed as f64
        }
    }

    /// Calculate processing speed (documents per second).
    #[must_use]
    pub fn documents_per_second(&self) -> f64 {
        if self.processing_time.is_zero() {
            0.0
        } else {
            self.documents_processed as f64 / self.processing_time.as_secs_f64()
        }
    }
}

impl Default for IndexingStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Progress information for indexing operations.
#[derive(Debug, Clone)]
pub struct IndexingProgress {
    /// Current stage of the pipeline.
    pub stage: String,

    /// Number of items processed in current stage.
    pub processed: usize,

    /// Total number of items to process (if known).
    pub total: Option<usize>,

    /// Name of the current item being processed.
    pub current_item: Option<String>,

    /// Estimated time remaining.
    pub estimated_remaining: Option<std::time::Duration>,

    /// Additional progress metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl IndexingProgress {
    /// Create new indexing progress.
    pub fn new<S: Into<String>>(stage: S, processed: usize) -> Self {
        Self {
            stage: stage.into(),
            processed,
            total: None,
            current_item: None,
            estimated_remaining: None,
            metadata: HashMap::new(),
        }
    }

    /// Calculate progress percentage if total is known.
    #[must_use]
    pub fn percentage(&self) -> Option<f64> {
        self.total.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.processed as f64 / total as f64) * 100.0
            }
        })
    }

    /// Check if the operation is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.total.is_some_and(|total| self.processed >= total)
    }
}

/// Statistics about query operations.
#[derive(Debug, Clone, Default)]
pub struct QueryStats {
    /// Total number of queries processed.
    pub queries_processed: usize,

    /// Number of queries that failed.
    pub queries_failed: usize,

    /// Average query processing time.
    pub avg_query_time: std::time::Duration,

    /// Average retrieval time.
    pub avg_retrieval_time: std::time::Duration,

    /// Average generation time.
    pub avg_generation_time: std::time::Duration,

    /// Total query processing time.
    pub total_query_time: std::time::Duration,

    /// Additional pipeline-specific statistics.
    pub additional_stats: HashMap<String, serde_json::Value>,
}

impl QueryStats {
    /// Create new query statistics.
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
        if self.total_query_time.is_zero() {
            0.0
        } else {
            self.queries_processed as f64 / self.total_query_time.as_secs_f64()
        }
    }
}

/// Status of a pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStatus {
    /// Pipeline is ready to run.
    Ready,

    /// Pipeline is currently running.
    Running,

    /// Pipeline has completed successfully.
    Completed,

    /// Pipeline has failed.
    Failed,

    /// Pipeline has been cancelled.
    Cancelled,

    /// Pipeline is in an unknown state.
    Unknown,
}

/// Options for query execution.
///
/// This is a placeholder that will be properly defined when we implement
/// the configuration system.
#[derive(Debug, Clone, Default)]
pub struct QueryOptions {
    /// Additional query options.
    pub additional_options: HashMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexing_stats() {
        let mut stats = IndexingStats::new();
        stats.documents_processed = 100;
        stats.nodes_created = 250;
        stats.nodes_stored = 240;
        stats.processing_time = std::time::Duration::from_secs(50);
        stats.errors = vec!["Error 1".to_string(), "Error 2".to_string()];

        assert_eq!(stats.storage_success_rate(), 96.0);
        assert_eq!(stats.avg_nodes_per_document(), 2.5);
        assert_eq!(stats.documents_per_second(), 2.0);
    }

    #[test]
    fn test_indexing_progress() {
        let mut progress = IndexingProgress::new("loading", 50);
        progress.total = Some(100);

        assert_eq!(progress.percentage(), Some(50.0));
        assert!(!progress.is_complete());

        progress.processed = 100;
        assert!(progress.is_complete());
    }

    #[test]
    fn test_query_stats() {
        let mut stats = QueryStats::new();
        stats.queries_processed = 100;
        stats.queries_failed = 5;
        stats.total_query_time = std::time::Duration::from_secs(50);

        assert_eq!(stats.success_rate(), 95.0);
        assert_eq!(stats.queries_per_second(), 2.0);
    }

    #[test]
    fn test_pipeline_status() {
        assert_eq!(PipelineStatus::Ready, PipelineStatus::Ready);
        assert_ne!(PipelineStatus::Ready, PipelineStatus::Running);
    }
}
