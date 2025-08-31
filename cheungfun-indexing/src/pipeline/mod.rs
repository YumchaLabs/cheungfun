//! Pipeline implementations for the Cheungfun indexing framework.
//!
//! This module provides different types of pipelines for various data processing needs:
//!
//! - **IngestionPipeline**: Data preprocessing pipeline that transforms documents into nodes
//! - **IndexingPipeline**: Complete indexing pipeline that builds searchable indices
//! - **QueryPipeline**: Query processing pipeline (future extension)
//!
//! The design follows LlamaIndex's architecture where different pipelines have distinct
//! responsibilities and can be composed together for complex workflows.

pub mod indexing;
pub mod ingestion;

// Re-export the main pipeline types for convenience
pub use indexing::{DefaultIndexingPipeline, PipelineConfig};
pub use ingestion::{IngestionConfig, IngestionPipeline, IngestionStats};

/// Common pipeline traits and utilities.
pub mod common {
    use std::collections::HashMap;
    use std::time::Duration;

    /// Progress information for pipeline operations.
    #[derive(Debug, Clone)]
    pub struct PipelineProgress {
        /// Current step being processed.
        pub current_step: String,
        /// Number of items processed so far.
        pub processed: usize,
        /// Total number of items to process.
        pub total: usize,
        /// Percentage completion (0.0 to 1.0).
        pub percentage: f64,
        /// Elapsed time since pipeline started.
        pub elapsed: Duration,
        /// Estimated time remaining.
        pub estimated_remaining: Option<Duration>,
    }

    impl PipelineProgress {
        /// Create a new progress instance.
        pub fn new(
            current_step: String,
            processed: usize,
            total: usize,
            elapsed: Duration,
        ) -> Self {
            let percentage = if total > 0 {
                processed as f64 / total as f64
            } else {
                0.0
            };

            let estimated_remaining = if processed > 0 && percentage < 1.0 {
                let rate = processed as f64 / elapsed.as_secs_f64();
                let remaining_items = total - processed;
                Some(Duration::from_secs_f64(remaining_items as f64 / rate))
            } else {
                None
            };

            Self {
                current_step,
                processed,
                total,
                percentage,
                elapsed,
                estimated_remaining,
            }
        }
    }

    /// Common statistics collected by pipelines.
    #[derive(Debug, Clone)]
    pub struct CommonStats {
        /// Total processing time.
        pub processing_time: Duration,
        /// Number of errors encountered.
        pub error_count: usize,
        /// List of error messages.
        pub errors: Vec<String>,
        /// Additional custom statistics.
        pub additional_stats: HashMap<String, serde_json::Value>,
    }

    impl Default for CommonStats {
        fn default() -> Self {
            Self {
                processing_time: Duration::from_secs(0),
                error_count: 0,
                errors: Vec::new(),
                additional_stats: HashMap::new(),
            }
        }
    }

    /// Trait for pipeline components that can report progress.
    pub trait ProgressReporter {
        /// Report progress during pipeline execution.
        fn report_progress(&self, progress: PipelineProgress);
    }

    /// Default progress reporter that logs to the console.
    pub struct ConsoleProgressReporter;

    impl ProgressReporter for ConsoleProgressReporter {
        fn report_progress(&self, progress: PipelineProgress) {
            use tracing::info;

            let remaining_str = if let Some(remaining) = progress.estimated_remaining {
                format!(" (ETA: {:?})", remaining)
            } else {
                String::new()
            };

            info!(
                "📊 {}: {}/{} ({:.1}%){} - Elapsed: {:?}",
                progress.current_step,
                progress.processed,
                progress.total,
                progress.percentage * 100.0,
                remaining_str,
                progress.elapsed
            );
        }
    }
}

/// Pipeline factory for creating different types of pipelines.
pub struct PipelineFactory;

impl PipelineFactory {
    /// Create a new ingestion pipeline builder.
    pub fn ingestion() -> ingestion::IngestionPipelineBuilder {
        ingestion::IngestionPipeline::builder()
    }

    /// Create a new indexing pipeline builder.
    pub fn indexing() -> indexing::PipelineBuilder {
        indexing::DefaultIndexingPipeline::builder()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_pipeline_progress() {
        let progress = common::PipelineProgress::new(
            "Processing documents".to_string(),
            50,
            100,
            Duration::from_secs(30),
        );

        assert_eq!(progress.processed, 50);
        assert_eq!(progress.total, 100);
        assert_eq!(progress.percentage, 0.5);
        assert!(progress.estimated_remaining.is_some());
    }

    #[test]
    fn test_common_stats() {
        let stats = common::CommonStats::default();
        assert_eq!(stats.error_count, 0);
        assert!(stats.errors.is_empty());
        assert!(stats.additional_stats.is_empty());
    }

    #[test]
    fn test_pipeline_factory() {
        let _ingestion_builder = PipelineFactory::ingestion();
        let _indexing_builder = PipelineFactory::indexing();
    }
}
