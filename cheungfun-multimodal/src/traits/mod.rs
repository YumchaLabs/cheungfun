//! Traits for multimodal operations.
//!
//! This module provides the core trait definitions for multimodal processing,
//! embedding generation, and cross-modal retrieval operations.

pub mod cross_modal_retriever;
pub mod media_processor;
pub mod multimodal_embedder;

// Re-export commonly used traits and types
pub use cross_modal_retriever::{
    CrossModalQuery, CrossModalRetriever, CrossModalRetrievalOptions,
    HybridMultimodalRetriever, HybridRetrievalOptions, QueryContent,
    QueryExpansionOptions, ScoredMultimodalNode, SimilarityBasedRetriever,
    FusionStrategy,
};

pub use media_processor::{
    BoundingBox, CompressionSettings, ContentAnalysis, ContentAnalyzer,
    ContentCategory, ConversionOptions, DetectedEntity, FeatureExtractor,
    FormatConverter, MediaProcessor, ProcessingOptions, TimeRange,
};

pub use multimodal_embedder::{
    ModalitySpecificEmbedder, MultimodalEmbedder, MultimodalEmbeddingConfig,
    MultimodalEmbeddingStats, MultimodalSimilarity,
};

/// Common trait bounds for multimodal components.
pub trait MultimodalComponent: Send + Sync + std::fmt::Debug {
    /// Get the name of this component.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Check if the component is healthy and ready.
    fn health_check(&self) -> crate::error::Result<()> {
        Ok(())
    }

    /// Get metadata about the component.
    fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("name".to_string(), self.name().into());
        metadata.insert("type".to_string(), "multimodal_component".into());
        metadata
    }
}

/// Trait for components that can be configured.
pub trait Configurable {
    /// Configuration type for this component.
    type Config;

    /// Apply configuration to the component.
    fn configure(&mut self, config: Self::Config) -> crate::error::Result<()>;

    /// Get the current configuration.
    fn get_config(&self) -> Self::Config;

    /// Validate a configuration before applying it.
    fn validate_config(_config: &Self::Config) -> crate::error::Result<()> {
        // Default implementation does no validation
        Ok(())
    }
}

/// Trait for components that support batch operations.
#[async_trait::async_trait]
pub trait BatchProcessor<Input, Output> {
    /// Process a batch of inputs.
    async fn process_batch(&self, inputs: Vec<Input>) -> crate::error::Result<Vec<Output>>;

    /// Get the optimal batch size for this processor.
    fn optimal_batch_size(&self) -> usize {
        32
    }

    /// Get the maximum batch size supported.
    fn max_batch_size(&self) -> Option<usize> {
        None
    }
}

/// Trait for components that support streaming operations.
#[async_trait::async_trait]
pub trait StreamProcessor<Input, Output> {
    /// Process a stream of inputs.
    async fn process_stream(
        &self,
        input_stream: tokio_stream::wrappers::ReceiverStream<Input>,
    ) -> crate::error::Result<tokio_stream::wrappers::ReceiverStream<Output>>;

    /// Get the buffer size for streaming operations.
    fn stream_buffer_size(&self) -> usize {
        1000
    }
}

/// Trait for components that support caching.
pub trait Cacheable {
    /// Cache key type.
    type Key: std::hash::Hash + Eq + Clone;

    /// Cache value type.
    type Value: Clone;

    /// Generate a cache key for the given input.
    fn cache_key(&self, input: &Self::Key) -> String;

    /// Check if caching is enabled.
    fn is_cache_enabled(&self) -> bool {
        true
    }

    /// Get the cache TTL in seconds.
    fn cache_ttl(&self) -> Option<u64> {
        Some(3600) // 1 hour default
    }
}

/// Trait for components that support metrics collection.
pub trait MetricsCollector {
    /// Record a metric value.
    fn record_metric(&self, name: &str, value: f64, tags: Option<&[(&str, &str)]>);

    /// Record a counter increment.
    fn increment_counter(&self, name: &str, tags: Option<&[(&str, &str)]>) {
        self.record_metric(name, 1.0, tags);
    }

    /// Record a timing metric.
    fn record_timing(&self, name: &str, duration: std::time::Duration, tags: Option<&[(&str, &str)]>) {
        self.record_metric(name, duration.as_secs_f64(), tags);
    }

    /// Record a gauge value.
    fn record_gauge(&self, name: &str, value: f64, tags: Option<&[(&str, &str)]>) {
        self.record_metric(name, value, tags);
    }
}

/// Trait for components that support progress reporting.
pub trait ProgressReporter {
    /// Progress information.
    type Progress;

    /// Report progress for an operation.
    fn report_progress(&self, progress: Self::Progress);

    /// Check if progress reporting is enabled.
    fn is_progress_enabled(&self) -> bool {
        false
    }
}

/// Common progress information for multimodal operations.
#[derive(Debug, Clone)]
pub struct MultimodalProgress {
    /// Current step in the operation.
    pub current_step: usize,

    /// Total number of steps.
    pub total_steps: Option<usize>,

    /// Current operation description.
    pub operation: String,

    /// Progress percentage (0.0 to 100.0).
    pub percentage: Option<f32>,

    /// Estimated time remaining.
    pub eta: Option<std::time::Duration>,

    /// Additional progress data.
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl MultimodalProgress {
    /// Create a new progress report.
    pub fn new<S: Into<String>>(current_step: usize, operation: S) -> Self {
        Self {
            current_step,
            total_steps: None,
            operation: operation.into(),
            percentage: None,
            eta: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set the total number of steps.
    pub fn with_total_steps(mut self, total: usize) -> Self {
        self.total_steps = Some(total);
        if total > 0 {
            self.percentage = Some((self.current_step as f32 / total as f32) * 100.0);
        }
        self
    }

    /// Set the estimated time remaining.
    pub fn with_eta(mut self, eta: std::time::Duration) -> Self {
        self.eta = Some(eta);
        self
    }

    /// Add metadata to the progress report.
    pub fn with_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if the operation is complete.
    pub fn is_complete(&self) -> bool {
        if let Some(total) = self.total_steps {
            self.current_step >= total
        } else {
            false
        }
    }
}

/// Utility functions for working with multimodal traits.
pub mod utils {
    use super::*;
    use crate::types::ModalityType;

    /// Check if two components are compatible for chaining.
    pub fn are_components_compatible<T1, T2>(_comp1: &T1, _comp2: &T2) -> bool
    where
        T1: MultimodalComponent,
        T2: MultimodalComponent,
    {
        // Basic compatibility check - can be extended with more sophisticated logic
        true
    }

    /// Get the processing order for multiple modalities.
    pub fn get_processing_order(modalities: &[ModalityType]) -> Vec<ModalityType> {
        let mut ordered = modalities.to_vec();
        ordered.sort_by_key(|m| crate::types::utils::modality_priority(*m));
        ordered
    }

    /// Calculate the optimal batch size for a given modality.
    pub fn optimal_batch_size_for_modality(modality: ModalityType) -> usize {
        match modality {
            ModalityType::Text => 64,
            ModalityType::Image => 16,
            ModalityType::Audio => 32,
            ModalityType::Video => 8,
            ModalityType::Document => 24,
            _ => 16,
        }
    }

    /// Estimate processing time for a modality and content size.
    pub fn estimate_processing_time(modality: ModalityType, content_size: u64) -> std::time::Duration {
        let base_time_ms = match modality {
            ModalityType::Text => 10,
            ModalityType::Image => 100,
            ModalityType::Audio => 200,
            ModalityType::Video => 500,
            ModalityType::Document => 150,
            _ => 100,
        };

        // Scale based on content size (rough estimate)
        let size_factor = (content_size as f64 / 1_000_000.0).max(1.0); // MB
        let estimated_ms = (base_time_ms as f64 * size_factor) as u64;

        std::time::Duration::from_millis(estimated_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_progress() {
        let progress = MultimodalProgress::new(5, "Processing images")
            .with_total_steps(10)
            .with_eta(std::time::Duration::from_secs(30));

        assert_eq!(progress.current_step, 5);
        assert_eq!(progress.total_steps, Some(10));
        assert_eq!(progress.percentage, Some(50.0));
        assert!(!progress.is_complete());

        let complete_progress = MultimodalProgress::new(10, "Done").with_total_steps(10);
        assert!(complete_progress.is_complete());
    }

    #[test]
    fn test_processing_order() {
        use crate::types::ModalityType;

        let modalities = vec![
            ModalityType::Video,
            ModalityType::Text,
            ModalityType::Image,
            ModalityType::Audio,
        ];

        let ordered = utils::get_processing_order(&modalities);
        assert_eq!(ordered[0], ModalityType::Text); // Highest priority
    }

    #[test]
    fn test_optimal_batch_size() {
        use crate::types::ModalityType;

        assert_eq!(utils::optimal_batch_size_for_modality(ModalityType::Text), 64);
        assert_eq!(utils::optimal_batch_size_for_modality(ModalityType::Image), 16);
        assert_eq!(utils::optimal_batch_size_for_modality(ModalityType::Video), 8);
    }

    #[test]
    fn test_processing_time_estimation() {
        use crate::types::ModalityType;

        let text_time = utils::estimate_processing_time(ModalityType::Text, 1_000_000);
        let video_time = utils::estimate_processing_time(ModalityType::Video, 1_000_000);

        assert!(video_time > text_time);
    }
}
