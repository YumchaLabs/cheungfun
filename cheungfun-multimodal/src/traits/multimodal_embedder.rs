//! Multimodal embedding generation traits.
//!
//! This module defines traits for generating embeddings from different modalities
//! and performing cross-modal embedding operations.

use crate::error::Result;
use crate::types::{MediaContent, ModalityType};
use async_trait::async_trait;
use std::collections::HashMap;

/// Trait for generating embeddings from multimodal content.
///
/// This trait extends the basic embedding functionality to support multiple
/// modalities and cross-modal operations.
#[async_trait]
pub trait MultimodalEmbedder: Send + Sync + std::fmt::Debug {
    /// Get the list of supported modalities.
    fn supported_modalities(&self) -> Vec<ModalityType>;

    /// Generate embedding for content of a specific modality.
    ///
    /// # Arguments
    ///
    /// * `content` - The media content to embed
    /// * `modality` - The modality type of the content
    ///
    /// # Returns
    ///
    /// A vector of floating-point numbers representing the embedding.
    async fn embed_modality(
        &self,
        content: &MediaContent,
        modality: ModalityType,
    ) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple pieces of content in batch.
    ///
    /// This method is more efficient than calling `embed_modality` multiple times
    /// as it can leverage batch processing capabilities.
    async fn embed_batch_multimodal(
        &self,
        items: Vec<(MediaContent, ModalityType)>,
    ) -> Result<Vec<Vec<f32>>>;

    /// Generate cross-modal embedding (unified vector space).
    ///
    /// This method generates embeddings that can be compared across different
    /// modalities, enabling cross-modal search and retrieval.
    async fn embed_cross_modal(
        &self,
        content: &MediaContent,
        modality: ModalityType,
    ) -> Result<Vec<f32>>;

    /// Get the dimension of embeddings produced by this embedder.
    fn embedding_dimension(&self) -> usize;

    /// Get the dimension of cross-modal embeddings.
    fn cross_modal_dimension(&self) -> usize {
        self.embedding_dimension()
    }

    /// Check if cross-modal operations are supported between two modalities.
    fn supports_cross_modal(&self, from: ModalityType, to: ModalityType) -> bool;

    /// Get the name/identifier of the embedding model.
    fn model_name(&self) -> &str;

    /// Get a human-readable name for this embedder.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Check if the embedder is healthy and ready to generate embeddings.
    async fn health_check(&self) -> Result<()> {
        Ok(())
    }

    /// Get metadata about the embedding model.
    fn metadata(&self) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();
        metadata.insert("model_name".to_string(), self.model_name().into());
        metadata.insert(
            "embedding_dimension".to_string(),
            self.embedding_dimension().into(),
        );
        metadata.insert(
            "cross_modal_dimension".to_string(),
            self.cross_modal_dimension().into(),
        );
        metadata.insert(
            "supported_modalities".to_string(),
            self.supported_modalities()
                .iter()
                .map(|m| m.to_string())
                .collect::<Vec<_>>()
                .into(),
        );
        metadata
    }

    /// Get the maximum content size this embedder can handle for a modality.
    fn max_content_size(&self, _modality: ModalityType) -> Option<u64> {
        None
    }

    /// Preprocess content before embedding.
    async fn preprocess_content(&self, content: &MediaContent) -> Result<MediaContent> {
        Ok(content.clone())
    }

    /// Postprocess embeddings after generation.
    fn postprocess_embedding(&self, embedding: Vec<f32>) -> Result<Vec<f32>> {
        Ok(embedding)
    }
}

/// Trait for embedders that support similarity computation.
#[async_trait]
pub trait MultimodalSimilarity: MultimodalEmbedder {
    /// Compute similarity between two embeddings.
    fn compute_similarity(&self, embedding1: &[f32], embedding2: &[f32]) -> Result<f32>;

    /// Compute cross-modal similarity between content of different modalities.
    async fn compute_cross_modal_similarity(
        &self,
        content1: &MediaContent,
        modality1: ModalityType,
        content2: &MediaContent,
        modality2: ModalityType,
    ) -> Result<f32> {
        let embedding1 = self.embed_cross_modal(content1, modality1).await?;
        let embedding2 = self.embed_cross_modal(content2, modality2).await?;
        self.compute_similarity(&embedding1, &embedding2)
    }

    /// Find the most similar content from a collection.
    async fn find_most_similar(
        &self,
        query_content: &MediaContent,
        query_modality: ModalityType,
        candidates: &[(MediaContent, ModalityType)],
    ) -> Result<Option<(usize, f32)>> {
        if candidates.is_empty() {
            return Ok(None);
        }

        let query_embedding = self
            .embed_cross_modal(query_content, query_modality)
            .await?;
        let mut best_index = 0;
        let mut best_similarity = f32::NEG_INFINITY;

        for (i, (content, modality)) in candidates.iter().enumerate() {
            let candidate_embedding = self.embed_cross_modal(content, *modality).await?;
            let similarity = self.compute_similarity(&query_embedding, &candidate_embedding)?;

            if similarity > best_similarity {
                best_similarity = similarity;
                best_index = i;
            }
        }

        Ok(Some((best_index, best_similarity)))
    }
}

/// Trait for embedders that support modality-specific operations.
#[async_trait]
pub trait ModalitySpecificEmbedder: MultimodalEmbedder {
    /// Get modality-specific configuration.
    fn get_modality_config(
        &self,
        modality: ModalityType,
    ) -> Option<HashMap<String, serde_json::Value>>;

    /// Set modality-specific configuration.
    fn set_modality_config(
        &mut self,
        modality: ModalityType,
        config: HashMap<String, serde_json::Value>,
    ) -> Result<()>;

    /// Get the optimal batch size for a specific modality.
    fn optimal_batch_size(&self, modality: ModalityType) -> usize {
        match modality {
            ModalityType::Text => 32,
            ModalityType::Image => 8,
            ModalityType::Audio => 16,
            ModalityType::Video => 4,
            ModalityType::Document => 16,
            _ => 8,
        }
    }

    /// Check if a specific modality requires special preprocessing.
    fn requires_preprocessing(&self, modality: ModalityType) -> bool {
        match modality {
            ModalityType::Image | ModalityType::Audio | ModalityType::Video => true,
            _ => false,
        }
    }
}

/// Configuration for multimodal embedding operations.
#[derive(Debug, Clone)]
pub struct MultimodalEmbeddingConfig {
    /// Maximum batch size for embedding operations.
    pub batch_size: Option<usize>,

    /// Timeout for embedding operations in seconds.
    pub timeout_seconds: Option<u64>,

    /// Number of retry attempts for failed embeddings.
    pub max_retries: Option<usize>,

    /// Whether to normalize embeddings to unit length.
    pub normalize: bool,

    /// Whether to enable cross-modal embeddings.
    pub enable_cross_modal: bool,

    /// Modality-specific configurations.
    pub modality_configs: HashMap<ModalityType, HashMap<String, serde_json::Value>>,

    /// Additional model-specific configuration.
    pub model_config: HashMap<String, serde_json::Value>,
}

impl Default for MultimodalEmbeddingConfig {
    fn default() -> Self {
        Self {
            batch_size: Some(16),
            timeout_seconds: Some(60),
            max_retries: Some(3),
            normalize: false,
            enable_cross_modal: true,
            modality_configs: HashMap::new(),
            model_config: HashMap::new(),
        }
    }
}

/// Statistics about multimodal embedding operations.
#[derive(Debug, Clone)]
pub struct MultimodalEmbeddingStats {
    /// Number of embeddings generated per modality.
    pub embeddings_per_modality: HashMap<ModalityType, usize>,

    /// Number of cross-modal embeddings generated.
    pub cross_modal_embeddings: usize,

    /// Number of embedding operations that failed per modality.
    pub failures_per_modality: HashMap<ModalityType, usize>,

    /// Total time taken for embedding operations.
    pub duration: std::time::Duration,

    /// Average time per embedding per modality.
    pub avg_time_per_modality: HashMap<ModalityType, std::time::Duration>,

    /// Total content size processed per modality.
    pub content_size_per_modality: HashMap<ModalityType, u64>,

    /// List of errors encountered.
    pub errors: Vec<String>,

    /// Additional statistics.
    pub additional_stats: HashMap<String, serde_json::Value>,
}

impl MultimodalEmbeddingStats {
    /// Create new multimodal embedding statistics.
    pub fn new() -> Self {
        Self {
            embeddings_per_modality: HashMap::new(),
            cross_modal_embeddings: 0,
            failures_per_modality: HashMap::new(),
            duration: std::time::Duration::ZERO,
            avg_time_per_modality: HashMap::new(),
            content_size_per_modality: HashMap::new(),
            errors: Vec::new(),
            additional_stats: HashMap::new(),
        }
    }

    /// Get the total number of embeddings generated.
    pub fn total_embeddings(&self) -> usize {
        self.embeddings_per_modality.values().sum::<usize>() + self.cross_modal_embeddings
    }

    /// Get the total number of failures.
    pub fn total_failures(&self) -> usize {
        self.failures_per_modality.values().sum()
    }

    /// Calculate the overall success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        let total = self.total_embeddings() + self.total_failures();
        if total == 0 {
            0.0
        } else {
            (self.total_embeddings() as f64 / total as f64) * 100.0
        }
    }

    /// Get the success rate for a specific modality.
    pub fn modality_success_rate(&self, modality: ModalityType) -> f64 {
        let successes = self.embeddings_per_modality.get(&modality).unwrap_or(&0);
        let failures = self.failures_per_modality.get(&modality).unwrap_or(&0);
        let total = successes + failures;

        if total == 0 {
            0.0
        } else {
            (*successes as f64 / total as f64) * 100.0
        }
    }

    /// Calculate throughput (embeddings per second).
    pub fn throughput(&self) -> f64 {
        if self.duration.is_zero() {
            0.0
        } else {
            self.total_embeddings() as f64 / self.duration.as_secs_f64()
        }
    }

    /// Calculate throughput for a specific modality.
    pub fn modality_throughput(&self, modality: ModalityType) -> f64 {
        if let (Some(count), Some(duration)) = (
            self.embeddings_per_modality.get(&modality),
            self.avg_time_per_modality.get(&modality),
        ) {
            if !duration.is_zero() {
                *count as f64 / duration.as_secs_f64()
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

impl Default for MultimodalEmbeddingStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_embedding_config_default() {
        let config = MultimodalEmbeddingConfig::default();
        assert_eq!(config.batch_size, Some(16));
        assert_eq!(config.timeout_seconds, Some(60));
        assert_eq!(config.max_retries, Some(3));
        assert!(!config.normalize);
        assert!(config.enable_cross_modal);
    }

    #[test]
    fn test_multimodal_embedding_stats() {
        let mut stats = MultimodalEmbeddingStats::new();
        stats.embeddings_per_modality.insert(ModalityType::Text, 10);
        stats.embeddings_per_modality.insert(ModalityType::Image, 5);
        stats.failures_per_modality.insert(ModalityType::Text, 1);
        stats.cross_modal_embeddings = 3;

        assert_eq!(stats.total_embeddings(), 18);
        assert_eq!(stats.total_failures(), 1);
        assert!((stats.success_rate() - 94.74).abs() < 0.01);
        assert!((stats.modality_success_rate(ModalityType::Text) - 90.91).abs() < 0.01);
        assert_eq!(stats.modality_success_rate(ModalityType::Image), 100.0);
    }
}
