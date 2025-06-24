//! Cross-modal retrieval traits for multimodal search.
//!
//! This module defines traits for performing cross-modal retrieval operations,
//! enabling search across different modalities (e.g., text query for images).

use crate::error::Result;
use crate::types::{MediaContent, ModalityType, MultimodalNode};
use async_trait::async_trait;
use cheungfun_core::types::ScoredNode;
use std::collections::HashMap;
use uuid::Uuid;

/// Trait for cross-modal retrieval operations.
///
/// This trait enables searching for content of one modality using queries
/// from another modality (e.g., finding images using text descriptions).
#[async_trait]
pub trait CrossModalRetriever: Send + Sync + std::fmt::Debug {
    /// Retrieve content across modalities.
    ///
    /// # Arguments
    ///
    /// * `query` - The query content and modality
    /// * `target_modality` - The modality of content to retrieve
    /// * `options` - Retrieval options and parameters
    ///
    /// # Returns
    ///
    /// A list of scored multimodal nodes matching the query.
    async fn retrieve_cross_modal(
        &self,
        query: &CrossModalQuery,
        target_modality: ModalityType,
        options: &CrossModalRetrievalOptions,
    ) -> Result<Vec<ScoredMultimodalNode>>;

    /// Retrieve content with multiple target modalities.
    async fn retrieve_multi_target(
        &self,
        query: &CrossModalQuery,
        target_modalities: &[ModalityType],
        options: &CrossModalRetrievalOptions,
    ) -> Result<HashMap<ModalityType, Vec<ScoredMultimodalNode>>>;

    /// Get supported query-target modality pairs.
    fn supported_cross_modal_pairs(&self) -> Vec<(ModalityType, ModalityType)>;

    /// Check if a specific cross-modal retrieval is supported.
    fn supports_cross_modal(
        &self,
        query_modality: ModalityType,
        target_modality: ModalityType,
    ) -> bool {
        self.supported_cross_modal_pairs()
            .contains(&(query_modality, target_modality))
    }

    /// Get the name of this retriever.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Check if the retriever is healthy and ready.
    async fn health_check(&self) -> Result<()> {
        Ok(())
    }

    /// Get metadata about the retriever.
    fn metadata(&self) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();
        metadata.insert("name".to_string(), self.name().into());
        metadata.insert(
            "supported_pairs".to_string(),
            self.supported_cross_modal_pairs()
                .iter()
                .map(|(q, t)| format!("{}→{}", q, t))
                .collect::<Vec<_>>()
                .into(),
        );
        metadata
    }
}

/// Trait for hybrid multimodal retrieval.
///
/// This trait combines multiple retrieval strategies and modalities
/// to provide comprehensive search capabilities.
#[async_trait]
pub trait HybridMultimodalRetriever: CrossModalRetriever {
    /// Perform hybrid retrieval combining multiple strategies.
    async fn retrieve_hybrid(
        &self,
        queries: &[CrossModalQuery],
        options: &HybridRetrievalOptions,
    ) -> Result<Vec<ScoredMultimodalNode>>;

    /// Retrieve with query expansion across modalities.
    async fn retrieve_with_expansion(
        &self,
        query: &CrossModalQuery,
        expansion_options: &QueryExpansionOptions,
    ) -> Result<Vec<ScoredMultimodalNode>>;

    /// Perform multimodal fusion of retrieval results.
    async fn fuse_multimodal_results(
        &self,
        results: HashMap<ModalityType, Vec<ScoredMultimodalNode>>,
        fusion_strategy: FusionStrategy,
    ) -> Result<Vec<ScoredMultimodalNode>>;
}

/// Trait for similarity-based cross-modal retrieval.
#[async_trait]
pub trait SimilarityBasedRetriever: CrossModalRetriever {
    /// Find similar content across modalities using embeddings.
    async fn find_similar_cross_modal(
        &self,
        query_embedding: &[f32],
        query_modality: ModalityType,
        target_modality: ModalityType,
        top_k: usize,
        threshold: Option<f32>,
    ) -> Result<Vec<ScoredMultimodalNode>>;

    /// Compute cross-modal similarity between two pieces of content.
    async fn compute_cross_modal_similarity(
        &self,
        content1: &MediaContent,
        modality1: ModalityType,
        content2: &MediaContent,
        modality2: ModalityType,
    ) -> Result<f32>;

    /// Get the similarity threshold for a modality pair.
    fn similarity_threshold(
        &self,
        query_modality: ModalityType,
        target_modality: ModalityType,
    ) -> f32 {
        match (query_modality, target_modality) {
            // Same modality should have high threshold
            (a, b) if a == b => 0.8,
            // Text-image pairs are well-supported
            (ModalityType::Text, ModalityType::Image)
            | (ModalityType::Image, ModalityType::Text) => 0.7,
            // Video-audio pairs are natural
            (ModalityType::Video, ModalityType::Audio)
            | (ModalityType::Audio, ModalityType::Video) => 0.75,
            // Other cross-modal pairs need lower threshold
            _ => 0.6,
        }
    }
}

/// Cross-modal query structure.
#[derive(Debug, Clone)]
pub struct CrossModalQuery {
    /// Query content (text, image, audio, etc.).
    pub content: QueryContent,

    /// Modality of the query.
    pub modality: ModalityType,

    /// Query metadata and context.
    pub metadata: HashMap<String, serde_json::Value>,

    /// Query ID for tracking.
    pub id: Option<Uuid>,
}

impl CrossModalQuery {
    /// Create a new cross-modal query from text.
    pub fn from_text<S: Into<String>>(text: S) -> Self {
        Self {
            content: QueryContent::Text(text.into()),
            modality: ModalityType::Text,
            metadata: HashMap::new(),
            id: Some(Uuid::new_v4()),
        }
    }

    /// Create a new cross-modal query from media content.
    pub fn from_media(content: MediaContent) -> Self {
        let modality = content.modality_type();
        Self {
            content: QueryContent::Media(content),
            modality,
            metadata: HashMap::new(),
            id: Some(Uuid::new_v4()),
        }
    }

    /// Create a new cross-modal query from embedding.
    pub fn from_embedding(embedding: Vec<f32>, modality: ModalityType) -> Self {
        Self {
            content: QueryContent::Embedding(embedding),
            modality,
            metadata: HashMap::new(),
            id: Some(Uuid::new_v4()),
        }
    }

    /// Add metadata to the query.
    pub fn with_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the query ID.
    pub fn with_id(mut self, id: Uuid) -> Self {
        self.id = Some(id);
        self
    }
}

/// Content of a cross-modal query.
#[derive(Debug, Clone)]
pub enum QueryContent {
    /// Text query.
    Text(String),

    /// Media content query.
    Media(MediaContent),

    /// Pre-computed embedding query.
    Embedding(Vec<f32>),

    /// Hybrid query with multiple content types.
    Hybrid(Vec<QueryContent>),
}

/// Scored multimodal node for retrieval results.
#[derive(Debug, Clone)]
pub struct ScoredMultimodalNode {
    /// The multimodal node.
    pub node: MultimodalNode,

    /// Relevance score.
    pub score: f32,

    /// Modality-specific scores.
    pub modality_scores: HashMap<ModalityType, f32>,

    /// Explanation of the score (optional).
    pub explanation: Option<String>,
}

impl From<ScoredMultimodalNode> for ScoredNode {
    fn from(scored_multimodal: ScoredMultimodalNode) -> Self {
        ScoredNode {
            node: scored_multimodal.node.base,
            score: scored_multimodal.score,
        }
    }
}

/// Options for cross-modal retrieval.
#[derive(Debug, Clone)]
pub struct CrossModalRetrievalOptions {
    /// Maximum number of results to return.
    pub top_k: usize,

    /// Minimum similarity threshold.
    pub threshold: Option<f32>,

    /// Whether to include cross-modal explanations.
    pub include_explanations: bool,

    /// Modality-specific weights for scoring.
    pub modality_weights: HashMap<ModalityType, f32>,

    /// Additional filters.
    pub filters: HashMap<String, serde_json::Value>,

    /// Timeout for retrieval operation.
    pub timeout_seconds: Option<u64>,
}

impl Default for CrossModalRetrievalOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            threshold: None,
            include_explanations: false,
            modality_weights: HashMap::new(),
            filters: HashMap::new(),
            timeout_seconds: Some(30),
        }
    }
}

/// Options for hybrid multimodal retrieval.
#[derive(Debug, Clone)]
pub struct HybridRetrievalOptions {
    /// Base retrieval options.
    pub base_options: CrossModalRetrievalOptions,

    /// Fusion strategy for combining results.
    pub fusion_strategy: FusionStrategy,

    /// Weight for each query in multi-query scenarios.
    pub query_weights: Vec<f32>,

    /// Whether to perform result diversification.
    pub diversify_results: bool,

    /// Diversification factor (0.0 to 1.0).
    pub diversification_factor: f32,
}

impl Default for HybridRetrievalOptions {
    fn default() -> Self {
        Self {
            base_options: CrossModalRetrievalOptions::default(),
            fusion_strategy: FusionStrategy::WeightedAverage,
            query_weights: Vec::new(),
            diversify_results: false,
            diversification_factor: 0.5,
        }
    }
}

/// Options for query expansion.
#[derive(Debug, Clone)]
pub struct QueryExpansionOptions {
    /// Number of expansion terms/concepts to generate.
    pub expansion_count: usize,

    /// Modalities to expand into.
    pub target_modalities: Vec<ModalityType>,

    /// Whether to use semantic expansion.
    pub semantic_expansion: bool,

    /// Whether to use synonym expansion.
    pub synonym_expansion: bool,

    /// Expansion weight (how much to weight expanded terms).
    pub expansion_weight: f32,
}

impl Default for QueryExpansionOptions {
    fn default() -> Self {
        Self {
            expansion_count: 5,
            target_modalities: vec![ModalityType::Text, ModalityType::Image],
            semantic_expansion: true,
            synonym_expansion: false,
            expansion_weight: 0.3,
        }
    }
}

/// Strategy for fusing multimodal retrieval results.
#[derive(Debug, Clone)]
pub enum FusionStrategy {
    /// Simple average of scores.
    Average,

    /// Weighted average with modality weights.
    WeightedAverage,

    /// Maximum score across modalities.
    Maximum,

    /// Minimum score across modalities.
    Minimum,

    /// Reciprocal rank fusion.
    ReciprocalRankFusion,

    /// Custom fusion function.
    Custom(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_modal_query_creation() {
        let query = CrossModalQuery::from_text("Find images of cats");
        assert_eq!(query.modality, ModalityType::Text);
        assert!(matches!(query.content, QueryContent::Text(_)));
        assert!(query.id.is_some());
    }

    #[test]
    fn test_cross_modal_retrieval_options_default() {
        let options = CrossModalRetrievalOptions::default();
        assert_eq!(options.top_k, 10);
        assert!(options.threshold.is_none());
        assert!(!options.include_explanations);
        assert_eq!(options.timeout_seconds, Some(30));
    }

    #[test]
    fn test_query_expansion_options_default() {
        let options = QueryExpansionOptions::default();
        assert_eq!(options.expansion_count, 5);
        assert!(options.semantic_expansion);
        assert!(!options.synonym_expansion);
        assert_eq!(options.expansion_weight, 0.3);
    }

    #[test]
    fn test_scored_multimodal_node_conversion() {
        use cheungfun_core::types::Node;

        let chunk_info = cheungfun_core::types::ChunkInfo::new(0, 12, 0);
        let base_node = Node::new("test content", Uuid::new_v4(), chunk_info);
        let multimodal_node = MultimodalNode::from_node(base_node);
        let scored = ScoredMultimodalNode {
            node: multimodal_node,
            score: 0.85,
            modality_scores: HashMap::new(),
            explanation: None,
        };

        let scored_node: ScoredNode = scored.into();
        assert_eq!(scored_node.score, 0.85);
    }
}
