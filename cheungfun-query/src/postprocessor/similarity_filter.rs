//! Similarity-based filtering postprocessor.
//!
//! This module implements LlamaIndex's SimilarityPostprocessor approach,
//! which filters nodes based on their similarity scores with the query.

use super::{NodePostprocessor, SimilarityFilterConfig};
use async_trait::async_trait;
use cheungfun_core::{Result, ScoredNode};
use tracing::debug;

/// Similarity-based node filter.
///
/// Based on LlamaIndex's SimilarityPostprocessor, this filter:
/// 1. Removes nodes below a similarity threshold
/// 2. Optionally limits the maximum number of nodes
/// 3. Preserves the original ranking order
#[derive(Debug)]
pub struct SimilarityFilter {
    /// Configuration for similarity filtering.
    config: SimilarityFilterConfig,
}

impl SimilarityFilter {
    /// Create a new similarity filter.
    pub fn new(config: SimilarityFilterConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn with_default_config() -> Self {
        Self::new(SimilarityFilterConfig::default())
    }

    /// Create with custom similarity cutoff.
    pub fn with_cutoff(similarity_cutoff: f32) -> Self {
        Self::new(SimilarityFilterConfig {
            similarity_cutoff,
            ..Default::default()
        })
    }

    /// Create with custom cutoff and max nodes.
    pub fn with_cutoff_and_limit(similarity_cutoff: f32, max_nodes: usize) -> Self {
        Self::new(SimilarityFilterConfig {
            similarity_cutoff,
            max_nodes: Some(max_nodes),
            ..Default::default()
        })
    }
}

#[async_trait]
impl NodePostprocessor for SimilarityFilter {
    async fn postprocess(&self, nodes: Vec<ScoredNode>, _query: &str) -> Result<Vec<ScoredNode>> {
        if nodes.is_empty() {
            return Ok(nodes);
        }

        debug!(
            "Filtering {} nodes with similarity cutoff: {:.3}",
            nodes.len(),
            self.config.similarity_cutoff
        );

        // Filter by similarity threshold
        let mut filtered_nodes: Vec<ScoredNode> = nodes
            .into_iter()
            .filter(|node| node.score >= self.config.similarity_cutoff)
            .collect();

        debug!("After similarity filtering: {} nodes", filtered_nodes.len());

        // Apply max nodes limit if specified
        if let Some(max_nodes) = self.config.max_nodes {
            if filtered_nodes.len() > max_nodes {
                filtered_nodes.truncate(max_nodes);
                debug!("After max nodes limit: {} nodes", filtered_nodes.len());
            }
        }

        Ok(filtered_nodes)
    }

    fn name(&self) -> &'static str {
        "SimilarityFilter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::{Node, ScoredNode};

    fn create_test_nodes() -> Vec<ScoredNode> {
        vec![
            ScoredNode {
                node: Node::new(
                    "High relevance content".to_string(),
                    Uuid::new_v4(),
                    ChunkInfo::new(Some(0), Some(22), 0),
                ),
                score: 0.9,
            },
            ScoredNode {
                node: Node::new(
                    "Medium relevance content".to_string(),
                    Uuid::new_v4(),
                    ChunkInfo::new(Some(0), Some(24), 1),
                ),
                score: 0.7,
            },
            ScoredNode {
                node: Node::new(
                    "Low relevance content".to_string(),
                    Uuid::new_v4(),
                    ChunkInfo::new(Some(0), Some(21), 2),
                ),
                score: 0.3,
            },
            ScoredNode {
                node: Node::new(
                    "Very low relevance content".to_string(),
                    Uuid::new_v4(),
                    ChunkInfo::new(Some(0), Some(26), 3),
                ),
                score: 0.1,
            },
        ]
    }

    #[tokio::test]
    async fn test_similarity_filter_with_cutoff() {
        let filter = SimilarityFilter::with_cutoff(0.5);
        let nodes = create_test_nodes();

        let filtered = filter.postprocess(nodes, "test query").await.unwrap();

        // Should keep nodes with score >= 0.5 (nodes 1 and 2)
        assert_eq!(filtered.len(), 2);
        // Note: We can't directly compare UUIDs with strings, so we check the content instead
        assert_eq!(filtered[0].node.content, "High relevance content");
        assert_eq!(filtered[1].node.content, "Medium relevance content");
    }

    #[tokio::test]
    async fn test_similarity_filter_with_limit() {
        let filter = SimilarityFilter::with_cutoff_and_limit(0.0, 2);
        let nodes = create_test_nodes();

        let filtered = filter.postprocess(nodes, "test query").await.unwrap();

        // Should keep only top 2 nodes
        assert_eq!(filtered.len(), 2);
        // Note: We can't directly compare UUIDs with strings, so we check the content instead
        assert_eq!(filtered[0].node.content, "High relevance content");
        assert_eq!(filtered[1].node.content, "Medium relevance content");
    }

    #[tokio::test]
    async fn test_similarity_filter_empty_input() {
        let filter = SimilarityFilter::with_cutoff(0.5);
        let nodes = vec![];

        let filtered = filter.postprocess(nodes, "test query").await.unwrap();

        assert_eq!(filtered.len(), 0);
    }

    #[tokio::test]
    async fn test_similarity_filter_no_matches() {
        let filter = SimilarityFilter::with_cutoff(0.95);
        let nodes = create_test_nodes();

        let filtered = filter.postprocess(nodes, "test query").await.unwrap();

        // No nodes should pass the high threshold
        assert_eq!(filtered.len(), 0);
    }
}
