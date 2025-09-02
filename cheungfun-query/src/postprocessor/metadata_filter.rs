//! Metadata-based filtering postprocessor.
//!
//! This module implements LlamaIndex's metadata filtering approach,
//! which filters nodes based on their metadata properties.

use super::{MetadataFilterConfig, NodePostprocessor};
use async_trait::async_trait;
use cheungfun_core::{Result, ScoredNode};
use std::collections::HashMap;
use tracing::debug;

/// Metadata-based node filter.
///
/// Based on LlamaIndex's metadata filtering, this filter:
/// 1. Filters nodes based on required metadata key-value pairs
/// 2. Excludes nodes with specific metadata values
/// 3. Supports both "require all" and "require any" matching modes
/// 4. Handles string-based metadata comparisons
#[derive(Debug)]
pub struct MetadataFilter {
    /// Configuration for metadata filtering.
    config: MetadataFilterConfig,
}

impl MetadataFilter {
    /// Create a new metadata filter.
    pub fn new(config: MetadataFilterConfig) -> Self {
        Self { config }
    }

    /// Create with required metadata only.
    pub fn with_required_metadata(metadata: HashMap<String, String>) -> Self {
        Self::new(MetadataFilterConfig {
            required_metadata: metadata,
            ..Default::default()
        })
    }

    /// Create with excluded metadata only.
    pub fn with_excluded_metadata(metadata: HashMap<String, String>) -> Self {
        Self::new(MetadataFilterConfig {
            excluded_metadata: metadata,
            ..Default::default()
        })
    }

    /// Create with both required and excluded metadata.
    pub fn with_metadata(
        required: HashMap<String, String>,
        excluded: HashMap<String, String>,
    ) -> Self {
        Self::new(MetadataFilterConfig {
            required_metadata: required,
            excluded_metadata: excluded,
            ..Default::default()
        })
    }

    /// Create with require_all mode.
    pub fn require_all(mut self) -> Self {
        self.config.require_all = true;
        self
    }

    /// Create with require_any mode.
    pub fn require_any(mut self) -> Self {
        self.config.require_all = false;
        self
    }

    /// Check if node metadata matches required criteria.
    fn matches_required_metadata(&self, node_metadata: &HashMap<String, String>) -> bool {
        if self.config.required_metadata.is_empty() {
            return true; // No required metadata means all nodes pass
        }

        let matches: Vec<bool> = self
            .config
            .required_metadata
            .iter()
            .map(|(key, value)| {
                node_metadata
                    .get(key)
                    .map(|node_value| node_value == value)
                    .unwrap_or(false)
            })
            .collect();

        if self.config.require_all {
            matches.iter().all(|&m| m)
        } else {
            matches.iter().any(|&m| m)
        }
    }

    /// Check if node metadata contains excluded values.
    fn contains_excluded_metadata(&self, node_metadata: &HashMap<String, String>) -> bool {
        if self.config.excluded_metadata.is_empty() {
            return false; // No excluded metadata means no nodes are excluded
        }

        self.config.excluded_metadata.iter().any(|(key, value)| {
            node_metadata
                .get(key)
                .map(|node_value| node_value == value)
                .unwrap_or(false)
        })
    }

    /// Filter a single node based on metadata criteria.
    fn should_keep_node(&self, node: &ScoredNode) -> bool {
        // Convert node metadata to string-based HashMap for comparison
        let node_metadata: HashMap<String, String> = node
            .node
            .metadata
            .iter()
            .filter_map(|(k, v)| {
                // Convert serde_json::Value to String for comparison
                match v {
                    serde_json::Value::String(s) => Some((k.clone(), s.clone())),
                    serde_json::Value::Number(n) => Some((k.clone(), n.to_string())),
                    serde_json::Value::Bool(b) => Some((k.clone(), b.to_string())),
                    _ => None, // Skip complex types for now
                }
            })
            .collect();

        // Check required metadata
        if !self.matches_required_metadata(&node_metadata) {
            debug!("Node filtered out: does not match required metadata");
            return false;
        }

        // Check excluded metadata
        if self.contains_excluded_metadata(&node_metadata) {
            debug!("Node filtered out: contains excluded metadata");
            return false;
        }

        true
    }
}

#[async_trait]
impl NodePostprocessor for MetadataFilter {
    async fn postprocess(&self, nodes: Vec<ScoredNode>, _query: &str) -> Result<Vec<ScoredNode>> {
        if nodes.is_empty() {
            return Ok(nodes);
        }

        debug!(
            "Filtering {} nodes with {} required metadata and {} excluded metadata (require_all: {})",
            nodes.len(),
            self.config.required_metadata.len(),
            self.config.excluded_metadata.len(),
            self.config.require_all
        );

        let filtered_nodes: Vec<ScoredNode> = nodes
            .into_iter()
            .filter(|node| self.should_keep_node(node))
            .collect();

        debug!(
            "Metadata filtering completed: {} nodes remaining",
            filtered_nodes.len()
        );

        Ok(filtered_nodes)
    }

    fn name(&self) -> &'static str {
        "MetadataFilter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::{ChunkInfo, Node};
    use serde_json::json;
    use uuid::Uuid;

    fn create_test_node_with_metadata(
        content: &str,
        metadata: HashMap<String, serde_json::Value>,
    ) -> ScoredNode {
        let chunk_info = ChunkInfo {
            start_char_idx: Some(0),
            end_char_idx: Some(content.len()),
            chunk_index: 0,
        };
        let mut node = Node::new(content.to_string(), Uuid::new_v4(), chunk_info);
        node.metadata = metadata;
        ScoredNode { node, score: 0.8 }
    }

    #[tokio::test]
    async fn test_required_metadata_filter() {
        let mut required = HashMap::new();
        required.insert("category".to_string(), "science".to_string());

        let filter = MetadataFilter::with_required_metadata(required);

        let nodes = vec![
            create_test_node_with_metadata(
                "Science content",
                [("category".to_string(), json!("science"))]
                    .into_iter()
                    .collect(),
            ),
            create_test_node_with_metadata(
                "Sports content",
                [("category".to_string(), json!("sports"))]
                    .into_iter()
                    .collect(),
            ),
            create_test_node_with_metadata("No category content", HashMap::new()),
        ];

        let filtered = filter.postprocess(nodes, "test query").await.unwrap();

        // Should keep only the science content
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].node.content, "Science content");
    }

    #[tokio::test]
    async fn test_excluded_metadata_filter() {
        let mut excluded = HashMap::new();
        excluded.insert("status".to_string(), "draft".to_string());

        let filter = MetadataFilter::with_excluded_metadata(excluded);

        let nodes = vec![
            create_test_node_with_metadata(
                "Published content",
                [("status".to_string(), json!("published"))]
                    .into_iter()
                    .collect(),
            ),
            create_test_node_with_metadata(
                "Draft content",
                [("status".to_string(), json!("draft"))]
                    .into_iter()
                    .collect(),
            ),
        ];

        let filtered = filter.postprocess(nodes, "test query").await.unwrap();

        // Should exclude the draft content
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].node.content, "Published content");
    }

    #[tokio::test]
    async fn test_require_all_vs_require_any() {
        let mut required = HashMap::new();
        required.insert("category".to_string(), "science".to_string());
        required.insert("language".to_string(), "english".to_string());

        // Test require_all (default)
        let filter_all = MetadataFilter::with_required_metadata(required.clone()).require_all();

        // Test require_any
        let filter_any = MetadataFilter::with_required_metadata(required).require_any();

        let nodes = vec![
            create_test_node_with_metadata(
                "Full match",
                [
                    ("category".to_string(), json!("science")),
                    ("language".to_string(), json!("english")),
                ]
                .into_iter()
                .collect(),
            ),
            create_test_node_with_metadata(
                "Partial match",
                [("category".to_string(), json!("science"))]
                    .into_iter()
                    .collect(),
            ),
        ];

        let filtered_all = filter_all
            .postprocess(nodes.clone(), "test query")
            .await
            .unwrap();
        let filtered_any = filter_any.postprocess(nodes, "test query").await.unwrap();

        // require_all should keep only full match
        assert_eq!(filtered_all.len(), 1);
        assert_eq!(filtered_all[0].node.content, "Full match");

        // require_any should keep both
        assert_eq!(filtered_any.len(), 2);
    }
}
