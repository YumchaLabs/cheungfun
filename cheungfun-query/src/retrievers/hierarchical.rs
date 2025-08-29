//! Hierarchical retriever with auto-merging capabilities.
//!
//! This module implements a hierarchical retriever that automatically merges
//! child nodes into parent nodes when a sufficient number of children are retrieved.
//!
//! **Reference**: LlamaIndex AutoMergingRetriever
//! - File: `llama-index-core/llama_index/core/retrievers/auto_merging_retriever.py`
//! - Lines: L26-L194

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, info, instrument, warn};

use cheungfun_core::{
    traits::Retriever,
    types::{Query, ScoredNode},
    Node, Result,
};

/// Storage context interface for accessing parent nodes.
///
/// **Reference**: auto_merging_retriever.py L38
#[async_trait]
pub trait StorageContext: Send + Sync + std::fmt::Debug {
    /// Get a node by its ID.
    async fn get_node(&self, node_id: &str) -> Result<Node>;

    /// Get multiple nodes by their IDs.
    async fn get_nodes(&self, node_ids: &[String]) -> Result<Vec<Node>>;
}

/// Merge candidate information.
///
/// **Reference**: auto_merging_retriever.py L61-L77
#[derive(Debug, Clone)]
pub struct MergeCandidate {
    pub parent_node: Node,
    pub child_results: Vec<ScoredNode>,
    pub merge_ratio: f32,
}

/// A hierarchical retriever that implements auto-merging functionality.
///
/// This retriever first retrieves chunks from a base retriever, then analyzes
/// parent-child relationships to merge child nodes into parent nodes when
/// a sufficient ratio of children are retrieved.
///
/// **Reference**: LlamaIndex AutoMergingRetriever
/// - File: `llama-index-core/llama_index/core/retrievers/auto_merging_retriever.py`
/// - Lines: L26-L55
#[derive(Debug)]
pub struct HierarchicalRetriever {
    /// Base retriever for leaf nodes (Reference: L37)
    leaf_retriever: Arc<dyn Retriever>,
    /// Storage context for accessing parent nodes (Reference: L38)
    storage_context: Arc<dyn StorageContext>,
    /// Threshold for merging child nodes into parent (Reference: L39)
    merge_threshold: f32,
    /// Enable verbose logging
    verbose: bool,
}

impl HierarchicalRetriever {
    /// Create a new hierarchical retriever.
    ///
    /// **Reference**: auto_merging_retriever.py L35-L54
    pub fn new(
        leaf_retriever: Arc<dyn Retriever>,
        storage_context: Arc<dyn StorageContext>,
        merge_threshold: f32,
    ) -> Self {
        Self {
            leaf_retriever,
            storage_context,
            merge_threshold,
            verbose: false,
        }
    }

    /// Create a builder for configuring the retriever.
    pub fn builder() -> HierarchicalRetrieverBuilder {
        HierarchicalRetrieverBuilder::new()
    }

    /// Analyze merge candidates from retrieved nodes.
    ///
    /// **Reference**: auto_merging_retriever.py L56-L78
    #[instrument(skip(self))]
    async fn analyze_merge_candidates(
        &self,
        nodes: &[ScoredNode],
    ) -> Result<HashMap<String, MergeCandidate>> {
        let mut parent_child_map: HashMap<String, Vec<&ScoredNode>> = HashMap::new();
        let mut parent_nodes: HashMap<String, Node> = HashMap::new();

        // Group nodes by their parent (Reference: L61-L77)
        for node in nodes {
            if let Some(parent_id_value) = node.node.metadata.get("parent_id") {
                if let Some(parent_id) = parent_id_value.as_str() {
                    parent_child_map
                        .entry(parent_id.to_string())
                        .or_default()
                        .push(node);

                    // Fetch parent node if not already cached (Reference: L68-L74)
                    if !parent_nodes.contains_key(parent_id) {
                        match self.storage_context.get_node(parent_id).await {
                            Ok(parent_node) => {
                                parent_nodes.insert(parent_id.to_string(), parent_node);
                            }
                            Err(e) => {
                                warn!("Failed to fetch parent node {}: {}", parent_id, e);
                                continue;
                            }
                        }
                    }
                }
            }
        }

        // Compute merge candidates (Reference: L79-L115)
        let mut candidates = HashMap::new();
        for (parent_id, child_results) in parent_child_map {
            if let Some(parent_node) = parent_nodes.get(&parent_id) {
                let total_children = self.get_total_children_count(parent_node).await?;
                let retrieved_children = child_results.len();
                let merge_ratio = retrieved_children as f32 / total_children as f32;

                // Check if merge threshold is met (Reference: L90)
                if merge_ratio > self.merge_threshold {
                    if self.verbose {
                        info!(
                            "Merge candidate: parent {} has {}/{} children (ratio: {:.2})",
                            parent_id, retrieved_children, total_children, merge_ratio
                        );
                    }

                    candidates.insert(
                        parent_id.clone(),
                        MergeCandidate {
                            parent_node: parent_node.clone(),
                            child_results: child_results.into_iter().cloned().collect(),
                            merge_ratio,
                        },
                    );
                }
            }
        }

        Ok(candidates)
    }

    /// Get the total number of children for a parent node.
    ///
    /// **Reference**: auto_merging_retriever.py L84-L85
    async fn get_total_children_count(&self, parent_node: &Node) -> Result<usize> {
        // Try to get from metadata first
        if let Some(child_count_value) = parent_node.metadata.get("child_count") {
            if let Some(child_count_str) = child_count_value.as_str() {
                if let Ok(count) = child_count_str.parse::<usize>() {
                    return Ok(count);
                }
            }
        }

        // Fallback: count from child_ids metadata
        if let Some(child_ids_value) = parent_node.metadata.get("child_ids") {
            if let Some(child_ids_str) = child_ids_value.as_str() {
                let child_ids: Vec<&str> = child_ids_str.split(',').collect();
                return Ok(child_ids.len());
            }
        }

        // Default to 1 if no information available
        Ok(1)
    }

    /// Perform merging of child nodes into parent nodes.
    ///
    /// **Reference**: auto_merging_retriever.py L166-L174
    #[instrument(skip(self))]
    async fn perform_merging(
        &self,
        mut results: Vec<ScoredNode>,
        candidates: HashMap<String, MergeCandidate>,
    ) -> Result<Vec<ScoredNode>> {
        let mut nodes_to_remove = HashSet::new();
        let mut nodes_to_add = Vec::new();

        for (parent_id, candidate) in candidates {
            // Mark child nodes for removal (Reference: L91-L93)
            for child_result in &candidate.child_results {
                nodes_to_remove.insert(child_result.node.id.clone());
            }

            // Calculate average score for parent node (Reference: L110-L112)
            let avg_score = candidate.child_results.iter().map(|r| r.score).sum::<f32>()
                / candidate.child_results.len() as f32;

            if self.verbose {
                info!(
                    "Merging {} child nodes into parent {} with score {:.3}",
                    candidate.child_results.len(),
                    parent_id,
                    avg_score
                );
            }

            // Add parent node with averaged score (Reference: L113-L115)
            nodes_to_add.push(ScoredNode {
                node: candidate.parent_node,
                score: avg_score,
            });
        }

        // Remove child nodes and add parent nodes
        results.retain(|r| !nodes_to_remove.contains(&r.node.id));
        results.extend(nodes_to_add);

        Ok(results)
    }

    /// Try merging nodes with their parents.
    ///
    /// **Reference**: auto_merging_retriever.py L166-L174
    #[instrument(skip(self))]
    async fn try_merging(&self, nodes: Vec<ScoredNode>) -> Result<(Vec<ScoredNode>, bool)> {
        // Analyze merge candidates
        let candidates = self.analyze_merge_candidates(&nodes).await?;

        if candidates.is_empty() {
            return Ok((nodes, false));
        }

        // Perform merging
        let merged_nodes = self.perform_merging(nodes, candidates).await?;
        Ok((merged_nodes, true))
    }
}

#[async_trait]
impl Retriever for HierarchicalRetriever {
    /// Retrieve nodes with hierarchical merging.
    ///
    /// **Reference**: auto_merging_retriever.py L176-L194
    #[instrument(skip(self), fields(retriever = "HierarchicalRetriever"))]
    async fn retrieve(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        info!("Starting hierarchical retrieval for query: {}", query.text);

        // 1. Initial retrieval from leaf nodes (Reference: L183)
        let initial_nodes = self.leaf_retriever.retrieve(query).await?;
        debug!("Retrieved {} initial nodes", initial_nodes.len());

        // 2. Iteratively try merging until no more changes (Reference: L185-L189)
        let mut current_nodes = initial_nodes;
        let mut iteration = 0;

        loop {
            let (merged_nodes, is_changed) = self.try_merging(current_nodes).await?;
            current_nodes = merged_nodes;
            iteration += 1;

            if !is_changed {
                break;
            }

            debug!(
                "Merge iteration {}: {} nodes remaining",
                iteration,
                current_nodes.len()
            );
        }

        // 3. Sort by relevance score (Reference: L192)
        current_nodes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        info!(
            "Hierarchical retrieval completed: {} final nodes after {} iterations",
            current_nodes.len(),
            iteration
        );

        Ok(current_nodes)
    }
}

/// Builder for creating hierarchical retrievers.
///
/// **Reference**: LlamaIndex builder patterns
#[derive(Debug, Default)]
pub struct HierarchicalRetrieverBuilder {
    leaf_retriever: Option<Arc<dyn Retriever>>,
    storage_context: Option<Arc<dyn StorageContext>>,
    merge_threshold: f32,
    verbose: bool,
}

impl HierarchicalRetrieverBuilder {
    pub fn new() -> Self {
        Self {
            merge_threshold: 0.5, // Default from auto_merging_retriever.py L39
            verbose: false,
            ..Default::default()
        }
    }

    pub fn leaf_retriever(mut self, retriever: Arc<dyn Retriever>) -> Self {
        self.leaf_retriever = Some(retriever);
        self
    }

    pub fn storage_context(mut self, context: Arc<dyn StorageContext>) -> Self {
        self.storage_context = Some(context);
        self
    }

    pub fn merge_threshold(mut self, threshold: f32) -> Self {
        self.merge_threshold = threshold;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn build(self) -> Result<HierarchicalRetriever> {
        let leaf_retriever =
            self.leaf_retriever
                .ok_or_else(|| cheungfun_core::CheungfunError::Configuration {
                    message: "leaf_retriever is required".to_string(),
                })?;

        let storage_context =
            self.storage_context
                .ok_or_else(|| cheungfun_core::CheungfunError::Configuration {
                    message: "storage_context is required".to_string(),
                })?;

        Ok(HierarchicalRetriever {
            leaf_retriever,
            storage_context,
            merge_threshold: self.merge_threshold,
            verbose: self.verbose,
        })
    }
}
