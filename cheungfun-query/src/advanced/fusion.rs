// Fusion Algorithms Implementation

use super::FusionMethod;
use cheungfun_core::ScoredNode;
use std::collections::HashMap;
use tracing::{debug, info};

/// Reciprocal Rank Fusion (RRF) implementation.
///
/// RRF is an effective result fusion algorithm, especially suitable for hybrid search scenarios.
/// Paper: "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods"
/// Formula: `RRF_score(d)` = Σ (1 / (k + rank(d))), where k is a smoothing parameter, typically set to 60.
#[derive(Debug, Clone)]
pub struct ReciprocalRankFusion {
    /// RRF parameter k, used to control the influence of outlier rankings.
    /// The original paper suggests k=60 for optimal results.
    pub k: f32,
}

impl ReciprocalRankFusion {
    /// Creates a new RRF fuser.
    #[must_use]
    pub fn new(k: f32) -> Self {
        Self { k }
    }

    /// Creates an RRF fuser with default parameters.
    #[must_use]
    pub fn with_default_k() -> Self {
        Self::new(60.0)
    }

    /// Fuses multiple search result lists.
    ///
    /// # Arguments
    /// * `results` - Multiple search result lists, each should be sorted in descending order of relevance.
    ///
    /// # Returns
    /// * A fused result list, sorted in descending order by RRF score.
    pub fn fuse_results(&self, results: Vec<Vec<ScoredNode>>) -> Vec<ScoredNode> {
        debug!(
            "Fusing {} result lists using RRF with k={}",
            results.len(),
            self.k
        );

        let mut fused_scores: HashMap<String, f32> = HashMap::new();
        let mut node_map: HashMap<String, ScoredNode> = HashMap::new();
        let mut result_list_sizes = Vec::new();

        // Calculate RRF scores for each result list
        for result_list in results {
            result_list_sizes.push(result_list.len());

            for (rank, node) in result_list.into_iter().enumerate() {
                let node_id = node.node.id.to_string();
                let rrf_score = 1.0 / (rank as f32 + self.k);

                // Accumulate RRF scores
                *fused_scores.entry(node_id.clone()).or_insert(0.0) += rrf_score;

                // Store node information (if it already exists, keep the version with the higher score)
                node_map
                    .entry(node_id.clone())
                    .and_modify(|existing| {
                        if node.score > existing.score {
                            *existing = node.clone();
                        }
                    })
                    .or_insert(node);
            }
        }

        // Sort by fused score and build the final result list
        let mut final_results: Vec<_> = fused_scores
            .into_iter()
            .filter_map(|(node_id, rrf_score)| {
                node_map.remove(&node_id).map(|mut node| {
                    // Use the RRF score as the final score
                    node.score = rrf_score;
                    node
                })
            })
            .collect();

        final_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        info!(
            "RRF fusion completed: {} unique results from {} lists (sizes: {:?})",
            final_results.len(),
            result_list_sizes.len(),
            result_list_sizes
        );

        final_results
    }

    /// Calculates the RRF score for a single document at a specific rank.
    #[must_use]
    pub fn calculate_rrf_score(&self, rank: usize) -> f32 {
        1.0 / (rank as f32 + self.k)
    }
}

impl Default for ReciprocalRankFusion {
    fn default() -> Self {
        Self::new(60.0)
    }
}

/// Weighted Average Fusion algorithm.
#[derive(Debug, Clone)]
pub struct WeightedAverageFusion {
    /// Weights for each result list.
    pub weights: Vec<f32>,
    /// Whether to normalize the scores.
    pub normalize_scores: bool,
}

impl WeightedAverageFusion {
    /// Creates a new Weighted Average fuser.
    #[must_use]
    pub fn new(weights: Vec<f32>) -> Self {
        Self {
            weights,
            normalize_scores: true,
        }
    }

    /// Sets whether to normalize scores.
    #[must_use]
    pub fn with_normalize_scores(mut self, normalize: bool) -> Self {
        self.normalize_scores = normalize;
        self
    }

    /// Fuses multiple search result lists.
    pub fn fuse_results(&self, results: Vec<Vec<ScoredNode>>) -> Vec<ScoredNode> {
        debug!(
            "Fusing {} result lists using weighted average",
            results.len()
        );

        assert!(
            results.len() == self.weights.len(),
            "Number of result lists ({}) must match number of weights ({})",
            results.len(),
            self.weights.len()
        );

        let mut fused_scores: HashMap<String, f32> = HashMap::new();
        let mut node_map: HashMap<String, ScoredNode> = HashMap::new();
        let mut weight_sums: HashMap<String, f32> = HashMap::new();

        // Normalize scores (if required)
        let normalized_results = if self.normalize_scores {
            self.normalize_result_scores(results)
        } else {
            results
        };

        // Calculate weighted average scores
        for (list_idx, result_list) in normalized_results.into_iter().enumerate() {
            let weight = self.weights[list_idx];

            for node in result_list {
                let node_id = node.node.id.to_string();
                let weighted_score = node.score * weight;

                *fused_scores.entry(node_id.clone()).or_insert(0.0) += weighted_score;
                *weight_sums.entry(node_id.clone()).or_insert(0.0) += weight;

                node_map.entry(node_id).or_insert(node);
            }
        }

        // Calculate the final weighted average score
        let mut final_results: Vec<_> = fused_scores
            .into_iter()
            .filter_map(|(node_id, total_weighted_score)| {
                let total_weight = weight_sums.get(&node_id)?;
                let avg_score = total_weighted_score / total_weight;

                node_map.remove(&node_id).map(|mut node| {
                    node.score = avg_score;
                    node
                })
            })
            .collect();

        final_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        info!(
            "Weighted average fusion completed: {} results",
            final_results.len()
        );
        final_results
    }

    /// Normalizes the result scores.
    fn normalize_result_scores(&self, results: Vec<Vec<ScoredNode>>) -> Vec<Vec<ScoredNode>> {
        results
            .into_iter()
            .map(|mut result_list| {
                if result_list.is_empty() {
                    return result_list;
                }

                // Find the maximum and minimum scores
                let max_score = result_list
                    .iter()
                    .map(|n| n.score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let min_score = result_list
                    .iter()
                    .map(|n| n.score)
                    .fold(f32::INFINITY, f32::min);

                // Avoid division by zero
                let score_range = max_score - min_score;
                if score_range > 0.0 {
                    for node in &mut result_list {
                        node.score = (node.score - min_score) / score_range;
                    }
                }

                result_list
            })
            .collect()
    }
}

/// Linear Combination Fusion algorithm.
#[derive(Debug, Clone)]
pub struct LinearCombinationFusion {
    /// Linear combination coefficients.
    pub coefficients: Vec<f32>,
    /// Whether to normalize the scores.
    pub normalize_scores: bool,
}

impl LinearCombinationFusion {
    /// Creates a new Linear Combination fuser.
    #[must_use]
    pub fn new(coefficients: Vec<f32>) -> Self {
        Self {
            coefficients,
            normalize_scores: true,
        }
    }

    /// Fuses multiple search result lists.
    pub fn fuse_results(&self, results: Vec<Vec<ScoredNode>>) -> Vec<ScoredNode> {
        debug!(
            "Fusing {} result lists using linear combination",
            results.len()
        );

        assert!(
            results.len() == self.coefficients.len(),
            "Number of result lists ({}) must match number of coefficients ({})",
            results.len(),
            self.coefficients.len()
        );

        let mut fused_scores: HashMap<String, f32> = HashMap::new();
        let mut node_map: HashMap<String, ScoredNode> = HashMap::new();

        // Normalize scores (if required)
        let normalized_results = if self.normalize_scores {
            self.normalize_result_scores(results)
        } else {
            results
        };

        // Calculate linear combination scores
        for (list_idx, result_list) in normalized_results.into_iter().enumerate() {
            let coefficient = self.coefficients[list_idx];

            for node in result_list {
                let node_id = node.node.id.to_string();
                let combined_score = node.score * coefficient;

                *fused_scores.entry(node_id.clone()).or_insert(0.0) += combined_score;
                node_map.entry(node_id).or_insert(node);
            }
        }

        // Build the final result list
        let mut final_results: Vec<_> = fused_scores
            .into_iter()
            .filter_map(|(node_id, combined_score)| {
                node_map.remove(&node_id).map(|mut node| {
                    node.score = combined_score;
                    node
                })
            })
            .collect();

        final_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        info!(
            "Linear combination fusion completed: {} results",
            final_results.len()
        );
        final_results
    }

    /// Normalizes the result scores (same implementation as `WeightedAverageFusion`).
    fn normalize_result_scores(&self, results: Vec<Vec<ScoredNode>>) -> Vec<Vec<ScoredNode>> {
        results
            .into_iter()
            .map(|mut result_list| {
                if result_list.is_empty() {
                    return result_list;
                }

                let max_score = result_list
                    .iter()
                    .map(|n| n.score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let min_score = result_list
                    .iter()
                    .map(|n| n.score)
                    .fold(f32::INFINITY, f32::min);

                let score_range = max_score - min_score;
                if score_range > 0.0 {
                    for node in &mut result_list {
                        node.score = (node.score - min_score) / score_range;
                    }
                }

                result_list
            })
            .collect()
    }
}

/// Distribution-Based Score Fusion implementation.
///
/// This fusion method scales scores based on the mean and standard deviation
/// of each result set, providing better normalization than simple relative scoring.
/// Based on: https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18
#[derive(Debug, Clone)]
pub struct DistributionBasedFusion {
    /// Weights for each result set
    pub weights: Vec<f32>,
}

impl DistributionBasedFusion {
    /// Create a new distribution-based fusion with equal weights.
    #[must_use]
    pub fn new(num_sources: usize) -> Self {
        let weight = 1.0 / num_sources as f32;
        Self {
            weights: vec![weight; num_sources],
        }
    }

    /// Create with custom weights.
    #[must_use]
    pub fn with_weights(weights: Vec<f32>) -> Self {
        Self { weights }
    }

    /// Calculate mean and standard deviation of scores.
    fn calculate_stats(scores: &[f32]) -> (f32, f32) {
        if scores.is_empty() {
            return (0.0, 1.0);
        }

        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores
            .iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f32>()
            / scores.len() as f32;
        let std_dev = variance.sqrt().max(1e-6); // Avoid division by zero

        (mean, std_dev)
    }

    /// Normalize scores using z-score normalization.
    fn normalize_scores(scores: &[f32], mean: f32, std_dev: f32) -> Vec<f32> {
        scores
            .iter()
            .map(|score| (score - mean) / std_dev)
            .collect()
    }

    /// Fuse multiple result sets using distribution-based scoring.
    pub fn fuse_results(&self, results: Vec<Vec<ScoredNode>>) -> Vec<ScoredNode> {
        debug!(
            "Fusing {} result lists using Distribution-Based Score Fusion",
            results.len()
        );

        if results.is_empty() {
            return Vec::new();
        }

        let mut fused_scores: HashMap<String, f32> = HashMap::new();
        let mut node_map: HashMap<String, ScoredNode> = HashMap::new();

        // Process each result set
        for (i, result_set) in results.iter().enumerate() {
            if result_set.is_empty() {
                continue;
            }

            let weight = self
                .weights
                .get(i)
                .copied()
                .unwrap_or(1.0 / results.len() as f32);
            let scores: Vec<f32> = result_set.iter().map(|node| node.score).collect();
            let (mean, std_dev) = Self::calculate_stats(&scores);
            let normalized_scores = Self::normalize_scores(&scores, mean, std_dev);

            for (node, &normalized_score) in result_set.iter().zip(normalized_scores.iter()) {
                let node_id = node.node.id.to_string();
                node_map.insert(node_id.clone(), node.clone());

                // Apply sigmoid to convert z-scores to 0-1 range
                let sigmoid_score = 1.0 / (1.0 + (-normalized_score).exp());
                let weighted_score = sigmoid_score * weight;

                *fused_scores.entry(node_id).or_insert(0.0) += weighted_score;
            }
        }

        // Create final result list
        let mut final_results: Vec<ScoredNode> = fused_scores
            .into_iter()
            .filter_map(|(node_id, score)| {
                node_map.get(&node_id).map(|node| ScoredNode {
                    node: node.node.clone(),
                    score,
                })
            })
            .collect();

        // Sort by fused score in descending order
        final_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        info!(
            "Distribution-based fusion completed: {} unique nodes",
            final_results.len()
        );

        final_results
    }
}

/// Fusion Algorithm Factory.
pub struct FusionAlgorithmFactory;

impl FusionAlgorithmFactory {
    /// Creates a corresponding fusion algorithm based on the fusion method.
    #[must_use]
    pub fn create_fusion_algorithm(method: &FusionMethod) -> Box<dyn FusionAlgorithm> {
        match method {
            FusionMethod::ReciprocalRankFusion { k } => Box::new(ReciprocalRankFusion::new(*k)),
            FusionMethod::WeightedAverage => {
                // Default to equal weights
                Box::new(WeightedAverageFusion::new(vec![0.5, 0.5]))
            }
            FusionMethod::LinearCombination => {
                // Default to equal coefficients
                Box::new(LinearCombinationFusion::new(vec![0.5, 0.5]))
            }
            FusionMethod::DistributionBasedScore => {
                // Default to equal weights for 2 sources
                Box::new(DistributionBasedFusion::new(2))
            }
            FusionMethod::Custom(name) => {
                // TODO: Implement registration and creation mechanism for custom fusion algorithms
                panic!("Custom fusion algorithm '{name}' not implemented");
            }
        }
    }
}

/// Generic interface for fusion algorithms.
pub trait FusionAlgorithm: Send + Sync + std::fmt::Debug {
    /// Fuses multiple search result lists.
    fn fuse_results(&self, results: Vec<Vec<ScoredNode>>) -> Vec<ScoredNode>;

    /// Gets the name of the algorithm.
    fn name(&self) -> &'static str;
}

impl FusionAlgorithm for ReciprocalRankFusion {
    fn fuse_results(&self, results: Vec<Vec<ScoredNode>>) -> Vec<ScoredNode> {
        self.fuse_results(results)
    }

    fn name(&self) -> &'static str {
        "ReciprocalRankFusion"
    }
}

impl FusionAlgorithm for WeightedAverageFusion {
    fn fuse_results(&self, results: Vec<Vec<ScoredNode>>) -> Vec<ScoredNode> {
        self.fuse_results(results)
    }

    fn name(&self) -> &'static str {
        "WeightedAverageFusion"
    }
}

impl FusionAlgorithm for LinearCombinationFusion {
    fn fuse_results(&self, results: Vec<Vec<ScoredNode>>) -> Vec<ScoredNode> {
        self.fuse_results(results)
    }

    fn name(&self) -> &'static str {
        "LinearCombinationFusion"
    }
}

impl FusionAlgorithm for DistributionBasedFusion {
    fn fuse_results(&self, results: Vec<Vec<ScoredNode>>) -> Vec<ScoredNode> {
        self.fuse_results(results)
    }

    fn name(&self) -> &'static str {
        "DistributionBasedFusion"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::{ChunkInfo, Node};
    use uuid::Uuid;

    fn create_test_node(id: &str, score: f32, content: &str) -> ScoredNode {
        ScoredNode {
            node: Node::new(
                content.to_string(),
                Uuid::new_v4(),
                ChunkInfo {
                    start_char_idx: Some(0),
                    end_char_idx: Some(content.len()),
                    chunk_index: 0,
                },
            ),
            score,
        }
    }

    #[test]
    fn test_rrf_fusion() {
        let rrf = ReciprocalRankFusion::new(60.0);

        let list1 = vec![
            create_test_node("00000000-0000-0000-0000-000000000001", 0.9, "doc1"),
            create_test_node("00000000-0000-0000-0000-000000000002", 0.8, "doc2"),
        ];

        let list2 = vec![
            create_test_node("00000000-0000-0000-0000-000000000002", 0.7, "doc2"),
            create_test_node("00000000-0000-0000-0000-000000000003", 0.6, "doc3"),
        ];

        let results = rrf.fuse_results(vec![list1, list2]);

        assert_eq!(results.len(), 3);
        // doc2 should be ranked first because it appears in both lists.
        assert_eq!(results[0].node.content, "doc2");
    }
}
