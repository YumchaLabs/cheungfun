//! Multi-Vector Retrieval Example
//!
//! This example demonstrates multi-vector retrieval techniques that use
//! multiple vector representations of the same content to improve retrieval
//! accuracy. This approach creates different embeddings for different aspects
//! of the content (summary, keywords, full text, etc.) and retrieves from
//! multiple vector spaces simultaneously.
//!
//! # Features Demonstrated
//!
//! - **Multiple Vector Representations**: Create different embeddings for same content
//! - **Multi-Space Retrieval**: Search across multiple vector spaces
//! - **Vector Space Fusion**: Combine results from different vector spaces
//! - **Aspect-based Embeddings**: Generate embeddings for different content aspects
//! - **Weighted Vector Combination**: Weight different vector spaces by importance
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin multi_vector_retrieval --features fastembed -- \
//!     --query "machine learning in medical diagnosis" \
//!     --vector-spaces "summary,keywords,full_text" \
//!     --fusion-method "weighted_average" \
//!     --space-weights "0.4,0.3,0.3"
//! ```

use cheungfun::prelude::*;
use cheungfun_query::advanced::fusion::DistributionBasedFusion;
use clap::Parser;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "multi-vector-retrieval")]
#[command(about = "Demonstrates multi-vector retrieval with multiple embeddings")]
struct Args {
    /// Query to search for
    #[arg(long, default_value = "machine learning in medical diagnosis")]
    query: String,

    /// Vector spaces to use (comma-separated)
    #[arg(long, default_value = "summary,keywords,full_text")]
    vector_spaces: String,

    /// Fusion method: weighted_average, max_score, rank_fusion
    #[arg(long, default_value = "weighted_average")]
    fusion_method: String,

    /// Weights for each vector space (comma-separated)
    #[arg(long, default_value = "0.4,0.3,0.3")]
    space_weights: String,

    /// Number of results per vector space
    #[arg(long, default_value = "8")]
    top_k: usize,

    /// Enable verbose output
    #[arg(long)]
    verbose: bool,
}

/// Vector space type for multi-vector retrieval.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorSpaceType {
    /// Summary-based embeddings
    Summary,
    /// Keyword-based embeddings
    Keywords,
    /// Full text embeddings
    FullText,
    /// Title/heading embeddings
    Title,
    /// Metadata embeddings
    Metadata,
}

impl VectorSpaceType {
    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "summary" => Some(Self::Summary),
            "keywords" => Some(Self::Keywords),
            "full_text" | "fulltext" => Some(Self::FullText),
            "title" => Some(Self::Title),
            "metadata" => Some(Self::Metadata),
            _ => None,
        }
    }

    /// Get display name.
    pub fn display_name(&self) -> &str {
        match self {
            Self::Summary => "Summary",
            Self::Keywords => "Keywords",
            Self::FullText => "Full Text",
            Self::Title => "Title",
            Self::Metadata => "Metadata",
        }
    }
}

/// Multi-vector retrieval result from a single vector space.
#[derive(Debug, Clone)]
pub struct VectorSpaceResult {
    /// Vector space type
    pub space_type: VectorSpaceType,
    /// Retrieved nodes
    pub nodes: Vec<ScoredNode>,
    /// Retrieval time
    pub retrieval_time: Duration,
    /// Average score
    pub avg_score: f32,
}

/// Multi-vector retrieval engine.
#[derive(Debug)]
pub struct MultiVectorRetriever {
    /// Vector spaces to use
    vector_spaces: Vec<VectorSpaceType>,
    /// Weights for each vector space
    space_weights: Vec<f32>,
    /// Fusion method
    fusion_method: String,
}

impl MultiVectorRetriever {
    /// Create a new multi-vector retriever.
    pub fn new(
        vector_spaces: Vec<VectorSpaceType>,
        space_weights: Vec<f32>,
        fusion_method: String,
    ) -> Self {
        Self {
            vector_spaces,
            space_weights,
            fusion_method,
        }
    }

    /// Perform multi-vector retrieval.
    pub fn retrieve_multi_vector(&self, query: &str, top_k: usize) -> Vec<VectorSpaceResult> {
        let mut results = Vec::new();

        info!(
            "Starting multi-vector retrieval across {} spaces",
            self.vector_spaces.len()
        );

        for (i, space_type) in self.vector_spaces.iter().enumerate() {
            let start_time = Instant::now();

            info!("Retrieving from {} vector space", space_type.display_name());

            // Generate space-specific query representation
            let space_query = self.adapt_query_for_space(query, space_type);

            // Perform retrieval in this vector space
            let nodes = self.retrieve_from_space(&space_query, space_type, top_k);

            let retrieval_time = start_time.elapsed();
            let avg_score = if nodes.is_empty() {
                0.0
            } else {
                nodes.iter().map(|n| n.score).sum::<f32>() / nodes.len() as f32
            };

            results.push(VectorSpaceResult {
                space_type: space_type.clone(),
                nodes,
                retrieval_time,
                avg_score,
            });

            info!(
                "Retrieved {} nodes from {} space (avg score: {:.3})",
                results[i].nodes.len(),
                space_type.display_name(),
                avg_score
            );
        }

        results
    }

    /// Adapt query for specific vector space.
    fn adapt_query_for_space(&self, query: &str, space_type: &VectorSpaceType) -> String {
        match space_type {
            VectorSpaceType::Summary => {
                format!("Summary: {}", query)
            }
            VectorSpaceType::Keywords => {
                // Extract key terms for keyword space
                let keywords: Vec<&str> = query
                    .split_whitespace()
                    .filter(|w| w.len() > 3 && !self.is_stop_word(w))
                    .collect();
                keywords.join(" ")
            }
            VectorSpaceType::FullText => query.to_string(),
            VectorSpaceType::Title => {
                format!("Title: {}", query)
            }
            VectorSpaceType::Metadata => {
                format!("Topic: {}", query)
            }
        }
    }

    /// Check if word is a stop word.
    fn is_stop_word(&self, word: &str) -> bool {
        matches!(
            word.to_lowercase().as_str(),
            "the"
                | "a"
                | "an"
                | "and"
                | "or"
                | "but"
                | "in"
                | "on"
                | "at"
                | "to"
                | "for"
                | "of"
                | "with"
                | "by"
                | "from"
                | "into"
                | "through"
                | "during"
                | "before"
                | "after"
        )
    }

    /// Retrieve from a specific vector space.
    fn retrieve_from_space(
        &self,
        query: &str,
        space_type: &VectorSpaceType,
        top_k: usize,
    ) -> Vec<ScoredNode> {
        // Simulate retrieval with different characteristics per space
        let base_score = match space_type {
            VectorSpaceType::Summary => 0.85,
            VectorSpaceType::Keywords => 0.80,
            VectorSpaceType::FullText => 0.90,
            VectorSpaceType::Title => 0.75,
            VectorSpaceType::Metadata => 0.70,
        };

        create_mock_vector_space_results(query, space_type, top_k, base_score)
    }

    /// Fuse results from multiple vector spaces.
    pub fn fuse_multi_vector_results(
        &self,
        space_results: &[VectorSpaceResult],
    ) -> Vec<ScoredNode> {
        match self.fusion_method.as_str() {
            "weighted_average" => self.weighted_average_fusion(space_results),
            "max_score" => self.max_score_fusion(space_results),
            "rank_fusion" => self.rank_fusion(space_results),
            _ => {
                warn!(
                    "Unknown fusion method: {}, using weighted_average",
                    self.fusion_method
                );
                self.weighted_average_fusion(space_results)
            }
        }
    }

    /// Weighted average fusion of vector space results.
    fn weighted_average_fusion(&self, space_results: &[VectorSpaceResult]) -> Vec<ScoredNode> {
        let result_sets: Vec<Vec<ScoredNode>> =
            space_results.iter().map(|r| r.nodes.clone()).collect();

        let weights = if self.space_weights.len() == space_results.len() {
            self.space_weights.clone()
        } else {
            // Equal weights if mismatch
            vec![1.0 / space_results.len() as f32; space_results.len()]
        };

        let fusion = DistributionBasedFusion::with_weights(weights);
        fusion.fuse_results(result_sets)
    }

    /// Max score fusion - take highest score for each document.
    fn max_score_fusion(&self, space_results: &[VectorSpaceResult]) -> Vec<ScoredNode> {
        let mut document_scores: HashMap<String, f32> = HashMap::new();
        let mut document_nodes: HashMap<String, ScoredNode> = HashMap::new();

        for space_result in space_results {
            for node in &space_result.nodes {
                let doc_key = node.node.content.clone();
                let current_score = document_scores.get(&doc_key).unwrap_or(&0.0);

                if node.score > *current_score {
                    document_scores.insert(doc_key.clone(), node.score);
                    document_nodes.insert(doc_key, node.clone());
                }
            }
        }

        let mut results: Vec<ScoredNode> = document_nodes.into_values().collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }

    /// Rank fusion - combine based on ranks rather than scores.
    fn rank_fusion(&self, space_results: &[VectorSpaceResult]) -> Vec<ScoredNode> {
        let mut document_ranks: HashMap<String, Vec<usize>> = HashMap::new();
        let mut document_nodes: HashMap<String, ScoredNode> = HashMap::new();

        for space_result in space_results {
            for (rank, node) in space_result.nodes.iter().enumerate() {
                let doc_key = node.node.content.clone();
                document_ranks
                    .entry(doc_key.clone())
                    .or_insert_with(Vec::new)
                    .push(rank);
                document_nodes.insert(doc_key, node.clone());
            }
        }

        // Calculate reciprocal rank fusion score
        let mut scored_results = Vec::new();
        for (doc_key, ranks) in document_ranks {
            let rrf_score: f32 = ranks.iter().map(|&rank| 1.0 / (60.0 + rank as f32)).sum();

            if let Some(mut node) = document_nodes.get(&doc_key).cloned() {
                node.score = rrf_score;
                scored_results.push(node);
            }
        }

        scored_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored_results
    }
}

/// Create mock results for vector space demonstration.
fn create_mock_vector_space_results(
    query: &str,
    space_type: &VectorSpaceType,
    top_k: usize,
    base_score: f32,
) -> Vec<ScoredNode> {
    use cheungfun_core::ChunkInfo;
    use uuid::Uuid;

    let doc_id = Uuid::new_v4();
    let chunk_info = ChunkInfo::new(0, 100, 0);

    (0..top_k)
        .map(|i| {
            let score = base_score - (i as f32 * 0.06);
            let content = format!(
                "{} result {} for: {}",
                space_type.display_name(),
                i + 1,
                query.chars().take(20).collect::<String>()
            );
            ScoredNode::new(Node::new(content, doc_id, chunk_info.clone()), score)
        })
        .collect()
}

#[derive(Debug)]
pub enum ExampleError {
    Other(String),
}

impl std::fmt::Display for ExampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExampleError::Other(e) => write!(f, "Error: {}", e),
        }
    }
}

impl std::error::Error for ExampleError {}

#[tokio::main]
async fn main() -> std::result::Result<(), ExampleError> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();

    println!("🎯 Multi-Vector Retrieval Demo");
    println!("==============================");
    println!("Query: {}", args.query);
    println!("Vector Spaces: {}", args.vector_spaces);
    println!("Fusion Method: {}", args.fusion_method);
    println!("Space Weights: {}", args.space_weights);
    println!();

    // Parse vector spaces
    let vector_spaces: Vec<VectorSpaceType> = args
        .vector_spaces
        .split(',')
        .filter_map(|s| VectorSpaceType::from_str(s.trim()))
        .collect();

    if vector_spaces.is_empty() {
        return Err(ExampleError::Other(
            "No valid vector spaces specified".to_string(),
        ));
    }

    // Parse weights
    let space_weights: Vec<f32> = args
        .space_weights
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // Create multi-vector retriever
    let retriever =
        MultiVectorRetriever::new(vector_spaces.clone(), space_weights, args.fusion_method);

    // Perform multi-vector retrieval
    let start_time = Instant::now();
    let space_results = retriever.retrieve_multi_vector(&args.query, args.top_k);
    let retrieval_time = start_time.elapsed();

    println!("🔍 Vector Space Results:");
    println!("{}", "=".repeat(50));

    for space_result in &space_results {
        println!("Vector Space: {}", space_result.space_type.display_name());
        println!("Results: {} nodes", space_result.nodes.len());
        println!("Avg Score: {:.3}", space_result.avg_score);
        println!(
            "Retrieval Time: {:.2}ms",
            space_result.retrieval_time.as_millis()
        );

        if args.verbose {
            for (i, node) in space_result.nodes.iter().take(3).enumerate() {
                println!(
                    "  {}. {} (Score: {:.3})",
                    i + 1,
                    node.node.content.chars().take(40).collect::<String>(),
                    node.score
                );
            }
        }
        println!("{}", "-".repeat(30));
    }

    // Fuse results from all vector spaces
    let fused_results = retriever.fuse_multi_vector_results(&space_results);

    println!("🎯 Final Fused Results:");
    println!("{}", "=".repeat(50));
    println!("Total retrieval time: {:.2}ms", retrieval_time.as_millis());
    println!("Vector spaces used: {}", space_results.len());
    println!("Final results: {} nodes", fused_results.len());
    println!();

    for (i, result) in fused_results.iter().take(args.top_k).enumerate() {
        println!(
            "{}. {} (Score: {:.3})",
            i + 1,
            result.node.content.chars().take(50).collect::<String>(),
            result.score
        );
    }

    // Show multi-vector statistics
    println!("\n📊 Multi-Vector Statistics:");
    println!("{}", "=".repeat(50));

    let total_nodes: usize = space_results.iter().map(|r| r.nodes.len()).sum();
    let avg_space_score =
        space_results.iter().map(|r| r.avg_score).sum::<f32>() / space_results.len() as f32;
    let total_space_time: u128 = space_results
        .iter()
        .map(|r| r.retrieval_time.as_millis())
        .sum();

    println!("Total nodes retrieved: {}", total_nodes);
    println!("Average space score: {:.3}", avg_space_score);
    println!("Total space retrieval time: {}ms", total_space_time);
    println!(
        "Fusion efficiency: {:.1}%",
        (fused_results.len() as f32 / total_nodes as f32) * 100.0
    );

    // Show space-by-space performance
    println!("\n📈 Space Performance:");
    for (i, space_result) in space_results.iter().enumerate() {
        let weight = retriever.space_weights.get(i).unwrap_or(&1.0);
        println!(
            "  {}: {} nodes, {:.3} avg score, weight {:.2}",
            space_result.space_type.display_name(),
            space_result.nodes.len(),
            space_result.avg_score,
            weight
        );
    }

    println!();
    println!("✅ Multi-vector retrieval demonstration completed!");
    println!("🎯 This technique leverages multiple vector representations for better coverage.");
    println!("📈 Different vector spaces capture different aspects of content relevance.");

    Ok(())
}
