//! Iterative Retrieval Example
//!
//! This example demonstrates iterative retrieval techniques that progressively
//! refine search results through multiple rounds of retrieval. This approach
//! is particularly effective for complex queries that require gathering
//! information from multiple sources or perspectives.
//!
//! # Features Demonstrated
//!
//! - **Multi-round Retrieval**: Progressive refinement through iterations
//! - **Query Expansion**: Expand queries based on initial results
//! - **Result Fusion**: Combine results from multiple iterations
//! - **Convergence Detection**: Stop when results stabilize
//! - **Adaptive Iteration**: Adjust strategy based on result quality
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin iterative_retrieval --features fastembed -- \
//!     --query "machine learning applications in healthcare" \
//!     --max-iterations 3 \
//!     --convergence-threshold 0.1 \
//!     --expansion-factor 2
//! ```

use cheungfun::prelude::*;
use cheungfun_query::advanced::fusion::DistributionBasedFusion;
use clap::Parser;
use std::{collections::HashSet, time::Instant};
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "iterative-retrieval")]
#[command(about = "Demonstrates iterative retrieval with progressive refinement")]
struct Args {
    /// Query to search for
    #[arg(long, default_value = "machine learning applications in healthcare")]
    query: String,

    /// Maximum number of iterations
    #[arg(long, default_value = "3")]
    max_iterations: usize,

    /// Convergence threshold (0.0-1.0)
    #[arg(long, default_value = "0.1")]
    convergence_threshold: f32,

    /// Query expansion factor
    #[arg(long, default_value = "2")]
    expansion_factor: usize,

    /// Number of results per iteration
    #[arg(long, default_value = "10")]
    top_k: usize,

    /// Enable verbose output
    #[arg(long)]
    verbose: bool,
}

/// Iterative retrieval engine that progressively refines results.
#[derive(Debug)]
pub struct IterativeRetriever {
    /// Maximum iterations allowed
    max_iterations: usize,
    /// Convergence threshold for stopping
    convergence_threshold: f32,
    /// Query expansion factor
    expansion_factor: usize,
}

/// Iteration result containing retrieved nodes and metadata.
#[derive(Debug, Clone)]
pub struct IterationResult {
    /// Retrieved nodes in this iteration
    pub nodes: Vec<ScoredNode>,
    /// Expanded query used
    pub expanded_query: String,
    /// Quality score for this iteration
    pub quality_score: f32,
    /// Iteration number
    pub iteration: usize,
}

impl IterativeRetriever {
    /// Create a new iterative retriever.
    pub fn new(max_iterations: usize, convergence_threshold: f32, expansion_factor: usize) -> Self {
        Self {
            max_iterations,
            convergence_threshold,
            expansion_factor,
        }
    }

    /// Perform iterative retrieval with progressive refinement.
    pub fn retrieve_iterative(&self, query: &str, top_k: usize) -> Vec<IterationResult> {
        let mut iterations = Vec::new();
        let mut current_query = query.to_string();
        let mut all_results: Vec<ScoredNode> = Vec::new();
        let mut previous_quality = 0.0f32;

        info!(
            "Starting iterative retrieval with max {} iterations",
            self.max_iterations
        );

        for iteration in 0..self.max_iterations {
            info!("Iteration {}: Query = '{}'", iteration + 1, current_query);

            // Perform retrieval for current iteration
            let iteration_nodes = self.retrieve_single_iteration(&current_query, top_k);

            // Calculate quality score for this iteration
            let quality_score = self.calculate_iteration_quality(&iteration_nodes);

            // Create iteration result
            let iteration_result = IterationResult {
                nodes: iteration_nodes.clone(),
                expanded_query: current_query.clone(),
                quality_score,
                iteration: iteration + 1,
            };

            iterations.push(iteration_result);

            // Add new results to cumulative collection
            all_results.extend(iteration_nodes);

            // Check for convergence
            let quality_improvement = quality_score - previous_quality;
            if iteration > 0 && quality_improvement.abs() < self.convergence_threshold {
                info!(
                    "Convergence detected at iteration {} (improvement: {:.3})",
                    iteration + 1,
                    quality_improvement
                );
                break;
            }

            previous_quality = quality_score;

            // Expand query for next iteration
            if iteration < self.max_iterations - 1 {
                current_query = self.expand_query(&current_query, &all_results);
                info!("Expanded query for next iteration: '{}'", current_query);
            }
        }

        iterations
    }

    /// Perform retrieval for a single iteration.
    fn retrieve_single_iteration(&self, query: &str, top_k: usize) -> Vec<ScoredNode> {
        // Simulate retrieval with different strategies based on iteration
        create_mock_iteration_results(query, top_k)
    }

    /// Calculate quality score for an iteration.
    fn calculate_iteration_quality(&self, nodes: &[ScoredNode]) -> f32 {
        if nodes.is_empty() {
            return 0.0;
        }

        // Calculate average score
        let avg_score = nodes.iter().map(|n| n.score).sum::<f32>() / nodes.len() as f32;

        // Calculate diversity (simplified)
        let unique_content_ratio = self.calculate_content_diversity(nodes);

        // Combine score and diversity
        (avg_score * 0.7 + unique_content_ratio * 0.3).clamp(0.0, 1.0)
    }

    /// Calculate content diversity in results.
    fn calculate_content_diversity(&self, nodes: &[ScoredNode]) -> f32 {
        if nodes.len() <= 1 {
            return 1.0;
        }

        let mut unique_words = HashSet::new();
        let mut total_words = 0;

        for node in nodes {
            let words: Vec<&str> = node.node.content.split_whitespace().collect();
            total_words += words.len();
            for word in words {
                unique_words.insert(word.to_lowercase());
            }
        }

        if total_words == 0 {
            1.0
        } else {
            unique_words.len() as f32 / total_words as f32
        }
    }

    /// Expand query based on previous results.
    fn expand_query(&self, original_query: &str, results: &[ScoredNode]) -> String {
        // Extract key terms from top results
        let mut term_frequency: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        // Analyze top results for expansion terms
        let top_results = results.iter().take(5);

        for result in top_results {
            let words: Vec<&str> = result
                .node
                .content
                .split_whitespace()
                .filter(|w| w.len() > 3) // Filter short words
                .collect();

            for word in words {
                let word_lower = word.to_lowercase();
                // Skip common words and original query terms
                if !self.is_common_word(&word_lower)
                    && !original_query.to_lowercase().contains(&word_lower)
                {
                    *term_frequency.entry(word_lower).or_insert(0) += 1;
                }
            }
        }

        // Select top expansion terms
        let mut expansion_terms: Vec<_> = term_frequency.into_iter().collect();
        expansion_terms.sort_by(|a, b| b.1.cmp(&a.1));

        let selected_terms: Vec<String> = expansion_terms
            .into_iter()
            .take(self.expansion_factor)
            .map(|(term, _)| term)
            .collect();

        if selected_terms.is_empty() {
            original_query.to_string()
        } else {
            format!("{} {}", original_query, selected_terms.join(" "))
        }
    }

    /// Check if a word is a common stop word.
    fn is_common_word(&self, word: &str) -> bool {
        matches!(
            word,
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
                | "up"
                | "about"
                | "into"
                | "through"
                | "during"
                | "before"
                | "after"
                | "above"
                | "below"
                | "between"
                | "among"
                | "this"
                | "that"
                | "these"
                | "those"
                | "is"
                | "are"
                | "was"
                | "were"
                | "be"
                | "been"
                | "being"
                | "have"
                | "has"
                | "had"
                | "do"
                | "does"
                | "did"
                | "will"
                | "would"
                | "could"
                | "should"
                | "may"
                | "might"
                | "must"
                | "can"
                | "shall"
        )
    }

    /// Fuse results from all iterations.
    pub fn fuse_iteration_results(&self, iterations: &[IterationResult]) -> Vec<ScoredNode> {
        if iterations.is_empty() {
            return Vec::new();
        }

        // Collect all result sets
        let result_sets: Vec<Vec<ScoredNode>> =
            iterations.iter().map(|iter| iter.nodes.clone()).collect();

        // Use distribution-based fusion with iteration weights
        let weights: Vec<f32> = iterations.iter().map(|iter| iter.quality_score).collect();

        let fusion = DistributionBasedFusion::with_weights(weights);
        fusion.fuse_results(result_sets)
    }
}

/// Create mock results for iteration demonstration.
fn create_mock_iteration_results(query: &str, top_k: usize) -> Vec<ScoredNode> {
    use cheungfun_core::ChunkInfo;
    use uuid::Uuid;

    let doc_id = Uuid::new_v4();
    let chunk_info = ChunkInfo::new(Some(0), Some(100), 0);

    // Simulate different result quality based on query complexity
    let base_score = if query.len() > 50 { 0.9 } else { 0.8 };

    (0..top_k)
        .map(|i| {
            let score = base_score - (i as f32 * 0.08);
            let content = format!(
                "Iteration result {} for: {}",
                i + 1,
                query.chars().take(30).collect::<String>()
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

    println!("🔄 Iterative Retrieval Demo");
    println!("===========================");
    println!("Query: {}", args.query);
    println!("Max Iterations: {}", args.max_iterations);
    println!("Convergence Threshold: {:.2}", args.convergence_threshold);
    println!("Expansion Factor: {}", args.expansion_factor);
    println!();

    // Create iterative retriever
    let retriever = IterativeRetriever::new(
        args.max_iterations,
        args.convergence_threshold,
        args.expansion_factor,
    );

    // Perform iterative retrieval
    let start_time = Instant::now();
    let iterations = retriever.retrieve_iterative(&args.query, args.top_k);
    let total_time = start_time.elapsed();

    println!("🔍 Iteration Results:");
    println!("{}", "=".repeat(50));

    for iteration in &iterations {
        println!(
            "Iteration {}: Quality Score = {:.3}",
            iteration.iteration, iteration.quality_score
        );
        println!("Expanded Query: {}", iteration.expanded_query);
        println!("Results: {} nodes", iteration.nodes.len());

        if args.verbose {
            for (i, node) in iteration.nodes.iter().take(3).enumerate() {
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

    // Fuse all iteration results
    let fused_results = retriever.fuse_iteration_results(&iterations);

    println!("\n🎯 Final Fused Results:");
    println!("{}", "=".repeat(50));
    println!("Total iterations: {}", iterations.len());
    println!("Total time: {:.2}ms", total_time.as_millis());
    println!("Final results: {} nodes", fused_results.len());
    println!();

    for (i, result) in fused_results.iter().take(args.top_k).enumerate() {
        println!(
            "{}. {} (Score: {:.3})",
            i + 1,
            result.node.content.chars().take(60).collect::<String>(),
            result.score
        );
    }

    // Show iteration statistics
    println!("\n📊 Iteration Statistics:");
    println!("{}", "=".repeat(50));

    let quality_scores: Vec<f32> = iterations.iter().map(|i| i.quality_score).collect();
    let avg_quality = quality_scores.iter().sum::<f32>() / quality_scores.len() as f32;
    let quality_improvement = if quality_scores.len() > 1 {
        quality_scores.last().unwrap() - quality_scores.first().unwrap()
    } else {
        0.0
    };

    println!("Average quality score: {:.3}", avg_quality);
    println!("Quality improvement: {:.3}", quality_improvement);
    println!(
        "Convergence achieved: {}",
        quality_improvement.abs() < args.convergence_threshold
    );

    println!();
    println!("✅ Iterative retrieval demonstration completed!");
    println!("🎯 This technique progressively refines results through multiple rounds.");
    println!("📈 Each iteration can expand the query and improve result quality.");

    Ok(())
}
