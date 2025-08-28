//! Self-Querying Example
//!
//! This example demonstrates self-querying techniques where the system
//! automatically generates and executes multiple related queries to gather
//! comprehensive information. This approach is particularly effective for
//! complex topics that require multiple perspectives or detailed exploration.
//!
//! # Features Demonstrated
//!
//! - **Automatic Query Generation**: Generate related queries from the original
//! - **Multi-perspective Retrieval**: Gather information from different angles
//! - **Query Decomposition**: Break complex queries into simpler sub-queries
//! - **Result Synthesis**: Combine results from multiple self-generated queries
//! - **Relevance Filtering**: Filter generated queries for relevance
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin self_querying --features fastembed -- \
//!     --query "How does machine learning impact modern healthcare?" \
//!     --num-generated-queries 5 \
//!     --min-relevance 0.6 \
//!     --synthesis-method "weighted_fusion"
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
#[command(name = "self-querying")]
#[command(about = "Demonstrates self-querying with automatic query generation")]
struct Args {
    /// Original query to expand
    #[arg(
        long,
        default_value = "How does machine learning impact modern healthcare?"
    )]
    query: String,

    /// Number of queries to generate
    #[arg(long, default_value = "5")]
    num_generated_queries: usize,

    /// Minimum relevance threshold for generated queries
    #[arg(long, default_value = "0.6")]
    min_relevance: f32,

    /// Synthesis method: weighted_fusion, equal_fusion, best_first
    #[arg(long, default_value = "weighted_fusion")]
    synthesis_method: String,

    /// Number of results per query
    #[arg(long, default_value = "8")]
    top_k: usize,

    /// Enable verbose output
    #[arg(long)]
    verbose: bool,
}

/// Self-querying engine that generates and executes related queries.
#[derive(Debug)]
pub struct SelfQueryingEngine {
    /// Number of queries to generate
    num_generated_queries: usize,
    /// Minimum relevance threshold
    min_relevance: f32,
    /// Synthesis method
    synthesis_method: String,
}

/// Generated query with metadata.
#[derive(Debug, Clone)]
pub struct GeneratedQuery {
    /// The generated query text
    pub text: String,
    /// Relevance score to original query
    pub relevance_score: f32,
    /// Query type/category
    pub query_type: String,
    /// Expected information type
    pub info_type: String,
}

/// Self-querying result containing all sub-query results.
#[derive(Debug)]
pub struct SelfQueryingResult {
    /// Original query
    pub original_query: String,
    /// Generated queries
    pub generated_queries: Vec<GeneratedQuery>,
    /// Results for each generated query
    pub query_results: HashMap<String, Vec<ScoredNode>>,
    /// Final synthesized results
    pub synthesized_results: Vec<ScoredNode>,
    /// Total processing time
    pub processing_time: Duration,
}

impl SelfQueryingEngine {
    /// Create a new self-querying engine.
    pub fn new(num_generated_queries: usize, min_relevance: f32, synthesis_method: String) -> Self {
        Self {
            num_generated_queries,
            min_relevance,
            synthesis_method,
        }
    }

    /// Perform self-querying retrieval.
    pub fn query_with_self_generation(
        &self,
        original_query: &str,
        top_k: usize,
    ) -> SelfQueryingResult {
        let start_time = Instant::now();

        info!("Starting self-querying for: '{}'", original_query);

        // Generate related queries
        let generated_queries = self.generate_related_queries(original_query);
        info!("Generated {} related queries", generated_queries.len());

        // Execute each generated query
        let mut query_results = HashMap::new();

        for generated_query in &generated_queries {
            info!("Executing generated query: '{}'", generated_query.text);
            let results = self.execute_single_query(&generated_query.text, top_k);
            query_results.insert(generated_query.text.clone(), results);
        }

        // Synthesize all results
        let synthesized_results = self.synthesize_results(&generated_queries, &query_results);

        let processing_time = start_time.elapsed();

        SelfQueryingResult {
            original_query: original_query.to_string(),
            generated_queries,
            query_results,
            synthesized_results,
            processing_time,
        }
    }

    /// Generate related queries from the original query.
    fn generate_related_queries(&self, original_query: &str) -> Vec<GeneratedQuery> {
        let mut generated_queries = Vec::new();

        // Extract key terms from original query
        let key_terms = self.extract_key_terms(original_query);
        let main_topic = if key_terms.is_empty() {
            "machine learning".to_string()
        } else {
            key_terms.join(" ")
        };

        // Query generation templates based on the original query
        let templates = vec![
            (
                format!("What are the applications of {}", main_topic),
                "applications",
            ),
            (format!("How does {} work", main_topic), "mechanism"),
            (
                format!("What are the benefits of {}", main_topic),
                "advantages",
            ),
            (
                format!("What are the challenges of {}", main_topic),
                "limitations",
            ),
            (
                format!("What are examples of {} in practice", main_topic),
                "examples",
            ),
            (format!("What is the future of {}", main_topic), "trends"),
            (format!("How to implement {}", main_topic), "implementation"),
            (
                format!("What are the latest developments in {}", main_topic),
                "recent",
            ),
        ];

        for (generated_text, info_type) in templates.iter().take(self.num_generated_queries) {
            let relevance_score = self.calculate_query_relevance(original_query, generated_text);

            // Lower the threshold for demonstration
            if relevance_score >= (self.min_relevance * 0.5) {
                generated_queries.push(GeneratedQuery {
                    text: generated_text.clone(),
                    relevance_score,
                    query_type: "template_based".to_string(),
                    info_type: info_type.to_string(),
                });
            }
        }

        // Add decomposed sub-queries
        let sub_queries = self.decompose_query(original_query);
        for sub_query in sub_queries {
            let relevance_score = self.calculate_query_relevance(original_query, &sub_query);
            if relevance_score >= self.min_relevance {
                generated_queries.push(GeneratedQuery {
                    text: sub_query,
                    relevance_score,
                    query_type: "decomposed".to_string(),
                    info_type: "sub_topic".to_string(),
                });
            }
        }

        generated_queries
    }

    /// Extract key terms from a query.
    fn extract_key_terms(&self, query: &str) -> Vec<String> {
        query
            .split_whitespace()
            .filter(|word| word.len() > 3 && !self.is_stop_word(word))
            .map(|word| word.to_lowercase())
            .collect()
    }

    /// Check if a word is a stop word.
    fn is_stop_word(&self, word: &str) -> bool {
        matches!(
            word.to_lowercase().as_str(),
            "what"
                | "how"
                | "why"
                | "when"
                | "where"
                | "who"
                | "which"
                | "does"
                | "is"
                | "are"
                | "the"
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

    /// Decompose complex query into simpler sub-queries.
    fn decompose_query(&self, query: &str) -> Vec<String> {
        let mut sub_queries = Vec::new();

        // Simple decomposition based on conjunctions
        if query.contains(" and ") {
            let parts: Vec<&str> = query.split(" and ").collect();
            for part in parts {
                if part.trim().len() > 10 {
                    sub_queries.push(part.trim().to_string());
                }
            }
        }

        // Decompose based on question words
        if query.contains("how") && query.contains("what") {
            sub_queries.push(query.replace("how", "what is").to_string());
            sub_queries.push(query.replace("what", "how to use").to_string());
        }

        sub_queries
    }

    /// Calculate relevance between original and generated query.
    fn calculate_query_relevance(&self, original: &str, generated: &str) -> f32 {
        let original_words: std::collections::HashSet<&str> = original.split_whitespace().collect();
        let generated_words: std::collections::HashSet<&str> =
            generated.split_whitespace().collect();

        let intersection = original_words.intersection(&generated_words).count();
        let union = original_words.union(&generated_words).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Execute a single query and return results.
    fn execute_single_query(&self, query: &str, top_k: usize) -> Vec<ScoredNode> {
        // Simulate query execution with varying quality
        create_mock_query_results(query, top_k)
    }

    /// Synthesize results from all generated queries.
    fn synthesize_results(
        &self,
        generated_queries: &[GeneratedQuery],
        query_results: &HashMap<String, Vec<ScoredNode>>,
    ) -> Vec<ScoredNode> {
        match self.synthesis_method.as_str() {
            "weighted_fusion" => self.weighted_fusion_synthesis(generated_queries, query_results),
            "equal_fusion" => self.equal_fusion_synthesis(query_results),
            "best_first" => self.best_first_synthesis(generated_queries, query_results),
            _ => {
                warn!(
                    "Unknown synthesis method: {}, using weighted_fusion",
                    self.synthesis_method
                );
                self.weighted_fusion_synthesis(generated_queries, query_results)
            }
        }
    }

    /// Synthesize using weighted fusion based on query relevance.
    fn weighted_fusion_synthesis(
        &self,
        generated_queries: &[GeneratedQuery],
        query_results: &HashMap<String, Vec<ScoredNode>>,
    ) -> Vec<ScoredNode> {
        let mut result_sets = Vec::new();
        let mut weights = Vec::new();

        for query in generated_queries {
            if let Some(results) = query_results.get(&query.text) {
                result_sets.push(results.clone());
                weights.push(query.relevance_score);
            }
        }

        if result_sets.is_empty() {
            return Vec::new();
        }

        let fusion = DistributionBasedFusion::with_weights(weights);
        fusion.fuse_results(result_sets)
    }

    /// Synthesize using equal weights for all queries.
    fn equal_fusion_synthesis(
        &self,
        query_results: &HashMap<String, Vec<ScoredNode>>,
    ) -> Vec<ScoredNode> {
        let result_sets: Vec<Vec<ScoredNode>> = query_results.values().cloned().collect();

        if result_sets.is_empty() {
            return Vec::new();
        }

        let fusion = DistributionBasedFusion::new(result_sets.len());
        fusion.fuse_results(result_sets)
    }

    /// Synthesize by taking best results from highest-relevance queries first.
    fn best_first_synthesis(
        &self,
        generated_queries: &[GeneratedQuery],
        query_results: &HashMap<String, Vec<ScoredNode>>,
    ) -> Vec<ScoredNode> {
        let mut sorted_queries = generated_queries.to_vec();
        sorted_queries.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());

        let mut all_results = Vec::new();

        for query in sorted_queries {
            if let Some(results) = query_results.get(&query.text) {
                all_results.extend(results.clone());
            }
        }

        // Sort by score and remove duplicates
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        all_results.dedup_by(|a, b| a.node.content == b.node.content);

        all_results
    }
}

/// Create mock results for query demonstration.
fn create_mock_query_results(query: &str, top_k: usize) -> Vec<ScoredNode> {
    use cheungfun_core::ChunkInfo;
    use uuid::Uuid;

    let doc_id = Uuid::new_v4();
    let chunk_info = ChunkInfo::new(0, 100, 0);

    // Vary base score based on query characteristics
    let base_score = if query.contains("What are") {
        0.85
    } else if query.contains("How does") {
        0.90
    } else if query.contains("Why is") {
        0.80
    } else {
        0.75
    };

    (0..top_k)
        .map(|i| {
            let score = base_score - (i as f32 * 0.07);
            let content = format!(
                "Result {} for self-query: {}",
                i + 1,
                query.chars().take(25).collect::<String>()
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

    println!("🤖 Self-Querying Demo");
    println!("=====================");
    println!("Original Query: {}", args.query);
    println!("Generated Queries: {}", args.num_generated_queries);
    println!("Min Relevance: {:.2}", args.min_relevance);
    println!("Synthesis Method: {}", args.synthesis_method);
    println!();

    // Create self-querying engine
    let engine = SelfQueryingEngine::new(
        args.num_generated_queries,
        args.min_relevance,
        args.synthesis_method,
    );

    // Perform self-querying
    let result = engine.query_with_self_generation(&args.query, args.top_k);

    println!("🔍 Generated Queries:");
    println!("{}", "=".repeat(50));

    for (i, query) in result.generated_queries.iter().enumerate() {
        println!(
            "{}. {} (Relevance: {:.3}, Type: {})",
            i + 1,
            query.text,
            query.relevance_score,
            query.query_type
        );

        if args.verbose {
            if let Some(results) = result.query_results.get(&query.text) {
                println!("   Results: {} nodes", results.len());
                for (j, node) in results.iter().take(2).enumerate() {
                    println!(
                        "     {}. {} (Score: {:.3})",
                        j + 1,
                        node.node.content.chars().take(30).collect::<String>(),
                        node.score
                    );
                }
            }
        }
        println!();
    }

    println!("🎯 Final Synthesized Results:");
    println!("{}", "=".repeat(50));
    println!(
        "Processing time: {:.2}ms",
        result.processing_time.as_millis()
    );
    println!("Total queries executed: {}", result.generated_queries.len());
    println!("Final results: {} nodes", result.synthesized_results.len());
    println!();

    for (i, node) in result
        .synthesized_results
        .iter()
        .take(args.top_k)
        .enumerate()
    {
        println!(
            "{}. {} (Score: {:.3})",
            i + 1,
            node.node.content.chars().take(60).collect::<String>(),
            node.score
        );
    }

    // Show synthesis statistics
    println!("\n📊 Self-Querying Statistics:");
    println!("{}", "=".repeat(50));

    let total_results: usize = result.query_results.values().map(|r| r.len()).sum();
    let avg_relevance = result
        .generated_queries
        .iter()
        .map(|q| q.relevance_score)
        .sum::<f32>()
        / result.generated_queries.len() as f32;

    println!("Total sub-query results: {}", total_results);
    println!("Average query relevance: {:.3}", avg_relevance);
    println!(
        "Synthesis efficiency: {:.1}%",
        (result.synthesized_results.len() as f32 / total_results as f32) * 100.0
    );

    println!();
    println!("✅ Self-querying demonstration completed!");
    println!("🎯 This technique automatically explores multiple aspects of a topic.");
    println!("📈 Generated queries provide comprehensive coverage of the subject matter.");

    Ok(())
}
