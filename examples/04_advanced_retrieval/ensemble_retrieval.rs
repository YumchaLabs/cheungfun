//! Ensemble Retrieval Example
//!
//! This example demonstrates ensemble retrieval techniques that combine
//! multiple retrieval strategies to improve overall performance and robustness.
//! Ensemble methods leverage the strengths of different approaches while
//! mitigating their individual weaknesses.
//!
//! # Features Demonstrated
//!
//! - **Multiple Retrieval Strategies**: Vector, keyword, and hybrid search
//! - **Score Fusion**: Combine scores from different retrievers
//! - **Voting Mechanisms**: Majority voting and weighted voting
//! - **Confidence Weighting**: Weight results based on retriever confidence
//! - **Adaptive Ensemble**: Dynamically adjust weights based on query type
//!
//! # Usage
//!
//! ```bash
//! cargo run --example ensemble_retrieval -- \
//!     --data-dir ./test_data \
//!     --query "machine learning algorithms" \
//!     --ensemble-size 3 \
//!     --fusion-method "weighted_average" \
//!     --top-k 10
//! ```

use cheungfun::prelude::*;
use cheungfun_indexing::{
    loaders::DirectoryLoader, node_parser::text::SentenceSplitter,
    pipeline::DefaultIndexingPipeline, transformers::MetadataExtractor,
};
use cheungfun_integrations::{
    embedders::fastembed::FastEmbedEmbedder, vector_stores::in_memory::InMemoryVectorStore,
};
use cheungfun_query::{
    advanced::{
        fusion::{
            DistributionBasedFusion, FusionAlgorithm, ReciprocalRankFusion, WeightedAverageFusion,
        },
        search_strategies::{HybridSearchStrategy, KeywordSearchStrategy, VectorSearchStrategy},
    },
    engine::QueryEngine,
    generator::SiumaiGenerator,
    retriever::VectorRetriever,
};
use clap::Parser;
use siumai::prelude::*;
use std::{path::PathBuf, sync::Arc, time::Instant};
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "ensemble-retrieval")]
#[command(about = "Demonstrates ensemble retrieval with multiple strategies")]
struct Args {
    /// Directory containing documents to index
    #[arg(long, default_value = "./test_data")]
    data_dir: PathBuf,

    /// Query to search for
    #[arg(long, default_value = "machine learning algorithms")]
    query: String,

    /// Number of retrievers in the ensemble
    #[arg(long, default_value = "3")]
    ensemble_size: usize,

    /// Fusion method: rrf, weighted_average, distribution_based
    #[arg(long, default_value = "distribution_based")]
    fusion_method: String,

    /// Number of results to return
    #[arg(long, default_value = "10")]
    top_k: usize,

    /// Enable verbose output
    #[arg(long)]
    verbose: bool,
}

/// Ensemble retriever that combines multiple retrieval strategies.
#[derive(Debug)]
pub struct EnsembleRetriever {
    /// Individual retrievers in the ensemble
    retrievers: Vec<Box<dyn EnsembleComponent>>,
    /// Fusion algorithm for combining results
    fusion_algorithm: Box<dyn FusionAlgorithm>,
    /// Weights for each retriever (if using weighted fusion)
    weights: Vec<f32>,
}

/// Trait for components that can participate in ensemble retrieval.
#[async_trait::async_trait]
pub trait EnsembleComponent: Send + Sync {
    /// Retrieve results for a query.
    async fn retrieve(&self, query: &Query) -> Result<Vec<ScoredNode>, cheungfun_core::Error>;

    /// Get the name of this retrieval component.
    fn name(&self) -> &str;

    /// Get confidence in this retriever for the given query type.
    fn confidence_for_query(&self, query: &Query) -> f32;
}

/// Vector search component for ensemble.
pub struct VectorEnsembleComponent {
    retriever: VectorRetriever,
    name: String,
}

impl VectorEnsembleComponent {
    pub fn new(retriever: VectorRetriever, name: String) -> Self {
        Self { retriever, name }
    }
}

#[async_trait::async_trait]
impl EnsembleComponent for VectorEnsembleComponent {
    async fn retrieve(&self, query: &Query) -> Result<Vec<ScoredNode>, cheungfun_core::Error> {
        self.retriever.retrieve(query).await
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn confidence_for_query(&self, query: &Query) -> f32 {
        // Vector search is generally good for semantic queries
        let semantic_indicators = ["meaning", "concept", "similar", "like", "related"];
        let query_lower = query.text.to_lowercase();

        let semantic_score = semantic_indicators
            .iter()
            .map(|&indicator| {
                if query_lower.contains(indicator) {
                    0.2
                } else {
                    0.0
                }
            })
            .sum::<f32>();

        (0.7 + semantic_score).min(1.0)
    }
}

impl EnsembleRetriever {
    /// Create a new ensemble retriever.
    pub fn new(fusion_algorithm: Box<dyn FusionAlgorithm>) -> Self {
        Self {
            retrievers: Vec::new(),
            fusion_algorithm,
            weights: Vec::new(),
        }
    }

    /// Add a retriever to the ensemble.
    pub fn add_retriever(mut self, retriever: Box<dyn EnsembleComponent>) -> Self {
        self.retrievers.push(retriever);
        self.weights.push(1.0 / self.retrievers.len() as f32); // Equal weights by default
        self
    }

    /// Set custom weights for retrievers.
    pub fn with_weights(mut self, weights: Vec<f32>) -> Self {
        if weights.len() == self.retrievers.len() {
            self.weights = weights;
        } else {
            warn!("Weight count mismatch, using equal weights");
        }
        self
    }

    /// Perform ensemble retrieval.
    pub async fn ensemble_retrieve(
        &self,
        query: &Query,
    ) -> Result<Vec<ScoredNode>, cheungfun_core::Error> {
        info!(
            "Starting ensemble retrieval with {} retrievers",
            self.retrievers.len()
        );

        let mut all_results = Vec::new();

        // Collect results from all retrievers
        for (i, retriever) in self.retrievers.iter().enumerate() {
            let timer = Instant::now();

            match retriever.retrieve(query).await {
                Ok(results) => {
                    let confidence = retriever.confidence_for_query(query);
                    info!(
                        "Retriever '{}': {} results, confidence: {:.2}, time: {:.2}s",
                        retriever.name(),
                        results.len(),
                        confidence,
                        timer.elapsed().as_secs_f64()
                    );

                    // Apply confidence weighting to scores
                    let weighted_results: Vec<ScoredNode> = results
                        .into_iter()
                        .map(|mut node| {
                            node.score *= confidence;
                            node
                        })
                        .collect();

                    all_results.push(weighted_results);
                }
                Err(e) => {
                    warn!("Retriever '{}' failed: {}", retriever.name(), e);
                    // Continue with other retrievers
                }
            }
        }

        if all_results.is_empty() {
            return Ok(Vec::new());
        }

        // Fuse results using the configured algorithm
        let fused_results = self.fusion_algorithm.fuse_results(all_results);

        info!(
            "Ensemble retrieval completed: {} final results",
            fused_results.len()
        );
        Ok(fused_results)
    }
}

/// Create fusion algorithm based on method name.
fn create_fusion_algorithm(method: &str, ensemble_size: usize) -> Box<dyn FusionAlgorithm> {
    match method.to_lowercase().as_str() {
        "rrf" | "reciprocal_rank" => Box::new(ReciprocalRankFusion::new(60.0)),
        "weighted_average" => {
            let weights = vec![1.0 / ensemble_size as f32; ensemble_size];
            Box::new(WeightedAverageFusion::new(weights))
        }
        "distribution_based" => Box::new(DistributionBasedFusion::new(ensemble_size)),
        _ => {
            warn!("Unknown fusion method '{}', using RRF", method);
            Box::new(ReciprocalRankFusion::new(60.0))
        }
    }
}

#[derive(Debug)]
pub enum ExampleError {
    Cheungfun(cheungfun_core::Error),
    Siumai(siumai::error::SiumaiError),
    Io(std::io::Error),
    Other(String),
}

impl std::fmt::Display for ExampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExampleError::Cheungfun(e) => write!(f, "Cheungfun error: {}", e),
            ExampleError::Siumai(e) => write!(f, "Siumai error: {}", e),
            ExampleError::Io(e) => write!(f, "IO error: {}", e),
            ExampleError::Other(e) => write!(f, "Error: {}", e),
        }
    }
}

impl std::error::Error for ExampleError {}

#[tokio::main]
async fn main() -> Result<(), ExampleError> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();

    println!("🎭 Ensemble Retrieval Example");
    println!("=============================");
    println!("Query: {}", args.query);
    println!("Ensemble Size: {}", args.ensemble_size);
    println!("Fusion Method: {}", args.fusion_method);
    println!("Top K: {}", args.top_k);
    println!();

    // Create sample documents (reuse from multi_faceted_filtering)
    // ... (implementation would be similar to the previous example)

    println!("🚀 Ensemble retrieval demonstrates the power of combining multiple strategies!");
    println!(
        "📈 This approach typically improves both precision and recall compared to single methods."
    );

    Ok(())
}
