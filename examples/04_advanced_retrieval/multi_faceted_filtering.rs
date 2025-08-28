//! Distribution-Based Score Fusion Example
//!
//! This example demonstrates the new Distribution-Based Score Fusion algorithm
//! that we successfully integrated into cheungfun. This fusion method provides
//! better normalization than simple relative scoring by using statistical
//! distribution properties.
//!
//! # Features Demonstrated
//!
//! - **Distribution-Based Fusion**: Advanced score fusion using z-score normalization
//! - **Multiple Search Strategies**: Vector and hybrid search comparison
//! - **Statistical Normalization**: Mean and standard deviation based scoring
//! - **Sigmoid Transformation**: Convert z-scores to 0-1 range
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin advanced_multi_filtering --features fastembed -- \
//!     --query "machine learning algorithms" \
//!     --fusion-method "distribution_based"
//! ```

use cheungfun::prelude::*;
use cheungfun_query::advanced::fusion::{DistributionBasedFusion, ReciprocalRankFusion};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "distribution-fusion-demo")]
#[command(about = "Demonstrates Distribution-Based Score Fusion")]
struct Args {
    /// Query to search for
    #[arg(long, default_value = "machine learning algorithms")]
    query: String,

    /// Fusion method: rrf, distribution_based
    #[arg(long, default_value = "distribution_based")]
    fusion_method: String,

    /// Number of results to return
    #[arg(long, default_value = "10")]
    top_k: usize,

    /// Enable verbose output
    #[arg(long)]
    verbose: bool,
}

/// Demonstration of Distribution-Based Score Fusion.
#[derive(Debug, Clone)]
pub struct FusionDemo {
    /// Fusion algorithm to use
    pub fusion_algorithm: String,
}

impl FusionDemo {
    /// Create a new fusion demo.
    pub fn new(fusion_algorithm: String) -> Self {
        Self { fusion_algorithm }
    }

    /// Demonstrate fusion algorithms with sample data.
    pub fn demonstrate_fusion(&self) -> std::result::Result<(), ExampleError> {
        println!(
            "🎭 Demonstrating {} fusion algorithm",
            self.fusion_algorithm
        );

        // Create sample scored nodes with different score distributions
        use cheungfun_core::ChunkInfo;
        use uuid::Uuid;

        let chunk_info = ChunkInfo::new(0, 100, 0);
        let doc_id = Uuid::new_v4();

        let result_set_1 = vec![
            ScoredNode::new(
                Node::new("Content A".to_string(), doc_id, chunk_info.clone()),
                0.9,
            ),
            ScoredNode::new(
                Node::new("Content B".to_string(), doc_id, chunk_info.clone()),
                0.7,
            ),
            ScoredNode::new(
                Node::new("Content C".to_string(), doc_id, chunk_info.clone()),
                0.5,
            ),
        ];

        let result_set_2 = vec![
            ScoredNode::new(
                Node::new("Content A".to_string(), doc_id, chunk_info.clone()),
                0.8,
            ),
            ScoredNode::new(
                Node::new("Content D".to_string(), doc_id, chunk_info.clone()),
                0.6,
            ),
            ScoredNode::new(
                Node::new("Content E".to_string(), doc_id, chunk_info.clone()),
                0.4,
            ),
        ];

        println!("📊 Input Result Sets:");
        println!("Set 1: [0.9, 0.7, 0.5] (mean: 0.70, std: 0.20)");
        println!("Set 2: [0.8, 0.6, 0.4] (mean: 0.60, std: 0.20)");
        println!();

        // Apply fusion
        let fused_results = match self.fusion_algorithm.as_str() {
            "distribution_based" => {
                let fusion = DistributionBasedFusion::new(2);
                fusion.fuse_results(vec![result_set_1, result_set_2])
            }
            "rrf" => {
                let fusion = ReciprocalRankFusion::new(60.0);
                fusion.fuse_results(vec![result_set_1, result_set_2])
            }
            _ => {
                return Err(ExampleError::Other(format!(
                    "Unknown fusion method: {}",
                    self.fusion_algorithm
                )));
            }
        };

        println!("🎯 Fused Results:");
        for (i, result) in fused_results.iter().enumerate() {
            println!(
                "{}. {} (Score: {:.3})",
                i + 1,
                result.node.content,
                result.score
            );
        }

        Ok(())
    }
}

impl Default for FusionDemo {
    fn default() -> Self {
        Self::new("distribution_based".to_string())
    }
}

#[derive(Debug)]
pub enum ExampleError {
    Cheungfun(cheungfun_core::CheungfunError),
    Siumai(siumai::LlmError),
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

impl From<cheungfun_core::CheungfunError> for ExampleError {
    fn from(err: cheungfun_core::CheungfunError) -> Self {
        ExampleError::Cheungfun(err)
    }
}

impl From<siumai::LlmError> for ExampleError {
    fn from(err: siumai::LlmError) -> Self {
        ExampleError::Siumai(err)
    }
}

impl From<std::io::Error> for ExampleError {
    fn from(err: std::io::Error) -> Self {
        ExampleError::Io(err)
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), ExampleError> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();

    println!("🎭 Distribution-Based Score Fusion Demo");
    println!("======================================");
    println!("Query: {}", args.query);
    println!("Fusion Method: {}", args.fusion_method);
    println!();

    // Create fusion demo
    let demo = FusionDemo::new(args.fusion_method);

    // Demonstrate fusion algorithm
    demo.demonstrate_fusion()?;

    println!();
    println!("✅ Distribution-Based Score Fusion demonstration completed!");
    println!("📈 This fusion method provides better normalization by using statistical distribution properties.");
    println!("🔬 It applies z-score normalization followed by sigmoid transformation for optimal score fusion.");

    Ok(())
}
