//! Adaptive Retrieval Example
//!
//! This example demonstrates adaptive retrieval techniques that dynamically
//! adjust retrieval strategies based on query characteristics, user context,
//! and system performance. Adaptive retrieval optimizes the retrieval process
//! by selecting the most appropriate strategy for each specific query.
//!
//! # Features Demonstrated
//!
//! - **Query Classification**: Classify queries to select optimal strategies
//! - **Dynamic Strategy Selection**: Choose retrieval strategy based on query type
//! - **Performance-based Adaptation**: Adjust based on response times and quality
//! - **Context-aware Retrieval**: Consider user context and history
//! - **Fallback Mechanisms**: Handle strategy failures gracefully
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin adaptive_retrieval --features fastembed -- \
//!     --query "What are the latest developments in machine learning?" \
//!     --enable-adaptation \
//!     --performance-threshold 100 \
//!     --quality-threshold 0.8
//! ```

use cheungfun::prelude::*;
use cheungfun_query::advanced::{
    fusion::{DistributionBasedFusion, ReciprocalRankFusion},
    search_strategies::{HybridSearchStrategy, KeywordSearchStrategy, VectorSearchStrategy},
};
use clap::Parser;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

#[derive(Parser, Debug)]
#[command(name = "adaptive-retrieval")]
#[command(about = "Demonstrates adaptive retrieval strategies")]
struct Args {
    /// Query to search for
    #[arg(
        long,
        default_value = "What are the latest developments in machine learning?"
    )]
    query: String,

    /// Enable adaptive strategy selection
    #[arg(long)]
    enable_adaptation: bool,

    /// Performance threshold in milliseconds
    #[arg(long, default_value = "100")]
    performance_threshold: u64,

    /// Quality threshold (0.0-1.0)
    #[arg(long, default_value = "0.8")]
    quality_threshold: f32,

    /// Number of results to return
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Enable verbose output
    #[arg(long)]
    verbose: bool,
}

/// Query classification types for adaptive retrieval.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    /// Factual questions requiring precise information
    Factual,
    /// Conceptual questions requiring broad understanding
    Conceptual,
    /// Comparative questions requiring multiple perspectives
    Comparative,
    /// Temporal questions about recent developments
    Temporal,
    /// Complex multi-part questions
    Complex,
}

/// Retrieval strategy performance metrics.
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Average response time
    pub avg_response_time: Duration,
    /// Average quality score
    pub avg_quality_score: f32,
    /// Success rate (0.0-1.0)
    pub success_rate: f32,
    /// Number of queries processed
    pub query_count: usize,
}

impl StrategyMetrics {
    pub fn new() -> Self {
        Self {
            avg_response_time: Duration::from_millis(0),
            avg_quality_score: 0.0,
            success_rate: 1.0,
            query_count: 0,
        }
    }

    /// Update metrics with new measurement.
    pub fn update(&mut self, response_time: Duration, quality_score: f32, success: bool) {
        self.query_count += 1;

        // Update average response time
        let total_time = self.avg_response_time.as_millis() as f64 * (self.query_count - 1) as f64;
        let new_total = total_time + response_time.as_millis() as f64;
        self.avg_response_time =
            Duration::from_millis((new_total / self.query_count as f64) as u64);

        // Update average quality score
        let total_quality = self.avg_quality_score * (self.query_count - 1) as f32;
        self.avg_quality_score = (total_quality + quality_score) / self.query_count as f32;

        // Update success rate
        let total_successes = self.success_rate * (self.query_count - 1) as f32;
        let new_successes = if success {
            total_successes + 1.0
        } else {
            total_successes
        };
        self.success_rate = new_successes / self.query_count as f32;
    }
}

/// Adaptive retrieval engine that selects optimal strategies.
#[derive(Debug)]
pub struct AdaptiveRetriever {
    /// Available retrieval strategies
    strategies: HashMap<String, Box<dyn RetrievalStrategy>>,
    /// Performance metrics for each strategy
    metrics: HashMap<String, StrategyMetrics>,
    /// Performance threshold for strategy selection
    performance_threshold: Duration,
    /// Quality threshold for strategy selection
    quality_threshold: f32,
    /// Enable adaptive behavior
    adaptive_enabled: bool,
}

/// Trait for retrieval strategies used in adaptive retrieval.
pub trait RetrievalStrategy: std::fmt::Debug + Send + Sync {
    /// Execute retrieval with this strategy.
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<ScoredNode>;

    /// Get strategy name.
    fn name(&self) -> &str;

    /// Check if strategy is suitable for query type.
    fn is_suitable_for(&self, query_type: &QueryType) -> bool;
}

/// Vector-based retrieval strategy.
#[derive(Debug)]
pub struct VectorStrategy {
    name: String,
}

impl VectorStrategy {
    pub fn new() -> Self {
        Self {
            name: "vector".to_string(),
        }
    }
}

impl RetrievalStrategy for VectorStrategy {
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<ScoredNode> {
        // Simulate vector retrieval
        info!("Using vector strategy for query: {}", query);
        create_mock_results(query, top_k, 0.8)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_suitable_for(&self, query_type: &QueryType) -> bool {
        matches!(query_type, QueryType::Factual | QueryType::Conceptual)
    }
}

/// Keyword-based retrieval strategy.
#[derive(Debug)]
pub struct KeywordStrategy {
    name: String,
}

impl KeywordStrategy {
    pub fn new() -> Self {
        Self {
            name: "keyword".to_string(),
        }
    }
}

impl RetrievalStrategy for KeywordStrategy {
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<ScoredNode> {
        // Simulate keyword retrieval
        info!("Using keyword strategy for query: {}", query);
        create_mock_results(query, top_k, 0.7)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_suitable_for(&self, query_type: &QueryType) -> bool {
        matches!(query_type, QueryType::Factual | QueryType::Temporal)
    }
}

/// Hybrid retrieval strategy.
#[derive(Debug)]
pub struct HybridStrategy {
    name: String,
}

impl HybridStrategy {
    pub fn new() -> Self {
        Self {
            name: "hybrid".to_string(),
        }
    }
}

impl RetrievalStrategy for HybridStrategy {
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<ScoredNode> {
        // Simulate hybrid retrieval
        info!("Using hybrid strategy for query: {}", query);
        create_mock_results(query, top_k, 0.9)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_suitable_for(&self, query_type: &QueryType) -> bool {
        matches!(query_type, QueryType::Comparative | QueryType::Complex)
    }
}

impl AdaptiveRetriever {
    /// Create a new adaptive retriever.
    pub fn new(
        performance_threshold: Duration,
        quality_threshold: f32,
        adaptive_enabled: bool,
    ) -> Self {
        let mut strategies: HashMap<String, Box<dyn RetrievalStrategy>> = HashMap::new();
        strategies.insert("vector".to_string(), Box::new(VectorStrategy::new()));
        strategies.insert("keyword".to_string(), Box::new(KeywordStrategy::new()));
        strategies.insert("hybrid".to_string(), Box::new(HybridStrategy::new()));

        let mut metrics = HashMap::new();
        for strategy_name in strategies.keys() {
            metrics.insert(strategy_name.clone(), StrategyMetrics::new());
        }

        Self {
            strategies,
            metrics,
            performance_threshold,
            quality_threshold,
            adaptive_enabled,
        }
    }

    /// Classify query to determine optimal strategy.
    pub fn classify_query(&self, query: &str) -> QueryType {
        let query_lower = query.to_lowercase();

        // Simple rule-based classification
        if query_lower.contains("what is") || query_lower.contains("define") {
            QueryType::Factual
        } else if query_lower.contains("compare") || query_lower.contains("difference") {
            QueryType::Comparative
        } else if query_lower.contains("latest")
            || query_lower.contains("recent")
            || query_lower.contains("new")
        {
            QueryType::Temporal
        } else if query_lower.contains("how") && query_lower.contains("why") {
            QueryType::Complex
        } else if query_lower.contains("concept") || query_lower.contains("understand") {
            QueryType::Conceptual
        } else {
            QueryType::Factual // Default
        }
    }

    /// Select optimal strategy based on query type and performance metrics.
    pub fn select_strategy(&self, query_type: &QueryType) -> String {
        if !self.adaptive_enabled {
            return "hybrid".to_string(); // Default strategy
        }

        // Find suitable strategies for this query type
        let suitable_strategies: Vec<_> = self
            .strategies
            .iter()
            .filter(|(_, strategy)| strategy.is_suitable_for(query_type))
            .collect();

        if suitable_strategies.is_empty() {
            return "hybrid".to_string(); // Fallback
        }

        // Select best performing suitable strategy
        let mut best_strategy = suitable_strategies[0].0.clone();
        let mut best_score = 0.0f32;

        for (strategy_name, _) in suitable_strategies {
            if let Some(metrics) = self.metrics.get(strategy_name) {
                // Calculate composite score based on performance and quality
                let performance_score = if metrics.avg_response_time <= self.performance_threshold {
                    1.0
                } else {
                    self.performance_threshold.as_millis() as f32
                        / metrics.avg_response_time.as_millis() as f32
                };

                let quality_score = metrics.avg_quality_score;
                let success_score = metrics.success_rate;

                let composite_score =
                    (performance_score * 0.3 + quality_score * 0.4 + success_score * 0.3);

                if composite_score > best_score {
                    best_score = composite_score;
                    best_strategy = strategy_name.clone();
                }
            }
        }

        best_strategy
    }

    /// Execute adaptive retrieval.
    pub fn retrieve_adaptive(&mut self, query: &str, top_k: usize) -> Vec<ScoredNode> {
        let start_time = Instant::now();

        // Classify query
        let query_type = self.classify_query(query);
        info!("Query classified as: {:?}", query_type);

        // Select strategy
        let strategy_name = self.select_strategy(&query_type);
        info!("Selected strategy: {}", strategy_name);

        // Execute retrieval
        let results = if let Some(strategy) = self.strategies.get(&strategy_name) {
            strategy.retrieve(query, top_k)
        } else {
            warn!(
                "Strategy '{}' not found, using hybrid fallback",
                strategy_name
            );
            self.strategies
                .get("hybrid")
                .unwrap()
                .retrieve(query, top_k)
        };

        let response_time = start_time.elapsed();

        // Calculate quality score (simplified)
        let quality_score = if results.is_empty() {
            0.0
        } else {
            results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32
        };

        let success = !results.is_empty() && quality_score >= self.quality_threshold;

        // Update metrics
        if let Some(metrics) = self.metrics.get_mut(&strategy_name) {
            metrics.update(response_time, quality_score, success);
        }

        info!(
            "Retrieval completed in {:.2}ms with quality score {:.3}",
            response_time.as_millis(),
            quality_score
        );

        results
    }

    /// Get current performance metrics.
    pub fn get_metrics(&self) -> &HashMap<String, StrategyMetrics> {
        &self.metrics
    }
}

/// Create mock results for demonstration.
fn create_mock_results(query: &str, top_k: usize, base_score: f32) -> Vec<ScoredNode> {
    use cheungfun_core::ChunkInfo;
    use uuid::Uuid;

    let doc_id = Uuid::new_v4();
    let chunk_info = ChunkInfo::new(0, 100, 0);

    (0..top_k)
        .map(|i| {
            let score = base_score - (i as f32 * 0.1);
            let content = format!("Mock result {} for query: {}", i + 1, query);
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

    println!("🧠 Adaptive Retrieval Demo");
    println!("==========================");
    println!("Query: {}", args.query);
    println!("Adaptive Enabled: {}", args.enable_adaptation);
    println!("Performance Threshold: {}ms", args.performance_threshold);
    println!("Quality Threshold: {:.2}", args.quality_threshold);
    println!();

    // Create adaptive retriever
    let mut retriever = AdaptiveRetriever::new(
        Duration::from_millis(args.performance_threshold),
        args.quality_threshold,
        args.enable_adaptation,
    );

    // Test different query types
    let test_queries = vec![
        ("What is machine learning?", QueryType::Factual),
        (
            "Compare supervised and unsupervised learning",
            QueryType::Comparative,
        ),
        (
            "What are the latest developments in AI?",
            QueryType::Temporal,
        ),
        ("How and why do neural networks work?", QueryType::Complex),
        (
            "Understand the concept of deep learning",
            QueryType::Conceptual,
        ),
    ];

    println!("🔍 Testing Adaptive Retrieval:");
    println!("{}", "=".repeat(50));

    for (query, expected_type) in test_queries {
        println!("\nQuery: {}", query);

        // Classify query
        let classified_type = retriever.classify_query(query);
        println!("Expected Type: {:?}", expected_type);
        println!("Classified Type: {:?}", classified_type);

        // Perform adaptive retrieval
        let results = retriever.retrieve_adaptive(query, args.top_k);

        println!("Results: {} nodes", results.len());
        if args.verbose {
            for (i, result) in results.iter().enumerate() {
                println!(
                    "  {}. {} (Score: {:.3})",
                    i + 1,
                    result.node.content.chars().take(50).collect::<String>(),
                    result.score
                );
            }
        }

        println!("{}", "-".repeat(30));
    }

    // Display performance metrics
    println!("\n📊 Strategy Performance Metrics:");
    println!("{}", "=".repeat(50));

    for (strategy_name, metrics) in retriever.get_metrics() {
        println!("Strategy: {}", strategy_name);
        println!("  Queries processed: {}", metrics.query_count);
        println!(
            "  Avg response time: {:.2}ms",
            metrics.avg_response_time.as_millis()
        );
        println!("  Avg quality score: {:.3}", metrics.avg_quality_score);
        println!("  Success rate: {:.1}%", metrics.success_rate * 100.0);
        println!();
    }

    println!("✅ Adaptive retrieval demonstration completed!");
    println!(
        "🎯 This technique optimizes retrieval by selecting the best strategy for each query type."
    );
    println!("📈 Performance metrics help improve strategy selection over time.");

    Ok(())
}
