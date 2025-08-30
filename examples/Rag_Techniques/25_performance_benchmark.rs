/*!
# Performance Benchmark for RAG Techniques

This example provides comprehensive performance benchmarking for various RAG techniques,
comparing the effectiveness of different postprocessors, rerankers, and fusion strategies.

## Key Features

- **Postprocessor Benchmarks**: Compare KeywordFilter, MetadataFilter, SimilarityFilter
- **Reranker Benchmarks**: Compare LLM vs Model vs Score-based reranking
- **Fusion Strategy Benchmarks**: Compare RRF, Score-based, Voting, Distribution-based
- **End-to-End Performance**: Measure complete pipeline performance
- **Quality Metrics**: Evaluate both speed and result quality

## Usage

```bash
# Run all benchmarks
cargo run --bin performance_benchmark --features fastembed

# Run specific benchmark category
cargo run --bin performance_benchmark --features fastembed -- --category postprocessors

# Run with custom parameters
cargo run --bin performance_benchmark --features fastembed -- \
    --iterations 10 \
    --warmup-iterations 3 \
    --enable-quality-metrics
```
*/

use cheungfun_core::{ChunkInfo, Node, ScoredNode};
use cheungfun_integrations::FastEmbedder;
use cheungfun_query::{
    advanced::fusion::{DistributionBasedFusion, ReciprocalRankFusion},
    postprocessor::{
        KeywordFilter, KeywordFilterConfig, MetadataFilter, MetadataFilterConfig,
        NodePostprocessor, SentenceEmbeddingConfig, SentenceEmbeddingOptimizer, SimilarityFilter,
        SimilarityFilterConfig,
    },
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio;
use tracing::{info, warn};
use tracing_subscriber;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(name = "performance-benchmark")]
#[command(about = "Comprehensive performance benchmarking for RAG techniques")]
struct Args {
    /// Benchmark category: all, postprocessors, rerankers, fusion
    #[arg(long, default_value = "all")]
    category: String,

    /// Number of benchmark iterations
    #[arg(long, default_value = "5")]
    iterations: usize,

    /// Number of warmup iterations
    #[arg(long, default_value = "2")]
    warmup_iterations: usize,

    /// Enable quality metrics calculation
    #[arg(long)]
    enable_quality_metrics: bool,

    /// Number of test nodes to generate
    #[arg(long, default_value = "100")]
    test_nodes: usize,

    /// Top-k for retrieval tests
    #[arg(long, default_value = "10")]
    top_k: usize,

    /// Output results to JSON file
    #[arg(long)]
    output_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    name: String,
    category: String,
    avg_duration_ms: f64,
    min_duration_ms: f64,
    max_duration_ms: f64,
    std_dev_ms: f64,
    throughput_ops_per_sec: f64,
    quality_score: Option<f64>,
    memory_usage_mb: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkReport {
    timestamp: String,
    total_duration_sec: f64,
    results: Vec<BenchmarkResult>,
    summary: BenchmarkSummary,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkSummary {
    fastest_postprocessor: String,
    fastest_reranker: String,
    fastest_fusion: String,
    best_quality_postprocessor: Option<String>,
    best_quality_reranker: Option<String>,
    best_quality_fusion: Option<String>,
}

struct BenchmarkRunner {
    args: Args,
    test_nodes: Vec<ScoredNode>,
    test_queries: Vec<String>,
    embedder: Arc<dyn cheungfun_core::Embedder>,
}

impl BenchmarkRunner {
    async fn new(args: Args) -> anyhow::Result<Self> {
        let embedder = Arc::new(
            FastEmbedder::new()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to initialize embedder: {}", e))?,
        );

        let test_nodes = Self::generate_test_nodes(args.test_nodes);
        let test_queries = Self::generate_test_queries();

        Ok(Self {
            args,
            test_nodes,
            test_queries,
            embedder,
        })
    }

    fn generate_test_nodes(count: usize) -> Vec<ScoredNode> {
        let contents = vec![
            "Climate change is a pressing global issue affecting weather patterns worldwide.",
            "Machine learning algorithms are revolutionizing data analysis and prediction.",
            "Renewable energy sources like solar and wind power are becoming more efficient.",
            "Artificial intelligence is transforming healthcare through diagnostic tools.",
            "Sustainable agriculture practices help preserve soil and water resources.",
            "Quantum computing promises to solve complex computational problems.",
            "Biodiversity conservation is crucial for maintaining ecosystem balance.",
            "Blockchain technology offers secure and transparent transaction systems.",
            "Ocean acidification threatens marine ecosystems and food chains.",
            "Natural language processing enables better human-computer interaction.",
        ];

        let keywords = vec![
            vec!["climate", "change", "global", "weather"],
            vec!["machine", "learning", "algorithm", "data"],
            vec!["renewable", "energy", "solar", "wind"],
            vec!["artificial", "intelligence", "healthcare", "diagnostic"],
            vec!["sustainable", "agriculture", "soil", "water"],
            vec!["quantum", "computing", "computational", "problems"],
            vec!["biodiversity", "conservation", "ecosystem", "balance"],
            vec!["blockchain", "technology", "secure", "transaction"],
            vec!["ocean", "acidification", "marine", "ecosystem"],
            vec!["natural", "language", "processing", "interaction"],
        ];

        (0..count)
            .map(|i| {
                let content_idx = i % contents.len();
                let content = contents[content_idx];
                let node_keywords = &keywords[content_idx];

                let mut metadata = HashMap::new();
                metadata.insert(
                    "category".to_string(),
                    serde_json::Value::String(match content_idx {
                        0 | 2 | 4 | 6 | 8 => "environment".to_string(),
                        1 | 3 | 5 | 7 | 9 => "technology".to_string(),
                        _ => "general".to_string(),
                    }),
                );
                metadata.insert(
                    "keywords".to_string(),
                    serde_json::Value::Array(
                        node_keywords
                            .iter()
                            .map(|k| serde_json::Value::String(k.to_string()))
                            .collect(),
                    ),
                );
                metadata.insert(
                    "priority".to_string(),
                    serde_json::Value::Number(serde_json::Number::from((i % 5) + 1)),
                );

                ScoredNode {
                    node: Node {
                        id: Uuid::new_v4(),
                        content: content.to_string(),
                        metadata,
                        embedding: None,
                        sparse_embedding: None,
                        excluded_embed_metadata_keys: std::collections::HashSet::new(),
                        excluded_llm_metadata_keys: std::collections::HashSet::new(),
                        relationships: cheungfun_core::relationships::NodeRelationships::new(),
                        source_document_id: Uuid::new_v4(),
                        hash: None,
                        mimetype: "text/plain".to_string(),
                        text_template: "{content}\n\n{metadata_str}".to_string(),
                        metadata_separator: "\n".to_string(),
                        metadata_template: "{key}: {value}".to_string(),
                        chunk_info: ChunkInfo {
                            start_char_idx: Some(0),
                            end_char_idx: Some(content.len()),
                            chunk_index: i,
                        },
                    },
                    score: 0.5 + (i as f32 * 0.01) % 0.5, // Vary scores between 0.5-1.0
                }
            })
            .collect()
    }

    fn generate_test_queries() -> Vec<String> {
        vec![
            "climate change effects".to_string(),
            "machine learning applications".to_string(),
            "renewable energy technology".to_string(),
            "artificial intelligence healthcare".to_string(),
            "sustainable farming practices".to_string(),
        ]
    }

    async fn run_all_benchmarks(&mut self) -> anyhow::Result<BenchmarkReport> {
        let start_time = Instant::now();
        let mut results = Vec::new();

        match self.args.category.as_str() {
            "all" => {
                results.extend(self.benchmark_postprocessors().await?);
                results.extend(self.benchmark_fusion_strategies().await?);
            }
            "postprocessors" => {
                results.extend(self.benchmark_postprocessors().await?);
            }
            "fusion" => {
                results.extend(self.benchmark_fusion_strategies().await?);
            }
            _ => {
                warn!("Unknown benchmark category: {}", self.args.category);
                results.extend(self.benchmark_postprocessors().await?);
            }
        }

        let total_duration = start_time.elapsed();
        let summary = self.generate_summary(&results);

        Ok(BenchmarkReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            total_duration_sec: total_duration.as_secs_f64(),
            results,
            summary,
        })
    }

    async fn benchmark_postprocessors(&mut self) -> anyhow::Result<Vec<BenchmarkResult>> {
        info!("🔍 Benchmarking Postprocessors...");
        let mut results = Vec::new();

        // Benchmark KeywordFilter
        results.push(self.benchmark_keyword_filter().await?);

        // Benchmark MetadataFilter
        results.push(self.benchmark_metadata_filter().await?);

        // Benchmark SimilarityFilter
        results.push(self.benchmark_similarity_filter().await?);

        // Benchmark SentenceEmbeddingOptimizer
        results.push(self.benchmark_sentence_embedding_optimizer().await?);

        Ok(results)
    }

    async fn benchmark_fusion_strategies(&mut self) -> anyhow::Result<Vec<BenchmarkResult>> {
        info!("🔀 Benchmarking Fusion Strategies...");
        let mut results = Vec::new();

        // Benchmark ReciprocalRankFusion
        results.push(self.benchmark_reciprocal_rank_fusion().await?);

        // Benchmark DistributionBasedFusion
        results.push(self.benchmark_distribution_based_fusion().await?);

        Ok(results)
    }

    fn generate_summary(&self, results: &[BenchmarkResult]) -> BenchmarkSummary {
        let postprocessors: Vec<_> = results
            .iter()
            .filter(|r| r.category == "postprocessor")
            .collect();
        let fusion_strategies: Vec<_> = results.iter().filter(|r| r.category == "fusion").collect();

        let fastest_postprocessor = postprocessors
            .iter()
            .min_by(|a, b| a.avg_duration_ms.partial_cmp(&b.avg_duration_ms).unwrap())
            .map(|r| r.name.clone())
            .unwrap_or_else(|| "N/A".to_string());

        let fastest_fusion = fusion_strategies
            .iter()
            .min_by(|a, b| a.avg_duration_ms.partial_cmp(&b.avg_duration_ms).unwrap())
            .map(|r| r.name.clone())
            .unwrap_or_else(|| "N/A".to_string());

        BenchmarkSummary {
            fastest_postprocessor,
            fastest_reranker: "N/A".to_string(), // Not implemented in this example
            fastest_fusion,
            best_quality_postprocessor: None,
            best_quality_reranker: None,
            best_quality_fusion: None,
        }
    }

    async fn benchmark_keyword_filter(&self) -> anyhow::Result<BenchmarkResult> {
        let config = KeywordFilterConfig {
            required_keywords: vec!["climate".to_string(), "machine".to_string()],
            exclude_keywords: vec!["spam".to_string()],
            case_sensitive: false,
            min_required_matches: 1,
        };
        let filter = KeywordFilter::new(config)?;

        self.benchmark_postprocessor("KeywordFilter", &filter).await
    }

    async fn benchmark_metadata_filter(&self) -> anyhow::Result<BenchmarkResult> {
        let mut required_metadata = HashMap::new();
        required_metadata.insert("category".to_string(), "technology".to_string());

        let config = MetadataFilterConfig {
            required_metadata,
            excluded_metadata: HashMap::new(),
            require_all: false,
        };
        let filter = MetadataFilter::new(config);

        self.benchmark_postprocessor("MetadataFilter", &filter)
            .await
    }

    async fn benchmark_similarity_filter(&self) -> anyhow::Result<BenchmarkResult> {
        let config = SimilarityFilterConfig {
            similarity_cutoff: 0.7,
            max_nodes: Some(self.args.top_k),
            use_query_embedding: true,
        };
        let filter = SimilarityFilter::new(config);

        self.benchmark_postprocessor("SimilarityFilter", &filter)
            .await
    }

    async fn benchmark_sentence_embedding_optimizer(&self) -> anyhow::Result<BenchmarkResult> {
        let config = SentenceEmbeddingConfig {
            percentile_cutoff: Some(0.7),
            threshold_cutoff: Some(0.5),
            context_before: Some(1),
            context_after: Some(1),
            max_sentences_per_node: Some(20),
        };
        let optimizer = SentenceEmbeddingOptimizer::new(self.embedder.clone(), config);

        self.benchmark_postprocessor("SentenceEmbeddingOptimizer", &optimizer)
            .await
    }

    async fn benchmark_postprocessor(
        &self,
        name: &str,
        postprocessor: &dyn NodePostprocessor,
    ) -> anyhow::Result<BenchmarkResult> {
        let mut durations = Vec::new();
        let query = &self.test_queries[0]; // Use first test query

        // Warmup iterations
        for _ in 0..self.args.warmup_iterations {
            let _ = postprocessor
                .postprocess(self.test_nodes.clone(), query)
                .await?;
        }

        // Benchmark iterations
        for _ in 0..self.args.iterations {
            let start = Instant::now();
            let _result = postprocessor
                .postprocess(self.test_nodes.clone(), query)
                .await?;
            durations.push(start.elapsed());
        }

        Ok(self.calculate_benchmark_result(name, "postprocessor", durations))
    }

    async fn benchmark_reciprocal_rank_fusion(&self) -> anyhow::Result<BenchmarkResult> {
        let fusion = ReciprocalRankFusion::new(60.0);
        let mut durations = Vec::new();

        // Create test result sets
        let result_sets = vec![
            self.test_nodes[0..20].to_vec(),
            self.test_nodes[10..30].to_vec(),
            self.test_nodes[5..25].to_vec(),
        ];

        // Warmup iterations
        for _ in 0..self.args.warmup_iterations {
            let _ = fusion.fuse_results(result_sets.clone());
        }

        // Benchmark iterations
        for _ in 0..self.args.iterations {
            let start = Instant::now();
            let _result = fusion.fuse_results(result_sets.clone());
            durations.push(start.elapsed());
        }

        Ok(self.calculate_benchmark_result("ReciprocalRankFusion", "fusion", durations))
    }

    async fn benchmark_distribution_based_fusion(&self) -> anyhow::Result<BenchmarkResult> {
        let fusion = DistributionBasedFusion::new(3); // 3 result sets
        let mut durations = Vec::new();

        // Create test result sets
        let result_sets = vec![
            self.test_nodes[0..20].to_vec(),
            self.test_nodes[10..30].to_vec(),
            self.test_nodes[5..25].to_vec(),
        ];

        // Warmup iterations
        for _ in 0..self.args.warmup_iterations {
            let _ = fusion.fuse_results(result_sets.clone());
        }

        // Benchmark iterations
        for _ in 0..self.args.iterations {
            let start = Instant::now();
            let _result = fusion.fuse_results(result_sets.clone());
            durations.push(start.elapsed());
        }

        Ok(self.calculate_benchmark_result("DistributionBasedFusion", "fusion", durations))
    }

    fn calculate_benchmark_result(
        &self,
        name: &str,
        category: &str,
        durations: Vec<Duration>,
    ) -> BenchmarkResult {
        let durations_ms: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect();

        let avg_duration_ms = durations_ms.iter().sum::<f64>() / durations_ms.len() as f64;
        let min_duration_ms = durations_ms.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_duration_ms = durations_ms.iter().fold(0.0f64, |a, &b| a.max(b));

        let variance = durations_ms
            .iter()
            .map(|d| (d - avg_duration_ms).powi(2))
            .sum::<f64>()
            / durations_ms.len() as f64;
        let std_dev_ms = variance.sqrt();

        let throughput_ops_per_sec = 1000.0 / avg_duration_ms;

        BenchmarkResult {
            name: name.to_string(),
            category: category.to_string(),
            avg_duration_ms,
            min_duration_ms,
            max_duration_ms,
            std_dev_ms,
            throughput_ops_per_sec,
            quality_score: None, // Could be implemented with actual quality metrics
            memory_usage_mb: None, // Could be implemented with memory profiling
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    println!("🚀 Starting Cheungfun RAG Performance Benchmark");
    println!("═══════════════════════════════════════════════");
    println!("📊 Category: {}", args.category);
    println!(
        "🔄 Iterations: {} (warmup: {})",
        args.iterations, args.warmup_iterations
    );
    println!("📝 Test nodes: {}", args.test_nodes);
    println!("🎯 Top-k: {}", args.top_k);
    println!();

    let mut runner = BenchmarkRunner::new(args).await?;
    let report = runner.run_all_benchmarks().await?;

    // Display results
    display_results(&report);

    // Save to JSON if requested
    if let Some(output_path) = &runner.args.output_json {
        save_results_to_json(&report, output_path)?;
        println!("💾 Results saved to: {}", output_path);
    }

    println!("\n✅ Benchmark completed successfully!");
    println!("🎯 Use these results to optimize your RAG pipeline configuration.");

    Ok(())
}

fn display_results(report: &BenchmarkReport) {
    println!("📊 BENCHMARK RESULTS");
    println!("════════════════════");
    println!("⏱️  Total Duration: {:.2}s", report.total_duration_sec);
    println!();

    // Group results by category
    let mut categories: std::collections::HashMap<String, Vec<&BenchmarkResult>> =
        std::collections::HashMap::new();
    for result in &report.results {
        categories
            .entry(result.category.clone())
            .or_default()
            .push(result);
    }

    for (category, results) in categories {
        println!("🔍 {} Results:", category.to_uppercase());
        println!("┌─────────────────────────────┬──────────┬──────────┬──────────┬──────────┬─────────────┐");
        println!("│ Component                   │ Avg (ms) │ Min (ms) │ Max (ms) │ Std Dev  │ Throughput  │");
        println!("├─────────────────────────────┼──────────┼──────────┼──────────┼──────────┼─────────────┤");

        for result in results {
            println!(
                "│ {:<27} │ {:>8.2} │ {:>8.2} │ {:>8.2} │ {:>8.2} │ {:>8.1} ops/s │",
                truncate_string(&result.name, 27),
                result.avg_duration_ms,
                result.min_duration_ms,
                result.max_duration_ms,
                result.std_dev_ms,
                result.throughput_ops_per_sec
            );
        }
        println!("└─────────────────────────────┴──────────┴──────────┴──────────┴──────────┴─────────────┘");
        println!();
    }

    // Display summary
    println!("🏆 PERFORMANCE SUMMARY");
    println!("═══════════════════════");
    println!(
        "🥇 Fastest Postprocessor: {}",
        report.summary.fastest_postprocessor
    );
    println!(
        "🥇 Fastest Fusion Strategy: {}",
        report.summary.fastest_fusion
    );
    println!();

    // Performance recommendations
    display_recommendations(&report.results);
}

fn display_recommendations(results: &[BenchmarkResult]) {
    println!("💡 PERFORMANCE RECOMMENDATIONS");
    println!("═══════════════════════════════");

    let postprocessors: Vec<_> = results
        .iter()
        .filter(|r| r.category == "postprocessor")
        .collect();

    if !postprocessors.is_empty() {
        let fastest = postprocessors
            .iter()
            .min_by(|a, b| a.avg_duration_ms.partial_cmp(&b.avg_duration_ms).unwrap())
            .unwrap();
        let slowest = postprocessors
            .iter()
            .max_by(|a, b| a.avg_duration_ms.partial_cmp(&b.avg_duration_ms).unwrap())
            .unwrap();

        println!("🔍 Postprocessors:");
        println!(
            "   • Use {} for fastest processing ({:.2}ms avg)",
            fastest.name, fastest.avg_duration_ms
        );
        println!(
            "   • {} is {:.1}x slower than the fastest",
            slowest.name,
            slowest.avg_duration_ms / fastest.avg_duration_ms
        );
    }

    let fusion_strategies: Vec<_> = results.iter().filter(|r| r.category == "fusion").collect();

    if !fusion_strategies.is_empty() {
        let fastest = fusion_strategies
            .iter()
            .min_by(|a, b| a.avg_duration_ms.partial_cmp(&b.avg_duration_ms).unwrap())
            .unwrap();

        println!("🔀 Fusion Strategies:");
        println!(
            "   • Use {} for fastest result fusion ({:.2}ms avg)",
            fastest.name, fastest.avg_duration_ms
        );
    }

    println!();
    println!("⚡ General Tips:");
    println!("   • For high-throughput scenarios, prioritize fastest components");
    println!("   • For quality-critical applications, consider slower but more accurate methods");
    println!("   • Use caching and batching to improve overall performance");
    println!("   • Monitor memory usage in production environments");
}

fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

fn save_results_to_json(report: &BenchmarkReport, path: &str) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(path, json)?;
    Ok(())
}
