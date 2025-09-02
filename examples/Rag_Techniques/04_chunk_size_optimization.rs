//! Chunk Size Optimization Example
//!
//! This example demonstrates how to optimize chunk sizes for RAG systems by:
//! - Experimenting with different chunk sizes and overlaps
//! - Measuring retrieval quality and performance for each configuration
//! - Providing recommendations for optimal chunk size based on your data
//! - Analyzing the trade-offs between context preservation and retrieval efficiency
//! - Supporting both automated optimization and manual configuration
//!
//! ## Usage
//!
//! ```bash
//! # Run with default optimization
//! cargo run --bin chunk_size_optimization --features fastembed
//!
//! # Run with custom chunk size range
//! cargo run --bin chunk_size_optimization --features fastembed -- --min-chunk-size 200 --max-chunk-size 2000
//!
//! # Run with specific chunk size for testing
//! cargo run --bin chunk_size_optimization --features fastembed -- --chunk-size 800 --no-optimization
//! ```

use clap::Parser;

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{
    constants::*, get_climate_test_queries, setup_logging, ExampleError, ExampleResult,
    PerformanceMetrics, Timer,
};
use std::{path::PathBuf, sync::Arc};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, VectorStore},
    DistanceMetric,
};
use cheungfun_indexing::{
    loaders::DirectoryLoader,
    node_parser::{config::SentenceSplitterConfig, text::SentenceSplitter},
    pipeline::DefaultIndexingPipeline,
    transformers::MetadataExtractor,
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    engine::QueryEngine, generator::SiumaiGenerator, prelude::QueryResponse,
    retriever::VectorRetriever,
};
use siumai::prelude::*;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug)]
#[command(name = "chunk_size_optimization")]
#[command(about = "Chunk Size Optimization Example - Find optimal chunk sizes for your RAG system")]
struct Args {
    /// Path to the document to process
    #[arg(long, default_value = "data/Understanding_Climate_Change.pdf")]
    document_path: PathBuf,

    /// Embedding provider (fastembed, openai)
    #[arg(long, default_value = "fastembed")]
    embedding_provider: String,

    /// Specific chunk size to test (disables optimization)
    #[arg(long)]
    chunk_size: Option<usize>,

    /// Minimum chunk size for optimization
    #[arg(long, default_value_t = 200)]
    min_chunk_size: usize,

    /// Maximum chunk size for optimization
    #[arg(long, default_value_t = 1500)]
    max_chunk_size: usize,

    /// Chunk size step for optimization
    #[arg(long, default_value_t = 200)]
    chunk_size_step: usize,

    /// Chunk overlap percentage (0.0 to 0.5)
    #[arg(long, default_value_t = 0.1)]
    overlap_ratio: f32,

    /// Number of top results to retrieve
    #[arg(long, default_value_t = DEFAULT_TOP_K)]
    top_k: usize,

    /// Skip optimization and use specific chunk size
    #[arg(long)]
    no_optimization: bool,

    /// Run in interactive mode after optimization
    #[arg(long)]
    interactive: bool,
}

/// Metrics for evaluating chunk size performance
#[derive(Debug, Clone)]
struct ChunkSizeMetrics {
    pub chunk_size: usize,
    pub overlap_size: usize,
    pub total_chunks: usize,
    pub avg_chunk_length: f64,
    pub indexing_time: std::time::Duration,
    pub avg_query_time: std::time::Duration,
    pub avg_similarity_score: f32,
    pub avg_response_length: usize,
    pub retrieval_consistency: f32,
    pub overall_score: f32,
}

impl ChunkSizeMetrics {
    pub fn new(chunk_size: usize, overlap_size: usize) -> Self {
        Self {
            chunk_size,
            overlap_size,
            total_chunks: 0,
            avg_chunk_length: 0.0,
            indexing_time: std::time::Duration::ZERO,
            avg_query_time: std::time::Duration::ZERO,
            avg_similarity_score: 0.0,
            avg_response_length: 0,
            retrieval_consistency: 0.0,
            overall_score: 0.0,
        }
    }

    pub fn calculate_overall_score(&mut self) {
        // Weighted scoring formula
        let similarity_weight = 0.4;
        let consistency_weight = 0.3;
        let efficiency_weight = 0.2;
        let response_quality_weight = 0.1;

        // Normalize metrics (assuming reasonable ranges)
        let similarity_score = (self.avg_similarity_score * 100.0).min(100.0);
        let consistency_score = (self.retrieval_consistency * 100.0).min(100.0);

        // Efficiency score (inverse of query time, normalized)
        let efficiency_score = if self.avg_query_time.as_millis() > 0 {
            (5000.0 / self.avg_query_time.as_millis() as f32 * 100.0).min(100.0)
        } else {
            100.0
        };

        // Response quality score (based on reasonable response length)
        let response_quality_score = if self.avg_response_length > 0 {
            let ideal_length = 500.0; // Ideal response length
            let length_ratio = self.avg_response_length as f32 / ideal_length;
            if length_ratio <= 1.0 {
                length_ratio * 100.0
            } else {
                (2.0 - length_ratio).max(0.0) * 100.0
            }
        } else {
            0.0
        };

        self.overall_score = similarity_score * similarity_weight
            + consistency_score * consistency_weight
            + efficiency_score * efficiency_weight
            + response_quality_score * response_quality_weight;
    }

    pub fn print_summary(&self) {
        println!(
            "📊 Chunk Size: {} (overlap: {})",
            self.chunk_size, self.overlap_size
        );
        println!("   📦 Total Chunks: {}", self.total_chunks);
        println!("   📏 Avg Chunk Length: {:.1} chars", self.avg_chunk_length);
        println!(
            "   ⏱️  Indexing Time: {:.2}s",
            self.indexing_time.as_secs_f64()
        );
        println!(
            "   🔍 Avg Query Time: {:.0}ms",
            self.avg_query_time.as_millis()
        );
        println!("   🎯 Avg Similarity: {:.3}", self.avg_similarity_score);
        println!(
            "   📝 Avg Response Length: {} chars",
            self.avg_response_length
        );
        println!(
            "   🔄 Retrieval Consistency: {:.3}",
            self.retrieval_consistency
        );
        println!("   🏆 Overall Score: {:.1}/100", self.overall_score);
        println!();
    }
}

/// Optimization results and recommendations
#[derive(Debug)]
struct OptimizationResults {
    pub best_config: ChunkSizeMetrics,
    pub all_results: Vec<ChunkSizeMetrics>,
    pub recommendations: Vec<String>,
}

impl OptimizationResults {
    pub fn print_summary(&self) {
        println!("🏆 OPTIMIZATION RESULTS");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        println!("\n🥇 Best Configuration:");
        self.best_config.print_summary();

        println!("📈 All Results (sorted by overall score):");
        for (i, result) in self.all_results.iter().enumerate() {
            let medal = match i {
                0 => "🥇",
                1 => "🥈",
                2 => "🥉",
                _ => "  ",
            };
            println!(
                "{} Chunk Size: {} | Score: {:.1} | Similarity: {:.3} | Time: {:.0}ms",
                medal,
                result.chunk_size,
                result.overall_score,
                result.avg_similarity_score,
                result.avg_query_time.as_millis()
            );
        }

        println!("\n💡 Recommendations:");
        for (i, rec) in self.recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, rec);
        }

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    // Setup logging
    setup_logging();

    let args = Args::parse();

    println!("🚀 Starting Chunk Size Optimization Example...");

    // Print configuration
    print_config(&args);

    let mut metrics = PerformanceMetrics::new();

    // Create embedder (reused across all experiments)
    let embedder = create_embedder(&args.embedding_provider).await?;
    println!("✅ Embedder initialized: {}", args.embedding_provider);

    // Create LLM client (reused across all experiments)
    let llm_client = create_llm_client().await?;
    let generator = Arc::new(SiumaiGenerator::new(llm_client));
    println!("✅ LLM client initialized");

    if args.no_optimization || args.chunk_size.is_some() {
        // Run with specific chunk size
        let chunk_size = args.chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);
        let overlap_size = (chunk_size as f32 * args.overlap_ratio) as usize;

        println!(
            "\n🎯 Testing specific chunk size: {} (overlap: {})",
            chunk_size, overlap_size
        );

        let result = run_chunk_size_experiment(
            chunk_size,
            overlap_size,
            &args,
            embedder.clone(),
            generator.clone(),
            &mut metrics,
        )
        .await?;

        result.print_summary();

        if args.interactive {
            let query_engine = create_query_engine_for_chunk_size(
                chunk_size,
                overlap_size,
                &args,
                embedder,
                generator,
            )
            .await?;

            run_interactive_mode(&query_engine, &mut metrics).await?;
        }
    } else {
        // Run optimization
        let optimization_results =
            run_chunk_size_optimization(&args, embedder.clone(), generator.clone(), &mut metrics)
                .await?;

        optimization_results.print_summary();

        if args.interactive {
            println!("\n🎮 Starting interactive mode with best configuration...");
            let best_config = &optimization_results.best_config;
            let query_engine = create_query_engine_for_chunk_size(
                best_config.chunk_size,
                best_config.overlap_size,
                &args,
                embedder,
                generator,
            )
            .await?;

            run_interactive_mode(&query_engine, &mut metrics).await?;
        }
    }

    // Print final metrics
    metrics.print_summary();

    Ok(())
}

fn print_config(args: &Args) {
    println!("📏 Chunk Size Optimization Example");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📄 Document: {}", args.document_path.display());
    println!("🔤 Embedding Provider: {}", args.embedding_provider);

    if args.no_optimization || args.chunk_size.is_some() {
        let chunk_size = args.chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);
        println!("📏 Chunk Size: {} (fixed)", chunk_size);
    } else {
        println!(
            "📏 Chunk Size Range: {} - {} (step: {})",
            args.min_chunk_size, args.max_chunk_size, args.chunk_size_step
        );
    }

    println!("🔄 Overlap Ratio: {:.1}%", args.overlap_ratio * 100.0);
    println!("🔍 Top-K: {}", args.top_k);
    println!();
}

async fn create_embedder(provider: &str) -> ExampleResult<Arc<dyn Embedder>> {
    match provider {
        "fastembed" => {
            println!("🔤 Using FastEmbed for embeddings (local)");
            let embedder = FastEmbedder::new()
                .await
                .map_err(|e| ExampleError::Config(format!("FastEmbed error: {}", e)))?;
            Ok(Arc::new(embedder))
        }
        "openai" => {
            // Check for API key
            if let Ok(_api_key) = std::env::var("OPENAI_API_KEY") {
                // Note: This would require implementing OpenAI embedder
                // For now, fall back to FastEmbed
                println!("⚠️  OpenAI embedder not yet implemented, using FastEmbed");
                let embedder = FastEmbedder::new()
                    .await
                    .map_err(|e| ExampleError::Config(format!("FastEmbed error: {}", e)))?;
                Ok(Arc::new(embedder))
            } else {
                println!("🔤 No OpenAI API key found, using FastEmbed for embeddings (local)");
                let embedder = FastEmbedder::new()
                    .await
                    .map_err(|e| ExampleError::Config(format!("FastEmbed error: {}", e)))?;
                Ok(Arc::new(embedder))
            }
        }
        _ => Err(ExampleError::Config(format!(
            "Unsupported embedding provider: {}",
            provider
        ))),
    }
}

async fn create_llm_client() -> ExampleResult<Siumai> {
    // Try OpenAI first
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        if !api_key.is_empty() && api_key != "test" && api_key.starts_with("sk-") {
            println!("🤖 Using OpenAI for LLM generation (cloud)");
            return Siumai::builder()
                .openai()
                .api_key(&api_key)
                .model("gpt-4o-mini")
                .temperature(0.0)
                .max_tokens(4000)
                .build()
                .await
                .map_err(|e| ExampleError::Config(format!("Failed to initialize OpenAI: {}", e)));
        }
    }

    // Fallback to Ollama
    println!("🤖 No valid OpenAI API key found, using Ollama for LLM generation (local)");
    println!("💡 Make sure Ollama is running with: ollama serve");
    println!("💡 And pull a model with: ollama pull llama3.2");

    Siumai::builder()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .temperature(0.0)
        .build()
        .await
        .map_err(|e| ExampleError::Config(format!("Failed to initialize Ollama: {}. Make sure Ollama is running with 'ollama serve' and you have pulled a model with 'ollama pull llama3.2'", e)))
}

/// Run optimization across multiple chunk sizes
async fn run_chunk_size_optimization(
    args: &Args,
    embedder: Arc<dyn Embedder>,
    generator: Arc<SiumaiGenerator>,
    metrics: &mut PerformanceMetrics,
) -> ExampleResult<OptimizationResults> {
    println!("🔬 Starting chunk size optimization...");
    println!(
        "📊 Testing chunk sizes from {} to {} (step: {})",
        args.min_chunk_size, args.max_chunk_size, args.chunk_size_step
    );
    println!();

    let mut all_results = Vec::new();
    let chunk_sizes: Vec<usize> = (args.min_chunk_size..=args.max_chunk_size)
        .step_by(args.chunk_size_step)
        .collect();

    for (i, chunk_size) in chunk_sizes.iter().enumerate() {
        let overlap_size = (*chunk_size as f32 * args.overlap_ratio) as usize;

        println!(
            "🧪 Experiment {}/{}: Chunk Size {} (overlap: {})",
            i + 1,
            chunk_sizes.len(),
            chunk_size,
            overlap_size
        );

        let result = run_chunk_size_experiment(
            *chunk_size,
            overlap_size,
            args,
            embedder.clone(),
            generator.clone(),
            metrics,
        )
        .await?;

        result.print_summary();
        all_results.push(result);
    }

    // Sort results by overall score (descending)
    all_results.sort_by(|a, b| b.overall_score.partial_cmp(&a.overall_score).unwrap());

    let best_config = all_results[0].clone();
    let recommendations = generate_recommendations(&all_results, args);

    Ok(OptimizationResults {
        best_config,
        all_results,
        recommendations,
    })
}

/// Run a single chunk size experiment
async fn run_chunk_size_experiment(
    chunk_size: usize,
    overlap_size: usize,
    args: &Args,
    embedder: Arc<dyn Embedder>,
    generator: Arc<SiumaiGenerator>,
    metrics: &mut PerformanceMetrics,
) -> ExampleResult<ChunkSizeMetrics> {
    let mut chunk_metrics = ChunkSizeMetrics::new(chunk_size, overlap_size);

    // Create query engine with specific chunk size
    let timer = Timer::new("Indexing");
    let query_engine =
        create_query_engine_for_chunk_size(chunk_size, overlap_size, args, embedder, generator)
            .await?;
    chunk_metrics.indexing_time = timer.finish();

    // Run test queries
    let queries = get_climate_test_queries();
    let mut query_times = Vec::new();
    let mut similarity_scores = Vec::new();
    let mut response_lengths = Vec::new();
    let mut all_retrieved_chunks = Vec::new();

    for query in queries {
        let timer = Timer::new("Query");

        let response = query_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;

        let query_time = timer.finish();
        query_times.push(query_time);

        // Collect metrics
        let max_similarity = response
            .retrieved_nodes
            .iter()
            .map(|node| node.score)
            .fold(0.0f32, |a, b| a.max(b));
        similarity_scores.push(max_similarity);

        response_lengths.push(response.response.content.len());

        // Collect retrieved chunk IDs for consistency analysis
        let chunk_ids: Vec<String> = response
            .retrieved_nodes
            .iter()
            .map(|node| node.node.id.to_string())
            .collect();
        all_retrieved_chunks.push(chunk_ids);
    }

    // Calculate averages
    chunk_metrics.avg_query_time =
        query_times.iter().sum::<std::time::Duration>() / query_times.len() as u32;
    chunk_metrics.avg_similarity_score =
        similarity_scores.iter().sum::<f32>() / similarity_scores.len() as f32;
    chunk_metrics.avg_response_length =
        response_lengths.iter().sum::<usize>() / response_lengths.len();

    // Calculate retrieval consistency (how often the same chunks are retrieved for similar queries)
    chunk_metrics.retrieval_consistency = calculate_retrieval_consistency(&all_retrieved_chunks);

    // Calculate overall score
    chunk_metrics.calculate_overall_score();

    // Update global metrics
    metrics.record_indexing_time(chunk_metrics.indexing_time);
    for query_time in query_times {
        metrics.record_query(query_time);
    }

    Ok(chunk_metrics)
}

/// Create a query engine with specific chunk size configuration
async fn create_query_engine_for_chunk_size(
    chunk_size: usize,
    overlap_size: usize,
    args: &Args,
    embedder: Arc<dyn Embedder>,
    generator: Arc<SiumaiGenerator>,
) -> ExampleResult<QueryEngine> {
    // Create vector store
    let vector_store = Arc::new(InMemoryVectorStore::new(
        DEFAULT_EMBEDDING_DIM,
        DistanceMetric::Cosine,
    ));

    // Get the directory containing the document
    let default_path = PathBuf::from(".");
    let data_dir = args.document_path.parent().unwrap_or(&default_path);

    let loader = Arc::new(DirectoryLoader::new(data_dir)?);

    // Create text splitter with specific chunk size
    let splitter_config = SentenceSplitterConfig::new(chunk_size, overlap_size);
    let splitter = Arc::new(SentenceSplitter::new(splitter_config)?);
    let metadata_extractor = Arc::new(MetadataExtractor::new());

    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(loader)
        .with_document_processor(splitter) // Documents -> Nodes
        .with_node_processor(metadata_extractor) // Nodes -> Nodes
        .with_embedder(embedder.clone())
        .with_vector_store(vector_store.clone())
        .build()?;

    // Run indexing pipeline
    let (_nodes, _indexing_stats) = pipeline.run(None, None, true, true, None, true).await?;

    // Create query engine
    let retriever = Arc::new(VectorRetriever::new(vector_store, embedder));
    let query_engine = QueryEngine::new(retriever, generator);

    Ok(query_engine)
}

/// Calculate retrieval consistency across queries
fn calculate_retrieval_consistency(all_retrieved_chunks: &[Vec<String>]) -> f32 {
    if all_retrieved_chunks.len() < 2 {
        return 1.0;
    }

    let mut total_overlap = 0.0;
    let mut comparisons = 0;

    for i in 0..all_retrieved_chunks.len() {
        for j in (i + 1)..all_retrieved_chunks.len() {
            let chunks_a = &all_retrieved_chunks[i];
            let chunks_b = &all_retrieved_chunks[j];

            let intersection: std::collections::HashSet<_> = chunks_a
                .iter()
                .filter(|chunk| chunks_b.contains(chunk))
                .collect();

            let union_size = chunks_a.len() + chunks_b.len() - intersection.len();
            let jaccard_similarity = if union_size > 0 {
                intersection.len() as f32 / union_size as f32
            } else {
                1.0
            };

            total_overlap += jaccard_similarity;
            comparisons += 1;
        }
    }

    if comparisons > 0 {
        total_overlap / comparisons as f32
    } else {
        1.0
    }
}

/// Generate optimization recommendations based on results
fn generate_recommendations(results: &[ChunkSizeMetrics], args: &Args) -> Vec<String> {
    let mut recommendations = Vec::new();

    let best = &results[0];
    let worst = results.last().unwrap();

    // Performance recommendation
    if best.overall_score > 80.0 {
        recommendations.push(format!(
            "🎯 Excellent performance with chunk size {}! This configuration provides optimal balance.",
            best.chunk_size
        ));
    } else if best.overall_score > 60.0 {
        recommendations.push(format!(
            "✅ Good performance with chunk size {}. Consider fine-tuning overlap ratio for better results.",
            best.chunk_size
        ));
    } else {
        recommendations.push(format!(
            "⚠️  Best chunk size {} shows moderate performance. Consider expanding the search range.",
            best.chunk_size
        ));
    }

    // Speed vs Quality trade-off
    let fastest = results.iter().min_by_key(|r| r.avg_query_time).unwrap();
    let most_accurate = results
        .iter()
        .max_by(|a, b| {
            a.avg_similarity_score
                .partial_cmp(&b.avg_similarity_score)
                .unwrap()
        })
        .unwrap();

    if fastest.chunk_size != best.chunk_size {
        recommendations.push(format!(
            "⚡ For fastest queries, use chunk size {} ({:.0}ms avg)",
            fastest.chunk_size,
            fastest.avg_query_time.as_millis()
        ));
    }

    if most_accurate.chunk_size != best.chunk_size {
        recommendations.push(format!(
            "🎯 For highest accuracy, use chunk size {} (similarity: {:.3})",
            most_accurate.chunk_size, most_accurate.avg_similarity_score
        ));
    }

    // Chunk size insights
    if best.chunk_size <= args.min_chunk_size + args.chunk_size_step {
        recommendations.push(
            "📏 Consider testing smaller chunk sizes for potentially better granularity."
                .to_string(),
        );
    } else if best.chunk_size >= args.max_chunk_size - args.chunk_size_step {
        recommendations.push(
            "📏 Consider testing larger chunk sizes for potentially better context.".to_string(),
        );
    }

    // Consistency insights
    if best.retrieval_consistency < 0.3 {
        recommendations.push("🔄 Low retrieval consistency detected. Consider increasing chunk overlap or using semantic chunking.".to_string());
    } else if best.retrieval_consistency > 0.8 {
        recommendations.push(
            "🔄 High retrieval consistency - good chunk size for stable results.".to_string(),
        );
    }

    // Performance difference insights
    let score_diff = best.overall_score - worst.overall_score;
    if score_diff > 30.0 {
        recommendations.push("📊 Significant performance variation detected. Chunk size optimization is crucial for your data.".to_string());
    } else if score_diff < 10.0 {
        recommendations.push(
            "📊 Minimal performance variation. Your data is relatively insensitive to chunk size."
                .to_string(),
        );
    }

    recommendations
}

async fn run_interactive_mode(
    query_engine: &QueryEngine,
    metrics: &mut PerformanceMetrics,
) -> ExampleResult<()> {
    println!("🎯 Interactive Chunk Size Testing Mode");
    println!("Type your questions, or 'quit' to exit.");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    loop {
        print!("❓ Your question: ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let query = input.trim();

        if query.to_lowercase() == "quit" {
            break;
        }

        let timer = Timer::new("Query processing");

        match query_engine.query(query).await {
            Ok(response) => {
                let query_time = timer.finish();
                metrics.record_query(query_time);

                // Calculate similarity score
                let max_similarity = response
                    .retrieved_nodes
                    .iter()
                    .map(|node| node.score)
                    .fold(0.0f32, |a, b| a.max(b));

                println!("\n🔍 Query: {}", query);
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("📊 Similarity Score: {:.3}", max_similarity);
                println!("⏱️  Query Time: {:.0}ms", query_time.as_millis());
                println!(
                    "📝 Response Length: {} chars",
                    response.response.content.len()
                );
                println!();
                println!("📝 Response: {}", response.response.content);
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            }
            Err(e) => {
                println!("❌ Error processing query: {}", e);
            }
        }

        println!();
    }

    println!("👋 Goodbye!");
    Ok(())
}
