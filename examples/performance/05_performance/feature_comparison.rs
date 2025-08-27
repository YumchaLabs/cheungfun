//! Feature Performance Comparison
//!
//! This example demonstrates the performance impact of different feature combinations
//! in the Cheungfun framework. It helps users understand which features to enable
//! for their specific use cases.
//!
//! ## Required Features
//!
//! This example can run with different feature combinations to show performance differences:
//! - `benchmarks` - Required for benchmark infrastructure
//! - `candle` - For embedding generation
//! - `simd` - SIMD acceleration
//! - `optimized-memory` - Memory optimizations
//! - `hnsw` - HNSW approximate nearest neighbor
//!
//! ## Usage
//!
//! ```bash
//! # Basic benchmark (CPU only)
//! cargo run --bin feature_comparison --features "benchmarks,candle"
//!
//! # With SIMD acceleration
//! cargo run --bin feature_comparison --features "benchmarks,candle,simd"
//!
//! # With all performance features
//! cargo run --bin feature_comparison --features "benchmarks,performance,candle"
//!
//! # With GPU acceleration
//! cargo run --bin feature_comparison --features "benchmarks,candle-cuda"
//! ```

use cheungfun_core::{traits::Embedder, Result};
use cheungfun_integrations::{CandleEmbedder, InMemoryVectorStore};
use std::time::{Duration, Instant};
use tracing::{info, Level};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    info!("🚀 Feature Performance Comparison");
    info!("=================================");

    // Display enabled features
    display_enabled_features();

    // Prepare test data
    let test_texts = generate_test_data();
    info!("📄 Generated {} test texts", test_texts.len());

    // Run embedding benchmarks
    let embedding_results = benchmark_embeddings(&test_texts).await?;
    display_embedding_results(&embedding_results);

    // Run vector operations benchmarks
    let vector_results = benchmark_vector_operations(&test_texts).await?;
    display_vector_results(&vector_results);

    // Provide recommendations
    provide_recommendations(&embedding_results, &vector_results);

    info!("🎉 Performance comparison completed!");

    Ok(())
}

/// Display which features are currently enabled
fn display_enabled_features() {
    info!("🔧 Enabled Features");
    info!("==================");

    // Check compile-time features
    #[cfg(feature = "simd")]
    info!("✅ SIMD acceleration enabled");
    #[cfg(not(feature = "simd"))]
    info!("❌ SIMD acceleration disabled");

    #[cfg(feature = "optimized-memory")]
    info!("✅ Optimized memory management enabled");
    #[cfg(not(feature = "optimized-memory"))]
    info!("❌ Optimized memory management disabled");

    #[cfg(feature = "hnsw")]
    info!("✅ HNSW approximate search enabled");
    #[cfg(not(feature = "hnsw"))]
    info!("❌ HNSW approximate search disabled");

    #[cfg(feature = "candle-cuda")]
    info!("✅ CUDA GPU acceleration enabled");
    #[cfg(not(feature = "candle-cuda"))]
    info!("❌ CUDA GPU acceleration disabled");

    #[cfg(feature = "candle-metal")]
    info!("✅ Metal GPU acceleration enabled");
    #[cfg(not(feature = "candle-metal"))]
    info!("❌ Metal GPU acceleration disabled");

    info!("");
}

/// Generate test data for benchmarking
fn generate_test_data() -> Vec<String> {
    vec![
        "Machine learning algorithms learn patterns from data automatically.",
        "Deep neural networks consist of multiple layers of interconnected nodes.",
        "Vector databases efficiently store and query high-dimensional embeddings.",
        "Retrieval-augmented generation combines search with language models.",
        "Transformer architectures use attention mechanisms for sequence processing.",
        "SIMD instructions enable parallel processing of multiple data elements.",
        "GPU acceleration significantly improves computational performance.",
        "Memory optimization reduces cache misses and improves throughput.",
        "Approximate nearest neighbor search trades accuracy for speed.",
        "Feature engineering extracts meaningful patterns from raw data.",
        "Distributed computing scales processing across multiple machines.",
        "Quantization reduces model size while maintaining accuracy.",
        "Batch processing improves efficiency for large-scale operations.",
        "Caching strategies reduce redundant computations and I/O operations.",
        "Parallel algorithms exploit multiple cores for faster execution.",
    ]
}

/// Benchmark embedding generation performance
async fn benchmark_embeddings(texts: &[String]) -> Result<EmbeddingBenchmarkResults> {
    info!("📊 Benchmarking embedding generation...");

    // Initialize embedder with appropriate device
    let device = if cfg!(feature = "candle-cuda") {
        Some("cuda:0".to_string())
    } else if cfg!(feature = "candle-metal") {
        Some("metal".to_string())
    } else {
        None
    };

    let embedder = if let Some(device) = device {
        CandleEmbedder::from_pretrained_with_device(
            "sentence-transformers/all-MiniLM-L6-v2",
            Some(device),
        )
        .await?
    } else {
        CandleEmbedder::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").await?
    };

    // Warm up
    let _ = embedder.embed(&texts[0]).await?;

    // Benchmark single embedding
    let start = Instant::now();
    let _ = embedder.embed(&texts[0]).await?;
    let single_time = start.elapsed();

    // Benchmark batch embeddings
    let start = Instant::now();
    for text in texts {
        let _ = embedder.embed(text).await?;
    }
    let batch_time = start.elapsed();

    Ok(EmbeddingBenchmarkResults {
        single_embedding_time: single_time,
        batch_time,
        texts_count: texts.len(),
        embeddings_per_second: texts.len() as f64 / batch_time.as_secs_f64(),
    })
}

/// Benchmark vector operations performance
async fn benchmark_vector_operations(texts: &[String]) -> Result<VectorBenchmarkResults> {
    info!("📊 Benchmarking vector operations...");

    // Initialize components
    let embedder =
        CandleEmbedder::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").await?;
    let vector_store = InMemoryVectorStore::new(
        embedder.dimension(),
        cheungfun_core::traits::DistanceMetric::Cosine,
    );

    // Generate embeddings
    let mut nodes = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        let embedding = embedder.embed(text).await?;
        let mut node = cheungfun_core::Node::new(format!("doc_{}", i), text.clone());
        node.embedding = Some(embedding);
        nodes.push(node);
    }

    // Benchmark insertion
    let start = Instant::now();
    let _ = vector_store.add(nodes.clone()).await?;
    let insertion_time = start.elapsed();

    // Benchmark search
    let query_embedding = embedder.embed(&texts[0]).await?;
    let start = Instant::now();
    let _ = vector_store.search(&query_embedding, 5).await?;
    let search_time = start.elapsed();

    Ok(VectorBenchmarkResults {
        insertion_time,
        search_time,
        nodes_count: nodes.len(),
        insertions_per_second: nodes.len() as f64 / insertion_time.as_secs_f64(),
    })
}

/// Display embedding benchmark results
fn display_embedding_results(results: &EmbeddingBenchmarkResults) {
    info!("");
    info!("📊 Embedding Performance Results");
    info!("================================");
    info!("Single embedding: {:?}", results.single_embedding_time);
    info!("Batch time: {:?}", results.batch_time);
    info!(
        "Average per embedding: {:?}",
        results.batch_time / results.texts_count as u32
    );
    info!(
        "Embeddings per second: {:.2}",
        results.embeddings_per_second
    );
}

/// Display vector operations benchmark results
fn display_vector_results(results: &VectorBenchmarkResults) {
    info!("");
    info!("📊 Vector Operations Performance Results");
    info!("========================================");
    info!("Insertion time: {:?}", results.insertion_time);
    info!("Search time: {:?}", results.search_time);
    info!(
        "Insertions per second: {:.2}",
        results.insertions_per_second
    );
}

/// Provide performance recommendations based on results
fn provide_recommendations(
    embedding_results: &EmbeddingBenchmarkResults,
    vector_results: &VectorBenchmarkResults,
) {
    info!("");
    info!("💡 Performance Recommendations");
    info!("==============================");

    if embedding_results.embeddings_per_second < 10.0 {
        info!("🐌 Embedding generation is slow. Consider:");
        info!("   - Enabling GPU acceleration (candle-cuda/candle-metal)");
        info!("   - Using SIMD optimizations");
        info!("   - Batching multiple texts together");
    } else {
        info!("✅ Embedding generation performance is good");
    }

    if vector_results.insertions_per_second < 100.0 {
        info!("🐌 Vector operations are slow. Consider:");
        info!("   - Enabling optimized-memory feature");
        info!("   - Using HNSW for approximate search");
        info!("   - Switching to a specialized vector database");
    } else {
        info!("✅ Vector operations performance is good");
    }

    info!("");
    info!("🚀 Feature Recommendations:");
    info!("   - For CPU workloads: enable 'performance' bundle");
    info!("   - For GPU workloads: enable 'candle-cuda' or 'candle-metal'");
    info!("   - For production: enable 'production' bundle");
    info!("   - For maximum performance: enable 'full' bundle");
}

/// Embedding benchmark results
#[derive(Debug)]
struct EmbeddingBenchmarkResults {
    single_embedding_time: Duration,
    batch_time: Duration,
    texts_count: usize,
    embeddings_per_second: f64,
}

/// Vector operations benchmark results
#[derive(Debug)]
struct VectorBenchmarkResults {
    insertion_time: Duration,
    search_time: Duration,
    nodes_count: usize,
    insertions_per_second: f64,
}
