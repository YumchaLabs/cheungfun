//! Comprehensive performance benchmark for Cheungfun vector stores
//!
//! This benchmark demonstrates performance improvements through:
//! - SIMD-accelerated vector operations (with `simd` feature)
//! - Optimized data structures (with `optimized-memory` feature)
//! - Parallel processing capabilities
//! - Memory usage analysis
//!
//! ## Usage Examples:
//!
//! Basic benchmark:
//! ```bash
//! cargo run --bin performance_benchmark
//! ```
//!
//! With SIMD acceleration:
//! ```bash
//! cargo run --bin performance_benchmark --features simd
//! ```
//!
//! With all performance features:
//! ```bash
//! cargo run --bin performance_benchmark --features performance
//! ```
//!
//! ## Feature Flags:
//! - `simd`: Enable SIMD vector operations for faster computation
//! - `optimized-memory`: Enable memory-optimized vector stores
//! - `performance`: Enable all performance optimizations (simd + optimized-memory + hnsw)

use cheungfun_core::{
    traits::VectorStore,
    types::{ChunkInfo, Node, Query},
    DistanceMetric, Result,
};
use cheungfun_integrations::vector_stores::{
    memory::InMemoryVectorStore, memory_optimized::OptimizedInMemoryVectorStore,
};
use std::time::Instant;
use uuid::Uuid;

// Feature-dependent imports
#[cfg(feature = "simd")]
use cheungfun_integrations::simd::SimdVectorOps;

#[cfg(feature = "hnsw")]
use cheungfun_integrations::vector_stores::hnsw::HnswVectorStore;

/// Benchmark configuration
#[derive(Debug, Clone)]
struct BenchmarkConfig {
    /// Number of vectors to generate for testing
    num_vectors: usize,
    /// Vector dimension
    dimension: usize,
    /// Number of search queries to perform
    num_queries: usize,
    /// Top-k results to retrieve
    top_k: usize,
    /// Distance metric to use
    distance_metric: DistanceMetric,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_vectors: 10_000,
            dimension: 384, // Common embedding dimension
            num_queries: 100,
            top_k: 10,
            distance_metric: DistanceMetric::Cosine,
        }
    }
}

/// Benchmark results for a single test
#[derive(Debug, Clone)]
struct BenchmarkResult {
    /// Test name
    name: String,
    /// Time taken for indexing (ms)
    indexing_time_ms: u64,
    /// Time taken for all queries (ms)
    total_query_time_ms: u64,
    /// Average time per query (ms)
    avg_query_time_ms: f64,
    /// Queries per second
    qps: f64,
    /// Memory usage estimate (bytes)
    memory_usage_bytes: usize,
    /// Additional metrics
    additional_metrics: std::collections::HashMap<String, String>,
}

impl BenchmarkResult {
    fn new(name: String) -> Self {
        Self {
            name,
            indexing_time_ms: 0,
            total_query_time_ms: 0,
            avg_query_time_ms: 0.0,
            qps: 0.0,
            memory_usage_bytes: 0,
            additional_metrics: std::collections::HashMap::new(),
        }
    }

    fn calculate_derived_metrics(&mut self, num_queries: usize) {
        self.avg_query_time_ms = self.total_query_time_ms as f64 / num_queries as f64;
        self.qps = if self.total_query_time_ms > 0 {
            (num_queries as f64 * 1000.0) / self.total_query_time_ms as f64
        } else {
            0.0
        };
    }

    fn print_summary(&self) {
        println!("\n=== {} ===", self.name);
        println!("Indexing time: {} ms", self.indexing_time_ms);
        println!("Total query time: {} ms", self.total_query_time_ms);
        println!("Average query time: {:.2} ms", self.avg_query_time_ms);
        println!("Queries per second: {:.2}", self.qps);
        println!(
            "Memory usage: {:.2} MB",
            self.memory_usage_bytes as f64 / 1_048_576.0
        );

        if !self.additional_metrics.is_empty() {
            println!("Additional metrics:");
            for (key, value) in &self.additional_metrics {
                println!("  {}: {}", key, value);
            }
        }
    }
}

/// Generate random vectors for testing
fn generate_test_vectors(num_vectors: usize, dimension: usize) -> Vec<Node> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..num_vectors)
        .map(|i| {
            let embedding: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();

            let mut metadata = std::collections::HashMap::new();
            metadata.insert("index".to_string(), serde_json::Value::Number(i.into()));
            metadata.insert(
                "category".to_string(),
                serde_json::Value::String(format!("cat_{}", i % 10)),
            );

            {
                let source_doc_id = Uuid::new_v4();
                let chunk_info = ChunkInfo::new(i * 100, (i + 1) * 100, i);
                let mut node = Node::new(format!("Document {}", i), source_doc_id, chunk_info);
                node.embedding = Some(embedding);
                node.metadata = metadata;
                node
            }
        })
        .collect()
}

/// Generate test queries
fn generate_test_queries(num_queries: usize, dimension: usize, top_k: usize) -> Vec<Query> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..num_queries)
        .map(|i| {
            let embedding: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();

            Query::new(format!("Query {}", i))
                .with_embedding(embedding)
                .with_top_k(top_k)
        })
        .collect()
}

/// Benchmark the basic in-memory vector store
async fn benchmark_basic_memory_store(
    config: &BenchmarkConfig,
    test_vectors: &[Node],
    test_queries: &[Query],
) -> Result<BenchmarkResult> {
    let mut result = BenchmarkResult::new("Basic InMemoryVectorStore".to_string());

    // Create store
    let store = InMemoryVectorStore::new(config.dimension, config.distance_metric.clone());

    // Benchmark indexing
    let start_time = Instant::now();
    store.add(test_vectors.to_vec()).await?;
    result.indexing_time_ms = start_time.elapsed().as_millis() as u64;

    // Benchmark queries
    let start_time = Instant::now();
    for query in test_queries {
        let _results = store.search(query).await?;
    }
    result.total_query_time_ms = start_time.elapsed().as_millis() as u64;

    result.calculate_derived_metrics(test_queries.len());

    // Estimate memory usage
    result.memory_usage_bytes = test_vectors.len() * config.dimension * 4 + // f32 vectors
                               test_vectors.len() * 200; // estimated node overhead

    Ok(result)
}

/// Benchmark the optimized in-memory vector store
async fn benchmark_optimized_memory_store(
    config: &BenchmarkConfig,
    test_vectors: &[Node],
    test_queries: &[Query],
) -> Result<BenchmarkResult> {
    let mut result = BenchmarkResult::new("Optimized InMemoryVectorStore".to_string());

    // Create store
    let store = OptimizedInMemoryVectorStore::new(config.dimension, config.distance_metric.clone());

    // Add SIMD capability info
    result.additional_metrics.insert(
        "SIMD Available".to_string(),
        store.is_simd_available().to_string(),
    );
    result.additional_metrics.insert(
        "SIMD Capabilities".to_string(),
        store.get_simd_capabilities(),
    );

    // Benchmark indexing
    let start_time = Instant::now();
    store.add(test_vectors.to_vec()).await?;
    result.indexing_time_ms = start_time.elapsed().as_millis() as u64;

    // Benchmark queries
    let start_time = Instant::now();
    for query in test_queries {
        let _results = store.search(query).await?;
    }
    result.total_query_time_ms = start_time.elapsed().as_millis() as u64;

    result.calculate_derived_metrics(test_queries.len());

    // Get performance statistics
    let stats = store.get_stats();
    result.additional_metrics.insert(
        "SIMD Operations".to_string(),
        stats.simd_operations.to_string(),
    );
    result.additional_metrics.insert(
        "Parallel Operations".to_string(),
        stats.parallel_operations.to_string(),
    );

    // Estimate memory usage (optimized layout)
    result.memory_usage_bytes = test_vectors.len() * config.dimension * 4 + // f32 vectors
                               test_vectors.len() * 4 + // f32 norms
                               test_vectors.len() * 200; // estimated node overhead

    Ok(result)
}

/// Benchmark HNSW vector store (if feature is enabled)
#[cfg(feature = "hnsw")]
async fn benchmark_hnsw_store(
    config: &BenchmarkConfig,
    test_vectors: &[Node],
    test_queries: &[Query],
) -> Result<BenchmarkResult> {
    let mut result = BenchmarkResult::new("HNSW VectorStore".to_string());

    // Create HNSW store with default parameters
    let store = HnswVectorStore::new(config.dimension, config.distance_metric.clone());

    // Benchmark indexing
    let start_time = Instant::now();
    store.add(test_vectors.to_vec()).await?;
    result.indexing_time_ms = start_time.elapsed().as_millis() as u64;

    // Benchmark queries
    let start_time = Instant::now();
    for query in test_queries {
        let _results = store.search(query).await?;
    }
    result.total_query_time_ms = start_time.elapsed().as_millis() as u64;

    result.calculate_derived_metrics(test_queries.len());

    // Get HNSW-specific metrics
    let hnsw_stats = store.get_stats();
    result
        .additional_metrics
        .insert("HNSW Layers".to_string(), hnsw_stats.num_layers.to_string());
    result.additional_metrics.insert(
        "HNSW Connections".to_string(),
        hnsw_stats.total_connections.to_string(),
    );
    result.additional_metrics.insert(
        "Search Efficiency".to_string(),
        format!("{:.2}%", hnsw_stats.search_efficiency * 100.0),
    );

    // Estimate memory usage (HNSW has overhead for graph structure)
    result.memory_usage_bytes = test_vectors.len() * config.dimension * 4 + // f32 vectors
                               test_vectors.len() * 200 + // estimated node overhead
                               hnsw_stats.total_connections * 8; // graph connections

    Ok(result)
}

/// Test SIMD operations directly (if feature is enabled)
#[cfg(feature = "simd")]
async fn test_simd_operations(config: &BenchmarkConfig) -> Result<()> {
    println!("\n🔥 SIMD Operations Test");
    println!("======================");

    let simd_ops = SimdVectorOps::new();
    println!("SIMD capabilities: {}", simd_ops.get_capabilities());

    if !simd_ops.is_simd_available() {
        println!("⚠️  SIMD not available on this system");
        return Ok(());
    }

    // Generate test vectors for SIMD operations
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let num_pairs = 1000;

    let vector_a: Vec<f32> = (0..config.dimension)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    let test_vectors: Vec<Vec<f32>> = (0..num_pairs)
        .map(|_| {
            (0..config.dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect();

    // Test one-to-many cosine similarity
    let vector_refs: Vec<&[f32]> = test_vectors.iter().map(|v| v.as_slice()).collect();

    let start = Instant::now();
    let _results = simd_ops.one_to_many_cosine_similarity_f32(&vector_a, &vector_refs)?;
    let simd_time = start.elapsed();

    println!("SIMD one-to-many operations:");
    println!("  Operations: {}", num_pairs);
    println!("  Time: {:?}", simd_time);
    println!(
        "  Ops/sec: {:.2}",
        num_pairs as f64 / simd_time.as_secs_f64()
    );

    Ok(())
}

/// Compare performance between different implementations
async fn run_performance_comparison() -> Result<()> {
    println!("🚀 Cheungfun Performance Benchmark");
    println!("===================================");

    // Display enabled features
    println!("Enabled features:");
    #[cfg(feature = "simd")]
    println!("  ✅ SIMD acceleration");
    #[cfg(not(feature = "simd"))]
    println!("  ❌ SIMD acceleration (use --features simd)");

    #[cfg(feature = "optimized-memory")]
    println!("  ✅ Optimized memory stores");
    #[cfg(not(feature = "optimized-memory"))]
    println!("  ❌ Optimized memory stores (use --features optimized-memory)");

    #[cfg(feature = "hnsw")]
    println!("  ✅ HNSW approximate search");
    #[cfg(not(feature = "hnsw"))]
    println!("  ❌ HNSW approximate search (use --features hnsw)");

    println!();

    let config = BenchmarkConfig::default();
    println!("Configuration:");
    println!("  Vectors: {}", config.num_vectors);
    println!("  Dimension: {}", config.dimension);
    println!("  Queries: {}", config.num_queries);
    println!("  Top-k: {}", config.top_k);
    println!("  Distance metric: {:?}", config.distance_metric);

    // Generate test data
    println!("\n📊 Generating test data...");
    let test_vectors = generate_test_vectors(config.num_vectors, config.dimension);
    let test_queries = generate_test_queries(config.num_queries, config.dimension, config.top_k);
    println!(
        "Generated {} vectors and {} queries",
        test_vectors.len(),
        test_queries.len()
    );

    // Run benchmarks
    println!("\n🔬 Running benchmarks...");

    let basic_result = benchmark_basic_memory_store(&config, &test_vectors, &test_queries).await?;
    let optimized_result =
        benchmark_optimized_memory_store(&config, &test_vectors, &test_queries).await?;

    // Run HNSW benchmark if feature is enabled
    #[cfg(feature = "hnsw")]
    let hnsw_result = {
        println!("Running HNSW benchmark...");
        Some(benchmark_hnsw_store(&config, &test_vectors, &test_queries).await?)
    };
    #[cfg(not(feature = "hnsw"))]
    let hnsw_result: Option<BenchmarkResult> = None;

    // Print results
    basic_result.print_summary();
    optimized_result.print_summary();

    #[cfg(feature = "hnsw")]
    if let Some(ref result) = hnsw_result {
        result.print_summary();
    }

    // Calculate improvements
    println!("\n📈 Performance Improvements");
    println!("===========================");

    if basic_result.total_query_time_ms > 0 {
        let query_speedup =
            basic_result.total_query_time_ms as f64 / optimized_result.total_query_time_ms as f64;
        println!("Optimized vs Basic - Query speedup: {:.2}x", query_speedup);

        #[cfg(feature = "hnsw")]
        if let Some(ref hnsw_result) = hnsw_result {
            if hnsw_result.total_query_time_ms > 0 {
                let hnsw_speedup = basic_result.total_query_time_ms as f64
                    / hnsw_result.total_query_time_ms as f64;
                println!("HNSW vs Basic - Query speedup: {:.2}x", hnsw_speedup);
            }
        }
    }

    if basic_result.indexing_time_ms > 0 {
        let indexing_speedup =
            basic_result.indexing_time_ms as f64 / optimized_result.indexing_time_ms as f64;
        println!(
            "Optimized vs Basic - Indexing speedup: {:.2}x",
            indexing_speedup
        );
    }

    let qps_improvement = (optimized_result.qps - basic_result.qps) / basic_result.qps * 100.0;
    println!(
        "Optimized vs Basic - QPS improvement: {:.1}%",
        qps_improvement
    );

    // Test SIMD operations if available
    #[cfg(feature = "simd")]
    test_simd_operations(&config).await?;

    Ok(())
}

/// Run different scale benchmarks
async fn run_scale_benchmarks() -> Result<()> {
    println!("\n🔍 Scale Benchmarks");
    println!("==================");

    let scales = vec![1_000, 5_000, 10_000, 50_000];

    for scale in scales {
        println!("\n--- Scale: {} vectors ---", scale);

        let config = BenchmarkConfig {
            num_vectors: scale,
            num_queries: 50,
            ..Default::default()
        };

        let test_vectors = generate_test_vectors(config.num_vectors, config.dimension);
        let test_queries =
            generate_test_queries(config.num_queries, config.dimension, config.top_k);

        let result =
            benchmark_optimized_memory_store(&config, &test_vectors, &test_queries).await?;

        println!(
            "Scale {}: {:.2} ms avg query, {:.2} QPS",
            scale, result.avg_query_time_ms, result.qps
        );
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("Starting Cheungfun performance benchmarks...\n");

    // Run main comparison
    run_performance_comparison().await?;

    // Run scale benchmarks
    run_scale_benchmarks().await?;

    println!("\n✅ Benchmarks completed!");

    Ok(())
}
