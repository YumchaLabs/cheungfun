//! Performance benchmark comparing different vector store implementations
//!
//! This example demonstrates the performance improvements achieved through:
//! - SIMD-accelerated vector operations
//! - Optimized data structures
//! - Parallel processing
//!
//! Run with: cargo run --example performance_benchmark --features simd

use cheungfun_core::{
    traits::VectorStore,
    types::{ChunkInfo, Node, Query},
    DistanceMetric, CheungfunError, Result,
};
use cheungfun_integrations::vector_stores::{
    memory::InMemoryVectorStore,
    memory_optimized::OptimizedInMemoryVectorStore,
};
use std::time::Instant;
use uuid::Uuid;

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
        println!("Memory usage: {:.2} MB", self.memory_usage_bytes as f64 / 1_048_576.0);
        
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
            let embedding: Vec<f32> = (0..dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("index".to_string(), serde_json::Value::Number(i.into()));
            metadata.insert("category".to_string(), serde_json::Value::String(format!("cat_{}", i % 10)));
            
            {
                let source_doc_id = Uuid::new_v4();
                let chunk_info = ChunkInfo::new(i * 100, (i + 1) * 100, i);
                let mut node = Node::new(
                    format!("Document {}", i),
                    source_doc_id,
                    chunk_info,
                );
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
            let embedding: Vec<f32> = (0..dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            
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

/// Compare performance between different implementations
async fn run_performance_comparison() -> Result<()> {
    println!("🚀 Cheungfun Performance Benchmark");
    println!("===================================");
    
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
    println!("Generated {} vectors and {} queries", test_vectors.len(), test_queries.len());
    
    // Run benchmarks
    println!("\n🔬 Running benchmarks...");
    
    let basic_result = benchmark_basic_memory_store(&config, &test_vectors, &test_queries).await?;
    let optimized_result = benchmark_optimized_memory_store(&config, &test_vectors, &test_queries).await?;
    
    // Print results
    basic_result.print_summary();
    optimized_result.print_summary();
    
    // Calculate improvements
    println!("\n📈 Performance Improvements");
    println!("===========================");
    
    if basic_result.total_query_time_ms > 0 {
        let query_speedup = basic_result.total_query_time_ms as f64 / optimized_result.total_query_time_ms as f64;
        println!("Query speedup: {:.2}x", query_speedup);
    }
    
    if basic_result.indexing_time_ms > 0 {
        let indexing_speedup = basic_result.indexing_time_ms as f64 / optimized_result.indexing_time_ms as f64;
        println!("Indexing speedup: {:.2}x", indexing_speedup);
    }
    
    let qps_improvement = (optimized_result.qps - basic_result.qps) / basic_result.qps * 100.0;
    println!("QPS improvement: {:.1}%", qps_improvement);
    
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
        let test_queries = generate_test_queries(config.num_queries, config.dimension, config.top_k);
        
        let result = benchmark_optimized_memory_store(&config, &test_vectors, &test_queries).await?;
        
        println!("Scale {}: {:.2} ms avg query, {:.2} QPS", 
                scale, result.avg_query_time_ms, result.qps);
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
