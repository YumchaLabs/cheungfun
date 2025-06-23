//! Simple performance test for Cheungfun
//!
//! This example demonstrates basic performance testing without complex dependencies

use cheungfun_core::{
    traits::{VectorStore, DistanceMetric},
    types::{Node, Query, ChunkInfo},
    Result,
};
use cheungfun_integrations::vector_stores::{
    memory::InMemoryVectorStore,
    memory_optimized::OptimizedInMemoryVectorStore,
    fast_memory::FastInMemoryVectorStore,
};
use std::time::Instant;
use uuid::Uuid;

/// Generate random test vectors
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
            
            Node::new(
                format!("Document {}", i),
                Uuid::new_v4(),
                ChunkInfo::new(0, 100, i),
            )
            .with_embedding(embedding)
            .with_metadata("index", i)
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
            
            let mut query = Query::new(format!("Query {}", i));
            query.embedding = Some(embedding);
            query.top_k = top_k;
            query
        })
        .collect()
}

/// Benchmark basic memory store
async fn benchmark_basic_store(
    vectors: &[Node],
    queries: &[Query],
    dimension: usize,
) -> Result<(u64, u64)> {
    let store = InMemoryVectorStore::new(dimension, DistanceMetric::Cosine);
    
    // Benchmark indexing
    let start = Instant::now();
    store.add(vectors.to_vec()).await?;
    let indexing_time = start.elapsed().as_millis() as u64;
    
    // Benchmark queries
    let start = Instant::now();
    for query in queries {
        let _results = store.search(query).await?;
    }
    let query_time = start.elapsed().as_millis() as u64;
    
    Ok((indexing_time, query_time))
}

/// Benchmark optimized memory store
async fn benchmark_optimized_store(
    vectors: &[Node],
    queries: &[Query],
    dimension: usize,
) -> Result<(u64, u64)> {
    let store = OptimizedInMemoryVectorStore::new(dimension, DistanceMetric::Cosine);

    // Benchmark indexing
    let start = Instant::now();
    store.add(vectors.to_vec()).await?;
    let indexing_time = start.elapsed().as_millis() as u64;

    // Benchmark queries
    let start = Instant::now();
    for query in queries {
        let _results = store.search(query).await?;
    }
    let query_time = start.elapsed().as_millis() as u64;

    // Get performance stats
    let stats = store.get_stats();
    println!("Optimized store stats:");
    println!("  SIMD operations: {}", stats.simd_operations);
    println!("  Parallel operations: {}", stats.parallel_operations);
    println!("  Average search time: {:.2} ms", stats.avg_search_time_ms);

    Ok((indexing_time, query_time))
}

/// Benchmark fast memory store
async fn benchmark_fast_store(
    vectors: &[Node],
    queries: &[Query],
    dimension: usize,
) -> Result<(u64, u64)> {
    let store = FastInMemoryVectorStore::new(dimension, DistanceMetric::Cosine);

    // Benchmark indexing
    let start = Instant::now();
    store.add(vectors.to_vec()).await?;
    let indexing_time = start.elapsed().as_millis() as u64;

    // Benchmark queries
    let start = Instant::now();
    for query in queries {
        let _results = store.search(query).await?;
    }
    let query_time = start.elapsed().as_millis() as u64;

    // Get performance stats
    let search_count = store.get_search_count();
    println!("Fast store stats:");
    println!("  Search operations: {}", search_count);

    Ok((indexing_time, query_time))
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Cheungfun 简单性能测试");
    println!("========================");
    
    // Test configuration
    let num_vectors = 1000;
    let dimension = 384;
    let num_queries = 100;
    let top_k = 10;
    
    println!("配置:");
    println!("  向量数量: {}", num_vectors);
    println!("  向量维度: {}", dimension);
    println!("  查询数量: {}", num_queries);
    println!("  Top-k: {}", top_k);
    
    // Generate test data
    println!("\n📊 生成测试数据...");
    let vectors = generate_test_vectors(num_vectors, dimension);
    let queries = generate_test_queries(num_queries, dimension, top_k);
    
    // Benchmark basic store
    println!("\n🔬 测试基础内存存储...");
    let (basic_indexing, basic_query) = benchmark_basic_store(&vectors, &queries, dimension).await?;

    // Benchmark optimized store
    println!("\n⚡ 测试优化内存存储...");
    let (opt_indexing, opt_query) = benchmark_optimized_store(&vectors, &queries, dimension).await?;

    // Benchmark fast store
    println!("\n🚀 测试快速内存存储...");
    let (fast_indexing, fast_query) = benchmark_fast_store(&vectors, &queries, dimension).await?;
    
    // Print results
    println!("\n📈 性能对比结果");
    println!("================");

    println!("\n索引性能:");
    println!("  基础版本: {} ms", basic_indexing);
    println!("  优化版本: {} ms", opt_indexing);
    println!("  快速版本: {} ms", fast_indexing);

    println!("\n查询性能:");
    println!("  基础版本: {} ms", basic_query);
    println!("  优化版本: {} ms", opt_query);
    println!("  快速版本: {} ms", fast_query);

    // Calculate speedups
    if basic_query > 0 {
        let opt_speedup = basic_query as f64 / opt_query as f64;
        let fast_speedup = basic_query as f64 / fast_query as f64;
        println!("\n加速比 (相对于基础版本):");
        println!("  优化版本: {:.2}x", opt_speedup);
        println!("  快速版本: {:.2}x", fast_speedup);
    }

    // Calculate QPS
    let basic_qps = if basic_query > 0 {
        (num_queries as f64 * 1000.0) / basic_query as f64
    } else {
        0.0
    };

    let opt_qps = if opt_query > 0 {
        (num_queries as f64 * 1000.0) / opt_query as f64
    } else {
        0.0
    };

    let fast_qps = if fast_query > 0 {
        (num_queries as f64 * 1000.0) / fast_query as f64
    } else {
        0.0
    };

    println!("\n查询吞吐量 (QPS):");
    println!("  基础版本: {:.2} queries/sec", basic_qps);
    println!("  优化版本: {:.2} queries/sec", opt_qps);
    println!("  快速版本: {:.2} queries/sec", fast_qps);
    
    println!("\n✅ 性能测试完成!");
    
    Ok(())
}
