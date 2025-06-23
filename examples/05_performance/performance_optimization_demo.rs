//! Performance optimization demonstration
//!
//! This example demonstrates the performance improvements achieved through:
//! 1. SIMD-accelerated vector operations
//! 2. Parallel processing with Rayon
//! 3. HNSW approximate nearest neighbor search
//!
//! Run with different feature flags to see the performance differences:
//! - Basic: `cargo run --example performance_optimization_demo`
//! - SIMD: `cargo run --example performance_optimization_demo --features simd`
//! - HNSW: `cargo run --example performance_optimization_demo --features hnsw`
//! - All: `cargo run --example performance_optimization_demo --features performance`

use anyhow::Result;
use cheungfun_core::{
    DistanceMetric,
    traits::VectorStore,
    types::{ChunkInfo, Node, Query, SearchMode},
};
use cheungfun_integrations::vector_stores::{
    memory::InMemoryVectorStore,
    memory_optimized::OptimizedInMemoryVectorStore,
};

#[cfg(feature = "hnsw")]
use cheungfun_integrations::vector_stores::hnsw::HnswVectorStore;

#[cfg(feature = "simd")]
use cheungfun_integrations::simd::SimdVectorOps;

#[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
use cheungfun_integrations::gpu::GpuVectorOps;

use rand::Rng;
use std::time::Instant;
use tokio;
use tracing::{info, warn};
use uuid::Uuid;

/// Generate random vectors for testing
fn generate_random_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            let mut vector: Vec<f32> = (0..dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            
            // Normalize the vector
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut vector {
                    *v /= norm;
                }
            }
            vector
        })
        .collect()
}

/// Create test nodes with embeddings
fn create_test_nodes(vectors: &[Vec<f32>]) -> Vec<Node> {
    vectors
        .iter()
        .enumerate()
        .map(|(i, embedding)| {
            let source_doc_id = Uuid::new_v4();
            let chunk_info = ChunkInfo::new(i * 100, (i + 1) * 100, i);
            let mut node = Node::new(
                format!("Test document content {}", i),
                source_doc_id,
                chunk_info,
            );
            node.embedding = Some(embedding.clone());
            node
        })
        .collect()
}

/// Benchmark vector similarity calculations
async fn benchmark_vector_operations() -> Result<()> {
    println!("\n🔬 Vector Operations Benchmark");
    println!("=====================================");

    let dimension = 384;
    let num_pairs = 10000;
    
    // Generate test vectors
    let vectors = generate_random_vectors(num_pairs * 2, dimension);
    let pairs: Vec<(&[f32], &[f32])> = vectors
        .chunks(2)
        .map(|chunk| (chunk[0].as_slice(), chunk[1].as_slice()))
        .collect();

    // Benchmark basic operations
    let start = Instant::now();
    let mut basic_results = Vec::new();
    for (a, b) in &pairs {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let similarity = if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        };
        basic_results.push(similarity);
    }
    let basic_time = start.elapsed();
    println!("📊 Basic scalar operations: {:?} ({} ops)", basic_time, num_pairs);

    // Benchmark SIMD operations
    #[cfg(feature = "simd")]
    {
        let simd_ops = SimdVectorOps::new();
        println!("🚀 SIMD capabilities: {}", simd_ops.get_capabilities());
        
        let start = Instant::now();
        let simd_results = simd_ops.batch_cosine_similarity_f32(&pairs)?;
        let simd_time = start.elapsed();
        
        println!("⚡ SIMD operations: {:?} ({} ops)", simd_time, num_pairs);
        println!("📈 SIMD speedup: {:.2}x", basic_time.as_nanos() as f64 / simd_time.as_nanos() as f64);
        
        // Verify results are similar
        let avg_diff: f32 = basic_results.iter()
            .zip(simd_results.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / num_pairs as f32;
        println!("🎯 Average difference: {:.6}", avg_diff);
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("⚠️  SIMD features not enabled. Use --features simd to enable.");
    }

    // Benchmark GPU operations
    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
    {
        let gpu_ops = GpuVectorOps::new();
        println!("🎮 GPU device: {}", gpu_ops.device_info());
        
        if gpu_ops.is_gpu_available() {
            let vectors_a: Vec<Vec<f32>> = pairs.iter().map(|(a, _)| a.to_vec()).collect();
            let vectors_b: Vec<Vec<f32>> = pairs.iter().map(|(_, b)| b.to_vec()).collect();
            
            let start = Instant::now();
            let gpu_results = gpu_ops.batch_cosine_similarity_f32(&vectors_a, &vectors_b)?;
            let gpu_time = start.elapsed();
            
            println!("🎮 GPU operations: {:?} ({} ops)", gpu_time, num_pairs);
            println!("📈 GPU speedup: {:.2}x", basic_time.as_nanos() as f64 / gpu_time.as_nanos() as f64);
            
            // Verify results are similar
            let avg_diff: f32 = basic_results.iter()
                .zip(gpu_results.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>() / num_pairs as f32;
            println!("🎯 Average difference: {:.6}", avg_diff);
        } else {
            println!("⚠️  GPU not available, using CPU fallback");
        }
    }

    #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
    {
        println!("⚠️  GPU features not enabled. Use --features gpu to enable.");
    }

    Ok(())
}

/// Benchmark vector store search performance
async fn benchmark_vector_stores() -> Result<()> {
    println!("\n🗄️  Vector Store Search Benchmark");
    println!("=====================================");

    let dimension = 384;
    let num_vectors = 10000;
    let num_queries = 100;
    let top_k = 10;

    // Generate test data
    let vectors = generate_random_vectors(num_vectors, dimension);
    let nodes = create_test_nodes(&vectors);
    let query_vectors = generate_random_vectors(num_queries, dimension);

    // Benchmark basic in-memory store
    println!("📦 Testing InMemoryVectorStore...");
    let basic_store = InMemoryVectorStore::new(dimension, DistanceMetric::Cosine);
    basic_store.add(nodes.clone()).await?;

    let start = Instant::now();
    for query_vector in &query_vectors {
        let query = Query::new("test query")
            .with_embedding(query_vector.clone())
            .with_top_k(top_k);
        let _results = basic_store.search(query).await?;
    }
    let basic_search_time = start.elapsed();
    println!("📊 Basic search: {:?} ({} queries)", basic_search_time, num_queries);

    // Benchmark optimized in-memory store
    println!("⚡ Testing OptimizedInMemoryVectorStore...");
    let optimized_store = OptimizedInMemoryVectorStore::new(dimension, DistanceMetric::Cosine);
    optimized_store.add(nodes.clone()).await?;

    let start = Instant::now();
    for query_vector in &query_vectors {
        let query = Query::new("test query")
            .with_embedding(query_vector.clone())
            .with_top_k(top_k);
        let _results = optimized_store.search(query).await?;
    }
    let optimized_search_time = start.elapsed();
    println!("⚡ Optimized search: {:?} ({} queries)", optimized_search_time, num_queries);
    println!("📈 Optimization speedup: {:.2}x", 
             basic_search_time.as_nanos() as f64 / optimized_search_time.as_nanos() as f64);

    // Benchmark HNSW store
    #[cfg(feature = "hnsw")]
    {
        println!("🌐 Testing HnswVectorStore...");
        let hnsw_store = HnswVectorStore::new(dimension, DistanceMetric::Cosine);
        hnsw_store.initialize_index(num_vectors)?;
        hnsw_store.add(nodes.clone()).await?;

        let start = Instant::now();
        for query_vector in &query_vectors {
            let query = Query::new("test query")
                .with_embedding(query_vector.clone())
                .with_top_k(top_k);
            let _results = hnsw_store.search(query).await?;
        }
        let hnsw_search_time = start.elapsed();
        println!("🌐 HNSW search: {:?} ({} queries)", hnsw_search_time, num_queries);
        println!("📈 HNSW speedup: {:.2}x", 
                 basic_search_time.as_nanos() as f64 / hnsw_search_time.as_nanos() as f64);

        let stats = hnsw_store.get_stats();
        println!("📊 HNSW stats: {} vectors indexed, {:.2}μs avg search time", 
                 stats.vectors_indexed, stats.avg_search_time_us);
    }

    #[cfg(not(feature = "hnsw"))]
    {
        println!("⚠️  HNSW features not enabled. Use --features hnsw to enable.");
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 Cheungfun Performance Optimization Demo");
    println!("==========================================");

    // Display enabled features
    println!("\n🔧 Enabled Features:");
    #[cfg(feature = "simd")]
    println!("  ✅ SIMD acceleration");
    #[cfg(not(feature = "simd"))]
    println!("  ❌ SIMD acceleration (use --features simd)");

    #[cfg(feature = "hnsw")]
    println!("  ✅ HNSW approximate search");
    #[cfg(not(feature = "hnsw"))]
    println!("  ❌ HNSW approximate search (use --features hnsw)");

    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
    println!("  ✅ GPU acceleration");
    #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
    println!("  ❌ GPU acceleration (use --features gpu)");

    // Run benchmarks
    benchmark_vector_operations().await?;
    benchmark_vector_stores().await?;

    println!("\n✨ Performance optimization demo completed!");
    println!("\n💡 Tips for maximum performance:");
    println!("  • Use --features performance to enable all optimizations");
    println!("  • Consider HNSW for large-scale vector search (>10k vectors)");
    println!("  • GPU acceleration works best with large batch sizes");
    println!("  • SIMD provides consistent speedups for all vector operations");

    Ok(())
}
