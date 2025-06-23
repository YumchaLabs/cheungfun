//! Performance benchmark for Cheungfun optimizations
//!
//! Run with: cargo run --example performance_benchmark --release --features simd

use std::time::Instant;
use cheungfun_core::{
    traits::VectorStore,
    types::{ChunkInfo, Node, Query},
    DistanceMetric,
};
use cheungfun_integrations::vector_stores::{
    memory::InMemoryVectorStore,
    memory_optimized::OptimizedInMemoryVectorStore,
};

#[cfg(feature = "simd")]
use cheungfun_integrations::simd::SimdVectorOps;

use uuid::Uuid;

fn generate_test_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    (0..count)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let seed = hasher.finish();
            
            let mut vector: Vec<f32> = (0..dimension)
                .map(|j| {
                    let mut h = DefaultHasher::new();
                    (seed + j as u64).hash(&mut h);
                    let val = (h.finish() % 2000) as f32 / 1000.0 - 1.0;
                    val
                })
                .collect();
            
            // Normalize
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

fn cosine_similarity_basic(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Cheungfun Performance Benchmark");
    println!("===================================");

    let dimension = 384;
    let num_vectors = 5000;
    let num_queries = 100;
    let top_k = 10;

    println!("📊 Test Configuration:");
    println!("  • Vector dimension: {}", dimension);
    println!("  • Number of vectors: {}", num_vectors);
    println!("  • Number of queries: {}", num_queries);
    println!("  • Top-K results: {}", top_k);

    // Generate test data
    println!("\n🔄 Generating test data...");
    let start = Instant::now();
    let vectors = generate_test_vectors(num_vectors, dimension);
    let nodes = create_test_nodes(&vectors);
    let query_vectors = generate_test_vectors(num_queries, dimension);
    println!("  ⏱️  Data generation: {:?}", start.elapsed());

    // Test 1: SIMD Performance
    println!("\n⚡ Test 1: SIMD Vector Operations");
    test_simd_performance(&query_vectors, &vectors).await?;

    // Test 2: Vector Store Performance
    println!("\n🗄️  Test 2: Vector Store Performance");
    test_vector_store_performance(&nodes, &query_vectors, top_k).await?;

    // Test 3: Parallel Processing
    println!("\n🔄 Test 3: Parallel Processing");
    test_parallel_performance(&query_vectors, &vectors).await?;

    println!("\n✨ Performance Benchmark Complete!");
    println!("\n💡 Performance Summary:");
    println!("  • SIMD provides significant speedups for vector operations");
    println!("  • Optimized vector stores outperform basic implementations");
    println!("  • Parallel processing scales with available CPU cores");
    println!("  • Enable --features performance for maximum optimization");

    Ok(())
}

async fn test_simd_performance(query_vectors: &[Vec<f32>], vectors: &[Vec<f32>]) -> Result<(), Box<dyn std::error::Error>> {
    let num_pairs = 1000.min(query_vectors.len() * vectors.len() / 100);
    let pairs: Vec<(&[f32], &[f32])> = query_vectors.iter()
        .take(num_pairs / 10)
        .flat_map(|q| vectors.iter().take(10).map(move |v| (q.as_slice(), v.as_slice())))
        .collect();

    // Basic scalar operations
    let start = Instant::now();
    let mut basic_results = Vec::new();
    for (a, b) in &pairs {
        basic_results.push(cosine_similarity_basic(a, b));
    }
    let basic_time = start.elapsed();
    println!("  📊 Basic scalar: {:?} ({:.2} ops/sec)", 
             basic_time, pairs.len() as f64 / basic_time.as_secs_f64());

    #[cfg(feature = "simd")]
    {
        let simd_ops = SimdVectorOps::new();
        println!("  🔧 SIMD capabilities: {}", simd_ops.get_capabilities());
        
        if simd_ops.is_simd_available() {
            let start = Instant::now();
            let simd_results = simd_ops.batch_cosine_similarity_f32(&pairs)?;
            let simd_time = start.elapsed();
            
            println!("  ⚡ SIMD operations: {:?} ({:.2} ops/sec)", 
                     simd_time, pairs.len() as f64 / simd_time.as_secs_f64());
            println!("  📈 SIMD speedup: {:.2}x", 
                     basic_time.as_nanos() as f64 / simd_time.as_nanos() as f64);
            
            // Verify accuracy
            let avg_diff: f32 = basic_results.iter()
                .zip(simd_results.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>() / pairs.len() as f32;
            println!("  🎯 Average difference: {:.6}", avg_diff);
        } else {
            println!("  ⚠️  SIMD not available on this system");
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("  ⚠️  SIMD features not enabled (use --features simd)");
    }

    Ok(())
}

async fn test_vector_store_performance(nodes: &[Node], query_vectors: &[Vec<f32>], top_k: usize) -> Result<(), Box<dyn std::error::Error>> {
    // Test basic in-memory store
    println!("  📦 Testing InMemoryVectorStore...");
    let basic_store = InMemoryVectorStore::new(384, DistanceMetric::Cosine);
    basic_store.add(nodes.to_vec()).await?;

    let start = Instant::now();
    for query_vector in query_vectors.iter().take(50) {
        let query = Query::new("test query")
            .with_embedding(query_vector.clone())
            .with_top_k(top_k);
        let _results = basic_store.search(query).await?;
    }
    let basic_time = start.elapsed();
    println!("    ⏱️  Time: {:?} ({:.2} queries/sec)", 
             basic_time, 50.0 / basic_time.as_secs_f64());

    // Test optimized in-memory store
    println!("  ⚡ Testing OptimizedInMemoryVectorStore...");
    let optimized_store = OptimizedInMemoryVectorStore::new(384, DistanceMetric::Cosine);
    optimized_store.add(nodes.to_vec()).await?;

    let start = Instant::now();
    for query_vector in query_vectors.iter().take(50) {
        let query = Query::new("test query")
            .with_embedding(query_vector.clone())
            .with_top_k(top_k);
        let _results = optimized_store.search(query).await?;
    }
    let optimized_time = start.elapsed();
    println!("    ⏱️  Time: {:?} ({:.2} queries/sec)", 
             optimized_time, 50.0 / optimized_time.as_secs_f64());
    println!("    📈 Speedup: {:.2}x", 
             basic_time.as_nanos() as f64 / optimized_time.as_nanos() as f64);

    Ok(())
}

async fn test_parallel_performance(query_vectors: &[Vec<f32>], vectors: &[Vec<f32>]) -> Result<(), Box<dyn std::error::Error>> {
    let query_vector = &query_vectors[0];
    let test_vectors = &vectors[..500.min(vectors.len())];

    #[cfg(feature = "simd")]
    {
        let simd_ops = SimdVectorOps::new();
        
        // Sequential processing
        let start = Instant::now();
        let mut sequential_results = Vec::new();
        for vector in test_vectors {
            sequential_results.push(simd_ops.cosine_similarity_f32(query_vector, vector)?);
        }
        let sequential_time = start.elapsed();
        
        // One-to-many processing (potentially parallel)
        let vector_refs: Vec<&[f32]> = test_vectors.iter().map(|v| v.as_slice()).collect();
        let start = Instant::now();
        let parallel_results = simd_ops.one_to_many_cosine_similarity_f32(query_vector, &vector_refs)?;
        let parallel_time = start.elapsed();
        
        println!("  📊 Operations: {}", test_vectors.len());
        println!("  ⏱️  Sequential: {:?} ({:.2} ops/sec)", 
                 sequential_time, test_vectors.len() as f64 / sequential_time.as_secs_f64());
        println!("  ⚡ One-to-many: {:?} ({:.2} ops/sec)", 
                 parallel_time, test_vectors.len() as f64 / parallel_time.as_secs_f64());
        println!("  📈 Speedup: {:.2}x", 
                 sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64);
        
        // Verify accuracy
        let avg_diff: f32 = sequential_results.iter()
            .zip(parallel_results.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / sequential_results.len() as f32;
        println!("  🎯 Average difference: {:.6}", avg_diff);
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("  ⚠️  SIMD features not enabled for parallel test");
    }

    Ok(())
}
