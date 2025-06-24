//! Integration performance tests for Cheungfun optimizations
//!
//! This test demonstrates the actual performance improvements from:
//! - SIMD acceleration with SimSIMD
//! - Parallel processing with Rayon
//! - HNSW approximate nearest neighbor search
//! - Optimized vector stores

use cheungfun_core::{
    DistanceMetric,
    traits::VectorStore,
    types::{ChunkInfo, Node, Query},
};
use cheungfun_integrations::vector_stores::{
    memory::InMemoryVectorStore, memory_optimized::OptimizedInMemoryVectorStore,
};
use std::time::Instant;

#[cfg(feature = "simd")]
use cheungfun_integrations::simd::SimdVectorOps;

#[cfg(feature = "hnsw")]
use cheungfun_integrations::vector_stores::hnsw::HnswVectorStore;

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

                    (h.finish() % 2000) as f32 / 1000.0 - 1.0
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

#[tokio::test]
async fn test_simd_performance() {
    println!("\n⚡ SIMD Performance Test");
    println!("========================");

    let dimension = 384;
    let num_pairs = 10000;

    let vectors = generate_test_vectors(num_pairs * 2, dimension);
    let pairs: Vec<(&[f32], &[f32])> = vectors
        .chunks(2)
        .map(|chunk| (chunk[0].as_slice(), chunk[1].as_slice()))
        .collect();

    // Basic scalar operations
    let start = Instant::now();
    let mut basic_results = Vec::new();
    for (a, b) in &pairs {
        basic_results.push(cosine_similarity_basic(a, b));
    }
    let basic_time = start.elapsed();
    println!(
        "📊 Basic scalar: {:?} ({:.2} ops/sec)",
        basic_time,
        num_pairs as f64 / basic_time.as_secs_f64()
    );

    #[cfg(feature = "simd")]
    {
        let simd_ops = SimdVectorOps::new();
        println!("🔧 SIMD capabilities: {}", simd_ops.get_capabilities());

        if simd_ops.is_simd_available() {
            let start = Instant::now();
            let simd_results = simd_ops.batch_cosine_similarity_f32(&pairs).unwrap();
            let simd_time = start.elapsed();

            println!(
                "⚡ SIMD operations: {:?} ({:.2} ops/sec)",
                simd_time,
                num_pairs as f64 / simd_time.as_secs_f64()
            );
            println!(
                "📈 SIMD speedup: {:.2}x",
                basic_time.as_nanos() as f64 / simd_time.as_nanos() as f64
            );

            // Verify accuracy
            let avg_diff: f32 = basic_results
                .iter()
                .zip(simd_results.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / num_pairs as f32;
            println!("🎯 Average difference: {:.6}", avg_diff);

            assert!(avg_diff < 0.001, "SIMD results should be accurate");
        } else {
            println!("⚠️  SIMD not available on this system");
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("⚠️  SIMD features not enabled");
    }
}

#[tokio::test]
async fn test_vector_store_performance() {
    println!("\n🗄️  Vector Store Performance Test");
    println!("==================================");

    let dimension = 384;
    let num_vectors = 5000;
    let num_queries = 100;
    let top_k = 10;

    let vectors = generate_test_vectors(num_vectors, dimension);
    let nodes = create_test_nodes(&vectors);
    let query_vectors = generate_test_vectors(num_queries, dimension);

    // Test basic in-memory store
    println!("📦 Testing InMemoryVectorStore...");
    let basic_store = InMemoryVectorStore::new(dimension, DistanceMetric::Cosine);
    basic_store.add(nodes.clone()).await.unwrap();

    let start = Instant::now();
    for query_vector in &query_vectors {
        let query = Query::new("test query")
            .with_embedding(query_vector.clone())
            .with_top_k(top_k);
        let _results = basic_store.search(&query).await.unwrap();
    }
    let basic_time = start.elapsed();
    println!(
        "  ⏱️  Time: {:?} ({:.2} queries/sec)",
        basic_time,
        num_queries as f64 / basic_time.as_secs_f64()
    );

    // Test optimized in-memory store
    println!("⚡ Testing OptimizedInMemoryVectorStore...");
    let optimized_store = OptimizedInMemoryVectorStore::new(dimension, DistanceMetric::Cosine);
    optimized_store.add(nodes.clone()).await.unwrap();

    let start = Instant::now();
    for query_vector in &query_vectors {
        let query = Query::new("test query")
            .with_embedding(query_vector.clone())
            .with_top_k(top_k);
        let _results = optimized_store.search(&query).await.unwrap();
    }
    let optimized_time = start.elapsed();
    println!(
        "  ⏱️  Time: {:?} ({:.2} queries/sec)",
        optimized_time,
        num_queries as f64 / optimized_time.as_secs_f64()
    );
    println!(
        "  📈 Speedup: {:.2}x",
        basic_time.as_nanos() as f64 / optimized_time.as_nanos() as f64
    );

    // Test HNSW store
    #[cfg(feature = "hnsw")]
    {
        println!("🌐 Testing HnswVectorStore...");
        let hnsw_store = HnswVectorStore::new(dimension, DistanceMetric::Cosine);
        hnsw_store.initialize_index(num_vectors).unwrap();
        hnsw_store.add(nodes.clone()).await.unwrap();

        let start = Instant::now();
        for query_vector in &query_vectors {
            let query = Query::new("test query")
                .with_embedding(query_vector.clone())
                .with_top_k(top_k);
            let _results = hnsw_store.search(&query).await.unwrap();
        }
        let hnsw_time = start.elapsed();
        println!(
            "  ⏱️  Time: {:?} ({:.2} queries/sec)",
            hnsw_time,
            num_queries as f64 / hnsw_time.as_secs_f64()
        );
        println!(
            "  📈 HNSW speedup vs basic: {:.2}x",
            basic_time.as_nanos() as f64 / hnsw_time.as_nanos() as f64
        );
        println!(
            "  📈 HNSW speedup vs optimized: {:.2}x",
            optimized_time.as_nanos() as f64 / hnsw_time.as_nanos() as f64
        );

        let stats = hnsw_store.get_stats();
        println!(
            "  📊 HNSW stats: {} vectors, {:.2}μs avg search",
            stats.vectors_indexed, stats.avg_search_time_us
        );
    }

    #[cfg(not(feature = "hnsw"))]
    {
        println!("⚠️  HNSW features not enabled");
    }
}

#[tokio::test]
async fn test_parallel_processing() {
    println!("\n🔄 Parallel Processing Test");
    println!("============================");

    let dimension = 384;
    let num_vectors = 1000;

    let vectors = generate_test_vectors(num_vectors, dimension);
    let query_vector = &vectors[0];

    #[cfg(feature = "simd")]
    {
        let simd_ops = SimdVectorOps::new();

        // Sequential processing
        let start = Instant::now();
        let mut sequential_results = Vec::new();
        for vector in &vectors[1..] {
            sequential_results.push(
                simd_ops
                    .cosine_similarity_f32(query_vector, vector)
                    .unwrap(),
            );
        }
        let sequential_time = start.elapsed();

        // One-to-many processing (potentially parallel)
        let vector_refs: Vec<&[f32]> = vectors[1..].iter().map(|v| v.as_slice()).collect();
        let start = Instant::now();
        let parallel_results = simd_ops
            .one_to_many_cosine_similarity_f32(query_vector, &vector_refs)
            .unwrap();
        let parallel_time = start.elapsed();

        println!("📊 Operations: {}", vectors.len() - 1);
        println!(
            "⏱️  Sequential: {:?} ({:.2} ops/sec)",
            sequential_time,
            (vectors.len() - 1) as f64 / sequential_time.as_secs_f64()
        );
        println!(
            "⚡ One-to-many: {:?} ({:.2} ops/sec)",
            parallel_time,
            (vectors.len() - 1) as f64 / parallel_time.as_secs_f64()
        );
        println!(
            "📈 Speedup: {:.2}x",
            sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64
        );

        // Verify accuracy
        let avg_diff: f32 = sequential_results
            .iter()
            .zip(parallel_results.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / sequential_results.len() as f32;
        println!("🎯 Average difference: {:.6}", avg_diff);

        assert!(avg_diff < 0.001, "Parallel results should be accurate");
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("⚠️  SIMD features not enabled for parallel test");
    }
}
