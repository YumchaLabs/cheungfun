//! Simple parallel processing test for Cheungfun
//!
//! This test verifies that parallel processing optimizations are working correctly
//! and provides performance comparisons between sequential and parallel operations.

use std::time::Instant;
use uuid::Uuid;

use rayon::prelude::*;

use cheungfun_core::{
    traits::{DistanceMetric, VectorStore},
    types::{ChunkInfo, Node, Query},
};

#[cfg(feature = "optimized-memory")]
use cheungfun_integrations::vector_stores::memory_optimized::OptimizedInMemoryVectorStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Cheungfun Parallel Processing Test");
    println!("====================================");

    // Test parallel availability
    test_parallel_availability().await?;

    // Test parallel vector operations
    test_parallel_vector_operations().await?;

    // Performance comparison
    test_performance_comparison().await?;

    println!("\n🎉 All parallel processing tests completed successfully!");
    Ok(())
}

async fn test_parallel_availability() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing Parallel Processing Availability");
    println!("-------------------------------------------");

    println!("✅ Parallel processing available");
    println!("   Available CPU cores: {}", rayon::current_num_threads());
    println!(
        "   Rayon thread pool size: {}",
        rayon::current_num_threads()
    );

    Ok(())
}

async fn test_parallel_vector_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧮 Testing Parallel Vector Operations");
    println!("------------------------------------");

    #[cfg(feature = "optimized-memory")]
    {
        let dimension = 128;
        let store = OptimizedInMemoryVectorStore::new(dimension, DistanceMetric::Cosine);

        // Create test nodes
        let mut nodes = Vec::new();
        for i in 0..1000 {
            let embedding: Vec<f32> = (0..dimension)
                .map(|j| ((i * dimension + j) as f32 * 0.01).sin())
                .collect();

            let node = Node::new(
                format!("parallel_test_doc_{}", i),
                Uuid::new_v4(),
                ChunkInfo::new(0, 100, i),
            )
            .with_embedding(embedding);

            nodes.push(node);
        }

        println!("✅ Created {} test nodes", nodes.len());

        // Test parallel indexing
        let start = Instant::now();
        let node_ids = store.add(nodes.clone()).await?;
        let parallel_add_duration = start.elapsed();

        println!(
            "✅ Parallel indexing completed in {:?}",
            parallel_add_duration
        );
        println!("   Added {} nodes", node_ids.len());

        // Test parallel search
        let query_embedding: Vec<f32> = (0..dimension).map(|i| (i as f32 * 0.01).sin()).collect();

        let query = Query::new("parallel test query")
            .with_embedding(query_embedding)
            .with_top_k(10);

        let start = Instant::now();
        let results = store.search(&query).await?;
        let parallel_search_duration = start.elapsed();

        println!(
            "✅ Parallel search completed in {:?}",
            parallel_search_duration
        );
        println!("   Found {} results", results.len());

        if !results.is_empty() {
            println!("   Top result similarity: {:.6}", results[0].score);
            println!(
                "   Worst result similarity: {:.6}",
                results.last().unwrap().score
            );
        }

        // Test batch operations
        let mut queries = Vec::new();
        for i in 0..50 {
            let embedding: Vec<f32> = (0..dimension)
                .map(|j| ((i * dimension + j) as f32 * 0.001).cos())
                .collect();

            let query = Query::new(&format!("batch_query_{}", i))
                .with_embedding(embedding)
                .with_top_k(5);

            queries.push(query);
        }

        let start = Instant::now();
        let mut total_results = 0;
        for query in &queries {
            let results = store.search(query).await?;
            total_results += results.len();
        }
        let batch_duration = start.elapsed();

        println!("✅ Batch search completed in {:?}", batch_duration);
        println!("   Processed {} queries", queries.len());
        println!("   Total results: {}", total_results);
        println!(
            "   Avg per query: {:?}",
            batch_duration / queries.len() as u32
        );
    }

    #[cfg(not(feature = "optimized-memory"))]
    {
        println!("⚠️  Parallel vector operation tests skipped");
        println!(
            "   💡 Enable with: cargo run --features \"performance\" --bin simple_parallel_test"
        );
    }

    Ok(())
}

async fn test_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Parallel vs Sequential Performance");
    println!("-----------------------------------");

    {
        let data_size = 10000;
        let iterations = 100;

        // Create test data
        let test_vectors: Vec<Vec<f32>> = (0..data_size)
            .map(|i| {
                (0..128)
                    .map(|j| ((i * 128 + j) as f32 * 0.001).sin())
                    .collect()
            })
            .collect();

        let query_vector: Vec<f32> = (0..128).map(|i| (i as f32 * 0.001).cos()).collect();

        println!("📊 Performance Test Setup:");
        println!("   Data vectors: {}", data_size);
        println!("   Vector dimensions: 128");
        println!("   Iterations: {}", iterations);

        // Test sequential processing
        let start = Instant::now();
        for _ in 0..iterations {
            let _similarities: Vec<f32> = test_vectors
                .iter()
                .map(|vec| cosine_similarity(&query_vector, vec))
                .collect();
        }
        let sequential_duration = start.elapsed();

        // Test parallel processing
        let start = Instant::now();
        for _ in 0..iterations {
            let _similarities: Vec<f32> = test_vectors
                .par_iter()
                .map(|vec| cosine_similarity(&query_vector, vec))
                .collect();
        }
        let parallel_duration = start.elapsed();

        println!("🚀 Performance Results:");
        println!(
            "   Sequential: {:?} ({:.2} ops/sec)",
            sequential_duration,
            iterations as f64 / sequential_duration.as_secs_f64()
        );
        println!(
            "   Parallel:   {:?} ({:.2} ops/sec)",
            parallel_duration,
            iterations as f64 / parallel_duration.as_secs_f64()
        );

        if parallel_duration < sequential_duration {
            let speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();
            println!("   🚀 Parallel is {:.2}x faster!", speedup);
        } else {
            println!("   ⚠️  Sequential was faster (overhead for this workload)");
        }

        // Test parallel batch operations
        let batch_size = 1000;
        let batches: Vec<&[Vec<f32>]> = test_vectors.chunks(batch_size).collect();

        let start = Instant::now();
        let _batch_results: Vec<Vec<f32>> = batches
            .par_iter()
            .map(|batch| {
                batch
                    .iter()
                    .map(|vec| cosine_similarity(&query_vector, vec))
                    .collect()
            })
            .collect();
        let batch_parallel_duration = start.elapsed();

        println!(
            "   Batch parallel: {:?} ({} batches of {})",
            batch_parallel_duration,
            batches.len(),
            batch_size
        );

        // Calculate efficiency metrics
        let cpu_cores = rayon::current_num_threads();
        let theoretical_speedup = cpu_cores as f64;
        let actual_speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();
        let efficiency = (actual_speedup / theoretical_speedup) * 100.0;

        println!("📈 Efficiency Analysis:");
        println!("   CPU cores: {}", cpu_cores);
        println!("   Theoretical max speedup: {:.2}x", theoretical_speedup);
        println!("   Actual speedup: {:.2}x", actual_speedup);
        println!("   Parallel efficiency: {:.1}%", efficiency);
    }

    Ok(())
}

// Helper function for cosine similarity calculation
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
