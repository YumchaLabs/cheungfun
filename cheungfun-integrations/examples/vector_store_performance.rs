//! Vector Store Performance Comparison
//!
//! Compares different vector store implementations in cheungfun-integrations:
//! - Memory-based stores
//! - HNSW approximate search
//! - Performance characteristics

use cheungfun_core::{
    traits::VectorStore,
    types::{Query, SearchMode},
    DistanceMetric, Node,
};
use cheungfun_integrations::vector_stores::{
    memory::InMemoryVectorStore, memory_optimized::OptimizedInMemoryVectorStore,
};
use rand::Rng;
use std::{collections::HashMap, sync::Arc, time::Instant};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Vector Store Performance Comparison");
    println!("======================================\n");

    // Configuration
    const DIMENSION: usize = 384;
    const NUM_VECTORS: usize = 1000;
    const QUERY_COUNT: usize = 100;

    // Create test data
    println!("📊 Generating test data...");
    let test_nodes = generate_test_nodes(NUM_VECTORS, DIMENSION);
    let test_queries = generate_test_queries(QUERY_COUNT, DIMENSION);
    println!(
        "  Generated {} nodes and {} queries\n",
        NUM_VECTORS, QUERY_COUNT
    );

    // Test stores
    let stores: Vec<(String, Box<dyn VectorStore>)> = vec![
        (
            "InMemoryVectorStore".to_string(),
            Box::new(InMemoryVectorStore::new(DIMENSION, DistanceMetric::Cosine)),
        ),
        (
            "OptimizedInMemoryVectorStore".to_string(),
            Box::new(OptimizedInMemoryVectorStore::new(
                DIMENSION,
                DistanceMetric::Cosine,
            )),
        ),
    ];

    for (store_name, store) in stores {
        println!("🔬 Testing: {}", store_name);
        println!("{}", "-".repeat(50));

        // Test insertion performance
        let start = Instant::now();
        for chunk in test_nodes.chunks(100) {
            store.add(chunk.to_vec()).await?;
        }
        let insert_duration = start.elapsed();
        let insert_rate = NUM_VECTORS as f64 / insert_duration.as_secs_f64();

        println!("📝 Insertion Performance:");
        println!("  Time: {:?}", insert_duration);
        println!("  Rate: {:.2} vectors/sec", insert_rate);

        // Test query performance
        let start = Instant::now();
        let mut total_results = 0;

        for query in &test_queries {
            let results = store.search(query).await?;
            total_results += results.len();
        }

        let query_duration = start.elapsed();
        let query_rate = QUERY_COUNT as f64 / query_duration.as_secs_f64();

        println!("🔍 Query Performance:");
        println!("  Time: {:?}", query_duration);
        println!("  Rate: {:.2} queries/sec", query_rate);
        println!(
            "  Avg results per query: {:.1}",
            total_results as f64 / QUERY_COUNT as f64
        );

        // Test memory usage (simplified)
        let memory_estimate = estimate_memory_usage(NUM_VECTORS, DIMENSION);
        println!("💾 Estimated Memory Usage: {:.2} MB", memory_estimate);

        // Health check
        match store.health_check().await {
            Ok(()) => println!("✅ Health check: PASSED"),
            Err(e) => println!("❌ Health check: FAILED ({})", e),
        }

        println!();
    }

    // Performance summary
    println!("📈 Performance Summary");
    println!("{}", "-".repeat(50));
    println!("Test Configuration:");
    println!("  Vectors: {}", NUM_VECTORS);
    println!("  Dimensions: {}", DIMENSION);
    println!("  Queries: {}", QUERY_COUNT);
    println!("  Distance Metric: Cosine");
    println!();

    println!("💡 Recommendations:");
    println!("  • InMemoryVectorStore: Good for prototyping and small datasets");
    println!(
        "  • OptimizedInMemoryVectorStore: Better for larger datasets with memory constraints"
    );
    println!("  • For production with >10K vectors, consider Qdrant or HNSW stores");

    Ok(())
}

fn generate_test_nodes(count: usize, dimension: usize) -> Vec<Node> {
    let mut rng = rand::thread_rng();
    let mut nodes = Vec::with_capacity(count);

    for i in 0..count {
        let embedding: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let node = Node::builder()
            .content(format!(
                "This is test document number {} with some sample content for testing purposes.",
                i
            ))
            .source_document_id(Uuid::new_v4())
            .chunk_info(cheungfun_core::types::ChunkInfo::with_char_indices(
                0, 100, 0,
            ))
            .metadata("doc_id", i)
            .metadata("category", format!("category_{}", i % 5))
            .embedding(embedding)
            .build()
            .expect("Failed to build node");

        nodes.push(node);
    }

    nodes
}

fn generate_test_queries(count: usize, dimension: usize) -> Vec<Query> {
    let mut rng = rand::thread_rng();
    let mut queries = Vec::with_capacity(count);

    for i in 0..count {
        let embedding: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();

        queries.push(Query {
            text: format!("Test query number {}", i),
            embedding: Some(embedding),
            filters: HashMap::new(),
            top_k: 5,
            similarity_threshold: Some(0.1),
            search_mode: SearchMode::Vector,
        });
    }

    queries
}

fn estimate_memory_usage(num_vectors: usize, dimension: usize) -> f64 {
    // Rough estimation: each vector takes dimension * 4 bytes (f32) + overhead
    let vector_size = dimension * 4; // 4 bytes per f32
    let overhead_per_vector = 200; // Estimated overhead for metadata, indices, etc.
    let total_bytes = num_vectors * (vector_size + overhead_per_vector);

    total_bytes as f64 / (1024.0 * 1024.0) // Convert to MB
}
