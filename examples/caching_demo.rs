//! Demonstration of caching functionality in Cheungfun pipelines.
//!
//! This example shows how to use the unified caching system to speed up
//! development workflows by caching expensive operations like embedding generation.

use cheungfun_core::{
    cache::{FileCache, MemoryCache},
    traits::PipelineCache,
    types::{ChunkInfo, Node},
};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{info, Level};
use tracing_subscriber;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🚀 Starting Cheungfun Caching Demo");

    // Demo 1: Memory Cache
    demo_memory_cache().await?;

    // Demo 2: File Cache
    demo_file_cache().await?;

    // Demo 3: Cache Performance Comparison
    demo_cache_performance().await?;

    info!("✅ Caching demo completed successfully!");
    Ok(())
}

/// Demonstrate memory cache functionality.
async fn demo_memory_cache() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n📝 Demo 1: Memory Cache");

    let cache = MemoryCache::new();

    // Cache some embeddings
    let embeddings = vec![
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.5, 0.6, 0.7, 0.8],
        vec![0.9, 1.0, 1.1, 1.2],
    ];

    let ttl = Duration::from_secs(3600); // 1 hour

    info!("Storing embeddings in memory cache...");
    for (i, embedding) in embeddings.iter().enumerate() {
        let key = format!("embedding_{}", i);
        cache.put_embedding(&key, embedding.clone(), ttl).await?;
    }

    info!("Retrieving embeddings from memory cache...");
    for i in 0..embeddings.len() {
        let key = format!("embedding_{}", i);
        if let Some(cached_embedding) = cache.get_embedding(&key).await? {
            info!("✅ Retrieved embedding {}: {:?}", i, cached_embedding);
        } else {
            info!("❌ Failed to retrieve embedding {}", i);
        }
    }

    // Check cache statistics
    let stats = cache.stats().await?;
    info!(
        "📊 Memory cache stats: hit rate = {:.1}%, total entries = {}",
        stats.hit_rate(),
        stats.total_entries
    );

    Ok(())
}

/// Demonstrate file cache functionality.
async fn demo_file_cache() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n💾 Demo 2: File Cache");

    let cache_dir = "./demo_cache";
    let cache = FileCache::with_default_config(cache_dir).await?;

    info!("Cache directory: {}", cache.cache_dir().display());

    // Cache some sample nodes
    let nodes = vec![
        create_sample_node("node_1", "This is the first sample document."),
        create_sample_node("node_2", "This is the second sample document."),
        create_sample_node("node_3", "This is the third sample document."),
    ];

    let ttl = Duration::from_secs(7200); // 2 hours

    info!("Storing nodes in file cache...");
    for (i, node_list) in nodes.iter().enumerate() {
        let key = format!("nodes_batch_{}", i);
        cache.put_nodes(&key, vec![node_list.clone()], ttl).await?;
    }

    info!("Retrieving nodes from file cache...");
    for i in 0..nodes.len() {
        let key = format!("nodes_batch_{}", i);
        if let Some(cached_nodes) = cache.get_nodes(&key).await? {
            info!("✅ Retrieved {} nodes for key: {}", cached_nodes.len(), key);
        } else {
            info!("❌ Failed to retrieve nodes for key: {}", key);
        }
    }

    // Cache arbitrary data
    #[derive(serde::Serialize, serde::Deserialize, Debug)]
    struct CustomData {
        name: String,
        value: i32,
        tags: Vec<String>,
    }

    let custom_data = CustomData {
        name: "sample_data".to_string(),
        value: 42,
        tags: vec!["demo".to_string(), "cache".to_string()],
    };

    info!("Storing custom data in file cache...");
    let serialized_data =
        bincode::serde::encode_to_vec(&custom_data, bincode::config::standard()).unwrap();
    cache
        .put_data_bytes("custom_data", serialized_data, ttl)
        .await?;

    info!("Retrieving custom data from file cache...");
    if let Some(cached_bytes) = cache.get_data_bytes("custom_data").await? {
        if let Ok((cached_data, _)) = bincode::serde::decode_from_slice::<CustomData, _>(
            &cached_bytes,
            bincode::config::standard(),
        ) {
            info!("✅ Retrieved custom data: {:?}", cached_data);
        } else {
            info!("❌ Failed to deserialize custom data");
        }
    } else {
        info!("❌ Failed to retrieve custom data");
    }

    // Check cache health
    let health = cache.health().await?;
    info!(
        "🏥 File cache health: {:?}, hit rate = {:.1}%",
        health.status, health.hit_rate
    );

    // Cleanup expired entries
    let removed = cache.cleanup().await?;
    info!("🧹 Cleaned up {} expired entries", removed);

    Ok(())
}

/// Demonstrate cache performance benefits.
async fn demo_cache_performance() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n⚡ Demo 3: Cache Performance Comparison");

    let cache = MemoryCache::new();
    let ttl = Duration::from_secs(3600);

    // Simulate expensive embedding computation
    async fn compute_expensive_embedding(text: &str) -> Vec<f32> {
        // Simulate computation time
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Generate a simple hash-based embedding
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        vec![
            (hash & 0xFF) as f32 / 255.0,
            ((hash >> 8) & 0xFF) as f32 / 255.0,
            ((hash >> 16) & 0xFF) as f32 / 255.0,
            ((hash >> 24) & 0xFF) as f32 / 255.0,
        ]
    }

    let test_texts = vec![
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret visual information.",
        "Reinforcement learning trains agents through trial and error.",
    ];

    // First run: No cache (cold start)
    info!("🥶 Cold start (no cache):");
    let start_time = Instant::now();

    for text in &test_texts {
        let _embedding = compute_expensive_embedding(text).await;
    }

    let cold_duration = start_time.elapsed();
    info!("   Time taken: {:?}", cold_duration);

    // Second run: Populate cache
    info!("💾 Populating cache:");
    let start_time = Instant::now();

    for text in &test_texts {
        let key = format!("perf_test_{}", text.len());
        let embedding = compute_expensive_embedding(text).await;
        cache.put_embedding(&key, embedding, ttl).await?;
    }

    let populate_duration = start_time.elapsed();
    info!("   Time taken: {:?}", populate_duration);

    // Third run: Use cache (warm start)
    info!("🔥 Warm start (with cache):");
    let start_time = Instant::now();

    for text in &test_texts {
        let key = format!("perf_test_{}", text.len());
        if let Some(_cached_embedding) = cache.get_embedding(&key).await? {
            // Cache hit - no computation needed
        } else {
            // Cache miss - compute and store
            let embedding = compute_expensive_embedding(text).await;
            cache.put_embedding(&key, embedding, ttl).await?;
        }
    }

    let warm_duration = start_time.elapsed();
    info!("   Time taken: {:?}", warm_duration);

    // Calculate performance improvement
    let speedup = cold_duration.as_millis() as f64 / warm_duration.as_millis() as f64;
    info!(
        "🚀 Performance improvement: {:.1}x faster with cache!",
        speedup
    );

    // Final cache statistics
    let stats = cache.stats().await?;
    info!(
        "📊 Final cache stats: {} hits, {} misses, {:.1}% hit rate",
        stats.hits,
        stats.misses,
        stats.hit_rate()
    );

    Ok(())
}

/// Create a sample node for testing.
fn create_sample_node(id: &str, content: &str) -> Node {
    let source_doc_id = Uuid::new_v4();
    let chunk_info = ChunkInfo::with_char_indices(0, content.len(), 0);

    Node::builder()
        .content(content)
        .source_document_id(source_doc_id)
        .chunk_info(chunk_info)
        .metadata("id", id)
        .metadata("type", "demo")
        .build()
        .expect("Failed to create sample node")
}
