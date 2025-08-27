//! Demonstration of the unified cache interface improvements.
//!
//! This example shows how to use the enhanced PipelineCache trait with batch operations,
//! cache adapters, and improved error handling.

use cheungfun_core::{
    cache::{EnhancedFileCache, FileCacheConfig, MemoryCache, UnifiedCache},
    traits::{CacheConfig, CacheKeyGenerator, EvictionPolicy, PipelineCache},
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

    info!("🚀 Starting Unified Cache Demo");

    // Demo 1: Batch Operations
    demo_batch_operations().await?;

    // Demo 2: Cache Configuration
    demo_cache_configuration().await?;

    // Demo 3: Cache Key Generation
    demo_cache_key_generation().await?;

    // Demo 4: Performance Comparison
    demo_performance_comparison().await?;

    info!("✅ Unified cache demo completed successfully!");
    Ok(())
}

/// Demonstrate batch operations for improved performance.
async fn demo_batch_operations() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n📦 Demo 1: Batch Operations");

    let cache = MemoryCache::new();

    // Prepare test data
    let embeddings = vec![
        ("doc1", vec![0.1, 0.2, 0.3, 0.4]),
        ("doc2", vec![0.5, 0.6, 0.7, 0.8]),
        ("doc3", vec![0.9, 1.0, 1.1, 1.2]),
        ("doc4", vec![1.3, 1.4, 1.5, 1.6]),
        ("doc5", vec![1.7, 1.8, 1.9, 2.0]),
    ];

    let ttl = Duration::from_secs(3600);

    // Batch store embeddings
    info!("Storing {} embeddings in batch...", embeddings.len());
    let start_time = Instant::now();

    let batch_items: Vec<(&str, Vec<f32>, Duration)> = embeddings
        .iter()
        .map(|(key, embedding)| (*key, embedding.clone(), ttl))
        .collect();

    cache.put_embeddings_batch(&batch_items).await?;
    let batch_store_time = start_time.elapsed();

    info!("Batch store completed in {:?}", batch_store_time);

    // Batch retrieve embeddings
    info!("Retrieving {} embeddings in batch...", embeddings.len());
    let start_time = Instant::now();

    let keys: Vec<&str> = embeddings.iter().map(|(key, _)| *key).collect();
    let retrieved_embeddings = cache.get_embeddings_batch(&keys).await?;
    let batch_retrieve_time = start_time.elapsed();

    info!("Batch retrieve completed in {:?}", batch_retrieve_time);

    // Verify results
    let mut successful_retrievals = 0;
    for (i, result) in retrieved_embeddings.iter().enumerate() {
        if let Some(embedding) = result {
            if embedding == &embeddings[i].1 {
                successful_retrievals += 1;
            }
        }
    }

    info!(
        "✅ Successfully retrieved {}/{} embeddings",
        successful_retrievals,
        embeddings.len()
    );

    // Compare with individual operations
    info!("Comparing with individual operations...");
    let start_time = Instant::now();

    for (key, embedding) in &embeddings {
        cache.put_embedding(key, embedding.clone(), ttl).await?;
    }

    let individual_store_time = start_time.elapsed();

    let start_time = Instant::now();
    for (key, _) in &embeddings {
        let _ = cache.get_embedding(key).await?;
    }
    let individual_retrieve_time = start_time.elapsed();

    info!(
        "Performance comparison:\n  Batch store: {:?} vs Individual: {:?} ({}x faster)\n  Batch retrieve: {:?} vs Individual: {:?} ({}x faster)",
        batch_store_time,
        individual_store_time,
        individual_store_time.as_millis() as f64 / batch_store_time.as_millis() as f64,
        batch_retrieve_time,
        individual_retrieve_time,
        individual_retrieve_time.as_millis() as f64 / batch_retrieve_time.as_millis() as f64
    );

    Ok(())
}

/// Demonstrate cache configuration options.
async fn demo_cache_configuration() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n⚙️ Demo 2: Cache Configuration");

    // Create cache configuration
    let config = CacheConfig::new()
        .with_default_ttl(Duration::from_secs(1800)) // 30 minutes
        .with_max_entries(5000)
        .with_compression(true)
        .with_eviction_policy(EvictionPolicy::Lru)
        .with_batch_size(50)
        .with_stats(true);

    info!("Cache configuration:");
    info!("  Default TTL: {:?}", config.default_ttl);
    info!("  Max entries: {}", config.max_entries);
    info!("  Compression: {}", config.enable_compression);
    info!("  Eviction policy: {:?}", config.eviction_policy);
    info!("  Batch size: {}", config.batch_size);
    info!("  Statistics: {}", config.enable_stats);

    // Create caches with different configurations
    let memory_cache = MemoryCache::with_config(config.max_entries, config.default_ttl);
    let file_config = FileCacheConfig {
        max_entries: config.max_entries,
        default_ttl: config.default_ttl,
        enable_compression: config.enable_compression,
        ..Default::default()
    };
    let file_cache = EnhancedFileCache::new("./demo_cache_config", file_config).await?;

    info!("✅ Created caches with custom configuration");

    // Test cache health monitoring
    let health = memory_cache.health().await?;
    info!("Memory cache health: {:?}", health.status);
    info!("  Hit rate: {:.1}%", health.hit_rate);
    info!("  Usage ratio: {:.1}%", health.usage_ratio * 100.0);

    Ok(())
}

/// Demonstrate intelligent cache key generation.
async fn demo_cache_key_generation() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n🔑 Demo 3: Cache Key Generation");

    // Generate embedding keys
    let embedding_key1 = CacheKeyGenerator::embedding_key(
        "This is a sample text for embedding",
        "sentence-transformers/all-MiniLM-L6-v2",
        Some("v1.0"),
    );

    let embedding_key2 = CacheKeyGenerator::embedding_key(
        "This is a sample text for embedding",
        "sentence-transformers/all-MiniLM-L6-v2",
        Some("v1.0"),
    );

    info!("Embedding key 1: {}", embedding_key1);
    info!("Embedding key 2: {}", embedding_key2);
    info!("Keys are identical: {}", embedding_key1 == embedding_key2);

    // Generate nodes key
    let nodes_key = CacheKeyGenerator::nodes_key("document_123", 1000, 200);
    info!("Nodes key: {}", nodes_key);

    // Generate query key
    let mut query_params = HashMap::new();
    query_params.insert("top_k".to_string(), "10".to_string());
    query_params.insert("threshold".to_string(), "0.8".to_string());

    let query_key = CacheKeyGenerator::query_key("What is machine learning?", &query_params);
    info!("Query key: {}", query_key);

    // Generate generic key
    #[derive(serde::Serialize)]
    struct CustomData {
        name: String,
        value: i32,
        tags: Vec<String>,
    }

    let custom_data = CustomData {
        name: "test".to_string(),
        value: 42,
        tags: vec!["ai".to_string(), "ml".to_string()],
    };

    let generic_key = CacheKeyGenerator::generic_key("custom", &custom_data)?;
    info!("Generic key: {}", generic_key);

    info!("✅ Cache key generation completed");

    Ok(())
}

/// Demonstrate performance improvements with unified cache.
async fn demo_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n⚡ Demo 4: Performance Comparison");

    // Create different cache types
    let memory_cache = UnifiedCache::Memory(MemoryCache::new());
    let file_cache = UnifiedCache::file("./demo_perf_cache").await?;

    // Test data
    let test_nodes = create_test_nodes(100);
    let ttl = Duration::from_secs(3600);

    // Test memory cache performance
    info!("Testing memory cache performance...");
    let start_time = Instant::now();

    for (i, node) in test_nodes.iter().enumerate() {
        let key = format!("node_{}", i);
        memory_cache
            .put_nodes(&key, vec![node.clone()], ttl)
            .await?;
    }

    let memory_store_time = start_time.elapsed();

    let start_time = Instant::now();
    for i in 0..test_nodes.len() {
        let key = format!("node_{}", i);
        let _ = memory_cache.get_nodes(&key).await?;
    }
    let memory_retrieve_time = start_time.elapsed();

    info!(
        "Memory cache: store={:?}, retrieve={:?}",
        memory_store_time, memory_retrieve_time
    );

    // Test file cache performance
    info!("Testing file cache performance...");
    let start_time = Instant::now();

    for (i, node) in test_nodes.iter().enumerate() {
        let key = format!("node_{}", i);
        file_cache.put_nodes(&key, vec![node.clone()], ttl).await?;
    }

    let file_store_time = start_time.elapsed();

    let start_time = Instant::now();
    for i in 0..test_nodes.len() {
        let key = format!("node_{}", i);
        let _ = file_cache.get_nodes(&key).await?;
    }
    let file_retrieve_time = start_time.elapsed();

    info!(
        "File cache: store={:?}, retrieve={:?}",
        file_store_time, file_retrieve_time
    );

    // Compare performance
    info!(
        "Performance comparison:\n  Memory vs File store: {:.2}x faster\n  Memory vs File retrieve: {:.2}x faster",
        file_store_time.as_millis() as f64 / memory_store_time.as_millis() as f64,
        file_retrieve_time.as_millis() as f64 / memory_retrieve_time.as_millis() as f64
    );

    // Test cache statistics
    let memory_stats = memory_cache.stats().await?;
    let file_stats = file_cache.stats().await?;

    info!("Cache statistics:");
    info!(
        "  Memory cache: {} hits, {} misses, {:.1}% hit rate",
        memory_stats.hits,
        memory_stats.misses,
        memory_stats.hit_rate()
    );
    info!(
        "  File cache: {} hits, {} misses, {:.1}% hit rate",
        file_stats.hits,
        file_stats.misses,
        file_stats.hit_rate()
    );

    Ok(())
}

/// Create test nodes for performance testing.
fn create_test_nodes(count: usize) -> Vec<Node> {
    (0..count)
        .map(|i| {
            let source_doc_id = Uuid::new_v4();
            let chunk_info = ChunkInfo {
                start_offset: i * 100,
                end_offset: (i + 1) * 100,
                chunk_index: i,
            };

            Node {
                id: Uuid::new_v4(),
                content: format!("This is test node number {} with some sample content.", i),
                embedding: Some(vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]),
                sparse_embedding: None,
                relationships: HashMap::new(),
                source_document_id: source_doc_id,
                chunk_info,
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("index".to_string(), serde_json::Value::Number(i.into()));
                    meta.insert(
                        "type".to_string(),
                        serde_json::Value::String("test".to_string()),
                    );
                    meta
                },
            }
        })
        .collect()
}
