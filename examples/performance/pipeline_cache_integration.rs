//! Demonstration of enhanced pipeline cache integration.
//!
//! This example shows how to use the new pipeline cache integration features
//! including batch operations, intelligent cache key generation, and cache warming.

use cheungfun_core::{
    cache::{
        EmbeddingCacheOps, MemoryCache, NodeCacheOps, PipelineCacheConfig, PipelineCacheManager,
        UnifiedCache,
    },
    traits::CacheKeyGenerator,
    types::{ChunkInfo, Node},
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, Level};
use tracing_subscriber;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🚀 Starting Pipeline Cache Integration Demo");

    // Demo 1: Pipeline Cache Manager Setup
    demo_cache_manager_setup().await?;

    // Demo 2: Embedding Cache Operations
    demo_embedding_cache_operations().await?;

    // Demo 3: Node Cache Operations
    demo_node_cache_operations().await?;

    // Demo 4: Batch Operations Performance
    demo_batch_operations_performance().await?;

    // Demo 5: Cache Key Generation
    demo_cache_key_generation().await?;

    // Demo 6: Cache Statistics and Monitoring
    demo_cache_statistics().await?;

    info!("✅ Pipeline cache integration demo completed successfully!");
    Ok(())
}

/// Demonstrate pipeline cache manager setup with different configurations.
async fn demo_cache_manager_setup() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n⚙️ Demo 1: Pipeline Cache Manager Setup");

    // Create different cache configurations
    let default_config = PipelineCacheConfig::default();
    info!("Default configuration:");
    info!("  Embedding TTL: {:?}", default_config.embedding_ttl);
    info!("  Node TTL: {:?}", default_config.node_ttl);
    info!("  Query TTL: {:?}", default_config.query_ttl);
    info!("  Batch operations: {}", default_config.enable_batch_operations);
    info!("  Batch size: {}", default_config.batch_size);

    // Create custom configuration
    let custom_config = PipelineCacheConfig {
        embedding_ttl: Duration::from_secs(7200), // 2 hours
        node_ttl: Duration::from_secs(3600),      // 1 hour
        query_ttl: Duration::from_secs(1800),     // 30 minutes
        enable_batch_operations: true,
        batch_size: 50,
        enable_cache_warming: true,
        model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        model_version: Some("v1.0".to_string()),
    };

    info!("\nCustom configuration:");
    info!("  Model: {}", custom_config.model_name);
    info!("  Model version: {:?}", custom_config.model_version);
    info!("  Cache warming: {}", custom_config.enable_cache_warming);

    // Create cache manager
    let cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let cache_manager = PipelineCacheManager::new(cache, custom_config);

    info!("✅ Pipeline cache manager created successfully");

    // Test basic functionality
    let stats = cache_manager.stats().await;
    info!("Initial cache statistics:");
    info!("  Overall hit rate: {:.1}%", stats.overall_hit_rate());
    info!("  Embedding operations: {} hits, {} misses", stats.embedding_hits, stats.embedding_misses);

    Ok(())
}

/// Demonstrate embedding cache operations with batch support.
async fn demo_embedding_cache_operations() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n🔤 Demo 2: Embedding Cache Operations");

    let cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let cache_manager = Arc::new(PipelineCacheManager::with_default_config(cache));

    // Test data
    let texts = vec![
        "Machine learning is a subset of artificial intelligence.".to_string(),
        "Deep learning uses neural networks with multiple layers.".to_string(),
        "Natural language processing enables computers to understand text.".to_string(),
        "Computer vision allows machines to interpret visual information.".to_string(),
        "Reinforcement learning trains agents through trial and error.".to_string(),
    ];

    let embeddings: Vec<Vec<f32>> = texts
        .iter()
        .enumerate()
        .map(|(i, _)| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
        .collect();

    // Store embeddings in cache
    info!("Storing {} embeddings in cache...", texts.len());
    let text_embeddings: Vec<(String, Vec<f32>)> = texts
        .iter()
        .cloned()
        .zip(embeddings.iter().cloned())
        .collect();

    let start_time = Instant::now();
    cache_manager
        .store_embeddings_cached(&text_embeddings)
        .await?;
    let store_time = start_time.elapsed();

    info!("Stored embeddings in {:?}", store_time);

    // Retrieve embeddings from cache
    info!("Retrieving {} embeddings from cache...", texts.len());
    let start_time = Instant::now();
    let cached_embeddings = cache_manager.get_embeddings_cached(&texts).await?;
    let retrieve_time = start_time.elapsed();

    info!("Retrieved embeddings in {:?}", retrieve_time);

    // Verify results
    let mut successful_retrievals = 0;
    for (i, cached_embedding) in cached_embeddings.iter().enumerate() {
        if let Some(embedding) = cached_embedding {
            if embedding == &embeddings[i] {
                successful_retrievals += 1;
            }
        }
    }

    info!(
        "✅ Successfully retrieved {}/{} embeddings from cache",
        successful_retrievals,
        texts.len()
    );

    // Check cache statistics
    let (hits, misses) = cache_manager.embedding_cache_stats().await?;
    info!("Embedding cache stats: {} hits, {} misses", hits, misses);

    Ok(())
}

/// Demonstrate node cache operations.
async fn demo_node_cache_operations() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n📄 Demo 3: Node Cache Operations");

    let cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let cache_manager = Arc::new(PipelineCacheManager::with_default_config(cache));

    // Create test nodes
    let document_id = "test_document_123";
    let chunk_size = 1000;
    let overlap = 200;
    let nodes = create_test_nodes(5);

    info!(
        "Storing {} nodes for document: {}",
        nodes.len(),
        document_id
    );

    // Store nodes in cache
    let start_time = Instant::now();
    cache_manager
        .store_nodes_cached(document_id, chunk_size, overlap, nodes.clone())
        .await?;
    let store_time = start_time.elapsed();

    info!("Stored nodes in {:?}", store_time);

    // Retrieve nodes from cache
    info!("Retrieving nodes from cache...");
    let start_time = Instant::now();
    let cached_nodes = cache_manager
        .get_nodes_cached(document_id, chunk_size, overlap)
        .await?;
    let retrieve_time = start_time.elapsed();

    info!("Retrieved nodes in {:?}", retrieve_time);

    // Verify results
    if let Some(retrieved_nodes) = cached_nodes {
        if retrieved_nodes.len() == nodes.len() {
            info!("✅ Successfully retrieved all {} nodes from cache", nodes.len());
        } else {
            info!(
                "⚠️ Retrieved {} nodes, expected {}",
                retrieved_nodes.len(),
                nodes.len()
            );
        }
    } else {
        info!("❌ Failed to retrieve nodes from cache");
    }

    // Check cache statistics
    let (hits, misses) = cache_manager.node_cache_stats().await?;
    info!("Node cache stats: {} hits, {} misses", hits, misses);

    Ok(())
}

/// Demonstrate batch operations performance benefits.
async fn demo_batch_operations_performance() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n⚡ Demo 4: Batch Operations Performance");

    // Create cache managers with and without batch operations
    let cache1 = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let batch_config = PipelineCacheConfig {
        enable_batch_operations: true,
        batch_size: 50,
        ..Default::default()
    };
    let batch_manager = Arc::new(PipelineCacheManager::new(cache1, batch_config));

    let cache2 = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let individual_config = PipelineCacheConfig {
        enable_batch_operations: false,
        ..Default::default()
    };
    let individual_manager = Arc::new(PipelineCacheManager::new(cache2, individual_config));

    // Test data
    let texts: Vec<String> = (0..100)
        .map(|i| format!("This is test text number {} for performance testing.", i))
        .collect();

    let embeddings: Vec<Vec<f32>> = (0..100)
        .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
        .collect();

    let text_embeddings: Vec<(String, Vec<f32>)> = texts
        .iter()
        .cloned()
        .zip(embeddings.iter().cloned())
        .collect();

    // Test batch operations
    info!("Testing batch operations...");
    let start_time = Instant::now();
    batch_manager
        .store_embeddings_cached(&text_embeddings)
        .await?;
    let batch_store_time = start_time.elapsed();

    let start_time = Instant::now();
    let _batch_results = batch_manager.get_embeddings_cached(&texts).await?;
    let batch_retrieve_time = start_time.elapsed();

    // Test individual operations
    info!("Testing individual operations...");
    let start_time = Instant::now();
    individual_manager
        .store_embeddings_cached(&text_embeddings)
        .await?;
    let individual_store_time = start_time.elapsed();

    let start_time = Instant::now();
    let _individual_results = individual_manager.get_embeddings_cached(&texts).await?;
    let individual_retrieve_time = start_time.elapsed();

    // Compare performance
    info!("Performance comparison:");
    info!(
        "  Store operations: Batch {:?} vs Individual {:?} ({:.2}x faster)",
        batch_store_time,
        individual_store_time,
        individual_store_time.as_millis() as f64 / batch_store_time.as_millis() as f64
    );
    info!(
        "  Retrieve operations: Batch {:?} vs Individual {:?} ({:.2}x faster)",
        batch_retrieve_time,
        individual_retrieve_time,
        individual_retrieve_time.as_millis() as f64 / batch_retrieve_time.as_millis() as f64
    );

    Ok(())
}

/// Demonstrate intelligent cache key generation.
async fn demo_cache_key_generation() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n🔑 Demo 5: Cache Key Generation");

    // Test embedding keys
    let text = "This is a sample text for embedding";
    let model = "sentence-transformers/all-MiniLM-L6-v2";
    let version = Some("v1.0");

    let key1 = CacheKeyGenerator::embedding_key(text, model, version);
    let key2 = CacheKeyGenerator::embedding_key(text, model, version);
    let key3 = CacheKeyGenerator::embedding_key(text, model, Some("v2.0"));

    info!("Embedding key generation:");
    info!("  Key 1: {}", key1);
    info!("  Key 2: {}", key2);
    info!("  Key 3 (different version): {}", key3);
    info!("  Keys 1 and 2 are identical: {}", key1 == key2);
    info!("  Keys 1 and 3 are different: {}", key1 != key3);

    // Test node keys
    let document_id = "document_123";
    let chunk_size = 1000;
    let overlap = 200;

    let node_key1 = CacheKeyGenerator::nodes_key(document_id, chunk_size, overlap);
    let node_key2 = CacheKeyGenerator::nodes_key(document_id, chunk_size, overlap);
    let node_key3 = CacheKeyGenerator::nodes_key(document_id, 500, overlap);

    info!("\nNode key generation:");
    info!("  Key 1: {}", node_key1);
    info!("  Key 2: {}", node_key2);
    info!("  Key 3 (different chunk size): {}", node_key3);
    info!("  Keys 1 and 2 are identical: {}", node_key1 == node_key2);
    info!("  Keys 1 and 3 are different: {}", node_key1 != node_key3);

    // Test query keys
    let query = "What is machine learning?";
    let mut params1 = HashMap::new();
    params1.insert("top_k".to_string(), "10".to_string());
    params1.insert("threshold".to_string(), "0.8".to_string());

    let mut params2 = HashMap::new();
    params2.insert("top_k".to_string(), "10".to_string());
    params2.insert("threshold".to_string(), "0.8".to_string());

    let mut params3 = HashMap::new();
    params3.insert("top_k".to_string(), "5".to_string());
    params3.insert("threshold".to_string(), "0.8".to_string());

    let query_key1 = CacheKeyGenerator::query_key(query, &params1);
    let query_key2 = CacheKeyGenerator::query_key(query, &params2);
    let query_key3 = CacheKeyGenerator::query_key(query, &params3);

    info!("\nQuery key generation:");
    info!("  Key 1: {}", query_key1);
    info!("  Key 2: {}", query_key2);
    info!("  Key 3 (different top_k): {}", query_key3);
    info!("  Keys 1 and 2 are identical: {}", query_key1 == query_key2);
    info!("  Keys 1 and 3 are different: {}", query_key1 != query_key3);

    Ok(())
}

/// Demonstrate cache statistics and monitoring.
async fn demo_cache_statistics() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n📊 Demo 6: Cache Statistics and Monitoring");

    let cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let cache_manager = Arc::new(PipelineCacheManager::with_default_config(cache));

    // Perform various cache operations
    let texts = vec![
        "First text".to_string(),
        "Second text".to_string(),
        "Third text".to_string(),
    ];
    let embeddings = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];

    // Store some embeddings
    let text_embeddings: Vec<(String, Vec<f32>)> = texts
        .iter()
        .cloned()
        .zip(embeddings.iter().cloned())
        .collect();
    cache_manager
        .store_embeddings_cached(&text_embeddings)
        .await?;

    // Retrieve embeddings (should be cache hits)
    let _cached_embeddings = cache_manager.get_embeddings_cached(&texts).await?;

    // Try to retrieve non-existent embeddings (should be cache misses)
    let missing_texts = vec!["Missing text 1".to_string(), "Missing text 2".to_string()];
    let _missing_embeddings = cache_manager.get_embeddings_cached(&missing_texts).await?;

    // Store and retrieve some nodes
    let nodes = create_test_nodes(3);
    cache_manager
        .store_nodes_cached("doc1", 1000, 200, nodes)
        .await?;
    let _cached_nodes = cache_manager.get_nodes_cached("doc1", 1000, 200).await?;
    let _missing_nodes = cache_manager.get_nodes_cached("doc2", 1000, 200).await?;

    // Get comprehensive statistics
    let stats = cache_manager.stats().await;

    info!("Comprehensive cache statistics:");
    info!("  Overall hit rate: {:.1}%", stats.overall_hit_rate());
    info!("  Embedding hit rate: {:.1}%", stats.embedding_hit_rate());
    info!("  Node hit rate: {:.1}%", stats.node_hit_rate());
    info!("  Query hit rate: {:.1}%", stats.query_hit_rate());
    info!("\nDetailed statistics:");
    info!(
        "  Embeddings: {} hits, {} misses, {} stores",
        stats.embedding_hits, stats.embedding_misses, stats.embedding_stores
    );
    info!(
        "  Nodes: {} hits, {} misses, {} stores",
        stats.node_hits, stats.node_misses, stats.node_stores
    );
    info!(
        "  Queries: {} hits, {} misses, {} stores",
        stats.query_hits, stats.query_misses, stats.query_stores
    );
    info!(
        "  Batch operations: {} operations, {} items processed",
        stats.batch_operations, stats.batch_items_processed
    );

    Ok(())
}

/// Create test nodes for demonstration purposes.
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
                content: format!("This is test node number {} with sample content.", i),
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
