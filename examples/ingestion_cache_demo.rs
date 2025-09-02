//! Ingestion Cache Demonstration
//!
//! This example demonstrates the new IngestionCache that provides
//! transformation-level caching following LlamaIndex's design exactly.
//! This significantly improves pipeline performance for repeated processing.

use std::sync::Arc;
use std::time::Instant;
use tracing::{info, Level};
use tracing_subscriber;

use cheungfun_core::{
    traits::{TypedData, TypedTransform},
    types::{ChunkInfo, Document, Node},
};
use cheungfun_indexing::{
    cache::{IngestionCache, SimpleCacheBackend, TransformationHasher},
    node_parser::text::SentenceSplitter,
    transformers::TitleExtractor,
};
use siumai::prelude::*;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🗄️ IngestionCache Demo");

    // Demo 1: Basic Cache Operations
    info!("\n📋 Demo 1: Basic Cache Operations");
    demo_basic_cache_operations().await?;

    // Demo 2: Transformation Caching
    info!("\n📋 Demo 2: Transformation Caching");
    demo_transformation_caching().await?;

    // Demo 3: Cache Persistence
    info!("\n📋 Demo 3: Cache Persistence");
    demo_cache_persistence().await?;

    // Demo 4: Performance Comparison
    info!("\n📋 Demo 4: Performance Comparison (With vs Without Cache)");
    demo_performance_comparison().await?;

    info!("\n✅ All demos completed successfully!");
    Ok(())
}

/// Demo 1: Basic cache operations
async fn demo_basic_cache_operations() -> Result<(), Box<dyn std::error::Error>> {
    info!("Creating IngestionCache with simple backend...");

    // Create cache with simple backend
    let backend = Arc::new(SimpleCacheBackend::new());
    let cache = IngestionCache::new(backend);

    // Create sample nodes
    let nodes = create_sample_nodes();
    let cache_key = "test_transformation_hash_123";

    info!("Storing {} nodes in cache...", nodes.len());

    // Store nodes in cache
    cache.put(cache_key, nodes.clone(), None).await?;

    // Retrieve nodes from cache
    let cached_nodes = cache.get(cache_key, None).await?;

    info!("✅ Cache operations completed!");
    match cached_nodes {
        Some(retrieved_nodes) => {
            info!("📄 Retrieved {} nodes from cache", retrieved_nodes.len());
            info!(
                "🔍 Content matches: {}",
                nodes[0].content == retrieved_nodes[0].content
            );
        }
        None => {
            info!("❌ No nodes found in cache");
        }
    }

    // Get cache statistics
    let stats = cache.stats(None).await?;
    info!(
        "📊 Cache stats: {} total, {} active entries",
        stats.total_entries, stats.active_entries
    );

    Ok(())
}

/// Demo 2: Transformation caching with hash generation
async fn demo_transformation_caching() -> Result<(), Box<dyn std::error::Error>> {
    info!("Creating transformation cache with hash generation...");

    let cache = IngestionCache::simple();
    let nodes = create_sample_nodes();

    // Create a metadata extractor transformation (processes nodes -> nodes)
    let metadata_extractor = cheungfun_indexing::transformers::MetadataExtractor::new();

    // Generate transformation hash (following LlamaIndex's approach)
    let transform_hash = TransformationHasher::hash(&nodes, "MetadataExtractor");
    info!(
        "🔑 Generated transformation hash: {}",
        &transform_hash[..16]
    );

    // Check if transformation result is cached
    if let Some(cached_result) = cache.get(&transform_hash, None).await? {
        info!("🎯 Cache hit! Using cached transformation result");
        info!("📄 Cached result has {} nodes", cached_result.len());
    } else {
        info!("❌ Cache miss. Running transformation...");

        // Run the transformation
        let start_time = Instant::now();
        let input = TypedData::from_nodes(nodes);
        let result = metadata_extractor.transform(input).await?;
        let result_nodes = result.into_nodes();
        let transform_time = start_time.elapsed();

        info!("⏱️ Transformation took: {:?}", transform_time);
        info!("📄 Transformation produced {} nodes", result_nodes.len());

        // Cache the result
        cache.put(&transform_hash, result_nodes, None).await?;
        info!("💾 Cached transformation result");
    }

    // Try the same transformation again (should hit cache)
    info!("\n🔄 Running same transformation again...");
    let nodes_again = create_sample_nodes();
    let hash_again = TransformationHasher::hash(&nodes_again, "MetadataExtractor");

    if let Some(cached_result) = cache.get(&hash_again, None).await? {
        info!(
            "🎯 Cache hit! Retrieved {} nodes instantly",
            cached_result.len()
        );
    } else {
        info!("❌ Unexpected cache miss");
    }

    Ok(())
}

/// Demo 3: Cache persistence
async fn demo_cache_persistence() -> Result<(), Box<dyn std::error::Error>> {
    info!("Testing cache persistence...");

    let cache_file = "./temp_cache.json";

    // Create cache and add some data
    {
        let cache = IngestionCache::simple();
        let nodes = create_sample_nodes();

        cache.put("persistent_key_1", nodes.clone(), None).await?;
        cache
            .put("persistent_key_2", nodes, Some("custom_collection"))
            .await?;

        info!("💾 Persisting cache to file...");
        cache.persist(cache_file).await?;
    }

    // Load cache from file
    {
        info!("📂 Loading cache from file...");
        let loaded_cache = IngestionCache::from_persist_path(cache_file).await?;

        // Check if data was persisted correctly
        if let Some(nodes) = loaded_cache.get("persistent_key_1", None).await? {
            info!(
                "✅ Successfully loaded {} nodes from default collection",
                nodes.len()
            );
        }

        if let Some(nodes) = loaded_cache
            .get("persistent_key_2", Some("custom_collection"))
            .await?
        {
            info!(
                "✅ Successfully loaded {} nodes from custom collection",
                nodes.len()
            );
        }

        let stats = loaded_cache.stats(None).await?;
        info!(
            "📊 Loaded cache stats: {} total entries",
            stats.total_entries
        );
    }

    // Clean up
    if std::path::Path::new(cache_file).exists() {
        std::fs::remove_file(cache_file)?;
        info!("🧹 Cleaned up cache file");
    }

    Ok(())
}

/// Demo 4: Performance comparison with and without cache
async fn demo_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // Check for API key for LLM-based transformations
    let has_api_key =
        std::env::var("OPENAI_API_KEY").is_ok() || std::env::var("ANTHROPIC_API_KEY").is_ok();

    if !has_api_key {
        info!("⚠️ No API key found. Skipping LLM-based performance demo");
        info!("💡 Set OPENAI_API_KEY or ANTHROPIC_API_KEY to see full demo");
        return demo_performance_with_splitter().await;
    }

    info!("Running performance comparison with LLM transformations...");

    let cache = IngestionCache::simple();
    let nodes = create_sample_nodes();

    // Create LLM client for transformations
    let llm_client = create_mock_llm_client().await?;
    let title_extractor = TitleExtractor::with_defaults(llm_client)?;

    // First run (no cache)
    info!("🏃 First run (no cache)...");
    let start_time = Instant::now();
    let transform_hash = TransformationHasher::hash(&nodes, "TitleExtractor");
    let input = TypedData::from_nodes(nodes.clone());
    let result = title_extractor.transform(input).await?;
    let result1 = result.into_nodes();
    let first_run_time = start_time.elapsed();

    // Cache the result
    cache.put(&transform_hash, result1, None).await?;

    info!("⏱️ First run took: {:?}", first_run_time);

    // Second run (with cache)
    info!("🏃 Second run (with cache)...");
    let start_time = Instant::now();
    let cached_result = cache.get(&transform_hash, None).await?;
    let second_run_time = start_time.elapsed();

    info!("⏱️ Second run took: {:?}", second_run_time);

    if cached_result.is_some() {
        let speedup = first_run_time.as_millis() as f64 / second_run_time.as_millis() as f64;
        info!("🚀 Cache speedup: {:.2}x faster", speedup);
    }

    Ok(())
}

/// Performance demo with text splitter (no API key required)
async fn demo_performance_with_splitter() -> Result<(), Box<dyn std::error::Error>> {
    info!("Running performance comparison with text splitter...");

    let cache = IngestionCache::simple();

    // Create larger document for more noticeable performance difference
    let large_content = "This is a sample document. ".repeat(1000);
    let large_doc = Document::new(&large_content);

    let splitter = SentenceSplitter::from_defaults(100, 20)?;

    // First run (no cache)
    info!("🏃 First run (no cache)...");
    let start_time = Instant::now();
    let transform_hash = TransformationHasher::content_hash(&large_content);
    let input = TypedData::from_documents(vec![large_doc.clone()]);
    let result = splitter.transform(input).await?;
    let result1 = result.into_nodes();
    let first_run_time = start_time.elapsed();

    // Cache the result
    cache.put(&transform_hash, result1, None).await?;

    info!("⏱️ First run took: {:?}", first_run_time);
    info!(
        "📄 Produced {} nodes",
        cache.get(&transform_hash, None).await?.unwrap().len()
    );

    // Second run (with cache)
    info!("🏃 Second run (with cache)...");
    let start_time = Instant::now();
    let cached_result = cache.get(&transform_hash, None).await?;
    let second_run_time = start_time.elapsed();

    info!("⏱️ Second run took: {:?}", second_run_time);

    if let Some(nodes) = cached_result {
        let speedup = first_run_time.as_millis() as f64 / second_run_time.as_millis() as f64;
        info!("🚀 Cache speedup: {:.2}x faster", speedup);
        info!("📄 Retrieved {} nodes from cache", nodes.len());
    }

    Ok(())
}

/// Create sample nodes for testing
fn create_sample_nodes() -> Vec<Node> {
    let contents = vec![
        "Artificial intelligence is transforming the world of technology.",
        "Machine learning algorithms can process vast amounts of data.",
        "Deep learning uses neural networks to understand complex patterns.",
    ];

    contents
        .into_iter()
        .enumerate()
        .map(|(i, content)| {
            Node::new(
                content.to_string(),
                Uuid::new_v4(),
                ChunkInfo {
                    start_char_idx: Some(i * 100),
                    end_char_idx: Some((i + 1) * 100),
                    chunk_index: i,
                },
            )
        })
        .collect()
}

/// Create a mock LLM client for testing
async fn create_mock_llm_client() -> Result<Arc<dyn LlmClient>, Box<dyn std::error::Error>> {
    // Try to create a real client if API key is available
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let client = Siumai::builder()
            .openai()
            .api_key(api_key)
            .model("gpt-4o-mini")
            .build()
            .await?;
        return Ok(Arc::new(client));
    }

    // Fallback to a mock implementation
    Err("No API key available for LLM client".into())
}
