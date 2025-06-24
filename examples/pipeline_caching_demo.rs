//! Demonstration of pipeline-level caching in Cheungfun.
//!
//! This example shows how to integrate caching into indexing pipelines
//! to dramatically speed up development workflows.

use cheungfun_core::{
    Document, Node,
    cache::{FileCache, MemoryCache},
    traits::PipelineCache,
};
use cheungfun_indexing::pipeline::{DefaultIndexingPipeline, PipelineConfig};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{Level, info};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🚀 Starting Pipeline Caching Demo");

    // Demo 1: Pipeline without cache
    demo_pipeline_without_cache().await?;

    // Demo 2: Pipeline with memory cache
    demo_pipeline_with_memory_cache().await?;

    // Demo 3: Pipeline with file cache
    demo_pipeline_with_file_cache().await?;

    // Demo 4: Cache performance comparison
    demo_cache_performance_comparison().await?;

    info!("✅ Pipeline caching demo completed successfully!");
    Ok(())
}

/// Demonstrate pipeline without caching.
async fn demo_pipeline_without_cache() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n🐌 Demo 1: Pipeline without cache");

    let documents = create_sample_documents();

    // Create a simple mock pipeline configuration
    let config = PipelineConfig {
        enable_caching: false,
        ..Default::default()
    };

    info!("Processing {} documents without cache...", documents.len());
    let start_time = Instant::now();

    // Simulate processing (in a real scenario, you'd build a complete pipeline)
    simulate_document_processing(&documents, None).await?;

    let duration = start_time.elapsed();
    info!("⏱️  Processing time without cache: {:?}", duration);

    Ok(())
}

/// Demonstrate pipeline with memory cache.
async fn demo_pipeline_with_memory_cache() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n🧠 Demo 2: Pipeline with memory cache");

    let documents = create_sample_documents();
    let cache = Arc::new(MemoryCache::new())
        as Arc<dyn PipelineCache<Error = cheungfun_core::CheungfunError>>;

    let config = PipelineConfig {
        enable_caching: true,
        cache_ttl_seconds: 3600, // 1 hour
        ..Default::default()
    };

    info!(
        "Processing {} documents with memory cache...",
        documents.len()
    );

    // First run (cold cache)
    info!("🥶 First run (cold cache):");
    let start_time = Instant::now();
    simulate_document_processing(&documents, Some(cache.clone())).await?;
    let cold_duration = start_time.elapsed();
    info!("   Time: {:?}", cold_duration);

    // Second run (warm cache)
    info!("🔥 Second run (warm cache):");
    let start_time = Instant::now();
    simulate_document_processing(&documents, Some(cache.clone())).await?;
    let warm_duration = start_time.elapsed();
    info!("   Time: {:?}", warm_duration);

    // Show cache statistics
    let stats = cache.stats().await?;
    info!(
        "📊 Memory cache stats: hit rate = {:.1}%, total entries = {}",
        stats.hit_rate(),
        stats.total_entries
    );

    let speedup = cold_duration.as_millis() as f64 / warm_duration.as_millis() as f64;
    info!("🚀 Speedup with memory cache: {:.1}x", speedup);

    Ok(())
}

/// Demonstrate pipeline with file cache.
async fn demo_pipeline_with_file_cache() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n💾 Demo 3: Pipeline with file cache");

    let documents = create_sample_documents();
    let cache_dir = "./pipeline_cache";
    let cache = Arc::new(FileCache::with_default_config(cache_dir).await?)
        as Arc<dyn PipelineCache<Error = cheungfun_core::CheungfunError>>;

    let config = PipelineConfig {
        enable_caching: true,
        cache_ttl_seconds: 7200, // 2 hours
        ..Default::default()
    };

    info!("Cache directory: {}", cache_dir);
    info!(
        "Processing {} documents with file cache...",
        documents.len()
    );

    // First run (populate cache)
    info!("💾 First run (populating file cache):");
    let start_time = Instant::now();
    simulate_document_processing(&documents, Some(cache.clone())).await?;
    let first_duration = start_time.elapsed();
    info!("   Time: {:?}", first_duration);

    // Second run (use cache)
    info!("⚡ Second run (using file cache):");
    let start_time = Instant::now();
    simulate_document_processing(&documents, Some(cache.clone())).await?;
    let second_duration = start_time.elapsed();
    info!("   Time: {:?}", second_duration);

    // Show cache health
    let health = cache.health().await?;
    info!(
        "🏥 File cache health: {:?}, hit rate = {:.1}%",
        health.status, health.hit_rate
    );

    let speedup = first_duration.as_millis() as f64 / second_duration.as_millis() as f64;
    info!("🚀 Speedup with file cache: {:.1}x", speedup);

    Ok(())
}

/// Compare cache performance across different scenarios.
async fn demo_cache_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n⚡ Demo 4: Cache Performance Comparison");

    let documents = create_sample_documents();

    // Test scenarios
    let scenarios = vec![
        ("No Cache", None),
        (
            "Memory Cache",
            Some(Arc::new(MemoryCache::new())
                as Arc<
                    dyn PipelineCache<Error = cheungfun_core::CheungfunError>,
                >),
        ),
        (
            "Enhanced File Cache",
            Some(
                Arc::new(FileCache::with_default_config("./comparison_cache").await?)
                    as Arc<dyn PipelineCache<Error = cheungfun_core::CheungfunError>>,
            ),
        ),
    ];

    let mut results = Vec::new();

    for (name, cache) in scenarios {
        info!("🧪 Testing scenario: {}", name);

        // Run twice to measure cache effectiveness
        let mut durations = Vec::new();

        for run in 1..=2 {
            let start_time = Instant::now();
            simulate_document_processing(&documents, cache.clone()).await?;
            let duration = start_time.elapsed();
            durations.push(duration);
            info!("   Run {}: {:?}", run, duration);
        }

        results.push((name, durations));
    }

    // Summary
    info!("\n📊 Performance Summary:");
    info!("┌─────────────┬─────────────┬─────────────┬─────────────┐");
    info!("│ Scenario    │ First Run   │ Second Run  │ Improvement │");
    info!("├─────────────┼─────────────┼─────────────┼─────────────┤");

    for (name, durations) in results {
        let first = durations[0];
        let second = durations[1];
        let improvement = if second.as_millis() > 0 {
            first.as_millis() as f64 / second.as_millis() as f64
        } else {
            1.0
        };

        info!(
            "│ {:11} │ {:9}ms │ {:9}ms │ {:9.1}x │",
            name,
            first.as_millis(),
            second.as_millis(),
            improvement
        );
    }

    info!("└─────────────┴─────────────┴─────────────┴─────────────┘");

    Ok(())
}

/// Create sample documents for testing.
fn create_sample_documents() -> Vec<Document> {
    vec![
        Document::new(
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        ),
        Document::new("Deep learning uses neural networks with multiple layers to process data."),
        Document::new(
            "Natural language processing helps computers understand and generate human language.",
        ),
        Document::new(
            "Computer vision enables machines to interpret and analyze visual information.",
        ),
        Document::new("Reinforcement learning trains agents through trial and error interactions."),
    ]
}

/// Simulate document processing with optional caching.
async fn simulate_document_processing(
    documents: &[Document],
    cache: Option<Arc<dyn PipelineCache<Error = cheungfun_core::CheungfunError>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let ttl = Duration::from_secs(3600);

    for (i, document) in documents.iter().enumerate() {
        let cache_key = format!("doc_embedding_{}", i);

        // Check cache first if available
        if let Some(ref cache) = cache {
            if let Ok(Some(_cached_embedding)) = cache.get_embedding(&cache_key).await {
                // Cache hit - no computation needed
                continue;
            }
        }

        // Simulate expensive embedding computation
        let embedding = simulate_embedding_computation(&document.content).await;

        // Store in cache if available
        if let Some(ref cache) = cache {
            let _ = cache.put_embedding(&cache_key, embedding, ttl).await;
        }
    }

    Ok(())
}

/// Simulate expensive embedding computation.
async fn simulate_embedding_computation(text: &str) -> Vec<f32> {
    // Simulate computation time
    tokio::time::sleep(Duration::from_millis(50)).await;

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
