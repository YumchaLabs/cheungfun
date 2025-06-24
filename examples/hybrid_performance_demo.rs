//! Demonstration of hybrid Tokio + Rayon performance optimization.
//!
//! This example shows the practical benefits of using the hybrid strategy
//! that combines Tokio for I/O operations and Rayon for CPU-intensive tasks.

use cheungfun_core::{
    cache::{MemoryCache, ParallelStrategy, PerformanceCache, PerformanceCacheConfig, UnifiedCache},
    traits::PipelineCache,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🚀 Starting Hybrid Performance Demo");

    // Demo 1: Strategy Comparison
    demo_strategy_comparison().await?;

    // Demo 2: SIMD Performance
    demo_simd_performance().await?;

    // Demo 3: Adaptive Thresholds
    demo_adaptive_thresholds().await?;

    // Demo 4: Real-world Workload Simulation
    demo_real_world_workload().await?;

    info!("✅ Hybrid performance demo completed successfully!");
    Ok(())
}

/// Compare different parallel processing strategies.
async fn demo_strategy_comparison() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n📊 Demo 1: Strategy Comparison");

    let test_size = 1000;
    let ttl = Duration::from_secs(3600);

    // Generate test data
    let embeddings: Vec<Vec<f32>> = (0..test_size)
        .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
        .collect();

    let keys: Vec<String> = (0..test_size).map(|i| format!("strategy_key_{}", i)).collect();

    let strategies = vec![
        ("Tokio-only", ParallelStrategy::TokioOnly),
        ("Rayon-only", ParallelStrategy::RayonOnly),
        ("Hybrid", ParallelStrategy::Hybrid),
        ("Adaptive", ParallelStrategy::Adaptive),
    ];

    for (name, strategy) in strategies {
        info!("\nTesting {} strategy:", name);

        let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
        let config = PerformanceCacheConfig {
            parallel_strategy: strategy,
            enable_simd: true,
            cpu_intensive_threshold: 100,
            io_intensive_threshold: 50,
            optimal_batch_size: 64,
            ..Default::default()
        };
        let cache = PerformanceCache::new(base_cache, config);

        // Test batch write operations
        let batch_items: Vec<(&str, Vec<f32>, Duration)> = keys
            .iter()
            .zip(embeddings.iter())
            .map(|(key, embedding)| (key.as_str(), embedding.clone(), ttl))
            .collect();

        let start_time = Instant::now();
        cache.put_embeddings_batch(&batch_items).await?;
        let write_time = start_time.elapsed();

        // Test batch read operations
        let key_refs: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
        let start_time = Instant::now();
        let _results = cache.get_embeddings_batch(&key_refs).await?;
        let read_time = start_time.elapsed();

        let total_time = write_time + read_time;
        let throughput = (test_size * 2) as f64 / total_time.as_secs_f64();

        info!("  Write time: {:?}", write_time);
        info!("  Read time: {:?}", read_time);
        info!("  Total time: {:?}", total_time);
        info!("  Throughput: {:.1} ops/sec", throughput);

        // Get performance metrics
        let metrics = cache.metrics().await;
        info!("  Parallel operations: {}", metrics.parallel_operations);
        info!("  SIMD operations: {}", metrics.simd_operations);
        info!("  Performance improvement: {:.1}%", metrics.performance_improvement());
    }

    Ok(())
}

/// Demonstrate SIMD performance benefits.
async fn demo_simd_performance() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n⚡ Demo 2: SIMD Performance");

    let vector_sizes = vec![64, 128, 256, 512];
    let num_vectors = 500;

    for vector_size in vector_sizes {
        info!("\nTesting {}-dimensional vectors:", vector_size);

        // Generate test embeddings
        let embeddings: Vec<Vec<f32>> = (0..num_vectors)
            .map(|i| (0..vector_size).map(|j| (i * vector_size + j) as f32 * 0.001).collect())
            .collect();

        let query_embedding: Vec<f32> = (0..vector_size).map(|i| i as f32 * 0.002).collect();

        // Test with SIMD enabled
        let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
        let simd_config = PerformanceCacheConfig {
            enable_simd: true,
            parallel_strategy: ParallelStrategy::Hybrid,
            cpu_intensive_threshold: 50,
            ..Default::default()
        };
        let simd_cache = PerformanceCache::new(base_cache, simd_config);

        let start_time = Instant::now();
        let _simd_results = simd_cache
            .batch_similarity_search(&query_embedding, &embeddings)
            .await;
        let simd_time = start_time.elapsed();

        // Test with SIMD disabled
        let base_cache2 = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
        let no_simd_config = PerformanceCacheConfig {
            enable_simd: false,
            parallel_strategy: ParallelStrategy::Hybrid,
            cpu_intensive_threshold: 50,
            ..Default::default()
        };
        let no_simd_cache = PerformanceCache::new(base_cache2, no_simd_config);

        let start_time = Instant::now();
        let _no_simd_results = no_simd_cache
            .batch_similarity_search(&query_embedding, &embeddings)
            .await;
        let no_simd_time = start_time.elapsed();

        // Compare performance
        let speedup = no_simd_time.as_millis() as f64 / simd_time.as_millis() as f64;
        info!("  SIMD enabled:  {:?}", simd_time);
        info!("  SIMD disabled: {:?}", no_simd_time);
        info!("  SIMD speedup:  {:.2}x", speedup);

        // Get SIMD utilization
        let simd_metrics = simd_cache.metrics().await;
        let no_simd_metrics = no_simd_cache.metrics().await;
        info!("  SIMD utilization: {:.1}% vs {:.1}%", 
            simd_metrics.simd_utilization(), 
            no_simd_metrics.simd_utilization()
        );
    }

    Ok(())
}

/// Demonstrate adaptive threshold tuning.
async fn demo_adaptive_thresholds() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n🎯 Demo 3: Adaptive Thresholds");

    let test_sizes = vec![10, 50, 100, 200, 500];
    let ttl = Duration::from_secs(3600);

    // Test different CPU-intensive thresholds
    let cpu_thresholds = vec![50, 100, 200];

    for threshold in cpu_thresholds {
        info!("\nTesting CPU-intensive threshold: {}", threshold);

        for size in &test_sizes {
            let embeddings: Vec<Vec<f32>> = (0..*size)
                .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
                .collect();

            let keys: Vec<String> = (0..*size).map(|i| format!("adaptive_key_{}", i)).collect();

            let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
            let config = PerformanceCacheConfig {
                parallel_strategy: ParallelStrategy::Adaptive,
                cpu_intensive_threshold: threshold,
                io_intensive_threshold: 20,
                enable_simd: true,
                ..Default::default()
            };
            let cache = PerformanceCache::new(base_cache, config);

            // Test batch operations
            let batch_items: Vec<(&str, Vec<f32>, Duration)> = keys
                .iter()
                .zip(embeddings.iter())
                .map(|(key, embedding)| (key.as_str(), embedding.clone(), ttl))
                .collect();

            let start_time = Instant::now();
            cache.put_embeddings_batch(&batch_items).await?;
            let operation_time = start_time.elapsed();

            let metrics = cache.metrics().await;
            let strategy_used = if metrics.parallel_operations > 0 {
                "Parallel (Rayon)"
            } else {
                "Sequential"
            };

            info!(
                "    Size {}: {:?} - Strategy: {}",
                size, operation_time, strategy_used
            );
        }
    }

    Ok(())
}

/// Simulate real-world workload with mixed operations.
async fn demo_real_world_workload() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n🌍 Demo 4: Real-world Workload Simulation");

    // Simulate a document processing pipeline
    let document_count = 100;
    let embeddings_per_doc = 10;
    let similarity_queries = 50;
    let ttl = Duration::from_secs(3600);

    info!("Simulating processing of {} documents", document_count);
    info!("Each document has {} embeddings", embeddings_per_doc);
    info!("Performing {} similarity queries", similarity_queries);

    let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let config = PerformanceCacheConfig {
        parallel_strategy: ParallelStrategy::Hybrid,
        enable_simd: true,
        cpu_intensive_threshold: 50,
        io_intensive_threshold: 20,
        optimal_batch_size: 32,
        enable_prefetching: true,
        prefetch_cache_size: 500,
        ..Default::default()
    };
    let cache = PerformanceCache::new(base_cache, config);

    let total_start = Instant::now();

    // Phase 1: Document ingestion (I/O intensive)
    info!("\nPhase 1: Document ingestion");
    let ingestion_start = Instant::now();

    for doc_id in 0..document_count {
        let doc_embeddings: Vec<Vec<f32>> = (0..embeddings_per_doc)
            .map(|i| {
                (0..128)
                    .map(|j| (doc_id * embeddings_per_doc + i * 128 + j) as f32 * 0.001)
                    .collect()
            })
            .collect();

        let doc_keys: Vec<String> = (0..embeddings_per_doc)
            .map(|i| format!("doc_{}_{}", doc_id, i))
            .collect();

        let batch_items: Vec<(&str, Vec<f32>, Duration)> = doc_keys
            .iter()
            .zip(doc_embeddings.iter())
            .map(|(key, embedding)| (key.as_str(), embedding.clone(), ttl))
            .collect();

        cache.put_embeddings_batch(&batch_items).await?;
    }

    let ingestion_time = ingestion_start.elapsed();
    info!("  Ingestion completed in: {:?}", ingestion_time);

    // Phase 2: Similarity search (CPU intensive)
    info!("\nPhase 2: Similarity search");
    let search_start = Instant::now();

    // Generate query embeddings
    let query_embeddings: Vec<Vec<f32>> = (0..similarity_queries)
        .map(|i| (0..128).map(|j| (i * 128 + j) as f32 * 0.002).collect())
        .collect();

    // Get all stored embeddings for similarity comparison
    let all_keys: Vec<String> = (0..document_count)
        .flat_map(|doc_id| {
            (0..embeddings_per_doc).map(move |i| format!("doc_{}_{}", doc_id, i))
        })
        .collect();

    let key_refs: Vec<&str> = all_keys.iter().map(|k| k.as_str()).collect();
    let stored_embeddings = cache.get_embeddings_batch(&key_refs).await?;

    // Extract valid embeddings
    let valid_embeddings: Vec<Vec<f32>> = stored_embeddings
        .into_iter()
        .filter_map(|opt| opt)
        .collect();

    info!("  Retrieved {} embeddings for similarity search", valid_embeddings.len());

    // Perform similarity searches
    for (i, query) in query_embeddings.iter().enumerate() {
        let _similarities = cache.batch_similarity_search(query, &valid_embeddings).await;
        
        if i % 10 == 0 {
            info!("    Completed {} similarity searches", i + 1);
        }
    }

    let search_time = search_start.elapsed();
    info!("  Similarity search completed in: {:?}", search_time);

    // Phase 3: Random access pattern (mixed I/O)
    info!("\nPhase 3: Random access pattern");
    let access_start = Instant::now();

    // Simulate random access to cached embeddings
    let random_keys: Vec<&str> = all_keys.iter().step_by(3).map(|k| k.as_str()).collect();
    let _random_embeddings = cache.get_embeddings_batch(&random_keys).await?;

    let access_time = access_start.elapsed();
    info!("  Random access completed in: {:?}", access_time);

    let total_time = total_start.elapsed();

    // Performance summary
    info!("\n📊 Performance Summary:");
    info!("  Total execution time: {:?}", total_time);
    info!("  Ingestion time: {:?} ({:.1}%)", 
        ingestion_time, 
        ingestion_time.as_millis() as f64 / total_time.as_millis() as f64 * 100.0
    );
    info!("  Search time: {:?} ({:.1}%)", 
        search_time, 
        search_time.as_millis() as f64 / total_time.as_millis() as f64 * 100.0
    );
    info!("  Access time: {:?} ({:.1}%)", 
        access_time, 
        access_time.as_millis() as f64 / total_time.as_millis() as f64 * 100.0
    );

    // Get comprehensive metrics
    let metrics = cache.metrics().await;
    info!("\n📈 Cache Metrics:");
    info!("  Total operations: {}", metrics.total_operations);
    info!("  Parallel operations: {}", metrics.parallel_operations);
    info!("  SIMD operations: {}", metrics.simd_operations);
    info!("  Operations per second: {:.1}", metrics.operations_per_second());
    info!("  SIMD utilization: {:.1}%", metrics.simd_utilization());
    info!("  Parallel utilization: {:.1}%", metrics.parallel_utilization());
    info!("  Prefetch hit rate: {:.1}%", metrics.prefetch_hit_rate());
    info!("  Performance improvement: {:.1}%", metrics.performance_improvement());

    // Get efficiency report
    let efficiency = cache.efficiency_report().await;
    info!("\n🎯 Efficiency Report:");
    info!("{}", efficiency.summary());

    Ok(())
}
