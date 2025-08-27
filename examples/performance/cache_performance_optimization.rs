//! Demonstration of cache performance optimization features.
//!
//! This example shows how to use the performance-optimized cache wrapper
//! with SIMD acceleration, parallel processing, and intelligent prefetching.

use cheungfun_core::{
    cache::{MemoryCache, PerformanceCache, PerformanceCacheConfig, UnifiedCache},
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

    info!("🚀 Starting Cache Performance Optimization Demo");

    // Demo 1: Performance Configuration
    demo_performance_configuration().await?;

    // Demo 2: SIMD Acceleration
    demo_simd_acceleration().await?;

    // Demo 3: Parallel Processing
    demo_parallel_processing().await?;

    // Demo 4: Intelligent Prefetching
    demo_intelligent_prefetching().await?;

    // Demo 5: Performance Comparison
    demo_performance_comparison().await?;

    // Demo 6: Efficiency Reporting
    demo_efficiency_reporting().await?;

    info!("✅ Cache performance optimization demo completed successfully!");
    Ok(())
}

/// Demonstrate different performance configurations.
async fn demo_performance_configuration() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n⚙️ Demo 1: Performance Configuration");

    // Default configuration
    let default_config = PerformanceCacheConfig::default();
    info!("Default performance configuration:");
    info!("  SIMD acceleration: {}", default_config.enable_simd);
    info!("  Parallel processing: {}", default_config.enable_parallel);
    info!("  Parallel threads: {}", default_config.parallel_threads);
    info!("  Prefetching: {}", default_config.enable_prefetching);
    info!(
        "  Prefetch cache size: {}",
        default_config.prefetch_cache_size
    );
    info!(
        "  Optimal batch size: {}",
        default_config.optimal_batch_size
    );

    // High-performance configuration
    let high_perf_config = PerformanceCacheConfig {
        enable_simd: true,
        enable_parallel: true,
        parallel_threads: num_cpus::get() * 2,
        enable_prefetching: true,
        prefetch_cache_size: 5000,
        prefetch_threshold: 3,
        enable_memory_optimization: true,
        optimal_batch_size: 128,
    };

    info!("\nHigh-performance configuration:");
    info!("  Parallel threads: {}", high_perf_config.parallel_threads);
    info!(
        "  Prefetch cache size: {}",
        high_perf_config.prefetch_cache_size
    );
    info!(
        "  Prefetch threshold: {}",
        high_perf_config.prefetch_threshold
    );
    info!(
        "  Optimal batch size: {}",
        high_perf_config.optimal_batch_size
    );

    // Memory-optimized configuration
    let memory_config = PerformanceCacheConfig {
        enable_simd: false,
        enable_parallel: false,
        parallel_threads: 1,
        enable_prefetching: false,
        prefetch_cache_size: 100,
        prefetch_threshold: 10,
        enable_memory_optimization: true,
        optimal_batch_size: 32,
    };

    info!("\nMemory-optimized configuration:");
    info!("  SIMD: {}", memory_config.enable_simd);
    info!("  Parallel: {}", memory_config.enable_parallel);
    info!("  Prefetching: {}", memory_config.enable_prefetching);
    info!(
        "  Memory optimization: {}",
        memory_config.enable_memory_optimization
    );

    Ok(())
}

/// Demonstrate SIMD acceleration benefits.
async fn demo_simd_acceleration() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n⚡ Demo 2: SIMD Acceleration");

    // Create cache with SIMD enabled
    let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let simd_config = PerformanceCacheConfig {
        enable_simd: true,
        enable_parallel: false,
        enable_prefetching: false,
        ..Default::default()
    };
    let simd_cache = PerformanceCache::new(base_cache.clone(), simd_config);

    // Create cache with SIMD disabled
    let base_cache2 = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let no_simd_config = PerformanceCacheConfig {
        enable_simd: false,
        enable_parallel: false,
        enable_prefetching: false,
        ..Default::default()
    };
    let no_simd_cache = PerformanceCache::new(base_cache2, no_simd_config);

    // Test data - vectors aligned for SIMD
    let test_embeddings: Vec<Vec<f32>> = (0..1000)
        .map(|i| {
            // Create 128-dimensional vectors (aligned for AVX2)
            (0..128).map(|j| (i * 128 + j) as f32 * 0.001).collect()
        })
        .collect();

    let ttl = Duration::from_secs(3600);

    // Test SIMD-enabled cache
    info!("Testing SIMD-enabled cache...");
    let start_time = Instant::now();
    for (i, embedding) in test_embeddings.iter().enumerate() {
        let key = format!("simd_embedding_{}", i);
        simd_cache
            .put_embedding(&key, embedding.clone(), ttl)
            .await?;
    }
    let simd_store_time = start_time.elapsed();

    let start_time = Instant::now();
    for i in 0..test_embeddings.len() {
        let key = format!("simd_embedding_{}", i);
        let _ = simd_cache.get_embedding(&key).await?;
    }
    let simd_retrieve_time = start_time.elapsed();

    // Test non-SIMD cache
    info!("Testing non-SIMD cache...");
    let start_time = Instant::now();
    for (i, embedding) in test_embeddings.iter().enumerate() {
        let key = format!("no_simd_embedding_{}", i);
        no_simd_cache
            .put_embedding(&key, embedding.clone(), ttl)
            .await?;
    }
    let no_simd_store_time = start_time.elapsed();

    let start_time = Instant::now();
    for i in 0..test_embeddings.len() {
        let key = format!("no_simd_embedding_{}", i);
        let _ = no_simd_cache.get_embedding(&key).await?;
    }
    let no_simd_retrieve_time = start_time.elapsed();

    // Compare performance
    info!("SIMD Performance Comparison:");
    info!(
        "  Store: SIMD {:?} vs No-SIMD {:?} ({:.2}x speedup)",
        simd_store_time,
        no_simd_store_time,
        no_simd_store_time.as_millis() as f64 / simd_store_time.as_millis() as f64
    );
    info!(
        "  Retrieve: SIMD {:?} vs No-SIMD {:?} ({:.2}x speedup)",
        simd_retrieve_time,
        no_simd_retrieve_time,
        no_simd_retrieve_time.as_millis() as f64 / simd_retrieve_time.as_millis() as f64
    );

    // Get SIMD utilization metrics
    let simd_metrics = simd_cache.metrics().await;
    let no_simd_metrics = no_simd_cache.metrics().await;

    info!("SIMD Utilization:");
    info!("  SIMD cache: {:.1}%", simd_metrics.simd_utilization());
    info!(
        "  No-SIMD cache: {:.1}%",
        no_simd_metrics.simd_utilization()
    );

    Ok(())
}

/// Demonstrate parallel processing benefits.
async fn demo_parallel_processing() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n🔄 Demo 3: Parallel Processing");

    // Create cache with parallel processing enabled
    let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let parallel_config = PerformanceCacheConfig {
        enable_parallel: true,
        parallel_threads: num_cpus::get(),
        optimal_batch_size: 50,
        enable_simd: false,
        enable_prefetching: false,
        ..Default::default()
    };
    let parallel_cache = PerformanceCache::new(base_cache, parallel_config);

    // Create cache with parallel processing disabled
    let base_cache2 = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let sequential_config = PerformanceCacheConfig {
        enable_parallel: false,
        parallel_threads: 1,
        optimal_batch_size: 50,
        enable_simd: false,
        enable_prefetching: false,
        ..Default::default()
    };
    let sequential_cache = PerformanceCache::new(base_cache2, sequential_config);

    // Large batch of embeddings
    let batch_size = 500;
    let embeddings: Vec<Vec<f32>> = (0..batch_size)
        .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
        .collect();

    let keys: Vec<String> = (0..batch_size)
        .map(|i| format!("batch_embedding_{}", i))
        .collect();

    let ttl = Duration::from_secs(3600);

    // Test parallel processing
    info!("Testing parallel batch operations...");
    let batch_items: Vec<(&str, Vec<f32>, Duration)> = keys
        .iter()
        .zip(embeddings.iter())
        .map(|(key, embedding)| (key.as_str(), embedding.clone(), ttl))
        .collect();

    let start_time = Instant::now();
    parallel_cache.put_embeddings_batch(&batch_items).await?;
    let parallel_store_time = start_time.elapsed();

    let key_refs: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
    let start_time = Instant::now();
    let _parallel_results = parallel_cache.get_embeddings_batch(&key_refs).await?;
    let parallel_retrieve_time = start_time.elapsed();

    // Test sequential processing
    info!("Testing sequential batch operations...");
    let start_time = Instant::now();
    sequential_cache.put_embeddings_batch(&batch_items).await?;
    let sequential_store_time = start_time.elapsed();

    let start_time = Instant::now();
    let _sequential_results = sequential_cache.get_embeddings_batch(&key_refs).await?;
    let sequential_retrieve_time = start_time.elapsed();

    // Compare performance
    info!("Parallel Processing Performance Comparison:");
    info!(
        "  Batch store: Parallel {:?} vs Sequential {:?} ({:.2}x speedup)",
        parallel_store_time,
        sequential_store_time,
        sequential_store_time.as_millis() as f64 / parallel_store_time.as_millis() as f64
    );
    info!(
        "  Batch retrieve: Parallel {:?} vs Sequential {:?} ({:.2}x speedup)",
        parallel_retrieve_time,
        sequential_retrieve_time,
        sequential_retrieve_time.as_millis() as f64 / parallel_retrieve_time.as_millis() as f64
    );

    // Get parallel utilization metrics
    let parallel_metrics = parallel_cache.metrics().await;
    let sequential_metrics = sequential_cache.metrics().await;

    info!("Parallel Utilization:");
    info!(
        "  Parallel cache: {:.1}%",
        parallel_metrics.parallel_utilization()
    );
    info!(
        "  Sequential cache: {:.1}%",
        sequential_metrics.parallel_utilization()
    );

    Ok(())
}

/// Demonstrate intelligent prefetching.
async fn demo_intelligent_prefetching() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n🔮 Demo 4: Intelligent Prefetching");

    let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let prefetch_config = PerformanceCacheConfig {
        enable_prefetching: true,
        prefetch_cache_size: 100,
        prefetch_threshold: 2,
        enable_simd: false,
        enable_parallel: false,
        ..Default::default()
    };
    let prefetch_cache = PerformanceCache::new(base_cache, prefetch_config);

    // Simulate access patterns
    let frequent_keys = vec!["frequent_1", "frequent_2", "frequent_3"];
    let infrequent_keys = vec!["rare_1", "rare_2", "rare_3"];

    let embedding = vec![1.0, 2.0, 3.0, 4.0];
    let ttl = Duration::from_secs(3600);

    // Store embeddings
    for key in &frequent_keys {
        prefetch_cache
            .put_embedding(key, embedding.clone(), ttl)
            .await?;
    }
    for key in &infrequent_keys {
        prefetch_cache
            .put_embedding(key, embedding.clone(), ttl)
            .await?;
    }

    // Simulate frequent access to some keys
    info!("Simulating access patterns...");
    for _ in 0..5 {
        for key in &frequent_keys {
            let _ = prefetch_cache.get_embedding(key).await?;
        }
    }

    // Access infrequent keys only once
    for key in &infrequent_keys {
        let _ = prefetch_cache.get_embedding(key).await?;
    }

    // Warm cache with predicted keys
    let predicted_keys: Vec<String> = frequent_keys.iter().map(|k| k.to_string()).collect();
    prefetch_cache.warm_cache(predicted_keys).await?;

    // Test prefetch performance
    info!("Testing prefetch performance...");
    let start_time = Instant::now();
    for key in &frequent_keys {
        let _ = prefetch_cache.get_embedding(key).await?;
    }
    let prefetch_time = start_time.elapsed();

    info!("Prefetch access time: {:?}", prefetch_time);

    // Get prefetch metrics
    let metrics = prefetch_cache.metrics().await;
    info!("Prefetch Statistics:");
    info!("  Prefetch hit rate: {:.1}%", metrics.prefetch_hit_rate());
    info!("  Prefetch hits: {}", metrics.prefetch_hits);
    info!("  Prefetch misses: {}", metrics.prefetch_misses);

    Ok(())
}

/// Compare performance between optimized and standard caches.
async fn demo_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n📊 Demo 5: Performance Comparison");

    // Standard cache
    let standard_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));

    // Optimized cache
    let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let optimized_config = PerformanceCacheConfig {
        enable_simd: true,
        enable_parallel: true,
        enable_prefetching: true,
        parallel_threads: num_cpus::get(),
        prefetch_cache_size: 1000,
        optimal_batch_size: 64,
        ..Default::default()
    };
    let optimized_cache = PerformanceCache::new(base_cache, optimized_config);

    // Test data
    let test_size = 1000;
    let embeddings: Vec<Vec<f32>> = (0..test_size)
        .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
        .collect();

    let ttl = Duration::from_secs(3600);

    // Test standard cache
    info!("Testing standard cache...");
    let start_time = Instant::now();
    for (i, embedding) in embeddings.iter().enumerate() {
        let key = format!("standard_{}", i);
        standard_cache
            .put_embedding(&key, embedding.clone(), ttl)
            .await?;
    }
    let standard_store_time = start_time.elapsed();

    let start_time = Instant::now();
    for i in 0..test_size {
        let key = format!("standard_{}", i);
        let _ = standard_cache.get_embedding(&key).await?;
    }
    let standard_retrieve_time = start_time.elapsed();

    // Test optimized cache
    info!("Testing optimized cache...");
    let start_time = Instant::now();
    for (i, embedding) in embeddings.iter().enumerate() {
        let key = format!("optimized_{}", i);
        optimized_cache
            .put_embedding(&key, embedding.clone(), ttl)
            .await?;
    }
    let optimized_store_time = start_time.elapsed();

    let start_time = Instant::now();
    for i in 0..test_size {
        let key = format!("optimized_{}", i);
        let _ = optimized_cache.get_embedding(&key).await?;
    }
    let optimized_retrieve_time = start_time.elapsed();

    // Performance comparison
    info!("Overall Performance Comparison:");
    info!(
        "  Store operations: Standard {:?} vs Optimized {:?} ({:.2}x speedup)",
        standard_store_time,
        optimized_store_time,
        standard_store_time.as_millis() as f64 / optimized_store_time.as_millis() as f64
    );
    info!(
        "  Retrieve operations: Standard {:?} vs Optimized {:?} ({:.2}x speedup)",
        standard_retrieve_time,
        optimized_retrieve_time,
        standard_retrieve_time.as_millis() as f64 / optimized_retrieve_time.as_millis() as f64
    );

    let total_standard = standard_store_time + standard_retrieve_time;
    let total_optimized = optimized_store_time + optimized_retrieve_time;

    info!(
        "  Total time: Standard {:?} vs Optimized {:?} ({:.2}x overall speedup)",
        total_standard,
        total_optimized,
        total_standard.as_millis() as f64 / total_optimized.as_millis() as f64
    );

    Ok(())
}

/// Demonstrate efficiency reporting.
async fn demo_efficiency_reporting() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n📈 Demo 6: Efficiency Reporting");

    let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let config = PerformanceCacheConfig {
        enable_simd: true,
        enable_parallel: true,
        enable_prefetching: true,
        ..Default::default()
    };
    let perf_cache = PerformanceCache::new(base_cache, config);

    // Perform various operations to generate metrics
    let embeddings: Vec<Vec<f32>> = (0..100)
        .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
        .collect();

    let ttl = Duration::from_secs(3600);

    // Store and retrieve embeddings
    for (i, embedding) in embeddings.iter().enumerate() {
        let key = format!("report_embedding_{}", i);
        perf_cache
            .put_embedding(&key, embedding.clone(), ttl)
            .await?;
    }

    for i in 0..embeddings.len() {
        let key = format!("report_embedding_{}", i);
        let _ = perf_cache.get_embedding(&key).await?;
    }

    // Generate efficiency report
    let report = perf_cache.efficiency_report().await;

    info!("Cache Efficiency Report:");
    info!("{}", report.summary());

    // Get detailed metrics
    let metrics = perf_cache.metrics().await;
    info!("\nDetailed Performance Metrics:");
    info!("  Total operations: {}", metrics.total_operations);
    info!("  SIMD operations: {}", metrics.simd_operations);
    info!("  Parallel operations: {}", metrics.parallel_operations);
    info!(
        "  Operations per second: {:.2}",
        metrics.operations_per_second()
    );
    info!(
        "  Performance improvement: {:.1}%",
        metrics.performance_improvement()
    );

    Ok(())
}
