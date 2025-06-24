//! Benchmark comparing Tokio vs Rayon performance for cache operations.
//!
//! This benchmark demonstrates the performance differences between using
//! Tokio async tasks vs Rayon parallel iterators for different types of
//! cache operations.

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

    info!("🚀 Starting Tokio vs Rayon Performance Benchmark");

    // Benchmark 1: CPU-intensive operations (SIMD similarity)
    benchmark_cpu_intensive_operations().await?;

    // Benchmark 2: I/O-intensive operations (cache reads/writes)
    benchmark_io_intensive_operations().await?;

    // Benchmark 3: Mixed workload
    benchmark_mixed_workload().await?;

    // Benchmark 4: Batch size scaling
    benchmark_batch_size_scaling().await?;

    // Benchmark 5: Adaptive strategy performance
    benchmark_adaptive_strategy().await?;

    info!("✅ Benchmark completed successfully!");
    Ok(())
}

/// Benchmark CPU-intensive operations (SIMD similarity computation).
async fn benchmark_cpu_intensive_operations() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n🔥 Benchmark 1: CPU-intensive Operations (SIMD Similarity)");

    let test_sizes = vec![100, 500, 1000, 2000];
    let query_embedding = vec![1.0; 128]; // 128-dimensional vector

    for size in test_sizes {
        info!("\nTesting with {} embeddings:", size);

        // Generate test embeddings
        let embeddings: Vec<Vec<f32>> = (0..size)
            .map(|i| (0..128).map(|j| (i * 128 + j) as f32 * 0.001).collect())
            .collect();

        // Test with Tokio-only strategy
        let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
        let tokio_config = PerformanceCacheConfig {
            parallel_strategy: ParallelStrategy::TokioOnly,
            enable_simd: true,
            ..Default::default()
        };
        let tokio_cache = PerformanceCache::new(base_cache, tokio_config);

        let start_time = Instant::now();
        let _tokio_results = tokio_cache
            .batch_similarity_search(&query_embedding, &embeddings)
            .await;
        let tokio_time = start_time.elapsed();

        // Test with Rayon-only strategy
        let base_cache2 = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
        let rayon_config = PerformanceCacheConfig {
            parallel_strategy: ParallelStrategy::RayonOnly,
            enable_simd: true,
            ..Default::default()
        };
        let rayon_cache = PerformanceCache::new(base_cache2, rayon_config);

        let start_time = Instant::now();
        let _rayon_results = rayon_cache
            .batch_similarity_search(&query_embedding, &embeddings)
            .await;
        let rayon_time = start_time.elapsed();

        // Test with Hybrid strategy
        let base_cache3 = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
        let hybrid_config = PerformanceCacheConfig {
            parallel_strategy: ParallelStrategy::Hybrid,
            enable_simd: true,
            cpu_intensive_threshold: 50,
            ..Default::default()
        };
        let hybrid_cache = PerformanceCache::new(base_cache3, hybrid_config);

        let start_time = Instant::now();
        let _hybrid_results = hybrid_cache
            .batch_similarity_search(&query_embedding, &embeddings)
            .await;
        let hybrid_time = start_time.elapsed();

        // Compare results
        info!("  Tokio-only:  {:?}", tokio_time);
        info!("  Rayon-only:  {:?}", rayon_time);
        info!("  Hybrid:      {:?}", hybrid_time);
        info!(
            "  Rayon speedup: {:.2}x vs Tokio",
            tokio_time.as_millis() as f64 / rayon_time.as_millis() as f64
        );
        info!(
            "  Hybrid speedup: {:.2}x vs Tokio",
            tokio_time.as_millis() as f64 / hybrid_time.as_millis() as f64
        );
    }

    Ok(())
}

/// Benchmark I/O-intensive operations (cache reads/writes).
async fn benchmark_io_intensive_operations() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n💾 Benchmark 2: I/O-intensive Operations (Cache Reads/Writes)");

    let test_sizes = vec![50, 200, 500, 1000];
    let ttl = Duration::from_secs(3600);

    for size in test_sizes {
        info!("\nTesting with {} cache operations:", size);

        // Generate test data
        let embeddings: Vec<Vec<f32>> = (0..size)
            .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
            .collect();

        let keys: Vec<String> = (0..size).map(|i| format!("key_{}", i)).collect();

        // Test with Tokio-only strategy
        let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
        let tokio_config = PerformanceCacheConfig {
            parallel_strategy: ParallelStrategy::TokioOnly,
            io_intensive_threshold: 10,
            ..Default::default()
        };
        let tokio_cache = PerformanceCache::new(base_cache, tokio_config);

        // Benchmark write operations
        let batch_items: Vec<(&str, Vec<f32>, Duration)> = keys
            .iter()
            .zip(embeddings.iter())
            .map(|(key, embedding)| (key.as_str(), embedding.clone(), ttl))
            .collect();

        let start_time = Instant::now();
        tokio_cache.put_embeddings_batch(&batch_items).await?;
        let tokio_write_time = start_time.elapsed();

        // Benchmark read operations
        let key_refs: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
        let start_time = Instant::now();
        let _tokio_results = tokio_cache.get_embeddings_batch(&key_refs).await?;
        let tokio_read_time = start_time.elapsed();

        // Test with Rayon-only strategy
        let base_cache2 = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
        let rayon_config = PerformanceCacheConfig {
            parallel_strategy: ParallelStrategy::RayonOnly,
            ..Default::default()
        };
        let rayon_cache = PerformanceCache::new(base_cache2, rayon_config);

        let start_time = Instant::now();
        rayon_cache.put_embeddings_batch(&batch_items).await?;
        let rayon_write_time = start_time.elapsed();

        let start_time = Instant::now();
        let _rayon_results = rayon_cache.get_embeddings_batch(&key_refs).await?;
        let rayon_read_time = start_time.elapsed();

        // Test with Hybrid strategy
        let base_cache3 = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
        let hybrid_config = PerformanceCacheConfig {
            parallel_strategy: ParallelStrategy::Hybrid,
            io_intensive_threshold: 10,
            cpu_intensive_threshold: 100,
            ..Default::default()
        };
        let hybrid_cache = PerformanceCache::new(base_cache3, hybrid_config);

        let start_time = Instant::now();
        hybrid_cache.put_embeddings_batch(&batch_items).await?;
        let hybrid_write_time = start_time.elapsed();

        let start_time = Instant::now();
        let _hybrid_results = hybrid_cache.get_embeddings_batch(&key_refs).await?;
        let hybrid_read_time = start_time.elapsed();

        // Compare results
        info!("  Write operations:");
        info!("    Tokio-only:  {:?}", tokio_write_time);
        info!("    Rayon-only:  {:?}", rayon_write_time);
        info!("    Hybrid:      {:?}", hybrid_write_time);

        info!("  Read operations:");
        info!("    Tokio-only:  {:?}", tokio_read_time);
        info!("    Rayon-only:  {:?}", rayon_read_time);
        info!("    Hybrid:      {:?}", hybrid_read_time);

        let total_tokio = tokio_write_time + tokio_read_time;
        let total_rayon = rayon_write_time + rayon_read_time;
        let total_hybrid = hybrid_write_time + hybrid_read_time;

        info!("  Total time:");
        info!("    Tokio-only:  {:?}", total_tokio);
        info!("    Rayon-only:  {:?}", total_rayon);
        info!("    Hybrid:      {:?}", total_hybrid);
        info!(
            "    Best strategy: {}",
            if total_hybrid <= total_tokio && total_hybrid <= total_rayon {
                "Hybrid"
            } else if total_tokio <= total_rayon {
                "Tokio"
            } else {
                "Rayon"
            }
        );
    }

    Ok(())
}

/// Benchmark mixed workload (CPU + I/O operations).
async fn benchmark_mixed_workload() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n🔄 Benchmark 3: Mixed Workload (CPU + I/O)");

    let cache_size = 500;
    let similarity_queries = 100;
    let ttl = Duration::from_secs(3600);

    // Generate test data
    let embeddings: Vec<Vec<f32>> = (0..cache_size)
        .map(|i| (0..64).map(|j| (i * 64 + j) as f32 * 0.001).collect())
        .collect();

    let keys: Vec<String> = (0..cache_size).map(|i| format!("mixed_key_{}", i)).collect();

    let query_embeddings: Vec<Vec<f32>> = (0..similarity_queries)
        .map(|i| (0..64).map(|j| (i * 64 + j) as f32 * 0.002).collect())
        .collect();

    // Test different strategies
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
            cpu_intensive_threshold: 50,
            io_intensive_threshold: 20,
            ..Default::default()
        };
        let cache = PerformanceCache::new(base_cache, config);

        let start_time = Instant::now();

        // Phase 1: Populate cache (I/O intensive)
        let batch_items: Vec<(&str, Vec<f32>, Duration)> = keys
            .iter()
            .zip(embeddings.iter())
            .map(|(key, embedding)| (key.as_str(), embedding.clone(), ttl))
            .collect();

        cache.put_embeddings_batch(&batch_items).await?;
        let populate_time = start_time.elapsed();

        // Phase 2: Perform similarity searches (CPU intensive)
        let search_start = Instant::now();
        for query in &query_embeddings {
            let _similarities = cache.batch_similarity_search(query, &embeddings).await;
        }
        let search_time = search_start.elapsed();

        // Phase 3: Random cache access (I/O intensive)
        let access_start = Instant::now();
        let random_keys: Vec<&str> = keys.iter().step_by(3).map(|k| k.as_str()).collect();
        let _cached_embeddings = cache.get_embeddings_batch(&random_keys).await?;
        let access_time = access_start.elapsed();

        let total_time = start_time.elapsed();

        info!("  Populate cache: {:?}", populate_time);
        info!("  Similarity search: {:?}", search_time);
        info!("  Random access: {:?}", access_time);
        info!("  Total time: {:?}", total_time);

        // Get performance metrics
        let metrics = cache.metrics().await;
        info!("  SIMD utilization: {:.1}%", metrics.simd_utilization());
        info!("  Parallel utilization: {:.1}%", metrics.parallel_utilization());
        info!("  Operations/sec: {:.1}", metrics.operations_per_second());
    }

    Ok(())
}

/// Benchmark batch size scaling.
async fn benchmark_batch_size_scaling() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n📏 Benchmark 4: Batch Size Scaling");

    let batch_sizes = vec![10, 50, 100, 500, 1000, 2000];
    let ttl = Duration::from_secs(3600);

    for batch_size in batch_sizes {
        info!("\nTesting batch size: {}", batch_size);

        // Generate test data
        let embeddings: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
            .collect();

        let keys: Vec<String> = (0..batch_size).map(|i| format!("scale_key_{}", i)).collect();

        // Test Hybrid strategy with different thresholds
        let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
        let config = PerformanceCacheConfig {
            parallel_strategy: ParallelStrategy::Hybrid,
            cpu_intensive_threshold: 100,
            io_intensive_threshold: 20,
            optimal_batch_size: 64,
            ..Default::default()
        };
        let cache = PerformanceCache::new(base_cache, config);

        // Benchmark batch operations
        let batch_items: Vec<(&str, Vec<f32>, Duration)> = keys
            .iter()
            .zip(embeddings.iter())
            .map(|(key, embedding)| (key.as_str(), embedding.clone(), ttl))
            .collect();

        let start_time = Instant::now();
        cache.put_embeddings_batch(&batch_items).await?;
        let write_time = start_time.elapsed();

        let key_refs: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
        let start_time = Instant::now();
        let _results = cache.get_embeddings_batch(&key_refs).await?;
        let read_time = start_time.elapsed();

        let total_time = write_time + read_time;
        let throughput = (batch_size * 2) as f64 / total_time.as_secs_f64(); // ops/sec

        info!("  Write: {:?}, Read: {:?}, Total: {:?}", write_time, read_time, total_time);
        info!("  Throughput: {:.1} ops/sec", throughput);

        // Get performance metrics
        let metrics = cache.metrics().await;
        info!("  Parallel operations: {}", metrics.parallel_operations);
        info!("  SIMD operations: {}", metrics.simd_operations);
    }

    Ok(())
}

/// Benchmark adaptive strategy performance.
async fn benchmark_adaptive_strategy() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n🧠 Benchmark 5: Adaptive Strategy Performance");

    // Test different workload patterns
    let workloads = vec![
        ("Small I/O", 20, false),
        ("Large I/O", 200, false),
        ("Small CPU", 20, true),
        ("Large CPU", 200, true),
    ];

    for (name, size, is_cpu_intensive) in workloads {
        info!("\nTesting workload: {} (size: {})", name, size);

        let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
        let config = PerformanceCacheConfig {
            parallel_strategy: ParallelStrategy::Adaptive,
            cpu_intensive_threshold: 100,
            io_intensive_threshold: 50,
            enable_simd: true,
            ..Default::default()
        };
        let cache = PerformanceCache::new(base_cache, config);

        let start_time = Instant::now();

        if is_cpu_intensive {
            // CPU-intensive workload: similarity computation
            let query = vec![1.0; 64];
            let embeddings: Vec<Vec<f32>> = (0..size)
                .map(|i| (0..64).map(|j| (i * 64 + j) as f32 * 0.001).collect())
                .collect();

            let _similarities = cache.batch_similarity_search(&query, &embeddings).await;
        } else {
            // I/O-intensive workload: cache operations
            let embeddings: Vec<Vec<f32>> = (0..size)
                .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32])
                .collect();

            let keys: Vec<String> = (0..size).map(|i| format!("adaptive_key_{}", i)).collect();

            let batch_items: Vec<(&str, Vec<f32>, Duration)> = keys
                .iter()
                .zip(embeddings.iter())
                .map(|(key, embedding)| (key.as_str(), embedding.clone(), Duration::from_secs(3600)))
                .collect();

            cache.put_embeddings_batch(&batch_items).await?;

            let key_refs: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
            let _results = cache.get_embeddings_batch(&key_refs).await?;
        }

        let total_time = start_time.elapsed();

        info!("  Execution time: {:?}", total_time);

        // Get performance metrics
        let metrics = cache.metrics().await;
        info!("  Strategy used: {}", 
            if metrics.parallel_operations > 0 { "Parallel" } else { "Sequential" }
        );
        info!("  SIMD utilization: {:.1}%", metrics.simd_utilization());
        info!("  Parallel utilization: {:.1}%", metrics.parallel_utilization());
    }

    Ok(())
}
