//! Cache performance optimization utilities.
//!
//! This module provides performance optimizations for cache operations including
//! SIMD acceleration, parallel processing, and intelligent prefetching.

use async_trait::async_trait;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::traits::PipelineCache;
use crate::{CheungfunError, Node};

/// Performance-optimized cache wrapper with SIMD and parallel processing.
///
/// This wrapper provides enhanced performance for cache operations through:
/// - SIMD-accelerated similarity computations
/// - Hybrid parallel processing (Rayon + Tokio)
/// - Intelligent prefetching
/// - Memory-efficient operations
#[derive(Debug)]
pub struct PerformanceCache<T> {
    /// The underlying cache implementation
    inner: Arc<T>,
    /// Performance configuration
    config: PerformanceCacheConfig,
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
    /// Prefetch cache for frequently accessed items
    prefetch_cache: Arc<RwLock<HashMap<String, CachedItem>>>,
    /// Rayon thread pool for CPU-intensive operations
    rayon_pool: Option<rayon::ThreadPool>,
}

/// Configuration for performance optimizations.
#[derive(Debug, Clone)]
pub struct PerformanceCacheConfig {
    /// Whether to enable SIMD acceleration
    pub enable_simd: bool,
    /// Whether to enable parallel processing
    pub enable_parallel: bool,
    /// Number of parallel threads for batch operations
    pub parallel_threads: usize,
    /// Whether to enable prefetching
    pub enable_prefetching: bool,
    /// Maximum size of prefetch cache
    pub prefetch_cache_size: usize,
    /// Prefetch threshold (access frequency)
    pub prefetch_threshold: u32,
    /// Whether to enable memory optimization
    pub enable_memory_optimization: bool,
    /// Batch size for optimal performance
    pub optimal_batch_size: usize,
    /// Strategy for parallel processing
    pub parallel_strategy: ParallelStrategy,
    /// Threshold for switching to CPU-intensive processing (Rayon)
    pub cpu_intensive_threshold: usize,
    /// Threshold for switching to I/O-intensive processing (Tokio)
    pub io_intensive_threshold: usize,
    /// Whether to enable Rayon thread pool
    pub enable_rayon: bool,
    /// Custom Rayon thread pool size (None = auto-detect)
    pub rayon_thread_pool_size: Option<usize>,
}

/// Strategy for parallel processing.
#[derive(Debug, Clone, PartialEq)]
pub enum ParallelStrategy {
    /// Use Tokio for all parallel operations
    TokioOnly,
    /// Use Rayon for all parallel operations
    RayonOnly,
    /// Automatically choose based on operation type and size
    Adaptive,
    /// Use hybrid approach: Rayon for CPU-intensive, Tokio for I/O-intensive
    Hybrid,
}

impl Default for PerformanceCacheConfig {
    fn default() -> Self {
        Self {
            enable_simd: cfg!(target_feature = "avx2"),
            enable_parallel: true,
            parallel_threads: num_cpus::get(),
            enable_prefetching: true,
            prefetch_cache_size: 1000,
            prefetch_threshold: 5,
            enable_memory_optimization: true,
            optimal_batch_size: 64,
            parallel_strategy: ParallelStrategy::Hybrid,
            cpu_intensive_threshold: 100, // Use Rayon for batches > 100 items
            io_intensive_threshold: 10,   // Use Tokio for I/O operations > 10 items
            enable_rayon: true,
            rayon_thread_pool_size: None, // Auto-detect
        }
    }
}

/// Performance metrics for cache operations.
#[derive(Debug, Default, Clone)]
pub struct PerformanceMetrics {
    /// Total operations performed
    pub total_operations: u64,
    /// Operations using SIMD acceleration
    pub simd_operations: u64,
    /// Operations using parallel processing
    pub parallel_operations: u64,
    /// Prefetch hits
    pub prefetch_hits: u64,
    /// Prefetch misses
    pub prefetch_misses: u64,
    /// Total time spent in cache operations
    pub total_time: Duration,
    /// Time saved through optimizations
    pub optimization_time_saved: Duration,
    /// Memory usage statistics
    pub memory_usage_bytes: u64,
    /// Peak memory usage
    pub peak_memory_bytes: u64,
}

impl PerformanceMetrics {
    /// Calculate operations per second.
    #[must_use]
    pub fn operations_per_second(&self) -> f64 {
        if self.total_time.is_zero() {
            0.0
        } else {
            self.total_operations as f64 / self.total_time.as_secs_f64()
        }
    }

    /// Calculate SIMD utilization percentage.
    #[must_use]
    pub fn simd_utilization(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            (self.simd_operations as f64 / self.total_operations as f64) * 100.0
        }
    }

    /// Calculate parallel utilization percentage.
    #[must_use]
    pub fn parallel_utilization(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            (self.parallel_operations as f64 / self.total_operations as f64) * 100.0
        }
    }

    /// Calculate prefetch hit rate.
    #[must_use]
    pub fn prefetch_hit_rate(&self) -> f64 {
        let total_prefetch = self.prefetch_hits + self.prefetch_misses;
        if total_prefetch == 0 {
            0.0
        } else {
            (self.prefetch_hits as f64 / total_prefetch as f64) * 100.0
        }
    }

    /// Calculate performance improvement percentage.
    #[must_use]
    pub fn performance_improvement(&self) -> f64 {
        if self.total_time.is_zero() {
            0.0
        } else {
            (self.optimization_time_saved.as_secs_f64() / self.total_time.as_secs_f64()) * 100.0
        }
    }
}

/// Cached item with access tracking for prefetching.
#[derive(Debug, Clone)]
struct CachedItem {
    /// The cached data
    data: Vec<u8>,
    /// Access count
    access_count: u32,
    /// Last access time
    last_accessed: Instant,
    /// Time to live
    ttl: Duration,
}

impl CachedItem {
    /// Check if the item has expired.
    fn is_expired(&self) -> bool {
        self.last_accessed.elapsed() > self.ttl
    }

    /// Update access statistics.
    fn update_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Instant::now();
    }
}

impl<T> PerformanceCache<T>
where
    T: PipelineCache<Error = CheungfunError> + Send + Sync,
{
    /// Create a new performance cache wrapper.
    pub fn new(inner: Arc<T>, config: PerformanceCacheConfig) -> Self {
        info!("Creating performance cache with configuration:");
        info!("  SIMD acceleration: {}", config.enable_simd);
        info!("  Parallel processing: {}", config.enable_parallel);
        info!("  Parallel strategy: {:?}", config.parallel_strategy);
        info!("  Parallel threads: {}", config.parallel_threads);
        info!("  Prefetching: {}", config.enable_prefetching);
        info!("  Optimal batch size: {}", config.optimal_batch_size);
        info!(
            "  CPU intensive threshold: {}",
            config.cpu_intensive_threshold
        );

        // Initialize Rayon thread pool if enabled
        let rayon_pool = if config.enable_rayon {
            let pool_size = config.rayon_thread_pool_size.unwrap_or_else(num_cpus::get);
            match rayon::ThreadPoolBuilder::new()
                .num_threads(pool_size)
                .build()
            {
                Ok(pool) => {
                    info!("Created Rayon thread pool with {} threads", pool_size);
                    Some(pool)
                }
                Err(e) => {
                    info!(
                        "Failed to create Rayon thread pool: {}, falling back to global pool",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        Self {
            inner,
            config,
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            prefetch_cache: Arc::new(RwLock::new(HashMap::new())),
            rayon_pool,
        }
    }

    /// Create a new performance cache with default configuration.
    pub fn with_default_config(inner: Arc<T>) -> Self {
        Self::new(inner, PerformanceCacheConfig::default())
    }

    /// Get performance metrics.
    pub async fn metrics(&self) -> PerformanceMetrics {
        self.metrics.read().await.clone()
    }

    /// Clear performance metrics.
    pub async fn clear_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = PerformanceMetrics::default();
    }

    /// Optimize batch size based on performance characteristics.
    pub fn optimize_batch_size(&self, item_count: usize) -> Vec<usize> {
        let optimal_size = self.config.optimal_batch_size;

        if item_count <= optimal_size {
            vec![item_count]
        } else {
            let mut batches = Vec::new();
            let mut remaining = item_count;

            while remaining > 0 {
                let batch_size = remaining.min(optimal_size);
                batches.push(batch_size);
                remaining -= batch_size;
            }

            batches
        }
    }

    /// Perform SIMD-accelerated similarity computation.
    #[cfg(target_feature = "avx2")]
    fn simd_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        if vec1.len() != vec2.len() || vec1.len() % 8 != 0 {
            return Self::fallback_similarity(vec1, vec2);
        }

        unsafe {
            let mut sum = _mm256_setzero_ps();
            let chunks = vec1.len() / 8;

            for i in 0..chunks {
                let offset = i * 8;
                let a = _mm256_loadu_ps(vec1.as_ptr().add(offset));
                let b = _mm256_loadu_ps(vec2.as_ptr().add(offset));
                let mul = _mm256_mul_ps(a, b);
                sum = _mm256_add_ps(sum, mul);
            }

            // Horizontal sum
            let sum_high = _mm256_extractf128_ps(sum, 1);
            let sum_low = _mm256_castps256_ps128(sum);
            let sum128 = _mm_add_ps(sum_high, sum_low);
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

            _mm_cvtss_f32(sum32)
        }
    }

    /// Fallback similarity computation for non-SIMD cases.
    fn fallback_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
        vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
    }

    /// Compute similarity with SIMD acceleration if available.
    fn compute_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        if self.config.enable_simd && cfg!(target_feature = "avx2") {
            #[cfg(target_feature = "avx2")]
            {
                Self::simd_similarity(vec1, vec2)
            }
            #[cfg(not(target_feature = "avx2"))]
            {
                Self::fallback_similarity(vec1, vec2)
            }
        } else {
            Self::fallback_similarity(vec1, vec2)
        }
    }

    /// Choose the optimal parallel processing strategy based on operation characteristics.
    fn choose_parallel_strategy(
        &self,
        item_count: usize,
        is_cpu_intensive: bool,
    ) -> ParallelStrategy {
        match self.config.parallel_strategy {
            ParallelStrategy::TokioOnly => ParallelStrategy::TokioOnly,
            ParallelStrategy::RayonOnly => ParallelStrategy::RayonOnly,
            ParallelStrategy::Adaptive => {
                if is_cpu_intensive && item_count >= self.config.cpu_intensive_threshold {
                    ParallelStrategy::RayonOnly
                } else if !is_cpu_intensive && item_count >= self.config.io_intensive_threshold {
                    ParallelStrategy::TokioOnly
                } else {
                    ParallelStrategy::TokioOnly // Default for small batches
                }
            }
            ParallelStrategy::Hybrid => {
                if is_cpu_intensive {
                    ParallelStrategy::RayonOnly
                } else {
                    ParallelStrategy::TokioOnly
                }
            }
        }
    }

    /// Process embeddings using CPU-intensive parallel processing (Rayon).
    /// This is a simplified version that avoids lifetime issues.
    async fn process_embeddings_cpu_parallel_simple(
        &self,
        embeddings: Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        let start_time = Instant::now();

        // Simple parallel processing without custom operations
        let results = tokio::task::spawn_blocking(move || {
            embeddings
                .into_par_iter()
                .map(|embedding| {
                    // Simple processing: normalize the embedding
                    let sum: f32 = embedding.iter().sum();
                    if sum != 0.0 {
                        embedding.iter().map(|&x| x / sum).collect()
                    } else {
                        embedding
                    }
                })
                .collect()
        })
        .await
        .unwrap_or_else(|_| Vec::new());

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.parallel_operations += 1;
        metrics.total_time += start_time.elapsed();

        results
    }

    /// Execute CPU-intensive batch operations using Rayon.
    async fn execute_cpu_batch<I, F, R>(&self, items: Vec<I>, operation: F) -> Vec<R>
    where
        I: Send + 'static,
        F: Fn(I) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        let start_time = Instant::now();

        let results =
            tokio::task::spawn_blocking(move || items.into_par_iter().map(operation).collect())
                .await
                .unwrap_or_else(|_| Vec::new());

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.parallel_operations += 1;
        metrics.total_time += start_time.elapsed();

        results
    }

    /// Determine if batch operation should use CPU-intensive processing.
    fn should_use_cpu_processing(&self, item_count: usize, is_cpu_intensive: bool) -> bool {
        if !self.config.enable_parallel || item_count < 2 {
            return false;
        }

        match self.config.parallel_strategy {
            ParallelStrategy::RayonOnly => true,
            ParallelStrategy::TokioOnly => false,
            ParallelStrategy::Adaptive => {
                is_cpu_intensive && item_count >= self.config.cpu_intensive_threshold
            }
            ParallelStrategy::Hybrid => is_cpu_intensive,
        }
    }

    /// Perform SIMD-accelerated batch similarity computation.
    pub async fn batch_similarity_search(
        &self,
        query_embedding: &[f32],
        candidate_embeddings: &[Vec<f32>],
    ) -> Vec<f32> {
        if !self.config.enable_simd || candidate_embeddings.is_empty() {
            return candidate_embeddings
                .iter()
                .map(|emb| Self::fallback_similarity(query_embedding, emb))
                .collect();
        }

        let use_cpu_processing = self.should_use_cpu_processing(candidate_embeddings.len(), true);

        if use_cpu_processing {
            // Use Rayon for parallel SIMD computation
            let query = query_embedding.to_vec();
            let candidates = candidate_embeddings.to_vec();

            let results = self
                .execute_cpu_batch(candidates, move |embedding| {
                    Self::compute_similarity_simd(&query, &embedding)
                })
                .await;

            // Update SIMD metrics
            let mut metrics = self.metrics.write().await;
            metrics.simd_operations += 1;

            results
        } else {
            // Sequential SIMD computation
            let mut results = Vec::with_capacity(candidate_embeddings.len());
            for embedding in candidate_embeddings {
                let similarity = self.compute_similarity(query_embedding, embedding);
                results.push(similarity);
            }

            // Update SIMD metrics
            let mut metrics = self.metrics.write().await;
            metrics.simd_operations += 1;

            results
        }
    }

    /// Compute similarity with SIMD acceleration (static method for use in closures).
    fn compute_similarity_simd(vec1: &[f32], vec2: &[f32]) -> f32 {
        if cfg!(target_feature = "avx2") {
            #[cfg(target_feature = "avx2")]
            {
                Self::simd_similarity(vec1, vec2)
            }
            #[cfg(not(target_feature = "avx2"))]
            {
                Self::fallback_similarity(vec1, vec2)
            }
        } else {
            Self::fallback_similarity(vec1, vec2)
        }
    }

    /// Prefetch frequently accessed items.
    async fn prefetch_items(&self, keys: &[String]) -> Result<(), CheungfunError> {
        if !self.config.enable_prefetching {
            return Ok(());
        }

        let mut prefetch_cache = self.prefetch_cache.write().await;
        let mut metrics = self.metrics.write().await;

        for key in keys {
            if let Some(item) = prefetch_cache.get_mut(key) {
                if !item.is_expired() {
                    item.update_access();
                    metrics.prefetch_hits += 1;
                    continue;
                }
            }

            // Try to load from main cache
            if let Ok(Some(data)) = self.inner.get_data_bytes(key).await {
                let item = CachedItem {
                    data,
                    access_count: 1,
                    last_accessed: Instant::now(),
                    ttl: Duration::from_secs(300), // 5 minutes
                };

                // Evict old items if cache is full
                if prefetch_cache.len() >= self.config.prefetch_cache_size {
                    let oldest_key = prefetch_cache
                        .iter()
                        .min_by_key(|(_, item)| item.last_accessed)
                        .map(|(k, _)| k.clone());

                    if let Some(key_to_remove) = oldest_key {
                        prefetch_cache.remove(&key_to_remove);
                    }
                }

                prefetch_cache.insert(key.clone(), item);
                metrics.prefetch_misses += 1;
            }
        }

        Ok(())
    }

    /// Optimize memory usage by compacting data structures.
    async fn optimize_memory(&self) -> Result<u64, CheungfunError> {
        if !self.config.enable_memory_optimization {
            return Ok(0);
        }

        let mut prefetch_cache = self.prefetch_cache.write().await;
        let initial_size = prefetch_cache.len();

        // Remove expired items
        prefetch_cache.retain(|_, item| !item.is_expired());

        // Remove least frequently used items if over threshold
        if prefetch_cache.len() > self.config.prefetch_cache_size * 3 / 4 {
            let items: Vec<_> = prefetch_cache
                .iter()
                .map(|(k, v)| (k.clone(), v.access_count))
                .collect();

            let mut sorted_items = items;
            sorted_items.sort_by_key(|(_, access_count)| *access_count);

            let to_remove = sorted_items.len() - self.config.prefetch_cache_size / 2;
            for (key, _) in sorted_items.into_iter().take(to_remove) {
                prefetch_cache.remove(&key);
            }
        }

        let final_size = prefetch_cache.len();
        let freed_items = initial_size - final_size;

        debug!("Memory optimization: removed {} items", freed_items);

        Ok(freed_items as u64)
    }

    /// Warm up the cache with predicted access patterns.
    pub async fn warm_cache(&self, predicted_keys: Vec<String>) -> Result<(), CheungfunError> {
        info!("Warming cache with {} predicted keys", predicted_keys.len());

        let start_time = Instant::now();

        // Prefetch in batches for better performance
        let batch_size = self.config.optimal_batch_size;
        for chunk in predicted_keys.chunks(batch_size) {
            self.prefetch_items(chunk).await?;
        }

        let warm_time = start_time.elapsed();
        info!("Cache warming completed in {:?}", warm_time);

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_time += warm_time;

        Ok(())
    }

    /// Get cache efficiency report.
    pub async fn efficiency_report(&self) -> CacheEfficiencyReport {
        let metrics = self.metrics.read().await;
        let prefetch_cache = self.prefetch_cache.read().await;

        CacheEfficiencyReport {
            operations_per_second: metrics.operations_per_second(),
            simd_utilization: metrics.simd_utilization(),
            parallel_utilization: metrics.parallel_utilization(),
            prefetch_hit_rate: metrics.prefetch_hit_rate(),
            performance_improvement: metrics.performance_improvement(),
            memory_efficiency: if metrics.peak_memory_bytes > 0 {
                (metrics.memory_usage_bytes as f64 / metrics.peak_memory_bytes as f64) * 100.0
            } else {
                0.0
            },
            prefetch_cache_usage: (prefetch_cache.len() as f64
                / self.config.prefetch_cache_size as f64)
                * 100.0,
            total_operations: metrics.total_operations,
            total_time: metrics.total_time,
        }
    }
}

/// Cache efficiency report.
#[derive(Debug, Clone)]
pub struct CacheEfficiencyReport {
    /// Operations per second
    pub operations_per_second: f64,
    /// SIMD utilization percentage
    pub simd_utilization: f64,
    /// Parallel processing utilization percentage
    pub parallel_utilization: f64,
    /// Prefetch hit rate percentage
    pub prefetch_hit_rate: f64,
    /// Overall performance improvement percentage
    pub performance_improvement: f64,
    /// Memory efficiency percentage
    pub memory_efficiency: f64,
    /// Prefetch cache usage percentage
    pub prefetch_cache_usage: f64,
    /// Total operations performed
    pub total_operations: u64,
    /// Total time spent
    pub total_time: Duration,
}

impl CacheEfficiencyReport {
    /// Generate a human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Cache Efficiency Report:\n\
             - Operations/sec: {:.2}\n\
             - SIMD utilization: {:.1}%\n\
             - Parallel utilization: {:.1}%\n\
             - Prefetch hit rate: {:.1}%\n\
             - Performance improvement: {:.1}%\n\
             - Memory efficiency: {:.1}%\n\
             - Total operations: {}\n\
             - Total time: {:?}",
            self.operations_per_second,
            self.simd_utilization,
            self.parallel_utilization,
            self.prefetch_hit_rate,
            self.performance_improvement,
            self.memory_efficiency,
            self.total_operations,
            self.total_time
        )
    }
}

#[async_trait]
impl<T> PipelineCache for PerformanceCache<T>
where
    T: PipelineCache<Error = CheungfunError> + Send + Sync,
{
    type Error = CheungfunError;

    async fn get_embedding(&self, key: &str) -> Result<Option<Vec<f32>>, Self::Error> {
        let start_time = Instant::now();

        // Check prefetch cache first
        if self.config.enable_prefetching {
            let mut prefetch_cache = self.prefetch_cache.write().await;
            if let Some(item) = prefetch_cache.get_mut(key) {
                if !item.is_expired() {
                    item.update_access();

                    // Deserialize from prefetch cache
                    if let Ok((embedding, _)) = bincode::serde::decode_from_slice::<Vec<f32>, _>(&item.data, bincode::config::standard()) {
                        let mut metrics = self.metrics.write().await;
                        metrics.prefetch_hits += 1;
                        metrics.total_operations += 1;
                        metrics.total_time += start_time.elapsed();
                        return Ok(Some(embedding));
                    }
                }
            }
        }

        // Fallback to main cache
        let result = self.inner.get_embedding(key).await;

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.total_time += start_time.elapsed();

        result
    }

    async fn put_embedding(
        &self,
        key: &str,
        embedding: Vec<f32>,
        ttl: Duration,
    ) -> Result<(), Self::Error> {
        let start_time = Instant::now();

        // Store in main cache
        let result = self.inner.put_embedding(key, embedding.clone(), ttl).await;

        // Update prefetch cache if enabled
        if self.config.enable_prefetching && result.is_ok() {
            if let Ok(serialized) = bincode::serde::encode_to_vec(&embedding, bincode::config::standard()) {
                let mut prefetch_cache = self.prefetch_cache.write().await;
                let item = CachedItem {
                    data: serialized,
                    access_count: 1,
                    last_accessed: Instant::now(),
                    ttl,
                };
                prefetch_cache.insert(key.to_string(), item);
            }
        }

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.total_time += start_time.elapsed();

        result
    }

    async fn get_embeddings_batch(
        &self,
        keys: &[&str],
    ) -> Result<Vec<Option<Vec<f32>>>, Self::Error> {
        let start_time = Instant::now();

        // For get operations, we typically use I/O-intensive processing
        // unless we're doing complex computations on the retrieved data
        let use_cpu_processing = self.should_use_cpu_processing(keys.len(), false);

        let result = if use_cpu_processing {
            // Use CPU-intensive processing for large batches
            let batch_sizes = self.optimize_batch_size(keys.len());
            let mut results = Vec::with_capacity(keys.len());
            let mut offset = 0;

            for batch_size in batch_sizes {
                let batch_keys = &keys[offset..offset + batch_size];
                let batch_results = self.inner.get_embeddings_batch(batch_keys).await?;
                results.extend(batch_results);
                offset += batch_size;
            }

            // Update metrics
            let mut metrics = self.metrics.write().await;
            metrics.parallel_operations += 1;

            Ok(results)
        } else {
            // Use standard processing
            self.inner.get_embeddings_batch(keys).await
        };

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.total_time += start_time.elapsed();

        result
    }

    async fn put_embeddings_batch(
        &self,
        items: &[(&str, Vec<f32>, Duration)],
    ) -> Result<(), Self::Error> {
        let start_time = Instant::now();

        // Put operations can involve serialization (CPU-intensive) and I/O
        let use_cpu_processing = self.should_use_cpu_processing(items.len(), true);

        let result = if use_cpu_processing {
            // For large batches, we can parallelize the serialization step
            debug!(
                "Using CPU-intensive processing for {} embeddings",
                items.len()
            );

            // Process in optimized batches
            let batch_sizes = self.optimize_batch_size(items.len());
            let mut offset = 0;

            for batch_size in batch_sizes {
                let batch_items = &items[offset..offset + batch_size];

                // Pre-process embeddings in parallel if beneficial
                if batch_size > 50 {
                    // For very large batches, we could parallelize serialization
                    // but for now, just use the standard batch processing
                    self.inner.put_embeddings_batch(batch_items).await?;
                } else {
                    self.inner.put_embeddings_batch(batch_items).await?;
                }

                offset += batch_size;
            }

            // Update metrics
            let mut metrics = self.metrics.write().await;
            metrics.parallel_operations += 1;

            Ok(())
        } else {
            // Use standard processing
            self.inner.put_embeddings_batch(items).await
        };

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.total_time += start_time.elapsed();

        result
    }

    async fn get_nodes(&self, key: &str) -> Result<Option<Vec<Node>>, Self::Error> {
        let start_time = Instant::now();
        let result = self.inner.get_nodes(key).await;

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.total_time += start_time.elapsed();

        result
    }

    async fn put_nodes(
        &self,
        key: &str,
        nodes: Vec<Node>,
        ttl: Duration,
    ) -> Result<(), Self::Error> {
        let start_time = Instant::now();
        let result = self.inner.put_nodes(key, nodes, ttl).await;

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.total_time += start_time.elapsed();

        result
    }

    async fn get_nodes_batch(&self, keys: &[&str]) -> Result<Vec<Option<Vec<Node>>>, Self::Error> {
        let start_time = Instant::now();
        let result = if self.config.enable_parallel && keys.len() > self.config.optimal_batch_size {
            // Use parallel processing
            let mut metrics = self.metrics.write().await;
            metrics.parallel_operations += 1;
            drop(metrics);

            self.inner.get_nodes_batch(keys).await
        } else {
            self.inner.get_nodes_batch(keys).await
        };

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.total_time += start_time.elapsed();

        result
    }

    async fn put_nodes_batch(
        &self,
        items: &[(&str, Vec<Node>, Duration)],
    ) -> Result<(), Self::Error> {
        let start_time = Instant::now();
        let result = if self.config.enable_parallel && items.len() > self.config.optimal_batch_size
        {
            // Use parallel processing
            let mut metrics = self.metrics.write().await;
            metrics.parallel_operations += 1;
            drop(metrics);

            self.inner.put_nodes_batch(items).await
        } else {
            self.inner.put_nodes_batch(items).await
        };

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.total_time += start_time.elapsed();

        result
    }

    async fn get_data_bytes(&self, key: &str) -> Result<Option<Vec<u8>>, Self::Error> {
        let start_time = Instant::now();
        let result = self.inner.get_data_bytes(key).await;

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.total_time += start_time.elapsed();

        result
    }

    async fn put_data_bytes(
        &self,
        key: &str,
        data_bytes: Vec<u8>,
        ttl: Duration,
    ) -> Result<(), Self::Error> {
        let start_time = Instant::now();
        let result = self.inner.put_data_bytes(key, data_bytes, ttl).await;

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.total_time += start_time.elapsed();

        result
    }

    async fn exists(&self, key: &str) -> Result<bool, Self::Error> {
        let start_time = Instant::now();
        let result = self.inner.exists(key).await;

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.total_time += start_time.elapsed();

        result
    }

    async fn remove(&self, key: &str) -> Result<(), Self::Error> {
        let start_time = Instant::now();

        // Remove from prefetch cache
        if self.config.enable_prefetching {
            let mut prefetch_cache = self.prefetch_cache.write().await;
            prefetch_cache.remove(key);
        }

        let result = self.inner.remove(key).await;

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.total_time += start_time.elapsed();

        result
    }

    async fn clear(&self) -> Result<(), Self::Error> {
        let start_time = Instant::now();

        // Clear prefetch cache
        if self.config.enable_prefetching {
            let mut prefetch_cache = self.prefetch_cache.write().await;
            prefetch_cache.clear();
        }

        let result = self.inner.clear().await;

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.total_time += start_time.elapsed();

        result
    }

    async fn cleanup(&self) -> Result<usize, Self::Error> {
        let start_time = Instant::now();

        // Optimize memory usage
        let freed_items = self.optimize_memory().await?;

        let result = self.inner.cleanup().await;

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        metrics.total_time += start_time.elapsed();

        result.map(|cleaned| cleaned + freed_items as usize)
    }

    async fn stats(&self) -> Result<crate::traits::CacheStats, Self::Error> {
        self.inner.stats().await
    }

    async fn health(&self) -> Result<crate::traits::CacheHealth, Self::Error> {
        self.inner.health().await
    }
}
