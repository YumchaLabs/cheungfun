//! Performance benchmarking framework for Cheungfun
//!
//! This module provides a unified framework for performance testing across all Cheungfun components.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Performance metrics collected during benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Unique identifier for this benchmark run
    pub run_id: Uuid,
    /// Name of the benchmark
    pub benchmark_name: String,
    /// Timestamp when the benchmark started
    pub start_time: std::time::SystemTime,
    /// Total duration of the benchmark
    pub duration: Duration,
    /// Number of operations performed
    pub operations_count: u64,
    /// Operations per second
    pub ops_per_second: f64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Average memory usage in bytes
    pub avg_memory_bytes: u64,
    /// Peak CPU usage percentage
    pub peak_cpu_percent: f64,
    /// Average CPU usage percentage
    pub avg_cpu_percent: f64,
    /// Percentile latencies (P50, P95, P99)
    pub latency_percentiles: HashMap<String, Duration>,
    /// Custom metrics specific to the benchmark
    pub custom_metrics: HashMap<String, f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl PerformanceMetrics {
    #[must_use]
    pub fn new(benchmark_name: String) -> Self {
        Self {
            run_id: Uuid::new_v4(),
            benchmark_name,
            start_time: std::time::SystemTime::now(),
            duration: Duration::default(),
            operations_count: 0,
            ops_per_second: 0.0,
            peak_memory_bytes: 0,
            avg_memory_bytes: 0,
            peak_cpu_percent: 0.0,
            avg_cpu_percent: 0.0,
            latency_percentiles: HashMap::new(),
            custom_metrics: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn calculate_ops_per_second(&mut self) {
        if self.duration.as_secs_f64() > 0.0 {
            self.ops_per_second = self.operations_count as f64 / self.duration.as_secs_f64();
        }
    }
}

/// Configuration for benchmarks
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub name: String,
    pub warmup_iterations: u32,
    pub measurement_iterations: u32,
    pub enable_memory_tracking: bool,
    pub enable_cpu_tracking: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            name: "default_benchmark".to_string(),
            warmup_iterations: 5,
            measurement_iterations: 100,
            enable_memory_tracking: true,
            enable_cpu_tracking: true,
        }
    }
}

/// Simple benchmark runner for basic performance testing
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    latencies: Vec<Duration>,
    start_time: Option<Instant>,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner with the given configuration
    #[must_use]
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            latencies: Vec::new(),
            start_time: None,
        }
    }

    /// Start the benchmark
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Record a single operation latency
    pub fn record_operation(&mut self, latency: Duration) {
        self.latencies.push(latency);
    }

    /// Finish the benchmark and return metrics
    #[must_use]
    pub fn finish(self) -> PerformanceMetrics {
        let total_duration = self
            .start_time
            .map(|start| start.elapsed())
            .unwrap_or_default();

        let mut metrics = PerformanceMetrics::new(self.config.name);
        metrics.duration = total_duration;
        metrics.operations_count = self.latencies.len() as u64;
        metrics.calculate_ops_per_second();

        // Calculate latency percentiles if we have data
        if !self.latencies.is_empty() {
            let mut sorted_latencies = self.latencies;
            sorted_latencies.sort();

            let len = sorted_latencies.len();
            metrics
                .latency_percentiles
                .insert("P50".to_string(), sorted_latencies[len / 2]);
            metrics
                .latency_percentiles
                .insert("P95".to_string(), sorted_latencies[(len * 95) / 100]);
            metrics
                .latency_percentiles
                .insert("P99".to_string(), sorted_latencies[(len * 99) / 100]);
        }

        metrics
    }
}

/// Utility functions for benchmark analysis
pub mod analysis {
    use super::Duration;

    /// Calculate the mean of a series of durations
    #[must_use]
    pub fn mean_duration(durations: &[Duration]) -> Duration {
        if durations.is_empty() {
            return Duration::default();
        }

        let total_nanos: u64 = durations.iter().map(|d| d.as_nanos() as u64).sum();
        Duration::from_nanos(total_nanos / durations.len() as u64)
    }

    /// Calculate standard deviation of durations
    #[must_use]
    pub fn std_dev_duration(durations: &[Duration]) -> Duration {
        if durations.len() < 2 {
            return Duration::default();
        }

        let mean = mean_duration(durations);
        let variance: f64 = durations
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>()
            / (durations.len() - 1) as f64;

        Duration::from_nanos(variance.sqrt() as u64)
    }

    /// Calculate percentile for a sorted list of durations
    #[must_use]
    pub fn percentile(sorted_durations: &[Duration], percentile: f64) -> Duration {
        if sorted_durations.is_empty() {
            return Duration::default();
        }

        let index = ((sorted_durations.len() as f64) * percentile / 100.0) as usize;
        let index = index.min(sorted_durations.len() - 1);
        sorted_durations[index]
    }
}
