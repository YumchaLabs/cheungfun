//! Performance benchmarking framework for Cheungfun
//!
//! This module provides a unified framework for performance testing across all Cheungfun components.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use sysinfo::System;
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
    pub operations: u64,
    /// Operations per second
    pub ops_per_second: f64,
    /// Average latency per operation
    pub avg_latency: Duration,
    /// Minimum latency observed
    pub min_latency: Duration,
    /// Maximum latency observed
    pub max_latency: Duration,
    /// 95th percentile latency
    pub p95_latency: Duration,
    /// 99th percentile latency
    pub p99_latency: Duration,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// CPU usage statistics
    pub cpu_stats: CpuStats,
    /// Custom metrics specific to the benchmark
    pub custom_metrics: HashMap<String, f64>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Average memory usage in bytes
    pub avg_memory_bytes: u64,
    /// Memory usage at start in bytes
    pub start_memory_bytes: u64,
    /// Memory usage at end in bytes
    pub end_memory_bytes: u64,
}

/// CPU usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuStats {
    /// Average CPU usage percentage
    pub avg_cpu_percent: f64,
    /// Peak CPU usage percentage
    pub peak_cpu_percent: f64,
    /// Number of CPU cores used
    pub cores_used: usize,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Name of the benchmark
    pub name: String,
    /// Number of warmup iterations
    pub warmup_iterations: u32,
    /// Number of measurement iterations
    pub measurement_iterations: u32,
    /// Duration to run the benchmark (alternative to iterations)
    pub duration: Option<Duration>,
    /// Whether to collect detailed system metrics
    pub collect_system_metrics: bool,
    /// Sample rate for system metrics (in milliseconds)
    pub metrics_sample_rate: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            name: "benchmark".to_string(),
            warmup_iterations: 10,
            measurement_iterations: 100,
            duration: None,
            collect_system_metrics: true,
            metrics_sample_rate: Duration::from_millis(100),
        }
    }
}

/// Benchmark runner that collects performance metrics
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    system: System,
    latencies: Vec<Duration>,
    memory_samples: Vec<u64>,
    cpu_samples: Vec<f64>,
    start_time: Option<Instant>,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner with the given configuration
    pub fn new(config: BenchmarkConfig) -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        Self {
            config,
            system,
            latencies: Vec::new(),
            memory_samples: Vec::new(),
            cpu_samples: Vec::new(),
            start_time: None,
        }
    }

    /// Start the benchmark
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.latencies.clear();
        self.memory_samples.clear();
        self.cpu_samples.clear();

        if self.config.collect_system_metrics {
            self.collect_system_sample();
        }
    }

    /// Record the latency of a single operation
    pub fn record_operation(&mut self, latency: Duration) {
        self.latencies.push(latency);

        if self.config.collect_system_metrics {
            self.collect_system_sample();
        }
    }

    /// Collect a system metrics sample
    fn collect_system_sample(&mut self) {
        self.system.refresh_all();

        // Collect memory usage
        let memory_usage = self.system.used_memory();
        self.memory_samples.push(memory_usage);

        // Collect CPU usage - use average of all CPUs
        let cpu_usage = self
            .system
            .cpus()
            .iter()
            .map(|cpu| cpu.cpu_usage())
            .sum::<f32>()
            / self.system.cpus().len() as f32;
        self.cpu_samples.push(cpu_usage as f64);
    }

    /// Finish the benchmark and return performance metrics
    pub fn finish(mut self) -> PerformanceMetrics {
        let total_duration = self
            .start_time
            .map(|start| start.elapsed())
            .unwrap_or_default();

        // Final system sample
        if self.config.collect_system_metrics {
            self.collect_system_sample();
        }

        // Calculate latency statistics
        let mut sorted_latencies = self.latencies.clone();
        sorted_latencies.sort();

        let avg_latency = if !sorted_latencies.is_empty() {
            let total: Duration = sorted_latencies.iter().sum();
            total / sorted_latencies.len() as u32
        } else {
            Duration::ZERO
        };

        let min_latency = sorted_latencies.first().copied().unwrap_or_default();
        let max_latency = sorted_latencies.last().copied().unwrap_or_default();

        let p95_latency = if !sorted_latencies.is_empty() {
            let index = (sorted_latencies.len() as f64 * 0.95) as usize;
            sorted_latencies.get(index).copied().unwrap_or_default()
        } else {
            Duration::ZERO
        };

        let p99_latency = if !sorted_latencies.is_empty() {
            let index = (sorted_latencies.len() as f64 * 0.99) as usize;
            sorted_latencies.get(index).copied().unwrap_or_default()
        } else {
            Duration::ZERO
        };

        // Calculate memory statistics
        let memory_stats = if !self.memory_samples.is_empty() {
            MemoryStats {
                peak_memory_bytes: *self.memory_samples.iter().max().unwrap_or(&0),
                avg_memory_bytes: self.memory_samples.iter().sum::<u64>()
                    / self.memory_samples.len() as u64,
                start_memory_bytes: self.memory_samples.first().copied().unwrap_or(0),
                end_memory_bytes: self.memory_samples.last().copied().unwrap_or(0),
            }
        } else {
            MemoryStats {
                peak_memory_bytes: 0,
                avg_memory_bytes: 0,
                start_memory_bytes: 0,
                end_memory_bytes: 0,
            }
        };

        // Calculate CPU statistics
        let cpu_stats = if !self.cpu_samples.is_empty() {
            CpuStats {
                avg_cpu_percent: self.cpu_samples.iter().sum::<f64>()
                    / self.cpu_samples.len() as f64,
                peak_cpu_percent: self.cpu_samples.iter().fold(0.0, |a, &b| a.max(b)),
                cores_used: self.system.cpus().len(),
            }
        } else {
            CpuStats {
                avg_cpu_percent: 0.0,
                peak_cpu_percent: 0.0,
                cores_used: 0,
            }
        };

        let operations = self.latencies.len() as u64;
        let ops_per_second = if total_duration.as_secs_f64() > 0.0 {
            operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        PerformanceMetrics {
            run_id: Uuid::new_v4(),
            benchmark_name: self.config.name.clone(),
            start_time: std::time::SystemTime::now(),
            duration: total_duration,
            operations,
            ops_per_second,
            avg_latency,
            min_latency,
            max_latency,
            p95_latency,
            p99_latency,
            memory_stats,
            cpu_stats,
            custom_metrics: HashMap::new(),
        }
    }
}

/// Utility function to run a simple benchmark
pub async fn run_benchmark<F, Fut, T>(
    config: BenchmarkConfig,
    mut operation: F,
) -> anyhow::Result<PerformanceMetrics>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<T>>,
{
    let mut runner = BenchmarkRunner::new(config.clone());

    // Warmup phase
    println!("🔥 Warming up ({} iterations)...", config.warmup_iterations);
    for _ in 0..config.warmup_iterations {
        let _ = operation().await?;
    }

    // Measurement phase
    println!(
        "📊 Running benchmark ({} iterations)...",
        config.measurement_iterations
    );
    runner.start();

    for _ in 0..config.measurement_iterations {
        let start = Instant::now();
        let _ = operation().await?;
        let latency = start.elapsed();
        runner.record_operation(latency);
    }

    Ok(runner.finish())
}

/// Format performance metrics for display
pub fn format_metrics(metrics: &PerformanceMetrics) -> String {
    format!(
        r#"
📊 Benchmark Results: {}
=====================================
🕐 Duration: {:?}
🔢 Operations: {}
⚡ Ops/sec: {:.2}
📈 Latency:
  • Average: {:?}
  • Min: {:?}
  • Max: {:?}
  • P95: {:?}
  • P99: {:?}
💾 Memory:
  • Peak: {:.2} MB
  • Average: {:.2} MB
  • Start: {:.2} MB
  • End: {:.2} MB
🖥️  CPU:
  • Average: {:.1}%
  • Peak: {:.1}%
  • Cores: {}
"#,
        metrics.benchmark_name,
        metrics.duration,
        metrics.operations,
        metrics.ops_per_second,
        metrics.avg_latency,
        metrics.min_latency,
        metrics.max_latency,
        metrics.p95_latency,
        metrics.p99_latency,
        metrics.memory_stats.peak_memory_bytes as f64 / 1024.0 / 1024.0,
        metrics.memory_stats.avg_memory_bytes as f64 / 1024.0 / 1024.0,
        metrics.memory_stats.start_memory_bytes as f64 / 1024.0 / 1024.0,
        metrics.memory_stats.end_memory_bytes as f64 / 1024.0 / 1024.0,
        metrics.cpu_stats.avg_cpu_percent,
        metrics.cpu_stats.peak_cpu_percent,
        metrics.cpu_stats.cores_used,
    )
}

/// Generate a CSV report from multiple metrics
pub fn generate_csv_report(metrics: &[PerformanceMetrics]) -> String {
    let mut csv = String::new();

    // Header
    csv.push_str("benchmark_name,duration_ms,operations,ops_per_second,avg_latency_ms,min_latency_ms,max_latency_ms,p95_latency_ms,p99_latency_ms,peak_memory_mb,avg_memory_mb,avg_cpu_percent,peak_cpu_percent\n");

    // Data rows
    for metric in metrics {
        csv.push_str(&format!(
            "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.1},{:.1}\n",
            metric.benchmark_name,
            metric.duration.as_millis(),
            metric.operations,
            metric.ops_per_second,
            metric.avg_latency.as_millis(),
            metric.min_latency.as_millis(),
            metric.max_latency.as_millis(),
            metric.p95_latency.as_millis(),
            metric.p99_latency.as_millis(),
            metric.memory_stats.peak_memory_bytes as f64 / 1024.0 / 1024.0,
            metric.memory_stats.avg_memory_bytes as f64 / 1024.0 / 1024.0,
            metric.cpu_stats.avg_cpu_percent,
            metric.cpu_stats.peak_cpu_percent,
        ));
    }

    csv
}

/// Compare two sets of metrics and generate a comparison report
pub fn compare_metrics(baseline: &[PerformanceMetrics], current: &[PerformanceMetrics]) -> String {
    let mut report = String::new();

    report.push_str("📊 Performance Comparison Report\n");
    report.push_str("===============================\n\n");

    // Group metrics by name for comparison
    let mut baseline_map = std::collections::HashMap::new();
    let mut current_map = std::collections::HashMap::new();

    for metric in baseline {
        baseline_map.insert(&metric.benchmark_name, metric);
    }

    for metric in current {
        current_map.insert(&metric.benchmark_name, metric);
    }

    // Compare matching benchmarks
    for (name, current_metric) in &current_map {
        if let Some(baseline_metric) = baseline_map.get(name) {
            report.push_str(&format!("## {}\n\n", name));

            let ops_change = ((current_metric.ops_per_second - baseline_metric.ops_per_second)
                / baseline_metric.ops_per_second)
                * 100.0;
            let latency_change = ((current_metric.avg_latency.as_millis() as f64
                - baseline_metric.avg_latency.as_millis() as f64)
                / baseline_metric.avg_latency.as_millis() as f64)
                * 100.0;
            let memory_change = ((current_metric.memory_stats.peak_memory_bytes as f64
                - baseline_metric.memory_stats.peak_memory_bytes as f64)
                / baseline_metric.memory_stats.peak_memory_bytes as f64)
                * 100.0;

            report.push_str(&format!(
                "- **Ops/sec:** {:.2} → {:.2} ({:+.1}%)\n",
                baseline_metric.ops_per_second, current_metric.ops_per_second, ops_change
            ));
            report.push_str(&format!(
                "- **Avg Latency:** {:?} → {:?} ({:+.1}%)\n",
                baseline_metric.avg_latency, current_metric.avg_latency, latency_change
            ));
            report.push_str(&format!(
                "- **Peak Memory:** {:.1} MB → {:.1} MB ({:+.1}%)\n\n",
                baseline_metric.memory_stats.peak_memory_bytes as f64 / 1024.0 / 1024.0,
                current_metric.memory_stats.peak_memory_bytes as f64 / 1024.0 / 1024.0,
                memory_change
            ));
        }
    }

    report
}
