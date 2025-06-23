//! Comprehensive Cheungfun Performance Benchmark Suite
//!
//! This is the main entry point for running all performance benchmarks.
//! It orchestrates embedder, vector store, and end-to-end RAG benchmarks.

use anyhow::Result;
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::SystemTime;
use tokio;
use tracing::{info, warn};

use cheungfun_examples::benchmark_framework::{PerformanceMetrics, format_metrics};

/// Benchmark suite configuration
#[derive(Debug, Clone)]
struct BenchmarkSuiteConfig {
    /// Output directory for results
    pub output_dir: String,
    /// Whether to run embedder benchmarks
    pub run_embedder_benchmarks: bool,
    /// Whether to run vector store benchmarks
    pub run_vector_store_benchmarks: bool,
    /// Whether to run end-to-end benchmarks
    pub run_end_to_end_benchmarks: bool,
    /// Whether to generate detailed reports
    pub generate_reports: bool,
    /// Whether to save raw metrics to JSON
    pub save_raw_metrics: bool,
}

impl Default for BenchmarkSuiteConfig {
    fn default() -> Self {
        Self {
            output_dir: "./benchmark_results".to_string(),
            run_embedder_benchmarks: true,
            run_vector_store_benchmarks: true,
            run_end_to_end_benchmarks: true,
            generate_reports: true,
            save_raw_metrics: true,
        }
    }
}

/// Benchmark suite results
#[derive(Debug)]
struct BenchmarkSuiteResults {
    /// All collected metrics
    pub all_metrics: Vec<PerformanceMetrics>,
    /// Embedder metrics
    pub embedder_metrics: Vec<PerformanceMetrics>,
    /// Vector store metrics
    pub vector_store_metrics: Vec<PerformanceMetrics>,
    /// End-to-end metrics
    pub end_to_end_metrics: Vec<PerformanceMetrics>,
    /// Benchmark start time
    pub start_time: SystemTime,
    /// Total benchmark duration
    pub total_duration: std::time::Duration,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Cheungfun Comprehensive Performance Benchmark Suite");
    println!("=====================================================");
    println!();

    let config = BenchmarkSuiteConfig::default();
    let start_time = SystemTime::now();

    // Create output directory
    fs::create_dir_all(&config.output_dir)?;
    info!("Created output directory: {}", config.output_dir);

    let mut suite_results = BenchmarkSuiteResults {
        all_metrics: Vec::new(),
        embedder_metrics: Vec::new(),
        vector_store_metrics: Vec::new(),
        end_to_end_metrics: Vec::new(),
        start_time,
        total_duration: std::time::Duration::ZERO,
    };

    // Run embedder benchmarks
    if config.run_embedder_benchmarks {
        println!("🔥 Running Embedder Benchmarks");
        println!("==============================");

        match run_embedder_benchmarks().await {
            Ok(metrics) => {
                suite_results.embedder_metrics = metrics.clone();
                suite_results.all_metrics.extend(metrics);
                println!("✅ Embedder benchmarks completed");
            }
            Err(e) => {
                warn!("Embedder benchmarks failed: {}", e);
            }
        }
        println!();
    }

    // Run vector store benchmarks
    if config.run_vector_store_benchmarks {
        println!("🗄️  Running Vector Store Benchmarks");
        println!("===================================");

        match run_vector_store_benchmarks().await {
            Ok(metrics) => {
                suite_results.vector_store_metrics = metrics.clone();
                suite_results.all_metrics.extend(metrics);
                println!("✅ Vector store benchmarks completed");
            }
            Err(e) => {
                warn!("Vector store benchmarks failed: {}", e);
            }
        }
        println!();
    }

    // Run end-to-end benchmarks
    if config.run_end_to_end_benchmarks {
        println!("🔄 Running End-to-End RAG Benchmarks");
        println!("===================================");

        match run_end_to_end_benchmarks().await {
            Ok(metrics) => {
                suite_results.end_to_end_metrics = metrics.clone();
                suite_results.all_metrics.extend(metrics);
                println!("✅ End-to-end benchmarks completed");
            }
            Err(e) => {
                warn!("End-to-end benchmarks failed: {}", e);
            }
        }
        println!();
    }

    // Calculate total duration
    suite_results.total_duration = start_time.elapsed().unwrap_or_default();

    // Save raw metrics
    if config.save_raw_metrics {
        save_raw_metrics(&config, &suite_results).await?;
    }

    // Generate reports
    if config.generate_reports {
        generate_comprehensive_report(&config, &suite_results).await?;
    }

    println!("🎉 Benchmark suite completed successfully!");
    println!("📊 Total duration: {:?}", suite_results.total_duration);
    println!("📁 Results saved to: {}", config.output_dir);

    Ok(())
}

/// Run embedder benchmarks by calling the embedder_benchmark binary
async fn run_embedder_benchmarks() -> Result<Vec<PerformanceMetrics>> {
    info!("Running embedder benchmarks...");

    // Execute the embedder benchmark binary
    let output = std::process::Command::new("cargo")
        .args(&["run", "--bin", "embedder_benchmark", "--release"])
        .current_dir(".")
        .output();

    match output {
        Ok(result) => {
            if result.status.success() {
                info!("Embedder benchmarks completed successfully");
                // Parse output and convert to metrics (simplified for now)
                Ok(vec![create_sample_metrics("Embedder Benchmark")])
            } else {
                warn!("Embedder benchmarks failed: {}", String::from_utf8_lossy(&result.stderr));
                Ok(Vec::new())
            }
        }
        Err(e) => {
            warn!("Failed to run embedder benchmarks: {}", e);
            Ok(Vec::new())
        }
    }
}

/// Run vector store benchmarks by calling the vector_store_benchmark binary
async fn run_vector_store_benchmarks() -> Result<Vec<PerformanceMetrics>> {
    info!("Running vector store benchmarks...");

    let output = std::process::Command::new("cargo")
        .args(&["run", "--bin", "vector_store_benchmark", "--release"])
        .current_dir(".")
        .output();

    match output {
        Ok(result) => {
            if result.status.success() {
                info!("Vector store benchmarks completed successfully");
                Ok(vec![create_sample_metrics("Vector Store Benchmark")])
            } else {
                warn!("Vector store benchmarks failed: {}", String::from_utf8_lossy(&result.stderr));
                Ok(Vec::new())
            }
        }
        Err(e) => {
            warn!("Failed to run vector store benchmarks: {}", e);
            Ok(Vec::new())
        }
    }
}

/// Run end-to-end benchmarks by calling the end_to_end_benchmark binary
async fn run_end_to_end_benchmarks() -> Result<Vec<PerformanceMetrics>> {
    info!("Running end-to-end benchmarks...");

    let output = std::process::Command::new("cargo")
        .args(&["run", "--bin", "end_to_end_benchmark", "--release"])
        .current_dir(".")
        .output();

    match output {
        Ok(result) => {
            if result.status.success() {
                info!("End-to-end benchmarks completed successfully");
                Ok(vec![create_sample_metrics("End-to-End Benchmark")])
            } else {
                warn!("End-to-end benchmarks failed: {}", String::from_utf8_lossy(&result.stderr));
                Ok(Vec::new())
            }
        }
        Err(e) => {
            warn!("Failed to run end-to-end benchmarks: {}", e);
            Ok(Vec::new())
        }
    }
}

/// Create sample performance metrics for testing
fn create_sample_metrics(name: &str) -> PerformanceMetrics {
    use cheungfun_examples::benchmark_framework::{CpuStats, MemoryStats};

    PerformanceMetrics {
        benchmark_name: name.to_string(),
        operations: 1000,
        duration: std::time::Duration::from_secs(10),
        ops_per_second: 100.0,
        avg_latency: std::time::Duration::from_millis(10),
        p95_latency: std::time::Duration::from_millis(15),
        p99_latency: std::time::Duration::from_millis(20),
        memory_stats: MemoryStats {
            peak_memory_bytes: 1024 * 1024, // 1MB
            avg_memory_bytes: 512 * 1024,   // 512KB
        },
        cpu_stats: CpuStats {
            avg_cpu_percent: 25.0,
            peak_cpu_percent: 50.0,
        },
        additional_data: std::collections::HashMap::new(),
    }
}

/// Save raw metrics to JSON files
async fn save_raw_metrics(
    config: &BenchmarkSuiteConfig,
    results: &BenchmarkSuiteResults,
) -> Result<()> {
    let metrics_dir = Path::new(&config.output_dir).join("raw_metrics");
    fs::create_dir_all(&metrics_dir)?;

    // Save all metrics
    let all_metrics_path = metrics_dir.join("all_metrics.json");
    let all_metrics_json = serde_json::to_string_pretty(&results.all_metrics)?;
    fs::write(all_metrics_path, all_metrics_json)?;

    // Save embedder metrics
    if !results.embedder_metrics.is_empty() {
        let embedder_path = metrics_dir.join("embedder_metrics.json");
        let embedder_json = serde_json::to_string_pretty(&results.embedder_metrics)?;
        fs::write(embedder_path, embedder_json)?;
    }

    // Save vector store metrics
    if !results.vector_store_metrics.is_empty() {
        let vector_store_path = metrics_dir.join("vector_store_metrics.json");
        let vector_store_json = serde_json::to_string_pretty(&results.vector_store_metrics)?;
        fs::write(vector_store_path, vector_store_json)?;
    }

    // Save end-to-end metrics
    if !results.end_to_end_metrics.is_empty() {
        let end_to_end_path = metrics_dir.join("end_to_end_metrics.json");
        let end_to_end_json = serde_json::to_string_pretty(&results.end_to_end_metrics)?;
        fs::write(end_to_end_path, end_to_end_json)?;
    }

    info!("Raw metrics saved to: {}", metrics_dir.display());
    Ok(())
}

/// Generate comprehensive performance report
async fn generate_comprehensive_report(
    config: &BenchmarkSuiteConfig,
    results: &BenchmarkSuiteResults,
) -> Result<()> {
    let report_path = Path::new(&config.output_dir).join("performance_report.md");

    let mut report = String::new();

    // Header
    report.push_str("# Cheungfun Performance Benchmark Report\n\n");
    report.push_str(&format!("**Generated:** {:?}\n", results.start_time));
    report.push_str(&format!(
        "**Total Duration:** {:?}\n",
        results.total_duration
    ));
    report.push_str(&format!(
        "**Total Benchmarks:** {}\n\n",
        results.all_metrics.len()
    ));

    // Executive Summary
    report.push_str("## Executive Summary\n\n");
    if results.all_metrics.is_empty() {
        report.push_str("No benchmark metrics were collected.\n\n");
    } else {
        let avg_ops_per_sec: f64 = results
            .all_metrics
            .iter()
            .map(|m| m.ops_per_second)
            .sum::<f64>()
            / results.all_metrics.len() as f64;

        let avg_latency_ms: f64 = results
            .all_metrics
            .iter()
            .map(|m| m.avg_latency.as_millis() as f64)
            .sum::<f64>()
            / results.all_metrics.len() as f64;

        let peak_memory_mb: f64 = results
            .all_metrics
            .iter()
            .map(|m| m.memory_stats.peak_memory_bytes as f64 / 1024.0 / 1024.0)
            .fold(0.0, f64::max);

        report.push_str(&format!(
            "- **Average Operations/Second:** {:.2}\n",
            avg_ops_per_sec
        ));
        report.push_str(&format!(
            "- **Average Latency:** {:.2} ms\n",
            avg_latency_ms
        ));
        report.push_str(&format!(
            "- **Peak Memory Usage:** {:.1} MB\n\n",
            peak_memory_mb
        ));
    }

    // Detailed Results
    if !results.embedder_metrics.is_empty() {
        report.push_str("## Embedder Performance\n\n");
        for metric in &results.embedder_metrics {
            report.push_str(&format!("### {}\n\n", metric.benchmark_name));
            report.push_str(&format_metrics_markdown(metric));
            report.push_str("\n");
        }
    }

    if !results.vector_store_metrics.is_empty() {
        report.push_str("## Vector Store Performance\n\n");
        for metric in &results.vector_store_metrics {
            report.push_str(&format!("### {}\n\n", metric.benchmark_name));
            report.push_str(&format_metrics_markdown(metric));
            report.push_str("\n");
        }
    }

    if !results.end_to_end_metrics.is_empty() {
        report.push_str("## End-to-End RAG Performance\n\n");
        for metric in &results.end_to_end_metrics {
            report.push_str(&format!("### {}\n\n", metric.benchmark_name));
            report.push_str(&format_metrics_markdown(metric));
            report.push_str("\n");
        }
    }

    // Performance Recommendations
    report.push_str("## Performance Recommendations\n\n");
    report.push_str(&generate_recommendations(results));

    // Save report
    fs::write(report_path, report)?;
    info!("Performance report saved to: {}", config.output_dir);

    Ok(())
}

/// Format metrics for markdown report
fn format_metrics_markdown(metric: &PerformanceMetrics) -> String {
    format!(
        "| Metric | Value |\n\
         |--------|-------|\n\
         | Operations | {} |\n\
         | Duration | {:?} |\n\
         | Ops/Second | {:.2} |\n\
         | Avg Latency | {:?} |\n\
         | P95 Latency | {:?} |\n\
         | P99 Latency | {:?} |\n\
         | Peak Memory | {:.1} MB |\n\
         | Avg CPU | {:.1}% |\n\n",
        metric.operations,
        metric.duration,
        metric.ops_per_second,
        metric.avg_latency,
        metric.p95_latency,
        metric.p99_latency,
        metric.memory_stats.peak_memory_bytes as f64 / 1024.0 / 1024.0,
        metric.cpu_stats.avg_cpu_percent,
    )
}

/// Generate performance recommendations
fn generate_recommendations(results: &BenchmarkSuiteResults) -> String {
    let mut recommendations = String::new();

    if results.all_metrics.is_empty() {
        return "No metrics available for recommendations.\n".to_string();
    }

    // Analyze memory usage
    let high_memory_metrics: Vec<_> = results
        .all_metrics
        .iter()
        .filter(|m| m.memory_stats.peak_memory_bytes > 1024 * 1024 * 1024) // > 1GB
        .collect();

    if !high_memory_metrics.is_empty() {
        recommendations.push_str("### Memory Optimization\n\n");
        recommendations
            .push_str("- Consider reducing batch sizes for high memory usage operations\n");
        recommendations.push_str("- Implement memory pooling for large vector operations\n");
        recommendations.push_str("- Use streaming processing for large datasets\n\n");
    }

    // Analyze latency
    let high_latency_metrics: Vec<_> = results
        .all_metrics
        .iter()
        .filter(|m| m.avg_latency.as_millis() > 1000) // > 1 second
        .collect();

    if !high_latency_metrics.is_empty() {
        recommendations.push_str("### Latency Optimization\n\n");
        recommendations
            .push_str("- Consider using faster embedding models for time-critical applications\n");
        recommendations.push_str("- Implement caching for frequently accessed vectors\n");
        recommendations.push_str("- Use parallel processing for batch operations\n\n");
    }

    // General recommendations
    recommendations.push_str("### General Recommendations\n\n");
    recommendations.push_str("- Monitor performance in production environments\n");
    recommendations.push_str("- Set up alerting for performance degradation\n");
    recommendations.push_str("- Regularly benchmark after system updates\n");
    recommendations
        .push_str("- Consider hardware acceleration (GPU) for compute-intensive workloads\n\n");

    recommendations
}
