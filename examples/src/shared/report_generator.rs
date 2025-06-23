//! Performance report generation with visualization
//!
//! This module provides functionality to generate comprehensive performance reports
//! with charts and visualizations using the plotters library.

use crate::benchmark_framework::PerformanceMetrics;
use anyhow::Result;
use plotters::prelude::*;
use std::path::Path;

/// Report generator for performance metrics
pub struct ReportGenerator {
    output_dir: String,
}

impl ReportGenerator {
    /// Create a new report generator
    pub fn new(output_dir: String) -> Self {
        Self { output_dir }
    }

    /// Generate a comprehensive performance report with visualizations
    pub async fn generate_comprehensive_report(
        &self,
        metrics: &[PerformanceMetrics],
    ) -> Result<()> {
        if metrics.is_empty() {
            return Ok(());
        }

        // Create charts directory
        let charts_dir = Path::new(&self.output_dir).join("charts");
        std::fs::create_dir_all(&charts_dir)?;

        // Generate various charts
        self.generate_ops_per_second_chart(metrics, &charts_dir)
            .await?;
        self.generate_latency_chart(metrics, &charts_dir).await?;
        self.generate_memory_usage_chart(metrics, &charts_dir)
            .await?;
        self.generate_cpu_usage_chart(metrics, &charts_dir).await?;
        self.generate_performance_matrix_chart(metrics, &charts_dir)
            .await?;

        // Generate HTML report
        self.generate_html_report(metrics).await?;

        Ok(())
    }

    /// Generate operations per second chart
    async fn generate_ops_per_second_chart(
        &self,
        metrics: &[PerformanceMetrics],
        charts_dir: &Path,
    ) -> Result<()> {
        let chart_path = charts_dir.join("ops_per_second.png");
        let root = BitMapBackend::new(&chart_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_ops = metrics.iter().map(|m| m.ops_per_second).fold(0.0, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Operations Per Second", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(0f64..metrics.len() as f64, 0f64..max_ops * 1.1)?;

        chart
            .configure_mesh()
            .x_desc("Benchmark")
            .y_desc("Operations/Second")
            .draw()?;

        chart
            .draw_series(metrics.iter().enumerate().map(|(i, metric)| {
                Rectangle::new(
                    [(i as f64, 0.0), (i as f64 + 0.8, metric.ops_per_second)],
                    BLUE.filled(),
                )
            }))?
            .label("Ops/Second")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

        chart.configure_series_labels().draw()?;
        root.present()?;

        Ok(())
    }

    /// Generate latency distribution chart
    async fn generate_latency_chart(
        &self,
        metrics: &[PerformanceMetrics],
        charts_dir: &Path,
    ) -> Result<()> {
        let chart_path = charts_dir.join("latency_distribution.png");
        let root = BitMapBackend::new(&chart_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_latency = metrics
            .iter()
            .map(|m| m.max_latency.as_millis() as f64)
            .fold(0.0, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Latency Distribution", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(0f64..metrics.len() as f64, 0f64..max_latency * 1.1)?;

        chart
            .configure_mesh()
            .x_desc("Benchmark")
            .y_desc("Latency (ms)")
            .draw()?;

        // Draw average latency
        chart
            .draw_series(metrics.iter().enumerate().map(|(i, metric)| {
                Circle::new(
                    (i as f64 + 0.4, metric.avg_latency.as_millis() as f64),
                    5,
                    BLUE.filled(),
                )
            }))?
            .label("Average")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

        // Draw P95 latency
        chart
            .draw_series(metrics.iter().enumerate().map(|(i, metric)| {
                Circle::new(
                    (i as f64 + 0.4, metric.p95_latency.as_millis() as f64),
                    5,
                    RED.filled(),
                )
            }))?
            .label("P95")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

        // Draw P99 latency
        chart
            .draw_series(metrics.iter().enumerate().map(|(i, metric)| {
                Circle::new(
                    (i as f64 + 0.4, metric.p99_latency.as_millis() as f64),
                    5,
                    GREEN.filled(),
                )
            }))?
            .label("P99")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &GREEN));

        chart.configure_series_labels().draw()?;
        root.present()?;

        Ok(())
    }

    /// Generate memory usage chart
    async fn generate_memory_usage_chart(
        &self,
        metrics: &[PerformanceMetrics],
        charts_dir: &Path,
    ) -> Result<()> {
        let chart_path = charts_dir.join("memory_usage.png");
        let root = BitMapBackend::new(&chart_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_memory = metrics
            .iter()
            .map(|m| m.memory_stats.peak_memory_bytes as f64 / 1024.0 / 1024.0)
            .fold(0.0, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Memory Usage", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(0f64..metrics.len() as f64, 0f64..max_memory * 1.1)?;

        chart
            .configure_mesh()
            .x_desc("Benchmark")
            .y_desc("Memory (MB)")
            .draw()?;

        chart
            .draw_series(metrics.iter().enumerate().map(|(i, metric)| {
                Rectangle::new(
                    [
                        (i as f64, 0.0),
                        (
                            i as f64 + 0.8,
                            metric.memory_stats.peak_memory_bytes as f64 / 1024.0 / 1024.0,
                        ),
                    ],
                    GREEN.filled(),
                )
            }))?
            .label("Peak Memory")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &GREEN));

        chart.configure_series_labels().draw()?;
        root.present()?;

        Ok(())
    }

    /// Generate CPU usage chart
    async fn generate_cpu_usage_chart(
        &self,
        metrics: &[PerformanceMetrics],
        charts_dir: &Path,
    ) -> Result<()> {
        let chart_path = charts_dir.join("cpu_usage.png");
        let root = BitMapBackend::new(&chart_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("CPU Usage", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(0f64..metrics.len() as f64, 0f64..100f64)?;

        chart
            .configure_mesh()
            .x_desc("Benchmark")
            .y_desc("CPU Usage (%)")
            .draw()?;

        chart
            .draw_series(metrics.iter().enumerate().map(|(i, metric)| {
                Rectangle::new(
                    [
                        (i as f64, 0.0),
                        (i as f64 + 0.8, metric.cpu_stats.avg_cpu_percent),
                    ],
                    RED.filled(),
                )
            }))?
            .label("Average CPU")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

        chart.configure_series_labels().draw()?;
        root.present()?;

        Ok(())
    }

    /// Generate performance matrix chart (ops/sec vs latency)
    async fn generate_performance_matrix_chart(
        &self,
        metrics: &[PerformanceMetrics],
        charts_dir: &Path,
    ) -> Result<()> {
        let chart_path = charts_dir.join("performance_matrix.png");
        let root = BitMapBackend::new(&chart_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_ops = metrics.iter().map(|m| m.ops_per_second).fold(0.0, f64::max);

        let max_latency = metrics
            .iter()
            .map(|m| m.avg_latency.as_millis() as f64)
            .fold(0.0, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Performance Matrix: Throughput vs Latency",
                ("sans-serif", 40),
            )
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(0f64..max_ops * 1.1, 0f64..max_latency * 1.1)?;

        chart
            .configure_mesh()
            .x_desc("Operations/Second")
            .y_desc("Average Latency (ms)")
            .draw()?;

        chart
            .draw_series(metrics.iter().map(|metric| {
                Circle::new(
                    (metric.ops_per_second, metric.avg_latency.as_millis() as f64),
                    8,
                    BLUE.filled(),
                )
            }))?
            .label("Benchmarks")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

        chart.configure_series_labels().draw()?;
        root.present()?;

        Ok(())
    }

    /// Generate HTML report with embedded charts
    async fn generate_html_report(&self, metrics: &[PerformanceMetrics]) -> Result<()> {
        let report_path = Path::new(&self.output_dir).join("performance_report.html");

        let mut html = String::new();

        // HTML header
        html.push_str(r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cheungfun Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .metric-card { background: #ecf0f1; padding: 20px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #3498db; }
        .chart { text-align: center; margin: 20px 0; }
        .chart img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }
        .summary { background: #e8f6f3; padding: 20px; border-radius: 8px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Cheungfun Performance Benchmark Report</h1>
"#);

        // Summary section
        if !metrics.is_empty() {
            let avg_ops_per_sec: f64 =
                metrics.iter().map(|m| m.ops_per_second).sum::<f64>() / metrics.len() as f64;

            let avg_latency_ms: f64 = metrics
                .iter()
                .map(|m| m.avg_latency.as_millis() as f64)
                .sum::<f64>()
                / metrics.len() as f64;

            let peak_memory_mb: f64 = metrics
                .iter()
                .map(|m| m.memory_stats.peak_memory_bytes as f64 / 1024.0 / 1024.0)
                .fold(0.0, f64::max);

            html.push_str(&format!(
                r#"
        <div class="summary">
            <h2>📊 Executive Summary</h2>
            <p><strong>Total Benchmarks:</strong> {}</p>
            <p><strong>Average Operations/Second:</strong> {:.2}</p>
            <p><strong>Average Latency:</strong> {:.2} ms</p>
            <p><strong>Peak Memory Usage:</strong> {:.1} MB</p>
        </div>
"#,
                metrics.len(),
                avg_ops_per_sec,
                avg_latency_ms,
                peak_memory_mb
            ));
        }

        // Charts section
        html.push_str(
            r#"
        <h2>📈 Performance Charts</h2>
        
        <div class="chart">
            <h3>Operations Per Second</h3>
            <img src="charts/ops_per_second.png" alt="Operations Per Second Chart">
        </div>
        
        <div class="chart">
            <h3>Latency Distribution</h3>
            <img src="charts/latency_distribution.png" alt="Latency Distribution Chart">
        </div>
        
        <div class="chart">
            <h3>Memory Usage</h3>
            <img src="charts/memory_usage.png" alt="Memory Usage Chart">
        </div>
        
        <div class="chart">
            <h3>CPU Usage</h3>
            <img src="charts/cpu_usage.png" alt="CPU Usage Chart">
        </div>
        
        <div class="chart">
            <h3>Performance Matrix</h3>
            <img src="charts/performance_matrix.png" alt="Performance Matrix Chart">
        </div>
"#,
        );

        // Detailed metrics table
        html.push_str(
            r#"
        <h2>📋 Detailed Metrics</h2>
        <table>
            <tr>
                <th>Benchmark</th>
                <th>Ops/Sec</th>
                <th>Avg Latency</th>
                <th>P95 Latency</th>
                <th>Peak Memory (MB)</th>
                <th>Avg CPU (%)</th>
            </tr>
"#,
        );

        for metric in metrics {
            html.push_str(&format!(
                r#"
            <tr>
                <td>{}</td>
                <td>{:.2}</td>
                <td>{:?}</td>
                <td>{:?}</td>
                <td>{:.1}</td>
                <td>{:.1}</td>
            </tr>
"#,
                metric.benchmark_name,
                metric.ops_per_second,
                metric.avg_latency,
                metric.p95_latency,
                metric.memory_stats.peak_memory_bytes as f64 / 1024.0 / 1024.0,
                metric.cpu_stats.avg_cpu_percent
            ));
        }

        html.push_str(
            r#"
        </table>
        
        <div class="summary">
            <h2>💡 Performance Recommendations</h2>
            <ul>
                <li>Monitor performance metrics in production environments</li>
                <li>Set up alerting for performance degradation</li>
                <li>Consider hardware acceleration for compute-intensive workloads</li>
                <li>Implement caching strategies for frequently accessed data</li>
                <li>Optimize batch sizes based on memory and latency requirements</li>
            </ul>
        </div>
    </div>
</body>
</html>
"#,
        );

        std::fs::write(report_path, html)?;
        Ok(())
    }
}
