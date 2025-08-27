//! Demonstration of cache monitoring and diagnostics system.
//!
//! This example shows how to use the comprehensive cache monitoring system
//! including real-time metrics, alerting, and performance analysis.

use cheungfun_core::{
    cache::{
        AlertSeverity, AlertThresholds, CacheMonitor, MemoryCache, MonitoringConfig, UnifiedCache,
    },
    traits::PipelineCache,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🚀 Starting Cache Monitoring Demo");

    // Demo 1: Basic Monitoring Setup
    demo_monitoring_setup().await?;

    // Demo 2: Real-time Metrics Collection
    demo_metrics_collection().await?;

    // Demo 3: Alert System
    demo_alert_system().await?;

    // Demo 4: Performance Analysis
    demo_performance_analysis().await?;

    // Demo 5: Diagnostic Reports
    demo_diagnostic_reports().await?;

    info!("✅ Cache monitoring demo completed successfully!");
    Ok(())
}

/// Demonstrate basic monitoring setup.
async fn demo_monitoring_setup() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n⚙️ Demo 1: Basic Monitoring Setup");

    // Create base cache
    let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));

    // Configure monitoring
    let monitoring_config = MonitoringConfig {
        collection_interval: Duration::from_secs(5), // Collect metrics every 5 seconds
        max_samples: 100,
        enable_alerting: true,
        alert_thresholds: AlertThresholds {
            min_hit_rate: 60.0,
            max_response_time: 50,
            max_error_rate: 2.0,
            max_memory_usage: 80.0,
            max_cache_size: 85.0,
        },
        enable_performance_analysis: true,
        analysis_window: Duration::from_secs(60),
        enable_trend_analysis: true,
        trend_period: Duration::from_secs(300),
    };

    info!("Monitoring configuration:");
    info!("  Collection interval: {:?}", monitoring_config.collection_interval);
    info!("  Max samples: {}", monitoring_config.max_samples);
    info!("  Alerting enabled: {}", monitoring_config.enable_alerting);
    info!("  Performance analysis: {}", monitoring_config.enable_performance_analysis);
    info!("  Trend analysis: {}", monitoring_config.enable_trend_analysis);

    // Create monitored cache
    let monitored_cache = CacheMonitor::new(base_cache, monitoring_config);

    // Start monitoring
    monitored_cache.start_monitoring().await?;

    info!("✅ Monitoring system started successfully");

    // Let it run for a short time
    sleep(Duration::from_secs(2)).await;

    // Check initial status
    let status = monitored_cache.get_monitoring_status().await;
    info!("Initial monitoring status:");
    info!("  Active: {}", status.is_active);
    info!("  Total samples: {}", status.total_samples);
    info!("  Active alerts: {}", status.active_alerts);

    Ok(())
}

/// Demonstrate real-time metrics collection.
async fn demo_metrics_collection() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n📊 Demo 2: Real-time Metrics Collection");

    let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let config = MonitoringConfig {
        collection_interval: Duration::from_secs(2),
        max_samples: 50,
        ..Default::default()
    };
    let monitored_cache = CacheMonitor::new(base_cache, config);

    // Start monitoring
    monitored_cache.start_monitoring().await?;

    // Perform cache operations to generate metrics
    info!("Performing cache operations to generate metrics...");

    let embeddings = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
    ];

    let ttl = Duration::from_secs(3600);

    // Store embeddings
    for (i, embedding) in embeddings.iter().enumerate() {
        let key = format!("metrics_embedding_{}", i);
        monitored_cache
            .put_embedding(&key, embedding.clone(), ttl)
            .await?;
    }

    // Retrieve embeddings (cache hits)
    for i in 0..embeddings.len() {
        let key = format!("metrics_embedding_{}", i);
        let _ = monitored_cache.get_embedding(&key).await?;
    }

    // Try to retrieve non-existent embeddings (cache misses)
    for i in 10..15 {
        let key = format!("missing_embedding_{}", i);
        let _ = monitored_cache.get_embedding(&key).await?;
    }

    // Wait for metrics collection
    sleep(Duration::from_secs(5)).await;

    // Get metrics history
    let metrics_history = monitored_cache.get_metrics_history(Some(5)).await;
    info!("Recent metrics samples: {}", metrics_history.len());

    for (i, sample) in metrics_history.iter().enumerate() {
        info!(
            "  Sample {}: Hit rate {:.1}%, Total ops: {}",
            i + 1,
            sample.cache_stats.hit_rate(),
            sample.cache_stats.hits + sample.cache_stats.misses
        );
    }

    // Get current status
    let status = monitored_cache.get_monitoring_status().await;
    info!("Current metrics:");
    info!("  Average hit rate: {:.1}%", status.current_metrics.avg_hit_rate);
    info!("  Average response time: {:.1}ms", status.current_metrics.avg_response_time);
    info!("  Total operations: {}", status.current_metrics.total_operations);
    info!("  Throughput: {:.1} ops/sec", status.current_metrics.throughput);

    Ok(())
}

/// Demonstrate alert system.
async fn demo_alert_system() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n🚨 Demo 3: Alert System");

    let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    
    // Configure with strict thresholds to trigger alerts
    let config = MonitoringConfig {
        collection_interval: Duration::from_secs(1),
        enable_alerting: true,
        alert_thresholds: AlertThresholds {
            min_hit_rate: 90.0, // Very high threshold to trigger alert
            max_response_time: 10, // Very low threshold to trigger alert
            max_error_rate: 1.0,
            max_memory_usage: 50.0,
            max_cache_size: 50.0,
        },
        ..Default::default()
    };

    let monitored_cache = CacheMonitor::new(base_cache, config);
    monitored_cache.start_monitoring().await?;

    info!("Configured strict alert thresholds:");
    info!("  Minimum hit rate: 90.0%");
    info!("  Maximum response time: 10ms");

    // Perform operations that will likely trigger alerts
    info!("Performing operations to trigger alerts...");

    // Create cache misses to lower hit rate
    for i in 0..20 {
        let key = format!("missing_key_{}", i);
        let _ = monitored_cache.get_embedding(&key).await?;
    }

    // Wait for metrics collection and alert processing
    sleep(Duration::from_secs(3)).await;

    // Check for alerts
    let active_alerts = monitored_cache.get_active_alerts().await;
    info!("Active alerts: {}", active_alerts.len());

    for alert in &active_alerts {
        info!("  Alert: {} - {}", alert.alert_type_string(), alert.message);
        info!("    Severity: {:?}", alert.severity);
        info!("    Metric value: {:.1}", alert.metric_value);
        info!("    Threshold: {:.1}", alert.threshold);
    }

    if active_alerts.is_empty() {
        info!("  No alerts triggered (cache performance is good!)");
    }

    // Get monitoring status
    let status = monitored_cache.get_monitoring_status().await;
    info!("Monitoring status:");
    info!("  Active alerts: {}", status.active_alerts);
    info!("  Current hit rate: {:.1}%", status.current_metrics.avg_hit_rate);

    Ok(())
}

/// Demonstrate performance analysis.
async fn demo_performance_analysis() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n📈 Demo 4: Performance Analysis");

    let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let config = MonitoringConfig {
        collection_interval: Duration::from_secs(1),
        enable_performance_analysis: true,
        enable_trend_analysis: true,
        analysis_window: Duration::from_secs(10),
        ..Default::default()
    };

    let monitored_cache = CacheMonitor::new(base_cache, config);
    monitored_cache.start_monitoring().await?;

    // Simulate varying performance patterns
    info!("Simulating performance patterns...");

    let ttl = Duration::from_secs(3600);

    // Phase 1: Good performance (high hit rate)
    info!("Phase 1: Establishing baseline with good performance");
    for i in 0..10 {
        let key = format!("baseline_key_{}", i);
        let embedding = vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32];
        monitored_cache
            .put_embedding(&key, embedding, ttl)
            .await?;
    }

    // Access the same keys multiple times (high hit rate)
    for _ in 0..3 {
        for i in 0..10 {
            let key = format!("baseline_key_{}", i);
            let _ = monitored_cache.get_embedding(&key).await?;
        }
    }

    sleep(Duration::from_secs(3)).await;

    // Phase 2: Degraded performance (low hit rate)
    info!("Phase 2: Simulating performance degradation");
    for i in 100..150 {
        let key = format!("random_key_{}", i);
        let _ = monitored_cache.get_embedding(&key).await?; // Cache misses
    }

    sleep(Duration::from_secs(3)).await;

    // Get trend analysis
    let trends = monitored_cache.get_trends().await;
    info!("Trend analysis:");
    for (metric_name, trend) in &trends {
        info!(
            "  {}: {:?} trend (strength: {:.2}, confidence: {:.2})",
            metric_name, trend.trend_direction, trend.trend_strength, trend.confidence
        );
    }

    // Get anomalies
    let anomalies = monitored_cache.get_anomalies().await;
    info!("Detected anomalies: {}", anomalies.len());
    for anomaly in &anomalies {
        info!(
            "  {}: value {:.1} vs expected {:.1} (deviation: {:.1})",
            anomaly.metric_name, anomaly.value, anomaly.expected_value, anomaly.deviation
        );
    }

    // Get performance recommendations
    let recommendations = monitored_cache.get_recommendations().await;
    info!("Performance recommendations: {}", recommendations.len());
    for rec in &recommendations {
        info!("  {:?}: {}", rec.priority, rec.message);
        info!("    Expected impact: {}", rec.expected_impact);
    }

    Ok(())
}

/// Demonstrate diagnostic reports.
async fn demo_diagnostic_reports() -> Result<(), Box<dyn std::error::Error>> {
    info!("\n📋 Demo 5: Diagnostic Reports");

    let base_cache = Arc::new(UnifiedCache::Memory(MemoryCache::new()));
    let config = MonitoringConfig {
        collection_interval: Duration::from_secs(1),
        max_samples: 20,
        enable_alerting: true,
        enable_performance_analysis: true,
        enable_trend_analysis: true,
        ..Default::default()
    };

    let monitored_cache = CacheMonitor::new(base_cache, config);
    monitored_cache.start_monitoring().await?;

    // Generate some activity
    info!("Generating cache activity for diagnostic report...");

    let ttl = Duration::from_secs(3600);
    let embeddings: Vec<Vec<f32>> = (0..20)
        .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
        .collect();

    // Store embeddings
    for (i, embedding) in embeddings.iter().enumerate() {
        let key = format!("report_embedding_{}", i);
        monitored_cache
            .put_embedding(&key, embedding.clone(), ttl)
            .await?;
    }

    // Mixed access pattern
    for i in 0..30 {
        let key = if i < 15 {
            format!("report_embedding_{}", i % 10) // Some hits
        } else {
            format!("missing_key_{}", i) // Some misses
        };
        let _ = monitored_cache.get_embedding(&key).await?;
    }

    // Wait for data collection
    sleep(Duration::from_secs(5)).await;

    // Generate comprehensive diagnostic report
    info!("Generating comprehensive diagnostic report...");

    let status = monitored_cache.get_monitoring_status().await;
    let metrics_history = monitored_cache.get_metrics_history(None).await;
    let active_alerts = monitored_cache.get_active_alerts().await;
    let recommendations = monitored_cache.get_recommendations().await;
    let trends = monitored_cache.get_trends().await;
    let anomalies = monitored_cache.get_anomalies().await;

    // Print diagnostic report
    info!("\n📊 CACHE DIAGNOSTIC REPORT");
    info!("=" .repeat(50));

    info!("\n🔍 MONITORING STATUS:");
    info!("  Monitoring active: {}", status.is_active);
    info!("  Total samples collected: {}", status.total_samples);
    info!("  Data collection period: {:?}", status.last_collection.elapsed());

    info!("\n📈 PERFORMANCE METRICS:");
    info!("  Average hit rate: {:.1}%", status.current_metrics.avg_hit_rate);
    info!("  Average response time: {:.1}ms", status.current_metrics.avg_response_time);
    info!("  Total operations: {}", status.current_metrics.total_operations);
    info!("  Throughput: {:.1} ops/sec", status.current_metrics.throughput);
    info!("  Error rate: {:.2}%", status.current_metrics.error_rate);

    info!("\n🚨 ALERTS ({}):", active_alerts.len());
    if active_alerts.is_empty() {
        info!("  No active alerts - system is healthy");
    } else {
        for alert in &active_alerts {
            info!("  {} - {}", alert.severity_string(), alert.message);
        }
    }

    info!("\n📊 TRENDS ({}):", trends.len());
    if trends.is_empty() {
        info!("  No trend data available yet");
    } else {
        for (metric, trend) in &trends {
            info!(
                "  {}: {:?} (strength: {:.1}%, confidence: {:.1}%)",
                metric,
                trend.trend_direction,
                trend.trend_strength * 100.0,
                trend.confidence * 100.0
            );
        }
    }

    info!("\n⚠️ ANOMALIES ({}):", anomalies.len());
    if anomalies.is_empty() {
        info!("  No anomalies detected");
    } else {
        for anomaly in &anomalies {
            info!(
                "  {}: {:.1} (expected: {:.1}, deviation: {:.1}σ)",
                anomaly.metric_name, anomaly.value, anomaly.expected_value, anomaly.deviation
            );
        }
    }

    info!("\n💡 RECOMMENDATIONS ({}):", recommendations.len());
    if recommendations.is_empty() {
        info!("  No recommendations - cache is performing optimally");
    } else {
        for rec in &recommendations {
            info!("  [{:?}] {}", rec.priority, rec.message);
            info!("    Impact: {}", rec.expected_impact);
        }
    }

    info!("\n📊 HISTORICAL DATA:");
    info!("  Samples available: {}", metrics_history.len());
    if !metrics_history.is_empty() {
        let first_sample = &metrics_history[0];
        let last_sample = &metrics_history[metrics_history.len() - 1];
        info!(
            "  Hit rate trend: {:.1}% → {:.1}%",
            first_sample.cache_stats.hit_rate(),
            last_sample.cache_stats.hit_rate()
        );
        info!(
            "  Operations trend: {} → {}",
            first_sample.cache_stats.hits + first_sample.cache_stats.misses,
            last_sample.cache_stats.hits + last_sample.cache_stats.misses
        );
    }

    info!("\n✅ DIAGNOSTIC REPORT COMPLETE");

    Ok(())
}

// Helper trait for better alert display
trait AlertDisplay {
    fn alert_type_string(&self) -> &str;
    fn severity_string(&self) -> &str;
}

impl AlertDisplay for cheungfun_core::cache::Alert {
    fn alert_type_string(&self) -> &str {
        match self.alert_type {
            cheungfun_core::cache::AlertType::LowHitRate => "Low Hit Rate",
            cheungfun_core::cache::AlertType::HighResponseTime => "High Response Time",
            cheungfun_core::cache::AlertType::HighErrorRate => "High Error Rate",
            cheungfun_core::cache::AlertType::HighMemoryUsage => "High Memory Usage",
            cheungfun_core::cache::AlertType::CacheSizeFull => "Cache Size Full",
            cheungfun_core::cache::AlertType::HealthDegraded => "Health Degraded",
        }
    }

    fn severity_string(&self) -> &str {
        match self.severity {
            AlertSeverity::Info => "INFO",
            AlertSeverity::Warning => "WARNING",
            AlertSeverity::Critical => "CRITICAL",
        }
    }
}
