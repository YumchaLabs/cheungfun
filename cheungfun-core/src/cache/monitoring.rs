//! Cache monitoring and diagnostics system.
//!
//! This module provides comprehensive monitoring, alerting, and diagnostic
//! capabilities for cache systems, including real-time metrics, health checks,
//! and performance analysis.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::traits::{CacheHealth, CacheStats, HealthStatus, PipelineCache};
use crate::CheungfunError;

/// Comprehensive cache monitoring system.
///
/// This system provides real-time monitoring, alerting, and diagnostic
/// capabilities for cache operations.
#[derive(Debug)]
pub struct CacheMonitor<T> {
    /// The monitored cache
    cache: Arc<T>,
    /// Monitoring configuration
    config: MonitoringConfig,
    /// Real-time metrics collector
    metrics_collector: Arc<RwLock<MetricsCollector>>,
    /// Alert manager
    alert_manager: Arc<RwLock<AlertManager>>,
    /// Diagnostic analyzer
    diagnostic_analyzer: Arc<RwLock<DiagnosticAnalyzer>>,
}

/// Configuration for cache monitoring.
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Metrics collection interval
    pub collection_interval: Duration,
    /// Maximum number of metric samples to keep
    pub max_samples: usize,
    /// Whether to enable real-time alerting
    pub enable_alerting: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Whether to enable performance analysis
    pub enable_performance_analysis: bool,
    /// Performance analysis window
    pub analysis_window: Duration,
    /// Whether to enable trend analysis
    pub enable_trend_analysis: bool,
    /// Trend analysis period
    pub trend_period: Duration,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(30),
            max_samples: 1000,
            enable_alerting: true,
            alert_thresholds: AlertThresholds::default(),
            enable_performance_analysis: true,
            analysis_window: Duration::from_secs(300), // 5 minutes
            enable_trend_analysis: true,
            trend_period: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Alert thresholds for various metrics.
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Minimum hit rate before alerting (percentage)
    pub min_hit_rate: f64,
    /// Maximum response time before alerting (milliseconds)
    pub max_response_time: u64,
    /// Maximum error rate before alerting (percentage)
    pub max_error_rate: f64,
    /// Maximum memory usage before alerting (percentage)
    pub max_memory_usage: f64,
    /// Maximum cache size before alerting (percentage)
    pub max_cache_size: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            min_hit_rate: 70.0,
            max_response_time: 100,
            max_error_rate: 5.0,
            max_memory_usage: 85.0,
            max_cache_size: 90.0,
        }
    }
}

/// Real-time metrics collector.
#[derive(Debug)]
struct MetricsCollector {
    /// Time-series metrics samples
    samples: VecDeque<MetricsSample>,
    /// Current aggregated metrics
    current_metrics: AggregatedMetrics,
    /// Last collection time
    last_collection: Instant,
}

/// A single metrics sample at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSample {
    /// Timestamp of the sample
    pub timestamp: u64,
    /// Cache statistics at this time
    pub cache_stats: CacheStats,
    /// Cache health at this time
    pub cache_health: CacheHealth,
    /// Response time metrics
    pub response_times: ResponseTimeMetrics,
    /// Error metrics
    pub error_metrics: ErrorMetrics,
}

/// Response time metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeMetrics {
    /// Average response time (milliseconds)
    pub avg_response_time: f64,
    /// 95th percentile response time
    pub p95_response_time: f64,
    /// 99th percentile response time
    pub p99_response_time: f64,
    /// Maximum response time
    pub max_response_time: f64,
}

/// Error metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Total number of errors
    pub total_errors: u64,
    /// Error rate (errors per second)
    pub error_rate: f64,
    /// Error types and their counts
    pub error_types: HashMap<String, u64>,
}

/// Aggregated metrics over a time period.
#[derive(Debug, Clone)]
pub struct AggregatedMetrics {
    /// Time period for aggregation
    pub period: Duration,
    /// Average hit rate over the period
    pub avg_hit_rate: f64,
    /// Average response time over the period
    pub avg_response_time: f64,
    /// Total operations over the period
    pub total_operations: u64,
    /// Error rate over the period
    pub error_rate: f64,
    /// Throughput (operations per second)
    pub throughput: f64,
}

/// Alert manager for cache monitoring.
#[derive(Debug)]
struct AlertManager {
    /// Active alerts
    active_alerts: HashMap<String, Alert>,
    /// Alert history
    alert_history: VecDeque<Alert>,
}

/// An alert generated by the monitoring system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Unique alert ID
    pub id: String,
    /// Alert severity level
    pub severity: AlertSeverity,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert message
    pub message: String,
    /// Timestamp when alert was triggered
    pub timestamp: u64,
    /// Metric value that triggered the alert
    pub metric_value: f64,
    /// Threshold that was exceeded
    pub threshold: f64,
    /// Whether the alert is resolved
    pub resolved: bool,
    /// Resolution timestamp
    pub resolved_at: Option<u64>,
}

/// Alert severity levels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Critical alert
    Critical,
}

/// Types of alerts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertType {
    /// Low hit rate alert
    LowHitRate,
    /// High response time alert
    HighResponseTime,
    /// High error rate alert
    HighErrorRate,
    /// High memory usage alert
    HighMemoryUsage,
    /// Cache size alert
    CacheSizeFull,
    /// Cache health degraded
    HealthDegraded,
}

/// Diagnostic analyzer for cache performance.
#[derive(Debug)]
struct DiagnosticAnalyzer {
    /// Performance trends
    trends: HashMap<String, TrendAnalysis>,
    /// Anomaly detection state
    anomaly_detector: AnomalyDetector,
    /// Performance recommendations
    recommendations: Vec<PerformanceRecommendation>,
}

/// Trend analysis for a specific metric.
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Metric name
    pub metric_name: String,
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub trend_strength: f64,
    /// Predicted future value
    pub predicted_value: f64,
    /// Confidence in prediction (0.0 to 1.0)
    pub confidence: f64,
}

/// Trend direction.
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// Stable trend
    Stable,
}

/// Anomaly detector for cache metrics.
#[derive(Debug)]
struct AnomalyDetector {
    /// Baseline metrics for comparison
    baseline: HashMap<String, f64>,
    /// Anomaly threshold (standard deviations)
    threshold: f64,
    /// Detected anomalies
    anomalies: VecDeque<Anomaly>,
}

/// A detected anomaly.
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Metric name
    pub metric_name: String,
    /// Anomalous value
    pub value: f64,
    /// Expected value
    pub expected_value: f64,
    /// Deviation from expected (standard deviations)
    pub deviation: f64,
    /// Timestamp of anomaly
    pub timestamp: u64,
}

/// Performance recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    /// Recommendation ID
    pub id: String,
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Recommendation message
    pub message: String,
    /// Expected impact
    pub expected_impact: String,
    /// Implementation priority
    pub priority: RecommendationPriority,
    /// Timestamp when recommendation was generated
    pub timestamp: u64,
}

/// Types of performance recommendations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Increase cache size
    IncreaseCacheSize,
    /// Adjust TTL settings
    AdjustTtl,
    /// Enable compression
    EnableCompression,
    /// Optimize batch size
    OptimizeBatchSize,
    /// Enable prefetching
    EnablePrefetching,
    /// Tune eviction policy
    TuneEvictionPolicy,
}

/// Recommendation priority levels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

impl<T> CacheMonitor<T>
where
    T: PipelineCache<Error = CheungfunError> + Send + Sync + 'static,
{
    /// Create a new cache monitor.
    pub fn new(cache: Arc<T>, config: MonitoringConfig) -> Self {
        let metrics_collector = MetricsCollector {
            samples: VecDeque::with_capacity(config.max_samples),
            current_metrics: AggregatedMetrics {
                period: config.collection_interval,
                avg_hit_rate: 0.0,
                avg_response_time: 0.0,
                total_operations: 0,
                error_rate: 0.0,
                throughput: 0.0,
            },
            last_collection: Instant::now(),
        };

        let alert_manager = AlertManager {
            active_alerts: HashMap::new(),
            alert_history: VecDeque::new(),
        };

        let diagnostic_analyzer = DiagnosticAnalyzer {
            trends: HashMap::new(),
            anomaly_detector: AnomalyDetector {
                baseline: HashMap::new(),
                threshold: 2.0, // 2 standard deviations
                anomalies: VecDeque::new(),
            },
            recommendations: Vec::new(),
        };

        Self {
            cache,
            config,
            metrics_collector: Arc::new(RwLock::new(metrics_collector)),
            alert_manager: Arc::new(RwLock::new(alert_manager)),
            diagnostic_analyzer: Arc::new(RwLock::new(diagnostic_analyzer)),
        }
    }

    /// Start the monitoring system.
    ///
    /// # Errors
    ///
    /// Returns an error if the monitoring system fails to start.
    pub async fn start_monitoring(&self) -> Result<(), CheungfunError> {
        info!("Starting cache monitoring system");

        // Start metrics collection task
        let cache = self.cache.clone();
        let metrics_collector = self.metrics_collector.clone();
        let alert_manager = self.alert_manager.clone();
        let diagnostic_analyzer = self.diagnostic_analyzer.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.collection_interval);

            loop {
                interval.tick().await;

                if let Err(e) = Self::collect_metrics(
                    &cache,
                    &metrics_collector,
                    &alert_manager,
                    &diagnostic_analyzer,
                    &config,
                )
                .await
                {
                    error!("Failed to collect metrics: {}", e);
                }
            }
        });

        info!("Cache monitoring system started");
        Ok(())
    }

    /// Collect metrics from the cache.
    async fn collect_metrics(
        cache: &Arc<T>,
        metrics_collector: &Arc<RwLock<MetricsCollector>>,
        alert_manager: &Arc<RwLock<AlertManager>>,
        diagnostic_analyzer: &Arc<RwLock<DiagnosticAnalyzer>>,
        config: &MonitoringConfig,
    ) -> Result<(), CheungfunError> {
        let start_time = Instant::now();

        // Collect cache statistics and health
        let cache_stats = cache.stats().await?;
        let cache_health = cache.health().await?;

        // Create metrics sample
        let sample = MetricsSample {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            cache_stats: cache_stats.clone(),
            cache_health: cache_health.clone(),
            response_times: ResponseTimeMetrics {
                avg_response_time: 10.0, // Placeholder - would be calculated from actual measurements
                p95_response_time: 15.0,
                p99_response_time: 25.0,
                max_response_time: 50.0,
            },
            error_metrics: ErrorMetrics {
                total_errors: 0,
                error_rate: 0.0,
                error_types: HashMap::new(),
            },
        };

        // Update metrics collector
        {
            let mut collector = metrics_collector.write().await;
            collector.samples.push_back(sample.clone());

            // Keep only the most recent samples
            while collector.samples.len() > config.max_samples {
                collector.samples.pop_front();
            }

            // Update aggregated metrics
            collector.current_metrics =
                Self::calculate_aggregated_metrics(&collector.samples, config.analysis_window);
            collector.last_collection = start_time;
        }

        // Check for alerts
        if config.enable_alerting {
            Self::check_alerts(&sample, alert_manager, &config.alert_thresholds).await;
        }

        // Perform diagnostic analysis
        if config.enable_performance_analysis {
            Self::analyze_performance(&sample, diagnostic_analyzer, config).await;
        }

        debug!("Metrics collection completed in {:?}", start_time.elapsed());
        Ok(())
    }

    /// Calculate aggregated metrics from samples.
    fn calculate_aggregated_metrics(
        samples: &VecDeque<MetricsSample>,
        window: Duration,
    ) -> AggregatedMetrics {
        if samples.is_empty() {
            return AggregatedMetrics {
                period: window,
                avg_hit_rate: 0.0,
                avg_response_time: 0.0,
                total_operations: 0,
                error_rate: 0.0,
                throughput: 0.0,
            };
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let window_start = now.saturating_sub(window.as_secs());

        let recent_samples: Vec<_> = samples
            .iter()
            .filter(|sample| sample.timestamp >= window_start)
            .collect();

        if recent_samples.is_empty() {
            return AggregatedMetrics {
                period: window,
                avg_hit_rate: 0.0,
                avg_response_time: 0.0,
                total_operations: 0,
                error_rate: 0.0,
                throughput: 0.0,
            };
        }

        let avg_hit_rate = recent_samples
            .iter()
            .map(|s| s.cache_stats.hit_rate())
            .sum::<f64>()
            / recent_samples.len() as f64;

        let avg_response_time = recent_samples
            .iter()
            .map(|s| s.response_times.avg_response_time)
            .sum::<f64>()
            / recent_samples.len() as f64;

        let total_operations = recent_samples
            .iter()
            .map(|s| s.cache_stats.hits + s.cache_stats.misses)
            .sum::<u64>();

        let error_rate = recent_samples
            .iter()
            .map(|s| s.error_metrics.error_rate)
            .sum::<f64>()
            / recent_samples.len() as f64;

        let throughput = total_operations as f64 / window.as_secs_f64();

        AggregatedMetrics {
            period: window,
            avg_hit_rate,
            avg_response_time,
            total_operations,
            error_rate,
            throughput,
        }
    }

    /// Check for alert conditions.
    async fn check_alerts(
        sample: &MetricsSample,
        alert_manager: &Arc<RwLock<AlertManager>>,
        thresholds: &AlertThresholds,
    ) {
        let mut manager = alert_manager.write().await;
        let timestamp = sample.timestamp;

        // Check hit rate
        let hit_rate = sample.cache_stats.hit_rate();
        if hit_rate < thresholds.min_hit_rate {
            let alert_id = format!("hit_rate_{timestamp}");
            if !manager.active_alerts.contains_key(&alert_id) {
                let alert = Alert {
                    id: alert_id.clone(),
                    severity: AlertSeverity::Warning,
                    alert_type: AlertType::LowHitRate,
                    message: format!(
                        "Cache hit rate ({:.1}%) is below threshold ({:.1}%)",
                        hit_rate, thresholds.min_hit_rate
                    ),
                    timestamp,
                    metric_value: hit_rate,
                    threshold: thresholds.min_hit_rate,
                    resolved: false,
                    resolved_at: None,
                };
                manager.active_alerts.insert(alert_id, alert.clone());
                manager.alert_history.push_back(alert);
                warn!("Alert triggered: Low hit rate ({:.1}%)", hit_rate);
            }
        }

        // Check response time
        let response_time = sample.response_times.avg_response_time;
        if response_time > thresholds.max_response_time as f64 {
            let alert_id = format!("response_time_{timestamp}");
            if !manager.active_alerts.contains_key(&alert_id) {
                let alert = Alert {
                    id: alert_id.clone(),
                    severity: AlertSeverity::Warning,
                    alert_type: AlertType::HighResponseTime,
                    message: format!(
                        "Average response time ({:.1}ms) exceeds threshold ({}ms)",
                        response_time, thresholds.max_response_time
                    ),
                    timestamp,
                    metric_value: response_time,
                    threshold: thresholds.max_response_time as f64,
                    resolved: false,
                    resolved_at: None,
                };
                manager.active_alerts.insert(alert_id, alert.clone());
                manager.alert_history.push_back(alert);
                warn!(
                    "Alert triggered: High response time ({:.1}ms)",
                    response_time
                );
            }
        }

        // Check cache health
        if sample.cache_health.status == HealthStatus::Critical {
            let alert_id = format!("health_critical_{timestamp}");
            if !manager.active_alerts.contains_key(&alert_id) {
                let alert = Alert {
                    id: alert_id.clone(),
                    severity: AlertSeverity::Critical,
                    alert_type: AlertType::HealthDegraded,
                    message: "Cache health status is critical".to_string(),
                    timestamp,
                    metric_value: 0.0, // Health is not numeric
                    threshold: 0.0,
                    resolved: false,
                    resolved_at: None,
                };
                manager.active_alerts.insert(alert_id, alert.clone());
                manager.alert_history.push_back(alert);
                error!("Alert triggered: Cache health is critical");
            }
        }
    }

    /// Perform performance analysis.
    async fn analyze_performance(
        sample: &MetricsSample,
        diagnostic_analyzer: &Arc<RwLock<DiagnosticAnalyzer>>,
        config: &MonitoringConfig,
    ) {
        let mut analyzer = diagnostic_analyzer.write().await;

        // Update trend analysis
        if config.enable_trend_analysis {
            Self::update_trend_analysis(&mut analyzer, sample);
        }

        // Detect anomalies
        Self::detect_anomalies(&mut analyzer, sample);

        // Generate recommendations
        Self::generate_recommendations(&mut analyzer, sample);
    }

    /// Update trend analysis.
    fn update_trend_analysis(analyzer: &mut DiagnosticAnalyzer, sample: &MetricsSample) {
        let hit_rate = sample.cache_stats.hit_rate();

        // Simple trend analysis for hit rate
        let trend = analyzer
            .trends
            .entry("hit_rate".to_string())
            .or_insert_with(|| TrendAnalysis {
                metric_name: "hit_rate".to_string(),
                trend_direction: TrendDirection::Stable,
                trend_strength: 0.0,
                predicted_value: hit_rate,
                confidence: 0.5,
            });

        // Update trend (simplified implementation)
        let diff = hit_rate - trend.predicted_value;
        if diff.abs() > 5.0 {
            trend.trend_direction = if diff > 0.0 {
                TrendDirection::Increasing
            } else {
                TrendDirection::Decreasing
            };
            trend.trend_strength = (diff.abs() / 100.0).min(1.0);
        } else {
            trend.trend_direction = TrendDirection::Stable;
            trend.trend_strength = 0.0;
        }

        trend.predicted_value = hit_rate;
        trend.confidence = 0.8; // Simplified confidence calculation
    }

    /// Detect anomalies in metrics.
    fn detect_anomalies(analyzer: &mut DiagnosticAnalyzer, sample: &MetricsSample) {
        let hit_rate = sample.cache_stats.hit_rate();
        let baseline_hit_rate = analyzer
            .anomaly_detector
            .baseline
            .get("hit_rate")
            .copied()
            .unwrap_or(hit_rate);

        let deviation = (hit_rate - baseline_hit_rate).abs();
        if deviation > analyzer.anomaly_detector.threshold * 10.0 {
            // 10% as simplified standard deviation
            let anomaly = Anomaly {
                metric_name: "hit_rate".to_string(),
                value: hit_rate,
                expected_value: baseline_hit_rate,
                deviation: deviation / 10.0,
                timestamp: sample.timestamp,
            };

            analyzer.anomaly_detector.anomalies.push_back(anomaly);

            // Keep only recent anomalies
            while analyzer.anomaly_detector.anomalies.len() > 100 {
                analyzer.anomaly_detector.anomalies.pop_front();
            }
        }

        // Update baseline
        analyzer
            .anomaly_detector
            .baseline
            .insert("hit_rate".to_string(), hit_rate);
    }

    /// Generate performance recommendations.
    fn generate_recommendations(analyzer: &mut DiagnosticAnalyzer, sample: &MetricsSample) {
        let hit_rate = sample.cache_stats.hit_rate();

        // Clear old recommendations
        analyzer.recommendations.clear();

        // Generate recommendations based on metrics
        if hit_rate < 50.0 {
            let recommendation = PerformanceRecommendation {
                id: format!("rec_{}", sample.timestamp),
                recommendation_type: RecommendationType::IncreaseCacheSize,
                message: "Consider increasing cache size to improve hit rate".to_string(),
                expected_impact: "Should increase hit rate by 10-20%".to_string(),
                priority: RecommendationPriority::High,
                timestamp: sample.timestamp,
            };
            analyzer.recommendations.push(recommendation);
        }

        if sample.response_times.avg_response_time > 50.0 {
            let recommendation = PerformanceRecommendation {
                id: format!("rec_batch_{}", sample.timestamp),
                recommendation_type: RecommendationType::OptimizeBatchSize,
                message: "Consider optimizing batch size to reduce response time".to_string(),
                expected_impact: "Should reduce response time by 20-30%".to_string(),
                priority: RecommendationPriority::Medium,
                timestamp: sample.timestamp,
            };
            analyzer.recommendations.push(recommendation);
        }
    }

    /// Get current monitoring status.
    pub async fn get_monitoring_status(&self) -> MonitoringStatus {
        let metrics_collector = self.metrics_collector.read().await;
        let alert_manager = self.alert_manager.read().await;
        let diagnostic_analyzer = self.diagnostic_analyzer.read().await;

        MonitoringStatus {
            is_active: true,
            last_collection: metrics_collector.last_collection,
            total_samples: metrics_collector.samples.len(),
            active_alerts: alert_manager.active_alerts.len(),
            current_metrics: metrics_collector.current_metrics.clone(),
            recent_anomalies: diagnostic_analyzer.anomaly_detector.anomalies.len(),
            recommendations: diagnostic_analyzer.recommendations.len(),
        }
    }

    /// Get active alerts.
    pub async fn get_active_alerts(&self) -> Vec<Alert> {
        let alert_manager = self.alert_manager.read().await;
        alert_manager.active_alerts.values().cloned().collect()
    }

    /// Get performance recommendations.
    pub async fn get_recommendations(&self) -> Vec<PerformanceRecommendation> {
        let diagnostic_analyzer = self.diagnostic_analyzer.read().await;
        diagnostic_analyzer.recommendations.clone()
    }

    /// Get trend analysis.
    pub async fn get_trends(&self) -> HashMap<String, TrendAnalysis> {
        let diagnostic_analyzer = self.diagnostic_analyzer.read().await;
        diagnostic_analyzer.trends.clone()
    }

    /// Get recent anomalies.
    pub async fn get_anomalies(&self) -> Vec<Anomaly> {
        let diagnostic_analyzer = self.diagnostic_analyzer.read().await;
        diagnostic_analyzer
            .anomaly_detector
            .anomalies
            .iter()
            .cloned()
            .collect()
    }

    /// Get metrics history.
    pub async fn get_metrics_history(&self, limit: Option<usize>) -> Vec<MetricsSample> {
        let metrics_collector = self.metrics_collector.read().await;
        let samples: Vec<_> = metrics_collector.samples.iter().cloned().collect();

        if let Some(limit) = limit {
            samples.into_iter().rev().take(limit).rev().collect()
        } else {
            samples
        }
    }
}

/// Current monitoring status.
#[derive(Debug, Clone)]
pub struct MonitoringStatus {
    /// Whether monitoring is active
    pub is_active: bool,
    /// Last metrics collection time
    pub last_collection: Instant,
    /// Total number of samples collected
    pub total_samples: usize,
    /// Number of active alerts
    pub active_alerts: usize,
    /// Current aggregated metrics
    pub current_metrics: AggregatedMetrics,
    /// Number of recent anomalies
    pub recent_anomalies: usize,
    /// Number of current recommendations
    pub recommendations: usize,
}

#[async_trait]
impl<T> PipelineCache for CacheMonitor<T>
where
    T: PipelineCache<Error = CheungfunError> + Send + Sync,
{
    type Error = CheungfunError;

    async fn get_embedding(&self, key: &str) -> Result<Option<Vec<f32>>, Self::Error> {
        let start_time = Instant::now();
        let result = self.cache.get_embedding(key).await;
        let duration = start_time.elapsed();

        // Record metrics
        self.record_operation("get_embedding", duration, result.is_ok())
            .await;

        result
    }

    async fn put_embedding(
        &self,
        key: &str,
        embedding: Vec<f32>,
        ttl: Duration,
    ) -> Result<(), Self::Error> {
        let start_time = Instant::now();
        let result = self.cache.put_embedding(key, embedding, ttl).await;
        let duration = start_time.elapsed();

        // Record metrics
        self.record_operation("put_embedding", duration, result.is_ok())
            .await;

        result
    }

    async fn get_embeddings_batch(
        &self,
        keys: &[&str],
    ) -> Result<Vec<Option<Vec<f32>>>, Self::Error> {
        let start_time = Instant::now();
        let result = self.cache.get_embeddings_batch(keys).await;
        let duration = start_time.elapsed();

        // Record metrics
        self.record_operation("get_embeddings_batch", duration, result.is_ok())
            .await;

        result
    }

    async fn put_embeddings_batch(
        &self,
        items: &[(&str, Vec<f32>, Duration)],
    ) -> Result<(), Self::Error> {
        let start_time = Instant::now();
        let result = self.cache.put_embeddings_batch(items).await;
        let duration = start_time.elapsed();

        // Record metrics
        self.record_operation("put_embeddings_batch", duration, result.is_ok())
            .await;

        result
    }

    async fn get_nodes(&self, key: &str) -> Result<Option<Vec<crate::Node>>, Self::Error> {
        let start_time = Instant::now();
        let result = self.cache.get_nodes(key).await;
        let duration = start_time.elapsed();

        // Record metrics
        self.record_operation("get_nodes", duration, result.is_ok())
            .await;

        result
    }

    async fn put_nodes(
        &self,
        key: &str,
        nodes: Vec<crate::Node>,
        ttl: Duration,
    ) -> Result<(), Self::Error> {
        let start_time = Instant::now();
        let result = self.cache.put_nodes(key, nodes, ttl).await;
        let duration = start_time.elapsed();

        // Record metrics
        self.record_operation("put_nodes", duration, result.is_ok())
            .await;

        result
    }

    async fn get_nodes_batch(
        &self,
        keys: &[&str],
    ) -> Result<Vec<Option<Vec<crate::Node>>>, Self::Error> {
        let start_time = Instant::now();
        let result = self.cache.get_nodes_batch(keys).await;
        let duration = start_time.elapsed();

        // Record metrics
        self.record_operation("get_nodes_batch", duration, result.is_ok())
            .await;

        result
    }

    async fn put_nodes_batch(
        &self,
        items: &[(&str, Vec<crate::Node>, Duration)],
    ) -> Result<(), Self::Error> {
        let start_time = Instant::now();
        let result = self.cache.put_nodes_batch(items).await;
        let duration = start_time.elapsed();

        // Record metrics
        self.record_operation("put_nodes_batch", duration, result.is_ok())
            .await;

        result
    }

    async fn get_data_bytes(&self, key: &str) -> Result<Option<Vec<u8>>, Self::Error> {
        let start_time = Instant::now();
        let result = self.cache.get_data_bytes(key).await;
        let duration = start_time.elapsed();

        // Record metrics
        self.record_operation("get_data_bytes", duration, result.is_ok())
            .await;

        result
    }

    async fn put_data_bytes(
        &self,
        key: &str,
        data_bytes: Vec<u8>,
        ttl: Duration,
    ) -> Result<(), Self::Error> {
        let start_time = Instant::now();
        let result = self.cache.put_data_bytes(key, data_bytes, ttl).await;
        let duration = start_time.elapsed();

        // Record metrics
        self.record_operation("put_data_bytes", duration, result.is_ok())
            .await;

        result
    }

    async fn exists(&self, key: &str) -> Result<bool, Self::Error> {
        let start_time = Instant::now();
        let result = self.cache.exists(key).await;
        let duration = start_time.elapsed();

        // Record metrics
        self.record_operation("exists", duration, result.is_ok())
            .await;

        result
    }

    async fn remove(&self, key: &str) -> Result<(), Self::Error> {
        let start_time = Instant::now();
        let result = self.cache.remove(key).await;
        let duration = start_time.elapsed();

        // Record metrics
        self.record_operation("remove", duration, result.is_ok())
            .await;

        result
    }

    async fn clear(&self) -> Result<(), Self::Error> {
        let start_time = Instant::now();
        let result = self.cache.clear().await;
        let duration = start_time.elapsed();

        // Record metrics
        self.record_operation("clear", duration, result.is_ok())
            .await;

        result
    }

    async fn cleanup(&self) -> Result<usize, Self::Error> {
        let start_time = Instant::now();
        let result = self.cache.cleanup().await;
        let duration = start_time.elapsed();

        // Record metrics
        self.record_operation("cleanup", duration, result.is_ok())
            .await;

        result
    }

    async fn stats(&self) -> Result<CacheStats, Self::Error> {
        self.cache.stats().await
    }

    async fn health(&self) -> Result<CacheHealth, Self::Error> {
        self.cache.health().await
    }
}

impl<T> CacheMonitor<T>
where
    T: PipelineCache<Error = CheungfunError> + Send + Sync,
{
    /// Record an operation for monitoring.
    async fn record_operation(&self, operation: &str, duration: Duration, success: bool) {
        debug!(
            "Cache operation: {} completed in {:?} (success: {})",
            operation, duration, success
        );

        // This would typically update internal metrics
        // For now, we just log the operation
    }
}
