//! Comprehensive embedder performance benchmarks
//!
//! This benchmark compares the performance of different embedder implementations:
//! - FastEmbedder (local model via fastembed)
//! - ApiEmbedder (cloud-based via siumai)
//! - CandleEmbedder (local model via candle)

use anyhow::Result;
use cheungfun_core::traits::Embedder;
use cheungfun_integrations::embedders::api::{ApiEmbedder, ApiEmbedderConfig};
use rand::Rng;
use std::time::Instant;
use tokio;
use tracing::{info, warn};

use cheungfun_examples::benchmark_framework::{
    format_metrics, run_benchmark, BenchmarkConfig, PerformanceMetrics,
};

#[cfg(feature = "fastembed")]
use cheungfun_integrations::embedders::fastembed::{FastEmbedder, FastEmbedderConfig};

#[cfg(feature = "candle")]
use cheungfun_integrations::embedders::candle::{CandleEmbedder, CandleEmbedderConfig};

/// Test data generator
struct TestDataGenerator {
    rng: rand::rngs::ThreadRng,
}

impl TestDataGenerator {
    fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }

    /// Generate a single test sentence
    fn generate_sentence(&mut self) -> String {
        let subjects = [
            "The cat",
            "A dog",
            "The scientist",
            "An engineer",
            "The student",
        ];
        let verbs = ["runs", "jumps", "studies", "develops", "analyzes"];
        let objects = [
            "quickly",
            "carefully",
            "efficiently",
            "thoroughly",
            "systematically",
        ];
        let contexts = [
            "in the park",
            "at home",
            "in the lab",
            "at work",
            "in the library",
        ];

        format!(
            "{} {} {} {}.",
            subjects[self.rng.gen_range(0..subjects.len())],
            verbs[self.rng.gen_range(0..verbs.len())],
            objects[self.rng.gen_range(0..objects.len())],
            contexts[self.rng.gen_range(0..contexts.len())]
        )
    }

    /// Generate multiple test sentences
    fn generate_sentences(&mut self, count: usize) -> Vec<String> {
        (0..count).map(|_| self.generate_sentence()).collect()
    }

    /// Generate sentences of varying lengths
    fn generate_varied_sentences(&mut self, count: usize) -> Vec<String> {
        let mut sentences = Vec::new();
        for i in 0..count {
            let length = match i % 4 {
                0 => "Short text.",
                1 => "This is a medium length sentence with some additional content.",
                2 => {
                    "This is a longer sentence that contains significantly more words and provides more context for embedding generation, which helps test performance with varying input sizes."
                }
                3 => {
                    "This is an extremely long sentence that goes on and on with lots of details, multiple clauses, and extensive information that really pushes the boundaries of what might be considered a reasonable input length for embedding generation, designed specifically to test how well the embedder handles very long inputs and whether performance degrades significantly with increased input length."
                }
                _ => unreachable!(),
            };
            sentences.push(format!("{} ({})", length, i));
        }
        sentences
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Cheungfun Embedder Performance Benchmark");
    println!("==========================================");
    println!();

    let mut generator = TestDataGenerator::new();
    let mut all_metrics = Vec::new();

    // Test configurations
    let single_text_config = BenchmarkConfig {
        name: "Single Text Embedding".to_string(),
        warmup_iterations: 5,
        measurement_iterations: 50,
        ..Default::default()
    };

    let batch_config = BenchmarkConfig {
        name: "Batch Embedding".to_string(),
        warmup_iterations: 3,
        measurement_iterations: 20,
        ..Default::default()
    };

    let large_batch_config = BenchmarkConfig {
        name: "Large Batch Embedding".to_string(),
        warmup_iterations: 2,
        measurement_iterations: 10,
        ..Default::default()
    };

    // Benchmark FastEmbedder
    #[cfg(feature = "fastembed")]
    {
        println!("🔥 Benchmarking FastEmbedder");
        println!("----------------------------");

        match benchmark_fastembed(
            &mut generator,
            &single_text_config,
            &batch_config,
            &large_batch_config,
        )
        .await
        {
            Ok(mut metrics) => {
                all_metrics.append(&mut metrics);
            }
            Err(e) => {
                warn!("FastEmbedder benchmark failed: {}", e);
            }
        }
        println!();
    }

    // Benchmark ApiEmbedder
    println!("☁️  Benchmarking ApiEmbedder");
    println!("---------------------------");

    match benchmark_api_embedder(&mut generator, &single_text_config, &batch_config).await {
        Ok(mut metrics) => {
            all_metrics.append(&mut metrics);
        }
        Err(e) => {
            warn!("ApiEmbedder benchmark failed: {}", e);
        }
    }
    println!();

    // Benchmark CandleEmbedder
    #[cfg(feature = "candle")]
    {
        println!("🕯️  Benchmarking CandleEmbedder");
        println!("------------------------------");

        match benchmark_candle_embedder(&mut generator, &single_text_config, &batch_config).await {
            Ok(mut metrics) => {
                all_metrics.append(&mut metrics);
            }
            Err(e) => {
                warn!("CandleEmbedder benchmark failed: {}", e);
            }
        }
        println!();
    }

    // Generate comparison report
    generate_comparison_report(&all_metrics);

    Ok(())
}

#[cfg(feature = "fastembed")]
async fn benchmark_fastembed(
    generator: &mut TestDataGenerator,
    single_config: &BenchmarkConfig,
    batch_config: &BenchmarkConfig,
    large_batch_config: &BenchmarkConfig,
) -> Result<Vec<PerformanceMetrics>> {
    let mut metrics = Vec::new();

    // Initialize FastEmbedder
    info!("Initializing FastEmbedder...");
    let config = FastEmbedderConfig::default();
    let embedder = FastEmbedder::new(config).await?;
    info!("FastEmbedder initialized successfully");

    // Single text benchmark
    let test_text = generator.generate_sentence();
    let single_metrics = run_benchmark(single_config.clone(), || {
        let embedder = &embedder;
        let text = &test_text;
        async move { embedder.embed(text).await.map_err(|e| anyhow::anyhow!(e)) }
    })
    .await?;

    println!("{}", format_metrics(&single_metrics));
    metrics.push(single_metrics);

    // Batch benchmark
    let batch_texts = generator.generate_sentences(10);
    let batch_refs: Vec<&str> = batch_texts.iter().map(|s| s.as_str()).collect();
    let batch_metrics = run_benchmark(batch_config.clone(), || {
        let embedder = &embedder;
        let texts = batch_refs.clone();
        async move {
            embedder
                .embed_batch(texts)
                .await
                .map_err(|e| anyhow::anyhow!(e))
        }
    })
    .await?;

    println!("{}", format_metrics(&batch_metrics));
    metrics.push(batch_metrics);

    // Large batch benchmark
    let large_batch_texts = generator.generate_sentences(100);
    let large_batch_refs: Vec<&str> = large_batch_texts.iter().map(|s| s.as_str()).collect();
    let large_batch_metrics = run_benchmark(large_batch_config.clone(), || {
        let embedder = &embedder;
        let texts = large_batch_refs.clone();
        async move {
            embedder
                .embed_batch(texts)
                .await
                .map_err(|e| anyhow::anyhow!(e))
        }
    })
    .await?;

    println!("{}", format_metrics(&large_batch_metrics));
    metrics.push(large_batch_metrics);

    Ok(metrics)
}

async fn benchmark_api_embedder(
    generator: &mut TestDataGenerator,
    single_config: &BenchmarkConfig,
    batch_config: &BenchmarkConfig,
) -> Result<Vec<PerformanceMetrics>> {
    let mut metrics = Vec::new();

    // Initialize ApiEmbedder (requires API key)
    let api_key = std::env::var("OPENAI_API_KEY")
        .or_else(|_| std::env::var("SIUMAI_API_KEY"))
        .unwrap_or_else(|_| {
            warn!("No API key found. Skipping ApiEmbedder benchmark.");
            return String::new();
        });

    if api_key.is_empty() {
        warn!("API key is empty. Skipping ApiEmbedder benchmark.");
        return Ok(metrics);
    }

    info!("Initializing ApiEmbedder...");
    let config = ApiEmbedderConfig::new()
        .with_api_key(api_key)
        .with_model("text-embedding-3-small")
        .with_batch_size(10);

    let embedder = ApiEmbedder::new(config).await?;
    info!("ApiEmbedder initialized successfully");

    // Single text benchmark
    let test_text = generator.generate_sentence();
    let single_metrics = run_benchmark(single_config.clone(), || {
        let embedder = &embedder;
        let text = &test_text;
        async move { embedder.embed(text).await.map_err(|e| anyhow::anyhow!(e)) }
    })
    .await?;

    println!("{}", format_metrics(&single_metrics));
    metrics.push(single_metrics);

    // Batch benchmark (smaller batch for API to avoid rate limits)
    let batch_texts = generator.generate_sentences(5);
    let batch_refs: Vec<&str> = batch_texts.iter().map(|s| s.as_str()).collect();
    let batch_metrics = run_benchmark(batch_config.clone(), || {
        let embedder = &embedder;
        let texts = batch_refs.clone();
        async move {
            embedder
                .embed_batch(texts)
                .await
                .map_err(|e| anyhow::anyhow!(e))
        }
    })
    .await?;

    println!("{}", format_metrics(&batch_metrics));
    metrics.push(batch_metrics);

    Ok(metrics)
}

#[cfg(feature = "candle")]
async fn benchmark_candle_embedder(
    generator: &mut TestDataGenerator,
    single_config: &BenchmarkConfig,
    batch_config: &BenchmarkConfig,
) -> Result<Vec<PerformanceMetrics>> {
    let mut metrics = Vec::new();

    // Initialize CandleEmbedder
    info!("Initializing CandleEmbedder...");
    let config = CandleEmbedderConfig::default();
    let embedder = CandleEmbedder::from_pretrained(config).await?;
    info!("CandleEmbedder initialized successfully");

    // Single text benchmark
    let test_text = generator.generate_sentence();
    let single_metrics = run_benchmark(single_config.clone(), || {
        let embedder = &embedder;
        let text = &test_text;
        async move { embedder.embed(text).await.map_err(|e| anyhow::anyhow!(e)) }
    })
    .await?;

    println!("{}", format_metrics(&single_metrics));
    metrics.push(single_metrics);

    // Batch benchmark
    let batch_texts = generator.generate_sentences(10);
    let batch_refs: Vec<&str> = batch_texts.iter().map(|s| s.as_str()).collect();
    let batch_metrics = run_benchmark(batch_config.clone(), || {
        let embedder = &embedder;
        let texts = batch_refs.clone();
        async move {
            embedder
                .embed_batch(texts)
                .await
                .map_err(|e| anyhow::anyhow!(e))
        }
    })
    .await?;

    println!("{}", format_metrics(&batch_metrics));
    metrics.push(batch_metrics);

    Ok(metrics)
}

fn generate_comparison_report(all_metrics: &[PerformanceMetrics]) {
    if all_metrics.is_empty() {
        println!("⚠️  No metrics collected for comparison");
        return;
    }

    println!("📊 Performance Comparison Report");
    println!("===============================");
    println!();

    // Group metrics by benchmark type
    let mut single_text_metrics = Vec::new();
    let mut batch_metrics = Vec::new();
    let mut large_batch_metrics = Vec::new();

    for metric in all_metrics {
        if metric.benchmark_name.contains("Single") {
            single_text_metrics.push(metric);
        } else if metric.benchmark_name.contains("Large") {
            large_batch_metrics.push(metric);
        } else if metric.benchmark_name.contains("Batch") {
            batch_metrics.push(metric);
        }
    }

    // Compare single text performance
    if !single_text_metrics.is_empty() {
        println!("🔍 Single Text Embedding Comparison:");
        for metric in &single_text_metrics {
            println!(
                "  • {}: {:.2} ops/sec, {:?} avg latency",
                metric.benchmark_name, metric.ops_per_second, metric.avg_latency
            );
        }
        println!();
    }

    // Compare batch performance
    if !batch_metrics.is_empty() {
        println!("📦 Batch Embedding Comparison:");
        for metric in &batch_metrics {
            println!(
                "  • {}: {:.2} ops/sec, {:?} avg latency",
                metric.benchmark_name, metric.ops_per_second, metric.avg_latency
            );
        }
        println!();
    }

    // Memory usage comparison
    println!("💾 Memory Usage Comparison:");
    for metric in all_metrics {
        println!(
            "  • {}: {:.1} MB peak",
            metric.benchmark_name,
            metric.memory_stats.peak_memory_bytes as f64 / 1024.0 / 1024.0
        );
    }
    println!();

    println!("✅ Benchmark completed successfully!");
}
