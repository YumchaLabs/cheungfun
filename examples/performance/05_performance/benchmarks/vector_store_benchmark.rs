//! Vector store performance benchmarks
//!
//! This benchmark compares the performance of different vector store implementations:
//! - InMemoryVectorStore (in-memory storage)
//! - QdrantVectorStore (persistent vector database)

use anyhow::Result;
use cheungfun_core::{
    traits::VectorStore,
    types::{ChunkInfo, Node, Query, SearchMode},
    DistanceMetric,
};
use cheungfun_integrations::vector_stores::{
    memory::InMemoryVectorStore,
    qdrant::{QdrantConfig, QdrantVectorStore},
};
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;
use tokio;
use tracing::{info, warn};
use uuid::Uuid;

use cheungfun_examples::benchmark_framework::{
    format_metrics, run_benchmark, BenchmarkConfig, PerformanceMetrics,
};

/// Test data generator for vector store benchmarks
struct VectorTestDataGenerator {
    rng: rand::rngs::ThreadRng,
    dimension: usize,
}

impl VectorTestDataGenerator {
    fn new(dimension: usize) -> Self {
        Self {
            rng: rand::thread_rng(),
            dimension,
        }
    }

    /// Generate a random vector
    fn generate_vector(&mut self) -> Vec<f32> {
        (0..self.dimension)
            .map(|_| self.rng.gen_range(-1.0..1.0))
            .collect()
    }

    /// Generate a normalized random vector
    fn generate_normalized_vector(&mut self) -> Vec<f32> {
        let mut vector = self.generate_vector();
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut vector {
                *x /= norm;
            }
        }
        vector
    }

    /// Generate a test node
    fn generate_node(&mut self, index: usize) -> Node {
        Node {
            id: Uuid::new_v4(),
            content: format!("Test document content number {}", index),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("index".to_string(), serde_json::Value::Number(index.into()));
                meta.insert(
                    "category".to_string(),
                    serde_json::Value::String(format!("category_{}", index % 5)),
                );
                meta
            },
            embedding: Some(self.generate_normalized_vector()),
            sparse_embedding: None,
            relationships: HashMap::new(),
            source_document_id: Uuid::new_v4(),
            chunk_info: ChunkInfo {
                start_offset: index * 100,
                end_offset: (index + 1) * 100,
                chunk_index: index,
            },
        }
    }

    /// Generate multiple test nodes
    fn generate_nodes(&mut self, count: usize) -> Vec<Node> {
        (0..count).map(|i| self.generate_node(i)).collect()
    }

    /// Generate a test query
    fn generate_query(&mut self, top_k: usize) -> Query {
        Query {
            text: "test query".to_string(),
            embedding: Some(self.generate_normalized_vector()),
            filters: HashMap::new(),
            top_k,
            similarity_threshold: Some(0.5),
            search_mode: SearchMode::Vector,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Cheungfun Vector Store Performance Benchmark");
    println!("==============================================");
    println!();

    let dimension = 384; // Common embedding dimension
    let mut generator = VectorTestDataGenerator::new(dimension);
    let mut all_metrics = Vec::new();

    // Test configurations
    let insert_config = BenchmarkConfig {
        name: "Vector Insert".to_string(),
        warmup_iterations: 5,
        measurement_iterations: 50,
        ..Default::default()
    };

    let batch_insert_config = BenchmarkConfig {
        name: "Batch Vector Insert".to_string(),
        warmup_iterations: 3,
        measurement_iterations: 20,
        ..Default::default()
    };

    let search_config = BenchmarkConfig {
        name: "Vector Search".to_string(),
        warmup_iterations: 10,
        measurement_iterations: 100,
        ..Default::default()
    };

    let large_search_config = BenchmarkConfig {
        name: "Large Dataset Search".to_string(),
        warmup_iterations: 5,
        measurement_iterations: 50,
        ..Default::default()
    };

    // Benchmark InMemoryVectorStore
    println!("🧠 Benchmarking InMemoryVectorStore");
    println!("----------------------------------");

    match benchmark_memory_store(
        &mut generator,
        &insert_config,
        &batch_insert_config,
        &search_config,
        &large_search_config,
    )
    .await
    {
        Ok(mut metrics) => {
            all_metrics.append(&mut metrics);
        }
        Err(e) => {
            warn!("InMemoryVectorStore benchmark failed: {}", e);
        }
    }
    println!();

    // Benchmark QdrantVectorStore
    println!("🗄️  Benchmarking QdrantVectorStore");
    println!("---------------------------------");

    match benchmark_qdrant_store(
        &mut generator,
        &insert_config,
        &batch_insert_config,
        &search_config,
    )
    .await
    {
        Ok(mut metrics) => {
            all_metrics.append(&mut metrics);
        }
        Err(e) => {
            warn!("QdrantVectorStore benchmark failed: {}", e);
        }
    }
    println!();

    // Generate comparison report
    generate_comparison_report(&all_metrics);

    Ok(())
}

async fn benchmark_memory_store(
    generator: &mut VectorTestDataGenerator,
    insert_config: &BenchmarkConfig,
    batch_insert_config: &BenchmarkConfig,
    search_config: &BenchmarkConfig,
    large_search_config: &BenchmarkConfig,
) -> Result<Vec<PerformanceMetrics>> {
    let mut metrics = Vec::new();

    // Initialize InMemoryVectorStore
    info!("Initializing InMemoryVectorStore...");
    let store = InMemoryVectorStore::new(generator.dimension, DistanceMetric::Cosine);
    info!("InMemoryVectorStore initialized successfully");

    // Single insert benchmark
    let test_node = generator.generate_node(0);
    let single_insert_metrics = run_benchmark(insert_config.clone(), || {
        let store = &store;
        let node = test_node.clone();
        async move { store.add(vec![node]).await.map_err(|e| anyhow::anyhow!(e)) }
    })
    .await?;

    println!("{}", format_metrics(&single_insert_metrics));
    metrics.push(single_insert_metrics);

    // Batch insert benchmark
    let batch_nodes = generator.generate_nodes(10);
    let batch_insert_metrics = run_benchmark(batch_insert_config.clone(), || {
        let store = &store;
        let nodes = batch_nodes.clone();
        async move { store.add(nodes).await.map_err(|e| anyhow::anyhow!(e)) }
    })
    .await?;

    println!("{}", format_metrics(&batch_insert_metrics));
    metrics.push(batch_insert_metrics);

    // Prepare data for search benchmarks
    let search_nodes = generator.generate_nodes(1000);
    store.add(search_nodes).await?;

    // Search benchmark
    let test_query = generator.generate_query(10);
    let search_metrics = run_benchmark(search_config.clone(), || {
        let store = &store;
        let query = test_query.clone();
        async move { store.search(&query).await.map_err(|e| anyhow::anyhow!(e)) }
    })
    .await?;

    println!("{}", format_metrics(&search_metrics));
    metrics.push(search_metrics);

    // Large dataset search benchmark
    let large_nodes = generator.generate_nodes(10000);
    store.add(large_nodes).await?;

    let large_search_metrics = run_benchmark(large_search_config.clone(), || {
        let store = &store;
        let query = test_query.clone();
        async move { store.search(&query).await.map_err(|e| anyhow::anyhow!(e)) }
    })
    .await?;

    println!("{}", format_metrics(&large_search_metrics));
    metrics.push(large_search_metrics);

    Ok(metrics)
}

async fn benchmark_qdrant_store(
    generator: &mut VectorTestDataGenerator,
    insert_config: &BenchmarkConfig,
    batch_insert_config: &BenchmarkConfig,
    search_config: &BenchmarkConfig,
) -> Result<Vec<PerformanceMetrics>> {
    let mut metrics = Vec::new();

    // Check if Qdrant is available
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());

    // Initialize QdrantVectorStore
    info!("Initializing QdrantVectorStore...");
    let config = QdrantConfig::new(&qdrant_url, "benchmark_collection", generator.dimension)
        .with_create_collection_if_missing(true);

    let store = match QdrantVectorStore::new(config).await {
        Ok(store) => store,
        Err(e) => {
            warn!(
                "Failed to connect to Qdrant: {}. Skipping Qdrant benchmarks.",
                e
            );
            return Ok(metrics);
        }
    };

    // Clear any existing data
    let _ = store.clear().await;

    info!("QdrantVectorStore initialized successfully");

    // Single insert benchmark
    let test_node = generator.generate_node(0);
    let single_insert_metrics = run_benchmark(insert_config.clone(), || {
        let store = &store;
        let node = test_node.clone();
        async move { store.add(vec![node]).await.map_err(|e| anyhow::anyhow!(e)) }
    })
    .await?;

    println!("{}", format_metrics(&single_insert_metrics));
    metrics.push(single_insert_metrics);

    // Batch insert benchmark
    let batch_nodes = generator.generate_nodes(10);
    let batch_insert_metrics = run_benchmark(batch_insert_config.clone(), || {
        let store = &store;
        let nodes = batch_nodes.clone();
        async move { store.add(nodes).await.map_err(|e| anyhow::anyhow!(e)) }
    })
    .await?;

    println!("{}", format_metrics(&batch_insert_metrics));
    metrics.push(batch_insert_metrics);

    // Prepare data for search benchmarks
    let search_nodes = generator.generate_nodes(1000);
    store.add(search_nodes).await?;

    // Search benchmark
    let test_query = generator.generate_query(10);
    let search_metrics = run_benchmark(search_config.clone(), || {
        let store = &store;
        let query = test_query.clone();
        async move { store.search(&query).await.map_err(|e| anyhow::anyhow!(e)) }
    })
    .await?;

    println!("{}", format_metrics(&search_metrics));
    metrics.push(search_metrics);

    Ok(metrics)
}

fn generate_comparison_report(all_metrics: &[PerformanceMetrics]) {
    if all_metrics.is_empty() {
        println!("⚠️  No metrics collected for comparison");
        return;
    }

    println!("📊 Vector Store Performance Comparison Report");
    println!("============================================");
    println!();

    // Group metrics by operation type
    let mut insert_metrics = Vec::new();
    let mut batch_insert_metrics = Vec::new();
    let mut search_metrics = Vec::new();
    let mut large_search_metrics = Vec::new();

    for metric in all_metrics {
        if metric.benchmark_name.contains("Large Dataset") {
            large_search_metrics.push(metric);
        } else if metric.benchmark_name.contains("Batch")
            && metric.benchmark_name.contains("Insert")
        {
            batch_insert_metrics.push(metric);
        } else if metric.benchmark_name.contains("Insert") {
            insert_metrics.push(metric);
        } else if metric.benchmark_name.contains("Search") {
            search_metrics.push(metric);
        }
    }

    // Compare insert performance
    if !insert_metrics.is_empty() {
        println!("📥 Single Insert Performance:");
        for metric in &insert_metrics {
            println!(
                "  • {}: {:.2} ops/sec, {:?} avg latency",
                metric.benchmark_name, metric.ops_per_second, metric.avg_latency
            );
        }
        println!();
    }

    // Compare batch insert performance
    if !batch_insert_metrics.is_empty() {
        println!("📦 Batch Insert Performance:");
        for metric in &batch_insert_metrics {
            println!(
                "  • {}: {:.2} ops/sec, {:?} avg latency",
                metric.benchmark_name, metric.ops_per_second, metric.avg_latency
            );
        }
        println!();
    }

    // Compare search performance
    if !search_metrics.is_empty() {
        println!("🔍 Search Performance:");
        for metric in &search_metrics {
            println!(
                "  • {}: {:.2} ops/sec, {:?} avg latency",
                metric.benchmark_name, metric.ops_per_second, metric.avg_latency
            );
        }
        println!();
    }

    // Compare large dataset search performance
    if !large_search_metrics.is_empty() {
        println!("🏋️  Large Dataset Search Performance:");
        for metric in &large_search_metrics {
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

    println!("✅ Vector store benchmark completed successfully!");
}
