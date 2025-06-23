//! End-to-end RAG performance benchmarks
//!
//! This benchmark tests the complete RAG pipeline performance:
//! - Document indexing (loading, chunking, embedding, storing)
//! - Query processing (embedding, retrieval, response generation)

use anyhow::Result;
use cheungfun_core::{
    traits::{Embedder, Loader, ResponseGenerator, Retriever, Transformer, VectorStore},
    types::{Document, GenerationOptions, Query, ResponseFormat, SearchMode},
};
use cheungfun_indexing::{
    loaders::text::TextLoader,
    pipeline::{IndexingPipeline, IndexingPipelineBuilder},
    transformers::text_splitter::TextSplitter,
};
use cheungfun_integrations::vector_stores::memory::InMemoryVectorStore;
use cheungfun_query::{
    generators::siumai::SiumaiResponseGenerator,
    pipeline::{QueryPipeline, QueryPipelineBuilder},
    retrievers::vector::VectorRetriever,
};
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tempfile::TempDir;
use tokio;
use tracing::{info, warn};
use uuid::Uuid;

use cheungfun_examples::benchmark_framework::{
    BenchmarkConfig, PerformanceMetrics, format_metrics, run_benchmark,
};

#[cfg(feature = "fastembed")]
use cheungfun_integrations::embedders::fastembed::{FastEmbedder, FastEmbedderConfig};

/// Test data generator for end-to-end benchmarks
struct EndToEndTestDataGenerator {
    rng: rand::rngs::ThreadRng,
}

impl EndToEndTestDataGenerator {
    fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }

    /// Generate test documents
    fn generate_documents(&mut self, count: usize) -> Vec<String> {
        let topics = [
            "artificial intelligence",
            "machine learning",
            "natural language processing",
            "computer vision",
            "robotics",
            "data science",
            "deep learning",
            "neural networks",
            "software engineering",
            "cloud computing",
            "cybersecurity",
            "blockchain",
        ];

        let templates = [
            "This document discusses the fundamentals of {}. It covers the basic concepts, applications, and recent developments in the field.",
            "An introduction to {} for beginners. This comprehensive guide explains the key principles and practical applications.",
            "Advanced techniques in {} are explored in this document. We examine cutting-edge research and industry best practices.",
            "The future of {} looks promising with new innovations and breakthrough technologies emerging rapidly.",
            "Understanding {} requires knowledge of mathematical foundations, algorithmic approaches, and real-world implementations.",
        ];

        (0..count)
            .map(|i| {
                let topic = topics[self.rng.gen_range(0..topics.len())];
                let template = templates[self.rng.gen_range(0..templates.len())];
                let content = template.replace("{}", topic);

                format!(
                    "Document {}: {}\n\nThis document contains detailed information about {}. \
                     It includes examples, case studies, and practical applications. \
                     The content is designed to be comprehensive and informative for readers \
                     interested in learning more about this important topic.\n\n\
                     Additional sections cover related concepts, implementation details, \
                     and future research directions. The document serves as a valuable \
                     resource for both beginners and advanced practitioners in the field.",
                    i + 1,
                    content,
                    topic
                )
            })
            .collect()
    }

    /// Generate test queries
    fn generate_queries(&mut self, count: usize) -> Vec<String> {
        let query_templates = [
            "What is {}?",
            "How does {} work?",
            "What are the applications of {}?",
            "Explain the benefits of {}",
            "What are the challenges in {}?",
            "How to implement {}?",
            "What is the future of {}?",
            "Compare {} with other approaches",
        ];

        let topics = [
            "machine learning",
            "artificial intelligence",
            "deep learning",
            "neural networks",
            "computer vision",
            "natural language processing",
            "data science",
            "robotics",
            "cloud computing",
            "cybersecurity",
        ];

        (0..count)
            .map(|_| {
                let template = query_templates[self.rng.gen_range(0..query_templates.len())];
                let topic = topics[self.rng.gen_range(0..topics.len())];
                template.replace("{}", topic)
            })
            .collect()
    }

    /// Create temporary files with test documents
    async fn create_test_files(
        &mut self,
        temp_dir: &TempDir,
        count: usize,
    ) -> Result<Vec<std::path::PathBuf>> {
        let documents = self.generate_documents(count);
        let mut file_paths = Vec::new();

        for (i, content) in documents.iter().enumerate() {
            let file_path = temp_dir.path().join(format!("doc_{}.txt", i));
            tokio::fs::write(&file_path, content).await?;
            file_paths.push(file_path);
        }

        Ok(file_paths)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Cheungfun End-to-End RAG Performance Benchmark");
    println!("================================================");
    println!();

    let mut generator = EndToEndTestDataGenerator::new();
    let mut all_metrics = Vec::new();

    // Test configurations
    let indexing_config = BenchmarkConfig {
        name: "Document Indexing".to_string(),
        warmup_iterations: 2,
        measurement_iterations: 10,
        ..Default::default()
    };

    let query_config = BenchmarkConfig {
        name: "Query Processing".to_string(),
        warmup_iterations: 5,
        measurement_iterations: 50,
        ..Default::default()
    };

    let large_indexing_config = BenchmarkConfig {
        name: "Large Document Indexing".to_string(),
        warmup_iterations: 1,
        measurement_iterations: 5,
        ..Default::default()
    };

    // Benchmark with FastEmbedder (if available)
    #[cfg(feature = "fastembed")]
    {
        println!("🔥 Benchmarking End-to-End RAG with FastEmbedder");
        println!("------------------------------------------------");

        match benchmark_rag_pipeline(
            &mut generator,
            &indexing_config,
            &query_config,
            &large_indexing_config,
        )
        .await
        {
            Ok(mut metrics) => {
                all_metrics.append(&mut metrics);
            }
            Err(e) => {
                warn!("End-to-end RAG benchmark failed: {}", e);
            }
        }
        println!();
    }

    // Generate comparison report
    generate_comparison_report(&all_metrics);

    Ok(())
}

#[cfg(feature = "fastembed")]
async fn benchmark_rag_pipeline(
    generator: &mut EndToEndTestDataGenerator,
    indexing_config: &BenchmarkConfig,
    query_config: &BenchmarkConfig,
    large_indexing_config: &BenchmarkConfig,
) -> Result<Vec<PerformanceMetrics>> {
    let mut metrics = Vec::new();

    // Create temporary directory for test files
    let temp_dir = TempDir::new()?;

    // Initialize components
    info!("Initializing RAG components...");

    // Embedder
    let embedder_config = FastEmbedderConfig::default();
    let embedder = Arc::new(FastEmbedder::new(embedder_config).await?);

    // Vector store
    let vector_store = Arc::new(InMemoryVectorStore::new());

    // Text splitter
    let text_splitter = Arc::new(TextSplitter::new(500, 50));

    info!("RAG components initialized successfully");

    // Small document indexing benchmark
    let small_files = generator.create_test_files(&temp_dir, 10).await?;
    let small_indexing_metrics = run_benchmark(indexing_config.clone(), || {
        let embedder = embedder.clone();
        let vector_store = vector_store.clone();
        let text_splitter = text_splitter.clone();
        let files = small_files.clone();

        async move {
            // Clear previous data
            let _ = vector_store.clear().await;

            // Create indexing pipeline
            let pipeline = IndexingPipelineBuilder::new()
                .with_loader(TextLoader::new(files))
                .with_transformer_arc(text_splitter)
                .with_embedder_arc(embedder)
                .with_vector_store_arc(vector_store)
                .build()?;

            // Run indexing
            pipeline.run().await.map_err(|e| anyhow::anyhow!(e))
        }
    })
    .await?;

    println!("{}", format_metrics(&small_indexing_metrics));
    metrics.push(small_indexing_metrics);

    // Large document indexing benchmark
    let large_files = generator.create_test_files(&temp_dir, 100).await?;
    let large_indexing_metrics = run_benchmark(large_indexing_config.clone(), || {
        let embedder = embedder.clone();
        let vector_store = vector_store.clone();
        let text_splitter = text_splitter.clone();
        let files = large_files.clone();

        async move {
            // Clear previous data
            let _ = vector_store.clear().await;

            // Create indexing pipeline
            let pipeline = IndexingPipelineBuilder::new()
                .with_loader(TextLoader::new(files))
                .with_transformer_arc(text_splitter)
                .with_embedder_arc(embedder)
                .with_vector_store_arc(vector_store)
                .build()?;

            // Run indexing
            pipeline.run().await.map_err(|e| anyhow::anyhow!(e))
        }
    })
    .await?;

    println!("{}", format_metrics(&large_indexing_metrics));
    metrics.push(large_indexing_metrics);

    // Prepare for query benchmarks - index some documents
    let query_files = generator.create_test_files(&temp_dir, 50).await?;
    let pipeline = IndexingPipelineBuilder::new()
        .with_loader(TextLoader::new(query_files))
        .with_transformer_arc(text_splitter.clone())
        .with_embedder_arc(embedder.clone())
        .with_vector_store_arc(vector_store.clone())
        .build()?;

    pipeline.run().await?;
    info!("Indexed documents for query benchmarks");

    // Query processing benchmark
    let test_queries = generator.generate_queries(10);
    let query_metrics = run_benchmark(query_config.clone(), || {
        let embedder = embedder.clone();
        let vector_store = vector_store.clone();
        let queries = test_queries.clone();

        async move {
            // Create retriever
            let retriever = Arc::new(VectorRetriever::new(vector_store, embedder));

            // Create mock response generator (since we don't have LLM API key)
            let response_generator = Arc::new(MockResponseGenerator::new());

            // Create query pipeline
            let pipeline = QueryPipelineBuilder::new()
                .with_retriever_arc(retriever)
                .with_response_generator_arc(response_generator)
                .build()?;

            // Run a random query
            let query = &queries[rand::thread_rng().gen_range(0..queries.len())];
            let options = Default::default();

            pipeline
                .query(query, &options)
                .await
                .map_err(|e| anyhow::anyhow!(e))
        }
    })
    .await?;

    println!("{}", format_metrics(&query_metrics));
    metrics.push(query_metrics);

    Ok(metrics)
}

/// Mock response generator for benchmarking
struct MockResponseGenerator;

impl MockResponseGenerator {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl ResponseGenerator for MockResponseGenerator {
    async fn generate_response(
        &self,
        query: &str,
        context_nodes: Vec<cheungfun_core::types::ScoredNode>,
        _options: &GenerationOptions,
    ) -> Result<cheungfun_core::types::GeneratedResponse, cheungfun_core::error::CheungfunError>
    {
        // Simulate response generation time
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let source_nodes = context_nodes.iter().map(|n| n.node.id).collect();

        Ok(cheungfun_core::types::GeneratedResponse {
            content: format!("Mock response for query: {}", query),
            source_nodes,
            metadata: HashMap::new(),
            usage: None,
        })
    }

    async fn generate_response_stream(
        &self,
        query: &str,
        context_nodes: Vec<cheungfun_core::types::ScoredNode>,
        options: &GenerationOptions,
    ) -> Result<
        impl futures::Stream<Item = Result<String, cheungfun_core::error::CheungfunError>> + Send,
        cheungfun_core::error::CheungfunError,
    > {
        // For simplicity, just return the non-streaming response as a single item stream
        let response = self
            .generate_response(query, context_nodes, options)
            .await?;
        Ok(futures::stream::once(async move { Ok(response.content) }))
    }

    fn name(&self) -> &'static str {
        "MockResponseGenerator"
    }
}

fn generate_comparison_report(all_metrics: &[PerformanceMetrics]) {
    if all_metrics.is_empty() {
        println!("⚠️  No metrics collected for comparison");
        return;
    }

    println!("📊 End-to-End RAG Performance Report");
    println!("===================================");
    println!();

    // Group metrics by operation type
    let mut indexing_metrics = Vec::new();
    let mut query_metrics = Vec::new();

    for metric in all_metrics {
        if metric.benchmark_name.contains("Indexing") {
            indexing_metrics.push(metric);
        } else if metric.benchmark_name.contains("Query") {
            query_metrics.push(metric);
        }
    }

    // Indexing performance
    if !indexing_metrics.is_empty() {
        println!("📥 Document Indexing Performance:");
        for metric in &indexing_metrics {
            let docs_per_sec = metric.ops_per_second;
            println!(
                "  • {}: {:.2} docs/sec, {:?} avg time per doc",
                metric.benchmark_name, docs_per_sec, metric.avg_latency
            );
        }
        println!();
    }

    // Query performance
    if !query_metrics.is_empty() {
        println!("🔍 Query Processing Performance:");
        for metric in &query_metrics {
            println!(
                "  • {}: {:.2} queries/sec, {:?} avg response time",
                metric.benchmark_name, metric.ops_per_second, metric.avg_latency
            );
        }
        println!();
    }

    // Overall system performance
    println!("🏆 Overall System Performance:");
    for metric in all_metrics {
        println!(
            "  • {}: {:.1} MB peak memory, {:.1}% avg CPU",
            metric.benchmark_name,
            metric.memory_stats.peak_memory_bytes as f64 / 1024.0 / 1024.0,
            metric.cpu_stats.avg_cpu_percent
        );
    }
    println!();

    println!("✅ End-to-end RAG benchmark completed successfully!");
}
