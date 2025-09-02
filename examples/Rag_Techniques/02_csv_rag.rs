//! CSV RAG Example
//!
//! This example demonstrates how to build a RAG system for structured CSV data.
//! It shows how to:
//! - Load and process CSV files
//! - Create meaningful text representations of structured data
//! - Query structured data using natural language
//! - Handle different data types and formats
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin csv_rag --features fastembed
//! cargo run --bin csv_rag --features fastembed -- --csv-file data/customers-100.csv --interactive
//! ```

use clap::Parser;

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{
    get_customer_test_queries, print_query_results, ExampleError, ExampleResult,
    PerformanceMetrics, Timer,
};
use std::{path::PathBuf, sync::Arc};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, VectorStore},
    DistanceMetric,
};
use cheungfun_indexing::{
    loaders::DirectoryLoader,
    node_parser::{
        config::{ChunkingStrategy, SentenceSplitterConfig},
        text::SentenceSplitter,
    },
    pipeline::DefaultIndexingPipeline,
    transformers::MetadataExtractor,
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    engine::QueryEngine, generator::SiumaiGenerator, retriever::VectorRetriever,
};
use siumai::prelude::*;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser)]
#[command(name = "csv_rag")]
#[command(about = "CSV RAG Example - Process structured CSV data with RAG")]
struct Args {
    /// Path to the CSV file to process
    #[arg(long, default_value = "data/customers-100.csv")]
    csv_file: PathBuf,

    /// Embedding provider to use
    #[arg(long, default_value = "fastembed")]
    embedding_provider: String,

    /// Chunk size for text splitting
    #[arg(long, default_value_t = 800)]
    chunk_size: usize,

    /// Chunk overlap for text splitting
    #[arg(long, default_value_t = 100)]
    chunk_overlap: usize,

    /// Number of top results to retrieve
    #[arg(long, default_value_t = 5)]
    top_k: usize,

    /// Run in interactive mode
    #[arg(long)]
    interactive: bool,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    // Setup logging
    setup_logging();

    let args = Args::parse();

    println!("🚀 Starting CSV RAG Example...");

    // Print configuration
    print_config(&args);

    let mut metrics = PerformanceMetrics::new();

    // Step 1: Create embedder
    let embedder = create_embedder(&args.embedding_provider).await?;
    println!("✅ Embedder initialized: {}", args.embedding_provider);

    // Step 2: Create vector store
    let vector_store = Arc::new(InMemoryVectorStore::new(
        DEFAULT_EMBEDDING_DIM,
        DistanceMetric::Cosine,
    ));
    println!("✅ Vector store initialized");

    // Step 3: Build indexing pipeline for CSV data
    let timer = Timer::new("CSV data indexing");

    // Get the directory containing the CSV file
    let default_path = PathBuf::from(".");
    let data_dir = args.csv_file.parent().unwrap_or(&default_path);
    println!("📂 Loading from directory: {}", data_dir.display());

    let loader = Arc::new(DirectoryLoader::new(data_dir)?);

    // Create specialized splitter for CSV data
    let csv_splitter_config = SentenceSplitterConfig::default();

    let csv_splitter = Arc::new(SentenceSplitter::new(csv_splitter_config)?);
    let metadata_extractor = Arc::new(MetadataExtractor::new());

    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(loader)
        .with_document_processor(csv_splitter) // Documents -> Nodes
        .with_node_processor(metadata_extractor) // Nodes -> Nodes
        .with_embedder(embedder.clone())
        .with_vector_store(vector_store.clone())
        .build()?;

    // Run indexing pipeline with progress reporting
    let (_nodes, indexing_stats) = pipeline
        .run_with_progress(
            None, // documents (will use loader)
            None, // nodes
            true, // store_doc_text
            None, // num_workers (use default)
            true, // in_place
            Box::new(|progress| {
                if let Some(percentage) = progress.percentage() {
                    println!(
                        "📊 {}: {:.1}% ({}/{})",
                        progress.stage,
                        percentage,
                        progress.processed,
                        progress.total.unwrap_or(0)
                    );
                } else {
                    println!(
                        "📊 {}: {} items processed",
                        progress.stage, progress.processed
                    );
                }

                if let Some(current_item) = &progress.current_item {
                    println!("   └─ {}", current_item);
                }
            }),
        )
        .await?;

    let indexing_time = timer.finish();

    metrics.record_indexing_time(indexing_time);
    metrics.total_documents = indexing_stats.documents_processed;
    metrics.total_nodes = indexing_stats.nodes_created;

    println!(
        "✅ Completed: CSV data indexing in {:.2}s",
        indexing_time.as_secs_f64()
    );
    println!("📊 Indexing completed:");
    println!("  📚 Documents: {}", indexing_stats.documents_processed);
    println!("  🔗 Nodes: {}", indexing_stats.nodes_created);
    println!("  ⏱️  Time: {:.2}s", indexing_time.as_secs_f64());

    // Step 4: Create query engine
    let retriever = Arc::new(VectorRetriever::new(vector_store, embedder));

    // Create LLM client - try OpenAI first, fallback to Ollama
    let llm_client = create_llm_client().await?;
    let generator = Arc::new(SiumaiGenerator::new(llm_client));

    let query_engine = QueryEngine::new(retriever, generator);

    println!("✅ Query engine initialized");
    println!();

    // Step 5: Run queries
    if args.interactive {
        run_interactive_mode(&query_engine, &mut metrics).await?;
    } else {
        run_demo_queries(&query_engine, &mut metrics).await?;
    }

    // Print final metrics
    metrics.print_summary();

    Ok(())
}

fn setup_logging() {
    tracing_subscriber::fmt()
        .with_env_filter("info,cheungfun=info")
        .init();
}

fn print_config(args: &Args) {
    println!("🗂️  CSV RAG Example");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📄 CSV File: {}", args.csv_file.display());
    println!("🔤 Embedding Provider: {}", args.embedding_provider);
    println!(
        "📏 Chunk Size: {} (overlap: {})",
        args.chunk_size, args.chunk_overlap
    );
    println!("🔍 Top-K: {}", args.top_k);
    println!();
}

async fn create_embedder(provider: &str) -> ExampleResult<Arc<dyn Embedder>> {
    match provider {
        "fastembed" => {
            println!("🔤 Using FastEmbed for embeddings (local)");
            let embedder = FastEmbedder::new()
                .await
                .map_err(|e| ExampleError::Config(format!("FastEmbed error: {}", e)))?;
            Ok(Arc::new(embedder))
        }
        "openai" => {
            // Check for API key
            if let Ok(_api_key) = std::env::var("OPENAI_API_KEY") {
                // Note: This would require implementing OpenAI embedder
                // For now, fall back to FastEmbed
                println!("⚠️  OpenAI embedder not yet implemented, using FastEmbed");
                let embedder = FastEmbedder::new()
                    .await
                    .map_err(|e| ExampleError::Config(format!("FastEmbed error: {}", e)))?;
                Ok(Arc::new(embedder))
            } else {
                println!("🔤 No OpenAI API key found, using FastEmbed for embeddings (local)");
                let embedder = FastEmbedder::new()
                    .await
                    .map_err(|e| ExampleError::Config(format!("FastEmbed error: {}", e)))?;
                Ok(Arc::new(embedder))
            }
        }
        _ => Err(ExampleError::Config(format!(
            "Unsupported embedding provider: {}",
            provider
        ))),
    }
}

async fn create_llm_client() -> ExampleResult<Siumai> {
    // Try OpenAI first
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        if !api_key.is_empty() && api_key != "test" && api_key.starts_with("sk-") {
            println!("🤖 Using OpenAI for LLM generation (cloud)");
            return Siumai::builder()
                .openai()
                .api_key(&api_key)
                .model("gpt-4o-mini")
                .temperature(0.0)
                .max_tokens(4000)
                .build()
                .await
                .map_err(|e| ExampleError::Config(format!("Failed to initialize OpenAI: {}", e)));
        }
    }

    // Fallback to Ollama
    println!("🤖 No valid OpenAI API key found, using Ollama for LLM generation (local)");
    println!("💡 Make sure Ollama is running with: ollama serve");
    println!("💡 And pull a model with: ollama pull llama3.2");

    Siumai::builder()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .temperature(0.0)
        .build()
        .await
        .map_err(|e| ExampleError::Config(format!("Failed to initialize Ollama: {}. Make sure Ollama is running with 'ollama serve' and you have pulled a model with 'ollama pull llama3.2'", e)))
}

async fn run_demo_queries(
    query_engine: &QueryEngine,
    metrics: &mut PerformanceMetrics,
) -> ExampleResult<()> {
    println!("🔍 Running demo queries...");
    println!();

    let queries = get_customer_test_queries();

    for query in queries {
        let timer = Timer::new(&format!("Query: {}", query));

        let response = query_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;

        let query_time = timer.finish();
        metrics.record_query(query_time);

        print_query_results(query, &response);
    }

    Ok(())
}

async fn run_interactive_mode(
    query_engine: &QueryEngine,
    metrics: &mut PerformanceMetrics,
) -> ExampleResult<()> {
    println!("🎯 Interactive CSV Query Mode");
    println!("Type your questions about the customer data, or 'quit' to exit.");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    loop {
        print!("❓ Your question: ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let query = input.trim();

        if query.to_lowercase() == "quit" {
            break;
        }

        let timer = Timer::new("Query processing");

        match query_engine.query(query).await {
            Ok(response) => {
                let query_time = timer.finish();
                metrics.record_query(query_time);
                print_query_results(query, &response);
            }
            Err(e) => {
                println!("❌ Error processing query: {}", e);
            }
        }

        println!();
    }

    println!("👋 Goodbye!");
    Ok(())
}
