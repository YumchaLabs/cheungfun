//! # Proposition Chunking - Advanced Text Segmentation for RAG
//!
//! This example demonstrates the Proposition Chunking technique for improving RAG systems.
//! Proposition chunking breaks down documents into atomic, factual, self-contained propositions
//! rather than using fixed-size chunks, enabling more precise and granular retrieval.
//!
//! ## Key Components
//!
//! 1. **Document Loading**: Load documents using the built-in loaders
//! 2. **Initial Chunking**: Split documents into manageable chunks for processing
//! 3. **Proposition Generation**: Use LLM to extract atomic propositions from chunks
//! 4. **Quality Assessment**: Evaluate propositions for accuracy and completeness
//! 5. **Vector Storage**: Store high-quality propositions in vector database
//! 6. **Retrieval Comparison**: Compare with traditional chunking approaches
//!
//! ## Usage
//!
//! ```bash
//! # Using FastEmbed (default, no API key required)
//! cargo run --bin proposition_chunking --features fastembed
//!
//! # Compare with traditional chunking
//! cargo run --bin proposition_chunking --features fastembed -- --compare-traditional
//!
//! # Interactive mode
//! cargo run --bin proposition_chunking --features fastembed -- --interactive
//! ```

use clap::Parser;

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{
    constants::*, get_climate_test_queries, print_query_results, setup_logging, ExampleError,
    ExampleResult, PerformanceMetrics, Timer,
};
use std::{path::PathBuf, sync::Arc};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, VectorStore},
    DistanceMetric,
};
use cheungfun_indexing::{
    loaders::DirectoryLoader,
    node_parser::{config::SentenceSplitterConfig, text::SentenceSplitter},
    pipeline::DefaultIndexingPipeline,
    transformers::MetadataExtractor,
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    engine::QueryEngine, generator::SiumaiGenerator, retriever::VectorRetriever,
};
use siumai::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "proposition_chunking")]
#[command(
    about = "Proposition Chunking RAG Example - Break documents into atomic, factual propositions"
)]
struct Args {
    /// Path to the document to process
    #[arg(long, default_value = "data/Understanding_Climate_Change.pdf")]
    document_path: PathBuf,

    /// Embedding provider (fastembed, openai)
    #[arg(long, default_value = "fastembed")]
    embedding_provider: String,

    /// Chunk size for initial document processing
    #[arg(long, default_value_t = DEFAULT_CHUNK_SIZE)]
    chunk_size: usize,

    /// Chunk overlap for initial processing
    #[arg(long, default_value_t = DEFAULT_CHUNK_OVERLAP)]
    chunk_overlap: usize,

    /// Number of documents to retrieve
    #[arg(long, default_value_t = DEFAULT_TOP_K)]
    top_k: usize,

    /// Compare with traditional chunking approach
    #[arg(long)]
    compare_traditional: bool,

    /// Enable interactive mode
    #[arg(long)]
    interactive: bool,

    /// Show detailed proposition information
    #[arg(long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    // Setup logging
    setup_logging();

    let args = Args::parse();

    println!("🚀 Starting Proposition Chunking RAG Example...");

    println!("🚀 Proposition Chunking RAG Example");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📄 Document: {}", args.document_path.display());
    println!("🔤 Embedding Provider: {}", args.embedding_provider);
    println!(
        "📏 Initial Chunk Size: {} (overlap: {})",
        args.chunk_size, args.chunk_overlap
    );
    println!("🔍 Top-K: {}", args.top_k);
    println!("🔍 Compare Traditional: {}", args.compare_traditional);
    println!("📊 Verbose: {}", args.verbose);
    println!();

    if args.compare_traditional {
        println!("📊 This run will compare proposition-based chunking with traditional chunking");
        println!();
    }

    // Initialize performance metrics
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

    // Step 3: Build indexing pipeline
    let timer = Timer::new("Document indexing");

    // Ensure we have the correct path to the data directory
    let data_dir = if args.document_path.is_absolute() {
        args.document_path
            .parent()
            .unwrap_or(&PathBuf::from("."))
            .to_path_buf()
    } else {
        // For relative paths, resolve from current working directory
        std::env::current_dir()?.join(args.document_path.parent().unwrap_or(&PathBuf::from(".")))
    };

    println!("📂 Loading from directory: {}", data_dir.display());
    let loader = Arc::new(DirectoryLoader::new(&data_dir)?);

    let splitter = Arc::new(SentenceSplitter::from_defaults(
        args.chunk_size,
        args.chunk_overlap,
    )?);

    let metadata_extractor = Arc::new(MetadataExtractor::new());

    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(loader)
        .with_document_processor(splitter) // Documents -> Nodes
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

    println!("📊 Indexing completed:");
    println!("  📚 Documents: {}", indexing_stats.documents_processed);
    println!("  🔗 Nodes: {}", indexing_stats.nodes_created);
    println!("  ⏱️  Time: {:.2?}", indexing_time);

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

    let queries = get_climate_test_queries();

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
    println!("🎯 Interactive mode - Enter your queries (type 'quit' to exit):");
    println!();

    loop {
        print!("Query: ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let query = input.trim();

        if query.is_empty() {
            continue;
        }

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
    }

    Ok(())
}
