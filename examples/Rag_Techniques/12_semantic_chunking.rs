/*!
# Semantic Chunking RAG Example

This example demonstrates semantic chunking, which divides documents based on
semantic coherence rather than fixed sizes, following the RAG_Techniques implementation.

Based on: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/semantic_chunking.ipynb

## Key Features

- **Semantic Boundary Detection**: Uses embeddings to identify topic boundaries
- **Adaptive Chunk Sizes**: Creates chunks of varying sizes based on content coherence
- **Performance Comparison**: Compares semantic vs traditional fixed-size chunking
- **Multiple Threshold Strategies**: Supports percentile, standard deviation, and gradient thresholds

## How It Works

1. **Split into Sentences**: Break document into individual sentences
2. **Calculate Embeddings**: Generate embeddings for sentence groups (buffer_size)
3. **Compute Similarities**: Calculate cosine similarity between adjacent groups
4. **Identify Breakpoints**: Find semantic boundaries using threshold strategies
5. **Form Chunks**: Group sentences between breakpoints into coherent chunks
6. **Compare Performance**: Evaluate against traditional chunking methods

## Usage

```bash
# Basic semantic chunking
cargo run --bin semantic_chunking --features fastembed

# Compare with traditional chunking
cargo run --bin semantic_chunking --features fastembed -- --compare-traditional

# Different threshold strategies
cargo run --bin semantic_chunking --features fastembed -- --threshold-type percentile --threshold-value 90
cargo run --bin semantic_chunking --features fastembed -- --threshold-type standard_deviation --threshold-value 1.5

# Interactive mode
cargo run --bin semantic_chunking --features fastembed -- --interactive
```
*/

use clap::Parser;
use std::{path::PathBuf, sync::Arc};

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{get_climate_test_queries, setup_logging, ExampleError, ExampleResult, Timer};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, VectorStore},
    DistanceMetric,
};
use cheungfun_indexing::{
    loaders::DirectoryLoader,
    node_parser::{config::SemanticSplitterConfig, text::SemanticSplitter, text::SentenceSplitter},
    pipeline::DefaultIndexingPipeline,
    transformers::MetadataExtractor,
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    engine::QueryEngine, generator::SiumaiGenerator, retriever::VectorRetriever,
};
use siumai::prelude::*;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug)]
#[command(
    name = "semantic_chunking",
    about = "Semantic Chunking RAG Example - Dividing documents based on semantic coherence"
)]
struct Args {
    /// Path to the document to process
    #[arg(long, default_value = "data/Understanding_Climate_Change.pdf")]
    document_path: PathBuf,

    /// Buffer size for semantic similarity calculation
    #[arg(long, default_value = "1")]
    buffer_size: usize,

    /// Threshold type for semantic boundary detection
    #[arg(long, default_value = "percentile")]
    threshold_type: String, // percentile, standard_deviation, gradient

    /// Threshold value
    #[arg(long, default_value = "95.0")]
    threshold_value: f32,

    /// Number of documents to retrieve
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Compare with traditional fixed-size chunking
    #[arg(long)]
    compare_traditional: bool,

    /// Enable interactive mode
    #[arg(long)]
    interactive: bool,

    /// Show detailed chunk information
    #[arg(long)]
    verbose: bool,

    /// Traditional chunk size for comparison
    #[arg(long, default_value = "800")]
    traditional_chunk_size: usize,

    /// Traditional chunk overlap for comparison
    #[arg(long, default_value = "100")]
    traditional_chunk_overlap: usize,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    // Setup logging
    setup_logging();

    let args = Args::parse();

    println!("🧠 Starting Semantic Chunking Example...");
    println!("📖 This example demonstrates semantic-based document chunking");
    println!("🎯 Based on the technique from RAG_Techniques repository\n");

    // Create embedder
    let embedder = create_embedder().await?;
    println!("✅ Embedder initialized");

    if args.compare_traditional {
        // Compare semantic vs traditional chunking
        compare_chunking_methods(&args, embedder).await?;
    } else {
        // Run semantic chunking only
        run_semantic_chunking(&args, embedder).await?;
    }

    Ok(())
}

/// Create embedder with fallback strategy
async fn create_embedder() -> ExampleResult<Arc<dyn Embedder>> {
    println!("🔧 Initializing embedder...");

    match FastEmbedder::new().await {
        Ok(embedder) => {
            println!("✅ Using FastEmbed (local)");
            Ok(Arc::new(embedder))
        }
        Err(e) => {
            println!("⚠️ FastEmbed failed: {}", e);
            Err(ExampleError::Config(format!(
                "Failed to create embedder: {}",
                e
            )))
        }
    }
}

/// Create Siumai client for LLM operations
async fn create_siumai_client() -> ExampleResult<Siumai> {
    // Try different providers based on available environment variables
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        println!("🔑 Using OpenAI with API key");
        Siumai::builder()
            .openai()
            .api_key(&api_key)
            .model("gpt-3.5-turbo")
            .temperature(0.7)
            .max_tokens(1000)
            .build()
            .await
            .map_err(|e| ExampleError::Config(format!("Failed to create OpenAI client: {}", e)))
    } else if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        println!("🔑 Using Anthropic with API key");
        Siumai::builder()
            .anthropic()
            .api_key(&api_key)
            .model("claude-3-haiku-20240307")
            .temperature(0.7)
            .max_tokens(1000)
            .build()
            .await
            .map_err(|e| ExampleError::Config(format!("Failed to create Anthropic client: {}", e)))
    } else {
        println!("⚠️ No API keys found, using demo mode");
        // For demo purposes, create a minimal client (this might not work without real API key)
        Siumai::builder()
            .openai()
            .api_key("demo-key")
            .model("gpt-3.5-turbo")
            .build()
            .await
            .map_err(|e| ExampleError::Config(format!("Failed to create demo client: {}", e)))
    }
}

/// Run semantic chunking demonstration
async fn run_semantic_chunking(args: &Args, embedder: Arc<dyn Embedder>) -> ExampleResult<()> {
    println!("🧠 Running Semantic Chunking...");

    let timer = Timer::new("Semantic chunking setup");

    // Create semantic splitter
    let semantic_splitter = create_semantic_splitter(args, embedder.clone())?;

    // Build indexing pipeline with semantic chunking
    let (vector_store, query_engine) =
        build_semantic_pipeline(args, semantic_splitter, embedder).await?;

    timer.finish();

    if args.interactive {
        run_interactive_mode(&query_engine).await?;
    } else {
        run_test_queries(&query_engine, args.verbose).await?;
    }

    Ok(())
}

/// Create semantic splitter with specified configuration
fn create_semantic_splitter(
    args: &Args,
    embedder: Arc<dyn Embedder>,
) -> ExampleResult<Arc<SemanticSplitter>> {
    println!("🔧 Creating semantic splitter...");
    println!("   📊 Buffer size: {}", args.buffer_size);
    println!("   🎯 Threshold type: {}", args.threshold_type);
    println!("   📈 Threshold value: {}", args.threshold_value);

    let splitter = SemanticSplitter::new(embedder)
        .with_buffer_size(args.buffer_size)
        .with_breakpoint_percentile_threshold(args.threshold_value);

    Ok(Arc::new(splitter))
}

/// Build indexing pipeline with semantic chunking
async fn build_semantic_pipeline(
    args: &Args,
    semantic_splitter: Arc<SemanticSplitter>,
    embedder: Arc<dyn Embedder>,
) -> ExampleResult<(Arc<dyn VectorStore>, QueryEngine)> {
    // Ensure we have the correct path to the data directory
    let data_dir = if args.document_path.is_absolute() {
        args.document_path
            .parent()
            .unwrap_or(&PathBuf::from("."))
            .to_path_buf()
    } else {
        std::env::current_dir()?.join(args.document_path.parent().unwrap_or(&PathBuf::from(".")))
    };

    println!("📂 Loading from directory: {}", data_dir.display());
    let loader = Arc::new(DirectoryLoader::new(&data_dir)?);
    let metadata_extractor = Arc::new(MetadataExtractor::new());

    // Create vector store
    let vector_store = Arc::new(InMemoryVectorStore::new(
        DEFAULT_EMBEDDING_DIM,
        DistanceMetric::Cosine,
    ));

    // Build pipeline with semantic splitter
    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(loader)
        .with_transformer(semantic_splitter)
        .with_transformer(metadata_extractor)
        .with_embedder(embedder.clone())
        .with_vector_store(vector_store.clone())
        .build()?;

    // Run indexing
    let indexing_timer = Timer::new("Semantic indexing");
    let index_result = pipeline
        .run()
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;
    let indexing_time = indexing_timer.finish();

    println!(
        "✅ Semantic indexing completed in {:.2}s",
        indexing_time.as_secs_f64()
    );
    println!("📊 Indexed {} nodes", index_result.nodes_created);

    if args.verbose {
        println!(
            "📊 Indexing statistics: {} documents processed, {} nodes created",
            index_result.documents_processed, index_result.nodes_created
        );
    }

    // Create Siumai client
    let siumai_client = create_siumai_client().await?;

    // Create query engine
    let retriever = Arc::new(VectorRetriever::new(vector_store.clone(), embedder));
    let generator = Arc::new(SiumaiGenerator::new(siumai_client));
    let query_engine = QueryEngine::new(retriever, generator);

    Ok((vector_store, query_engine))
}

/// Compare semantic chunking with traditional fixed-size chunking
async fn compare_chunking_methods(args: &Args, embedder: Arc<dyn Embedder>) -> ExampleResult<()> {
    println!("⚖️ Comparing Semantic vs Traditional Chunking...");
    println!("═══════════════════════════════════════════════════════════════\n");

    // 1. Build semantic chunking system
    println!("🧠 1. Building Semantic Chunking System...");
    let semantic_timer = Timer::new("Semantic chunking");
    let semantic_splitter = create_semantic_splitter(args, embedder.clone())?;
    let (semantic_store, semantic_engine) =
        build_semantic_pipeline(args, semantic_splitter, embedder.clone()).await?;
    let semantic_time = semantic_timer.finish();

    // 2. Build traditional chunking system
    println!("\n📏 2. Building Traditional Chunking System...");
    let traditional_timer = Timer::new("Traditional chunking");
    let traditional_splitter = Arc::new(SentenceSplitter::from_defaults(
        args.traditional_chunk_size,
        args.traditional_chunk_overlap,
    )?);
    let (traditional_store, traditional_engine) =
        build_traditional_pipeline(args, traditional_splitter, embedder).await?;
    let traditional_time = traditional_timer.finish();

    // 3. Compare performance on test queries
    println!("\n🎯 3. Performance Comparison...");
    compare_retrieval_performance(&semantic_engine, &traditional_engine, args.verbose).await?;

    // 4. Show indexing time comparison
    println!("\n⏱️ 4. Indexing Time Comparison:");
    println!(
        "   🧠 Semantic Chunking: {:.2}s",
        semantic_time.as_secs_f64()
    );
    println!(
        "   📏 Traditional Chunking: {:.2}s",
        traditional_time.as_secs_f64()
    );
    println!(
        "   📊 Semantic Overhead: {:.1}%",
        ((semantic_time.as_secs_f64() - traditional_time.as_secs_f64())
            / traditional_time.as_secs_f64())
            * 100.0
    );

    Ok(())
}

/// Build traditional chunking pipeline for comparison
async fn build_traditional_pipeline(
    args: &Args,
    traditional_splitter: Arc<SentenceSplitter>,
    embedder: Arc<dyn Embedder>,
) -> ExampleResult<(Arc<dyn VectorStore>, QueryEngine)> {
    let data_dir = if args.document_path.is_absolute() {
        args.document_path
            .parent()
            .unwrap_or(&PathBuf::from("."))
            .to_path_buf()
    } else {
        std::env::current_dir()?.join(args.document_path.parent().unwrap_or(&PathBuf::from(".")))
    };

    let loader = Arc::new(DirectoryLoader::new(&data_dir)?);
    let metadata_extractor = Arc::new(MetadataExtractor::new());

    // Create vector store
    let vector_store = Arc::new(InMemoryVectorStore::new(
        DEFAULT_EMBEDDING_DIM,
        DistanceMetric::Cosine,
    ));

    // Build pipeline with traditional splitter
    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(loader)
        .with_transformer(traditional_splitter)
        .with_transformer(metadata_extractor)
        .with_embedder(embedder.clone())
        .with_vector_store(vector_store.clone())
        .build()?;

    // Run indexing
    let index_result = pipeline
        .run()
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;
    println!("✅ Traditional indexing completed");
    println!("📊 Indexed {} nodes", index_result.nodes_created);

    // Create Siumai client
    let siumai_client = create_siumai_client().await?;

    // Create query engine
    let retriever = Arc::new(VectorRetriever::new(vector_store.clone(), embedder));
    let generator = Arc::new(SiumaiGenerator::new(siumai_client));
    let query_engine = QueryEngine::new(retriever, generator);

    Ok((vector_store, query_engine))
}

/// Compare retrieval performance between semantic and traditional chunking
async fn compare_retrieval_performance(
    semantic_engine: &QueryEngine,
    traditional_engine: &QueryEngine,
    verbose: bool,
) -> ExampleResult<()> {
    let test_queries = get_climate_test_queries();

    println!("🔍 Testing {} queries...", test_queries.len());

    let mut semantic_total_time = std::time::Duration::ZERO;
    let mut traditional_total_time = std::time::Duration::ZERO;
    let mut semantic_scores = Vec::new();
    let mut traditional_scores = Vec::new();

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        // Test semantic chunking
        let semantic_timer = Timer::new("semantic_query");
        let semantic_result = semantic_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;
        let semantic_time = semantic_timer.finish();
        semantic_total_time += semantic_time;

        // Test traditional chunking
        let traditional_timer = Timer::new("traditional_query");
        let traditional_result = traditional_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;
        let traditional_time = traditional_timer.finish();
        traditional_total_time += traditional_time;

        if verbose {
            println!(
                "   🧠 Semantic result: {}",
                semantic_result
                    .response
                    .content
                    .chars()
                    .take(100)
                    .collect::<String>()
            );
            println!(
                "   📏 Traditional result: {}",
                traditional_result
                    .response
                    .content
                    .chars()
                    .take(100)
                    .collect::<String>()
            );
        }

        println!(
            "   ⏱️ Semantic time: {:.2}s, Traditional time: {:.2}s",
            semantic_time.as_secs_f64(),
            traditional_time.as_secs_f64()
        );

        // Calculate average similarity scores (if available)
        if !semantic_result.retrieved_nodes.is_empty() {
            let avg_score = semantic_result
                .retrieved_nodes
                .iter()
                .map(|n| n.score)
                .sum::<f32>()
                / semantic_result.retrieved_nodes.len() as f32;
            semantic_scores.push(avg_score);
        }

        if !traditional_result.retrieved_nodes.is_empty() {
            let avg_score = traditional_result
                .retrieved_nodes
                .iter()
                .map(|n| n.score)
                .sum::<f32>()
                / traditional_result.retrieved_nodes.len() as f32;
            traditional_scores.push(avg_score);
        }
    }

    // Summary statistics
    println!("\n📈 Performance Summary:");
    println!("   ⏱️ Average query time:");
    println!(
        "      🧠 Semantic: {:.2}s",
        semantic_total_time.as_secs_f64() / test_queries.len() as f64
    );
    println!(
        "      📏 Traditional: {:.2}s",
        traditional_total_time.as_secs_f64() / test_queries.len() as f64
    );

    if !semantic_scores.is_empty() && !traditional_scores.is_empty() {
        let semantic_avg = semantic_scores.iter().sum::<f32>() / semantic_scores.len() as f32;
        let traditional_avg =
            traditional_scores.iter().sum::<f32>() / traditional_scores.len() as f32;

        println!("   📊 Average similarity scores:");
        println!("      🧠 Semantic: {:.3}", semantic_avg);
        println!("      📏 Traditional: {:.3}", traditional_avg);
        println!(
            "      📈 Improvement: {:.1}%",
            ((semantic_avg - traditional_avg) / traditional_avg) * 100.0
        );
    }

    Ok(())
}

/// Run test queries on the semantic chunking system
async fn run_test_queries(query_engine: &QueryEngine, verbose: bool) -> ExampleResult<()> {
    let test_queries = get_climate_test_queries();

    println!("🔍 Running test queries...");

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        let timer = Timer::new("query");
        let result = query_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;
        let query_time = timer.finish();

        println!("💬 Response: {}", result.response.content);

        if verbose {
            if !result.retrieved_nodes.is_empty() {
                println!(
                    "📚 Retrieved {} context chunks:",
                    result.retrieved_nodes.len()
                );
                for (j, node) in result.retrieved_nodes.iter().enumerate() {
                    println!(
                        "   {}. Score: {:.3}, Length: {} chars",
                        j + 1,
                        node.score,
                        node.node.content.len()
                    );
                    println!(
                        "      Content: {}...",
                        node.node.content.chars().take(100).collect::<String>()
                    );
                }
            }
        }

        println!("⏱️ Query time: {:.2}s", query_time.as_secs_f64());
    }

    Ok(())
}

/// Run interactive mode for testing custom queries
async fn run_interactive_mode(query_engine: &QueryEngine) -> ExampleResult<()> {
    println!("\n🎯 Interactive Mode - Enter your queries (type 'quit' to exit):");

    loop {
        print!("\n❓ Your question: ");
        use std::io::{self, Write};
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let query = input.trim();

        if query.is_empty() {
            continue;
        }

        if query.to_lowercase() == "quit" {
            println!("👋 Goodbye!");
            break;
        }

        let timer = Timer::new("interactive_query");
        match query_engine.query(query).await {
            Ok(result) => {
                let query_time = timer.finish();
                println!("\n💬 Response: {}", result.response.content);
                println!("⏱️ Query time: {:.2}s", query_time.as_secs_f64());

                if !result.retrieved_nodes.is_empty() {
                    println!(
                        "📚 Used {} context chunks with avg score: {:.3}",
                        result.retrieved_nodes.len(),
                        result.retrieved_nodes.iter().map(|n| n.score).sum::<f32>()
                            / result.retrieved_nodes.len() as f32
                    );
                }
            }
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }
    }

    Ok(())
}
