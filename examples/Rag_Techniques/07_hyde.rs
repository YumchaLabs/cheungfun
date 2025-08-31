//! HyDE (Hypothetical Document Embedding) Example
//!
//! This example demonstrates the HyDE technique for improving retrieval in RAG systems.
//! HyDE works by generating hypothetical documents that would answer the user's query,
//! then using these hypothetical documents for embedding-based retrieval instead of
//! the original query.
//!
//! ## How HyDE Works
//!
//! 1. **Query Analysis**: Take the user's original query
//! 2. **Hypothetical Document Generation**: Use an LLM to generate a hypothetical document
//!    that would contain the answer to the query
//! 3. **Embedding**: Create embeddings for the hypothetical document
//! 4. **Retrieval**: Use the hypothetical document embeddings to find similar real documents
//! 5. **Generation**: Generate the final answer using the retrieved real documents
//!
//! ## Benefits
//!
//! - **Better Semantic Matching**: Hypothetical documents often match the style and content
//!   of real documents better than queries do
//! - **Improved Recall**: Can find relevant documents that don't contain the exact query terms
//! - **Domain Adaptation**: Works well across different domains and document types
//!
//! ## Usage
//!
//! ```bash
//! # Run with default HyDE strategy
//! cargo run --bin hyde --features fastembed
//!
//! # Run with specific HyDE strategy
//! cargo run --bin hyde --features fastembed -- --strategy single
//! cargo run --bin hyde --features fastembed -- --strategy multiple
//! cargo run --bin hyde --features fastembed -- --strategy iterative
//!
//! # Compare with baseline (no HyDE)
//! cargo run --bin hyde --features fastembed -- --compare-baseline
//!
//! # Interactive mode
//! cargo run --bin hyde --features fastembed -- --interactive
//! ```

use clap::Parser;
use serde::{Deserialize, Serialize};

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{
    constants::*, get_climate_test_queries, setup_logging, ExampleError, ExampleResult,
    PerformanceMetrics, Timer,
};
use std::{path::PathBuf, sync::Arc, time::Duration};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline},
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
    advanced::{query_transformers::HyDETransformer, AdvancedQuery, QueryTransformer},
    engine::QueryEngine,
    generator::SiumaiGenerator,
    prelude::QueryResponse,
    retriever::VectorRetriever,
};
use siumai::prelude::*;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug)]
#[command(name = "hyde")]
#[command(
    about = "HyDE (Hypothetical Document Embedding) Example - Improve retrieval with hypothetical documents"
)]
struct Args {
    /// Path to the document to process
    #[arg(long, default_value = "data/Understanding_Climate_Change.pdf")]
    document_path: PathBuf,

    /// Embedding provider (fastembed, openai)
    #[arg(long, default_value = "fastembed")]
    embedding_provider: String,

    /// Chunk size for text splitting
    #[arg(long, default_value_t = DEFAULT_CHUNK_SIZE)]
    chunk_size: usize,

    /// Chunk overlap for text splitting
    #[arg(long, default_value_t = DEFAULT_CHUNK_OVERLAP)]
    chunk_overlap: usize,

    /// Number of top results to retrieve
    #[arg(long, default_value_t = DEFAULT_TOP_K)]
    top_k: usize,

    /// HyDE strategy to use
    #[arg(long, value_enum, default_value = "single")]
    strategy: HydeStrategy,

    /// Compare with baseline (no HyDE)
    #[arg(long)]
    compare_baseline: bool,

    /// Run in interactive mode
    #[arg(long)]
    interactive: bool,

    /// Show detailed HyDE process
    #[arg(long)]
    verbose: bool,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum HydeStrategy {
    /// Generate a single hypothetical document (default)
    Single,
    /// Generate multiple hypothetical documents for better coverage
    Multiple,
    /// Include original query alongside hypothetical documents
    WithOriginal,
}

/// Represents a hypothetical document generated for HyDE
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HypotheticalDocument {
    pub original_query: String,
    pub hypothetical_content: String,
    pub strategy: String,
    pub confidence: f32,
    pub generation_time: std::time::Duration,
}

/// Results from HyDE processing
#[derive(Debug)]
struct HydeResults {
    pub original_query: String,
    pub hypothetical_documents: Vec<HypotheticalDocument>,
    pub hyde_response: QueryResponse,
    pub baseline_response: Option<QueryResponse>,
    pub performance_metrics: HydeMetrics,
}

/// Performance metrics for HyDE
#[derive(Debug, Default)]
struct HydeMetrics {
    pub total_generation_time: std::time::Duration,
    pub total_retrieval_time: std::time::Duration,
    pub hyde_similarity_score: f32,
    pub baseline_similarity_score: f32,
    pub improvement_percentage: f32,
    pub num_hypothetical_docs: usize,
}

impl HydeResults {
    pub fn print_summary(&self, verbose: bool) {
        println!("\n🔮 HYDE RESULTS");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        println!("📝 Original Query: {}", self.original_query);
        println!(
            "🎯 Generated {} hypothetical documents",
            self.hypothetical_documents.len()
        );
        println!();

        if verbose {
            println!("📄 Hypothetical Documents:");
            for (i, hyp_doc) in self.hypothetical_documents.iter().enumerate() {
                println!(
                    "  {}. Strategy: {} (Confidence: {:.2})",
                    i + 1,
                    hyp_doc.strategy,
                    hyp_doc.confidence
                );
                println!(
                    "     Generation Time: {:.0}ms",
                    hyp_doc.generation_time.as_millis()
                );
                println!(
                    "     Content: {}",
                    if hyp_doc.hypothetical_content.len() > 200 {
                        format!("{}...", &hyp_doc.hypothetical_content[..200])
                    } else {
                        hyp_doc.hypothetical_content.clone()
                    }
                );
                println!();
            }
        }

        println!("📊 Performance Metrics:");
        println!(
            "   ⏱️  Total Generation Time: {:.0}ms",
            self.performance_metrics.total_generation_time.as_millis()
        );
        println!(
            "   🔍 Total Retrieval Time: {:.0}ms",
            self.performance_metrics.total_retrieval_time.as_millis()
        );
        println!(
            "   🎯 HyDE Similarity Score: {:.3}",
            self.performance_metrics.hyde_similarity_score
        );

        if let Some(ref _baseline) = self.baseline_response {
            println!(
                "   📊 Baseline Similarity Score: {:.3}",
                self.performance_metrics.baseline_similarity_score
            );
            println!(
                "   📈 Improvement: {:.1}%",
                self.performance_metrics.improvement_percentage
            );

            if self.performance_metrics.improvement_percentage > 0.0 {
                println!("   ✅ HyDE improved retrieval quality!");
            } else if self.performance_metrics.improvement_percentage < -5.0 {
                println!("   ⚠️  HyDE performed worse than baseline");
            } else {
                println!("   ➖ HyDE performance similar to baseline");
            }
        }

        println!("\n📝 HyDE Response:");
        println!("{}", self.hyde_response.response.content);

        if let Some(ref baseline) = self.baseline_response {
            println!("\n📝 Baseline Response:");
            println!("{}", baseline.response.content);
        }

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    // Setup logging
    setup_logging();

    let args = Args::parse();

    println!("🚀 Starting HyDE (Hypothetical Document Embedding) Example...");

    // Print configuration
    print_config(&args);

    let mut metrics = PerformanceMetrics::new();

    // Step 1: Create embedder
    let embedder = create_embedder(&args.embedding_provider).await?;
    println!("✅ Embedder initialized: {}", args.embedding_provider);

    // Step 2: Create query engine
    let query_engine = create_query_engine(&args, embedder).await?;
    println!("✅ Query engine initialized");

    // Step 3: Create LLM client for HyDE generation
    let llm_client = create_llm_client().await?;
    println!("✅ LLM client for HyDE generation initialized");

    if args.interactive {
        run_interactive_mode(&query_engine, &llm_client, &args, &mut metrics).await?;
    } else {
        run_demo_queries(&query_engine, &llm_client, &args, &mut metrics).await?;
    }

    // Print final metrics
    metrics.print_summary();

    Ok(())
}

fn print_config(args: &Args) {
    println!("🔮 HyDE (Hypothetical Document Embedding) Example");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📄 Document: {}", args.document_path.display());
    println!("🔤 Embedding Provider: {}", args.embedding_provider);
    println!(
        "📏 Chunk Size: {} (overlap: {})",
        args.chunk_size, args.chunk_overlap
    );
    println!("🔍 Top-K: {}", args.top_k);
    println!("🎯 HyDE Strategy: {:?}", args.strategy);
    println!("📊 Compare Baseline: {}", args.compare_baseline);
    println!("🔍 Verbose: {}", args.verbose);
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
            println!("🤖 Using OpenAI for HyDE generation (cloud)");
            return Siumai::builder()
                .openai()
                .api_key(&api_key)
                .model("gpt-4o-mini")
                .temperature(0.7) // Higher temperature for creative hypothetical documents
                .max_tokens(2000)
                .build()
                .await
                .map_err(|e| ExampleError::Config(format!("Failed to initialize OpenAI: {}", e)));
        }
    }

    // Fallback to Ollama
    println!("🤖 No valid OpenAI API key found, using Ollama for HyDE generation (local)");
    println!("💡 Make sure Ollama is running with: ollama serve");
    println!("💡 And pull a model with: ollama pull llama3.2");

    Siumai::builder()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .temperature(0.7)
        .build()
        .await
        .map_err(|e| ExampleError::Config(format!("Failed to initialize Ollama: {}. Make sure Ollama is running with 'ollama serve' and you have pulled a model with 'ollama pull llama3.2'", e)))
}

async fn create_query_engine(
    args: &Args,
    embedder: Arc<dyn Embedder>,
) -> ExampleResult<QueryEngine> {
    // Create vector store
    let vector_store = Arc::new(InMemoryVectorStore::new(
        DEFAULT_EMBEDDING_DIM,
        DistanceMetric::Cosine,
    ));

    // Build indexing pipeline
    let timer = Timer::new("Document indexing");

    // Get the directory containing the document
    let default_path = PathBuf::from(".");
    let data_dir = args.document_path.parent().unwrap_or(&default_path);
    println!("📂 Loading from directory: {}", data_dir.display());

    let loader = Arc::new(DirectoryLoader::new(data_dir)?);

    // Create text splitter with custom configuration
    let splitter_config = SentenceSplitterConfig::default();
    let splitter = Arc::new(SentenceSplitter::new(splitter_config)?);
    let metadata_extractor = Arc::new(MetadataExtractor::new());

    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(loader)
        .with_document_processor(splitter)  // Documents -> Nodes
        .with_node_processor(metadata_extractor)  // Nodes -> Nodes
        .with_embedder(embedder.clone())
        .with_vector_store(vector_store.clone())
        .build()?;

    // Run indexing pipeline with progress reporting
    let (_nodes, indexing_stats) = pipeline
        .run_with_progress(
            None,  // documents (will use loader)
            None,  // nodes
            true,  // store_doc_text
            None,  // num_workers (use default)
            true,  // in_place
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
            })
        )
        .await?;

    let indexing_time = timer.finish();

    println!(
        "✅ Completed: Document indexing in {:.2}s",
        indexing_time.as_secs_f64()
    );
    println!("📊 Indexing completed:");
    println!("  📚 Documents: {}", indexing_stats.documents_processed);
    println!("  🔗 Nodes: {}", indexing_stats.nodes_created);
    println!("  ⏱️  Time: {:.2}s", indexing_time.as_secs_f64());

    // Create query engine
    let retriever = Arc::new(VectorRetriever::new(vector_store, embedder));

    // Create LLM client for generation
    let generation_llm = create_generation_llm().await?;
    let generator = Arc::new(SiumaiGenerator::new(generation_llm));

    let query_engine = QueryEngine::new(retriever, generator);

    Ok(query_engine)
}

async fn create_generation_llm() -> ExampleResult<Siumai> {
    // Try OpenAI first
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        if !api_key.is_empty() && api_key != "test" && api_key.starts_with("sk-") {
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
    Siumai::builder()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .temperature(0.0)
        .build()
        .await
        .map_err(|e| ExampleError::Config(format!("Failed to initialize Ollama: {}", e)))
}

/// Generate hypothetical documents using our HyDETransformer
async fn generate_hypothetical_documents(
    query: &str,
    strategy: &HydeStrategy,
    llm_client: &Siumai,
) -> ExampleResult<Vec<HypotheticalDocument>> {
    // Create HyDETransformer based on strategy
    let (num_docs, include_original) = match strategy {
        HydeStrategy::Single => (1, false),
        HydeStrategy::Multiple => (3, false),
        HydeStrategy::WithOriginal => (2, true),
    };

    // TODO: Once Siumai supports Clone, we can use llm_client.clone() instead of creating a new instance
    // For now, we need to create a new Siumai client since it doesn't implement Clone
    let new_llm_client = create_llm_client().await?;

    // Create HyDETransformer using our library
    let hyde_transformer = HyDETransformer::new(Arc::new(SiumaiGenerator::new(new_llm_client)))
        .with_num_hypothetical_docs(num_docs)
        .with_include_original(include_original);

    // Create AdvancedQuery
    let mut advanced_query = AdvancedQuery::from_text(query.to_string());

    // Apply HyDE transformation
    hyde_transformer
        .transform(&mut advanced_query)
        .await
        .map_err(|e| ExampleError::DataProcessing(format!("HyDE transformation failed: {}", e)))?;

    // Convert to our HypotheticalDocument format for compatibility
    let mut hypothetical_docs = Vec::new();
    for (i, transformed_query) in advanced_query.transformed_queries.iter().enumerate() {
        if i == 0 && include_original && transformed_query == &query {
            // Skip the original query in the hypothetical documents list
            continue;
        }
        hypothetical_docs.push(HypotheticalDocument {
            original_query: query.to_string(),
            hypothetical_content: transformed_query.clone(),
            strategy: format!("{:?}", strategy),
            confidence: 0.8,                             // Default confidence
            generation_time: Duration::from_millis(100), // Placeholder
        });
    }

    Ok(hypothetical_docs)
}

/// Perform HyDE query processing using the transformed queries
async fn perform_hyde_query(
    query: &str,
    strategy: &HydeStrategy,
    query_engine: &QueryEngine,
    llm_client: &Siumai,
    compare_baseline: bool,
) -> ExampleResult<HydeResults> {
    let timer = Timer::new("HyDE query processing");

    // Step 1: Generate hypothetical documents using our library
    let hypothetical_docs = generate_hypothetical_documents(query, strategy, llm_client).await?;

    println!(
        "📝 Generated {} hypothetical documents",
        hypothetical_docs.len()
    );
    if hypothetical_docs.is_empty() {
        return Err(ExampleError::DataProcessing(
            "No hypothetical documents generated".to_string(),
        ));
    }

    // Step 2: Use the first hypothetical document for retrieval (or combine multiple)
    let retrieval_query = if hypothetical_docs.len() == 1 {
        &hypothetical_docs[0].hypothetical_content
    } else {
        // For multiple documents, use the first one (could be enhanced to combine)
        &hypothetical_docs[0].hypothetical_content
    };

    // Step 3: Perform retrieval using the hypothetical document
    let hyde_response = query_engine
        .query(retrieval_query)
        .await
        .map_err(ExampleError::Cheungfun)?;

    // Step 4: Optionally compare with baseline (original query)
    let baseline_response = if compare_baseline {
        Some(
            query_engine
                .query(query)
                .await
                .map_err(ExampleError::Cheungfun)?,
        )
    } else {
        None
    };

    let total_time = timer.finish();

    // Step 5: Calculate performance metrics
    let mut metrics = HydeMetrics::default();
    metrics.total_generation_time = hypothetical_docs
        .iter()
        .map(|doc| doc.generation_time)
        .sum();
    metrics.total_retrieval_time = total_time - metrics.total_generation_time;
    metrics.hyde_similarity_score = hyde_response
        .retrieved_nodes
        .first()
        .map(|node| node.score)
        .unwrap_or(0.0);

    if let Some(ref baseline) = baseline_response {
        metrics.baseline_similarity_score = baseline
            .retrieved_nodes
            .first()
            .map(|node| node.score)
            .unwrap_or(0.0);

        if metrics.baseline_similarity_score > 0.0 {
            metrics.improvement_percentage = ((metrics.hyde_similarity_score
                - metrics.baseline_similarity_score)
                / metrics.baseline_similarity_score)
                * 100.0;
        }
    }

    Ok(HydeResults {
        original_query: query.to_string(),
        hypothetical_documents: hypothetical_docs,
        hyde_response,
        baseline_response,
        performance_metrics: metrics,
    })
}

/// Run demo queries with HyDE processing
async fn run_demo_queries(
    query_engine: &QueryEngine,
    llm_client: &Siumai,
    args: &Args,
    metrics: &mut PerformanceMetrics,
) -> ExampleResult<()> {
    println!("🔮 Running HyDE demo queries...");
    println!();

    let queries = get_climate_test_queries();

    for (i, query) in queries.iter().enumerate() {
        println!("🧪 Demo Query {}/{}: {}", i + 1, queries.len(), query);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let timer = Timer::new("HyDE query processing");

        let results = perform_hyde_query(
            query,
            &args.strategy,
            query_engine,
            llm_client,
            args.compare_baseline,
        )
        .await?;

        let processing_time = timer.finish();
        metrics.record_query(processing_time);

        results.print_summary(args.verbose);

        println!();
    }

    Ok(())
}

/// Run interactive mode with HyDE processing
async fn run_interactive_mode(
    query_engine: &QueryEngine,
    llm_client: &Siumai,
    args: &Args,
    metrics: &mut PerformanceMetrics,
) -> ExampleResult<()> {
    println!("🎯 Interactive HyDE Mode");
    println!("Type your questions, or 'quit' to exit.");
    println!("Use 'strategy <name>' to change HyDE strategy.");
    println!("Available strategies: single, multiple, with-original");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let mut current_strategy = args.strategy.clone();

    loop {
        println!("Current strategy: {:?}", current_strategy);
        print!("❓ Your question (or command): ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input.to_lowercase() == "quit" {
            break;
        }

        // Handle strategy change commands
        if input.starts_with("strategy ") {
            let strategy_name = input.strip_prefix("strategy ").unwrap().trim();
            match strategy_name.to_lowercase().as_str() {
                "single" => current_strategy = HydeStrategy::Single,
                "multiple" => current_strategy = HydeStrategy::Multiple,
                "with-original" | "withoriginal" => current_strategy = HydeStrategy::WithOriginal,
                _ => {
                    println!("❌ Unknown strategy. Available: single, multiple, with-original");
                    continue;
                }
            }
            println!("✅ Strategy changed to: {:?}", current_strategy);
            continue;
        }

        let timer = Timer::new("HyDE query processing");

        match perform_hyde_query(
            input,
            &current_strategy,
            query_engine,
            llm_client,
            args.compare_baseline,
        )
        .await
        {
            Ok(results) => {
                let processing_time = timer.finish();
                metrics.record_query(processing_time);
                results.print_summary(args.verbose);
            }
            Err(e) => {
                println!("❌ Error processing query: {}", e);
            }
        }

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();
    }

    println!("👋 Goodbye!");
    Ok(())
}
