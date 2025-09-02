//! LlamaIndex-Style Postprocessor Chain Example
//!
//! This example demonstrates the new LlamaIndex-compatible API for postprocessor chains
//! in Cheungfun. It shows how to use the `with_postprocessors()` method to create
//! query engines with automatic postprocessor chain management.
//!
//! ## Key Features
//!
//! - **LlamaIndex-Compatible API**: Use `query_engine.with_postprocessors(vec![...])`
//! - **Automatic Chain Management**: No manual chain creation required
//! - **Builder Pattern Support**: Works with QueryEngineBuilder
//! - **Error Handling**: Configurable error handling strategies
//! - **Verbose Logging**: Optional detailed processing logs
//!
//! ## Usage
//!
//! ```bash
//! # Basic usage with postprocessor chain
//! cargo run --bin llamaindex_style_postprocessors --features fastembed
//!
//! # With verbose logging
//! cargo run --bin llamaindex_style_postprocessors --features fastembed -- --verbose
//!
//! # With error handling configuration
//! cargo run --bin llamaindex_style_postprocessors --features fastembed -- --continue-on-error
//! ```

use clap::Parser;
use std::{path::PathBuf, sync::Arc};

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{constants::*, setup_logging, ExampleError, ExampleResult, PerformanceMetrics, Timer};

use cheungfun_core::traits::{Embedder, IndexingPipeline};
use cheungfun_indexing::{
    loaders::DirectoryLoader,
    node_parser::{config::SentenceSplitterConfig, text::SentenceSplitter},
    pipeline::DefaultIndexingPipeline,
    transformers::MetadataExtractor,
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    engine::QueryEngine,
    generator::SiumaiGenerator,
    postprocessor::{
        KeywordFilter, KeywordFilterConfig, MetadataFilter, MetadataFilterConfig,
        SentenceEmbeddingConfig, SentenceEmbeddingOptimizer, SimilarityFilter,
        SimilarityFilterConfig,
    },
    retriever::VectorRetriever,
};
use siumai::prelude::*;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug, Clone)]
#[command(name = "llamaindex_style_postprocessors")]
#[command(about = "LlamaIndex-Style Postprocessor Chain Example")]
struct Args {
    /// Path to the document to process
    #[arg(long, default_value = "data/Understanding_Climate_Change.pdf")]
    document_path: PathBuf,

    /// Embedding provider (fastembed, openai)
    #[arg(long, default_value = "fastembed")]
    embedding_provider: String,

    /// Number of top results to retrieve
    #[arg(long, default_value_t = DEFAULT_TOP_K)]
    top_k: usize,

    /// Enable verbose logging for postprocessor chain
    #[arg(long)]
    verbose: bool,

    /// Continue processing even if a postprocessor fails
    #[arg(long)]
    continue_on_error: bool,

    /// Query to test with
    #[arg(long, default_value = "What are the main impacts of climate change?")]
    query: String,

    /// Run interactive mode
    #[arg(long)]
    interactive: bool,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    let args = Args::parse();
    setup_logging();

    println!("🚀 LlamaIndex-Style Postprocessor Chain Example");
    println!("================================================");
    println!();

    // Create embedder
    let embedder = create_embedder(&args.embedding_provider).await?;

    // Create and run the demo
    let demo = LlamaIndexStyleDemo::new(args.clone(), embedder).await?;
    demo.run().await?;

    Ok(())
}

/// Demo showcasing LlamaIndex-style postprocessor API
pub struct LlamaIndexStyleDemo {
    args: Args,
    query_engine_basic: QueryEngine,
    query_engine_advanced: QueryEngine,
    query_engine_builder: QueryEngine,
    performance_metrics: PerformanceMetrics,
}

impl LlamaIndexStyleDemo {
    /// Create a new demo instance
    pub async fn new(args: Args, embedder: Arc<dyn Embedder>) -> ExampleResult<Self> {
        let performance_metrics = PerformanceMetrics::new();

        // Create vector store and index documents
        let vector_store = Arc::new(InMemoryVectorStore::new(
            DEFAULT_EMBEDDING_DIM,
            cheungfun_core::DistanceMetric::Cosine,
        ));

        // Build indexing pipeline
        let timer = Timer::new("Document indexing");
        let default_path = PathBuf::from(".");
        let data_dir = args.document_path.parent().unwrap_or(&default_path);
        let loader = Arc::new(DirectoryLoader::new(data_dir)?);
        let splitter_config = SentenceSplitterConfig::default();
        let splitter = Arc::new(SentenceSplitter::new(splitter_config)?);
        let metadata_extractor = Arc::new(MetadataExtractor::new());

        let pipeline = DefaultIndexingPipeline::builder()
            .with_loader(loader)
            .with_document_processor(splitter) // Documents -> Nodes
            .with_node_processor(metadata_extractor) // Nodes -> Nodes
            .with_embedder(embedder.clone())
            .with_vector_store(vector_store.clone())
            .build()?;

        let (_nodes, _indexing_stats) = pipeline.run(None, None, true, true, None, true).await?;
        let _indexing_time = timer.finish();

        // Create LLM client and generator
        let llm_client = create_llm_client().await?;
        let generator = Arc::new(SiumaiGenerator::new(llm_client.clone()));
        let retriever = Arc::new(VectorRetriever::new(vector_store.clone(), embedder.clone()));

        // Create postprocessors
        let keyword_filter = Arc::new(KeywordFilter::new(KeywordFilterConfig {
            required_keywords: vec!["climate".to_string(), "change".to_string()],
            exclude_keywords: vec![],
            case_sensitive: false,
            min_required_matches: 1,
        })?);

        let metadata_filter = Arc::new(MetadataFilter::new(MetadataFilterConfig {
            required_metadata: {
                let mut map = std::collections::HashMap::new();
                map.insert("source".to_string(), "document".to_string());
                map
            },
            excluded_metadata: std::collections::HashMap::new(),
            require_all: false,
        }));

        let similarity_filter = Arc::new(SimilarityFilter::new(SimilarityFilterConfig {
            similarity_cutoff: 0.3,
            max_nodes: Some(args.top_k),
            use_query_embedding: true,
        }));

        let sentence_optimizer = Arc::new(SentenceEmbeddingOptimizer::new(
            embedder.clone(),
            SentenceEmbeddingConfig {
                percentile_cutoff: Some(0.7),
                threshold_cutoff: Some(0.5),
                context_before: Some(1),
                context_after: Some(1),
                max_sentences_per_node: Some(10),
            },
        ));

        // Method 1: Basic QueryEngine with postprocessors (LlamaIndex style)
        let query_engine_basic = QueryEngine::new(retriever.clone(), generator.clone())
            .with_postprocessors(vec![keyword_filter.clone(), similarity_filter.clone()]);

        // Method 2: Advanced configuration with all postprocessors
        let query_engine_advanced = QueryEngine::new(retriever.clone(), generator.clone())
            .with_postprocessors_config(
                vec![
                    keyword_filter.clone(),
                    metadata_filter.clone(),
                    similarity_filter.clone(),
                    sentence_optimizer.clone(),
                ],
                args.continue_on_error,
                args.verbose,
            );

        // Method 3: Using QueryEngineBuilder (also LlamaIndex style)
        let query_engine_builder = QueryEngine::builder()
            .retriever(retriever.clone())
            .generator(generator.clone())
            .postprocessors(vec![
                metadata_filter.clone(),
                similarity_filter.clone(),
                sentence_optimizer.clone(),
            ])
            .build()?;

        Ok(Self {
            args,
            query_engine_basic,
            query_engine_advanced,
            query_engine_builder,
            performance_metrics,
        })
    }

    /// Run the demo
    pub async fn run(&self) -> ExampleResult<()> {
        if self.args.interactive {
            self.run_interactive().await
        } else {
            self.run_comparison().await
        }
    }

    /// Run comparison between different approaches
    async fn run_comparison(&self) -> ExampleResult<()> {
        println!("🔍 Comparing Different Postprocessor Approaches");
        println!("===============================================");
        println!();

        let queries = vec![
            self.args.query.clone(),
            "How does climate change affect biodiversity?".to_string(),
            "What are renewable energy solutions?".to_string(),
        ];

        for (i, query) in queries.iter().enumerate() {
            println!("📝 Query {}: {}", i + 1, query);
            println!("{}", "─".repeat(60));

            // Test basic approach
            println!("🔧 Method 1: Basic with_postprocessors()");
            let timer = Timer::new("Basic query");
            let response1 = self.query_engine_basic.query(query).await?;
            let time1 = timer.finish();
            println!(
                "   ✅ Response: {}",
                response1.content().chars().take(100).collect::<String>() + "..."
            );
            println!("   ⏱️  Time: {:.2}s", time1.as_secs_f64());
            println!(
                "   📊 Context nodes: {}",
                response1.response.source_nodes.len()
            );
            println!();

            // Test advanced approach
            println!("🔧 Method 2: Advanced with_postprocessors_config()");
            let timer = Timer::new("Advanced query");
            let response2 = self.query_engine_advanced.query(query).await?;
            let time2 = timer.finish();
            println!(
                "   ✅ Response: {}",
                response2.content().chars().take(100).collect::<String>() + "..."
            );
            println!("   ⏱️  Time: {:.2}s", time2.as_secs_f64());
            println!(
                "   📊 Context nodes: {}",
                response2.response.source_nodes.len()
            );
            println!();

            // Test builder approach
            println!("🔧 Method 3: QueryEngineBuilder.postprocessors()");
            let timer = Timer::new("Builder query");
            let response3 = self.query_engine_builder.query(query).await?;
            let time3 = timer.finish();
            println!(
                "   ✅ Response: {}",
                response3.content().chars().take(100).collect::<String>() + "..."
            );
            println!("   ⏱️  Time: {:.2}s", time3.as_secs_f64());
            println!(
                "   📊 Context nodes: {}",
                response3.response.source_nodes.len()
            );
            println!();

            println!("{}", "═".repeat(60));
            println!();
        }

        self.print_summary();
        Ok(())
    }

    /// Run interactive mode
    async fn run_interactive(&self) -> ExampleResult<()> {
        println!("🎮 Interactive Mode - LlamaIndex-Style Postprocessors");
        println!("====================================================");
        println!("Enter queries to test different postprocessor approaches.");
        println!("Type 'quit' to exit.");
        println!();

        loop {
            print!("🔍 Enter your query: ");
            use std::io::{self, Write};
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let query = input.trim();

            if query.is_empty() {
                continue;
            }

            if query.to_lowercase() == "quit" {
                break;
            }

            println!("\n🚀 Processing query: {}", query);
            println!("{}", "─".repeat(50));

            // Use the advanced query engine for interactive mode
            let timer = Timer::new("Interactive query");
            match self.query_engine_advanced.query(query).await {
                Ok(response) => {
                    let time = timer.finish();
                    println!("✅ Response:");
                    println!("{}", response.content());
                    println!();
                    println!("📊 Metadata:");
                    println!("   ⏱️  Processing time: {:.2}s", time.as_secs_f64());
                    println!(
                        "   📄 Source nodes: {}",
                        response.response.source_nodes.len()
                    );
                    if !response.response.source_nodes.is_empty() {
                        println!(
                            "   🎯 Source node IDs: {}",
                            response.response.source_nodes.len()
                        );
                    }
                }
                Err(e) => {
                    println!("❌ Error: {}", e);
                }
            }

            println!("\n{}\n", "═".repeat(50));
        }

        Ok(())
    }

    /// Print summary of the demo
    fn print_summary(&self) {
        println!("📋 LlamaIndex-Style API Summary");
        println!("==============================");
        println!();
        println!("✅ Successfully demonstrated three approaches:");
        println!("   1. QueryEngine::new().with_postprocessors(vec![...])");
        println!("   2. QueryEngine::new().with_postprocessors_config(vec![...], continue_on_error, verbose)");
        println!("   3. QueryEngine::builder().postprocessors(vec![...]).build()");
        println!();
        println!("🎯 Key Benefits:");
        println!("   • LlamaIndex-compatible API design");
        println!("   • Automatic postprocessor chain management");
        println!("   • Configurable error handling and logging");
        println!("   • Builder pattern support");
        println!("   • Type-safe postprocessor composition");
        println!();
        println!("🚀 This brings Cheungfun's API in line with LlamaIndex standards!");
    }
}

/// Create embedder based on provider
async fn create_embedder(provider: &str) -> ExampleResult<Arc<dyn Embedder>> {
    match provider {
        "fastembed" => {
            println!("🤖 Using FastEmbed for embeddings");
            let embedder = FastEmbedder::new().await.map_err(|e| {
                ExampleError::Config(format!("FastEmbed initialization failed: {}", e))
            })?;
            Ok(Arc::new(embedder))
        }
        _ => Err(ExampleError::Config(format!(
            "Unsupported embedding provider: {}",
            provider
        ))),
    }
}

/// Create LLM client
async fn create_llm_client() -> ExampleResult<Siumai> {
    // Try OpenAI first
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        if !api_key.is_empty() {
            println!("🤖 Using OpenAI for generation");
            return Siumai::builder()
                .openai()
                .api_key(&api_key)
                .model("gpt-3.5-turbo")
                .temperature(0.0)
                .max_tokens(4000)
                .build()
                .await
                .map_err(|e| ExampleError::Config(format!("Failed to initialize OpenAI: {}", e)));
        }
    }

    // Fallback to Ollama
    println!("🤖 No valid OpenAI API key found, using Ollama for generation (local)");
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
