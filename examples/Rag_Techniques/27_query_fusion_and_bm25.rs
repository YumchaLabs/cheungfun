//! Query Fusion and BM25 Retriever Example
//!
//! This example demonstrates the new QueryFusionRetriever and BM25Retriever components,
//! showing how to combine multiple retrieval strategies for enhanced search performance.
//!
//! ## Key Features
//!
//! - **BM25Retriever**: Standalone BM25-based keyword search
//! - **QueryFusionRetriever**: Unified interface for multi-retriever fusion
//! - **Multiple Fusion Modes**: RRF, weighted average, distance-based scoring
//! - **Performance Comparison**: Compare different retrieval strategies
//!
//! ## Usage
//!
//! ```bash
//! # Basic fusion retrieval
//! cargo run --bin query_fusion_and_bm25 --features fastembed
//!
//! # Compare different fusion modes
//! cargo run --bin query_fusion_and_bm25 --features fastembed -- --compare-modes
//!
//! # Interactive mode
//! cargo run --bin query_fusion_and_bm25 --features fastembed -- --interactive
//! ```

use clap::Parser;
use std::{path::PathBuf, sync::Arc};

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{constants::*, setup_logging, ExampleError, ExampleResult, PerformanceMetrics, Timer};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, Loader, Retriever},
    types::Query,
};
use cheungfun_indexing::{
    loaders::DirectoryLoader,
    node_parser::{config::SentenceSplitterConfig, text::SentenceSplitter, NodeParser},
    pipeline::DefaultIndexingPipeline,
    transformers::MetadataExtractor,
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    engine::QueryEngine,
    generator::SiumaiGenerator,
    retriever::VectorRetriever,
    retrievers::{BM25Config, BM25Retriever, QueryFusionRetriever},
};
use siumai::prelude::*;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug, Clone)]
#[command(name = "query_fusion_and_bm25")]
#[command(about = "Query Fusion and BM25 Retriever Example")]
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

    /// Query to test with
    #[arg(long, default_value = "What are the main impacts of climate change?")]
    query: String,

    /// Compare different fusion modes
    #[arg(long)]
    compare_modes: bool,

    /// Run interactive mode
    #[arg(long)]
    interactive: bool,

    /// Vector retriever weight for fusion
    #[arg(long, default_value_t = 0.6)]
    vector_weight: f32,

    /// BM25 retriever weight for fusion
    #[arg(long, default_value_t = 0.4)]
    bm25_weight: f32,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    let args = Args::parse();
    setup_logging();

    println!("🚀 Query Fusion and BM25 Retriever Example");
    println!("==========================================");
    println!();

    // Create embedder
    let embedder = create_embedder(&args.embedding_provider).await?;

    // Create and run the demo
    let demo = QueryFusionDemo::new(args.clone(), embedder).await?;
    demo.run().await?;

    Ok(())
}

/// Demo showcasing QueryFusionRetriever and BM25Retriever
pub struct QueryFusionDemo {
    args: Args,
    vector_retriever: Arc<VectorRetriever>,
    bm25_retriever: Arc<BM25Retriever>,
    fusion_retriever_rrf: Arc<QueryFusionRetriever>,
    fusion_retriever_weighted: Arc<QueryFusionRetriever>,
    query_engine: QueryEngine,
    performance_metrics: PerformanceMetrics,
}

impl QueryFusionDemo {
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

        // Load documents first for BM25 indexing
        let documents = loader.load().await?;

        let pipeline = DefaultIndexingPipeline::builder()
            .with_loader(loader)
            .with_document_processor(splitter) // Documents -> Nodes
            .with_node_processor(metadata_extractor) // Nodes -> Nodes
            .with_embedder(embedder.clone())
            .with_vector_store(vector_store.clone())
            .build()?;

        let (_nodes, _indexing_stats) = pipeline.run(None, None, true, true, None, true).await?;
        let _indexing_time = timer.finish();

        // Create nodes for BM25 indexing using a separate splitter
        let splitter_config = SentenceSplitterConfig::default();
        let bm25_splitter = SentenceSplitter::new(splitter_config)?;

        // Use the splitter to create nodes for BM25
        let nodes = bm25_splitter.parse_nodes(&documents, false).await?;

        // Create retrievers
        let vector_retriever =
            Arc::new(VectorRetriever::new(vector_store.clone(), embedder.clone()));

        // Create BM25 retriever
        let bm25_config = BM25Config {
            max_results: args.top_k * 2, // Get more intermediate results
            ..Default::default()
        };
        let bm25_retriever = Arc::new(BM25Retriever::from_nodes(nodes, bm25_config).await?);

        // Create fusion retrievers with different modes
        let fusion_retriever_rrf = Arc::new(QueryFusionRetriever::with_rrf(
            vec![vector_retriever.clone(), bm25_retriever.clone()],
            60.0, // RRF k parameter
        )?);

        let fusion_retriever_weighted = Arc::new(QueryFusionRetriever::with_weighted_average(
            vec![vector_retriever.clone(), bm25_retriever.clone()],
            vec![args.vector_weight, args.bm25_weight],
            true, // normalize scores
        )?);

        // Create LLM client and query engine
        let llm_client = create_llm_client().await?;
        let generator = Arc::new(SiumaiGenerator::new(llm_client));
        let query_engine = QueryEngine::new(fusion_retriever_rrf.clone(), generator);

        Ok(Self {
            args,
            vector_retriever,
            bm25_retriever,
            fusion_retriever_rrf,
            fusion_retriever_weighted,
            query_engine,
            performance_metrics,
        })
    }

    /// Run the demo
    pub async fn run(&self) -> ExampleResult<()> {
        if self.args.interactive {
            self.run_interactive().await
        } else if self.args.compare_modes {
            self.run_mode_comparison().await
        } else {
            self.run_basic_demo().await
        }
    }

    /// Run basic demonstration
    async fn run_basic_demo(&self) -> ExampleResult<()> {
        println!("🔍 Basic Query Fusion Demonstration");
        println!("===================================");
        println!();

        let query = Query::new(&self.args.query).with_top_k(self.args.top_k);

        // Test individual retrievers
        println!("📊 Individual Retriever Performance:");
        println!("-----------------------------------");

        // Vector retriever
        let timer = Timer::new("Vector retrieval");
        let vector_results = self.vector_retriever.retrieve(&query).await?;
        let vector_time = timer.finish();
        println!(
            "🔢 Vector Retriever: {} results in {:.2}s",
            vector_results.len(),
            vector_time.as_secs_f64()
        );

        // BM25 retriever
        let timer = Timer::new("BM25 retrieval");
        let bm25_results = self.bm25_retriever.retrieve(&query).await?;
        let bm25_time = timer.finish();
        println!(
            "📝 BM25 Retriever: {} results in {:.2}s",
            bm25_results.len(),
            bm25_time.as_secs_f64()
        );

        println!();

        // Test fusion retrievers
        println!("🔀 Fusion Retriever Performance:");
        println!("--------------------------------");

        // RRF fusion
        let timer = Timer::new("RRF fusion");
        let rrf_results = self.fusion_retriever_rrf.retrieve(&query).await?;
        let rrf_time = timer.finish();
        println!(
            "⚡ RRF Fusion: {} results in {:.2}s",
            rrf_results.len(),
            rrf_time.as_secs_f64()
        );

        // Weighted average fusion
        let timer = Timer::new("Weighted fusion");
        let weighted_results = self.fusion_retriever_weighted.retrieve(&query).await?;
        let weighted_time = timer.finish();
        println!(
            "⚖️  Weighted Fusion: {} results in {:.2}s",
            weighted_results.len(),
            weighted_time.as_secs_f64()
        );

        println!();

        // Generate response using fusion retriever
        println!("💬 Query Engine Response (using RRF fusion):");
        println!("--------------------------------------------");
        let timer = Timer::new("Query engine");
        let response = self.query_engine.query(&self.args.query).await?;
        let response_time = timer.finish();

        println!("✅ Response: {}", response.content());
        println!("⏱️  Total time: {:.2}s", response_time.as_secs_f64());
        println!("📄 Source nodes: {}", response.response.source_nodes.len());

        Ok(())
    }

    /// Run mode comparison
    async fn run_mode_comparison(&self) -> ExampleResult<()> {
        println!("🔀 Fusion Mode Comparison");
        println!("========================");
        println!();

        let query = Query::new(&self.args.query).with_top_k(self.args.top_k);

        let fusion_modes = vec![
            ("RRF Fusion", self.fusion_retriever_rrf.clone()),
            ("Weighted Average", self.fusion_retriever_weighted.clone()),
        ];

        for (mode_name, retriever) in fusion_modes {
            println!("🧪 Testing: {}", mode_name);
            println!("{}", "-".repeat(40));

            let timer = Timer::new(mode_name);
            let results = retriever.retrieve(&query).await?;
            let time = timer.finish();

            println!("   📊 Results: {}", results.len());
            println!("   ⏱️  Time: {:.3}s", time.as_secs_f64());

            if !results.is_empty() {
                let avg_score = results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32;
                let max_score = results.iter().map(|r| r.score).fold(0.0f32, f32::max);
                let min_score = results
                    .iter()
                    .map(|r| r.score)
                    .fold(f32::INFINITY, f32::min);

                println!(
                    "   🎯 Score stats: avg={:.3}, max={:.3}, min={:.3}",
                    avg_score, max_score, min_score
                );
            }

            println!();
        }

        Ok(())
    }

    /// Run interactive mode
    async fn run_interactive(&self) -> ExampleResult<()> {
        println!("🎮 Interactive Query Fusion Mode");
        println!("===============================");
        println!("Enter queries to test different retrieval strategies.");
        println!("Type 'quit' to exit.");
        println!();

        loop {
            print!("🔍 Enter your query: ");
            use std::io::{self, Write};
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let query_text = input.trim();

            if query_text.is_empty() {
                continue;
            }

            if query_text.to_lowercase() == "quit" {
                break;
            }

            println!("\n🚀 Processing query: {}", query_text);
            println!("{}", "─".repeat(50));

            let query = Query::new(query_text).with_top_k(self.args.top_k);

            // Test all retrieval strategies
            let strategies = vec![
                (
                    "Vector Only",
                    self.vector_retriever.clone() as Arc<dyn Retriever>,
                ),
                (
                    "BM25 Only",
                    self.bm25_retriever.clone() as Arc<dyn Retriever>,
                ),
                (
                    "RRF Fusion",
                    self.fusion_retriever_rrf.clone() as Arc<dyn Retriever>,
                ),
                (
                    "Weighted Fusion",
                    self.fusion_retriever_weighted.clone() as Arc<dyn Retriever>,
                ),
            ];

            for (strategy_name, retriever) in strategies {
                let timer = Timer::new(strategy_name);
                match retriever.retrieve(&query).await {
                    Ok(results) => {
                        let time = timer.finish();
                        println!(
                            "✅ {}: {} results in {:.3}s",
                            strategy_name,
                            results.len(),
                            time.as_secs_f64()
                        );
                    }
                    Err(e) => {
                        println!("❌ {}: Error - {}", strategy_name, e);
                    }
                }
            }

            println!("\n{}\n", "═".repeat(50));
        }

        Ok(())
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
