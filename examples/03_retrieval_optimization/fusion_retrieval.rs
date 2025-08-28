/*!
# Fusion Retrieval Example

This example demonstrates fusion retrieval techniques using cheungfun's HybridSearchStrategy
and ReciprocalRankFusion implementations. Based on LlamaIndex's QueryFusionRetriever.

## Key Features

- **HybridSearchStrategy**: Uses our library's built-in hybrid search with vector + keyword
- **ReciprocalRankFusion**: Implements the RRF algorithm from research papers
- **Multiple Fusion Methods**: RRF, Weighted Average, Linear Combination
- **Performance Comparison**: Compare hybrid vs individual retrieval methods

## How It Works

1. **Vector Search**: Semantic similarity using embeddings
2. **Keyword Search**: BM25-style keyword matching (simulated)
3. **Result Fusion**: Combine results using RRF (k=60) or other methods
4. **Score Normalization**: Handle different score scales across methods

## Fusion Methods

### Reciprocal Rank Fusion (RRF)
- Formula: `score = Σ(1 / (k + rank_i))` where k=60
- Robust to score scale differences
- Based on "Reciprocal rank fusion outperforms Condorcet" paper

### Weighted Average
- Combines normalized scores with configurable weights
- Good when you know relative quality of retrievers

### Linear Combination
- Linear combination of scores with coefficients
- More flexible than weighted average

## Usage

```bash
# Basic usage with RRF
cargo run --bin fusion_retrieval --features fastembed

# Different fusion methods
cargo run --bin fusion_retrieval --features fastembed -- --fusion-method rrf
cargo run --bin fusion_retrieval --features fastembed -- --fusion-method weighted
cargo run --bin fusion_retrieval --features fastembed -- --fusion-method linear

# Compare with individual methods
cargo run --bin fusion_retrieval --features fastembed -- --compare-individual

# Interactive mode
cargo run --bin fusion_retrieval --features fastembed -- --interactive
```
*/

use clap::Parser;
// use serde::{Deserialize, Serialize};

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{
    constants::*, get_climate_test_queries, setup_logging, ExampleError, ExampleResult,
    PerformanceMetrics, Timer,
};
use std::{path::PathBuf, sync::Arc};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, VectorStore},
    ScoredNode,
};
use cheungfun_indexing::{
    loaders::DirectoryLoader,
    node_parser::{config::SentenceSplitterConfig, text::SentenceSplitter},
    pipeline::DefaultIndexingPipeline,
    transformers::MetadataExtractor,
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    advanced::{
        search_strategies::{
            BM25Params, HybridSearchStrategy, KeywordSearchConfig, KeywordSearchStrategy,
            VectorSearchConfig, VectorSearchStrategy,
        },
        AdvancedQuery, DistanceMetric, FusionMethod, NormalizationMethod, SearchStrategy,
    },
    engine::QueryEngine,
    generator::SiumaiGenerator,
    prelude::QueryResponse,
    retriever::VectorRetriever,
};
use siumai::prelude::*;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug, Clone)]
#[command(name = "fusion_retrieval")]
#[command(
    about = "Fusion Retrieval Example - Combine multiple retrieval methods for better results"
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

    /// Number of top results to retrieve from each method
    #[arg(long, default_value_t = DEFAULT_TOP_K)]
    top_k: usize,

    /// Fusion method to use
    #[arg(long, value_enum, default_value = "rrf")]
    fusion_method: FusionMethodArg,

    /// Compare with individual retrievers
    #[arg(long)]
    compare_individual: bool,

    /// Run in interactive mode
    #[arg(long)]
    interactive: bool,

    /// Show detailed fusion process
    #[arg(long)]
    verbose: bool,

    /// RRF constant k (for RRF fusion method)
    #[arg(long, default_value_t = 60.0)]
    rrf_k: f32,

    /// Vector retriever weight (for weighted fusion)
    #[arg(long, default_value_t = 0.7)]
    vector_weight: f32,

    /// Keyword retriever weight (for weighted fusion)
    #[arg(long, default_value_t = 0.3)]
    keyword_weight: f32,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum FusionMethodArg {
    /// Reciprocal Rank Fusion (k=60)
    Rrf,
    /// Weighted Average fusion
    Weighted,
    /// Linear Combination fusion
    Linear,
}

impl FusionMethodArg {
    /// Get method name for display
    fn name(&self) -> &'static str {
        match self {
            FusionMethodArg::Rrf => "Reciprocal Rank Fusion",
            FusionMethodArg::Weighted => "Weighted Average",
            FusionMethodArg::Linear => "Linear Combination",
        }
    }
}

/// Fusion retrieval results using HybridSearchStrategy
#[derive(Debug)]
struct FusionResults {
    pub original_query: String,
    pub fusion_method: String,
    pub vector_results: Vec<ScoredNode>,
    pub keyword_results: Vec<ScoredNode>,
    pub fused_results: Vec<ScoredNode>,
    pub final_response: QueryResponse,
    pub performance_metrics: FusionMetrics,
}

/// Performance metrics for fusion retrieval
#[derive(Debug, Default)]
struct FusionMetrics {
    pub total_retrieval_time: std::time::Duration,
    pub fusion_time: std::time::Duration,
    pub generation_time: std::time::Duration,
    pub vector_count: usize,
    pub keyword_count: usize,
    pub fused_count: usize,
    pub overlap_count: usize,
    pub overlap_percentage: f32,
}

impl FusionResults {
    pub fn print_summary(&self, verbose: bool) {
        println!("\n🔀 FUSION RETRIEVAL RESULTS");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        println!("📝 Original Query: {}", self.original_query);
        println!("🎯 Fusion Method: {}", self.fusion_method);
        println!(
            "📊 Retrieved {} fused documents from vector + keyword search",
            self.performance_metrics.fused_count
        );
        println!();

        if verbose {
            println!("🔍 Individual Retriever Results:");

            println!("  📋 Vector: {} results", self.vector_results.len());
            for (i, scored_node) in self.vector_results.iter().take(3).enumerate() {
                println!(
                    "     {}. Score: {:.3} | {}",
                    i + 1,
                    scored_node.score,
                    if scored_node.node.content.len() > 80 {
                        format!("{}...", &scored_node.node.content[..80])
                    } else {
                        scored_node.node.content.clone()
                    }
                );
            }
            println!();

            println!("  📋 Keyword: {} results", self.keyword_results.len());
            for (i, scored_node) in self.keyword_results.iter().take(3).enumerate() {
                println!(
                    "     {}. Score: {:.3} | {}",
                    i + 1,
                    scored_node.score,
                    if scored_node.node.content.len() > 80 {
                        format!("{}...", &scored_node.node.content[..80])
                    } else {
                        scored_node.node.content.clone()
                    }
                );
            }
            println!();

            println!("🔀 Fused Results (Top 5):");
            for (i, scored_node) in self.fused_results.iter().take(5).enumerate() {
                println!(
                    "  {}. Score: {:.3} | {}",
                    i + 1,
                    scored_node.score,
                    if scored_node.node.content.len() > 100 {
                        format!("{}...", &scored_node.node.content[..100])
                    } else {
                        scored_node.node.content.clone()
                    }
                );
            }
            println!();
        }

        println!("📊 Performance Metrics:");
        println!(
            "   ⏱️  Total Time: {:.0}ms",
            self.performance_metrics.total_retrieval_time.as_millis()
        );
        println!(
            "   🔀 Fusion Time: {:.0}ms",
            self.performance_metrics.fusion_time.as_millis()
        );
        println!(
            "   💬 Generation Time: {:.0}ms",
            self.performance_metrics.generation_time.as_millis()
        );
        println!(
            "   📊 Vector Results: {}",
            self.performance_metrics.vector_count
        );
        println!(
            "   📊 Keyword Results: {}",
            self.performance_metrics.keyword_count
        );
        println!(
            "   📊 Fused Results: {}",
            self.performance_metrics.fused_count
        );
        println!(
            "   🔗 Overlap: {} documents ({:.1}%)",
            self.performance_metrics.overlap_count, self.performance_metrics.overlap_percentage
        );

        println!("\n📝 Final Response:");
        println!("{}", self.final_response.response.content);

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    // Setup logging
    setup_logging();

    let args = Args::parse();

    println!("🚀 Starting Fusion Retrieval Example...");

    // Print configuration
    print_config(&args);

    let mut metrics = PerformanceMetrics::new();

    // Step 1: Create embedder
    let embedder = create_embedder(&args.embedding_provider).await?;
    println!("✅ Embedder initialized: {}", args.embedding_provider);

    // Step 2: Create vector store and query engine
    let (vector_store, query_engine) = create_retrievers_and_engine(&args, embedder).await?;
    println!("✅ Vector store and query engine initialized");

    if args.interactive {
        run_interactive_mode(vector_store.as_ref(), &query_engine, &args, &mut metrics).await?;
    } else {
        run_demo_queries(vector_store.as_ref(), &query_engine, &args, &mut metrics).await?;
    }

    // Print final metrics
    metrics.print_summary();

    Ok(())
}

fn print_config(args: &Args) {
    println!("🔀 Fusion Retrieval Example");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📄 Document: {}", args.document_path.display());
    println!("🔤 Embedding Provider: {}", args.embedding_provider);
    println!(
        "📏 Chunk Size: {} (overlap: {})",
        args.chunk_size, args.chunk_overlap
    );
    println!("🔍 Top-K per method: {}", args.top_k);
    println!("🎯 Fusion Method: {:?}", args.fusion_method);
    println!("📊 Compare Individual: {}", args.compare_individual);
    println!("🔍 Verbose: {}", args.verbose);

    match args.fusion_method {
        FusionMethodArg::Rrf => println!("⚙️  RRF K: {}", args.rrf_k),
        FusionMethodArg::Weighted => println!(
            "⚙️  Weights: Vector={}, Keyword={}",
            args.vector_weight, args.keyword_weight
        ),
        _ => {}
    }

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
            println!("🤖 Using OpenAI for generation (cloud)");
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

async fn create_retrievers_and_engine(
    args: &Args,
    embedder: Arc<dyn Embedder>,
) -> ExampleResult<(Arc<dyn VectorStore>, QueryEngine)> {
    // Create vector store and index documents
    let vector_store = Arc::new(InMemoryVectorStore::new(
        DEFAULT_EMBEDDING_DIM,
        cheungfun_core::DistanceMetric::Cosine,
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
        .with_transformer(splitter)
        .with_transformer(metadata_extractor)
        .with_embedder(embedder.clone())
        .with_vector_store(vector_store.clone())
        .build()?;

    // Run indexing pipeline with progress reporting
    let indexing_stats = pipeline
        .run_with_progress(Box::new(|progress| {
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
        }))
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

    // Create vector retriever for query engine
    let vector_retriever = Arc::new(VectorRetriever::new(vector_store.clone(), embedder.clone()));

    // Create query engine with vector retriever
    let llm_client = create_llm_client().await?;
    let generator = Arc::new(SiumaiGenerator::new(llm_client));
    let query_engine = QueryEngine::new(vector_retriever, generator);

    Ok((vector_store, query_engine))
}

/// Perform fusion retrieval using HybridSearchStrategy
async fn perform_fusion_retrieval(
    query: &str,
    args: &Args,
    vector_store: &dyn VectorStore,
    query_engine: &QueryEngine,
) -> ExampleResult<FusionResults> {
    let timer = Timer::new("Total fusion retrieval");

    // Create hybrid search strategy using new preset API
    let hybrid_strategy = match args.fusion_method {
        FusionMethodArg::Rrf => HybridSearchStrategy::for_general_qa()
            .fusion_method(FusionMethod::ReciprocalRankFusion { k: args.rrf_k })
            .vector_weight(args.vector_weight)
            .keyword_weight(args.keyword_weight)
            .top_k(args.top_k)
            .build(),
        FusionMethodArg::Weighted => HybridSearchStrategy::for_academic_papers()
            .fusion_method(FusionMethod::WeightedAverage)
            .vector_weight(args.vector_weight)
            .keyword_weight(args.keyword_weight)
            .top_k(args.top_k)
            .build(),
        FusionMethodArg::Linear => HybridSearchStrategy::for_code_search()
            .fusion_method(FusionMethod::LinearCombination)
            .vector_weight(args.vector_weight)
            .keyword_weight(args.keyword_weight)
            .top_k(args.top_k)
            .build(),
    };

    // Create advanced query
    let advanced_query = AdvancedQuery::from_text(query.to_string());

    // Perform hybrid search
    let search_timer = Timer::new("Hybrid search");
    let fused_results = hybrid_strategy
        .search(&advanced_query, vector_store)
        .await
        .map_err(|e| ExampleError::DataProcessing(format!("Hybrid search failed: {}", e)))?;
    let search_time = search_timer.finish();

    // For comparison, create individual strategies to get separate results
    let vector_config = VectorSearchConfig {
        vector_field: "embedding".to_string(),
        distance_metric: DistanceMetric::Cosine,
        search_params: std::collections::HashMap::new(),
        pre_filter: None,
    };

    let keyword_config = KeywordSearchConfig {
        search_fields: vec!["content".to_string()],
        analyzer: Some("standard".to_string()),
        bm25_params: Some(BM25Params { k1: 1.2, b: 0.75 }),
        minimum_should_match: None,
    };

    let vector_timer = Timer::new("Vector search");
    let vector_strategy = VectorSearchStrategy::new(vector_config);
    let vector_results = vector_strategy
        .search(&advanced_query, vector_store)
        .await
        .map_err(|e| ExampleError::DataProcessing(format!("Vector search failed: {}", e)))?;
    let vector_time = vector_timer.finish();

    let keyword_timer = Timer::new("Keyword search");
    let keyword_strategy = KeywordSearchStrategy::new(keyword_config);
    let keyword_results = keyword_strategy
        .search(&advanced_query, vector_store)
        .await
        .map_err(|e| ExampleError::DataProcessing(format!("Keyword search failed: {}", e)))?;
    let keyword_time = keyword_timer.finish();

    // Generate final response
    let generation_timer = Timer::new("Response generation");
    let final_response = generate_response_from_nodes(query, &fused_results, query_engine).await?;
    let generation_time = generation_timer.finish();

    let total_time = timer.finish();

    // Calculate metrics
    let overlap_count = calculate_overlap(&vector_results, &keyword_results);
    let overlap_percentage = if vector_results.len() + keyword_results.len() > 0 {
        (overlap_count as f32 * 2.0) / (vector_results.len() + keyword_results.len()) as f32 * 100.0
    } else {
        0.0
    };

    let mut performance_metrics = FusionMetrics {
        total_retrieval_time: total_time,
        fusion_time: search_time - vector_time - keyword_time,
        generation_time: std::time::Duration::from_millis(0), // Will be updated after generation
        vector_count: vector_results.len(),
        keyword_count: keyword_results.len(),
        fused_count: fused_results.len(),
        overlap_count,
        overlap_percentage,
    };

    // Generate final response using query engine
    let generation_timer = Timer::new("Response generation");
    let final_response = generate_response_from_nodes(query, &fused_results, query_engine).await?;
    let generation_time = generation_timer.finish();
    performance_metrics.generation_time = generation_time;

    Ok(FusionResults {
        original_query: query.to_string(),
        fusion_method: args.fusion_method.name().to_string(),
        vector_results,
        keyword_results,
        fused_results,
        final_response,
        performance_metrics,
    })
}

/// Generate response from retrieved nodes using query engine
async fn generate_response_from_nodes(
    query: &str,
    nodes: &[ScoredNode],
    query_engine: &QueryEngine,
) -> ExampleResult<QueryResponse> {
    // Create a simple query with the retrieved nodes
    // In a real implementation, you might want to use the query engine more directly
    let response = query_engine
        .query(query)
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;

    // Replace the retrieved nodes with our fusion results
    Ok(QueryResponse {
        response: response.response,
        retrieved_nodes: nodes.to_vec(),
        query_metadata: response.query_metadata,
    })
}

/// Calculate overlap between two result sets
fn calculate_overlap(results1: &[ScoredNode], results2: &[ScoredNode]) -> usize {
    let ids1: std::collections::HashSet<_> = results1.iter().map(|n| &n.node.id).collect();
    let ids2: std::collections::HashSet<_> = results2.iter().map(|n| &n.node.id).collect();
    ids1.intersection(&ids2).count()
}

/// Run fusion retrieval experiments on demo queries
async fn run_demo_queries(
    vector_store: &dyn VectorStore,
    query_engine: &QueryEngine,
    args: &Args,
    metrics: &mut PerformanceMetrics,
) -> ExampleResult<()> {
    println!("🔀 Running Fusion Retrieval demo queries...");
    println!();

    let queries = get_climate_test_queries();

    for (i, query) in queries.iter().enumerate() {
        println!("🧪 Demo Query {}/{}: {}", i + 1, queries.len(), query);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let timer = Timer::new("Fusion retrieval processing");

        let results = perform_fusion_retrieval(query, args, vector_store, &query_engine).await?;

        let total_time = timer.finish();
        metrics.record_query(total_time);

        results.print_summary(args.verbose);

        // Compare with individual retrievers if requested
        if args.compare_individual {
            println!("\n📊 Individual Retriever Comparison:");
            println!(
                "  🔍 Vector: {} results, best score: {:.3}",
                results.vector_results.len(),
                results
                    .vector_results
                    .first()
                    .map(|n| n.score)
                    .unwrap_or(0.0)
            );
            println!(
                "  🔍 Keyword: {} results, best score: {:.3}",
                results.keyword_results.len(),
                results
                    .keyword_results
                    .first()
                    .map(|n| n.score)
                    .unwrap_or(0.0)
            );
            println!(
                "  🔀 Fused: {} results, best score: {:.3}",
                results.fused_results.len(),
                results
                    .fused_results
                    .first()
                    .map(|n| n.score)
                    .unwrap_or(0.0)
            );
        }

        println!();
    }

    Ok(())
}

/// Run interactive mode with fusion retrieval
async fn run_interactive_mode(
    vector_store: &dyn VectorStore,
    query_engine: &QueryEngine,
    args: &Args,
    metrics: &mut PerformanceMetrics,
) -> ExampleResult<()> {
    println!("🎯 Interactive Fusion Retrieval Mode");
    println!("Type your questions, or 'quit' to exit.");
    println!("Use 'fusion <method>' to change fusion method.");
    println!("Available methods: rrf, weighted, rank-based, adaptive");
    println!("Use 'weights <vector> <keyword>' to adjust fusion weights.");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let mut current_args = args.clone();

    loop {
        println!("Current fusion method: {:?}", current_args.fusion_method);
        if matches!(current_args.fusion_method, FusionMethodArg::Weighted) {
            println!(
                "Current weights: Vector={:.1}, Keyword={:.1}",
                current_args.vector_weight, current_args.keyword_weight
            );
        }
        print!("❓ Your question (or command): ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.to_lowercase() == "quit" {
            break;
        }

        // Handle fusion method change commands
        if input.starts_with("fusion ") {
            let method_name = input.strip_prefix("fusion ").unwrap().trim();
            match method_name.to_lowercase().as_str() {
                "rrf" => current_args.fusion_method = FusionMethodArg::Rrf,
                "weighted" => current_args.fusion_method = FusionMethodArg::Weighted,
                "linear" => current_args.fusion_method = FusionMethodArg::Linear,
                _ => {
                    println!("❌ Unknown fusion method. Available: rrf, weighted, linear");
                    continue;
                }
            }
            println!(
                "✅ Fusion method changed to: {:?}",
                current_args.fusion_method
            );
            continue;
        }

        // Handle weight adjustment commands
        if input.starts_with("weights ") {
            let weights_str = input.strip_prefix("weights ").unwrap().trim();
            let weights: Vec<&str> = weights_str.split_whitespace().collect();
            if weights.len() == 2 {
                if let (Ok(vector_weight), Ok(keyword_weight)) =
                    (weights[0].parse::<f32>(), weights[1].parse::<f32>())
                {
                    current_args.vector_weight = vector_weight;
                    current_args.keyword_weight = keyword_weight;
                    println!(
                        "✅ Weights updated: Vector={:.1}, Keyword={:.1}",
                        vector_weight, keyword_weight
                    );
                } else {
                    println!(
                        "❌ Invalid weight values. Use: weights <vector_weight> <keyword_weight>"
                    );
                }
            } else {
                println!("❌ Use: weights <vector_weight> <keyword_weight>");
            }
            continue;
        }

        let timer = Timer::new("Fusion retrieval processing");

        match perform_fusion_retrieval(input, &current_args, vector_store, &query_engine).await {
            Ok(results) => {
                let total_time = timer.finish();
                metrics.record_query(total_time);
                results.print_summary(current_args.verbose);
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
