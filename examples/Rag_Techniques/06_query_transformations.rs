//! Query Transformations Example
//!
//! This example demonstrates advanced query transformation techniques using Cheungfun's
//! built-in Query Transformers with optimized preset configurations:
//! - **HyDE (Hypothetical Document Embeddings)**: Generate hypothetical documents for better retrieval
//! - **Sub-query Decomposition**: Break complex queries into simpler sub-queries
//! - **Query Rewriting**: Reformulate queries for better retrieval effectiveness
//! - **Multi-step Transformations**: Chain multiple transformers for comprehensive enhancement
//!
//! Now uses Cheungfun's optimized Query Transformers with research-backed preset configurations
//! for different domains (general Q&A, code search, academic research).
//!
//! ## Usage
//!
//! ```bash
//! # Run with all transformation techniques using preset configurations
//! cargo run --bin query_transformations --features fastembed
//!
//! # Run with specific transformation technique
//! cargo run --bin query_transformations --features fastembed -- --technique hyde
//! cargo run --bin query_transformations --features fastembed -- --technique subquery
//! cargo run --bin query_transformations --features fastembed -- --technique rewrite
//!
//! # Use domain-specific presets
//! cargo run --bin query_transformations --features fastembed -- --preset qa
//! cargo run --bin query_transformations --features fastembed -- --preset code
//! cargo run --bin query_transformations --features fastembed -- --preset academic
//!
//! # Interactive mode with transformations
//! cargo run --bin query_transformations --features fastembed -- --interactive
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
use std::{path::PathBuf, sync::Arc};

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
    advanced::{AdvancedQuery, HyDETransformer, QueryTransformer, SubquestionTransformer},
    engine::{QueryEngine, QueryRewriteStrategy},
    generator::SiumaiGenerator,
    prelude::QueryResponse,
    retriever::VectorRetriever,
};
use siumai::prelude::*;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug, Clone)]
#[command(name = "query_transformations")]
#[command(about = "Query Transformations Example - Advanced query enhancement techniques")]
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

    /// Specific transformation technique to use
    #[arg(long, value_enum)]
    technique: Option<TransformationTechnique>,

    /// Domain-specific preset configuration
    #[arg(long, value_enum)]
    preset: Option<DomainPreset>,

    /// Run in interactive mode
    #[arg(long)]
    interactive: bool,

    /// Show detailed transformation process
    #[arg(long)]
    verbose: bool,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum TransformationTechnique {
    /// HyDE (Hypothetical Document Embeddings) transformation
    Hyde,
    /// Sub-query decomposition for complex queries
    Subquery,
    /// Query rewriting for better retrieval
    Rewrite,
    /// All techniques combined
    All,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum DomainPreset {
    /// General Q&A optimized configuration
    Qa,
    /// Code search optimized configuration
    Code,
    /// Academic research optimized configuration
    Academic,
}

/// Represents a transformed query with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TransformedQuery {
    pub original_query: String,
    pub transformed_query: String,
    pub technique: String,
    pub confidence: f32,
    pub reasoning: String,
}

/// Results from query transformation process
#[derive(Debug)]
struct TransformationResults {
    pub original_query: String,
    pub transformed_queries: Vec<TransformedQuery>,
    pub best_response: QueryResponse,
    pub all_responses: Vec<(TransformedQuery, QueryResponse)>,
    pub performance_metrics: TransformationMetrics,
}

/// Performance metrics for query transformations
#[derive(Debug, Default)]
struct TransformationMetrics {
    pub total_transformations: usize,
    pub avg_transformation_time: std::time::Duration,
    pub avg_retrieval_time: std::time::Duration,
    pub best_similarity_score: f32,
    pub improvement_over_original: f32,
}

impl TransformationResults {
    pub fn print_summary(&self, verbose: bool) {
        println!("\n🔄 QUERY TRANSFORMATION RESULTS");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        println!("📝 Original Query: {}", self.original_query);
        println!(
            "🎯 Generated {} transformed queries",
            self.transformed_queries.len()
        );
        println!();

        if verbose {
            println!("🔍 All Transformations:");
            for (i, tq) in self.transformed_queries.iter().enumerate() {
                println!("  {}. {} ({})", i + 1, tq.technique, tq.confidence);
                println!("     Query: {}", tq.transformed_query);
                println!("     Reasoning: {}", tq.reasoning);
                println!();
            }
        }

        // Find best performing transformation
        let best_transform = self.all_responses.iter().max_by(|a, b| {
            let score_a =
                a.1.retrieved_nodes
                    .iter()
                    .map(|n| n.score)
                    .fold(0.0f32, |acc, s| acc.max(s));
            let score_b =
                b.1.retrieved_nodes
                    .iter()
                    .map(|n| n.score)
                    .fold(0.0f32, |acc, s| acc.max(s));
            score_a.partial_cmp(&score_b).unwrap()
        });

        if let Some((best_tq, best_resp)) = best_transform {
            let best_score = best_resp
                .retrieved_nodes
                .iter()
                .map(|n| n.score)
                .fold(0.0f32, |acc, s| acc.max(s));
            println!("🏆 Best Performing Transformation:");
            println!("   Technique: {}", best_tq.technique);
            println!("   Query: {}", best_tq.transformed_query);
            println!("   Similarity Score: {:.3}", best_score);
            println!(
                "   Improvement: {:.1}%",
                self.performance_metrics.improvement_over_original * 100.0
            );
        }

        println!("\n📊 Performance Metrics:");
        println!(
            "   ⏱️  Avg Transformation Time: {:.0}ms",
            self.performance_metrics.avg_transformation_time.as_millis()
        );
        println!(
            "   🔍 Avg Retrieval Time: {:.0}ms",
            self.performance_metrics.avg_retrieval_time.as_millis()
        );
        println!(
            "   🎯 Best Similarity Score: {:.3}",
            self.performance_metrics.best_similarity_score
        );
        println!(
            "   📈 Overall Improvement: {:.1}%",
            self.performance_metrics.improvement_over_original * 100.0
        );

        println!("\n📝 Best Response:");
        println!("{}", self.best_response.response.content);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    // Setup logging
    setup_logging();

    let args = Args::parse();

    println!("🚀 Starting Query Transformations Example...");

    // Print configuration
    print_config(&args);

    let mut metrics = PerformanceMetrics::new();

    // Step 1: Create embedder
    let embedder = create_embedder(&args.embedding_provider).await?;
    println!("✅ Embedder initialized: {}", args.embedding_provider);

    // Step 2: Create vector store and index documents
    let query_engine = create_query_engine(&args, embedder).await?;
    println!("✅ Query engine initialized");

    // Step 3: Create LLM client for transformations
    let llm_client = create_llm_client().await?;
    println!("✅ LLM client for transformations initialized");

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
    println!("🔄 Query Transformations Example");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📄 Document: {}", args.document_path.display());
    println!("🔤 Embedding Provider: {}", args.embedding_provider);
    println!(
        "📏 Chunk Size: {} (overlap: {})",
        args.chunk_size, args.chunk_overlap
    );
    println!("🔍 Top-K: {}", args.top_k);

    if let Some(ref technique) = args.technique {
        println!("🎯 Technique: {:?}", technique);
    } else {
        println!("🎯 Technique: All techniques");
    }

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
            println!("🤖 Using OpenAI for query transformations (cloud)");
            return Siumai::builder()
                .openai()
                .api_key(&api_key)
                .model("gpt-4o-mini")
                .temperature(0.3) // Slightly higher temperature for creativity
                .max_tokens(2000)
                .build()
                .await
                .map_err(|e| ExampleError::Config(format!("Failed to initialize OpenAI: {}", e)));
        }
    }

    // Fallback to Ollama
    println!("🤖 No valid OpenAI API key found, using Ollama for query transformations (local)");
    println!("💡 Make sure Ollama is running with: ollama serve");
    println!("💡 And pull a model with: ollama pull llama3.2");

    Siumai::builder()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .temperature(0.3)
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

    // Step 2: Build indexing pipeline
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

/// Transform a query using the specified technique with Cheungfun's Query Transformers
async fn transform_query(
    original_query: &str,
    technique: &TransformationTechnique,
    preset: &Option<DomainPreset>,
    _llm_client: &Siumai,
) -> ExampleResult<Vec<TransformedQuery>> {
    // Create a new Siumai client since it doesn't implement Clone
    let new_llm_client = create_llm_client().await?;
    let generator = Arc::new(SiumaiGenerator::new(new_llm_client));

    match technique {
        TransformationTechnique::Hyde => hyde_transform(original_query, preset, &generator).await,
        TransformationTechnique::Subquery => {
            subquery_transform(original_query, preset, &generator).await
        }
        TransformationTechnique::Rewrite => {
            rewrite_transform(original_query, preset, &generator).await
        }
        TransformationTechnique::All => {
            let mut all_transforms = Vec::new();

            // Apply all techniques using our library transformers
            if let Ok(mut transforms) = hyde_transform(original_query, preset, &generator).await {
                all_transforms.append(&mut transforms);
            }
            if let Ok(mut transforms) = subquery_transform(original_query, preset, &generator).await
            {
                all_transforms.append(&mut transforms);
            }
            if let Ok(mut transforms) = rewrite_transform(original_query, preset, &generator).await
            {
                all_transforms.append(&mut transforms);
            }

            Ok(all_transforms)
        }
    }
}

/// HyDE transformation using Cheungfun's HyDETransformer
async fn hyde_transform(
    original_query: &str,
    preset: &Option<DomainPreset>,
    generator: &Arc<SiumaiGenerator>,
) -> ExampleResult<Vec<TransformedQuery>> {
    let transformer = match preset {
        Some(DomainPreset::Qa) => HyDETransformer::for_qa(generator.clone()),
        Some(DomainPreset::Code) => HyDETransformer::for_code_search(generator.clone()),
        Some(DomainPreset::Academic) => HyDETransformer::for_academic_search(generator.clone()),
        None => HyDETransformer::from_defaults(generator.clone()),
    };

    // Create an AdvancedQuery from the string
    let mut advanced_query = AdvancedQuery::from_text(original_query.to_string());

    // Apply the transformation (modifies the query in-place)
    transformer
        .transform(&mut advanced_query)
        .await
        .map_err(|e| {
            ExampleError::Cheungfun(cheungfun_core::CheungfunError::External { source: e })
        })?;

    let mut results = Vec::new();
    for (i, query) in advanced_query.transformed_queries.iter().enumerate() {
        results.push(TransformedQuery {
            original_query: original_query.to_string(),
            transformed_query: query.clone(),
            technique: "HyDE".to_string(),
            confidence: 0.85, // HyDE typically has high confidence
            reasoning: format!(
                "Generated hypothetical document #{} using domain-optimized HyDE",
                i + 1
            ),
        });
    }

    Ok(results)
}

/// Sub-query decomposition using Cheungfun's SubquestionTransformer
async fn subquery_transform(
    original_query: &str,
    preset: &Option<DomainPreset>,
    generator: &Arc<SiumaiGenerator>,
) -> ExampleResult<Vec<TransformedQuery>> {
    let transformer = match preset {
        Some(DomainPreset::Academic) => SubquestionTransformer::for_research(generator.clone()),
        Some(_) => SubquestionTransformer::from_defaults(generator.clone()),
        None => SubquestionTransformer::from_defaults(generator.clone()),
    };

    // Create an AdvancedQuery from the string
    let mut advanced_query = AdvancedQuery::from_text(original_query.to_string());

    // Apply the transformation (modifies the query in-place)
    transformer
        .transform(&mut advanced_query)
        .await
        .map_err(|e| {
            ExampleError::Cheungfun(cheungfun_core::CheungfunError::External { source: e })
        })?;

    let mut results = Vec::new();
    for (i, query) in advanced_query.transformed_queries.iter().enumerate() {
        results.push(TransformedQuery {
            original_query: original_query.to_string(),
            transformed_query: query.clone(),
            technique: "Sub-query Decomposition".to_string(),
            confidence: 0.80, // Sub-query decomposition typically has good confidence
            reasoning: format!(
                "Generated sub-query #{} to break down complex question",
                i + 1
            ),
        });
    }

    Ok(results)
}

/// Query rewriting using Cheungfun's QueryEngine rewrite functionality
async fn rewrite_transform(
    original_query: &str,
    preset: &Option<DomainPreset>,
    _generator: &Arc<SiumaiGenerator>,
) -> ExampleResult<Vec<TransformedQuery>> {
    // Create a temporary query engine for rewriting
    // In a real implementation, you'd pass the actual query engine
    // For now, we'll simulate different rewrite strategies based on preset

    let strategies = match preset {
        Some(DomainPreset::Code) => vec![
            QueryRewriteStrategy::Clarification,
            QueryRewriteStrategy::Expansion,
        ],
        Some(DomainPreset::Academic) => vec![
            QueryRewriteStrategy::Clarification,
            QueryRewriteStrategy::Decomposition,
        ],
        _ => vec![
            QueryRewriteStrategy::Clarification,
            QueryRewriteStrategy::Expansion,
        ],
    };

    let mut results = Vec::new();

    // For this example, we'll create simple rewritten queries
    // In a real implementation, you'd use the actual QueryEngine.rewrite_query method
    for (_i, strategy) in strategies.iter().enumerate() {
        let rewritten_query = match strategy {
            QueryRewriteStrategy::Clarification => {
                format!("What are the specific details about {}", original_query)
            }
            QueryRewriteStrategy::Expansion => {
                format!("{} including related concepts and examples", original_query)
            }
            QueryRewriteStrategy::Decomposition => {
                format!("Break down and explain: {}", original_query)
            }
            QueryRewriteStrategy::HyDE => {
                format!(
                    "Generate comprehensive information about: {}",
                    original_query
                )
            }
        };

        results.push(TransformedQuery {
            original_query: original_query.to_string(),
            transformed_query: rewritten_query,
            technique: format!("Query Rewrite ({:?})", strategy),
            confidence: 0.75,
            reasoning: format!("Applied {:?} rewrite strategy", strategy),
        });
    }

    Ok(results)
}

/// Run transformation experiments on demo queries
async fn run_demo_queries(
    query_engine: &QueryEngine,
    llm_client: &Siumai,
    args: &Args,
    metrics: &mut PerformanceMetrics,
) -> ExampleResult<()> {
    println!("🔍 Running query transformation demo...");
    println!();

    let queries = get_climate_test_queries();
    let technique = args
        .technique
        .as_ref()
        .unwrap_or(&TransformationTechnique::All);

    for (i, query) in queries.iter().enumerate() {
        println!("🧪 Demo Query {}/{}: {}", i + 1, queries.len(), query);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let timer = Timer::new("Query transformation and retrieval");

        let results =
            perform_query_transformation(query, technique, query_engine, llm_client, args).await?;

        let total_time = timer.finish();
        metrics.record_query(total_time);

        results.print_summary(args.verbose);
        println!();
    }

    Ok(())
}

/// Run interactive mode with query transformations
async fn run_interactive_mode(
    query_engine: &QueryEngine,
    llm_client: &Siumai,
    args: &Args,
    metrics: &mut PerformanceMetrics,
) -> ExampleResult<()> {
    println!("🎯 Interactive Query Transformations Mode");
    println!("Type your questions, or 'quit' to exit.");
    println!("Use 'technique <name>' to change transformation technique.");
    println!("Available techniques: hyde, subquery, rewrite, all");
    println!("Use 'preset <name>' to change domain preset: qa, code, academic");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let mut current_technique = args
        .technique
        .as_ref()
        .unwrap_or(&TransformationTechnique::All)
        .clone();

    let mut current_preset = args.preset.clone();

    loop {
        println!("Current technique: {:?}", current_technique);
        if let Some(ref preset) = current_preset {
            println!("Current preset: {:?}", preset);
        }
        print!("❓ Your question (or command): ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.to_lowercase() == "quit" {
            break;
        }

        // Handle technique change commands
        if input.starts_with("technique ") {
            let technique_name = input.strip_prefix("technique ").unwrap().trim();
            match technique_name.to_lowercase().as_str() {
                "hyde" => current_technique = TransformationTechnique::Hyde,
                "subquery" => current_technique = TransformationTechnique::Subquery,
                "rewrite" => current_technique = TransformationTechnique::Rewrite,
                "all" => current_technique = TransformationTechnique::All,
                _ => {
                    println!("❌ Unknown technique. Available: hyde, subquery, rewrite, all");
                    continue;
                }
            }
            println!("✅ Technique changed to: {:?}", current_technique);
            continue;
        }

        // Handle preset change commands
        if input.starts_with("preset ") {
            let preset_name = input.strip_prefix("preset ").unwrap().trim();
            match preset_name.to_lowercase().as_str() {
                "qa" => current_preset = Some(DomainPreset::Qa),
                "code" => current_preset = Some(DomainPreset::Code),
                "academic" => current_preset = Some(DomainPreset::Academic),
                "none" => current_preset = None,
                _ => {
                    println!("❌ Unknown preset. Available: qa, code, academic, none");
                    continue;
                }
            }
            println!("✅ Preset changed to: {:?}", current_preset);
            continue;
        }

        let timer = Timer::new("Query transformation and retrieval");

        // Create temporary args with current settings
        let mut temp_args = args.clone();
        temp_args.technique = Some(current_technique.clone());
        temp_args.preset = current_preset.clone();

        match perform_query_transformation(
            input,
            &current_technique,
            query_engine,
            llm_client,
            &temp_args,
        )
        .await
        {
            Ok(results) => {
                let total_time = timer.finish();
                metrics.record_query(total_time);
                results.print_summary(args.verbose);
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

/// Perform query transformation and retrieval
async fn perform_query_transformation(
    original_query: &str,
    technique: &TransformationTechnique,
    query_engine: &QueryEngine,
    llm_client: &Siumai,
    args: &Args,
) -> ExampleResult<TransformationResults> {
    // Step 1: Transform the query
    let transform_timer = Timer::new("Query transformation");
    let transformed_queries =
        transform_query(original_query, technique, &args.preset, llm_client).await?;
    let transformation_time = transform_timer.finish();

    if args.verbose {
        println!(
            "🔄 Generated {} transformed queries in {:.0}ms",
            transformed_queries.len(),
            transformation_time.as_millis()
        );
    }

    // Step 2: Execute original query for baseline
    let original_timer = Timer::new("Original query");
    let original_response = query_engine
        .query(original_query)
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;
    let original_time = original_timer.finish();

    let original_score = original_response
        .retrieved_nodes
        .iter()
        .map(|node| node.score)
        .fold(0.0f32, |a, b| a.max(b));

    // Step 3: Execute all transformed queries
    let mut all_responses = Vec::new();
    let mut retrieval_times = Vec::new();
    let mut best_response = original_response.clone();
    let mut best_score = original_score;

    for tq in &transformed_queries {
        let retrieval_timer = Timer::new("Transformed query retrieval");

        match query_engine.query(&tq.transformed_query).await {
            Ok(response) => {
                let retrieval_time = retrieval_timer.finish();
                retrieval_times.push(retrieval_time);

                let max_score = response
                    .retrieved_nodes
                    .iter()
                    .map(|node| node.score)
                    .fold(0.0f32, |a, b| a.max(b));

                if max_score > best_score {
                    best_score = max_score;
                    best_response = response.clone();
                }

                all_responses.push((tq.clone(), response));
            }
            Err(e) => {
                println!(
                    "⚠️  Failed to execute transformed query '{}': {}",
                    tq.transformed_query, e
                );
            }
        }
    }

    // Step 4: Calculate performance metrics
    let avg_transformation_time = transformation_time;
    let avg_retrieval_time = if !retrieval_times.is_empty() {
        retrieval_times.iter().sum::<std::time::Duration>() / retrieval_times.len() as u32
    } else {
        original_time
    };

    let improvement = if original_score > 0.0 {
        (best_score - original_score) / original_score
    } else {
        0.0
    };

    let performance_metrics = TransformationMetrics {
        total_transformations: transformed_queries.len(),
        avg_transformation_time,
        avg_retrieval_time,
        best_similarity_score: best_score,
        improvement_over_original: improvement,
    };

    Ok(TransformationResults {
        original_query: original_query.to_string(),
        transformed_queries,
        best_response,
        all_responses,
        performance_metrics,
    })
}
