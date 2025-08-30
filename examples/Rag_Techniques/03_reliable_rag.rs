//! Reliable RAG Example
//!
//! This example demonstrates how to build a reliable RAG system with quality assurance,
//! confidence scoring, and fallback mechanisms. It shows how to:
//! - Implement confidence scoring for retrieved content
//! - Add quality checks for generated responses
//! - Provide fallback mechanisms when confidence is low
//! - Track and report reliability metrics
//! - Handle edge cases gracefully
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin reliable_rag --features fastembed
//! cargo run --bin reliable_rag --features fastembed -- --confidence-threshold 0.8 --interactive
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
    engine::QueryEngine, generator::SiumaiGenerator, prelude::QueryResponse,
    retriever::VectorRetriever,
};
use siumai::prelude::*;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug)]
#[command(name = "reliable_rag")]
#[command(about = "Reliable RAG Example - RAG with quality assurance and confidence scoring")]
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

    /// Minimum confidence threshold for reliable answers
    #[arg(long, default_value_t = 0.7)]
    confidence_threshold: f32,

    /// Minimum similarity score for retrieved content
    #[arg(long, default_value_t = 0.6)]
    similarity_threshold: f32,

    /// Run in interactive mode
    #[arg(long)]
    interactive: bool,
}

/// Reliability metrics for tracking system performance
#[derive(Debug, Default)]
struct ReliabilityMetrics {
    pub total_queries: usize,
    pub high_confidence_responses: usize,
    pub low_confidence_responses: usize,
    pub fallback_responses: usize,
    pub avg_confidence_score: f32,
    pub avg_similarity_score: f32,
}

impl ReliabilityMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_query(
        &mut self,
        confidence: f32,
        similarity: f32,
        is_fallback: bool,
        confidence_threshold: f32,
    ) {
        self.total_queries += 1;

        if is_fallback {
            self.fallback_responses += 1;
        } else if confidence >= confidence_threshold {
            self.high_confidence_responses += 1;
        } else {
            self.low_confidence_responses += 1;
        }

        // Update running averages
        let n = self.total_queries as f32;
        self.avg_confidence_score = ((self.avg_confidence_score * (n - 1.0)) + confidence) / n;
        self.avg_similarity_score = ((self.avg_similarity_score * (n - 1.0)) + similarity) / n;
    }

    pub fn print_summary(&self) {
        println!("\n📊 Reliability Summary");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📈 Total Queries: {}", self.total_queries);
        println!(
            "✅ High Confidence: {} ({:.1}%)",
            self.high_confidence_responses,
            (self.high_confidence_responses as f32 / self.total_queries as f32) * 100.0
        );
        println!(
            "⚠️  Low Confidence: {} ({:.1}%)",
            self.low_confidence_responses,
            (self.low_confidence_responses as f32 / self.total_queries as f32) * 100.0
        );
        println!(
            "🔄 Fallback Used: {} ({:.1}%)",
            self.fallback_responses,
            (self.fallback_responses as f32 / self.total_queries as f32) * 100.0
        );
        println!("📊 Avg Confidence: {:.3}", self.avg_confidence_score);
        println!("📊 Avg Similarity: {:.3}", self.avg_similarity_score);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

/// Enhanced response with reliability information
#[derive(Debug)]
struct ReliableResponse {
    pub content: String,
    pub confidence_score: f32,
    pub max_similarity_score: f32,
    pub is_reliable: bool,
    pub is_fallback: bool,
    pub quality_checks: Vec<String>,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    // Setup logging
    setup_logging();

    let args = Args::parse();

    println!("🚀 Starting Reliable RAG Example...");

    // Print configuration
    print_config(&args);

    let mut metrics = PerformanceMetrics::new();
    let mut reliability_metrics = ReliabilityMetrics::new();

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

    metrics.record_indexing_time(indexing_time);
    metrics.total_documents = indexing_stats.documents_processed;
    metrics.total_nodes = indexing_stats.nodes_created;

    println!(
        "✅ Completed: Document indexing in {:.2}s",
        indexing_time.as_secs_f64()
    );
    println!("📊 Indexing completed:");
    println!("  📚 Documents: {}", indexing_stats.documents_processed);
    println!("  🔗 Nodes: {}", indexing_stats.nodes_created);
    println!("  ⏱️  Time: {:.2}s", indexing_time.as_secs_f64());

    // Step 4: Create reliable query engine
    let retriever = Arc::new(VectorRetriever::new(vector_store, embedder));

    // Create LLM client - try OpenAI first, fallback to Ollama
    let llm_client = create_llm_client().await?;
    let generator = Arc::new(SiumaiGenerator::new(llm_client));

    let query_engine = QueryEngine::new(retriever, generator);

    println!("✅ Reliable query engine initialized");
    println!("🎯 Confidence threshold: {:.2}", args.confidence_threshold);
    println!("🎯 Similarity threshold: {:.2}", args.similarity_threshold);
    println!();

    // Step 5: Run queries with reliability checks
    if args.interactive {
        run_interactive_mode(&query_engine, &mut metrics, &mut reliability_metrics, &args).await?;
    } else {
        run_demo_queries(&query_engine, &mut metrics, &mut reliability_metrics, &args).await?;
    }

    // Print final metrics
    metrics.print_summary();
    reliability_metrics.print_summary();

    Ok(())
}

fn print_config(args: &Args) {
    println!("🛡️  Reliable RAG Example");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📄 Document: {}", args.document_path.display());
    println!("🔤 Embedding Provider: {}", args.embedding_provider);
    println!(
        "📏 Chunk Size: {} (overlap: {})",
        args.chunk_size, args.chunk_overlap
    );
    println!("🔍 Top-K: {}", args.top_k);
    println!("🎯 Confidence Threshold: {:.2}", args.confidence_threshold);
    println!("🎯 Similarity Threshold: {:.2}", args.similarity_threshold);
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

/// Perform a reliable query with confidence scoring and quality checks
async fn reliable_query(
    query_engine: &QueryEngine,
    query: &str,
    args: &Args,
) -> ExampleResult<ReliableResponse> {
    // Perform the query
    let response = query_engine
        .query(query)
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;

    // Extract similarity scores from retrieved context
    let max_similarity = response
        .retrieved_nodes
        .iter()
        .map(|node| node.score)
        .fold(0.0f32, |a, b| a.max(b));

    // Calculate confidence score based on multiple factors
    let confidence_score = calculate_confidence_score(&response, max_similarity);

    // Perform quality checks
    let quality_checks = perform_quality_checks(&response, max_similarity, args);

    // Determine if response is reliable
    let is_reliable = confidence_score >= args.confidence_threshold
        && max_similarity >= args.similarity_threshold
        && quality_checks.is_empty();

    // Generate fallback response if needed
    let (final_content, is_fallback) = if !is_reliable {
        let fallback = generate_fallback_response(query, &quality_checks);
        (fallback, true)
    } else {
        (response.response.content, false)
    };

    Ok(ReliableResponse {
        content: final_content,
        confidence_score,
        max_similarity_score: max_similarity,
        is_reliable,
        is_fallback,
        quality_checks,
    })
}

/// Calculate confidence score based on multiple factors
fn calculate_confidence_score(response: &QueryResponse, max_similarity: f32) -> f32 {
    let mut confidence = 0.0f32;

    // Factor 1: Maximum similarity score (40% weight)
    confidence += max_similarity * 0.4;

    // Factor 2: Number of retrieved contexts (20% weight)
    let context_factor = (response.retrieved_nodes.len() as f32 / 5.0).min(1.0);
    confidence += context_factor * 0.2;

    // Factor 3: Response length (reasonable length indicates good content) (20% weight)
    let response_length = response.response.content.len() as f32;
    let length_factor = if response_length < 50.0 {
        response_length / 50.0
    } else if response_length > 2000.0 {
        1.0 - ((response_length - 2000.0) / 1000.0).min(0.5)
    } else {
        1.0
    };
    confidence += length_factor * 0.2;

    // Factor 4: Consistency across retrieved contexts (20% weight)
    let consistency_factor = calculate_consistency_factor(&response.retrieved_nodes);
    confidence += consistency_factor * 0.2;

    confidence.min(1.0)
}

/// Calculate consistency factor based on similarity scores distribution
fn calculate_consistency_factor(context_nodes: &[cheungfun_core::ScoredNode]) -> f32 {
    if context_nodes.len() < 2 {
        return 1.0;
    }

    let scores: Vec<f32> = context_nodes.iter().map(|node| node.score).collect();
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    let variance = scores
        .iter()
        .map(|score| (score - mean).powi(2))
        .sum::<f32>()
        / scores.len() as f32;
    let std_dev = variance.sqrt();

    // Lower standard deviation means more consistent scores
    (1.0 - std_dev).max(0.0)
}

/// Perform quality checks on the response
fn perform_quality_checks(
    response: &QueryResponse,
    max_similarity: f32,
    args: &Args,
) -> Vec<String> {
    let mut issues = Vec::new();

    // Check 1: Minimum similarity threshold
    if max_similarity < args.similarity_threshold {
        issues.push(format!(
            "Low similarity score: {:.3} < {:.3}",
            max_similarity, args.similarity_threshold
        ));
    }

    // Check 2: Response length
    if response.response.content.len() < 20 {
        issues.push("Response too short".to_string());
    }

    // Check 3: Generic responses
    let generic_phrases = [
        "I don't know",
        "I cannot answer",
        "The provided context does not",
        "I don't have enough information",
    ];

    for phrase in &generic_phrases {
        if response
            .response
            .content
            .to_lowercase()
            .contains(&phrase.to_lowercase())
        {
            issues.push("Generic or uncertain response detected".to_string());
            break;
        }
    }

    // Check 4: Number of retrieved contexts
    if response.retrieved_nodes.len() < 2 {
        issues.push("Insufficient context retrieved".to_string());
    }

    issues
}

/// Generate a fallback response when confidence is low
fn generate_fallback_response(query: &str, quality_issues: &[String]) -> String {
    let mut fallback = format!(
        "I'm not confident enough to provide a reliable answer to: \"{}\"\n\n",
        query
    );

    fallback.push_str("Quality concerns identified:\n");
    for (i, issue) in quality_issues.iter().enumerate() {
        fallback.push_str(&format!("  {}. {}\n", i + 1, issue));
    }

    fallback.push_str("\n💡 Suggestions:\n");
    fallback.push_str("  • Try rephrasing your question\n");
    fallback.push_str("  • Ask for more specific information\n");
    fallback.push_str("  • Check if the topic is covered in the source documents\n");

    fallback
}

async fn run_demo_queries(
    query_engine: &QueryEngine,
    metrics: &mut PerformanceMetrics,
    reliability_metrics: &mut ReliabilityMetrics,
    args: &Args,
) -> ExampleResult<()> {
    println!("🔍 Running reliable demo queries...");
    println!();

    let queries = get_climate_test_queries();

    for query in queries {
        let timer = Timer::new(&format!("Reliable Query: {}", query));

        let reliable_response = reliable_query(query_engine, query, args).await?;

        let query_time = timer.finish();
        metrics.record_query(query_time);
        reliability_metrics.record_query(
            reliable_response.confidence_score,
            reliable_response.max_similarity_score,
            reliable_response.is_fallback,
            args.confidence_threshold,
        );

        print_reliable_results(query, &reliable_response, args);
    }

    Ok(())
}

async fn run_interactive_mode(
    query_engine: &QueryEngine,
    metrics: &mut PerformanceMetrics,
    reliability_metrics: &mut ReliabilityMetrics,
    args: &Args,
) -> ExampleResult<()> {
    println!("🎯 Interactive Reliable RAG Mode");
    println!("Type your questions, or 'quit' to exit.");
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

        let timer = Timer::new("Reliable Query processing");

        match reliable_query(query_engine, query, args).await {
            Ok(reliable_response) => {
                let query_time = timer.finish();
                metrics.record_query(query_time);
                reliability_metrics.record_query(
                    reliable_response.confidence_score,
                    reliable_response.max_similarity_score,
                    reliable_response.is_fallback,
                    args.confidence_threshold,
                );
                print_reliable_results(query, &reliable_response, args);
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

fn print_reliable_results(query: &str, response: &ReliableResponse, args: &Args) {
    println!("\n🔍 Query: {}", query);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Reliability indicators
    let reliability_icon = if response.is_fallback {
        "🔄"
    } else if response.is_reliable {
        "✅"
    } else {
        "⚠️"
    };

    println!(
        "{} Reliability: {} (Confidence: {:.3}, Similarity: {:.3})",
        reliability_icon,
        if response.is_fallback {
            "Fallback"
        } else if response.is_reliable {
            "High"
        } else {
            "Low"
        },
        response.confidence_score,
        response.max_similarity_score
    );

    // Quality issues
    if !response.quality_checks.is_empty() {
        println!("⚠️  Quality Issues:");
        for issue in &response.quality_checks {
            println!("   • {}", issue);
        }
    }

    println!();
    println!("📝 Response: {}", response.content);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
