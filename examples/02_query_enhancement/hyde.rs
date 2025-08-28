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
    engine::QueryEngine, generator::SiumaiGenerator, prelude::QueryResponse,
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
    /// Generate a single hypothetical document
    Single,
    /// Generate multiple hypothetical documents and combine
    Multiple,
    /// Iterative refinement of hypothetical documents
    Iterative,
    /// Adaptive strategy based on query complexity
    Adaptive,
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

/// Generate hypothetical documents using the specified strategy
async fn generate_hypothetical_documents(
    query: &str,
    strategy: &HydeStrategy,
    llm_client: &Siumai,
) -> ExampleResult<Vec<HypotheticalDocument>> {
    match strategy {
        HydeStrategy::Single => generate_single_hypothetical_document(query, llm_client).await,
        HydeStrategy::Multiple => generate_multiple_hypothetical_documents(query, llm_client).await,
        HydeStrategy::Iterative => {
            generate_iterative_hypothetical_documents(query, llm_client).await
        }
        HydeStrategy::Adaptive => generate_adaptive_hypothetical_documents(query, llm_client).await,
    }
}

/// Generate a single hypothetical document
async fn generate_single_hypothetical_document(
    query: &str,
    llm_client: &Siumai,
) -> ExampleResult<Vec<HypotheticalDocument>> {
    let timer = Timer::new("Single hypothetical document generation");

    let prompt = format!(
        r#"You are an expert writer tasked with creating a hypothetical document that would contain the answer to the following question.

Question: "{}"

Please write a comprehensive, informative document that would naturally contain the answer to this question. The document should:
1. Be written in a formal, informative style similar to academic or reference materials
2. Include relevant background information and context
3. Provide detailed explanations and examples
4. Be approximately 200-400 words long
5. Focus on factual, well-structured content

Write only the document content, without any meta-commentary or explanations about the task.
"#,
        query
    );

    let response = llm_client
        .chat(vec![ChatMessage::user(prompt).build()])
        .await
        .map_err(|e| ExampleError::Config(format!("LLM error: {}", e)))?;

    let generation_time = timer.finish();

    let content = match &response.content {
        siumai::MessageContent::Text(text) => text.clone(),
        _ => "".to_string(),
    };

    Ok(vec![HypotheticalDocument {
        original_query: query.to_string(),
        hypothetical_content: content,
        strategy: "Single".to_string(),
        confidence: 0.8, // Default confidence for single document
        generation_time,
    }])
}

/// Generate multiple hypothetical documents with different perspectives
async fn generate_multiple_hypothetical_documents(
    query: &str,
    llm_client: &Siumai,
) -> ExampleResult<Vec<HypotheticalDocument>> {
    let timer = Timer::new("Multiple hypothetical documents generation");

    let perspectives = [
        ("Scientific", "Write from a scientific research perspective with technical details and evidence"),
        ("Educational", "Write from an educational perspective suitable for students and general learning"),
        ("Practical", "Write from a practical perspective focusing on real-world applications and implications"),
    ];

    let mut hypothetical_docs = Vec::new();
    let start_time = std::time::Instant::now();

    for (perspective_name, perspective_instruction) in &perspectives {
        let prompt = format!(
            r#"You are an expert writer creating a hypothetical document to answer the following question.

Question: "{}"

Instructions: {}

Please write a comprehensive document (200-300 words) that would naturally contain the answer to this question. Focus on providing accurate, detailed information in a well-structured format.

Write only the document content, without any meta-commentary.
"#,
            query, perspective_instruction
        );

        let response = llm_client
            .chat(vec![ChatMessage::user(prompt).build()])
            .await
            .map_err(|e| ExampleError::Config(format!("LLM error: {}", e)))?;

        let content = match &response.content {
            siumai::MessageContent::Text(text) => text.clone(),
            _ => "".to_string(),
        };

        let current_time = start_time.elapsed();
        hypothetical_docs.push(HypotheticalDocument {
            original_query: query.to_string(),
            hypothetical_content: content,
            strategy: format!("Multiple-{}", perspective_name),
            confidence: 0.75, // Slightly lower confidence for multiple docs
            generation_time: current_time,
        });
    }

    let _total_time = timer.finish();
    Ok(hypothetical_docs)
}

/// Generate hypothetical documents with iterative refinement
async fn generate_iterative_hypothetical_documents(
    query: &str,
    llm_client: &Siumai,
) -> ExampleResult<Vec<HypotheticalDocument>> {
    let timer = Timer::new("Iterative hypothetical documents generation");

    // First iteration: Basic document
    let initial_prompt = format!(
        r#"Create a comprehensive document that answers this question: "{}"

Write a well-structured, informative document (200-300 words) that provides a complete answer.
"#,
        query
    );

    let initial_response = llm_client
        .chat(vec![ChatMessage::user(initial_prompt).build()])
        .await
        .map_err(|e| ExampleError::Config(format!("LLM error: {}", e)))?;

    let initial_content = match &initial_response.content {
        siumai::MessageContent::Text(text) => text.clone(),
        _ => "".to_string(),
    };

    // Second iteration: Refined document
    let refinement_prompt = format!(
        r#"Here is a document that answers the question: "{}"

Document:
{}

Please create an improved version of this document that:
1. Adds more specific details and examples
2. Improves the structure and flow
3. Includes additional relevant information
4. Maintains the same approximate length (200-300 words)

Write only the improved document content.
"#,
        query, initial_content
    );

    let refined_response = llm_client
        .chat(vec![ChatMessage::user(refinement_prompt).build()])
        .await
        .map_err(|e| ExampleError::Config(format!("LLM error: {}", e)))?;

    let refined_content = match &refined_response.content {
        siumai::MessageContent::Text(text) => text.clone(),
        _ => "".to_string(),
    };

    let generation_time = timer.finish();

    Ok(vec![
        HypotheticalDocument {
            original_query: query.to_string(),
            hypothetical_content: initial_content,
            strategy: "Iterative-Initial".to_string(),
            confidence: 0.7,
            generation_time: generation_time / 2, // Approximate split
        },
        HypotheticalDocument {
            original_query: query.to_string(),
            hypothetical_content: refined_content,
            strategy: "Iterative-Refined".to_string(),
            confidence: 0.85, // Higher confidence for refined version
            generation_time: generation_time / 2,
        },
    ])
}

/// Generate hypothetical documents using adaptive strategy based on query complexity
async fn generate_adaptive_hypothetical_documents(
    query: &str,
    llm_client: &Siumai,
) -> ExampleResult<Vec<HypotheticalDocument>> {
    // Analyze query complexity
    let query_complexity = analyze_query_complexity(query);

    match query_complexity {
        QueryComplexity::Simple => generate_single_hypothetical_document(query, llm_client).await,
        QueryComplexity::Moderate => {
            generate_multiple_hypothetical_documents(query, llm_client).await
        }
        QueryComplexity::Complex => {
            generate_iterative_hypothetical_documents(query, llm_client).await
        }
    }
}

#[derive(Debug)]
enum QueryComplexity {
    Simple,
    Moderate,
    Complex,
}

/// Analyze query complexity based on various factors
fn analyze_query_complexity(query: &str) -> QueryComplexity {
    let word_count = query.split_whitespace().count();
    let has_multiple_questions = query.matches('?').count() > 1;
    let has_complex_terms = query.contains("how") && query.contains("why");
    let has_comparisons =
        query.contains("compare") || query.contains("difference") || query.contains("versus");

    if word_count > 15 || has_multiple_questions || has_complex_terms || has_comparisons {
        QueryComplexity::Complex
    } else if word_count > 8 || query.contains("explain") || query.contains("describe") {
        QueryComplexity::Moderate
    } else {
        QueryComplexity::Simple
    }
}

/// Combine multiple hypothetical documents for enhanced retrieval
async fn combine_multiple_hypothetical_retrievals(
    _original_query: &str,
    hypothetical_docs: &[HypotheticalDocument],
    query_engine: &QueryEngine,
) -> ExampleResult<QueryResponse> {
    // Strategy 1: Use the highest confidence hypothetical document
    let best_doc = hypothetical_docs
        .iter()
        .max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    // Strategy 2: For multiple documents, we could also try:
    // - Concatenating all hypothetical documents
    // - Using each document separately and merging results
    // - Weighted combination based on confidence scores

    // For now, we'll use the best document but also try a combined approach
    if hypothetical_docs.len() >= 3 {
        // For 3+ documents, create a combined hypothetical document
        let combined_content = create_combined_hypothetical_document(hypothetical_docs);

        // Try both the best individual document and the combined document
        let best_response = query_engine
            .query(&best_doc.hypothetical_content)
            .await
            .map_err(ExampleError::Cheungfun)?;

        let combined_response = query_engine
            .query(&combined_content)
            .await
            .map_err(ExampleError::Cheungfun)?;

        // Choose the response with higher average similarity score
        let best_avg_score = best_response
            .retrieved_nodes
            .iter()
            .map(|n| n.score)
            .sum::<f32>()
            / best_response.retrieved_nodes.len() as f32;

        let combined_avg_score = combined_response
            .retrieved_nodes
            .iter()
            .map(|n| n.score)
            .sum::<f32>()
            / combined_response.retrieved_nodes.len() as f32;

        if combined_avg_score > best_avg_score {
            Ok(combined_response)
        } else {
            Ok(best_response)
        }
    } else {
        // For 2 documents, just use the best one
        query_engine
            .query(&best_doc.hypothetical_content)
            .await
            .map_err(ExampleError::Cheungfun)
    }
}

/// Create a combined hypothetical document from multiple documents
fn create_combined_hypothetical_document(hypothetical_docs: &[HypotheticalDocument]) -> String {
    // Sort by confidence (highest first)
    let mut sorted_docs = hypothetical_docs.to_vec();
    sorted_docs.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take the top 3 documents and combine them intelligently
    let top_docs: Vec<_> = sorted_docs.iter().take(3).collect();

    let mut combined = String::new();
    combined.push_str("Comprehensive Analysis:\n\n");

    for (i, doc) in top_docs.iter().enumerate() {
        combined.push_str(&format!(
            "Perspective {} ({} approach):\n",
            i + 1,
            doc.strategy
        ));

        // Take the first 200 words of each document to avoid too much repetition
        let words: Vec<&str> = doc.hypothetical_content.split_whitespace().collect();
        let excerpt = if words.len() > 200 {
            format!("{}...", words[..200].join(" "))
        } else {
            doc.hypothetical_content.clone()
        };

        combined.push_str(&excerpt);
        combined.push_str("\n\n");
    }

    combined.push_str("This comprehensive analysis incorporates multiple perspectives to provide a thorough understanding of the topic.");

    combined
}

/// Perform HyDE-enhanced retrieval
async fn perform_hyde_retrieval(
    query: &str,
    strategy: &HydeStrategy,
    query_engine: &QueryEngine,
    llm_client: &Siumai,
    compare_baseline: bool,
) -> ExampleResult<HydeResults> {
    // Step 1: Generate hypothetical documents
    let generation_timer = Timer::new("HyDE generation");
    let hypothetical_docs = generate_hypothetical_documents(query, strategy, llm_client).await?;
    let total_generation_time = generation_timer.finish();

    // Step 2: Perform retrieval using hypothetical documents
    let retrieval_timer = Timer::new("HyDE retrieval");

    let hyde_response = if hypothetical_docs.is_empty() {
        // Fallback to original query if no hypothetical documents generated
        query_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?
    } else if hypothetical_docs.len() == 1 {
        // Single hypothetical document - use it directly
        query_engine
            .query(&hypothetical_docs[0].hypothetical_content)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?
    } else {
        // Multiple hypothetical documents - combine them intelligently
        combine_multiple_hypothetical_retrievals(query, &hypothetical_docs, query_engine).await?
    };

    let total_retrieval_time = retrieval_timer.finish();

    // Step 3: Get baseline response if requested
    let baseline_response = if compare_baseline {
        Some(
            query_engine
                .query(query)
                .await
                .map_err(|e| ExampleError::Cheungfun(e))?,
        )
    } else {
        None
    };

    // Step 4: Calculate performance metrics
    let hyde_similarity = hyde_response
        .retrieved_nodes
        .iter()
        .map(|node| node.score)
        .fold(0.0f32, |a, b| a.max(b));

    let baseline_similarity = baseline_response
        .as_ref()
        .map(|resp| {
            resp.retrieved_nodes
                .iter()
                .map(|node| node.score)
                .fold(0.0f32, |a, b| a.max(b))
        })
        .unwrap_or(0.0);

    let improvement_percentage = if baseline_similarity > 0.0 {
        ((hyde_similarity - baseline_similarity) / baseline_similarity) * 100.0
    } else {
        0.0
    };

    let performance_metrics = HydeMetrics {
        total_generation_time,
        total_retrieval_time,
        hyde_similarity_score: hyde_similarity,
        baseline_similarity_score: baseline_similarity,
        improvement_percentage,
        num_hypothetical_docs: hypothetical_docs.len(),
    };

    Ok(HydeResults {
        original_query: query.to_string(),
        hypothetical_documents: hypothetical_docs,
        hyde_response,
        baseline_response,
        performance_metrics,
    })
}

/// Run HyDE experiments on demo queries
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

        let timer = Timer::new("HyDE processing");

        let results = perform_hyde_retrieval(
            query,
            &args.strategy,
            query_engine,
            llm_client,
            args.compare_baseline,
        )
        .await?;

        let total_time = timer.finish();
        metrics.record_query(total_time);

        results.print_summary(args.verbose);
        println!();
    }

    Ok(())
}

/// Run interactive mode with HyDE
async fn run_interactive_mode(
    query_engine: &QueryEngine,
    llm_client: &Siumai,
    args: &Args,
    metrics: &mut PerformanceMetrics,
) -> ExampleResult<()> {
    println!("🎯 Interactive HyDE Mode");
    println!("Type your questions, or 'quit' to exit.");
    println!("Use 'strategy <name>' to change HyDE strategy.");
    println!("Available strategies: single, multiple, iterative, adaptive");
    println!("Use 'baseline on/off' to toggle baseline comparison.");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let mut current_strategy = args.strategy.clone();
    let mut compare_baseline = args.compare_baseline;

    loop {
        println!(
            "Current strategy: {:?} | Baseline comparison: {}",
            current_strategy, compare_baseline
        );
        print!("❓ Your question (or command): ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.to_lowercase() == "quit" {
            break;
        }

        // Handle strategy change commands
        if input.starts_with("strategy ") {
            let strategy_name = input.strip_prefix("strategy ").unwrap().trim();
            match strategy_name.to_lowercase().as_str() {
                "single" => current_strategy = HydeStrategy::Single,
                "multiple" => current_strategy = HydeStrategy::Multiple,
                "iterative" => current_strategy = HydeStrategy::Iterative,
                "adaptive" => current_strategy = HydeStrategy::Adaptive,
                _ => {
                    println!(
                        "❌ Unknown strategy. Available: single, multiple, iterative, adaptive"
                    );
                    continue;
                }
            }
            println!("✅ Strategy changed to: {:?}", current_strategy);
            continue;
        }

        // Handle baseline toggle commands
        if input.starts_with("baseline ") {
            let baseline_setting = input.strip_prefix("baseline ").unwrap().trim();
            match baseline_setting.to_lowercase().as_str() {
                "on" | "true" => compare_baseline = true,
                "off" | "false" => compare_baseline = false,
                _ => {
                    println!("❌ Use 'baseline on' or 'baseline off'");
                    continue;
                }
            }
            println!("✅ Baseline comparison: {}", compare_baseline);
            continue;
        }

        let timer = Timer::new("HyDE processing");

        match perform_hyde_retrieval(
            input,
            &current_strategy,
            query_engine,
            llm_client,
            compare_baseline,
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
