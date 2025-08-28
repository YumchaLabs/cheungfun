//! Fusion Retrieval Example
//!
//! This example demonstrates fusion retrieval techniques that combine multiple retrieval methods
//! to improve both precision and recall in RAG systems. Fusion retrieval works by:
//!
//! 1. **Multiple Retrieval Methods**: Using different retrieval approaches (vector, keyword, hybrid)
//! 2. **Result Fusion**: Combining results using algorithms like RRF (Reciprocal Rank Fusion)
//! 3. **Score Normalization**: Normalizing scores across different retrieval methods
//! 4. **Rank Aggregation**: Intelligently merging ranked lists from different retrievers
//!
//! ## Fusion Algorithms
//!
//! ### Reciprocal Rank Fusion (RRF)
//! - Combines rankings by taking reciprocal of ranks
//! - Formula: `score = Σ(1 / (k + rank_i))` where k is a constant (usually 60)
//! - Robust to score scale differences between retrievers
//!
//! ### Weighted Score Fusion
//! - Combines normalized scores with learned weights
//! - Allows fine-tuning of retriever importance
//! - Better for when you have quality metrics for each retriever
//!
//! ### Rank-based Fusion
//! - Uses only ranking information, ignoring raw scores
//! - More robust to score distribution differences
//! - Good for combining very different retrieval methods
//!
//! ## Benefits
//!
//! - **Improved Recall**: Different methods find different relevant documents
//! - **Better Precision**: Fusion can filter out false positives
//! - **Robustness**: Less dependent on any single retrieval method
//! - **Adaptability**: Can adjust to different query types and domains
//!
//! ## Usage
//!
//! ```bash
//! # Run with default RRF fusion
//! cargo run --bin fusion_retrieval --features fastembed
//! 
//! # Run with specific fusion method
//! cargo run --bin fusion_retrieval --features fastembed -- --fusion-method rrf
//! cargo run --bin fusion_retrieval --features fastembed -- --fusion-method weighted
//! cargo run --bin fusion_retrieval --features fastembed -- --fusion-method rank-based
//! 
//! # Compare with individual retrievers
//! cargo run --bin fusion_retrieval --features fastembed -- --compare-individual
//! 
//! # Interactive mode
//! cargo run --bin fusion_retrieval --features fastembed -- --interactive
//! 
//! # Verbose output showing fusion process
//! cargo run --bin fusion_retrieval --features fastembed -- --verbose
//! ```

use clap::Parser;
// use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{
    Timer, PerformanceMetrics,
    get_climate_test_queries, setup_logging,
    ExampleResult, ExampleError,
    constants::*,
};
use std::{path::PathBuf, sync::Arc};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, Retriever, VectorStore},
    DistanceMetric, ScoredNode,
};
use cheungfun_indexing::{
    loaders::DirectoryLoader,
    node_parser::{text::SentenceSplitter, config::SentenceSplitterConfig},
    pipeline::DefaultIndexingPipeline,
    transformers::MetadataExtractor,
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    engine::QueryEngine,
    generator::SiumaiGenerator,
    retriever::VectorRetriever,
    prelude::{QueryResponse, Query},
};
use siumai::prelude::*;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug, Clone)]
#[command(name = "fusion_retrieval")]
#[command(about = "Fusion Retrieval Example - Combine multiple retrieval methods for better results")]
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
    fusion_method: FusionMethod,

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
enum FusionMethod {
    /// Reciprocal Rank Fusion
    Rrf,
    /// Weighted score fusion
    Weighted,
    /// Rank-based fusion
    RankBased,
    /// Adaptive fusion based on query characteristics
    Adaptive,
}

/// Individual retriever results
#[derive(Debug, Clone)]
struct RetrieverResults {
    pub method_name: String,
    pub nodes: Vec<ScoredNode>,
    pub retrieval_time: std::time::Duration,
}

/// Fused retrieval results
#[derive(Debug)]
struct FusionResults {
    pub original_query: String,
    pub fusion_method: String,
    pub individual_results: Vec<RetrieverResults>,
    pub fused_nodes: Vec<ScoredNode>,
    pub final_response: QueryResponse,
    pub performance_metrics: FusionMetrics,
}

/// Performance metrics for fusion retrieval
#[derive(Debug, Default)]
struct FusionMetrics {
    pub total_retrieval_time: std::time::Duration,
    pub fusion_time: std::time::Duration,
    pub generation_time: std::time::Duration,
    pub vector_retrieval_time: std::time::Duration,
    pub keyword_retrieval_time: std::time::Duration,
    pub num_unique_documents: usize,
    pub fusion_score: f32,
    pub individual_scores: HashMap<String, f32>,
    pub overlap_percentage: f32,
}

impl FusionResults {
    pub fn print_summary(&self, verbose: bool) {
        println!("\n🔀 FUSION RETRIEVAL RESULTS");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        println!("📝 Original Query: {}", self.original_query);
        println!("🎯 Fusion Method: {}", self.fusion_method);
        println!("📊 Retrieved {} unique documents from {} methods", 
            self.performance_metrics.num_unique_documents,
            self.individual_results.len()
        );
        println!();
        
        if verbose {
            println!("🔍 Individual Retriever Results:");
            for result in &self.individual_results {
                println!("  📋 {}: {} results in {:.0}ms", 
                    result.method_name, 
                    result.nodes.len(),
                    result.retrieval_time.as_millis()
                );
                
                if let Some(score) = self.performance_metrics.individual_scores.get(&result.method_name) {
                    println!("     Best Score: {:.3}", score);
                }
                
                // Show top 3 results
                for (i, scored_node) in result.nodes.iter().take(3).enumerate() {
                    println!("     {}. Score: {:.3} | {}",
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
            }
            
            println!("🔀 Fused Results (Top 5):");
            for (i, scored_node) in self.fused_nodes.iter().take(5).enumerate() {
                println!("  {}. Score: {:.3} | {}",
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
        println!("   ⏱️  Vector Retrieval: {:.0}ms", self.performance_metrics.vector_retrieval_time.as_millis());
        println!("   ⏱️  Keyword Retrieval: {:.0}ms", self.performance_metrics.keyword_retrieval_time.as_millis());
        println!("   🔀 Fusion Time: {:.0}ms", self.performance_metrics.fusion_time.as_millis());
        println!("   💬 Generation Time: {:.0}ms", self.performance_metrics.generation_time.as_millis());
        println!("   🎯 Fusion Score: {:.3}", self.performance_metrics.fusion_score);
        println!("   📈 Result Overlap: {:.1}%", self.performance_metrics.overlap_percentage);
        
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

    // Step 2: Create retrievers and query engine
    let (vector_retriever, keyword_retriever, query_engine) = create_retrievers_and_engine(&args, embedder).await?;
    println!("✅ Retrievers and query engine initialized");

    if args.interactive {
        run_interactive_mode(&vector_retriever, &keyword_retriever, &query_engine, &args, &mut metrics).await?;
    } else {
        run_demo_queries(&vector_retriever, &keyword_retriever, &query_engine, &args, &mut metrics).await?;
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
    println!("📏 Chunk Size: {} (overlap: {})", args.chunk_size, args.chunk_overlap);
    println!("🔍 Top-K per method: {}", args.top_k);
    println!("🎯 Fusion Method: {:?}", args.fusion_method);
    println!("📊 Compare Individual: {}", args.compare_individual);
    println!("🔍 Verbose: {}", args.verbose);
    
    match args.fusion_method {
        FusionMethod::Rrf => println!("⚙️  RRF K: {}", args.rrf_k),
        FusionMethod::Weighted => println!("⚙️  Weights: Vector={}, Keyword={}", args.vector_weight, args.keyword_weight),
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
    embedder: Arc<dyn Embedder>
) -> ExampleResult<(Arc<VectorRetriever>, Arc<KeywordRetriever>, QueryEngine)> {
    // Create vector store and index documents
    let vector_store = Arc::new(InMemoryVectorStore::new(DEFAULT_EMBEDDING_DIM, DistanceMetric::Cosine));

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
    let indexing_stats = pipeline.run_with_progress(Box::new(|progress| {
        if let Some(percentage) = progress.percentage() {
            println!("📊 {}: {:.1}% ({}/{})",
                progress.stage,
                percentage,
                progress.processed,
                progress.total.unwrap_or(0)
            );
        } else {
            println!("📊 {}: {} items processed",
                progress.stage,
                progress.processed
            );
        }

        if let Some(current_item) = &progress.current_item {
            println!("   └─ {}", current_item);
        }
    })).await?;

    let indexing_time = timer.finish();

    println!("✅ Completed: Document indexing in {:.2}s", indexing_time.as_secs_f64());
    println!("📊 Indexing completed:");
    println!("  📚 Documents: {}", indexing_stats.documents_processed);
    println!("  🔗 Nodes: {}", indexing_stats.nodes_created);
    println!("  ⏱️  Time: {:.2}s", indexing_time.as_secs_f64());

    // Create retrievers
    let vector_retriever = Arc::new(VectorRetriever::new(vector_store.clone(), embedder.clone()));

    // Create a proper keyword retriever with TF-IDF-like scoring
    let keyword_retriever = Arc::new(KeywordRetriever::new(vector_store.clone(), embedder.clone()));

    // Create query engine with vector retriever (we'll handle fusion separately)
    let llm_client = create_llm_client().await?;
    let generator = Arc::new(SiumaiGenerator::new(llm_client));
    let query_engine = QueryEngine::new(vector_retriever.clone(), generator);

    Ok((vector_retriever, keyword_retriever, query_engine))
}

/// Simple keyword retriever implementation using TF-IDF-like scoring
/// In a production system, you'd want to use a proper keyword search engine like BM25
pub struct KeywordRetriever {
    vector_store: Arc<InMemoryVectorStore>,
    embedder: Arc<dyn Embedder>,
}

impl KeywordRetriever {
    pub fn new(vector_store: Arc<InMemoryVectorStore>, embedder: Arc<dyn Embedder>) -> Self {
        Self { vector_store, embedder }
    }

    /// Perform keyword-based retrieval using a simple TF-IDF-like approach
    pub async fn retrieve(&self, query: &str, top_k: usize) -> ExampleResult<Vec<ScoredNode>> {
        // Extract query terms
        let query_terms: Vec<String> = query.to_lowercase()
            .split_whitespace()
            .filter(|term| term.len() > 2) // Filter out very short words
            .map(|s| s.to_string())
            .collect();

        if query_terms.is_empty() {
            return Ok(vec![]);
        }

        // For demonstration, we'll create a keyword-focused query by emphasizing important terms
        // In a real implementation, you'd have an inverted index and proper TF-IDF scoring

        // Create multiple keyword-focused queries
        let mut all_results = Vec::new();

        // Query 1: All terms together
        let combined_query = format!("important keywords: {}", query_terms.join(" "));
        let embedding1 = self.embedder.embed(&combined_query).await.map_err(|e| ExampleError::Config(format!("Embedding error: {}", e)))?;
        let query1 = Query::new(combined_query).with_embedding(embedding1).with_top_k(top_k);
        if let Ok(results1) = self.vector_store.search(&query1).await {
            all_results.extend(results1);
        }

        // Query 2: Individual important terms
        for term in &query_terms {
            let term_query = format!("key term: {}", term);
            let embedding = self.embedder.embed(&term_query).await.map_err(|e| ExampleError::Config(format!("Embedding error: {}", e)))?;
            let query = Query::new(term_query).with_embedding(embedding).with_top_k(top_k / 2);
            if let Ok(results) = self.vector_store.search(&query).await {
                all_results.extend(results);
            }
        }

        // Deduplicate and re-score based on keyword matching
        let mut scored_results: HashMap<String, ScoredNode> = HashMap::new();

        for scored_node in all_results {
            let node_id = format!("{}_{}", scored_node.node.id, scored_node.node.content.len());
            let content_lower = scored_node.node.content.to_lowercase();

            // Calculate keyword matching score
            let mut keyword_score = 0.0;
            let mut matched_terms = 0;

            for term in &query_terms {
                let term_count = content_lower.matches(term).count() as f32;
                if term_count > 0.0 {
                    matched_terms += 1;
                    // Simple TF-IDF-like scoring: term frequency * inverse document frequency (approximated)
                    keyword_score += term_count * (1.0 + (query_terms.len() as f32 / term_count).ln());
                }
            }

            // Boost score based on the number of matched terms
            if matched_terms > 0 {
                keyword_score *= matched_terms as f32 / query_terms.len() as f32;

                // Combine with original vector similarity score
                let combined_score = (keyword_score * 0.7) + (scored_node.score * 0.3);

                let entry = scored_results.entry(node_id).or_insert_with(|| scored_node.clone());
                if combined_score > entry.score {
                    entry.score = combined_score;
                }
            }
        }

        let mut final_results: Vec<ScoredNode> = scored_results.into_values().collect();
        final_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        final_results.truncate(top_k);

        Ok(final_results)
    }
}

/// Perform fusion retrieval combining multiple retrieval methods
async fn perform_fusion_retrieval(
    query: &str,
    vector_retriever: &VectorRetriever,
    keyword_retriever: &KeywordRetriever,
    query_engine: &QueryEngine,
    args: &Args,
) -> ExampleResult<FusionResults> {
    let total_timer = Timer::new("Total fusion retrieval");

    // Step 1: Perform individual retrievals
    let mut individual_results = Vec::new();

    // Vector retrieval
    let vector_timer = Timer::new("Vector retrieval");
    let vector_query = Query::new(query.to_string())
        .with_top_k(args.top_k);
    let vector_nodes = vector_retriever.retrieve(&vector_query).await
        .map_err(|e| ExampleError::Cheungfun(e))?;
    let vector_time = vector_timer.finish();

    individual_results.push(RetrieverResults {
        method_name: "Vector".to_string(),
        nodes: vector_nodes.clone(),
        retrieval_time: vector_time,
    });

    // Keyword retrieval using our TF-IDF-like approach
    let keyword_timer = Timer::new("Keyword retrieval");
    let keyword_nodes = keyword_retriever.retrieve(query, args.top_k).await?;
    let keyword_time = keyword_timer.finish();

    individual_results.push(RetrieverResults {
        method_name: "Keyword".to_string(),
        nodes: keyword_nodes.clone(),
        retrieval_time: keyword_time,
    });

    // Step 2: Perform fusion
    let fusion_timer = Timer::new("Result fusion");
    let fused_nodes = match args.fusion_method {
        FusionMethod::Rrf => reciprocal_rank_fusion(&individual_results, args.rrf_k),
        FusionMethod::Weighted => weighted_score_fusion(&individual_results, args.vector_weight, args.keyword_weight),
        FusionMethod::RankBased => rank_based_fusion(&individual_results),
        FusionMethod::Adaptive => adaptive_fusion(&individual_results, query),
    };
    let fusion_time = fusion_timer.finish();

    // Step 3: Generate response using fused results
    let generation_timer = Timer::new("Response generation");
    let final_response = generate_response_from_nodes(query, &fused_nodes, query_engine).await?;
    let generation_time = generation_timer.finish();

    let total_time = total_timer.finish();

    // Calculate metrics
    let mut individual_scores = HashMap::new();
    for result in &individual_results {
        if let Some(node) = result.nodes.first() {
            individual_scores.insert(result.method_name.clone(), node.score);
        }
    }

    let fusion_score = fused_nodes.first().map(|n| n.score).unwrap_or(0.0);
    let overlap_percentage = calculate_overlap_percentage(&individual_results);
    let unique_docs = count_unique_documents(&individual_results);

    let performance_metrics = FusionMetrics {
        total_retrieval_time: total_time,
        fusion_time,
        generation_time,
        vector_retrieval_time: vector_time,
        keyword_retrieval_time: keyword_time,
        num_unique_documents: unique_docs,
        fusion_score,
        individual_scores,
        overlap_percentage,
    };

    Ok(FusionResults {
        original_query: query.to_string(),
        fusion_method: format!("{:?}", args.fusion_method),
        individual_results,
        fused_nodes,
        final_response,
        performance_metrics,
    })
}

/// Reciprocal Rank Fusion (RRF) algorithm
fn reciprocal_rank_fusion(results: &[RetrieverResults], k: f32) -> Vec<ScoredNode> {
    let mut score_map: HashMap<String, f32> = HashMap::new();
    let mut node_map: HashMap<String, ScoredNode> = HashMap::new();

    for result in results {
        for (rank, scored_node) in result.nodes.iter().enumerate() {
            let node_id = format!("{}_{}", scored_node.node.id, scored_node.node.content.len()); // Simple ID generation
            let rrf_score = 1.0 / (k + rank as f32 + 1.0);

            *score_map.entry(node_id.clone()).or_insert(0.0) += rrf_score;
            node_map.entry(node_id).or_insert_with(|| scored_node.clone());
        }
    }

    let mut fused_nodes: Vec<_> = score_map.into_iter()
        .filter_map(|(id, score)| {
            node_map.get(&id).map(|scored_node| {
                let mut fused_node = scored_node.clone();
                fused_node.score = score;
                fused_node
            })
        })
        .collect();

    fused_nodes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    fused_nodes
}

/// Weighted score fusion
fn weighted_score_fusion(results: &[RetrieverResults], vector_weight: f32, keyword_weight: f32) -> Vec<ScoredNode> {
    let mut score_map: HashMap<String, f32> = HashMap::new();
    let mut node_map: HashMap<String, ScoredNode> = HashMap::new();

    let weights = vec![vector_weight, keyword_weight];

    for (i, result) in results.iter().enumerate() {
        let weight = weights.get(i).copied().unwrap_or(1.0 / results.len() as f32);

        // Normalize scores within this result set first
        let max_score = result.nodes.iter().map(|n| n.score).fold(0.0f32, f32::max);
        let min_score = result.nodes.iter().map(|n| n.score).fold(f32::INFINITY, f32::min);
        let score_range = max_score - min_score;

        for scored_node in &result.nodes {
            let node_id = format!("{}_{}", scored_node.node.id, scored_node.node.content.len());

            // Normalize score to [0, 1] range
            let normalized_score = if score_range > 0.0 {
                (scored_node.score - min_score) / score_range
            } else {
                1.0 // All scores are the same
            };

            let weighted_score = normalized_score * weight;

            *score_map.entry(node_id.clone()).or_insert(0.0) += weighted_score;
            node_map.entry(node_id).or_insert_with(|| scored_node.clone());
        }
    }

    let mut fused_nodes: Vec<_> = score_map.into_iter()
        .filter_map(|(id, score)| {
            node_map.get(&id).map(|scored_node| {
                let mut fused_node = scored_node.clone();
                fused_node.score = score;
                fused_node
            })
        })
        .collect();

    fused_nodes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    fused_nodes
}

/// Rank-based fusion (ignores scores, uses only ranks)
fn rank_based_fusion(results: &[RetrieverResults]) -> Vec<ScoredNode> {
    let mut rank_map: HashMap<String, f32> = HashMap::new();
    let mut node_map: HashMap<String, ScoredNode> = HashMap::new();

    for result in results {
        for (rank, scored_node) in result.nodes.iter().enumerate() {
            let node_id = format!("{}_{}", scored_node.node.id, scored_node.node.content.len());

            // Use reciprocal rank as score (higher rank = lower score)
            let rank_score = 1.0 / (rank as f32 + 1.0);

            *rank_map.entry(node_id.clone()).or_insert(0.0) += rank_score;
            node_map.entry(node_id).or_insert_with(|| scored_node.clone());
        }
    }

    let mut fused_nodes: Vec<_> = rank_map.into_iter()
        .filter_map(|(id, score)| {
            node_map.get(&id).map(|scored_node| {
                let mut fused_node = scored_node.clone();
                fused_node.score = score;
                fused_node
            })
        })
        .collect();

    fused_nodes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    fused_nodes
}

/// Adaptive fusion based on query characteristics
fn adaptive_fusion(results: &[RetrieverResults], query: &str) -> Vec<ScoredNode> {
    // Analyze query to determine best fusion strategy
    let query_length = query.split_whitespace().count();
    let has_specific_terms = query.contains("what") || query.contains("how") || query.contains("when");

    if query_length <= 3 && has_specific_terms {
        // Short, specific queries: favor vector search
        weighted_score_fusion(results, 0.8, 0.2)
    } else if query_length > 10 {
        // Long queries: use RRF for balanced approach
        reciprocal_rank_fusion(results, 60.0)
    } else {
        // Medium queries: balanced weights
        weighted_score_fusion(results, 0.6, 0.4)
    }
}

/// Generate response from fused nodes
async fn generate_response_from_nodes(
    query: &str,
    _nodes: &[ScoredNode],
    query_engine: &QueryEngine,
) -> ExampleResult<QueryResponse> {
    // For simplicity, we'll use the query engine's generation capability
    // In a more sophisticated implementation, you might want to create a custom context
    // from the fused nodes

    let response = query_engine.query(query).await
        .map_err(|e| ExampleError::Cheungfun(e))?;

    Ok(response)
}

/// Calculate overlap percentage between retrieval results
fn calculate_overlap_percentage(results: &[RetrieverResults]) -> f32 {
    if results.len() < 2 {
        return 0.0;
    }

    let mut all_ids = std::collections::HashSet::new();
    let mut overlapping_ids = std::collections::HashSet::new();

    // Collect all unique document IDs
    for result in results {
        for scored_node in &result.nodes {
            let node_id = format!("{}_{}", scored_node.node.id, scored_node.node.content.len());
            if all_ids.contains(&node_id) {
                overlapping_ids.insert(node_id.clone());
            }
            all_ids.insert(node_id);
        }
    }

    if all_ids.is_empty() {
        0.0
    } else {
        (overlapping_ids.len() as f32 / all_ids.len() as f32) * 100.0
    }
}

/// Count unique documents across all retrieval results
fn count_unique_documents(results: &[RetrieverResults]) -> usize {
    let mut unique_ids = std::collections::HashSet::new();

    for result in results {
        for scored_node in &result.nodes {
            let node_id = format!("{}_{}", scored_node.node.id, scored_node.node.content.len());
            unique_ids.insert(node_id);
        }
    }

    unique_ids.len()
}

/// Run fusion retrieval experiments on demo queries
async fn run_demo_queries(
    vector_retriever: &VectorRetriever,
    keyword_retriever: &KeywordRetriever,
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

        let results = perform_fusion_retrieval(
            query,
            vector_retriever,
            keyword_retriever,
            query_engine,
            args,
        ).await?;

        let total_time = timer.finish();
        metrics.record_query(total_time);

        results.print_summary(args.verbose);

        // Compare with individual retrievers if requested
        if args.compare_individual {
            println!("\n📊 Individual Retriever Comparison:");
            for individual_result in &results.individual_results {
                println!("  🔍 {}: {} results, best score: {:.3}",
                    individual_result.method_name,
                    individual_result.nodes.len(),
                    individual_result.nodes.first().map(|n| n.score).unwrap_or(0.0)
                );
            }
        }

        println!();
    }

    Ok(())
}

/// Run interactive mode with fusion retrieval
async fn run_interactive_mode(
    vector_retriever: &VectorRetriever,
    keyword_retriever: &KeywordRetriever,
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
        if matches!(current_args.fusion_method, FusionMethod::Weighted) {
            println!("Current weights: Vector={:.1}, Keyword={:.1}",
                current_args.vector_weight, current_args.keyword_weight);
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
                "rrf" => current_args.fusion_method = FusionMethod::Rrf,
                "weighted" => current_args.fusion_method = FusionMethod::Weighted,
                "rank-based" => current_args.fusion_method = FusionMethod::RankBased,
                "adaptive" => current_args.fusion_method = FusionMethod::Adaptive,
                _ => {
                    println!("❌ Unknown fusion method. Available: rrf, weighted, rank-based, adaptive");
                    continue;
                }
            }
            println!("✅ Fusion method changed to: {:?}", current_args.fusion_method);
            continue;
        }

        // Handle weight adjustment commands
        if input.starts_with("weights ") {
            let weights_str = input.strip_prefix("weights ").unwrap().trim();
            let weights: Vec<&str> = weights_str.split_whitespace().collect();
            if weights.len() == 2 {
                if let (Ok(vector_weight), Ok(keyword_weight)) = (weights[0].parse::<f32>(), weights[1].parse::<f32>()) {
                    current_args.vector_weight = vector_weight;
                    current_args.keyword_weight = keyword_weight;
                    println!("✅ Weights updated: Vector={:.1}, Keyword={:.1}", vector_weight, keyword_weight);
                } else {
                    println!("❌ Invalid weight values. Use: weights <vector_weight> <keyword_weight>");
                }
            } else {
                println!("❌ Use: weights <vector_weight> <keyword_weight>");
            }
            continue;
        }

        let timer = Timer::new("Fusion retrieval processing");

        match perform_fusion_retrieval(
            input,
            vector_retriever,
            keyword_retriever,
            query_engine,
            &current_args,
        ).await {
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
