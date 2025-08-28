//! Query Transformations Example
//!
//! This example demonstrates advanced query transformation techniques to improve RAG retrieval:
//! - **Query Rewriting**: Reformulate queries for better retrieval effectiveness
//! - **Step-back Prompting**: Generate broader queries for better context retrieval
//! - **Sub-query Decomposition**: Break complex queries into simpler sub-queries
//! - **Query Expansion**: Add related terms and synonyms to improve coverage
//! - **Multi-perspective Queries**: Generate multiple viewpoints of the same question
//!
//! These techniques help overcome the limitations of direct query-document matching
//! by creating more effective search queries that better align with document content.
//!
//! ## Usage
//!
//! ```bash
//! # Run with all transformation techniques
//! cargo run --bin query_transformations --features fastembed
//! 
//! # Run with specific transformation technique
//! cargo run --bin query_transformations --features fastembed -- --technique rewrite
//! cargo run --bin query_transformations --features fastembed -- --technique stepback
//! cargo run --bin query_transformations --features fastembed -- --technique decompose
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
    Timer, PerformanceMetrics,
    get_climate_test_queries, setup_logging,
    ExampleResult, ExampleError,
    constants::*,
};
use std::{path::PathBuf, sync::Arc};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline},
    DistanceMetric,
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
    prelude::QueryResponse,
};
use siumai::prelude::*;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug)]
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

    /// Run in interactive mode
    #[arg(long)]
    interactive: bool,

    /// Show detailed transformation process
    #[arg(long)]
    verbose: bool,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum TransformationTechnique {
    /// Query rewriting for better retrieval
    Rewrite,
    /// Step-back prompting for broader context
    Stepback,
    /// Sub-query decomposition for complex queries
    Decompose,
    /// Query expansion with related terms
    Expand,
    /// Multi-perspective query generation
    Multiperspective,
    /// All techniques combined
    All,
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
        println!("🎯 Generated {} transformed queries", self.transformed_queries.len());
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
        let best_transform = self.all_responses
            .iter()
            .max_by(|a, b| {
                let score_a = a.1.retrieved_nodes.iter().map(|n| n.score).fold(0.0f32, |acc, s| acc.max(s));
                let score_b = b.1.retrieved_nodes.iter().map(|n| n.score).fold(0.0f32, |acc, s| acc.max(s));
                score_a.partial_cmp(&score_b).unwrap()
            });
            
        if let Some((best_tq, best_resp)) = best_transform {
            let best_score = best_resp.retrieved_nodes.iter().map(|n| n.score).fold(0.0f32, |acc, s| acc.max(s));
            println!("🏆 Best Performing Transformation:");
            println!("   Technique: {}", best_tq.technique);
            println!("   Query: {}", best_tq.transformed_query);
            println!("   Similarity Score: {:.3}", best_score);
            println!("   Improvement: {:.1}%", self.performance_metrics.improvement_over_original * 100.0);
        }
        
        println!("\n📊 Performance Metrics:");
        println!("   ⏱️  Avg Transformation Time: {:.0}ms", self.performance_metrics.avg_transformation_time.as_millis());
        println!("   🔍 Avg Retrieval Time: {:.0}ms", self.performance_metrics.avg_retrieval_time.as_millis());
        println!("   🎯 Best Similarity Score: {:.3}", self.performance_metrics.best_similarity_score);
        println!("   📈 Overall Improvement: {:.1}%", self.performance_metrics.improvement_over_original * 100.0);
        
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
    println!("📏 Chunk Size: {} (overlap: {})", args.chunk_size, args.chunk_overlap);
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

async fn create_query_engine(args: &Args, embedder: Arc<dyn Embedder>) -> ExampleResult<QueryEngine> {
    // Create vector store
    let vector_store = Arc::new(InMemoryVectorStore::new(DEFAULT_EMBEDDING_DIM, DistanceMetric::Cosine));

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

/// Transform a query using the specified technique
async fn transform_query(
    original_query: &str,
    technique: &TransformationTechnique,
    llm_client: &Siumai,
) -> ExampleResult<Vec<TransformedQuery>> {
    match technique {
        TransformationTechnique::Rewrite => query_rewrite(original_query, llm_client).await,
        TransformationTechnique::Stepback => step_back_prompting(original_query, llm_client).await,
        TransformationTechnique::Decompose => sub_query_decomposition(original_query, llm_client).await,
        TransformationTechnique::Expand => query_expansion(original_query, llm_client).await,
        TransformationTechnique::Multiperspective => multi_perspective_queries(original_query, llm_client).await,
        TransformationTechnique::All => {
            let mut all_transforms = Vec::new();

            // Apply all techniques directly to avoid recursion
            if let Ok(mut transforms) = query_rewrite(original_query, llm_client).await {
                all_transforms.append(&mut transforms);
            }
            if let Ok(mut transforms) = step_back_prompting(original_query, llm_client).await {
                all_transforms.append(&mut transforms);
            }
            if let Ok(mut transforms) = sub_query_decomposition(original_query, llm_client).await {
                all_transforms.append(&mut transforms);
            }
            if let Ok(mut transforms) = query_expansion(original_query, llm_client).await {
                all_transforms.append(&mut transforms);
            }
            if let Ok(mut transforms) = multi_perspective_queries(original_query, llm_client).await {
                all_transforms.append(&mut transforms);
            }

            Ok(all_transforms)
        }
    }
}

/// Query rewriting technique
async fn query_rewrite(original_query: &str, llm_client: &Siumai) -> ExampleResult<Vec<TransformedQuery>> {
    let prompt = format!(
        r#"You are an expert at rewriting search queries to improve retrieval effectiveness.

Original query: "{}"

Please rewrite this query in 2-3 different ways that would be more effective for document retrieval.
Focus on:
1. Using more specific terminology
2. Adding context that might be implicit
3. Reformulating to match how information might be presented in documents

For each rewritten query, provide:
- The rewritten query
- A confidence score (0.0-1.0)
- Brief reasoning for the rewrite

Format your response as JSON:
{{
  "rewrites": [
    {{
      "query": "rewritten query here",
      "confidence": 0.85,
      "reasoning": "explanation here"
    }}
  ]
}}
"#,
        original_query
    );

    let response = llm_client
        .chat(vec![ChatMessage::user(prompt).build()])
        .await
        .map_err(|e| ExampleError::Config(format!("LLM error: {}", e)))?;

    let content_str = match &response.content {
        siumai::MessageContent::Text(text) => text.clone(),
        _ => "".to_string(),
    };
    parse_transformation_response(&content_str, original_query, "Query Rewrite")
}

/// Step-back prompting technique
async fn step_back_prompting(original_query: &str, llm_client: &Siumai) -> ExampleResult<Vec<TransformedQuery>> {
    let prompt = format!(
        r#"You are an expert at creating step-back prompts for better information retrieval.

Original query: "{}"

Create 2-3 step-back queries that are broader and more general than the original query.
These should help retrieve relevant background context and foundational information.

For example:
- If asked about "effects of CO2 on ocean pH", step back to "ocean acidification" or "carbon cycle"
- If asked about "specific climate policy", step back to "climate change mitigation strategies"

For each step-back query, provide:
- The step-back query
- A confidence score (0.0-1.0)
- Brief reasoning for why this broader query helps

Format your response as JSON:
{{
  "stepbacks": [
    {{
      "query": "broader query here",
      "confidence": 0.80,
      "reasoning": "explanation here"
    }}
  ]
}}
"#,
        original_query
    );

    let response = llm_client
        .chat(vec![ChatMessage::user(prompt).build()])
        .await
        .map_err(|e| ExampleError::Config(format!("LLM error: {}", e)))?;

    let content_str = match &response.content {
        siumai::MessageContent::Text(text) => text.clone(),
        _ => "".to_string(),
    };
    parse_transformation_response(&content_str, original_query, "Step-back Prompting")
}

/// Sub-query decomposition technique
async fn sub_query_decomposition(original_query: &str, llm_client: &Siumai) -> ExampleResult<Vec<TransformedQuery>> {
    let prompt = format!(
        r#"You are an expert at breaking down complex queries into simpler sub-queries.

Original query: "{}"

If this query is complex, break it down into 2-4 simpler sub-queries that together would help answer the original question.
If the query is already simple, create related queries that would provide supporting information.

For each sub-query, provide:
- The sub-query
- A confidence score (0.0-1.0)
- Brief reasoning for how this sub-query contributes to answering the original

Format your response as JSON:
{{
  "subqueries": [
    {{
      "query": "sub-query here",
      "confidence": 0.75,
      "reasoning": "explanation here"
    }}
  ]
}}
"#,
        original_query
    );

    let response = llm_client
        .chat(vec![ChatMessage::user(prompt).build()])
        .await
        .map_err(|e| ExampleError::Config(format!("LLM error: {}", e)))?;

    let content_str = match &response.content {
        siumai::MessageContent::Text(text) => text.clone(),
        _ => "".to_string(),
    };
    parse_transformation_response(&content_str, original_query, "Sub-query Decomposition")
}

/// Query expansion technique
async fn query_expansion(original_query: &str, llm_client: &Siumai) -> ExampleResult<Vec<TransformedQuery>> {
    let prompt = format!(
        r#"You are an expert at expanding queries with related terms and synonyms.

Original query: "{}"

Create 2-3 expanded versions of this query by adding:
1. Synonyms and related terms
2. Technical terminology that might be used in documents
3. Alternative phrasings that convey the same meaning

For each expanded query, provide:
- The expanded query
- A confidence score (0.0-1.0)
- Brief reasoning for the expansion choices

Format your response as JSON:
{{
  "expansions": [
    {{
      "query": "expanded query here",
      "confidence": 0.70,
      "reasoning": "explanation here"
    }}
  ]
}}
"#,
        original_query
    );

    let response = llm_client
        .chat(vec![ChatMessage::user(prompt).build()])
        .await
        .map_err(|e| ExampleError::Config(format!("LLM error: {}", e)))?;

    let content_str = match &response.content {
        siumai::MessageContent::Text(text) => text.clone(),
        _ => "".to_string(),
    };
    parse_transformation_response(&content_str, original_query, "Query Expansion")
}

/// Multi-perspective queries technique
async fn multi_perspective_queries(original_query: &str, llm_client: &Siumai) -> ExampleResult<Vec<TransformedQuery>> {
    let prompt = format!(
        r#"You are an expert at creating multi-perspective queries for comprehensive information retrieval.

Original query: "{}"

Create 2-3 queries that approach the same topic from different perspectives or angles:
1. Different stakeholder viewpoints (scientific, policy, economic, social)
2. Different time frames (historical, current, future)
3. Different scales (local, national, global)
4. Different aspects (causes, effects, solutions)

For each perspective query, provide:
- The query from that perspective
- A confidence score (0.0-1.0)
- Brief reasoning for why this perspective is valuable

Format your response as JSON:
{{
  "perspectives": [
    {{
      "query": "perspective query here",
      "confidence": 0.80,
      "reasoning": "explanation here"
    }}
  ]
}}
"#,
        original_query
    );

    let response = llm_client
        .chat(vec![ChatMessage::user(prompt).build()])
        .await
        .map_err(|e| ExampleError::Config(format!("LLM error: {}", e)))?;

    let content_str = match &response.content {
        siumai::MessageContent::Text(text) => text.clone(),
        _ => "".to_string(),
    };
    parse_transformation_response(&content_str, original_query, "Multi-perspective Queries")
}

/// Parse LLM response into TransformedQuery objects
fn parse_transformation_response(
    response_content: &str,
    original_query: &str,
    technique: &str,
) -> ExampleResult<Vec<TransformedQuery>> {
    // Try to extract JSON from the response
    let json_start = response_content.find('{');
    let json_end = response_content.rfind('}');

    if let (Some(start), Some(end)) = (json_start, json_end) {
        let json_str = &response_content[start..=end];

        // Try to parse as JSON
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
            let mut transformed_queries = Vec::new();

            // Handle different response formats
            let queries_array = parsed.get("rewrites")
                .or_else(|| parsed.get("stepbacks"))
                .or_else(|| parsed.get("subqueries"))
                .or_else(|| parsed.get("expansions"))
                .or_else(|| parsed.get("perspectives"));

            if let Some(queries) = queries_array.and_then(|v| v.as_array()) {
                for query_obj in queries {
                    if let (Some(query), Some(confidence), Some(reasoning)) = (
                        query_obj.get("query").and_then(|v| v.as_str()),
                        query_obj.get("confidence").and_then(|v| v.as_f64()),
                        query_obj.get("reasoning").and_then(|v| v.as_str()),
                    ) {
                        transformed_queries.push(TransformedQuery {
                            original_query: original_query.to_string(),
                            transformed_query: query.to_string(),
                            technique: technique.to_string(),
                            confidence: confidence as f32,
                            reasoning: reasoning.to_string(),
                        });
                    }
                }
            }

            if !transformed_queries.is_empty() {
                return Ok(transformed_queries);
            }
        }
    }

    // Fallback: create a simple transformation if JSON parsing fails
    Ok(vec![TransformedQuery {
        original_query: original_query.to_string(),
        transformed_query: original_query.to_string(), // Use original as fallback
        technique: format!("{} (fallback)", technique),
        confidence: 0.5,
        reasoning: "JSON parsing failed, using original query".to_string(),
    }])
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
    let technique = args.technique.as_ref().unwrap_or(&TransformationTechnique::All);

    for (i, query) in queries.iter().enumerate() {
        println!("🧪 Demo Query {}/{}: {}", i + 1, queries.len(), query);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let timer = Timer::new("Query transformation and retrieval");

        let results = perform_query_transformation(
            query,
            technique,
            query_engine,
            llm_client,
            args,
        ).await?;

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
    println!("Available techniques: rewrite, stepback, decompose, expand, multiperspective, all");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let mut current_technique = args.technique.as_ref().unwrap_or(&TransformationTechnique::All).clone();

    loop {
        println!("Current technique: {:?}", current_technique);
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
                "rewrite" => current_technique = TransformationTechnique::Rewrite,
                "stepback" => current_technique = TransformationTechnique::Stepback,
                "decompose" => current_technique = TransformationTechnique::Decompose,
                "expand" => current_technique = TransformationTechnique::Expand,
                "multiperspective" => current_technique = TransformationTechnique::Multiperspective,
                "all" => current_technique = TransformationTechnique::All,
                _ => {
                    println!("❌ Unknown technique. Available: rewrite, stepback, decompose, expand, multiperspective, all");
                    continue;
                }
            }
            println!("✅ Technique changed to: {:?}", current_technique);
            continue;
        }

        let timer = Timer::new("Query transformation and retrieval");

        match perform_query_transformation(
            input,
            &current_technique,
            query_engine,
            llm_client,
            args,
        ).await {
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
    let transformed_queries = transform_query(original_query, technique, llm_client).await?;
    let transformation_time = transform_timer.finish();

    if args.verbose {
        println!("🔄 Generated {} transformed queries in {:.0}ms",
            transformed_queries.len(), transformation_time.as_millis());
    }

    // Step 2: Execute original query for baseline
    let original_timer = Timer::new("Original query");
    let original_response = query_engine
        .query(original_query)
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;
    let original_time = original_timer.finish();

    let original_score = original_response.retrieved_nodes
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

                let max_score = response.retrieved_nodes
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
                println!("⚠️  Failed to execute transformed query '{}': {}", tq.transformed_query, e);
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
