/*!
# Self-RAG Example

This example demonstrates Self-Reflective RAG, which uses LLM-based reflection
to evaluate and improve retrieval quality through iterative refinement.

Based on: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/self_rag.ipynb

## Key Features

- **Self-Reflection**: LLM evaluates its own retrieval and generation quality
- **Iterative Refinement**: Automatically improves responses through multiple iterations
- **Quality Assessment**: Scores relevance, completeness, and accuracy
- **Adaptive Retrieval**: Adjusts retrieval strategy based on self-assessment

## How It Works

1. **Initial Retrieval**: Perform standard RAG retrieval and generation
2. **Self-Assessment**: LLM evaluates the quality of its own response
3. **Reflection Analysis**: Identify gaps, inaccuracies, or missing information
4. **Iterative Improvement**: Refine retrieval and regenerate if needed
5. **Quality Convergence**: Continue until quality threshold is met

## Usage

```bash
# Basic self-RAG
cargo run --bin self_rag --features fastembed

# Custom reflection iterations
cargo run --bin self_rag --features fastembed -- --max-iterations 5

# Detailed reflection analysis
cargo run --bin self_rag --features fastembed -- --verbose

# Interactive mode
cargo run --bin self_rag --features fastembed -- --interactive
```
*/

use clap::Parser;
use std::{path::PathBuf, sync::Arc};

#[path = "../shared/mod.rs"]
mod shared;

use shared::{get_climate_test_queries, setup_logging, ExampleError, ExampleResult, Timer};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, VectorStore},
    types::{GenerationOptions, Query, SearchMode},
    DistanceMetric, ScoredNode,
};
use cheungfun_indexing::{
    loaders::DirectoryLoader, node_parser::text::SentenceSplitter,
    pipeline::DefaultIndexingPipeline, transformers::MetadataExtractor,
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    engine::QueryEngine, generator::SiumaiGenerator, retriever::VectorRetriever,
};
use siumai::prelude::*;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug)]
#[command(name = "self_rag")]
struct Args {
    #[arg(long, default_value = "data/Understanding_Climate_Change.pdf")]
    document_path: PathBuf,
    #[arg(long, default_value = "800")]
    chunk_size: usize,
    #[arg(long, default_value = "100")]
    chunk_overlap: usize,
    #[arg(long, default_value = "10")]
    top_k: usize,
    #[arg(long, default_value = "3")]
    max_iterations: usize,
    #[arg(long, default_value = "0.8")]
    quality_threshold: f32,
    #[arg(long)]
    interactive: bool,
    #[arg(long)]
    verbose: bool,
}

#[derive(Debug, Clone)]
struct ReflectionResult {
    iteration: usize,
    response: String,
    quality_score: f32,
    reflection_feedback: String,
    retrieval_adjustments: Vec<String>,
    should_continue: bool,
}

#[derive(Debug, Clone)]
struct SelfRAGResult {
    final_response: String,
    iterations: Vec<ReflectionResult>,
    total_time: f64,
    convergence_achieved: bool,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    setup_logging();
    let args = Args::parse();

    println!("🪞 Starting Self-RAG Example...");
    println!("📖 This example demonstrates self-reflective RAG with iterative improvement");
    println!("🎯 Based on Self-RAG techniques from RAG_Techniques repository\n");

    let embedder = create_embedder().await?;
    println!("✅ Embedder initialized");

    let (vector_store, query_engine) = build_indexing_pipeline(&args, embedder.clone()).await?;

    if args.interactive {
        run_interactive_mode(&vector_store, &query_engine, &args).await?;
    } else {
        run_test_queries(&vector_store, &query_engine, &args).await?;
    }

    Ok(())
}

async fn create_embedder() -> ExampleResult<Arc<dyn Embedder>> {
    match FastEmbedder::new().await {
        Ok(embedder) => Ok(Arc::new(embedder)),
        Err(e) => Err(ExampleError::Config(format!(
            "Failed to create embedder: {}",
            e
        ))),
    }
}

async fn create_siumai_client() -> ExampleResult<Siumai> {
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        Siumai::builder()
            .openai()
            .api_key(&api_key)
            .model("gpt-3.5-turbo")
            .temperature(0.7)
            .max_tokens(1500)
            .build()
            .await
            .map_err(|e| ExampleError::Config(format!("Failed to create client: {}", e)))
    } else {
        Siumai::builder()
            .openai()
            .api_key("demo-key")
            .model("gpt-3.5-turbo")
            .build()
            .await
            .map_err(|e| ExampleError::Config(format!("Failed to create demo client: {}", e)))
    }
}

async fn build_indexing_pipeline(
    args: &Args,
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

    println!("📂 Loading from directory: {}", data_dir.display());
    let loader = Arc::new(DirectoryLoader::new(&data_dir)?);
    let splitter = Arc::new(SentenceSplitter::from_defaults(
        args.chunk_size,
        args.chunk_overlap,
    )?);
    let metadata_extractor = Arc::new(MetadataExtractor::new());
    let vector_store = Arc::new(InMemoryVectorStore::new(
        DEFAULT_EMBEDDING_DIM,
        DistanceMetric::Cosine,
    ));

    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(loader)
        .with_document_processor(splitter)  // Documents -> Nodes
        .with_node_processor(metadata_extractor)  // Nodes -> Nodes
        .with_embedder(embedder.clone())
        .with_vector_store(vector_store.clone())
        .build()?;

    let indexing_timer = Timer::new("Indexing");
    let (_nodes, index_result) = pipeline
        .run(None, None, true, true, None, true)
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;
    let indexing_time = indexing_timer.finish();

    println!(
        "✅ Indexing completed in {:.2}s",
        indexing_time.as_secs_f64()
    );
    println!("📊 Indexed {} nodes", index_result.nodes_created);

    let siumai_client = create_siumai_client().await?;
    let retriever = Arc::new(VectorRetriever::new(vector_store.clone(), embedder));
    let generator = Arc::new(SiumaiGenerator::new(siumai_client));
    let query_engine = QueryEngine::new(retriever, generator);

    Ok((vector_store, query_engine))
}

async fn perform_self_rag(
    vector_store: &Arc<dyn VectorStore>,
    query_engine: &QueryEngine,
    query: &str,
    args: &Args,
) -> ExampleResult<SelfRAGResult> {
    let total_timer = Timer::new("Self-RAG process");
    let mut iterations = Vec::new();
    let mut current_top_k = args.top_k;

    for iteration in 0..args.max_iterations {
        println!("🔄 Iteration {}/{}", iteration + 1, args.max_iterations);

        let iter_timer = Timer::new(&format!("Iteration {}", iteration + 1));

        // Perform retrieval and generation
        let search_query = Query::builder()
            .text(query.to_string())
            .top_k(current_top_k)
            .search_mode(SearchMode::Vector)
            .build();

        let retrieved_chunks = vector_store
            .search(&search_query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;

        let options = GenerationOptions::default();
        let response = query_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;

        // Self-reflection: evaluate the response quality
        let reflection =
            perform_self_reflection(query, &response.response.content, &retrieved_chunks, args)
                .await?;

        let iter_time = iter_timer.finish();

        iterations.push(ReflectionResult {
            iteration,
            response: response.response.content.clone(),
            quality_score: reflection.quality_score,
            reflection_feedback: reflection.reflection_feedback.clone(),
            retrieval_adjustments: reflection.retrieval_adjustments.clone(),
            should_continue: reflection.should_continue,
        });

        if args.verbose {
            println!("   📊 Quality score: {:.2}", reflection.quality_score);
            println!("   💭 Feedback: {}", reflection.reflection_feedback);
        }

        // Check if we should continue
        if reflection.quality_score >= args.quality_threshold || !reflection.should_continue {
            println!("✅ Convergence achieved at iteration {}", iteration + 1);
            break;
        }

        // Adjust retrieval parameters for next iteration
        if reflection
            .retrieval_adjustments
            .contains(&"increase_scope".to_string())
        {
            current_top_k = (current_top_k * 3 / 2).min(50); // Increase by 50%, max 50
            println!(
                "   🔍 Increasing retrieval scope to {} chunks",
                current_top_k
            );
        }
    }

    let total_time = total_timer.finish();
    let final_response = iterations.last().unwrap().response.clone();
    let convergence_achieved = iterations.last().unwrap().quality_score >= args.quality_threshold;

    Ok(SelfRAGResult {
        final_response,
        iterations,
        total_time: total_time.as_secs_f64(),
        convergence_achieved,
    })
}

#[derive(Debug, Clone)]
struct ReflectionFeedback {
    quality_score: f32,
    reflection_feedback: String,
    retrieval_adjustments: Vec<String>,
    should_continue: bool,
}

async fn perform_self_reflection(
    query: &str,
    response: &str,
    retrieved_chunks: &[ScoredNode],
    _args: &Args,
) -> ExampleResult<ReflectionFeedback> {
    let reflection_client = create_siumai_client().await?;

    let reflection_prompt = format!(
        r#"You are an AI assistant evaluating the quality of a RAG response. Please assess the following:

ORIGINAL QUERY: {}

GENERATED RESPONSE: {}

RETRIEVED CONTEXT: {} chunks with average score: {:.3}

Please evaluate this response on a scale of 0.0 to 1.0 and provide feedback:

1. RELEVANCE: How well does the response address the query?
2. COMPLETENESS: Does the response fully answer the question?
3. ACCURACY: Is the information factually correct based on the context?
4. CLARITY: Is the response clear and well-structured?

Respond in this exact format:
QUALITY_SCORE: [0.0-1.0]
FEEDBACK: [Your detailed feedback]
ADJUSTMENTS: [comma-separated list of adjustments like: increase_scope, refine_query, filter_context]
CONTINUE: [YES/NO]"#,
        query,
        response,
        retrieved_chunks.len(),
        if retrieved_chunks.is_empty() {
            0.0
        } else {
            retrieved_chunks.iter().map(|c| c.score).sum::<f32>() / retrieved_chunks.len() as f32
        }
    );

    let messages = vec![siumai::ChatMessage::user(&reflection_prompt).build()];
    let reflection_response = reflection_client
        .chat(messages)
        .await
        .map_err(|e| ExampleError::Config(format!("Reflection failed: {}", e)))?;

    let content_str = match &reflection_response.content {
        siumai::MessageContent::Text(text) => text.as_str(),
        _ => "No text content available",
    };

    parse_reflection_response(content_str)
}

fn parse_reflection_response(response: &str) -> ExampleResult<ReflectionFeedback> {
    let mut quality_score = 0.5; // Default
    let mut feedback = "No feedback provided".to_string();
    let mut adjustments = Vec::new();
    let mut should_continue = false;

    for line in response.lines() {
        if line.starts_with("QUALITY_SCORE:") {
            if let Ok(score) = line
                .split(':')
                .nth(1)
                .unwrap_or("0.5")
                .trim()
                .parse::<f32>()
            {
                quality_score = score.clamp(0.0, 1.0);
            }
        } else if line.starts_with("FEEDBACK:") {
            feedback = line
                .split(':')
                .skip(1)
                .collect::<Vec<_>>()
                .join(":")
                .trim()
                .to_string();
        } else if line.starts_with("ADJUSTMENTS:") {
            let adj_parts: Vec<&str> = line.split(':').skip(1).collect();
            let adj_str = adj_parts.join(":").trim().to_string();
            adjustments = adj_str.split(',').map(|s| s.trim().to_string()).collect();
        } else if line.starts_with("CONTINUE:") {
            let continue_str = line.split(':').nth(1).unwrap_or("NO").trim().to_uppercase();
            should_continue = continue_str == "YES";
        }
    }

    Ok(ReflectionFeedback {
        quality_score,
        reflection_feedback: feedback,
        retrieval_adjustments: adjustments,
        should_continue,
    })
}

async fn run_test_queries(
    vector_store: &Arc<dyn VectorStore>,
    query_engine: &QueryEngine,
    args: &Args,
) -> ExampleResult<()> {
    let test_queries = get_climate_test_queries();

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);
        println!("═══════════════════════════════════════════════════════════════");

        let self_rag_result = perform_self_rag(vector_store, query_engine, query, args).await?;

        println!("\n🎯 Final Result:");
        println!("💬 Response: {}", self_rag_result.final_response);
        println!("⏱️ Total time: {:.2}s", self_rag_result.total_time);
        println!("🔄 Iterations: {}", self_rag_result.iterations.len());
        println!("✅ Converged: {}", self_rag_result.convergence_achieved);

        if args.verbose {
            display_self_rag_details(&self_rag_result);
        }
    }

    Ok(())
}

fn display_self_rag_details(result: &SelfRAGResult) {
    println!("\n🪞 Self-RAG Iteration Details:");

    for iteration in &result.iterations {
        println!("\n   🔄 Iteration {}:", iteration.iteration + 1);
        println!("      📊 Quality Score: {:.2}", iteration.quality_score);
        println!("      💭 Feedback: {}", iteration.reflection_feedback);

        if !iteration.retrieval_adjustments.is_empty() {
            println!(
                "      🔧 Adjustments: {}",
                iteration.retrieval_adjustments.join(", ")
            );
        }

        println!(
            "      ➡️ Continue: {}",
            if iteration.should_continue {
                "Yes"
            } else {
                "No"
            }
        );
    }

    let quality_progression: Vec<f32> = result.iterations.iter().map(|i| i.quality_score).collect();
    if quality_progression.len() > 1 {
        let improvement =
            quality_progression.last().unwrap() - quality_progression.first().unwrap();
        println!(
            "\n📈 Quality Improvement: {:.2} → {:.2} ({:+.2})",
            quality_progression.first().unwrap(),
            quality_progression.last().unwrap(),
            improvement
        );
    }
}

async fn run_interactive_mode(
    vector_store: &Arc<dyn VectorStore>,
    query_engine: &QueryEngine,
    args: &Args,
) -> ExampleResult<()> {
    println!("\n🎯 Interactive Self-RAG Mode - Enter your queries (type 'quit' to exit):");

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
            break;
        }

        println!("\n🪞 Starting Self-RAG process...");

        match perform_self_rag(vector_store, query_engine, query, args).await {
            Ok(self_rag_result) => {
                println!("\n🎯 Final Result:");
                println!("💬 Response: {}", self_rag_result.final_response);
                println!("⏱️ Total time: {:.2}s", self_rag_result.total_time);
                println!("🔄 Iterations: {}", self_rag_result.iterations.len());
                println!("✅ Converged: {}", self_rag_result.convergence_achieved);

                if args.verbose {
                    display_self_rag_details(&self_rag_result);
                }
            }
            Err(e) => println!("❌ Self-RAG error: {}", e),
        }
    }

    Ok(())
}
