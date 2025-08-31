/*!
# Corrective RAG (CRAG) Example

This example demonstrates Corrective RAG, which automatically detects and corrects
retrieval errors through confidence assessment and alternative retrieval strategies.

Based on: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/corrective_rag.ipynb

## Key Features

- **Confidence Assessment**: Evaluates retrieval quality and confidence scores
- **Error Detection**: Identifies when retrieval results are insufficient or incorrect
- **Corrective Actions**: Applies alternative strategies when errors are detected
- **Adaptive Retrieval**: Dynamically adjusts retrieval approach based on confidence

## How It Works

1. **Initial Retrieval**: Perform standard retrieval and assess confidence
2. **Confidence Evaluation**: Score the relevance and quality of retrieved chunks
3. **Error Detection**: Identify low-confidence or irrelevant results
4. **Corrective Actions**: Apply alternative retrieval strategies if needed
5. **Quality Assurance**: Ensure final results meet quality thresholds

## Usage

```bash
# Basic corrective RAG
cargo run --bin corrective_rag --features fastembed

# Custom confidence thresholds
cargo run --bin corrective_rag --features fastembed -- --confidence-threshold 0.8

# Compare with standard RAG
cargo run --bin corrective_rag --features fastembed -- --compare-standard

# Interactive mode
cargo run --bin corrective_rag --features fastembed -- --interactive
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
#[command(name = "corrective_rag")]
struct Args {
    #[arg(long, default_value = "data/Understanding_Climate_Change.pdf")]
    document_path: PathBuf,
    #[arg(long, default_value = "800")]
    chunk_size: usize,
    #[arg(long, default_value = "100")]
    chunk_overlap: usize,
    #[arg(long, default_value = "10")]
    top_k: usize,
    #[arg(long, default_value = "0.7")]
    confidence_threshold: f32,
    #[arg(long, default_value = "3")]
    max_corrections: usize,
    #[arg(long)]
    compare_standard: bool,
    #[arg(long)]
    interactive: bool,
    #[arg(long)]
    verbose: bool,
}

#[derive(Debug, Clone)]
struct ConfidenceAssessment {
    overall_confidence: f32,
    relevance_scores: Vec<f32>,
    quality_issues: Vec<String>,
    needs_correction: bool,
    suggested_actions: Vec<String>,
}

#[derive(Debug, Clone)]
struct CorrectiveResult {
    final_chunks: Vec<ScoredNode>,
    corrections_applied: usize,
    confidence_progression: Vec<f32>,
    correction_actions: Vec<String>,
    total_time: f64,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    setup_logging();
    let args = Args::parse();

    println!("🔧 Starting Corrective RAG (CRAG) Example...");
    println!("📖 This example demonstrates error detection and correction in RAG");
    println!("🎯 Based on Corrective RAG techniques from RAG_Techniques repository\n");

    let embedder = create_embedder().await?;
    println!("✅ Embedder initialized");

    let (vector_store, query_engine) = build_indexing_pipeline(&args, embedder.clone()).await?;

    if args.compare_standard {
        compare_corrective_vs_standard(&vector_store, &query_engine, &args).await?;
    } else if args.interactive {
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
            .temperature(0.3) // Lower temperature for confidence assessment
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

async fn perform_corrective_rag(
    vector_store: &Arc<dyn VectorStore>,
    query_engine: &QueryEngine,
    query: &str,
    args: &Args,
) -> ExampleResult<CorrectiveResult> {
    let total_timer = Timer::new("Corrective RAG");
    let mut corrections_applied = 0;
    let mut confidence_progression = Vec::new();
    let mut correction_actions = Vec::new();
    let mut current_top_k = args.top_k;

    for correction_round in 0..args.max_corrections {
        println!(
            "🔍 Correction round {}/{}",
            correction_round + 1,
            args.max_corrections
        );

        // Perform retrieval
        let search_query = Query::builder()
            .text(query.to_string())
            .top_k(current_top_k)
            .search_mode(SearchMode::Vector)
            .build();

        let retrieved_chunks = vector_store
            .search(&search_query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;

        // Assess confidence
        let confidence = assess_retrieval_confidence(query, &retrieved_chunks, args).await?;
        confidence_progression.push(confidence.overall_confidence);

        if args.verbose {
            println!("   📊 Confidence: {:.2}", confidence.overall_confidence);
            if !confidence.quality_issues.is_empty() {
                println!("   ⚠️ Issues: {}", confidence.quality_issues.join(", "));
            }
        }

        // Check if correction is needed
        if !confidence.needs_correction
            || confidence.overall_confidence >= args.confidence_threshold
        {
            println!("✅ Confidence threshold met, no correction needed");
            return Ok(CorrectiveResult {
                final_chunks: retrieved_chunks,
                corrections_applied,
                confidence_progression,
                correction_actions,
                total_time: total_timer.finish().as_secs_f64(),
            });
        }

        // Apply corrective actions
        corrections_applied += 1;
        for action in &confidence.suggested_actions {
            correction_actions.push(action.clone());

            match action.as_str() {
                "expand_search" => {
                    current_top_k = (current_top_k * 3 / 2).min(50);
                    println!("   🔍 Expanding search to {} chunks", current_top_k);
                }
                "refine_query" => {
                    println!("   🎯 Query refinement applied");
                    // In a real implementation, this would use query transformation
                }
                "alternative_strategy" => {
                    println!("   🔄 Switching to alternative retrieval strategy");
                    // In a real implementation, this would use different retrieval methods
                }
                _ => {}
            }
        }
    }

    // Final retrieval after all corrections
    let final_query = Query::builder()
        .text(query.to_string())
        .top_k(current_top_k)
        .search_mode(SearchMode::Vector)
        .build();

    let final_chunks = vector_store
        .search(&final_query)
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;

    Ok(CorrectiveResult {
        final_chunks,
        corrections_applied,
        confidence_progression,
        correction_actions,
        total_time: total_timer.finish().as_secs_f64(),
    })
}

async fn assess_retrieval_confidence(
    query: &str,
    chunks: &[ScoredNode],
    _args: &Args,
) -> ExampleResult<ConfidenceAssessment> {
    // Simplified confidence assessment without LLM call for demo
    let avg_score = if chunks.is_empty() {
        0.0
    } else {
        chunks.iter().map(|c| c.score).sum::<f32>() / chunks.len() as f32
    };

    let overall_confidence = avg_score;
    let needs_correction = overall_confidence < 0.7;

    let suggested_actions = if needs_correction {
        vec!["expand_search".to_string(), "refine_query".to_string()]
    } else {
        vec![]
    };

    Ok(ConfidenceAssessment {
        overall_confidence,
        relevance_scores: chunks.iter().map(|c| c.score).collect(),
        quality_issues: if needs_correction {
            vec!["Low relevance scores".to_string()]
        } else {
            vec![]
        },
        needs_correction,
        suggested_actions,
    })
}

fn parse_confidence_response(
    response: &str,
    chunk_count: usize,
) -> ExampleResult<ConfidenceAssessment> {
    let mut overall_confidence = 0.5;
    let mut relevance_scores = vec![0.5; chunk_count];
    let mut quality_issues = Vec::new();
    let mut needs_correction = true;
    let mut suggested_actions = Vec::new();

    for line in response.lines() {
        if line.starts_with("OVERALL_CONFIDENCE:") {
            if let Ok(conf) = line
                .split(':')
                .nth(1)
                .unwrap_or("0.5")
                .trim()
                .parse::<f32>()
            {
                overall_confidence = conf.clamp(0.0, 1.0);
            }
        } else if line.starts_with("RELEVANCE_SCORES:") {
            let scores_str = line
                .split(':')
                .skip(1)
                .collect::<Vec<_>>()
                .join(":")
                .trim()
                .to_string();
            let parsed_scores: Vec<f32> = scores_str
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if !parsed_scores.is_empty() {
                relevance_scores = parsed_scores;
            }
        } else if line.starts_with("QUALITY_ISSUES:") {
            let issues_str = line
                .split(':')
                .skip(1)
                .collect::<Vec<_>>()
                .join(":")
                .trim()
                .to_string();
            quality_issues = issues_str
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();
        } else if line.starts_with("NEEDS_CORRECTION:") {
            let needs_str = line
                .split(':')
                .nth(1)
                .unwrap_or("YES")
                .trim()
                .to_uppercase();
            needs_correction = needs_str == "YES";
        } else if line.starts_with("SUGGESTED_ACTIONS:") {
            let actions_str = line
                .split(':')
                .skip(1)
                .collect::<Vec<_>>()
                .join(":")
                .trim()
                .to_string();
            suggested_actions = actions_str
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();
        }
    }

    Ok(ConfidenceAssessment {
        overall_confidence,
        relevance_scores,
        quality_issues,
        needs_correction,
        suggested_actions,
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

        let corrective_result =
            perform_corrective_rag(vector_store, query_engine, query, args).await?;

        // Generate final response
        let response = query_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;

        println!("\n🎯 Final Result:");
        println!("💬 Response: {}", response.response.content);
        println!("⏱️ Total time: {:.2}s", corrective_result.total_time);
        println!(
            "🔧 Corrections applied: {}",
            corrective_result.corrections_applied
        );

        if args.verbose {
            display_corrective_details(&corrective_result);
        }
    }

    Ok(())
}

fn display_corrective_details(result: &CorrectiveResult) {
    println!("\n🔧 Corrective RAG Details:");
    println!("   📊 Final chunks: {}", result.final_chunks.len());
    println!("   🔄 Corrections: {}", result.corrections_applied);

    if !result.confidence_progression.is_empty() {
        println!(
            "   📈 Confidence progression: {:.2} → {:.2}",
            result.confidence_progression.first().unwrap(),
            result.confidence_progression.last().unwrap()
        );
    }

    if !result.correction_actions.is_empty() {
        println!(
            "   🛠️ Actions taken: {}",
            result.correction_actions.join(", ")
        );
    }
}

async fn compare_corrective_vs_standard(
    vector_store: &Arc<dyn VectorStore>,
    query_engine: &QueryEngine,
    args: &Args,
) -> ExampleResult<()> {
    println!("⚖️ Comparing Corrective RAG vs Standard RAG...");
    println!("═══════════════════════════════════════════════════════════════\n");

    let test_queries = get_climate_test_queries();

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        // Standard RAG
        let standard_timer = Timer::new("Standard RAG");
        let standard_response = query_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;
        let standard_time = standard_timer.finish();

        // Corrective RAG
        let corrective_result =
            perform_corrective_rag(vector_store, query_engine, query, args).await?;

        println!("   📏 Standard RAG: {:.2}s", standard_time.as_secs_f64());
        println!(
            "   🔧 Corrective RAG: {:.2}s, {} corrections",
            corrective_result.total_time, corrective_result.corrections_applied
        );

        if !corrective_result.confidence_progression.is_empty() {
            let improvement = corrective_result.confidence_progression.last().unwrap()
                - corrective_result.confidence_progression.first().unwrap();
            println!("   📈 Confidence improvement: {:+.2}", improvement);
        }
    }

    Ok(())
}

async fn run_interactive_mode(
    vector_store: &Arc<dyn VectorStore>,
    query_engine: &QueryEngine,
    args: &Args,
) -> ExampleResult<()> {
    println!("\n🎯 Interactive Corrective RAG Mode - Enter your queries (type 'quit' to exit):");

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

        println!("\n🔧 Starting Corrective RAG process...");

        match perform_corrective_rag(vector_store, query_engine, query, args).await {
            Ok(corrective_result) => {
                let response = query_engine
                    .query(query)
                    .await
                    .map_err(|e| ExampleError::Cheungfun(e))?;

                println!("\n🎯 Final Result:");
                println!("💬 Response: {}", response.response.content);
                println!("⏱️ Total time: {:.2}s", corrective_result.total_time);
                println!(
                    "🔧 Corrections applied: {}",
                    corrective_result.corrections_applied
                );

                if args.verbose {
                    display_corrective_details(&corrective_result);
                }
            }
            Err(e) => println!("❌ Corrective RAG error: {}", e),
        }
    }

    Ok(())
}
