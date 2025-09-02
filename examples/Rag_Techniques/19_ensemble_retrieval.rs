/*!
# Ensemble Retrieval RAG Example

This example demonstrates ensemble retrieval, which combines multiple retrieval methods
to achieve better accuracy and robustness than any single method alone.

Based on: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/ensemble_retrieval.ipynb

## Key Features

- **Multiple Retrieval Methods**: Vector search, keyword search, hybrid search
- **Ensemble Fusion**: Combines results using various fusion strategies
- **Adaptive Weighting**: Dynamically adjusts weights based on query characteristics
- **Performance Analysis**: Compare ensemble vs individual methods

## How It Works

1. **Multi-Method Retrieval**: Run multiple retrieval strategies in parallel
2. **Result Fusion**: Combine results using rank fusion, score fusion, or voting
3. **Quality Assessment**: Evaluate ensemble effectiveness vs individual methods
4. **Adaptive Selection**: Choose best ensemble strategy based on query type

## Usage

```bash
# Basic ensemble retrieval
cargo run --bin ensemble_retrieval --features fastembed

# Compare fusion strategies
cargo run --bin ensemble_retrieval --features fastembed -- --compare-fusion

# Custom ensemble weights
cargo run --bin ensemble_retrieval --features fastembed -- --vector-weight 0.6 --keyword-weight 0.4

# Interactive mode
cargo run --bin ensemble_retrieval --features fastembed -- --interactive
```
*/

use clap::Parser;
use std::{path::PathBuf, sync::Arc};

#[path = "../shared/mod.rs"]
mod shared;

use shared::{get_climate_test_queries, setup_logging, ExampleError, ExampleResult, Timer};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, ResponseGenerator, VectorStore},
    types::{GenerationOptions, Query, SearchMode},
    DistanceMetric, ScoredNode,
};
use cheungfun_indexing::{
    loaders::DirectoryLoader, node_parser::text::SentenceSplitter,
    pipeline::DefaultIndexingPipeline, transformers::MetadataExtractor,
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    advanced::fusion::{DistributionBasedFusion, ReciprocalRankFusion},
    engine::QueryEngine,
    generator::SiumaiGenerator,
    retriever::VectorRetriever,
};
use siumai::prelude::*;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug)]
#[command(name = "ensemble_retrieval")]
struct Args {
    #[arg(long, default_value = "data/Understanding_Climate_Change.pdf")]
    document_path: PathBuf,
    #[arg(long, default_value = "800")]
    chunk_size: usize,
    #[arg(long, default_value = "100")]
    chunk_overlap: usize,
    #[arg(long, default_value = "10")]
    top_k: usize,
    #[arg(long, default_value = "0.6")]
    vector_weight: f32,
    #[arg(long, default_value = "0.4")]
    keyword_weight: f32,
    #[arg(long, default_value = "rank")]
    fusion_method: String, // rank, score, voting, adaptive
    #[arg(long)]
    compare_fusion: bool,
    #[arg(long)]
    interactive: bool,
    #[arg(long)]
    verbose: bool,
}

#[derive(Debug, Clone)]
struct EnsembleResult {
    method_name: String,
    chunks: Vec<ScoredNode>,
    retrieval_time: f64,
    avg_score: f32,
}

#[derive(Debug, Clone)]
struct FusedResult {
    fusion_method: String,
    chunks: Vec<ScoredNode>,
    fusion_time: f64,
    component_results: Vec<EnsembleResult>,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    setup_logging();
    let args = Args::parse();

    println!("🎭 Starting Ensemble Retrieval Example...");
    println!("📖 This example demonstrates combining multiple retrieval methods");
    println!("🎯 Based on ensemble techniques from RAG_Techniques repository\n");

    let embedder = create_embedder().await?;
    println!("✅ Embedder initialized");

    if args.compare_fusion {
        compare_fusion_strategies(&args, embedder).await?;
    } else {
        run_ensemble_retrieval(&args, embedder).await?;
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

async fn run_ensemble_retrieval(args: &Args, embedder: Arc<dyn Embedder>) -> ExampleResult<()> {
    println!(
        "🎭 Running ensemble retrieval with {} fusion...",
        args.fusion_method
    );

    let (vector_store, _) = build_indexing_pipeline(args, embedder.clone()).await?;

    if args.interactive {
        run_interactive_mode(&vector_store, embedder, args).await?;
    } else {
        run_test_queries(&vector_store, embedder, args).await?;
    }

    Ok(())
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
        .with_document_processor(splitter) // Documents -> Nodes
        .with_node_processor(metadata_extractor) // Nodes -> Nodes
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

async fn run_ensemble_methods(
    vector_store: &Arc<dyn VectorStore>,
    query: &str,
    top_k: usize,
) -> ExampleResult<Vec<EnsembleResult>> {
    let mut results = Vec::new();

    // Method 1: Vector Search
    let vector_timer = Timer::new("Vector search");
    let vector_query = Query::builder()
        .text(query.to_string())
        .top_k(top_k)
        .search_mode(SearchMode::Vector)
        .build();

    let vector_chunks = vector_store
        .search(&vector_query)
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;
    let vector_time = vector_timer.finish();
    let vector_avg = if vector_chunks.is_empty() {
        0.0
    } else {
        vector_chunks.iter().map(|c| c.score).sum::<f32>() / vector_chunks.len() as f32
    };

    results.push(EnsembleResult {
        method_name: "Vector".to_string(),
        chunks: vector_chunks,
        retrieval_time: vector_time.as_secs_f64(),
        avg_score: vector_avg,
    });

    // Method 2: Keyword Search (simulated with text matching)
    let keyword_timer = Timer::new("Keyword search");
    let keyword_query = Query::builder()
        .text(query.to_string())
        .top_k(top_k * 2) // Get more for keyword filtering
        .search_mode(SearchMode::Vector) // We'll filter these
        .build();

    let all_chunks = vector_store
        .search(&keyword_query)
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;
    let keyword_chunks = simulate_keyword_search(all_chunks, query, top_k);
    let keyword_time = keyword_timer.finish();
    let keyword_avg = if keyword_chunks.is_empty() {
        0.0
    } else {
        keyword_chunks.iter().map(|c| c.score).sum::<f32>() / keyword_chunks.len() as f32
    };

    results.push(EnsembleResult {
        method_name: "Keyword".to_string(),
        chunks: keyword_chunks,
        retrieval_time: keyword_time.as_secs_f64(),
        avg_score: keyword_avg,
    });

    // Method 3: Hybrid Search (combination of above)
    let hybrid_timer = Timer::new("Hybrid search");
    let hybrid_chunks = simulate_hybrid_search(&results[0].chunks, &results[1].chunks, top_k);
    let hybrid_time = hybrid_timer.finish();
    let hybrid_avg = if hybrid_chunks.is_empty() {
        0.0
    } else {
        hybrid_chunks.iter().map(|c| c.score).sum::<f32>() / hybrid_chunks.len() as f32
    };

    results.push(EnsembleResult {
        method_name: "Hybrid".to_string(),
        chunks: hybrid_chunks,
        retrieval_time: hybrid_time.as_secs_f64(),
        avg_score: hybrid_avg,
    });

    Ok(results)
}

fn simulate_keyword_search(chunks: Vec<ScoredNode>, query: &str, top_k: usize) -> Vec<ScoredNode> {
    let query_lower = query.to_lowercase();
    let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

    let mut scored_chunks: Vec<(ScoredNode, f32)> = chunks
        .into_iter()
        .map(|chunk| {
            let content_lower = chunk.node.content.to_lowercase();
            let keyword_score = query_terms
                .iter()
                .map(|term| {
                    let count = content_lower.matches(term).count() as f32;
                    count / content_lower.split_whitespace().count() as f32
                })
                .sum::<f32>();

            (chunk, keyword_score)
        })
        .collect();

    scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored_chunks
        .into_iter()
        .take(top_k)
        .map(|(mut chunk, keyword_score)| {
            // Combine original vector score with keyword score
            chunk.score = (chunk.score + keyword_score) / 2.0;
            chunk
        })
        .collect()
}

fn simulate_hybrid_search(
    vector_chunks: &[ScoredNode],
    keyword_chunks: &[ScoredNode],
    top_k: usize,
) -> Vec<ScoredNode> {
    let mut combined_chunks = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();

    // Add vector results with higher weight
    for chunk in vector_chunks.iter().take(top_k) {
        if seen_ids.insert(chunk.node.id) {
            let mut weighted_chunk = chunk.clone();
            weighted_chunk.score *= 0.7; // Vector weight
            combined_chunks.push(weighted_chunk);
        }
    }

    // Add keyword results with lower weight
    for chunk in keyword_chunks.iter().take(top_k) {
        if seen_ids.insert(chunk.node.id) {
            let mut weighted_chunk = chunk.clone();
            weighted_chunk.score *= 0.3; // Keyword weight
            combined_chunks.push(weighted_chunk);
        }
    }

    // Sort by combined score and take top-k
    combined_chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    combined_chunks.truncate(top_k);
    combined_chunks
}

async fn fuse_ensemble_results(
    results: &[EnsembleResult],
    fusion_method: &str,
    args: &Args,
) -> ExampleResult<FusedResult> {
    let fusion_timer = Timer::new(&format!("{} fusion", fusion_method));

    // Convert EnsembleResult to the format expected by fusion strategies
    let result_sets: Vec<Vec<ScoredNode>> = results.iter().map(|r| r.chunks.clone()).collect();

    let fused_chunks = match fusion_method {
        "rank" => {
            let fusion = ReciprocalRankFusion::new(60.0); // k=60 parameter
            let mut results = fusion.fuse_results(result_sets);
            results.truncate(args.top_k);
            results
        }
        "score" => {
            // Use weighted score fusion (manual implementation for now)
            let weights = vec![args.vector_weight, args.keyword_weight, 0.5]; // Default weights
            weighted_score_fusion(result_sets, weights, args.top_k)
        }
        "voting" => {
            // Use voting-based fusion (manual implementation for now)
            voting_based_fusion(result_sets, args.top_k)
        }
        "adaptive" => {
            // Use distribution-based fusion for adaptive strategy
            let fusion = DistributionBasedFusion::new(result_sets.len());
            let mut results = fusion.fuse_results(result_sets);
            results.truncate(args.top_k);
            results
        }
        _ => {
            // Default to rank fusion
            let fusion = ReciprocalRankFusion::new(60.0);
            let mut results = fusion.fuse_results(result_sets);
            results.truncate(args.top_k);
            results
        }
    };

    let fusion_time = fusion_timer.finish();

    Ok(FusedResult {
        fusion_method: fusion_method.to_string(),
        chunks: fused_chunks,
        fusion_time: fusion_time.as_secs_f64(),
        component_results: results.to_vec(),
    })
}

/// Weighted score fusion implementation.
fn weighted_score_fusion(
    result_sets: Vec<Vec<ScoredNode>>,
    weights: Vec<f32>,
    top_k: usize,
) -> Vec<ScoredNode> {
    let mut score_map: std::collections::HashMap<String, (ScoredNode, f32, usize)> =
        std::collections::HashMap::new();

    for (method_idx, result_set) in result_sets.into_iter().enumerate() {
        let weight = weights.get(method_idx).unwrap_or(&0.33);

        for chunk in result_set {
            let id = chunk.node.id.to_string();
            let weighted_score = chunk.score * weight;

            match score_map.get_mut(&id) {
                Some((_, total_score, count)) => {
                    *total_score += weighted_score;
                    *count += 1;
                }
                None => {
                    score_map.insert(id, (chunk, weighted_score, 1));
                }
            }
        }
    }

    let mut fused: Vec<(ScoredNode, f32)> = score_map
        .into_iter()
        .map(|(_, (chunk, total_score, count))| {
            let mut fused_chunk = chunk;
            let final_score = total_score / count as f32;
            fused_chunk.score = final_score;
            (fused_chunk, final_score)
        })
        .collect();

    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    fused
        .into_iter()
        .take(top_k)
        .map(|(chunk, _)| chunk)
        .collect()
}

/// Voting-based fusion implementation.
fn voting_based_fusion(result_sets: Vec<Vec<ScoredNode>>, top_k: usize) -> Vec<ScoredNode> {
    let mut vote_counts: std::collections::HashMap<String, (ScoredNode, usize)> =
        std::collections::HashMap::new();

    for result_set in result_sets {
        for chunk in result_set {
            let id = chunk.node.id.to_string();
            match vote_counts.get_mut(&id) {
                Some((_, count)) => *count += 1,
                None => {
                    vote_counts.insert(id, (chunk, 1));
                }
            }
        }
    }

    let mut voted: Vec<(ScoredNode, usize)> = vote_counts.into_iter().map(|(_, v)| v).collect();
    voted.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| b.0.score.partial_cmp(&a.0.score).unwrap())
    });

    voted
        .into_iter()
        .take(top_k)
        .map(|(chunk, _)| chunk)
        .collect()
}

async fn run_test_queries(
    vector_store: &Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedder>,
    args: &Args,
) -> ExampleResult<()> {
    let test_queries = get_climate_test_queries();
    let generator = SiumaiGenerator::new(create_siumai_client().await?);

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        let timer = Timer::new("Ensemble retrieval");

        // Run ensemble methods
        let ensemble_results = run_ensemble_methods(vector_store, query, args.top_k).await?;

        // Fuse results
        let fused_result =
            fuse_ensemble_results(&ensemble_results, &args.fusion_method, args).await?;

        let query_time = timer.finish();

        // Generate response
        let options = GenerationOptions::default();
        let response = generator
            .generate_response(query, fused_result.chunks.clone(), &options)
            .await
            .map_err(|e| ExampleError::Config(format!("Generation failed: {}", e)))?;

        println!("💬 Response: {}", response.content);
        println!("⏱️ Query time: {:.2}s", query_time.as_secs_f64());

        if args.verbose {
            display_ensemble_details(&fused_result);
        }
    }

    Ok(())
}

fn display_ensemble_details(result: &FusedResult) {
    println!("\n🎭 Ensemble Details:");
    println!("   🔀 Fusion method: {}", result.fusion_method);
    println!("   ⏱️ Fusion time: {:.3}s", result.fusion_time);
    println!("   📊 Final chunks: {}", result.chunks.len());

    for component in &result.component_results {
        println!(
            "   📈 {}: {} chunks, avg score: {:.3}, time: {:.3}s",
            component.method_name,
            component.chunks.len(),
            component.avg_score,
            component.retrieval_time
        );
    }
}

async fn compare_fusion_strategies(args: &Args, embedder: Arc<dyn Embedder>) -> ExampleResult<()> {
    println!("⚖️ Comparing Ensemble Fusion Strategies...");
    println!("═══════════════════════════════════════════════════════════════\n");

    let (vector_store, _) = build_indexing_pipeline(args, embedder.clone()).await?;
    let test_queries = get_climate_test_queries();
    let fusion_methods = vec!["rank", "score", "voting", "adaptive"];

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        // Run ensemble methods once
        let ensemble_results = run_ensemble_methods(&vector_store, query, args.top_k).await?;

        // Test each fusion strategy
        for fusion_method in &fusion_methods {
            let fused_result =
                fuse_ensemble_results(&ensemble_results, fusion_method, args).await?;

            let avg_score = if fused_result.chunks.is_empty() {
                0.0
            } else {
                fused_result.chunks.iter().map(|c| c.score).sum::<f32>()
                    / fused_result.chunks.len() as f32
            };

            println!(
                "   🔀 {}: {} chunks, avg score: {:.3}, fusion time: {:.3}s",
                fusion_method,
                fused_result.chunks.len(),
                avg_score,
                fused_result.fusion_time
            );
        }
    }

    Ok(())
}

async fn run_interactive_mode(
    vector_store: &Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedder>,
    args: &Args,
) -> ExampleResult<()> {
    println!("\n🎯 Interactive Mode - Enter your queries (type 'quit' to exit):");

    let generator = SiumaiGenerator::new(create_siumai_client().await?);

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

        let timer = Timer::new("Ensemble query");

        // Run ensemble methods
        let ensemble_results = run_ensemble_methods(vector_store, query, args.top_k).await?;

        // Fuse results
        let fused_result =
            fuse_ensemble_results(&ensemble_results, &args.fusion_method, args).await?;

        let query_time = timer.finish();

        // Generate response
        let options = GenerationOptions::default();
        match generator
            .generate_response(query, fused_result.chunks.clone(), &options)
            .await
        {
            Ok(response) => {
                println!("\n💬 Response: {}", response.content);
                println!("⏱️ Query time: {:.2}s", query_time.as_secs_f64());

                if args.verbose {
                    display_ensemble_details(&fused_result);
                }
            }
            Err(e) => println!("❌ Generation error: {}", e),
        }
    }

    Ok(())
}
