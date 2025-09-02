/*!
# Multi-faceted Filtering RAG Example

This example demonstrates various filtering strategies for improving RAG retrieval quality.

## Usage

```bash
cargo run --bin multi_faceted_filtering --features fastembed -- --compare-strategies
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
    engine::QueryEngine,
    generator::SiumaiGenerator,
    postprocessor::{
        NodePostprocessor, SentenceEmbeddingConfig, SentenceEmbeddingOptimizer, SimilarityFilter,
        SimilarityFilterConfig,
    },
    retriever::VectorRetriever,
};
use siumai::prelude::*;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug)]
#[command(name = "multi_faceted_filtering")]
struct Args {
    #[arg(long, default_value = "data/Understanding_Climate_Change.pdf")]
    document_path: PathBuf,
    #[arg(long, default_value = "800")]
    chunk_size: usize,
    #[arg(long, default_value = "100")]
    chunk_overlap: usize,
    #[arg(long, default_value = "20")]
    initial_retrieval_count: usize,
    #[arg(long, default_value = "5")]
    top_k: usize,
    #[arg(long, default_value = "0.7")]
    score_threshold: f32,
    #[arg(long, default_value = "100")]
    min_length: usize,
    #[arg(long, default_value = "2000")]
    max_length: usize,
    #[arg(long, default_value = "sentence_embedding")]
    strategy: String,
    #[arg(long)]
    compare_strategies: bool,
    #[arg(long)]
    interactive: bool,
    #[arg(long)]
    verbose: bool,
}

#[derive(Debug, Clone)]
pub struct FilteredResult {
    pub chunks: Vec<ScoredNode>,
    pub strategy_used: String,
    pub original_count: usize,
    pub filtered_count: usize,
    pub filtering_time: f64,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    setup_logging();
    let args = Args::parse();

    println!("🔍 Starting Multi-faceted Filtering Example...");

    let embedder = create_embedder().await?;
    println!("✅ Embedder initialized");

    if args.compare_strategies {
        compare_filtering_strategies(&args, embedder).await?;
    } else {
        run_filtering_strategy(&args, embedder).await?;
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

async fn run_filtering_strategy(args: &Args, embedder: Arc<dyn Embedder>) -> ExampleResult<()> {
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

    let (_nodes, index_result) = pipeline
        .run(None, None, true, true, None, true)
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;
    println!("✅ Indexed {} nodes", index_result.nodes_created);

    let siumai_client = create_siumai_client().await?;
    let retriever = Arc::new(VectorRetriever::new(vector_store.clone(), embedder));
    let generator = Arc::new(SiumaiGenerator::new(siumai_client));
    let query_engine = QueryEngine::new(retriever, generator);

    Ok((vector_store, query_engine))
}

fn apply_metadata_filtering(chunks: Vec<ScoredNode>, _args: &Args) -> Vec<ScoredNode> {
    chunks
        .into_iter()
        .filter(|chunk| {
            if let Some(source) = chunk.node.metadata.get("source") {
                if let Some(source_str) = source.as_str() {
                    return source_str.ends_with(".pdf");
                }
            }
            true
        })
        .collect()
}

async fn apply_score_filtering(
    chunks: Vec<ScoredNode>,
    args: &Args,
    query: &str,
) -> ExampleResult<Vec<ScoredNode>> {
    let similarity_filter = SimilarityFilter::new(SimilarityFilterConfig {
        similarity_cutoff: args.score_threshold,
        max_nodes: Some(args.top_k),
        use_query_embedding: true,
    });

    similarity_filter
        .postprocess(chunks, query)
        .await
        .map_err(|e| ExampleError::Config(format!("Similarity filtering failed: {}", e)))
}

fn apply_quality_filtering(chunks: Vec<ScoredNode>, args: &Args) -> Vec<ScoredNode> {
    chunks
        .into_iter()
        .filter(|chunk| {
            let content = &chunk.node.content;
            let length = content.len();

            if length < args.min_length || length > args.max_length {
                return false;
            }

            let trimmed = content.trim();
            if trimmed.is_empty() {
                return false;
            }

            let word_count = content.split_whitespace().count();
            word_count >= 10
        })
        .collect()
}

async fn apply_filtering_strategy(
    chunks: Vec<ScoredNode>,
    strategy: &str,
    args: &Args,
    query: &str,
    embedder: Arc<dyn Embedder>,
) -> ExampleResult<FilteredResult> {
    let timer = Timer::new(&format!("{} filtering", strategy));
    let original_count = chunks.len();

    let filtered_chunks = match strategy {
        "metadata" => apply_metadata_filtering(chunks, args),
        "score" => apply_score_filtering(chunks, args, query).await?,
        "quality" => apply_quality_filtering(chunks, args),
        "sentence_embedding" => {
            // Use SentenceEmbeddingOptimizer for advanced filtering
            let config = SentenceEmbeddingConfig {
                percentile_cutoff: Some(0.7), // Keep top 70% of sentences
                threshold_cutoff: Some(0.3),  // Minimum similarity threshold
                context_before: Some(1),
                context_after: Some(1),
                max_sentences_per_node: Some(20),
            };

            let optimizer = SentenceEmbeddingOptimizer::new(embedder, config);
            optimizer.postprocess(chunks, query).await.map_err(|e| {
                ExampleError::Config(format!("Sentence embedding filtering failed: {}", e))
            })?
        }
        "combined" => {
            let after_metadata = apply_metadata_filtering(chunks, args);
            let after_score = apply_score_filtering(after_metadata, args, query).await?;
            apply_quality_filtering(after_score, args)
        }
        _ => chunks,
    };

    let filtering_time = timer.finish();
    let filtered_count = filtered_chunks.len();

    Ok(FilteredResult {
        chunks: filtered_chunks,
        strategy_used: strategy.to_string(),
        original_count,
        filtered_count,
        filtering_time: filtering_time.as_secs_f64(),
    })
}

async fn compare_filtering_strategies(
    args: &Args,
    embedder: Arc<dyn Embedder>,
) -> ExampleResult<()> {
    println!("⚖️ Comparing Multi-faceted Filtering Strategies...");

    let (vector_store, _) = build_indexing_pipeline(args, embedder.clone()).await?;
    let test_queries = get_climate_test_queries();
    let strategies = vec![
        "metadata",
        "score",
        "quality",
        "sentence_embedding",
        "combined",
    ];

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        for strategy in &strategies {
            let search_query = Query::builder()
                .text(query.to_string())
                .top_k(args.initial_retrieval_count)
                .search_mode(SearchMode::Vector)
                .build();

            let initial_chunks = vector_store
                .search(&search_query)
                .await
                .map_err(|e| ExampleError::Cheungfun(e))?;
            let filtered_result =
                apply_filtering_strategy(initial_chunks, strategy, args, query, embedder.clone())
                    .await?;

            println!(
                "   🎯 {}: {} → {} chunks",
                strategy, filtered_result.original_count, filtered_result.filtered_count
            );
        }
    }

    Ok(())
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

        let search_query = Query::builder()
            .text(query.to_string())
            .top_k(args.initial_retrieval_count)
            .search_mode(SearchMode::Vector)
            .build();

        let initial_chunks = vector_store
            .search(&search_query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;
        let filtered_result = apply_filtering_strategy(
            initial_chunks,
            &args.strategy,
            args,
            query,
            embedder.clone(),
        )
        .await?;

        let options = GenerationOptions::default();
        let response = generator
            .generate_response(query, filtered_result.chunks.clone(), &options)
            .await
            .map_err(|e| ExampleError::Config(format!("Generation failed: {}", e)))?;

        println!("💬 Response: {}", response.content);
        println!("📊 Used {} filtered chunks", filtered_result.filtered_count);
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

        let search_query = Query::builder()
            .text(query.to_string())
            .top_k(args.initial_retrieval_count)
            .search_mode(SearchMode::Vector)
            .build();

        match vector_store.search(&search_query).await {
            Ok(initial_chunks) => {
                let filtered_result = apply_filtering_strategy(
                    initial_chunks,
                    &args.strategy,
                    args,
                    query,
                    embedder.clone(),
                )
                .await?;

                let options = GenerationOptions::default();

                match generator
                    .generate_response(query, filtered_result.chunks.clone(), &options)
                    .await
                {
                    Ok(response) => {
                        println!("\n💬 Response: {}", response.content);
                        println!("📊 Used {} filtered chunks", filtered_result.filtered_count);
                    }
                    Err(e) => println!("❌ Generation error: {}", e),
                }
            }
            Err(e) => println!("❌ Retrieval error: {}", e),
        }
    }

    Ok(())
}
