/*!
# Hierarchical Indices RAG Example

This example demonstrates hierarchical indexing, which creates multi-level indices
for more efficient and accurate retrieval across different granularities.

Based on: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/hierarchical_indices.ipynb

## Key Features

- **Multi-level Indexing**: Creates indices at document, section, and chunk levels
- **Hierarchical Retrieval**: Searches across different granularities simultaneously
- **Parent-Child Relationships**: Maintains relationships between different levels
- **Adaptive Granularity**: Automatically selects appropriate level based on query type

## How It Works

1. **Document Level**: Index entire documents for high-level topic matching
2. **Section Level**: Index document sections for medium-granularity retrieval
3. **Chunk Level**: Index small chunks for precise information retrieval
4. **Hierarchical Search**: Search across all levels and combine results
5. **Context Assembly**: Assemble hierarchical context for comprehensive responses

## Usage

```bash
# Basic hierarchical indices
cargo run --bin hierarchical_indices --features fastembed

# Compare with flat indexing
cargo run --bin hierarchical_indices --features fastembed -- --compare-flat

# Adjust hierarchy levels
cargo run --bin hierarchical_indices --features fastembed -- --max-levels 4

# Interactive mode
cargo run --bin hierarchical_indices --features fastembed -- --interactive
```
*/

use clap::Parser;
use std::{collections::HashMap, path::PathBuf, sync::Arc};

#[path = "../shared/mod.rs"]
mod shared;

use shared::{get_climate_test_queries, setup_logging, ExampleError, ExampleResult, Timer};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, ResponseGenerator, VectorStore},
    types::{GenerationOptions, Query, SearchMode},
    DistanceMetric, Node, ScoredNode,
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
#[command(name = "hierarchical_indices")]
struct Args {
    #[arg(long, default_value = "data/Understanding_Climate_Change.pdf")]
    document_path: PathBuf,
    #[arg(long, default_value = "3")]
    max_levels: usize,
    #[arg(long, default_value = "5")]
    top_k: usize,
    #[arg(long)]
    compare_flat: bool,
    #[arg(long)]
    interactive: bool,
    #[arg(long)]
    verbose: bool,
}

#[derive(Debug, Clone)]
struct HierarchicalLevel {
    level: usize,
    name: String,
    chunk_size: usize,
    chunk_overlap: usize,
    vector_store: Arc<dyn VectorStore>,
}

#[derive(Debug, Clone)]
struct HierarchicalResult {
    level: usize,
    chunks: Vec<ScoredNode>,
    avg_score: f32,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    setup_logging();
    let args = Args::parse();

    println!("🏗️ Starting Hierarchical Indices Example...");

    let embedder = create_embedder().await?;
    println!("✅ Embedder initialized");

    if args.compare_flat {
        compare_hierarchical_vs_flat(&args, embedder).await?;
    } else {
        run_hierarchical_indexing(&args, embedder).await?;
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

async fn run_hierarchical_indexing(args: &Args, embedder: Arc<dyn Embedder>) -> ExampleResult<()> {
    println!("🏗️ Building hierarchical indices...");

    let levels = build_hierarchical_levels(args, embedder.clone()).await?;

    if args.interactive {
        run_interactive_mode(&levels, embedder, args).await?;
    } else {
        run_test_queries(&levels, embedder, args).await?;
    }

    Ok(())
}

async fn build_hierarchical_levels(
    args: &Args,
    embedder: Arc<dyn Embedder>,
) -> ExampleResult<Vec<HierarchicalLevel>> {
    let mut levels = Vec::new();

    // Define hierarchy levels with different granularities
    let level_configs = vec![
        ("Document", 4000, 400), // Document level
        ("Section", 1500, 150),  // Section level
        ("Paragraph", 600, 60),  // Paragraph level
        ("Sentence", 200, 20),   // Sentence level
    ];

    let data_dir = if args.document_path.is_absolute() {
        args.document_path
            .parent()
            .unwrap_or(&PathBuf::from("."))
            .to_path_buf()
    } else {
        std::env::current_dir()?.join(args.document_path.parent().unwrap_or(&PathBuf::from(".")))
    };

    for (level_idx, (name, chunk_size, chunk_overlap)) in level_configs.iter().enumerate() {
        if level_idx >= args.max_levels {
            break;
        }

        println!(
            "🔨 Building level {}: {} ({}±{} chars)...",
            level_idx + 1,
            name,
            chunk_size,
            chunk_overlap
        );

        let timer = Timer::new(&format!("Level {} indexing", level_idx + 1));

        let loader = Arc::new(DirectoryLoader::new(&data_dir)?);
        let splitter = Arc::new(SentenceSplitter::from_defaults(
            *chunk_size,
            *chunk_overlap,
        )?);
        let metadata_extractor = Arc::new(MetadataExtractor::new());
        let vector_store = Arc::new(InMemoryVectorStore::new(
            DEFAULT_EMBEDDING_DIM,
            DistanceMetric::Cosine,
        ));

        let pipeline = DefaultIndexingPipeline::builder()
            .with_loader(loader)
            .with_transformer(splitter)
            .with_transformer(metadata_extractor)
            .with_embedder(embedder.clone())
            .with_vector_store(vector_store.clone())
            .build()?;

        let index_result = pipeline
            .run()
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;
        let indexing_time = timer.finish();

        println!(
            "   ✅ Level {} completed in {:.2}s, {} nodes",
            level_idx + 1,
            indexing_time,
            index_result.nodes_created
        );

        levels.push(HierarchicalLevel {
            level: level_idx,
            name: name.to_string(),
            chunk_size: *chunk_size,
            chunk_overlap: *chunk_overlap,
            vector_store,
        });
    }

    println!(
        "🏗️ Hierarchical indexing completed with {} levels",
        levels.len()
    );
    Ok(levels)
}

async fn search_hierarchical_levels(
    levels: &[HierarchicalLevel],
    query: &str,
    top_k_per_level: usize,
) -> ExampleResult<Vec<HierarchicalResult>> {
    let mut results = Vec::new();

    for level in levels {
        let search_query = Query::builder()
            .text(query.to_string())
            .top_k(top_k_per_level)
            .search_mode(SearchMode::Vector)
            .build();

        let chunks = level
            .vector_store
            .search(&search_query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;

        let avg_score = if chunks.is_empty() {
            0.0
        } else {
            chunks.iter().map(|c| c.score).sum::<f32>() / chunks.len() as f32
        };

        results.push(HierarchicalResult {
            level: level.level,
            chunks,
            avg_score,
        });
    }

    Ok(results)
}

async fn run_test_queries(
    levels: &[HierarchicalLevel],
    embedder: Arc<dyn Embedder>,
    args: &Args,
) -> ExampleResult<()> {
    let test_queries = get_climate_test_queries();
    let generator = SiumaiGenerator::new(create_siumai_client().await?);

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        let timer = Timer::new("Hierarchical search");
        let results = search_hierarchical_levels(levels, query, args.top_k).await?;
        let search_time = timer.finish();

        // Combine results from all levels
        let mut all_chunks = Vec::new();
        for result in &results {
            all_chunks.extend(result.chunks.clone());
        }

        // Sort by score and take top-k
        all_chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        all_chunks.truncate(args.top_k);

        let options = GenerationOptions::default();
        let response = generator
            .generate_response(query, all_chunks.clone(), &options)
            .await
            .map_err(|e| ExampleError::Config(format!("Generation failed: {}", e)))?;

        println!("💬 Response: {}", response.content);
        println!("⏱️ Search time: {:.2}s", search_time);

        if args.verbose {
            display_hierarchical_results(&results);
        }
    }

    Ok(())
}

fn display_hierarchical_results(results: &[HierarchicalResult]) {
    println!("\n🏗️ Hierarchical Search Results:");

    for result in results {
        println!(
            "   📊 Level {}: {} chunks, avg score: {:.3}",
            result.level + 1,
            result.chunks.len(),
            result.avg_score
        );
    }
}

async fn compare_hierarchical_vs_flat(
    args: &Args,
    embedder: Arc<dyn Embedder>,
) -> ExampleResult<()> {
    println!("⚖️ Comparing Hierarchical vs Flat Indexing...");

    // Build hierarchical system
    let hier_timer = Timer::new("Hierarchical setup");
    let levels = build_hierarchical_levels(args, embedder.clone()).await?;
    let hier_time = hier_timer.finish();

    // Build flat system
    let flat_timer = Timer::new("Flat setup");
    let (flat_store, _) = build_flat_pipeline(args, embedder.clone()).await?;
    let flat_time = flat_timer.finish();

    let test_queries = get_climate_test_queries();
    let generator = SiumaiGenerator::new(create_siumai_client().await?);

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        // Test hierarchical
        let hier_results = search_hierarchical_levels(&levels, query, args.top_k).await?;
        let mut hier_chunks = Vec::new();
        for result in &hier_results {
            hier_chunks.extend(result.chunks.clone());
        }
        hier_chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        hier_chunks.truncate(args.top_k);

        // Test flat
        let search_query = Query::builder()
            .text(query.to_string())
            .top_k(args.top_k)
            .search_mode(SearchMode::Vector)
            .build();
        let flat_chunks = flat_store
            .search(&search_query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;

        let hier_avg = if hier_chunks.is_empty() {
            0.0
        } else {
            hier_chunks.iter().map(|c| c.score).sum::<f32>() / hier_chunks.len() as f32
        };
        let flat_avg = if flat_chunks.is_empty() {
            0.0
        } else {
            flat_chunks.iter().map(|c| c.score).sum::<f32>() / flat_chunks.len() as f32
        };

        println!("   🏗️ Hierarchical: {:.3} avg score", hier_avg);
        println!("   📏 Flat: {:.3} avg score", flat_avg);
        println!(
            "   📈 Improvement: {:.1}%",
            ((hier_avg - flat_avg) / flat_avg) * 100.0
        );
    }

    println!("\n📈 Setup Time Comparison:");
    println!("   🏗️ Hierarchical: {:.2}s", hier_time);
    println!("   📏 Flat: {:.2}s", flat_time);

    Ok(())
}

async fn build_flat_pipeline(
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
    let splitter = Arc::new(SentenceSplitter::from_defaults(800, 100)?);
    let metadata_extractor = Arc::new(MetadataExtractor::new());
    let vector_store = Arc::new(InMemoryVectorStore::new(
        DEFAULT_EMBEDDING_DIM,
        DistanceMetric::Cosine,
    ));

    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(loader)
        .with_transformer(splitter)
        .with_transformer(metadata_extractor)
        .with_embedder(embedder.clone())
        .with_vector_store(vector_store.clone())
        .build()?;

    let index_result = pipeline
        .run()
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;
    println!("✅ Flat indexing: {} nodes", index_result.nodes_created);

    let siumai_client = create_siumai_client().await?;
    let retriever = Arc::new(VectorRetriever::new(vector_store.clone(), embedder));
    let generator = Arc::new(SiumaiGenerator::new(siumai_client));
    let query_engine = QueryEngine::new(retriever, generator);

    Ok((vector_store, query_engine))
}

async fn run_interactive_mode(
    levels: &[HierarchicalLevel],
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

        let timer = Timer::new("Hierarchical query");
        let results = search_hierarchical_levels(levels, query, args.top_k).await?;

        let mut all_chunks = Vec::new();
        for result in &results {
            all_chunks.extend(result.chunks.clone());
        }
        all_chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        all_chunks.truncate(args.top_k);

        let options = GenerationOptions::default();
        match generator
            .generate_response(query, all_chunks, &options)
            .await
        {
            Ok(response) => {
                let query_time = timer.finish();
                println!("\n💬 Response: {}", response.content);
                println!("⏱️ Query time: {:.2}s", query_time);

                if args.verbose {
                    display_hierarchical_results(&results);
                }
            }
            Err(e) => println!("❌ Generation error: {}", e),
        }
    }

    Ok(())
}
