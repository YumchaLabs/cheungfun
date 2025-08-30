/*!
# Relevant Segment Extraction (RSE) RAG Example

This example demonstrates Relevant Segment Extraction, which dynamically constructs
multi-chunk segments of text that are relevant to a given query.

Based on: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/relevant_segment_extraction.ipynb

## Key Features

- **Multi-chunk Segments**: Combines adjacent relevant chunks into longer segments
- **Dynamic Construction**: Builds segments based on query relevance rather than fixed boundaries
- **Context Expansion**: Provides more complete context by including neighboring chunks
- **Relevance Scoring**: Uses similarity scores to determine segment boundaries

## How It Works

1. **Initial Retrieval**: Retrieve top-N chunks using standard vector search
2. **Adjacency Analysis**: Identify chunks that are adjacent in the original document
3. **Relevance Scoring**: Calculate relevance scores for potential segments
4. **Segment Construction**: Merge adjacent chunks that exceed relevance threshold
5. **Context Enhancement**: Provide expanded context for better LLM responses

## Usage

```bash
# Basic relevant segment extraction
cargo run --bin relevant_segment_extraction --features fastembed

# Compare with standard chunk retrieval
cargo run --bin relevant_segment_extraction --features fastembed -- --compare-standard

# Adjust segment parameters
cargo run --bin relevant_segment_extraction --features fastembed -- --segment-threshold 0.7 --max-segment-size 3

# Interactive mode
cargo run --bin relevant_segment_extraction --features fastembed -- --interactive
```
*/

use clap::Parser;
use std::{collections::HashMap, path::PathBuf, sync::Arc};

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{get_climate_test_queries, setup_logging, ExampleError, ExampleResult, Timer};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, ResponseGenerator, VectorStore},
    types::{Query, SearchMode},
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
#[command(
    name = "relevant_segment_extraction",
    about = "Relevant Segment Extraction RAG Example - Dynamic multi-chunk segment construction"
)]
struct Args {
    /// Path to the document to process
    #[arg(long, default_value = "data/Understanding_Climate_Change.pdf")]
    document_path: PathBuf,

    /// Chunk size for document processing
    #[arg(long, default_value = "600")]
    chunk_size: usize,

    /// Chunk overlap
    #[arg(long, default_value = "100")]
    chunk_overlap: usize,

    /// Number of initial chunks to retrieve
    #[arg(long, default_value = "10")]
    initial_retrieval_count: usize,

    /// Final number of segments to return
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Relevance threshold for segment construction
    #[arg(long, default_value = "0.75")]
    segment_threshold: f32,

    /// Maximum number of chunks per segment
    #[arg(long, default_value = "4")]
    max_segment_size: usize,

    /// Compare with standard chunk retrieval
    #[arg(long)]
    compare_standard: bool,

    /// Enable interactive mode
    #[arg(long)]
    interactive: bool,

    /// Show detailed segment information
    #[arg(long)]
    verbose: bool,
}

/// Represents a constructed segment from multiple chunks
#[derive(Debug, Clone)]
struct RelevantSegment {
    /// Combined content from multiple chunks
    content: String,
    /// Average relevance score
    avg_score: f32,
    /// Number of chunks in this segment
    chunk_count: usize,
    /// Source chunk indices
    source_indices: Vec<usize>,
    /// Metadata from constituent chunks
    metadata: HashMap<String, serde_json::Value>,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    // Setup logging
    setup_logging();

    let args = Args::parse();

    println!("🎯 Starting Relevant Segment Extraction Example...");
    println!("📖 This example demonstrates dynamic multi-chunk segment construction");
    println!("🔗 Based on the technique from RAG_Techniques repository\n");

    // Create embedder
    let embedder = create_embedder().await?;
    println!("✅ Embedder initialized");

    if args.compare_standard {
        // Compare RSE with standard chunk retrieval
        compare_retrieval_methods(&args, embedder).await?;
    } else {
        // Run RSE only
        run_segment_extraction(&args, embedder).await?;
    }

    Ok(())
}

/// Create embedder with fallback strategy
async fn create_embedder() -> ExampleResult<Arc<dyn Embedder>> {
    println!("🔧 Initializing embedder...");

    match FastEmbedder::new().await {
        Ok(embedder) => {
            println!("✅ Using FastEmbed (local)");
            Ok(Arc::new(embedder))
        }
        Err(e) => {
            println!("⚠️ FastEmbed failed: {}", e);
            Err(ExampleError::Embedder(format!(
                "Failed to create embedder: {}",
                e
            )))
        }
    }
}

/// Run relevant segment extraction demonstration
async fn run_segment_extraction(args: &Args, embedder: Arc<dyn Embedder>) -> ExampleResult<()> {
    println!("🎯 Running Relevant Segment Extraction...");

    let timer = Timer::new("RSE setup");

    // Build indexing pipeline
    let (vector_store, _) = build_indexing_pipeline(args, embedder.clone()).await?;

    timer.finish();

    if args.interactive {
        run_interactive_mode(&vector_store, embedder, args).await?;
    } else {
        run_test_queries(&vector_store, embedder, args).await?;
    }

    Ok(())
}

/// Build indexing pipeline
async fn build_indexing_pipeline(
    args: &Args,
    embedder: Arc<dyn Embedder>,
) -> ExampleResult<(Arc<dyn VectorStore>, QueryEngine)> {
    // Ensure we have the correct path to the data directory
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

    // Create vector store
    let vector_store = Arc::new(InMemoryVectorStore::new(
        DEFAULT_EMBEDDING_DIM,
        DistanceMetric::Cosine,
    ));

    // Build pipeline
    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(loader)
        .with_transformer(splitter)
        .with_transformer(metadata_extractor)
        .with_embedder(embedder.clone())
        .with_vector_store(vector_store.clone())
        .build()?;

    // Run indexing
    let indexing_timer = Timer::new("Indexing");
    let index_result = pipeline
        .run()
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;
    let indexing_time = indexing_timer.finish();

    println!(
        "✅ Indexing completed in {:.2}s",
        indexing_time.as_secs_f64()
    );
    println!("📊 Indexed {} nodes", index_result.nodes_created);

    // Create query engine
    let retriever = VectorRetriever::new(vector_store.clone(), embedder.clone());

    // Create Siumai client for generation
    let siumai_client = siumai::prelude::Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()))
        .model("gpt-4o-mini")
        .temperature(0.7)
        .max_tokens(1500)
        .build()
        .await
        .map_err(|e| ExampleError::Siumai(e.to_string()))?;

    let generator = SiumaiGenerator::new(siumai_client);
    let query_engine = QueryEngine::new(Arc::new(retriever), Arc::new(generator));

    Ok((vector_store, query_engine))
}

/// Extract relevant segments from retrieved chunks
async fn extract_relevant_segments(
    chunks: Vec<ScoredNode>,
    args: &Args,
) -> ExampleResult<Vec<RelevantSegment>> {
    println!(
        "🔗 Extracting relevant segments from {} chunks...",
        chunks.len()
    );

    // Group chunks by document source for adjacency analysis
    let mut doc_chunks: HashMap<String, Vec<(usize, ScoredNode)>> = HashMap::new();

    for (idx, chunk) in chunks.into_iter().enumerate() {
        let doc_id = chunk
            .node
            .metadata
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        doc_chunks
            .entry(doc_id)
            .or_insert_with(Vec::new)
            .push((idx, chunk));
    }

    let mut segments = Vec::new();

    // Process each document separately
    for (doc_id, mut chunks) in doc_chunks {
        // Sort chunks by their position in the document (using chunk_index if available)
        chunks.sort_by_key(|(_, chunk)| {
            chunk
                .node
                .metadata
                .get("chunk_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize
        });

        println!(
            "📄 Processing document: {} ({} chunks)",
            doc_id,
            chunks.len()
        );

        // Build segments from adjacent chunks
        let doc_segments = build_segments_from_chunks(chunks, args)?;
        segments.extend(doc_segments);
    }

    // Sort segments by average score and take top-k
    segments.sort_by(|a, b| b.avg_score.partial_cmp(&a.avg_score).unwrap());
    segments.truncate(args.top_k);

    println!("✅ Constructed {} relevant segments", segments.len());

    Ok(segments)
}

/// Build segments from adjacent chunks
fn build_segments_from_chunks(
    chunks: Vec<(usize, ScoredNode)>,
    args: &Args,
) -> ExampleResult<Vec<RelevantSegment>> {
    let mut segments = Vec::new();
    let mut i = 0;

    while i < chunks.len() {
        let mut current_segment_chunks = vec![chunks[i].clone()];
        let mut j = i + 1;

        // Try to extend the segment with adjacent chunks
        while j < chunks.len() && current_segment_chunks.len() < args.max_segment_size {
            let current_score = chunks[j].1.score;

            // Check if this chunk should be included in the segment
            if current_score >= args.segment_threshold {
                current_segment_chunks.push(chunks[j].clone());
                j += 1;
            } else {
                break;
            }
        }

        // Create segment if it meets criteria
        if !current_segment_chunks.is_empty() {
            let segment = create_segment_from_chunks(current_segment_chunks)?;
            segments.push(segment);
        }

        i = j.max(i + 1); // Move to next unprocessed chunk
    }

    Ok(segments)
}

/// Create a segment from a collection of chunks
fn create_segment_from_chunks(chunks: Vec<(usize, ScoredNode)>) -> ExampleResult<RelevantSegment> {
    let combined_content = chunks
        .iter()
        .map(|(_, chunk)| chunk.node.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    let avg_score = chunks.iter().map(|(_, chunk)| chunk.score).sum::<f32>() / chunks.len() as f32;

    let source_indices = chunks.iter().map(|(idx, _)| *idx).collect();

    // Combine metadata from all chunks
    let mut combined_metadata = HashMap::new();
    for (_, chunk) in &chunks {
        for (key, value) in &chunk.node.metadata {
            combined_metadata.insert(key.clone(), value.clone());
        }
    }

    Ok(RelevantSegment {
        content: combined_content,
        avg_score,
        chunk_count: chunks.len(),
        source_indices,
        metadata: combined_metadata,
    })
}

/// Run test queries using relevant segment extraction
async fn run_test_queries(
    vector_store: &Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedder>,
    args: &Args,
) -> ExampleResult<()> {
    let test_queries = get_climate_test_queries();

    println!("🔍 Running test queries with Relevant Segment Extraction...");

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        let timer = Timer::new("RSE query");

        // Step 1: Initial retrieval
        let search_query = Query::builder()
            .text(query.to_string())
            .top_k(args.initial_retrieval_count)
            .search_mode(SearchMode::Vector)
            .build();

        let initial_chunks = vector_store
            .search(&search_query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;

        // Step 2: Extract relevant segments
        let segments = extract_relevant_segments(initial_chunks, args).await?;

        let query_time = timer.finish();

        // Display results
        display_segment_results(&segments, args.verbose);

        println!("⏱️ Query time: {:.2}s", query_time.as_secs_f64());
    }

    Ok(())
}

/// Display segment extraction results
fn display_segment_results(segments: &[RelevantSegment], verbose: bool) {
    println!("🔗 Extracted {} relevant segments:", segments.len());

    for (i, segment) in segments.iter().enumerate() {
        println!(
            "\n   📄 Segment {}: {} chunks, avg score: {:.3}",
            i + 1,
            segment.chunk_count,
            segment.avg_score
        );

        if verbose {
            println!("      Content length: {} chars", segment.content.len());
            println!("      Source indices: {:?}", segment.source_indices);
            println!(
                "      Preview: {}...",
                segment.content.chars().take(150).collect::<String>()
            );
        } else {
            println!(
                "      Preview: {}...",
                segment.content.chars().take(100).collect::<String>()
            );
        }
    }
}

/// Compare RSE with standard chunk retrieval
async fn compare_retrieval_methods(args: &Args, embedder: Arc<dyn Embedder>) -> ExampleResult<()> {
    println!("⚖️ Comparing Relevant Segment Extraction vs Standard Retrieval...");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Build indexing pipeline
    let (vector_store, standard_engine) = build_indexing_pipeline(args, embedder.clone()).await?;

    let test_queries = get_climate_test_queries();

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        // 1. Standard retrieval
        println!("\n📏 Standard Chunk Retrieval:");
        let standard_timer = Timer::new("standard_retrieval");
        let standard_result = standard_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;
        let standard_time = standard_timer.finish();

        println!(
            "   💬 Response: {}",
            standard_result
                .response
                .content
                .chars()
                .take(100)
                .collect::<String>()
        );
        println!(
            "   📊 Retrieved {} chunks, avg score: {:.3}",
            standard_result.retrieved_nodes.len(),
            standard_result
                .retrieved_nodes
                .iter()
                .map(|n| n.score)
                .sum::<f32>()
                / standard_result.retrieved_nodes.len() as f32
        );
        println!("   ⏱️ Time: {:.2}s", standard_time.as_secs_f64());

        // 2. Relevant Segment Extraction
        println!("\n🔗 Relevant Segment Extraction:");
        let rse_timer = Timer::new("rse_retrieval");

        let search_query = Query::builder()
            .text(query.to_string())
            .top_k(args.initial_retrieval_count)
            .search_mode(SearchMode::Vector)
            .build();

        let initial_chunks = vector_store
            .search(&search_query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;

        let segments = extract_relevant_segments(initial_chunks, args).await?;
        let rse_time = rse_timer.finish();

        // Generate response using segments
        let siumai_client = siumai::prelude::Siumai::builder()
            .openai()
            .api_key(&std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()))
            .model("gpt-4o-mini")
            .temperature(0.7)
            .max_tokens(1500)
            .build()
            .await
            .map_err(|e| ExampleError::Siumai(e.to_string()))?;

        let generator = SiumaiGenerator::new(siumai_client);
        let segment_context = segments
            .iter()
            .map(|s| s.content.clone())
            .collect::<Vec<_>>()
            .join("\n\n");

        let rse_response = generator
            .generate_response(
                query,
                vec![],
                &cheungfun_core::types::GenerationOptions::default(),
            )
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;

        println!(
            "   💬 Response: {}",
            rse_response.content.chars().take(100).collect::<String>()
        );
        println!(
            "   📊 Constructed {} segments, avg score: {:.3}",
            segments.len(),
            segments.iter().map(|s| s.avg_score).sum::<f32>() / segments.len() as f32
        );
        println!("   ⏱️ Time: {:.2}s", rse_time.as_secs_f64());

        // Show improvement
        let standard_avg_score = standard_result
            .retrieved_nodes
            .iter()
            .map(|n| n.score)
            .sum::<f32>()
            / standard_result.retrieved_nodes.len() as f32;
        let rse_avg_score =
            segments.iter().map(|s| s.avg_score).sum::<f32>() / segments.len() as f32;

        println!(
            "   📈 Score improvement: {:.1}%",
            ((rse_avg_score - standard_avg_score) / standard_avg_score) * 100.0
        );
    }

    Ok(())
}

/// Run interactive mode for testing custom queries
async fn run_interactive_mode(
    vector_store: &Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedder>,
    args: &Args,
) -> ExampleResult<()> {
    println!("\n🎯 Interactive Mode - Enter your queries (type 'quit' to exit):");

    let siumai_client = siumai::prelude::Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()))
        .model("gpt-4o-mini")
        .temperature(0.7)
        .max_tokens(1500)
        .build()
        .await
        .map_err(|e| ExampleError::Siumai(e.to_string()))?;

    let generator = SiumaiGenerator::new(siumai_client);

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
            println!("👋 Goodbye!");
            break;
        }

        let timer = Timer::new("interactive_query");

        // Perform relevant segment extraction
        let search_query = Query::builder()
            .text(query.to_string())
            .top_k(args.initial_retrieval_count)
            .search_mode(SearchMode::Vector)
            .build();

        match vector_store.search(&search_query).await {
            Ok(initial_chunks) => {
                let segments = extract_relevant_segments(initial_chunks, args).await?;
                let query_time = timer.finish();

                // Generate response using segments
                let segment_context = segments
                    .iter()
                    .map(|s| s.content.clone())
                    .collect::<Vec<_>>()
                    .join("\n\n");

                match generator
                    .generate_response(
                        query,
                        vec![],
                        &cheungfun_core::types::GenerationOptions::default(),
                    )
                    .await
                {
                    Ok(response) => {
                        println!("\n💬 Response: {}", response.content);
                        println!("⏱️ Query time: {:.2}s", query_time.as_secs_f64());

                        display_segment_results(&segments, args.verbose);
                    }
                    Err(e) => {
                        println!("❌ Generation error: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("❌ Retrieval error: {}", e);
            }
        }
    }

    Ok(())
}
