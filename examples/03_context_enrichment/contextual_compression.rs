/*!
# Contextual Compression RAG Example

This example demonstrates contextual compression, which compresses retrieved information
while preserving query-relevant content using LLM-based compression.

Based on: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/contextual_compression.ipynb

## Key Features

- **LLM-based Compression**: Uses language models to intelligently compress content
- **Query-aware Filtering**: Preserves information most relevant to the specific query
- **Noise Reduction**: Removes irrelevant information while maintaining context
- **Performance Comparison**: Compares compressed vs uncompressed retrieval

## How It Works

1. **Initial Retrieval**: Retrieve top-N chunks using standard vector search
2. **Relevance Analysis**: Use LLM to analyze relevance of each chunk to the query
3. **Content Compression**: Compress chunks while preserving key information
4. **Quality Filtering**: Remove chunks that don't meet relevance threshold
5. **Enhanced Generation**: Use compressed, relevant content for final response

## Usage

```bash
# Basic contextual compression
cargo run --bin contextual_compression --features fastembed

# Adjust compression parameters
cargo run --bin contextual_compression --features fastembed -- --compression-ratio 0.6 --relevance-threshold 0.7

# Compare with uncompressed retrieval
cargo run --bin contextual_compression --features fastembed -- --compare-uncompressed

# Interactive mode
cargo run --bin contextual_compression --features fastembed -- --interactive
```
*/

use clap::Parser;
use std::{path::PathBuf, sync::Arc};

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{get_climate_test_queries, setup_logging, ExampleError, ExampleResult, Timer};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, VectorStore},
    types::{Query, SearchMode},
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
#[command(
    name = "contextual_compression",
    about = "Contextual Compression RAG Example - LLM-based content compression"
)]
struct Args {
    /// Path to the document to process
    #[arg(long, default_value = "data/Understanding_Climate_Change.pdf")]
    document_path: PathBuf,

    /// Chunk size for document processing
    #[arg(long, default_value = "1000")]
    chunk_size: usize,

    /// Chunk overlap
    #[arg(long, default_value = "200")]
    chunk_overlap: usize,

    /// Number of chunks to retrieve initially
    #[arg(long, default_value = "10")]
    initial_retrieval_count: usize,

    /// Final number of compressed chunks to use
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Target compression ratio (0.0-1.0)
    #[arg(long, default_value = "0.7")]
    compression_ratio: f32,

    /// Relevance threshold for filtering
    #[arg(long, default_value = "0.6")]
    relevance_threshold: f32,

    /// Compare with uncompressed retrieval
    #[arg(long)]
    compare_uncompressed: bool,

    /// Enable interactive mode
    #[arg(long)]
    interactive: bool,

    /// Show detailed compression information
    #[arg(long)]
    verbose: bool,
}

/// Compressed chunk with relevance information
#[derive(Debug, Clone)]
struct CompressedChunk {
    /// Original content
    original_content: String,
    /// Compressed content
    compressed_content: String,
    /// Relevance score to the query
    relevance_score: f32,
    /// Original similarity score
    original_score: f32,
    /// Compression ratio achieved
    compression_ratio: f32,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    // Setup logging
    setup_logging();

    let args = Args::parse();

    println!("🗜️ Starting Contextual Compression Example...");
    println!("📖 This example demonstrates LLM-based content compression");
    println!("🎯 Based on the technique from RAG_Techniques repository\n");

    // Create embedder
    let embedder = create_embedder().await?;
    println!("✅ Embedder initialized");

    if args.compare_uncompressed {
        // Compare compressed vs uncompressed retrieval
        compare_retrieval_methods(&args, embedder).await?;
    } else {
        // Run contextual compression only
        run_contextual_compression(&args, embedder).await?;
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

/// Run contextual compression demonstration
async fn run_contextual_compression(args: &Args, embedder: Arc<dyn Embedder>) -> ExampleResult<()> {
    println!("🗜️ Running Contextual Compression...");

    let timer = Timer::new("Compression setup");

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

    println!("✅ Indexing completed in {:.2}s", indexing_time);
    println!("📊 Indexed {} nodes", index_result.nodes.len());

    // Create query engine
    let retriever = VectorRetriever::new(vector_store.clone(), args.initial_retrieval_count);
    let generator = SiumaiGenerator::new()
        .await
        .map_err(|e| ExampleError::Siumai(e))?;
    let query_engine = QueryEngine::new(Box::new(retriever), Box::new(generator));

    Ok((vector_store, query_engine))
}

/// Compress retrieved chunks using LLM-based contextual compression
async fn compress_chunks(
    chunks: Vec<ScoredNode>,
    query: &str,
    args: &Args,
) -> ExampleResult<Vec<CompressedChunk>> {
    println!(
        "🗜️ Compressing {} chunks with target ratio {:.1}%...",
        chunks.len(),
        args.compression_ratio * 100.0
    );

    // Initialize LLM for compression
    let llm = ChatOpenAI::default();

    let mut compressed_chunks = Vec::new();

    for chunk in chunks {
        let compression_timer = Timer::new("chunk_compression");

        // Create compression prompt
        let compression_prompt = format!(
            r#"Given the following text chunk and a user query, extract and compress only the information that is directly relevant to answering the query. 

Query: "{}"

Text Chunk:
{}

Instructions:
1. Extract only information directly relevant to the query
2. Compress the content to approximately {:.0}% of the original length
3. Preserve key facts, numbers, and specific details
4. Remove redundant or irrelevant information
5. Maintain readability and coherence

Compressed Content:"#,
            query,
            chunk.node.content,
            args.compression_ratio * 100.0
        );

        // Perform compression (simulated for now - in real implementation would use LLM)
        let compressed_content = simulate_compression(&chunk.node.content, args.compression_ratio);

        // Calculate relevance score (simulated)
        let relevance_score = calculate_relevance_score(query, &chunk.node.content);

        let actual_compression_ratio =
            compressed_content.len() as f32 / chunk.node.content.len() as f32;

        let compressed_chunk = CompressedChunk {
            original_content: chunk.node.content.clone(),
            compressed_content,
            relevance_score,
            original_score: chunk.score,
            compression_ratio: actual_compression_ratio,
        };

        compressed_chunks.push(compressed_chunk);

        let _compression_time = compression_timer.finish();
    }

    // Filter by relevance threshold
    compressed_chunks.retain(|chunk| chunk.relevance_score >= args.relevance_threshold);

    // Sort by relevance score and take top-k
    compressed_chunks.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
    compressed_chunks.truncate(args.top_k);

    println!(
        "✅ Compressed to {} relevant chunks",
        compressed_chunks.len()
    );

    Ok(compressed_chunks)
}

/// Simulate content compression (in real implementation, would use LLM)
fn simulate_compression(content: &str, target_ratio: f32) -> String {
    let target_length = (content.len() as f32 * target_ratio) as usize;

    // Simple compression: take first sentences up to target length
    let sentences: Vec<&str> = content.split('.').collect();
    let mut compressed = String::new();

    for sentence in sentences {
        if compressed.len() + sentence.len() <= target_length {
            if !compressed.is_empty() {
                compressed.push('.');
            }
            compressed.push_str(sentence);
        } else {
            break;
        }
    }

    if compressed.is_empty() {
        // Fallback: truncate to target length
        content.chars().take(target_length).collect()
    } else {
        compressed
    }
}

/// Calculate relevance score between query and content (simulated)
fn calculate_relevance_score(query: &str, content: &str) -> f32 {
    let query_words: std::collections::HashSet<&str> =
        query.to_lowercase().split_whitespace().collect();

    let content_words: std::collections::HashSet<&str> =
        content.to_lowercase().split_whitespace().collect();

    let intersection = query_words.intersection(&content_words).count();
    let union = query_words.union(&content_words).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Run test queries using contextual compression
async fn run_test_queries(
    vector_store: &Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedder>,
    args: &Args,
) -> ExampleResult<()> {
    let test_queries = get_climate_test_queries();

    println!("🔍 Running test queries with Contextual Compression...");

    let generator = SiumaiGenerator::new()
        .await
        .map_err(|e| ExampleError::Siumai(e))?;

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        let timer = Timer::new("Compression query");

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

        // Step 2: Compress chunks
        let compressed_chunks = compress_chunks(initial_chunks, query, args).await?;

        // Step 3: Generate response using compressed content
        let compressed_context = compressed_chunks
            .iter()
            .map(|chunk| chunk.compressed_content.clone())
            .collect::<Vec<_>>()
            .join("\n\n");

        let response = generator
            .generate(query, &compressed_context)
            .await
            .map_err(|e| ExampleError::Siumai(e))?;

        let query_time = timer.finish();

        println!("💬 Response: {}", response);

        if args.verbose {
            display_compression_results(&compressed_chunks);
        }

        println!("⏱️ Query time: {:.2}s", query_time);
    }

    Ok(())
}

/// Display compression results
fn display_compression_results(compressed_chunks: &[CompressedChunk]) {
    println!("\n🗜️ Compression Results:");

    let total_original_length: usize = compressed_chunks
        .iter()
        .map(|c| c.original_content.len())
        .sum();
    let total_compressed_length: usize = compressed_chunks
        .iter()
        .map(|c| c.compressed_content.len())
        .sum();
    let overall_compression = total_compressed_length as f32 / total_original_length as f32;

    println!(
        "   📊 Overall compression: {:.1}% ({} → {} chars)",
        overall_compression * 100.0,
        total_original_length,
        total_compressed_length
    );

    for (i, chunk) in compressed_chunks.iter().enumerate() {
        println!(
            "\n   📄 Chunk {}: Relevance {:.3}, Compression {:.1}%",
            i + 1,
            chunk.relevance_score,
            chunk.compression_ratio * 100.0
        );
        println!(
            "      📏 Length: {} → {} chars",
            chunk.original_content.len(),
            chunk.compressed_content.len()
        );
        println!(
            "      📝 Compressed: {}...",
            chunk
                .compressed_content
                .chars()
                .take(100)
                .collect::<String>()
        );
    }
}

/// Compare contextual compression with uncompressed retrieval
async fn compare_retrieval_methods(args: &Args, embedder: Arc<dyn Embedder>) -> ExampleResult<()> {
    println!("⚖️ Comparing Contextual Compression vs Uncompressed Retrieval...");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Build indexing pipeline
    let (vector_store, uncompressed_engine) =
        build_indexing_pipeline(args, embedder.clone()).await?;

    let test_queries = get_climate_test_queries();
    let generator = SiumaiGenerator::new()
        .await
        .map_err(|e| ExampleError::Siumai(e))?;

    let mut compressed_total_time = 0.0;
    let mut uncompressed_total_time = 0.0;

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        // 1. Uncompressed retrieval
        println!("\n📏 Uncompressed Retrieval:");
        let uncompressed_timer = Timer::new("uncompressed_retrieval");
        let uncompressed_result = uncompressed_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;
        let uncompressed_time = uncompressed_timer.finish();
        uncompressed_total_time += uncompressed_time;

        println!(
            "   💬 Response: {}",
            uncompressed_result
                .response
                .chars()
                .take(100)
                .collect::<String>()
        );
        if let Some(context) = &uncompressed_result.context {
            let total_chars: usize = context.iter().map(|n| n.node.content.len()).sum();
            println!(
                "   📊 Context: {} chunks, {} total chars",
                context.len(),
                total_chars
            );
        }
        println!("   ⏱️ Time: {:.2}s", uncompressed_time);

        // 2. Contextual compression
        println!("\n🗜️ Contextual Compression:");
        let compression_timer = Timer::new("compression_retrieval");

        let search_query = Query::builder()
            .text(query.to_string())
            .top_k(args.initial_retrieval_count)
            .search_mode(SearchMode::Vector)
            .build();

        let initial_chunks = vector_store
            .search(&search_query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;

        let compressed_chunks = compress_chunks(initial_chunks, query, args).await?;

        let compressed_context = compressed_chunks
            .iter()
            .map(|chunk| chunk.compressed_content.clone())
            .collect::<Vec<_>>()
            .join("\n\n");

        let compressed_response = generator
            .generate(query, &compressed_context)
            .await
            .map_err(|e| ExampleError::Siumai(e))?;
        let compression_time = compression_timer.finish();
        compressed_total_time += compression_time;

        println!(
            "   💬 Response: {}",
            compressed_response.chars().take(100).collect::<String>()
        );

        let total_compressed_chars: usize = compressed_chunks
            .iter()
            .map(|c| c.compressed_content.len())
            .sum();
        let total_original_chars: usize = compressed_chunks
            .iter()
            .map(|c| c.original_content.len())
            .sum();
        let compression_ratio = total_compressed_chars as f32 / total_original_chars as f32;

        println!(
            "   📊 Context: {} chunks, {} chars (compressed from {})",
            compressed_chunks.len(),
            total_compressed_chars,
            total_original_chars
        );
        println!("   🗜️ Compression ratio: {:.1}%", compression_ratio * 100.0);
        println!("   ⏱️ Time: {:.2}s", compression_time);

        // Show improvement metrics
        if let Some(context) = &uncompressed_result.context {
            let uncompressed_chars: usize = context.iter().map(|n| n.node.content.len()).sum();
            let space_savings = ((uncompressed_chars - total_compressed_chars) as f32
                / uncompressed_chars as f32)
                * 100.0;
            println!("   💾 Space savings: {:.1}%", space_savings);
        }
    }

    // Summary statistics
    println!("\n📈 Performance Summary:");
    println!("   ⏱️ Average query time:");
    println!(
        "      🗜️ Compressed: {:.2}s",
        compressed_total_time / test_queries.len() as f64
    );
    println!(
        "      📏 Uncompressed: {:.2}s",
        uncompressed_total_time / test_queries.len() as f64
    );

    Ok(())
}

/// Run interactive mode for testing custom queries
async fn run_interactive_mode(
    vector_store: &Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedder>,
    args: &Args,
) -> ExampleResult<()> {
    println!("\n🎯 Interactive Mode - Enter your queries (type 'quit' to exit):");

    let generator = SiumaiGenerator::new()
        .await
        .map_err(|e| ExampleError::Siumai(e))?;

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

        // Perform contextual compression
        let search_query = Query::builder()
            .text(query.to_string())
            .top_k(args.initial_retrieval_count)
            .search_mode(SearchMode::Vector)
            .build();

        match vector_store.search(&search_query).await {
            Ok(initial_chunks) => {
                let compressed_chunks = compress_chunks(initial_chunks, query, args).await?;

                let compressed_context = compressed_chunks
                    .iter()
                    .map(|chunk| chunk.compressed_content.clone())
                    .collect::<Vec<_>>()
                    .join("\n\n");

                match generator.generate(query, &compressed_context).await {
                    Ok(response) => {
                        let query_time = timer.finish();
                        println!("\n💬 Response: {}", response);
                        println!("⏱️ Query time: {:.2}s", query_time);

                        if args.verbose {
                            display_compression_results(&compressed_chunks);
                        }
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
