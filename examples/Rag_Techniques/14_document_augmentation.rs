/*!
# Document Augmentation RAG Example

This example demonstrates document augmentation through question generation, which
improves document retrieval by generating and incorporating various questions
related to each text fragment.

Based on: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/document_augmentation.ipynb

## Key Features

- **Question Generation**: Uses LLM to generate multiple questions for each document chunk
- **Enhanced Retrieval**: Stores both original content and generated questions
- **Improved Matching**: Better alignment between user queries and document content
- **Performance Analysis**: Compares augmented vs standard retrieval

## How It Works

1. **Document Processing**: Split documents into chunks
2. **Question Generation**: Generate 3-5 questions per chunk using LLM
3. **Augmented Storage**: Store chunks with their generated questions
4. **Enhanced Retrieval**: Search against both content and questions
5. **Performance Comparison**: Evaluate improvement over standard retrieval

## Usage

```bash
# Basic document augmentation
cargo run --bin document_augmentation --features fastembed

# Adjust question generation parameters
cargo run --bin document_augmentation --features fastembed -- --questions-per-chunk 5

# Compare with standard retrieval
cargo run --bin document_augmentation --features fastembed -- --compare-standard

# Interactive mode
cargo run --bin document_augmentation --features fastembed -- --interactive
```
*/

use clap::Parser;
use std::{collections::HashMap, path::PathBuf, sync::Arc};

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{get_climate_test_queries, setup_logging, ExampleError, ExampleResult, Timer};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, Loader, VectorStore},
    types::ChunkInfo,
    DistanceMetric, Node,
};
use cheungfun_indexing::{
    loaders::DirectoryLoader, node_parser::text::SentenceSplitter,
    pipeline::DefaultIndexingPipeline, transformers::MetadataExtractor, NodeParser,
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    engine::QueryEngine, generator::SiumaiGenerator, retriever::VectorRetriever,
};
use siumai::prelude::*;
use uuid::Uuid;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug)]
#[command(
    name = "document_augmentation",
    about = "Document Augmentation RAG Example - Question generation for enhanced retrieval"
)]
struct Args {
    /// Path to the document to process
    #[arg(long, default_value = "data/Understanding_Climate_Change.pdf")]
    document_path: PathBuf,

    /// Chunk size for document processing
    #[arg(long, default_value = "800")]
    chunk_size: usize,

    /// Chunk overlap
    #[arg(long, default_value = "100")]
    chunk_overlap: usize,

    /// Number of questions to generate per chunk
    #[arg(long, default_value = "3")]
    questions_per_chunk: usize,

    /// Number of documents to retrieve
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Compare with standard retrieval (no augmentation)
    #[arg(long)]
    compare_standard: bool,

    /// Enable interactive mode
    #[arg(long)]
    interactive: bool,

    /// Show detailed augmentation information
    #[arg(long)]
    verbose: bool,
}

/// Augmented document chunk with generated questions
#[derive(Debug, Clone)]
struct AugmentedChunk {
    /// Original content
    original_content: String,
    /// Generated questions
    generated_questions: Vec<String>,
    /// Combined content (original + questions)
    augmented_content: String,
    /// Metadata from original chunk
    metadata: HashMap<String, serde_json::Value>,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    // Setup logging
    setup_logging();

    let args = Args::parse();

    println!("📚 Starting Document Augmentation Example...");
    println!("📖 This example demonstrates question generation for enhanced retrieval");
    println!("🎯 Based on the technique from RAG_Techniques repository\n");

    // Create embedder
    let embedder = create_embedder().await?;
    println!("✅ Embedder initialized");

    if args.compare_standard {
        // Compare augmented vs standard retrieval
        compare_retrieval_methods(&args, embedder).await?;
    } else {
        // Run document augmentation only
        run_document_augmentation(&args, embedder).await?;
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

/// Run document augmentation demonstration
async fn run_document_augmentation(args: &Args, embedder: Arc<dyn Embedder>) -> ExampleResult<()> {
    println!("📚 Running Document Augmentation...");

    let timer = Timer::new("Document augmentation setup");

    // Build augmented indexing pipeline
    let (_augmented_store, query_engine) = build_augmented_pipeline(args, embedder.clone()).await?;

    timer.finish();

    if args.interactive {
        run_interactive_mode(&query_engine).await?;
    } else {
        run_test_queries(&query_engine, args.verbose).await?;
    }

    Ok(())
}

/// Generate questions for a text chunk using LLM
async fn generate_questions_for_chunk(
    content: &str,
    num_questions: usize,
) -> ExampleResult<Vec<String>> {
    // Create question generation prompt
    let _prompt = format!(
        r#"Generate {} diverse, specific questions that can be answered using the following text chunk.
The questions should:
1. Be directly answerable from the content
2. Cover different aspects of the information
3. Use varied question types (what, how, why, when, etc.)
4. Be clear and specific

Text chunk:
{}

Generate exactly {} questions, one per line:"#,
        num_questions, content, num_questions
    );

    // Simulate question generation (in real implementation, would use LLM)
    let questions = simulate_question_generation(content, num_questions);

    Ok(questions)
}

/// Simulate question generation (placeholder for real LLM implementation)
fn simulate_question_generation(content: &str, num_questions: usize) -> Vec<String> {
    let mut questions = Vec::new();

    // Extract key terms and concepts
    let words: Vec<&str> = content.split_whitespace().collect();
    let key_terms: Vec<&str> = words
        .iter()
        .filter(|word| word.len() > 4 && !is_common_word(word))
        .take(10)
        .cloned()
        .collect();

    // Generate different types of questions
    let question_templates = vec![
        "What is {}?",
        "How does {} work?",
        "Why is {} important?",
        "What are the effects of {}?",
        "What causes {}?",
    ];

    for i in 0..num_questions {
        if let Some(term) = key_terms.get(i % key_terms.len()) {
            let template = question_templates[i % question_templates.len()];
            let question = template.replace("{}", term);
            questions.push(question);
        }
    }

    // Ensure we have the requested number of questions
    while questions.len() < num_questions {
        questions.push(format!(
            "What information is provided about the topic in this section?"
        ));
    }

    questions.truncate(num_questions);
    questions
}

/// Check if a word is a common word that shouldn't be used for question generation
fn is_common_word(word: &str) -> bool {
    let common_words = vec![
        "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
        "about", "into", "through", "during", "before", "after", "above", "below", "between",
        "among", "this", "that", "these", "those", "they", "them", "their", "there", "where",
        "when", "what", "which", "who", "whom", "whose", "how", "why", "can", "could", "should",
        "would", "will", "shall", "may", "might", "must", "have", "has", "had", "do", "does",
        "did", "is", "are", "was", "were", "be", "been", "being", "very", "more", "most", "much",
        "many", "some", "any", "all", "each", "every", "other", "another", "such",
    ];

    common_words.contains(&word.to_lowercase().as_str())
}

/// Build augmented indexing pipeline with question generation
async fn build_augmented_pipeline(
    args: &Args,
    embedder: Arc<dyn Embedder>,
) -> ExampleResult<(Arc<dyn VectorStore>, QueryEngine)> {
    // Load documents first to generate questions
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

    // Build initial pipeline to get chunks
    let initial_pipeline = DefaultIndexingPipeline::builder()
        .with_loader(loader)
        .with_document_processor(splitter)  // Documents -> Nodes
        .with_node_processor(metadata_extractor)  // Nodes -> Nodes
        .build()?;

    // Run initial processing to get chunks
    let (_nodes, _initial_result) = initial_pipeline
        .run(None, None, true, true, None, true)
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;

    // We need to get the actual nodes from the pipeline result
    // Since IndexingStats doesn't contain nodes, we need to load them separately
    let loader = Arc::new(DirectoryLoader::new(&data_dir)?);
    let documents = loader
        .load()
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;

    let splitter = Arc::new(SentenceSplitter::from_defaults(
        args.chunk_size,
        args.chunk_overlap,
    )?);

    // Parse documents into nodes
    let nodes = splitter
        .parse_nodes(&documents, false)
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;

    println!("📊 Processing {} chunks for augmentation...", nodes.len());

    // Generate questions for each chunk
    let augmentation_timer = Timer::new("Question generation");
    let mut augmented_chunks = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        if i % 10 == 0 {
            println!("   📝 Processing chunk {} of {}...", i + 1, nodes.len());
        }

        let questions =
            generate_questions_for_chunk(&node.content, args.questions_per_chunk).await?;

        // Create augmented content by combining original content with questions
        let questions_text = questions.join("\n");
        let augmented_content =
            format!("{}\n\nRelated Questions:\n{}", node.content, questions_text);

        let augmented_chunk = AugmentedChunk {
            original_content: node.content.clone(),
            generated_questions: questions,
            augmented_content,
            metadata: node.metadata.clone(),
        };

        augmented_chunks.push(augmented_chunk);
    }

    let augmentation_time = augmentation_timer.finish();
    println!(
        "✅ Question generation completed in {:.2}s",
        augmentation_time.as_secs_f64()
    );
    println!(
        "📊 Generated {} questions total",
        augmented_chunks.len() * args.questions_per_chunk
    );

    // Create nodes from augmented chunks and embed them
    let embedding_timer = Timer::new("Augmented embedding");
    let mut augmented_nodes = Vec::new();

    for (i, augmented_chunk) in augmented_chunks.iter().enumerate() {
        // Create embedding for augmented content
        let embedding = embedder
            .embed(&augmented_chunk.augmented_content)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;

        // Create node with augmented content
        let chunk_info = ChunkInfo::new(Some(0), Some(augmented_chunk.augmented_content.len()), i);
        let mut node = Node::new(
            augmented_chunk.augmented_content.clone(),
            Uuid::new_v4(),
            chunk_info,
        );
        node.metadata = augmented_chunk.metadata.clone();
        node.metadata.insert("augmented".to_string(), true.into());
        node.metadata.insert(
            "questions_count".to_string(),
            augmented_chunk.generated_questions.len().into(),
        );

        // Set embedding on node
        node = node.with_embedding(embedding);

        // Store in vector store
        vector_store
            .add(vec![node.clone()])
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;

        augmented_nodes.push(node);
    }

    let embedding_time = embedding_timer.finish();
    println!(
        "✅ Augmented embedding completed in {:.2}s",
        embedding_time.as_secs_f64()
    );

    // Create query engine
    let retriever = VectorRetriever::new(vector_store.clone(), embedder.clone());

    // Create Siumai client
    let siumai_client = Siumai::builder()
        .openai()
        .build()
        .await
        .map_err(|e| ExampleError::Siumai(e.to_string()))?;

    let generator = SiumaiGenerator::new(siumai_client);
    let query_engine = QueryEngine::new(Arc::new(retriever), Arc::new(generator));

    Ok((vector_store, query_engine))
}

/// Run test queries using document augmentation
async fn run_test_queries(query_engine: &QueryEngine, verbose: bool) -> ExampleResult<()> {
    let test_queries = get_climate_test_queries();

    println!("🔍 Running test queries with Document Augmentation...");

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        let timer = Timer::new("Augmented query");
        let result = query_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;
        let query_time = timer.finish();

        println!("💬 Response: {}", result.response.content);

        if verbose {
            let context = &result.retrieved_nodes;
            println!("📚 Retrieved {} augmented chunks:", context.len());
            for (j, node) in context.iter().enumerate() {
                println!(
                    "   {}. Score: {:.3}, Questions: {}",
                    j + 1,
                    node.score,
                    node.node
                        .metadata
                        .get("questions_count")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0)
                );

                // Show original content (before augmentation)
                let content_preview = if node.node.content.contains("Related Questions:") {
                    node.node
                        .content
                        .split("Related Questions:")
                        .next()
                        .unwrap_or(&node.node.content)
                } else {
                    &node.node.content
                };

                println!(
                    "      Content: {}...",
                    content_preview.chars().take(100).collect::<String>()
                );
            }
        }

        println!("⏱️ Query time: {:.2}s", query_time.as_secs_f64());
    }

    Ok(())
}

/// Compare document augmentation with standard retrieval
async fn compare_retrieval_methods(args: &Args, embedder: Arc<dyn Embedder>) -> ExampleResult<()> {
    println!("⚖️ Comparing Document Augmentation vs Standard Retrieval...");
    println!("═══════════════════════════════════════════════════════════════\n");

    // 1. Build augmented system
    println!("📚 1. Building Document Augmentation System...");
    let augmented_timer = Timer::new("Augmented system setup");
    let (_augmented_store, augmented_engine) =
        build_augmented_pipeline(args, embedder.clone()).await?;
    let augmented_time = augmented_timer.finish();

    // 2. Build standard system
    println!("\n📏 2. Building Standard Retrieval System...");
    let standard_timer = Timer::new("Standard system setup");
    let (_standard_store, standard_engine) =
        build_standard_pipeline(args, embedder.clone()).await?;
    let standard_time = standard_timer.finish();

    // 3. Compare performance on test queries
    println!("\n🎯 3. Performance Comparison...");

    let test_queries = get_climate_test_queries();
    let mut augmented_total_time = 0.0;
    let mut standard_total_time = 0.0;
    let mut augmented_scores = Vec::new();
    let mut standard_scores = Vec::new();

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        // Test augmented retrieval
        let augmented_timer = Timer::new("augmented_query");
        let augmented_result = augmented_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;
        let augmented_time = augmented_timer.finish();
        augmented_total_time += augmented_time.as_secs_f64();

        // Test standard retrieval
        let standard_timer = Timer::new("standard_query");
        let standard_result = standard_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;
        let standard_time = standard_timer.finish();
        standard_total_time += standard_time.as_secs_f64();

        if args.verbose {
            println!(
                "   📚 Augmented response: {}",
                augmented_result
                    .response
                    .content
                    .chars()
                    .take(100)
                    .collect::<String>()
            );
            println!(
                "   📏 Standard response: {}",
                standard_result
                    .response
                    .content
                    .chars()
                    .take(100)
                    .collect::<String>()
            );
        }

        println!(
            "   ⏱️ Augmented time: {:.2}s, Standard time: {:.2}s",
            augmented_time.as_secs_f64(),
            standard_time.as_secs_f64()
        );

        // Calculate average scores
        let context = &augmented_result.retrieved_nodes;
        if !context.is_empty() {
            let avg_score = context.iter().map(|n| n.score).sum::<f32>() / context.len() as f32;
            augmented_scores.push(avg_score);
        }

        let context = &standard_result.retrieved_nodes;
        if !context.is_empty() {
            let avg_score = context.iter().map(|n| n.score).sum::<f32>() / context.len() as f32;
            standard_scores.push(avg_score);
        }
    }

    // Summary statistics
    println!("\n📈 Performance Summary:");
    println!("   ⏱️ Setup time:");
    println!("      📚 Augmented: {:.2}s", augmented_time.as_secs_f64());
    println!("      📏 Standard: {:.2}s", standard_time.as_secs_f64());

    println!("   ⏱️ Average query time:");
    println!(
        "      📚 Augmented: {:.2}s",
        augmented_total_time / test_queries.len() as f64
    );
    println!(
        "      📏 Standard: {:.2}s",
        standard_total_time / test_queries.len() as f64
    );

    if !augmented_scores.is_empty() && !standard_scores.is_empty() {
        let augmented_avg = augmented_scores.iter().sum::<f32>() / augmented_scores.len() as f32;
        let standard_avg = standard_scores.iter().sum::<f32>() / standard_scores.len() as f32;

        println!("   📊 Average similarity scores:");
        println!("      📚 Augmented: {:.3}", augmented_avg);
        println!("      📏 Standard: {:.3}", standard_avg);
        println!(
            "      📈 Improvement: {:.1}%",
            ((augmented_avg - standard_avg) / standard_avg) * 100.0
        );
    }

    Ok(())
}

/// Build standard pipeline for comparison
async fn build_standard_pipeline(
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

    // Create vector store
    let vector_store = Arc::new(InMemoryVectorStore::new(
        DEFAULT_EMBEDDING_DIM,
        DistanceMetric::Cosine,
    ));

    // Build standard pipeline
    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(loader)
        .with_document_processor(splitter)  // Documents -> Nodes
        .with_node_processor(metadata_extractor)  // Nodes -> Nodes
        .with_embedder(embedder.clone())
        .with_vector_store(vector_store.clone())
        .build()?;

    // Run indexing
    let (_nodes, index_result) = pipeline
        .run(None, None, true, true, None, true)
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;
    println!("✅ Standard indexing completed");
    println!("📊 Indexed {} standard chunks", index_result.nodes_created);

    // Create query engine
    let retriever = VectorRetriever::new(vector_store.clone(), embedder.clone());

    // Create Siumai client
    let siumai_client = Siumai::builder()
        .openai()
        .build()
        .await
        .map_err(|e| ExampleError::Siumai(e.to_string()))?;

    let generator = SiumaiGenerator::new(siumai_client);
    let query_engine = QueryEngine::new(Arc::new(retriever), Arc::new(generator));

    Ok((vector_store, query_engine))
}

/// Run interactive mode for testing custom queries
async fn run_interactive_mode(query_engine: &QueryEngine) -> ExampleResult<()> {
    println!("\n🎯 Interactive Mode - Enter your queries (type 'quit' to exit):");

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
        match query_engine.query(query).await {
            Ok(result) => {
                let query_time = timer.finish();
                println!("\n💬 Response: {}", result.response.content);
                println!("⏱️ Query time: {:.2}s", query_time.as_secs_f64());

                let context = &result.retrieved_nodes;
                if !context.is_empty() {
                    println!(
                        "📚 Used {} augmented chunks with avg score: {:.3}",
                        context.len(),
                        context.iter().map(|n| n.score).sum::<f32>() / context.len() as f32
                    );
                }
            }
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }
    }

    Ok(())
}
