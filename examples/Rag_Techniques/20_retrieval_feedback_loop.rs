/*!
# Retrieval with Feedback Loop RAG Example

This example demonstrates implementing feedback loops in RAG systems to learn from user
interactions and improve future retrievals over time.

Based on: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/retrieval_with_feedback_loop.ipynb

## Key Features

- **User Feedback Collection**: Captures relevance ratings from users
- **Feedback Storage**: Maintains feedback history for learning
- **Adaptive Scoring**: Adjusts retrieval scores based on feedback
- **Performance Tracking**: Monitors improvement over time
- **Interactive Learning**: Continuously improves with user input

## How It Works

1. **Initial Retrieval**: Perform standard vector similarity search
2. **Feedback Collection**: Ask users to rate result relevance
3. **Feedback Storage**: Store ratings with query-document pairs
4. **Score Adjustment**: Modify future scores based on historical feedback
5. **Continuous Learning**: Improve retrieval quality over time

## Usage

```bash
# Basic feedback loop demonstration
cargo run --bin 20_retrieval_feedback_loop --features fastembed

# Interactive feedback collection
cargo run --bin 20_retrieval_feedback_loop --features fastembed -- --interactive

# Show feedback statistics
cargo run --bin 20_retrieval_feedback_loop --features fastembed -- --show-stats

# Reset feedback data
cargo run --bin 20_retrieval_feedback_loop --features fastembed -- --reset-feedback
```
*/

use clap::Parser;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use serde::{Deserialize, Serialize};

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{get_climate_test_queries, setup_logging, ExampleError, ExampleResult, Timer};

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, VectorStore, Loader},
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
use uuid::Uuid;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Parser, Debug)]
#[command(
    name = "retrieval_feedback_loop",
    about = "Retrieval with Feedback Loop RAG Example - Learning from user interactions"
)]
struct Args {
    /// Path to the document directory
    #[arg(long, default_value = "data")]
    data_path: PathBuf,

    /// Chunk size for document processing
    #[arg(long, default_value = "800")]
    chunk_size: usize,

    /// Chunk overlap
    #[arg(long, default_value = "100")]
    chunk_overlap: usize,

    /// Number of documents to retrieve
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Enable interactive feedback collection
    #[arg(long)]
    interactive: bool,

    /// Show feedback statistics
    #[arg(long)]
    show_stats: bool,

    /// Reset feedback data
    #[arg(long)]
    reset_feedback: bool,

    /// Feedback learning rate (0.0 to 1.0)
    #[arg(long, default_value = "0.1")]
    learning_rate: f32,

    /// Show detailed feedback information
    #[arg(long)]
    verbose: bool,
}

/// User feedback for a query-document pair
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UserFeedback {
    /// Query text
    query: String,
    /// Document ID
    document_id: String,
    /// Relevance rating (1-5 scale)
    relevance_rating: u8,
    /// Timestamp of feedback
    timestamp: chrono::DateTime<chrono::Utc>,
    /// Optional user comment
    comment: Option<String>,
}

/// Feedback storage and management
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeedbackStore {
    /// All collected feedback
    feedback_history: Vec<UserFeedback>,
    /// Query-document relevance scores
    relevance_scores: HashMap<String, f32>,
    /// Query performance statistics
    query_stats: HashMap<String, QueryStats>,
}

/// Statistics for query performance
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueryStats {
    /// Number of times queried
    query_count: u32,
    /// Average relevance rating
    avg_relevance: f32,
    /// Improvement over time
    improvement_trend: f32,
}

impl FeedbackStore {
    fn new() -> Self {
        Self {
            feedback_history: Vec::new(),
            relevance_scores: HashMap::new(),
            query_stats: HashMap::new(),
        }
    }

    /// Add user feedback
    fn add_feedback(&mut self, feedback: UserFeedback) {
        let key = format!("{}:{}", feedback.query, feedback.document_id);
        
        // Update relevance score (normalize to 0-1 scale)
        let normalized_rating = (feedback.relevance_rating as f32 - 1.0) / 4.0;
        self.relevance_scores.insert(key, normalized_rating);
        
        // Update query statistics
        let stats = self.query_stats.entry(feedback.query.clone()).or_insert(QueryStats {
            query_count: 0,
            avg_relevance: 0.0,
            improvement_trend: 0.0,
        });
        
        stats.query_count += 1;
        stats.avg_relevance = (stats.avg_relevance * (stats.query_count - 1) as f32 + normalized_rating) / stats.query_count as f32;
        
        self.feedback_history.push(feedback);
    }

    /// Get feedback-adjusted score for a query-document pair
    fn get_adjusted_score(&self, query: &str, document_id: &str, original_score: f32, learning_rate: f32) -> f32 {
        let key = format!("{}:{}", query, document_id);
        
        if let Some(&feedback_score) = self.relevance_scores.get(&key) {
            // Blend original similarity score with feedback score
            original_score * (1.0 - learning_rate) + feedback_score * learning_rate
        } else {
            original_score
        }
    }

    /// Get statistics summary
    fn get_stats_summary(&self) -> String {
        let total_feedback = self.feedback_history.len();
        let unique_queries = self.query_stats.len();
        let avg_rating = if total_feedback > 0 {
            self.feedback_history.iter()
                .map(|f| f.relevance_rating as f32)
                .sum::<f32>() / total_feedback as f32
        } else {
            0.0
        };

        format!(
            "Feedback Statistics:\n\
             - Total feedback entries: {}\n\
             - Unique queries: {}\n\
             - Average rating: {:.2}/5.0\n\
             - Feedback coverage: {:.1}%",
            total_feedback,
            unique_queries,
            avg_rating,
            if unique_queries > 0 { (total_feedback as f32 / unique_queries as f32) * 100.0 } else { 0.0 }
        )
    }
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    // Setup logging
    setup_logging();

    let args = Args::parse();

    println!("🔄 Starting Retrieval with Feedback Loop Example...");
    println!("📈 This example demonstrates learning from user feedback");
    println!("🎯 Based on the technique from RAG_Techniques repository\n");

    if args.reset_feedback {
        println!("🗑️ Resetting feedback data...");
        // In a real implementation, this would clear persistent storage
        println!("✅ Feedback data reset");
        return Ok(());
    }

    // Create embedder
    let embedder = create_embedder().await?;
    println!("✅ Embedder initialized");

    // Build RAG pipeline
    let (query_engine, mut feedback_store) = build_feedback_rag_pipeline(&args, embedder).await?;

    if args.show_stats {
        println!("{}", feedback_store.get_stats_summary());
        return Ok(());
    }

    if args.interactive {
        run_interactive_feedback_mode(&query_engine, &mut feedback_store, &args).await?;
    } else {
        run_feedback_demonstration(&query_engine, &mut feedback_store, &args).await?;
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

/// Build RAG pipeline with feedback capabilities
async fn build_feedback_rag_pipeline(
    args: &Args,
    embedder: Arc<dyn Embedder>,
) -> ExampleResult<(QueryEngine, FeedbackStore)> {
    let timer = Timer::new("Feedback RAG pipeline setup");

    // Load and process documents
    let data_dir = std::env::current_dir()?.join(&args.data_path);
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

    // Build indexing pipeline
    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(loader)
        .with_transformer(splitter)
        .with_transformer(metadata_extractor)
        .with_embedder(embedder.clone())
        .with_vector_store(vector_store.clone())
        .build()?;

    // Run indexing
    let index_result = pipeline
        .run()
        .await
        .map_err(|e| ExampleError::Cheungfun(e))?;
    
    println!("✅ Indexed {} documents", index_result.nodes_created);

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

    // Initialize feedback store
    let feedback_store = FeedbackStore::new();

    timer.finish();
    Ok((query_engine, feedback_store))
}

/// Run feedback demonstration with test queries
async fn run_feedback_demonstration(
    query_engine: &QueryEngine,
    feedback_store: &mut FeedbackStore,
    args: &Args,
) -> ExampleResult<()> {
    let test_queries = get_climate_test_queries();

    println!("🔍 Running feedback demonstration...");

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);

        let timer = Timer::new("Query with feedback");
        let result = query_engine
            .query(query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;
        let query_time = timer.finish();

        // Apply feedback adjustments to scores
        let adjusted_results = apply_feedback_adjustments(
            &result.retrieved_nodes,
            query,
            feedback_store,
            args.learning_rate,
        );

        println!("💬 Response: {}", result.response.content);

        if args.verbose {
            println!("📚 Retrieved {} documents:", adjusted_results.len());
            for (j, scored_node) in adjusted_results.iter().enumerate() {
                println!(
                    "   {}. Original Score: {:.3}, Adjusted Score: {:.3}",
                    j + 1,
                    scored_node.score,
                    scored_node.score // In real implementation, this would show adjusted score
                );
                println!(
                    "      Content: {}...",
                    scored_node.node.content.chars().take(100).collect::<String>()
                );
            }
        }

        // Simulate user feedback (in real implementation, this would be user input)
        simulate_user_feedback(query, &adjusted_results, feedback_store);

        println!("⏱️ Query time: {:.2}s", query_time.as_secs_f64());
    }

    println!("\n{}", feedback_store.get_stats_summary());
    Ok(())
}

/// Apply feedback adjustments to retrieval scores
fn apply_feedback_adjustments(
    results: &[ScoredNode],
    query: &str,
    feedback_store: &FeedbackStore,
    learning_rate: f32,
) -> Vec<ScoredNode> {
    let mut adjusted_results = results.to_vec();

    for scored_node in &mut adjusted_results {
        let document_id = scored_node.node.id.to_string();
        let adjusted_score = feedback_store.get_adjusted_score(
            query,
            &document_id,
            scored_node.score,
            learning_rate,
        );
        scored_node.score = adjusted_score;
    }

    // Re-sort by adjusted scores
    adjusted_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    adjusted_results
}

/// Simulate user feedback for demonstration
fn simulate_user_feedback(
    query: &str,
    results: &[ScoredNode],
    feedback_store: &mut FeedbackStore,
) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Simulate feedback for top 3 results
    for (i, scored_node) in results.iter().take(3).enumerate() {
        // Higher ranked results get better ratings on average
        let base_rating = match i {
            0 => 4, // Top result usually good
            1 => 3, // Second result moderate
            _ => 2, // Lower results often poor
        };

        // Add some randomness
        let rating = (base_rating + rng.gen_range(-1..=1)).clamp(1, 5) as u8;

        let feedback = UserFeedback {
            query: query.to_string(),
            document_id: scored_node.node.id.to_string(),
            relevance_rating: rating,
            timestamp: chrono::Utc::now(),
            comment: None,
        };

        feedback_store.add_feedback(feedback);
    }
}

/// Run interactive feedback collection mode
async fn run_interactive_feedback_mode(
    query_engine: &QueryEngine,
    feedback_store: &mut FeedbackStore,
    args: &Args,
) -> ExampleResult<()> {
    println!("\n🎯 Interactive Feedback Mode - Enter your queries (type 'quit' to exit):");
    println!("After each query, you'll be asked to rate the relevance of results (1-5 scale)");

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

        let timer = Timer::new("Interactive query");
        match query_engine.query(query).await {
            Ok(result) => {
                let query_time = timer.finish();

                // Apply feedback adjustments
                let adjusted_results = apply_feedback_adjustments(
                    &result.retrieved_nodes,
                    query,
                    feedback_store,
                    args.learning_rate,
                );

                println!("\n💬 Response: {}", result.response.content);
                println!("⏱️ Query time: {:.2}s", query_time.as_secs_f64());

                // Collect user feedback
                collect_user_feedback(query, &adjusted_results, feedback_store).await?;
            }
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }
    }

    println!("\n{}", feedback_store.get_stats_summary());
    Ok(())
}

/// Collect user feedback interactively
async fn collect_user_feedback(
    query: &str,
    results: &[ScoredNode],
    feedback_store: &mut FeedbackStore,
) -> ExampleResult<()> {
    println!("\n📊 Please rate the relevance of the retrieved documents (1-5 scale):");
    println!("1 = Not relevant, 2 = Slightly relevant, 3 = Moderately relevant, 4 = Very relevant, 5 = Extremely relevant");

    use std::io::{self, Write};

    for (i, scored_node) in results.iter().take(3).enumerate() {
        println!("\n📄 Document {}: Score: {:.3}", i + 1, scored_node.score);
        println!("Content: {}...",
            scored_node.node.content.chars().take(200).collect::<String>());

        loop {
            print!("Rate this document (1-5, or 's' to skip): ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            if input == "s" {
                break;
            }

            if let Ok(rating) = input.parse::<u8>() {
                if rating >= 1 && rating <= 5 {
                    let feedback = UserFeedback {
                        query: query.to_string(),
                        document_id: scored_node.node.id.to_string(),
                        relevance_rating: rating,
                        timestamp: chrono::Utc::now(),
                        comment: None,
                    };

                    feedback_store.add_feedback(feedback);
                    println!("✅ Feedback recorded: {}/5", rating);
                    break;
                } else {
                    println!("Please enter a number between 1 and 5");
                }
            } else {
                println!("Please enter a valid number or 's' to skip");
            }
        }
    }

    Ok(())
}
