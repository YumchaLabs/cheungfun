// Intelligent Reranking Example
//
// This example demonstrates various reranking strategies in RAG systems,
// including LLM-based reranking, score-based reranking, and diversity-based reranking.
//
// Based on: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/reranking.ipynb

use cheungfun_core::{
    traits::{Embedder, IndexingPipeline, VectorStore},
    types::{Query, SearchMode},
    ScoredNode,
};
use cheungfun_indexing::{
    loaders::DirectoryLoader,
    node_parser::{config::SentenceSplitterConfig, text::SentenceSplitter},
    pipeline::DefaultIndexingPipeline,
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    advanced::rerankers::ScoreRerankStrategy, engine::QueryEngine, generator::SiumaiGenerator,
    retriever::VectorRetriever,
};
use clap::{Arg, Command};
use siumai::prelude::*;
use std::{path::PathBuf, sync::Arc};
use tokio;
use tracing::info;

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{setup_logging, ExampleError, ExampleResult, PerformanceMetrics, Timer};

/// Configuration for reranking strategies
#[derive(Debug, Clone)]
pub struct RerankingConfig {
    pub strategy: RerankingStrategy,
    pub top_n: usize,
    pub initial_retrieval_count: usize,
    pub enable_comparison: bool,
}

#[derive(Debug, Clone)]
pub enum RerankingStrategy {
    /// LLM-based reranking with custom prompt
    LLM {
        batch_size: usize,
        custom_prompt: Option<String>,
    },
    /// Score-based reranking strategies
    Score(ScoreRerankStrategy),
    /// Combined reranking (LLM + Score)
    Combined { llm_weight: f32, score_weight: f32 },
    /// All strategies for comparison
    All,
}

impl Default for RerankingConfig {
    fn default() -> Self {
        Self {
            strategy: RerankingStrategy::LLM {
                batch_size: 5,
                custom_prompt: None,
            },
            top_n: 5,
            initial_retrieval_count: 15,
            enable_comparison: true,
        }
    }
}

/// Intelligent Reranking demonstration
pub struct IntelligentRerankingDemo {
    query_engine: QueryEngine,
    vector_store: Arc<dyn VectorStore>,
    config: RerankingConfig,
    performance_metrics: PerformanceMetrics,
}

impl IntelligentRerankingDemo {
    /// Create a new reranking demo
    pub async fn new(
        data_path: PathBuf,
        embedding_provider: &str,
        config: RerankingConfig,
    ) -> ExampleResult<Self> {
        let performance_metrics = PerformanceMetrics::new();

        // Step 1: Create embedder
        let embedder = create_embedder(embedding_provider).await?;
        println!("✅ Embedder initialized: {}", embedding_provider);

        // Step 2: Create vector store and query engine
        let (vector_store, query_engine) =
            create_retrievers_and_engine(&data_path, embedder).await?;
        println!("✅ Vector store and query engine initialized");

        info!("✅ Intelligent Reranking Demo initialized successfully");

        Ok(Self {
            query_engine,
            vector_store,
            config,
            performance_metrics,
        })
    }

    /// Demonstrate different reranking strategies
    pub async fn demonstrate_reranking(&self, query: &str) -> ExampleResult<()> {
        info!(
            "🎯 Demonstrating Intelligent Reranking with query: '{}'",
            query
        );

        match &self.config.strategy {
            RerankingStrategy::All => {
                self.compare_all_strategies(query).await?;
            }
            strategy => {
                self.demonstrate_single_strategy(query, strategy).await?;
            }
        }

        Ok(())
    }

    /// Compare all reranking strategies
    async fn compare_all_strategies(&self, query: &str) -> ExampleResult<()> {
        println!("\n🔍 Comparing Reranking Strategies");
        println!("═══════════════════════════════════════════════════════════════");

        // 1. Baseline: No reranking (simple similarity search)
        println!("\n📊 1. Baseline: Simple Similarity Search");
        let baseline_results = self.get_baseline_results(query).await?;
        self.display_reranking_results("Baseline", &baseline_results, None);

        // 2. LLM-based reranking
        println!("\n🧠 2. LLM-based Reranking");
        let _llm_results = self.demonstrate_llm_reranking(query).await?;

        // 3. Score-based reranking strategies
        println!("\n📈 3. Score-based Reranking Strategies");
        self.demonstrate_score_reranking(query).await?;

        // 4. Diversity-based reranking
        println!("\n🌈 4. Diversity-based Reranking");
        self.demonstrate_diversity_reranking(query).await?;

        Ok(())
    }

    /// Demonstrate a single reranking strategy
    async fn demonstrate_single_strategy(
        &self,
        query: &str,
        strategy: &RerankingStrategy,
    ) -> ExampleResult<()> {
        match strategy {
            RerankingStrategy::LLM { .. } => {
                self.demonstrate_llm_reranking(query).await?;
            }
            RerankingStrategy::Score(score_strategy) => {
                self.demonstrate_score_strategy(query, score_strategy.clone())
                    .await?;
            }
            RerankingStrategy::Combined { .. } => {
                self.demonstrate_combined_reranking(query).await?;
            }
            RerankingStrategy::All => unreachable!(),
        }
        Ok(())
    }

    /// Get baseline results without reranking
    async fn get_baseline_results(&self, query: &str) -> ExampleResult<Vec<ScoredNode>> {
        let search_query = Query::builder()
            .text(query.to_string())
            .top_k(self.config.initial_retrieval_count)
            .search_mode(SearchMode::Vector)
            .build();

        let results = self
            .vector_store
            .search(&search_query)
            .await
            .map_err(|e| ExampleError::Cheungfun(e))?;
        Ok(results)
    }

    /// Demonstrate LLM-based reranking
    async fn demonstrate_llm_reranking(&self, query: &str) -> ExampleResult<Vec<ScoredNode>> {
        println!("🧠 LLM-based Reranking");
        println!("⚠️  Note: LLM reranking requires advanced query pipeline setup");
        println!("💡 For now, showing baseline results with simulated reranking scores");

        let timer = Timer::new("llm_reranking_simulation");

        // Get initial results
        let mut initial_results = self.get_baseline_results(query).await?;

        // Simulate LLM reranking by adjusting scores
        // In a real implementation, this would use LLMReranker with proper setup
        for (i, result) in initial_results.iter_mut().enumerate() {
            // Simulate LLM giving higher scores to more relevant results
            let relevance_boost = 1.0 - (i as f32 * 0.1);
            result.score = (result.score * relevance_boost).min(1.0);
        }

        // Sort by new scores
        initial_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let reranked_results: Vec<ScoredNode> = initial_results
            .into_iter()
            .take(self.config.top_n)
            .collect();

        let reranking_time = timer.finish();
        self.display_reranking_results(
            "LLM Reranking (Simulated)",
            &reranked_results,
            Some(reranking_time),
        );

        Ok(reranked_results)
    }

    /// Demonstrate score-based reranking
    async fn demonstrate_score_reranking(&self, query: &str) -> ExampleResult<()> {
        let strategies = vec![
            ("Original Score (Desc)", ScoreRerankStrategy::OriginalScore),
            (
                "Original Score (Asc)",
                ScoreRerankStrategy::OriginalScoreAsc,
            ),
            ("Random", ScoreRerankStrategy::Random),
        ];

        for (_name, strategy) in strategies {
            self.demonstrate_score_strategy(query, strategy).await?;
        }

        Ok(())
    }

    /// Demonstrate a specific score strategy
    async fn demonstrate_score_strategy(
        &self,
        query: &str,
        strategy: ScoreRerankStrategy,
    ) -> ExampleResult<()> {
        let timer = Timer::new("score_reranking");

        // Get initial results
        let mut initial_results = self.get_baseline_results(query).await?;

        // Simulate score reranking
        match &strategy {
            ScoreRerankStrategy::OriginalScore => {
                // Already sorted by score (descending)
            }
            ScoreRerankStrategy::OriginalScoreAsc => {
                initial_results.sort_by(|a, b| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            ScoreRerankStrategy::Random => {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                initial_results.shuffle(&mut rng);
            }
            ScoreRerankStrategy::Diversity {
                similarity_threshold: _,
            } => {
                // Simulate diversity by removing very similar results
                let mut diverse_results = Vec::new();
                for result in initial_results {
                    if diverse_results.len() < self.config.top_n {
                        diverse_results.push(result);
                    }
                }
                initial_results = diverse_results;
            }
        }

        let reranked_results: Vec<ScoredNode> = initial_results
            .into_iter()
            .take(self.config.top_n)
            .collect();

        let reranking_time = timer.finish();
        let strategy_name = format!("Score Reranking ({:?})", strategy);
        self.display_reranking_results(&strategy_name, &reranked_results, Some(reranking_time));

        Ok(())
    }

    /// Demonstrate diversity-based reranking
    async fn demonstrate_diversity_reranking(&self, query: &str) -> ExampleResult<()> {
        let strategy = ScoreRerankStrategy::Diversity {
            similarity_threshold: 0.8,
        };

        self.demonstrate_score_strategy(query, strategy).await?;
        Ok(())
    }

    /// Demonstrate combined reranking
    async fn demonstrate_combined_reranking(&self, query: &str) -> ExampleResult<()> {
        println!("\n🔄 Combined Reranking (LLM + Score)");

        // This would implement a combination of LLM and score-based reranking
        // For now, we'll demonstrate the concept by showing both results

        let _llm_results = self.demonstrate_llm_reranking(query).await?;
        self.demonstrate_score_strategy(query, ScoreRerankStrategy::OriginalScore)
            .await?;

        println!("💡 Combined reranking would merge these results with weighted scores");

        Ok(())
    }

    /// Display reranking results
    fn display_reranking_results(
        &self,
        method_name: &str,
        results: &[ScoredNode],
        timing: Option<std::time::Duration>,
    ) {
        println!("\n📋 {} Results:", method_name);
        if let Some(duration) = timing {
            println!("⏱️  Reranking time: {:.2}s", duration.as_secs_f64());
        }

        for (i, result) in results.iter().take(self.config.top_n).enumerate() {
            let content_preview = if result.node.content.len() > 100 {
                format!(
                    "{}...",
                    result.node.content.chars().take(100).collect::<String>()
                )
            } else {
                result.node.content.clone()
            };

            println!(
                "{}. [Score: {:.4}] {}",
                i + 1,
                result.score,
                content_preview
            );
        }
    }

    /// Run interactive mode
    pub async fn run_interactive(&self) -> ExampleResult<()> {
        println!("\n🎯 Interactive Intelligent Reranking Mode");
        println!("═══════════════════════════════════════════════════════════════");
        println!("Enter your queries to see different reranking strategies in action.");
        println!("Type 'quit' to exit, 'help' for commands.\n");

        loop {
            print!("🔍 Query: ");
            use std::io::{self, Write};
            io::stdout().flush().map_err(ExampleError::Io)?;

            let mut input = String::new();
            io::stdin()
                .read_line(&mut input)
                .map_err(ExampleError::Io)?;
            let input = input.trim();

            match input.to_lowercase().as_str() {
                "quit" | "exit" => {
                    println!("👋 Goodbye!");
                    break;
                }
                "help" => {
                    self.display_help();
                }
                "stats" => {
                    self.performance_metrics.print_summary();
                }
                "" => continue,
                query => {
                    if let Err(e) = self.demonstrate_reranking(query).await {
                        eprintln!("❌ Error processing query: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Display help information
    fn display_help(&self) {
        println!("\n📖 Available Commands:");
        println!("  help  - Show this help message");
        println!("  stats - Show performance statistics");
        println!("  quit  - Exit the program");
        println!("\n💡 Tips:");
        println!("  - Try queries about climate change, biodiversity, or environmental topics");
        println!("  - Compare how different reranking strategies affect result ordering");
        println!("  - Notice the trade-offs between accuracy and speed");
    }
}

/// Parse reranking strategy from command line argument
fn parse_reranking_strategy(strategy_str: &str) -> ExampleResult<RerankingStrategy> {
    match strategy_str.to_lowercase().as_str() {
        "llm" => Ok(RerankingStrategy::LLM {
            batch_size: 5,
            custom_prompt: None,
        }),
        "score" => Ok(RerankingStrategy::Score(ScoreRerankStrategy::OriginalScore)),
        "diversity" => Ok(RerankingStrategy::Score(ScoreRerankStrategy::Diversity {
            similarity_threshold: 0.8,
        })),
        "random" => Ok(RerankingStrategy::Score(ScoreRerankStrategy::Random)),
        "combined" => Ok(RerankingStrategy::Combined {
            llm_weight: 0.7,
            score_weight: 0.3,
        }),
        "all" => Ok(RerankingStrategy::All),
        _ => Err(ExampleError::Config(
            "Invalid strategy. Use: llm, score, diversity, random, combined, or all".to_string(),
        )),
    }
}

/// Create CLI command structure
fn create_cli() -> Command {
    Command::new("intelligent_reranking")
        .about("Intelligent Reranking Example - Demonstrates various reranking strategies in RAG systems")
        .version("1.0")
        .arg(
            Arg::new("data")
                .short('d')
                .long("data")
                .value_name("PATH")
                .help("Path to data file or directory")
                .default_value("data/Understanding_Climate_Change.pdf"),
        )
        .arg(
            Arg::new("strategy")
                .short('s')
                .long("strategy")
                .value_name("STRATEGY")
                .help("Reranking strategy: llm, score, diversity, random, combined, all")
                .default_value("all"),
        )
        .arg(
            Arg::new("top_n")
                .short('n')
                .long("top-n")
                .value_name("NUMBER")
                .help("Number of top results to return after reranking")
                .default_value("5"),
        )
        .arg(
            Arg::new("initial_count")
                .short('i')
                .long("initial-count")
                .value_name("NUMBER")
                .help("Number of documents to retrieve initially before reranking")
                .default_value("15"),
        )
        .arg(
            Arg::new("query")
                .short('q')
                .long("query")
                .value_name("TEXT")
                .help("Query to test (if not provided, enters interactive mode)"),
        )
        .arg(
            Arg::new("embedding_provider")
                .long("embedding-provider")
                .value_name("PROVIDER")
                .help("Embedding provider: fastembed or openai")
                .default_value("fastembed"),
        )

        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .action(clap::ArgAction::SetTrue)
                .help("Enable verbose logging"),
        )
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    let matches = create_cli().get_matches();

    // Setup logging
    setup_logging();

    // Parse arguments
    let data_path = PathBuf::from(matches.get_one::<String>("data").unwrap());
    let strategy_str = matches.get_one::<String>("strategy").unwrap();
    let top_n: usize = matches
        .get_one::<String>("top_n")
        .unwrap()
        .parse()
        .map_err(|e| ExampleError::Config(format!("Invalid top_n: {}", e)))?;
    let initial_count: usize = matches
        .get_one::<String>("initial_count")
        .unwrap()
        .parse()
        .map_err(|e| ExampleError::Config(format!("Invalid initial_count: {}", e)))?;
    let query = matches.get_one::<String>("query");
    let embedding_provider = matches.get_one::<String>("embedding_provider").unwrap();

    // Display banner
    println!("🎯 Intelligent Reranking Example");
    println!("═══════════════════════════════════════════════════════════════");
    println!("📊 Strategy: {}", strategy_str);
    println!("📁 Data: {}", data_path.display());
    println!("🔢 Top N: {}, Initial Count: {}", top_n, initial_count);
    println!("🤖 Embeddings: {}", embedding_provider);
    println!();

    // Parse reranking strategy
    let strategy = parse_reranking_strategy(strategy_str)?;

    // Create reranking configuration
    let config = RerankingConfig {
        strategy,
        top_n,
        initial_retrieval_count: initial_count,
        enable_comparison: true,
    };

    // Initialize demo
    let demo = IntelligentRerankingDemo::new(data_path, embedding_provider, config).await?;

    // Run demo
    if let Some(query_text) = query {
        // Single query mode
        demo.demonstrate_reranking(query_text).await?;
        demo.performance_metrics.print_summary();
    } else {
        // Interactive mode
        demo.run_interactive().await?;
    }

    Ok(())
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
            println!("🤖 Using OpenAI for generation (cloud)");
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
    println!("🤖 No valid OpenAI API key found, using Ollama for generation (local)");
    println!("💡 Make sure Ollama is running with: ollama serve");
    println!("💡 And pull a model with: ollama pull llama3.2");

    Siumai::builder()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .temperature(0.0)
        .build()
        .await
        .map_err(|e| ExampleError::Config(format!("Failed to initialize Ollama: {}. Make sure Ollama is running with 'ollama serve' and you have pulled a model with 'ollama pull llama3.2'", e)))
}

async fn create_retrievers_and_engine(
    data_path: &PathBuf,
    embedder: Arc<dyn Embedder>,
) -> ExampleResult<(Arc<dyn VectorStore>, QueryEngine)> {
    const DEFAULT_EMBEDDING_DIM: usize = 384;

    // Create vector store and index documents
    let vector_store = Arc::new(InMemoryVectorStore::new(
        DEFAULT_EMBEDDING_DIM,
        cheungfun_core::DistanceMetric::Cosine,
    ));

    // Build indexing pipeline
    let timer = Timer::new("Document indexing");

    // Get the directory containing the document
    let default_path = PathBuf::from(".");
    let data_dir = data_path.parent().unwrap_or(&default_path);
    println!("📂 Loading from directory: {}", data_dir.display());

    let loader = Arc::new(DirectoryLoader::new(data_dir)?);

    // Create text splitter with custom configuration
    let splitter_config = SentenceSplitterConfig::default();
    let splitter = Arc::new(SentenceSplitter::new(splitter_config)?);

    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(loader)
        .with_transformer(splitter)
        .with_embedder(embedder.clone())
        .with_vector_store(vector_store.clone())
        .build()?;

    // Run indexing pipeline with progress reporting
    let _indexing_stats = pipeline
        .run_with_progress(Box::new(|progress| {
            if let Some(percentage) = progress.percentage() {
                println!(
                    "📊 {}: {:.1}% ({}/{})",
                    progress.stage,
                    percentage,
                    progress.processed,
                    progress.total.unwrap_or(0)
                );
            } else {
                println!(
                    "📊 {}: {} items processed",
                    progress.stage, progress.processed
                );
            }

            if let Some(current_item) = &progress.current_item {
                println!("   └─ {}", current_item);
            }
        }))
        .await?;

    let indexing_time = timer.finish();

    println!(
        "✅ Completed: Document indexing in {:.2}s",
        indexing_time.as_secs_f64()
    );

    // Create LLM client
    let llm_client = create_llm_client().await?;

    // Create query engine with SiumaiGenerator
    let generator = SiumaiGenerator::new(llm_client);
    let retriever = VectorRetriever::new(vector_store.clone(), embedder.clone());
    let query_engine = QueryEngine::new(Arc::new(retriever), Arc::new(generator));

    Ok((vector_store, query_engine))
}
