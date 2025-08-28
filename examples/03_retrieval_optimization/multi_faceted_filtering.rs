// Multi-faceted Filtering Example
//
// This example demonstrates various filtering strategies for improving RAG retrieval quality,
// including contextual compression, metadata filtering, score-based filtering, and content quality filtering.
//
// Based on: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/contextual_compression.ipynb
//           https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/relevant_segment_extraction.ipynb

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
    engine::QueryEngine, generator::SiumaiGenerator, retriever::VectorRetriever,
};
use clap::{Arg, Command};
use siumai::prelude::*;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tokio;
use tracing::info;

// Add the shared module
#[path = "../shared/mod.rs"]
mod shared;

use shared::{setup_logging, ExampleError, ExampleResult, PerformanceMetrics, Timer};

/// Configuration for multi-faceted filtering strategies
#[derive(Debug, Clone)]
pub struct FilteringConfig {
    pub strategies: Vec<FilteringStrategy>,
    pub top_n: usize,
    pub initial_retrieval_count: usize,
    pub enable_comparison: bool,
}

#[derive(Debug, Clone)]
pub enum FilteringStrategy {
    /// Metadata-based filtering
    Metadata { filters: HashMap<String, String> },
    /// Score-based filtering with threshold
    ScoreThreshold { min_score: f32 },
    /// Content quality filtering
    ContentQuality {
        min_length: usize,
        max_length: usize,
        require_complete_sentences: bool,
    },
    /// Contextual compression filtering
    ContextualCompression { compression_ratio: f32 },
    /// Relevant segment extraction
    RelevantSegments {
        segment_threshold: f32,
        max_segment_size: usize,
    },
    /// Combined filtering (all strategies)
    Combined,
}

impl Default for FilteringConfig {
    fn default() -> Self {
        Self {
            strategies: vec![FilteringStrategy::Combined],
            top_n: 5,
            initial_retrieval_count: 20,
            enable_comparison: true,
        }
    }
}

/// Multi-faceted Filtering demonstration
pub struct MultiFacetedFilteringDemo {
    query_engine: QueryEngine,
    vector_store: Arc<dyn VectorStore>,
    config: FilteringConfig,
    performance_metrics: PerformanceMetrics,
}

impl MultiFacetedFilteringDemo {
    /// Create a new filtering demo
    pub async fn new(
        data_path: PathBuf,
        embedding_provider: &str,
        config: FilteringConfig,
    ) -> ExampleResult<Self> {
        let performance_metrics = PerformanceMetrics::new();

        // Step 1: Create embedder
        let embedder = create_embedder(embedding_provider).await?;
        println!("✅ Embedder initialized: {}", embedding_provider);

        // Step 2: Create vector store and query engine
        let (vector_store, query_engine) =
            create_retrievers_and_engine(&data_path, embedder).await?;
        println!("✅ Vector store and query engine initialized");

        info!("✅ Multi-faceted Filtering Demo initialized successfully");

        Ok(Self {
            query_engine,
            vector_store,
            config,
            performance_metrics,
        })
    }

    /// Demonstrate different filtering strategies
    pub async fn demonstrate_filtering(&self, query: &str) -> ExampleResult<()> {
        info!(
            "🎯 Demonstrating Multi-faceted Filtering with query: '{}'",
            query
        );

        if self.config.strategies.len() == 1
            && matches!(self.config.strategies[0], FilteringStrategy::Combined)
        {
            self.compare_all_strategies(query).await?;
        } else {
            for strategy in &self.config.strategies {
                self.demonstrate_single_strategy(query, strategy).await?;
            }
        }

        Ok(())
    }

    /// Compare all filtering strategies
    async fn compare_all_strategies(&self, query: &str) -> ExampleResult<()> {
        println!("\n🔍 Comparing Multi-faceted Filtering Strategies");
        println!("═══════════════════════════════════════════════════════════════");

        // 1. Baseline: No filtering (simple similarity search)
        println!("\n📊 1. Baseline: No Filtering");
        let baseline_results = self.get_baseline_results(query).await?;
        self.display_filtering_results("Baseline", &baseline_results, None);

        // 2. Metadata filtering
        println!("\n🏷️ 2. Metadata Filtering");
        self.demonstrate_metadata_filtering(query).await?;

        // 3. Score-based filtering
        println!("\n📈 3. Score-based Filtering");
        self.demonstrate_score_filtering(query).await?;

        // 4. Content quality filtering
        println!("\n✨ 4. Content Quality Filtering");
        self.demonstrate_content_quality_filtering(query).await?;

        // 5. Contextual compression
        println!("\n🗜️ 5. Contextual Compression");
        self.demonstrate_contextual_compression(query).await?;

        // 6. Relevant segment extraction
        println!("\n🎯 6. Relevant Segment Extraction");
        self.demonstrate_segment_extraction(query).await?;

        Ok(())
    }

    /// Demonstrate a single filtering strategy
    async fn demonstrate_single_strategy(
        &self,
        query: &str,
        strategy: &FilteringStrategy,
    ) -> ExampleResult<()> {
        match strategy {
            FilteringStrategy::Metadata { .. } => {
                self.demonstrate_metadata_filtering(query).await?;
            }
            FilteringStrategy::ScoreThreshold { .. } => {
                self.demonstrate_score_filtering(query).await?;
            }
            FilteringStrategy::ContentQuality { .. } => {
                self.demonstrate_content_quality_filtering(query).await?;
            }
            FilteringStrategy::ContextualCompression { .. } => {
                self.demonstrate_contextual_compression(query).await?;
            }
            FilteringStrategy::RelevantSegments { .. } => {
                self.demonstrate_segment_extraction(query).await?;
            }
            FilteringStrategy::Combined => {
                self.compare_all_strategies(query).await?;
            }
        }
        Ok(())
    }

    /// Get baseline results without filtering
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

    /// Demonstrate metadata filtering
    async fn demonstrate_metadata_filtering(&self, query: &str) -> ExampleResult<()> {
        println!("🏷️ Metadata-based Filtering");
        println!("💡 Filtering based on document metadata (source, type, etc.)");

        let timer = Timer::new("metadata_filtering");

        // Get initial results
        let mut initial_results = self.get_baseline_results(query).await?;

        // Apply metadata filtering (simulate filtering by document source)
        let filtered_results: Vec<ScoredNode> = initial_results
            .into_iter()
            .filter(|result| {
                // Example: Filter by document source or other metadata
                if let Some(source) = result.node.metadata.get("source") {
                    // Keep documents from PDF sources (more authoritative)
                    if let Some(source_str) = source.as_str() {
                        source_str.contains(".pdf")
                            || source_str.contains("Understanding_Climate_Change")
                    } else {
                        true
                    }
                } else {
                    true // Keep if no source metadata
                }
            })
            .take(self.config.top_n)
            .collect();

        let filtering_time = timer.finish();
        self.display_filtering_results(
            "Metadata Filtering",
            &filtered_results,
            Some(filtering_time),
        );

        Ok(())
    }

    /// Demonstrate score-based filtering
    async fn demonstrate_score_filtering(&self, query: &str) -> ExampleResult<()> {
        println!("📈 Score-based Filtering");
        println!("💡 Filtering based on similarity score threshold");

        let timer = Timer::new("score_filtering");

        // Get initial results
        let initial_results = self.get_baseline_results(query).await?;

        // Apply score threshold filtering
        let score_threshold = 0.7; // Only keep results with score > 0.7
        let filtered_results: Vec<ScoredNode> = initial_results
            .into_iter()
            .filter(|result| result.score > score_threshold)
            .take(self.config.top_n)
            .collect();

        let filtering_time = timer.finish();
        println!("   📊 Score threshold: {:.2}", score_threshold);
        println!("   📉 Results after filtering: {}", filtered_results.len());
        self.display_filtering_results("Score Filtering", &filtered_results, Some(filtering_time));

        Ok(())
    }

    /// Demonstrate content quality filtering
    async fn demonstrate_content_quality_filtering(&self, query: &str) -> ExampleResult<()> {
        println!("✨ Content Quality Filtering");
        println!("💡 Filtering based on content length and completeness");

        let timer = Timer::new("quality_filtering");

        // Get initial results
        let initial_results = self.get_baseline_results(query).await?;

        // Apply content quality filtering
        let min_length = 50; // Minimum content length
        let max_length = 2000; // Maximum content length

        let filtered_results: Vec<ScoredNode> = initial_results
            .into_iter()
            .filter(|result| {
                let content_len = result.node.content.len();
                let has_complete_sentences = result.node.content.contains('.')
                    || result.node.content.contains('!')
                    || result.node.content.contains('?');

                content_len >= min_length && content_len <= max_length && has_complete_sentences
            })
            .take(self.config.top_n)
            .collect();

        let filtering_time = timer.finish();
        println!(
            "   📏 Length range: {}-{} characters",
            min_length, max_length
        );
        println!("   📝 Require complete sentences: Yes");
        println!("   📉 Results after filtering: {}", filtered_results.len());
        self.display_filtering_results(
            "Content Quality Filtering",
            &filtered_results,
            Some(filtering_time),
        );

        Ok(())
    }

    /// Demonstrate contextual compression
    async fn demonstrate_contextual_compression(&self, query: &str) -> ExampleResult<()> {
        println!("🗜️ Contextual Compression");
        println!("💡 Compressing content to most relevant parts");

        let timer = Timer::new("contextual_compression");

        // Get initial results
        let mut initial_results = self.get_baseline_results(query).await?;

        // Simulate contextual compression by extracting key sentences
        let compressed_results: Vec<ScoredNode> = initial_results
            .into_iter()
            .map(|mut result| {
                // Simple compression: extract sentences containing query keywords
                let query_lower = query.to_lowercase();
                let query_words: Vec<&str> = query_lower.split_whitespace().collect();
                let sentences: Vec<&str> = result.node.content.split('.').collect();

                let relevant_sentences: Vec<&str> = sentences
                    .into_iter()
                    .filter(|sentence| {
                        let sentence_lower = sentence.to_lowercase();
                        query_words.iter().any(|word| sentence_lower.contains(word))
                    })
                    .take(3) // Keep top 3 relevant sentences
                    .collect();

                if !relevant_sentences.is_empty() {
                    result.node.content = relevant_sentences.join(". ") + ".";
                    // Boost score for compressed content
                    result.score = (result.score * 1.1).min(1.0);
                }

                result
            })
            .take(self.config.top_n)
            .collect();

        let filtering_time = timer.finish();
        self.display_filtering_results(
            "Contextual Compression",
            &compressed_results,
            Some(filtering_time),
        );

        Ok(())
    }

    /// Demonstrate relevant segment extraction
    async fn demonstrate_segment_extraction(&self, query: &str) -> ExampleResult<()> {
        println!("🎯 Relevant Segment Extraction");
        println!("💡 Extracting contiguous relevant segments");

        let timer = Timer::new("segment_extraction");

        // Get initial results
        let initial_results = self.get_baseline_results(query).await?;

        // Simulate segment extraction by grouping related content
        let segment_results: Vec<ScoredNode> = initial_results
            .into_iter()
            .map(|mut result| {
                // Simple segment extraction: focus on paragraphs containing query terms
                let query_lower = query.to_lowercase();
                let query_words: Vec<&str> = query_lower.split_whitespace().collect();
                let paragraphs: Vec<&str> = result.node.content.split("\n\n").collect();

                let relevant_paragraphs: Vec<&str> = paragraphs
                    .into_iter()
                    .filter(|para| {
                        let para_lower = para.to_lowercase();
                        query_words.iter().any(|word| para_lower.contains(word))
                    })
                    .take(2) // Keep top 2 relevant paragraphs
                    .collect();

                if !relevant_paragraphs.is_empty() {
                    result.node.content = relevant_paragraphs.join("\n\n");
                    // Boost score for segment extraction
                    result.score = (result.score * 1.05).min(1.0);
                }

                result
            })
            .take(self.config.top_n)
            .collect();

        let filtering_time = timer.finish();
        self.display_filtering_results(
            "Relevant Segment Extraction",
            &segment_results,
            Some(filtering_time),
        );

        Ok(())
    }

    /// Display filtering results
    fn display_filtering_results(
        &self,
        method_name: &str,
        results: &[ScoredNode],
        timing: Option<std::time::Duration>,
    ) {
        println!("\n📋 {} Results:", method_name);
        if let Some(duration) = timing {
            println!("⏱️  Filtering time: {:.2}s", duration.as_secs_f64());
        }

        for (i, result) in results.iter().take(self.config.top_n).enumerate() {
            let content_preview = if result.node.content.len() > 150 {
                format!(
                    "{}...",
                    result.node.content.chars().take(150).collect::<String>()
                )
            } else {
                result.node.content.clone()
            };

            println!(
                "{}. [Score: {:.4}] [Length: {}] {}",
                i + 1,
                result.score,
                result.node.content.len(),
                content_preview
            );
        }
    }

    /// Run interactive mode
    pub async fn run_interactive(&self) -> ExampleResult<()> {
        println!("\n🎯 Interactive Multi-faceted Filtering Mode");
        println!("═══════════════════════════════════════════════════════════════");
        println!("Enter your queries to see different filtering strategies in action.");
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
                    if let Err(e) = self.demonstrate_filtering(query).await {
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
        println!("  - Compare how different filtering strategies affect result quality");
        println!("  - Notice the trade-offs between filtering strictness and result count");
        println!("\n🔧 Available Filtering Strategies:");
        println!("  1. Metadata Filtering - Filter by document source, type, etc.");
        println!("  2. Score Filtering - Filter by similarity score threshold");
        println!("  3. Content Quality - Filter by content length and completeness");
        println!("  4. Contextual Compression - Extract most relevant content parts");
        println!("  5. Segment Extraction - Extract contiguous relevant segments");
    }
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

/// Parse filtering strategy from command line argument
fn parse_filtering_strategy(strategy_str: &str) -> ExampleResult<FilteringStrategy> {
    match strategy_str.to_lowercase().as_str() {
        "metadata" => Ok(FilteringStrategy::Metadata {
            filters: HashMap::new(),
        }),
        "score" => Ok(FilteringStrategy::ScoreThreshold { min_score: 0.7 }),
        "quality" => Ok(FilteringStrategy::ContentQuality {
            min_length: 50,
            max_length: 2000,
            require_complete_sentences: true,
        }),
        "compression" => Ok(FilteringStrategy::ContextualCompression {
            compression_ratio: 0.5,
        }),
        "segments" => Ok(FilteringStrategy::RelevantSegments {
            segment_threshold: 0.6,
            max_segment_size: 1000,
        }),
        "combined" | "all" => Ok(FilteringStrategy::Combined),
        _ => Err(ExampleError::Config(
            "Invalid strategy. Use: metadata, score, quality, compression, segments, or combined"
                .to_string(),
        )),
    }
}

/// Create CLI command structure
fn create_cli() -> Command {
    Command::new("multi_faceted_filtering")
        .about("Multi-faceted Filtering Example - Demonstrates various filtering strategies for improving RAG retrieval quality")
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
                .help("Filtering strategy: metadata, score, quality, compression, segments, combined")
                .default_value("combined"),
        )
        .arg(
            Arg::new("top_n")
                .short('n')
                .long("top-n")
                .value_name("NUMBER")
                .help("Number of top results to return after filtering")
                .default_value("5"),
        )
        .arg(
            Arg::new("initial_count")
                .short('i')
                .long("initial-count")
                .value_name("NUMBER")
                .help("Number of documents to retrieve initially before filtering")
                .default_value("20"),
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
    println!("🎯 Multi-faceted Filtering Example");
    println!("═══════════════════════════════════════════════════════════════");
    println!("📊 Strategy: {}", strategy_str);
    println!("📁 Data: {}", data_path.display());
    println!("🔢 Top N: {}, Initial Count: {}", top_n, initial_count);
    println!("🤖 Embeddings: {}", embedding_provider);
    println!();

    // Parse filtering strategy
    let strategy = parse_filtering_strategy(strategy_str)?;

    // Create filtering configuration
    let config = FilteringConfig {
        strategies: vec![strategy],
        top_n,
        initial_retrieval_count: initial_count,
        enable_comparison: true,
    };

    // Initialize demo
    let demo = MultiFacetedFilteringDemo::new(data_path, embedding_provider, config).await?;

    // Run demo
    if let Some(query_text) = query {
        // Single query mode
        demo.demonstrate_filtering(query_text).await?;
        demo.performance_metrics.print_summary();
    } else {
        // Interactive mode
        demo.run_interactive().await?;
    }

    Ok(())
}
