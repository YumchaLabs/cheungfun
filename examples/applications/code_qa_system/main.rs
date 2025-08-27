//! Code Q&A System - Complete RAG Application for Code Analysis
//!
//! This application demonstrates a comprehensive code question-answering system
//! using Cheungfun's capabilities for code indexing and intelligent querying.
//!
//! Features:
//! - Multi-language code parsing (Rust, C#, Python, JavaScript, etc.)
//! - AST-based intelligent code splitting
//! - Semantic search with code structure awareness
//! - Interactive CLI interface
//!
//! Usage:
//! ```bash
//! cargo run --bin code_qa_system -- /path/to/project
//! cargo run --bin code_qa_system -- /path/to/project --lang rust --model gpt-4
//! ```

use anyhow::Result;
use cheungfun_core::{
    traits::{Embedder, Loader, Transform, VectorStore},
    Document,
};
use cheungfun_indexing::{
    loaders::{
        filter::FilterConfig, CodeLoader, CodeLoaderConfig, LoaderConfig, ProgrammingLanguage,
    },
    node_parser::{
        config::{ChunkingStrategy, CodeSplitterConfig},
        text::CodeSplitter,
    },
};
use cheungfun_integrations::{
    embedders::fastembed::{FastEmbedConfig, FastEmbedEmbedder},
    vector_stores::memory::InMemoryVectorStore,
};
use clap::{Arg, Command};
use std::{
    io::{self, Write},
    path::Path,
    sync::Arc,
};
use tokio;
use tracing::{info, Level};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let matches = Command::new("Code Q&A System")
        .version("1.0")
        .about("Intelligent code question-answering system using RAG")
        .arg(
            Arg::new("path")
                .help("Path to the codebase directory")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("language")
                .long("lang")
                .help("Primary programming language (rust, csharp, python, javascript)")
                .default_value("auto"),
        )
        .arg(
            Arg::new("model")
                .long("model")
                .help("Embedding model to use")
                .default_value("sentence-transformers/all-MiniLM-L6-v2"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    let project_path = matches.get_one::<String>("path").unwrap();
    let language = matches.get_one::<String>("language").unwrap();
    let model = matches.get_one::<String>("model").unwrap();
    let verbose = matches.get_flag("verbose");

    println!("🤖 Code Q&A System");
    println!("==================");
    println!("📁 Project: {}", project_path);
    println!("🔤 Language: {}", language);
    println!("🧠 Model: {}", model);
    println!();

    // Step 1: Setup components
    info!("Setting up components...");

    let embedder = Arc::new(
        FastEmbedEmbedder::from_config(&FastEmbedConfig {
            model_name: model.clone(),
            ..Default::default()
        })
        .await?,
    );

    let vector_store = Arc::new(InMemoryVectorStore::new(
        embedder.dimension(),
        cheungfun_core::types::DistanceMetric::Cosine,
    ));

    // Step 2: Configure code loader
    info!("Configuring code loader...");

    let programming_lang = match language.as_str() {
        "rust" => Some(ProgrammingLanguage::Rust),
        "csharp" | "cs" => Some(ProgrammingLanguage::CSharp),
        "python" | "py" => Some(ProgrammingLanguage::Python),
        "javascript" | "js" => Some(ProgrammingLanguage::JavaScript),
        "typescript" | "ts" => Some(ProgrammingLanguage::TypeScript),
        _ => None,
    };

    let filter_config = FilterConfig::source_code_only()
        .with_respect_gitignore(true)
        .with_exclude_hidden(true);

    let loader_config = LoaderConfig::new().with_filter_config(filter_config);

    let code_config = CodeLoaderConfig {
        programming_language: programming_lang,
        extract_functions: true,
        extract_classes: true,
        extract_imports: true,
        extract_comments: true,
        max_file_size: Some(1024 * 1024), // 1MB limit
        ..Default::default()
    };

    let loader = Arc::new(CodeLoader::with_configs(
        project_path,
        loader_config,
        code_config,
    )?);

    // Step 3: Configure code splitter
    info!("Setting up AST-based code splitter...");

    let splitter_config = CodeSplitterConfig {
        chunk_size: 1000,
        chunk_overlap: 200,
        chunking_strategy: ChunkingStrategy::Semantic,
        respect_function_boundaries: true,
        include_metadata: true,
        ..Default::default()
    };

    let code_splitter = Arc::new(CodeSplitter::from_config(splitter_config)?);

    // Step 4: Index the codebase
    println!("📊 Indexing codebase...");

    let documents = loader.load().await?;
    println!("✅ Loaded {} files", documents.len());

    let mut total_nodes = 0;
    for doc in documents {
        let input = cheungfun_core::types::TransformInput::Documents(vec![doc]);
        let nodes = code_splitter.transform(input).await?;

        if verbose {
            println!("  📄 Processed document with {} nodes", nodes.len());
        }

        // Generate embeddings and store
        for node in &nodes {
            let embedding = embedder.embed(&node.content).await?;
            let mut node_with_embedding = node.clone();
            node_with_embedding.embedding = Some(embedding);
            vector_store.add(vec![node_with_embedding]).await?;
        }

        total_nodes += nodes.len();
    }

    println!("✅ Indexed {} code nodes", total_nodes);
    println!();

    // Step 5: Interactive Q&A loop
    println!("🎯 Ready for questions! Type 'quit' to exit.");
    println!("Examples:");
    println!("  - How does authentication work in this codebase?");
    println!("  - What are the main classes and their responsibilities?");
    println!("  - Show me error handling patterns");
    println!();

    loop {
        print!("❓ Your question: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let question = input.trim();

        if question.is_empty() {
            continue;
        }

        if question.to_lowercase() == "quit" {
            println!("👋 Goodbye!");
            break;
        }

        // Search for relevant code
        let query_embedding = embedder.embed(question).await?;
        let query = cheungfun_core::types::Query {
            text: question.to_string(),
            embedding: Some(query_embedding),
            filters: std::collections::HashMap::new(),
            top_k: 5,
            similarity_threshold: Some(0.3),
            search_mode: cheungfun_core::types::SearchMode::Vector,
        };

        let results = vector_store.search(&query).await?;

        if results.is_empty() {
            println!("🤷 No relevant code found for your question.");
            println!();
            continue;
        }

        println!("🔍 Found {} relevant code sections:", results.len());
        println!();

        for (i, scored_node) in results.iter().enumerate() {
            println!("📋 Result {} (Score: {:.3}):", i + 1, scored_node.score);

            // Extract metadata for better context
            if let Some(file_path) = scored_node.node.metadata.get("file_path") {
                println!("📁 File: {}", file_path.as_str().unwrap_or("unknown"));
            }

            if let Some(function_name) = scored_node.node.metadata.get("function_name") {
                println!(
                    "⚡ Function: {}",
                    function_name.as_str().unwrap_or("unknown")
                );
            }

            println!("```");
            println!(
                "{}",
                scored_node
                    .node
                    .content
                    .lines()
                    .take(20)
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            if scored_node.node.content.lines().count() > 20 {
                println!("... (truncated)");
            }
            println!("```");
            println!();
        }
    }

    Ok(())
}
