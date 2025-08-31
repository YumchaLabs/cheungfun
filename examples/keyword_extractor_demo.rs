//! Keyword Extractor Demonstration
//!
//! This example demonstrates the new KeywordExtractor transformer that uses LLM
//! to extract relevant keywords from content, following LlamaIndex's
//! KeywordExtractor design exactly.

use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber;

use cheungfun_core::{
    traits::{Transform, TransformInput},
    types::{ChunkInfo, Document, Node},
};
use cheungfun_indexing::{
    node_parser::text::SentenceSplitter,
    transformers::{KeywordExtractor, KeywordExtractorConfig},
};
use siumai::prelude::*;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("🔑 KeywordExtractor Demo");

    // Check for API key
    let api_key = std::env::var("OPENAI_API_KEY")
        .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
        .or_else(|_| std::env::var("OLLAMA_API_KEY"));

    if api_key.is_err() {
        info!("⚠️ No API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or use Ollama");
        info!("📝 Running with mock data only...");
        demo_with_mock_data().await?;
        return Ok(());
    }

    // Demo 1: Basic Keyword Extraction
    info!("\n📋 Demo 1: Basic Keyword Extraction");
    demo_basic_keyword_extraction().await?;

    // Demo 2: Multiple Nodes Processing
    info!("\n📋 Demo 2: Multiple Nodes Keyword Extraction");
    demo_multiple_nodes().await?;

    // Demo 3: Custom Configuration
    info!("\n📋 Demo 3: Custom Configuration");
    demo_custom_configuration().await?;

    // Demo 4: Integration with Pipeline
    info!("\n📋 Demo 4: Integration with Processing Pipeline");
    demo_pipeline_integration().await?;

    info!("\n✅ All demos completed successfully!");
    Ok(())
}

/// Demo 1: Basic keyword extraction from a single node
async fn demo_basic_keyword_extraction() -> Result<(), Box<dyn std::error::Error>> {
    info!("Creating LLM client...");

    // Create LLM client (try different providers)
    let llm_client = create_llm_client().await?;

    info!("Creating KeywordExtractor...");

    // Create keyword extractor with default configuration
    let keyword_extractor = KeywordExtractor::with_defaults(Arc::new(llm_client))?;

    // Create sample node content
    let node_content = r#"
Artificial Intelligence and Machine Learning have revolutionized the way we approach complex problems in technology. 
Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions with remarkable accuracy.
Deep learning, a subset of machine learning, uses neural networks with multiple layers to model and understand complex patterns in data.
These networks have been particularly successful in areas such as image recognition, natural language processing, and speech recognition.
"#;

    // Create a sample node
    let node = create_sample_node(node_content, "ai_ml_node");

    info!("Processing node for keyword extraction...");

    // Extract keywords
    let enhanced_nodes = keyword_extractor.transform(TransformInput::Node(node)).await?;

    // Display results
    info!("✅ Keyword extraction completed!");
    if let Some(node) = enhanced_nodes.first() {
        if let Some(keywords) = node.metadata.get("excerpt_keywords") {
            info!("🔑 Extracted keywords: '{}'", keywords.as_str().unwrap_or("N/A"));
        }
    }

    Ok(())
}

/// Demo 2: Multiple nodes with different content types
async fn demo_multiple_nodes() -> Result<(), Box<dyn std::error::Error>> {
    let llm_client = create_llm_client().await?;
    let keyword_extractor = KeywordExtractor::with_defaults(Arc::new(llm_client))?;

    // Create multiple nodes with different topics
    let node_contents = vec![
        r#"
Climate change represents one of the most pressing challenges of our time. Rising global temperatures are causing 
widespread environmental impacts including melting ice caps, rising sea levels, and extreme weather events.
The primary driver of climate change is the increase in greenhouse gas emissions from human activities.
"#,
        r#"
Quantum computing represents a paradigm shift in computational technology, leveraging the principles of quantum mechanics
to process information in fundamentally new ways. Unlike classical computers that use bits, quantum computers use quantum bits or qubits.
Qubits can exist in superposition states, allowing quantum computers to perform certain calculations exponentially faster.
"#,
        r#"
Blockchain technology provides a decentralized and secure way to record transactions and store data.
It uses cryptographic hashing and distributed consensus mechanisms to ensure data integrity and prevent tampering.
Applications include cryptocurrencies, smart contracts, and supply chain management.
"#,
    ];

    let nodes: Vec<Node> = node_contents
        .into_iter()
        .enumerate()
        .map(|(i, content)| create_sample_node(content, &format!("node_{}", i)))
        .collect();

    info!("Processing {} nodes for keyword extraction...", nodes.len());

    let enhanced_nodes = keyword_extractor.transform(TransformInput::Nodes(nodes)).await?;

    // Display results
    info!("✅ Multiple node keyword extraction completed!");
    for (i, node) in enhanced_nodes.iter().enumerate() {
        if let Some(keywords) = node.metadata.get("excerpt_keywords") {
            info!("🔑 Node {}: '{}'", i + 1, keywords.as_str().unwrap_or("N/A"));
        }
    }

    Ok(())
}

/// Demo 3: Custom configuration options
async fn demo_custom_configuration() -> Result<(), Box<dyn std::error::Error>> {
    let llm_client = create_llm_client().await?;

    // Create custom configuration
    let config = KeywordExtractorConfig::new()
        .with_keywords(8) // Extract 8 keywords instead of default 5
        .with_show_progress(true)
        .with_max_context_length(2000) // Limit context length
        .with_lowercase_keywords(false) // Keep original case
        .with_remove_duplicates(true); // Remove duplicates

    let keyword_extractor = KeywordExtractor::new(Arc::new(llm_client), config)?;

    // Create a technical document
    let technical_content = r#"
The field of biotechnology has emerged as one of the most promising areas of scientific research and commercial application.
Biotechnology involves the use of living organisms, cells, and biological processes to develop products and technologies
that improve human life and the environment. Modern biotechnology encompasses various disciplines including genetic engineering,
molecular biology, biochemistry, and bioinformatics. These fields work together to create innovative solutions for challenges
in medicine, agriculture, and environmental science. Gene therapy and personalized medicine are emerging as revolutionary
approaches to treating genetic disorders and cancer. CRISPR-Cas9 technology has revolutionized gene editing capabilities.
"#;

    let node = create_sample_node(technical_content, "biotech_node");
    info!("Processing node with custom configuration...");

    let enhanced_nodes = keyword_extractor.transform(TransformInput::Node(node)).await?;

    info!("✅ Custom configuration keyword extraction completed!");
    if let Some(node) = enhanced_nodes.first() {
        if let Some(keywords) = node.metadata.get("excerpt_keywords") {
            info!("🔑 Extracted keywords (8 max, original case): '{}'", keywords.as_str().unwrap_or("N/A"));
        }
    }

    Ok(())
}

/// Demo 4: Integration with document processing pipeline
async fn demo_pipeline_integration() -> Result<(), Box<dyn std::error::Error>> {
    let llm_client = create_llm_client().await?;

    // Create document splitter
    let splitter = SentenceSplitter::from_defaults(300, 30)?;

    // Create keyword extractor
    let keyword_extractor = KeywordExtractor::builder(Arc::new(llm_client))
        .keywords(6)
        .show_progress(true)
        .build()?;

    // Sample document
    let document = Document::new(
        r#"
Space exploration has captured human imagination for centuries, but it wasn't until the 20th century that we developed
the technology to actually venture beyond Earth's atmosphere. The space race between the United States and Soviet Union
drove rapid advances in rocket technology and spacecraft design. Today, space exploration has evolved from a competition
between superpowers to a collaborative international effort. The International Space Station serves as a symbol of this
cooperation, hosting astronauts from multiple countries working together on scientific research. Private companies are
now playing an increasingly important role in space exploration. Companies like SpaceX and Blue Origin are developing
reusable rockets that dramatically reduce the cost of space access, opening up new possibilities for commercial space
activities and eventual human settlement of other planets. Mars exploration missions continue to provide valuable data
about the Red Planet's geology, atmosphere, and potential for past or present life.
"#,
        Some("space_exploration_doc".to_string()),
    );

    info!("Processing document through complete pipeline...");

    // Step 1: Split document into nodes
    let nodes = splitter.transform(TransformInput::Document(document)).await?;
    info!("📄 Document split into {} nodes", nodes.len());

    // Step 2: Extract keywords
    let enhanced_nodes = keyword_extractor.transform(TransformInput::Nodes(nodes)).await?;
    info!("🔑 Keyword extraction completed");

    // Display results
    info!("✅ Pipeline integration completed!");
    for (i, node) in enhanced_nodes.iter().enumerate() {
        if let Some(keywords) = node.metadata.get("excerpt_keywords") {
            info!("🔑 Node {}: '{}'", i + 1, keywords.as_str().unwrap_or("N/A"));
        }
    }
    info!("📊 Total enhanced nodes: {}", enhanced_nodes.len());

    Ok(())
}

/// Demo with mock data when no API key is available
async fn demo_with_mock_data() -> Result<(), Box<dyn std::error::Error>> {
    info!("📝 Demonstrating KeywordExtractor structure with mock data...");

    // Show configuration options
    let config = KeywordExtractorConfig::new()
        .with_keywords(7)
        .with_show_progress(true)
        .with_num_workers(2)
        .with_max_context_length(3000)
        .with_lowercase_keywords(true)
        .with_remove_duplicates(true);

    info!("📋 KeywordExtractor Configuration:");
    info!("  - Keywords to extract: {}", config.keywords);
    info!("  - Show progress: {}", config.show_progress);
    info!("  - Number of workers: {}", config.num_workers);
    info!("  - Max context length: {}", config.max_context_length);
    info!("  - Lowercase keywords: {}", config.lowercase_keywords);
    info!("  - Remove duplicates: {}", config.remove_duplicates);

    // Show what the extractor would do
    let sample_content = "Artificial Intelligence, Machine Learning, Deep Learning, Neural Networks, Data Science";
    let node = create_sample_node(sample_content, "sample_node");

    info!("📄 Sample node created with content length: {}", node.content().len());
    info!("🔑 KeywordExtractor would process this node to extract relevant keywords");
    info!("💡 The extractor uses LLM to analyze content and generate contextual keywords");
    info!("📝 Output would be stored in 'excerpt_keywords' metadata field");

    Ok(())
}

/// Create LLM client with fallback options
async fn create_llm_client() -> Result<Box<dyn LlmClient>, Box<dyn std::error::Error>> {
    // Try OpenAI first
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        info!("🤖 Using OpenAI client");
        let client = Siumai::builder()
            .openai()
            .api_key(api_key)
            .model("gpt-4o-mini")
            .temperature(0.3) // Lower temperature for more consistent keywords
            .build()
            .await?;
        return Ok(Box::new(client));
    }

    // Try Anthropic
    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        info!("🤖 Using Anthropic client");
        let client = Siumai::builder()
            .anthropic()
            .api_key(api_key)
            .model("claude-3-haiku-20240307")
            .temperature(0.3)
            .build()
            .await?;
        return Ok(Box::new(client));
    }

    // Try Ollama as fallback
    info!("🤖 Using Ollama client (local)");
    let client = Siumai::builder()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .temperature(0.3)
        .build()
        .await?;
    Ok(Box::new(client))
}

/// Create a sample node from content
fn create_sample_node(content: &str, node_id: &str) -> Node {
    Node::new(
        content.trim().to_string(),
        Uuid::new_v4(),
        ChunkInfo {
            start_offset: 0,
            end_offset: content.len(),
            chunk_index: 0,
        },
    )
    .with_ref_doc_id(node_id.to_string())
}
