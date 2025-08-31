//! Title Extractor Demonstration
//!
//! This example demonstrates the new TitleExtractor transformer that uses LLM
//! to generate intelligent document titles from content, following LlamaIndex's
//! TitleExtractor design exactly.

use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber;

use cheungfun_core::{
    traits::{Transform, TransformInput},
    types::{ChunkInfo, Document, Node},
};
use cheungfun_indexing::{
    node_parser::text::SentenceSplitter,
    transformers::{TitleExtractor, TitleExtractorConfig},
};
use siumai::prelude::*;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("🏷️ TitleExtractor Demo");

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

    // Demo 1: Basic Title Extraction
    info!("\n📋 Demo 1: Basic Title Extraction");
    demo_basic_title_extraction().await?;

    // Demo 2: Multiple Documents
    info!("\n📋 Demo 2: Multiple Documents Title Extraction");
    demo_multiple_documents().await?;

    // Demo 3: Custom Configuration
    info!("\n📋 Demo 3: Custom Configuration");
    demo_custom_configuration().await?;

    // Demo 4: Integration with Pipeline
    info!("\n📋 Demo 4: Integration with Processing Pipeline");
    demo_pipeline_integration().await?;

    info!("\n✅ All demos completed successfully!");
    Ok(())
}

/// Demo 1: Basic title extraction from a single document
async fn demo_basic_title_extraction() -> Result<(), Box<dyn std::error::Error>> {
    info!("Creating LLM client...");

    // Create LLM client (try different providers)
    let llm_client = create_llm_client().await?;

    info!("Creating TitleExtractor...");

    // Create title extractor with default configuration
    let title_extractor = TitleExtractor::with_defaults(Arc::new(llm_client))?;

    // Create sample document content
    let document_content = r#"
Artificial Intelligence and Machine Learning have revolutionized the way we approach complex problems in technology. 
Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions with remarkable accuracy.

Deep learning, a subset of machine learning, uses neural networks with multiple layers to model and understand complex patterns in data.
These networks have been particularly successful in areas such as image recognition, natural language processing, and speech recognition.

The applications of AI and ML span across various industries including healthcare, finance, automotive, and entertainment.
In healthcare, AI systems can assist in medical diagnosis and drug discovery. In finance, they help with fraud detection and algorithmic trading.
"#;

    // Create nodes from the content (simulating document processing)
    let nodes = create_sample_nodes(document_content, "ai_ml_doc");

    info!("Processing {} nodes for title extraction...", nodes.len());

    // Extract titles
    let enhanced_nodes = title_extractor.transform(TransformInput::Nodes(nodes)).await?;

    // Display results
    info!("✅ Title extraction completed!");
    for (i, node) in enhanced_nodes.iter().enumerate() {
        if let Some(title) = node.metadata.get("document_title") {
            info!("📄 Node {}: Document title = '{}'", i + 1, title.as_str().unwrap_or("N/A"));
        }
    }

    Ok(())
}

/// Demo 2: Multiple documents with different content types
async fn demo_multiple_documents() -> Result<(), Box<dyn std::error::Error>> {
    let llm_client = create_llm_client().await?;
    let title_extractor = TitleExtractor::with_defaults(Arc::new(llm_client))?;

    // Create multiple documents with different topics
    let documents = vec![
        (
            "climate_change_doc",
            r#"
Climate change represents one of the most pressing challenges of our time. Rising global temperatures are causing 
widespread environmental impacts including melting ice caps, rising sea levels, and extreme weather events.

The primary driver of climate change is the increase in greenhouse gas emissions from human activities, particularly
the burning of fossil fuels for energy production and transportation.

Mitigation strategies include transitioning to renewable energy sources, improving energy efficiency, and implementing
carbon capture technologies. Adaptation measures are also crucial for communities to cope with changing climate conditions.
"#,
        ),
        (
            "quantum_computing_doc",
            r#"
Quantum computing represents a paradigm shift in computational technology, leveraging the principles of quantum mechanics
to process information in fundamentally new ways. Unlike classical computers that use bits, quantum computers use quantum bits or qubits.

Qubits can exist in superposition states, allowing quantum computers to perform certain calculations exponentially faster
than classical computers. This has profound implications for cryptography, optimization, and scientific simulation.

Major technology companies and research institutions are racing to develop practical quantum computers that can solve
real-world problems in areas such as drug discovery, financial modeling, and artificial intelligence.
"#,
        ),
    ];

    let mut all_nodes = Vec::new();
    for (doc_id, content) in documents {
        let nodes = create_sample_nodes(content, doc_id);
        all_nodes.extend(nodes);
    }

    info!("Processing {} nodes from {} documents...", all_nodes.len(), 2);

    let enhanced_nodes = title_extractor.transform(TransformInput::Nodes(all_nodes)).await?;

    // Group results by document
    let mut titles_by_doc: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for node in &enhanced_nodes {
        if let (Some(doc_id), Some(title)) = (
            node.ref_doc_id.as_ref(),
            node.metadata.get("document_title"),
        ) {
            titles_by_doc.insert(doc_id.clone(), title.as_str().unwrap_or("N/A").to_string());
        }
    }

    info!("✅ Multiple document title extraction completed!");
    for (doc_id, title) in titles_by_doc {
        info!("📄 Document '{}': '{}'", doc_id, title);
    }

    Ok(())
}

/// Demo 3: Custom configuration options
async fn demo_custom_configuration() -> Result<(), Box<dyn std::error::Error>> {
    let llm_client = create_llm_client().await?;

    // Create custom configuration
    let config = TitleExtractorConfig::new()
        .with_nodes(3) // Use only first 3 nodes
        .with_show_progress(true)
        .with_max_context_length(2000) // Limit context length
        .with_in_place(false); // Create copies instead of modifying in place

    let title_extractor = TitleExtractor::new(Arc::new(llm_client), config)?;

    // Create a longer document
    let long_content = r#"
The field of biotechnology has emerged as one of the most promising areas of scientific research and commercial application.
Biotechnology involves the use of living organisms, cells, and biological processes to develop products and technologies
that improve human life and the environment.

Modern biotechnology encompasses various disciplines including genetic engineering, molecular biology, biochemistry,
and bioinformatics. These fields work together to create innovative solutions for challenges in medicine, agriculture,
and environmental science.

In medicine, biotechnology has led to the development of life-saving drugs, vaccines, and diagnostic tools.
Gene therapy and personalized medicine are emerging as revolutionary approaches to treating genetic disorders and cancer.

Agricultural biotechnology has produced crops with enhanced nutritional content, resistance to pests and diseases,
and improved yield. These developments are crucial for addressing global food security challenges.

Environmental biotechnology offers solutions for pollution control, waste management, and sustainable energy production.
Bioremediation uses microorganisms to clean up contaminated environments, while biofuels provide renewable energy alternatives.
"#;

    let nodes = create_sample_nodes(long_content, "biotech_doc");
    info!("Processing {} nodes with custom configuration...", nodes.len());

    let enhanced_nodes = title_extractor.transform(TransformInput::Nodes(nodes)).await?;

    info!("✅ Custom configuration title extraction completed!");
    if let Some(node) = enhanced_nodes.first() {
        if let Some(title) = node.metadata.get("document_title") {
            info!("📄 Generated title: '{}'", title.as_str().unwrap_or("N/A"));
        }
    }

    Ok(())
}

/// Demo 4: Integration with document processing pipeline
async fn demo_pipeline_integration() -> Result<(), Box<dyn std::error::Error>> {
    let llm_client = create_llm_client().await?;

    // Create document splitter
    let splitter = SentenceSplitter::from_defaults(500, 50)?;

    // Create title extractor
    let title_extractor = TitleExtractor::builder(Arc::new(llm_client))
        .nodes(2)
        .show_progress(true)
        .build()?;

    // Sample document
    let document = Document::new(
        r#"
Space exploration has captured human imagination for centuries, but it wasn't until the 20th century that we developed
the technology to actually venture beyond Earth's atmosphere. The space race between the United States and Soviet Union
drove rapid advances in rocket technology and spacecraft design.

Today, space exploration has evolved from a competition between superpowers to a collaborative international effort.
The International Space Station serves as a symbol of this cooperation, hosting astronauts from multiple countries
working together on scientific research.

Private companies are now playing an increasingly important role in space exploration. Companies like SpaceX and Blue Origin
are developing reusable rockets that dramatically reduce the cost of space access, opening up new possibilities for
commercial space activities and eventual human settlement of other planets.
"#,
        Some("space_exploration_doc".to_string()),
    );

    info!("Processing document through complete pipeline...");

    // Step 1: Split document into nodes
    let nodes = splitter.transform(TransformInput::Document(document)).await?;
    info!("📄 Document split into {} nodes", nodes.len());

    // Step 2: Extract titles
    let enhanced_nodes = title_extractor.transform(TransformInput::Nodes(nodes)).await?;
    info!("🏷️ Title extraction completed");

    // Display results
    info!("✅ Pipeline integration completed!");
    if let Some(node) = enhanced_nodes.first() {
        if let Some(title) = node.metadata.get("document_title") {
            info!("📄 Final document title: '{}'", title.as_str().unwrap_or("N/A"));
        }
        info!("📊 Total enhanced nodes: {}", enhanced_nodes.len());
    }

    Ok(())
}

/// Demo with mock data when no API key is available
async fn demo_with_mock_data() -> Result<(), Box<dyn std::error::Error>> {
    info!("📝 Demonstrating TitleExtractor structure with mock data...");

    // Show configuration options
    let config = TitleExtractorConfig::new()
        .with_nodes(5)
        .with_show_progress(true)
        .with_num_workers(2)
        .with_max_context_length(3000);

    info!("📋 TitleExtractor Configuration:");
    info!("  - Nodes for extraction: {}", config.nodes);
    info!("  - Show progress: {}", config.show_progress);
    info!("  - Number of workers: {}", config.num_workers);
    info!("  - Max context length: {}", config.max_context_length);

    // Show what the extractor would do
    let sample_content = "Artificial Intelligence and Machine Learning in Modern Applications";
    let nodes = create_sample_nodes(sample_content, "sample_doc");

    info!("📄 Sample nodes created: {}", nodes.len());
    info!("🏷️ TitleExtractor would process these nodes to generate intelligent titles");
    info!("💡 The extractor uses LLM to analyze content and generate contextual titles");

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
            .temperature(0.3) // Lower temperature for more consistent titles
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

/// Create sample nodes from content
fn create_sample_nodes(content: &str, doc_id: &str) -> Vec<Node> {
    // Split content into sentences for realistic node simulation
    let sentences: Vec<&str> = content.split('.').filter(|s| !s.trim().is_empty()).collect();
    
    let mut nodes = Vec::new();
    let mut offset = 0;
    
    for (i, sentence) in sentences.iter().enumerate() {
        let sentence_content = format!("{}.", sentence.trim());
        let start_offset = offset;
        let end_offset = offset + sentence_content.len();
        
        let node = Node::new(
            sentence_content,
            Uuid::new_v4(),
            ChunkInfo {
                start_offset,
                end_offset,
                chunk_index: i,
            },
        )
        .with_ref_doc_id(doc_id.to_string());
        
        nodes.push(node);
        offset = end_offset + 1; // +1 for the period
    }
    
    // Ensure we have at least one node
    if nodes.is_empty() {
        let node = Node::new(
            content.to_string(),
            Uuid::new_v4(),
            ChunkInfo {
                start_offset: 0,
                end_offset: content.len(),
                chunk_index: 0,
            },
        )
        .with_ref_doc_id(doc_id.to_string());
        nodes.push(node);
    }
    
    nodes
}
