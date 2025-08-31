//! Summary Extractor Demonstration
//!
//! This example demonstrates the new SummaryExtractor transformer that uses LLM
//! to generate intelligent summaries for nodes and their adjacent context,
//! following LlamaIndex's SummaryExtractor design exactly.

use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber;

use cheungfun_core::{
    traits::{Transform, TransformInput},
    types::{ChunkInfo, Document, Node},
};
use cheungfun_indexing::{
    node_parser::text::SentenceSplitter,
    transformers::{SummaryExtractor, SummaryExtractorConfig, SummaryType},
};
use siumai::prelude::*;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("📝 SummaryExtractor Demo");

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

    // Demo 1: Basic Summary Extraction (Self Only)
    info!("\n📋 Demo 1: Basic Summary Extraction (Self Only)");
    demo_basic_summary_extraction().await?;

    // Demo 2: Adjacent Context Summaries
    info!("\n📋 Demo 2: Adjacent Context Summaries (Prev/Self/Next)");
    demo_adjacent_context_summaries().await?;

    // Demo 3: Custom Configuration
    info!("\n📋 Demo 3: Custom Configuration");
    demo_custom_configuration().await?;

    // Demo 4: Integration with Pipeline
    info!("\n📋 Demo 4: Integration with Processing Pipeline");
    demo_pipeline_integration().await?;

    info!("\n✅ All demos completed successfully!");
    Ok(())
}

/// Demo 1: Basic summary extraction (self only)
async fn demo_basic_summary_extraction() -> Result<(), Box<dyn std::error::Error>> {
    info!("Creating LLM client...");

    // Create LLM client (try different providers)
    let llm_client = create_llm_client().await?;

    info!("Creating SummaryExtractor...");

    // Create summary extractor with default configuration (self only)
    let summary_extractor = SummaryExtractor::with_defaults(Arc::new(llm_client))?;

    // Create sample nodes with different content
    let nodes = create_sample_nodes();

    info!("Processing {} nodes for summary extraction...", nodes.len());

    // Extract summaries
    let enhanced_nodes = summary_extractor.transform(TransformInput::Nodes(nodes)).await?;

    // Display results
    info!("✅ Summary extraction completed!");
    for (i, node) in enhanced_nodes.iter().enumerate() {
        if let Some(summary) = node.metadata.get("section_summary") {
            info!("📄 Node {}: Summary = '{}'", i + 1, 
                  summary.as_str().unwrap_or("N/A").chars().take(100).collect::<String>());
        }
    }

    Ok(())
}

/// Demo 2: Adjacent context summaries (prev/self/next)
async fn demo_adjacent_context_summaries() -> Result<(), Box<dyn std::error::Error>> {
    let llm_client = create_llm_client().await?;

    // Create summary extractor with all summary types
    let summary_extractor = SummaryExtractor::builder(Arc::new(llm_client))
        .summaries(vec![
            SummaryType::PrevSummary,
            SummaryType::SelfSummary,
            SummaryType::NextSummary,
        ])
        .show_progress(true)
        .build()?;

    // Create a sequence of related nodes
    let nodes = create_sequential_nodes();

    info!("Processing {} sequential nodes with adjacent context...", nodes.len());

    let enhanced_nodes = summary_extractor.transform(TransformInput::Nodes(nodes)).await?;

    // Display results with adjacent context
    info!("✅ Adjacent context summary extraction completed!");
    for (i, node) in enhanced_nodes.iter().enumerate() {
        info!("📄 Node {}:", i + 1);
        
        if let Some(prev_summary) = node.metadata.get("prev_section_summary") {
            info!("  ⬅️ Prev: '{}'", 
                  prev_summary.as_str().unwrap_or("N/A").chars().take(80).collect::<String>());
        }
        
        if let Some(self_summary) = node.metadata.get("section_summary") {
            info!("  📄 Self: '{}'", 
                  self_summary.as_str().unwrap_or("N/A").chars().take(80).collect::<String>());
        }
        
        if let Some(next_summary) = node.metadata.get("next_section_summary") {
            info!("  ➡️ Next: '{}'", 
                  next_summary.as_str().unwrap_or("N/A").chars().take(80).collect::<String>());
        }
    }

    Ok(())
}

/// Demo 3: Custom configuration options
async fn demo_custom_configuration() -> Result<(), Box<dyn std::error::Error>> {
    let llm_client = create_llm_client().await?;

    // Create custom configuration with custom template
    let custom_template = r#"Content: {context_str}

Create a concise summary focusing on the main concepts and key information. Keep it under 50 words.

Summary: "#;

    let config = SummaryExtractorConfig::new()
        .with_summaries(vec![SummaryType::SelfSummary])
        .with_prompt_template(custom_template.to_string())
        .with_show_progress(true)
        .with_max_context_length(1500) // Shorter context
        .with_in_place(false); // Create copies

    let summary_extractor = SummaryExtractor::new(Arc::new(llm_client), config)?;

    // Create a technical document
    let technical_content = r#"
The field of quantum computing represents a fundamental shift in computational paradigms. Unlike classical computers
that use bits representing either 0 or 1, quantum computers use quantum bits (qubits) that can exist in superposition
states. This allows quantum computers to perform certain calculations exponentially faster than classical computers.
Key quantum phenomena include superposition, entanglement, and quantum interference. These properties enable quantum
algorithms like Shor's algorithm for factoring large numbers and Grover's algorithm for searching unsorted databases.
Current challenges include quantum decoherence, error correction, and scaling to larger numbers of qubits.
"#;

    let node = create_sample_node(technical_content, "quantum_computing");
    info!("Processing technical content with custom configuration...");

    let enhanced_nodes = summary_extractor.transform(TransformInput::Node(node)).await?;

    info!("✅ Custom configuration summary extraction completed!");
    if let Some(node) = enhanced_nodes.first() {
        if let Some(summary) = node.metadata.get("section_summary") {
            info!("📄 Custom summary: '{}'", summary.as_str().unwrap_or("N/A"));
        }
    }

    Ok(())
}

/// Demo 4: Integration with document processing pipeline
async fn demo_pipeline_integration() -> Result<(), Box<dyn std::error::Error>> {
    let llm_client = create_llm_client().await?;

    // Create document splitter
    let splitter = SentenceSplitter::from_defaults(400, 50)?;

    // Create summary extractor
    let summary_extractor = SummaryExtractor::builder(Arc::new(llm_client))
        .with_self_summary()
        .with_prev_summary()
        .show_progress(true)
        .build()?;

    // Sample document about AI development
    let document = Document::new(
        r#"
Artificial Intelligence has undergone remarkable evolution since its inception in the 1950s. Early AI research focused
on symbolic reasoning and expert systems, attempting to encode human knowledge into rule-based systems. These systems
showed promise in narrow domains but struggled with the complexity and ambiguity of real-world problems.

The 1980s and 1990s saw the rise of machine learning approaches, which shifted focus from hand-coded rules to algorithms
that could learn patterns from data. Neural networks, inspired by biological brain structures, gained attention but were
limited by computational constraints and the lack of large datasets.

The breakthrough came in the 2010s with deep learning, enabled by powerful GPUs and vast amounts of data. Deep neural
networks achieved superhuman performance in image recognition, natural language processing, and game playing. This led
to practical applications in autonomous vehicles, medical diagnosis, and language translation.

Today, large language models like GPT and BERT have revolutionized natural language understanding and generation.
These models, trained on massive text corpora, demonstrate emergent capabilities in reasoning, creativity, and
problem-solving. The field continues to evolve rapidly with advances in multimodal AI, reinforcement learning,
and AI safety research.
"#,
        Some("ai_evolution_doc".to_string()),
    );

    info!("Processing document through complete pipeline...");

    // Step 1: Split document into nodes
    let nodes = splitter.transform(TransformInput::Document(document)).await?;
    info!("📄 Document split into {} nodes", nodes.len());

    // Step 2: Extract summaries
    let enhanced_nodes = summary_extractor.transform(TransformInput::Nodes(nodes)).await?;
    info!("📝 Summary extraction completed");

    // Display results
    info!("✅ Pipeline integration completed!");
    for (i, node) in enhanced_nodes.iter().enumerate() {
        info!("📄 Node {}:", i + 1);
        
        if let Some(self_summary) = node.metadata.get("section_summary") {
            info!("  📝 Summary: '{}'", 
                  self_summary.as_str().unwrap_or("N/A").chars().take(120).collect::<String>());
        }
        
        if let Some(prev_summary) = node.metadata.get("prev_section_summary") {
            info!("  ⬅️ Previous context: '{}'", 
                  prev_summary.as_str().unwrap_or("N/A").chars().take(80).collect::<String>());
        }
    }
    info!("📊 Total enhanced nodes: {}", enhanced_nodes.len());

    Ok(())
}

/// Demo with mock data when no API key is available
async fn demo_with_mock_data() -> Result<(), Box<dyn std::error::Error>> {
    info!("📝 Demonstrating SummaryExtractor structure with mock data...");

    // Show configuration options
    let config = SummaryExtractorConfig::new()
        .with_summaries(vec![
            SummaryType::PrevSummary,
            SummaryType::SelfSummary,
            SummaryType::NextSummary,
        ])
        .with_show_progress(true)
        .with_num_workers(2)
        .with_max_context_length(3000);

    info!("📋 SummaryExtractor Configuration:");
    info!("  - Summary types: {:?}", config.summaries.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    info!("  - Show progress: {}", config.show_progress);
    info!("  - Number of workers: {}", config.num_workers);
    info!("  - Max context length: {}", config.max_context_length);

    // Show what the extractor would do
    let sample_nodes = create_sample_nodes();
    info!("📄 Sample nodes created: {}", sample_nodes.len());
    info!("📝 SummaryExtractor would process these nodes to generate intelligent summaries");
    info!("💡 The extractor uses LLM to analyze content and generate contextual summaries");
    info!("🔗 Adjacent context summaries provide rich context for each node");

    // Show metadata keys that would be added
    info!("📊 Metadata keys that would be added:");
    for summary_type in &config.summaries {
        info!("  - {}: Summary for {}", summary_type.metadata_key(), summary_type.as_str());
    }

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
            .temperature(0.3) // Lower temperature for more consistent summaries
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

/// Create sample nodes with different topics
fn create_sample_nodes() -> Vec<Node> {
    let contents = vec![
        "Artificial intelligence and machine learning are transforming industries across the globe.",
        "Climate change represents one of the most significant challenges facing humanity today.",
        "Quantum computing promises to revolutionize computational capabilities in the coming decades.",
    ];

    contents
        .into_iter()
        .enumerate()
        .map(|(i, content)| create_sample_node(content, &format!("node_{}", i)))
        .collect()
}

/// Create sequential nodes that form a coherent narrative
fn create_sequential_nodes() -> Vec<Node> {
    let contents = vec![
        "The history of computing began with mechanical calculators and evolved through several distinct phases.",
        "The invention of the transistor in 1947 marked the beginning of the electronic computing era.",
        "Personal computers emerged in the 1970s and 1980s, bringing computing power to individual users.",
        "The internet revolution of the 1990s connected computers globally and transformed communication.",
        "Mobile computing and smartphones have made computing ubiquitous in the 21st century.",
    ];

    contents
        .into_iter()
        .enumerate()
        .map(|(i, content)| create_sample_node(content, &format!("sequential_node_{}", i)))
        .collect()
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
