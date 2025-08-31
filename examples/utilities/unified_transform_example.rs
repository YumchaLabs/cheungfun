//! Unified Transform Interface Example
//!
//! This example demonstrates the new unified Transform interface architecture
//! that follows LlamaIndex's TransformComponent design pattern.
//!
//! Key features demonstrated:
//! 1. Unified Transform trait for all processing components
//! 2. Type-safe TransformInput enum
//! 3. Polymorphic processing capabilities
//! 4. Pipeline integration with unified interface
//!
//! To run this example:
//! ```bash
//! cargo run --example unified_transform_example
//! ```

use cheungfun_core::{
    traits::{TypedTransform, TypedData, DocumentState, NodeState},
    Document,
};
use cheungfun_indexing::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 Cheungfun Unified Transform Interface Example");
    println!("===============================================\n");

    // Create sample documents
    let documents = create_sample_documents();
    println!("📄 Created {} sample documents", documents.len());

    // Example 1: Direct Transform interface usage
    println!("\n📝 Example 1: Direct Transform Interface Usage");
    println!("----------------------------------------------");

    // Create different transform components
    let sentence_splitter = SentenceSplitter::from_defaults(300, 75)?;
    let _token_splitter = TokenTextSplitter::from_defaults(250, 50)?;
    let metadata_extractor = MetadataExtractor::new();

    // Test sentence splitter using unified TypedTransform interface
    println!("🔧 Testing SentenceSplitter...");
    let input = TypedData::from_documents(documents.clone());
    let result = sentence_splitter.transform(input).await?;
    let nodes = result.into_nodes();
    println!("   ✅ Created {} nodes", nodes.len());

    // Test metadata extractor on the nodes
    println!("🔧 Testing MetadataExtractor...");
    let input = TypedData::from_nodes(nodes);
    let result = metadata_extractor.transform(input).await?;
    let enriched_nodes = result.into_nodes();
    println!(
        "   ✅ Enriched {} nodes with metadata",
        enriched_nodes.len()
    );

    // Example 2: Polymorphic processing
    println!("\n🔄 Example 2: Polymorphic Processing");
    println!("-----------------------------------");

    // Create a vector of different transforms
    let transforms: Vec<Box<dyn TypedTransform<DocumentState, NodeState>>> = vec![
        Box::new(SentenceSplitter::from_defaults(400, 80)?),
        Box::new(TokenTextSplitter::from_defaults(300, 60)?),
    ];

    let test_input = TypedData::from_documents(documents.clone());
    for transform in transforms {
        let result = transform.transform(test_input.clone()).await?;
        let nodes = result.into_nodes();
        println!("   📊 {}: {} nodes", transform.name(), nodes.len());

        if !nodes.is_empty() {
            let avg_length = nodes.iter().map(|n| n.content.len()).sum::<usize>() / nodes.len();
            println!("      Average node length: {} characters", avg_length);
        }
    }

    // Example 3: Pipeline integration
    println!("\n🔧 Example 3: Pipeline Integration");
    println!("----------------------------------");

    // Create a temporary directory with sample files
    let temp_dir = create_temp_directory_with_files()?;

    // Build pipeline using unified TypedTransform interface
    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(Arc::new(DirectoryLoader::new(temp_dir.path())?))
        .with_transformer(Arc::new(SentenceSplitter::from_defaults(500, 100)?))
        .with_node_transformer(Arc::new(MetadataExtractor::new()))
        .build()?;

    println!("   ✅ Pipeline built successfully");

    // Validate pipeline
    pipeline.validate()?;
    println!("   ✅ Pipeline validation passed");

    println!("\n🎉 Unified Transform Interface Example Complete!");
    println!("===============================================");
    println!("✅ Demonstrated unified Transform interface");
    println!("✅ Showed polymorphic processing");
    println!("✅ Integrated with pipeline architecture");
    println!("\n📚 Key Benefits:");
    println!("   • Single interface for all transformations");
    println!("   • Type-safe input handling");
    println!("   • LlamaIndex-compatible design");
    println!("   • Simplified pipeline integration");

    Ok(())
}

/// Create sample documents for testing
fn create_sample_documents() -> Vec<Document> {
    let mut documents = Vec::new();

    // Document 1: Technical content
    let mut metadata1 = HashMap::new();
    metadata1.insert(
        "type".to_string(),
        serde_json::Value::String("technical".to_string()),
    );
    metadata1.insert(
        "source".to_string(),
        serde_json::Value::String("example".to_string()),
    );

    documents.push(Document {
        id: uuid::Uuid::new_v4(),
        content: "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety. It accomplishes these goals by being memory safe without using garbage collection. Rust has great documentation, a friendly compiler with useful error messages, and top-notch tooling.".to_string(),
        metadata: metadata1,
        embedding: None,
    });

    // Document 2: General content
    let mut metadata2 = HashMap::new();
    metadata2.insert(
        "type".to_string(),
        serde_json::Value::String("general".to_string()),
    );
    metadata2.insert(
        "source".to_string(),
        serde_json::Value::String("example".to_string()),
    );

    documents.push(Document {
        id: uuid::Uuid::new_v4(),
        content: "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Through the use of algorithms and statistical models, machine learning enables computers to improve their performance on a specific task through experience.".to_string(),
        metadata: metadata2,
        embedding: None,
    });

    documents
}

/// Create a temporary directory with sample files for pipeline testing
fn create_temp_directory_with_files() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let temp_dir = tempfile::TempDir::new()?;

    // Create sample files
    std::fs::write(
        temp_dir.path().join("doc1.txt"),
        "This is the first sample document. It contains information about Rust programming language and its features."
    )?;

    std::fs::write(
        temp_dir.path().join("doc2.txt"),
        "This is the second sample document. It discusses machine learning concepts and applications in modern technology."
    )?;

    Ok(temp_dir)
}
