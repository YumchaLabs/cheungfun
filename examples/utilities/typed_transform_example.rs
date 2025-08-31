//! TypedTransform System Example
//!
//! This example demonstrates the new TypedTransform system that provides
//! compile-time type safety for document processing pipelines.
//!
//! Key features demonstrated:
//! 1. Type-safe document and node processing
//! 2. Compile-time pipeline validation
//! 3. Zero runtime overhead type checking
//! 4. Modern Rust async/await patterns
//!
//! To run this example:
//! ```bash
//! cargo run --example typed_transform_example
//! ```

use cheungfun_core::{
    traits::{TypedTransform, TypedData, NodeState},
    Document,
};
use cheungfun_indexing::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 TypedTransform System Example");
    println!("================================");

    // Create sample documents
    let documents = create_sample_documents();
    println!("📄 Created {} sample documents", documents.len());

    // Demonstrate type-safe document processing
    await_document_processing_example(&documents).await?;

    // Demonstrate type-safe node processing  
    await_node_processing_example(&documents).await?;

    // Demonstrate compile-time type safety
    demonstrate_compile_time_safety().await?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Create sample documents for testing
fn create_sample_documents() -> Vec<Document> {
    vec![
        Document::new("This is the first document. It contains multiple sentences. Each sentence provides valuable information for processing."),
        Document::new("The second document focuses on technical content. It discusses advanced algorithms and data structures. Performance optimization is a key topic."),
        Document::new("Document three explores machine learning concepts. Neural networks and deep learning are central themes. The content is highly technical."),
    ]
}

/// Demonstrate type-safe document processing
async fn await_document_processing_example(documents: &[Document]) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📝 Document Processing Example");
    println!("------------------------------");

    // Create a sentence splitter (Documents -> Nodes)
    let splitter = SentenceSplitter::from_defaults(100, 20)?;
    
    // Create typed input from documents
    let typed_input = TypedData::from_documents(documents.to_vec());
    println!("📥 Input: {} documents", typed_input.documents().len());

    // Apply transformation with compile-time type safety
    let typed_output = splitter.transform(typed_input).await?;
    let nodes = typed_output.nodes();
    
    println!("📤 Output: {} nodes", nodes.len());
    
    // Display first few nodes
    for (i, node) in nodes.iter().take(3).enumerate() {
        println!("  Node {}: {} chars", i + 1, node.content.len());
    }

    Ok(())
}

/// Demonstrate type-safe node processing
async fn await_node_processing_example(documents: &[Document]) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Node Processing Example");
    println!("--------------------------");

    // First, convert documents to nodes
    let splitter = SentenceSplitter::from_defaults(100, 20)?;
    let typed_input = TypedData::from_documents(documents.to_vec());
    let typed_nodes = splitter.transform(typed_input).await?;

    // Now process nodes with metadata extractor (Nodes -> Nodes)
    let metadata_extractor = MetadataExtractor::new();
    let enhanced_nodes = metadata_extractor.transform(typed_nodes).await?;
    let final_nodes = enhanced_nodes.nodes();

    println!("📤 Enhanced {} nodes with metadata", final_nodes.len());
    
    // Display metadata for first node
    if let Some(first_node) = final_nodes.first() {
        println!("  First node metadata:");
        for (key, value) in &first_node.metadata {
            println!("    {}: {}", key, value);
        }
    }

    Ok(())
}

/// Demonstrate compile-time type safety features
async fn demonstrate_compile_time_safety() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔒 Compile-Time Type Safety");
    println!("---------------------------");

    println!("✅ The following code compiles successfully:");
    println!("   Documents -> SentenceSplitter -> Nodes");
    println!("   Nodes -> MetadataExtractor -> Nodes");
    
    println!("\n❌ The following would cause compile errors:");
    println!("   // Nodes -> SentenceSplitter  // Error: SentenceSplitter expects Documents");
    println!("   // Documents -> MetadataExtractor  // Error: MetadataExtractor expects Nodes");
    
    println!("\n🎯 Benefits:");
    println!("   • Type errors caught at compile time");
    println!("   • Zero runtime overhead");
    println!("   • Better IDE support and autocomplete");
    println!("   • Impossible to create invalid pipelines");

    Ok(())
}

/// Example of how to create a custom TypedTransform
#[derive(Debug)]
pub struct CustomNodeProcessor {
    prefix: String,
}

impl CustomNodeProcessor {
    pub fn new(prefix: String) -> Self {
        Self { prefix }
    }
}

#[async_trait::async_trait]
impl TypedTransform<NodeState, NodeState> for CustomNodeProcessor {
    async fn transform(&self, input: TypedData<NodeState>) -> cheungfun_core::Result<TypedData<NodeState>> {
        let nodes = input.nodes();
        let processed_nodes = nodes
            .iter()
            .map(|node| {
                let mut new_node = node.clone();
                new_node.content = format!("{}: {}", self.prefix, node.content);
                new_node
            })
            .collect();
        
        Ok(TypedData::from_nodes(processed_nodes))
    }

    fn name(&self) -> &'static str {
        "CustomNodeProcessor"
    }

    fn description(&self) -> &'static str {
        "Adds a custom prefix to all node content"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_typed_transform_system() {
        let documents = create_sample_documents();
        
        // Test document processing
        let splitter = SentenceSplitter::from_defaults(100, 20).unwrap();
        let typed_input = TypedData::from_documents(documents);
        let result = splitter.transform(typed_input).await;
        
        assert!(result.is_ok());
        let nodes = result.unwrap().nodes();
        assert!(!nodes.is_empty());
    }

    #[tokio::test]
    async fn test_custom_processor() {
        let documents = create_sample_documents();
        
        // Convert to nodes first
        let splitter = SentenceSplitter::from_defaults(100, 20).unwrap();
        let typed_input = TypedData::from_documents(documents);
        let nodes_data = splitter.transform(typed_input).await.unwrap();
        
        // Apply custom processor
        let custom_processor = CustomNodeProcessor::new("PROCESSED".to_string());
        let result = custom_processor.transform(nodes_data).await.unwrap();
        let processed_nodes = result.nodes();
        
        // Verify prefix was added
        for node in processed_nodes.iter() {
            assert!(node.content.starts_with("PROCESSED:"));
        }
    }
}
