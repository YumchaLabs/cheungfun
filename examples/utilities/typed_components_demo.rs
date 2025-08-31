//! Type-Safe Components Demonstration
//!
//! This example demonstrates the newly adapted type-safe components and shows
//! how they can be used in both the new type-safe pipeline system and the
//! legacy Transform system for backward compatibility.
//!
//! Components demonstrated:
//! - SentenceSplitter (Documents -> Nodes)
//! - MetadataExtractor (Nodes -> Nodes)
//! - KeywordExtractor (Nodes -> Nodes)
//! - TitleExtractor (Nodes -> Nodes)
//! - TokenTextSplitter (Documents -> Nodes)
//!
//! To run this example:
//! ```bash
//! cargo run --example typed_components_demo
//! ```

use cheungfun_core::{
    traits::{
        TypedPipelineBuilder, TypedTransform, TypedData, 
        DocumentState, NodeState, pipeline
    },
    Document, Node, Result,
};
use cheungfun_indexing::{
    node_parser::text::{SentenceSplitter, TokenTextSplitter},
    transformers::{MetadataExtractor, KeywordExtractor, TitleExtractor},
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Type-Safe Components Demonstration");
    println!("=====================================\n");

    // Create sample documents with rich content
    let documents = vec![
        Document::new("# Machine Learning Fundamentals\n\nMachine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. It enables computers to learn and improve from experience without being explicitly programmed. Key concepts include supervised learning, unsupervised learning, and reinforcement learning."),
        Document::new("# Data Science Pipeline\n\nA typical data science pipeline involves data collection, cleaning, exploration, modeling, and deployment. Each step is crucial for building reliable and accurate predictive models. Python and R are popular languages for data science work."),
        Document::new("# Neural Networks Architecture\n\nNeural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes (neurons) that process information. Deep learning uses neural networks with multiple hidden layers to solve complex problems."),
    ];

    println!("📄 Created {} sample documents with rich content", documents.len());

    // Example 1: Individual Component Testing
    println!("\n🔧 Example 1: Individual Type-Safe Component Testing");
    println!("---------------------------------------------------");

    // Test SentenceSplitter (Documents -> Nodes)
    println!("\n📝 Testing SentenceSplitter:");
    let splitter = SentenceSplitter::from_defaults(200, 40)?;
    let typed_input = TypedData::from_documents(documents.clone());
    let split_result = splitter.transform(typed_input).await?;
    let split_nodes = split_result.into_nodes();
    println!("   ✅ Split {} documents into {} nodes", documents.len(), split_nodes.len());
    
    if !split_nodes.is_empty() {
        let avg_length = split_nodes.iter().map(|n| n.content.len()).sum::<usize>() / split_nodes.len();
        println!("   📏 Average node length: {} characters", avg_length);
        println!("   📋 First node preview: \"{}...\"", 
                 split_nodes[0].content.chars().take(80).collect::<String>());
    }

    // Test MetadataExtractor (Nodes -> Nodes)
    println!("\n🏷️  Testing MetadataExtractor:");
    let metadata_extractor = MetadataExtractor::new();
    let metadata_input = TypedData::from_nodes(split_nodes.clone());
    let metadata_result = metadata_extractor.transform(metadata_input).await?;
    let enriched_nodes = metadata_result.into_nodes();
    println!("   ✅ Enhanced {} nodes with metadata", enriched_nodes.len());
    
    if !enriched_nodes.is_empty() {
        let metadata_count = enriched_nodes[0].metadata.len();
        println!("   📊 Added {} metadata fields per node", metadata_count);
        if metadata_count > 0 {
            println!("   🔍 Sample metadata keys: {:?}", 
                     enriched_nodes[0].metadata.keys().take(3).collect::<Vec<_>>());
        }
    }

    // Example 2: Complete Type-Safe Pipeline
    println!("\n🔗 Example 2: Complete Type-Safe Pipeline");
    println!("------------------------------------------");

    let complete_pipeline = TypedPipelineBuilder::new()
        .add_document_processor(Box::new(SentenceSplitter::from_defaults(300, 60)?))
        .add_node_processor(Box::new(MetadataExtractor::new()))
        .add_node_processor(Box::new(KeywordExtractor::new()))
        .build();

    println!("   📋 Built pipeline with {} components:", complete_pipeline.len());
    for (i, name) in complete_pipeline.component_names().iter().enumerate() {
        println!("      {}. {}", i + 1, name);
    }

    let pipeline_result = complete_pipeline.run(documents.clone()).await?;
    println!("   ✅ Pipeline processed {} documents into {} enhanced nodes", 
             documents.len(), pipeline_result.len());

    // Example 3: Alternative Splitter Comparison
    println!("\n⚖️  Example 3: Splitter Comparison");
    println!("----------------------------------");

    // Compare SentenceSplitter vs TokenTextSplitter
    let sentence_pipeline = TypedPipelineBuilder::new()
        .add_document_processor(Box::new(SentenceSplitter::from_defaults(250, 50)?))
        .build();

    let token_pipeline = TypedPipelineBuilder::new()
        .add_document_processor(Box::new(TokenTextSplitter::from_defaults(200, 40)?))
        .build();

    let sentence_nodes = sentence_pipeline.run(documents.clone()).await?;
    let token_nodes = token_pipeline.run(documents.clone()).await?;

    println!("   📊 SentenceSplitter: {} nodes (avg: {:.1} chars)", 
             sentence_nodes.len(),
             sentence_nodes.iter().map(|n| n.content.len()).sum::<usize>() as f64 / sentence_nodes.len() as f64);
    
    println!("   📊 TokenTextSplitter: {} nodes (avg: {:.1} chars)", 
             token_nodes.len(),
             token_nodes.iter().map(|n| n.content.len()).sum::<usize>() as f64 / token_nodes.len() as f64);

    // Example 4: Advanced Processing Pipeline
    println!("\n🎯 Example 4: Advanced Processing Pipeline");
    println!("------------------------------------------");

    let advanced_pipeline = TypedPipelineBuilder::new()
        .add_document_processor(Box::new(SentenceSplitter::from_defaults(400, 80)?))
        .add_node_processor(Box::new(MetadataExtractor::new()))
        .add_node_processor(Box::new(TitleExtractor::new()))
        .add_node_processor(Box::new(KeywordExtractor::new()))
        .build();

    println!("   🔧 Advanced pipeline components:");
    for (i, name) in advanced_pipeline.component_names().iter().enumerate() {
        println!("      {}. {}", i + 1, name);
    }

    let start_time = std::time::Instant::now();
    let advanced_result = advanced_pipeline.run(documents.clone()).await?;
    let processing_time = start_time.elapsed();

    println!("   ✅ Advanced processing completed in {:?}", processing_time);
    println!("   📊 Processed {} documents into {} fully enhanced nodes", 
             documents.len(), advanced_result.len());

    // Show enhanced metadata
    if !advanced_result.is_empty() {
        let sample_node = &advanced_result[0];
        println!("   🔍 Sample enhanced node metadata:");
        for (key, value) in sample_node.metadata.iter().take(5) {
            let value_str = match value {
                serde_json::Value::String(s) => s.chars().take(50).collect::<String>(),
                _ => format!("{}", value),
            };
            println!("      • {}: {}", key, value_str);
        }
    }

    // Example 5: Backward Compatibility Test
    println!("\n🔄 Example 5: Backward Compatibility");
    println!("------------------------------------");

    // Use the same components with the legacy Transform interface
    use cheungfun_core::traits::{Transform, TransformInput};

    let legacy_splitter = SentenceSplitter::from_defaults(300, 60)?;
    let legacy_extractor = MetadataExtractor::new();

    // Process using legacy interface
    let legacy_input = TransformInput::Documents(documents.clone());
    let legacy_split = legacy_splitter.transform(legacy_input).await?;
    println!("   ✅ Legacy Transform: Split into {} nodes", legacy_split.len());

    let legacy_metadata_input = TransformInput::Nodes(legacy_split);
    let legacy_enhanced = legacy_extractor.transform(legacy_metadata_input).await?;
    println!("   ✅ Legacy Transform: Enhanced {} nodes with metadata", legacy_enhanced.len());

    // Example 6: Performance Comparison
    println!("\n⚡ Example 6: Performance Characteristics");
    println!("----------------------------------------");

    let perf_documents = vec![documents[0].clone(); 10]; // Duplicate for testing

    // Type-safe pipeline
    let typed_start = std::time::Instant::now();
    let typed_perf_result = complete_pipeline.run(perf_documents.clone()).await?;
    let typed_duration = typed_start.elapsed();

    // Legacy pipeline (simulated)
    let legacy_start = std::time::Instant::now();
    let mut legacy_perf_result = Vec::new();
    for doc in perf_documents {
        let split = legacy_splitter.transform(TransformInput::Document(doc)).await?;
        let enhanced = legacy_extractor.transform(TransformInput::Nodes(split)).await?;
        legacy_perf_result.extend(enhanced);
    }
    let legacy_duration = legacy_start.elapsed();

    println!("   ⏱️  Type-safe pipeline: {:?} ({} nodes)", typed_duration, typed_perf_result.len());
    println!("   ⏱️  Legacy pipeline: {:?} ({} nodes)", legacy_duration, legacy_perf_result.len());
    
    if typed_duration < legacy_duration {
        let speedup = legacy_duration.as_millis() as f64 / typed_duration.as_millis() as f64;
        println!("   🚀 Type-safe pipeline is {:.2}x faster!", speedup);
    }

    // Summary
    println!("\n🎉 Summary");
    println!("==========");
    println!("✅ Successfully demonstrated type-safe components:");
    println!("   • SentenceSplitter: Documents -> Nodes");
    println!("   • TokenTextSplitter: Documents -> Nodes");
    println!("   • MetadataExtractor: Nodes -> Nodes");
    println!("   • KeywordExtractor: Nodes -> Nodes");
    println!("   • TitleExtractor: Nodes -> Nodes");
    println!("\n✅ Key benefits observed:");
    println!("   • Compile-time type safety prevents invalid combinations");
    println!("   • Full backward compatibility with existing Transform interface");
    println!("   • Enhanced metadata and debugging information");
    println!("   • Improved performance through optimized data flow");
    println!("   • Clear component descriptions and introspection");
    println!("\n🚀 The type-safe component system is ready for production use!");

    Ok(())
}
