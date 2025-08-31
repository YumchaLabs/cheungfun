//! Type-Safe Pipeline System Example
//!
//! This example demonstrates the new type-safe pipeline system that provides
//! compile-time guarantees for component compatibility while maintaining full
//! backward compatibility with existing Transform implementations.
//!
//! Key features demonstrated:
//! 1. Compile-time type safety for pipeline composition
//! 2. Automatic prevention of invalid component combinations
//! 3. Seamless integration with existing Transform components
//! 4. Enhanced error messages and debugging capabilities
//!
//! To run this example:
//! ```bash
//! cargo run --example typed_pipeline_example
//! ```

use cheungfun_core::{
    traits::{
        TypedPipelineBuilder, TypedTransform, TypedData, 
        DocumentState, NodeState, pipeline
    },
    Document, Node, Result,
};
use cheungfun_indexing::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Type-Safe Pipeline System Example");
    println!("=====================================\n");

    // Create sample documents
    let documents = vec![
        Document::new("This is the first document. It contains multiple sentences. Each sentence will be processed separately."),
        Document::new("Here is the second document. It also has several sentences. The pipeline will handle them efficiently."),
        Document::new("The third document demonstrates batch processing. Multiple documents are processed together. This improves performance significantly."),
    ];

    println!("📄 Created {} sample documents", documents.len());

    // Example 1: Type-safe pipeline construction
    println!("\n🔧 Example 1: Type-Safe Pipeline Construction");
    println!("----------------------------------------------");

    // ✅ This compiles successfully - valid pipeline composition
    let pipeline = TypedPipelineBuilder::new()
        .add_document_processor(SentenceSplitter::from_defaults(200, 40)?)
        .add_node_processor(MetadataExtractor::new())
        .build();

    println!("   ✅ Created valid pipeline with {} components", pipeline.len());
    println!("   📋 Components: {:?}", pipeline.component_names());

    // Execute the pipeline
    let nodes = pipeline.run(documents.clone()).await?;
    println!("   📊 Pipeline processed {} documents into {} nodes", documents.len(), nodes.len());

    if !nodes.is_empty() {
        let avg_length = nodes.iter().map(|n| n.content.len()).sum::<usize>() / nodes.len();
        println!("   📏 Average node length: {} characters", avg_length);
    }

    // Example 2: Compile-time error prevention
    println!("\n🛡️  Example 2: Compile-Time Error Prevention");
    println!("---------------------------------------------");
    
    println!("   The following code would cause a COMPILE ERROR:");
    println!("   ```rust");
    println!("   // ❌ This won't compile!");
    println!("   let invalid_pipeline = TypedPipelineBuilder::new()");
    println!("       .add_node_processor(MetadataExtractor::new())  // Error!");
    println!("       .add_document_processor(SentenceSplitter::new());");
    println!("   ```");
    println!("   💡 The type system prevents adding document processors after node processors!");

    // Example 3: Backward compatibility
    println!("\n🔄 Example 3: Backward Compatibility");
    println!("------------------------------------");

    // Create pipeline using legacy Transform components
    let legacy_transforms: Vec<Box<dyn cheungfun_core::traits::Transform>> = vec![
        Box::new(SentenceSplitter::from_defaults(300, 60)?),
        Box::new(KeywordExtractor::new()),
    ];

    let legacy_pipeline = cheungfun_core::traits::pipeline_from_transforms(legacy_transforms);
    println!("   📦 Created pipeline from legacy Transform components");

    let legacy_nodes = legacy_pipeline.run(documents.clone()).await?;
    println!("   📊 Legacy pipeline processed {} documents into {} nodes", 
             documents.len(), legacy_nodes.len());

    // Example 4: Mixed pipeline (new + legacy)
    println!("\n🔀 Example 4: Mixed Pipeline (Type-Safe + Legacy)");
    println!("-------------------------------------------------");

    let mixed_pipeline = TypedPipelineBuilder::new()
        .add_document_processor(SentenceSplitter::from_defaults(250, 50)?)
        .add_transform(Box::new(TitleExtractor::new()))  // Legacy component
        .add_node_processor(MetadataExtractor::new())    // Type-safe component
        .build();

    println!("   🔗 Created mixed pipeline with {} components", mixed_pipeline.len());
    
    let mixed_nodes = mixed_pipeline.run(documents.clone()).await?;
    println!("   📊 Mixed pipeline processed {} documents into {} nodes", 
             documents.len(), mixed_nodes.len());

    // Example 5: Pipeline introspection
    println!("\n🔍 Example 5: Pipeline Introspection");
    println!("------------------------------------");

    println!("   📋 Pipeline components:");
    for (i, name) in mixed_pipeline.component_names().iter().enumerate() {
        println!("      {}. {}", i + 1, name);
    }
    
    println!("   📏 Pipeline length: {}", mixed_pipeline.len());
    println!("   🔍 Is empty: {}", mixed_pipeline.is_empty());

    // Example 6: Error handling and debugging
    println!("\n🐛 Example 6: Enhanced Error Handling");
    println!("-------------------------------------");

    // Create a pipeline that might fail
    let error_prone_pipeline = TypedPipelineBuilder::new()
        .add_document_processor(SentenceSplitter::from_defaults(100, 20)?)
        .build();

    // Try with empty documents to demonstrate error handling
    match error_prone_pipeline.run(vec![]).await {
        Ok(nodes) => println!("   ✅ Processed {} nodes", nodes.len()),
        Err(e) => println!("   ⚠️  Pipeline error (expected): {}", e),
    }

    // Example 7: Performance comparison
    println!("\n⚡ Example 7: Performance Characteristics");
    println!("----------------------------------------");

    let start_time = std::time::Instant::now();
    let performance_pipeline = pipeline();
        .add_document_processor(SentenceSplitter::from_defaults(400, 80)?)
        .add_node_processor(MetadataExtractor::new())
        .build();

    let perf_nodes = performance_pipeline.run(documents.clone()).await?;
    let duration = start_time.elapsed();

    println!("   ⏱️  Processing time: {:?}", duration);
    println!("   📊 Throughput: {:.2} nodes/ms", 
             perf_nodes.len() as f64 / duration.as_millis() as f64);

    // Example 8: Advanced pipeline patterns
    println!("\n🎯 Example 8: Advanced Pipeline Patterns");
    println!("----------------------------------------");

    // Conditional pipeline construction
    let use_advanced_processing = true;
    
    let mut builder = TypedPipelineBuilder::new()
        .add_document_processor(SentenceSplitter::from_defaults(300, 60)?);
    
    if use_advanced_processing {
        builder = builder
            .add_node_processor(MetadataExtractor::new())
            .add_node_processor(KeywordExtractor::new());
    }
    
    let conditional_pipeline = builder.build();
    println!("   🔀 Built conditional pipeline with {} components", 
             conditional_pipeline.len());

    let final_nodes = conditional_pipeline.run(documents).await?;
    println!("   📊 Final processing: {} nodes with enhanced metadata", final_nodes.len());

    // Summary
    println!("\n🎉 Summary");
    println!("==========");
    println!("✅ Type-safe pipeline system provides:");
    println!("   • Compile-time component compatibility validation");
    println!("   • Zero runtime overhead for type checking");
    println!("   • Full backward compatibility with existing code");
    println!("   • Enhanced error messages and debugging");
    println!("   • Flexible pipeline composition patterns");
    println!("\n🚀 The type-safe pipeline system makes RAG development safer and more efficient!");

    Ok(())
}

// ============================================================================
// Helper Functions for Demonstration
// ============================================================================

/// Demonstrate compile-time error prevention (this function won't compile if uncommented)
#[allow(dead_code)]
fn demonstrate_compile_error() {
    // Uncomment the following code to see compile-time error:
    /*
    let _invalid_pipeline = TypedPipelineBuilder::new()
        .add_node_processor(MetadataExtractor::new())      // ❌ Compile error!
        .add_document_processor(SentenceSplitter::new());  // ❌ Can't add after node processor!
    */
}

/// Show how the type system guides correct usage
#[allow(dead_code)]
fn demonstrate_correct_usage() -> Result<(), Box<dyn std::error::Error>> {
    // ✅ This compiles correctly:
    let _correct_pipeline = TypedPipelineBuilder::new()
        .add_document_processor(SentenceSplitter::from_defaults(512, 100)?)
        .add_node_processor(MetadataExtractor::new())
        .add_node_processor(KeywordExtractor::new())
        .build();
    
    Ok(())
}
