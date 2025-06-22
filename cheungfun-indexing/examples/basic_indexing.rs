//! Basic indexing example demonstrating the core functionality.

use cheungfun_core::traits::{Loader, NodeTransformer, Transformer};
use cheungfun_indexing::loaders::LoaderConfig;
use cheungfun_indexing::prelude::*;
use std::fs;
use std::sync::Arc;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Note: In a real application, you might want to initialize logging
    // tracing_subscriber::init();

    println!("🚀 Cheungfun Indexing Example");
    println!("==============================");

    // Create a temporary directory with sample documents
    let temp_dir = create_sample_documents()?;
    println!(
        "📁 Created sample documents in: {}",
        temp_dir.path().display()
    );

    // Example 1: Load a single file
    println!("\n📄 Example 1: Loading a single file");
    let file_path = temp_dir.path().join("document1.txt");
    let file_loader = FileLoader::new(&file_path)?;
    let documents = file_loader.load().await?;
    println!("   Loaded {} document(s) from file", documents.len());
    for doc in &documents {
        println!("   - Content length: {} characters", doc.content.len());
        println!(
            "   - Metadata keys: {:?}",
            doc.metadata.keys().collect::<Vec<_>>()
        );
    }

    // Example 2: Load all files from a directory
    println!("\n📂 Example 2: Loading from directory");
    let dir_loader = DirectoryLoader::new(temp_dir.path())?;
    let all_documents = dir_loader.load().await?;
    println!(
        "   Loaded {} document(s) from directory",
        all_documents.len()
    );

    // Example 3: Text splitting
    println!("\n✂️  Example 3: Text splitting");
    let splitter = TextSplitter::new(200, 50); // 200 chars with 50 char overlap
    let mut all_nodes = Vec::new();

    for doc in all_documents {
        let nodes = splitter.transform(doc).await?;
        println!("   Split document into {} chunks", nodes.len());
        all_nodes.extend(nodes);
    }

    println!("   Total nodes created: {}", all_nodes.len());

    // Example 4: Metadata extraction
    println!("\n🏷️  Example 4: Metadata extraction");
    let metadata_extractor = MetadataExtractor::new();
    let enriched_nodes = metadata_extractor.transform_batch(all_nodes).await?;

    for (i, node) in enriched_nodes.iter().take(3).enumerate() {
        println!("   Node {}: {} metadata fields", i + 1, node.metadata.len());
        if let Some(word_count) = node.metadata.get("word_count") {
            println!("     - Word count: {}", word_count);
        }
        if let Some(char_count) = node.metadata.get("character_count") {
            println!("     - Character count: {}", char_count);
        }
    }

    // Example 5: Complete pipeline
    println!("\n🔄 Example 5: Complete indexing pipeline");

    // Create a new temporary directory for pipeline example
    let pipeline_temp_dir = create_sample_documents()?;

    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(Arc::new(DirectoryLoader::new(pipeline_temp_dir.path())?))
        .with_transformer(Arc::new(TextSplitter::new(300, 75)))
        .with_node_transformer(Arc::new(MetadataExtractor::new()))
        .build()?;

    // Validate pipeline
    pipeline.validate()?;
    println!("   ✅ Pipeline validation passed");

    // Run pipeline with progress reporting
    let stats = pipeline
        .run_with_progress(Box::new(|progress| {
            println!(
                "   📊 {}: {}/{}",
                progress.stage,
                progress.processed,
                progress.total.unwrap_or(0)
            );
        }))
        .await?;

    println!("   📈 Pipeline Results:");
    println!("     - Documents processed: {}", stats.documents_processed);
    println!("     - Nodes created: {}", stats.nodes_created);
    println!("     - Processing time: {:?}", stats.processing_time);
    println!("     - Errors: {}", stats.errors.len());

    // Example 6: Configuration options
    println!("\n⚙️  Example 6: Configuration options");

    let custom_config = LoaderConfig::new()
        .with_max_file_size(1024 * 1024) // 1MB limit
        .with_include_extensions(vec!["txt".to_string(), "md".to_string()])
        .with_continue_on_error(true);

    let configured_loader = DirectoryLoader::with_config(temp_dir.path(), custom_config)?;
    let filtered_docs = configured_loader.load().await?;
    println!(
        "   Loaded {} documents with custom config",
        filtered_docs.len()
    );

    let custom_splitter_config = SplitterConfig::new(500, 100)
        .with_respect_sentence_boundaries(true)
        .with_min_chunk_size(50);

    let configured_splitter = TextSplitter::with_config(custom_splitter_config);
    println!("   Created splitter with custom configuration");

    let custom_metadata_config = MetadataConfig::new()
        .with_title_extraction(true)
        .with_statistics(true);

    let configured_extractor = MetadataExtractor::with_config(custom_metadata_config);
    println!("   Created metadata extractor with custom configuration");

    println!("\n✨ Example completed successfully!");
    Ok(())
}

/// Create sample documents for demonstration.
fn create_sample_documents() -> Result<TempDir, Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;

    // Document 1: Simple text
    let doc1_content = r#"
This is the first sample document for the Cheungfun indexing example.
It contains multiple paragraphs and sentences to demonstrate text splitting.

The document discusses various topics including natural language processing,
information retrieval, and document indexing techniques.

This paragraph contains technical terms like embeddings, vector databases,
and semantic search that are commonly used in RAG systems.
"#;

    // Document 2: Markdown format
    let doc2_content = r#"
# Technical Documentation

## Introduction

This is a **markdown document** that demonstrates how the indexing system
handles structured content with headers and formatting.

## Features

- Text extraction from various formats
- Intelligent chunking strategies  
- Metadata enrichment
- Pipeline processing

## Code Example

```rust
let loader = FileLoader::new("document.txt")?;
let documents = loader.load().await?;
```

## Conclusion

The indexing system provides a flexible and powerful way to process
documents for retrieval-augmented generation applications.
"#;

    // Document 3: Longer content
    let doc3_content = r#"
Understanding Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is a powerful technique that combines
the strengths of large language models with external knowledge sources.
This approach allows AI systems to access up-to-date information and
domain-specific knowledge that may not be present in their training data.

The RAG process typically involves several key steps:

1. Document Ingestion: Raw documents are loaded from various sources
2. Text Processing: Documents are cleaned and preprocessed
3. Chunking: Large documents are split into manageable pieces
4. Embedding Generation: Text chunks are converted to vector representations
5. Vector Storage: Embeddings are stored in a searchable database
6. Query Processing: User queries are embedded and used to retrieve relevant chunks
7. Response Generation: Retrieved context is used to generate informed responses

Each step in this pipeline is crucial for the overall effectiveness of the
RAG system. The quality of document processing directly impacts the quality
of retrieved information and, consequently, the generated responses.

Modern RAG systems employ sophisticated techniques for each stage, including
advanced text splitting algorithms, high-quality embedding models, and
efficient vector databases optimized for similarity search.
"#;

    // Write documents to files
    fs::write(temp_dir.path().join("document1.txt"), doc1_content.trim())?;
    fs::write(temp_dir.path().join("document2.md"), doc2_content.trim())?;
    fs::write(temp_dir.path().join("document3.txt"), doc3_content.trim())?;

    Ok(temp_dir)
}
