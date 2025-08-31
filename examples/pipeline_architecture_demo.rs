//! Demonstration of the new pipeline architecture.
//!
//! This example shows the correct usage of IngestionPipeline vs IndexingPipeline,
//! matching LlamaIndex's architectural patterns.

use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber;

use cheungfun_core::{
    traits::{VectorStore, StorageContext},
    Document,
};
use cheungfun_indexing::{
    loaders::StringLoader,
    node_parser::text::SentenceSplitter,
    pipeline::{IngestionPipeline, DefaultIndexingPipeline, PipelineFactory},
    transformers::MetadataExtractor,
};
use cheungfun_integrations::{
    embedders::fastembed::FastEmbedEmbedder,
    storage::{
        docstore::KVDocumentStore,
        kvstore::InMemoryKVStore,
        vector_store::InMemoryVectorStore,
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("🏗️ Pipeline Architecture Demo");

    // Create sample documents
    let documents = vec![
        Document::new("This is the first document about machine learning and AI.", None),
        Document::new("The second document discusses natural language processing.", None),
        Document::new("This third document covers computer vision and deep learning.", None),
    ];

    // Demo 1: IngestionPipeline (Data Preprocessing Only)
    info!("\n📋 Demo 1: IngestionPipeline - Data Preprocessing");
    demo_ingestion_pipeline(documents.clone()).await?;

    // Demo 2: Complete IndexingPipeline (End-to-End)
    info!("\n📋 Demo 2: IndexingPipeline - Complete End-to-End");
    demo_indexing_pipeline(documents.clone()).await?;

    // Demo 3: Composed Workflow (IngestionPipeline + VectorStoreIndex)
    info!("\n📋 Demo 3: Composed Workflow - Matching LlamaIndex Pattern");
    demo_composed_workflow(documents.clone()).await?;

    info!("\n✅ All demos completed successfully!");
    Ok(())
}

/// Demo 1: IngestionPipeline for data preprocessing only
async fn demo_ingestion_pipeline(documents: Vec<Document>) -> Result<(), Box<dyn std::error::Error>> {
    info!("Creating IngestionPipeline for data preprocessing...");

    // Create ingestion pipeline using the factory
    let pipeline = PipelineFactory::ingestion()
        .with_name("preprocessing_pipeline".to_string())
        .with_transformations(vec![
            Arc::new(SentenceSplitter::from_defaults(100, 20)?),
            Arc::new(MetadataExtractor::new()),
        ])
        .build()?;

    // Process documents into nodes (no indexing)
    let processed_nodes = pipeline.run(Some(documents), None).await?;

    info!("✅ IngestionPipeline Results:");
    info!("  - Processed {} nodes", processed_nodes.len());
    info!("  - Nodes are ready for indexing");
    
    // Show sample node
    if let Some(first_node) = processed_nodes.first() {
        info!("  - Sample node content: '{}'", 
              first_node.content().chars().take(50).collect::<String>());
    }

    Ok(())
}

/// Demo 2: Complete IndexingPipeline for end-to-end processing
async fn demo_indexing_pipeline(documents: Vec<Document>) -> Result<(), Box<dyn std::error::Error>> {
    info!("Creating complete IndexingPipeline...");

    // Create storage components
    let kv_store = Arc::new(InMemoryKVStore::new());
    let doc_store = Arc::new(KVDocumentStore::new(kv_store.clone(), Some("documents".to_string())));
    let vector_store = Arc::new(InMemoryVectorStore::new());
    
    // Create storage context
    let storage_context = Arc::new(StorageContext::from_defaults(
        Some(doc_store),
        None,
        Some(vector_store.clone()),
        None,
        None,
    ));

    // Create embedder
    let embedder = Arc::new(FastEmbedEmbedder::try_new_default()?);

    // Create complete indexing pipeline using the factory
    let pipeline = PipelineFactory::indexing()
        .with_loader(Arc::new(StringLoader::from_documents(documents)))
        .with_transformer(Arc::new(SentenceSplitter::from_defaults(100, 20)?))
        .with_transformer(Arc::new(MetadataExtractor::new()))
        .with_embedder(embedder)
        .with_storage_context(storage_context)
        .with_deduplication()
        .build()?;

    // Run complete pipeline (documents -> nodes -> embeddings -> storage)
    let (processed_nodes, stats) = pipeline.run(None, None, true, true, None, true).await?;

    info!("✅ IndexingPipeline Results:");
    info!("  - Documents processed: {}", stats.documents_processed);
    info!("  - Nodes created: {}", stats.nodes_created);
    info!("  - Nodes stored: {}", stats.nodes_stored);
    info!("  - Processing time: {:?}", stats.processing_time);
    info!("  - Final nodes count: {}", processed_nodes.len());

    Ok(())
}

/// Demo 3: Composed workflow matching LlamaIndex pattern
async fn demo_composed_workflow(documents: Vec<Document>) -> Result<(), Box<dyn std::error::Error>> {
    info!("Creating composed workflow (IngestionPipeline + VectorStoreIndex)...");

    // Step 1: Use IngestionPipeline for preprocessing
    let ingestion_pipeline = PipelineFactory::ingestion()
        .with_name("preprocessing_step".to_string())
        .with_transformations(vec![
            Arc::new(SentenceSplitter::from_defaults(100, 20)?),
            Arc::new(MetadataExtractor::new()),
        ])
        .build()?;

    // Process documents into nodes
    let processed_nodes = ingestion_pipeline.run(Some(documents), None).await?;
    info!("📄 Preprocessing completed: {} nodes ready", processed_nodes.len());

    // Step 2: Create VectorStoreIndex for indexing (simulated)
    let vector_store = Arc::new(InMemoryVectorStore::new());
    let embedder = Arc::new(FastEmbedEmbedder::try_new_default()?);

    // Generate embeddings for nodes
    let texts: Vec<String> = processed_nodes.iter().map(|n| n.content().to_string()).collect();
    let embeddings = embedder.embed_texts(texts).await?;

    // Create nodes with embeddings
    let mut embedded_nodes = processed_nodes;
    for (node, embedding) in embedded_nodes.iter_mut().zip(embeddings.iter()) {
        node.embedding = Some(embedding.clone());
    }

    // Store in vector store
    let stored_ids = vector_store.add_nodes(&embedded_nodes).await?;

    info!("✅ Composed Workflow Results:");
    info!("  - Preprocessing: {} nodes created", embedded_nodes.len());
    info!("  - Indexing: {} nodes stored", stored_ids.len());
    info!("  - Vector store ready for querying");

    // This matches LlamaIndex pattern:
    // 1. IngestionPipeline.run(documents) -> nodes
    // 2. VectorStoreIndex.from_nodes(nodes) -> index
    info!("🎯 This workflow matches LlamaIndex's separation of concerns:");
    info!("   - IngestionPipeline: Data preprocessing");
    info!("   - VectorStoreIndex: Index construction and querying");

    Ok(())
}

/// Demonstrate pipeline factory usage
#[allow(dead_code)]
async fn demo_pipeline_factory() -> Result<(), Box<dyn std::error::Error>> {
    info!("Demonstrating PipelineFactory usage...");

    // Create different types of pipelines using the factory
    let _ingestion_pipeline = PipelineFactory::ingestion()
        .with_name("data_preprocessing".to_string())
        .with_transformations(vec![
            Arc::new(SentenceSplitter::from_defaults(512, 50)?),
        ])
        .build()?;

    let _indexing_pipeline = PipelineFactory::indexing()
        .with_loader(Arc::new(StringLoader::from_documents(vec![])))
        .with_transformer(Arc::new(SentenceSplitter::from_defaults(512, 50)?))
        .build()?;

    info!("✅ Pipeline factory can create different pipeline types");
    Ok(())
}

/// Show the architectural benefits
#[allow(dead_code)]
fn show_architectural_benefits() {
    info!("🏗️ Architectural Benefits:");
    info!("  1. Clear Separation of Concerns:");
    info!("     - IngestionPipeline: Data preprocessing only");
    info!("     - IndexingPipeline: Complete end-to-end solution");
    info!("  2. Composability:");
    info!("     - Mix and match components as needed");
    info!("     - Reuse preprocessing pipelines across different indices");
    info!("  3. LlamaIndex Compatibility:");
    info!("     - Same API patterns and concepts");
    info!("     - Easy migration from Python to Rust");
    info!("  4. Flexibility:");
    info!("     - Use IngestionPipeline for custom workflows");
    info!("     - Use IndexingPipeline for simple end-to-end cases");
}
