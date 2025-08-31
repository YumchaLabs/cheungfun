//! Example demonstrating document deduplication in IngestionPipeline.
//!
//! This example shows how to use the enhanced IngestionPipeline with document
//! deduplication capabilities, similar to LlamaIndex's document management features.

use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber;

use cheungfun_core::{
    deduplication::{DocumentDeduplicator, DocumentHasher, DocstoreStrategy},
    traits::{DocumentStore, IndexingPipeline, StorageContext},
    Document,
};
use cheungfun_indexing::{
    loaders::StringLoader,
    node_parser::text::SentenceSplitter,
    pipeline::{DefaultIndexingPipeline, PipelineConfig},
    transformers::MetadataExtractor,
};
use cheungfun_integrations::storage::{
    docstore::KVDocumentStore,
    kvstore::InMemoryKVStore,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("🚀 Starting IngestionPipeline Deduplication Demo");

    // Create storage components
    let kv_store = Arc::new(InMemoryKVStore::new());
    let doc_store = Arc::new(KVDocumentStore::new(kv_store.clone(), Some("documents".to_string())));
    
    // Create storage context
    let storage_context = Arc::new(StorageContext::from_defaults(
        Some(doc_store.clone()),
        None,
        None,
        None,
        None,
    ));

    // Demo 1: Basic deduplication
    info!("\n📋 Demo 1: Basic Document Deduplication");
    demo_basic_deduplication(storage_context.clone()).await?;

    // Demo 2: Different deduplication strategies
    info!("\n📋 Demo 2: Different Deduplication Strategies");
    demo_deduplication_strategies(storage_context.clone()).await?;

    // Demo 3: Document updates and upserts
    info!("\n📋 Demo 3: Document Updates and Upserts");
    demo_document_updates(storage_context.clone()).await?;

    info!("\n✅ All demos completed successfully!");
    Ok(())
}

async fn demo_basic_deduplication(
    storage_context: Arc<StorageContext>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create sample documents with some duplicates
    let documents = vec![
        Document::new("This is the first document about machine learning.", None),
        Document::new("This is the second document about artificial intelligence.", None),
        Document::new("This is the first document about machine learning.", None), // Duplicate
        Document::new("This is a third document about deep learning.", None),
    ];

    info!("Created {} documents (including 1 duplicate)", documents.len());

    // Create pipeline with deduplication enabled
    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(Arc::new(StringLoader::from_documents(documents)))
        .with_transformer(Arc::new(SentenceSplitter::from_defaults(100, 20)?))
        .with_transformer(Arc::new(MetadataExtractor::new()))
        .with_storage_context(storage_context.clone())
        .with_deduplication() // Enable deduplication with default settings
        .build()?;

    // Run the pipeline
    let stats = pipeline.run().await?;

    info!("Pipeline completed:");
    info!("  Documents processed: {}", stats.documents_processed);
    info!("  Nodes created: {}", stats.nodes_created);
    info!("  Processing time: {:?}", stats.processing_time);

    // Check document store
    let doc_count = storage_context.doc_store.count_documents().await?;
    info!("Documents in store: {}", doc_count);

    Ok(())
}

async fn demo_deduplication_strategies(
    storage_context: Arc<StorageContext>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Clear previous data
    storage_context.doc_store.clear().await?;

    let documents = vec![
        Document::new("Strategy demo document 1", None),
        Document::new("Strategy demo document 2", None),
    ];

    // Test different strategies
    let strategies = vec![
        (DocstoreStrategy::DuplicatesOnly, "DuplicatesOnly"),
        (DocstoreStrategy::Upserts, "Upserts"),
        (DocstoreStrategy::UpsertsAndDelete, "UpsertsAndDelete"),
    ];

    for (strategy, name) in strategies {
        info!("\nTesting strategy: {}", name);

        let pipeline = DefaultIndexingPipeline::builder()
            .with_loader(Arc::new(StringLoader::from_documents(documents.clone())))
            .with_transformer(Arc::new(SentenceSplitter::from_defaults(100, 20)?))
            .with_storage_context(storage_context.clone())
            .with_deduplication_strategy(strategy)
            .build()?;

        let stats = pipeline.run().await?;
        info!("  Processed {} documents", stats.documents_processed);

        // Clear for next test
        storage_context.doc_store.clear().await?;
    }

    Ok(())
}

async fn demo_document_updates(
    storage_context: Arc<StorageContext>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Clear previous data
    storage_context.doc_store.clear().await?;

    // First run: Add initial documents
    let initial_docs = vec![
        Document::new("Initial version of document 1", None),
        Document::new("Initial version of document 2", None),
    ];

    info!("First run: Adding initial documents");
    let pipeline1 = DefaultIndexingPipeline::builder()
        .with_loader(Arc::new(StringLoader::from_documents(initial_docs)))
        .with_transformer(Arc::new(SentenceSplitter::from_defaults(100, 20)?))
        .with_storage_context(storage_context.clone())
        .with_deduplication_strategy(DocstoreStrategy::Upserts)
        .build()?;

    let stats1 = pipeline1.run().await?;
    info!("  Processed {} documents", stats1.documents_processed);

    let doc_count_after_first = storage_context.doc_store.count_documents().await?;
    info!("  Documents in store: {}", doc_count_after_first);

    // Second run: Update one document, add one new
    let mut updated_doc = Document::new("Updated version of document 1", None);
    updated_doc.id = initial_docs[0].id; // Same ID as first document

    let updated_docs = vec![
        updated_doc,
        Document::new("Initial version of document 2", None), // Same content
        Document::new("This is a completely new document", None), // New document
    ];

    info!("\nSecond run: Updating documents");
    let pipeline2 = DefaultIndexingPipeline::builder()
        .with_loader(Arc::new(StringLoader::from_documents(updated_docs)))
        .with_transformer(Arc::new(SentenceSplitter::from_defaults(100, 20)?))
        .with_storage_context(storage_context.clone())
        .with_deduplication_strategy(DocstoreStrategy::Upserts)
        .build()?;

    let stats2 = pipeline2.run().await?;
    info!("  Processed {} documents", stats2.documents_processed);

    let doc_count_after_second = storage_context.doc_store.count_documents().await?;
    info!("  Documents in store: {}", doc_count_after_second);

    // Show document hashes
    let hashes = storage_context.doc_store.get_all_document_hashes().await?;
    info!("Document hashes:");
    for (doc_id, hash) in hashes {
        info!("  {}: {}", doc_id, &hash[..16]); // Show first 16 chars of hash
    }

    Ok(())
}

/// Demonstrate custom document hasher
#[allow(dead_code)]
async fn demo_custom_hasher() -> Result<(), Box<dyn std::error::Error>> {
    // Create custom hasher that excludes metadata
    let custom_hasher = DocumentHasher::with_options(false, true);

    let doc1 = Document::builder()
        .content("Same content")
        .metadata("source", "file1.txt")
        .build();

    let doc2 = Document::builder()
        .content("Same content")
        .metadata("source", "file2.txt") // Different metadata
        .build();

    let hash1 = custom_hasher.calculate_hash(&doc1);
    let hash2 = custom_hasher.calculate_hash(&doc2);

    info!("Custom hasher demo:");
    info!("  Document 1 hash: {}", &hash1[..16]);
    info!("  Document 2 hash: {}", &hash2[..16]);
    info!("  Hashes equal (ignoring metadata): {}", hash1 == hash2);

    Ok(())
}

/// Demonstrate deduplicator filtering
#[allow(dead_code)]
async fn demo_deduplicator_filtering() -> Result<(), Box<dyn std::error::Error>> {
    let deduplicator = DocumentDeduplicator::new();

    let documents = vec![
        Document::new("Document 1", None),
        Document::new("Document 2", None),
        Document::new("Document 3", None),
    ];

    // Simulate existing hashes (document 2 already exists)
    let mut existing_hashes = std::collections::HashMap::new();
    let doc2_hash = deduplicator.hasher().calculate_hash(&documents[1]);
    existing_hashes.insert(documents[1].id.to_string(), doc2_hash);

    let (to_process, to_skip, to_update) = deduplicator.filter_documents(documents, &existing_hashes);

    info!("Deduplicator filtering demo:");
    info!("  Documents to process: {}", to_process.len());
    info!("  Documents to skip: {}", to_skip.len());
    info!("  Documents to update: {}", to_update.len());

    Ok(())
}
