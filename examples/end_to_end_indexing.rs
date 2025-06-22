//! End-to-end indexing example.
//!
//! This example demonstrates how to build a complete indexing pipeline
//! from documents to vector storage using Cheungfun components.

use cheungfun_core::{
    Result,
    traits::{Embedder, Transformer, VectorStore},
    types::Document,
};
use cheungfun_indexing::prelude::{SplitterConfig, TextSplitter};
use cheungfun_integrations::{CandleEmbedder, InMemoryVectorStore};
use std::sync::Arc;
use tracing::{Level, info};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("Starting end-to-end indexing example");

    // Step 1: Create sample documents
    let documents = create_sample_documents().await?;
    info!("Created {} sample documents", documents.len());

    // Step 2: Initialize components
    let embedder = create_embedder().await?;
    let splitter = create_splitter();
    let vector_store = create_vector_store(&embedder).await?;

    // Step 3: Process documents through the pipeline
    let mut all_nodes = Vec::new();

    for document in documents {
        info!(
            "Processing document: {}",
            document.metadata.get("title").unwrap_or(&"Unknown".into())
        );

        // Split document into chunks
        let nodes = splitter.transform(document).await?;
        info!("Split document into {} chunks", nodes.len());

        // Generate embeddings for each chunk
        let mut embedded_nodes = Vec::new();
        for mut node in nodes {
            let embedding = embedder.embed(&node.content).await?;
            node.embedding = Some(embedding);
            embedded_nodes.push(node);
        }

        all_nodes.extend(embedded_nodes);
    }

    info!("Generated embeddings for {} total nodes", all_nodes.len());

    // Step 4: Store nodes in vector store
    let node_ids = vector_store.add(all_nodes.clone()).await?;
    info!("Stored {} nodes in vector store", node_ids.len());

    // Step 5: Verify storage and demonstrate search
    let count = vector_store.count().await?;
    info!("Vector store now contains {} nodes", count);

    // Demonstrate search functionality
    demonstrate_search(&vector_store, &embedder).await?;

    // Step 6: Show statistics
    show_statistics(&vector_store, &embedder).await?;

    info!("End-to-end indexing example completed successfully!");
    Ok(())
}

/// Create sample documents for demonstration.
async fn create_sample_documents() -> Result<Vec<Document>> {
    let mut documents = Vec::new();

    // Document 1: Machine Learning Overview
    let mut doc1 = Document::new(
        "Machine learning is a subset of artificial intelligence that focuses on algorithms \
         that can learn and make decisions from data. It includes supervised learning, where \
         models learn from labeled examples, unsupervised learning, where patterns are found \
         in unlabeled data, and reinforcement learning, where agents learn through interaction \
         with an environment. Popular algorithms include linear regression, decision trees, \
         neural networks, and support vector machines.",
    );
    doc1.metadata
        .insert("title".to_string(), "Machine Learning Overview".into());
    doc1.metadata.insert("category".to_string(), "AI/ML".into());
    doc1.metadata
        .insert("difficulty".to_string(), "beginner".into());
    documents.push(doc1);

    // Document 2: Deep Learning Fundamentals
    let mut doc2 = Document::new(
        "Deep learning is a specialized branch of machine learning that uses neural networks \
         with multiple layers to model and understand complex patterns in data. It has \
         revolutionized fields like computer vision, natural language processing, and speech \
         recognition. Key architectures include convolutional neural networks (CNNs) for \
         image processing, recurrent neural networks (RNNs) for sequential data, and \
         transformers for language understanding. Training deep networks requires large \
         datasets and significant computational resources.",
    );
    doc2.metadata
        .insert("title".to_string(), "Deep Learning Fundamentals".into());
    doc2.metadata.insert("category".to_string(), "AI/ML".into());
    doc2.metadata
        .insert("difficulty".to_string(), "intermediate".into());
    documents.push(doc2);

    // Document 3: Natural Language Processing
    let mut doc3 = Document::new(
        "Natural Language Processing (NLP) is the field of AI that focuses on enabling \
         computers to understand, interpret, and generate human language. It combines \
         computational linguistics with machine learning and deep learning. Common NLP \
         tasks include text classification, named entity recognition, sentiment analysis, \
         machine translation, and question answering. Modern NLP heavily relies on \
         transformer architectures like BERT, GPT, and T5, which have achieved \
         state-of-the-art results across many language understanding tasks.",
    );
    doc3.metadata
        .insert("title".to_string(), "Natural Language Processing".into());
    doc3.metadata.insert("category".to_string(), "AI/ML".into());
    doc3.metadata
        .insert("difficulty".to_string(), "intermediate".into());
    documents.push(doc3);

    // Document 4: Vector Databases
    let mut doc4 = Document::new(
        "Vector databases are specialized databases designed to store and query high-dimensional \
         vectors efficiently. They are essential for applications like semantic search, \
         recommendation systems, and retrieval-augmented generation (RAG). Vector databases \
         use techniques like approximate nearest neighbor (ANN) search, indexing methods \
         such as HNSW and IVF, and distance metrics like cosine similarity and Euclidean \
         distance. Popular vector databases include Pinecone, Weaviate, Qdrant, and Chroma.",
    );
    doc4.metadata
        .insert("title".to_string(), "Vector Databases".into());
    doc4.metadata
        .insert("category".to_string(), "Database".into());
    doc4.metadata
        .insert("difficulty".to_string(), "advanced".into());
    documents.push(doc4);

    // Document 5: Retrieval-Augmented Generation
    let mut doc5 = Document::new(
        "Retrieval-Augmented Generation (RAG) is a technique that combines information \
         retrieval with text generation to create more accurate and informative responses. \
         It works by first retrieving relevant documents or passages from a knowledge base, \
         then using this context to generate responses with a language model. RAG helps \
         address limitations of large language models such as hallucination and outdated \
         knowledge. The process typically involves embedding documents, storing them in \
         a vector database, retrieving relevant context based on queries, and generating \
         responses conditioned on the retrieved information.",
    );
    doc5.metadata
        .insert("title".to_string(), "Retrieval-Augmented Generation".into());
    doc5.metadata.insert("category".to_string(), "AI/ML".into());
    doc5.metadata
        .insert("difficulty".to_string(), "advanced".into());
    documents.push(doc5);

    Ok(documents)
}

/// Create and configure the embedder.
async fn create_embedder() -> Result<Arc<dyn Embedder>> {
    info!("Initializing CandleEmbedder...");
    let embedder =
        CandleEmbedder::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").await?;
    info!(
        "CandleEmbedder initialized with dimension: {}",
        embedder.dimension()
    );
    Ok(Arc::new(embedder))
}

/// Create and configure the text splitter.
fn create_splitter() -> Arc<dyn Transformer> {
    info!("Initializing TextSplitter...");
    let config = SplitterConfig {
        chunk_size: 200,
        chunk_overlap: 50,
        separators: vec!["\n\n".to_string(), "\n".to_string(), ". ".to_string()],
        keep_separators: true,
        ..Default::default()
    };
    let splitter = TextSplitter::with_config(config);
    Arc::new(splitter)
}

/// Create and configure the vector store.
async fn create_vector_store(embedder: &Arc<dyn Embedder>) -> Result<Arc<dyn VectorStore>> {
    info!("Initializing InMemoryVectorStore...");
    use cheungfun_core::traits::DistanceMetric;
    let vector_store = InMemoryVectorStore::new(embedder.dimension(), DistanceMetric::Cosine);
    Ok(Arc::new(vector_store))
}

/// Demonstrate search functionality.
async fn demonstrate_search(
    vector_store: &Arc<dyn VectorStore>,
    embedder: &Arc<dyn Embedder>,
) -> Result<()> {
    info!("Demonstrating search functionality...");

    let queries = vec![
        "What is machine learning?",
        "How do neural networks work?",
        "What are vector databases used for?",
        "Explain retrieval-augmented generation",
    ];

    for query_text in queries {
        info!("Searching for: '{}'", query_text);

        // Generate query embedding
        let query_embedding = embedder.embed(query_text).await?;

        // Create query
        let query = cheungfun_core::types::Query::builder()
            .text(query_text)
            .embedding(query_embedding)
            .top_k(3)
            .build();

        // Perform search
        let results = vector_store.search(&query).await?;

        info!("Found {} results:", results.len());
        for (i, result) in results.iter().enumerate() {
            let title = result
                .node
                .metadata
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown");
            info!("  {}. {} (score: {:.4})", i + 1, title, result.score);
            info!(
                "     Content preview: {}...",
                result.node.content.chars().take(100).collect::<String>()
            );
        }
        info!("");
    }

    Ok(())
}

/// Show statistics about the indexing process.
async fn show_statistics(
    vector_store: &Arc<dyn VectorStore>,
    embedder: &Arc<dyn Embedder>,
) -> Result<()> {
    info!("=== Indexing Statistics ===");

    // Vector store statistics
    let count = vector_store.count().await?;
    let metadata = vector_store.metadata().await?;
    info!("Vector Store:");
    info!("  - Total nodes: {}", count);
    info!(
        "  - Type: {}",
        metadata.get("type").unwrap_or(&"unknown".into())
    );
    info!(
        "  - Dimension: {}",
        metadata.get("dimension").unwrap_or(&"unknown".into())
    );

    // Embedder statistics
    let embedder_metadata = embedder.metadata();
    info!("Embedder:");
    info!("  - Model: {}", embedder.model_name());
    info!("  - Dimension: {}", embedder.dimension());
    info!(
        "  - Texts embedded: {}",
        embedder_metadata.get("texts_embedded").unwrap_or(&0.into())
    );

    // Health checks
    info!("Health Checks:");
    match vector_store.health_check().await {
        Ok(()) => info!("  - Vector store: ✓ Healthy"),
        Err(e) => info!("  - Vector store: ✗ Error: {}", e),
    }

    match embedder.health_check().await {
        Ok(()) => info!("  - Embedder: ✓ Healthy"),
        Err(e) => info!("  - Embedder: ✗ Error: {}", e),
    }

    Ok(())
}
