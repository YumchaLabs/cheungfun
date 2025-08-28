//! End-to-end query example.
//!
//! This example demonstrates how to build a complete query pipeline
//! from user questions to generated responses using Cheungfun components.

use cheungfun_core::{
    traits::{Embedder, ResponseGenerator, Retriever, Transformer, VectorStore},
    types::{Document, Query, SearchMode},
    GenerationOptions, Result,
};
use cheungfun_indexing::prelude::{SplitterConfig, TextSplitter};
use cheungfun_integrations::{CandleEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    generator::{SiumaiGenerator, SiumaiGeneratorConfig},
    retriever::{VectorRetriever, VectorRetrieverConfig},
};
use siumai::prelude::*;
use std::sync::Arc;
use tracing::{info, Level};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("Starting end-to-end query example");

    // Step 1: Set up the knowledge base (same as indexing example)
    let (vector_store, embedder) = setup_knowledge_base().await?;

    // Step 2: Create retriever
    let retriever = create_retriever(vector_store.clone(), embedder.clone()).await?;

    // Step 3: Create generator
    let generator = create_generator().await?;

    // Step 4: Demonstrate different types of queries
    let queries = vec![
        "What is machine learning and how does it work?",
        "Explain the difference between supervised and unsupervised learning",
        "What are the main applications of vector databases?",
        "How does retrieval-augmented generation improve language models?",
        "What are the key components of a deep learning system?",
    ];

    for query_text in queries {
        info!("Processing query: '{}'", query_text);

        // Step 5: Retrieve relevant context
        let query = Query::builder()
            .text(query_text)
            .top_k(5)
            .search_mode(SearchMode::Vector)
            .similarity_threshold(0.1)
            .build();

        let retrieved_nodes = retriever.retrieve(&query).await?;
        info!("Retrieved {} relevant documents", retrieved_nodes.len());

        // Step 6: Generate response using retrieved context
        let response = generate_response(&generator, query_text, &retrieved_nodes).await?;

        // Step 7: Display results
        display_results(query_text, &retrieved_nodes, &response).await?;

        println!("\n{}", "=".repeat(80));
    }

    // Step 8: Show final statistics
    show_final_statistics(&vector_store, &embedder, &retriever, &generator).await?;

    info!("End-to-end query example completed successfully!");
    Ok(())
}

/// Set up the knowledge base with sample documents.
async fn setup_knowledge_base() -> Result<(Arc<dyn VectorStore>, Arc<dyn Embedder>)> {
    info!("Setting up knowledge base...");

    // Create embedder
    let embedder =
        Arc::new(CandleEmbedder::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").await?);

    // Create vector store
    use cheungfun_core::traits::DistanceMetric;
    let vector_store = Arc::new(InMemoryVectorStore::new(
        embedder.dimension(),
        DistanceMetric::Cosine,
    ));

    // Create sample documents (same as indexing example)
    let documents = create_sample_documents();

    // Create text splitter
    let config = SplitterConfig {
        chunk_size: 300,
        chunk_overlap: 50,
        ..Default::default()
    };
    let splitter = TextSplitter::with_config(config);

    // Process documents
    let mut all_nodes = Vec::new();
    for document in documents {
        let nodes = splitter.transform(document).await?;
        for mut node in nodes {
            let embedding = embedder.embed(&node.content).await?;
            node.embedding = Some(embedding);
            all_nodes.push(node);
        }
    }

    // Store in vector store
    vector_store.add(all_nodes).await?;
    info!(
        "Knowledge base setup complete with {} nodes",
        vector_store.count().await?
    );

    Ok((vector_store, embedder))
}

/// Create sample documents for the knowledge base.
fn create_sample_documents() -> Vec<Document> {
    let mut documents = Vec::new();

    // Document 1: Machine Learning Overview
    let mut doc1 = Document::new(
        "Machine learning is a subset of artificial intelligence that focuses on algorithms \
         that can learn and make decisions from data. It includes supervised learning, where \
         models learn from labeled examples, unsupervised learning, where patterns are found \
         in unlabeled data, and reinforcement learning, where agents learn through interaction \
         with an environment. Popular algorithms include linear regression, decision trees, \
         neural networks, and support vector machines. Machine learning is used in applications \
         like image recognition, natural language processing, recommendation systems, and \
         autonomous vehicles.",
    );
    doc1.metadata
        .insert("title".to_string(), "Machine Learning Overview".into());
    doc1.metadata.insert("category".to_string(), "AI/ML".into());
    documents.push(doc1);

    // Document 2: Deep Learning
    let mut doc2 = Document::new(
        "Deep learning is a specialized branch of machine learning that uses neural networks \
         with multiple layers to model complex patterns in data. It has revolutionized fields \
         like computer vision, natural language processing, and speech recognition. Key \
         architectures include convolutional neural networks (CNNs) for image processing, \
         recurrent neural networks (RNNs) for sequential data, and transformers for language \
         understanding. Deep learning requires large datasets and significant computational \
         resources, often using GPUs for training. Applications include image classification, \
         machine translation, and generative AI.",
    );
    doc2.metadata
        .insert("title".to_string(), "Deep Learning".into());
    doc2.metadata.insert("category".to_string(), "AI/ML".into());
    documents.push(doc2);

    // Document 3: Vector Databases
    let mut doc3 = Document::new(
        "Vector databases are specialized databases designed to store and query high-dimensional \
         vectors efficiently. They are essential for applications like semantic search, \
         recommendation systems, and retrieval-augmented generation (RAG). Vector databases \
         use techniques like approximate nearest neighbor (ANN) search, indexing methods \
         such as HNSW and IVF, and distance metrics like cosine similarity and Euclidean \
         distance. Popular vector databases include Pinecone, Weaviate, Qdrant, and Chroma. \
         They enable fast similarity search across millions of vectors, making them crucial \
         for modern AI applications.",
    );
    doc3.metadata
        .insert("title".to_string(), "Vector Databases".into());
    doc3.metadata
        .insert("category".to_string(), "Database".into());
    documents.push(doc3);

    // Document 4: RAG Systems
    let mut doc4 = Document::new(
        "Retrieval-Augmented Generation (RAG) is a technique that combines information \
         retrieval with text generation to create more accurate and informative responses. \
         It works by first retrieving relevant documents from a knowledge base, then using \
         this context to generate responses with a language model. RAG helps address \
         limitations of large language models such as hallucination and outdated knowledge. \
         The process involves embedding documents, storing them in a vector database, \
         retrieving relevant context based on queries, and generating responses conditioned \
         on the retrieved information. RAG systems are widely used in chatbots, question \
         answering, and knowledge management applications.",
    );
    doc4.metadata
        .insert("title".to_string(), "RAG Systems".into());
    doc4.metadata.insert("category".to_string(), "AI/ML".into());
    documents.push(doc4);

    documents
}

/// Create and configure the retriever.
async fn create_retriever(
    vector_store: Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedder>,
) -> Result<Arc<dyn Retriever>> {
    info!("Creating vector retriever...");

    let config = VectorRetrieverConfig {
        default_top_k: 5,
        max_top_k: 20,
        default_similarity_threshold: Some(0.1),
        enable_query_expansion: false,
        enable_reranking: false,
        timeout_seconds: Some(30),
    };

    let retriever = VectorRetriever::with_config(vector_store, embedder, config);
    Ok(Arc::new(retriever))
}

/// Create and configure the generator.
async fn create_generator() -> Result<Arc<dyn ResponseGenerator>> {
    info!("Creating Siumai generator...");

    let config = SiumaiGeneratorConfig {
        default_model: Some("gpt-3.5-turbo".to_string()),
        default_temperature: 0.7,
        default_max_tokens: 500,
        default_system_prompt:
            "You are a helpful AI assistant that provides accurate and informative answers \
             based on the given context. Use the provided context to answer questions, but \
             also draw on your general knowledge when appropriate. Be concise but thorough."
                .to_string(),
        ..Default::default()
    };

    // Note: In a real application, you would need to provide actual API credentials
    // For this example, we'll create a mock client and generator
    let client = Siumai::builder().openai().build().await.map_err(|e| {
        cheungfun_core::CheungfunError::Configuration {
            message: format!("Failed to create Siumai client: {}", e),
        }
    })?;

    let generator = SiumaiGenerator::with_config(client, config);
    Ok(Arc::new(generator))
}

/// Generate a response using the retrieved context.
async fn generate_response(
    generator: &Arc<dyn ResponseGenerator>,
    query: &str,
    retrieved_nodes: &[cheungfun_core::types::ScoredNode],
) -> Result<String> {
    // Note: The SiumaiGenerator will automatically build the prompt from the query and context nodes

    // Generate response
    let generation_options = GenerationOptions {
        max_tokens: Some(300),
        temperature: Some(0.7),
        ..Default::default()
    };

    let response = generator
        .generate_response(query, retrieved_nodes.to_vec(), &generation_options)
        .await?;
    Ok(response.content.clone())
}

/// Display the results of a query.
async fn display_results(
    query: &str,
    retrieved_nodes: &[cheungfun_core::types::ScoredNode],
    response: &str,
) -> Result<()> {
    println!("\n🔍 Query: {}", query);
    println!("\n📚 Retrieved Context:");

    for (i, node) in retrieved_nodes.iter().enumerate() {
        let title = node
            .node
            .metadata
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        println!("  {}. {} (Score: {:.3})", i + 1, title, node.score);
        println!(
            "     {}",
            node.node.content.chars().take(100).collect::<String>() + "..."
        );
    }

    println!("\n🤖 Generated Response:");
    println!("{}", response);

    Ok(())
}

/// Show final statistics.
async fn show_final_statistics(
    vector_store: &Arc<dyn VectorStore>,
    embedder: &Arc<dyn Embedder>,
    retriever: &Arc<dyn Retriever>,
    generator: &Arc<dyn ResponseGenerator>,
) -> Result<()> {
    info!("=== Final Statistics ===");

    // Vector store stats
    let node_count = vector_store.count().await?;
    info!("Vector Store: {} nodes indexed", node_count);

    // Embedder stats
    let embedder_metadata = embedder.metadata();
    info!(
        "Embedder: {} texts embedded",
        embedder_metadata.get("texts_embedded").unwrap_or(&0.into())
    );

    // Retriever stats
    let retriever_config = retriever.config();
    info!(
        "Retriever: max_top_k = {}",
        retriever_config.get("max_top_k").unwrap_or(&0.into())
    );

    // Generator stats
    let generator_config = generator.config();
    info!(
        "Generator: {} (config keys: {})",
        generator.name(),
        generator_config.len()
    );

    // Health checks
    info!("Health Checks:");
    match vector_store.health_check().await {
        Ok(()) => info!("  ✓ Vector store healthy"),
        Err(e) => info!("  ✗ Vector store error: {}", e),
    }

    match retriever.health_check().await {
        Ok(()) => info!("  ✓ Retriever healthy"),
        Err(e) => info!("  ✗ Retriever error: {}", e),
    }

    match generator.health_check().await {
        Ok(()) => info!("  ✓ Generator healthy"),
        Err(e) => info!("  ✗ Generator error: {}", e),
    }

    Ok(())
}
