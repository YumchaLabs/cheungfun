//! End-to-end retrieval demonstration.
//!
//! This example demonstrates the retrieval part of the RAG pipeline
//! without requiring external API keys. It shows how to:
//! 1. Set up a knowledge base with documents
//! 2. Create embeddings and store them in a vector store
//! 3. Perform semantic search queries
//! 4. Display retrieved results

use cheungfun_core::{
    traits::{Embedder, Retriever, Transformer, VectorStore},
    types::{Document, Query, SearchMode},
    Result,
};
use cheungfun_indexing::prelude::{SplitterConfig, TextSplitter};
use cheungfun_integrations::{CandleEmbedder, InMemoryVectorStore};
use cheungfun_query::retriever::{VectorRetriever, VectorRetrieverConfig};
use std::sync::Arc;
use tracing::{info, Level};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("Starting end-to-end retrieval demonstration");

    // Step 1: Set up the knowledge base
    let (vector_store, embedder) = setup_knowledge_base().await?;

    // Step 2: Create retriever
    let retriever = create_retriever(vector_store.clone(), embedder.clone()).await?;

    // Step 3: Demonstrate different types of queries
    let queries = vec![
        "What is machine learning and how does it work?",
        "Explain the difference between supervised and unsupervised learning",
        "What are the main applications of vector databases?",
        "How does retrieval-augmented generation improve language models?",
        "What are the key components of a deep learning system?",
        "Tell me about neural networks and transformers",
    ];

    for query_text in queries {
        info!("Processing query: '{}'", query_text);

        // Step 4: Retrieve relevant context
        let query = Query::builder()
            .text(query_text)
            .top_k(3)
            .search_mode(SearchMode::Vector)
            .similarity_threshold(0.1)
            .build();

        let retrieved_nodes = retriever.retrieve(&query).await?;
        info!("Retrieved {} relevant documents", retrieved_nodes.len());

        // Step 5: Display results
        display_retrieval_results(query_text, &retrieved_nodes).await?;

        println!("\n{}", "=".repeat(80));
    }

    // Step 6: Show final statistics
    show_final_statistics(&vector_store, &embedder, &retriever).await?;

    info!("End-to-end retrieval demonstration completed successfully!");
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

    // Create sample documents
    let documents = create_comprehensive_documents();

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

/// Create comprehensive sample documents for the knowledge base.
fn create_comprehensive_documents() -> Vec<Document> {
    let mut documents = Vec::new();

    // Document 1: Machine Learning Fundamentals
    let mut doc1 = Document::new(
        "Machine learning is a subset of artificial intelligence that enables computers to learn \
         and make decisions from data without being explicitly programmed. There are three main \
         types of machine learning: supervised learning, where models learn from labeled training \
         data to make predictions on new data; unsupervised learning, where algorithms find hidden \
         patterns in unlabeled data; and reinforcement learning, where agents learn optimal actions \
         through trial and error in an environment. Common supervised learning algorithms include \
         linear regression for predicting continuous values, logistic regression for classification, \
         decision trees for interpretable models, and support vector machines for complex boundaries. \
         Unsupervised learning includes clustering algorithms like k-means and hierarchical clustering, \
         and dimensionality reduction techniques like PCA and t-SNE.",
    );
    doc1.metadata
        .insert("title".to_string(), "Machine Learning Fundamentals".into());
    doc1.metadata.insert("category".to_string(), "AI/ML".into());
    doc1.metadata
        .insert("difficulty".to_string(), "beginner".into());
    documents.push(doc1);

    // Document 2: Deep Learning and Neural Networks
    let mut doc2 = Document::new(
        "Deep learning is a specialized branch of machine learning that uses artificial neural \
         networks with multiple layers to model and understand complex patterns in data. Neural \
         networks are inspired by the human brain and consist of interconnected nodes (neurons) \
         organized in layers. The input layer receives data, hidden layers process information \
         through weighted connections and activation functions, and the output layer produces \
         results. Deep learning has revolutionized computer vision with Convolutional Neural \
         Networks (CNNs), which use filters to detect features like edges and textures in images. \
         For sequential data like text and speech, Recurrent Neural Networks (RNNs) and Long \
         Short-Term Memory (LSTM) networks maintain memory of previous inputs. The transformer \
         architecture, introduced in 'Attention is All You Need', uses self-attention mechanisms \
         to process sequences in parallel, leading to breakthrough models like BERT and GPT.",
    );
    doc2.metadata.insert(
        "title".to_string(),
        "Deep Learning and Neural Networks".into(),
    );
    doc2.metadata.insert("category".to_string(), "AI/ML".into());
    doc2.metadata
        .insert("difficulty".to_string(), "intermediate".into());
    documents.push(doc2);

    // Document 3: Vector Databases and Embeddings
    let mut doc3 = Document::new(
        "Vector databases are specialized databases designed to store, index, and query \
         high-dimensional vectors efficiently. They are essential for modern AI applications \
         that work with embeddings - dense numerical representations of data like text, images, \
         or audio. Vector embeddings capture semantic meaning, allowing similar items to have \
         similar vector representations. Vector databases use approximate nearest neighbor (ANN) \
         algorithms like HNSW (Hierarchical Navigable Small World) and IVF (Inverted File) to \
         perform fast similarity searches across millions of vectors. Distance metrics like \
         cosine similarity, Euclidean distance, and dot product determine how similarity is \
         calculated. Popular vector databases include Pinecone, Weaviate, Qdrant, Chroma, and \
         Milvus. These databases enable applications like semantic search, recommendation systems, \
         image similarity search, and retrieval-augmented generation (RAG) systems.",
    );
    doc3.metadata.insert(
        "title".to_string(),
        "Vector Databases and Embeddings".into(),
    );
    doc3.metadata
        .insert("category".to_string(), "Database".into());
    doc3.metadata
        .insert("difficulty".to_string(), "intermediate".into());
    documents.push(doc3);

    // Document 4: Retrieval-Augmented Generation (RAG)
    let mut doc4 = Document::new(
        "Retrieval-Augmented Generation (RAG) is a powerful technique that combines information \
         retrieval with text generation to create more accurate, informative, and up-to-date \
         responses from language models. RAG addresses key limitations of large language models \
         such as hallucination (generating false information), knowledge cutoffs, and inability \
         to access real-time information. The RAG process involves several steps: first, documents \
         are processed and converted into embeddings using models like BERT or sentence transformers; \
         these embeddings are stored in a vector database; when a user asks a question, the query \
         is also embedded and used to retrieve the most relevant documents; finally, the retrieved \
         context is provided to a language model along with the original question to generate \
         a grounded response. RAG systems are widely used in chatbots, question-answering systems, \
         customer support, and knowledge management applications.",
    );
    doc4.metadata
        .insert("title".to_string(), "Retrieval-Augmented Generation".into());
    doc4.metadata.insert("category".to_string(), "AI/ML".into());
    doc4.metadata
        .insert("difficulty".to_string(), "advanced".into());
    documents.push(doc4);

    // Document 5: Natural Language Processing
    let mut doc5 = Document::new(
        "Natural Language Processing (NLP) is the field of artificial intelligence that focuses \
         on enabling computers to understand, interpret, and generate human language. NLP combines \
         computational linguistics with machine learning and deep learning techniques. Key NLP \
         tasks include tokenization (breaking text into words or subwords), part-of-speech tagging, \
         named entity recognition (identifying people, places, organizations), sentiment analysis, \
         machine translation, text summarization, and question answering. Modern NLP heavily relies \
         on transformer-based models like BERT (Bidirectional Encoder Representations from Transformers) \
         for understanding tasks and GPT (Generative Pre-trained Transformer) for generation tasks. \
         These models are pre-trained on large text corpora and can be fine-tuned for specific \
         applications. Recent advances include large language models like GPT-3, GPT-4, and ChatGPT \
         that demonstrate remarkable capabilities in text generation, reasoning, and conversation.",
    );
    doc5.metadata
        .insert("title".to_string(), "Natural Language Processing".into());
    doc5.metadata.insert("category".to_string(), "AI/ML".into());
    doc5.metadata
        .insert("difficulty".to_string(), "intermediate".into());
    documents.push(doc5);

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

/// Display the results of a retrieval query.
async fn display_retrieval_results(
    query: &str,
    retrieved_nodes: &[cheungfun_core::types::ScoredNode],
) -> Result<()> {
    println!("\n🔍 Query: {}", query);
    println!("\n📚 Retrieved Documents:");

    if retrieved_nodes.is_empty() {
        println!("  No relevant documents found.");
        return Ok(());
    }

    for (i, node) in retrieved_nodes.iter().enumerate() {
        let title = node
            .node
            .metadata
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown Document");
        let category = node
            .node
            .metadata
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        let difficulty = node
            .node
            .metadata
            .get("difficulty")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");

        println!("  {}. {} (Score: {:.4})", i + 1, title, node.score);
        println!("     Category: {} | Difficulty: {}", category, difficulty);
        println!(
            "     Content: {}",
            node.node.content.chars().take(150).collect::<String>() + "..."
        );
        println!();
    }

    // Show a simple mock response based on the retrieved content
    println!("🤖 Mock Response:");
    println!("Based on the retrieved documents, I can provide information about your query.");
    println!(
        "The most relevant document is '{}' with a similarity score of {:.4}.",
        retrieved_nodes[0]
            .node
            .metadata
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown"),
        retrieved_nodes[0].score
    );
    println!(
        "This information comes from {} category content.",
        retrieved_nodes[0]
            .node
            .metadata
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
    );

    Ok(())
}

/// Show final statistics.
async fn show_final_statistics(
    vector_store: &Arc<dyn VectorStore>,
    embedder: &Arc<dyn Embedder>,
    retriever: &Arc<dyn Retriever>,
) -> Result<()> {
    info!("=== Final Statistics ===");

    // Vector store stats
    let node_count = vector_store.count().await?;
    let vs_metadata = vector_store.metadata().await?;
    info!("Vector Store:");
    info!("  - Total nodes indexed: {}", node_count);
    info!(
        "  - Store type: {}",
        vs_metadata.get("type").unwrap_or(&"unknown".into())
    );
    info!(
        "  - Vector dimension: {}",
        vs_metadata.get("dimension").unwrap_or(&"unknown".into())
    );

    // Embedder stats
    let embedder_metadata = embedder.metadata();
    info!("Embedder:");
    info!("  - Model: {}", embedder.model_name());
    info!("  - Dimension: {}", embedder.dimension());
    info!(
        "  - Texts embedded: {}",
        embedder_metadata.get("texts_embedded").unwrap_or(&0.into())
    );

    // Retriever stats
    let retriever_config = retriever.config();
    info!("Retriever:");
    info!(
        "  - Max top-k: {}",
        retriever_config.get("max_top_k").unwrap_or(&0.into())
    );
    info!(
        "  - Default top-k: {}",
        retriever_config.get("default_top_k").unwrap_or(&0.into())
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

    Ok(())
}
