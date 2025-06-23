//! Complete RAG (Retrieval-Augmented Generation) Demo
//!
//! This example demonstrates a complete end-to-end RAG workflow using Cheungfun:
//! 1. Document Loading - Load documents from various sources
//! 2. Text Processing - Split documents into chunks and extract metadata
//! 3. Embedding Generation - Generate vector embeddings for text chunks
//! 4. Vector Storage - Store embeddings in a vector database
//! 5. Query Processing - Process user queries and retrieve relevant context
//! 6. Response Generation - Generate responses using LLM with retrieved context
//!
//! This demo showcases the complete integration of all Cheungfun components.
//!
//! ## Required Features
//!
//! This example requires the `production-examples` feature bundle, which includes:
//! - `candle` - Candle ML framework for embeddings
//! - `performance` - CPU performance optimizations (SIMD, optimized memory, HNSW)
//! - `qdrant` - Qdrant vector database integration
//!
//! ## Usage
//!
//! ```bash
//! # Run with production features
//! cargo run --bin complete_rag_system --features production-examples
//!
//! # Or run with specific features
//! cargo run --bin complete_rag_system --features "candle,performance"
//!
//! # For GPU acceleration (if available)
//! cargo run --bin complete_rag_system --features "production-examples,candle-cuda"
//! ```
//!
//! ## Environment Setup
//!
//! Set up your LLM API key (optional - will fall back to Ollama):
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! ```
//!
//! Or ensure Ollama is running for local LLM:
//! ```bash
//! ollama serve
//! ollama pull llama2
//! ```

use cheungfun_core::{
    Result,
    traits::{Embedder, Loader, NodeTransformer, QueryPipeline, Transformer, VectorStore},
};
use cheungfun_indexing::{
    loaders::DirectoryLoader,
    prelude::SplitterConfig,
    transformers::{MetadataExtractor, TextSplitter},
};
use cheungfun_integrations::{CandleEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    generator::SiumaiGenerator, pipeline::DefaultQueryPipeline, retriever::VectorRetriever,
};
use siumai::prelude::*;
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::fs;
use tracing::{Level, info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    info!("🚀 Starting Complete RAG Demo");
    info!("===============================");

    // Step 1: Setup demo environment
    let demo_env = setup_demo_environment().await?;
    info!("✅ Demo environment setup complete");

    // Step 2: Initialize components
    let components = initialize_components().await?;
    info!("✅ All components initialized");

    // Step 3: Build and run indexing pipeline
    let indexing_stats = run_indexing_pipeline(&demo_env, &components).await?;
    info!("✅ Indexing pipeline completed");
    print_indexing_stats(&indexing_stats);

    // Step 4: Build query pipeline
    let query_pipeline = build_query_pipeline(&components).await?;
    info!("✅ Query pipeline built");

    // Step 5: Demonstrate interactive querying
    demonstrate_querying(&query_pipeline).await?;
    info!("✅ Query demonstration completed");

    // Step 6: Show system statistics
    show_system_statistics(&components).await?;

    info!("🎉 Complete RAG Demo finished successfully!");
    info!("===============================");

    Ok(())
}

/// Demo environment containing sample documents and configuration
struct DemoEnvironment {
    temp_dir: TempDir,
    documents_path: std::path::PathBuf,
}

/// All initialized components needed for the RAG system
struct RagComponents {
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    text_splitter: Arc<dyn Transformer>,
    metadata_extractor: Arc<MetadataExtractor>,
}

/// Statistics from the indexing process
#[derive(Debug)]
struct IndexingStats {
    documents_loaded: usize,
    chunks_created: usize,
    embeddings_generated: usize,
    nodes_stored: usize,
    processing_time: std::time::Duration,
}

/// Setup demo environment with sample documents
async fn setup_demo_environment() -> Result<DemoEnvironment> {
    info!("📁 Setting up demo environment...");

    let temp_dir = tempfile::tempdir().map_err(|e| cheungfun_core::CheungfunError::Io(e))?;

    let documents_path = temp_dir.path().join("documents");
    fs::create_dir_all(&documents_path)
        .await
        .map_err(|e| cheungfun_core::CheungfunError::Io(e))?;

    // Create sample documents
    create_sample_documents(&documents_path).await?;

    info!(
        "📄 Created sample documents in: {}",
        documents_path.display()
    );

    Ok(DemoEnvironment {
        temp_dir,
        documents_path,
    })
}

/// Create sample documents for the demo
async fn create_sample_documents(docs_path: &Path) -> Result<()> {
    let documents = vec![
        (
            "machine_learning_basics.txt",
            "Machine Learning Fundamentals\n\n\
            Machine learning is a subset of artificial intelligence (AI) that focuses on \
            algorithms that can learn and make decisions from data without being explicitly \
            programmed for every scenario. It enables computers to automatically improve \
            their performance on a specific task through experience.\n\n\
            There are three main types of machine learning:\n\n\
            1. Supervised Learning: Uses labeled training data to learn a mapping from \
            inputs to outputs. Examples include classification and regression tasks.\n\n\
            2. Unsupervised Learning: Finds hidden patterns in data without labeled examples. \
            Common techniques include clustering and dimensionality reduction.\n\n\
            3. Reinforcement Learning: Learns through interaction with an environment, \
            receiving rewards or penalties for actions taken.\n\n\
            Popular algorithms include linear regression, decision trees, neural networks, \
            support vector machines, and ensemble methods like random forests.",
        ),
        (
            "deep_learning_guide.txt",
            "Deep Learning: A Comprehensive Guide\n\n\
            Deep learning is a specialized branch of machine learning that uses artificial \
            neural networks with multiple layers (hence 'deep') to model and understand \
            complex patterns in data. It has revolutionized many fields including computer \
            vision, natural language processing, and speech recognition.\n\n\
            Key Deep Learning Architectures:\n\n\
            1. Convolutional Neural Networks (CNNs): Primarily used for image processing \
            and computer vision tasks. They use convolutional layers to detect features \
            like edges, shapes, and textures.\n\n\
            2. Recurrent Neural Networks (RNNs): Designed for sequential data like text \
            and time series. Variants include LSTM and GRU networks.\n\n\
            3. Transformers: The foundation of modern NLP, using attention mechanisms \
            to process sequences. Examples include BERT, GPT, and T5.\n\n\
            Training deep networks requires large datasets, significant computational \
            resources (often GPUs), and careful hyperparameter tuning.",
        ),
        (
            "vector_databases.txt",
            "Vector Databases: The Foundation of Modern AI Applications\n\n\
            Vector databases are specialized databases designed to store, index, and query \
            high-dimensional vectors efficiently. They have become essential infrastructure \
            for AI applications, particularly those involving semantic search, recommendation \
            systems, and retrieval-augmented generation (RAG).\n\n\
            Key Features of Vector Databases:\n\n\
            1. High-dimensional vector storage: Can handle vectors with hundreds or thousands \
            of dimensions representing embeddings from machine learning models.\n\n\
            2. Similarity search: Use distance metrics like cosine similarity, Euclidean \
            distance, or dot product to find similar vectors quickly.\n\n\
            3. Approximate Nearest Neighbor (ANN): Employ algorithms like HNSW, IVF, or \
            LSH to perform fast approximate searches on large datasets.\n\n\
            4. Scalability: Designed to handle millions or billions of vectors with \
            horizontal scaling capabilities.\n\n\
            Popular vector databases include Pinecone, Weaviate, Qdrant, Chroma, and Milvus. \
            Each offers different trade-offs in terms of performance, scalability, and features.",
        ),
        (
            "rag_systems.txt",
            "Retrieval-Augmented Generation: Enhancing AI with External Knowledge\n\n\
            Retrieval-Augmented Generation (RAG) is a powerful technique that combines \
            information retrieval with text generation to create more accurate, informative, \
            and up-to-date AI responses. It addresses key limitations of large language \
            models such as hallucination, outdated knowledge, and lack of domain-specific \
            information.\n\n\
            How RAG Works:\n\n\
            1. Knowledge Base Creation: Documents are processed, chunked, and converted \
            into vector embeddings, then stored in a vector database.\n\n\
            2. Query Processing: User queries are converted into embeddings using the \
            same model used for the knowledge base.\n\n\
            3. Retrieval: The system searches the vector database to find the most \
            relevant document chunks based on semantic similarity.\n\n\
            4. Context Assembly: Retrieved chunks are combined and formatted as context \
            for the language model.\n\n\
            5. Generation: The language model generates a response using both the original \
            query and the retrieved context.\n\n\
            RAG systems are widely used in chatbots, question-answering systems, \
            documentation assistants, and knowledge management platforms.",
        ),
    ];

    for (filename, content) in documents {
        let file_path = docs_path.join(filename);
        fs::write(&file_path, content)
            .await
            .map_err(|e| cheungfun_core::CheungfunError::Io(e))?;
    }

    Ok(())
}

/// Initialize all RAG components
async fn initialize_components() -> Result<RagComponents> {
    info!("🔧 Initializing RAG components...");

    // Initialize embedder
    info!("  📊 Initializing embedder...");
    let embedder =
        Arc::new(CandleEmbedder::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").await?);
    info!(
        "    ✅ Embedder ready (dimension: {})",
        embedder.dimension()
    );

    // Initialize vector store
    info!("  🗄️  Initializing vector store...");
    let vector_store = Arc::new(InMemoryVectorStore::new(
        embedder.dimension(),
        cheungfun_core::traits::DistanceMetric::Cosine,
    ));
    info!("    ✅ Vector store ready");

    // Initialize text splitter
    info!("  ✂️  Initializing text splitter...");
    let splitter_config = SplitterConfig {
        chunk_size: 300,
        chunk_overlap: 50,
        separators: vec!["\n\n".to_string(), "\n".to_string(), ". ".to_string()],
        keep_separators: true,
        ..Default::default()
    };
    let text_splitter = Arc::new(TextSplitter::with_config(splitter_config));
    info!("    ✅ Text splitter ready");

    // Initialize metadata extractor
    info!("  🏷️  Initializing metadata extractor...");
    let metadata_extractor = Arc::new(MetadataExtractor::new());
    info!("    ✅ Metadata extractor ready");

    Ok(RagComponents {
        embedder,
        vector_store,
        text_splitter,
        metadata_extractor,
    })
}

/// Run the complete indexing pipeline
async fn run_indexing_pipeline(
    demo_env: &DemoEnvironment,
    components: &RagComponents,
) -> Result<IndexingStats> {
    info!("🔄 Running indexing pipeline...");
    let start_time = std::time::Instant::now();

    // Step 1: Load documents
    info!("  📂 Loading documents...");
    let loader = DirectoryLoader::new(&demo_env.documents_path)?;
    let documents = loader.load().await?;
    info!("    ✅ Loaded {} documents", documents.len());

    // Step 2: Process documents through pipeline
    let mut all_nodes = Vec::new();
    let mut total_chunks = 0;

    for (i, document) in documents.iter().enumerate() {
        info!("  📄 Processing document {}/{}", i + 1, documents.len());

        // Split document into chunks
        let nodes = components.text_splitter.transform(document.clone()).await?;
        info!("    ✂️  Split into {} chunks", nodes.len());
        total_chunks += nodes.len();

        // Extract metadata for each node
        let mut enriched_nodes = Vec::new();
        for node in nodes {
            let enriched_node = components.metadata_extractor.transform_node(node).await?;
            enriched_nodes.push(enriched_node);
        }

        // Generate embeddings for each chunk
        let mut embedded_nodes = Vec::new();
        for mut node in enriched_nodes {
            let embedding = components.embedder.embed(&node.content).await?;
            node.embedding = Some(embedding);
            embedded_nodes.push(node);
        }

        all_nodes.extend(embedded_nodes);
    }

    info!("  🧮 Generated embeddings for {} chunks", all_nodes.len());

    // Step 3: Store in vector database
    info!("  💾 Storing in vector database...");
    let stored_ids = components.vector_store.add(all_nodes).await?;
    info!("    ✅ Stored {} nodes", stored_ids.len());

    let processing_time = start_time.elapsed();

    Ok(IndexingStats {
        documents_loaded: documents.len(),
        chunks_created: total_chunks,
        embeddings_generated: stored_ids.len(),
        nodes_stored: stored_ids.len(),
        processing_time,
    })
}

/// Build the query pipeline
async fn build_query_pipeline(components: &RagComponents) -> Result<Box<dyn QueryPipeline>> {
    info!("🔍 Building query pipeline...");

    // Create retriever
    let retriever =
        VectorRetriever::new(components.vector_store.clone(), components.embedder.clone());

    // Create response generator (check for API key)
    let generator = if std::env::var("OPENAI_API_KEY").is_ok() {
        info!("  🤖 Using OpenAI for response generation");
        let api_key = std::env::var("OPENAI_API_KEY").unwrap();
        let client = Siumai::builder()
            .openai()
            .api_key(&api_key)
            .model("gpt-3.5-turbo")
            .temperature(0.7)
            .max_tokens(1000)
            .build()
            .await
            .map_err(|e| cheungfun_core::CheungfunError::Configuration {
                message: format!("Failed to create OpenAI client: {}", e),
            })?;
        SiumaiGenerator::new(client)
    } else {
        warn!("  ⚠️  No OpenAI API key found, using Ollama");
        let client = Siumai::builder()
            .ollama()
            .base_url("http://localhost:11434")
            .model("llama2")
            .temperature(0.7)
            .max_tokens(1000)
            .build()
            .await
            .map_err(|e| cheungfun_core::CheungfunError::Configuration {
                message: format!("Failed to create Ollama client: {}", e),
            })?;
        SiumaiGenerator::new(client)
    };

    // Build query pipeline
    let query_pipeline = DefaultQueryPipeline::new(Arc::new(retriever), Arc::new(generator));

    Ok(Box::new(query_pipeline))
}

/// Demonstrate interactive querying with various questions
async fn demonstrate_querying(query_pipeline: &Box<dyn QueryPipeline>) -> Result<()> {
    info!("❓ Demonstrating query capabilities...");

    let test_queries = vec![
        "What is machine learning and what are its main types?",
        "How do convolutional neural networks work?",
        "What are vector databases used for?",
        "Explain how RAG systems work step by step",
        "What is the difference between supervised and unsupervised learning?",
        "What are the key features of vector databases?",
    ];

    for (i, query_text) in test_queries.iter().enumerate() {
        info!("");
        info!("🔍 Query {}/{}: {}", i + 1, test_queries.len(), query_text);
        info!("{}", "=".repeat(60));

        let query_options = cheungfun_core::QueryOptions::default();
        match query_pipeline.query(query_text, &query_options).await {
            Ok(response) => {
                info!("✅ Response generated successfully");
                info!("📝 Content: {}", response.response.content);
                info!(
                    "📊 Retrieved {} relevant chunks",
                    response.retrieved_nodes.len()
                );

                // Show top retrieved chunks
                for (j, scored_node) in response.retrieved_nodes.iter().take(2).enumerate() {
                    let preview = scored_node
                        .node
                        .content
                        .chars()
                        .take(100)
                        .collect::<String>();
                    info!(
                        "  {}. Score: {:.3} - {}...",
                        j + 1,
                        scored_node.score,
                        preview
                    );
                }

                if let Some(usage) = &response.response.usage {
                    info!(
                        "🔢 Token usage: {} prompt + {} completion = {} total",
                        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                    );
                }
            }
            Err(e) => {
                warn!("❌ Query failed: {}", e);
            }
        }
    }

    Ok(())
}

/// Print indexing statistics
fn print_indexing_stats(stats: &IndexingStats) {
    info!("");
    info!("📊 Indexing Statistics");
    info!("=====================");
    info!("📄 Documents loaded: {}", stats.documents_loaded);
    info!("✂️  Chunks created: {}", stats.chunks_created);
    info!("🧮 Embeddings generated: {}", stats.embeddings_generated);
    info!("💾 Nodes stored: {}", stats.nodes_stored);
    info!("⏱️  Processing time: {:?}", stats.processing_time);
    info!(
        "⚡ Average time per document: {:?}",
        stats.processing_time / stats.documents_loaded as u32
    );
}

/// Show comprehensive system statistics
async fn show_system_statistics(components: &RagComponents) -> Result<()> {
    info!("");
    info!("📈 System Statistics");
    info!("===================");

    // Vector store statistics
    let store_count = components.vector_store.count().await?;
    let store_metadata = components.vector_store.metadata().await?;
    info!("🗄️  Vector Store:");
    info!("   - Total nodes: {}", store_count);
    info!(
        "   - Type: {}",
        store_metadata.get("type").unwrap_or(&"unknown".into())
    );
    info!(
        "   - Dimension: {}",
        store_metadata.get("dimension").unwrap_or(&"unknown".into())
    );

    // Embedder statistics
    let embedder_metadata = components.embedder.metadata();
    info!("📊 Embedder:");
    info!("   - Model: {}", components.embedder.model_name());
    info!("   - Dimension: {}", components.embedder.dimension());
    info!(
        "   - Texts embedded: {}",
        embedder_metadata.get("texts_embedded").unwrap_or(&0.into())
    );

    // Health checks
    info!("🏥 Health Checks:");
    match components.vector_store.health_check().await {
        Ok(()) => info!("   - Vector store: ✅ Healthy"),
        Err(e) => info!("   - Vector store: ❌ Error: {}", e),
    }

    match components.embedder.health_check().await {
        Ok(()) => info!("   - Embedder: ✅ Healthy"),
        Err(e) => info!("   - Embedder: ❌ Error: {}", e),
    }

    Ok(())
}
