//! Basic Querying - Learn how to search and retrieve information
//!
//! This example demonstrates:
//! 1. Different types of queries
//! 2. Similarity search parameters
//! 3. Result filtering and ranking
//! 4. Response generation
//! 5. Query optimization techniques

use anyhow::Result;
use cheungfun_core::{
    DistanceMetric,
    types::{GenerationOptions, Query, SearchMode},
};
use cheungfun_indexing::{
    loaders::text::TextLoader, pipeline::IndexingPipelineBuilder,
    transformers::text_splitter::TextSplitter,
};
use cheungfun_integrations::vector_stores::memory::InMemoryVectorStore;
use cheungfun_query::{
    generators::mock::MockResponseGenerator, pipeline::QueryPipelineBuilder,
    retrievers::vector::VectorRetriever,
};
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;
use tokio;
use tracing::{Level, info};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    println!("🔍 Cheungfun Basic Querying Example");
    println!("===================================");
    println!();

    // Step 1: Set up a knowledge base
    println!("📚 Step 1: Setting up knowledge base...");
    let (vector_store, embedder) = setup_knowledge_base().await?;
    println!("✅ Knowledge base ready with sample documents");
    println!();

    // Step 2: Create query pipeline
    println!("🔧 Step 2: Creating query pipeline...");
    let retriever = Arc::new(VectorRetriever::new(vector_store.clone(), embedder.clone()));
    let response_generator = Arc::new(MockResponseGenerator::new());

    let query_pipeline = QueryPipelineBuilder::new()
        .with_retriever_arc(retriever.clone())
        .with_response_generator_arc(response_generator)
        .build()?;

    println!("✅ Query pipeline ready");
    println!();

    // Step 3: Basic queries
    println!("💬 Step 3: Basic queries...");
    let basic_queries = vec![
        "What is Rust programming language?",
        "How does machine learning work with Rust?",
        "What are the benefits of web development in Rust?",
        "Tell me about Rust's memory safety features",
    ];

    for (i, question) in basic_queries.iter().enumerate() {
        println!("❓ Query {}: {}", i + 1, question);

        let options = GenerationOptions::default();
        let response = query_pipeline.query(question, &options).await?;

        println!("💡 Answer: {}", response.content);
        println!("📄 Sources: {} chunks", response.source_nodes.len());
        println!();
    }

    // Step 4: Advanced query parameters
    println!("🎯 Step 4: Advanced query parameters...");
    demonstrate_query_parameters(&retriever).await?;
    println!();

    // Step 5: Query optimization
    println!("⚡ Step 5: Query optimization techniques...");
    demonstrate_query_optimization().await?;
    println!();

    // Step 6: Performance analysis
    println!("📊 Step 6: Query performance analysis...");
    analyze_query_performance(&query_pipeline).await?;

    println!("🎉 Basic querying example completed!");
    println!();
    println!("🚀 Next Steps:");
    println!("  1. Explore 03_advanced_features/ for hybrid search and reranking");
    println!("  2. Check 02_core_components/vector_stores/ for different storage options");
    println!("  3. See 05_performance/optimization/ for query performance tips");
    println!("  4. Try 07_use_cases/ for real-world application examples");

    Ok(())
}

/// Set up a knowledge base with sample documents
async fn setup_knowledge_base() -> Result<(Arc<InMemoryVectorStore>, Arc<MockEmbedder>)> {
    // Create sample documents
    let temp_dir = TempDir::new()?;
    create_sample_documents(&temp_dir).await?;

    // Set up components
    let embedder = Arc::new(MockEmbedder::new(384));
    let vector_store = Arc::new(InMemoryVectorStore::new(384, DistanceMetric::Cosine));
    let text_splitter = Arc::new(TextSplitter::new(500, 50));

    // Index documents
    let pipeline = IndexingPipelineBuilder::new()
        .with_loader(TextLoader::new(get_document_paths(&temp_dir)?))
        .with_transformer_arc(text_splitter)
        .with_embedder_arc(embedder.clone())
        .with_vector_store_arc(vector_store.clone())
        .build()?;

    pipeline.run().await?;

    Ok((vector_store, embedder))
}

/// Demonstrate different query parameters
async fn demonstrate_query_parameters(retriever: &VectorRetriever) -> Result<()> {
    println!("🔧 Query Parameter Examples:");

    // Different top_k values
    let query_text = "Rust memory safety";

    for top_k in [1, 3, 5, 10] {
        let query = Query {
            text: query_text.to_string(),
            embedding: None,
            filters: HashMap::new(),
            top_k,
            similarity_threshold: Some(0.5),
            search_mode: SearchMode::Vector,
        };

        let results = retriever.retrieve(&query).await?;
        println!("  📊 top_k={}: Found {} results", top_k, results.len());
    }

    // Different similarity thresholds
    println!();
    println!("🎯 Similarity Threshold Examples:");

    for threshold in [0.3, 0.5, 0.7, 0.9] {
        let query = Query {
            text: query_text.to_string(),
            embedding: None,
            filters: HashMap::new(),
            top_k: 10,
            similarity_threshold: Some(threshold),
            search_mode: SearchMode::Vector,
        };

        let results = retriever.retrieve(&query).await?;
        println!(
            "  🎯 threshold={}: Found {} results",
            threshold,
            results.len()
        );
    }

    Ok(())
}

/// Demonstrate query optimization techniques
async fn demonstrate_query_optimization() -> Result<()> {
    println!("⚡ Query Optimization Techniques:");
    println!("  🔄 Query Expansion: Add related terms to improve recall");
    println!("  📝 Query Rewriting: Rephrase queries for better matching");
    println!("  🎯 Semantic Caching: Cache embeddings for repeated queries");
    println!("  📊 Result Reranking: Improve result quality with secondary scoring");
    println!("  🔍 Hybrid Search: Combine vector and keyword search");
    println!("  ⚖️  Score Fusion: Merge multiple relevance signals");

    Ok(())
}

/// Analyze query performance
async fn analyze_query_performance(
    query_pipeline: &cheungfun_query::pipeline::DefaultQueryPipeline,
) -> Result<()> {
    println!("📊 Performance Analysis:");

    let test_queries = vec![
        "What is Rust?",
        "Machine learning in Rust",
        "Web development frameworks",
        "Memory safety features",
        "Performance characteristics",
    ];

    let mut total_time = std::time::Duration::ZERO;

    for (i, query) in test_queries.iter().enumerate() {
        let start = std::time::Instant::now();
        let options = GenerationOptions::default();
        let _response = query_pipeline.query(query, &options).await?;
        let duration = start.elapsed();

        total_time += duration;
        println!("  ⏱️  Query {}: {:?}", i + 1, duration);
    }

    let avg_time = total_time / test_queries.len() as u32;
    println!("  📈 Average query time: {:?}", avg_time);
    println!(
        "  🎯 Queries per second: {:.1}",
        1.0 / avg_time.as_secs_f64()
    );

    Ok(())
}

/// Create sample documents
async fn create_sample_documents(temp_dir: &TempDir) -> Result<()> {
    let documents = vec![
        (
            "rust_basics.txt",
            "Rust is a systems programming language focused on safety and performance.",
        ),
        (
            "ml_rust.txt",
            "Machine learning in Rust offers memory safety and high performance for ML workloads.",
        ),
        (
            "web_rust.txt",
            "Rust web frameworks like Axum and Actix provide excellent performance for web services.",
        ),
    ];

    for (filename, content) in documents {
        let file_path = temp_dir.path().join(filename);
        tokio::fs::write(file_path, content).await?;
    }

    Ok(())
}

/// Get document paths
fn get_document_paths(temp_dir: &TempDir) -> Result<Vec<std::path::PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(temp_dir.path())? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            paths.push(entry.path());
        }
    }
    Ok(paths)
}

/// Mock embedder for demonstration
struct MockEmbedder {
    dimension: usize,
}

impl MockEmbedder {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait::async_trait]
impl cheungfun_core::traits::Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, cheungfun_core::error::CheungfunError> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        let mut embedding = vec![0.0; self.dimension];
        for i in 0..self.dimension {
            embedding[i] = ((hash.wrapping_add(i as u64) % 1000) as f32 - 500.0) / 500.0;
        }

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(embedding)
    }

    async fn embed_batch(
        &self,
        texts: Vec<&str>,
    ) -> Result<Vec<Vec<f32>>, cheungfun_core::error::CheungfunError> {
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.embed(text).await?);
        }
        Ok(embeddings)
    }

    fn name(&self) -> &'static str {
        "MockEmbedder"
    }
}
