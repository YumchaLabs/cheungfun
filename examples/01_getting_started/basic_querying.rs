//! Basic Querying - Learn how to search and retrieve information
//!
//! This example demonstrates:
//! 1. Different types of queries
//! 2. Similarity search parameters
//! 3. Result filtering and ranking
//! 4. Response generation with mock components
//! 5. Query optimization techniques
//!
//! To run this example:
//! ```bash
//! cargo run --bin basic_querying
//! ```

use anyhow::Result;
use cheungfun_core::{
    traits::{DistanceMetric, Embedder, VectorStore},
    types::{ChunkInfo, Node, Query, ScoredNode},
};
use cheungfun_integrations::InMemoryVectorStore;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;
use tracing::{info, Level};
use uuid::Uuid;

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

    // Step 2: Demonstrate basic queries
    println!("💬 Step 2: Basic queries...");
    let basic_queries = vec![
        "What is Rust programming language?",
        "How does machine learning work with Rust?",
        "What are the benefits of web development in Rust?",
        "Tell me about Rust's memory safety features",
    ];

    for (i, question) in basic_queries.iter().enumerate() {
        println!("❓ Query {}: {}", i + 1, question);

        // Create query and search
        let query_embedding = embedder.embed(question).await?;
        let query = Query::new(*question)
            .with_embedding(query_embedding)
            .with_top_k(3);

        let results = vector_store.search(&query).await?;

        println!("💡 Found {} relevant chunks:", results.len());
        for (j, scored_node) in results.iter().enumerate() {
            println!(
                "  {}. Score: {:.3} - {}",
                j + 1,
                scored_node.score,
                &scored_node.node.content[..100.min(scored_node.node.content.len())]
            );
        }
        println!();
    }

    // Step 3: Advanced query parameters
    println!("🎯 Step 3: Advanced query parameters...");
    demonstrate_query_parameters(&vector_store, &embedder).await?;
    println!();

    // Step 4: Query optimization
    println!("⚡ Step 4: Query optimization techniques...");
    demonstrate_query_optimization().await?;
    println!();

    // Step 5: Performance analysis
    println!("📊 Step 5: Query performance analysis...");
    analyze_query_performance(&vector_store, &embedder).await?;

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
    // Set up components
    let embedder = Arc::new(MockEmbedder::new(384));
    let vector_store = Arc::new(InMemoryVectorStore::new(384, DistanceMetric::Cosine));

    // Create sample documents and index them
    let sample_texts = vec![
        "Rust is a systems programming language focused on safety, speed, and concurrency. It prevents segfaults and guarantees thread safety.",
        "Machine learning in Rust offers memory safety and high performance for ML workloads. Libraries like Candle provide ML capabilities.",
        "Rust web frameworks like Axum and Actix provide excellent performance for web services with strong type safety.",
        "Rust's ownership system ensures memory safety without garbage collection, making it ideal for system programming.",
        "The Rust ecosystem includes powerful tools like Cargo for package management and excellent documentation.",
    ];

    let mut nodes = Vec::new();
    let source_doc_id = Uuid::new_v4();

    for (i, text) in sample_texts.iter().enumerate() {
        let embedding = embedder.embed(text).await?;
        let node = Node {
            id: Uuid::new_v4(),
            content: text.to_string(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert(
                    "source".to_string(),
                    serde_json::Value::String(format!("doc_{}.txt", i)),
                );
                meta.insert(
                    "chunk_index".to_string(),
                    serde_json::Value::Number(i.into()),
                );
                meta
            },
            embedding: Some(embedding),
            sparse_embedding: None,
            relationships: HashMap::new(),
            source_document_id: source_doc_id,
            chunk_info: ChunkInfo {
                start_offset: i * 100,
                end_offset: (i + 1) * 100,
                chunk_index: i,
            },
        };
        nodes.push(node);
    }

    vector_store.add(nodes).await?;

    Ok((vector_store, embedder))
}

/// Demonstrate different query parameters
async fn demonstrate_query_parameters(
    vector_store: &InMemoryVectorStore,
    embedder: &MockEmbedder,
) -> Result<()> {
    println!("🔧 Query Parameter Examples:");

    // Different top_k values
    let query_text = "Rust memory safety";
    let query_embedding = embedder.embed(query_text).await?;

    for top_k in [1, 3, 5] {
        let query = Query::new(query_text)
            .with_embedding(query_embedding.clone())
            .with_top_k(top_k);

        let results = vector_store.search(&query).await?;
        println!("  📊 top_k={}: Found {} results", top_k, results.len());
    }

    // Different similarity thresholds (simulated)
    println!();
    println!("🎯 Similarity Threshold Examples:");

    for threshold in [0.3, 0.5, 0.7, 0.9] {
        let query = Query::new(query_text)
            .with_embedding(query_embedding.clone())
            .with_top_k(10);

        let results = vector_store.search(&query).await?;
        // Filter by threshold (simulated)
        let filtered_results: Vec<_> = results
            .into_iter()
            .filter(|r| r.score >= threshold)
            .collect();

        println!(
            "  🎯 threshold={}: Found {} results",
            threshold,
            filtered_results.len()
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
    vector_store: &InMemoryVectorStore,
    embedder: &MockEmbedder,
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

    for (i, query_text) in test_queries.iter().enumerate() {
        let start = std::time::Instant::now();

        // Embed query and search
        let query_embedding = embedder.embed(query_text).await?;
        let query = Query::new(*query_text)
            .with_embedding(query_embedding)
            .with_top_k(3);
        let _results = vector_store.search(&query).await?;

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

/// Mock embedder for demonstration
#[derive(Debug)]
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

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        "MockEmbedder"
    }

    async fn health_check(&self) -> Result<(), cheungfun_core::error::CheungfunError> {
        Ok(())
    }
}
