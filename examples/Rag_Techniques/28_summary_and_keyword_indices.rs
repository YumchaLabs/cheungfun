//! Summary Index and Keyword Table Index demonstration.
//!
//! This example demonstrates the usage of two new index types:
//! 1. SummaryIndex - Simple list-based index for small datasets and prototyping
//! 2. KeywordTableIndex - Keyword-based inverted index for exact term matching
//!
//! These indices complement the existing vector-based indices and provide
//! different retrieval strategies suitable for various use cases.

use cheungfun_core::{
    traits::{KeywordExtractionConfig, KeywordStore, Retriever, SimpleKeywordExtractor},
    types::{Document, Query},
    Result,
};
use cheungfun_integrations::InMemoryKeywordStore;
use cheungfun_query::indices::{
    KeywordTableIndex, KeywordTableIndexConfig, SummaryIndex, SummaryIndexConfig,
};
use std::sync::Arc;
use tracing::{info, Level};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🚀 Starting Summary Index and Keyword Table Index demonstration");

    // Create sample documents
    let documents = create_sample_documents();
    info!("📚 Created {} sample documents", documents.len());

    // Demonstrate SummaryIndex
    info!("\n=== 📝 SummaryIndex Demonstration ===");
    demonstrate_summary_index(&documents).await?;

    // Demonstrate KeywordTableIndex
    info!("\n=== 🔑 KeywordTableIndex Demonstration ===");
    demonstrate_keyword_table_index(&documents).await?;

    // Compare performance and use cases
    info!("\n=== ⚖️ Comparison and Use Cases ===");
    compare_indices(&documents).await?;

    info!("\n✅ Demonstration completed successfully!");
    Ok(())
}

/// Create sample documents for demonstration.
fn create_sample_documents() -> Vec<Document> {
    vec![
        Document::new("Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.")
            .with_metadata("category", "programming")
            .with_metadata("language", "rust")
            .with_metadata("difficulty", "intermediate"),

        Document::new("Python is an interpreted, high-level programming language with dynamic semantics. Its high-level built-in data structures make it attractive for Rapid Application Development.")
            .with_metadata("category", "programming")
            .with_metadata("language", "python")
            .with_metadata("difficulty", "beginner"),

        Document::new("Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data.")
            .with_metadata("category", "ai")
            .with_metadata("topic", "machine_learning")
            .with_metadata("difficulty", "advanced"),

        Document::new("Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They are essential for similarity search and recommendation systems.")
            .with_metadata("category", "database")
            .with_metadata("topic", "vector_search")
            .with_metadata("difficulty", "intermediate"),

        Document::new("Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.")
            .with_metadata("category", "ai")
            .with_metadata("topic", "nlp")
            .with_metadata("difficulty", "advanced"),

        Document::new("RAG (Retrieval-Augmented Generation) combines the power of retrieval systems with generative models to provide more accurate and contextual responses.")
            .with_metadata("category", "ai")
            .with_metadata("topic", "rag")
            .with_metadata("difficulty", "advanced"),
    ]
}

/// Demonstrate SummaryIndex usage.
async fn demonstrate_summary_index(documents: &[Document]) -> Result<()> {
    info!("Creating SummaryIndex with custom configuration...");

    // Create SummaryIndex with custom configuration
    let config = SummaryIndexConfig {
        show_progress: true,
        max_nodes: Some(100),
        deduplicate: true,
        min_content_length: 20,
    };

    let summary_index = SummaryIndex::from_documents(documents.to_vec(), Some(config)).await?;

    info!("✅ SummaryIndex created successfully!");
    info!("📊 Index stats: {} nodes", summary_index.len());
    info!("📈 Statistics: {:?}", summary_index.stats());

    // Create retriever and test queries
    let retriever = summary_index.as_retriever();

    // Test different types of queries
    let test_queries = vec![
        "programming languages",
        "artificial intelligence",
        "machine learning systems",
        "database technology",
    ];

    for query_text in test_queries {
        info!("\n🔍 Query: '{}'", query_text);
        let query = Query::new(query_text).with_top_k(3);

        let results = retriever.retrieve(&query).await?;
        info!("📋 Found {} results:", results.len());

        for (i, result) in results.iter().enumerate() {
            info!(
                "  {}. Score: {:.3} | Content: {}...",
                i + 1,
                result.score,
                result.node.content.chars().take(80).collect::<String>()
            );
        }
    }

    Ok(())
}

/// Demonstrate KeywordTableIndex usage.
async fn demonstrate_keyword_table_index(documents: &[Document]) -> Result<()> {
    info!("Creating KeywordTableIndex with custom configuration...");

    // Create keyword store
    let keyword_store = Arc::new(InMemoryKeywordStore::new());

    // Create custom keyword extractor
    let extractor_config = KeywordExtractionConfig {
        min_keyword_length: 3,
        max_keyword_length: 20,
        lowercase: true,
        remove_stop_words: true,
        custom_stop_words: vec!["the".to_string(), "and".to_string(), "with".to_string()],
        min_frequency: 1,
        max_keywords_per_node: Some(50),
    };
    let keyword_extractor = Box::new(SimpleKeywordExtractor::new(extractor_config));

    // Create KeywordTableIndex with custom configuration
    let config = KeywordTableIndexConfig {
        show_progress: true,
        max_nodes: Some(100),
        deduplicate: true,
        min_content_length: 20,
        max_results: 10,
        score_threshold: Some(0.01),
        use_and_logic: false, // Use OR logic for multiple keywords
    };

    let keyword_index = KeywordTableIndex::from_documents(
        documents.to_vec(),
        keyword_store.clone(),
        Some(keyword_extractor),
        Some(config),
    )
    .await?;

    info!("✅ KeywordTableIndex created successfully!");
    info!("📊 Index stats: {} nodes", keyword_index.len());
    info!("📈 Statistics: {:?}", keyword_index.stats());

    // Show some extracted keywords
    let keyword_stats = keyword_store.stats().await?;
    info!("🔑 Total keywords: {}", keyword_stats.total_keywords);
    info!("🔗 Total mappings: {}", keyword_stats.total_mappings);
    info!("📊 Top keywords: {:?}", keyword_stats.top_keywords);

    // Create retriever and test queries
    let retriever = keyword_index.as_retriever();

    // Test keyword-based queries
    let test_queries = vec![
        "rust programming",
        "machine learning",
        "python language",
        "vector database",
        "artificial intelligence",
        "nlp processing",
    ];

    for query_text in test_queries {
        info!("\n🔍 Keyword Query: '{}'", query_text);
        let query = Query::new(query_text).with_top_k(3);

        let results = retriever.retrieve(&query).await?;
        info!("📋 Found {} results:", results.len());

        for (i, result) in results.iter().enumerate() {
            info!(
                "  {}. Score: {:.3} | Content: {}...",
                i + 1,
                result.score,
                result.node.content.chars().take(80).collect::<String>()
            );
        }
    }

    // Test AND logic
    info!("\n🔍 Testing AND logic (all keywords must match):");
    let mut and_config = KeywordTableIndexConfig::default();
    and_config.use_and_logic = true;

    let and_keyword_store = Arc::new(InMemoryKeywordStore::new());
    let and_index = KeywordTableIndex::from_documents(
        documents.to_vec(),
        and_keyword_store,
        None,
        Some(and_config),
    )
    .await?;

    let and_retriever = and_index.as_retriever();
    let and_query = Query::new("programming language").with_top_k(5);
    let and_results = and_retriever.retrieve(&and_query).await?;

    info!(
        "📋 AND logic results for 'programming language': {} matches",
        and_results.len()
    );
    for (i, result) in and_results.iter().enumerate() {
        info!(
            "  {}. Score: {:.3} | Content: {}...",
            i + 1,
            result.score,
            result.node.content.chars().take(80).collect::<String>()
        );
    }

    Ok(())
}

/// Compare the two indices and their use cases.
async fn compare_indices(documents: &[Document]) -> Result<()> {
    info!("Comparing SummaryIndex vs KeywordTableIndex...");

    // Create both indices
    let summary_index = SummaryIndex::from_documents(documents.to_vec(), None).await?;
    let keyword_store = Arc::new(InMemoryKeywordStore::new());
    let keyword_index =
        KeywordTableIndex::from_documents(documents.to_vec(), keyword_store, None, None).await?;

    let summary_retriever = summary_index.as_retriever();
    let keyword_retriever = keyword_index.as_retriever();

    // Test the same query on both indices
    let test_query = "machine learning artificial intelligence";
    let query = Query::new(test_query).with_top_k(3);

    info!("\n🔍 Comparing results for query: '{}'", test_query);

    // SummaryIndex results
    let summary_results = summary_retriever.retrieve(&query).await?;
    info!(
        "\n📝 SummaryIndex Results ({} found):",
        summary_results.len()
    );
    for (i, result) in summary_results.iter().enumerate() {
        info!(
            "  {}. Score: {:.3} | {}...",
            i + 1,
            result.score,
            result.node.content.chars().take(60).collect::<String>()
        );
    }

    // KeywordTableIndex results
    let keyword_results = keyword_retriever.retrieve(&query).await?;
    info!(
        "\n🔑 KeywordTableIndex Results ({} found):",
        keyword_results.len()
    );
    for (i, result) in keyword_results.iter().enumerate() {
        info!(
            "  {}. Score: {:.3} | {}...",
            i + 1,
            result.score,
            result.node.content.chars().take(60).collect::<String>()
        );
    }

    // Print use case recommendations
    info!("\n📋 Use Case Recommendations:");
    info!("📝 SummaryIndex:");
    info!("  ✅ Small datasets (< 1000 documents)");
    info!("  ✅ Prototyping and development");
    info!("  ✅ When you need comprehensive search (no information missed)");
    info!("  ✅ Simple applications without complex indexing needs");
    info!("  ❌ Large datasets (performance issues)");
    info!("  ❌ When you need fast keyword-based filtering");

    info!("\n🔑 KeywordTableIndex:");
    info!("  ✅ Exact term matching and keyword-based search");
    info!("  ✅ Technical documentation and legal documents");
    info!("  ✅ Fast filtering and initial document screening");
    info!("  ✅ When you know specific terms to search for");
    info!("  ✅ Complement to vector-based semantic search");
    info!("  ❌ When you need semantic understanding");
    info!("  ❌ Fuzzy or conceptual queries");

    info!("\n💡 Best Practice: Use KeywordTableIndex for initial filtering, then apply semantic search!");

    Ok(())
}
