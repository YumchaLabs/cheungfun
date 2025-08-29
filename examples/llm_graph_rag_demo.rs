//! LLM-driven Graph RAG demonstration.
//!
//! This example demonstrates how to use Cheungfun's LLM-powered knowledge graph
//! extraction capabilities to build and query a graph-based RAG system.

use cheungfun_core::{
    traits::{PropertyGraphStore, Retriever},
    types::Document,
    Query,
};
use cheungfun_indexing::transformers::{LlmExtractor, LlmExtractionConfig, ExtractionFormat};
use cheungfun_integrations::SimplePropertyGraphStore;
use cheungfun_query::{PropertyGraphIndex, PropertyGraphIndexConfig};
use siumai::prelude::*;
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::init();

    println!("🚀 Cheungfun LLM-driven Graph RAG Demo");
    println!("=====================================\n");

    // Step 1: Set up LLM client
    println!("📡 Setting up LLM client...");
    let llm_client = Siumai::builder()
        .openai()
        .api_key(std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"))
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Step 2: Configure LLM extraction
    println!("⚙️  Configuring LLM extraction...");
    let extraction_config = LlmExtractionConfig {
        max_triplets_per_chunk: 15,
        format: ExtractionFormat::Json, // Use JSON for better reliability
        temperature: 0.1, // Low temperature for consistent extraction
        enable_validation: true,
        enable_deduplication: true,
        min_confidence: 0.6,
        show_progress: true,
        ..Default::default()
    };

    // Step 3: Create LLM extractor
    let llm_extractor = Arc::new(LlmExtractor::new(llm_client, extraction_config)?);

    // Step 4: Set up graph storage
    println!("🗄️  Setting up graph storage...");
    let graph_store = Arc::new(SimplePropertyGraphStore::new());

    // Step 5: Create PropertyGraphIndex with LLM extraction
    println!("🏗️  Creating PropertyGraphIndex with LLM extraction...");
    let index_config = PropertyGraphIndexConfig {
        enable_llm_extraction: true,
        show_progress: true,
        embed_kg_nodes: false, // Focus on graph structure for this demo
        ..Default::default()
    };

    let mut index = PropertyGraphIndex::with_llm_extractor(
        graph_store.clone(),
        None, // No vector store for this demo
        llm_extractor,
        Some(index_config),
    );

    // Step 6: Prepare sample documents
    println!("📄 Preparing sample documents...");
    let documents = vec![
        Document::new("Alice is a software engineer at Microsoft. She works on Azure cloud services and lives in Seattle."),
        Document::new("Bob is Alice's colleague at Microsoft. He is the team lead for the Azure AI platform and previously worked at Google."),
        Document::new("Microsoft is a technology company founded by Bill Gates and Paul Allen in 1975. It is headquartered in Redmond, Washington."),
        Document::new("Azure is Microsoft's cloud computing platform. It provides services like virtual machines, databases, and AI tools."),
        Document::new("Seattle is a major city in Washington state. It is known for its tech industry and is home to companies like Microsoft and Amazon."),
    ];

    // Step 7: Index documents with LLM extraction
    println!("🔍 Indexing documents with LLM extraction...");
    index.insert_documents(documents).await?;

    // Step 8: Display extraction results
    println!("\n📊 Extraction Results:");
    let stats = index.stats().await?;
    println!("  • Entities: {}", stats.graph_stats.entity_count);
    println!("  • Relations: {}", stats.graph_stats.relation_count);
    println!("  • Triplets: {}", stats.graph_stats.triplet_count);

    // Step 9: Query the knowledge graph
    println!("\n🔎 Querying the knowledge graph...");
    let retriever = index.as_retriever();

    let queries = vec![
        "Where does Alice work?",
        "What is Azure?",
        "Who founded Microsoft?",
        "What city is Microsoft located in?",
        "Who is Bob's colleague?",
    ];

    for query_text in queries {
        println!("\n❓ Query: {}", query_text);
        let query = Query::new(query_text);
        let results = retriever.retrieve(&query).await?;

        if results.is_empty() {
            println!("   No results found.");
        } else {
            println!("   Found {} results:", results.len());
            for (i, result) in results.iter().take(3).enumerate() {
                println!("   {}. {} (score: {:.3})", i + 1, result.node.get_text(), result.score);
            }
        }
    }

    // Step 10: Display some extracted triplets
    println!("\n🕸️  Sample extracted triplets:");
    let all_triplets = graph_store.get_triplets(None, None, None, Some(10)).await?;
    for (i, triplet) in all_triplets.iter().take(10).enumerate() {
        println!("   {}. ({}, {}, {})", 
                 i + 1, 
                 triplet.subject.name, 
                 triplet.relation.label, 
                 triplet.object.name);
    }

    println!("\n✅ Demo completed successfully!");
    println!("\n💡 Key Features Demonstrated:");
    println!("   • LLM-powered entity and relationship extraction");
    println!("   • JSON-based structured output parsing");
    println!("   • Automatic validation and deduplication");
    println!("   • Graph-based knowledge retrieval");
    println!("   • Integration with PropertyGraphIndex");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_llm_extraction_config() {
        let config = LlmExtractionConfig {
            format: ExtractionFormat::Json,
            enable_validation: true,
            enable_deduplication: true,
            ..Default::default()
        };

        assert_eq!(config.max_triplets_per_chunk, 10);
        assert!(config.enable_validation);
        assert!(config.enable_deduplication);
        assert!(matches!(config.format, ExtractionFormat::Json));
    }

    #[tokio::test]
    async fn test_property_graph_index_with_llm() {
        // This test requires an actual LLM client, so we'll just test the structure
        let graph_store = Arc::new(SimplePropertyGraphStore::new());
        
        // Create a mock LLM client for testing
        // In a real test, you'd use a mock or test LLM client
        let config = PropertyGraphIndexConfig {
            enable_llm_extraction: true,
            show_progress: false,
            ..Default::default()
        };

        // Test that the config is set correctly
        assert!(config.enable_llm_extraction);
        assert!(!config.show_progress);
    }
}
