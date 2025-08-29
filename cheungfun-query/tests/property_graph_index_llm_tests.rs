//! Integration tests for PropertyGraphIndex with LLM extraction.
//!
//! These tests verify the complete pipeline from documents to knowledge graph
//! using LLM-powered entity extraction.

use cheungfun_core::{
    traits::{PropertyGraphStore, Retriever},
    types::Document,
    Query,
};
use cheungfun_indexing::transformers::{ExtractionFormat, LlmExtractionConfig, LlmExtractor};
use cheungfun_integrations::SimplePropertyGraphStore;
use cheungfun_query::{PropertyGraphIndex, PropertyGraphIndexConfig};
use siumai::prelude::*;
use std::sync::Arc;
use tokio;

/// Mock LLM client for testing PropertyGraphIndex integration.
pub struct MockGraphLlmClient {
    pub response: String,
}

impl MockGraphLlmClient {
    pub fn new() -> Self {
        Self {
            response: r#"[
                {"subject": "Alice", "predicate": "works at", "object": "Microsoft"},
                {"subject": "Alice", "predicate": "lives in", "object": "Seattle"},
                {"subject": "Microsoft", "predicate": "located in", "object": "Seattle"},
                {"subject": "Bob", "predicate": "works at", "object": "Microsoft"},
                {"subject": "Bob", "predicate": "colleague of", "object": "Alice"}
            ]"#
            .to_string(),
        }
    }
}

#[async_trait::async_trait]
impl LlmClient for MockGraphLlmClient {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, Box<dyn std::error::Error + Send + Sync>> {
        Ok(CompletionResponse {
            content: self.response.clone(),
            model: "mock-model".to_string(),
            usage: None,
        })
    }

    async fn chat(
        &self,
        _request: ChatRequest,
    ) -> Result<ChatResponse, Box<dyn std::error::Error + Send + Sync>> {
        unimplemented!("Chat not implemented for mock")
    }

    async fn stream_complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<
        Box<
            dyn futures::Stream<
                    Item = Result<CompletionResponse, Box<dyn std::error::Error + Send + Sync>>,
                > + Send
                + Unpin,
        >,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        unimplemented!("Stream complete not implemented for mock")
    }

    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> Result<
        Box<
            dyn futures::Stream<
                    Item = Result<ChatResponse, Box<dyn std::error::Error + Send + Sync>>,
                > + Send
                + Unpin,
        >,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        unimplemented!("Stream chat not implemented for mock")
    }
}

#[tokio::test]
async fn test_property_graph_index_with_llm_extraction() {
    let mock_client = Arc::new(MockGraphLlmClient::new());
    let graph_store = Arc::new(SimplePropertyGraphStore::new());

    // Configure LLM extraction
    let extraction_config = LlmExtractionConfig {
        format: ExtractionFormat::Json,
        max_triplets_per_chunk: 10,
        enable_validation: true,
        enable_deduplication: true,
        show_progress: false,
        ..Default::default()
    };

    let llm_extractor = Arc::new(LlmExtractor::new(mock_client, extraction_config).unwrap());

    // Create PropertyGraphIndex with LLM extraction
    let index_config = PropertyGraphIndexConfig {
        enable_llm_extraction: true,
        show_progress: false,
        embed_kg_nodes: false,
        ..Default::default()
    };

    let mut index = PropertyGraphIndex::with_llm_extractor(
        graph_store.clone(),
        None,
        llm_extractor,
        Some(index_config),
    );

    // Test documents
    let documents = vec![
        Document::new("Alice works at Microsoft and lives in Seattle."),
        Document::new("Bob is Alice's colleague at Microsoft."),
    ];

    // Index documents
    index.insert_documents(documents).await.unwrap();

    // Verify extraction results
    let stats = index.stats().await.unwrap();
    assert!(
        stats.graph_stats.entity_count > 0,
        "Should have extracted entities"
    );
    assert!(
        stats.graph_stats.relation_count > 0,
        "Should have extracted relations"
    );
    assert!(
        stats.graph_stats.triplet_count > 0,
        "Should have extracted triplets"
    );

    println!(
        "Extracted {} entities, {} relations, {} triplets",
        stats.graph_stats.entity_count,
        stats.graph_stats.relation_count,
        stats.graph_stats.triplet_count
    );

    // Test retrieval
    let retriever = index.as_retriever();
    let query = Query::new("Where does Alice work?");
    let results = retriever.retrieve(&query).await.unwrap();

    // Should find relevant results
    assert!(
        !results.is_empty(),
        "Should find results for Alice work query"
    );

    // Test specific entity queries
    let alice_query = Query::new("Alice");
    let alice_results = retriever.retrieve(&alice_query).await.unwrap();
    assert!(!alice_results.is_empty(), "Should find results for Alice");

    let microsoft_query = Query::new("Microsoft");
    let microsoft_results = retriever.retrieve(&microsoft_query).await.unwrap();
    assert!(
        !microsoft_results.is_empty(),
        "Should find results for Microsoft"
    );
}

#[tokio::test]
async fn test_property_graph_index_from_documents_with_llm() {
    let mock_client = Arc::new(MockGraphLlmClient::new());
    let graph_store = Arc::new(SimplePropertyGraphStore::new());

    let extraction_config = LlmExtractionConfig {
        format: ExtractionFormat::Json,
        max_triplets_per_chunk: 10,
        enable_validation: true,
        ..Default::default()
    };

    let llm_extractor = Arc::new(LlmExtractor::new(mock_client, extraction_config).unwrap());

    let documents = vec![
        Document::new("Alice is a software engineer at Microsoft."),
        Document::new("Microsoft is headquartered in Redmond, Washington."),
        Document::new("Seattle is a major city in Washington state."),
    ];

    // Create index with LLM extraction using from_documents pattern
    let index_config = PropertyGraphIndexConfig {
        enable_llm_extraction: true,
        show_progress: false,
        ..Default::default()
    };

    let mut index = PropertyGraphIndex::with_llm_extractor(
        graph_store.clone(),
        None,
        llm_extractor,
        Some(index_config),
    );

    index.insert_documents(documents).await.unwrap();

    // Verify the knowledge graph was built
    let stats = index.stats().await.unwrap();
    assert!(
        stats.graph_stats.entity_count >= 3,
        "Should extract at least 3 entities"
    );

    // Test that we can retrieve specific triplets
    let all_triplets = graph_store
        .get_triplets(None, None, None, Some(20))
        .await
        .unwrap();
    assert!(!all_triplets.is_empty(), "Should have extracted triplets");

    // Print extracted triplets for verification
    println!("Extracted triplets:");
    for (i, triplet) in all_triplets.iter().take(10).enumerate() {
        println!(
            "  {}. ({}, {}, {})",
            i + 1,
            triplet.subject.name,
            triplet.relation.label,
            triplet.object.name
        );
    }

    // Test specific entity retrieval
    let alice_triplets = graph_store
        .get_triplets(Some(vec!["Alice".to_string()]), None, None, None)
        .await
        .unwrap();
    assert!(
        !alice_triplets.is_empty(),
        "Should find triplets involving Alice"
    );

    let microsoft_triplets = graph_store
        .get_triplets(Some(vec!["Microsoft".to_string()]), None, None, None)
        .await
        .unwrap();
    assert!(
        !microsoft_triplets.is_empty(),
        "Should find triplets involving Microsoft"
    );
}

#[tokio::test]
async fn test_property_graph_index_retrieval_strategies() {
    let mock_client = Arc::new(MockGraphLlmClient::new());
    let graph_store = Arc::new(SimplePropertyGraphStore::new());

    let extraction_config = LlmExtractionConfig {
        format: ExtractionFormat::Json,
        enable_validation: true,
        enable_deduplication: true,
        ..Default::default()
    };

    let llm_extractor = Arc::new(LlmExtractor::new(mock_client, extraction_config).unwrap());

    let mut index =
        PropertyGraphIndex::with_llm_extractor(graph_store.clone(), None, llm_extractor, None);

    let documents = vec![
        Document::new("Alice works at Microsoft in Seattle."),
        Document::new("Bob is a colleague of Alice at Microsoft."),
        Document::new("Microsoft develops software and cloud services."),
    ];

    index.insert_documents(documents).await.unwrap();

    let retriever = index.as_retriever();

    // Test different query types
    let queries = vec![
        ("Alice", "Should find Alice-related information"),
        ("Microsoft", "Should find Microsoft-related information"),
        ("works at", "Should find work relationships"),
        ("colleague", "Should find colleague relationships"),
        ("Seattle", "Should find location information"),
    ];

    for (query_text, description) in queries {
        let query = Query::new(query_text);
        let results = retriever.retrieve(&query).await.unwrap();

        println!("Query '{}': {} results", query_text, results.len());
        for (i, result) in results.iter().take(3).enumerate() {
            println!(
                "  {}. Score: {:.3}, Content: {}",
                i + 1,
                result.score,
                &result.node.get_text()[..std::cmp::min(50, result.node.get_text().len())]
            );
        }

        // Most queries should return some results given our mock data
        if query_text != "nonexistent" {
            assert!(!results.is_empty(), "{}", description);
        }
    }
}

// Integration test with real LLM (requires API key)
#[tokio::test]
#[ignore] // Ignored by default, run with --ignored to test with real API
async fn test_real_llm_property_graph_integration() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let llm_client = Siumai::builder()
        .openai()
        .api_key(&api_key)
        .model("gpt-4o-mini")
        .build()
        .await
        .unwrap();

    let graph_store = Arc::new(SimplePropertyGraphStore::new());

    let extraction_config = LlmExtractionConfig {
        format: ExtractionFormat::Json,
        max_triplets_per_chunk: 15,
        temperature: 0.1,
        enable_validation: true,
        enable_deduplication: true,
        show_progress: true,
        ..Default::default()
    };

    let llm_extractor = Arc::new(LlmExtractor::new(llm_client, extraction_config).unwrap());

    let index_config = PropertyGraphIndexConfig {
        enable_llm_extraction: true,
        show_progress: true,
        embed_kg_nodes: false,
        ..Default::default()
    };

    let mut index = PropertyGraphIndex::with_llm_extractor(
        graph_store.clone(),
        None,
        llm_extractor,
        Some(index_config),
    );

    let documents = vec![
        Document::new("Alice is a senior software engineer at Microsoft. She specializes in cloud computing and works on Azure services. Alice graduated from Stanford University with a degree in Computer Science."),
        Document::new("Bob Johnson is Alice's manager at Microsoft. He leads the Azure AI team and has been with the company for 8 years. Bob previously worked at Google as a principal engineer."),
        Document::new("Microsoft Corporation is a multinational technology company headquartered in Redmond, Washington. It was founded by Bill Gates and Paul Allen in 1975 and is known for Windows, Office, and Azure."),
        Document::new("Azure is Microsoft's cloud computing platform that provides services like virtual machines, databases, AI tools, and storage solutions. It competes with Amazon Web Services and Google Cloud Platform."),
    ];

    println!(
        "🔍 Starting LLM extraction for {} documents...",
        documents.len()
    );
    index.insert_documents(documents).await.unwrap();

    let stats = index.stats().await.unwrap();
    println!("📊 Extraction Results:");
    println!("  • Entities: {}", stats.graph_stats.entity_count);
    println!("  • Relations: {}", stats.graph_stats.relation_count);
    println!("  • Triplets: {}", stats.graph_stats.triplet_count);

    assert!(
        stats.graph_stats.entity_count >= 5,
        "Should extract at least 5 entities"
    );
    assert!(
        stats.graph_stats.relation_count >= 5,
        "Should extract at least 5 relations"
    );
    assert!(
        stats.graph_stats.triplet_count >= 10,
        "Should extract at least 10 triplets"
    );

    // Test retrieval with various queries
    let retriever = index.as_retriever();
    let test_queries = vec![
        "Where does Alice work?",
        "Who is Bob?",
        "What is Azure?",
        "Who founded Microsoft?",
        "What does Alice specialize in?",
    ];

    println!("\n🔎 Testing retrieval queries:");
    for query_text in test_queries {
        let query = Query::new(query_text);
        let results = retriever.retrieve(&query).await.unwrap();

        println!("\n❓ Query: {}", query_text);
        if results.is_empty() {
            println!("   No results found.");
        } else {
            println!("   Found {} results:", results.len());
            for (i, result) in results.iter().take(3).enumerate() {
                println!(
                    "   {}. Score: {:.3} - {}",
                    i + 1,
                    result.score,
                    &result.node.get_text()[..std::cmp::min(80, result.node.get_text().len())]
                );
            }
        }
    }

    // Display some extracted triplets
    println!("\n🕸️  Sample extracted triplets:");
    let all_triplets = graph_store
        .get_triplets(None, None, None, Some(15))
        .await
        .unwrap();
    for (i, triplet) in all_triplets.iter().take(15).enumerate() {
        println!(
            "   {}. ({}, {}, {})",
            i + 1,
            triplet.subject.name,
            triplet.relation.label,
            triplet.object.name
        );
    }

    println!("\n✅ Real LLM integration test completed successfully!");
}
