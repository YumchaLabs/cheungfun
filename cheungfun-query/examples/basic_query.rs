//! Basic query engine example.
//!
//! This example demonstrates how to set up and use the basic query engine
//! components for a simple RAG application with real siumai integration.
//!
//! To run this example:
//! ```bash
//! # With OpenAI (recommended)
//! export OPENAI_API_KEY="your-openai-api-key"
//! cargo run --example basic_query --features siumai
//!
//! # With Anthropic
//! export ANTHROPIC_API_KEY="your-anthropic-api-key"
//! cargo run --example basic_query --features siumai
//!
//! # With Ollama (local)
//! export OLLAMA_BASE_URL="http://localhost:11434"
//! cargo run --example basic_query --features siumai
//! ```
//!
//! # Environment Variables
//!
//! Set the following environment variables to use real LLM providers:
//! - `OPENAI_API_KEY`: For OpenAI models (default provider)
//! - `ANTHROPIC_API_KEY`: For Anthropic models
//! - `OLLAMA_BASE_URL`: For Ollama (defaults to http://localhost:11434)
//!
//! If no API keys are provided, the example will fail with helpful error messages.

use std::collections::HashMap;
use std::sync::Arc;

use cheungfun_core::{
    traits::{Embedder, VectorStore},
    types::{Node, Query, ScoredNode},
    Result,
};
use cheungfun_query::prelude::*;
use siumai::prelude::*;

/// Mock embedder for demonstration purposes.
#[derive(Debug)]
struct MockEmbedder;

#[async_trait::async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Simple mock embedding: convert text to vector based on length and characters
        let mut embedding = vec![0.0; 384]; // Common embedding dimension
        let bytes = text.as_bytes();
        for (i, &byte) in bytes.iter().enumerate() {
            if i < embedding.len() {
                embedding[i] = (byte as f32) / 255.0;
            }
        }
        Ok(embedding)
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.embed(text).await?);
        }
        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        384
    }

    fn model_name(&self) -> &str {
        "mock-embedder"
    }
}

/// Mock vector store for demonstration purposes.
#[derive(Debug)]
struct MockVectorStore {
    nodes: std::sync::RwLock<Vec<Node>>,
}

impl MockVectorStore {
    fn new() -> Self {
        Self {
            nodes: std::sync::RwLock::new(Vec::new()),
        }
    }

    fn add_sample_data(&self) -> Result<()> {
        let sample_nodes = vec![
            Node::builder()
                .content("Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.")
                .source_document_id(uuid::Uuid::new_v4())
                .chunk_info(cheungfun_core::types::ChunkInfo::with_char_indices(0, 100, 0))
                .metadata("source", "ml_basics.txt")
                .embedding(vec![0.1; 384])
                .build()
                .expect("Failed to build node"),
            Node::builder()
                .content("Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.")
                .source_document_id(uuid::Uuid::new_v4())
                .chunk_info(cheungfun_core::types::ChunkInfo::with_char_indices(0, 100, 0))
                .metadata("source", "deep_learning.txt")
                .embedding(vec![0.2; 384])
                .build()
                .expect("Failed to build node"),
        ];

        if let Ok(mut nodes) = self.nodes.write() {
            nodes.extend(sample_nodes);
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl VectorStore for MockVectorStore {
    async fn add(&self, nodes: Vec<Node>) -> Result<Vec<uuid::Uuid>> {
        let ids: Vec<uuid::Uuid> = nodes.iter().map(|n| n.id).collect();
        if let Ok(mut store_nodes) = self.nodes.write() {
            store_nodes.extend(nodes);
        }
        Ok(ids)
    }

    async fn update(&self, _nodes: Vec<Node>) -> Result<()> {
        // Mock implementation
        Ok(())
    }

    async fn delete(&self, _node_ids: Vec<uuid::Uuid>) -> Result<()> {
        // Mock implementation
        Ok(())
    }

    async fn search(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        if let Ok(nodes) = self.nodes.read() {
            // Simple mock search: return all nodes with mock scores
            let scored_nodes: Vec<ScoredNode> = nodes
                .iter()
                .take(query.top_k)
                .enumerate()
                .map(|(i, node)| ScoredNode {
                    node: node.clone(),
                    score: 1.0 - (i as f32 * 0.1), // Decreasing scores
                })
                .collect();
            Ok(scored_nodes)
        } else {
            Ok(Vec::new())
        }
    }

    async fn get(&self, node_ids: Vec<uuid::Uuid>) -> Result<Vec<Option<Node>>> {
        if let Ok(nodes) = self.nodes.read() {
            let mut result = Vec::new();
            for id in node_ids {
                let found = nodes.iter().find(|n| n.id == id).cloned();
                result.push(found);
            }
            Ok(result)
        } else {
            Ok(Vec::new())
        }
    }

    async fn health_check(&self) -> Result<()> {
        Ok(())
    }
}

/// Create a siumai client based on available environment variables.
async fn create_siumai_client() -> Result<Siumai> {
    // Try different providers based on available environment variables
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        println!("🔑 Using OpenAI with API key");
        Siumai::builder()
            .openai()
            .api_key(&api_key)
            .model("gpt-3.5-turbo")
            .temperature(0.7)
            .max_tokens(1000)
            .build()
            .await
            .map_err(|e| {
                cheungfun_core::CheungfunError::configuration(format!(
                    "Failed to create OpenAI client: {e}"
                ))
            })
    } else if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        println!("🔑 Using Anthropic with API key");
        Siumai::builder()
            .anthropic()
            .api_key(&api_key)
            .model("claude-3-haiku-20240307")
            .temperature(0.7)
            .max_tokens(1000)
            .build()
            .await
            .map_err(|e| {
                cheungfun_core::CheungfunError::configuration(format!(
                    "Failed to create Anthropic client: {e}"
                ))
            })
    } else {
        // Try Ollama as fallback
        let base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        println!("🦙 Using Ollama at {}", base_url);
        Siumai::builder()
            .ollama()
            .base_url(&base_url)
            .model("llama2")
            .temperature(0.7)
            .max_tokens(1000)
            .build()
            .await
            .map_err(|e| {
                cheungfun_core::CheungfunError::configuration(format!(
                    "Failed to create Ollama client: {e}"
                ))
            })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🚀 Starting basic query engine example...");

    // Create mock components
    let embedder = Arc::new(MockEmbedder);
    let vector_store = Arc::new(MockVectorStore::new());

    // Test siumai client creation to ensure it works
    match create_siumai_client().await {
        Ok(_) => {
            println!("✅ Successfully validated siumai client configuration");
        }
        Err(e) => {
            eprintln!("❌ Failed to create siumai client: {}", e);
            eprintln!(
                "💡 Make sure to set OPENAI_API_KEY, ANTHROPIC_API_KEY, or run Ollama locally"
            );
            return Err(e);
        }
    }

    // Add sample data to vector store
    vector_store.add_sample_data()?;
    println!("📚 Added sample data to vector store");

    // Create retriever
    let retriever = Arc::new(VectorRetriever::new(vector_store.clone(), embedder.clone()));
    println!("🔍 Created vector retriever");

    // Create generator using the factory
    let llm_config = if std::env::var("OPENAI_API_KEY").is_ok() {
        LlmConfig::openai("gpt-3.5-turbo", &std::env::var("OPENAI_API_KEY").unwrap())
    } else if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        LlmConfig::anthropic(
            "claude-3-haiku-20240307",
            &std::env::var("ANTHROPIC_API_KEY").unwrap(),
        )
    } else {
        let mut config = LlmConfig::new("ollama", "llama2");
        config.base_url = Some(
            std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
        );
        config
    };

    let factory = SiumaiLlmFactory::new();
    let generator = factory.create_llm(&llm_config).await?;
    println!("🤖 Created response generator using factory");

    // Create query engine
    let query_engine = QueryEngine::new(retriever, generator);
    println!("⚙️ Created query engine");

    // Execute a query
    let query = "What is machine learning?";
    println!("\n❓ Query: {}", query);

    let response = query_engine.query(query).await?;

    println!("\n✅ Response:");
    println!("Content: {}", response.response.content);
    println!("Source nodes: {}", response.retrieved_nodes.len());

    if let Some(usage) = &response.response.usage {
        println!(
            "Token usage: {} prompt + {} completion = {} total",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }

    for (i, scored_node) in response.retrieved_nodes.iter().enumerate() {
        println!(
            "  {}. Score: {:.3} - {}",
            i + 1,
            scored_node.score,
            &scored_node.node.content[..100.min(scored_node.node.content.len())]
        );
    }

    println!("\n🎉 Example completed successfully!");
    Ok(())
}
