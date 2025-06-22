//! Basic query engine example.
//!
//! This example demonstrates how to set up and use the basic query engine
//! components for a simple RAG application.

use std::collections::HashMap;
use std::sync::Arc;

use cheungfun_core::{
    types::{Node, Query, ScoredNode},
    traits::{Embedder, VectorStore},
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
            Node {
                id: uuid::Uuid::new_v4(),
                content: "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.".to_string(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("source".to_string(), serde_json::Value::String("ml_basics.txt".to_string()));
                    meta
                },
                embedding: Some(vec![0.1; 384]),
                sparse_embedding: None,
                relationships: HashMap::new(),
                source_document_id: uuid::Uuid::new_v4(),
                chunk_info: cheungfun_core::types::ChunkInfo {
                    start_offset: 0,
                    end_offset: 100,
                    chunk_index: 0,
                },
            },
            Node {
                id: uuid::Uuid::new_v4(),
                content: "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.".to_string(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("source".to_string(), serde_json::Value::String("deep_learning.txt".to_string()));
                    meta
                },
                embedding: Some(vec![0.2; 384]),
                sparse_embedding: None,
                relationships: HashMap::new(),
                source_document_id: uuid::Uuid::new_v4(),
                chunk_info: cheungfun_core::types::ChunkInfo {
                    start_offset: 0,
                    end_offset: 100,
                    chunk_index: 0,
                },
            },
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

/// Mock LLM client for demonstration.
#[derive(Debug)]
struct MockSiumai;

impl MockSiumai {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl siumai::traits::ChatCompletion for MockSiumai {
    async fn chat(&self, _messages: Vec<ChatMessage>) -> Result<ChatResponse, siumai::Error> {
        Ok(ChatResponse {
            id: "mock-response".to_string(),
            content: MessageContent::Text("This is a mock response based on the provided context. Machine learning is indeed a powerful technology for learning from data.".to_string()),
            model: Some("mock-model".to_string()),
            usage: Some(Usage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
            }),
            finish_reason: Some(FinishReason::Stop),
            created: chrono::Utc::now(),
            system_fingerprint: None,
        })
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _options: Option<StreamOptions>,
    ) -> Result<impl futures::Stream<Item = Result<ChatStreamEvent, siumai::Error>> + Send, siumai::Error> {
        // Return empty stream for mock
        Ok(futures::stream::empty())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::init();

    println!("🚀 Starting basic query engine example...");

    // Create mock components
    let embedder = Arc::new(MockEmbedder);
    let vector_store = Arc::new(MockVectorStore::new());
    let siumai_client = Siumai::new(Box::new(MockSiumai::new()));

    // Add sample data to vector store
    vector_store.add_sample_data()?;
    println!("📚 Added sample data to vector store");

    // Create retriever
    let retriever = Arc::new(VectorRetriever::new(vector_store.clone(), embedder.clone()));
    println!("🔍 Created vector retriever");

    // Create generator
    let generator = Arc::new(SiumaiGenerator::new(siumai_client));
    println!("🤖 Created response generator");

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
    
    for (i, scored_node) in response.retrieved_nodes.iter().enumerate() {
        println!("  {}. Score: {:.3} - {}", 
                 i + 1, 
                 scored_node.score, 
                 &scored_node.node.content[..100.min(scored_node.node.content.len())]);
    }

    println!("\n🎉 Example completed successfully!");
    Ok(())
}
