//! Configuration utilities for examples.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Example configuration for RAG systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleConfig {
    /// Embedding configuration
    pub embedding: EmbeddingConfig,
    
    /// Vector store configuration
    pub vector_store: VectorStoreConfig,
    
    /// LLM configuration
    pub llm: LlmConfig,
    
    /// Chunking configuration
    pub chunking: ChunkingConfig,
    
    /// Retrieval configuration
    pub retrieval: RetrievalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding provider (fastembed, openai, candle)
    pub provider: String,
    
    /// Model name
    pub model: String,
    
    /// Embedding dimension
    pub dimension: usize,
    
    /// Additional parameters
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    /// Vector store type (memory, qdrant)
    pub store_type: String,
    
    /// Connection parameters
    pub connection: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// LLM provider (openai, anthropic, groq)
    pub provider: String,
    
    /// Model name
    pub model: String,
    
    /// Temperature
    pub temperature: f32,
    
    /// Max tokens
    pub max_tokens: Option<u32>,
    
    /// Additional parameters
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Chunk size
    pub chunk_size: usize,
    
    /// Chunk overlap
    pub chunk_overlap: usize,
    
    /// Chunking strategy
    pub strategy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Number of documents to retrieve
    pub top_k: usize,
    
    /// Search mode (vector, hybrid, bm25)
    pub search_mode: String,
    
    /// Reranking enabled
    pub rerank: bool,
    
    /// Reranking top k
    pub rerank_top_k: Option<usize>,
}

impl Default for ExampleConfig {
    fn default() -> Self {
        Self {
            embedding: EmbeddingConfig {
                provider: "fastembed".to_string(),
                model: "BAAI/bge-small-en-v1.5".to_string(),
                dimension: 384,
                params: HashMap::new(),
            },
            vector_store: VectorStoreConfig {
                store_type: "memory".to_string(),
                connection: HashMap::new(),
            },
            llm: LlmConfig {
                provider: "openai".to_string(),
                model: "gpt-4o".to_string(),
                temperature: 0.0,
                max_tokens: Some(4000),
                params: HashMap::new(),
            },
            chunking: ChunkingConfig {
                chunk_size: 1000,
                chunk_overlap: 200,
                strategy: "recursive".to_string(),
            },
            retrieval: RetrievalConfig {
                top_k: 5,
                search_mode: "vector".to_string(),
                rerank: false,
                rerank_top_k: None,
            },
        }
    }
}

impl ExampleConfig {
    /// Load configuration from file
    pub fn from_file(path: &str) -> crate::ExampleResult<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn to_file(&self, path: &str) -> crate::ExampleResult<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Create configuration for OpenAI-based examples
    pub fn openai_config() -> Self {
        let mut config = Self::default();
        config.embedding.provider = "openai".to_string();
        config.embedding.model = "text-embedding-ada-002".to_string();
        config.embedding.dimension = 1536;
        config
    }
    
    /// Create configuration for local FastEmbed examples
    pub fn fastembed_config() -> Self {
        Self::default()
    }
    
    /// Create configuration for Qdrant vector store
    pub fn with_qdrant(mut self, url: &str) -> Self {
        self.vector_store.store_type = "qdrant".to_string();
        self.vector_store.connection.insert(
            "url".to_string(),
            serde_json::Value::String(url.to_string()),
        );
        self
    }
}
