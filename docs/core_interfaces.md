# Cheungfun Core Interfaces Design

## Overview

This document defines the core interfaces for the Cheungfun RAG framework, inspired by LlamaIndex and Swiftide architectures. The design emphasizes type safety, async operations, and modular extensibility.

## Core Data Structures

### Document
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Represents a raw document from data sources
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Document {
    /// Unique identifier for the document
    pub id: Uuid,
    /// Raw content of the document
    pub content: String,
    /// Document metadata (source, creation time, etc.)
    pub metadata: HashMap<String, serde_json::Value>,
    /// Optional pre-computed embedding
    pub embedding: Option<Vec<f32>>,
}
```

### Node
```rust
/// Enhanced Node structure based on LlamaIndex BaseNode
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Node {
    /// Unique identifier for the node
    pub id: Uuid,
    /// Text content of the chunk
    pub content: String,
    /// Node metadata (chunk info, extracted metadata, etc.)
    pub metadata: HashMap<String, serde_json::Value>,
    /// Dense vector embedding
    pub embedding: Option<Vec<f32>>,
    /// Sparse vector embedding (for hybrid search)
    pub sparse_embedding: Option<HashMap<u32, f32>>,
    /// Structured relationships to other nodes
    pub relationships: NodeRelationships,
    /// Original document reference
    pub source_document_id: Uuid,
    /// Chunk position info
    pub chunk_info: ChunkInfo,
    /// Content hash for deduplication and caching
    pub hash: Option<String>,
    /// MIME type of the node content
    pub mimetype: String,
    /// Metadata keys to exclude from embeddings
    pub excluded_embed_metadata_keys: HashSet<String>,
    /// Metadata keys to exclude from LLM
    pub excluded_llm_metadata_keys: HashSet<String>,
    /// Template for formatting content with metadata
    pub text_template: String,
    /// Separator for metadata entries
    pub metadata_separator: String,
    /// Template for individual metadata entries
    pub metadata_template: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkInfo {
    /// Start position in original document
    pub start_offset: usize,
    /// End position in original document  
    pub end_offset: usize,
    /// Chunk index in document
    pub chunk_index: usize,
}
```

### Query
```rust
/// Represents a search query with various parameters
#[derive(Debug, Clone)]
pub struct Query {
    /// Query text
    pub text: String,
    /// Pre-computed query embedding
    pub embedding: Option<Vec<f32>>,
    /// Metadata filters
    pub filters: HashMap<String, serde_json::Value>,
    /// Number of results to return
    pub top_k: usize,
    /// Similarity threshold
    pub similarity_threshold: Option<f32>,
    /// Search mode (vector, keyword, hybrid)
    pub search_mode: SearchMode,
}

#[derive(Debug, Clone)]
pub enum SearchMode {
    Vector,
    Keyword,
    Hybrid { alpha: f32 }, // alpha balances vector vs keyword
}
```

## Core Traits

### Data Loading
```rust
use async_trait::async_trait;
use anyhow::Result;

/// Loads documents from various data sources
#[async_trait]
pub trait Loader: Send + Sync + std::fmt::Debug {
    /// Load documents from the data source
    async fn load(&self) -> Result<Vec<Document>>;
    
    /// Get loader name for logging/debugging
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Loads documents incrementally/streaming
#[async_trait]
pub trait StreamingLoader: Send + Sync + std::fmt::Debug {
    /// Create a stream of documents
    fn into_stream(self) -> impl futures::Stream<Item = Result<Document>> + Send;
    
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}
```

### Document Processing

现在所有的文档和节点处理都使用统一的 `Transform` 接口：

```rust
/// Unified transformation trait for all processing components.
#[async_trait]
pub trait Transform: Send + Sync + std::fmt::Debug {
    /// Transform input into nodes.
    async fn transform(&self, input: TransformInput) -> Result<Vec<Node>>;

    /// Transform multiple inputs in batch for better performance.
    async fn transform_batch(&self, inputs: Vec<TransformInput>) -> Result<Vec<Node>> {
        let mut all_nodes = Vec::new();
        for input in inputs {
            let nodes = self.transform(input).await?;
            all_nodes.extend(nodes);
        }
        Ok(all_nodes)
    }

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}
```

### Embedding Generation
```rust
/// Generates embeddings for text content
#[async_trait]
pub trait Embedder: Send + Sync + std::fmt::Debug {
    /// Embed a single text
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    
    /// Embed multiple texts in batch (more efficient)
    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>>;
    
    /// Get embedding dimension
    fn dimension(&self) -> usize;
    
    /// Get model name/identifier
    fn model_name(&self) -> &str;
    
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Generates sparse embeddings (for keyword/hybrid search)
#[async_trait]
pub trait SparseEmbedder: Send + Sync + std::fmt::Debug {
    /// Generate sparse embedding for text
    async fn embed_sparse(&self, text: &str) -> Result<HashMap<u32, f32>>;
    
    /// Generate sparse embeddings for multiple texts
    async fn embed_sparse_batch(&self, texts: Vec<&str>) -> Result<Vec<HashMap<u32, f32>>>;
    
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}
```

### Vector Storage
```rust
/// Stores and retrieves vector embeddings
#[async_trait]
pub trait VectorStore: Send + Sync + std::fmt::Debug {
    /// Add nodes to the store
    async fn add(&self, nodes: Vec<Node>) -> Result<Vec<Uuid>>;
    
    /// Update existing nodes
    async fn update(&self, nodes: Vec<Node>) -> Result<()>;
    
    /// Delete nodes by IDs
    async fn delete(&self, node_ids: Vec<Uuid>) -> Result<()>;
    
    /// Search for similar nodes
    async fn search(&self, query: &Query) -> Result<Vec<ScoredNode>>;
    
    /// Get nodes by IDs
    async fn get(&self, node_ids: Vec<Uuid>) -> Result<Vec<Option<Node>>>;
    
    /// Check if store is healthy/connected
    async fn health_check(&self) -> Result<()>;
    
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

#[derive(Debug, Clone)]
pub struct ScoredNode {
    pub node: Node,
    pub score: f32,
}
```

### Retrieval
```rust
/// Retrieves relevant nodes for a query
#[async_trait]
pub trait Retriever: Send + Sync + std::fmt::Debug {
    /// Retrieve nodes for a query
    async fn retrieve(&self, query: &Query) -> Result<Vec<ScoredNode>>;

    /// Retrieve with additional context/filters
    async fn retrieve_with_context(
        &self,
        query: &Query,
        context: &RetrievalContext
    ) -> Result<Vec<ScoredNode>> {
        // Default implementation ignores context
        self.retrieve(query).await
    }

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

#[derive(Debug, Clone, Default)]
pub struct RetrievalContext {
    /// Previous conversation history
    pub chat_history: Vec<ChatMessage>,
    /// User context/preferences
    pub user_context: HashMap<String, serde_json::Value>,
    /// Session information
    pub session_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}
```

### LLM Integration
```rust
/// Response generation using LLMs
#[async_trait]
pub trait ResponseGenerator: Send + Sync + std::fmt::Debug {
    /// Generate response from retrieved context
    async fn generate_response(
        &self,
        query: &str,
        context_nodes: Vec<ScoredNode>,
        options: &GenerationOptions,
    ) -> Result<GeneratedResponse>;

    /// Generate streaming response
    async fn generate_response_stream(
        &self,
        query: &str,
        context_nodes: Vec<ScoredNode>,
        options: &GenerationOptions,
    ) -> Result<impl futures::Stream<Item = Result<String>> + Send>;

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

#[derive(Debug, Clone)]
pub struct GenerationOptions {
    /// Maximum tokens to generate
    pub max_tokens: Option<usize>,
    /// Temperature for generation
    pub temperature: Option<f32>,
    /// System prompt override
    pub system_prompt: Option<String>,
    /// Include source citations
    pub include_citations: bool,
    /// Response format
    pub format: ResponseFormat,
}

#[derive(Debug, Clone)]
pub enum ResponseFormat {
    Text,
    Markdown,
    Json,
}

#[derive(Debug, Clone)]
pub struct GeneratedResponse {
    /// Generated text content
    pub content: String,
    /// Source nodes used for generation
    pub source_nodes: Vec<Uuid>,
    /// Generation metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Token usage information
    pub usage: Option<TokenUsage>,
}

#[derive(Debug, Clone)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}
```

### Pipeline Interfaces
```rust
/// Indexing pipeline for processing documents
#[async_trait]
pub trait IndexingPipeline: Send + Sync {
    /// Run the complete indexing pipeline
    async fn run(&self) -> Result<IndexingStats>;

    /// Run pipeline with progress reporting
    async fn run_with_progress(
        &self,
        progress_callback: impl Fn(IndexingProgress) + Send + Sync,
    ) -> Result<IndexingStats>;

    /// Validate pipeline configuration
    fn validate(&self) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct IndexingStats {
    pub documents_processed: usize,
    pub nodes_created: usize,
    pub nodes_stored: usize,
    pub processing_time: std::time::Duration,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IndexingProgress {
    pub stage: String,
    pub processed: usize,
    pub total: Option<usize>,
    pub current_item: Option<String>,
}

/// Query pipeline for answering questions
#[async_trait]
pub trait QueryPipeline: Send + Sync {
    /// Execute a query and return response
    async fn query(&self, query: &str, options: &QueryOptions) -> Result<QueryResponse>;

    /// Execute query with streaming response
    async fn query_stream(
        &self,
        query: &str,
        options: &QueryOptions,
    ) -> Result<impl futures::Stream<Item = Result<String>> + Send>;
}

#[derive(Debug, Clone, Default)]
pub struct QueryOptions {
    pub retrieval_options: Query,
    pub generation_options: GenerationOptions,
    pub context: RetrievalContext,
}

#[derive(Debug, Clone)]
pub struct QueryResponse {
    pub response: GeneratedResponse,
    pub retrieved_nodes: Vec<ScoredNode>,
    pub query_metadata: HashMap<String, serde_json::Value>,
}
```

## Error Handling

```rust
use thiserror::Error;

/// Core error types for Cheungfun
#[derive(Error, Debug)]
pub enum CheungfunError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Embedding error: {message}")]
    Embedding { message: String },

    #[error("Vector store error: {message}")]
    VectorStore { message: String },

    #[error("LLM error: {message}")]
    Llm { message: String },

    #[error("Pipeline error: {message}")]
    Pipeline { message: String },

    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("Validation error: {message}")]
    Validation { message: String },

    #[error("Not found: {resource}")]
    NotFound { resource: String },

    #[error("Timeout: {operation}")]
    Timeout { operation: String },

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Authentication failed")]
    Authentication,

    #[error("Permission denied")]
    Permission,

    #[error("Internal error: {message}")]
    Internal { message: String },
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, CheungfunError>;
```

## Configuration Interfaces

```rust
/// Configuration for embedders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderConfig {
    pub model_name: String,
    pub dimension: usize,
    pub batch_size: Option<usize>,
    pub max_retries: Option<usize>,
    pub timeout_seconds: Option<u64>,
    pub provider_config: HashMap<String, serde_json::Value>,
}

/// Configuration for vector stores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    pub store_type: String,
    pub connection_string: String,
    pub collection_name: String,
    pub dimension: usize,
    pub distance_metric: DistanceMetric,
    pub index_config: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

/// Configuration for LLM providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String,
    pub model_name: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub timeout_seconds: Option<u64>,
    pub provider_config: HashMap<String, serde_json::Value>,
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub name: String,
    pub description: Option<String>,
    pub concurrency: Option<usize>,
    pub batch_size: Option<usize>,
    pub retry_config: RetryConfig,
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: usize,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub enable_tracing: bool,
    pub log_level: String,
    pub metrics_endpoint: Option<String>,
}
```

## Factory Interfaces

```rust
/// Factory for creating embedders
#[async_trait]
pub trait EmbedderFactory: Send + Sync {
    /// Create an embedder from configuration
    async fn create_embedder(&self, config: &EmbedderConfig) -> Result<Box<dyn Embedder>>;

    /// List supported embedder types
    fn supported_types(&self) -> Vec<&'static str>;

    /// Validate configuration
    fn validate_config(&self, config: &EmbedderConfig) -> Result<()>;
}

/// Factory for creating vector stores
#[async_trait]
pub trait VectorStoreFactory: Send + Sync {
    /// Create a vector store from configuration
    async fn create_vector_store(&self, config: &VectorStoreConfig) -> Result<Box<dyn VectorStore>>;

    /// List supported store types
    fn supported_types(&self) -> Vec<&'static str>;

    /// Validate configuration
    fn validate_config(&self, config: &VectorStoreConfig) -> Result<()>;
}

/// Factory for creating LLM clients
#[async_trait]
pub trait LlmFactory: Send + Sync {
    /// Create an LLM client from configuration
    async fn create_llm(&self, config: &LlmConfig) -> Result<Box<dyn ResponseGenerator>>;

    /// List supported providers
    fn supported_providers(&self) -> Vec<&'static str>;

    /// Validate configuration
    fn validate_config(&self, config: &LlmConfig) -> Result<()>;
}
```

## Builder Patterns

```rust
/// Builder for indexing pipelines
pub struct IndexingPipelineBuilder {
    loader: Option<Arc<dyn Loader>>,
    transforms: Vec<Arc<dyn Transform>>,
    embedder: Option<Arc<dyn Embedder>>,
    vector_store: Option<Arc<dyn VectorStore>>,
    config: PipelineConfig,
}

impl IndexingPipelineBuilder {
    pub fn new() -> Self {
        Self {
            loader: None,
            transforms: Vec::new(),
            embedder: None,
            vector_store: None,
            config: PipelineConfig::default(),
        }
    }

    // 支持直接传入组件（会自动包装为Arc）
    pub fn with_loader(mut self, loader: impl Loader + 'static) -> Self {
        self.loader = Some(Arc::new(loader));
        self
    }

    // 支持传入Arc包装的组件（用于共享）
    pub fn with_loader_arc(mut self, loader: Arc<dyn Loader>) -> Self {
        self.loader = Some(loader);
        self
    }

    pub fn with_transform(mut self, transform: impl Transform + 'static) -> Self {
        self.transforms.push(Arc::new(transform));
        self
    }

    pub fn with_transform_arc(mut self, transform: Arc<dyn Transform>) -> Self {
        self.transforms.push(transform);
        self
    }

    pub fn with_embedder(mut self, embedder: impl Embedder + 'static) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    pub fn with_embedder_arc(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    pub fn with_vector_store(mut self, store: impl VectorStore + 'static) -> Self {
        self.vector_store = Some(Arc::new(store));
        self
    }

    pub fn with_vector_store_arc(mut self, store: Arc<dyn VectorStore>) -> Self {
        self.vector_store = Some(store);
        self
    }

    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn build(self) -> Result<Box<dyn IndexingPipeline>> {
        // Validation logic here
        if self.loader.is_none() {
            return Err(CheungfunError::Configuration {
                message: "Loader is required".to_string(),
            });
        }

        if self.embedder.is_none() {
            return Err(CheungfunError::Configuration {
                message: "Embedder is required".to_string(),
            });
        }

        if self.vector_store.is_none() {
            return Err(CheungfunError::Configuration {
                message: "Vector store is required".to_string(),
            });
        }

        // Create concrete pipeline implementation
        todo!("Implement concrete pipeline")
    }
}

/// Builder for query pipelines
pub struct QueryPipelineBuilder {
    retriever: Option<Box<dyn Retriever>>,
    response_generator: Option<Box<dyn ResponseGenerator>>,
    config: PipelineConfig,
}

impl QueryPipelineBuilder {
    pub fn new() -> Self {
        Self {
            retriever: None,
            response_generator: None,
            config: PipelineConfig::default(),
        }
    }

    pub fn with_retriever(mut self, retriever: Box<dyn Retriever>) -> Self {
        self.retriever = Some(retriever);
        self
    }

    pub fn with_response_generator(mut self, generator: Box<dyn ResponseGenerator>) -> Self {
        self.response_generator = Some(generator);
        self
    }

    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn build(self) -> Result<Box<dyn QueryPipeline>> {
        // Validation and construction logic
        todo!("Implement concrete query pipeline")
    }
}
```

## Usage Examples

```rust
// Example 1: Simple usage (components are automatically wrapped in Arc)
let pipeline = IndexingPipelineBuilder::new()
    .with_loader(FileLoader::new("./docs"))
    .with_transformer(TextSplitter::new(1000, 200))
    .with_node_transformer(MetadataExtractor::new())
    .with_embedder(CandleEmbedder::new("sentence-transformers/all-MiniLM-L6-v2"))
    .with_vector_store(QdrantStore::new("localhost:6334", "my_collection"))
    .build()?;

// Run the pipeline
let stats = pipeline.run().await?;
println!("Processed {} documents, created {} nodes",
         stats.documents_processed, stats.nodes_created);

// Example 2: Shared components (explicit Arc usage)
let embedder = Arc::new(CandleEmbedder::new("sentence-transformers/all-MiniLM-L6-v2"));
let vector_store = Arc::new(QdrantStore::new("localhost:6334", "my_collection"));

// Use shared embedder in indexing
let indexing_pipeline = IndexingPipelineBuilder::new()
    .with_loader(FileLoader::new("./docs"))
    .with_transformer(TextSplitter::new(1000, 200))
    .with_embedder_arc(embedder.clone())  // Share embedder
    .with_vector_store_arc(vector_store.clone())  // Share store
    .build()?;

// Use same embedder and store in query pipeline
let query_pipeline = QueryPipelineBuilder::new()
    .with_retriever_arc(Arc::new(VectorRetriever::new(vector_store.clone(), embedder.clone())))
    .with_response_generator(SiumaiGenerator::new(llm_config))
    .build()?;

// Execute a query
let response = query_pipeline.query(
    "What is the main topic of the documents?",
    &QueryOptions::default()
).await?;

println!("Answer: {}", response.response.content);
```

## Implementation Guidelines

### Trait Object Safety
All core traits are designed to be object-safe, allowing for dynamic dispatch:

```rust
// This should work
let embedder: Box<dyn Embedder> = Box::new(CandleEmbedder::new(...));
let store: Box<dyn VectorStore> = Box::new(QdrantStore::new(...));
```

### Error Handling Best Practices
1. Use `CheungfunError` for all public APIs
2. Provide detailed error messages with context
3. Use `anyhow::Error` internally for convenience
4. Implement proper error conversion with `From` traits

### Async Considerations
1. All I/O operations should be async
2. Use `tokio` as the async runtime
3. Implement proper cancellation support
4. Use `futures::Stream` for streaming operations

### Performance Guidelines
1. Implement batch operations where possible
2. Use connection pooling for external services
3. Implement proper caching strategies
4. Consider memory usage for large documents

### Testing Strategy
1. All traits should be mockable (use `mockall` crate)
2. Provide test utilities for common scenarios
3. Include integration tests with real services
4. Benchmark critical paths

## Module Organization

```
cheungfun-core/
├── src/
│   ├── lib.rs                 # Public API exports
│   ├── error.rs               # Error types
│   ├── types/
│   │   ├── mod.rs
│   │   ├── document.rs        # Document struct
│   │   ├── node.rs           # Node struct
│   │   ├── query.rs          # Query types
│   │   └── response.rs       # Response types
│   ├── traits/
│   │   ├── mod.rs
│   │   ├── loader.rs         # Loader traits
│   │   ├── transformer.rs    # Transformer traits
│   │   ├── embedder.rs       # Embedder traits
│   │   ├── storage.rs        # Storage traits
│   │   ├── retriever.rs      # Retriever traits
│   │   ├── generator.rs      # Response generator traits
│   │   └── pipeline.rs       # Pipeline traits
│   ├── config/
│   │   ├── mod.rs
│   │   ├── embedder.rs       # Embedder configs
│   │   ├── storage.rs        # Storage configs
│   │   ├── llm.rs           # LLM configs
│   │   └── pipeline.rs       # Pipeline configs
│   ├── factory/
│   │   ├── mod.rs
│   │   ├── embedder.rs       # Embedder factory
│   │   ├── storage.rs        # Storage factory
│   │   └── llm.rs           # LLM factory
│   └── builder/
│       ├── mod.rs
│       ├── indexing.rs       # Indexing pipeline builder
│       └── query.rs          # Query pipeline builder
└── tests/
    ├── integration/
    └── unit/
```

## Next Steps

1. **Implement cheungfun-core**: Start with basic data structures and traits
2. **Create concrete implementations**: Begin with simple in-memory implementations
3. **Add Candle integration**: Implement embedders using Candle
4. **Integrate siumai**: Add LLM response generation
5. **Build storage adapters**: Implement vector store integrations
6. **Create pipeline implementations**: Build the actual pipeline logic
7. **Add comprehensive tests**: Ensure reliability and correctness
8. **Performance optimization**: Profile and optimize critical paths
9. **Documentation**: Add detailed API documentation and examples
10. **Integration examples**: Create real-world usage examples
