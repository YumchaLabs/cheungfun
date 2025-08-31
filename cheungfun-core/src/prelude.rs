//! Prelude module for convenient imports.
//!
//! This module re-exports the most commonly used types and traits
//! from the Cheungfun core library for easy importing.
//!
//! # Examples
//!
//! ```rust
//! use cheungfun_core::prelude::*;
//!
//! // Now you can use all the common types and traits
//! let doc = Document::new("Hello, world!");
//! let query = Query::new("What is this about?");
//! ```

// Re-export core error types
pub use crate::error::{CheungfunError, Result};

// Re-export all data types
pub use crate::types::{
    ChatMessage,
    ChunkInfo,
    // Document types
    Document,
    DocumentBuilder,

    GeneratedResponse,
    GenerationOptions,
    MessageRole,
    // Node types
    Node,
    NodeBuilder,
    // Query types
    Query,
    QueryBuilder,
    // Response types
    QueryResponse,
    ResponseFormat,
    RetrievalContext,
    ScoredNode,

    SearchMode,

    TokenUsage,
};

// Re-export core traits
pub use crate::traits::{
    DistanceMetric, Embedder, EmbeddingConfig, EmbeddingStats, EnsembleRetriever, GenerationCost,
    GenerationStats, GeneratorConfig, IndexConfig, IndexInfo, IndexingPipeline, IndexingProgress,
    IndexingStats, Loader, LoaderConfig, LoadingStats, ModelFeature, ModelInfo, PipelineStatus,
    QueryOptions, QueryPipeline, QueryStats, ResponseGenerator, RetrievalConfig,
    RetrievalExplanation, RetrievalStats, RetrievalStep, Retriever, SparseEmbedder, StorageStats,
    StreamingLoader, TransformConfig, TransformStats, VectorStore,
};

// Re-export configuration types
pub use crate::config::{
    EmbedderConfig, IndexingPipelineConfig, LlmConfig, QueryPipelineConfig, VectorStoreConfig,
};

// Re-export factory types
pub use crate::factory::{
    EmbedderFactory, EmbedderFactoryRegistry, LlmFactory, LlmFactoryRegistry, VectorStoreFactory,
    VectorStoreFactoryRegistry,
};

// Re-export builder types
pub use crate::builder::{
    BuilderInfo, IndexingPipelineBuilder, QueryBuilderInfo, QueryPipelineBuilder,
};

// Re-export deduplication types
pub use crate::deduplication::{DocstoreStrategy, DocumentDeduplicator, DocumentHasher};
