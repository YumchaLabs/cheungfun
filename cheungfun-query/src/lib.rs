//! Query processing and retrieval for the Cheungfun RAG framework.
//!
//! This crate provides components for processing queries, retrieving relevant
//! content, and generating responses using LLMs. It includes:
//!
//! - **Retrievers**: Components for finding relevant nodes based on queries
//! - **Generators**: LLM-based response generation with streaming support
//! - **Query Engines**: High-level interfaces combining retrieval and generation
//! - **Query Pipelines**: Complete query processing workflows
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use cheungfun_query::prelude::*;
//! use cheungfun_core::prelude::*;
//!
//! # async fn example() -> Result<()> {
//! // Create a query engine
//! let query_engine = QueryEngine::builder()
//!     .retriever(vector_retriever)
//!     .generator(siumai_generator)
//!     .build()?;
//!
//! // Execute a query
//! let response = query_engine.query("What is machine learning?").await?;
//! println!("Answer: {}", response.content);
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! The query module follows a modular architecture:
//!
//! ```text
//! Query → Retriever → Vector Store
//!   ↓
//! Generator → LLM → Response
//! ```
//!
//! # Features
//!
//! - **Multiple Search Modes**: Vector, keyword, and hybrid search
//! - **Streaming Responses**: Real-time response generation
//! - **Context Management**: Conversation history and user context
//! - **Flexible Configuration**: Customizable retrieval and generation parameters
//! - **Error Handling**: Comprehensive error types and recovery strategies

#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

/// Advanced query processing capabilities including transformers, rerankers, and fusion algorithms.
pub mod advanced;
pub mod engine;
pub mod factory;
pub mod generator;
pub mod indices;
pub mod memory;
pub mod pipeline;
pub mod retriever;
pub mod retrievers;
pub mod utils;

/// Re-export commonly used types and traits.
pub mod prelude {
    pub use crate::engine::{
        QueryEngine, QueryEngineBuilder, QueryEngineConfig, QueryEngineOptions,
        QueryRewriteStrategy,
    };
    pub use crate::factory::SiumaiLlmFactory;
    pub use crate::generator::{SiumaiGenerator, SiumaiGeneratorBuilder, SiumaiGeneratorConfig};
    pub use crate::indices::{PropertyGraphIndex, PropertyGraphIndexConfig, PropertyGraphIndexStats};
    pub use crate::memory::{
        ApproximateTokenCounter, BaseMemory, ChatMemoryBuffer, ChatMemoryConfig, MemoryStats,
        TokenCounter,
    };
    pub use crate::pipeline::{
        DefaultQueryPipeline, QueryOptions, QueryPipelineBuilder, QueryPipelineConfig,
        RetrievalOptions,
    };
    pub use crate::retriever::{VectorRetriever, VectorRetrieverBuilder, VectorRetrieverConfig};
    pub use crate::retrievers::{GraphRetriever, GraphRetrievalConfig, GraphRetrievalStrategy};
    pub use crate::utils::{
        CacheStats, QueryCache, QueryOptimizer, QueryOptimizerConfig, ResponsePostProcessor,
        ResponsePostProcessorConfig,
    };

    // Re-export core types
    pub use cheungfun_core::prelude::*;
}
