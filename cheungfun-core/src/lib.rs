//! # Cheungfun Core
//!
//! Core traits, types, and interfaces for the Cheungfun RAG (Retrieval-Augmented Generation) framework.
//!
//! This crate provides the foundational building blocks for building RAG applications in Rust,
//! including:
//!
//! - **Data structures**: Document, Node, Query, and Response types
//! - **Core traits**: Loader, Transformer, Embedder, `VectorStore`, Retriever, `ResponseGenerator`
//! - **Pipeline interfaces**: `IndexingPipeline` and `QueryPipeline`
//! - **Configuration**: Type-safe configuration structures
//! - **Error handling**: Comprehensive error types with context
//! - **Builder patterns**: Fluent APIs for constructing pipelines
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use cheungfun_core::prelude::*;
//!
//! // Define a simple document
//! let doc = Document::builder()
//!     .content("This is a sample document")
//!     .metadata("source", "example.txt")
//!     .build();
//! ```
//!
//! ## Architecture
//!
//! The core architecture follows a modular design where each component implements
//! well-defined traits, allowing for easy composition and testing:
//!
//! - **Loaders** read documents from various sources
//! - **Transformers** process documents into searchable nodes
//! - **Embedders** generate vector representations
//! - **Vector Stores** persist and search embeddings
//! - **Retrievers** find relevant content for queries
//! - **Response Generators** create final answers using LLMs

#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

// Re-export commonly used types and traits
pub mod prelude;

// Core modules
pub mod builder;
pub mod cache;
pub mod config;
pub mod error;
pub mod factory;
pub mod traits;
pub mod types;

// Re-export key types at crate root for convenience
pub use error::{CheungfunError, Result};
pub use types::{
    ChatMessage, ChunkInfo, Document, GeneratedResponse, GenerationOptions, MessageRole, Node,
    Query, QueryResponse, ResponseFormat, RetrievalContext, ScoredNode, TokenUsage,
};

// Re-export traits for convenience
pub use traits::*;

/// Version information for the Cheungfun core library.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Name of the Cheungfun core library.
pub const NAME: &str = env!("CARGO_PKG_NAME");
