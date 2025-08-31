//! Core traits for the Cheungfun framework.
//!
//! This module defines the fundamental traits that components must implement
//! to participate in the RAG pipeline. These traits provide a consistent
//! interface for different implementations while maintaining type safety
//! and async support.

pub mod cache;
pub mod embedder;
pub mod generator;
pub mod keyword_store;
pub mod kvstore;
pub mod loader;
pub mod memory;
pub mod pipeline;
pub mod retriever;
pub mod storage;
pub mod transformer;
pub mod typed_pipeline;

// Re-export all traits for convenience
pub use cache::*;
pub use embedder::*;
pub use generator::*;
pub use keyword_store::*;
pub use kvstore::*;
pub use loader::*;
pub use memory::*;
pub use pipeline::*;
pub use retriever::*;
pub use storage::*;
pub use transformer::*;
pub use typed_pipeline::*;
