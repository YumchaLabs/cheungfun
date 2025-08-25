//! Configuration types for the Cheungfun framework.
//!
//! This module provides type-safe configuration structures for all
//! components in the RAG pipeline. Configurations are designed to be
//! serializable and validatable.

pub mod embedder;
pub mod extensions;
pub mod llm;
pub mod manager;
pub mod pipeline;
pub mod storage;

// Re-export all config types for convenience
pub use embedder::*;
pub use extensions::*;
pub use llm::*;
pub use manager::*;
pub use pipeline::*;
pub use storage::*;
