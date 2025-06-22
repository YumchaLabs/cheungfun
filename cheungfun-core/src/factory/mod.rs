//! Factory traits for creating components from configuration.
//!
//! This module provides factory patterns for creating embedders,
//! vector stores, and LLM clients from configuration. Factories
//! enable dependency injection and make testing easier.

pub mod embedder;
pub mod llm;
pub mod storage;

// Re-export all factory traits for convenience
pub use embedder::*;
pub use llm::*;
pub use storage::*;
