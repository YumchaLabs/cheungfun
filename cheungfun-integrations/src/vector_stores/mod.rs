//! Vector store implementations.
//!
//! This module provides concrete implementations of the VectorStore trait
//! for different storage backends.

pub mod memory;
pub mod qdrant;

// Re-export implementations
pub use memory::InMemoryVectorStore;
pub use qdrant::{QdrantConfig, QdrantVectorStore};
