//! Vector store implementations.
//!
//! This module provides concrete implementations of the VectorStore trait
//! for different storage backends.

pub mod memory;
pub mod memory_optimized;
pub mod fast_memory;
pub mod qdrant;

// Re-export implementations
pub use memory::InMemoryVectorStore;
pub use memory_optimized::OptimizedInMemoryVectorStore;
pub use fast_memory::FastInMemoryVectorStore;
pub use qdrant::{QdrantConfig, QdrantVectorStore};
