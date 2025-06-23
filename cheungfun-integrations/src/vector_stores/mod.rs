//! Vector store implementations.
//!
//! This module provides concrete implementations of the VectorStore trait
//! for different storage backends.

pub mod memory;
pub mod memory_optimized;
pub mod fast_memory;

#[cfg(feature = "qdrant")]
pub mod qdrant;

#[cfg(feature = "hnsw")]
pub mod hnsw;

// Re-export implementations
pub use memory::InMemoryVectorStore;
pub use memory_optimized::OptimizedInMemoryVectorStore;
pub use fast_memory::FastInMemoryVectorStore;

#[cfg(feature = "qdrant")]
pub use qdrant::{QdrantConfig, QdrantVectorStore};

#[cfg(feature = "hnsw")]
pub use hnsw::{HnswConfig, HnswVectorStore};
