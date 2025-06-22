//! Vector store implementations.
//!
//! This module provides concrete implementations of the VectorStore trait
//! for different storage backends.

pub mod memory;

// Re-export implementations
pub use memory::InMemoryVectorStore;
