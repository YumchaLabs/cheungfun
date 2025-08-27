//! Key-Value store implementations for the Cheungfun framework.
//!
//! This module provides concrete implementations of the KVStore trait using
//! various storage backends. The design follows LlamaIndex's KVStore pattern,
//! providing a unified interface for different storage systems.

pub mod memory;
pub mod sqlx;

// Re-export implementations for convenience
pub use memory::InMemoryKVStore;
pub use sqlx::SqlxKVStore;

// Re-export the trait and constants from core
pub use cheungfun_core::traits::{KVStore, DEFAULT_BATCH_SIZE, DEFAULT_COLLECTION};
