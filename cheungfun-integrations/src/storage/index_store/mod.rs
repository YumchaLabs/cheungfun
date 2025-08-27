//! Index store implementations for the Cheungfun framework.
//!
//! This module provides index storage implementations based on the KVStore
//! abstraction. Index stores are responsible for persisting and retrieving
//! index metadata and structures in the RAG pipeline.

pub mod keyval_index_store;

// Re-export implementations for convenience
pub use keyval_index_store::KVIndexStore;
