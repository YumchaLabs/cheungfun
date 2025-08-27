//! Document store implementations for the Cheungfun framework.
//!
//! This module provides document storage implementations based on the KVStore
//! abstraction. Document stores are responsible for persisting and retrieving
//! Document objects in the RAG pipeline.

pub mod keyval_docstore;

// Re-export implementations for convenience
pub use keyval_docstore::KVDocumentStore;
