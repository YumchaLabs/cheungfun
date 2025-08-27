//! Chat store implementations for the Cheungfun framework.
//!
//! This module provides chat storage implementations based on the KVStore
//! abstraction. Chat stores are responsible for persisting and retrieving
//! conversation history and chat messages in the RAG pipeline.

pub mod keyval_chat_store;

// Re-export implementations for convenience
pub use keyval_chat_store::KVChatStore;
