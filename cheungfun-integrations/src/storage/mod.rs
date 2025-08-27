//! Storage implementations for the Cheungfun framework.
//!
//! This module provides concrete implementations of storage traits using
//! various backends. The architecture follows LlamaIndex's pattern with
//! KVStore as the foundation and specialized stores built on top.

// KVStore implementations (always available)
pub mod kvstore;

// Specialized store implementations based on KVStore
pub mod docstore;
pub mod index_store;
pub mod chat_store;

// Legacy SQLx implementations (feature-gated)
#[cfg(feature = "storage")]
pub mod sqlx;

// Re-export KVStore implementations
pub use kvstore::{InMemoryKVStore, SqlxKVStore};

// Re-export specialized store implementations
pub use docstore::KVDocumentStore;
pub use index_store::KVIndexStore;
pub use chat_store::KVChatStore;

// Re-export legacy implementations for backward compatibility
#[cfg(feature = "storage")]
pub use sqlx::*;

// Re-export storage traits from core for convenience
pub use cheungfun_core::traits::{
    ChatStore, DocumentStore, IndexStore, IndexStruct, KVStore, StorageContext, StorageContextStats,
    VectorStore, DEFAULT_COLLECTION,
};
