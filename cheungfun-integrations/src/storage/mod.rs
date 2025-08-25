//! Storage implementations for the Cheungfun framework.
//!
//! This module provides concrete implementations of storage traits using
//! various database backends, following LlamaIndex's StorageContext pattern.

#[cfg(feature = "storage")]
pub mod sqlx;

#[cfg(feature = "storage")]
pub use sqlx::*;

// Re-export storage traits from core for convenience
pub use cheungfun_core::traits::{
    ChatStore, DocumentStore, IndexStore, IndexStruct, StorageContext, StorageContextStats,
    VectorStore,
};
