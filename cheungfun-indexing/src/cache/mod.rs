//! Cache implementations for the indexing pipeline.
//!
//! This module provides caching capabilities specifically designed for
//! the indexing pipeline, including transformation-level caching that
//! follows LlamaIndex's IngestionCache design.

pub mod ingestion_cache;

pub use ingestion_cache::{
    CacheBackend, CacheEntry, CacheStats, IngestionCache, SimpleCacheBackend, TransformationHasher,
    DEFAULT_CACHE_COLLECTION, DEFAULT_NODES_KEY,
};
