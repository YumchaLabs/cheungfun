//! Keyword store implementations for Cheungfun.
//!
//! This module provides various keyword store implementations that implement
//! the KeywordStore trait, enabling keyword-based indexing and retrieval.

pub mod memory;

pub use memory::InMemoryKeywordStore;
