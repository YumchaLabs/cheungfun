//! External service integrations for Cheungfun.
//!
//! This crate provides integrations with vector databases, embedding models,
//! and other external services used in RAG applications.

#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod embedders;
pub mod vector_stores;

// Re-export commonly used types
pub use vector_stores::InMemoryVectorStore;

// Feature-gated embedder exports
#[cfg(feature = "candle")]
pub use embedders::CandleEmbedder;

#[cfg(feature = "fastembed")]
pub use embedders::FastEmbedder;

#[cfg(feature = "api")]
pub use embedders::ApiEmbedder;
