//! Embedding model implementations.
//!
//! This module provides concrete implementations of the Embedder trait
//! for different embedding models and services.

pub mod candle;

// Re-export implementations
pub use candle::CandleEmbedder;
