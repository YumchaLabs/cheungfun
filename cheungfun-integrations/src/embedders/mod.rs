//! Embedding model implementations.
//!
//! This module provides concrete implementations of the Embedder trait
//! for different embedding models and services.

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "fastembed")]
pub mod fastembed;

// Re-export implementations based on features
#[cfg(feature = "candle")]
pub use candle::{CandleEmbedder, CandleEmbedderConfig};

#[cfg(feature = "fastembed")]
pub use fastembed::{FastEmbedder, FastEmbedConfig, ModelPreset};
