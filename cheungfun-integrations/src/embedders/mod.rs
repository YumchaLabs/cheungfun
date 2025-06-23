//! Embedding model implementations.
//!
//! This module provides concrete implementations of the Embedder trait
//! for different embedding models and services.

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "fastembed")]
pub mod fastembed;

#[cfg(feature = "api")]
pub mod api;

// Re-export implementations based on features
#[cfg(feature = "candle")]
pub use candle::{CandleEmbedder, CandleEmbedderConfig};

#[cfg(feature = "fastembed")]
pub use fastembed::{FastEmbedConfig, FastEmbedder, ModelPreset};

#[cfg(feature = "api")]
pub use api::{ApiEmbedder, ApiEmbedderConfig, ApiProvider};
