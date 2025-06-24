//! FastEmbed-based embedding implementation.
//!
//! This module provides a simple, fast, and reliable embedding solution using `FastEmbed`.
//! It focuses on ease of use while maintaining high performance for production workloads.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use cheungfun_integrations::embedders::fastembed::FastEmbedder;
//! use cheungfun_core::traits::Embedder;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Simple usage with defaults
//! let embedder = FastEmbedder::new().await?;
//! let embedding = embedder.embed("Hello, world!").await?;
//!
//! // With specific model
//! let embedder = FastEmbedder::with_model("BAAI/bge-large-en-v1.5").await?;
//! let embeddings = embedder.embed_batch(vec!["Hello", "World"]).await?;
//!
//! // With preset configuration
//! let embedder = FastEmbedder::multilingual().await?;
//! let embedding = embedder.embed("Bonjour le monde!").await?;
//! # Ok(())
//! # }
//! ```

mod config;
mod embedder;
mod error;

pub use config::{FastEmbedConfig, ModelPreset};
pub use embedder::FastEmbedder;
pub use error::FastEmbedError;
