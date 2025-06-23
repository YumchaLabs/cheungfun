//! Candle-based embedding implementation.
//!
//! This module provides a complete implementation of local embedding generation
//! using the Candle ML framework. It supports loading sentence-transformers models
//! from HuggingFace Hub and generating embeddings locally with CPU/GPU acceleration.
//!
//! # Architecture
//!
//! The module is organized into several components:
//! - `device`: Device management (CPU/CUDA detection and selection)
//! - `model`: Model loading and management from HuggingFace Hub
//! - `tokenizer`: Text tokenization using HuggingFace tokenizers
//! - `embedder`: Main embedder implementation
//!
//! # Examples
//!
//! ```rust,no_run
//! use cheungfun_integrations::embedders::candle::CandleEmbedder;
//! use cheungfun_core::traits::Embedder;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create embedder with default model
//! let embedder = CandleEmbedder::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").await?;
//!
//! // Generate single embedding
//! let embedding = embedder.embed("Hello, world!").await?;
//! println!("Embedding dimension: {}", embedding.len());
//!
//! // Generate batch embeddings
//! let texts = vec!["Hello", "World", "Rust is great!"];
//! let embeddings = embedder.embed_batch(texts).await?;
//! println!("Generated {} embeddings", embeddings.len());
//! # Ok(())
//! # }
//! ```

pub mod device;
pub mod model;
pub mod tokenizer;
pub mod embedder;
pub mod config;
pub mod error;

// Re-export main types
pub use embedder::CandleEmbedder;
pub use config::CandleEmbedderConfig;
pub use error::CandleError;
