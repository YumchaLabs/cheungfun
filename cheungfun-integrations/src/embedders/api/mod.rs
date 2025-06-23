//! API-based embedding implementation using siumai.
//!
//! This module provides a cloud-based embedding solution using the siumai library
//! to access various embedding APIs like OpenAI, Anthropic, and others.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use cheungfun_integrations::embedders::api::ApiEmbedder;
//! use cheungfun_core::traits::Embedder;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Simple usage with OpenAI
//! let embedder = ApiEmbedder::builder()
//!     .openai("your-api-key")
//!     .model("text-embedding-3-small")
//!     .build()
//!     .await?;
//!
//! let embedding = embedder.embed("Hello, world!").await?;
//! println!("Embedding dimension: {}", embedding.len());
//!
//! // Batch processing
//! let texts = vec!["Hello", "World", "Rust is amazing!"];
//! let embeddings = embedder.embed_batch(texts).await?;
//! println!("Generated {} embeddings", embeddings.len());
//! # Ok(())
//! # }
//! ```
//!
//! # Features
//!
//! - **Multi-provider support**: OpenAI, Anthropic, and more through siumai
//! - **Intelligent caching**: Reduce API costs with built-in caching
//! - **Batch processing**: Efficient handling of multiple texts
//! - **Error handling**: Robust retry mechanisms and error classification
//! - **Cost tracking**: Monitor API usage and costs
//!
//! # Architecture
//!
//! The module is organized into several components:
//! - `config`: Configuration structures and builder patterns
//! - `embedder`: Main embedder implementation
//! - `cache`: Embedding caching mechanisms
//! - `error`: Error types and handling
//!

pub mod cache;
pub mod config;
pub mod embedder;
pub mod error;

#[cfg(test)]
mod tests;

// Re-export main types
pub use cache::{EmbeddingCache, InMemoryCache};
pub use config::{ApiEmbedderConfig, ApiProvider};
pub use embedder::{ApiEmbedder, ApiEmbedderBuilder};
pub use error::{ApiEmbedderError, Result};
