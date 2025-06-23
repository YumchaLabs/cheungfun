//! Qdrant vector store implementation.
//!
//! This module provides a production-grade vector store implementation using
//! Qdrant as the backend. It supports all VectorStore operations with high
//! performance and scalability.
//!
//! # Module Structure
//!
//! This module is organized into several sub-modules for better maintainability:
//!
//! - [`config`] - Configuration management and builder patterns
//! - [`client`] - Qdrant client management and connection handling
//! - [`conversion`] - Data type conversions between Cheungfun and Qdrant types
//! - [`store`] - Core VectorStore trait implementation
//! - [`advanced`] - Advanced features like batch operations and filtering
//! - [`error`] - Error handling and mapping utilities
//!
//! # Examples
//!
//! ```rust,no_run
//! use cheungfun_integrations::vector_stores::qdrant::{QdrantVectorStore, QdrantConfig};
//! use cheungfun_core::traits::{VectorStore, DistanceMetric};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = QdrantConfig::new("http://localhost:6334", "my_collection", 384)
//!     .with_distance_metric(DistanceMetric::Cosine);
//!
//! let store = QdrantVectorStore::new(config).await?;
//! # Ok(())
//! # }
//! ```

pub mod advanced;
pub mod client;
pub mod config;
pub mod conversion;
pub mod error;
pub mod store;

// Re-export main types for backward compatibility
pub use config::QdrantConfig;
pub use error::map_qdrant_error;
pub use store::QdrantVectorStore;
