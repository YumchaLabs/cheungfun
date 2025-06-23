//! Error types for FastEmbed embedder.

use thiserror::Error;

/// Simplified error types for FastEmbed operations.
#[derive(Error, Debug)]
pub enum FastEmbedError {
    /// Model initialization failed
    #[error("Failed to initialize model '{model}': {reason}")]
    ModelInit {
        /// The model name that failed to initialize
        model: String,
        /// The reason for the failure
        reason: String
    },

    /// Embedding generation failed
    #[error("Failed to generate embeddings: {reason}")]
    Embedding {
        /// The reason for the failure
        reason: String
    },

    /// Configuration error
    #[error("Invalid configuration: {reason}")]
    Config {
        /// The reason for the configuration error
        reason: String
    },

    /// I/O error (file operations, network, etc.)
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// FastEmbed library error
    #[error("FastEmbed error: {0}")]
    FastEmbed(#[from] fastembed::Error),
}

impl From<FastEmbedError> for cheungfun_core::CheungfunError {
    fn from(err: FastEmbedError) -> Self {
        cheungfun_core::CheungfunError::Embedding {
            message: err.to_string(),
        }
    }
}

/// Result type for FastEmbed operations.
pub type Result<T> = std::result::Result<T, FastEmbedError>;
