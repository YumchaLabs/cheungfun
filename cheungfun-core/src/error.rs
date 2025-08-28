//! Error types for the Cheungfun framework.
//!
//! This module provides comprehensive error handling with context-aware error types
//! that help with debugging and error reporting in RAG applications.

use thiserror::Error;

/// Core error types for Cheungfun framework.
///
/// This enum covers all possible error conditions that can occur during
/// document processing, indexing, querying, and response generation.
#[derive(Error, Debug)]
pub enum CheungfunError {
    /// I/O related errors (file reading, network operations, etc.)
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Embedding generation errors
    #[error("Embedding error: {message}")]
    Embedding {
        /// Detailed error message
        message: String,
    },

    /// Vector store operation errors
    #[error("Vector store error: {message}")]
    VectorStore {
        /// Detailed error message
        message: String,
    },

    /// Storage operation errors
    #[error("Storage error: {0}")]
    Storage(String),

    /// LLM/Response generation errors
    #[error("LLM error: {message}")]
    Llm {
        /// Detailed error message
        message: String,
    },

    /// Pipeline execution errors
    #[error("Pipeline error: {message}")]
    Pipeline {
        /// Detailed error message
        message: String,
    },

    /// Configuration validation errors
    #[error("Configuration error: {message}")]
    Configuration {
        /// Detailed error message
        message: String,
    },

    /// Input validation errors
    #[error("Validation error: {message}")]
    Validation {
        /// Detailed error message
        message: String,
    },

    /// Resource not found errors
    #[error("Not found: {resource}")]
    NotFound {
        /// Name of the missing resource
        resource: String,
    },

    /// Operation timeout errors
    #[error("Timeout: {operation}")]
    Timeout {
        /// Name of the operation that timed out
        operation: String,
    },

    /// Rate limiting errors
    #[error("Rate limit exceeded")]
    RateLimit,

    /// Authentication failures
    #[error("Authentication failed")]
    Authentication,

    /// Permission denied errors
    #[error("Permission denied")]
    Permission,

    /// Internal framework errors
    #[error("Internal error: {message}")]
    Internal {
        /// Detailed error message
        message: String,
    },

    /// Generic errors from external dependencies
    #[error("External error: {source}")]
    External {
        /// The underlying error
        #[source]
        source: anyhow::Error,
    },
}

impl CheungfunError {
    /// Create a new embedding error with a message.
    pub fn embedding<S: Into<String>>(message: S) -> Self {
        Self::Embedding {
            message: message.into(),
        }
    }

    /// Create a new vector store error with a message.
    pub fn vector_store<S: Into<String>>(message: S) -> Self {
        Self::VectorStore {
            message: message.into(),
        }
    }

    /// Create a new LLM error with a message.
    pub fn llm<S: Into<String>>(message: S) -> Self {
        Self::Llm {
            message: message.into(),
        }
    }

    /// Create a new pipeline error with a message.
    pub fn pipeline<S: Into<String>>(message: S) -> Self {
        Self::Pipeline {
            message: message.into(),
        }
    }

    /// Create a new configuration error with a message.
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create a new validation error with a message.
    pub fn validation<S: Into<String>>(message: S) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    /// Create a new not found error with a resource name.
    pub fn not_found<S: Into<String>>(resource: S) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }

    /// Create a new timeout error with an operation name.
    pub fn timeout<S: Into<String>>(operation: S) -> Self {
        Self::Timeout {
            operation: operation.into(),
        }
    }

    /// Create a new internal error with a message.
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Create a new external error from any error that implements `Into<anyhow::Error>`.
    pub fn external<E: Into<anyhow::Error>>(error: E) -> Self {
        Self::External {
            source: error.into(),
        }
    }

    /// Check if this error is retryable.
    ///
    /// Returns `true` for transient errors that might succeed on retry,
    /// such as network timeouts or rate limits.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Timeout { .. } | Self::RateLimit | Self::Io(_))
    }

    /// Check if this error is a client error (4xx-style).
    ///
    /// Returns `true` for errors caused by invalid input or configuration
    /// that won't be fixed by retrying.
    #[must_use]
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            Self::Validation { .. }
                | Self::Configuration { .. }
                | Self::NotFound { .. }
                | Self::Authentication
                | Self::Permission
        )
    }
}

/// Convert from `anyhow::Error` to `CheungfunError`.
impl From<anyhow::Error> for CheungfunError {
    fn from(error: anyhow::Error) -> Self {
        Self::External { source: error }
    }
}

#[cfg(feature = "storage")]
impl From<sqlx::Error> for CheungfunError {
    fn from(error: sqlx::Error) -> Self {
        Self::Storage(format!("Database error: {error}"))
    }
}

/// Result type alias for convenience.
///
/// This is the standard result type used throughout the Cheungfun framework.
pub type Result<T> = std::result::Result<T, CheungfunError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = CheungfunError::embedding("Failed to generate embedding");
        assert!(matches!(err, CheungfunError::Embedding { .. }));
        assert_eq!(
            err.to_string(),
            "Embedding error: Failed to generate embedding"
        );
    }

    #[test]
    fn test_error_retryable() {
        assert!(CheungfunError::timeout("network").is_retryable());
        assert!(CheungfunError::RateLimit.is_retryable());
        assert!(!CheungfunError::validation("invalid input").is_retryable());
    }

    #[test]
    fn test_error_client_error() {
        assert!(CheungfunError::validation("invalid").is_client_error());
        assert!(CheungfunError::Authentication.is_client_error());
        assert!(!CheungfunError::timeout("network").is_client_error());
    }
}
