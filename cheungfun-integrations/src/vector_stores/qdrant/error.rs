//! Error handling and mapping utilities for Qdrant integration.
//!
//! This module provides functions to convert Qdrant-specific errors into
//! the unified CheungfunError type used throughout the framework.

use qdrant_client::QdrantError;

/// Convert QdrantError to CheungfunError.
///
/// This function maps various Qdrant error types to appropriate CheungfunError
/// variants, providing consistent error handling across the framework.
///
/// # Arguments
///
/// * `error` - The QdrantError to convert
///
/// # Returns
///
/// A CheungfunError that represents the equivalent error condition
///
/// # Examples
///
/// ```rust,no_run
/// use qdrant_client::QdrantError;
/// use cheungfun_integrations::vector_stores::qdrant::error::map_qdrant_error;
///
/// // This would typically be called internally when handling Qdrant operations
/// let qdrant_err = QdrantError::ResponseError {
///     status: tonic::Status::not_found("Collection not found")
/// };
/// let cheungfun_err = map_qdrant_error(qdrant_err);
/// ```
pub fn map_qdrant_error(error: QdrantError) -> cheungfun_core::CheungfunError {
    match error {
        QdrantError::ResponseError { status } => {
            if status.message().contains("not found") {
                cheungfun_core::CheungfunError::NotFound {
                    resource: "Qdrant resource".to_string(),
                }
            } else if status.message().contains("already exists") {
                cheungfun_core::CheungfunError::Validation {
                    message: format!("Qdrant resource already exists: {}", status.message()),
                }
            } else if status.message().contains("permission")
                || status.message().contains("unauthorized")
            {
                cheungfun_core::CheungfunError::VectorStore {
                    message: format!("Qdrant authentication error: {}", status),
                }
            } else if status.message().contains("timeout") {
                cheungfun_core::CheungfunError::VectorStore {
                    message: format!("Qdrant timeout error: {}", status),
                }
            } else {
                cheungfun_core::CheungfunError::VectorStore {
                    message: format!("Qdrant HTTP error: {}", status),
                }
            }
        }
        QdrantError::ConversionError(source) => cheungfun_core::CheungfunError::VectorStore {
            message: format!("Qdrant conversion error: {}", source),
        },
        _ => cheungfun_core::CheungfunError::VectorStore {
            message: format!("Qdrant error: {}", error),
        },
    }
}

/// Check if a QdrantError indicates a "not found" condition.
///
/// This is a utility function to quickly check if an error represents
/// a missing resource without converting to CheungfunError.
///
/// # Arguments
///
/// * `error` - The QdrantError to check
///
/// # Returns
///
/// `true` if the error indicates a "not found" condition, `false` otherwise
pub fn is_not_found_error(error: &QdrantError) -> bool {
    match error {
        QdrantError::ResponseError { status } => status.message().contains("not found"),
        _ => false,
    }
}

/// Check if a QdrantError indicates a connection problem.
///
/// This is a utility function to quickly identify connection-related errors
/// that might be recoverable with retry logic.
///
/// # Arguments
///
/// * `error` - The QdrantError to check
///
/// # Returns
///
/// `true` if the error indicates a connection problem, `false` otherwise
pub fn is_connection_error(error: &QdrantError) -> bool {
    match error {
        QdrantError::ResponseError { status } => {
            status.message().contains("timeout")
                || status.message().contains("connection")
                || status.message().contains("network")
        }
        _ => false,
    }
}

/// Check if a QdrantError is potentially recoverable with retry.
///
/// This function helps determine whether an operation should be retried
/// based on the type of error encountered.
///
/// # Arguments
///
/// * `error` - The QdrantError to check
///
/// # Returns
///
/// `true` if the error might be recoverable with retry, `false` otherwise
pub fn is_retryable_error(error: &QdrantError) -> bool {
    match error {
        QdrantError::ResponseError { status } => {
            // Retry on temporary server errors, timeouts, and rate limits
            let message = status.message();
            message.contains("unavailable")
                || message.contains("timeout")
                || message.contains("deadline")
                || message.contains("resource exhausted")
                || message.contains("internal")
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_conversion_error() {
        let qdrant_error = QdrantError::ConversionError("Invalid data format".to_string());

        let cheungfun_error = map_qdrant_error(qdrant_error);

        match cheungfun_error {
            cheungfun_core::CheungfunError::VectorStore { message } => {
                assert!(message.contains("conversion error"));
            }
            _ => panic!("Expected VectorStore error"),
        }
    }

    #[test]
    fn test_is_connection_error() {
        // Since ConnectionError doesn't exist, we test with ResponseError containing connection-related message
        // This test would need to be updated when we have access to actual QdrantError variants
        // For now, we just test that the function doesn't panic
        let conversion_error = QdrantError::ConversionError("Connection failed".to_string());
        assert!(!is_connection_error(&conversion_error)); // ConversionError is not a connection error
    }

    // Note: More comprehensive tests would require mocking tonic::Status
    // which is not easily available without the tonic dependency
}
