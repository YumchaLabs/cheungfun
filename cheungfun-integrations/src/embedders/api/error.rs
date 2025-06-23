//! Error types for API embedders.

use thiserror::Error;

/// Result type alias for API embedder operations.
pub type Result<T> = std::result::Result<T, ApiEmbedderError>;

/// Errors that can occur when using API embedders.
#[derive(Error, Debug)]
pub enum ApiEmbedderError {
    /// Configuration error
    #[error("Configuration error: {message}")]
    Configuration {
        /// Error message
        message: String,
    },

    /// API authentication error
    #[error("Authentication failed: {message}")]
    Authentication {
        /// Error message
        message: String,
    },

    /// API rate limit exceeded
    #[error("Rate limit exceeded: {message}")]
    RateLimit {
        /// Error message
        message: String,
    },

    /// API quota exceeded
    #[error("Quota exceeded: {message}")]
    QuotaExceeded {
        /// Error message
        message: String,
    },

    /// Invalid model or unsupported model
    #[error("Invalid model: {model}")]
    InvalidModel {
        /// Model name
        model: String,
    },

    /// Network or connection error
    #[error("Network error: {message}")]
    Network {
        /// Error message
        message: String,
    },

    /// API server error
    #[error("Server error: {status_code} - {message}")]
    Server {
        /// HTTP status code
        status_code: u16,
        /// Error message
        message: String,
    },

    /// Request timeout
    #[error("Request timeout after {seconds} seconds")]
    Timeout {
        /// Timeout duration in seconds
        seconds: u64,
    },

    /// Invalid input text
    #[error("Invalid input: {message}")]
    InvalidInput {
        /// Error message
        message: String,
    },

    /// Serialization/deserialization error
    #[error("Serialization error: {message}")]
    Serialization {
        /// Error message
        message: String,
    },

    /// Cache error
    #[error("Cache error: {message}")]
    Cache {
        /// Error message
        message: String,
    },

    /// Siumai library error
    #[error("Siumai error: {0}")]
    Siumai(String),

    /// Generic internal error
    #[error("Internal error: {message}")]
    Internal {
        /// Error message
        message: String,
    },
}

impl ApiEmbedderError {
    /// Create a configuration error.
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create an authentication error.
    pub fn authentication<S: Into<String>>(message: S) -> Self {
        Self::Authentication {
            message: message.into(),
        }
    }

    /// Create a rate limit error.
    pub fn rate_limit<S: Into<String>>(message: S) -> Self {
        Self::RateLimit {
            message: message.into(),
        }
    }

    /// Create a quota exceeded error.
    pub fn quota_exceeded<S: Into<String>>(message: S) -> Self {
        Self::QuotaExceeded {
            message: message.into(),
        }
    }

    /// Create an invalid model error.
    pub fn invalid_model<S: Into<String>>(model: S) -> Self {
        Self::InvalidModel {
            model: model.into(),
        }
    }

    /// Create a network error.
    pub fn network<S: Into<String>>(message: S) -> Self {
        Self::Network {
            message: message.into(),
        }
    }

    /// Create a server error.
    pub fn server<S: Into<String>>(status_code: u16, message: S) -> Self {
        Self::Server {
            status_code,
            message: message.into(),
        }
    }

    /// Create a timeout error.
    pub fn timeout(seconds: u64) -> Self {
        Self::Timeout { seconds }
    }

    /// Create an invalid input error.
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// Create a serialization error.
    pub fn serialization<S: Into<String>>(message: S) -> Self {
        Self::Serialization {
            message: message.into(),
        }
    }

    /// Create a cache error.
    pub fn cache<S: Into<String>>(message: S) -> Self {
        Self::Cache {
            message: message.into(),
        }
    }

    /// Create an internal error.
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Check if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Network { .. } => true,
            Self::Server { status_code, .. } if *status_code >= 500 => true,
            Self::Timeout { .. } => true,
            Self::RateLimit { .. } => true,
            _ => false,
        }
    }

    /// Get the retry delay in seconds for retryable errors.
    pub fn retry_delay(&self) -> Option<u64> {
        match self {
            Self::RateLimit { .. } => Some(60), // Wait 1 minute for rate limits
            Self::Network { .. } | Self::Server { .. } | Self::Timeout { .. } => Some(5), // Wait 5 seconds for other retryable errors
            _ => None,
        }
    }
}

// Convert to cheungfun core error
impl From<ApiEmbedderError> for cheungfun_core::CheungfunError {
    fn from(error: ApiEmbedderError) -> Self {
        match error {
            ApiEmbedderError::Configuration { message } => Self::configuration(message),
            ApiEmbedderError::Authentication { message } => {
                Self::embedding(format!("Authentication failed: {}", message))
            }
            ApiEmbedderError::RateLimit { message } => {
                Self::embedding(format!("Rate limit exceeded: {}", message))
            }
            ApiEmbedderError::QuotaExceeded { message } => {
                Self::embedding(format!("Quota exceeded: {}", message))
            }
            ApiEmbedderError::InvalidModel { model } => {
                Self::embedding(format!("Invalid model: {}", model))
            }
            ApiEmbedderError::Network { message } => {
                Self::embedding(format!("Network error: {}", message))
            }
            ApiEmbedderError::Server {
                status_code,
                message,
            } => Self::embedding(format!("Server error {}: {}", status_code, message)),
            ApiEmbedderError::Timeout { seconds } => {
                Self::timeout(format!("API request timeout after {} seconds", seconds))
            }
            ApiEmbedderError::InvalidInput { message } => Self::validation(message),
            ApiEmbedderError::Serialization { message } => {
                Self::embedding(format!("Serialization error: {}", message))
            }
            ApiEmbedderError::Cache { message } => {
                Self::internal(format!("Cache error: {}", message))
            }
            ApiEmbedderError::Siumai(siumai_error) => {
                Self::embedding(format!("Siumai error: {}", siumai_error))
            }
            ApiEmbedderError::Internal { message } => Self::internal(message),
        }
    }
}
