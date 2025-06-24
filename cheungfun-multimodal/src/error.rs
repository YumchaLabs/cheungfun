//! Error types for multimodal operations.
//!
//! This module defines comprehensive error types for all multimodal operations
//! including media processing, format conversion, and cross-modal operations.

use std::path::PathBuf;
use thiserror::Error;

/// Main error type for multimodal operations.
#[derive(Error, Debug)]
pub enum MultimodalError {
    /// Core framework errors
    #[error("Core framework error: {0}")]
    Core(#[from] cheungfun_core::error::CheungfunError),

    /// Generic errors
    #[error("Generic error: {0}")]
    Generic(#[from] anyhow::Error),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Media processing errors
    #[error("Media processing error: {message}")]
    MediaProcessing { message: String },

    /// Format detection errors
    #[error("Format detection error: {message}")]
    FormatDetection { message: String },

    /// Unsupported format
    #[error("Unsupported format: {format} for modality {modality}")]
    UnsupportedFormat { format: String, modality: String },

    /// Unsupported modality
    #[error("Unsupported modality: {modality}")]
    UnsupportedModality { modality: String },

    /// Cross-modal operation errors
    #[error("Cross-modal operation error: {message}")]
    CrossModalConversion { message: String },

    /// Embedding generation errors
    #[error("Embedding generation error: {message}")]
    EmbeddingGeneration { message: String },

    /// File not found
    #[error("File not found: {path}")]
    FileNotFound { path: PathBuf },

    /// Invalid file format
    #[error("Invalid file format: expected {expected}, got {actual}")]
    InvalidFormat { expected: String, actual: String },

    /// Media content too large
    #[error("Media content too large: {size} bytes exceeds limit of {limit} bytes")]
    ContentTooLarge { size: u64, limit: u64 },

    /// Network errors (for remote resources)
    #[cfg(feature = "remote-resources")]
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// URL parsing errors
    #[error("URL parsing error: {message}")]
    UrlParse { message: String },

    /// Base64 decoding errors
    #[error("Base64 decoding error: {0}")]
    Base64Decode(#[from] base64::DecodeError),

    /// Image processing errors
    #[cfg(feature = "image-support")]
    #[error("Image processing error: {0}")]
    ImageProcessing(#[from] image::ImageError),

    /// Audio processing errors
    #[cfg(feature = "audio-support")]
    #[error("Audio processing error: {message}")]
    AudioProcessing { message: String },

    /// Video processing errors
    #[cfg(feature = "video-support")]
    #[error("Video processing error: {message}")]
    VideoProcessing { message: String },

    /// Machine learning model errors
    #[cfg(feature = "candle")]
    #[error("ML model error: {0}")]
    ModelError(#[from] candle_core::Error),

    /// Tokenization errors
    #[cfg(feature = "candle")]
    #[error("Tokenization error: {0}")]
    TokenizationError(#[from] tokenizers::Error),

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Validation errors
    #[error("Validation error: {message}")]
    Validation { message: String },

    /// Resource not available
    #[error("Resource not available: {resource}")]
    ResourceNotAvailable { resource: String },

    /// Operation timeout
    #[error("Operation timeout: {operation} took longer than {timeout_seconds} seconds")]
    Timeout {
        operation: String,
        timeout_seconds: u64,
    },

    /// Insufficient memory
    #[error(
        "Insufficient memory: operation requires {required} bytes, only {available} available"
    )]
    InsufficientMemory { required: u64, available: u64 },

    /// Concurrent access error
    #[error("Concurrent access error: {message}")]
    ConcurrentAccess { message: String },

    /// Feature not enabled
    #[error("Feature not enabled: {feature}. Enable with --features {feature}")]
    FeatureNotEnabled { feature: String },

    /// Internal error
    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl MultimodalError {
    /// Create a media processing error.
    pub fn media_processing<S: Into<String>>(message: S) -> Self {
        Self::MediaProcessing {
            message: message.into(),
        }
    }

    /// Create a format detection error.
    pub fn format_detection<S: Into<String>>(message: S) -> Self {
        Self::FormatDetection {
            message: message.into(),
        }
    }

    /// Create an unsupported format error.
    pub fn unsupported_format<S1: Into<String>, S2: Into<String>>(
        format: S1,
        modality: S2,
    ) -> Self {
        Self::UnsupportedFormat {
            format: format.into(),
            modality: modality.into(),
        }
    }

    /// Create an unsupported modality error.
    pub fn unsupported_modality<S: Into<String>>(modality: S) -> Self {
        Self::UnsupportedModality {
            modality: modality.into(),
        }
    }

    /// Create a cross-modal conversion error.
    pub fn cross_modal_conversion<S1: Into<String>, S2: Into<String>>(
        source: S1,
        target: S2,
    ) -> Self {
        Self::CrossModalConversion {
            message: format!("cannot convert from {} to {}", source.into(), target.into()),
        }
    }

    /// Create an embedding generation error.
    pub fn embedding_generation<S: Into<String>>(message: S) -> Self {
        Self::EmbeddingGeneration {
            message: message.into(),
        }
    }

    /// Create a file not found error.
    pub fn file_not_found<P: Into<PathBuf>>(path: P) -> Self {
        Self::FileNotFound { path: path.into() }
    }

    /// Create an invalid format error.
    pub fn invalid_format<S1: Into<String>, S2: Into<String>>(expected: S1, actual: S2) -> Self {
        Self::InvalidFormat {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create a content too large error.
    pub fn content_too_large(size: u64, limit: u64) -> Self {
        Self::ContentTooLarge { size, limit }
    }

    /// Create an audio processing error.
    #[cfg(feature = "audio-support")]
    pub fn audio_processing<S: Into<String>>(message: S) -> Self {
        Self::AudioProcessing {
            message: message.into(),
        }
    }

    /// Create a video processing error.
    #[cfg(feature = "video-support")]
    pub fn video_processing<S: Into<String>>(message: S) -> Self {
        Self::VideoProcessing {
            message: message.into(),
        }
    }

    /// Create a configuration error.
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create a validation error.
    pub fn validation<S: Into<String>>(message: S) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    /// Create a resource not available error.
    pub fn resource_not_available<S: Into<String>>(resource: S) -> Self {
        Self::ResourceNotAvailable {
            resource: resource.into(),
        }
    }

    /// Create a timeout error.
    pub fn timeout<S: Into<String>>(operation: S, timeout_seconds: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            timeout_seconds,
        }
    }

    /// Create an insufficient memory error.
    pub fn insufficient_memory(required: u64, available: u64) -> Self {
        Self::InsufficientMemory {
            required,
            available,
        }
    }

    /// Create a concurrent access error.
    pub fn concurrent_access<S: Into<String>>(message: S) -> Self {
        Self::ConcurrentAccess {
            message: message.into(),
        }
    }

    /// Create a feature not enabled error.
    pub fn feature_not_enabled<S: Into<String>>(feature: S) -> Self {
        Self::FeatureNotEnabled {
            feature: feature.into(),
        }
    }

    /// Create an internal error.
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Permanent errors that cannot be recovered
            Self::UnsupportedFormat { .. }
            | Self::UnsupportedModality { .. }
            | Self::InvalidFormat { .. }
            | Self::FeatureNotEnabled { .. }
            | Self::Validation { .. } => false,

            // Temporary errors that might be recoverable
            Self::Io(_)
            | Self::Timeout { .. }
            | Self::InsufficientMemory { .. }
            | Self::ConcurrentAccess { .. } => true,

            #[cfg(feature = "remote-resources")]
            Self::Network(_) => true,

            // Other errors depend on the specific case
            _ => false,
        }
    }

    /// Get the error category for logging and metrics.
    pub fn category(&self) -> &'static str {
        match self {
            Self::Core(_) => "core",
            Self::Generic(_) => "generic",
            Self::Io(_) => "io",
            Self::Serialization(_) => "serialization",
            Self::MediaProcessing { .. } => "media_processing",
            Self::FormatDetection { .. } => "format_detection",
            Self::UnsupportedFormat { .. } => "unsupported_format",
            Self::UnsupportedModality { .. } => "unsupported_modality",
            Self::CrossModalConversion { .. } => "cross_modal_conversion",
            Self::EmbeddingGeneration { .. } => "embedding_generation",
            Self::FileNotFound { .. } => "file_not_found",
            Self::InvalidFormat { .. } => "invalid_format",
            Self::ContentTooLarge { .. } => "content_too_large",
            Self::UrlParse { .. } => "url_parse",
            Self::Base64Decode(_) => "base64_decode",
            Self::Configuration { .. } => "configuration",
            Self::Validation { .. } => "validation",
            Self::ResourceNotAvailable { .. } => "resource_not_available",
            Self::Timeout { .. } => "timeout",
            Self::InsufficientMemory { .. } => "insufficient_memory",
            Self::ConcurrentAccess { .. } => "concurrent_access",
            Self::FeatureNotEnabled { .. } => "feature_not_enabled",
            Self::Internal { .. } => "internal",

            #[cfg(feature = "remote-resources")]
            Self::Network(_) => "network",

            #[cfg(feature = "image-support")]
            Self::ImageProcessing(_) => "image_processing",

            #[cfg(feature = "audio-support")]
            Self::AudioProcessing { .. } => "audio_processing",

            #[cfg(feature = "video-support")]
            Self::VideoProcessing { .. } => "video_processing",

            #[cfg(feature = "candle")]
            Self::ModelError(_) => "model_error",

            #[cfg(feature = "candle")]
            Self::TokenizationError(_) => "tokenization_error",
        }
    }
}

/// Result type alias for multimodal operations.
pub type Result<T> = std::result::Result<T, MultimodalError>;

/// Helper macro for creating multimodal errors.
#[macro_export]
macro_rules! multimodal_error {
    ($variant:ident, $($arg:expr),*) => {
        $crate::error::MultimodalError::$variant($($arg),*)
    };
    ($variant:ident { $($field:ident: $value:expr),* }) => {
        $crate::error::MultimodalError::$variant { $($field: $value),* }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = MultimodalError::media_processing("Test error");
        assert!(matches!(error, MultimodalError::MediaProcessing { .. }));
        assert_eq!(error.category(), "media_processing");
    }

    #[test]
    fn test_error_recoverability() {
        let recoverable = MultimodalError::timeout("test_op", 30);
        let non_recoverable = MultimodalError::unsupported_format("unknown", "test");

        assert!(recoverable.is_recoverable());
        assert!(!non_recoverable.is_recoverable());
    }

    #[test]
    fn test_error_categories() {
        let errors = vec![
            MultimodalError::media_processing("test"),
            MultimodalError::format_detection("test"),
            MultimodalError::configuration("test"),
        ];

        let categories: Vec<_> = errors.iter().map(|e| e.category()).collect();
        assert_eq!(
            categories,
            vec!["media_processing", "format_detection", "configuration"]
        );
    }
}
