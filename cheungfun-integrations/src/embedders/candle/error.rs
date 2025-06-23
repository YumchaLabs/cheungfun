//! Error types for Candle embedder.

use thiserror::Error;

/// Errors that can occur in Candle embedder operations.
#[derive(Error, Debug)]
pub enum CandleError {
    /// Model loading errors
    #[error("Model loading failed: {message}")]
    ModelLoading {
        /// Error message
        message: String
    },

    /// Tokenization errors
    #[error("Tokenization failed: {message}")]
    Tokenization {
        /// Error message
        message: String
    },

    /// Device initialization errors
    #[error("Device initialization failed: {message}")]
    Device {
        /// Error message
        message: String
    },

    /// Inference errors
    #[error("Model inference failed: {message}")]
    Inference {
        /// Error message
        message: String
    },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration {
        /// Error message
        message: String
    },

    /// HuggingFace Hub errors
    #[error("HuggingFace Hub error: {message}")]
    HuggingFaceHub {
        /// Error message
        message: String
    },

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Candle core errors
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Tokenizer errors
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),

    /// HF Hub errors
    #[error("HF Hub error: {0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),
}

impl From<CandleError> for cheungfun_core::CheungfunError {
    fn from(err: CandleError) -> Self {
        cheungfun_core::CheungfunError::Embedding {
            message: err.to_string(),
        }
    }
}
