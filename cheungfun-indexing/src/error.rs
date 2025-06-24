//! Error types for the indexing module.

use thiserror::Error;

/// Errors that can occur during document loading and indexing operations.
#[derive(Error, Debug)]
pub enum IndexingError {
    /// IO error occurred while reading files or directories.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Error parsing or processing document content.
    #[error("Document parsing error: {message}")]
    DocumentParsing {
        /// Error message describing the parsing issue.
        message: String,
    },

    /// Error extracting text from binary documents (PDF, Word, etc.).
    #[error("Text extraction error: {message}")]
    TextExtraction {
        /// Error message describing the extraction issue.
        message: String,
    },

    /// Error during text splitting or chunking.
    #[error("Text splitting error: {message}")]
    TextSplitting {
        /// Error message describing the splitting issue.
        message: String,
    },

    /// Error during metadata extraction.
    #[error("Metadata extraction error: {message}")]
    MetadataExtraction {
        /// Error message describing the metadata extraction issue.
        message: String,
    },

    /// Unsupported file format.
    #[error("Unsupported file format: {format}")]
    UnsupportedFormat {
        /// The unsupported file format.
        format: String,
    },

    /// File not found or inaccessible.
    #[error("File not found: {path}")]
    FileNotFound {
        /// Path to the file that was not found.
        path: String,
    },

    /// Directory not found or inaccessible.
    #[error("Directory not found: {path}")]
    DirectoryNotFound {
        /// Path to the directory that was not found.
        path: String,
    },

    /// Network error during web content loading.
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Configuration error.
    #[error("Configuration error: {message}")]
    Configuration {
        /// Error message describing the configuration issue.
        message: String,
    },

    /// Pipeline execution error.
    #[error("Pipeline error: {message}")]
    Pipeline {
        /// Error message describing the pipeline issue.
        message: String,
    },

    /// Core framework error.
    #[error("Core error: {0}")]
    Core(#[from] cheungfun_core::error::CheungfunError),

    /// Generic error with custom message.
    #[error("Indexing error: {message}")]
    Generic {
        /// Generic error message.
        message: String,
    },
}

/// Result type alias for indexing operations.
pub type Result<T> = std::result::Result<T, IndexingError>;

impl IndexingError {
    /// Create a new document parsing error.
    pub fn document_parsing<S: Into<String>>(message: S) -> Self {
        Self::DocumentParsing {
            message: message.into(),
        }
    }

    /// Create a new text extraction error.
    pub fn text_extraction<S: Into<String>>(message: S) -> Self {
        Self::TextExtraction {
            message: message.into(),
        }
    }

    /// Create a new text splitting error.
    pub fn text_splitting<S: Into<String>>(message: S) -> Self {
        Self::TextSplitting {
            message: message.into(),
        }
    }

    /// Create a new metadata extraction error.
    pub fn metadata_extraction<S: Into<String>>(message: S) -> Self {
        Self::MetadataExtraction {
            message: message.into(),
        }
    }

    /// Create a new unsupported format error.
    pub fn unsupported_format<S: Into<String>>(format: S) -> Self {
        Self::UnsupportedFormat {
            format: format.into(),
        }
    }

    /// Create a new file not found error.
    pub fn file_not_found<S: Into<String>>(path: S) -> Self {
        Self::FileNotFound { path: path.into() }
    }

    /// Create a new directory not found error.
    pub fn directory_not_found<S: Into<String>>(path: S) -> Self {
        Self::DirectoryNotFound { path: path.into() }
    }

    /// Create a new configuration error.
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create a new pipeline error.
    pub fn pipeline<S: Into<String>>(message: S) -> Self {
        Self::Pipeline {
            message: message.into(),
        }
    }

    /// Create a new generic error.
    pub fn generic<S: Into<String>>(message: S) -> Self {
        Self::Generic {
            message: message.into(),
        }
    }
}

// Convert from anyhow::Error for convenience
impl From<anyhow::Error> for IndexingError {
    fn from(err: anyhow::Error) -> Self {
        Self::Generic {
            message: err.to_string(),
        }
    }
}

// Convert to CheungfunError for trait compatibility
impl From<IndexingError> for cheungfun_core::error::CheungfunError {
    fn from(err: IndexingError) -> Self {
        match err {
            IndexingError::Io(e) => Self::Io(e),
            IndexingError::Core(e) => e,
            IndexingError::Network(e) => Self::Internal {
                message: format!("Network error: {e}"),
            },
            IndexingError::Configuration { message } => Self::Configuration { message },
            IndexingError::Pipeline { message } => Self::Pipeline { message },
            _ => Self::Internal {
                message: err.to_string(),
            },
        }
    }
}
