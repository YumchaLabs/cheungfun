//! Document loaders for various data sources.
//!
//! This module provides implementations of the `Loader` trait for different
//! data sources including files, directories, web content, and more.

pub mod directory;
pub mod file;
pub mod filter;
pub mod web;

pub use directory::DirectoryLoader;
pub use file::FileLoader;
pub use filter::{FileFilter, Filter, FilterConfig};
pub use web::WebLoader;

use crate::error::{IndexingError, Result};
use cheungfun_core::Document;

/// Utility functions for document loading.
pub mod utils {
    use super::*;
    use std::path::Path;

    /// Detect the content type of a file based on its extension.
    pub fn detect_content_type(path: &Path) -> Option<String> {
        let extension = path.extension()?.to_str()?.to_lowercase();

        match extension.as_str() {
            "txt" => Some("text/plain".to_string()),
            "md" | "markdown" => Some("text/markdown".to_string()),
            "html" | "htm" => Some("text/html".to_string()),
            "pdf" => Some("application/pdf".to_string()),
            "doc" => Some("application/msword".to_string()),
            "docx" => Some(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    .to_string(),
            ),
            "csv" => Some("text/csv".to_string()),
            "json" => Some("application/json".to_string()),
            "xml" => Some("application/xml".to_string()),
            "rtf" => Some("application/rtf".to_string()),
            _ => None,
        }
    }

    /// Check if a file type is supported for text extraction.
    pub fn is_supported_file_type(path: &Path) -> bool {
        detect_content_type(path).is_some()
    }

    /// Get the file size in bytes.
    pub async fn get_file_size(path: &Path) -> Result<u64> {
        let metadata = tokio::fs::metadata(path)
            .await
            .map_err(|_| IndexingError::file_not_found(path.display().to_string()))?;
        Ok(metadata.len())
    }

    /// Read file content as UTF-8 string.
    pub async fn read_text_file(path: &Path) -> Result<String> {
        tokio::fs::read_to_string(path)
            .await
            .map_err(|e| IndexingError::Io(e))
    }

    /// Create a document from file content with basic metadata.
    pub fn create_document_from_file(
        content: String,
        path: &Path,
        content_type: Option<String>,
        file_size: Option<u64>,
    ) -> Document {
        let mut doc = Document::new(content);

        // Add basic file metadata
        doc.metadata.insert(
            "source".to_string(),
            serde_json::Value::String(path.display().to_string()),
        );
        doc.metadata.insert(
            "filename".to_string(),
            serde_json::Value::String(
                path.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
            ),
        );

        if let Some(parent) = path.parent() {
            doc.metadata.insert(
                "directory".to_string(),
                serde_json::Value::String(parent.display().to_string()),
            );
        }

        if let Some(content_type) = content_type {
            doc.metadata.insert(
                "content_type".to_string(),
                serde_json::Value::String(content_type),
            );
        }

        if let Some(size) = file_size {
            doc.metadata.insert(
                "file_size".to_string(),
                serde_json::Value::Number(size.into()),
            );
        }

        // Add timestamps if available
        if let Ok(metadata) = std::fs::metadata(path) {
            if let Ok(created) = metadata.created() {
                if let Ok(duration) = created.duration_since(std::time::UNIX_EPOCH) {
                    doc.metadata.insert(
                        "created_at".to_string(),
                        serde_json::Value::Number(duration.as_secs().into()),
                    );
                }
            }

            if let Ok(modified) = metadata.modified() {
                if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                    doc.metadata.insert(
                        "modified_at".to_string(),
                        serde_json::Value::Number(duration.as_secs().into()),
                    );
                }
            }
        }

        doc
    }
}

/// Configuration for loader behavior.
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Maximum file size to process (in bytes).
    pub max_file_size: Option<u64>,

    /// File extensions to include (if None, all supported types are included).
    pub include_extensions: Option<Vec<String>>,

    /// File extensions to exclude.
    pub exclude_extensions: Vec<String>,

    /// Whether to follow symbolic links.
    pub follow_symlinks: bool,

    /// Maximum depth for recursive directory traversal.
    pub max_depth: Option<usize>,

    /// Whether to continue processing if some files fail.
    pub continue_on_error: bool,

    /// Timeout for individual file operations (in seconds).
    pub timeout_seconds: Option<u64>,

    /// Enhanced file filtering configuration.
    pub filter_config: Option<FilterConfig>,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            max_file_size: Some(100 * 1024 * 1024), // 100MB default limit
            include_extensions: None,
            exclude_extensions: vec![
                "exe".to_string(),
                "bin".to_string(),
                "dll".to_string(),
                "so".to_string(),
                "dylib".to_string(),
            ],
            follow_symlinks: false,
            max_depth: Some(10),
            continue_on_error: true,
            timeout_seconds: Some(30),
            filter_config: None, // Use enhanced filtering when specified
        }
    }
}

impl LoaderConfig {
    /// Create a new loader configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum file size.
    pub fn with_max_file_size(mut self, size: u64) -> Self {
        self.max_file_size = Some(size);
        self
    }

    /// Set included file extensions.
    pub fn with_include_extensions(mut self, extensions: Vec<String>) -> Self {
        self.include_extensions = Some(extensions);
        self
    }

    /// Set excluded file extensions.
    pub fn with_exclude_extensions(mut self, extensions: Vec<String>) -> Self {
        self.exclude_extensions = extensions;
        self
    }

    /// Set whether to follow symbolic links.
    pub fn with_follow_symlinks(mut self, follow: bool) -> Self {
        self.follow_symlinks = follow;
        self
    }

    /// Set maximum directory traversal depth.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Set whether to continue on errors.
    pub fn with_continue_on_error(mut self, continue_on_error: bool) -> Self {
        self.continue_on_error = continue_on_error;
        self
    }

    /// Set timeout for file operations.
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = Some(seconds);
        self
    }

    /// Set enhanced filter configuration.
    pub fn with_filter_config(mut self, filter_config: FilterConfig) -> Self {
        self.filter_config = Some(filter_config);
        self
    }

    /// Enable enhanced filtering with default configuration.
    pub fn with_enhanced_filtering(mut self) -> Self {
        self.filter_config = Some(FilterConfig::default());
        self
    }

    /// Enable enhanced filtering for source code only.
    pub fn with_source_code_filtering(mut self) -> Self {
        self.filter_config = Some(FilterConfig::source_code_only());
        self
    }

    /// Enable enhanced filtering for text files only.
    pub fn with_text_files_filtering(mut self) -> Self {
        self.filter_config = Some(FilterConfig::text_files_only());
        self
    }

    /// Check if enhanced filtering is enabled.
    pub fn has_enhanced_filtering(&self) -> bool {
        self.filter_config.is_some()
    }

    /// Get the filter configuration if available.
    pub fn get_filter_config(&self) -> Option<&FilterConfig> {
        self.filter_config.as_ref()
    }
}
