//! Enhanced file filtering system for cheungfun-indexing.
//!
//! This module provides a comprehensive file filtering system that supports:
//! - Gitignore pattern matching
//! - Custom glob patterns
//! - File extension filtering
//! - Hidden file exclusion
//! - Performance-optimized matching

pub mod config;
pub mod file_filter;
pub mod gitignore;
pub mod glob_matcher;

pub use config::FilterConfig;
pub use file_filter::FileFilter;
pub use gitignore::GitignoreMatcher;
pub use glob_matcher::GlobMatcher;

use std::path::Path;

/// Trait for file filtering implementations.
pub trait Filter: Send + Sync {
    /// Check if a file should be included based on its path.
    fn should_include(&self, path: &Path) -> bool;

    /// Check if a directory should be traversed.
    fn should_traverse_dir(&self, path: &Path) -> bool {
        // By default, allow directory traversal unless explicitly excluded
        self.should_include(path)
    }
}

/// Result type for filter operations.
pub type FilterResult<T> = Result<T, FilterError>;

/// Errors that can occur during filtering operations.
#[derive(Debug, thiserror::Error)]
pub enum FilterError {
    /// IO error occurred during file operations.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid glob pattern provided.
    #[error("Invalid glob pattern: {0}")]
    InvalidGlob(String),

    /// Error parsing gitignore file or patterns.
    #[error("Gitignore parsing error: {0}")]
    GitignoreParse(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
}
