//! Document loading traits.
//!
//! This module defines traits for loading documents from various data sources
//! such as files, databases, APIs, and streaming sources.

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

use crate::{Document, Result};

/// Loads documents from various data sources.
///
/// This trait provides a unified interface for loading documents from different
/// sources like files, databases, web APIs, etc. Implementations should handle
/// the specifics of each data source while providing a consistent API.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::Loader;
/// use cheungfun_core::{Document, Result};
/// use async_trait::async_trait;
///
/// struct FileLoader {
///     path: String,
/// }
///
/// #[async_trait]
/// impl Loader for FileLoader {
///     async fn load(&self) -> Result<Vec<Document>> {
///         // Implementation would read files from the path
///         Ok(vec![Document::new("Sample content")])
///     }
/// }
/// ```
#[async_trait]
pub trait Loader: Send + Sync + std::fmt::Debug {
    /// Load documents from the data source.
    ///
    /// This method should load all available documents from the source
    /// and return them as a vector. For large datasets, consider using
    /// `StreamingLoader` instead.
    ///
    /// # Errors
    ///
    /// Returns an error if the data source cannot be accessed or if
    /// document parsing fails.
    async fn load(&self) -> Result<Vec<Document>>;

    /// Get a human-readable name for this loader.
    ///
    /// This is used for logging and debugging purposes.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Check if the loader can access its data source.
    ///
    /// This method can be used to validate configuration before
    /// attempting to load documents.
    async fn health_check(&self) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Get metadata about the data source.
    ///
    /// This can include information like the number of documents,
    /// last update time, etc.
    async fn metadata(&self) -> Result<std::collections::HashMap<String, serde_json::Value>> {
        // Default implementation returns empty metadata
        Ok(std::collections::HashMap::new())
    }
}

/// Loads documents incrementally using streaming.
///
/// This trait is useful for large datasets where loading all documents
/// at once would consume too much memory. It provides a stream of documents
/// that can be processed one at a time.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::StreamingLoader;
/// use cheungfun_core::{Document, Result};
/// use futures::Stream;
/// use std::pin::Pin;
///
/// struct StreamingFileLoader {
///     directory: String,
/// }
///
/// impl StreamingLoader for StreamingFileLoader {
///     fn into_stream(self) -> Pin<Box<dyn Stream<Item = Result<Document>> + Send>> {
///         // Implementation would return a stream of documents
///         Box::pin(futures::stream::empty())
///     }
/// }
/// ```
pub trait StreamingLoader: Send + Sync + std::fmt::Debug {
    /// Create a stream of documents from the data source.
    ///
    /// This method consumes the loader and returns a stream that yields
    /// documents one at a time. This is more memory-efficient for large
    /// datasets.
    ///
    /// # Returns
    ///
    /// A stream that yields `Result<Document>` items. The stream should
    /// handle errors gracefully and continue processing when possible.
    fn into_stream(self) -> Pin<Box<dyn Stream<Item = Result<Document>> + Send>>;

    /// Get a human-readable name for this loader.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Configuration for batch loading operations.
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Maximum number of documents to load in a single batch.
    pub batch_size: Option<usize>,

    /// Maximum number of concurrent loading operations.
    pub concurrency: Option<usize>,

    /// Timeout for loading operations in seconds.
    pub timeout_seconds: Option<u64>,

    /// Whether to continue loading if some documents fail.
    pub continue_on_error: bool,

    /// Additional loader-specific configuration.
    pub additional_config: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: Some(100),
            concurrency: Some(4),
            timeout_seconds: Some(300), // 5 minutes
            continue_on_error: true,
            additional_config: std::collections::HashMap::new(),
        }
    }
}

/// Statistics about a loading operation.
#[derive(Debug, Clone)]
pub struct LoadingStats {
    /// Total number of documents loaded successfully.
    pub documents_loaded: usize,

    /// Number of documents that failed to load.
    pub documents_failed: usize,

    /// Total time taken for the loading operation.
    pub duration: std::time::Duration,

    /// Average loading time per document.
    pub avg_time_per_document: std::time::Duration,

    /// List of errors encountered during loading.
    pub errors: Vec<String>,

    /// Additional statistics specific to the loader.
    pub additional_stats: std::collections::HashMap<String, serde_json::Value>,
}

impl LoadingStats {
    /// Create new loading statistics.
    pub fn new() -> Self {
        Self {
            documents_loaded: 0,
            documents_failed: 0,
            duration: std::time::Duration::ZERO,
            avg_time_per_document: std::time::Duration::ZERO,
            errors: Vec::new(),
            additional_stats: std::collections::HashMap::new(),
        }
    }

    /// Calculate the success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        let total = self.documents_loaded + self.documents_failed;
        if total == 0 {
            0.0
        } else {
            (self.documents_loaded as f64 / total as f64) * 100.0
        }
    }

    /// Check if the loading operation was successful.
    pub fn is_successful(&self) -> bool {
        self.documents_loaded > 0 && self.documents_failed == 0
    }

    /// Update average time per document.
    pub fn update_avg_time(&mut self) {
        let total_docs = self.documents_loaded + self.documents_failed;
        if total_docs > 0 {
            self.avg_time_per_document = self.duration / total_docs as u32;
        }
    }
}

impl Default for LoadingStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loading_stats() {
        let mut stats = LoadingStats::new();
        stats.documents_loaded = 80;
        stats.documents_failed = 20;
        stats.duration = std::time::Duration::from_secs(100);
        stats.update_avg_time();

        assert_eq!(stats.success_rate(), 80.0);
        assert!(!stats.is_successful()); // Has failures
        assert_eq!(
            stats.avg_time_per_document,
            std::time::Duration::from_secs(1)
        );
    }

    #[test]
    fn test_loader_config_default() {
        let config = LoaderConfig::default();
        assert_eq!(config.batch_size, Some(100));
        assert_eq!(config.concurrency, Some(4));
        assert_eq!(config.timeout_seconds, Some(300));
        assert!(config.continue_on_error);
    }
}
