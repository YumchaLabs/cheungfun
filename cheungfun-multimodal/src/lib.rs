//! # Cheungfun Multimodal
//!
//! A comprehensive multimodal extension for the Cheungfun RAG framework, providing
//! unified processing, embedding, and retrieval capabilities across text, image,
//! audio, video, and document modalities.
//!
//! ## Features
//!
//! - **Unified Data Structures**: Common interfaces for all media types
//! - **Cross-Modal Embeddings**: Generate embeddings that work across modalities
//! - **Cross-Modal Retrieval**: Search for images using text, audio using images, etc.
//! - **Format Support**: Wide range of media formats with automatic detection
//! - **Performance Optimized**: Batch processing, streaming, and GPU acceleration
//! - **Extensible Architecture**: Plugin system for custom processors and embedders
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use cheungfun_multimodal::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create a multimodal document from an image
//!     let image_content = MediaContent::from_file_path("image.jpg", MediaFormat::Jpeg);
//!     let doc = MultimodalDocument::with_media(
//!         Document::new("A beautiful sunset"),
//!         ModalityType::Image,
//!         image_content,
//!     );
//!
//!     // Create a cross-modal query
//!     let query = CrossModalQuery::from_text("Find images of sunsets");
//!
//!     println!("Created multimodal document and query");
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! The library is organized into several key modules:
//!
//! - [`types`]: Core data structures for multimodal content
//! - [`traits`]: Trait definitions for processors, embedders, and retrievers
//! - [`processors`]: Implementations for different media processing tasks
//! - [`embedders`]: Multimodal embedding generation
//! - [`retrievers`]: Cross-modal search and retrieval
//! - [`utils`]: Utility functions and helpers
//!
//! ## Modality Support
//!
//! | Modality | Read | Write | Embed | Search | Extract Text |
//! |----------|------|-------|-------|--------|--------------|
//! | Text     | ✅   | ✅    | ✅    | ✅     | ✅           |
//! | Image    | ✅   | ✅    | ✅    | ✅     | ❌           |
//! | Audio    | ✅   | ❌    | ✅    | ✅     | ✅*          |
//! | Video    | ✅   | ❌    | ✅    | ✅     | ✅*          |
//! | Document | ✅   | ❌    | ✅    | ✅     | ✅           |
//!
//! *Requires speech recognition features
//!
//! ## Feature Flags
//!
//! - `image-support`: Enable image processing capabilities
//! - `audio-support`: Enable audio processing capabilities  
//! - `video-support`: Enable video processing capabilities
//! - `format-detection`: Enable automatic format detection
//! - `remote-resources`: Enable fetching from URLs
//! - `candle`: Enable Candle ML framework integration
//! - `clip`: Enable CLIP multimodal embeddings
//! - `full`: Enable all features

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_inception)]

// Core modules
pub mod error;
pub mod traits;
pub mod types;

// Feature-gated modules
#[cfg(any(feature = "image-support", feature = "audio-support", feature = "video-support"))]
pub mod processors;

#[cfg(feature = "candle")]
pub mod embedders;

// #[cfg(any(feature = "image-support", feature = "audio-support", feature = "video-support"))]
// pub mod retrievers;

pub mod utils;

// Re-export commonly used items
pub use error::{MultimodalError, Result};

/// Prelude module for convenient imports.
///
/// This module re-exports the most commonly used types and traits
/// for easy importing with `use cheungfun_multimodal::prelude::*;`.
pub mod prelude {
    // Core types
    pub use crate::types::{
        HasModality, MediaContent, MediaContentBuilder, MediaData, MediaFormat,
        MediaMetadata, ModalityType, MultimodalDocument, MultimodalNode,
        MultimodalNodeBuilder, TextExtraction,
    };

    // Core traits
    pub use crate::traits::{
        CrossModalQuery, CrossModalRetriever, CrossModalRetrievalOptions,
        MediaProcessor, MultimodalComponent, MultimodalEmbedder,
        ProcessingOptions, QueryContent, ScoredMultimodalNode,
    };

    // Error handling
    pub use crate::error::{MultimodalError, Result};

    // Utility functions
    pub use crate::types::utils::{
        are_modalities_compatible, detect_format_from_extension,
        detect_modality_from_extension,
    };

    // Re-export core types for convenience
    pub use cheungfun_core::types::{Document, Node};
}

/// Version information for the multimodal library.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get version information for the multimodal library.
pub fn version() -> &'static str {
    VERSION
}

/// Get build information for the multimodal library.
pub fn build_info() -> std::collections::HashMap<&'static str, &'static str> {
    let mut info = std::collections::HashMap::new();
    info.insert("version", VERSION);
    info.insert("target", "unknown"); // Will be set at compile time in real builds
    info.insert("profile", "unknown"); // Will be set at compile time in real builds
    
    // Feature flags
    #[cfg(feature = "image-support")]
    info.insert("image-support", "enabled");
    #[cfg(not(feature = "image-support"))]
    info.insert("image-support", "disabled");
    
    #[cfg(feature = "audio-support")]
    info.insert("audio-support", "enabled");
    #[cfg(not(feature = "audio-support"))]
    info.insert("audio-support", "disabled");
    
    #[cfg(feature = "video-support")]
    info.insert("video-support", "enabled");
    #[cfg(not(feature = "video-support"))]
    info.insert("video-support", "disabled");
    
    #[cfg(feature = "candle")]
    info.insert("candle", "enabled");
    #[cfg(not(feature = "candle"))]
    info.insert("candle", "disabled");
    
    #[cfg(feature = "clip")]
    info.insert("clip", "enabled");
    #[cfg(not(feature = "clip"))]
    info.insert("clip", "disabled");
    
    info
}

/// Initialize the multimodal library with default settings.
///
/// This function sets up logging, validates feature availability,
/// and performs any necessary initialization.
pub fn init() -> Result<()> {
    // Initialize logging if not already done
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "cheungfun_multimodal=info");
    }
    
    // Initialize tracing subscriber if not already initialized
    let _ = tracing_subscriber::fmt::try_init();
    
    tracing::info!("Initializing Cheungfun Multimodal v{}", VERSION);
    
    // Log enabled features
    let build_info = build_info();
    for (feature, status) in build_info.iter() {
        if feature.ends_with("-support") || *feature == "candle" || *feature == "clip" {
            tracing::debug!("Feature {}: {}", feature, status);
        }
    }
    
    // Validate critical dependencies
    validate_dependencies()?;
    
    tracing::info!("Cheungfun Multimodal initialized successfully");
    Ok(())
}

/// Validate that required dependencies are available.
fn validate_dependencies() -> Result<()> {
    // Check if core framework is available
    // Note: cheungfun_core::init() might not exist, so we'll do basic validation
    // if let Err(e) = cheungfun_core::init() {
    //     return Err(MultimodalError::Core(e));
    // }
    
    // Validate feature-specific dependencies
    #[cfg(feature = "image-support")]
    {
        // Image processing dependencies are compile-time checked
        tracing::debug!("Image support validated");
    }
    
    #[cfg(feature = "audio-support")]
    {
        // Audio processing dependencies are compile-time checked
        tracing::debug!("Audio support validated");
    }
    
    #[cfg(feature = "video-support")]
    {
        // Video processing dependencies are compile-time checked
        tracing::debug!("Video support validated");
    }
    
    #[cfg(feature = "candle")]
    {
        // ML framework dependencies are compile-time checked
        tracing::debug!("Candle ML framework validated");
    }
    
    Ok(())
}

/// Configuration for the multimodal library.
#[derive(Debug, Clone)]
pub struct MultimodalConfig {
    /// Maximum content size in bytes for embedded data.
    pub max_embedded_size: u64,
    
    /// Default timeout for operations in seconds.
    pub default_timeout: u64,
    
    /// Enable automatic format detection.
    pub auto_format_detection: bool,
    
    /// Enable cross-modal operations.
    pub enable_cross_modal: bool,
    
    /// Default batch size for processing operations.
    pub default_batch_size: usize,
    
    /// Enable metrics collection.
    pub enable_metrics: bool,
    
    /// Enable progress reporting.
    pub enable_progress: bool,
    
    /// Custom configuration values.
    pub custom: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for MultimodalConfig {
    fn default() -> Self {
        Self {
            max_embedded_size: 10 * 1024 * 1024, // 10 MB
            default_timeout: 60,                  // 60 seconds
            auto_format_detection: true,
            enable_cross_modal: true,
            default_batch_size: 16,
            enable_metrics: false,
            enable_progress: false,
            custom: std::collections::HashMap::new(),
        }
    }
}

impl MultimodalConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the maximum embedded content size.
    pub fn with_max_embedded_size(mut self, size: u64) -> Self {
        self.max_embedded_size = size;
        self
    }
    
    /// Set the default timeout.
    pub fn with_default_timeout(mut self, timeout: u64) -> Self {
        self.default_timeout = timeout;
        self
    }
    
    /// Enable or disable automatic format detection.
    pub fn with_auto_format_detection(mut self, enabled: bool) -> Self {
        self.auto_format_detection = enabled;
        self
    }
    
    /// Enable or disable cross-modal operations.
    pub fn with_cross_modal(mut self, enabled: bool) -> Self {
        self.enable_cross_modal = enabled;
        self
    }
    
    /// Set the default batch size.
    pub fn with_default_batch_size(mut self, size: usize) -> Self {
        self.default_batch_size = size;
        self
    }
    
    /// Enable or disable metrics collection.
    pub fn with_metrics(mut self, enabled: bool) -> Self {
        self.enable_metrics = enabled;
        self
    }
    
    /// Enable or disable progress reporting.
    pub fn with_progress(mut self, enabled: bool) -> Self {
        self.enable_progress = enabled;
        self
    }
    
    /// Add a custom configuration value.
    pub fn with_custom<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.custom.insert(key.into(), value.into());
        self
    }
    
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.max_embedded_size == 0 {
            return Err(MultimodalError::validation("max_embedded_size must be greater than 0"));
        }
        
        if self.default_timeout == 0 {
            return Err(MultimodalError::validation("default_timeout must be greater than 0"));
        }
        
        if self.default_batch_size == 0 {
            return Err(MultimodalError::validation("default_batch_size must be greater than 0"));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let version = version();
        assert!(!version.is_empty());
        assert!(version.contains('.'));
    }

    #[test]
    fn test_build_info() {
        let info = build_info();
        assert!(info.contains_key("version"));
        assert!(info.contains_key("target"));
        assert!(info.contains_key("profile"));
    }

    #[test]
    fn test_multimodal_config() {
        let config = MultimodalConfig::new()
            .with_max_embedded_size(5 * 1024 * 1024)
            .with_default_timeout(30)
            .with_auto_format_detection(false)
            .with_metrics(true);

        assert_eq!(config.max_embedded_size, 5 * 1024 * 1024);
        assert_eq!(config.default_timeout, 30);
        assert!(!config.auto_format_detection);
        assert!(config.enable_metrics);
        
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let invalid_config = MultimodalConfig::new().with_max_embedded_size(0);
        assert!(invalid_config.validate().is_err());
        
        let valid_config = MultimodalConfig::new();
        assert!(valid_config.validate().is_ok());
    }
}
