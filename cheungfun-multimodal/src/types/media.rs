//! Media content data structures for multimodal processing.
//!
//! This module defines the core data structures for representing media content
//! including binary data, metadata, and various storage formats.

use crate::types::modality::{MediaFormat, ModalityType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Unified representation of media content across different modalities.
///
/// This structure can represent any type of media content (text, image, audio, video)
/// with associated metadata and features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaContent {
    /// The actual media data
    pub data: MediaData,

    /// Format of the media content
    pub format: MediaFormat,

    /// Media metadata (file info, creation time, etc.)
    pub metadata: MediaMetadata,

    /// Extracted text content (if applicable)
    pub extracted_text: Option<String>,

    /// Media-specific features and properties
    pub features: HashMap<String, serde_json::Value>,

    /// Checksum for content verification and deduplication
    pub checksum: Option<String>,
}

impl MediaContent {
    /// Create new media content from binary data.
    pub fn from_bytes(data: Vec<u8>, format: MediaFormat) -> Self {
        Self {
            data: MediaData::Embedded(data),
            format,
            metadata: MediaMetadata::default(),
            extracted_text: None,
            features: HashMap::new(),
            checksum: None,
        }
    }

    /// Create new media content from file path.
    pub fn from_file_path<P: Into<PathBuf>>(path: P, format: MediaFormat) -> Self {
        let path = path.into();
        let mut metadata = MediaMetadata::default();
        metadata.filename = path.file_name().and_then(|n| n.to_str()).map(String::from);

        Self {
            data: MediaData::FilePath(path),
            format,
            metadata,
            extracted_text: None,
            features: HashMap::new(),
            checksum: None,
        }
    }

    /// Create new media content from URL.
    pub fn from_url<S: Into<String>>(url: S, format: MediaFormat) -> Self {
        Self {
            data: MediaData::Url(url.into()),
            format,
            metadata: MediaMetadata::default(),
            extracted_text: None,
            features: HashMap::new(),
            checksum: None,
        }
    }

    /// Create new media content from base64 encoded data.
    pub fn from_base64<S: Into<String>>(data: S, format: MediaFormat) -> Self {
        Self {
            data: MediaData::Base64(data.into()),
            format,
            metadata: MediaMetadata::default(),
            extracted_text: None,
            features: HashMap::new(),
            checksum: None,
        }
    }

    /// Get the modality type of this media content.
    pub fn modality_type(&self) -> ModalityType {
        self.format.modality_type()
    }

    /// Check if this media content has embedded binary data.
    pub fn has_embedded_data(&self) -> bool {
        matches!(self.data, MediaData::Embedded(_) | MediaData::Base64(_))
    }

    /// Check if this media content references external resources.
    pub fn is_external(&self) -> bool {
        matches!(self.data, MediaData::FilePath(_) | MediaData::Url(_))
    }

    /// Get the estimated size of the media content in bytes.
    pub fn estimated_size(&self) -> Option<u64> {
        match &self.data {
            MediaData::Embedded(data) => Some(data.len() as u64),
            MediaData::Base64(data) => Some((data.len() * 3 / 4) as u64), // Approximate
            MediaData::FilePath(_) | MediaData::Url(_) => self.metadata.size,
        }
    }

    /// Add a feature to the media content.
    pub fn with_feature<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.features.insert(key.into(), value.into());
        self
    }

    /// Add extracted text to the media content.
    pub fn with_extracted_text<S: Into<String>>(mut self, text: S) -> Self {
        self.extracted_text = Some(text.into());
        self
    }

    /// Add metadata to the media content.
    pub fn with_metadata(mut self, metadata: MediaMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set the checksum for the media content.
    pub fn with_checksum<S: Into<String>>(mut self, checksum: S) -> Self {
        self.checksum = Some(checksum.into());
        self
    }
}

/// Different ways to store media data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MediaData {
    /// Binary data embedded directly in the structure (for small files)
    Embedded(Vec<u8>),

    /// Reference to a local file path (for large files)
    FilePath(PathBuf),

    /// Reference to a remote URL
    Url(String),

    /// Base64 encoded data (for API compatibility)
    Base64(String),
}

impl MediaData {
    /// Get a human-readable description of the data storage type.
    pub fn storage_type(&self) -> &'static str {
        match self {
            Self::Embedded(_) => "embedded",
            Self::FilePath(_) => "file",
            Self::Url(_) => "url",
            Self::Base64(_) => "base64",
        }
    }

    /// Check if the data is stored locally.
    pub fn is_local(&self) -> bool {
        matches!(
            self,
            Self::Embedded(_) | Self::FilePath(_) | Self::Base64(_)
        )
    }

    /// Get the data as bytes if available locally.
    pub async fn as_bytes(&self) -> anyhow::Result<Vec<u8>> {
        match self {
            Self::Embedded(data) => Ok(data.clone()),
            Self::Base64(data) => {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .map_err(|e| anyhow::anyhow!("Base64 decode error: {}", e))
            }
            Self::FilePath(path) => tokio::fs::read(path)
                .await
                .map_err(|e| anyhow::anyhow!("File read error: {}", e)),
            Self::Url(_) => {
                #[cfg(feature = "remote-resources")]
                {
                    // This would require implementing HTTP client functionality
                    Err(anyhow::anyhow!("URL data fetching not implemented yet"))
                }
                #[cfg(not(feature = "remote-resources"))]
                {
                    Err(anyhow::anyhow!("Remote resource support not enabled"))
                }
            }
        }
    }
}

/// Metadata associated with media content.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MediaMetadata {
    /// File size in bytes
    pub size: Option<u64>,

    /// Creation timestamp
    pub created_at: Option<DateTime<Utc>>,

    /// Last modification timestamp
    pub modified_at: Option<DateTime<Utc>>,

    /// MIME type
    pub mime_type: Option<String>,

    /// Original filename
    pub filename: Option<String>,

    /// Content description or title
    pub title: Option<String>,

    /// Content author or creator
    pub author: Option<String>,

    /// Content description
    pub description: Option<String>,

    /// Content tags or keywords
    pub tags: Vec<String>,

    /// Content language (ISO 639-1 code)
    pub language: Option<String>,

    /// Copyright information
    pub copyright: Option<String>,

    /// Additional format-specific metadata
    pub extra: HashMap<String, serde_json::Value>,
}

impl MediaMetadata {
    /// Create a new metadata instance with current timestamp.
    pub fn new() -> Self {
        Self {
            created_at: Some(Utc::now()),
            ..Default::default()
        }
    }

    /// Add a tag to the metadata.
    pub fn with_tag<S: Into<String>>(mut self, tag: S) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add multiple tags to the metadata.
    pub fn with_tags<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tags.extend(tags.into_iter().map(|t| t.into()));
        self
    }

    /// Set the title of the content.
    pub fn with_title<S: Into<String>>(mut self, title: S) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the author of the content.
    pub fn with_author<S: Into<String>>(mut self, author: S) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set the description of the content.
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the language of the content.
    pub fn with_language<S: Into<String>>(mut self, language: S) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Add extra metadata.
    pub fn with_extra<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// Builder for creating media content with a fluent API.
#[derive(Debug, Default)]
pub struct MediaContentBuilder {
    data: Option<MediaData>,
    format: Option<MediaFormat>,
    metadata: MediaMetadata,
    extracted_text: Option<String>,
    features: HashMap<String, serde_json::Value>,
    checksum: Option<String>,
}

impl MediaContentBuilder {
    /// Create a new media content builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the media data.
    pub fn data(mut self, data: MediaData) -> Self {
        self.data = Some(data);
        self
    }

    /// Set the media format.
    pub fn format(mut self, format: MediaFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Set the metadata.
    pub fn metadata(mut self, metadata: MediaMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set extracted text.
    pub fn extracted_text<S: Into<String>>(mut self, text: S) -> Self {
        self.extracted_text = Some(text.into());
        self
    }

    /// Add a feature.
    pub fn feature<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.features.insert(key.into(), value.into());
        self
    }

    /// Set the checksum.
    pub fn checksum<S: Into<String>>(mut self, checksum: S) -> Self {
        self.checksum = Some(checksum.into());
        self
    }

    /// Build the media content.
    pub fn build(self) -> anyhow::Result<MediaContent> {
        let data = self
            .data
            .ok_or_else(|| anyhow::anyhow!("Media data is required"))?;
        let format = self
            .format
            .ok_or_else(|| anyhow::anyhow!("Media format is required"))?;

        Ok(MediaContent {
            data,
            format,
            metadata: self.metadata,
            extracted_text: self.extracted_text,
            features: self.features,
            checksum: self.checksum,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_media_content_creation() {
        let content = MediaContent::from_bytes(vec![1, 2, 3], MediaFormat::Jpeg);
        assert_eq!(content.modality_type(), ModalityType::Image);
        assert!(content.has_embedded_data());
        assert!(!content.is_external());
        assert_eq!(content.estimated_size(), Some(3));
    }

    #[test]
    fn test_media_content_builder() {
        let content = MediaContentBuilder::new()
            .data(MediaData::Embedded(vec![1, 2, 3]))
            .format(MediaFormat::Mp3)
            .extracted_text("test audio")
            .feature("duration", 120)
            .build()
            .unwrap();

        assert_eq!(content.modality_type(), ModalityType::Audio);
        assert_eq!(content.extracted_text, Some("test audio".to_string()));
        assert_eq!(
            content.features.get("duration"),
            Some(&serde_json::Value::Number(120.into()))
        );
    }

    #[test]
    fn test_media_metadata_builder() {
        let metadata = MediaMetadata::new()
            .with_title("Test Content")
            .with_author("Test Author")
            .with_tags(vec!["tag1", "tag2"])
            .with_language("en");

        assert_eq!(metadata.title, Some("Test Content".to_string()));
        assert_eq!(metadata.author, Some("Test Author".to_string()));
        assert_eq!(metadata.tags, vec!["tag1", "tag2"]);
        assert_eq!(metadata.language, Some("en".to_string()));
    }
}
