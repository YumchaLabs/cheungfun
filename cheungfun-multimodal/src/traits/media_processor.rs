//! Media processing traits for different modalities.
//!
//! This module defines traits for processing media content including
//! format conversion, feature extraction, and content analysis.

use crate::types::{MediaContent, MediaFormat, ModalityType};
use crate::error::Result;
use async_trait::async_trait;
use std::collections::HashMap;

/// Trait for processing media content.
///
/// This trait provides a unified interface for processing different types
/// of media content including format conversion, feature extraction, and
/// content analysis.
#[async_trait]
pub trait MediaProcessor: Send + Sync + std::fmt::Debug {
    /// Get the supported input modalities.
    fn supported_input_modalities(&self) -> Vec<ModalityType>;

    /// Get the supported output modalities.
    fn supported_output_modalities(&self) -> Vec<ModalityType>;

    /// Get the supported input formats for a modality.
    fn supported_input_formats(&self, modality: ModalityType) -> Vec<MediaFormat>;

    /// Get the supported output formats for a modality.
    fn supported_output_formats(&self, modality: ModalityType) -> Vec<MediaFormat>;

    /// Process media content.
    ///
    /// # Arguments
    ///
    /// * `content` - The input media content
    /// * `options` - Processing options
    ///
    /// # Returns
    ///
    /// The processed media content.
    async fn process(&self, content: &MediaContent, options: &ProcessingOptions) -> Result<MediaContent>;

    /// Process multiple media contents in batch.
    async fn process_batch(
        &self,
        contents: Vec<&MediaContent>,
        options: &ProcessingOptions,
    ) -> Result<Vec<MediaContent>> {
        let mut results = Vec::new();
        for content in contents {
            results.push(self.process(content, options).await?);
        }
        Ok(results)
    }

    /// Extract features from media content.
    async fn extract_features(&self, content: &MediaContent) -> Result<HashMap<String, serde_json::Value>>;

    /// Extract text content from media (if applicable).
    async fn extract_text(&self, content: &MediaContent) -> Result<Option<String>>;

    /// Get metadata about the processor.
    fn metadata(&self) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();
        metadata.insert("name".to_string(), self.name().into());
        metadata.insert("supported_input_modalities".to_string(), 
                        self.supported_input_modalities().iter().map(|m| m.to_string()).collect::<Vec<_>>().into());
        metadata.insert("supported_output_modalities".to_string(), 
                        self.supported_output_modalities().iter().map(|m| m.to_string()).collect::<Vec<_>>().into());
        metadata
    }

    /// Get a human-readable name for this processor.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Check if the processor is healthy and ready.
    async fn health_check(&self) -> Result<()> {
        Ok(())
    }

    /// Validate input content before processing.
    fn validate_input(&self, content: &MediaContent) -> Result<()> {
        let modality = content.modality_type();
        if !self.supported_input_modalities().contains(&modality) {
            return Err(crate::error::MultimodalError::unsupported_modality(modality.to_string()));
        }
        
        if !self.supported_input_formats(modality).contains(&content.format) {
            return Err(crate::error::MultimodalError::unsupported_format(
                content.format.to_string(),
                modality.to_string(),
            ));
        }
        
        Ok(())
    }
}

/// Trait for format conversion between different media formats.
#[async_trait]
pub trait FormatConverter: MediaProcessor {
    /// Convert media content from one format to another.
    async fn convert_format(
        &self,
        content: &MediaContent,
        target_format: MediaFormat,
        options: &ConversionOptions,
    ) -> Result<MediaContent>;

    /// Get the conversion quality for a specific format pair.
    fn conversion_quality(&self, from: MediaFormat, to: MediaFormat) -> Option<f32>;

    /// Check if conversion between two formats is supported.
    fn supports_conversion(&self, from: MediaFormat, to: MediaFormat) -> bool {
        self.conversion_quality(from, to).is_some()
    }

    /// Get all supported conversion paths.
    fn supported_conversions(&self) -> Vec<(MediaFormat, MediaFormat)>;
}

/// Trait for extracting features from specific modalities.
#[async_trait]
pub trait FeatureExtractor: MediaProcessor {
    /// Extract modality-specific features.
    async fn extract_modality_features(
        &self,
        content: &MediaContent,
        modality: ModalityType,
    ) -> Result<HashMap<String, serde_json::Value>>;

    /// Get the list of extractable features for a modality.
    fn extractable_features(&self, modality: ModalityType) -> Vec<String>;

    /// Extract a specific feature by name.
    async fn extract_feature(
        &self,
        content: &MediaContent,
        feature_name: &str,
    ) -> Result<Option<serde_json::Value>>;
}

/// Trait for content analysis and understanding.
#[async_trait]
pub trait ContentAnalyzer: MediaProcessor {
    /// Analyze content and provide insights.
    async fn analyze_content(&self, content: &MediaContent) -> Result<ContentAnalysis>;

    /// Detect objects, entities, or concepts in the content.
    async fn detect_entities(&self, content: &MediaContent) -> Result<Vec<DetectedEntity>>;

    /// Generate a description or summary of the content.
    async fn generate_description(&self, content: &MediaContent) -> Result<String>;

    /// Classify the content into categories.
    async fn classify_content(&self, content: &MediaContent) -> Result<Vec<ContentCategory>>;
}

/// Options for media processing operations.
#[derive(Debug, Clone)]
pub struct ProcessingOptions {
    /// Target output format (if conversion is needed).
    pub target_format: Option<MediaFormat>,

    /// Quality settings (0.0 to 1.0).
    pub quality: Option<f32>,

    /// Maximum output size in bytes.
    pub max_output_size: Option<u64>,

    /// Whether to preserve metadata.
    pub preserve_metadata: bool,

    /// Custom processing parameters.
    pub custom_params: HashMap<String, serde_json::Value>,
}

impl Default for ProcessingOptions {
    fn default() -> Self {
        Self {
            target_format: None,
            quality: Some(0.8),
            max_output_size: None,
            preserve_metadata: true,
            custom_params: HashMap::new(),
        }
    }
}

/// Options for format conversion operations.
#[derive(Debug, Clone)]
pub struct ConversionOptions {
    /// Quality settings (0.0 to 1.0).
    pub quality: f32,

    /// Whether to preserve original metadata.
    pub preserve_metadata: bool,

    /// Compression settings.
    pub compression: Option<CompressionSettings>,

    /// Custom conversion parameters.
    pub custom_params: HashMap<String, serde_json::Value>,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            quality: 0.8,
            preserve_metadata: true,
            compression: None,
            custom_params: HashMap::new(),
        }
    }
}

/// Compression settings for media conversion.
#[derive(Debug, Clone)]
pub struct CompressionSettings {
    /// Compression level (0-100).
    pub level: u8,

    /// Whether to use lossless compression.
    pub lossless: bool,

    /// Target bitrate (for audio/video).
    pub bitrate: Option<u32>,
}

/// Result of content analysis.
#[derive(Debug, Clone)]
pub struct ContentAnalysis {
    /// Confidence score of the analysis (0.0 to 1.0).
    pub confidence: f32,

    /// Primary content type detected.
    pub primary_type: String,

    /// Secondary content types.
    pub secondary_types: Vec<String>,

    /// Extracted features and their values.
    pub features: HashMap<String, serde_json::Value>,

    /// Content description or summary.
    pub description: Option<String>,

    /// Detected language (for text content).
    pub language: Option<String>,

    /// Content quality assessment.
    pub quality_score: Option<f32>,

    /// Additional analysis results.
    pub additional_data: HashMap<String, serde_json::Value>,
}

/// Detected entity in content.
#[derive(Debug, Clone)]
pub struct DetectedEntity {
    /// Entity type (person, object, concept, etc.).
    pub entity_type: String,

    /// Entity label or name.
    pub label: String,

    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,

    /// Bounding box (for visual content).
    pub bounding_box: Option<BoundingBox>,

    /// Time range (for temporal content).
    pub time_range: Option<TimeRange>,

    /// Additional entity attributes.
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Bounding box for spatial localization.
#[derive(Debug, Clone)]
pub struct BoundingBox {
    /// X coordinate of top-left corner.
    pub x: f32,

    /// Y coordinate of top-left corner.
    pub y: f32,

    /// Width of the bounding box.
    pub width: f32,

    /// Height of the bounding box.
    pub height: f32,
}

/// Time range for temporal localization.
#[derive(Debug, Clone)]
pub struct TimeRange {
    /// Start time in seconds.
    pub start: f32,

    /// End time in seconds.
    pub end: f32,
}

/// Content category classification.
#[derive(Debug, Clone)]
pub struct ContentCategory {
    /// Category name.
    pub name: String,

    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,

    /// Parent category (if hierarchical).
    pub parent: Option<String>,

    /// Category description.
    pub description: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_processing_options_default() {
        let options = ProcessingOptions::default();
        assert_eq!(options.quality, Some(0.8));
        assert!(options.preserve_metadata);
        assert!(options.target_format.is_none());
    }

    #[test]
    fn test_conversion_options_default() {
        let options = ConversionOptions::default();
        assert_eq!(options.quality, 0.8);
        assert!(options.preserve_metadata);
        assert!(options.compression.is_none());
    }

    #[test]
    fn test_bounding_box() {
        let bbox = BoundingBox {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 50.0,
        };
        assert_eq!(bbox.x, 10.0);
        assert_eq!(bbox.width, 100.0);
    }

    #[test]
    fn test_time_range() {
        let time_range = TimeRange {
            start: 1.5,
            end: 3.2,
        };
        assert_eq!(time_range.start, 1.5);
        assert_eq!(time_range.end, 3.2);
    }
}
