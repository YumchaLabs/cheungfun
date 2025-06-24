//! Core types for multimodal content processing.
//!
//! This module provides the fundamental data structures for representing
//! and working with multimodal content in the Cheungfun framework.

pub mod media;
pub mod modality;
pub mod multimodal_node;

// Re-export commonly used types
pub use media::{MediaContent, MediaContentBuilder, MediaData, MediaMetadata};
pub use modality::{MediaFormat, ModalityType};
pub use multimodal_node::{MultimodalDocument, MultimodalNode, MultimodalNodeBuilder};

/// Common result type for multimodal operations.
pub type Result<T> = std::result::Result<T, crate::error::MultimodalError>;

/// Trait for types that have an associated modality.
pub trait HasModality {
    /// Get the primary modality type of this item.
    fn modality_type(&self) -> ModalityType;

    /// Check if this item supports a specific modality.
    fn supports_modality(&self, modality: ModalityType) -> bool {
        self.modality_type() == modality
    }
}

impl HasModality for MediaContent {
    fn modality_type(&self) -> ModalityType {
        self.format.modality_type()
    }
}

impl HasModality for MultimodalDocument {
    fn modality_type(&self) -> ModalityType {
        self.primary_modality
    }

    fn supports_modality(&self, modality: ModalityType) -> bool {
        self.has_modality(modality)
    }
}

impl HasModality for MultimodalNode {
    fn modality_type(&self) -> ModalityType {
        self.modality
    }

    fn supports_modality(&self, modality: ModalityType) -> bool {
        self.modality == modality || self.cross_modal_relations.contains_key(&modality)
    }
}

/// Trait for types that can be converted to different modalities.
pub trait ModalityConversion {
    /// Convert content to a different modality if possible.
    fn convert_to_modality(&self, target: ModalityType) -> Result<Self>
    where
        Self: Sized;

    /// Check if conversion to a specific modality is supported.
    fn can_convert_to(&self, target: ModalityType) -> bool;

    /// Get all supported conversion targets.
    fn supported_conversions(&self) -> Vec<ModalityType>;
}

/// Trait for extracting text content from multimodal data.
pub trait TextExtraction {
    /// Extract text content from the multimodal data.
    fn extract_text(&self) -> Option<String>;

    /// Check if text extraction is supported for this content.
    fn supports_text_extraction(&self) -> bool;
}

impl TextExtraction for MediaContent {
    fn extract_text(&self) -> Option<String> {
        self.extracted_text.clone()
    }

    fn supports_text_extraction(&self) -> bool {
        self.modality_type().supports_text_extraction()
    }
}

impl TextExtraction for MultimodalDocument {
    fn extract_text(&self) -> Option<String> {
        // First try the base document content
        if !self.base.content.is_empty() {
            return Some(self.base.content.clone());
        }

        // Then try media content
        if let Some(ref media) = self.media_content {
            if let Some(text) = media.extract_text() {
                return Some(text);
            }
        }

        // Finally try related media
        for media in &self.related_media {
            if let Some(text) = media.extract_text() {
                return Some(text);
            }
        }

        None
    }

    fn supports_text_extraction(&self) -> bool {
        // Check if any modality supports text extraction
        self.get_modalities()
            .iter()
            .any(|m| m.supports_text_extraction())
    }
}

impl TextExtraction for MultimodalNode {
    fn extract_text(&self) -> Option<String> {
        // First try the base node content
        if !self.base.content.is_empty() {
            return Some(self.base.content.clone());
        }

        // Then try media content
        if let Some(ref media) = self.media_content {
            return media.extract_text();
        }

        None
    }

    fn supports_text_extraction(&self) -> bool {
        self.modality.supports_text_extraction()
    }
}

/// Utility functions for working with multimodal content.
pub mod utils {
    use super::*;

    /// Detect the modality type from a file extension.
    pub fn detect_modality_from_extension(extension: &str) -> Option<ModalityType> {
        let ext = extension.to_lowercase();

        for modality in ModalityType::all() {
            if modality.typical_extensions().contains(&ext.as_str()) {
                return Some(modality);
            }
        }

        None
    }

    /// Detect the media format from a file extension.
    pub fn detect_format_from_extension(extension: &str) -> Option<MediaFormat> {
        match extension.to_lowercase().as_str() {
            "jpg" | "jpeg" => Some(MediaFormat::Jpeg),
            "png" => Some(MediaFormat::Png),
            "gif" => Some(MediaFormat::Gif),
            "webp" => Some(MediaFormat::WebP),
            "svg" => Some(MediaFormat::Svg),
            "bmp" => Some(MediaFormat::Bmp),
            "tiff" | "tif" => Some(MediaFormat::Tiff),
            "ico" => Some(MediaFormat::Ico),

            "mp3" => Some(MediaFormat::Mp3),
            "wav" => Some(MediaFormat::Wav),
            "flac" => Some(MediaFormat::Flac),
            "ogg" => Some(MediaFormat::Ogg),
            "m4a" => Some(MediaFormat::M4a),
            "aac" => Some(MediaFormat::Aac),
            "wma" => Some(MediaFormat::Wma),

            "mp4" => Some(MediaFormat::Mp4),
            "avi" => Some(MediaFormat::Avi),
            "mkv" => Some(MediaFormat::Mkv),
            "webm" => Some(MediaFormat::WebM),
            "mov" => Some(MediaFormat::Mov),
            "wmv" => Some(MediaFormat::Wmv),
            "flv" => Some(MediaFormat::Flv),

            "pdf" => Some(MediaFormat::Pdf),
            "docx" => Some(MediaFormat::Docx),
            "doc" => Some(MediaFormat::Doc),
            "pptx" => Some(MediaFormat::Pptx),
            "ppt" => Some(MediaFormat::Ppt),
            "xlsx" => Some(MediaFormat::Xlsx),
            "xls" => Some(MediaFormat::Xls),
            "odt" => Some(MediaFormat::Odt),
            "rtf" => Some(MediaFormat::Rtf),

            "txt" => Some(MediaFormat::PlainText),
            "md" => Some(MediaFormat::Markdown),
            "html" | "htm" => Some(MediaFormat::Html),
            "xml" => Some(MediaFormat::Xml),
            "json" => Some(MediaFormat::Json),
            "yaml" | "yml" => Some(MediaFormat::Yaml),
            "csv" => Some(MediaFormat::Csv),

            _ => None,
        }
    }

    /// Check if two modalities are compatible for cross-modal operations.
    pub fn are_modalities_compatible(source: ModalityType, target: ModalityType) -> bool {
        match (source, target) {
            // Same modality is always compatible
            (a, b) if a == b => true,

            // Text can be associated with any modality
            (ModalityType::Text, _) | (_, ModalityType::Text) => true,

            // Visual modalities are compatible
            (ModalityType::Image, ModalityType::Video)
            | (ModalityType::Video, ModalityType::Image) => true,
            (ModalityType::Image, ModalityType::Document)
            | (ModalityType::Document, ModalityType::Image) => true,
            (ModalityType::Video, ModalityType::Document)
            | (ModalityType::Document, ModalityType::Video) => true,

            // Audio and video are compatible (video contains audio)
            (ModalityType::Audio, ModalityType::Video)
            | (ModalityType::Video, ModalityType::Audio) => true,

            // Mixed modality is compatible with everything
            (ModalityType::Mixed, _) | (_, ModalityType::Mixed) => true,

            // Other combinations are not directly compatible
            _ => false,
        }
    }

    /// Get the priority order for modalities (for conflict resolution).
    pub fn modality_priority(modality: ModalityType) -> u8 {
        match modality {
            ModalityType::Text => 1,     // Highest priority (most universal)
            ModalityType::Document => 2, // High priority (structured)
            ModalityType::Image => 3,    // Medium priority
            ModalityType::Video => 4,    // Medium priority (complex)
            ModalityType::Audio => 5,    // Lower priority
            #[cfg(feature = "3d-support")]
            ModalityType::ThreeD => 6, // Specialized
            ModalityType::Mixed => 7,    // Lowest priority (ambiguous)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::utils::*;
    use super::*;

    #[test]
    fn test_modality_detection() {
        assert_eq!(
            detect_modality_from_extension("jpg"),
            Some(ModalityType::Image)
        );
        assert_eq!(
            detect_modality_from_extension("mp3"),
            Some(ModalityType::Audio)
        );
        assert_eq!(
            detect_modality_from_extension("mp4"),
            Some(ModalityType::Video)
        );
        assert_eq!(
            detect_modality_from_extension("pdf"),
            Some(ModalityType::Document)
        );
        assert_eq!(
            detect_modality_from_extension("txt"),
            Some(ModalityType::Text)
        );
        assert_eq!(detect_modality_from_extension("unknown"), None);
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(detect_format_from_extension("jpg"), Some(MediaFormat::Jpeg));
        assert_eq!(detect_format_from_extension("PNG"), Some(MediaFormat::Png));
        assert_eq!(detect_format_from_extension("unknown"), None);
    }

    #[test]
    fn test_modality_compatibility() {
        assert!(are_modalities_compatible(
            ModalityType::Text,
            ModalityType::Image
        ));
        assert!(are_modalities_compatible(
            ModalityType::Image,
            ModalityType::Video
        ));
        assert!(are_modalities_compatible(
            ModalityType::Audio,
            ModalityType::Video
        ));
        assert!(!are_modalities_compatible(
            ModalityType::Image,
            ModalityType::Audio
        ));
    }

    #[test]
    fn test_modality_priority() {
        assert!(modality_priority(ModalityType::Text) < modality_priority(ModalityType::Image));
        assert!(modality_priority(ModalityType::Document) < modality_priority(ModalityType::Video));
    }
}
