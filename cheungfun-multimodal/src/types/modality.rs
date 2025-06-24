//! Modality type definitions for multimodal content.
//!
//! This module defines the core types for representing different modalities
//! (text, image, audio, video) and their associated metadata.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported modality types in the multimodal framework.
///
/// Each modality represents a different type of content that can be processed,
/// embedded, and searched within the RAG system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalityType {
    /// Plain text content
    Text,

    /// Image content (photos, diagrams, charts, etc.)
    Image,

    /// Audio content (speech, music, sound effects, etc.)
    Audio,

    /// Video content (movies, presentations, animations, etc.)
    Video,

    /// Structured documents (PDF, Word, PowerPoint, etc.)
    Document,

    /// 3D models and spatial data
    #[cfg(feature = "3d-support")]
    ThreeD,

    /// Mixed or unknown modality
    Mixed,
}

impl ModalityType {
    /// Get all supported modality types.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Text,
            Self::Image,
            Self::Audio,
            Self::Video,
            Self::Document,
            #[cfg(feature = "3d-support")]
            Self::ThreeD,
            Self::Mixed,
        ]
    }

    /// Check if this modality supports text extraction.
    pub fn supports_text_extraction(&self) -> bool {
        matches!(self, Self::Text | Self::Document | Self::Video)
    }

    /// Check if this modality is visual.
    pub fn is_visual(&self) -> bool {
        matches!(self, Self::Image | Self::Video | Self::Document)
    }

    /// Check if this modality is temporal (has duration).
    pub fn is_temporal(&self) -> bool {
        matches!(self, Self::Audio | Self::Video)
    }

    /// Get the typical file extensions for this modality.
    pub fn typical_extensions(&self) -> Vec<&'static str> {
        match self {
            Self::Text => vec!["txt", "md", "rst"],
            Self::Image => vec!["jpg", "jpeg", "png", "gif", "webp", "svg", "bmp"],
            Self::Audio => vec!["mp3", "wav", "flac", "ogg", "m4a", "aac"],
            Self::Video => vec!["mp4", "avi", "mkv", "webm", "mov", "wmv"],
            Self::Document => vec!["pdf", "docx", "pptx", "xlsx", "odt"],
            #[cfg(feature = "3d-support")]
            Self::ThreeD => vec!["obj", "stl", "ply", "gltf", "fbx"],
            Self::Mixed => vec![],
        }
    }
}

impl fmt::Display for ModalityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Image => write!(f, "image"),
            Self::Audio => write!(f, "audio"),
            Self::Video => write!(f, "video"),
            Self::Document => write!(f, "document"),
            #[cfg(feature = "3d-support")]
            Self::ThreeD => write!(f, "3d"),
            Self::Mixed => write!(f, "mixed"),
        }
    }
}

impl std::str::FromStr for ModalityType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "image" => Ok(Self::Image),
            "audio" => Ok(Self::Audio),
            "video" => Ok(Self::Video),
            "document" => Ok(Self::Document),
            #[cfg(feature = "3d-support")]
            "3d" | "threed" => Ok(Self::ThreeD),
            "mixed" => Ok(Self::Mixed),
            _ => Err(anyhow::anyhow!("Unknown modality type: {}", s)),
        }
    }
}

/// Media format enumeration for different file types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MediaFormat {
    // Image formats
    Jpeg,
    Png,
    Gif,
    WebP,
    Svg,
    Bmp,
    Tiff,
    Ico,

    // Audio formats
    Mp3,
    Wav,
    Flac,
    Ogg,
    M4a,
    Aac,
    Wma,

    // Video formats
    Mp4,
    Avi,
    Mkv,
    WebM,
    Mov,
    Wmv,
    Flv,

    // Document formats
    Pdf,
    Docx,
    Doc,
    Pptx,
    Ppt,
    Xlsx,
    Xls,
    Odt,
    Rtf,

    // Text formats
    PlainText,
    Markdown,
    Html,
    Xml,
    Json,
    Yaml,
    Csv,

    // 3D formats
    #[cfg(feature = "3d-support")]
    Obj,
    #[cfg(feature = "3d-support")]
    Stl,
    #[cfg(feature = "3d-support")]
    Ply,
    #[cfg(feature = "3d-support")]
    Gltf,
    #[cfg(feature = "3d-support")]
    Fbx,

    // Unknown or custom format
    Unknown(String),
}

impl MediaFormat {
    /// Get the modality type for this format.
    pub fn modality_type(&self) -> ModalityType {
        match self {
            Self::Jpeg
            | Self::Png
            | Self::Gif
            | Self::WebP
            | Self::Svg
            | Self::Bmp
            | Self::Tiff
            | Self::Ico => ModalityType::Image,

            Self::Mp3 | Self::Wav | Self::Flac | Self::Ogg | Self::M4a | Self::Aac | Self::Wma => {
                ModalityType::Audio
            }

            Self::Mp4 | Self::Avi | Self::Mkv | Self::WebM | Self::Mov | Self::Wmv | Self::Flv => {
                ModalityType::Video
            }

            Self::Pdf
            | Self::Docx
            | Self::Doc
            | Self::Pptx
            | Self::Ppt
            | Self::Xlsx
            | Self::Xls
            | Self::Odt
            | Self::Rtf => ModalityType::Document,

            Self::PlainText
            | Self::Markdown
            | Self::Html
            | Self::Xml
            | Self::Json
            | Self::Yaml
            | Self::Csv => ModalityType::Text,

            #[cfg(feature = "3d-support")]
            Self::Obj | Self::Stl | Self::Ply | Self::Gltf | Self::Fbx => ModalityType::ThreeD,

            Self::Unknown(_) => ModalityType::Mixed,
        }
    }

    /// Get the typical file extension for this format.
    pub fn extension(&self) -> &str {
        match self {
            Self::Jpeg => "jpg",
            Self::Png => "png",
            Self::Gif => "gif",
            Self::WebP => "webp",
            Self::Svg => "svg",
            Self::Bmp => "bmp",
            Self::Tiff => "tiff",
            Self::Ico => "ico",

            Self::Mp3 => "mp3",
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Ogg => "ogg",
            Self::M4a => "m4a",
            Self::Aac => "aac",
            Self::Wma => "wma",

            Self::Mp4 => "mp4",
            Self::Avi => "avi",
            Self::Mkv => "mkv",
            Self::WebM => "webm",
            Self::Mov => "mov",
            Self::Wmv => "wmv",
            Self::Flv => "flv",

            Self::Pdf => "pdf",
            Self::Docx => "docx",
            Self::Doc => "doc",
            Self::Pptx => "pptx",
            Self::Ppt => "ppt",
            Self::Xlsx => "xlsx",
            Self::Xls => "xls",
            Self::Odt => "odt",
            Self::Rtf => "rtf",

            Self::PlainText => "txt",
            Self::Markdown => "md",
            Self::Html => "html",
            Self::Xml => "xml",
            Self::Json => "json",
            Self::Yaml => "yaml",
            Self::Csv => "csv",

            #[cfg(feature = "3d-support")]
            Self::Obj => "obj",
            #[cfg(feature = "3d-support")]
            Self::Stl => "stl",
            #[cfg(feature = "3d-support")]
            Self::Ply => "ply",
            #[cfg(feature = "3d-support")]
            Self::Gltf => "gltf",
            #[cfg(feature = "3d-support")]
            Self::Fbx => "fbx",

            Self::Unknown(ext) => ext,
        }
    }

    /// Get the MIME type for this format.
    pub fn mime_type(&self) -> &str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Gif => "image/gif",
            Self::WebP => "image/webp",
            Self::Svg => "image/svg+xml",
            Self::Bmp => "image/bmp",
            Self::Tiff => "image/tiff",
            Self::Ico => "image/x-icon",

            Self::Mp3 => "audio/mpeg",
            Self::Wav => "audio/wav",
            Self::Flac => "audio/flac",
            Self::Ogg => "audio/ogg",
            Self::M4a => "audio/mp4",
            Self::Aac => "audio/aac",
            Self::Wma => "audio/x-ms-wma",

            Self::Mp4 => "video/mp4",
            Self::Avi => "video/x-msvideo",
            Self::Mkv => "video/x-matroska",
            Self::WebM => "video/webm",
            Self::Mov => "video/quicktime",
            Self::Wmv => "video/x-ms-wmv",
            Self::Flv => "video/x-flv",

            Self::Pdf => "application/pdf",
            Self::Docx => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            Self::Doc => "application/msword",
            Self::Pptx => {
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            }
            Self::Ppt => "application/vnd.ms-powerpoint",
            Self::Xlsx => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            Self::Xls => "application/vnd.ms-excel",
            Self::Odt => "application/vnd.oasis.opendocument.text",
            Self::Rtf => "application/rtf",

            Self::PlainText => "text/plain",
            Self::Markdown => "text/markdown",
            Self::Html => "text/html",
            Self::Xml => "application/xml",
            Self::Json => "application/json",
            Self::Yaml => "application/x-yaml",
            Self::Csv => "text/csv",

            #[cfg(feature = "3d-support")]
            Self::Obj => "model/obj",
            #[cfg(feature = "3d-support")]
            Self::Stl => "model/stl",
            #[cfg(feature = "3d-support")]
            Self::Ply => "model/ply",
            #[cfg(feature = "3d-support")]
            Self::Gltf => "model/gltf+json",
            #[cfg(feature = "3d-support")]
            Self::Fbx => "model/fbx",

            Self::Unknown(_) => "application/octet-stream",
        }
    }
}

impl fmt::Display for MediaFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.extension())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modality_type_display() {
        assert_eq!(ModalityType::Text.to_string(), "text");
        assert_eq!(ModalityType::Image.to_string(), "image");
        assert_eq!(ModalityType::Audio.to_string(), "audio");
        assert_eq!(ModalityType::Video.to_string(), "video");
    }

    #[test]
    fn test_modality_type_from_str() {
        assert_eq!("text".parse::<ModalityType>().unwrap(), ModalityType::Text);
        assert_eq!(
            "IMAGE".parse::<ModalityType>().unwrap(),
            ModalityType::Image
        );
        assert!("invalid".parse::<ModalityType>().is_err());
    }

    #[test]
    fn test_media_format_modality_mapping() {
        assert_eq!(MediaFormat::Jpeg.modality_type(), ModalityType::Image);
        assert_eq!(MediaFormat::Mp3.modality_type(), ModalityType::Audio);
        assert_eq!(MediaFormat::Mp4.modality_type(), ModalityType::Video);
        assert_eq!(MediaFormat::Pdf.modality_type(), ModalityType::Document);
        assert_eq!(MediaFormat::PlainText.modality_type(), ModalityType::Text);
    }

    #[test]
    fn test_modality_properties() {
        assert!(ModalityType::Text.supports_text_extraction());
        assert!(ModalityType::Image.is_visual());
        assert!(ModalityType::Audio.is_temporal());
        assert!(ModalityType::Video.is_visual() && ModalityType::Video.is_temporal());
    }
}
