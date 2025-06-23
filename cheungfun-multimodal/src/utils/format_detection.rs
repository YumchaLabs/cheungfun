//! Format detection utilities for multimodal content.
//!
//! This module provides functions to automatically detect media formats
//! and modalities from file content, extensions, and MIME types.

use crate::types::{MediaFormat, ModalityType};
use crate::error::Result;
use std::path::Path;

/// Detect media format from file content (magic bytes).
#[cfg(feature = "format-detection")]
pub fn detect_format_from_bytes(data: &[u8]) -> Option<MediaFormat> {
    if let Some(kind) = infer::get(data) {
        match kind.mime_type() {
            // Image formats
            "image/jpeg" => Some(MediaFormat::Jpeg),
            "image/png" => Some(MediaFormat::Png),
            "image/gif" => Some(MediaFormat::Gif),
            "image/webp" => Some(MediaFormat::WebP),
            "image/bmp" => Some(MediaFormat::Bmp),
            "image/tiff" => Some(MediaFormat::Tiff),
            "image/x-icon" => Some(MediaFormat::Ico),

            // Audio formats
            "audio/mpeg" => Some(MediaFormat::Mp3),
            "audio/wav" => Some(MediaFormat::Wav),
            "audio/flac" => Some(MediaFormat::Flac),
            "audio/ogg" => Some(MediaFormat::Ogg),
            "audio/mp4" => Some(MediaFormat::M4a),
            "audio/aac" => Some(MediaFormat::Aac),

            // Video formats
            "video/mp4" => Some(MediaFormat::Mp4),
            "video/x-msvideo" => Some(MediaFormat::Avi),
            "video/x-matroska" => Some(MediaFormat::Mkv),
            "video/webm" => Some(MediaFormat::WebM),
            "video/quicktime" => Some(MediaFormat::Mov),
            "video/x-flv" => Some(MediaFormat::Flv),

            // Document formats
            "application/pdf" => Some(MediaFormat::Pdf),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => Some(MediaFormat::Docx),
            "application/msword" => Some(MediaFormat::Doc),
            "application/vnd.openxmlformats-officedocument.presentationml.presentation" => Some(MediaFormat::Pptx),
            "application/vnd.ms-powerpoint" => Some(MediaFormat::Ppt),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => Some(MediaFormat::Xlsx),
            "application/vnd.ms-excel" => Some(MediaFormat::Xls),

            _ => None,
        }
    } else {
        // Try to detect text formats by content analysis
        detect_text_format_from_content(data)
    }
}

/// Detect format from file extension.
pub fn detect_format_from_extension(extension: &str) -> Option<MediaFormat> {
    crate::types::utils::detect_format_from_extension(extension)
}

/// Detect format from file path.
pub fn detect_format_from_path<P: AsRef<Path>>(path: P) -> Option<MediaFormat> {
    path.as_ref()
        .extension()
        .and_then(|ext| ext.to_str())
        .and_then(detect_format_from_extension)
}

/// Detect modality from file path.
pub fn detect_modality_from_path<P: AsRef<Path>>(path: P) -> Option<ModalityType> {
    detect_format_from_path(path).map(|format| format.modality_type())
}

/// Comprehensive format detection using multiple methods.
pub async fn detect_format(data: &[u8], filename: Option<&str>) -> MediaFormat {
    // First try magic bytes detection
    #[cfg(feature = "format-detection")]
    if let Some(format) = detect_format_from_bytes(data) {
        return format;
    }
    
    // Then try filename extension
    if let Some(filename) = filename {
        if let Some(format) = detect_format_from_path(filename) {
            return format;
        }
    }
    
    // Fallback to content analysis
    if let Some(format) = detect_text_format_from_content(data) {
        return format;
    }
    
    // Default to unknown
    MediaFormat::Unknown("unknown".to_string())
}

/// Detect text format from content analysis.
fn detect_text_format_from_content(data: &[u8]) -> Option<MediaFormat> {
    // Check if it's valid UTF-8 text
    if let Ok(text) = std::str::from_utf8(data) {
        let text = text.trim();
        
        // Check for specific text formats
        if text.starts_with("<!DOCTYPE html") || text.starts_with("<html") {
            return Some(MediaFormat::Html);
        }
        
        if text.starts_with("<?xml") {
            return Some(MediaFormat::Xml);
        }
        
        if text.starts_with('{') && text.ends_with('}') {
            // Try to parse as JSON
            if serde_json::from_str::<serde_json::Value>(text).is_ok() {
                return Some(MediaFormat::Json);
            }
        }
        
        // Check for YAML indicators
        if text.contains("---") || text.contains(": ") {
            return Some(MediaFormat::Yaml);
        }
        
        // Check for Markdown indicators
        if text.contains("# ") || text.contains("## ") || text.contains("```") {
            return Some(MediaFormat::Markdown);
        }
        
        // Check for CSV (simple heuristic)
        let lines: Vec<&str> = text.lines().take(5).collect();
        if lines.len() > 1 {
            let first_line_commas = lines[0].matches(',').count();
            if first_line_commas > 0 && lines.iter().all(|line| line.matches(',').count() == first_line_commas) {
                return Some(MediaFormat::Csv);
            }
        }
        
        // Default to plain text
        Some(MediaFormat::PlainText)
    } else {
        None
    }
}

/// Validate that detected format matches expected modality.
pub fn validate_format_modality(format: MediaFormat, expected_modality: ModalityType) -> bool {
    format.modality_type() == expected_modality
}

/// Get confidence score for format detection.
pub fn format_detection_confidence(data: &[u8], detected_format: &MediaFormat) -> f32 {
    match detected_format {
        // High confidence for formats with magic bytes
        MediaFormat::Jpeg | MediaFormat::Png | MediaFormat::Gif | MediaFormat::WebP
        | MediaFormat::Mp3 | MediaFormat::Mp4 | MediaFormat::Pdf => {
            #[cfg(feature = "format-detection")]
            {
                if detect_format_from_bytes(data).is_some() {
                    0.95
                } else {
                    0.3
                }
            }
            #[cfg(not(feature = "format-detection"))]
            {
                0.5
            }
        }
        
        // Medium confidence for text formats
        MediaFormat::PlainText | MediaFormat::Html | MediaFormat::Json | MediaFormat::Xml => 0.7,
        
        // Lower confidence for formats without clear indicators
        MediaFormat::Csv | MediaFormat::Yaml | MediaFormat::Markdown => 0.6,
        
        // Very low confidence for unknown formats
        MediaFormat::Unknown(_) => 0.1,
        
        // Default medium confidence
        _ => 0.5,
    }
}

/// Format detection statistics.
#[derive(Debug, Clone)]
pub struct FormatDetectionStats {
    /// Total number of detections performed.
    pub total_detections: usize,
    
    /// Number of successful detections by format.
    pub detections_by_format: std::collections::HashMap<MediaFormat, usize>,
    
    /// Number of detections by modality.
    pub detections_by_modality: std::collections::HashMap<ModalityType, usize>,
    
    /// Average confidence scores.
    pub average_confidence: f32,
    
    /// Number of unknown format detections.
    pub unknown_detections: usize,
}

impl Default for FormatDetectionStats {
    fn default() -> Self {
        Self {
            total_detections: 0,
            detections_by_format: std::collections::HashMap::new(),
            detections_by_modality: std::collections::HashMap::new(),
            average_confidence: 0.0,
            unknown_detections: 0,
        }
    }
}

impl FormatDetectionStats {
    /// Create new format detection statistics.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Record a format detection.
    pub fn record_detection(&mut self, format: MediaFormat, confidence: f32) {
        self.total_detections += 1;
        
        // Update format counts
        *self.detections_by_format.entry(format.clone()).or_insert(0) += 1;
        
        // Update modality counts
        let modality = format.modality_type();
        *self.detections_by_modality.entry(modality).or_insert(0) += 1;
        
        // Update average confidence
        let total_confidence = self.average_confidence * (self.total_detections - 1) as f32 + confidence;
        self.average_confidence = total_confidence / self.total_detections as f32;
        
        // Count unknown detections
        if matches!(format, MediaFormat::Unknown(_)) {
            self.unknown_detections += 1;
        }
    }
    
    /// Get the success rate (non-unknown detections).
    pub fn success_rate(&self) -> f32 {
        if self.total_detections == 0 {
            0.0
        } else {
            (self.total_detections - self.unknown_detections) as f32 / self.total_detections as f32
        }
    }
    
    /// Get the most common format.
    pub fn most_common_format(&self) -> Option<MediaFormat> {
        self.detections_by_format
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(format, _)| format.clone())
    }
    
    /// Get the most common modality.
    pub fn most_common_modality(&self) -> Option<ModalityType> {
        self.detections_by_modality
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(modality, _)| *modality)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_from_extension() {
        assert_eq!(detect_format_from_extension("jpg"), Some(MediaFormat::Jpeg));
        assert_eq!(detect_format_from_extension("PNG"), Some(MediaFormat::Png));
        assert_eq!(detect_format_from_extension("mp3"), Some(MediaFormat::Mp3));
        assert_eq!(detect_format_from_extension("unknown"), None);
    }

    #[test]
    fn test_detect_format_from_path() {
        assert_eq!(detect_format_from_path("image.jpg"), Some(MediaFormat::Jpeg));
        assert_eq!(detect_format_from_path("/path/to/audio.mp3"), Some(MediaFormat::Mp3));
        assert_eq!(detect_format_from_path("document.pdf"), Some(MediaFormat::Pdf));
        assert_eq!(detect_format_from_path("no_extension"), None);
    }

    #[test]
    fn test_detect_modality_from_path() {
        assert_eq!(detect_modality_from_path("image.jpg"), Some(ModalityType::Image));
        assert_eq!(detect_modality_from_path("audio.mp3"), Some(ModalityType::Audio));
        assert_eq!(detect_modality_from_path("video.mp4"), Some(ModalityType::Video));
        assert_eq!(detect_modality_from_path("document.pdf"), Some(ModalityType::Document));
    }

    #[test]
    fn test_detect_text_format_from_content() {
        assert_eq!(detect_text_format_from_content(b"Hello, world!"), Some(MediaFormat::PlainText));
        assert_eq!(detect_text_format_from_content(b"<!DOCTYPE html><html></html>"), Some(MediaFormat::Html));
        assert_eq!(detect_text_format_from_content(b"<?xml version=\"1.0\"?>"), Some(MediaFormat::Xml));
        assert_eq!(detect_text_format_from_content(b"{\"key\": \"value\"}"), Some(MediaFormat::Json));
        assert_eq!(detect_text_format_from_content(b"# Markdown Header"), Some(MediaFormat::Markdown));
        assert_eq!(detect_text_format_from_content(b"name,age,city\nJohn,30,NYC"), Some(MediaFormat::Csv));
    }

    #[test]
    fn test_validate_format_modality() {
        assert!(validate_format_modality(MediaFormat::Jpeg, ModalityType::Image));
        assert!(validate_format_modality(MediaFormat::Mp3, ModalityType::Audio));
        assert!(!validate_format_modality(MediaFormat::Jpeg, ModalityType::Audio));
    }

    #[test]
    fn test_format_detection_confidence() {
        let jpeg_data = b"\xFF\xD8\xFF"; // JPEG magic bytes
        let confidence = format_detection_confidence(jpeg_data, &MediaFormat::Jpeg);
        
        #[cfg(feature = "format-detection")]
        assert!(confidence > 0.9);
        
        #[cfg(not(feature = "format-detection"))]
        assert_eq!(confidence, 0.5);
        
        let unknown_confidence = format_detection_confidence(b"random", &MediaFormat::Unknown("test".to_string()));
        assert_eq!(unknown_confidence, 0.1);
    }

    #[test]
    fn test_format_detection_stats() {
        let mut stats = FormatDetectionStats::new();
        
        stats.record_detection(MediaFormat::Jpeg, 0.9);
        stats.record_detection(MediaFormat::Jpeg, 0.8);
        stats.record_detection(MediaFormat::Mp3, 0.7);
        stats.record_detection(MediaFormat::Unknown("test".to_string()), 0.1);
        
        assert_eq!(stats.total_detections, 4);
        assert_eq!(stats.unknown_detections, 1);
        assert_eq!(stats.success_rate(), 0.75);
        assert_eq!(stats.most_common_format(), Some(MediaFormat::Jpeg));
        assert_eq!(stats.most_common_modality(), Some(ModalityType::Image));
    }
}
