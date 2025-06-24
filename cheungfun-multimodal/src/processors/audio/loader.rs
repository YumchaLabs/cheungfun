//! Audio file loading and format detection.
//!
//! This module provides functionality to load audio files from various sources
//! and detect their format and properties.

use crate::error::{MultimodalError, Result};
use crate::types::{MediaContent, MediaData, MediaFormat, MediaMetadata};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Audio file loader that supports multiple audio formats.
#[derive(Debug, Clone)]
pub struct AudioLoader {
    /// Maximum file size to load (in bytes)
    max_file_size: Option<u64>,

    /// Whether to validate audio format
    validate_format: bool,

    /// Supported audio formats
    supported_formats: Vec<MediaFormat>,
}

impl Default for AudioLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioLoader {
    /// Create a new audio loader with default settings.
    pub fn new() -> Self {
        Self {
            max_file_size: Some(100 * 1024 * 1024), // 100MB default
            validate_format: true,
            supported_formats: vec![
                MediaFormat::Mp3,
                MediaFormat::Wav,
                MediaFormat::Flac,
                MediaFormat::Ogg,
                MediaFormat::M4a,
                MediaFormat::Aac,
                MediaFormat::Wma,
            ],
        }
    }

    /// Create a builder for configuring the audio loader.
    pub fn builder() -> AudioLoaderBuilder {
        AudioLoaderBuilder::new()
    }

    /// Load audio from a file path.
    pub async fn load_from_path<P: AsRef<Path>>(&self, path: P) -> Result<MediaContent> {
        let path = path.as_ref();

        // Check if file exists
        if !path.exists() {
            return Err(MultimodalError::file_not_found(path.to_path_buf()));
        }

        // Check file size
        let metadata = fs::metadata(path).map_err(|e| {
            MultimodalError::media_processing(format!("Failed to read file metadata: {}", e))
        })?;

        if let Some(max_size) = self.max_file_size {
            if metadata.len() > max_size {
                return Err(MultimodalError::content_too_large(metadata.len(), max_size));
            }
        }

        // Read file content
        let data = fs::read(path).map_err(|e| {
            MultimodalError::media_processing(format!("Failed to read audio file: {}", e))
        })?;

        // Detect format
        let format = self.detect_audio_format(&data, Some(path))?;

        // Validate format if enabled
        if self.validate_format && !self.supported_formats.contains(&format) {
            return Err(MultimodalError::unsupported_format(
                format.to_string(),
                "audio".to_string(),
            ));
        }

        // Create media metadata
        let mut media_metadata = MediaMetadata::new();
        media_metadata.filename = path.file_name().and_then(|n| n.to_str()).map(String::from);
        media_metadata.size = Some(metadata.len());
        // Convert SystemTime to DateTime<Utc>
        if let Ok(created) = metadata.created() {
            media_metadata.created_at = Some(chrono::DateTime::from(created));
        }
        if let Ok(modified) = metadata.modified() {
            media_metadata.modified_at = Some(chrono::DateTime::from(modified));
        }

        // Create media content
        Ok(MediaContent {
            data: MediaData::Embedded(data),
            format,
            metadata: media_metadata,
            extracted_text: None,
            features: HashMap::new(),
            checksum: None,
        })
    }

    /// Load audio from bytes.
    pub async fn load_from_bytes(
        &self,
        data: Vec<u8>,
        filename: Option<&str>,
    ) -> Result<MediaContent> {
        // Check file size
        if let Some(max_size) = self.max_file_size {
            if data.len() as u64 > max_size {
                return Err(MultimodalError::content_too_large(
                    data.len() as u64,
                    max_size,
                ));
            }
        }

        // Detect format
        let format = self.detect_audio_format(&data, filename.map(Path::new))?;

        // Validate format if enabled
        if self.validate_format && !self.supported_formats.contains(&format) {
            return Err(MultimodalError::unsupported_format(
                format.to_string(),
                "audio".to_string(),
            ));
        }

        // Create media metadata
        let mut media_metadata = MediaMetadata::new();
        media_metadata.size = Some(data.len() as u64);
        if let Some(filename) = filename {
            media_metadata.filename = Some(filename.to_string());
        }

        // Create media content
        Ok(MediaContent {
            data: MediaData::Embedded(data),
            format,
            metadata: media_metadata,
            extracted_text: None,
            features: HashMap::new(),
            checksum: None,
        })
    }

    /// Load audio from a URL.
    #[cfg(feature = "remote-resources")]
    pub async fn load_from_url(&self, url: &str) -> Result<MediaContent> {
        use reqwest;

        let response = reqwest::get(url).await.map_err(|e| {
            MultimodalError::network(format!("Failed to fetch audio from URL: {}", e))
        })?;

        if !response.status().is_success() {
            return Err(MultimodalError::network(format!(
                "HTTP error {}: {}",
                response.status(),
                response
                    .status()
                    .canonical_reason()
                    .unwrap_or("Unknown error")
            )));
        }

        let data = response
            .bytes()
            .await
            .map_err(|e| MultimodalError::network(format!("Failed to read response body: {}", e)))?
            .to_vec();

        // Extract filename from URL
        let filename = url
            .split('/')
            .last()
            .map(|s| s.split('?').next().unwrap_or(s));

        self.load_from_bytes(data, filename).await
    }

    /// Detect audio format from data and optional filename.
    fn detect_audio_format(&self, data: &[u8], path: Option<&Path>) -> Result<MediaFormat> {
        // First try magic bytes detection
        #[cfg(feature = "format-detection")]
        if let Some(format) = self.detect_format_from_magic_bytes(data) {
            return Ok(format);
        }

        // Then try filename extension
        if let Some(path) = path {
            if let Some(format) = self.detect_format_from_extension(path) {
                return Ok(format);
            }
        }

        // Fallback to content analysis
        if let Some(format) = self.detect_format_from_content(data) {
            return Ok(format);
        }

        Err(MultimodalError::format_detection(
            "Unable to detect audio format".to_string(),
        ))
    }

    /// Detect format from magic bytes.
    #[cfg(feature = "format-detection")]
    fn detect_format_from_magic_bytes(&self, data: &[u8]) -> Option<MediaFormat> {
        use infer;

        if let Some(kind) = infer::get(data) {
            match kind.mime_type() {
                "audio/mpeg" => Some(MediaFormat::Mp3),
                "audio/wav" | "audio/wave" => Some(MediaFormat::Wav),
                "audio/flac" => Some(MediaFormat::Flac),
                "audio/ogg" => Some(MediaFormat::Ogg),
                "audio/mp4" | "audio/m4a" => Some(MediaFormat::M4a),
                "audio/aac" => Some(MediaFormat::Aac),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Detect format from file extension.
    fn detect_format_from_extension(&self, path: &Path) -> Option<MediaFormat> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext| match ext.to_lowercase().as_str() {
                "mp3" => Some(MediaFormat::Mp3),
                "wav" | "wave" => Some(MediaFormat::Wav),
                "flac" => Some(MediaFormat::Flac),
                "ogg" => Some(MediaFormat::Ogg),
                "m4a" => Some(MediaFormat::M4a),
                "aac" => Some(MediaFormat::Aac),
                "wma" => Some(MediaFormat::Wma),
                _ => None,
            })
    }

    /// Detect format from content analysis.
    fn detect_format_from_content(&self, data: &[u8]) -> Option<MediaFormat> {
        if data.len() < 4 {
            return None;
        }

        // Check for common audio file signatures
        match &data[0..4] {
            // MP3 with ID3v2 tag
            [0x49, 0x44, 0x33, _] => Some(MediaFormat::Mp3),
            // MP3 frame sync
            [0xFF, 0xFB, _, _] | [0xFF, 0xFA, _, _] => Some(MediaFormat::Mp3),
            // WAV/RIFF
            [0x52, 0x49, 0x46, 0x46] => {
                if data.len() >= 12 && &data[8..12] == b"WAVE" {
                    Some(MediaFormat::Wav)
                } else {
                    None
                }
            }
            // FLAC
            [0x66, 0x4C, 0x61, 0x43] => Some(MediaFormat::Flac),
            // OGG
            [0x4F, 0x67, 0x67, 0x53] => Some(MediaFormat::Ogg),
            _ => None,
        }
    }
}

/// Builder for configuring AudioLoader.
#[derive(Debug)]
pub struct AudioLoaderBuilder {
    max_file_size: Option<u64>,
    validate_format: bool,
    supported_formats: Vec<MediaFormat>,
}

impl AudioLoaderBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            max_file_size: Some(100 * 1024 * 1024), // 100MB default
            validate_format: true,
            supported_formats: vec![
                MediaFormat::Mp3,
                MediaFormat::Wav,
                MediaFormat::Flac,
                MediaFormat::Ogg,
                MediaFormat::M4a,
                MediaFormat::Aac,
                MediaFormat::Wma,
            ],
        }
    }

    /// Set maximum file size.
    pub fn max_file_size(mut self, size: Option<u64>) -> Self {
        self.max_file_size = size;
        self
    }

    /// Set whether to validate format.
    pub fn validate_format(mut self, validate: bool) -> Self {
        self.validate_format = validate;
        self
    }

    /// Set supported formats.
    pub fn supported_formats(mut self, formats: Vec<MediaFormat>) -> Self {
        self.supported_formats = formats;
        self
    }

    /// Build the AudioLoader.
    pub fn build(self) -> AudioLoader {
        AudioLoader {
            max_file_size: self.max_file_size,
            validate_format: self.validate_format,
            supported_formats: self.supported_formats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_audio_loader_creation() {
        let loader = AudioLoader::new();
        assert!(loader.supported_formats.contains(&MediaFormat::Mp3));
        assert!(loader.supported_formats.contains(&MediaFormat::Wav));
        assert_eq!(loader.max_file_size, Some(100 * 1024 * 1024));
    }

    #[tokio::test]
    async fn test_audio_loader_builder() {
        let loader = AudioLoader::builder()
            .max_file_size(Some(50 * 1024 * 1024))
            .validate_format(false)
            .supported_formats(vec![MediaFormat::Mp3, MediaFormat::Wav])
            .build();

        assert_eq!(loader.max_file_size, Some(50 * 1024 * 1024));
        assert!(!loader.validate_format);
        assert_eq!(loader.supported_formats.len(), 2);
    }

    #[tokio::test]
    async fn test_load_from_bytes() {
        let loader = AudioLoader::new();

        // Create dummy WAV data with proper header
        let mut wav_data = Vec::new();
        wav_data.extend_from_slice(b"RIFF");
        wav_data.extend_from_slice(&[36, 0, 0, 0]); // File size - 8
        wav_data.extend_from_slice(b"WAVE");
        wav_data.extend_from_slice(b"fmt ");
        wav_data.extend_from_slice(&[16, 0, 0, 0]); // Subchunk1Size
        wav_data.extend_from_slice(&[1, 0]); // AudioFormat (PCM)
        wav_data.extend_from_slice(&[1, 0]); // NumChannels
        wav_data.extend_from_slice(&[0x44, 0xAC, 0, 0]); // SampleRate (44100)
        wav_data.extend_from_slice(&[0x88, 0x58, 1, 0]); // ByteRate
        wav_data.extend_from_slice(&[2, 0]); // BlockAlign
        wav_data.extend_from_slice(&[16, 0]); // BitsPerSample
        wav_data.extend_from_slice(b"data");
        wav_data.extend_from_slice(&[0, 0, 0, 0]); // Subchunk2Size

        let content = loader
            .load_from_bytes(wav_data, Some("test.wav"))
            .await
            .unwrap();
        assert_eq!(content.format, MediaFormat::Wav);
    }

    #[test]
    fn test_format_detection_from_extension() {
        let loader = AudioLoader::new();

        assert_eq!(
            loader.detect_format_from_extension(Path::new("test.mp3")),
            Some(MediaFormat::Mp3)
        );
        assert_eq!(
            loader.detect_format_from_extension(Path::new("test.wav")),
            Some(MediaFormat::Wav)
        );
        assert_eq!(
            loader.detect_format_from_extension(Path::new("test.unknown")),
            None
        );
    }
}
