//! Audio feature extraction utilities.
//!
//! This module provides functionality to extract various features from audio content,
//! including basic properties (duration, sample rate, channels) and advanced spectral
//! features (MFCCs, spectral centroid, tempo, etc.).

use crate::error::{MultimodalError, Result};
use crate::types::{MediaContent, MediaFormat};
use std::collections::HashMap;

/// Audio feature extractor that can analyze various audio properties.
#[derive(Debug, Clone)]
pub struct AudioFeatureExtractor {
    /// Whether to extract advanced spectral features
    extract_spectral_features: bool,

    /// Whether to extract tempo and rhythm features
    extract_tempo_features: bool,

    /// Whether to extract harmonic features
    extract_harmonic_features: bool,

    /// Frame size for spectral analysis
    frame_size: usize,

    /// Hop size for spectral analysis
    hop_size: usize,
}

impl Default for AudioFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFeatureExtractor {
    /// Create a new audio feature extractor with default settings.
    pub fn new() -> Self {
        Self {
            extract_spectral_features: true,
            extract_tempo_features: false,
            extract_harmonic_features: false,
            frame_size: 2048,
            hop_size: 512,
        }
    }

    /// Create a builder for configuring the feature extractor.
    pub fn builder() -> AudioFeatureExtractorBuilder {
        AudioFeatureExtractorBuilder::new()
    }

    /// Extract all available features from audio content.
    pub async fn extract_features(
        &self,
        content: &MediaContent,
    ) -> Result<HashMap<String, serde_json::Value>> {
        self.validate_audio_content(content)?;

        let mut features = HashMap::new();

        // Extract basic features
        self.extract_basic_features(content, &mut features).await?;

        // Extract spectral features if enabled
        if self.extract_spectral_features {
            self.extract_spectral_features_impl(content, &mut features)
                .await?;
        }

        // Extract tempo features if enabled
        if self.extract_tempo_features {
            self.extract_tempo_features_impl(content, &mut features)
                .await?;
        }

        // Extract harmonic features if enabled
        if self.extract_harmonic_features {
            self.extract_harmonic_features_impl(content, &mut features)
                .await?;
        }

        Ok(features)
    }

    /// Extract basic audio properties.
    pub async fn extract_basic_features(
        &self,
        content: &MediaContent,
        features: &mut HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        // Add format information
        features.insert("format".to_string(), content.format.to_string().into());

        // Add file size if available
        if let Some(size) = content.estimated_size() {
            features.insert("file_size".to_string(), size.into());
        }

        #[cfg(feature = "audio-support")]
        {
            // TODO: Implement actual audio analysis using symphonia or rodio
            // This would involve:
            // 1. Decoding the audio data
            // 2. Extracting sample rate, channels, duration
            // 3. Calculating bit rate and other basic properties

            // Placeholder values - these should be extracted from actual audio data
            match content.format {
                MediaFormat::Mp3 => {
                    features.insert("sample_rate".to_string(), 44100.into());
                    features.insert("channels".to_string(), 2.into());
                    features.insert("bitrate".to_string(), 128000.into());
                    features.insert(
                        "duration".to_string(),
                        self.estimate_duration(content).into(),
                    );
                }
                MediaFormat::Wav => {
                    features.insert("sample_rate".to_string(), 44100.into());
                    features.insert("channels".to_string(), 2.into());
                    features.insert("bit_depth".to_string(), 16.into());
                    features.insert(
                        "duration".to_string(),
                        self.estimate_duration(content).into(),
                    );
                }
                MediaFormat::Flac => {
                    features.insert("sample_rate".to_string(), 44100.into());
                    features.insert("channels".to_string(), 2.into());
                    features.insert("bit_depth".to_string(), 24.into());
                    features.insert(
                        "duration".to_string(),
                        self.estimate_duration(content).into(),
                    );
                }
                _ => {
                    features.insert("sample_rate".to_string(), 44100.into());
                    features.insert("channels".to_string(), 2.into());
                    features.insert(
                        "duration".to_string(),
                        self.estimate_duration(content).into(),
                    );
                }
            }
        }

        #[cfg(not(feature = "audio-support"))]
        {
            // Basic fallback features
            features.insert(
                "estimated_duration".to_string(),
                self.estimate_duration(content).into(),
            );
        }

        Ok(())
    }

    /// Extract spectral features (MFCCs, spectral centroid, etc.).
    async fn extract_spectral_features_impl(
        &self,
        content: &MediaContent,
        features: &mut HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        #[cfg(feature = "audio-support")]
        {
            // TODO: Implement actual spectral feature extraction
            // This would involve:
            // 1. Computing Short-Time Fourier Transform (STFT)
            // 2. Calculating spectral centroid, rolloff, flux
            // 3. Computing Mel-Frequency Cepstral Coefficients (MFCCs)
            // 4. Extracting zero-crossing rate

            // Placeholder spectral features
            features.insert("spectral_centroid".to_string(), 2000.0.into());
            features.insert("spectral_rolloff".to_string(), 8000.0.into());
            features.insert("spectral_flux".to_string(), 0.15.into());
            features.insert("zero_crossing_rate".to_string(), 0.08.into());

            // MFCC coefficients (typically 12-13 coefficients)
            let mfcc_coeffs: Vec<f64> = vec![
                -15.2, 8.4, -2.1, 1.8, -0.9, 0.6, -0.4, 0.3, -0.2, 0.1, -0.1, 0.05, -0.03,
            ];
            features.insert("mfcc".to_string(), mfcc_coeffs.into());
        }

        Ok(())
    }

    /// Extract tempo and rhythm features.
    async fn extract_tempo_features_impl(
        &self,
        content: &MediaContent,
        features: &mut HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        #[cfg(feature = "audio-support")]
        {
            // TODO: Implement actual tempo detection
            // This would involve:
            // 1. Computing onset detection function
            // 2. Analyzing beat patterns
            // 3. Estimating tempo in BPM
            // 4. Detecting time signature

            // Placeholder tempo features
            features.insert("tempo_bpm".to_string(), 120.0.into());
            features.insert("time_signature".to_string(), "4/4".into());
            features.insert("rhythm_regularity".to_string(), 0.85.into());
        }

        Ok(())
    }

    /// Extract harmonic features.
    async fn extract_harmonic_features_impl(
        &self,
        content: &MediaContent,
        features: &mut HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        #[cfg(feature = "audio-support")]
        {
            // TODO: Implement actual harmonic analysis
            // This would involve:
            // 1. Computing chromagram
            // 2. Detecting key and mode
            // 3. Analyzing harmonic content
            // 4. Computing tonal centroid features

            // Placeholder harmonic features
            features.insert("key".to_string(), "C major".into());
            features.insert("mode".to_string(), "major".into());
            features.insert("harmonic_ratio".to_string(), 0.75.into());
            features.insert("tonal_centroid_x".to_string(), 0.2.into());
            features.insert("tonal_centroid_y".to_string(), 0.8.into());
        }

        Ok(())
    }

    /// Estimate audio duration based on file size and format.
    fn estimate_duration(&self, content: &MediaContent) -> f64 {
        if let Some(size) = content.estimated_size() {
            // Rough estimation based on typical bitrates
            let estimated_bitrate = match content.format {
                MediaFormat::Mp3 => 128000,  // 128 kbps
                MediaFormat::Wav => 1411200, // 44.1kHz * 16bit * 2ch
                MediaFormat::Flac => 800000, // Compressed lossless
                MediaFormat::Ogg => 160000,  // 160 kbps
                MediaFormat::M4a => 128000,  // 128 kbps
                MediaFormat::Aac => 128000,  // 128 kbps
                _ => 128000,                 // Default
            };

            // Duration = (file_size_bits) / (bitrate)
            (size as f64 * 8.0) / (estimated_bitrate as f64)
        } else {
            0.0
        }
    }

    /// Validate that the content is audio.
    fn validate_audio_content(&self, content: &MediaContent) -> Result<()> {
        if !matches!(
            content.format,
            MediaFormat::Mp3
                | MediaFormat::Wav
                | MediaFormat::Flac
                | MediaFormat::Ogg
                | MediaFormat::M4a
                | MediaFormat::Aac
                | MediaFormat::Wma
        ) {
            return Err(MultimodalError::invalid_format(
                "audio".to_string(),
                content.format.to_string(),
            ));
        }
        Ok(())
    }

    /// Get list of extractable features.
    pub fn extractable_features(&self) -> Vec<String> {
        let mut features = vec![
            "format".to_string(),
            "file_size".to_string(),
            "duration".to_string(),
            "sample_rate".to_string(),
            "channels".to_string(),
            "bitrate".to_string(),
        ];

        if self.extract_spectral_features {
            features.extend(vec![
                "spectral_centroid".to_string(),
                "spectral_rolloff".to_string(),
                "spectral_flux".to_string(),
                "zero_crossing_rate".to_string(),
                "mfcc".to_string(),
            ]);
        }

        if self.extract_tempo_features {
            features.extend(vec![
                "tempo_bpm".to_string(),
                "time_signature".to_string(),
                "rhythm_regularity".to_string(),
            ]);
        }

        if self.extract_harmonic_features {
            features.extend(vec![
                "key".to_string(),
                "mode".to_string(),
                "harmonic_ratio".to_string(),
                "tonal_centroid_x".to_string(),
                "tonal_centroid_y".to_string(),
            ]);
        }

        features
    }
}

/// Builder for configuring AudioFeatureExtractor.
#[derive(Debug)]
pub struct AudioFeatureExtractorBuilder {
    extract_spectral_features: bool,
    extract_tempo_features: bool,
    extract_harmonic_features: bool,
    frame_size: usize,
    hop_size: usize,
}

impl AudioFeatureExtractorBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            extract_spectral_features: true,
            extract_tempo_features: false,
            extract_harmonic_features: false,
            frame_size: 2048,
            hop_size: 512,
        }
    }

    /// Set whether to extract spectral features.
    pub fn extract_spectral_features(mut self, extract: bool) -> Self {
        self.extract_spectral_features = extract;
        self
    }

    /// Set whether to extract tempo features.
    pub fn extract_tempo_features(mut self, extract: bool) -> Self {
        self.extract_tempo_features = extract;
        self
    }

    /// Set whether to extract harmonic features.
    pub fn extract_harmonic_features(mut self, extract: bool) -> Self {
        self.extract_harmonic_features = extract;
        self
    }

    /// Set frame size for spectral analysis.
    pub fn frame_size(mut self, size: usize) -> Self {
        self.frame_size = size;
        self
    }

    /// Set hop size for spectral analysis.
    pub fn hop_size(mut self, size: usize) -> Self {
        self.hop_size = size;
        self
    }

    /// Build the AudioFeatureExtractor.
    pub fn build(self) -> AudioFeatureExtractor {
        AudioFeatureExtractor {
            extract_spectral_features: self.extract_spectral_features,
            extract_tempo_features: self.extract_tempo_features,
            extract_harmonic_features: self.extract_harmonic_features,
            frame_size: self.frame_size,
            hop_size: self.hop_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MediaData, MediaMetadata};

    #[tokio::test]
    async fn test_audio_feature_extractor_creation() {
        let extractor = AudioFeatureExtractor::new();
        assert!(extractor.extract_spectral_features);
        assert!(!extractor.extract_tempo_features);
        assert!(!extractor.extract_harmonic_features);
    }

    #[tokio::test]
    async fn test_audio_feature_extractor_builder() {
        let extractor = AudioFeatureExtractor::builder()
            .extract_spectral_features(false)
            .extract_tempo_features(true)
            .extract_harmonic_features(true)
            .frame_size(4096)
            .hop_size(1024)
            .build();

        assert!(!extractor.extract_spectral_features);
        assert!(extractor.extract_tempo_features);
        assert!(extractor.extract_harmonic_features);
        assert_eq!(extractor.frame_size, 4096);
        assert_eq!(extractor.hop_size, 1024);
    }

    #[tokio::test]
    async fn test_extract_basic_features() {
        let extractor = AudioFeatureExtractor::new();
        let content = MediaContent {
            data: MediaData::Bytes(vec![0; 1024]),
            format: MediaFormat::Mp3,
            metadata: MediaMetadata::new(),
            extracted_text: None,
            features: HashMap::new(),
            checksum: None,
        };

        let mut features = HashMap::new();
        extractor
            .extract_basic_features(&content, &mut features)
            .await
            .unwrap();

        assert!(features.contains_key("format"));
        assert!(features.contains_key("file_size"));
    }

    #[test]
    fn test_extractable_features() {
        let extractor = AudioFeatureExtractor::builder()
            .extract_spectral_features(true)
            .extract_tempo_features(true)
            .extract_harmonic_features(true)
            .build();

        let features = extractor.extractable_features();

        assert!(features.contains(&"format".to_string()));
        assert!(features.contains(&"spectral_centroid".to_string()));
        assert!(features.contains(&"tempo_bpm".to_string()));
        assert!(features.contains(&"key".to_string()));
    }

    #[test]
    fn test_estimate_duration() {
        let extractor = AudioFeatureExtractor::new();
        let content = MediaContent {
            data: MediaData::Bytes(vec![0; 1024 * 1024]), // 1MB
            format: MediaFormat::Mp3,
            metadata: MediaMetadata::new(),
            extracted_text: None,
            features: HashMap::new(),
            checksum: None,
        };

        let duration = extractor.estimate_duration(&content);
        assert!(duration > 0.0);
    }
}
