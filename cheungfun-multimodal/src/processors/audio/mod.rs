//! Audio processing module.
//!
//! This module provides audio processing capabilities including format conversion,
//! feature extraction, and basic audio analysis operations.

use crate::traits::{MediaProcessor, ProcessingOptions, FeatureExtractor};
use crate::types::{MediaContent, MediaFormat, ModalityType};
use crate::error::Result;
use async_trait::async_trait;
use std::collections::HashMap;

/// Audio processor for handling various audio formats and operations.
#[derive(Debug, Clone)]
pub struct AudioProcessor {
    /// Maximum audio duration for processing (in seconds)
    max_duration: Option<f32>,
    
    /// Default sample rate for processing
    default_sample_rate: u32,
    
    /// Whether to normalize audio levels
    normalize_audio: bool,
}

impl AudioProcessor {
    /// Create a new audio processor with default settings.
    pub fn new() -> Self {
        Self {
            max_duration: Some(600.0), // 10 minutes
            default_sample_rate: 44100,
            normalize_audio: true,
        }
    }
    
    /// Create a builder for configuring the processor.
    pub fn builder() -> AudioProcessorBuilder {
        AudioProcessorBuilder::new()
    }
}

impl Default for AudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MediaProcessor for AudioProcessor {
    fn supported_input_modalities(&self) -> Vec<ModalityType> {
        vec![ModalityType::Audio]
    }
    
    fn supported_output_modalities(&self) -> Vec<ModalityType> {
        vec![ModalityType::Audio]
    }
    
    fn supported_input_formats(&self, modality: ModalityType) -> Vec<MediaFormat> {
        if modality == ModalityType::Audio {
            vec![
                MediaFormat::Mp3,
                MediaFormat::Wav,
                MediaFormat::Flac,
                MediaFormat::Ogg,
                MediaFormat::M4a,
                MediaFormat::Aac,
            ]
        } else {
            vec![]
        }
    }
    
    fn supported_output_formats(&self, modality: ModalityType) -> Vec<MediaFormat> {
        if modality == ModalityType::Audio {
            vec![
                MediaFormat::Wav, // Most commonly supported output format
                MediaFormat::Mp3,
                MediaFormat::Flac,
            ]
        } else {
            vec![]
        }
    }
    
    async fn process(&self, content: &MediaContent, _options: &ProcessingOptions) -> Result<MediaContent> {
        self.validate_input(content)?;
        
        #[cfg(feature = "audio-support")]
        {
            // TODO: Implement actual audio processing
            // This would involve:
            // 1. Loading audio data using rodio/symphonia
            // 2. Converting sample rates if needed
            // 3. Applying normalization
            // 4. Format conversion
            
            // For now, return the content as-is
            Ok(content.clone())
        }
        
        #[cfg(not(feature = "audio-support"))]
        {
            Err(crate::error::MultimodalError::feature_not_enabled("audio-support"))
        }
    }
    
    async fn extract_features(&self, content: &MediaContent) -> Result<HashMap<String, serde_json::Value>> {
        #[cfg(feature = "audio-support")]
        {
            let mut features = HashMap::new();
            
            // TODO: Implement actual audio feature extraction
            // This would involve:
            // 1. Loading audio data
            // 2. Extracting duration, sample rate, channels
            // 3. Computing spectral features (MFCCs, spectral centroid, etc.)
            // 4. Detecting tempo, key, loudness
            
            // Basic placeholder features
            features.insert("format".to_string(), content.format.to_string().into());
            if let Some(size) = content.estimated_size() {
                features.insert("file_size".to_string(), size.into());
            }
            
            Ok(features)
        }
        
        #[cfg(not(feature = "audio-support"))]
        {
            Ok(HashMap::new())
        }
    }
    
    async fn extract_text(&self, _content: &MediaContent) -> Result<Option<String>> {
        #[cfg(feature = "audio-support")]
        {
            // TODO: Implement speech-to-text functionality
            // This would require integration with speech recognition services
            // or libraries like whisper-rs
            Ok(None)
        }
        
        #[cfg(not(feature = "audio-support"))]
        {
            Ok(None)
        }
    }
}

#[async_trait]
impl FeatureExtractor for AudioProcessor {
    async fn extract_modality_features(
        &self,
        content: &MediaContent,
        modality: ModalityType,
    ) -> Result<HashMap<String, serde_json::Value>> {
        if modality == ModalityType::Audio {
            self.extract_features(content).await
        } else {
            Ok(HashMap::new())
        }
    }
    
    fn extractable_features(&self, modality: ModalityType) -> Vec<String> {
        if modality == ModalityType::Audio {
            vec![
                "duration".to_string(),
                "sample_rate".to_string(),
                "channels".to_string(),
                "bitrate".to_string(),
                "format".to_string(),
                "file_size".to_string(),
                // Advanced features (when implemented):
                // "tempo".to_string(),
                // "key".to_string(),
                // "loudness".to_string(),
                // "spectral_centroid".to_string(),
                // "mfcc".to_string(),
            ]
        } else {
            vec![]
        }
    }
    
    async fn extract_feature(
        &self,
        content: &MediaContent,
        feature_name: &str,
    ) -> Result<Option<serde_json::Value>> {
        let features = self.extract_features(content).await?;
        Ok(features.get(feature_name).cloned())
    }
}

/// Builder for configuring an audio processor.
#[derive(Debug)]
pub struct AudioProcessorBuilder {
    max_duration: Option<f32>,
    default_sample_rate: u32,
    normalize_audio: bool,
}

impl AudioProcessorBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            max_duration: Some(600.0),
            default_sample_rate: 44100,
            normalize_audio: true,
        }
    }
    
    /// Set maximum audio duration.
    pub fn max_duration(mut self, duration: f32) -> Self {
        self.max_duration = Some(duration);
        self
    }
    
    /// Remove duration limits.
    pub fn unlimited_duration(mut self) -> Self {
        self.max_duration = None;
        self
    }
    
    /// Set default sample rate.
    pub fn default_sample_rate(mut self, sample_rate: u32) -> Self {
        self.default_sample_rate = sample_rate;
        self
    }
    
    /// Set whether to normalize audio levels.
    pub fn normalize_audio(mut self, normalize: bool) -> Self {
        self.normalize_audio = normalize;
        self
    }
    
    /// Build the audio processor.
    pub fn build(self) -> AudioProcessor {
        AudioProcessor {
            max_duration: self.max_duration,
            default_sample_rate: self.default_sample_rate,
            normalize_audio: self.normalize_audio,
        }
    }
}

impl Default for AudioProcessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_processor_creation() {
        let processor = AudioProcessor::new();
        assert_eq!(processor.supported_input_modalities(), vec![ModalityType::Audio]);
        
        let formats = processor.supported_input_formats(ModalityType::Audio);
        assert!(formats.contains(&MediaFormat::Mp3));
        assert!(formats.contains(&MediaFormat::Wav));
        assert!(formats.contains(&MediaFormat::Flac));
    }

    #[test]
    fn test_audio_processor_builder() {
        let processor = AudioProcessor::builder()
            .max_duration(300.0)
            .default_sample_rate(48000)
            .normalize_audio(false)
            .build();
        
        assert_eq!(processor.max_duration, Some(300.0));
        assert_eq!(processor.default_sample_rate, 48000);
        assert!(!processor.normalize_audio);
    }

    #[test]
    fn test_extractable_features() {
        let processor = AudioProcessor::new();
        let features = processor.extractable_features(ModalityType::Audio);
        
        assert!(features.contains(&"duration".to_string()));
        assert!(features.contains(&"sample_rate".to_string()));
        assert!(features.contains(&"channels".to_string()));
    }

    #[tokio::test]
    async fn test_audio_feature_extraction() {
        let content = MediaContent::from_bytes(
            vec![0; 1024], // Dummy audio data
            MediaFormat::Wav,
        );
        
        let processor = AudioProcessor::new();
        let features = processor.extract_features(&content).await.unwrap();
        
        // Should at least have format information
        assert!(features.contains_key("format"));
    }
}
