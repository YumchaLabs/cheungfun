//! Media processors for different modalities.
//!
//! This module provides implementations of media processors for various
//! content types including images, audio, video, and documents.

#[cfg(feature = "image-support")]
pub mod image;

#[cfg(feature = "audio-support")]
pub mod audio;

#[cfg(feature = "video-support")]
pub mod video;

// Re-export commonly used processors
#[cfg(feature = "image-support")]
pub use image::ImageProcessor;

#[cfg(feature = "audio-support")]
pub use audio::AudioProcessor;

#[cfg(feature = "video-support")]
pub use video::VideoProcessor;

use crate::error::Result;
use crate::traits::{MediaProcessor, ProcessingOptions};
use crate::types::{MediaContent, MediaFormat, ModalityType};
use async_trait::async_trait;
use std::collections::HashMap;

/// A composite processor that can handle multiple modalities.
#[derive(Debug)]
pub struct MultimodalProcessor {
    #[cfg(feature = "image-support")]
    image_processor: Option<ImageProcessor>,

    #[cfg(feature = "audio-support")]
    audio_processor: Option<AudioProcessor>,

    #[cfg(feature = "video-support")]
    video_processor: Option<VideoProcessor>,
}

impl MultimodalProcessor {
    /// Create a new multimodal processor with default settings.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "image-support")]
            image_processor: Some(ImageProcessor::new()),

            #[cfg(feature = "audio-support")]
            audio_processor: Some(AudioProcessor::new()),

            #[cfg(feature = "video-support")]
            video_processor: Some(VideoProcessor::new()),
        }
    }

    /// Create a builder for configuring the processor.
    pub fn builder() -> MultimodalProcessorBuilder {
        MultimodalProcessorBuilder::new()
    }
}

impl Default for MultimodalProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MediaProcessor for MultimodalProcessor {
    fn supported_input_modalities(&self) -> Vec<ModalityType> {
        let mut modalities = vec![ModalityType::Text];

        #[cfg(feature = "image-support")]
        if self.image_processor.is_some() {
            modalities.push(ModalityType::Image);
        }

        #[cfg(feature = "audio-support")]
        if self.audio_processor.is_some() {
            modalities.push(ModalityType::Audio);
        }

        #[cfg(feature = "video-support")]
        if self.video_processor.is_some() {
            modalities.push(ModalityType::Video);
        }

        modalities
    }

    fn supported_output_modalities(&self) -> Vec<ModalityType> {
        self.supported_input_modalities()
    }

    fn supported_input_formats(&self, modality: ModalityType) -> Vec<MediaFormat> {
        match modality {
            ModalityType::Text => vec![
                MediaFormat::PlainText,
                MediaFormat::Markdown,
                MediaFormat::Html,
                MediaFormat::Json,
                MediaFormat::Xml,
                MediaFormat::Csv,
            ],

            #[cfg(feature = "image-support")]
            ModalityType::Image => {
                if self.image_processor.is_some() {
                    self.image_processor
                        .as_ref()
                        .unwrap()
                        .supported_input_formats(modality)
                } else {
                    vec![]
                }
            }

            #[cfg(feature = "audio-support")]
            ModalityType::Audio => {
                if self.audio_processor.is_some() {
                    self.audio_processor
                        .as_ref()
                        .unwrap()
                        .supported_input_formats(modality)
                } else {
                    vec![]
                }
            }

            #[cfg(feature = "video-support")]
            ModalityType::Video => {
                if self.video_processor.is_some() {
                    self.video_processor
                        .as_ref()
                        .unwrap()
                        .supported_input_formats(modality)
                } else {
                    vec![]
                }
            }

            _ => vec![],
        }
    }

    fn supported_output_formats(&self, modality: ModalityType) -> Vec<MediaFormat> {
        self.supported_input_formats(modality)
    }

    async fn process(
        &self,
        content: &MediaContent,
        options: &ProcessingOptions,
    ) -> Result<MediaContent> {
        self.validate_input(content)?;

        match content.modality_type() {
            ModalityType::Text => {
                // Text processing (basic implementation)
                Ok(content.clone())
            }

            #[cfg(feature = "image-support")]
            ModalityType::Image => {
                if let Some(ref processor) = self.image_processor {
                    processor.process(content, options).await
                } else {
                    Err(crate::error::MultimodalError::feature_not_enabled(
                        "image-support",
                    ))
                }
            }

            #[cfg(feature = "audio-support")]
            ModalityType::Audio => {
                if let Some(ref processor) = self.audio_processor {
                    processor.process(content, options).await
                } else {
                    Err(crate::error::MultimodalError::feature_not_enabled(
                        "audio-support",
                    ))
                }
            }

            #[cfg(feature = "video-support")]
            ModalityType::Video => {
                if let Some(ref processor) = self.video_processor {
                    processor.process(content, options).await
                } else {
                    Err(crate::error::MultimodalError::feature_not_enabled(
                        "video-support",
                    ))
                }
            }

            _ => Err(crate::error::MultimodalError::unsupported_modality(
                content.modality_type().to_string(),
            )),
        }
    }

    async fn extract_features(
        &self,
        content: &MediaContent,
    ) -> Result<HashMap<String, serde_json::Value>> {
        match content.modality_type() {
            ModalityType::Text => {
                let mut features = HashMap::new();

                if let Some(text) = &content.extracted_text {
                    features.insert("length".to_string(), text.len().into());
                    features.insert(
                        "word_count".to_string(),
                        text.split_whitespace().count().into(),
                    );
                    features.insert("line_count".to_string(), text.lines().count().into());
                }

                Ok(features)
            }

            #[cfg(feature = "image-support")]
            ModalityType::Image => {
                if let Some(ref processor) = self.image_processor {
                    processor.extract_features(content).await
                } else {
                    Ok(HashMap::new())
                }
            }

            #[cfg(feature = "audio-support")]
            ModalityType::Audio => {
                if let Some(ref processor) = self.audio_processor {
                    processor.extract_features(content).await
                } else {
                    Ok(HashMap::new())
                }
            }

            #[cfg(feature = "video-support")]
            ModalityType::Video => {
                if let Some(ref processor) = self.video_processor {
                    processor.extract_features(content).await
                } else {
                    Ok(HashMap::new())
                }
            }

            _ => Ok(HashMap::new()),
        }
    }

    async fn extract_text(&self, content: &MediaContent) -> Result<Option<String>> {
        match content.modality_type() {
            ModalityType::Text => {
                if let Some(ref text) = content.extracted_text {
                    Ok(Some(text.clone()))
                } else {
                    // Try to extract from data
                    let data = content.data.as_bytes().await?;
                    let text = String::from_utf8_lossy(&data).to_string();
                    Ok(Some(text))
                }
            }

            #[cfg(feature = "image-support")]
            ModalityType::Image => {
                if let Some(ref processor) = self.image_processor {
                    processor.extract_text(content).await
                } else {
                    Ok(None)
                }
            }

            #[cfg(feature = "audio-support")]
            ModalityType::Audio => {
                if let Some(ref processor) = self.audio_processor {
                    processor.extract_text(content).await
                } else {
                    Ok(None)
                }
            }

            #[cfg(feature = "video-support")]
            ModalityType::Video => {
                if let Some(ref processor) = self.video_processor {
                    processor.extract_text(content).await
                } else {
                    Ok(None)
                }
            }

            _ => Ok(None),
        }
    }
}

/// Builder for configuring a multimodal processor.
#[derive(Debug, Default)]
pub struct MultimodalProcessorBuilder {
    #[cfg(feature = "image-support")]
    enable_image: bool,

    #[cfg(feature = "audio-support")]
    enable_audio: bool,

    #[cfg(feature = "video-support")]
    enable_video: bool,
}

impl MultimodalProcessorBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable image processing.
    #[cfg(feature = "image-support")]
    pub fn with_image_support(mut self) -> Self {
        self.enable_image = true;
        self
    }

    /// Enable audio processing.
    #[cfg(feature = "audio-support")]
    pub fn with_audio_support(mut self) -> Self {
        self.enable_audio = true;
        self
    }

    /// Enable video processing.
    #[cfg(feature = "video-support")]
    pub fn with_video_support(mut self) -> Self {
        self.enable_video = true;
        self
    }

    /// Build the multimodal processor.
    pub fn build(self) -> MultimodalProcessor {
        MultimodalProcessor {
            #[cfg(feature = "image-support")]
            image_processor: if self.enable_image {
                Some(ImageProcessor::new())
            } else {
                None
            },

            #[cfg(feature = "audio-support")]
            audio_processor: if self.enable_audio {
                Some(AudioProcessor::new())
            } else {
                None
            },

            #[cfg(feature = "video-support")]
            video_processor: if self.enable_video {
                Some(VideoProcessor::new())
            } else {
                None
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MediaContent;

    #[test]
    fn test_multimodal_processor_creation() {
        let processor = MultimodalProcessor::new();
        let modalities = processor.supported_input_modalities();

        // Text should always be supported
        assert!(modalities.contains(&ModalityType::Text));

        // Other modalities depend on feature flags
        #[cfg(feature = "image-support")]
        assert!(modalities.contains(&ModalityType::Image));

        #[cfg(feature = "audio-support")]
        assert!(modalities.contains(&ModalityType::Audio));

        #[cfg(feature = "video-support")]
        assert!(modalities.contains(&ModalityType::Video));
    }

    #[test]
    fn test_multimodal_processor_builder() {
        let mut builder = MultimodalProcessorBuilder::new();

        #[cfg(feature = "image-support")]
        {
            builder = builder.with_image_support();
        }

        #[cfg(feature = "audio-support")]
        {
            builder = builder.with_audio_support();
        }

        let processor = builder.build();
        let modalities = processor.supported_input_modalities();
        assert!(modalities.contains(&ModalityType::Text));
    }

    #[tokio::test]
    async fn test_text_processing() {
        let processor = MultimodalProcessor::new();
        let mut content =
            MediaContent::from_bytes(b"Hello, world!".to_vec(), MediaFormat::PlainText);
        // Set extracted_text for text content
        content.extracted_text = Some("Hello, world!".to_string());

        let options = ProcessingOptions::default();
        let result = processor.process(&content, &options).await;
        assert!(result.is_ok());

        let features = processor.extract_features(&content).await.unwrap();
        assert!(features.contains_key("length"));

        let text = processor.extract_text(&content).await.unwrap();
        assert!(text.is_some());
    }
}
