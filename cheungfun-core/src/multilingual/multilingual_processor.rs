//! Multilingual content processing
//!
//! This module provides comprehensive multilingual content processing capabilities,
//! including language-aware text splitting, preprocessing, and content analysis.

use crate::error::CheungfunError;
use crate::multilingual::{
    LanguageDetectionResult, LanguageDetector, MultilingualConfig, SupportedLanguage,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Result of multilingual content processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultilingualProcessingResult {
    /// Original text
    pub original_text: String,
    /// Detected primary language
    pub primary_language: Option<LanguageDetectionResult>,
    /// Language segments for mixed-language content
    pub language_segments: Vec<LanguageSegment>,
    /// Processed chunks with language information
    pub processed_chunks: Vec<ProcessedChunk>,
    /// Processing metadata
    pub metadata: ProcessingMetadata,
}

/// A segment of text in a specific language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageSegment {
    /// The text segment
    pub text: String,
    /// Detected language
    pub language: LanguageDetectionResult,
    /// Start position in original text
    pub start_index: usize,
    /// End position in original text
    pub end_index: usize,
}

/// A processed text chunk with language information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedChunk {
    /// The processed text chunk
    pub text: String,
    /// Language of this chunk
    pub language: SupportedLanguage,
    /// Confidence in language detection
    pub language_confidence: f64,
    /// Chunk index in the document
    pub chunk_index: usize,
    /// Whether this chunk spans multiple languages
    pub is_mixed_language: bool,
    /// Language-specific metadata
    pub language_metadata: HashMap<String, serde_json::Value>,
}

/// Processing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    /// Total number of languages detected
    pub languages_detected: usize,
    /// Primary language confidence
    pub primary_language_confidence: f64,
    /// Whether the content is mixed-language
    pub is_mixed_language: bool,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Number of chunks created
    pub total_chunks: usize,
}

/// Multilingual content processor
pub struct MultilingualProcessor {
    /// Language detector
    detector: Arc<LanguageDetector>,
    /// Multilingual configuration
    config: Arc<MultilingualConfig>,
    /// Language-specific processors
    language_processors: HashMap<SupportedLanguage, Box<dyn LanguageSpecificProcessor>>,
}

/// Trait for language-specific processing
pub trait LanguageSpecificProcessor: Send + Sync {
    /// Process text for this specific language
    fn process_text(
        &self,
        text: &str,
        language: SupportedLanguage,
    ) -> Result<Vec<String>, CheungfunError>;

    /// Apply language-specific preprocessing
    fn preprocess(&self, text: &str, language: SupportedLanguage)
        -> Result<String, CheungfunError>;

    /// Get language-specific stop words
    fn get_stop_words(&self, language: SupportedLanguage) -> Vec<String>;

    /// Check if text needs special handling for this language
    fn needs_special_handling(&self, text: &str, language: SupportedLanguage) -> bool;
}

/// Default language processor implementation
#[derive(Clone)]
pub struct DefaultLanguageProcessor;

impl LanguageSpecificProcessor for DefaultLanguageProcessor {
    fn process_text(
        &self,
        text: &str,
        language: SupportedLanguage,
    ) -> Result<Vec<String>, CheungfunError> {
        // Basic text splitting - can be enhanced with language-specific logic
        let chunk_size = match language {
            // CJK languages typically need smaller chunks
            SupportedLanguage::Chinese
            | SupportedLanguage::Japanese
            | SupportedLanguage::Korean => 256,
            // Other languages use standard chunk size
            _ => 512,
        };

        let chunks = self.split_text_by_size(text, chunk_size, 50);
        Ok(chunks)
    }

    fn preprocess(
        &self,
        text: &str,
        language: SupportedLanguage,
    ) -> Result<String, CheungfunError> {
        let mut processed = text.to_string();

        // Apply language-specific preprocessing
        match language {
            SupportedLanguage::Chinese | SupportedLanguage::Japanese => {
                // For CJK languages, normalize whitespace more aggressively
                processed = processed
                    .chars()
                    .map(|c| if c.is_whitespace() { ' ' } else { c })
                    .collect::<String>()
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
            }
            SupportedLanguage::Arabic | SupportedLanguage::Hebrew => {
                // For RTL languages, preserve text direction
                processed = processed.trim().to_string();
            }
            _ => {
                // Standard preprocessing for other languages
                processed = processed.trim().to_string();
            }
        }

        Ok(processed)
    }

    fn get_stop_words(&self, language: SupportedLanguage) -> Vec<String> {
        // Basic stop words - can be enhanced with comprehensive lists
        match language {
            SupportedLanguage::English => vec![
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
                "by",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            SupportedLanguage::Chinese => vec![
                "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "个", "上",
                "也", "很", "到", "说", "要", "去",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            SupportedLanguage::Japanese => vec![
                "の",
                "に",
                "は",
                "を",
                "が",
                "で",
                "と",
                "から",
                "まで",
                "より",
                "も",
                "だ",
                "である",
                "です",
                "ます",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            _ => vec![], // Add more languages as needed
        }
    }

    fn needs_special_handling(&self, _text: &str, language: SupportedLanguage) -> bool {
        match language {
            // CJK languages need special tokenization
            SupportedLanguage::Chinese
            | SupportedLanguage::Japanese
            | SupportedLanguage::Korean => true,
            // RTL languages need special handling
            SupportedLanguage::Arabic | SupportedLanguage::Hebrew => true,
            // Languages with complex scripts
            SupportedLanguage::Thai | SupportedLanguage::Burmese | SupportedLanguage::Khmer => true,
            _ => false,
        }
    }
}

impl DefaultLanguageProcessor {
    fn split_text_by_size(&self, text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();

        if chars.len() <= chunk_size {
            chunks.push(text.to_string());
            return chunks;
        }

        let mut start = 0;
        while start < chars.len() {
            let end = std::cmp::min(start + chunk_size, chars.len());
            let chunk: String = chars[start..end].iter().collect();
            chunks.push(chunk);

            if end >= chars.len() {
                break;
            }

            start = end - overlap;
        }

        chunks
    }
}

impl MultilingualProcessor {
    /// Create a new multilingual processor
    pub fn new(detector: Arc<LanguageDetector>, config: Arc<MultilingualConfig>) -> Self {
        let mut language_processors = HashMap::new();

        // Add default processor for all supported languages
        let default_processor = Box::new(DefaultLanguageProcessor);
        for language in SupportedLanguage::all() {
            language_processors.insert(
                language,
                default_processor.clone() as Box<dyn LanguageSpecificProcessor>,
            );
        }

        Self {
            detector,
            config,
            language_processors,
        }
    }

    /// Process multilingual content
    pub async fn process(
        &self,
        text: &str,
    ) -> Result<MultilingualProcessingResult, CheungfunError> {
        let start_time = std::time::Instant::now();

        if text.trim().is_empty() {
            return Ok(MultilingualProcessingResult {
                original_text: text.to_string(),
                primary_language: None,
                language_segments: vec![],
                processed_chunks: vec![],
                metadata: ProcessingMetadata {
                    languages_detected: 0,
                    primary_language_confidence: 0.0,
                    is_mixed_language: false,
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    total_chunks: 0,
                },
            });
        }

        // Detect primary language
        let primary_language = self.detector.detect(text)?;
        debug!("Primary language detected: {:?}", primary_language);

        // Detect multiple languages if enabled
        let language_segments = if self.config.global_settings.enable_cross_language_search {
            self.detect_language_segments(text).await?
        } else {
            // Create a single segment for the entire text
            if let Some(ref primary) = primary_language {
                vec![LanguageSegment {
                    text: text.to_string(),
                    language: primary.clone(),
                    start_index: 0,
                    end_index: text.len(),
                }]
            } else {
                // Fallback to default language
                vec![LanguageSegment {
                    text: text.to_string(),
                    language: LanguageDetectionResult {
                        language: self.config.default_language,
                        confidence: 0.5,
                        is_reliable: false,
                    },
                    start_index: 0,
                    end_index: text.len(),
                }]
            }
        };

        // Process chunks for each language segment
        let mut processed_chunks = Vec::new();
        let mut chunk_index = 0;

        for segment in &language_segments {
            let chunks = self
                .process_language_segment(segment, &mut chunk_index)
                .await?;
            processed_chunks.extend(chunks);
        }

        let processing_time = start_time.elapsed().as_millis() as u64;
        let languages_detected = language_segments
            .iter()
            .map(|s| s.language.language)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let metadata = ProcessingMetadata {
            languages_detected,
            primary_language_confidence: primary_language
                .as_ref()
                .map(|p| p.confidence)
                .unwrap_or(0.0),
            is_mixed_language: languages_detected > 1,
            processing_time_ms: processing_time,
            total_chunks: processed_chunks.len(),
        };

        info!(
            "Processed multilingual content: {} languages, {} chunks, {}ms",
            languages_detected,
            processed_chunks.len(),
            processing_time
        );

        Ok(MultilingualProcessingResult {
            original_text: text.to_string(),
            primary_language,
            language_segments,
            processed_chunks,
            metadata,
        })
    }

    /// Detect language segments in mixed-language text
    async fn detect_language_segments(
        &self,
        text: &str,
    ) -> Result<Vec<LanguageSegment>, CheungfunError> {
        let detection_results = self.detector.detect_multiple(text)?;

        let mut segments = Vec::new();
        for (segment_text, detection_result) in detection_results {
            // Find the position of this segment in the original text
            if let Some(start_pos) = text.find(&segment_text) {
                segments.push(LanguageSegment {
                    text: segment_text.clone(),
                    language: detection_result,
                    start_index: start_pos,
                    end_index: start_pos + segment_text.len(),
                });
            }
        }

        // If no segments found, create one for the entire text
        if segments.is_empty() {
            if let Some(primary_detection) = self.detector.detect(text)? {
                segments.push(LanguageSegment {
                    text: text.to_string(),
                    language: primary_detection,
                    start_index: 0,
                    end_index: text.len(),
                });
            }
        }

        Ok(segments)
    }

    /// Process a single language segment
    async fn process_language_segment(
        &self,
        segment: &LanguageSegment,
        chunk_index: &mut usize,
    ) -> Result<Vec<ProcessedChunk>, CheungfunError> {
        let language = segment.language.language;

        // Get language-specific processor
        let processor = self.language_processors.get(&language).ok_or_else(|| {
            CheungfunError::Configuration {
                message: format!("No processor found for language: {}", language.name()),
            }
        })?;

        // Preprocess the text
        let preprocessed_text = processor.preprocess(&segment.text, language)?;

        // Split into chunks
        let text_chunks = processor.process_text(&preprocessed_text, language)?;

        let mut processed_chunks = Vec::new();
        for chunk_text in text_chunks {
            let mut language_metadata = HashMap::new();
            language_metadata.insert(
                "stop_words".to_string(),
                serde_json::to_value(processor.get_stop_words(language))?,
            );
            language_metadata.insert(
                "needs_special_handling".to_string(),
                serde_json::to_value(processor.needs_special_handling(&chunk_text, language))?,
            );

            processed_chunks.push(ProcessedChunk {
                text: chunk_text,
                language,
                language_confidence: segment.language.confidence,
                chunk_index: *chunk_index,
                is_mixed_language: false, // Single language segment
                language_metadata,
            });

            *chunk_index += 1;
        }

        Ok(processed_chunks)
    }

    /// Add a custom language processor
    pub fn add_language_processor(
        &mut self,
        language: SupportedLanguage,
        processor: Box<dyn LanguageSpecificProcessor>,
    ) {
        self.language_processors.insert(language, processor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multilingual::LanguageDetectionConfig;

    #[tokio::test]
    async fn test_multilingual_processing() {
        let detector = Arc::new(LanguageDetector::new().unwrap());
        let config = Arc::new(MultilingualConfig::default());
        let processor = MultilingualProcessor::new(detector, config);

        let result = processor
            .process("Hello world! This is a test.")
            .await
            .unwrap();

        assert!(!result.processed_chunks.is_empty());
        assert!(result.primary_language.is_some());
        assert_eq!(
            result.primary_language.unwrap().language,
            SupportedLanguage::English
        );
    }

    #[tokio::test]
    async fn test_empty_text_processing() {
        let detector = Arc::new(LanguageDetector::new().unwrap());
        let config = Arc::new(MultilingualConfig::default());
        let processor = MultilingualProcessor::new(detector, config);

        let result = processor.process("").await.unwrap();

        assert!(result.processed_chunks.is_empty());
        assert!(result.primary_language.is_none());
        assert_eq!(result.metadata.languages_detected, 0);
    }
}
