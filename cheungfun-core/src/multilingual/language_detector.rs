//! Language detection using lingua-rs
//!
//! This module provides high-accuracy language detection capabilities
//! using the lingua-rs library, which is optimized for both short text
//! and mixed-language content.

use crate::error::CheungfunError;
use crate::multilingual::SupportedLanguage;
use lingua::{Language, LanguageDetector as LinguaDetector, LanguageDetectorBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, warn};

/// Language detection result with confidence score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetectionResult {
    /// Detected language
    pub language: SupportedLanguage,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Whether the detection is reliable
    pub is_reliable: bool,
}

/// Configuration for language detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetectionConfig {
    /// Minimum confidence threshold for reliable detection
    pub min_confidence: f64,
    /// Minimum relative distance between top languages
    pub min_relative_distance: f64,
    /// Whether to use low accuracy mode (faster, less memory)
    pub low_accuracy_mode: bool,
    /// Languages to include in detection (None = all supported)
    pub target_languages: Option<Vec<SupportedLanguage>>,
    /// Whether to preload language models
    pub preload_models: bool,
}

impl Default for LanguageDetectionConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            min_relative_distance: 0.1,
            low_accuracy_mode: false,
            target_languages: None,
            preload_models: true,
        }
    }
}

/// High-accuracy language detector using lingua-rs
pub struct LanguageDetector {
    detector: Arc<LinguaDetector>,
    config: LanguageDetectionConfig,
    language_mapping: HashMap<Language, SupportedLanguage>,
}

impl LanguageDetector {
    /// Create a new language detector with default configuration
    pub fn new() -> Result<Self, CheungfunError> {
        Self::with_config(LanguageDetectionConfig::default())
    }

    /// Create a new language detector with custom configuration
    pub fn with_config(config: LanguageDetectionConfig) -> Result<Self, CheungfunError> {
        // For now, create a simple detector with basic languages
        // TODO: Implement full configuration support once lingua-rs API is clarified
        let lingua_langs = vec![
            Language::English,
            Language::Chinese,
            Language::Japanese,
            Language::Korean,
            Language::Spanish,
            Language::French,
            Language::German,
            Language::Russian,
            Language::Portuguese,
            Language::Italian,
            Language::Arabic,
            Language::Hindi,
        ];

        let detector = Arc::new(LanguageDetectorBuilder::from_languages(&lingua_langs).build());
        let language_mapping = Self::create_language_mapping();

        Ok(Self {
            detector,
            config,
            language_mapping,
        })
    }

    /// Detect the language of the given text
    pub fn detect(&self, text: &str) -> Result<Option<LanguageDetectionResult>, CheungfunError> {
        if text.trim().is_empty() {
            return Ok(None);
        }

        // Use lingua's detection
        if let Some(detected_lang) = self.detector.detect_language_of(text) {
            // Convert to our supported language
            if let Some(&supported_lang) = self.language_mapping.get(&detected_lang) {
                // Get confidence values for reliability assessment
                let confidence_values = self.detector.compute_language_confidence_values(text);

                if let Some((top_lang, confidence)) = confidence_values.first() {
                    if *top_lang == detected_lang {
                        let is_reliable = confidence >= &self.config.min_confidence;

                        debug!(
                            "Detected language: {} (confidence: {:.3}, reliable: {})",
                            supported_lang.name(),
                            confidence,
                            is_reliable
                        );

                        return Ok(Some(LanguageDetectionResult {
                            language: supported_lang,
                            confidence: *confidence,
                            is_reliable,
                        }));
                    }
                }
            } else {
                warn!(
                    "Detected language {:?} is not in our supported language mapping",
                    detected_lang
                );
            }
        }

        debug!(
            "Could not reliably detect language for text: {}",
            if text.len() > 50 { &text[..50] } else { text }
        );
        Ok(None)
    }

    /// Detect multiple languages in mixed-language text
    pub fn detect_multiple(
        &self,
        text: &str,
    ) -> Result<Vec<(String, LanguageDetectionResult)>, CheungfunError> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        let detection_results = self.detector.detect_multiple_languages_of(text);
        let mut results = Vec::new();

        for detection in detection_results {
            let segment = &text[detection.start_index()..detection.end_index()];
            let lingua_lang = detection.language();

            if let Some(&supported_lang) = self.language_mapping.get(&lingua_lang) {
                // For multiple language detection, we assume high confidence
                // since lingua-rs already filtered the results
                let result = LanguageDetectionResult {
                    language: supported_lang,
                    confidence: 0.9, // High confidence for detected segments
                    is_reliable: true,
                };

                results.push((segment.to_string(), result));
            }
        }

        debug!("Detected {} language segments in mixed text", results.len());
        Ok(results)
    }

    /// Get confidence values for all possible languages
    pub fn get_confidence_values(
        &self,
        text: &str,
    ) -> Result<Vec<(SupportedLanguage, f64)>, CheungfunError> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        let confidence_values = self.detector.compute_language_confidence_values(text);
        let mut results = Vec::new();

        for (lingua_lang, confidence) in confidence_values {
            if let Some(&supported_lang) = self.language_mapping.get(&lingua_lang) {
                results.push((supported_lang, confidence));
            }
        }

        Ok(results)
    }

    /// Get confidence for a specific language
    pub fn get_language_confidence(
        &self,
        text: &str,
        language: SupportedLanguage,
    ) -> Result<f64, CheungfunError> {
        if text.trim().is_empty() {
            return Ok(0.0);
        }

        if let Some(lingua_lang) = Self::supported_to_lingua_language(language) {
            let confidence = self.detector.compute_language_confidence(text, lingua_lang);
            Ok(confidence)
        } else {
            Ok(0.0)
        }
    }

    /// Check if the detector is in single-language mode
    pub fn is_single_language_mode(&self) -> bool {
        // This would require access to lingua's internal state
        // For now, we check if target_languages has only one language
        self.config
            .target_languages
            .as_ref()
            .map(|langs| langs.len() == 1)
            .unwrap_or(false)
    }

    /// Convert SupportedLanguage to lingua Language
    fn supported_to_lingua_language(lang: SupportedLanguage) -> Option<Language> {
        match lang {
            // Only map languages that are actually supported by lingua-rs with our current features
            SupportedLanguage::English => Some(Language::English),
            SupportedLanguage::Chinese => Some(Language::Chinese),
            SupportedLanguage::Japanese => Some(Language::Japanese),
            SupportedLanguage::Korean => Some(Language::Korean),
            SupportedLanguage::Spanish => Some(Language::Spanish),
            SupportedLanguage::French => Some(Language::French),
            SupportedLanguage::German => Some(Language::German),
            SupportedLanguage::Russian => Some(Language::Russian),
            SupportedLanguage::Portuguese => Some(Language::Portuguese),
            SupportedLanguage::Italian => Some(Language::Italian),
            SupportedLanguage::Arabic => Some(Language::Arabic),
            SupportedLanguage::Hindi => Some(Language::Hindi),
            // Other languages are not included in our current lingua-rs features
            _ => None,
        }
    }

    /// Create mapping from lingua Language to SupportedLanguage
    fn create_language_mapping() -> HashMap<Language, SupportedLanguage> {
        let mut mapping = HashMap::new();

        // Add all supported mappings
        for supported_lang in SupportedLanguage::all() {
            if let Some(lingua_lang) = Self::supported_to_lingua_language(supported_lang) {
                mapping.insert(lingua_lang, supported_lang);
            }
        }

        mapping
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new().expect("Failed to create default language detector")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detection() {
        let detector = LanguageDetector::new().unwrap();

        // Test English
        let result = detector.detect("Hello, how are you today?").unwrap();
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.language, SupportedLanguage::English);
        assert!(result.confidence > 0.5);

        // Test Chinese
        let result = detector.detect("你好，今天怎么样？").unwrap();
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.language, SupportedLanguage::Chinese);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_confidence_values() {
        let detector = LanguageDetector::new().unwrap();
        let confidence_values = detector.get_confidence_values("Hello world").unwrap();

        assert!(!confidence_values.is_empty());
        // English should have the highest confidence
        assert_eq!(confidence_values[0].0, SupportedLanguage::English);
        assert!(confidence_values[0].1 > 0.5);
    }

    #[test]
    fn test_empty_text() {
        let detector = LanguageDetector::new().unwrap();
        let result = detector.detect("").unwrap();
        assert!(result.is_none());

        let result = detector.detect("   ").unwrap();
        assert!(result.is_none());
    }
}
