//! Language configuration management
//!
//! This module provides configuration management for multilingual features,
//! including language-specific settings, default language configuration,
//! and language mapping utilities.

use crate::error::CheungfunError;
use crate::multilingual::SupportedLanguage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Language-specific configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageSettings {
    /// Language code (ISO 639-1)
    pub code: String,
    /// Human-readable language name
    pub name: String,
    /// Whether this language is enabled
    pub enabled: bool,
    /// Language-specific text processing settings
    pub text_processing: TextProcessingSettings,
    /// Language-specific embedding settings
    pub embedding: EmbeddingSettings,
}

/// Text processing settings for a specific language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextProcessingSettings {
    /// Chunk size for text splitting
    pub chunk_size: usize,
    /// Chunk overlap for text splitting
    pub chunk_overlap: usize,
    /// Whether to use language-specific tokenization
    pub use_language_tokenizer: bool,
    /// Custom stop words for this language
    pub stop_words: Option<Vec<String>>,
    /// Whether to apply language-specific preprocessing
    pub apply_preprocessing: bool,
}

impl Default for TextProcessingSettings {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
            use_language_tokenizer: true,
            stop_words: None,
            apply_preprocessing: true,
        }
    }
}

/// Embedding settings for a specific language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingSettings {
    /// Preferred embedding model for this language
    pub preferred_model: Option<String>,
    /// Embedding dimension override
    pub dimension_override: Option<usize>,
    /// Whether to use multilingual embeddings
    pub use_multilingual: bool,
    /// Language-specific embedding parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

impl Default for EmbeddingSettings {
    fn default() -> Self {
        Self {
            preferred_model: None,
            dimension_override: None,
            use_multilingual: true,
            parameters: HashMap::new(),
        }
    }
}

/// Comprehensive multilingual configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultilingualConfig {
    /// Default language for the system
    pub default_language: SupportedLanguage,
    /// Language-specific settings
    pub languages: HashMap<SupportedLanguage, LanguageSettings>,
    /// Global multilingual settings
    pub global_settings: GlobalMultilingualSettings,
    /// Language detection configuration
    pub detection: LanguageDetectionSettings,
}

/// Global multilingual settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMultilingualSettings {
    /// Whether to enable automatic language detection
    pub auto_detect_language: bool,
    /// Whether to enable cross-language search
    pub enable_cross_language_search: bool,
    /// Whether to translate queries for better matching
    pub enable_query_translation: bool,
    /// Maximum number of languages to consider in detection
    pub max_detection_languages: usize,
    /// Whether to cache language detection results
    pub cache_detection_results: bool,
}

impl Default for GlobalMultilingualSettings {
    fn default() -> Self {
        Self {
            auto_detect_language: true,
            enable_cross_language_search: true,
            enable_query_translation: false, // Requires translation service
            max_detection_languages: 10,
            cache_detection_results: true,
        }
    }
}

/// Language detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetectionSettings {
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Minimum relative distance between languages
    pub min_relative_distance: f64,
    /// Whether to use low accuracy mode
    pub low_accuracy_mode: bool,
    /// Whether to preload language models
    pub preload_models: bool,
    /// Languages to include in detection
    pub target_languages: Option<Vec<SupportedLanguage>>,
}

impl Default for LanguageDetectionSettings {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            min_relative_distance: 0.1,
            low_accuracy_mode: false,
            preload_models: true,
            target_languages: None,
        }
    }
}

impl Default for MultilingualConfig {
    fn default() -> Self {
        let mut languages = HashMap::new();

        // Add default settings for major languages
        for lang in [
            SupportedLanguage::English,
            SupportedLanguage::Chinese,
            SupportedLanguage::Japanese,
            SupportedLanguage::Korean,
            SupportedLanguage::Spanish,
            SupportedLanguage::French,
            SupportedLanguage::German,
            SupportedLanguage::Russian,
        ] {
            languages.insert(
                lang,
                LanguageSettings {
                    code: lang.iso_code().to_string(),
                    name: lang.name().to_string(),
                    enabled: true,
                    text_processing: TextProcessingSettings::default(),
                    embedding: EmbeddingSettings::default(),
                },
            );
        }

        Self {
            default_language: SupportedLanguage::English,
            languages,
            global_settings: GlobalMultilingualSettings::default(),
            detection: LanguageDetectionSettings::default(),
        }
    }
}

impl MultilingualConfig {
    /// Create a new multilingual configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from a JSON file
    pub async fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, CheungfunError> {
        let content =
            tokio::fs::read_to_string(path)
                .await
                .map_err(|e| CheungfunError::Configuration {
                    message: format!("Failed to read multilingual config file: {}", e),
                })?;

        let config: Self =
            serde_json::from_str(&content).map_err(|e| CheungfunError::Configuration {
                message: format!("Failed to parse multilingual config: {}", e),
            })?;

        Ok(config)
    }

    /// Save configuration to a JSON file
    pub async fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), CheungfunError> {
        let content =
            serde_json::to_string_pretty(self).map_err(|e| CheungfunError::Configuration {
                message: format!("Failed to serialize multilingual config: {}", e),
            })?;

        tokio::fs::write(path, content)
            .await
            .map_err(|e| CheungfunError::Configuration {
                message: format!("Failed to write multilingual config file: {}", e),
            })?;

        Ok(())
    }

    /// Get settings for a specific language
    pub fn get_language_settings(&self, language: SupportedLanguage) -> Option<&LanguageSettings> {
        self.languages.get(&language)
    }

    /// Get settings for a specific language, falling back to default
    pub fn get_language_settings_or_default(
        &self,
        language: SupportedLanguage,
    ) -> &LanguageSettings {
        self.languages
            .get(&language)
            .or_else(|| self.languages.get(&self.default_language))
            .expect("Default language settings should always exist")
    }

    /// Check if a language is enabled
    pub fn is_language_enabled(&self, language: SupportedLanguage) -> bool {
        self.languages
            .get(&language)
            .map(|settings| settings.enabled)
            .unwrap_or(false)
    }

    /// Get all enabled languages
    pub fn get_enabled_languages(&self) -> Vec<SupportedLanguage> {
        self.languages
            .iter()
            .filter(|(_, settings)| settings.enabled)
            .map(|(lang, _)| *lang)
            .collect()
    }

    /// Add or update language settings
    pub fn set_language_settings(
        &mut self,
        language: SupportedLanguage,
        settings: LanguageSettings,
    ) {
        self.languages.insert(language, settings);
    }

    /// Enable a language
    pub fn enable_language(&mut self, language: SupportedLanguage) {
        if let Some(settings) = self.languages.get_mut(&language) {
            settings.enabled = true;
        } else {
            // Create default settings for the language
            let settings = LanguageSettings {
                code: language.iso_code().to_string(),
                name: language.name().to_string(),
                enabled: true,
                text_processing: TextProcessingSettings::default(),
                embedding: EmbeddingSettings::default(),
            };
            self.languages.insert(language, settings);
        }
    }

    /// Disable a language
    pub fn disable_language(&mut self, language: SupportedLanguage) {
        if let Some(settings) = self.languages.get_mut(&language) {
            settings.enabled = false;
        }
    }

    /// Set the default language
    pub fn set_default_language(&mut self, language: SupportedLanguage) {
        self.default_language = language;
        // Ensure the default language is enabled
        self.enable_language(language);
    }

    /// Get the default language
    pub fn get_default_language(&self) -> SupportedLanguage {
        self.default_language
    }

    /// Create a language mapping for external systems
    pub fn create_language_mapping(&self) -> HashMap<String, SupportedLanguage> {
        let mut mapping = HashMap::new();

        for (lang, settings) in &self.languages {
            if settings.enabled {
                mapping.insert(settings.code.clone(), *lang);
                mapping.insert(lang.iso_code().to_string(), *lang);
            }
        }

        mapping
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), CheungfunError> {
        // Check if default language is enabled
        if !self.is_language_enabled(self.default_language) {
            return Err(CheungfunError::Configuration {
                message: format!(
                    "Default language {} is not enabled",
                    self.default_language.name()
                ),
            });
        }

        // Check if at least one language is enabled
        if self.get_enabled_languages().is_empty() {
            return Err(CheungfunError::Configuration {
                message: "At least one language must be enabled".to_string(),
            });
        }

        // Validate detection settings
        if self.detection.min_confidence < 0.0 || self.detection.min_confidence > 1.0 {
            return Err(CheungfunError::Configuration {
                message: "Detection min_confidence must be between 0.0 and 1.0".to_string(),
            });
        }

        if self.detection.min_relative_distance < 0.0 || self.detection.min_relative_distance > 1.0
        {
            return Err(CheungfunError::Configuration {
                message: "Detection min_relative_distance must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MultilingualConfig::default();
        assert_eq!(config.default_language, SupportedLanguage::English);
        assert!(config.is_language_enabled(SupportedLanguage::English));
        assert!(!config.get_enabled_languages().is_empty());
    }

    #[test]
    fn test_language_management() {
        let mut config = MultilingualConfig::default();

        // Test enabling a new language
        config.enable_language(SupportedLanguage::Vietnamese);
        assert!(config.is_language_enabled(SupportedLanguage::Vietnamese));

        // Test disabling a language
        config.disable_language(SupportedLanguage::Vietnamese);
        assert!(!config.is_language_enabled(SupportedLanguage::Vietnamese));
    }

    #[test]
    fn test_validation() {
        let config = MultilingualConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.detection.min_confidence = 1.5; // Invalid
        assert!(invalid_config.validate().is_err());
    }
}
