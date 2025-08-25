//! Tests for multilingual configuration system

use cheungfun_core::multilingual::{
    EmbeddingSettings, LanguageSettings, MultilingualConfig, SupportedLanguage,
    TextProcessingSettings,
};
use std::collections::HashMap;

#[test]
fn test_default_multilingual_config() {
    let config = MultilingualConfig::default();

    // Test default language
    assert_eq!(config.default_language, SupportedLanguage::English);

    // Test that major languages are enabled by default
    assert!(config.is_language_enabled(SupportedLanguage::English));
    assert!(config.is_language_enabled(SupportedLanguage::Chinese));
    assert!(config.is_language_enabled(SupportedLanguage::Japanese));
    assert!(config.is_language_enabled(SupportedLanguage::Korean));
    assert!(config.is_language_enabled(SupportedLanguage::Spanish));
    assert!(config.is_language_enabled(SupportedLanguage::French));
    assert!(config.is_language_enabled(SupportedLanguage::German));
    assert!(config.is_language_enabled(SupportedLanguage::Russian));

    // Test enabled languages list
    let enabled = config.get_enabled_languages();
    assert!(!enabled.is_empty());
    assert!(enabled.len() >= 8); // At least the major languages

    // Test global settings
    assert!(config.global_settings.auto_detect_language);
    assert!(config.global_settings.enable_cross_language_search);
    assert!(!config.global_settings.enable_query_translation); // Disabled by default
    assert_eq!(config.global_settings.max_detection_languages, 10);
    assert!(config.global_settings.cache_detection_results);

    // Test detection settings
    assert_eq!(config.detection.min_confidence, 0.7);
    assert_eq!(config.detection.min_relative_distance, 0.1);
    assert!(!config.detection.low_accuracy_mode);
    assert!(config.detection.preload_models);
    assert!(config.detection.target_languages.is_none());

    // Test validation
    assert!(config.validate().is_ok());

    println!("✅ Default config validation passed");
}

#[test]
fn test_language_settings_customization() {
    let mut config = MultilingualConfig::default();

    // Test getting language settings
    let english_settings = config.get_language_settings(SupportedLanguage::English);
    assert!(english_settings.is_some());

    if let Some(settings) = english_settings {
        assert_eq!(settings.code, "en");
        assert_eq!(settings.name, "English");
        assert!(settings.enabled);
        assert_eq!(settings.text_processing.chunk_size, 512);
        assert_eq!(settings.text_processing.chunk_overlap, 50);
        assert!(settings.text_processing.use_language_tokenizer);
        assert!(settings.text_processing.apply_preprocessing);
        assert!(settings.embedding.use_multilingual);
    }

    // Test customizing language settings
    let custom_settings = LanguageSettings {
        code: "en".to_string(),
        name: "English (Custom)".to_string(),
        enabled: true,
        text_processing: TextProcessingSettings {
            chunk_size: 1024,
            chunk_overlap: 100,
            use_language_tokenizer: false,
            stop_words: Some(vec![
                "custom".to_string(),
                "stop".to_string(),
                "words".to_string(),
            ]),
            apply_preprocessing: false,
        },
        embedding: EmbeddingSettings {
            preferred_model: Some("custom-model".to_string()),
            dimension_override: Some(768),
            use_multilingual: false,
            parameters: {
                let mut params = HashMap::new();
                params.insert("temperature".to_string(), serde_json::json!(0.5));
                params
            },
        },
    };

    config.set_language_settings(SupportedLanguage::English, custom_settings);

    // Verify the custom settings
    let updated_settings = config
        .get_language_settings(SupportedLanguage::English)
        .unwrap();
    assert_eq!(updated_settings.name, "English (Custom)");
    assert_eq!(updated_settings.text_processing.chunk_size, 1024);
    assert_eq!(updated_settings.text_processing.chunk_overlap, 100);
    assert!(!updated_settings.text_processing.use_language_tokenizer);
    assert!(!updated_settings.text_processing.apply_preprocessing);
    assert_eq!(
        updated_settings.embedding.preferred_model,
        Some("custom-model".to_string())
    );
    assert_eq!(updated_settings.embedding.dimension_override, Some(768));
    assert!(!updated_settings.embedding.use_multilingual);

    println!("✅ Language settings customization passed");
}

#[test]
fn test_language_enable_disable() {
    let mut config = MultilingualConfig::default();

    // Test enabling a new language
    config.enable_language(SupportedLanguage::Vietnamese);
    assert!(config.is_language_enabled(SupportedLanguage::Vietnamese));

    let enabled = config.get_enabled_languages();
    assert!(enabled.contains(&SupportedLanguage::Vietnamese));

    // Test disabling a language
    config.disable_language(SupportedLanguage::Vietnamese);
    assert!(!config.is_language_enabled(SupportedLanguage::Vietnamese));

    let enabled = config.get_enabled_languages();
    assert!(!enabled.contains(&SupportedLanguage::Vietnamese));

    // Test that disabling doesn't remove the settings
    let settings = config.get_language_settings(SupportedLanguage::Vietnamese);
    assert!(settings.is_some());
    assert!(!settings.unwrap().enabled);

    println!("✅ Language enable/disable passed");
}

#[test]
fn test_default_language_management() {
    let mut config = MultilingualConfig::default();

    // Test initial default language
    assert_eq!(config.get_default_language(), SupportedLanguage::English);

    // Test changing default language
    config.set_default_language(SupportedLanguage::Chinese);
    assert_eq!(config.get_default_language(), SupportedLanguage::Chinese);

    // Test that setting default language enables it
    config.disable_language(SupportedLanguage::Spanish);
    assert!(!config.is_language_enabled(SupportedLanguage::Spanish));

    config.set_default_language(SupportedLanguage::Spanish);
    assert!(config.is_language_enabled(SupportedLanguage::Spanish));
    assert_eq!(config.get_default_language(), SupportedLanguage::Spanish);

    println!("✅ Default language management passed");
}

#[test]
fn test_language_mapping() {
    let config = MultilingualConfig::default();

    let mapping = config.create_language_mapping();
    assert!(!mapping.is_empty());

    // Test that enabled languages are in the mapping
    assert_eq!(mapping.get("en"), Some(&SupportedLanguage::English));
    assert_eq!(mapping.get("zh"), Some(&SupportedLanguage::Chinese));
    assert_eq!(mapping.get("ja"), Some(&SupportedLanguage::Japanese));
    assert_eq!(mapping.get("ko"), Some(&SupportedLanguage::Korean));
    assert_eq!(mapping.get("es"), Some(&SupportedLanguage::Spanish));
    assert_eq!(mapping.get("fr"), Some(&SupportedLanguage::French));
    assert_eq!(mapping.get("de"), Some(&SupportedLanguage::German));
    assert_eq!(mapping.get("ru"), Some(&SupportedLanguage::Russian));

    println!("✅ Language mapping has {} entries", mapping.len());
}

#[test]
fn test_config_validation() {
    let mut config = MultilingualConfig::default();

    // Test valid configuration
    assert!(config.validate().is_ok());

    // Test invalid configuration - disable default language
    config.disable_language(config.default_language);
    let result = config.validate();
    assert!(result.is_err());

    // Re-enable default language
    config.enable_language(config.default_language);
    assert!(config.validate().is_ok());

    // Test invalid detection settings
    config.detection.min_confidence = 1.5; // Invalid - should be <= 1.0
    let result = config.validate();
    assert!(result.is_err());

    config.detection.min_confidence = 0.7; // Valid
    assert!(config.validate().is_ok());

    config.detection.min_relative_distance = -0.1; // Invalid - should be >= 0.0
    let result = config.validate();
    assert!(result.is_err());

    config.detection.min_relative_distance = 0.1; // Valid
    assert!(config.validate().is_ok());

    println!("✅ Config validation passed");
}

#[test]
fn test_get_language_settings_or_default() {
    let config = MultilingualConfig::default();

    // Test getting settings for enabled language
    let english_settings = config.get_language_settings_or_default(SupportedLanguage::English);
    assert_eq!(english_settings.code, "en");

    // Test getting settings for disabled/non-configured language
    // Should fall back to default language settings
    let vietnamese_settings =
        config.get_language_settings_or_default(SupportedLanguage::Vietnamese);
    // Should get default language (English) settings
    assert_eq!(vietnamese_settings.code, "en");

    println!("✅ Get language settings or default passed");
}

#[tokio::test]
async fn test_config_serialization() {
    let config = MultilingualConfig::default();

    // Test JSON serialization
    let json = serde_json::to_string_pretty(&config).expect("Should serialize to JSON");
    assert!(!json.is_empty());

    // Test JSON deserialization
    let deserialized: MultilingualConfig =
        serde_json::from_str(&json).expect("Should deserialize from JSON");

    assert_eq!(deserialized.default_language, config.default_language);
    assert_eq!(deserialized.languages.len(), config.languages.len());
    assert_eq!(
        deserialized.global_settings.auto_detect_language,
        config.global_settings.auto_detect_language
    );

    println!("✅ Config serialization passed");
}
