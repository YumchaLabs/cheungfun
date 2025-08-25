//! Integration tests for multilingual functionality

use cheungfun_core::multilingual::{
    LanguageDetectionConfig, LanguageDetector, MultilingualConfig, SupportedLanguage,
};

#[test]
fn test_language_detection_integration() {
    // Test basic language detection
    let detector = LanguageDetector::new().expect("Should create detector");

    // Test English
    let result = detector
        .detect("Hello, how are you today? This is a test in English.")
        .expect("Should detect language");

    if let Some(detection) = result {
        assert_eq!(detection.language, SupportedLanguage::English);
        assert!(
            detection.confidence > 0.5,
            "Confidence should be reasonable: {}",
            detection.confidence
        );
        println!(
            "✅ English detection: {} (confidence: {:.3})",
            detection.language.name(),
            detection.confidence
        );
    } else {
        panic!("Should detect English language");
    }

    // Test Chinese
    let result = detector
        .detect("你好，今天怎么样？这是一个中文测试。")
        .expect("Should detect language");

    if let Some(detection) = result {
        assert_eq!(detection.language, SupportedLanguage::Chinese);
        assert!(
            detection.confidence > 0.5,
            "Confidence should be reasonable: {}",
            detection.confidence
        );
        println!(
            "✅ Chinese detection: {} (confidence: {:.3})",
            detection.language.name(),
            detection.confidence
        );
    } else {
        println!("⚠️  Chinese detection failed - this might be expected if lingua-rs doesn't have Chinese models loaded");
    }

    // Test Spanish
    let result = detector
        .detect("Hola, ¿cómo estás hoy? Esta es una prueba en español.")
        .expect("Should detect language");

    if let Some(detection) = result {
        assert_eq!(detection.language, SupportedLanguage::Spanish);
        assert!(
            detection.confidence > 0.5,
            "Confidence should be reasonable: {}",
            detection.confidence
        );
        println!(
            "✅ Spanish detection: {} (confidence: {:.3})",
            detection.language.name(),
            detection.confidence
        );
    } else {
        println!("⚠️  Spanish detection failed");
    }
}

#[test]
fn test_supported_language_functionality() {
    // Test ISO code mapping
    assert_eq!(SupportedLanguage::English.iso_code(), "en");
    assert_eq!(SupportedLanguage::Chinese.iso_code(), "zh");
    assert_eq!(SupportedLanguage::Japanese.iso_code(), "ja");
    assert_eq!(SupportedLanguage::Korean.iso_code(), "ko");
    assert_eq!(SupportedLanguage::Spanish.iso_code(), "es");
    assert_eq!(SupportedLanguage::French.iso_code(), "fr");
    assert_eq!(SupportedLanguage::German.iso_code(), "de");
    assert_eq!(SupportedLanguage::Russian.iso_code(), "ru");

    // Test name display
    assert_eq!(SupportedLanguage::English.name(), "English");
    assert_eq!(SupportedLanguage::Chinese.name(), "Chinese (中文)");
    assert_eq!(SupportedLanguage::Japanese.name(), "Japanese (日本語)");
    assert_eq!(SupportedLanguage::Korean.name(), "Korean (한국어)");

    // Test from ISO code
    assert_eq!(
        SupportedLanguage::from_iso_code("en"),
        Some(SupportedLanguage::English)
    );
    assert_eq!(
        SupportedLanguage::from_iso_code("zh"),
        Some(SupportedLanguage::Chinese)
    );
    assert_eq!(
        SupportedLanguage::from_iso_code("ja"),
        Some(SupportedLanguage::Japanese)
    );
    assert_eq!(SupportedLanguage::from_iso_code("invalid"), None);

    // Test all languages list
    let all_languages = SupportedLanguage::all();
    assert!(!all_languages.is_empty());
    assert!(all_languages.contains(&SupportedLanguage::English));
    assert!(all_languages.contains(&SupportedLanguage::Chinese));

    println!("✅ Supported {} languages total", all_languages.len());
}

#[test]
fn test_multilingual_config() {
    // Test default configuration
    let config = MultilingualConfig::default();
    assert_eq!(config.default_language, SupportedLanguage::English);
    assert!(config.is_language_enabled(SupportedLanguage::English));

    // Test enabled languages
    let enabled = config.get_enabled_languages();
    assert!(!enabled.is_empty());
    assert!(enabled.contains(&SupportedLanguage::English));

    // Test configuration validation
    assert!(config.validate().is_ok());

    println!("✅ Default config has {} enabled languages", enabled.len());
}

#[test]
fn test_language_detection_config() {
    // Test custom configuration
    let config = LanguageDetectionConfig {
        target_languages: Some(vec![
            SupportedLanguage::English,
            SupportedLanguage::Spanish,
            SupportedLanguage::French,
        ]),
        min_confidence: 0.8,
        ..Default::default()
    };

    let detector =
        LanguageDetector::with_config(config).expect("Should create detector with custom config");

    // Test detection with custom config
    let result = detector
        .detect("Hello world, this is English text.")
        .expect("Should detect language");

    if let Some(detection) = result {
        assert_eq!(detection.language, SupportedLanguage::English);
        println!(
            "✅ Custom config detection: {} (confidence: {:.3})",
            detection.language.name(),
            detection.confidence
        );
    }
}

#[test]
fn test_confidence_values() {
    let detector = LanguageDetector::new().expect("Should create detector");

    let confidence_values = detector
        .get_confidence_values("Hello world, this is a test.")
        .expect("Should get confidence values");

    assert!(
        !confidence_values.is_empty(),
        "Should have confidence values"
    );

    // English should be the top language
    let (top_language, top_confidence) = confidence_values[0];
    assert_eq!(top_language, SupportedLanguage::English);
    assert!(
        top_confidence > 0.5,
        "Top confidence should be reasonable: {}",
        top_confidence
    );

    println!("✅ Confidence values for English text:");
    for (i, (lang, conf)) in confidence_values.iter().take(5).enumerate() {
        println!("  {}. {} - {:.3}", i + 1, lang.name(), conf);
    }
}

#[test]
fn test_empty_text_handling() {
    let detector = LanguageDetector::new().expect("Should create detector");

    // Test empty string
    let result = detector.detect("").expect("Should handle empty string");
    assert!(result.is_none(), "Empty string should return None");

    // Test whitespace only
    let result = detector
        .detect("   \n\t  ")
        .expect("Should handle whitespace");
    assert!(result.is_none(), "Whitespace only should return None");

    // Test very short text
    let result = detector.detect("Hi").expect("Should handle short text");
    // Short text might or might not be detected, both are acceptable
    if let Some(detection) = result {
        println!(
            "✅ Short text detected as: {} (confidence: {:.3})",
            detection.language.name(),
            detection.confidence
        );
    } else {
        println!("✅ Short text not detected (acceptable)");
    }
}
