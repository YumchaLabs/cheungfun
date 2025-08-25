//! Simple test for multilingual functionality

#[cfg(test)]
mod tests {
    use super::super::{LanguageDetectionConfig, LanguageDetector, SupportedLanguage};

    #[test]
    fn test_basic_language_detection() {
        // Test with a simple configuration that only includes major languages
        let config = LanguageDetectionConfig {
            target_languages: Some(vec![
                SupportedLanguage::English,
                SupportedLanguage::Chinese,
                SupportedLanguage::Spanish,
                SupportedLanguage::French,
            ]),
            ..Default::default()
        };

        let detector = LanguageDetector::with_config(config);

        // For now, just test that the detector can be created
        assert!(
            detector.is_ok(),
            "Should be able to create language detector"
        );

        if let Ok(detector) = detector {
            // Test English detection
            let result = detector.detect("Hello, how are you today?");
            assert!(result.is_ok(), "Should be able to detect language");

            if let Ok(Some(detection)) = result {
                assert_eq!(detection.language, SupportedLanguage::English);
                println!("Detected English with confidence: {}", detection.confidence);
            }
        }
    }

    #[test]
    fn test_supported_language_iso_codes() {
        assert_eq!(SupportedLanguage::English.iso_code(), "en");
        assert_eq!(SupportedLanguage::Chinese.iso_code(), "zh");
        assert_eq!(SupportedLanguage::Japanese.iso_code(), "ja");
        assert_eq!(SupportedLanguage::Korean.iso_code(), "ko");
        assert_eq!(SupportedLanguage::Spanish.iso_code(), "es");
        assert_eq!(SupportedLanguage::French.iso_code(), "fr");
    }

    #[test]
    fn test_supported_language_names() {
        assert_eq!(SupportedLanguage::English.name(), "English");
        assert_eq!(SupportedLanguage::Chinese.name(), "Chinese (中文)");
        assert_eq!(SupportedLanguage::Japanese.name(), "Japanese (日本語)");
        assert_eq!(SupportedLanguage::Korean.name(), "Korean (한국어)");
    }

    #[test]
    fn test_language_from_iso_code() {
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
    }
}
