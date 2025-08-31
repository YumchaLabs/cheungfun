//! Document transformers for processing and chunking content.
//!
//! This module provides implementations of the unified `Transform` trait
//! for various document processing tasks.

pub mod entity_extractor;
pub mod keyword_extractor;
pub mod llm_extractor;
pub mod metadata_extractor;
pub mod summary_extractor;
pub mod title_extractor;

pub use entity_extractor::{
    EntityExtractionConfig, EntityExtractor, ExtractedEntity, ExtractedRelationship,
};
pub use keyword_extractor::{KeywordExtractor, KeywordExtractorBuilder, KeywordExtractorConfig};
pub use llm_extractor::{LlmExtractionConfig, LlmExtractor};
pub use metadata_extractor::MetadataExtractor;
pub use summary_extractor::{
    SummaryExtractor, SummaryExtractorBuilder, SummaryExtractorConfig, SummaryType,
};
pub use title_extractor::{TitleExtractor, TitleExtractorBuilder, TitleExtractorConfig};

/// Configuration for metadata extraction.
#[derive(Debug, Clone)]
pub struct MetadataConfig {
    /// Whether to extract title from content.
    pub extract_title: bool,
    /// Whether to extract language information.
    pub extract_language: bool,
    /// Whether to extract content statistics.
    pub extract_statistics: bool,
    /// Whether to extract keywords.
    pub extract_keywords: bool,
    /// Maximum number of keywords to extract.
    pub max_keywords: usize,
    /// Whether to extract entities (requires NLP processing).
    pub extract_entities: bool,
    /// Custom metadata extractors.
    pub custom_extractors: Vec<String>,
}

impl Default for MetadataConfig {
    fn default() -> Self {
        Self {
            extract_title: true,
            extract_language: false, // Requires additional dependencies
            extract_statistics: true,
            extract_keywords: false, // Requires additional processing
            max_keywords: 10,
            extract_entities: false, // Requires NLP models
            custom_extractors: Vec::new(),
        }
    }
}

impl MetadataConfig {
    /// Create a new metadata configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable title extraction.
    #[must_use]
    pub fn with_title_extraction(mut self, extract: bool) -> Self {
        self.extract_title = extract;
        self
    }

    /// Enable language detection.
    #[must_use]
    pub fn with_language_detection(mut self, extract: bool) -> Self {
        self.extract_language = extract;
        self
    }

    /// Enable statistics extraction.
    #[must_use]
    pub fn with_statistics(mut self, extract: bool) -> Self {
        self.extract_statistics = extract;
        self
    }

    /// Enable keyword extraction.
    #[must_use]
    pub fn with_keyword_extraction(mut self, extract: bool, max_keywords: usize) -> Self {
        self.extract_keywords = extract;
        self.max_keywords = max_keywords;
        self
    }

    /// Enable entity extraction.
    #[must_use]
    pub fn with_entity_extraction(mut self, extract: bool) -> Self {
        self.extract_entities = extract;
        self
    }
}

/// Utility functions for text processing.
pub mod utils {
    use regex::Regex;

    /// Clean and normalize text content.
    #[must_use]
    pub fn clean_text(text: &str) -> String {
        // Remove excessive whitespace
        let whitespace_regex = Regex::new(r"\s+").unwrap();
        let cleaned = whitespace_regex.replace_all(text.trim(), " ");

        // Remove control characters except newlines and tabs
        cleaned
            .chars()
            .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
            .collect()
    }

    /// Extract title from text content (first line or heading).
    #[must_use]
    pub fn extract_title(text: &str) -> Option<String> {
        let lines: Vec<&str> = text.lines().collect();

        if lines.is_empty() {
            return None;
        }

        // Check for markdown heading
        if let Some(first_line) = lines.first() {
            if first_line.starts_with('#') {
                let title = first_line.trim_start_matches('#').trim();
                if !title.is_empty() {
                    return Some(title.to_string());
                }
            }
        }

        // Use first non-empty line as title
        for line in lines {
            let trimmed = line.trim();
            if !trimmed.is_empty() && trimmed.len() <= 200 {
                return Some(trimmed.to_string());
            }
        }

        None
    }

    /// Calculate basic text statistics.
    #[must_use]
    pub fn calculate_statistics(
        text: &str,
    ) -> std::collections::HashMap<String, serde_json::Value> {
        let mut stats = std::collections::HashMap::new();

        stats.insert(
            "character_count".to_string(),
            serde_json::Value::Number(text.len().into()),
        );
        stats.insert(
            "word_count".to_string(),
            serde_json::Value::Number(text.split_whitespace().count().into()),
        );
        stats.insert(
            "line_count".to_string(),
            serde_json::Value::Number(text.lines().count().into()),
        );
        stats.insert(
            "paragraph_count".to_string(),
            serde_json::Value::Number(
                text.split("\n\n")
                    .filter(|p| !p.trim().is_empty())
                    .count()
                    .into(),
            ),
        );

        // Calculate average word length
        let words: Vec<&str> = text.split_whitespace().collect();
        if !words.is_empty() {
            let avg_word_length =
                words.iter().map(|w| w.len()).sum::<usize>() as f64 / words.len() as f64;
            stats.insert(
                "avg_word_length".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(avg_word_length).unwrap_or_else(|| 0.into()),
                ),
            );
        }

        stats
    }

    /// Check if text appears to be in a specific language (basic heuristics).
    #[must_use]
    pub fn detect_language_simple(text: &str) -> Option<String> {
        // Very basic language detection based on common words
        let text_lower = text.to_lowercase();

        // English indicators
        let english_words = [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        ];
        let english_count = english_words
            .iter()
            .map(|word| text_lower.matches(word).count())
            .sum::<usize>();

        // Chinese indicators (simplified check)
        let chinese_chars = text
            .chars()
            .filter(|c| {
                let code = *c as u32;
                (0x4E00..=0x9FFF).contains(&code) // CJK Unified Ideographs
            })
            .count();

        if chinese_chars > text.len() / 4 {
            Some("zh".to_string())
        } else if english_count > 5 {
            Some("en".to_string())
        } else {
            None
        }
    }
}
