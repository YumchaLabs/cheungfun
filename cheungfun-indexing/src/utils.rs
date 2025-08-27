//! Unified utility functions for the indexing module.
//!
//! This module consolidates common utility functions used across different
//! components of the indexing system, reducing code duplication and providing
//! a central location for shared functionality.

use regex::Regex;
use std::path::Path;

/// Text processing utilities.
pub mod text {
    use super::*;

    /// Clean and normalize text content.
    ///
    /// This function removes excessive whitespace and control characters
    /// while preserving newlines and tabs.
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
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }

        None
    }

    /// Count words in text.
    #[must_use]
    pub fn count_words(text: &str) -> usize {
        text.split_whitespace().count()
    }

    /// Count sentences in text (basic heuristic).
    #[must_use]
    pub fn count_sentences(text: &str) -> usize {
        let sentence_endings = ['.', '!', '?'];
        text.chars()
            .filter(|c| sentence_endings.contains(c))
            .count()
            .max(1) // At least 1 sentence
    }

    /// Calculate basic text statistics.
    #[derive(Debug, Clone)]
    pub struct TextStats {
        /// Number of characters in the text.
        pub char_count: usize,
        /// Number of words in the text.
        pub word_count: usize,
        /// Number of sentences in the text.
        pub sentence_count: usize,
        /// Number of lines in the text.
        pub line_count: usize,
    }

    impl TextStats {
        /// Create text statistics from text.
        #[must_use]
        pub fn from_text(text: &str) -> Self {
            Self {
                char_count: text.chars().count(),
                word_count: count_words(text),
                sentence_count: count_sentences(text),
                line_count: text.lines().count(),
            }
        }
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

        // Simple decision logic
        if chinese_chars > text.len() / 10 {
            Some("zh".to_string())
        } else if english_count > 0 {
            Some("en".to_string())
        } else {
            None
        }
    }

    /// Estimate reading time for text (in minutes).
    #[must_use]
    pub fn estimate_reading_time(text: &str) -> f64 {
        let stats = TextStats::from_text(text);
        // Average reading speed: 200-250 words per minute
        stats.word_count as f64 / 225.0
    }

    /// Check if text appears to be code based on heuristics.
    #[must_use]
    pub fn is_likely_code(text: &str) -> bool {
        let stats = TextStats::from_text(text);

        // Calculate punctuation ratio
        let punctuation_chars = text.chars().filter(|c| c.is_ascii_punctuation()).count();
        let punctuation_ratio = if stats.char_count > 0 {
            punctuation_chars as f64 / stats.char_count as f64
        } else {
            0.0
        };

        // Calculate average characters per word
        let avg_chars_per_word = if stats.word_count > 0 {
            stats.char_count as f64 / stats.word_count as f64
        } else {
            0.0
        };

        // Code typically has higher punctuation ratio and longer "words"
        punctuation_ratio > 0.15 || avg_chars_per_word > 8.0
    }
}

/// File and path utilities.
pub mod file {
    use super::*;

    /// Detect the content type of a file based on its extension.
    #[must_use]
    pub fn detect_content_type(path: &Path) -> Option<String> {
        let extension = path.extension()?.to_str()?.to_lowercase();

        match extension.as_str() {
            "txt" => Some("text/plain".to_string()),
            "md" | "markdown" => Some("text/markdown".to_string()),
            "html" | "htm" => Some("text/html".to_string()),
            "pdf" => Some("application/pdf".to_string()),
            "doc" => Some("application/msword".to_string()),
            "docx" => Some(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    .to_string(),
            ),
            "csv" => Some("text/csv".to_string()),
            "json" => Some("application/json".to_string()),
            "xml" => Some("application/xml".to_string()),
            "rtf" => Some("application/rtf".to_string()),
            // Programming languages
            "rs" => Some("text/x-rust".to_string()),
            "py" | "pyw" => Some("text/x-python".to_string()),
            "js" | "mjs" => Some("text/javascript".to_string()),
            "ts" | "tsx" => Some("text/typescript".to_string()),
            "java" => Some("text/x-java".to_string()),
            "c" => Some("text/x-c".to_string()),
            "cpp" | "cc" | "cxx" => Some("text/x-c++".to_string()),
            "h" | "hpp" => Some("text/x-c-header".to_string()),
            "cs" => Some("text/x-csharp".to_string()),
            "go" => Some("text/x-go".to_string()),
            "php" => Some("text/x-php".to_string()),
            "rb" => Some("text/x-ruby".to_string()),
            "swift" => Some("text/x-swift".to_string()),
            "kt" => Some("text/x-kotlin".to_string()),
            "scala" => Some("text/x-scala".to_string()),
            _ => None,
        }
    }

    /// Check if a file extension indicates a text file.
    #[must_use]
    pub fn is_text_file(path: &Path) -> bool {
        detect_content_type(path)
            .map(|content_type| content_type.starts_with("text/"))
            .unwrap_or(false)
    }

    /// Check if a file extension indicates a code file.
    #[must_use]
    pub fn is_code_file(path: &Path) -> bool {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase());

        if let Some(ext) = extension {
            matches!(
                ext.as_str(),
                "rs" | "py"
                    | "pyw"
                    | "js"
                    | "mjs"
                    | "ts"
                    | "tsx"
                    | "java"
                    | "c"
                    | "cpp"
                    | "cc"
                    | "cxx"
                    | "h"
                    | "hpp"
                    | "cs"
                    | "go"
                    | "php"
                    | "rb"
                    | "swift"
                    | "kt"
                    | "scala"
            )
        } else {
            false
        }
    }
}

/// Metadata utilities.
pub mod metadata {
    use std::collections::HashMap;

    /// Add chunk-specific metadata to a metadata map.
    pub fn add_chunk_metadata(
        metadata: &mut HashMap<String, serde_json::Value>,
        chunk_index: usize,
        chunk_text: &str,
        start_offset: usize,
        end_offset: usize,
    ) {
        use super::text::TextStats;

        metadata.insert(
            "chunk_index".to_string(),
            serde_json::Value::Number(chunk_index.into()),
        );
        metadata.insert(
            "start_offset".to_string(),
            serde_json::Value::Number(start_offset.into()),
        );
        metadata.insert(
            "end_offset".to_string(),
            serde_json::Value::Number(end_offset.into()),
        );

        let stats = TextStats::from_text(chunk_text);
        metadata.insert(
            "char_count".to_string(),
            serde_json::Value::Number(stats.char_count.into()),
        );
        metadata.insert(
            "word_count".to_string(),
            serde_json::Value::Number(stats.word_count.into()),
        );
        metadata.insert(
            "sentence_count".to_string(),
            serde_json::Value::Number(stats.sentence_count.into()),
        );
        metadata.insert(
            "line_count".to_string(),
            serde_json::Value::Number(stats.line_count.into()),
        );
    }
}
