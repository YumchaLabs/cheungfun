//! Utility functions for text splitting operations.
//!
//! This module provides helper functions and utilities that are shared
//! across different text splitter implementations.

use cheungfun_core::Result as CoreResult;
use regex::Regex;
use std::collections::HashMap;
use unicode_segmentation::UnicodeSegmentation;

/// Text statistics for a given text chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct TextStats {
    /// Number of characters in the text.
    pub char_count: usize,
    /// Number of words in the text.
    pub word_count: usize,
    /// Number of lines in the text.
    pub line_count: usize,
    /// Number of sentences (estimated).
    pub sentence_count: usize,
    /// Number of paragraphs.
    pub paragraph_count: usize,
}

impl TextStats {
    /// Calculate statistics for the given text.
    pub fn from_text(text: &str) -> Self {
        let char_count = text.graphemes(true).count();
        let word_count = text.split_whitespace().count();
        let line_count = text.lines().count().max(1); // At least 1 line

        // Estimate sentence count by counting sentence-ending punctuation
        let sentence_count = count_sentences(text);

        // Count paragraphs by double newlines
        let paragraph_count = text.split("\n\n").count().max(1);

        Self {
            char_count,
            word_count,
            line_count,
            sentence_count,
            paragraph_count,
        }
    }
}

/// Count estimated number of sentences in text.
fn count_sentences(text: &str) -> usize {
    // Simple sentence counting based on punctuation
    let sentence_endings = ['.', '!', '?'];
    let mut count = 0;
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if sentence_endings.contains(&ch) {
            // Check if next character is whitespace or end of text
            if let Some(&next_ch) = chars.peek() {
                if next_ch.is_whitespace() || next_ch.is_uppercase() {
                    count += 1;
                }
            } else {
                // End of text
                count += 1;
            }
        }
    }

    count.max(1) // At least 1 sentence
}

/// Split text by multiple separators in order of preference.
///
/// This function tries each separator in order and returns the result
/// from the first separator that produces multiple splits.
pub fn split_by_separators(text: &str, separators: &[&str]) -> Vec<String> {
    for separator in separators {
        if separator.is_empty() {
            continue;
        }

        let splits: Vec<&str> = text.split(separator).collect();
        if splits.len() > 1 {
            return splits
                .into_iter()
                .map(|s| s.to_string())
                .filter(|s| !s.trim().is_empty())
                .collect();
        }
    }

    // If no separator worked, return the original text
    vec![text.to_string()]
}

/// Split text by regex pattern.
pub fn split_by_regex(text: &str, pattern: &str) -> CoreResult<Vec<String>> {
    let regex =
        Regex::new(pattern).map_err(|e| cheungfun_core::error::CheungfunError::Pipeline {
            message: format!("Invalid regex pattern '{}': {}", pattern, e),
        })?;

    let splits: Vec<String> = regex
        .split(text)
        .map(|s| s.to_string())
        .filter(|s| !s.trim().is_empty())
        .collect();

    if splits.is_empty() {
        Ok(vec![text.to_string()])
    } else {
        Ok(splits)
    }
}

/// Find optimal split points in text based on various criteria.
pub fn find_split_points(text: &str, max_chunk_size: usize) -> Vec<usize> {
    let mut split_points = Vec::new();
    let chars: Vec<char> = text.chars().collect();

    if chars.len() <= max_chunk_size {
        return split_points;
    }

    let mut current_pos = 0;

    while current_pos + max_chunk_size < chars.len() {
        let search_start = current_pos + max_chunk_size;
        let search_end = (search_start + max_chunk_size / 4).min(chars.len());

        // Look for good split points in the search window
        let mut best_split = search_start;

        // Prefer sentence endings
        for i in search_start..search_end {
            if matches!(chars[i], '.' | '!' | '?')
                && i + 1 < chars.len()
                && chars[i + 1].is_whitespace()
            {
                best_split = i + 1;
                break;
            }
        }

        // If no sentence ending, look for paragraph breaks
        if best_split == search_start {
            for i in search_start..search_end {
                if i + 1 < chars.len() && chars[i] == '\n' && chars[i + 1] == '\n' {
                    best_split = i + 2;
                    break;
                }
            }
        }

        // If no paragraph break, look for line breaks
        if best_split == search_start {
            for i in search_start..search_end {
                if chars[i] == '\n' {
                    best_split = i + 1;
                    break;
                }
            }
        }

        // If no line break, look for word boundaries
        if best_split == search_start {
            for i in search_start..search_end {
                if chars[i].is_whitespace() {
                    best_split = i + 1;
                    break;
                }
            }
        }

        split_points.push(best_split);
        current_pos = best_split;
    }

    split_points
}

/// Clean and normalize text for splitting.
pub fn normalize_text(text: &str) -> String {
    // Remove excessive whitespace while preserving structure
    let mut result = String::new();
    let mut prev_was_space = false;
    let mut prev_was_newline = false;

    for ch in text.chars() {
        match ch {
            ' ' | '\t' => {
                if !prev_was_space && !prev_was_newline {
                    result.push(' ');
                    prev_was_space = true;
                }
                prev_was_newline = false;
            }
            '\n' => {
                if !prev_was_newline {
                    result.push('\n');
                    prev_was_newline = true;
                }
                prev_was_space = false;
            }
            '\r' => {
                // Skip carriage returns
                continue;
            }
            _ => {
                result.push(ch);
                prev_was_space = false;
                prev_was_newline = false;
            }
        }
    }

    result.trim().to_string()
}

/// Extract text features for analysis.
pub fn extract_text_features(text: &str) -> HashMap<String, f64> {
    let mut features = HashMap::new();
    let stats = TextStats::from_text(text);

    features.insert("char_count".to_string(), stats.char_count as f64);
    features.insert("word_count".to_string(), stats.word_count as f64);
    features.insert("line_count".to_string(), stats.line_count as f64);
    features.insert("sentence_count".to_string(), stats.sentence_count as f64);
    features.insert("paragraph_count".to_string(), stats.paragraph_count as f64);

    // Calculate ratios
    if stats.word_count > 0 {
        features.insert(
            "avg_chars_per_word".to_string(),
            stats.char_count as f64 / stats.word_count as f64,
        );
    }

    if stats.sentence_count > 0 {
        features.insert(
            "avg_words_per_sentence".to_string(),
            stats.word_count as f64 / stats.sentence_count as f64,
        );
    }

    if stats.line_count > 0 {
        features.insert(
            "avg_words_per_line".to_string(),
            stats.word_count as f64 / stats.line_count as f64,
        );
    }

    // Text density features
    let whitespace_count = text.chars().filter(|c| c.is_whitespace()).count();
    features.insert(
        "whitespace_ratio".to_string(),
        whitespace_count as f64 / stats.char_count as f64,
    );

    let punctuation_count = text.chars().filter(|c| c.is_ascii_punctuation()).count();
    features.insert(
        "punctuation_ratio".to_string(),
        punctuation_count as f64 / stats.char_count as f64,
    );

    features
}

/// Check if text appears to be code based on heuristics.
pub fn is_likely_code(text: &str) -> bool {
    let features = extract_text_features(text);

    // Heuristics for code detection
    let punctuation_ratio = features.get("punctuation_ratio").copied().unwrap_or(0.0);
    let avg_chars_per_word = features.get("avg_chars_per_word").copied().unwrap_or(0.0);

    // Code typically has higher punctuation ratio and longer "words"
    punctuation_ratio > 0.15 || avg_chars_per_word > 8.0
}

/// Estimate reading time for text (in minutes).
pub fn estimate_reading_time(text: &str) -> f64 {
    let stats = TextStats::from_text(text);
    // Average reading speed: 200-250 words per minute
    stats.word_count as f64 / 225.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_stats() {
        let text = "Hello world! This is a test. How are you?\n\nThis is a new paragraph.";
        let stats = TextStats::from_text(text);

        assert!(stats.word_count > 0);
        assert!(stats.sentence_count >= 3);
        assert_eq!(stats.paragraph_count, 2);
    }

    #[test]
    fn test_split_by_separators() {
        let text = "Hello world. This is a test. How are you?";
        let separators = &[". ", " "];
        let splits = split_by_separators(text, separators);

        assert!(splits.len() > 1);
        assert!(splits.iter().any(|s| s.contains("Hello")));
    }

    #[test]
    fn test_normalize_text() {
        let text = "Hello    world!\n\n\nThis   is  a   test.\r\n";
        let normalized = normalize_text(text);

        assert!(!normalized.contains("    "));
        assert!(!normalized.contains("\r"));
        assert!(normalized.contains("Hello world!"));
    }

    #[test]
    fn test_is_likely_code() {
        let code_text = "fn main() { println!(\"Hello, world!\"); }";
        let natural_text = "This is a natural language sentence with normal punctuation.";

        assert!(is_likely_code(code_text));
        assert!(!is_likely_code(natural_text));
    }
}
