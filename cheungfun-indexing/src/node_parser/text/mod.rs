//! Text-based node parsers and splitters.
//!
//! This module contains implementations of various text splitting strategies,
//! including sentence-based, token-based, code-aware, and semantic splitting.
//! All implementations follow the LlamaIndex patterns while leveraging Rust's
//! performance and safety features.

pub mod code;
pub mod sentence;
pub mod token;
pub mod utils;

// Re-export commonly used types
pub use code::CodeSplitter;
pub use sentence::SentenceSplitter;
pub use token::TokenTextSplitter;
pub use utils::*;

use cheungfun_core::Result as CoreResult;

use std::fmt::Debug;

/// Split represents a text fragment with metadata.
///
/// This structure is used internally by text splitters to track
/// text fragments along with their properties during the splitting process.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    /// The text content of this split.
    pub text: String,
    /// Whether this split represents a complete sentence.
    pub is_sentence: bool,
    /// The token count for this split.
    pub token_size: usize,
}

impl Split {
    /// Create a new split.
    pub fn new(text: String, is_sentence: bool, token_size: usize) -> Self {
        Self {
            text,
            is_sentence,
            token_size,
        }
    }
}

/// Tokenizer trait for counting tokens in text.
///
/// This trait abstracts different tokenization strategies,
/// allowing for pluggable tokenizers in text splitters.
pub trait Tokenizer: Send + Sync + Debug {
    /// Count the number of tokens in the given text.
    fn count_tokens(&self, text: &str) -> CoreResult<usize>;

    /// Encode text into tokens.
    fn encode(&self, text: &str) -> CoreResult<Vec<u32>>;

    /// Decode tokens back into text.
    fn decode(&self, tokens: &[u32]) -> CoreResult<String>;
}

/// Simple whitespace tokenizer.
///
/// This tokenizer splits text on whitespace and counts words as tokens.
/// It's fast but not as accurate as more sophisticated tokenizers.
#[derive(Debug, Clone)]
pub struct WhitespaceTokenizer;

impl Tokenizer for WhitespaceTokenizer {
    fn count_tokens(&self, text: &str) -> CoreResult<usize> {
        Ok(text.split_whitespace().count())
    }

    fn encode(&self, text: &str) -> CoreResult<Vec<u32>> {
        // Simple implementation: hash each word
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let tokens = text
            .split_whitespace()
            .map(|word| {
                let mut hasher = DefaultHasher::new();
                word.hash(&mut hasher);
                hasher.finish() as u32
            })
            .collect();

        Ok(tokens)
    }

    fn decode(&self, _tokens: &[u32]) -> CoreResult<String> {
        // Simple tokenizer doesn't support decoding
        Err(cheungfun_core::error::CheungfunError::Pipeline {
            message: "WhitespaceTokenizer doesn't support decoding".to_string(),
        })
    }
}

/// Character-based tokenizer.
///
/// This tokenizer treats each character as a token.
/// Useful for character-level processing or when working with languages
/// that don't use whitespace separation.
#[derive(Debug, Clone)]
pub struct CharacterTokenizer;

impl Tokenizer for CharacterTokenizer {
    fn count_tokens(&self, text: &str) -> CoreResult<usize> {
        Ok(text.chars().count())
    }

    fn encode(&self, text: &str) -> CoreResult<Vec<u32>> {
        let tokens = text.chars().map(|c| c as u32).collect();
        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> CoreResult<String> {
        let text: String = tokens
            .iter()
            .filter_map(|&token| char::from_u32(token))
            .collect();
        Ok(text)
    }
}

/// Split function trait for different splitting strategies.
///
/// This trait allows for pluggable splitting functions that can be
/// combined in different ways to achieve various splitting behaviors.
pub trait SplitFunction: Send + Sync + Debug {
    /// Split text using this function's strategy.
    fn split(&self, text: &str) -> CoreResult<Vec<String>>;

    /// Get the name of this split function.
    fn name(&self) -> &str;

    /// Clone this split function.
    fn clone_box(&self) -> Box<dyn SplitFunction>;
}

/// Split by separator function.
#[derive(Debug, Clone)]
pub struct SplitBySeparator {
    separator: String,
    keep_separator: bool,
}

impl SplitBySeparator {
    /// Create a new split by separator function.
    pub fn new<S: Into<String>>(separator: S, keep_separator: bool) -> Self {
        Self {
            separator: separator.into(),
            keep_separator,
        }
    }
}

impl SplitFunction for SplitBySeparator {
    fn split(&self, text: &str) -> CoreResult<Vec<String>> {
        if self.separator.is_empty() {
            return Ok(vec![text.to_string()]);
        }

        let parts: Vec<&str> = text.split(&self.separator).collect();

        if self.keep_separator && parts.len() > 1 {
            let mut result = Vec::new();
            for (i, part) in parts.iter().enumerate() {
                if i == 0 {
                    result.push(part.to_string());
                } else {
                    result.push(format!("{}{}", self.separator, part));
                }
            }
            Ok(result)
        } else {
            Ok(parts.into_iter().map(|s| s.to_string()).collect())
        }
    }

    fn name(&self) -> &str {
        "split_by_separator"
    }

    fn clone_box(&self) -> Box<dyn SplitFunction> {
        Box::new(self.clone())
    }
}

/// Split by regex function.
#[derive(Debug, Clone)]
pub struct SplitByRegex {
    pattern: String,
    keep_separator: bool,
}

impl SplitByRegex {
    /// Create a new split by regex function.
    pub fn new<S: Into<String>>(pattern: S, keep_separator: bool) -> Self {
        Self {
            pattern: pattern.into(),
            keep_separator,
        }
    }
}

impl SplitFunction for SplitByRegex {
    fn split(&self, text: &str) -> CoreResult<Vec<String>> {
        use regex::Regex;

        let regex = Regex::new(&self.pattern).map_err(|e| {
            cheungfun_core::error::CheungfunError::Pipeline {
                message: format!("Invalid regex pattern '{}': {}", self.pattern, e),
            }
        })?;

        if self.keep_separator {
            // Split while keeping the separator
            let mut result = Vec::new();
            let mut last_end = 0;

            for mat in regex.find_iter(text) {
                if mat.start() > last_end {
                    result.push(text[last_end..mat.start()].to_string());
                }
                result.push(text[mat.start()..mat.end()].to_string());
                last_end = mat.end();
            }

            if last_end < text.len() {
                result.push(text[last_end..].to_string());
            }

            Ok(result.into_iter().filter(|s| !s.is_empty()).collect())
        } else {
            Ok(regex
                .split(text)
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
                .collect())
        }
    }

    fn name(&self) -> &str {
        "split_by_regex"
    }

    fn clone_box(&self) -> Box<dyn SplitFunction> {
        Box::new(self.clone())
    }
}

/// Split by character function.
#[derive(Debug, Clone)]
pub struct SplitByCharacter;

impl SplitFunction for SplitByCharacter {
    fn split(&self, text: &str) -> CoreResult<Vec<String>> {
        Ok(text.chars().map(|c| c.to_string()).collect())
    }

    fn name(&self) -> &str {
        "split_by_character"
    }

    fn clone_box(&self) -> Box<dyn SplitFunction> {
        Box::new(SplitByCharacter)
    }
}

/// Factory functions for creating common tokenizers and split functions.

/// Create a tokenizer by name.
pub fn create_tokenizer(name: &str) -> CoreResult<Box<dyn Tokenizer>> {
    match name {
        "whitespace" => Ok(Box::new(WhitespaceTokenizer)),
        "character" => Ok(Box::new(CharacterTokenizer)),
        _ => Err(cheungfun_core::error::CheungfunError::Pipeline {
            message: format!("Unknown tokenizer: {}", name),
        }),
    }
}

/// Create common split functions for sentence splitting.
pub fn create_sentence_split_functions() -> CoreResult<Vec<Box<dyn SplitFunction>>> {
    Ok(vec![
        Box::new(SplitBySeparator::new("\n\n\n", false)), // Paragraphs
        Box::new(SplitBySeparator::new("\n\n", false)),   // Double newlines
        Box::new(SplitBySeparator::new("\n", false)),     // Single newlines
        Box::new(SplitByRegex::new(r"[.!?]+\s+", true)),  // Sentence endings
        Box::new(SplitBySeparator::new(". ", true)),      // Period + space
        Box::new(SplitBySeparator::new("! ", true)),      // Exclamation + space
        Box::new(SplitBySeparator::new("? ", true)),      // Question + space
        Box::new(SplitBySeparator::new(" ", false)),      // Words
        Box::new(SplitByCharacter),                       // Characters
    ])
}

/// Create common split functions for token splitting.
pub fn create_token_split_functions() -> CoreResult<Vec<Box<dyn SplitFunction>>> {
    Ok(vec![
        Box::new(SplitBySeparator::new(" ", false)),  // Words
        Box::new(SplitBySeparator::new("\n", false)), // Lines
        Box::new(SplitByCharacter),                   // Characters
    ])
}
