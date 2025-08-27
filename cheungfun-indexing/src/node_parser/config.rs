//! Configuration system for node parsers.
//!
//! This module provides a unified configuration system for all node parsers,
//! following the design patterns from LlamaIndex while leveraging Rust's
//! type system for better safety and performance.

use crate::loaders::ProgrammingLanguage;
use crate::parsers::AstParserConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Base configuration for all node parsers.
///
/// This configuration contains common settings that apply to all types
/// of node parsers, such as metadata handling and relationship management.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeParserConfig {
    /// Whether to include metadata when processing nodes.
    pub include_metadata: bool,

    /// Whether to include previous/next node relationships.
    pub include_prev_next_rel: bool,

    /// Custom ID generation strategy identifier.
    /// This is a string identifier that maps to a specific ID generation function.
    pub id_func: Option<String>,

    /// Additional custom configuration parameters.
    pub custom_config: HashMap<String, serde_json::Value>,
}

impl Default for NodeParserConfig {
    fn default() -> Self {
        Self {
            include_metadata: true,
            include_prev_next_rel: true,
            id_func: None,
            custom_config: HashMap::new(),
        }
    }
}

impl NodeParserConfig {
    /// Create a new node parser configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to include metadata.
    pub fn with_include_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Set whether to include prev/next relationships.
    pub fn with_include_prev_next_rel(mut self, include: bool) -> Self {
        self.include_prev_next_rel = include;
        self
    }

    /// Set custom ID generation function.
    pub fn with_id_func(mut self, id_func: String) -> Self {
        self.id_func = Some(id_func);
        self
    }

    /// Add custom configuration parameter.
    pub fn with_custom_config<K: Into<String>>(mut self, key: K, value: serde_json::Value) -> Self {
        self.custom_config.insert(key.into(), value);
        self
    }
}

/// Configuration for text splitters.
///
/// This extends the base node parser configuration with text-specific
/// settings like chunk size, overlap, and separators.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextSplitterConfig {
    /// Base node parser configuration.
    pub base: NodeParserConfig,

    /// Target chunk size in tokens.
    pub chunk_size: usize,

    /// Overlap between chunks in tokens.
    pub chunk_overlap: usize,

    /// Primary separator for splitting text.
    pub separator: String,

    /// Backup separators to use if primary separator doesn't work.
    pub backup_separators: Vec<String>,

    /// Whether to preserve whitespace in chunks.
    pub keep_whitespaces: bool,

    /// Tokenizer configuration identifier.
    pub tokenizer: Option<String>,
}

impl Default for TextSplitterConfig {
    fn default() -> Self {
        Self {
            base: NodeParserConfig::default(),
            chunk_size: 1000,
            chunk_overlap: 200,
            separator: " ".to_string(),
            backup_separators: vec!["\n".to_string()],
            keep_whitespaces: false,
            tokenizer: None,
        }
    }
}

impl TextSplitterConfig {
    /// Create a new text splitter configuration.
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            ..Default::default()
        }
    }

    /// Set the separator.
    pub fn with_separator<S: Into<String>>(mut self, separator: S) -> Self {
        self.separator = separator.into();
        self
    }

    /// Set backup separators.
    pub fn with_backup_separators(mut self, separators: Vec<String>) -> Self {
        self.backup_separators = separators;
        self
    }

    /// Set whether to keep whitespaces.
    pub fn with_keep_whitespaces(mut self, keep: bool) -> Self {
        self.keep_whitespaces = keep;
        self
    }

    /// Set tokenizer configuration.
    pub fn with_tokenizer<S: Into<String>>(mut self, tokenizer: S) -> Self {
        self.tokenizer = Some(tokenizer.into());
        self
    }
}

/// Configuration for sentence splitters.
///
/// This extends text splitter configuration with sentence-specific
/// settings like paragraph separators and chunking strategies.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SentenceSplitterConfig {
    /// Base text splitter configuration.
    pub base: TextSplitterConfig,

    /// Separator between paragraphs.
    pub paragraph_separator: String,

    /// Secondary chunking regex pattern.
    pub secondary_chunking_regex: Option<String>,

    /// Chunking tokenizer function identifier.
    pub chunking_tokenizer: Option<String>,
}

impl Default for SentenceSplitterConfig {
    fn default() -> Self {
        Self {
            base: TextSplitterConfig::default(),
            paragraph_separator: "\n\n\n".to_string(),
            secondary_chunking_regex: Some("[^,.;。？！]+[,.;。？！]?".to_string()),
            chunking_tokenizer: None,
        }
    }
}

impl SentenceSplitterConfig {
    /// Create a new sentence splitter configuration.
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            base: TextSplitterConfig::new(chunk_size, chunk_overlap),
            ..Default::default()
        }
    }

    /// Set paragraph separator.
    pub fn with_paragraph_separator<S: Into<String>>(mut self, separator: S) -> Self {
        self.paragraph_separator = separator.into();
        self
    }

    /// Set secondary chunking regex.
    pub fn with_secondary_chunking_regex<S: Into<String>>(mut self, regex: S) -> Self {
        self.secondary_chunking_regex = Some(regex.into());
        self
    }

    /// Set chunking tokenizer.
    pub fn with_chunking_tokenizer<S: Into<String>>(mut self, tokenizer: S) -> Self {
        self.chunking_tokenizer = Some(tokenizer.into());
        self
    }
}

/// Configuration for token text splitters.
///
/// This configuration is specifically for token-based splitting,
/// which focuses on raw token counts rather than semantic boundaries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenTextSplitterConfig {
    /// Base text splitter configuration.
    pub base: TextSplitterConfig,

    /// Metadata format length reserve.
    pub metadata_format_len: usize,
}

impl Default for TokenTextSplitterConfig {
    fn default() -> Self {
        Self {
            base: TextSplitterConfig::default(),
            metadata_format_len: 2,
        }
    }
}

impl TokenTextSplitterConfig {
    /// Create a new token text splitter configuration.
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            base: TextSplitterConfig::new(chunk_size, chunk_overlap),
            ..Default::default()
        }
    }

    /// Set metadata format length.
    pub fn with_metadata_format_len(mut self, len: usize) -> Self {
        self.metadata_format_len = len;
        self
    }
}

/// Configuration for code splitters.
///
/// This configuration combines text splitting with AST-based code analysis
/// for intelligent code chunking that respects syntactic boundaries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodeSplitterConfig {
    /// Base text splitter configuration.
    pub base: TextSplitterConfig,

    /// Programming language for the code.
    pub language: ProgrammingLanguage,

    /// Maximum number of lines per chunk.
    pub chunk_lines: usize,

    /// Number of lines to overlap between chunks.
    pub chunk_lines_overlap: usize,

    /// Maximum number of characters per chunk.
    pub max_chars: usize,

    /// Whether to use AST-based splitting (recommended).
    pub use_ast_splitting: bool,

    /// AST parser configuration (used when use_ast_splitting is true).
    pub ast_config: Option<AstParserConfig>,

    /// Whether to respect function boundaries.
    pub respect_function_boundaries: bool,

    /// Whether to respect class boundaries.
    pub respect_class_boundaries: bool,

    /// Whether to preserve indentation context.
    pub preserve_indentation: bool,

    /// Minimum lines for a standalone chunk.
    pub min_chunk_lines: usize,
}

impl Default for CodeSplitterConfig {
    fn default() -> Self {
        Self {
            base: TextSplitterConfig::default(),
            language: ProgrammingLanguage::Rust,
            chunk_lines: 40,
            chunk_lines_overlap: 15,
            max_chars: 1500,
            use_ast_splitting: true,
            ast_config: None,
            respect_function_boundaries: true,
            respect_class_boundaries: true,
            preserve_indentation: true,
            min_chunk_lines: 5,
        }
    }
}

impl CodeSplitterConfig {
    /// Create a new code splitter configuration.
    pub fn new(
        language: ProgrammingLanguage,
        chunk_lines: usize,
        chunk_lines_overlap: usize,
        max_chars: usize,
    ) -> Self {
        Self {
            language,
            chunk_lines,
            chunk_lines_overlap,
            max_chars,
            ..Default::default()
        }
    }

    /// Enable or disable AST-based splitting.
    pub fn with_ast_splitting(mut self, enabled: bool) -> Self {
        self.use_ast_splitting = enabled;
        self
    }

    /// Set AST parser configuration.
    pub fn with_ast_config(mut self, config: AstParserConfig) -> Self {
        self.ast_config = Some(config);
        self
    }

    /// Set whether to respect function boundaries.
    pub fn with_respect_function_boundaries(mut self, respect: bool) -> Self {
        self.respect_function_boundaries = respect;
        self
    }

    /// Set whether to respect class boundaries.
    pub fn with_respect_class_boundaries(mut self, respect: bool) -> Self {
        self.respect_class_boundaries = respect;
        self
    }

    /// Set whether to preserve indentation.
    pub fn with_preserve_indentation(mut self, preserve: bool) -> Self {
        self.preserve_indentation = preserve;
        self
    }
}

/// Configuration for semantic splitters.
///
/// This configuration is for embedding-based semantic splitting,
/// which groups text based on semantic similarity rather than
/// syntactic or token-based boundaries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SemanticSplitterConfig {
    /// Base node parser configuration.
    pub base: NodeParserConfig,

    /// Number of sentences to group together when evaluating semantic similarity.
    pub buffer_size: usize,

    /// Percentile threshold for breakpoint distance.
    pub breakpoint_percentile_threshold: f32,

    /// Embedding model configuration identifier.
    pub embed_model: String,

    /// Sentence splitter function identifier.
    pub sentence_splitter: Option<String>,
}

impl Default for SemanticSplitterConfig {
    fn default() -> Self {
        Self {
            base: NodeParserConfig::default(),
            buffer_size: 1,
            breakpoint_percentile_threshold: 95.0,
            embed_model: "default".to_string(),
            sentence_splitter: None,
        }
    }
}

impl SemanticSplitterConfig {
    /// Create a new semantic splitter configuration.
    pub fn new<S: Into<String>>(embed_model: S) -> Self {
        Self {
            embed_model: embed_model.into(),
            ..Default::default()
        }
    }

    /// Set buffer size.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set breakpoint percentile threshold.
    pub fn with_breakpoint_percentile_threshold(mut self, threshold: f32) -> Self {
        self.breakpoint_percentile_threshold = threshold;
        self
    }

    /// Set sentence splitter.
    pub fn with_sentence_splitter<S: Into<String>>(mut self, splitter: S) -> Self {
        self.sentence_splitter = Some(splitter.into());
        self
    }
}
