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

/// Configuration for semantic splitters.
///
/// This configuration is for semantic-based splitting that uses embeddings
/// to group semantically related sentences together.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SemanticSplitterConfig {
    /// Base node parser configuration.
    pub base: NodeParserConfig,

    /// Number of sentences to group together when evaluating semantic similarity.
    /// Set to 1 to consider each sentence individually.
    /// Set to >1 to group sentences together for context.
    pub buffer_size: usize,

    /// The percentile of cosine dissimilarity that must be exceeded between
    /// a group of sentences and the next to form a node. The smaller this
    /// number is, the more nodes will be generated.
    pub breakpoint_percentile_threshold: f32,

    /// Sentence splitter function identifier.
    pub sentence_splitter: Option<String>,

    /// Embedder configuration identifier for semantic similarity calculation.
    pub embedder_config: Option<String>,
}

impl Default for SemanticSplitterConfig {
    fn default() -> Self {
        Self {
            base: NodeParserConfig::default(),
            buffer_size: 1,
            breakpoint_percentile_threshold: 95.0,
            sentence_splitter: None,
            embedder_config: None,
        }
    }
}

impl SemanticSplitterConfig {
    /// Create a new semantic splitter configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set buffer size.
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
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

    /// Set embedder configuration.
    pub fn with_embedder_config<S: Into<String>>(mut self, config: S) -> Self {
        self.embedder_config = Some(config.into());
        self
    }
}

/// Configuration for sentence window splitters.
///
/// This configuration is for sentence window splitting that creates nodes
/// for individual sentences while preserving surrounding context in metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SentenceWindowConfig {
    /// Base node parser configuration.
    pub base: NodeParserConfig,

    /// The number of sentences on each side of a sentence to capture.
    pub window_size: usize,

    /// The metadata key to store the sentence window under.
    pub window_metadata_key: String,

    /// The metadata key to store the original sentence in.
    pub original_text_metadata_key: String,

    /// Sentence splitter function identifier.
    pub sentence_splitter: Option<String>,
}

impl Default for SentenceWindowConfig {
    fn default() -> Self {
        Self {
            base: NodeParserConfig::default(),
            window_size: 3,
            window_metadata_key: "window".to_string(),
            original_text_metadata_key: "original_text".to_string(),
            sentence_splitter: None,
        }
    }
}

impl SentenceWindowConfig {
    /// Create a new sentence window configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set window size.
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Set window metadata key.
    pub fn with_window_metadata_key<S: Into<String>>(mut self, key: S) -> Self {
        self.window_metadata_key = key.into();
        self
    }

    /// Set original text metadata key.
    pub fn with_original_text_metadata_key<S: Into<String>>(mut self, key: S) -> Self {
        self.original_text_metadata_key = key.into();
        self
    }

    /// Set sentence splitter.
    pub fn with_sentence_splitter<S: Into<String>>(mut self, splitter: S) -> Self {
        self.sentence_splitter = Some(splitter.into());
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

/// Predefined chunking strategies for different use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkingStrategy {
    /// Optimal strategy for RAG applications using SweepAI-enhanced AST chunking (40 lines, 15 overlap, 1500 chars)
    /// Best for: General code analysis, knowledge retrieval, documentation
    /// Note: This strategy uses the SweepAI algorithm for intelligent code splitting
    Optimal,
    /// Fine-grained strategy (15 lines, 5 overlap, 800 chars)
    /// Best for: Detailed code analysis, function-level processing
    Fine,
    /// Balanced strategy (30 lines, 10 overlap, 1200 chars)
    /// Best for: Most general use cases, medium-sized projects
    Balanced,
    /// Coarse strategy (50 lines, 15 overlap, 2000 chars)
    /// Best for: High-level code overview, large file processing
    Coarse,
    /// Minimal strategy (10 lines, 3 overlap, 500 chars)
    /// Best for: Very detailed analysis, small functions
    Minimal,
    /// Enterprise strategy for large codebases (60 lines, 20 overlap, 2500 chars)
    /// Best for: Large projects like Unity3D, enterprise applications
    Enterprise,
}

impl ChunkingStrategy {
    /// Get the parameters for this chunking strategy
    pub fn params(self) -> (usize, usize, usize) {
        match self {
            ChunkingStrategy::Optimal => (40, 15, 1500),
            ChunkingStrategy::Fine => (15, 5, 800),
            ChunkingStrategy::Balanced => (30, 10, 1200),
            ChunkingStrategy::Coarse => (50, 15, 2000),
            ChunkingStrategy::Minimal => (10, 3, 500),
            ChunkingStrategy::Enterprise => (60, 20, 2500),
        }
    }

    /// Get a description of this strategy
    pub fn description(self) -> &'static str {
        match self {
            ChunkingStrategy::Optimal => "Optimal for general RAG applications (SweepAI-enhanced)",
            ChunkingStrategy::Fine => "Fine-grained for detailed code analysis",
            ChunkingStrategy::Balanced => "Balanced for most general use cases",
            ChunkingStrategy::Coarse => "Coarse for high-level code overview",
            ChunkingStrategy::Minimal => "Minimal for very detailed analysis",
            ChunkingStrategy::Enterprise => "Enterprise for large codebases like Unity3D",
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

    /// Create a configuration with a predefined chunking strategy.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cheungfun_indexing::node_parser::config::{CodeSplitterConfig, ChunkingStrategy};
    /// use cheungfun_indexing::loaders::ProgrammingLanguage;
    ///
    /// // Use SweepAI-optimized strategy for Rust code
    /// let config = CodeSplitterConfig::with_strategy(
    ///     ProgrammingLanguage::Rust,
    ///     ChunkingStrategy::SweepAI
    /// );
    ///
    /// // Use fine-grained strategy for detailed analysis
    /// let config = CodeSplitterConfig::with_strategy(
    ///     ProgrammingLanguage::Python,
    ///     ChunkingStrategy::Fine
    /// );
    /// ```
    pub fn with_strategy(language: ProgrammingLanguage, strategy: ChunkingStrategy) -> Self {
        let (chunk_lines, chunk_lines_overlap, max_chars) = strategy.params();
        Self {
            language,
            chunk_lines,
            chunk_lines_overlap,
            max_chars,
            ..Default::default()
        }
    }

    /// Create an optimal configuration for RAG applications (recommended default).
    ///
    /// This uses research-backed parameters optimized for code retrieval:
    /// - 40 lines per chunk
    /// - 15 lines overlap
    /// - 1500 max characters
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cheungfun_indexing::node_parser::config::CodeSplitterConfig;
    /// use cheungfun_indexing::loaders::ProgrammingLanguage;
    ///
    /// let config = CodeSplitterConfig::optimal(ProgrammingLanguage::Rust);
    /// ```
    pub fn optimal(language: ProgrammingLanguage) -> Self {
        Self::with_strategy(language, ChunkingStrategy::Optimal)
    }

    /// Create an enterprise configuration for large codebases.
    ///
    /// This is specifically designed for large projects like Unity3D:
    /// - 60 lines per chunk (larger context)
    /// - 20 lines overlap (better continuity)
    /// - 2500 max characters (handles complex classes)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cheungfun_indexing::node_parser::config::CodeSplitterConfig;
    /// use cheungfun_indexing::loaders::ProgrammingLanguage;
    ///
    /// let config = CodeSplitterConfig::enterprise(ProgrammingLanguage::CSharp);
    /// ```
    pub fn enterprise(language: ProgrammingLanguage) -> Self {
        Self::with_strategy(language, ChunkingStrategy::Enterprise)
    }

    /// Create a fine-grained configuration for detailed analysis.
    pub fn fine_grained(language: ProgrammingLanguage) -> Self {
        Self::with_strategy(language, ChunkingStrategy::Fine)
    }

    /// Create a balanced configuration for general use.
    pub fn balanced(language: ProgrammingLanguage) -> Self {
        Self::with_strategy(language, ChunkingStrategy::Balanced)
    }

    /// Create a coarse configuration for high-level overview.
    pub fn coarse_grained(language: ProgrammingLanguage) -> Self {
        Self::with_strategy(language, ChunkingStrategy::Coarse)
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

/// Configuration for markdown node parser.
///
/// This configuration is for parsing Markdown documents using header-based splitting logic.
/// Each node contains its text content and the path of headers leading to it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MarkdownConfig {
    /// Base node parser configuration.
    pub base: NodeParserConfig,

    /// Separator character used for section header path metadata.
    /// Default is "/" to create paths like "/Header1/Header2/".
    pub header_path_separator: String,

    /// Whether to include code blocks in parsing.
    /// When false, content inside ```code blocks``` is treated as regular text.
    pub include_code_blocks: bool,

    /// Whether to preserve header hierarchy in metadata.
    /// When true, adds header_path metadata with the full path to the section.
    pub preserve_header_hierarchy: bool,

    /// Maximum header depth to process (1-6, where 1 is #, 6 is ######).
    /// Headers deeper than this level will be treated as regular text.
    pub max_header_depth: usize,

    /// Whether to include the header text in the node content.
    /// When true, the header line is included in the node text.
    pub include_header_in_content: bool,

    /// Minimum content length for a section to be considered a separate node.
    /// Sections shorter than this will be merged with adjacent sections.
    pub min_section_length: usize,
}

/// Configuration for hierarchical node parser.
///
/// This configuration defines multiple levels of chunking with different sizes,
/// creating a hierarchy where larger chunks contain smaller chunks as children.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HierarchicalConfig {
    /// Base node parser configuration.
    pub base: NodeParserConfig,

    /// Chunk sizes for each hierarchy level (ordered from largest to smallest).
    /// For example: [2048, 512, 128] creates 3 levels of hierarchy.
    pub chunk_sizes: Vec<usize>,

    /// Overlap between chunks at each level.
    pub chunk_overlap: usize,

    /// Whether to include metadata about hierarchy level.
    pub include_hierarchy_metadata: bool,

    /// Maximum depth of hierarchy to create.
    pub max_depth: Option<usize>,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            base: NodeParserConfig::default(),
            chunk_sizes: vec![2048, 512, 128], // Default 3-level hierarchy
            chunk_overlap: 20,
            include_hierarchy_metadata: true,
            max_depth: None,
        }
    }
}

impl HierarchicalConfig {
    /// Create a new hierarchical configuration with the given chunk sizes.
    pub fn new(chunk_sizes: Vec<usize>) -> Self {
        Self {
            chunk_sizes,
            ..Default::default()
        }
    }

    /// Set the chunk overlap for all levels.
    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }

    /// Set whether to include hierarchy metadata.
    pub fn with_hierarchy_metadata(mut self, include: bool) -> Self {
        self.include_hierarchy_metadata = include;
        self
    }

    /// Set the maximum hierarchy depth.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }
}

impl Default for MarkdownConfig {
    fn default() -> Self {
        Self {
            base: NodeParserConfig::default(),
            header_path_separator: "/".to_string(),
            include_code_blocks: true,
            preserve_header_hierarchy: true,
            max_header_depth: 6,
            include_header_in_content: true,
            min_section_length: 0,
        }
    }
}

impl MarkdownConfig {
    /// Create a new markdown configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the header path separator.
    pub fn with_header_path_separator<S: Into<String>>(mut self, separator: S) -> Self {
        self.header_path_separator = separator.into();
        self
    }

    /// Set whether to include code blocks.
    pub fn with_include_code_blocks(mut self, include: bool) -> Self {
        self.include_code_blocks = include;
        self
    }

    /// Set whether to preserve header hierarchy.
    pub fn with_preserve_header_hierarchy(mut self, preserve: bool) -> Self {
        self.preserve_header_hierarchy = preserve;
        self
    }

    /// Set the maximum header depth.
    pub fn with_max_header_depth(mut self, depth: usize) -> Self {
        self.max_header_depth = depth.min(6).max(1);
        self
    }

    /// Set whether to include header in content.
    pub fn with_include_header_in_content(mut self, include: bool) -> Self {
        self.include_header_in_content = include;
        self
    }

    /// Set the minimum section length.
    pub fn with_min_section_length(mut self, length: usize) -> Self {
        self.min_section_length = length;
        self
    }

    /// Create a configuration optimized for documentation parsing.
    pub fn for_documentation() -> Self {
        Self::new()
            .with_header_path_separator("/")
            .with_preserve_header_hierarchy(true)
            .with_include_header_in_content(true)
            .with_max_header_depth(4)
            .with_min_section_length(50)
    }

    /// Create a configuration optimized for blog posts.
    pub fn for_blog_posts() -> Self {
        Self::new()
            .with_header_path_separator(" > ")
            .with_preserve_header_hierarchy(true)
            .with_include_header_in_content(true)
            .with_max_header_depth(3)
            .with_min_section_length(100)
    }

    /// Create a configuration optimized for README files.
    pub fn for_readme() -> Self {
        Self::new()
            .with_header_path_separator("/")
            .with_preserve_header_hierarchy(true)
            .with_include_header_in_content(true)
            .with_max_header_depth(6)
            .with_min_section_length(20)
    }
}
