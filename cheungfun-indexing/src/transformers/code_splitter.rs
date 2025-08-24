//! Code-aware text splitter that preserves syntactic structure.
//!
//! This module provides a specialized text splitter for source code that attempts
//! to maintain syntactic boundaries when splitting code into chunks.

use crate::error::Result;
use crate::loaders::ProgrammingLanguage;
use cheungfun_core::{ChunkInfo, Document, Node, Result as CoreResult, traits::NodeTransformer};
use tracing::debug;

/// Configuration for code-aware splitting.
#[derive(Debug, Clone)]
pub struct CodeSplitterConfig {
    /// Maximum number of lines per chunk.
    pub max_chunk_lines: usize,

    /// Number of lines to overlap between chunks.
    pub chunk_overlap_lines: usize,

    /// Maximum number of characters per chunk.
    pub max_chunk_chars: usize,

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
            max_chunk_lines: 40,
            chunk_overlap_lines: 15,
            max_chunk_chars: 1500,
            respect_function_boundaries: true,
            respect_class_boundaries: true,
            preserve_indentation: true,
            min_chunk_lines: 5,
        }
    }
}

/// Code-aware text splitter that preserves syntactic structure.
#[derive(Debug, Clone)]
pub struct CodeSplitter {
    config: CodeSplitterConfig,
}

impl CodeSplitter {
    /// Create a new code splitter with default configuration.
    pub fn new() -> Self {
        Self {
            config: CodeSplitterConfig::default(),
        }
    }

    /// Create a code splitter with custom configuration.
    pub fn with_config(config: CodeSplitterConfig) -> Self {
        Self { config }
    }

    /// Split a document into code-aware chunks.
    pub async fn split_document(&self, document: Document) -> Result<Vec<Node>> {
        debug!("Splitting document with code splitter");

        // Detect programming language from metadata
        let language = self.detect_language_from_document(&document);

        // Split the content based on the language
        let chunks = self.split_code_content(&document.content, &language)?;

        // Convert chunks to nodes
        let mut nodes = Vec::new();
        for (i, chunk) in chunks.into_iter().enumerate() {
            let chunk_info = ChunkInfo::new(0, chunk.len(), i);
            let mut node = Node::new(chunk, document.id, chunk_info);

            // Copy metadata from document
            for (key, value) in &document.metadata {
                node.metadata.insert(key.clone(), value.clone());
            }

            // Add chunk-specific metadata
            node.metadata.insert(
                "chunk_index".to_string(),
                serde_json::Value::Number(i.into()),
            );

            node.metadata.insert(
                "language".to_string(),
                serde_json::Value::String(language.as_str().to_string()),
            );

            node.metadata.insert(
                "splitter_type".to_string(),
                serde_json::Value::String("code_splitter".to_string()),
            );

            nodes.push(node);
        }

        debug!("Split document into {} code chunks", nodes.len());
        Ok(nodes)
    }

    /// Detect programming language from document metadata.
    fn detect_language_from_document(&self, document: &Document) -> ProgrammingLanguage {
        // Try to get language from metadata
        if let Some(lang_value) = document.metadata.get("language") {
            if let Some(lang_str) = lang_value.as_str() {
                return self.parse_language_string(lang_str);
            }
        }

        // Try to detect from content type
        if let Some(content_type) = document.metadata.get("content_type") {
            if let Some(ct_str) = content_type.as_str() {
                return self.language_from_content_type(ct_str);
            }
        }

        // Try to detect from filename
        if let Some(filename) = document.metadata.get("filename") {
            if let Some(name_str) = filename.as_str() {
                if let Some(extension) = name_str.split('.').last() {
                    return ProgrammingLanguage::from_extension(extension);
                }
            }
        }

        ProgrammingLanguage::Unknown
    }

    /// Parse language string to ProgrammingLanguage enum.
    fn parse_language_string(&self, lang_str: &str) -> ProgrammingLanguage {
        match lang_str.to_lowercase().as_str() {
            "rust" => ProgrammingLanguage::Rust,
            "python" => ProgrammingLanguage::Python,
            "javascript" => ProgrammingLanguage::JavaScript,
            "typescript" => ProgrammingLanguage::TypeScript,
            "java" => ProgrammingLanguage::Java,
            "csharp" | "c#" => ProgrammingLanguage::CSharp,
            "cpp" | "c++" => ProgrammingLanguage::Cpp,
            "c" => ProgrammingLanguage::C,
            "go" => ProgrammingLanguage::Go,
            "ruby" => ProgrammingLanguage::Ruby,
            "php" => ProgrammingLanguage::PHP,
            "swift" => ProgrammingLanguage::Swift,
            "kotlin" => ProgrammingLanguage::Kotlin,
            "scala" => ProgrammingLanguage::Scala,
            "haskell" => ProgrammingLanguage::Haskell,
            "clojure" => ProgrammingLanguage::Clojure,
            "erlang" => ProgrammingLanguage::Erlang,
            "elixir" => ProgrammingLanguage::Elixir,
            "lua" => ProgrammingLanguage::Lua,
            "shell" | "bash" | "sh" => ProgrammingLanguage::Shell,
            "sql" => ProgrammingLanguage::SQL,
            "html" => ProgrammingLanguage::HTML,
            "css" => ProgrammingLanguage::CSS,
            "xml" => ProgrammingLanguage::XML,
            "json" => ProgrammingLanguage::JSON,
            "yaml" => ProgrammingLanguage::YAML,
            "toml" => ProgrammingLanguage::TOML,
            "markdown" => ProgrammingLanguage::Markdown,
            _ => ProgrammingLanguage::Unknown,
        }
    }

    /// Get programming language from content type.
    fn language_from_content_type(&self, content_type: &str) -> ProgrammingLanguage {
        match content_type {
            "text/x-rust" => ProgrammingLanguage::Rust,
            "text/x-python" => ProgrammingLanguage::Python,
            "text/javascript" => ProgrammingLanguage::JavaScript,
            "text/typescript" => ProgrammingLanguage::TypeScript,
            "text/x-java" => ProgrammingLanguage::Java,
            "text/x-csharp" => ProgrammingLanguage::CSharp,
            "text/x-c++" => ProgrammingLanguage::Cpp,
            "text/x-c" => ProgrammingLanguage::C,
            "text/x-go" => ProgrammingLanguage::Go,
            "text/x-ruby" => ProgrammingLanguage::Ruby,
            "text/x-php" => ProgrammingLanguage::PHP,
            "text/x-swift" => ProgrammingLanguage::Swift,
            "text/x-kotlin" => ProgrammingLanguage::Kotlin,
            "text/x-scala" => ProgrammingLanguage::Scala,
            "text/x-haskell" => ProgrammingLanguage::Haskell,
            "text/x-clojure" => ProgrammingLanguage::Clojure,
            "text/x-erlang" => ProgrammingLanguage::Erlang,
            "text/x-elixir" => ProgrammingLanguage::Elixir,
            "text/x-lua" => ProgrammingLanguage::Lua,
            "text/x-shellscript" => ProgrammingLanguage::Shell,
            "text/x-sql" => ProgrammingLanguage::SQL,
            "text/html" => ProgrammingLanguage::HTML,
            "text/css" => ProgrammingLanguage::CSS,
            "text/xml" | "application/xml" => ProgrammingLanguage::XML,
            "application/json" => ProgrammingLanguage::JSON,
            "text/x-yaml" => ProgrammingLanguage::YAML,
            "text/x-toml" => ProgrammingLanguage::TOML,
            "text/markdown" => ProgrammingLanguage::Markdown,
            _ => ProgrammingLanguage::Unknown,
        }
    }

    /// Split code content into chunks while preserving structure.
    fn split_code_content(
        &self,
        content: &str,
        language: &ProgrammingLanguage,
    ) -> Result<Vec<String>> {
        let lines: Vec<&str> = content.lines().collect();

        if lines.is_empty() {
            return Ok(vec![]);
        }

        // For now, use a simple line-based splitting approach
        // TODO: Implement AST-based splitting for better structure preservation
        let chunks = self.split_by_lines(&lines, language)?;

        Ok(chunks)
    }

    /// Split content by lines with language-aware boundaries.
    fn split_by_lines(
        &self,
        lines: &[&str],
        language: &ProgrammingLanguage,
    ) -> Result<Vec<String>> {
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_chars = 0;

        for (_i, line) in lines.iter().enumerate() {
            let line_chars = line.len();

            // Check if adding this line would exceed limits
            if current_chunk.len() >= self.config.max_chunk_lines
                || current_chars + line_chars > self.config.max_chunk_chars
            {
                // Try to find a good break point
                if let Some(break_point) = self.find_break_point(&current_chunk, language) {
                    // Split at the break point
                    let chunk_content = current_chunk[..break_point].join("\n");
                    if !chunk_content.trim().is_empty() {
                        chunks.push(chunk_content);
                    }

                    // Start new chunk with overlap
                    let overlap_start = if break_point > self.config.chunk_overlap_lines {
                        break_point - self.config.chunk_overlap_lines
                    } else {
                        0
                    };

                    current_chunk = current_chunk[overlap_start..].to_vec();
                    current_chars = current_chunk.iter().map(|l| l.len()).sum();
                } else {
                    // No good break point found, force split
                    let chunk_content = current_chunk.join("\n");
                    if !chunk_content.trim().is_empty() {
                        chunks.push(chunk_content);
                    }

                    current_chunk.clear();
                    current_chars = 0;
                }
            }

            current_chunk.push(line);
            current_chars += line_chars;
        }

        // Add the last chunk if it has content
        if !current_chunk.is_empty() {
            let chunk_content = current_chunk.join("\n");
            if !chunk_content.trim().is_empty() {
                chunks.push(chunk_content);
            }
        }

        Ok(chunks)
    }

    /// Find a good break point in the current chunk based on language syntax.
    fn find_break_point(&self, lines: &[&str], language: &ProgrammingLanguage) -> Option<usize> {
        if lines.len() < self.config.min_chunk_lines {
            return None;
        }

        // Look for natural break points from the end backwards
        for i in (self.config.min_chunk_lines..lines.len()).rev() {
            let line = lines[i].trim();

            if self.is_good_break_point(line, language) {
                return Some(i + 1); // Include the break line in the chunk
            }
        }

        None
    }

    /// Check if a line is a good break point for the given language.
    fn is_good_break_point(&self, line: &str, language: &ProgrammingLanguage) -> bool {
        if line.is_empty() {
            return true; // Empty lines are always good break points
        }

        match language {
            ProgrammingLanguage::Rust
            | ProgrammingLanguage::JavaScript
            | ProgrammingLanguage::TypeScript
            | ProgrammingLanguage::Java
            | ProgrammingLanguage::CSharp
            | ProgrammingLanguage::Cpp
            | ProgrammingLanguage::C
            | ProgrammingLanguage::Go
            | ProgrammingLanguage::PHP
            | ProgrammingLanguage::Swift
            | ProgrammingLanguage::Kotlin
            | ProgrammingLanguage::Scala => {
                // Look for closing braces or semicolons
                line.ends_with('}') || line.ends_with(';')
            }
            ProgrammingLanguage::Python | ProgrammingLanguage::Ruby => {
                // Look for function/class definitions or comments
                line.starts_with("def ")
                    || line.starts_with("class ")
                    || line.starts_with('#')
                    || !line.starts_with(' ') && !line.starts_with('\t') // Top-level statements
            }
            _ => {
                // Generic break points
                line.starts_with("//") || line.starts_with('#') || line.starts_with("/*")
            }
        }
    }
}

impl Default for CodeSplitter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl NodeTransformer for CodeSplitter {
    async fn transform_node(&self, node: Node) -> CoreResult<Node> {
        // For single node transformation, we don't split
        // This is mainly used for batch processing
        Ok(node)
    }
}
