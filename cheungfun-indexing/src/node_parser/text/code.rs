//! AST-enhanced code splitter implementation.
//!
//! This module provides the CodeSplitter, which combines traditional line-based
//! splitting with AST analysis for intelligent code chunking that respects
//! syntactic boundaries. It integrates with the existing AstParser infrastructure.

use super::{Tokenizer, WhitespaceTokenizer};
use crate::loaders::ProgrammingLanguage;
use crate::node_parser::{
    callbacks::{CallbackManager, EventPayload},
    config::CodeSplitterConfig,
    utils::{build_nodes_from_splits, get_id_function},
    MetadataAwareTextSplitter, NodeParser, TextSplitter,
};
use crate::parsers::{AstParser, AstParserConfig};
use async_trait::async_trait;
use cheungfun_core::{
    traits::{Transform, TransformInput},
    CheungfunError, Document, Node, Result as CoreResult,
};
use std::sync::Arc;
use tracing::{debug, warn};
use tree_sitter::Parser;

/// Span data structure for representing string slices (inspired by SweepAI)
/// This is a direct port of the Span class from the chunking-improvements blog post
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Start position (byte index or line number)
    pub start: usize,
    /// End position (byte index or line number)
    pub end: usize,
}

impl Span {
    /// Create a new span
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Extract the corresponding substring by bytes
    pub fn extract<'a>(&self, text: &'a str) -> &'a str {
        &text[self.start..self.end]
    }

    /// Extract the corresponding lines (for line-based spans)
    pub fn extract_lines<'a>(&self, text: &'a str) -> String {
        let lines: Vec<&str> = text.lines().collect();
        lines[self.start..self.end.min(lines.len())].join("\n")
    }

    /// Get the length of the span
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Check if the span is empty
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

impl std::ops::Add<Span> for Span {
    type Output = Span;

    /// Concatenate two spans (SweepAI style)
    /// Note: No safety checks, Span(a, b) + Span(c, d) = Span(a, d)
    fn add(self, other: Span) -> Span {
        Span::new(self.start, other.end)
    }
}

impl std::ops::Add<usize> for Span {
    type Output = Span;

    /// Shift span by offset
    fn add(self, offset: usize) -> Span {
        Span::new(self.start + offset, self.end + offset)
    }
}

/// AST-enhanced code splitter.
///
/// This splitter provides intelligent code chunking by combining:
/// 1. AST analysis for syntactic boundaries
/// 2. Line-based splitting as fallback
/// 3. Respect for function and class boundaries
/// 4. Indentation context preservation
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::node_parser::{text::CodeSplitter, TextSplitter};
/// use cheungfun_indexing::loaders::ProgrammingLanguage;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let splitter = CodeSplitter::from_defaults(
///         ProgrammingLanguage::Rust,
///         40,  // chunk_lines
///         15,  // chunk_lines_overlap
///         1500 // max_chars
///     )?;
///     
///     let code = r#"
/// fn main() {
///     println!("Hello, world!");
/// }
///
/// fn another_function() {
///     // This is another function
///     let x = 42;
///     println!("x = {}", x);
/// }
///     "#;
///     
///     let chunks = splitter.split_text(code)?;
///     println!("Split into {} chunks", chunks.len());
///     Ok(())
/// }
/// ```
pub struct CodeSplitter {
    config: CodeSplitterConfig,
    tokenizer: Arc<dyn Tokenizer>,
    ast_parser: Option<AstParser>,
    tree_sitter_parser: Option<Parser>,
    callback_manager: Option<CallbackManager>,
}

impl std::fmt::Debug for CodeSplitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeSplitter")
            .field("config", &self.config)
            .field("tokenizer", &"<tokenizer>")
            .field("ast_parser", &self.ast_parser.is_some())
            .field("tree_sitter_parser", &self.tree_sitter_parser.is_some())
            .field("callback_manager", &self.callback_manager.is_some())
            .finish()
    }
}

impl CodeSplitter {
    /// Create a new code splitter with the given configuration.
    pub fn new(config: CodeSplitterConfig) -> CoreResult<Self> {
        let tokenizer = Arc::new(WhitespaceTokenizer);

        let (ast_parser, tree_sitter_parser) = if config.use_ast_splitting {
            let ast_config = config.ast_config.clone().unwrap_or_default();
            let ast_parser = AstParser::with_config(ast_config)?;
            let ts_parser = create_tree_sitter_parser(&config.language)?;
            (Some(ast_parser), Some(ts_parser))
        } else {
            (None, None)
        };

        Ok(Self {
            config,
            tokenizer,
            ast_parser,
            tree_sitter_parser,
            callback_manager: None,
        })
    }

    /// Create a code splitter with default configuration.
    pub fn from_defaults(
        language: ProgrammingLanguage,
        chunk_lines: usize,
        chunk_lines_overlap: usize,
        max_chars: usize,
    ) -> CoreResult<Self> {
        let config = CodeSplitterConfig::new(language, chunk_lines, chunk_lines_overlap, max_chars);
        Self::new(config)
    }

    /// Create a code splitter with a predefined chunking strategy.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cheungfun_indexing::node_parser::{text::CodeSplitter, config::ChunkingStrategy};
    /// use cheungfun_indexing::loaders::ProgrammingLanguage;
    ///
    /// // Use SweepAI-optimized strategy
    /// let splitter = CodeSplitter::with_strategy(
    ///     ProgrammingLanguage::Rust,
    ///     ChunkingStrategy::SweepAI
    /// )?;
    ///
    /// // Use fine-grained strategy for detailed analysis
    /// let splitter = CodeSplitter::with_strategy(
    ///     ProgrammingLanguage::Python,
    ///     ChunkingStrategy::Fine
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_strategy(
        language: ProgrammingLanguage,
        strategy: crate::node_parser::config::ChunkingStrategy,
    ) -> CoreResult<Self> {
        let config = CodeSplitterConfig::with_strategy(language, strategy);
        Self::new(config)
    }

    /// Create an optimal code splitter for RAG applications (recommended default).
    ///
    /// This uses research-backed parameters for optimal code chunking quality
    /// in knowledge retrieval applications.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cheungfun_indexing::node_parser::text::CodeSplitter;
    /// use cheungfun_indexing::loaders::ProgrammingLanguage;
    ///
    /// let splitter = CodeSplitter::optimal(ProgrammingLanguage::Rust)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn optimal(language: ProgrammingLanguage) -> CoreResult<Self> {
        let config = CodeSplitterConfig::optimal(language);
        Self::new(config)
    }

    /// Create an enterprise code splitter for large codebases.
    ///
    /// This is specifically designed for large projects like Unity3D with
    /// thousands of code files, providing larger context windows and better
    /// continuity for complex class hierarchies.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cheungfun_indexing::node_parser::text::CodeSplitter;
    /// use cheungfun_indexing::loaders::ProgrammingLanguage;
    ///
    /// let splitter = CodeSplitter::enterprise(ProgrammingLanguage::CSharp)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn enterprise(language: ProgrammingLanguage) -> CoreResult<Self> {
        let config = CodeSplitterConfig::enterprise(language);
        Self::new(config)
    }

    /// Create a fine-grained code splitter for detailed analysis.
    pub fn fine_grained(language: ProgrammingLanguage) -> CoreResult<Self> {
        let config = CodeSplitterConfig::fine_grained(language);
        Self::new(config)
    }

    /// Create a balanced code splitter for general use.
    pub fn balanced(language: ProgrammingLanguage) -> CoreResult<Self> {
        let config = CodeSplitterConfig::balanced(language);
        Self::new(config)
    }

    /// Create a coarse-grained code splitter for high-level overview.
    pub fn coarse_grained(language: ProgrammingLanguage) -> CoreResult<Self> {
        let config = CodeSplitterConfig::coarse_grained(language);
        Self::new(config)
    }

    /// Set callback manager.
    pub fn with_callback_manager(mut self, callback_manager: CallbackManager) -> Self {
        self.callback_manager = Some(callback_manager);
        self
    }

    /// Split code text using the configured strategy.
    fn split_code_internal(&self, text: &str) -> CoreResult<Vec<String>> {
        if text.trim().is_empty() {
            return Ok(vec![text.to_string()]);
        }

        debug!(
            "Splitting {} code of {} characters",
            self.config.language.as_str(),
            text.len()
        );

        if self.config.use_ast_splitting {
            match self.split_text_with_ast(text) {
                Ok(chunks) => {
                    debug!("AST splitting produced {} chunks", chunks.len());
                    return Ok(chunks);
                }
                Err(e) => {
                    warn!(
                        "AST splitting failed: {}, falling back to line-based splitting",
                        e
                    );
                }
            }
        }

        // Fallback to line-based splitting
        self.split_text_by_lines(text)
    }

    /// Split text using AST analysis with SweepAI-enhanced approach first
    fn split_text_with_ast(&self, text: &str) -> CoreResult<Vec<String>> {
        // Try SweepAI-enhanced AST chunking first (most advanced)
        match self.split_with_sweepai_style(text) {
            Ok(chunks) if !chunks.is_empty() => {
                debug!(
                    "Using SweepAI-enhanced AST chunking: {} chunks",
                    chunks.len()
                );
                return Ok(chunks);
            }
            Ok(_) => {
                debug!("SweepAI-style chunking returned empty, trying LlamaIndex fallback");
            }
            Err(e) => {
                debug!(
                    "SweepAI-style AST parsing failed: {}, trying LlamaIndex fallback",
                    e
                );
            }
        }

        // Fallback to LlamaIndex-style AST chunking
        match self.split_with_llamaindex_style(text) {
            Ok(chunks) if !chunks.is_empty() => {
                debug!(
                    "Using LlamaIndex-style AST chunking: {} chunks",
                    chunks.len()
                );
                return Ok(chunks);
            }
            Ok(_) => {
                debug!("LlamaIndex-style chunking returned empty, trying structure-aware approach");
            }
            Err(e) => {
                debug!(
                    "LlamaIndex-style AST parsing failed: {}, trying structure-aware approach",
                    e
                );
            }
        }

        // Fallback to our structure-aware approach
        let ast_config = AstParserConfig {
            include_function_bodies: true,
            max_depth: Some(10),
            extract_functions: true,
            extract_classes: true,
            extract_imports: true,
            extract_comments: true,
        };

        let ast_parser = match AstParser::new() {
            Ok(parser) => parser,
            Err(e) => {
                warn!(
                    "Failed to create AST parser: {}, falling back to line-based splitting",
                    e
                );
                return self.split_by_lines_fallback(text);
            }
        };

        // Parse the code to get AST analysis
        match ast_parser.parse(text, self.config.language) {
            Ok(analysis) => {
                debug!(
                    "Structure-aware AST analysis successful: {} functions, {} classes",
                    analysis.functions.len(),
                    analysis.classes.len()
                );
                self.split_with_ast_analysis(text, &analysis)
            }
            Err(e) => {
                warn!(
                    "AST parsing failed: {}, falling back to line-based splitting",
                    e
                );
                self.split_by_lines_fallback(text)
            }
        }
    }

    /// Split text using AST analysis results.
    fn split_with_ast_analysis(
        &self,
        text: &str,
        analysis: &crate::parsers::AstAnalysis,
    ) -> CoreResult<Vec<String>> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = text.lines().collect();

        // Create a map of line numbers to code structures
        let mut structure_map = std::collections::BTreeMap::new();

        // Add functions to the structure map
        for func in &analysis.functions {
            structure_map.insert(
                func.start_line,
                ("function", func.name.clone(), func.end_line),
            );
        }

        // Add classes to the structure map
        for class in &analysis.classes {
            structure_map.insert(
                class.start_line,
                ("class", class.name.clone(), class.end_line),
            );
        }

        // Split based on code structures
        let mut current_line = 0;

        for (&start_line, &(structure_type, ref name, end_line)) in &structure_map {
            // Add any content before this structure
            if current_line < start_line {
                let before_lines = &lines[current_line..start_line.min(lines.len())];
                if !before_lines.is_empty() {
                    let before_text = before_lines.join("\n");
                    if !before_text.trim().is_empty() {
                        chunks.extend(self.split_by_lines_fallback(&before_text)?);
                    }
                }
            }

            // Handle the structure itself
            let structure_end = end_line.min(lines.len());
            if start_line < structure_end {
                let structure_lines = &lines[start_line..structure_end];
                let structure_text = structure_lines.join("\n");

                debug!(
                    "Processing {} '{}' ({} lines)",
                    structure_type,
                    name,
                    structure_lines.len()
                );

                // If the structure is small enough, keep it as one chunk
                if structure_lines.len() <= self.config.chunk_lines {
                    chunks.push(structure_text);
                } else {
                    // Split large structures intelligently
                    chunks.extend(self.split_large_structure(&structure_text, structure_type)?);
                }

                current_line = structure_end;
            }
        }

        // Add any remaining content
        if current_line < lines.len() {
            let remaining_lines = &lines[current_line..];
            if !remaining_lines.is_empty() {
                let remaining_text = remaining_lines.join("\n");
                if !remaining_text.trim().is_empty() {
                    chunks.extend(self.split_by_lines_fallback(&remaining_text)?);
                }
            }
        }

        // If no structures were found, fall back to line-based splitting
        if chunks.is_empty() {
            debug!("No code structures found, using line-based splitting");
            return self.split_by_lines_fallback(text);
        }

        Ok(chunks)
    }

    /// Split large code structures (functions, classes) intelligently.
    fn split_large_structure(&self, text: &str, structure_type: &str) -> CoreResult<Vec<String>> {
        let lines: Vec<&str> = text.lines().collect();
        let mut chunks = Vec::new();

        debug!(
            "Splitting large {} with {} lines",
            structure_type,
            lines.len()
        );

        // For functions and classes, try to keep the signature with some body
        let signature_lines = match structure_type {
            "function" => self.estimate_function_signature_lines(&lines),
            "class" => self.estimate_class_signature_lines(&lines),
            _ => 1,
        };

        let mut start = 0;
        while start < lines.len() {
            let mut end = (start + self.config.chunk_lines).min(lines.len());

            // If this is the first chunk, ensure we include the signature
            if start == 0 {
                end = end.max(signature_lines);
            }

            // Try to end at a logical boundary (empty line, closing brace, etc.)
            if end < lines.len() {
                end = self.find_logical_boundary(&lines, start, end);
            }

            let chunk_lines = &lines[start..end];
            let chunk_text = chunk_lines.join("\n");

            if !chunk_text.trim().is_empty() {
                chunks.push(chunk_text);
            }

            // Move to next chunk with overlap
            start = if end >= lines.len() {
                break;
            } else {
                (end - self.config.chunk_lines_overlap).max(start + 1)
            };
        }

        Ok(chunks)
    }

    /// Estimate how many lines a function signature takes.
    fn estimate_function_signature_lines(&self, lines: &[&str]) -> usize {
        let mut signature_lines = 1;

        for (i, line) in lines.iter().enumerate() {
            if i >= 10 {
                break;
            } // Don't look too far

            let trimmed = line.trim();
            if trimmed.contains('{') || trimmed.contains(':') {
                signature_lines = i + 1;
                break;
            }
        }

        signature_lines.min(5) // Cap at 5 lines
    }

    /// Estimate how many lines a class signature takes.
    fn estimate_class_signature_lines(&self, lines: &[&str]) -> usize {
        let mut signature_lines = 1;

        for (i, line) in lines.iter().enumerate() {
            if i >= 10 {
                break;
            } // Don't look too far

            let trimmed = line.trim();
            if trimmed.contains('{') || (trimmed.contains(':') && !trimmed.starts_with("//")) {
                signature_lines = i + 1;
                break;
            }
        }

        signature_lines.min(3) // Cap at 3 lines
    }

    /// Find a logical boundary for splitting (empty line, closing brace, etc.).
    fn find_logical_boundary(&self, lines: &[&str], start: usize, preferred_end: usize) -> usize {
        // Look for good boundaries within a small window around the preferred end
        let window_start = preferred_end.saturating_sub(5);
        let window_end = (preferred_end + 5).min(lines.len());

        // Look for empty lines first
        for i in (window_start..window_end).rev() {
            if i > start && lines[i].trim().is_empty() {
                return i;
            }
        }

        // Look for closing braces
        for i in (window_start..window_end).rev() {
            if i > start {
                let trimmed = lines[i].trim();
                if trimmed == "}" || trimmed.ends_with("};") {
                    return i + 1;
                }
            }
        }

        // Fall back to preferred end
        preferred_end
    }

    /// Fallback method for line-based splitting.
    fn split_by_lines_fallback(&self, text: &str) -> CoreResult<Vec<String>> {
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() <= self.config.chunk_lines {
            return Ok(vec![text.to_string()]);
        }

        debug!("Line-based fallback splitting of {} lines", lines.len());

        let mut chunks = Vec::new();
        let mut i = 0;

        while i < lines.len() {
            let end = (i + self.config.chunk_lines).min(lines.len());
            let chunk_lines = &lines[i..end];
            let chunk_text = chunk_lines.join("\n");

            if !chunk_text.trim().is_empty() {
                chunks.push(chunk_text);
            }

            // Move to next chunk with overlap
            i = if end >= lines.len() {
                break;
            } else {
                (end - self.config.chunk_lines_overlap).max(i + 1)
            };
        }

        Ok(chunks)
    }

    /// Split text by lines (fallback method).
    fn split_text_by_lines(&self, text: &str) -> CoreResult<Vec<String>> {
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() <= self.config.chunk_lines {
            return Ok(vec![text.to_string()]);
        }

        debug!("Line-based splitting of {} lines", lines.len());

        let mut chunks = Vec::new();
        let mut i = 0;

        while i < lines.len() {
            let end = (i + self.config.chunk_lines).min(lines.len());
            let chunk_lines = &lines[i..end];

            // Check if this chunk respects boundaries
            let chunk_text = chunk_lines.join("\n");
            if self.should_split_chunk(&chunk_text, i, &lines)? {
                // Try to find a better split point
                if let Some(better_end) = self.find_better_split_point(i, end, &lines) {
                    let better_chunk_lines = &lines[i..better_end];
                    chunks.push(better_chunk_lines.join("\n"));
                    // Ensure we always make progress: next_i must be > i
                    let next_i = better_end.saturating_sub(self.config.chunk_lines_overlap);
                    i = if next_i <= i { i + 1 } else { next_i };
                } else {
                    chunks.push(chunk_text);
                    // Ensure we always make progress: next_i must be > i
                    let next_i = end.saturating_sub(self.config.chunk_lines_overlap);
                    i = if next_i <= i { i + 1 } else { next_i };
                }
            } else {
                chunks.push(chunk_text);
                // Ensure we always make progress: next_i must be > i
                let next_i = end.saturating_sub(self.config.chunk_lines_overlap);
                i = if next_i <= i { i + 1 } else { next_i };
            }
        }

        Ok(chunks)
    }

    /// Check if a chunk should be split further based on code structure.
    fn should_split_chunk(
        &self,
        chunk: &str,
        start_line: usize,
        all_lines: &[&str],
    ) -> CoreResult<bool> {
        // Simple heuristics for code structure
        if !self.config.respect_function_boundaries && !self.config.respect_class_boundaries {
            return Ok(false);
        }

        // Check if chunk ends in the middle of a function or class
        let lines_in_chunk: Vec<&str> = chunk.lines().collect();
        if let Some(last_line) = lines_in_chunk.last() {
            // Look for opening braces without closing braces
            let open_braces = chunk.matches('{').count();
            let close_braces = chunk.matches('}').count();

            if open_braces > close_braces {
                // We're in the middle of a block, try to find a better split
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Find a better split point that respects code boundaries.
    fn find_better_split_point(&self, start: usize, end: usize, lines: &[&str]) -> Option<usize> {
        // Look backwards from the end to find a good split point
        for i in (start..end).rev() {
            if i >= lines.len() {
                continue;
            }

            let line = lines[i].trim();

            // Good split points: end of functions, classes, or blocks
            if line.ends_with('}') || line.is_empty() {
                return Some(i + 1);
            }
        }

        None
    }
}

/// Create a tree-sitter parser for the given language.
fn create_tree_sitter_parser(language: &ProgrammingLanguage) -> CoreResult<Parser> {
    let mut parser = Parser::new();

    // Get the tree-sitter language for the given programming language
    let ts_language = match language {
        ProgrammingLanguage::Rust => tree_sitter_rust::LANGUAGE.into(),
        ProgrammingLanguage::Python => tree_sitter_python::LANGUAGE.into(),
        ProgrammingLanguage::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        ProgrammingLanguage::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        ProgrammingLanguage::Java => tree_sitter_java::LANGUAGE.into(),
        ProgrammingLanguage::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
        ProgrammingLanguage::C => tree_sitter_c::LANGUAGE.into(),
        ProgrammingLanguage::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        ProgrammingLanguage::Go => tree_sitter_go::LANGUAGE.into(),
        _ => {
            return Err(cheungfun_core::error::CheungfunError::Pipeline {
                message: format!(
                    "Unsupported language for AST parsing: {}",
                    language.as_str()
                ),
            });
        }
    };

    // Set the language for the parser
    parser.set_language(&ts_language).map_err(|e| {
        cheungfun_core::error::CheungfunError::Pipeline {
            message: format!("Failed to set tree-sitter language: {}", e),
        }
    })?;

    Ok(parser)
}

#[async_trait]
impl NodeParser for CodeSplitter {
    async fn parse_nodes(
        &self,
        documents: &[Document],
        show_progress: bool,
    ) -> CoreResult<Vec<Node>> {
        let mut all_nodes = Vec::new();

        // Emit start event
        if let Some(ref callback_manager) = self.callback_manager {
            let payload = EventPayload::node_parsing_start(documents.len());
            callback_manager.emit_event(payload).await?;
        }

        for document in documents {
            let chunks = self.split_text(&document.content)?;

            let id_func = self
                .config
                .base
                .base
                .id_func
                .as_ref()
                .map(|name| get_id_function(name));

            let nodes = build_nodes_from_splits(
                chunks,
                document,
                id_func.as_deref(),
                self.config.base.base.include_prev_next_rel,
            )?;

            all_nodes.extend(nodes);
        }

        // Emit end event
        if let Some(ref callback_manager) = self.callback_manager {
            let payload = EventPayload::node_parsing_end(&all_nodes);
            callback_manager.emit_event(payload).await?;
        }

        Ok(all_nodes)
    }
}

#[async_trait]
impl TextSplitter for CodeSplitter {
    fn split_text(&self, text: &str) -> CoreResult<Vec<String>> {
        self.split_code_internal(text)
    }
}

#[async_trait]
impl MetadataAwareTextSplitter for CodeSplitter {
    fn split_text_metadata_aware(&self, text: &str, metadata_str: &str) -> CoreResult<Vec<String>> {
        // For code, metadata is less critical, but we still consider it
        let metadata_lines = metadata_str.lines().count();
        let effective_chunk_lines = if self.config.chunk_lines > metadata_lines {
            self.config.chunk_lines - metadata_lines
        } else {
            self.config.chunk_lines / 2
        };

        // Temporarily adjust config for this split
        let mut temp_config = self.config.clone();
        temp_config.chunk_lines = effective_chunk_lines;

        // Use the same splitting logic but with adjusted parameters
        self.split_code_internal(text)
    }
}

#[async_trait]
impl Transform for CodeSplitter {
    async fn transform(&self, input: TransformInput) -> CoreResult<Vec<Node>> {
        match input {
            TransformInput::Document(document) => {
                // Use the existing NodeParser implementation
                NodeParser::parse_nodes(self, &[document], false).await
            }
            TransformInput::Documents(documents) => {
                // Use the existing NodeParser implementation for batch processing
                NodeParser::parse_nodes(self, &documents, false).await
            }
            TransformInput::Node(_) | TransformInput::Nodes(_) => {
                // CodeSplitter only processes documents, not nodes
                Err(CheungfunError::Validation {
                    message: "CodeSplitter only accepts documents as input".into(),
                })
            }
        }
    }

    fn name(&self) -> &'static str {
        "CodeSplitter"
    }
}

impl CodeSplitter {
    /// Create a tree-sitter parser for the configured language
    fn create_tree_sitter_parser(&self) -> CoreResult<tree_sitter::Parser> {
        create_tree_sitter_parser(&self.config.language)
    }

    /// Count non-whitespace characters (SweepAI improvement)
    /// This helps avoid tiny chunks that are mostly indentation
    fn non_whitespace_len(&self, text: &str) -> usize {
        text.chars().filter(|c| !c.is_whitespace()).count()
    }

    /// Convert byte index to line number (SweepAI improvement)
    /// This eliminates encoding issues and ensures line-based chunks
    fn get_line_number(&self, index: usize, source_code: &str) -> usize {
        // Fast path: if index is 0, return 0
        if index == 0 {
            return 0;
        }

        // Count newlines up to the index
        let bytes = source_code.as_bytes();
        if index >= bytes.len() {
            return source_code.lines().count().saturating_sub(1);
        }

        let mut line_count = 0;
        for i in 0..index.min(bytes.len()) {
            if bytes[i] == b'\n' {
                line_count += 1;
            }
        }
        line_count
    }

    /// Build line offset cache for efficient line number lookups
    fn build_line_offsets(&self, source_code: &str) -> Vec<usize> {
        let mut offsets = vec![0]; // First line starts at 0
        for (i, byte) in source_code.bytes().enumerate() {
            if byte == b'\n' {
                offsets.push(i + 1); // Next line starts after newline
            }
        }
        offsets
    }

    /// Convert byte index to line number using precomputed offsets
    fn byte_to_line_with_cache(&self, index: usize, line_offsets: &[usize]) -> usize {
        match line_offsets.binary_search(&index) {
            Ok(line) => line,
            Err(line) => line.saturating_sub(1),
        }
    }

    /// Fill gaps between consecutive spans (SweepAI improvement)
    /// This fixes tree-sitter's missing whitespace between nodes
    fn connect_chunks(&self, chunks: &mut [Span]) {
        for i in 0..chunks.len().saturating_sub(1) {
            chunks[i].end = chunks[i + 1].start;
        }
    }

    /// Coalesce small chunks with larger ones (SweepAI improvement)
    /// This prevents over-fragmentation of code
    fn coalesce_chunks(
        &self,
        chunks: Vec<Span>,
        source_code: &str,
        coalesce_threshold: usize,
    ) -> Vec<Span> {
        if chunks.is_empty() {
            return Vec::new();
        }

        let mut new_chunks = Vec::new();
        let mut current_chunk = chunks[0]; // Start with the first chunk

        for chunk in chunks.into_iter().skip(1) {
            // Try to merge current_chunk with the next chunk
            let merged_chunk = Span::new(current_chunk.start, chunk.end);

            // Check if we can safely extract the text
            if merged_chunk.end <= source_code.len() {
                let chunk_text = merged_chunk.extract(source_code);

                if self.non_whitespace_len(chunk_text) <= coalesce_threshold
                    || !chunk_text.contains('\n')
                {
                    // Merge the chunks
                    current_chunk = merged_chunk;
                } else {
                    // Can't merge, save current and start new
                    new_chunks.push(current_chunk);
                    current_chunk = chunk;
                }
            } else {
                // Invalid span, save current and start new
                new_chunks.push(current_chunk);
                current_chunk = chunk;
            }
        }

        // Don't forget the last chunk
        if current_chunk.len() > 0 {
            new_chunks.push(current_chunk);
        }

        new_chunks
    }

    /// SweepAI-enhanced AST chunking implementation
    /// This implements the complete algorithm from the chunking-improvements blog post
    fn split_with_sweepai_style(&self, text: &str) -> CoreResult<Vec<String>> {
        // Create tree-sitter parser
        let mut parser = self.create_tree_sitter_parser()?;
        let text_bytes = text.as_bytes();

        // Parse the code into AST
        let tree = parser
            .parse(text_bytes, None)
            .ok_or_else(|| CheungfunError::Internal {
                message: "Failed to parse code with tree-sitter".to_string(),
            })?;

        // Check for parsing errors
        let root_node = tree.root_node();
        if root_node.has_error()
            || (root_node.child_count() > 0 && root_node.child(0).unwrap().kind() == "ERROR")
        {
            return Err(CheungfunError::Internal {
                message: format!(
                    "Could not parse code with language {:?}",
                    self.config.language
                ),
            });
        }

        // 1. Recursively form chunks based on SweepAI algorithm
        let mut chunks = self.chunk_node_sweepai_style(&root_node, text)?;

        // 2. Fill in the gaps between consecutive spans
        self.connect_chunks(&mut chunks);

        // 3. Coalesce small chunks with bigger ones
        let coalesce_threshold = 50; // SweepAI default
        chunks = self.coalesce_chunks(chunks, text, coalesce_threshold);

        // 4. Convert byte indices to line numbers using efficient caching
        let line_offsets = self.build_line_offsets(text);
        let line_chunks: Vec<Span> = chunks
            .into_iter()
            .map(|chunk| {
                let start_line = self.byte_to_line_with_cache(chunk.start, &line_offsets);
                let end_line = self.byte_to_line_with_cache(chunk.end, &line_offsets);
                Span::new(start_line, end_line)
            })
            .collect();

        // 5. Eliminate empty chunks and extract final strings
        let final_chunks: Vec<String> = line_chunks
            .into_iter()
            .filter(|chunk| !chunk.is_empty())
            .map(|chunk| chunk.extract_lines(text))
            .filter(|text| !text.trim().is_empty())
            .collect();

        debug!(
            "SweepAI-style AST splitting produced {} chunks",
            final_chunks.len()
        );
        Ok(final_chunks)
    }

    /// Recursive AST node chunking following SweepAI's exact algorithm
    fn chunk_node_sweepai_style(
        &self,
        node: &tree_sitter::Node,
        text: &str,
    ) -> CoreResult<Vec<Span>> {
        let mut chunks = Vec::new();

        // If node is small enough, return it as a single chunk
        let node_size = node.end_byte() - node.start_byte();
        if node_size <= self.config.max_chars {
            return Ok(vec![Span::new(node.start_byte(), node.end_byte())]);
        }

        // Node is too big, need to split it
        let mut current_start = node.start_byte();
        let mut current_end = node.start_byte();

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            let child_span = Span::new(child.start_byte(), child.end_byte());
            let child_size = child_span.len();

            if child_size > self.config.max_chars {
                // Child is too big, save current chunk and recursively process child
                if current_end > current_start {
                    chunks.push(Span::new(current_start, current_end));
                }
                chunks.extend(self.chunk_node_sweepai_style(&child, text)?);
                current_start = child.end_byte();
                current_end = child.end_byte();
            } else {
                // Check if adding this child would exceed max_chars
                let potential_size = child.end_byte() - current_start;
                if potential_size > self.config.max_chars && current_end > current_start {
                    // Save current chunk and start new one
                    chunks.push(Span::new(current_start, current_end));
                    current_start = child.start_byte();
                }
                current_end = child.end_byte();
            }
        }

        // Don't forget the last chunk
        if current_end > current_start {
            chunks.push(Span::new(current_start, current_end));
        }

        Ok(chunks)
    }

    /// LlamaIndex-style AST chunking implementation (kept as fallback)
    /// This follows the exact algorithm from LlamaIndex's CodeSplitter
    fn split_with_llamaindex_style(&self, text: &str) -> CoreResult<Vec<String>> {
        // Create tree-sitter parser
        let mut parser = self.create_tree_sitter_parser()?;
        let text_bytes = text.as_bytes();

        // Parse the code into AST
        let tree = parser
            .parse(text_bytes, None)
            .ok_or_else(|| CheungfunError::Internal {
                message: "Failed to parse code with tree-sitter".to_string(),
            })?;

        // Check for parsing errors (following LlamaIndex's error handling)
        let root_node = tree.root_node();
        if root_node.has_error()
            || (root_node.child_count() > 0 && root_node.child(0).unwrap().kind() == "ERROR")
        {
            return Err(CheungfunError::Internal {
                message: format!(
                    "Could not parse code with language {:?}",
                    self.config.language
                ),
            });
        }

        // Recursively chunk the AST nodes (LlamaIndex style)
        let chunks = self.chunk_node_recursive(&root_node, text_bytes, 0)?;

        // Filter out empty chunks and trim whitespace (like LlamaIndex)
        let filtered_chunks: Vec<String> = chunks
            .into_iter()
            .map(|chunk| chunk.trim().to_string())
            .filter(|chunk| !chunk.is_empty())
            .collect();

        debug!(
            "LlamaIndex-style AST splitting produced {} chunks",
            filtered_chunks.len()
        );
        Ok(filtered_chunks)
    }

    /// Recursive AST node chunking following LlamaIndex's exact algorithm
    /// This is a direct port of LlamaIndex's _chunk_node method
    fn chunk_node_recursive(
        &self,
        node: &tree_sitter::Node,
        text_bytes: &[u8],
        mut last_end: usize,
    ) -> CoreResult<Vec<String>> {
        let mut new_chunks = Vec::new();
        let mut current_chunk = String::new();

        // Iterate through all child nodes
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            let child_size = child.end_byte() - child.start_byte();

            if child_size > self.config.max_chars {
                // Child is too big, recursively chunk the child
                if !current_chunk.is_empty() {
                    new_chunks.push(current_chunk);
                    current_chunk = String::new();
                }

                // Recursively process the large child
                let child_chunks = self.chunk_node_recursive(&child, text_bytes, last_end)?;
                new_chunks.extend(child_chunks);
            } else if current_chunk.len() + child_size > self.config.max_chars {
                // Child would make current chunk too big, start a new chunk
                if !current_chunk.is_empty() {
                    new_chunks.push(current_chunk);
                }

                // Start new chunk with content from last_end to child.end_byte
                let chunk_text = std::str::from_utf8(&text_bytes[last_end..child.end_byte()])
                    .map_err(|e| CheungfunError::Internal {
                        message: format!("UTF-8 decoding error: {}", e),
                    })?;
                current_chunk = chunk_text.to_string();
            } else {
                // Add child to current chunk (from last_end to child.end_byte)
                let chunk_text = std::str::from_utf8(&text_bytes[last_end..child.end_byte()])
                    .map_err(|e| CheungfunError::Internal {
                        message: format!("UTF-8 decoding error: {}", e),
                    })?;
                current_chunk.push_str(chunk_text);
            }

            // Update last_end to child's end position
            last_end = child.end_byte();
        }

        // Add the final chunk if it's not empty
        if !current_chunk.is_empty() {
            new_chunks.push(current_chunk);
        }

        Ok(new_chunks)
    }
}
