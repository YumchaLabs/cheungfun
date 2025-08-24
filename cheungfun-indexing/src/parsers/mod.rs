//! AST parsing module with tree-sitter integration.
//!
//! This module provides a modular architecture for parsing source code using tree-sitter,
//! with support for multiple programming languages and extensible language support.

pub mod ast_parser;

pub use ast_parser::{AstParser, AstParserConfig};

use crate::loaders::ProgrammingLanguage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Extracted function information from AST.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionInfo {
    /// Function name.
    pub name: String,
    /// Full function signature.
    pub signature: String,

    /// Return type if available.
    pub return_type: Option<String>,
    /// Visibility modifier (pub, private, etc.).
    pub visibility: Option<String>,
    /// Whether the function is async.
    pub is_async: bool,
    /// Whether the function is static/class method.
    pub is_static: bool,
    /// Start line number (1-based).
    pub start_line: usize,
    /// End line number (1-based).
    pub end_line: usize,
    /// Docstring or comments.
    pub documentation: Option<String>,
}

/// Extracted class/struct information from AST.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassInfo {
    /// Class/struct name.
    pub name: String,
    /// Kind of definition (class, struct, interface, enum, etc.).
    pub kind: String,
    /// Visibility modifier.
    pub visibility: Option<String>,
    /// Start line number (1-based).
    pub start_line: usize,
    /// End line number (1-based).
    pub end_line: usize,
    /// Docstring or comments.
    pub documentation: Option<String>,
}

/// Extracted import information from AST.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportInfo {
    /// Module or package name.
    pub module: String,
    /// Specific items imported.
    pub items: Vec<String>,
    /// Alias for the import.
    pub alias: Option<String>,
    /// Whether it's a wildcard import.
    pub is_wildcard: bool,
    /// Start line number (1-based).
    pub start_line: usize,
}

/// Extracted comment information from AST.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommentInfo {
    /// Comment content.
    pub content: String,
    /// Comment kind (line, block, doc).
    pub kind: CommentKind,
    /// Start line number (1-based).
    pub start_line: usize,
    /// End line number (1-based).
    pub end_line: usize,
}

/// Types of comments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommentKind {
    /// Single-line comment (// or #).
    Line,
    /// Multi-line block comment (/* */ or """ """).
    Block,
    /// Documentation comment (/// or /** */).
    Documentation,
}

/// Complete AST analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstAnalysis {
    /// Programming language.
    pub language: ProgrammingLanguage,
    /// Extracted functions.
    pub functions: Vec<FunctionInfo>,
    /// Extracted classes/structs.
    pub classes: Vec<ClassInfo>,
    /// Extracted imports.
    pub imports: Vec<ImportInfo>,
    /// Extracted comments.
    pub comments: Vec<CommentInfo>,
    /// Additional metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl AstAnalysis {
    /// Create a new empty AST analysis.
    pub fn new(language: ProgrammingLanguage) -> Self {
        Self {
            language,
            functions: Vec::new(),
            classes: Vec::new(),
            imports: Vec::new(),
            comments: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the analysis.
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get total lines of code (excluding comments and blank lines).
    pub fn lines_of_code(&self) -> usize {
        self.metadata
            .get("lines_of_code")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize
    }

    /// Get cyclomatic complexity if available.
    pub fn complexity(&self) -> Option<u32> {
        self.metadata
            .get("complexity")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
    }
}

/// Error types for AST parsing.
#[derive(Debug, thiserror::Error)]
pub enum AstError {
    /// Language is not supported by the AST parser.
    #[error("Language not supported: {0:?}")]
    UnsupportedLanguage(ProgrammingLanguage),

    /// Tree-sitter parsing failed.
    #[error("Tree-sitter parsing failed: {0}")]
    ParseError(String),

    /// Query compilation failed.
    #[error("Query compilation failed: {0}")]
    QueryError(String),

    /// Language initialization failed.
    #[error("Language initialization failed: {0}")]
    LanguageInitError(String),
}

/// Result type for AST operations.
pub type AstResult<T> = std::result::Result<T, AstError>;
