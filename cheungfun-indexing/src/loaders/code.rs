//! Code-specific document loader with AST parsing support.
//!
//! This module provides specialized loading for source code files with enhanced
//! metadata extraction using tree-sitter AST parsing.

use crate::loaders::{utils, LoaderConfig};
use crate::parsers::AstParser;
use async_trait::async_trait;
use cheungfun_core::Loader;
use cheungfun_core::{Document, Result as CoreResult};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{error, warn};

use crate::error::{IndexingError, Result};

/// Configuration for code-specific loading behavior.
#[derive(Debug, Clone)]
pub struct CodeLoaderConfig {
    /// Base loader configuration.
    pub base: LoaderConfig,

    /// Whether to extract function/method signatures.
    pub extract_functions: bool,

    /// Whether to extract class/struct definitions.
    pub extract_classes: bool,

    /// Whether to extract import/include statements.
    pub extract_imports: bool,

    /// Whether to extract comments and docstrings.
    pub extract_comments: bool,

    /// Maximum file size to process (in bytes).
    pub max_file_size: Option<u64>,

    /// Whether to include function bodies in the analysis.
    pub include_function_bodies: bool,

    /// Maximum number of lines to process per chunk.
    pub max_chunk_lines: usize,

    /// Maximum number of characters to process per chunk.
    pub max_chunk_chars: usize,
}

impl Default for CodeLoaderConfig {
    fn default() -> Self {
        Self {
            base: LoaderConfig::default(),
            extract_functions: true,
            extract_classes: true,
            extract_imports: true,
            extract_comments: false, // Comments can be expensive
            max_file_size: Some(10 * 1024 * 1024), // 10MB
            include_function_bodies: false,
            max_chunk_lines: 1000,
            max_chunk_chars: 50_000,
        }
    }
}

/// Metadata extracted from code files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeMetadata {
    /// Programming language detected.
    pub language: ProgrammingLanguage,
    /// Function/method names found.
    pub functions: Vec<String>,
    /// Class/struct names found.
    pub classes: Vec<String>,
    /// Import/include statements found.
    pub imports: Vec<String>,
    /// Comments and docstrings found.
    pub comments: Vec<String>,
    /// Lines of code (excluding comments and blank lines).
    pub loc: usize,
    /// Total lines in the file.
    pub total_lines: usize,
    /// Code complexity score (if calculated).
    pub complexity: Option<u32>,
}

/// Supported programming languages for enhanced parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProgrammingLanguage {
    /// Rust programming language.
    Rust,
    /// Python programming language.
    Python,
    /// JavaScript programming language.
    JavaScript,
    /// TypeScript programming language.
    TypeScript,
    /// Java programming language.
    Java,
    /// C# programming language.
    CSharp,
    /// C++ programming language.
    Cpp,
    /// C programming language.
    C,
    /// Go programming language.
    Go,
    /// PHP programming language.
    PHP,
    /// Ruby programming language.
    Ruby,
    /// Swift programming language.
    Swift,
    /// Kotlin programming language.
    Kotlin,
    /// Scala programming language.
    Scala,
    /// Haskell programming language.
    Haskell,
    /// Clojure programming language.
    Clojure,
    /// Erlang programming language.
    Erlang,
    /// Elixir programming language.
    Elixir,
    /// Lua programming language.
    Lua,
    /// SQL query language.
    SQL,
    /// HTML markup language.
    HTML,
    /// XML markup language.
    XML,
    /// CSS stylesheet language.
    CSS,
    /// Shell scripting language.
    Shell,
    /// JSON data format.
    JSON,
    /// YAML data format.
    YAML,
    /// TOML configuration format.
    TOML,
    /// Markdown markup language.
    Markdown,
    /// Unknown or unsupported language.
    Unknown,
}

impl ProgrammingLanguage {
    /// Detect programming language from file extension.
    #[must_use]
    pub fn from_extension(extension: &str) -> Self {
        match extension.to_lowercase().as_str() {
            "rs" => Self::Rust,
            "py" | "pyw" => Self::Python,
            "js" | "mjs" => Self::JavaScript,
            "ts" | "tsx" => Self::TypeScript,
            "java" => Self::Java,
            "cs" => Self::CSharp,
            "cpp" | "cxx" | "cc" => Self::Cpp,
            "c" | "h" => Self::C,
            "go" => Self::Go,
            "php" => Self::PHP,
            "rb" => Self::Ruby,
            "swift" => Self::Swift,
            "kt" | "kts" => Self::Kotlin,
            "scala" | "sc" => Self::Scala,
            "hs" => Self::Haskell,
            "clj" | "cljs" => Self::Clojure,
            "erl" => Self::Erlang,
            "ex" | "exs" => Self::Elixir,
            "lua" => Self::Lua,
            "sql" => Self::SQL,
            "html" | "htm" => Self::HTML,
            "xml" => Self::XML,
            "css" => Self::CSS,
            "sh" | "bash" | "zsh" => Self::Shell,
            "json" => Self::JSON,
            "yaml" | "yml" => Self::YAML,
            "toml" => Self::TOML,
            "md" | "markdown" => Self::Markdown,
            _ => Self::Unknown,
        }
    }

    /// Get the language name as a string.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Java => "java",
            Self::CSharp => "csharp",
            Self::Cpp => "cpp",
            Self::C => "c",
            Self::Go => "go",
            Self::PHP => "php",
            Self::Ruby => "ruby",
            Self::Swift => "swift",
            Self::Kotlin => "kotlin",
            Self::Scala => "scala",
            Self::Haskell => "haskell",
            Self::Clojure => "clojure",
            Self::Erlang => "erlang",
            Self::Elixir => "elixir",
            Self::Lua => "lua",
            Self::SQL => "sql",
            Self::HTML => "html",
            Self::XML => "xml",
            Self::CSS => "css",
            Self::Shell => "shell",
            Self::JSON => "json",
            Self::YAML => "yaml",
            Self::TOML => "toml",
            Self::Markdown => "markdown",
            Self::Unknown => "unknown",
        }
    }
}

/// Code-specific document loader.
#[derive(Debug)]
pub struct CodeLoader {
    /// Path to the code file or directory.
    path: PathBuf,
    /// Configuration for the loader.
    config: CodeLoaderConfig,
}

impl CodeLoader {
    /// Create a new code loader for the specified path.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(IndexingError::file_not_found(path.display().to_string()));
        }

        Ok(Self {
            path,
            config: CodeLoaderConfig::default(),
        })
    }

    /// Create a new code loader with custom configuration.
    pub fn with_config<P: AsRef<Path>>(path: P, config: CodeLoaderConfig) -> Result<Self> {
        let mut loader = Self::new(path)?;
        loader.config = config;
        Ok(loader)
    }

    /// Detect the programming language from file extension.
    fn detect_language(&self, path: &Path) -> ProgrammingLanguage {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(ProgrammingLanguage::from_extension)
            .unwrap_or(ProgrammingLanguage::Unknown)
    }

    /// Check if a file should be processed as a code file.
    fn is_code_file(&self, path: &Path) -> bool {
        let language = self.detect_language(path);
        !matches!(language, ProgrammingLanguage::Unknown)
    }

    /// Extract code metadata using AST parsing.
    fn extract_metadata(&self, content: &str, language: ProgrammingLanguage) -> CodeMetadata {
        // Use AST parser for enhanced metadata extraction
        match AstParser::new().and_then(|parser| parser.parse(content, language)) {
            Ok(analysis) => {
                let functions: Vec<String> =
                    analysis.functions.iter().map(|f| f.name.clone()).collect();
                let classes: Vec<String> =
                    analysis.classes.iter().map(|c| c.name.clone()).collect();
                let imports: Vec<String> =
                    analysis.imports.iter().map(|i| i.module.clone()).collect();
                let comments: Vec<String> = analysis
                    .comments
                    .iter()
                    .map(|c| c.content.clone())
                    .collect();
                let loc = analysis.lines_of_code();
                let total_lines = analysis
                    .metadata
                    .get("total_lines")
                    .and_then(|v| v.as_u64())
                    .map_or(0, |v| v as usize);
                let complexity = analysis.complexity();

                CodeMetadata {
                    language,
                    functions,
                    classes,
                    imports,
                    comments,
                    loc,
                    total_lines,
                    complexity,
                }
            }
            Err(e) => {
                warn!(
                    "AST parsing failed for {:?}: {}, using empty metadata",
                    language, e
                );
                // Return empty metadata if AST parsing fails
                CodeMetadata {
                    language,
                    functions: Vec::new(),
                    classes: Vec::new(),
                    imports: Vec::new(),
                    comments: Vec::new(),
                    loc: content
                        .lines()
                        .filter(|line| !line.trim().is_empty())
                        .count(),
                    total_lines: content.lines().count(),
                    complexity: None,
                }
            }
        }
    }

    /// Create a document with code-specific metadata.
    fn create_code_document(
        &self,
        content: String,
        path: &Path,
        metadata: CodeMetadata,
    ) -> Document {
        let mut doc = utils::create_document_from_file(content, path, None, None);

        // Add code-specific metadata
        doc.metadata.insert(
            "language".to_string(),
            serde_json::Value::String(metadata.language.as_str().to_string()),
        );

        doc.metadata.insert(
            "loc".to_string(),
            serde_json::Value::Number(metadata.loc.into()),
        );

        doc.metadata.insert(
            "total_lines".to_string(),
            serde_json::Value::Number(metadata.total_lines.into()),
        );

        if !metadata.functions.is_empty() {
            doc.metadata.insert(
                "functions".to_string(),
                serde_json::Value::Array(
                    metadata
                        .functions
                        .into_iter()
                        .map(serde_json::Value::String)
                        .collect(),
                ),
            );
        }

        if !metadata.classes.is_empty() {
            doc.metadata.insert(
                "classes".to_string(),
                serde_json::Value::Array(
                    metadata
                        .classes
                        .into_iter()
                        .map(serde_json::Value::String)
                        .collect(),
                ),
            );
        }

        if !metadata.imports.is_empty() {
            doc.metadata.insert(
                "imports".to_string(),
                serde_json::Value::Array(
                    metadata
                        .imports
                        .into_iter()
                        .map(serde_json::Value::String)
                        .collect(),
                ),
            );
        }

        if !metadata.comments.is_empty() {
            doc.metadata.insert(
                "comments".to_string(),
                serde_json::Value::Array(
                    metadata
                        .comments
                        .into_iter()
                        .map(serde_json::Value::String)
                        .collect(),
                ),
            );
        }

        if let Some(complexity) = metadata.complexity {
            doc.metadata.insert(
                "complexity".to_string(),
                serde_json::Value::Number(complexity.into()),
            );
        }

        doc
    }
}

#[async_trait]
impl Loader for CodeLoader {
    async fn load(&self) -> CoreResult<Vec<Document>> {
        let mut documents = Vec::new();

        if self.path.is_file() {
            if self.is_code_file(&self.path) {
                match self.load_file(&self.path).await {
                    Ok(doc) => documents.push(doc),
                    Err(e) => {
                        error!("Failed to load code file {:?}: {}", self.path, e);
                        if !self.config.base.continue_on_error {
                            return Err(e.into());
                        }
                    }
                }
            }
        } else if self.path.is_dir() {
            documents = self.load_directory(&self.path).await?;
        }

        Ok(documents)
    }
}

impl CodeLoader {
    /// Load a single code file.
    async fn load_file(&self, path: &Path) -> Result<Document> {
        let content = fs::read_to_string(path)
            .await
            .map_err(|e| IndexingError::from(e))?;

        // Check file size limit
        if let Some(max_size) = self.config.max_file_size {
            if content.len() as u64 > max_size {
                return Err(IndexingError::configuration(format!(
                    "File {:?} exceeds maximum size limit of {} bytes",
                    path, max_size
                )));
            }
        }

        let language = self.detect_language(path);
        let metadata = self.extract_metadata(&content, language);

        Ok(self.create_code_document(content, path, metadata))
    }

    /// Load all code files from a directory.
    fn load_directory<'a>(
        &'a self,
        dir: &'a Path,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Document>>> + Send + 'a>>
    {
        Box::pin(async move {
            let mut documents = Vec::new();
            let mut entries = fs::read_dir(dir)
                .await
                .map_err(|e| IndexingError::from(e))?;

            while let Some(entry) = entries
                .next_entry()
                .await
                .map_err(|e| IndexingError::from(e))?
            {
                let path = entry.path();

                if path.is_file() && self.is_code_file(&path) {
                    match self.load_file(&path).await {
                        Ok(doc) => documents.push(doc),
                        Err(e) => {
                            error!("Failed to load code file {:?}: {}", path, e);
                            if !self.config.base.continue_on_error {
                                return Err(e);
                            }
                        }
                    }
                } else if path.is_dir() && self.config.base.max_depth.map_or(true, |d| d > 0) {
                    match self.load_directory(&path).await {
                        Ok(mut sub_docs) => documents.append(&mut sub_docs),
                        Err(e) => {
                            error!("Failed to load subdirectory {:?}: {}", path, e);
                            if !self.config.base.continue_on_error {
                                return Err(e);
                            }
                        }
                    }
                }
            }

            Ok(documents)
        })
    }
}
