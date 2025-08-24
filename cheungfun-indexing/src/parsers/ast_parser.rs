//! AST parser implementation using tree-sitter.
//!
//! This module provides the main AST parser that uses the language registry
//! and query system to extract code structures from source files.

use crate::loaders::ProgrammingLanguage;
use crate::parsers::{
    AstAnalysis, AstError, AstResult, ClassInfo, CommentInfo, CommentKind, FunctionInfo, ImportInfo,
};
use tracing::{debug, error, warn};
use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator, Tree};

/// Configuration for AST parsing.
#[derive(Debug, Clone)]
pub struct AstParserConfig {
    /// Whether to extract function information.
    pub extract_functions: bool,
    /// Whether to extract class/struct information.
    pub extract_classes: bool,
    /// Whether to extract import information.
    pub extract_imports: bool,
    /// Whether to extract comment information.
    pub extract_comments: bool,
    /// Whether to include function bodies in analysis.
    pub include_function_bodies: bool,
    /// Maximum depth for AST traversal.
    pub max_depth: Option<usize>,
}

impl Default for AstParserConfig {
    fn default() -> Self {
        Self {
            extract_functions: true,
            extract_classes: true,
            extract_imports: true,
            extract_comments: false, // Comments can be expensive to extract
            include_function_bodies: false,
            max_depth: None,
        }
    }
}

/// AST parser that uses tree-sitter for code analysis.
pub struct AstParser {
    /// Parser configuration.
    config: AstParserConfig,
}

impl AstParser {
    /// Create a new AST parser with default language support.
    pub fn new() -> AstResult<Self> {
        Ok(Self {
            config: AstParserConfig::default(),
        })
    }

    /// Create an AST parser with custom configuration.
    pub fn with_config(config: AstParserConfig) -> AstResult<Self> {
        Ok(Self { config })
    }

    /// Check if a language is supported.
    pub fn is_supported(&self, language: &ProgrammingLanguage) -> bool {
        matches!(
            language,
            ProgrammingLanguage::Rust
                | ProgrammingLanguage::Python
                | ProgrammingLanguage::JavaScript
                | ProgrammingLanguage::TypeScript
                | ProgrammingLanguage::Java
                | ProgrammingLanguage::CSharp
                | ProgrammingLanguage::C
                | ProgrammingLanguage::Cpp
                | ProgrammingLanguage::Go
        )
    }

    /// Get language for a specific programming language.
    fn get_language(&self, language: &ProgrammingLanguage) -> AstResult<Language> {
        match language {
            ProgrammingLanguage::Rust => Ok(tree_sitter_rust::LANGUAGE.into()),
            ProgrammingLanguage::Python => Ok(tree_sitter_python::LANGUAGE.into()),
            ProgrammingLanguage::JavaScript => Ok(tree_sitter_javascript::LANGUAGE.into()),
            ProgrammingLanguage::TypeScript => {
                Ok(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
            }
            ProgrammingLanguage::Java => Ok(tree_sitter_java::LANGUAGE.into()),
            ProgrammingLanguage::CSharp => Ok(tree_sitter_c_sharp::LANGUAGE.into()),
            ProgrammingLanguage::C => Ok(tree_sitter_c::LANGUAGE.into()),
            ProgrammingLanguage::Cpp => Ok(tree_sitter_cpp::LANGUAGE.into()),
            ProgrammingLanguage::Go => Ok(tree_sitter_go::LANGUAGE.into()),
            _ => Err(AstError::UnsupportedLanguage(*language)),
        }
    }

    /// Create a parser for a specific language.
    fn create_parser(&self, language: &ProgrammingLanguage) -> AstResult<Parser> {
        let lang = self.get_language(language)?;
        let mut parser = Parser::new();
        parser
            .set_language(&lang)
            .map_err(|e| AstError::LanguageInitError(e.to_string()))?;
        Ok(parser)
    }

    /// Parse source code and extract AST information.
    pub fn parse(&self, content: &str, language: ProgrammingLanguage) -> AstResult<AstAnalysis> {
        if !self.is_supported(&language) {
            return Err(AstError::UnsupportedLanguage(language));
        }

        let mut parser = self.create_parser(&language)?;
        let tree = parser
            .parse(content, None)
            .ok_or_else(|| AstError::ParseError("Failed to parse source code".to_string()))?;

        let mut analysis = AstAnalysis::new(language);

        // Extract different code structures based on configuration
        if self.config.extract_functions {
            analysis.functions = self.extract_functions(&tree, content, language)?;
        }

        if self.config.extract_classes {
            analysis.classes = self.extract_classes(&tree, content, language)?;
        }

        if self.config.extract_imports {
            analysis.imports = self.extract_imports(&tree, content, language)?;
        }

        if self.config.extract_comments {
            analysis.comments = self.extract_comments(&tree, content, language)?;
        }

        // Add metadata
        analysis = analysis.with_metadata(
            "total_lines".to_string(),
            serde_json::Value::Number(content.lines().count().into()),
        );

        let loc = self.calculate_lines_of_code(content);
        analysis = analysis.with_metadata(
            "lines_of_code".to_string(),
            serde_json::Value::Number(loc.into()),
        );

        debug!(
            "Parsed {} code: {} functions, {} classes, {} imports",
            language.as_str(),
            analysis.functions.len(),
            analysis.classes.len(),
            analysis.imports.len()
        );

        Ok(analysis)
    }

    /// Extract function information from the AST.
    fn extract_functions(
        &self,
        tree: &Tree,
        content: &str,
        language: ProgrammingLanguage,
    ) -> AstResult<Vec<FunctionInfo>> {
        let query_str = match language {
            ProgrammingLanguage::Rust => "(function_item name: (identifier) @name) @function",
            ProgrammingLanguage::Python => {
                "(function_definition name: (identifier) @name) @function"
            }
            ProgrammingLanguage::JavaScript | ProgrammingLanguage::TypeScript => {
                "[(function_declaration name: (identifier) @name) (method_definition name: (property_identifier) @name)] @function"
            }
            ProgrammingLanguage::Java => "(method_declaration name: (identifier) @name) @function",
            ProgrammingLanguage::CSharp => {
                "(method_declaration name: (identifier) @name) @function"
            }
            ProgrammingLanguage::C | ProgrammingLanguage::Cpp => {
                "(function_definition declarator: (function_declarator declarator: (identifier) @name)) @function"
            }
            ProgrammingLanguage::Go => "(function_declaration name: (identifier) @name) @function",
            _ => {
                warn!("Function extraction not implemented for {:?}", language);
                return Ok(Vec::new());
            }
        };

        let lang = self.get_language(&language)?;
        let query =
            Query::new(&lang, query_str).map_err(|e| AstError::QueryError(e.to_string()))?;

        let mut cursor = QueryCursor::new();
        let mut functions = Vec::new();

        // Use the new StreamingIterator API
        let mut matches = cursor.matches(&query, tree.root_node(), content.as_bytes());

        while let Some(query_match) = matches.next() {
            if let Some(function_info) =
                self.extract_function_from_match(query_match, &query, content)
            {
                functions.push(function_info);
            }
        }

        debug!("Extracted {} functions for {:?}", functions.len(), language);
        Ok(functions)
    }

    /// Extract class information from the AST.
    fn extract_classes(
        &self,
        tree: &Tree,
        content: &str,
        language: ProgrammingLanguage,
    ) -> AstResult<Vec<ClassInfo>> {
        let query_str = match language {
            ProgrammingLanguage::Rust => {
                "[(struct_item name: (type_identifier) @name) (enum_item name: (type_identifier) @name) (trait_item name: (type_identifier) @name)] @class"
            }
            ProgrammingLanguage::Python => "(class_definition name: (identifier) @name) @class",
            ProgrammingLanguage::JavaScript | ProgrammingLanguage::TypeScript => {
                "(class_declaration name: (identifier) @name) @class"
            }
            ProgrammingLanguage::Java => {
                "[(class_declaration name: (identifier) @name) (interface_declaration name: (identifier) @name) (enum_declaration name: (identifier) @name)] @class"
            }
            ProgrammingLanguage::CSharp => {
                "[(class_declaration name: (identifier) @name) (interface_declaration name: (identifier) @name) (struct_declaration name: (identifier) @name) (enum_declaration name: (identifier) @name)] @class"
            }
            ProgrammingLanguage::C | ProgrammingLanguage::Cpp => {
                "[(struct_specifier name: (type_identifier) @name) (class_specifier name: (type_identifier) @name) (enum_specifier name: (type_identifier) @name)] @class"
            }
            ProgrammingLanguage::Go => {
                "[(type_declaration (type_spec name: (type_identifier) @name type: (struct_type))) (type_declaration (type_spec name: (type_identifier) @name type: (interface_type)))] @class"
            }
            _ => {
                warn!("Class extraction not implemented for {:?}", language);
                return Ok(Vec::new());
            }
        };

        let lang = self.get_language(&language)?;
        let query =
            Query::new(&lang, query_str).map_err(|e| AstError::QueryError(e.to_string()))?;

        let mut cursor = QueryCursor::new();
        let mut classes = Vec::new();

        let mut matches = cursor.matches(&query, tree.root_node(), content.as_bytes());

        while let Some(query_match) = matches.next() {
            if let Some(class_info) = self.extract_class_from_match(query_match, &query, content) {
                classes.push(class_info);
            }
        }

        debug!("Extracted {} classes for {:?}", classes.len(), language);
        Ok(classes)
    }

    /// Extract import information from the AST.
    fn extract_imports(
        &self,
        tree: &Tree,
        content: &str,
        language: ProgrammingLanguage,
    ) -> AstResult<Vec<ImportInfo>> {
        let query_str = match language {
            ProgrammingLanguage::Rust => "(use_declaration) @import",
            ProgrammingLanguage::Python => "[(import_statement) (import_from_statement)] @import",
            ProgrammingLanguage::JavaScript | ProgrammingLanguage::TypeScript => {
                "(import_statement) @import"
            }
            ProgrammingLanguage::Java => "(import_declaration) @import",
            ProgrammingLanguage::CSharp => "(using_directive) @import",
            ProgrammingLanguage::C | ProgrammingLanguage::Cpp => "(preproc_include) @import",
            ProgrammingLanguage::Go => "(import_declaration) @import",
            _ => {
                warn!("Import extraction not implemented for {:?}", language);
                return Ok(Vec::new());
            }
        };

        let lang = self.get_language(&language)?;
        let query =
            Query::new(&lang, query_str).map_err(|e| AstError::QueryError(e.to_string()))?;

        let mut cursor = QueryCursor::new();
        let mut imports = Vec::new();

        let mut matches = cursor.matches(&query, tree.root_node(), content.as_bytes());

        while let Some(query_match) = matches.next() {
            if let Some(import_info) = self.extract_import_from_match(query_match, &query, content)
            {
                imports.push(import_info);
            }
        }

        debug!("Extracted {} imports for {:?}", imports.len(), language);
        Ok(imports)
    }

    /// Extract comment information from the AST.
    fn extract_comments(
        &self,
        tree: &Tree,
        content: &str,
        language: ProgrammingLanguage,
    ) -> AstResult<Vec<CommentInfo>> {
        let query_str = match language {
            ProgrammingLanguage::Rust => "[(line_comment) (block_comment) (doc_comment)] @comment",
            ProgrammingLanguage::Python => "(comment) @comment",
            ProgrammingLanguage::JavaScript | ProgrammingLanguage::TypeScript => {
                "(comment) @comment"
            }
            ProgrammingLanguage::Java => "[(line_comment) (block_comment)] @comment",
            ProgrammingLanguage::CSharp => "(comment) @comment",
            ProgrammingLanguage::C | ProgrammingLanguage::Cpp => "(comment) @comment",
            ProgrammingLanguage::Go => "(comment) @comment",
            _ => {
                warn!("Comment extraction not implemented for {:?}", language);
                return Ok(Vec::new());
            }
        };

        let lang = self.get_language(&language)?;
        let query =
            Query::new(&lang, query_str).map_err(|e| AstError::QueryError(e.to_string()))?;

        let mut cursor = QueryCursor::new();
        let mut comments = Vec::new();

        let mut matches = cursor.matches(&query, tree.root_node(), content.as_bytes());

        while let Some(query_match) = matches.next() {
            if let Some(comment_info) =
                self.extract_comment_from_match(query_match, &query, content)
            {
                comments.push(comment_info);
            }
        }

        debug!("Extracted {} comments for {:?}", comments.len(), language);
        Ok(comments)
    }

    /// Extract function information from a query match.
    fn extract_function_from_match(
        &self,
        query_match: &tree_sitter::QueryMatch,
        query: &Query,
        content: &str,
    ) -> Option<FunctionInfo> {
        let mut function_info = FunctionInfo {
            name: String::new(),
            signature: String::new(),
            return_type: None,
            visibility: None,
            is_async: false,
            is_static: false,
            start_line: 0,
            end_line: 0,
            documentation: None,
        };

        for capture in query_match.captures {
            let node = capture.node;
            let capture_name = query.capture_names()[capture.index as usize];
            let node_text = node.utf8_text(content.as_bytes()).ok()?;

            match capture_name {
                "name" => function_info.name = node_text.to_string(),
                "function" => {
                    function_info.start_line = node.start_position().row + 1;
                    function_info.end_line = node.end_position().row + 1;
                    function_info.signature =
                        node_text.lines().next().unwrap_or("").trim().to_string();
                }
                _ => {}
            }
        }

        if function_info.name.is_empty() {
            None
        } else {
            Some(function_info)
        }
    }

    /// Extract class information from a query match.
    fn extract_class_from_match(
        &self,
        query_match: &tree_sitter::QueryMatch,
        query: &Query,
        content: &str,
    ) -> Option<ClassInfo> {
        let mut class_info = ClassInfo {
            name: String::new(),
            kind: String::new(),
            visibility: None,
            start_line: 0,
            end_line: 0,
            documentation: None,
        };

        for capture in query_match.captures {
            let node = capture.node;
            let capture_name = query.capture_names()[capture.index as usize];
            let node_text = node.utf8_text(content.as_bytes()).ok()?;

            match capture_name {
                "name" => class_info.name = node_text.to_string(),
                "class" => {
                    class_info.start_line = node.start_position().row + 1;
                    class_info.end_line = node.end_position().row + 1;
                    class_info.kind = "class".to_string(); // Default kind
                }
                _ => {}
            }
        }

        if class_info.name.is_empty() {
            None
        } else {
            Some(class_info)
        }
    }

    /// Extract import information from a query match.
    fn extract_import_from_match(
        &self,
        query_match: &tree_sitter::QueryMatch,
        query: &Query,
        content: &str,
    ) -> Option<ImportInfo> {
        let mut import_info = ImportInfo {
            module: String::new(),
            items: Vec::new(),
            alias: None,
            is_wildcard: false,
            start_line: 0,
        };

        for capture in query_match.captures {
            let node = capture.node;
            let capture_name = query.capture_names()[capture.index as usize];
            let node_text = node.utf8_text(content.as_bytes()).ok()?;

            match capture_name {
                "import" => {
                    import_info.module = node_text.trim_matches('"').trim_matches('\'').to_string();
                    import_info.start_line = node.start_position().row + 1;
                }
                _ => {}
            }
        }

        if import_info.module.is_empty() {
            None
        } else {
            Some(import_info)
        }
    }

    /// Extract comment information from a query match.
    fn extract_comment_from_match(
        &self,
        query_match: &tree_sitter::QueryMatch,
        _query: &Query,
        content: &str,
    ) -> Option<CommentInfo> {
        for capture in query_match.captures {
            let node = capture.node;
            let node_text = node.utf8_text(content.as_bytes()).ok()?;

            let kind = if node_text.starts_with("///") || node_text.starts_with("/**") {
                CommentKind::Documentation
            } else if node_text.starts_with("//") || node_text.starts_with("#") {
                CommentKind::Line
            } else if node_text.starts_with("/*") || node_text.starts_with("\"\"\"") {
                CommentKind::Block
            } else {
                CommentKind::Line
            };

            return Some(CommentInfo {
                content: node_text.to_string(),
                kind,
                start_line: node.start_position().row + 1,
                end_line: node.end_position().row + 1,
            });
        }

        None
    }

    /// Calculate lines of code (excluding comments and blank lines).
    fn calculate_lines_of_code(&self, content: &str) -> usize {
        content
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && !self.is_likely_comment(trimmed)
            })
            .count()
    }

    /// Simple heuristic to detect comment lines.
    fn is_likely_comment(&self, line: &str) -> bool {
        line.starts_with("//")
            || line.starts_with("#")
            || line.starts_with("/*")
            || line.starts_with("*")
            || line.starts_with("\"\"\"")
            || line.starts_with("'''")
    }

    /// Get supported languages.
    pub fn supported_languages(&self) -> Vec<ProgrammingLanguage> {
        vec![
            ProgrammingLanguage::Rust,
            ProgrammingLanguage::Python,
            ProgrammingLanguage::JavaScript,
            ProgrammingLanguage::TypeScript,
            ProgrammingLanguage::Java,
            ProgrammingLanguage::CSharp,
            ProgrammingLanguage::C,
            ProgrammingLanguage::Cpp,
            ProgrammingLanguage::Go,
        ]
    }
}

impl Default for AstParser {
    fn default() -> Self {
        Self::new().unwrap_or_else(|e| {
            error!("Failed to create default AST parser: {}", e);
            panic!("Cannot create AST parser without language support");
        })
    }
}
