//! Markdown node parser implementation.
//!
//! This module provides Markdown-aware text parsing that splits documents
//! based on header structure while preserving the hierarchical context,
//! following the LlamaIndex MarkdownNodeParser pattern.

use crate::node_parser::{config::MarkdownConfig, NodeParser, TextSplitter};
use async_trait::async_trait;
use cheungfun_core::{CheungfunError, Document, Node, Result as CoreResult};
use regex::Regex;
use tracing::debug;

/// Markdown node parser that splits text based on header structure.
///
/// This parser analyzes Markdown documents and creates nodes based on the header
/// hierarchy. Each node contains the content of a section along with metadata
/// about its position in the document structure.
///
/// The parser supports:
/// - Header-based section splitting (# ## ### etc.)
/// - Hierarchical metadata with header paths
/// - Code block handling
/// - Configurable header depth limits
/// - Section length filtering
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::node_parser::text::MarkdownNodeParser;
/// use cheungfun_core::Document;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let parser = MarkdownNodeParser::new()
///         .with_max_header_depth(3)
///         .with_preserve_header_hierarchy(true);
///     
///     let markdown = r#"
/// # Introduction
/// This is the introduction section.
///
/// ## Getting Started
/// Here's how to get started.
///
/// ### Installation
/// Run the following command:
/// ```bash
/// cargo install cheungfun
/// ```
///     "#;
///     
///     let document = Document::new(markdown);
///     let nodes = parser.parse_nodes(&[document], false).await?;
///     
///     println!("Generated {} nodes", nodes.len());
///     for (i, node) in nodes.iter().enumerate() {
///         println!("Node {}: {}", i + 1, node.content.lines().next().unwrap_or(""));
///         if let Some(path) = node.metadata.get("header_path") {
///             println!("  Path: {}", path);
///         }
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct MarkdownNodeParser {
    /// Configuration for the markdown parser.
    config: MarkdownConfig,
    /// Regex for matching headers.
    header_regex: Regex,
    /// Regex for matching code blocks.
    code_block_regex: Regex,
}

/// Represents a markdown section with its metadata.
#[derive(Debug, Clone)]
struct MarkdownSection {
    /// The content of the section.
    content: String,
    /// The header level (1-6).
    level: usize,
    /// The header text.
    header: String,
    /// The full path to this section.
    header_path: String,
    /// Start position in the original text.
    start_pos: usize,
    /// End position in the original text.
    end_pos: usize,
}

impl MarkdownNodeParser {
    /// Create a new markdown node parser with default configuration.
    pub fn new() -> Self {
        let config = MarkdownConfig::default();
        Self::from_config(config).unwrap()
    }

    /// Create a markdown parser from configuration.
    pub fn from_config(config: MarkdownConfig) -> CoreResult<Self> {
        let header_regex =
            Regex::new(r"^(#{1,6})\s+(.+)$").map_err(|e| CheungfunError::Validation {
                message: format!("Invalid header regex: {}", e),
            })?;

        let code_block_regex =
            Regex::new(r"```[\s\S]*?```").map_err(|e| CheungfunError::Validation {
                message: format!("Invalid code block regex: {}", e),
            })?;

        Ok(Self {
            config,
            header_regex,
            code_block_regex,
        })
    }

    /// Create a markdown parser with default settings.
    pub fn from_defaults(
        max_header_depth: usize,
        preserve_hierarchy: bool,
        include_header_in_content: bool,
    ) -> CoreResult<Self> {
        let config = MarkdownConfig::new()
            .with_max_header_depth(max_header_depth)
            .with_preserve_header_hierarchy(preserve_hierarchy)
            .with_include_header_in_content(include_header_in_content);

        Self::from_config(config)
    }

    /// Set maximum header depth.
    pub fn with_max_header_depth(mut self, depth: usize) -> Self {
        self.config.max_header_depth = depth.min(6).max(1);
        self
    }

    /// Set whether to preserve header hierarchy.
    pub fn with_preserve_header_hierarchy(mut self, preserve: bool) -> Self {
        self.config.preserve_header_hierarchy = preserve;
        self
    }

    /// Set whether to include header in content.
    pub fn with_include_header_in_content(mut self, include: bool) -> Self {
        self.config.include_header_in_content = include;
        self
    }

    /// Set header path separator.
    pub fn with_header_path_separator<S: Into<String>>(mut self, separator: S) -> Self {
        self.config.header_path_separator = separator.into();
        self
    }

    /// Set minimum section length.
    pub fn with_min_section_length(mut self, length: usize) -> Self {
        self.config.min_section_length = length;
        self
    }

    /// Parse markdown text into sections.
    fn parse_markdown_sections(&self, text: &str) -> CoreResult<Vec<MarkdownSection>> {
        let lines: Vec<&str> = text.lines().collect();
        let mut sections = Vec::new();
        let mut current_section = String::new();
        let mut header_stack: Vec<(usize, String)> = Vec::new(); // (level, header_text)
        let mut current_header = String::new();
        let mut current_level = 0;
        let mut section_start = 0;

        for (line_idx, line) in lines.iter().enumerate() {
            if let Some(captures) = self.header_regex.captures(line) {
                let header_marks = captures.get(1).unwrap().as_str();
                let header_text = captures.get(2).unwrap().as_str().trim();
                let level = header_marks.len();

                // Skip headers deeper than max depth
                if level > self.config.max_header_depth {
                    current_section.push_str(line);
                    current_section.push('\n');
                    continue;
                }

                // Save the previous section if it has content
                if !current_section.trim().is_empty() && !current_header.is_empty() {
                    let header_path = self.build_header_path(&header_stack);
                    let section = MarkdownSection {
                        content: current_section.trim().to_string(),
                        level: current_level,
                        header: current_header.clone(),
                        header_path,
                        start_pos: section_start,
                        end_pos: line_idx,
                    };
                    sections.push(section);
                }

                // Update header stack
                self.update_header_stack(&mut header_stack, level, header_text.to_string());

                // Start new section
                current_header = header_text.to_string();
                current_level = level;
                section_start = line_idx;
                current_section.clear();

                // Include header in content if configured
                if self.config.include_header_in_content {
                    current_section.push_str(line);
                    current_section.push('\n');
                }
            } else {
                current_section.push_str(line);
                current_section.push('\n');
            }
        }

        // Add the final section
        if !current_section.trim().is_empty() {
            let header_path = if current_header.is_empty() {
                String::new()
            } else {
                self.build_header_path(&header_stack)
            };

            let section = MarkdownSection {
                content: current_section.trim().to_string(),
                level: current_level,
                header: current_header,
                header_path,
                start_pos: section_start,
                end_pos: lines.len(),
            };
            sections.push(section);
        }

        // Filter sections by minimum length
        if self.config.min_section_length > 0 {
            sections.retain(|section| section.content.len() >= self.config.min_section_length);
        }

        debug!("Parsed {} markdown sections", sections.len());
        Ok(sections)
    }

    /// Update the header stack based on the current header level.
    fn update_header_stack(&self, stack: &mut Vec<(usize, String)>, level: usize, header: String) {
        // Remove headers at the same or deeper level
        stack.retain(|(l, _)| *l < level);

        // Add the current header
        stack.push((level, header));
    }

    /// Build header path from the header stack.
    fn build_header_path(&self, stack: &[(usize, String)]) -> String {
        if stack.is_empty() {
            return String::new();
        }

        let path_parts: Vec<&str> = stack.iter().map(|(_, header)| header.as_str()).collect();
        format!(
            "{}{}{}",
            self.config.header_path_separator,
            path_parts.join(&self.config.header_path_separator),
            self.config.header_path_separator
        )
    }

    /// Build nodes from markdown sections.
    fn build_nodes_from_sections(
        &self,
        sections: Vec<MarkdownSection>,
        document: &Document,
    ) -> CoreResult<Vec<Node>> {
        let mut nodes = Vec::new();

        for (i, section) in sections.into_iter().enumerate() {
            let chunk_info =
                cheungfun_core::types::ChunkInfo::new(section.start_pos, section.end_pos, i);

            let mut node = Node::new(section.content, document.id, chunk_info);

            // Copy metadata from document
            node.metadata = document.metadata.clone();

            // Add markdown-specific metadata
            if self.config.preserve_header_hierarchy && !section.header_path.is_empty() {
                node.metadata
                    .insert("header_path".to_string(), section.header_path.into());
            }

            if !section.header.is_empty() {
                node.metadata
                    .insert("header".to_string(), section.header.into());
                node.metadata
                    .insert("header_level".to_string(), section.level.into());
            }

            node.metadata.insert("section_index".to_string(), i.into());
            node.metadata
                .insert("parser_type".to_string(), "markdown".into());

            nodes.push(node);
        }

        debug!("Generated {} nodes from markdown sections", nodes.len());
        Ok(nodes)
    }

    /// Split text into sections (for TextSplitter interface).
    fn split_into_sections(&self, text: &str) -> CoreResult<Vec<String>> {
        let sections = self.parse_markdown_sections(text)?;
        Ok(sections.into_iter().map(|s| s.content).collect())
    }
}

impl Default for MarkdownNodeParser {
    fn default() -> Self {
        Self::new()
    }
}

impl TextSplitter for MarkdownNodeParser {
    fn split_text(&self, text: &str) -> CoreResult<Vec<String>> {
        self.split_into_sections(text)
    }
}

#[async_trait]
impl NodeParser for MarkdownNodeParser {
    async fn parse_nodes(
        &self,
        documents: &[Document],
        _show_progress: bool,
    ) -> CoreResult<Vec<Node>> {
        let mut all_nodes = Vec::new();

        for document in documents {
            let sections = self.parse_markdown_sections(&document.content)?;
            let nodes = self.build_nodes_from_sections(sections, document)?;
            all_nodes.extend(nodes);
        }

        Ok(all_nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::Document;

    #[tokio::test]
    async fn test_markdown_basic_parsing() {
        let parser = MarkdownNodeParser::new();

        let markdown = r#"# Introduction
This is the introduction section.

## Getting Started
Here's how to get started.

### Installation
Run the following command to install.

## Configuration
This section covers configuration.
"#;

        let document = Document::new(markdown);
        let nodes = <MarkdownNodeParser as NodeParser>::parse_nodes(&parser, &[document], false)
            .await
            .unwrap();

        assert_eq!(nodes.len(), 4);

        // Check first node (Introduction)
        assert!(nodes[0]
            .content
            .contains("This is the introduction section"));
        assert_eq!(
            nodes[0].metadata.get("header").unwrap().as_str().unwrap(),
            "Introduction"
        );
        assert_eq!(
            nodes[0]
                .metadata
                .get("header_level")
                .unwrap()
                .as_u64()
                .unwrap(),
            1
        );
        assert_eq!(
            nodes[0]
                .metadata
                .get("header_path")
                .unwrap()
                .as_str()
                .unwrap(),
            "/Introduction/"
        );

        // Check second node (Getting Started)
        assert!(nodes[1].content.contains("Here's how to get started"));
        assert_eq!(
            nodes[1].metadata.get("header").unwrap().as_str().unwrap(),
            "Getting Started"
        );
        assert_eq!(
            nodes[1]
                .metadata
                .get("header_level")
                .unwrap()
                .as_u64()
                .unwrap(),
            2
        );
        assert_eq!(
            nodes[1]
                .metadata
                .get("header_path")
                .unwrap()
                .as_str()
                .unwrap(),
            "/Introduction/Getting Started/"
        );

        // Check third node (Installation)
        assert!(nodes[2].content.contains("Run the following command"));
        assert_eq!(
            nodes[2].metadata.get("header").unwrap().as_str().unwrap(),
            "Installation"
        );
        assert_eq!(
            nodes[2]
                .metadata
                .get("header_level")
                .unwrap()
                .as_u64()
                .unwrap(),
            3
        );
        assert_eq!(
            nodes[2]
                .metadata
                .get("header_path")
                .unwrap()
                .as_str()
                .unwrap(),
            "/Introduction/Getting Started/Installation/"
        );

        // Check fourth node (Configuration)
        assert!(nodes[3]
            .content
            .contains("This section covers configuration"));
        assert_eq!(
            nodes[3].metadata.get("header").unwrap().as_str().unwrap(),
            "Configuration"
        );
        assert_eq!(
            nodes[3]
                .metadata
                .get("header_level")
                .unwrap()
                .as_u64()
                .unwrap(),
            2
        );
        assert_eq!(
            nodes[3]
                .metadata
                .get("header_path")
                .unwrap()
                .as_str()
                .unwrap(),
            "/Introduction/Configuration/"
        );
    }

    #[tokio::test]
    async fn test_markdown_max_header_depth() {
        let parser = MarkdownNodeParser::new().with_max_header_depth(2);

        let markdown = r#"# Level 1
Content 1

## Level 2
Content 2

### Level 3
This should be treated as regular text.

#### Level 4
This should also be treated as regular text.
"#;

        let document = Document::new(markdown);
        let nodes = <MarkdownNodeParser as NodeParser>::parse_nodes(&parser, &[document], false)
            .await
            .unwrap();

        // Should only have 2 nodes (Level 1 and Level 2)
        // Level 3 and 4 headers should be included in Level 2 content
        assert_eq!(nodes.len(), 2);

        assert_eq!(
            nodes[0].metadata.get("header").unwrap().as_str().unwrap(),
            "Level 1"
        );
        assert_eq!(
            nodes[1].metadata.get("header").unwrap().as_str().unwrap(),
            "Level 2"
        );

        // Level 2 content should include the Level 3 and 4 headers as regular text
        assert!(nodes[1].content.contains("### Level 3"));
        assert!(nodes[1].content.contains("#### Level 4"));
    }

    #[tokio::test]
    async fn test_markdown_without_header_in_content() {
        let parser = MarkdownNodeParser::new().with_include_header_in_content(false);

        let markdown = r#"# Header 1
Content under header 1.

## Header 2
Content under header 2.
"#;

        let document = Document::new(markdown);
        let nodes = <MarkdownNodeParser as NodeParser>::parse_nodes(&parser, &[document], false)
            .await
            .unwrap();

        assert_eq!(nodes.len(), 2);

        // Content should not include the header lines
        assert!(!nodes[0].content.contains("# Header 1"));
        assert!(nodes[0].content.contains("Content under header 1"));

        assert!(!nodes[1].content.contains("## Header 2"));
        assert!(nodes[1].content.contains("Content under header 2"));
    }

    #[tokio::test]
    async fn test_markdown_custom_separator() {
        let parser = MarkdownNodeParser::new().with_header_path_separator(" > ");

        let markdown = r#"# Parent
Parent content.

## Child
Child content.
"#;

        let document = Document::new(markdown);
        let nodes = <MarkdownNodeParser as NodeParser>::parse_nodes(&parser, &[document], false)
            .await
            .unwrap();

        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes[0]
                .metadata
                .get("header_path")
                .unwrap()
                .as_str()
                .unwrap(),
            " > Parent > "
        );
        assert_eq!(
            nodes[1]
                .metadata
                .get("header_path")
                .unwrap()
                .as_str()
                .unwrap(),
            " > Parent > Child > "
        );
    }

    #[tokio::test]
    async fn test_markdown_min_section_length() {
        let parser = MarkdownNodeParser::new().with_min_section_length(20);

        let markdown = r#"# Long Section
This is a long section with enough content to meet the minimum length requirement.

## Short
Short.

# Another Long Section
This is another long section with sufficient content.
"#;

        let document = Document::new(markdown);
        let nodes = <MarkdownNodeParser as NodeParser>::parse_nodes(&parser, &[document], false)
            .await
            .unwrap();

        // Should only have 2 nodes (the long sections)
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes[0].metadata.get("header").unwrap().as_str().unwrap(),
            "Long Section"
        );
        assert_eq!(
            nodes[1].metadata.get("header").unwrap().as_str().unwrap(),
            "Another Long Section"
        );
    }

    #[tokio::test]
    async fn test_markdown_empty_content() {
        let parser = MarkdownNodeParser::new();

        let document = Document::new("");
        let nodes = <MarkdownNodeParser as NodeParser>::parse_nodes(&parser, &[document], false)
            .await
            .unwrap();

        assert!(nodes.is_empty());
    }

    #[tokio::test]
    async fn test_markdown_no_headers() {
        let parser = MarkdownNodeParser::new();

        let markdown = r#"This is just regular text without any headers.

It has multiple paragraphs but no markdown headers.

So it should be treated as a single section.
"#;

        let document = Document::new(markdown);
        let nodes = <MarkdownNodeParser as NodeParser>::parse_nodes(&parser, &[document], false)
            .await
            .unwrap();

        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].content.contains("This is just regular text"));
        assert!(nodes[0]
            .content
            .contains("So it should be treated as a single section"));

        // Should not have header metadata
        assert!(!nodes[0].metadata.contains_key("header"));
        assert!(!nodes[0].metadata.contains_key("header_level"));
        assert!(!nodes[0].metadata.contains_key("header_path"));
    }

    #[tokio::test]
    async fn test_text_splitter_interface() {
        let parser = MarkdownNodeParser::new();

        let markdown = r#"# Section 1
Content 1

## Section 2
Content 2
"#;

        let sections = parser.split_text(markdown).unwrap();
        assert_eq!(sections.len(), 2);
        assert!(sections[0].contains("Content 1"));
        assert!(sections[1].contains("Content 2"));
    }

    #[tokio::test]
    async fn test_markdown_preset_configs() {
        // Test documentation config
        let doc_parser =
            MarkdownNodeParser::from_config(MarkdownConfig::for_documentation()).unwrap();
        assert_eq!(doc_parser.config.max_header_depth, 4);
        assert_eq!(doc_parser.config.min_section_length, 50);

        // Test blog config
        let blog_parser =
            MarkdownNodeParser::from_config(MarkdownConfig::for_blog_posts()).unwrap();
        assert_eq!(blog_parser.config.header_path_separator, " > ");
        assert_eq!(blog_parser.config.max_header_depth, 3);

        // Test README config
        let readme_parser = MarkdownNodeParser::from_config(MarkdownConfig::for_readme()).unwrap();
        assert_eq!(readme_parser.config.max_header_depth, 6);
        assert_eq!(readme_parser.config.min_section_length, 20);
    }
}
