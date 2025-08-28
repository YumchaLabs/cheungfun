//! HTML node parser implementation.
//!
//! This module provides HTML-specific document parsing that extracts text
//! from HTML tags while preserving structure and metadata.

use crate::node_parser::{utils::build_nodes_from_splits, NodeParser};
use async_trait::async_trait;
use cheungfun_core::{
    traits::{Transform, TransformInput},
    Document, Node, Result as CoreResult,
};
use scraper::{Html, Selector};
use tracing::{debug, warn};

/// HTML node parser that extracts text from HTML documents.
///
/// This parser uses CSS selectors to extract text from specific HTML tags,
/// creating nodes for each section while preserving tag information in metadata.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::node_parser::file::HTMLNodeParser;
/// use cheungfun_core::Document;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let parser = HTMLNodeParser::new();
///     
///     let html_content = r#"
///         <html>
///             <body>
///                 <h1>Title</h1>
///                 <p>First paragraph.</p>
///                 <p>Second paragraph.</p>
///             </body>
///         </html>
///     "#;
///     
///     let document = Document::new(html_content);
///     let nodes = parser.parse_nodes(&[document], false).await?;
///     
///     println!("Generated {} nodes from HTML", nodes.len());
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct HTMLNodeParser {
    /// Configuration for the HTML parser.
    config: HTMLConfig,
}

impl HTMLNodeParser {
    /// Create a new HTML node parser with default configuration.
    pub fn new() -> Self {
        Self {
            config: HTMLConfig::default(),
        }
    }

    /// Create an HTML parser with custom configuration.
    pub fn with_config(config: HTMLConfig) -> Self {
        Self { config }
    }

    /// Create an HTML parser with custom tags.
    pub fn with_tags(tags: Vec<String>) -> Self {
        let config = HTMLConfig::default().with_tags(tags);
        Self { config }
    }

    /// Extract text from HTML document using configured tags.
    fn extract_html_sections(&self, html_content: &str) -> CoreResult<Vec<(String, String)>> {
        let document = Html::parse_document(html_content);
        let mut sections = Vec::new();
        let mut current_tag: Option<String> = None;
        let mut current_content = String::new();

        for tag_name in &self.config.tags {
            let selector = Selector::parse(tag_name).map_err(|e| {
                cheungfun_core::CheungfunError::Internal {
                    message: format!("Invalid CSS selector '{}': {}", tag_name, e),
                }
            })?;

            for element in document.select(&selector) {
                let text = element
                    .text()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .trim()
                    .to_string();

                if text.is_empty() {
                    continue;
                }

                // If we're switching to a different tag type, save the current section
                if let Some(ref current) = current_tag {
                    if current != tag_name && !current_content.is_empty() {
                        sections.push((current.clone(), current_content.trim().to_string()));
                        current_content.clear();
                    }
                }

                current_tag = Some(tag_name.clone());

                if !current_content.is_empty() {
                    current_content.push('\n');
                }
                current_content.push_str(&text);
            }
        }

        // Don't forget the last section
        if let Some(tag) = current_tag {
            if !current_content.is_empty() {
                sections.push((tag, current_content.trim().to_string()));
            }
        }

        debug!("Extracted {} sections from HTML", sections.len());
        Ok(sections)
    }

    /// Build nodes from HTML sections.
    fn build_nodes_from_html_sections(
        &self,
        sections: Vec<(String, String)>,
        document: &Document,
    ) -> CoreResult<Vec<Node>> {
        let mut nodes = Vec::new();

        for (i, (tag, content)) in sections.into_iter().enumerate() {
            if content.len() < self.config.min_section_length {
                continue;
            }

            let chunk_info = cheungfun_core::types::ChunkInfo::new(0, content.len(), i);
            let mut node = Node::new(content, document.id, chunk_info);

            // Copy metadata from document
            node.metadata = document.metadata.clone();

            // Add HTML-specific metadata
            node.metadata.insert("tag".to_string(), tag.into());
            node.metadata.insert("section_index".to_string(), i.into());

            if self.config.preserve_tag_attributes {
                // In a more complete implementation, we'd extract and preserve attributes
                node.metadata
                    .insert("has_attributes".to_string(), false.into());
            }

            nodes.push(node);
        }

        Ok(nodes)
    }
}

impl Default for HTMLNodeParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl NodeParser for HTMLNodeParser {
    async fn parse_nodes(
        &self,
        documents: &[Document],
        _show_progress: bool,
    ) -> CoreResult<Vec<Node>> {
        let mut all_nodes = Vec::new();

        for document in documents {
            let sections = self.extract_html_sections(&document.content)?;
            let nodes = self.build_nodes_from_html_sections(sections, document)?;
            all_nodes.extend(nodes);
        }

        Ok(all_nodes)
    }
}

#[async_trait]
impl Transform for HTMLNodeParser {
    async fn transform(&self, input: TransformInput) -> CoreResult<Vec<Node>> {
        match input {
            TransformInput::Documents(documents) => self.parse_nodes(&documents, false).await,
            TransformInput::Document(document) => self.parse_nodes(&[document], false).await,
            TransformInput::Node(node) => {
                // Convert single node back to document and re-parse
                let document = Document::new(&node.content);
                self.parse_nodes(&[document], false).await
            }
            TransformInput::Nodes(nodes) => {
                // Convert nodes back to documents and re-parse
                let documents: Vec<Document> = nodes
                    .into_iter()
                    .map(|node| Document::new(&node.content))
                    .collect();
                self.parse_nodes(&documents, false).await
            }
        }
    }
}

/// Configuration for HTML node parser.
#[derive(Debug, Clone)]
pub struct HTMLConfig {
    /// HTML tags to extract text from.
    pub tags: Vec<String>,
    /// Minimum section length to create a node.
    pub min_section_length: usize,
    /// Whether to preserve tag attributes in metadata.
    pub preserve_tag_attributes: bool,
    /// Whether to include tag hierarchy in metadata.
    pub include_tag_hierarchy: bool,
}

impl Default for HTMLConfig {
    fn default() -> Self {
        Self {
            tags: vec![
                "p".to_string(),
                "h1".to_string(),
                "h2".to_string(),
                "h3".to_string(),
                "h4".to_string(),
                "h5".to_string(),
                "h6".to_string(),
                "li".to_string(),
                "b".to_string(),
                "i".to_string(),
                "u".to_string(),
                "section".to_string(),
            ],
            min_section_length: 10,
            preserve_tag_attributes: false,
            include_tag_hierarchy: false,
        }
    }
}

impl HTMLConfig {
    /// Set the HTML tags to extract.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set the minimum section length.
    pub fn with_min_section_length(mut self, length: usize) -> Self {
        self.min_section_length = length;
        self
    }

    /// Set whether to preserve tag attributes.
    pub fn with_preserve_attributes(mut self, preserve: bool) -> Self {
        self.preserve_tag_attributes = preserve;
        self
    }

    /// Set whether to include tag hierarchy.
    pub fn with_tag_hierarchy(mut self, include: bool) -> Self {
        self.include_tag_hierarchy = include;
        self
    }
}
