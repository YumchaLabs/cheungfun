//! JSON node parser implementation.
//!
//! This module provides JSON-specific document parsing that extracts structured
//! data from JSON documents and creates nodes with hierarchical metadata.

use crate::node_parser::NodeParser;
use async_trait::async_trait;
use cheungfun_core::{
    traits::{DocumentState, NodeState, TypedData, TypedTransform},
    Document, Node, Result as CoreResult,
};
use serde_json::{Map, Value};
use tracing::debug;

/// JSON node parser that extracts structured data from JSON documents.
///
/// This parser traverses JSON structures and creates nodes for each object
/// or array, preserving the hierarchical structure in metadata.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::node_parser::file::JSONNodeParser;
/// use cheungfun_core::Document;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let parser = JSONNodeParser::new();
///     
///     let json_content = r#"
///         {
///             "users": [
///                 {"name": "Alice", "age": 30},
///                 {"name": "Bob", "age": 25}
///             ],
///             "metadata": {
///                 "version": "1.0",
///                 "created": "2024-01-01"
///             }
///         }
///     "#;
///     
///     let document = Document::new(json_content);
///     let nodes = parser.parse_nodes(&[document], false).await?;
///     
///     println!("Generated {} nodes from JSON", nodes.len());
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct JSONNodeParser {
    /// Configuration for the JSON parser.
    config: JSONConfig,
}

impl JSONNodeParser {
    /// Create a new JSON node parser with default configuration.
    pub fn new() -> Self {
        Self {
            config: JSONConfig::default(),
        }
    }

    /// Create a JSON parser with custom configuration.
    pub fn with_config(config: JSONConfig) -> Self {
        Self { config }
    }

    /// Extract structured data from JSON document.
    fn extract_json_sections(&self, json_content: &str) -> CoreResult<Vec<String>> {
        let value: Value = serde_json::from_str(json_content).map_err(|e| {
            cheungfun_core::CheungfunError::Internal {
                message: format!("Failed to parse JSON: {}", e),
            }
        })?;

        let mut sections = Vec::new();
        self.traverse_json_value(&value, &mut Vec::new(), &mut sections)?;

        debug!("Extracted {} sections from JSON", sections.len());
        Ok(sections)
    }

    /// Recursively traverse JSON value and extract sections.
    fn traverse_json_value(
        &self,
        value: &Value,
        path: &mut Vec<String>,
        sections: &mut Vec<String>,
    ) -> CoreResult<()> {
        match value {
            Value::Object(obj) => {
                if self.config.create_nodes_for_objects {
                    let lines = self.object_to_lines(obj, path)?;
                    if !lines.is_empty() {
                        sections.push(lines.join("\n"));
                    }
                }

                // Recursively process nested objects and arrays
                for (key, nested_value) in obj {
                    path.push(key.clone());
                    self.traverse_json_value(nested_value, path, sections)?;
                    path.pop();
                }
            }
            Value::Array(arr) => {
                if self.config.create_nodes_for_arrays {
                    for (index, item) in arr.iter().enumerate() {
                        path.push(index.to_string());
                        self.traverse_json_value(item, path, sections)?;
                        path.pop();
                    }
                }
            }
            _ => {
                // For primitive values, we might want to include them in parent objects
                // This is handled in object_to_lines
            }
        }

        Ok(())
    }

    /// Convert JSON object to lines of text.
    fn object_to_lines(
        &self,
        obj: &Map<String, Value>,
        path: &[String],
    ) -> CoreResult<Vec<String>> {
        let mut lines = Vec::new();

        if !path.is_empty() && self.config.include_path_in_content {
            lines.push(format!("Path: {}", path.join(".")));
        }

        for (key, value) in obj {
            match value {
                Value::String(s) => lines.push(format!("{}: {}", key, s)),
                Value::Number(n) => lines.push(format!("{}: {}", key, n)),
                Value::Bool(b) => lines.push(format!("{}: {}", key, b)),
                Value::Null => lines.push(format!("{}: null", key)),
                Value::Object(_) => {
                    if self.config.include_nested_objects {
                        lines.push(format!("{}: [Object]", key));
                    }
                }
                Value::Array(arr) => {
                    if self.config.include_array_summaries {
                        lines.push(format!("{}: [Array with {} items]", key, arr.len()));
                    }
                }
            }
        }

        Ok(lines)
    }

    /// Build nodes from JSON sections.
    fn build_nodes_from_json_sections(
        &self,
        sections: Vec<String>,
        document: &Document,
    ) -> CoreResult<Vec<Node>> {
        let mut nodes = Vec::new();

        for (i, content) in sections.into_iter().enumerate() {
            if content.len() < self.config.min_section_length {
                continue;
            }

            let chunk_info =
                cheungfun_core::types::ChunkInfo::with_char_indices(0, content.len(), i);
            let mut node = Node::new(content, document.id, chunk_info);

            // Copy metadata from document
            node.metadata = document.metadata.clone();

            // Add JSON-specific metadata
            node.metadata
                .insert("section_type".to_string(), "json_object".into());
            node.metadata.insert("section_index".to_string(), i.into());

            nodes.push(node);
        }

        Ok(nodes)
    }
}

impl Default for JSONNodeParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl NodeParser for JSONNodeParser {
    async fn parse_nodes(
        &self,
        documents: &[Document],
        _show_progress: bool,
    ) -> CoreResult<Vec<Node>> {
        let mut all_nodes = Vec::new();

        for document in documents {
            let sections = self.extract_json_sections(&document.content)?;
            let nodes = self.build_nodes_from_json_sections(sections, document)?;
            all_nodes.extend(nodes);
        }

        Ok(all_nodes)
    }
}

// ============================================================================
// Type-Safe Transform Implementation
// ============================================================================

#[async_trait]
impl TypedTransform<DocumentState, NodeState> for JSONNodeParser {
    async fn transform(&self, input: TypedData<DocumentState>) -> CoreResult<TypedData<NodeState>> {
        let documents = input.documents();
        let nodes = self.parse_nodes(documents, false).await?;
        Ok(TypedData::from_nodes(nodes))
    }

    fn name(&self) -> &'static str {
        "JSONNodeParser"
    }

    fn description(&self) -> &'static str {
        "Extracts structured data from JSON documents while preserving hierarchical metadata"
    }
}

// Legacy Transform implementation has been removed.
// JSONNodeParser now only uses the type-safe TypedTransform system.

/// Configuration for JSON node parser.
#[derive(Debug, Clone)]
pub struct JSONConfig {
    /// Whether to create nodes for JSON objects.
    pub create_nodes_for_objects: bool,
    /// Whether to create nodes for JSON arrays.
    pub create_nodes_for_arrays: bool,
    /// Whether to include the JSON path in node content.
    pub include_path_in_content: bool,
    /// Whether to include nested objects in summaries.
    pub include_nested_objects: bool,
    /// Whether to include array summaries.
    pub include_array_summaries: bool,
    /// Minimum section length to create a node.
    pub min_section_length: usize,
    /// Maximum depth to traverse.
    pub max_depth: Option<usize>,
}

impl Default for JSONConfig {
    fn default() -> Self {
        Self {
            create_nodes_for_objects: true,
            create_nodes_for_arrays: true,
            include_path_in_content: true,
            include_nested_objects: true,
            include_array_summaries: true,
            min_section_length: 10,
            max_depth: None,
        }
    }
}

impl JSONConfig {
    /// Set whether to create nodes for objects.
    pub fn with_create_object_nodes(mut self, create: bool) -> Self {
        self.create_nodes_for_objects = create;
        self
    }

    /// Set whether to create nodes for arrays.
    pub fn with_create_array_nodes(mut self, create: bool) -> Self {
        self.create_nodes_for_arrays = create;
        self
    }

    /// Set whether to include paths in content.
    pub fn with_include_paths(mut self, include: bool) -> Self {
        self.include_path_in_content = include;
        self
    }

    /// Set the minimum section length.
    pub fn with_min_section_length(mut self, length: usize) -> Self {
        self.min_section_length = length;
        self
    }

    /// Set the maximum traversal depth.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }
}
