//! Hierarchical node parser implementation.
//!
//! This module provides hierarchical document parsing that creates multiple levels
//! of nodes with parent-child relationships, enabling advanced retrieval patterns
//! like auto-merging when multiple child nodes are retrieved.

use crate::node_parser::{
    config::HierarchicalConfig, NodeParser, TextSplitter,
};
use async_trait::async_trait;
use cheungfun_core::{
    traits::{Transform, TransformInput},
    Document, Node, Result as CoreResult,
};
use std::collections::HashMap;
use tracing::{debug, info};
use uuid::Uuid;

/// Hierarchical node parser that creates multiple levels of nodes.
///
/// This parser creates a hierarchy of nodes with different chunk sizes,
/// where larger chunks become parents of smaller chunks. This enables
/// advanced retrieval patterns where retrieving multiple child nodes
/// can trigger auto-merging with their parent for better context.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::node_parser::relational::HierarchicalNodeParser;
/// use cheungfun_core::Document;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let parser = HierarchicalNodeParser::from_defaults(vec![2048, 512, 128])?;
///     
///     let document = Document::new("Long document content...");
///     let nodes = parser.parse_nodes(&[document], false).await?;
///     
///     println!("Generated {} hierarchical nodes", nodes.len());
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct HierarchicalNodeParser {
    /// Configuration for the hierarchical parser.
    config: HierarchicalConfig,
    /// Text splitters for each level (ordered from largest to smallest).
    splitters: Vec<Box<dyn TextSplitter>>,
}

impl HierarchicalNodeParser {
    /// Create a new hierarchical node parser with the given configuration.
    pub fn new(config: HierarchicalConfig) -> CoreResult<Self> {
        let splitters = Self::create_splitters(&config)?;
        
        Ok(Self {
            config,
            splitters,
        })
    }

    /// Create a hierarchical parser with default settings.
    pub fn from_defaults(chunk_sizes: Vec<usize>) -> CoreResult<Self> {
        let config = HierarchicalConfig::new(chunk_sizes);
        Self::new(config)
    }

    /// Create text splitters for each hierarchy level.
    fn create_splitters(config: &HierarchicalConfig) -> CoreResult<Vec<Box<dyn TextSplitter>>> {
        use crate::node_parser::text::SentenceSplitter;
        
        let mut splitters: Vec<Box<dyn TextSplitter>> = Vec::new();
        
        for &chunk_size in &config.chunk_sizes {
            let splitter = SentenceSplitter::from_defaults(chunk_size, config.chunk_overlap)?;
            splitters.push(Box::new(splitter));
        }
        
        Ok(splitters)
    }

    /// Recursively create nodes from documents at each hierarchy level.
    fn create_hierarchical_nodes<'a>(
        &'a self,
        documents: &'a [Document],
        level: usize,
        parent_nodes: Option<&'a [Node]>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = CoreResult<Vec<Node>>> + Send + 'a>> {
        Box::pin(async move {
        if level >= self.splitters.len() {
            return Ok(Vec::new());
        }

        debug!("Creating nodes at hierarchy level {}", level);

        // Get the appropriate splitter for this level
        let splitter = &self.splitters[level];
        
        // Create nodes at this level
        let current_nodes = if let Some(parents) = parent_nodes {
            // Split parent nodes into smaller chunks
            let mut all_nodes = Vec::new();
            for parent in parents {
                let parent_doc = Document::new(&parent.content);
                let child_nodes = TextSplitter::parse_nodes(&**splitter, &[parent_doc], false).await?;
                
                // Set up parent-child relationships
                for mut child in child_nodes {
                    child.metadata.insert(
                        "parent_id".to_string(),
                        parent.id.to_string().into(),
                    );
                    all_nodes.push(child);
                }
            }
            all_nodes
        } else {
            // First level - split documents directly
            TextSplitter::parse_nodes(&**splitter, documents, false).await?
        };

        info!("Created {} nodes at level {}", current_nodes.len(), level);

        // Recursively create nodes for the next level
        let mut all_nodes = current_nodes.clone();
        if level + 1 < self.splitters.len() {
            let child_nodes = self.create_hierarchical_nodes(
                documents,
                level + 1,
                Some(&current_nodes),
            ).await?;
            all_nodes.extend(child_nodes);
        }

        Ok(all_nodes)
        })
    }
}

#[async_trait]
impl NodeParser for HierarchicalNodeParser {
    async fn parse_nodes(
        &self,
        documents: &[Document],
        show_progress: bool,
    ) -> CoreResult<Vec<Node>> {
        if show_progress {
            info!("Starting hierarchical parsing of {} documents", documents.len());
        }

        let nodes = self.create_hierarchical_nodes(documents, 0, None).await?;

        if show_progress {
            info!("Hierarchical parsing completed: {} total nodes", nodes.len());
        }

        Ok(nodes)
    }
}

#[async_trait]
impl Transform for HierarchicalNodeParser {
    async fn transform(&self, input: TransformInput) -> CoreResult<Vec<Node>> {
        match input {
            TransformInput::Documents(documents) => {
                self.parse_nodes(&documents, false).await
            }
            TransformInput::Document(document) => {
                self.parse_nodes(&[document], false).await
            }
            TransformInput::Node(node) => {
                // Convert single node back to document and re-parse hierarchically
                let document = Document::new(&node.content);
                self.parse_nodes(&[document], false).await
            }
            TransformInput::Nodes(nodes) => {
                // Convert nodes back to documents and re-parse hierarchically
                let documents: Vec<Document> = nodes
                    .into_iter()
                    .map(|node| Document::new(&node.content))
                    .collect();
                self.parse_nodes(&documents, false).await
            }
        }
    }
}

/// Utility functions for working with hierarchical nodes.

/// Get root nodes (nodes without parents) from a list of nodes.
pub fn get_root_nodes(nodes: &[Node]) -> Vec<&Node> {
    nodes
        .iter()
        .filter(|node| !node.metadata.contains_key("parent_id"))
        .collect()
}

/// Get leaf nodes (nodes without children) from a list of nodes.
pub fn get_leaf_nodes(nodes: &[Node]) -> Vec<&Node> {
    let parent_ids: std::collections::HashSet<String> = nodes
        .iter()
        .filter_map(|node| {
            node.metadata.get("parent_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .collect();

    nodes
        .iter()
        .filter(|node| !parent_ids.contains(&node.id.to_string()))
        .collect()
}

/// Get child nodes of a specific parent node.
pub fn get_child_nodes<'a>(nodes: &'a [Node], parent_id: &str) -> Vec<&'a Node> {
    nodes
        .iter()
        .filter(|node| {
            node.metadata
                .get("parent_id")
                .and_then(|v| v.as_str())
                .map_or(false, |pid| pid == parent_id)
        })
        .collect()
}

/// Get nodes at deeper levels (higher hierarchy levels).
pub fn get_deeper_nodes(nodes: &[Node], max_depth: usize) -> Vec<&Node> {
    // This is a simplified implementation
    // In practice, you'd track depth during parsing
    nodes
        .iter()
        .filter(|node| {
            // Count the depth by following parent relationships
            let mut depth = 0;
            let mut current_id = node.id.to_string();

            while depth < max_depth {
                if let Some(parent_node) = nodes.iter().find(|n| n.id.to_string() == current_id) {
                    if let Some(parent_id) = parent_node.metadata.get("parent_id")
                        .and_then(|v| v.as_str()) {
                        current_id = parent_id.to_string();
                        depth += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            depth >= max_depth
        })
        .collect()
}
