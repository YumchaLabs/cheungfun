//! Node relationship management system.
//!
//! This module provides a structured system for managing relationships between nodes,
//! similar to LlamaIndex's NodeRelationship and RelatedNodeInfo system.
//!
//! **Reference**: LlamaIndex schema.py L206-L259

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Node relationship types.
///
/// **Reference**: LlamaIndex NodeRelationship enum
/// - File: `llama-index-core/llama_index/core/schema.py`
/// - Lines: L206-L223
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeRelationship {
    /// The node is the source document.
    Source,
    /// The node is the previous node in the document.
    Previous,
    /// The node is the next node in the document.
    Next,
    /// The node is the parent node in the document.
    Parent,
    /// The node is a child node in the document.
    Child,
}

impl std::fmt::Display for NodeRelationship {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeRelationship::Source => write!(f, "SOURCE"),
            NodeRelationship::Previous => write!(f, "PREVIOUS"),
            NodeRelationship::Next => write!(f, "NEXT"),
            NodeRelationship::Parent => write!(f, "PARENT"),
            NodeRelationship::Child => write!(f, "CHILD"),
        }
    }
}

/// Information about a related node.
///
/// **Reference**: LlamaIndex RelatedNodeInfo class
/// - File: `llama-index-core/llama_index/core/schema.py`
/// - Lines: L248-L256
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedNodeInfo {
    /// ID of the related node.
    pub node_id: Uuid,
    /// Type of the related node (optional).
    pub node_type: Option<String>,
    /// Additional metadata for the relationship.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Hash of the related node (optional).
    pub hash: Option<String>,
}

impl RelatedNodeInfo {
    /// Create a new RelatedNodeInfo.
    pub fn new(node_id: Uuid) -> Self {
        Self {
            node_id,
            node_type: None,
            metadata: HashMap::new(),
            hash: None,
        }
    }

    /// Create a RelatedNodeInfo with type.
    pub fn with_type(node_id: Uuid, node_type: String) -> Self {
        Self {
            node_id,
            node_type: Some(node_type),
            metadata: HashMap::new(),
            hash: None,
        }
    }

    /// Add metadata to the relationship.
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set the hash of the related node.
    pub fn with_hash(mut self, hash: String) -> Self {
        self.hash = Some(hash);
        self
    }
}

/// Type for relationship values (single or multiple).
///
/// **Reference**: LlamaIndex RelatedNodeType
/// - File: `llama-index-core/llama_index/core/schema.py`
/// - Line: L259
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelatedNodeType {
    /// Single related node.
    Single(RelatedNodeInfo),
    /// Multiple related nodes (for CHILD relationships).
    Multiple(Vec<RelatedNodeInfo>),
}

impl RelatedNodeType {
    /// Create a single relationship.
    pub fn single(info: RelatedNodeInfo) -> Self {
        RelatedNodeType::Single(info)
    }

    /// Create a multiple relationship.
    pub fn multiple(infos: Vec<RelatedNodeInfo>) -> Self {
        RelatedNodeType::Multiple(infos)
    }

    /// Get as single RelatedNodeInfo (returns None if multiple).
    pub fn as_single(&self) -> Option<&RelatedNodeInfo> {
        match self {
            RelatedNodeType::Single(info) => Some(info),
            RelatedNodeType::Multiple(_) => None,
        }
    }

    /// Get as multiple RelatedNodeInfo (returns None if single).
    pub fn as_multiple(&self) -> Option<&Vec<RelatedNodeInfo>> {
        match self {
            RelatedNodeType::Single(_) => None,
            RelatedNodeType::Multiple(infos) => Some(infos),
        }
    }

    /// Add a child to multiple relationship (converts single to multiple if needed).
    pub fn add_child(&mut self, info: RelatedNodeInfo) {
        match self {
            RelatedNodeType::Single(existing) => {
                // Convert to multiple
                let existing = existing.clone();
                *self = RelatedNodeType::Multiple(vec![existing, info]);
            }
            RelatedNodeType::Multiple(infos) => {
                infos.push(info);
            }
        }
    }
}

/// Node relationships manager.
///
/// **Reference**: LlamaIndex BaseNode.relationships field
/// - File: `llama-index-core/llama_index/core/schema.py`
/// - Lines: L301-L307
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeRelationships {
    relationships: HashMap<NodeRelationship, RelatedNodeType>,
}

impl NodeRelationships {
    /// Create a new empty relationships manager.
    pub fn new() -> Self {
        Self {
            relationships: HashMap::new(),
        }
    }

    /// Set a single relationship.
    pub fn set_single(&mut self, rel_type: NodeRelationship, info: RelatedNodeInfo) {
        self.relationships
            .insert(rel_type, RelatedNodeType::Single(info));
    }

    /// Set multiple relationships.
    pub fn set_multiple(&mut self, rel_type: NodeRelationship, infos: Vec<RelatedNodeInfo>) {
        self.relationships
            .insert(rel_type, RelatedNodeType::Multiple(infos));
    }

    /// Get a relationship.
    pub fn get(&self, rel_type: &NodeRelationship) -> Option<&RelatedNodeType> {
        self.relationships.get(rel_type)
    }

    /// Get source node.
    pub fn source_node(&self) -> Option<&RelatedNodeInfo> {
        self.get(&NodeRelationship::Source)?.as_single()
    }

    /// Get parent node.
    pub fn parent_node(&self) -> Option<&RelatedNodeInfo> {
        self.get(&NodeRelationship::Parent)?.as_single()
    }

    /// Get previous node.
    pub fn prev_node(&self) -> Option<&RelatedNodeInfo> {
        self.get(&NodeRelationship::Previous)?.as_single()
    }

    /// Get next node.
    pub fn next_node(&self) -> Option<&RelatedNodeInfo> {
        self.get(&NodeRelationship::Next)?.as_single()
    }

    /// Get child nodes.
    pub fn child_nodes(&self) -> Option<&Vec<RelatedNodeInfo>> {
        self.get(&NodeRelationship::Child)?.as_multiple()
    }

    /// Add a child node.
    pub fn add_child(&mut self, info: RelatedNodeInfo) {
        match self.relationships.get_mut(&NodeRelationship::Child) {
            Some(existing) => existing.add_child(info),
            None => {
                self.relationships.insert(
                    NodeRelationship::Child,
                    RelatedNodeType::Multiple(vec![info]),
                );
            }
        }
    }

    /// Remove a relationship.
    pub fn remove(&mut self, rel_type: &NodeRelationship) -> Option<RelatedNodeType> {
        self.relationships.remove(rel_type)
    }

    /// Check if a relationship exists.
    pub fn has_relationship(&self, rel_type: &NodeRelationship) -> bool {
        self.relationships.contains_key(rel_type)
    }

    /// Get all relationship types.
    pub fn relationship_types(&self) -> Vec<&NodeRelationship> {
        self.relationships.keys().collect()
    }

    /// Clear all relationships.
    pub fn clear(&mut self) {
        self.relationships.clear();
    }
}

/// Utility functions for managing node relationships.
///
/// **Reference**: LlamaIndex _add_parent_child_relationship function
/// - File: `llama-index-core/llama_index/core/node_parser/relational/hierarchical.py`
/// - Lines: L14-L22
pub mod utils {
    use super::*;
    use crate::Node;

    /// Add parent-child relationship between nodes.
    ///
    /// **Reference**: LlamaIndex _add_parent_child_relationship
    pub fn add_parent_child_relationship(parent: &mut Node, child: &mut Node) {
        // For now, use the existing relationships HashMap
        // Add parent relationship to child
        child.relationships.insert("parent".to_string(), parent.id);

        // Add child relationship to parent (we'll use metadata for child list)
        parent.relationships.insert("child".to_string(), child.id);

        // Store parent_id in child metadata for backward compatibility
        child.metadata.insert(
            "parent_id".to_string(),
            serde_json::Value::String(parent.id.to_string()),
        );
    }

    /// Add previous-next relationship between nodes.
    pub fn add_prev_next_relationship(prev: &mut Node, next: &mut Node) {
        // Add next relationship to previous node
        prev.relationships.insert("next".to_string(), next.id);

        // Add previous relationship to next node
        next.relationships.insert("previous".to_string(), prev.id);
    }

    /// Get leaf nodes (nodes without children).
    ///
    /// **Reference**: LlamaIndex get_leaf_nodes function
    /// - File: `llama-index-core/llama_index/core/node_parser/relational/hierarchical.py`
    /// - Lines: L25-L31
    pub fn get_leaf_nodes(nodes: &[Node]) -> Vec<&Node> {
        nodes
            .iter()
            .filter(|node| !node.relationships.contains_key("child"))
            .collect()
    }

    /// Get root nodes (nodes without parents).
    pub fn get_root_nodes(nodes: &[Node]) -> Vec<&Node> {
        nodes
            .iter()
            .filter(|node| !node.relationships.contains_key("parent"))
            .collect()
    }
}
