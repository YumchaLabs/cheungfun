//! In-memory labelled property graph implementation.
//!
//! This module provides the core in-memory graph data structure that powers
//! the SimplePropertyGraphStore, following LlamaIndex's LabelledPropertyGraph design exactly.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::graph::{ChunkNode, EntityNode, LabelledNode, Relation, Triplet};
use crate::Result;

/// Enum to represent different types of labelled nodes.
///
/// This enum allows us to store different node types in a single collection
/// while maintaining type safety, avoiding the complexity of trait objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LabelledNodeType {
    /// An entity node representing a person, place, organization, concept, etc.
    Entity(EntityNode),
    /// A chunk node representing a text fragment from a document.
    Chunk(ChunkNode),
}

impl LabelledNodeType {
    /// Get the node ID.
    pub fn id(&self) -> String {
        match self {
            LabelledNodeType::Entity(node) => node.id(),
            LabelledNodeType::Chunk(node) => node.id(),
        }
    }

    /// Get the node label.
    pub fn label(&self) -> &str {
        match self {
            LabelledNodeType::Entity(node) => node.node_label(),
            LabelledNodeType::Chunk(node) => node.node_label(),
        }
    }

    /// Get the node properties.
    pub fn properties(&self) -> &HashMap<String, serde_json::Value> {
        match self {
            LabelledNodeType::Entity(node) => node.node_properties(),
            LabelledNodeType::Chunk(node) => node.node_properties(),
        }
    }

    /// Check if this is an entity node.
    pub fn is_entity(&self) -> bool {
        matches!(self, LabelledNodeType::Entity(_))
    }

    /// Check if this is a chunk node.
    pub fn is_chunk(&self) -> bool {
        matches!(self, LabelledNodeType::Chunk(_))
    }

    /// Get as entity node if possible.
    pub fn as_entity(&self) -> Option<&EntityNode> {
        match self {
            LabelledNodeType::Entity(node) => Some(node),
            _ => None,
        }
    }

    /// Get as chunk node if possible.
    pub fn as_chunk(&self) -> Option<&ChunkNode> {
        match self {
            LabelledNodeType::Chunk(node) => Some(node),
            _ => None,
        }
    }
}

impl From<EntityNode> for LabelledNodeType {
    fn from(node: EntityNode) -> Self {
        LabelledNodeType::Entity(node)
    }
}

impl From<ChunkNode> for LabelledNodeType {
    fn from(node: ChunkNode) -> Self {
        LabelledNodeType::Chunk(node)
    }
}

/// In-memory labelled property graph containing entities and relations.
///
/// This struct provides the core graph data structure that stores nodes, relations,
/// and triplets in memory, following LlamaIndex's LabelledPropertyGraph design exactly.
///
/// # Examples
///
/// ```rust
/// use cheungfun_core::types::labelled_property_graph::LabelledPropertyGraph;
/// use cheungfun_core::types::graph::{EntityNode, Relation};
/// use cheungfun_core::types::LabelledNodeType;
/// use std::collections::HashMap;
///
/// let mut graph = LabelledPropertyGraph::new();
///
/// let entity1 = EntityNode::new("Alice".to_string(), "Person".to_string(), HashMap::new());
/// let entity2 = EntityNode::new("Bob".to_string(), "Person".to_string(), HashMap::new());
/// let relation = Relation::new(
///     "rel_1".to_string(),
///     "knows".to_string(),
///     entity1.id(),
///     entity2.id(),
///     HashMap::new(),
/// );
///
/// graph.add_node(LabelledNodeType::Entity(entity1));
/// graph.add_node(LabelledNodeType::Entity(entity2));
/// graph.add_relation(relation);
///
/// let triplets = graph.get_triplets();
/// assert_eq!(triplets.len(), 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelledPropertyGraph {
    /// All nodes in the graph, indexed by their ID.
    pub nodes: HashMap<String, LabelledNodeType>,

    /// All relations in the graph, indexed by their composite key.
    pub relations: HashMap<String, Relation>,

    /// Set of triplet IDs (subject_id, relation_id, object_id) for efficient lookup.
    pub triplets: HashSet<(String, String, String)>,
}

impl LabelledPropertyGraph {
    /// Create a new empty labelled property graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            relations: HashMap::new(),
            triplets: HashSet::new(),
        }
    }

    /// Get relation key for indexing relations.
    ///
    /// This follows LlamaIndex's pattern: "{source_id}_{relation_label}_{target_id}"
    ///
    /// # Arguments
    ///
    /// * `relation` - The relation to get key for
    /// * `subj_id` - Subject ID (if relation is None)
    /// * `obj_id` - Object ID (if relation is None)  
    /// * `rel_id` - Relation ID (if relation is None)
    fn get_relation_key(
        &self,
        relation: Option<&Relation>,
        subj_id: Option<&str>,
        obj_id: Option<&str>,
        rel_id: Option<&str>,
    ) -> String {
        if let Some(rel) = relation {
            format!("{}_{}_{}", rel.source_id, rel.label, rel.target_id)
        } else if let (Some(subj), Some(obj), Some(rel)) = (subj_id, obj_id, rel_id) {
            format!("{}_{}_{}", subj, rel, obj)
        } else {
            panic!("Either relation or all three IDs must be provided")
        }
    }

    /// Get all nodes in the graph.
    pub fn get_all_nodes(&self) -> Vec<&LabelledNodeType> {
        self.nodes.values().collect()
    }

    /// Get all relations in the graph.
    pub fn get_all_relations(&self) -> Vec<&Relation> {
        self.relations.values().collect()
    }

    /// Get all triplets in the graph.
    ///
    /// This reconstructs triplets from the stored node and relation data,
    /// following LlamaIndex's exact pattern.
    pub fn get_triplets(&self) -> Vec<Triplet> {
        let mut triplets = Vec::new();

        for (subj_id, rel_id, obj_id) in &self.triplets {
            if let (Some(subj_node), Some(obj_node)) =
                (self.nodes.get(subj_id), self.nodes.get(obj_id))
            {
                // We need to find the relation by matching the triplet components
                // Since we store relations by {source_id}_{label}_{target_id}, we need to find the matching relation
                let rel_key = self
                    .relations
                    .keys()
                    .find(|key| {
                        if let Some(relation) = self.relations.get(*key) {
                            relation.source_id == *subj_id
                                && relation.target_id == *obj_id
                                && relation.id == *rel_id
                        } else {
                            false
                        }
                    })
                    .cloned();
                if let Some(rel_key) = rel_key {
                    if let Some(relation) = self.relations.get(&rel_key) {
                        // Convert nodes to EntityNodes for triplet creation
                        if let (Some(source_entity), Some(target_entity)) = (
                            self.node_to_entity(subj_node),
                            self.node_to_entity(obj_node),
                        ) {
                            triplets.push(Triplet::new(
                                source_entity,
                                relation.clone(),
                                target_entity,
                            ));
                        }
                    }
                }
            }
        }

        triplets
    }

    /// Convert a LabelledNodeType to EntityNode.
    ///
    /// For ChunkNodes, we create a synthetic EntityNode representation.
    fn node_to_entity(&self, node: &LabelledNodeType) -> Option<EntityNode> {
        match node {
            LabelledNodeType::Entity(entity) => Some(entity.clone()),
            LabelledNodeType::Chunk(chunk) => {
                // Convert ChunkNode to EntityNode representation
                let mut properties = chunk.properties.clone();
                properties.insert(
                    "text".to_string(),
                    serde_json::Value::String(chunk.text.clone()),
                );

                Some(EntityNode::with_id(
                    chunk.id(),
                    chunk.text.clone(), // Use text as name
                    chunk.label.clone(),
                    properties,
                ))
            }
        }
    }

    /// Add a triplet to the graph.
    ///
    /// This adds both nodes and the relation, following LlamaIndex's pattern.
    pub fn add_triplet(&mut self, triplet: Triplet) -> Result<()> {
        let subj_id = triplet.source.id();
        let obj_id = triplet.target.id();
        let rel_id = triplet.relation.id.clone();

        // Check if triplet already exists
        if self
            .triplets
            .contains(&(subj_id.clone(), rel_id.clone(), obj_id.clone()))
        {
            return Ok(());
        }

        // Add the triplet
        self.triplets
            .insert((subj_id.clone(), rel_id.clone(), obj_id.clone()));

        // Add nodes
        self.nodes
            .insert(subj_id, LabelledNodeType::Entity(triplet.source));
        self.nodes
            .insert(obj_id, LabelledNodeType::Entity(triplet.target));

        // Add relation
        let rel_key = self.get_relation_key(Some(&triplet.relation), None, None, None);
        self.relations.insert(rel_key, triplet.relation);

        Ok(())
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: LabelledNodeType) -> Result<()> {
        let node_id = node.id();
        self.nodes.insert(node_id, node);
        Ok(())
    }

    /// Add an entity node to the graph.
    pub fn add_entity_node(&mut self, node: EntityNode) -> Result<()> {
        self.add_node(LabelledNodeType::Entity(node))
    }

    /// Add a chunk node to the graph.
    pub fn add_chunk_node(&mut self, node: ChunkNode) -> Result<()> {
        self.add_node(LabelledNodeType::Chunk(node))
    }

    /// Add a relation to the graph.
    ///
    /// This creates the relation but doesn't automatically create a triplet.
    /// The nodes must exist for the relation to be meaningful.
    pub fn add_relation(&mut self, relation: Relation) -> Result<()> {
        // Check if both nodes exist
        if !self.nodes.contains_key(&relation.source_id)
            || !self.nodes.contains_key(&relation.target_id)
        {
            return Err(crate::CheungfunError::Internal {
                message: format!(
                    "Cannot add relation: source '{}' or target '{}' node not found",
                    relation.source_id, relation.target_id
                ),
            });
        }

        let subj_id = relation.source_id.clone();
        let obj_id = relation.target_id.clone();
        let rel_id = relation.id.clone();

        // Add to triplets set
        self.triplets.insert((subj_id, rel_id, obj_id));

        // Add to relations map
        let rel_key = self.get_relation_key(Some(&relation), None, None, None);
        self.relations.insert(rel_key, relation);

        Ok(())
    }

    /// Delete a triplet from the graph.
    pub fn delete_triplet(&mut self, triplet: Triplet) -> Result<()> {
        let subj_id = triplet.source.id();
        let obj_id = triplet.target.id();
        let rel_id = triplet.relation.id.clone();

        // Remove from triplets set
        if !self
            .triplets
            .remove(&(subj_id.clone(), rel_id.clone(), obj_id.clone()))
        {
            return Ok(()); // Triplet didn't exist
        }

        // Remove nodes (following LlamaIndex's aggressive deletion pattern)
        self.nodes.remove(&subj_id);
        self.nodes.remove(&obj_id);

        // Remove relation
        let rel_key = self.get_relation_key(Some(&triplet.relation), None, None, None);
        self.relations.remove(&rel_key);

        Ok(())
    }

    /// Delete a node from the graph by ID.
    pub fn delete_node_by_id(&mut self, node_id: &str) -> Result<()> {
        self.nodes.remove(node_id);
        Ok(())
    }

    /// Delete a node from the graph.
    pub fn delete_node(&mut self, node: &LabelledNodeType) -> Result<()> {
        let node_id = node.id();
        self.nodes.remove(&node_id);
        Ok(())
    }

    /// Delete a relation from the graph.
    pub fn delete_relation(&mut self, relation: &Relation) -> Result<()> {
        let rel_key = self.get_relation_key(Some(relation), None, None, None);
        self.relations.remove(&rel_key);
        Ok(())
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of relations in the graph.
    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }

    /// Get the number of triplets in the graph.
    pub fn triplet_count(&self) -> usize {
        self.triplets.len()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.relations.is_empty() && self.triplets.is_empty()
    }

    /// Clear all data from the graph.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.relations.clear();
        self.triplets.clear();
    }
}

impl Default for LabelledPropertyGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_empty_graph() {
        let graph = LabelledPropertyGraph::new();
        assert!(graph.is_empty());
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.relation_count(), 0);
        assert_eq!(graph.triplet_count(), 0);
    }

    #[test]
    fn test_add_nodes() {
        let mut graph = LabelledPropertyGraph::new();

        let entity1 = EntityNode::new("Alice".to_string(), "Person".to_string(), HashMap::new());
        let entity2 = EntityNode::new("Bob".to_string(), "Person".to_string(), HashMap::new());

        graph.add_node(LabelledNodeType::Entity(entity1)).unwrap();
        graph.add_node(LabelledNodeType::Entity(entity2)).unwrap();

        assert_eq!(graph.node_count(), 2);
        assert!(!graph.is_empty());
    }

    #[test]
    fn test_add_relation() {
        let mut graph = LabelledPropertyGraph::new();

        let entity1 = EntityNode::new("Alice".to_string(), "Person".to_string(), HashMap::new());
        let entity2 = EntityNode::new("Bob".to_string(), "Person".to_string(), HashMap::new());
        let entity1_id = entity1.id();
        let entity2_id = entity2.id();

        graph.add_node(LabelledNodeType::Entity(entity1)).unwrap();
        graph.add_node(LabelledNodeType::Entity(entity2)).unwrap();

        let relation = Relation::new(
            "rel_1".to_string(),
            "knows".to_string(),
            entity1_id,
            entity2_id,
            HashMap::new(),
        );

        graph.add_relation(relation).unwrap();

        assert_eq!(graph.relation_count(), 1);
        assert_eq!(graph.triplet_count(), 1);
    }
}
