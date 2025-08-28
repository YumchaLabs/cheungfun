//! Graph data types for property graph storage and retrieval.
//!
//! This module defines the core data structures for representing property graphs,
//! following LlamaIndex's PropertyGraphStore design patterns exactly. It provides types
//! for entities, relations, chunks, and triplets that form the foundation of Graph RAG.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Base trait for all labeled nodes in a property graph.
///
/// This trait provides the common interface for all node types in the graph,
/// following LlamaIndex's LabelledNode design pattern.
pub trait LabelledNode: Send + Sync + std::fmt::Debug {
    /// Get the unique identifier for this node.
    fn node_id(&self) -> String;

    /// Get the label/type of this node.
    fn node_label(&self) -> &str;

    /// Get the properties of this node.
    fn node_properties(&self) -> &HashMap<String, serde_json::Value>;

    /// Get a mutable reference to the properties.
    fn node_properties_mut(&mut self) -> &mut HashMap<String, serde_json::Value>;

    /// Set a property value.
    fn set_node_property(&mut self, key: String, value: serde_json::Value) {
        self.node_properties_mut().insert(key, value);
    }

    /// Get a property value by key.
    fn get_node_property(&self, key: &str) -> Option<&serde_json::Value> {
        self.node_properties().get(key)
    }
}

/// A labeled node in a property graph representing an entity.
///
/// EntityNode represents entities extracted from documents, such as people,
/// places, organizations, concepts, etc. Each entity has a unique ID, a name,
/// a label (type), and arbitrary properties. This follows LlamaIndex's EntityNode design exactly.
///
/// # Examples
///
/// ```rust
/// use cheungfun_core::types::graph::{EntityNode, LabelledNode};
/// use std::collections::HashMap;
///
/// let mut properties = HashMap::new();
/// properties.insert("age".to_string(), serde_json::Value::Number(30.into()));
/// properties.insert("occupation".to_string(), serde_json::Value::String("Engineer".to_string()));
///
/// let entity = EntityNode::new(
///     "John Doe".to_string(),
///     "Person".to_string(),
///     properties,
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityNode {
    /// Unique identifier for this entity (auto-generated if not provided).
    pub id_: Option<String>,

    /// Human-readable name of the entity.
    pub name: String,

    /// Label/type of the entity (e.g., "Person", "Organization", "Concept").
    pub label: String,

    /// Additional properties/metadata for this entity.
    pub properties: HashMap<String, serde_json::Value>,

    /// Optional embedding vector for this entity.
    pub embedding: Option<Vec<f32>>,
}

impl EntityNode {
    /// Create a new entity node.
    ///
    /// # Arguments
    ///
    /// * `name` - Human-readable name
    /// * `label` - Entity type/label (defaults to "__Entity__" if empty)
    /// * `properties` - Additional metadata
    pub fn new(
        name: String,
        label: String,
        mut properties: HashMap<String, serde_json::Value>,
    ) -> Self {
        let label = if label.is_empty() { "__Entity__".to_string() } else { label };

        // Store name in properties for trait object compatibility
        properties.insert("name".to_string(), serde_json::Value::String(name.clone()));

        Self {
            id_: None,
            name,
            label,
            properties,
            embedding: None,
        }
    }

    /// Create a new entity node with a specific ID.
    pub fn with_id(
        id: String,
        name: String,
        label: String,
        mut properties: HashMap<String, serde_json::Value>,
    ) -> Self {
        let label = if label.is_empty() { "__Entity__".to_string() } else { label };

        // Store name in properties for trait object compatibility
        properties.insert("name".to_string(), serde_json::Value::String(name.clone()));

        Self {
            id_: Some(id),
            name,
            label,
            properties,
            embedding: None,
        }
    }

    /// Create a new entity node with embedding.
    pub fn with_embedding(
        name: String,
        label: String,
        mut properties: HashMap<String, serde_json::Value>,
        embedding: Vec<f32>,
    ) -> Self {
        let label = if label.is_empty() { "__Entity__".to_string() } else { label };

        // Store name in properties for trait object compatibility
        properties.insert("name".to_string(), serde_json::Value::String(name.clone()));

        Self {
            id_: None,
            name,
            label,
            properties,
            embedding: Some(embedding),
        }
    }

    /// Get the unique ID for this entity (auto-generated if not set).
    pub fn id(&self) -> String {
        self.id_.clone().unwrap_or_else(|| {
            // Generate ID based on name hash, similar to LlamaIndex
            format!("{:x}", md5::compute(&self.name))
        })
    }

    /// Check if this entity has an embedding.
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }
}

impl LabelledNode for EntityNode {
    fn node_id(&self) -> String {
        self.id()
    }

    fn node_label(&self) -> &str {
        &self.label
    }

    fn node_properties(&self) -> &HashMap<String, serde_json::Value> {
        &self.properties
    }

    fn node_properties_mut(&mut self) -> &mut HashMap<String, serde_json::Value> {
        &mut self.properties
    }
}

/// A text chunk node in a property graph.
///
/// ChunkNode represents text chunks from documents that are stored in the graph
/// alongside entities. This follows LlamaIndex's ChunkNode design exactly.
///
/// # Examples
///
/// ```rust
/// use cheungfun_core::types::graph::{ChunkNode, LabelledNode};
/// use std::collections::HashMap;
///
/// let chunk = ChunkNode::new(
///     "This is a sample text chunk from a document.".to_string(),
///     HashMap::new(),
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkNode {
    /// The text content of the chunk.
    pub text: String,

    /// Optional unique identifier (auto-generated from text hash if not provided).
    pub id_: Option<String>,

    /// Label for the chunk (defaults to "text_chunk").
    pub label: String,

    /// Additional properties/metadata for this chunk.
    pub properties: HashMap<String, serde_json::Value>,

    /// Optional embedding vector for this chunk.
    pub embedding: Option<Vec<f32>>,
}

impl ChunkNode {
    /// Create a new chunk node.
    ///
    /// # Arguments
    ///
    /// * `text` - The text content of the chunk
    /// * `properties` - Additional metadata
    pub fn new(text: String, properties: HashMap<String, serde_json::Value>) -> Self {
        Self {
            text,
            id_: None,
            label: "text_chunk".to_string(),
            properties,
            embedding: None,
        }
    }

    /// Create a new chunk node with a specific ID.
    pub fn with_id(
        id: String,
        text: String,
        properties: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            text,
            id_: Some(id),
            label: "text_chunk".to_string(),
            properties,
            embedding: None,
        }
    }

    /// Create a new chunk node with embedding.
    pub fn with_embedding(
        text: String,
        properties: HashMap<String, serde_json::Value>,
        embedding: Vec<f32>,
    ) -> Self {
        Self {
            text,
            id_: None,
            label: "text_chunk".to_string(),
            properties,
            embedding: Some(embedding),
        }
    }

    /// Get the unique ID for this chunk (auto-generated if not set).
    pub fn id(&self) -> String {
        self.id_.clone().unwrap_or_else(|| {
            // Generate ID based on text hash, similar to LlamaIndex
            format!("{:x}", md5::compute(&self.text))
        })
    }

    /// Check if this chunk has an embedding.
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }
}

impl LabelledNode for ChunkNode {
    fn node_id(&self) -> String {
        self.id()
    }

    fn node_label(&self) -> &str {
        &self.label
    }

    fn node_properties(&self) -> &HashMap<String, serde_json::Value> {
        &self.properties
    }

    fn node_properties_mut(&mut self) -> &mut HashMap<String, serde_json::Value> {
        &mut self.properties
    }
}

impl std::fmt::Display for ChunkNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text)
    }
}

/// A directed relationship between two entities in a property graph.
///
/// Relation represents the connections between entities, such as "works_at",
/// "located_in", "is_a", etc. Each relation has a label, source and target
/// entity IDs, and optional properties.
///
/// # Examples
///
/// ```rust
/// use cheungfun_core::types::graph::Relation;
/// use std::collections::HashMap;
///
/// let mut properties = HashMap::new();
/// properties.insert("since".to_string(), serde_json::Value::String("2020".to_string()));
///
/// let relation = Relation::new(
///     "rel_456".to_string(),
///     "works_at".to_string(),
///     "person_123".to_string(),
///     "company_789".to_string(),
///     properties,
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Relation {
    /// Unique identifier for this relation.
    pub id: String,
    
    /// Label/type of the relation (e.g., "works_at", "located_in").
    pub label: String,
    
    /// ID of the source entity.
    pub source_id: String,
    
    /// ID of the target entity.
    pub target_id: String,
    
    /// Additional properties/metadata for this relation.
    pub properties: HashMap<String, serde_json::Value>,
    
    /// Source document ID that this relation was extracted from.
    pub source_doc_id: Option<Uuid>,
}

impl Relation {
    /// Create a new relation.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the relation
    /// * `label` - Relation type/label
    /// * `source_id` - ID of the source entity
    /// * `target_id` - ID of the target entity
    /// * `properties` - Additional metadata
    pub fn new(
        id: String,
        label: String,
        source_id: String,
        target_id: String,
        properties: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            id,
            label,
            source_id,
            target_id,
            properties,
            source_doc_id: None,
        }
    }

    /// Set the source document ID.
    pub fn with_source_doc_id(mut self, source_doc_id: Uuid) -> Self {
        self.source_doc_id = Some(source_doc_id);
        self
    }

    /// Get a property value by key.
    pub fn get_property(&self, key: &str) -> Option<&serde_json::Value> {
        self.properties.get(key)
    }

    /// Set a property value.
    pub fn set_property(&mut self, key: String, value: serde_json::Value) {
        self.properties.insert(key, value);
    }

    /// Check if this relation connects the given entities.
    pub fn connects(&self, entity1_id: &str, entity2_id: &str) -> bool {
        (self.source_id == entity1_id && self.target_id == entity2_id)
            || (self.source_id == entity2_id && self.target_id == entity1_id)
    }

    /// Get the other entity ID given one entity ID.
    pub fn get_other_entity_id(&self, entity_id: &str) -> Option<&str> {
        if self.source_id == entity_id {
            Some(&self.target_id)
        } else if self.target_id == entity_id {
            Some(&self.source_id)
        } else {
            None
        }
    }
}

/// A triplet representing a complete relationship: (source_entity, relation, target_entity).
///
/// Triplet is the fundamental unit of knowledge in a property graph, representing
/// a complete fact or relationship. It combines two entities with their connecting
/// relation to form a semantic statement.
///
/// # Examples
///
/// ```rust
/// use cheungfun_core::types::graph::{EntityNode, Relation, Triplet};
/// use std::collections::HashMap;
///
/// let person = EntityNode::new(
///     "person_123".to_string(),
///     "John Doe".to_string(),
///     "Person".to_string(),
///     HashMap::new(),
/// );
///
/// let company = EntityNode::new(
///     "company_789".to_string(),
///     "Acme Corp".to_string(),
///     "Organization".to_string(),
///     HashMap::new(),
/// );
///
/// let works_at = Relation::new(
///     "rel_456".to_string(),
///     "works_at".to_string(),
///     "person_123".to_string(),
///     "company_789".to_string(),
///     HashMap::new(),
/// );
///
/// let triplet = Triplet::new(person, works_at, company);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Triplet {
    /// The source entity.
    pub source: EntityNode,
    
    /// The relation connecting source and target.
    pub relation: Relation,
    
    /// The target entity.
    pub target: EntityNode,
}

impl Triplet {
    /// Create a new triplet.
    ///
    /// # Arguments
    ///
    /// * `source` - Source entity
    /// * `relation` - Connecting relation
    /// * `target` - Target entity
    pub fn new(source: EntityNode, relation: Relation, target: EntityNode) -> Self {
        Self {
            source,
            relation,
            target,
        }
    }

    /// Get the relation label.
    pub fn relation_label(&self) -> &str {
        &self.relation.label
    }

    /// Get the source entity name.
    pub fn source_name(&self) -> &str {
        &self.source.name
    }

    /// Get the target entity name.
    pub fn target_name(&self) -> &str {
        &self.target.name
    }

    /// Convert to a string representation.
    pub fn to_string(&self) -> String {
        format!(
            "({}) -[{}]-> ({})",
            self.source.name,
            self.relation.label,
            self.target.name
        )
    }

    /// Check if this triplet involves the given entity ID.
    pub fn involves_entity(&self, entity_id: &str) -> bool {
        self.source.id() == entity_id || self.target.id() == entity_id
    }

    /// Get all entity IDs involved in this triplet.
    pub fn entity_ids(&self) -> Vec<String> {
        vec![self.source.id(), self.target.id()]
    }
}

/// Query parameters for graph traversal and search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQuery {
    /// Starting entity IDs for traversal.
    pub start_entities: Vec<String>,
    
    /// Relation labels to follow (empty means all relations).
    pub relation_labels: Vec<String>,
    
    /// Entity labels to include (empty means all entities).
    pub entity_labels: Vec<String>,
    
    /// Maximum traversal depth.
    pub max_depth: usize,
    
    /// Maximum number of results to return.
    pub limit: Option<usize>,
    
    /// Additional filters on entity properties.
    pub entity_filters: HashMap<String, serde_json::Value>,
    
    /// Additional filters on relation properties.
    pub relation_filters: HashMap<String, serde_json::Value>,
}

impl GraphQuery {
    /// Create a new graph query starting from given entities.
    pub fn from_entities(start_entities: Vec<String>) -> Self {
        Self {
            start_entities,
            relation_labels: Vec::new(),
            entity_labels: Vec::new(),
            max_depth: 2,
            limit: None,
            entity_filters: HashMap::new(),
            relation_filters: HashMap::new(),
        }
    }

    /// Set the maximum traversal depth.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the result limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Filter by relation labels.
    pub fn with_relation_labels(mut self, labels: Vec<String>) -> Self {
        self.relation_labels = labels;
        self
    }

    /// Filter by entity labels.
    pub fn with_entity_labels(mut self, labels: Vec<String>) -> Self {
        self.entity_labels = labels;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_node_creation() {
        let mut properties = HashMap::new();
        properties.insert("age".to_string(), serde_json::Value::Number(30.into()));

        let entity = EntityNode::new(
            "test_id".to_string(),
            "Test Entity".to_string(),
            "TestType".to_string(),
            properties,
        );

        assert_eq!(entity.id, "test_id");
        assert_eq!(entity.name, "Test Entity");
        assert_eq!(entity.label, "TestType");
        assert_eq!(entity.get_property("age"), Some(&serde_json::Value::Number(30.into())));
        assert!(!entity.has_embedding());
    }

    #[test]
    fn test_relation_creation() {
        let relation = Relation::new(
            "rel_id".to_string(),
            "test_relation".to_string(),
            "source_id".to_string(),
            "target_id".to_string(),
            HashMap::new(),
        );

        assert_eq!(relation.id, "rel_id");
        assert_eq!(relation.label, "test_relation");
        assert!(relation.connects("source_id", "target_id"));
        assert_eq!(relation.get_other_entity_id("source_id"), Some("target_id"));
    }

    #[test]
    fn test_triplet_creation() {
        let source = EntityNode::new(
            "source".to_string(),
            "Source".to_string(),
            "Entity".to_string(),
            HashMap::new(),
        );
        let target = EntityNode::new(
            "target".to_string(),
            "Target".to_string(),
            "Entity".to_string(),
            HashMap::new(),
        );
        let relation = Relation::new(
            "rel".to_string(),
            "connects".to_string(),
            "source".to_string(),
            "target".to_string(),
            HashMap::new(),
        );

        let triplet = Triplet::new(source, relation, target);
        assert_eq!(triplet.to_string(), "(Source) -[connects]-> (Target)");
        assert!(triplet.involves_entity("source"));
        assert!(triplet.involves_entity("target"));
        assert!(!triplet.involves_entity("other"));
    }
}
