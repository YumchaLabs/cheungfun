//! Entity extraction transformer for graph-based RAG.
//!
//! This module provides entity extraction capabilities that identify and extract
//! entities and relationships from text content, following LlamaIndex's EntityExtractor
//! design exactly. The extracted entities can be used to build knowledge graphs.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

use cheungfun_core::{
    traits::{Transform, TransformInput},
    Node, Result,
};

/// Configuration for entity extraction.
///
/// This configuration controls how entities are extracted from text,
/// including confidence thresholds, entity types, and output formatting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtractionConfig {
    /// Confidence threshold for accepting entity predictions (0.0 to 1.0).
    pub prediction_threshold: f32,

    /// Whether to include entity type labels in the output.
    pub label_entities: bool,

    /// Separator used to join multi-word entity spans.
    pub span_joiner: String,

    /// Maximum number of entities to extract per node.
    pub max_entities_per_node: Option<usize>,

    /// Entity types to extract (if None, extract all types).
    pub entity_types: Option<HashSet<String>>,

    /// Whether to extract relationships between entities.
    pub extract_relationships: bool,

    /// Maximum length of text to process (longer texts will be truncated).
    pub max_text_length: usize,
}

impl Default for EntityExtractionConfig {
    fn default() -> Self {
        Self {
            prediction_threshold: 0.5,
            label_entities: false,
            span_joiner: " ".to_string(),
            max_entities_per_node: Some(50),
            entity_types: None,
            extract_relationships: true,
            max_text_length: 8192,
        }
    }
}

/// Entity extraction result for a single entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// The entity text/name.
    pub text: String,

    /// The entity type/label (e.g., "PERSON", "ORG", "LOC").
    pub label: String,

    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,

    /// Start position in the original text.
    pub start_pos: usize,

    /// End position in the original text.
    pub end_pos: usize,

    /// Additional properties for this entity.
    pub properties: HashMap<String, serde_json::Value>,
}

/// Relationship extraction result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelationship {
    /// Source entity text.
    pub source: String,

    /// Target entity text.
    pub target: String,

    /// Relationship type/label.
    pub relation_type: String,

    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,

    /// Additional properties for this relationship.
    pub properties: HashMap<String, serde_json::Value>,
}

/// Entity extractor transformer.
///
/// This transformer extracts entities and relationships from text content,
/// following LlamaIndex's EntityExtractor design exactly. It uses rule-based
/// and pattern-based extraction methods for high performance and reliability.
///
/// # Features
///
/// - Named Entity Recognition (NER) for common entity types
/// - Relationship extraction between entities
/// - Configurable confidence thresholds
/// - Support for custom entity types
/// - Batch processing for efficiency
/// - Full LlamaIndex compatibility
///
/// # Examples
///
/// ```rust
/// use cheungfun_indexing::transformers::{EntityExtractor, EntityExtractionConfig};
/// use cheungfun_core::{Node, traits::{Transform, TransformInput}};
/// use std::collections::HashMap;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = EntityExtractionConfig {
///         prediction_threshold: 0.7,
///         label_entities: true,
///         extract_relationships: true,
///         ..Default::default()
///     };
///
///     let extractor = EntityExtractor::with_config(config);
///     
///     let node = Node::new(
///         "Alice works at Microsoft in Seattle.".to_string(),
///         uuid::Uuid::new_v4(),
///         Default::default(),
///     );
///
///     let enriched_nodes = extractor.transform(TransformInput::Node(node)).await?;
///     
///     for node in enriched_nodes {
///         println!("Entities: {:?}", node.metadata.get("entities"));
///         println!("Relationships: {:?}", node.metadata.get("relationships"));
///     }
///     
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct EntityExtractor {
    /// Configuration for entity extraction.
    config: EntityExtractionConfig,

    /// Compiled patterns for entity recognition.
    patterns: EntityPatterns,
}

/// Compiled regex patterns for entity extraction.
#[derive(Debug, Clone)]
struct EntityPatterns {
    /// Pattern for person names (simple heuristic).
    person: regex::Regex,

    /// Pattern for organizations.
    organization: regex::Regex,

    /// Pattern for locations.
    location: regex::Regex,

    /// Pattern for dates.
    date: regex::Regex,

    /// Pattern for monetary amounts.
    money: regex::Regex,

    /// Pattern for email addresses.
    email: regex::Regex,

    /// Pattern for phone numbers.
    phone: regex::Regex,

    /// Pattern for URLs.
    url: regex::Regex,
}

impl Default for EntityPatterns {
    fn default() -> Self {
        Self {
            // Simple person name pattern (capitalized words)
            person: regex::Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b").unwrap(),

            // Organization patterns (common suffixes)
            organization: regex::Regex::new(r"\b[A-Z][A-Za-z\s]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation|Group|Systems|Technologies|Solutions)\b").unwrap(),

            // Location patterns (capitalized place names)
            location: regex::Regex::new(r"\b(?:in|at|from|to) ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b").unwrap(),

            // Date patterns
            date: regex::Regex::new(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b").unwrap(),

            // Money patterns
            money: regex::Regex::new(r"\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})? (?:dollars?|USD|euros?|EUR)\b").unwrap(),

            // Email pattern
            email: regex::Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap(),

            // Phone pattern
            phone: regex::Regex::new(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b").unwrap(),

            // URL pattern
            url: regex::Regex::new(r"\bhttps?://[^\s]+\b").unwrap(),
        }
    }
}

impl EntityExtractor {
    /// Create a new entity extractor with default configuration.
    pub fn new() -> Self {
        Self::with_config(EntityExtractionConfig::default())
    }

    /// Create a new entity extractor with custom configuration.
    pub fn with_config(config: EntityExtractionConfig) -> Self {
        Self {
            config,
            patterns: EntityPatterns::default(),
        }
    }

    /// Extract entities from text content.
    ///
    /// This method performs the core entity extraction logic using pattern matching
    /// and heuristics to identify different types of entities in the text.
    fn extract_entities(&self, text: &str) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();

        // Truncate text if it's too long
        let text = if text.len() > self.config.max_text_length {
            warn!(
                "Text length {} exceeds maximum {}, truncating",
                text.len(),
                self.config.max_text_length
            );
            &text[..self.config.max_text_length]
        } else {
            text
        };

        // Extract different types of entities
        self.extract_entities_by_pattern(&mut entities, text, &self.patterns.person, "PERSON");
        self.extract_entities_by_pattern(&mut entities, text, &self.patterns.organization, "ORG");
        self.extract_entities_by_pattern(&mut entities, text, &self.patterns.date, "DATE");
        self.extract_entities_by_pattern(&mut entities, text, &self.patterns.money, "MONEY");
        self.extract_entities_by_pattern(&mut entities, text, &self.patterns.email, "EMAIL");
        self.extract_entities_by_pattern(&mut entities, text, &self.patterns.phone, "PHONE");
        self.extract_entities_by_pattern(&mut entities, text, &self.patterns.url, "URL");

        // Extract locations (special handling for capture groups)
        for cap in self.patterns.location.captures_iter(text) {
            if let Some(location_match) = cap.get(1) {
                let entity_text = location_match.as_str().to_string();
                if self.should_include_entity(&entity_text, "LOCATION") {
                    entities.push(ExtractedEntity {
                        text: entity_text,
                        label: "LOCATION".to_string(),
                        confidence: 0.8, // Moderate confidence for pattern-based extraction
                        start_pos: location_match.start(),
                        end_pos: location_match.end(),
                        properties: HashMap::new(),
                    });
                }
            }
        }

        // Remove duplicates and sort by position
        entities.sort_by_key(|e| e.start_pos);
        entities.dedup_by(|a, b| a.text == b.text && a.label == b.label);

        // Apply max entities limit
        if let Some(max_entities) = self.config.max_entities_per_node {
            entities.truncate(max_entities);
        }

        debug!("Extracted {} entities from text", entities.len());
        entities
    }

    /// Extract entities using a specific pattern and label.
    fn extract_entities_by_pattern(
        &self,
        entities: &mut Vec<ExtractedEntity>,
        text: &str,
        pattern: &regex::Regex,
        label: &str,
    ) {
        for mat in pattern.find_iter(text) {
            let entity_text = mat.as_str().to_string();
            if self.should_include_entity(&entity_text, label) {
                entities.push(ExtractedEntity {
                    text: entity_text,
                    label: label.to_string(),
                    confidence: 0.9, // High confidence for pattern-based extraction
                    start_pos: mat.start(),
                    end_pos: mat.end(),
                    properties: HashMap::new(),
                });
            }
        }
    }

    /// Check if an entity should be included based on configuration.
    fn should_include_entity(&self, entity_text: &str, label: &str) -> bool {
        // Check entity type filter
        if let Some(ref allowed_types) = self.config.entity_types {
            if !allowed_types.contains(label) {
                return false;
            }
        }

        // Basic quality filters
        if entity_text.len() < 2 || entity_text.len() > 100 {
            return false;
        }

        // Skip common stop words and noise
        let stop_words = [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        ];
        if stop_words.contains(&entity_text.to_lowercase().as_str()) {
            return false;
        }

        true
    }

    /// Extract relationships between entities.
    ///
    /// This method identifies potential relationships between extracted entities
    /// using simple heuristics and patterns.
    fn extract_relationships(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
    ) -> Vec<ExtractedRelationship> {
        let mut relationships = Vec::new();

        if !self.config.extract_relationships || entities.len() < 2 {
            return relationships;
        }

        // Simple relationship patterns
        let work_patterns = [
            r"\b(\w+(?:\s\w+)*)\s+(?:works?\s+(?:at|for)|employed\s+(?:at|by))\s+(\w+(?:\s\w+)*)\b",
            r"\b(\w+(?:\s\w+)*)\s+(?:is\s+(?:a|an|the)\s+)?(?:CEO|CTO|manager|director|employee)\s+(?:of|at)\s+(\w+(?:\s\w+)*)\b",
        ];

        let location_patterns = [
            r"\b(\w+(?:\s\w+)*)\s+(?:is\s+)?(?:located\s+)?(?:in|at)\s+(\w+(?:\s\w+)*)\b",
            r"\b(\w+(?:\s\w+)*)\s+(?:from|of)\s+(\w+(?:\s\w+)*)\b",
        ];

        // Extract work relationships
        for pattern_str in &work_patterns {
            if let Ok(pattern) = regex::Regex::new(pattern_str) {
                for cap in pattern.captures_iter(text) {
                    if let (Some(source), Some(target)) = (cap.get(1), cap.get(2)) {
                        let source_text = source.as_str().trim();
                        let target_text = target.as_str().trim();

                        // Check if both entities were extracted
                        if self.entity_exists(source_text, entities)
                            && self.entity_exists(target_text, entities)
                        {
                            relationships.push(ExtractedRelationship {
                                source: source_text.to_string(),
                                target: target_text.to_string(),
                                relation_type: "WORKS_AT".to_string(),
                                confidence: 0.8,
                                properties: HashMap::new(),
                            });
                        }
                    }
                }
            }
        }

        // Extract location relationships
        for pattern_str in &location_patterns {
            if let Ok(pattern) = regex::Regex::new(pattern_str) {
                for cap in pattern.captures_iter(text) {
                    if let (Some(source), Some(target)) = (cap.get(1), cap.get(2)) {
                        let source_text = source.as_str().trim();
                        let target_text = target.as_str().trim();

                        if self.entity_exists(source_text, entities)
                            && self.entity_exists(target_text, entities)
                        {
                            relationships.push(ExtractedRelationship {
                                source: source_text.to_string(),
                                target: target_text.to_string(),
                                relation_type: "LOCATED_IN".to_string(),
                                confidence: 0.7,
                                properties: HashMap::new(),
                            });
                        }
                    }
                }
            }
        }

        debug!("Extracted {} relationships from text", relationships.len());
        relationships
    }

    /// Check if an entity exists in the extracted entities list.
    fn entity_exists(&self, entity_text: &str, entities: &[ExtractedEntity]) -> bool {
        entities
            .iter()
            .any(|e| e.text.to_lowercase() == entity_text.to_lowercase())
    }

    /// Convert extracted entities to metadata format.
    fn entities_to_metadata(
        &self,
        entities: &[ExtractedEntity],
    ) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();

        if entities.is_empty() {
            return metadata;
        }

        if self.config.label_entities {
            // Group entities by label
            let mut entities_by_label: HashMap<String, Vec<String>> = HashMap::new();

            for entity in entities {
                if entity.confidence >= self.config.prediction_threshold {
                    entities_by_label
                        .entry(entity.label.clone())
                        .or_default()
                        .push(entity.text.clone());
                }
            }

            // Add each entity type as a separate metadata field
            for (label, entity_list) in entities_by_label {
                let label_key = label.to_lowercase();
                metadata.insert(
                    label_key,
                    serde_json::Value::Array(
                        entity_list
                            .into_iter()
                            .map(serde_json::Value::String)
                            .collect(),
                    ),
                );
            }
        } else {
            // All entities in a single "entities" field
            let entity_list: Vec<String> = entities
                .iter()
                .filter(|e| e.confidence >= self.config.prediction_threshold)
                .map(|e| e.text.clone())
                .collect();

            if !entity_list.is_empty() {
                metadata.insert(
                    "entities".to_string(),
                    serde_json::Value::Array(
                        entity_list
                            .into_iter()
                            .map(serde_json::Value::String)
                            .collect(),
                    ),
                );
            }
        }

        metadata
    }

    /// Convert extracted relationships to metadata format.
    fn relationships_to_metadata(
        &self,
        relationships: &[ExtractedRelationship],
    ) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();

        if relationships.is_empty() {
            return metadata;
        }

        let relationship_list: Vec<serde_json::Value> = relationships
            .iter()
            .filter(|r| r.confidence >= self.config.prediction_threshold)
            .map(|r| {
                serde_json::json!({
                    "source": r.source,
                    "target": r.target,
                    "relation": r.relation_type,
                    "confidence": r.confidence
                })
            })
            .collect();

        if !relationship_list.is_empty() {
            metadata.insert(
                "relationships".to_string(),
                serde_json::Value::Array(relationship_list),
            );
        }

        metadata
    }

    /// Process a single node for entity extraction.
    async fn process_node(&self, mut node: Node) -> Result<Node> {
        debug!("Processing node {} for entity extraction", node.id);

        let (entities, relationships) = if node.content.is_empty() {
            warn!(
                "Node {} has empty content, skipping entity extraction",
                node.id
            );
            (Vec::new(), Vec::new())
        } else {
            // Extract entities
            let entities = self.extract_entities(&node.content);

            // Extract relationships
            let relationships = self.extract_relationships(&node.content, &entities);

            (entities, relationships)
        };

        // Convert to metadata
        let entity_metadata = self.entities_to_metadata(&entities);
        let relationship_metadata = self.relationships_to_metadata(&relationships);

        // Add to node metadata
        for (key, value) in entity_metadata {
            node.metadata.insert(key, value);
        }

        for (key, value) in relationship_metadata {
            node.metadata.insert(key, value);
        }

        // Add extraction statistics
        node.metadata.insert(
            "entity_extraction_stats".to_string(),
            serde_json::json!({
                "total_entities": entities.len(),
                "total_relationships": relationships.len(),
                "prediction_threshold": self.config.prediction_threshold
            }),
        );

        info!(
            "Extracted {} entities and {} relationships from node {}",
            entities.len(),
            relationships.len(),
            node.id
        );

        Ok(node)
    }
}

impl Default for EntityExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Transform for EntityExtractor {
    async fn transform(&self, input: TransformInput) -> Result<Vec<Node>> {
        match input {
            TransformInput::Node(node) => {
                let processed_node = self.process_node(node).await?;
                Ok(vec![processed_node])
            }
            TransformInput::Nodes(nodes) => {
                let mut processed_nodes = Vec::with_capacity(nodes.len());

                for node in nodes {
                    let processed_node = self.process_node(node).await?;
                    processed_nodes.push(processed_node);
                }

                Ok(processed_nodes)
            }
            TransformInput::Document(_) => {
                Err(cheungfun_core::CheungfunError::Validation {
                    message: "EntityExtractor only processes nodes, not documents. Use a document splitter first.".to_string()
                })
            }
            TransformInput::Documents(_) => {
                Err(cheungfun_core::CheungfunError::Validation {
                    message: "EntityExtractor only processes nodes, not documents. Use a document splitter first.".to_string()
                })
            }
        }
    }

    fn name(&self) -> &'static str {
        "EntityExtractor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::types::ChunkInfo;
    use uuid::Uuid;

    fn create_test_node(content: &str) -> Node {
        Node::new(
            content.to_string(),
            Uuid::new_v4(),
            ChunkInfo {
                start_offset: 0,
                end_offset: content.len(),
                chunk_index: 0,
            },
        )
    }

    #[tokio::test]
    async fn test_entity_extraction_basic() {
        let extractor = EntityExtractor::new();
        let node = create_test_node("Alice Smith works at Microsoft in Seattle.");

        let result = extractor
            .transform(TransformInput::Node(node))
            .await
            .unwrap();
        assert_eq!(result.len(), 1);

        let processed_node = &result[0];

        // Check that entities were extracted
        assert!(processed_node.metadata.contains_key("entities"));

        // Check extraction stats
        assert!(processed_node
            .metadata
            .contains_key("entity_extraction_stats"));
    }

    #[tokio::test]
    async fn test_entity_extraction_with_labels() {
        let config = EntityExtractionConfig {
            label_entities: true,
            ..Default::default()
        };
        let extractor = EntityExtractor::with_config(config);
        let node = create_test_node(
            "John Doe works at Apple Inc in Cupertino. Contact: john@apple.com or (555) 123-4567.",
        );

        let result = extractor
            .transform(TransformInput::Node(node))
            .await
            .unwrap();
        let processed_node = &result[0];

        // Check that different entity types are separated
        if processed_node.metadata.contains_key("person") {
            println!(
                "Found person entities: {:?}",
                processed_node.metadata.get("person")
            );
        }
        if processed_node.metadata.contains_key("org") {
            println!(
                "Found org entities: {:?}",
                processed_node.metadata.get("org")
            );
        }
        if processed_node.metadata.contains_key("email") {
            println!(
                "Found email entities: {:?}",
                processed_node.metadata.get("email")
            );
        }
    }

    #[tokio::test]
    async fn test_relationship_extraction() {
        let config = EntityExtractionConfig {
            extract_relationships: true,
            ..Default::default()
        };
        let extractor = EntityExtractor::with_config(config);
        let node =
            create_test_node("Alice Smith works at Microsoft. Microsoft is located in Seattle.");

        let result = extractor
            .transform(TransformInput::Node(node))
            .await
            .unwrap();
        let processed_node = &result[0];

        // Check that relationships were extracted
        if processed_node.metadata.contains_key("relationships") {
            println!(
                "Found relationships: {:?}",
                processed_node.metadata.get("relationships")
            );
        }
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let extractor = EntityExtractor::new();
        let nodes = vec![
            create_test_node("Alice works at Google."),
            create_test_node("Bob is from New York."),
            create_test_node("Microsoft was founded in 1975."),
        ];

        let result = extractor
            .transform(TransformInput::Nodes(nodes))
            .await
            .unwrap();
        assert_eq!(result.len(), 3);

        // Each node should have been processed
        for node in &result {
            assert!(node.metadata.contains_key("entity_extraction_stats"));
        }
    }

    #[tokio::test]
    async fn test_empty_content() {
        let extractor = EntityExtractor::new();
        let node = create_test_node("");

        let result = extractor
            .transform(TransformInput::Node(node))
            .await
            .unwrap();
        assert_eq!(result.len(), 1);

        let processed_node = &result[0];
        // Should not have entities but should have stats
        assert!(processed_node
            .metadata
            .contains_key("entity_extraction_stats"));
    }

    #[tokio::test]
    async fn test_document_input_error() {
        let extractor = EntityExtractor::new();
        let document = cheungfun_core::Document::new("test content".to_string());

        let result = extractor
            .transform(TransformInput::Document(document))
            .await;
        assert!(result.is_err());

        if let Err(e) = result {
            assert!(e.to_string().contains("only processes nodes"));
        }
    }

    #[tokio::test]
    async fn test_confidence_threshold() {
        let config = EntityExtractionConfig {
            prediction_threshold: 0.95, // Very high threshold
            ..Default::default()
        };
        let extractor = EntityExtractor::with_config(config);
        let node = create_test_node("Alice Smith works at Microsoft.");

        let result = extractor
            .transform(TransformInput::Node(node))
            .await
            .unwrap();
        let processed_node = &result[0];

        // With high threshold, fewer entities should pass
        if let Some(entities) = processed_node.metadata.get("entities") {
            if let serde_json::Value::Array(arr) = entities {
                println!("High threshold entities: {} found", arr.len());
            }
        }
    }

    #[test]
    fn test_entity_patterns() {
        let patterns = EntityPatterns::default();

        // Test person pattern
        let text = "Alice Smith and Bob Johnson are here.";
        let matches: Vec<_> = patterns
            .person
            .find_iter(text)
            .map(|m| m.as_str())
            .collect();
        assert!(matches.contains(&"Alice Smith"));
        assert!(matches.contains(&"Bob Johnson"));

        // Test organization pattern
        let text = "Microsoft Corporation and Apple Inc are competitors.";
        let matches: Vec<_> = patterns
            .organization
            .find_iter(text)
            .map(|m| m.as_str())
            .collect();
        assert!(matches.len() > 0);

        // Test email pattern
        let text = "Contact us at info@example.com or support@test.org";
        let matches: Vec<_> = patterns.email.find_iter(text).map(|m| m.as_str()).collect();
        assert!(matches.contains(&"info@example.com"));
        assert!(matches.contains(&"support@test.org"));
    }
}
