//! Multimodal node structures extending the core Node type.
//!
//! This module provides extensions to the core Node structure to support
//! multimodal content, cross-modal relationships, and multimodal embeddings.

use crate::types::{
    media::MediaContent,
    modality::ModalityType,
};
use cheungfun_core::types::{Document, Node};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Extended document structure with multimodal support.
///
/// This structure extends the core Document to include media content
/// and multimodal metadata while maintaining backward compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalDocument {
    /// Base document information (maintains compatibility)
    pub base: Document,
    
    /// Primary modality type of this document
    pub primary_modality: ModalityType,
    
    /// Main media content (if not pure text)
    pub media_content: Option<MediaContent>,
    
    /// Additional related media content
    pub related_media: Vec<MediaContent>,
    
    /// Multimodal-specific metadata
    pub multimodal_metadata: HashMap<String, serde_json::Value>,
}

impl MultimodalDocument {
    /// Create a new multimodal document from a base document.
    pub fn from_document(document: Document) -> Self {
        Self {
            base: document,
            primary_modality: ModalityType::Text,
            media_content: None,
            related_media: Vec::new(),
            multimodal_metadata: HashMap::new(),
        }
    }
    
    /// Create a new multimodal document with media content.
    pub fn with_media(
        base: Document,
        modality: ModalityType,
        media: MediaContent,
    ) -> Self {
        Self {
            base,
            primary_modality: modality,
            media_content: Some(media),
            related_media: Vec::new(),
            multimodal_metadata: HashMap::new(),
        }
    }
    
    /// Add related media content.
    pub fn add_related_media(mut self, media: MediaContent) -> Self {
        self.related_media.push(media);
        self
    }
    
    /// Add multimodal metadata.
    pub fn with_multimodal_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.multimodal_metadata.insert(key.into(), value.into());
        self
    }
    
    /// Get all modalities present in this document.
    pub fn get_modalities(&self) -> Vec<ModalityType> {
        let mut modalities = vec![self.primary_modality];
        
        if let Some(ref media) = self.media_content {
            let media_modality = media.modality_type();
            if !modalities.contains(&media_modality) {
                modalities.push(media_modality);
            }
        }
        
        for media in &self.related_media {
            let media_modality = media.modality_type();
            if !modalities.contains(&media_modality) {
                modalities.push(media_modality);
            }
        }
        
        modalities
    }
    
    /// Check if this document contains a specific modality.
    pub fn has_modality(&self, modality: ModalityType) -> bool {
        self.get_modalities().contains(&modality)
    }
    
    /// Get the total estimated size of all media content.
    pub fn total_media_size(&self) -> u64 {
        let mut total = 0;
        
        if let Some(ref media) = self.media_content {
            total += media.estimated_size().unwrap_or(0);
        }
        
        for media in &self.related_media {
            total += media.estimated_size().unwrap_or(0);
        }
        
        total
    }
}

/// Extended node structure with multimodal support.
///
/// This structure extends the core Node to include multimodal embeddings,
/// cross-modal relationships, and media content while maintaining compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalNode {
    /// Base node information (maintains compatibility)
    pub base: Node,
    
    /// Primary modality type of this node
    pub modality: ModalityType,
    
    /// Media content associated with this node (if not pure text)
    pub media_content: Option<MediaContent>,
    
    /// Multimodal embeddings keyed by embedding model/type
    pub multimodal_embeddings: HashMap<String, Vec<f32>>,
    
    /// Cross-modal relationships to other nodes
    pub cross_modal_relations: HashMap<ModalityType, Vec<Uuid>>,
    
    /// Modality-specific features and metadata
    pub modality_features: HashMap<String, serde_json::Value>,
    
    /// Confidence scores for different modality interpretations
    pub modality_confidence: HashMap<ModalityType, f32>,
}

impl MultimodalNode {
    /// Create a new multimodal node from a base node.
    pub fn from_node(node: Node) -> Self {
        Self {
            base: node,
            modality: ModalityType::Text,
            media_content: None,
            multimodal_embeddings: HashMap::new(),
            cross_modal_relations: HashMap::new(),
            modality_features: HashMap::new(),
            modality_confidence: HashMap::new(),
        }
    }
    
    /// Create a new multimodal node with media content.
    pub fn with_media(
        base: Node,
        modality: ModalityType,
        media: MediaContent,
    ) -> Self {
        Self {
            base,
            modality,
            media_content: Some(media),
            multimodal_embeddings: HashMap::new(),
            cross_modal_relations: HashMap::new(),
            modality_features: HashMap::new(),
            modality_confidence: HashMap::new(),
        }
    }
    
    /// Add a multimodal embedding.
    pub fn add_embedding<K: Into<String>>(mut self, key: K, embedding: Vec<f32>) -> Self {
        self.multimodal_embeddings.insert(key.into(), embedding);
        self
    }
    
    /// Add a cross-modal relationship.
    pub fn add_cross_modal_relation(mut self, modality: ModalityType, node_id: Uuid) -> Self {
        self.cross_modal_relations
            .entry(modality)
            .or_insert_with(Vec::new)
            .push(node_id);
        self
    }
    
    /// Add modality-specific features.
    pub fn with_modality_feature<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.modality_features.insert(key.into(), value.into());
        self
    }
    
    /// Set confidence score for a modality.
    pub fn with_modality_confidence(mut self, modality: ModalityType, confidence: f32) -> Self {
        self.modality_confidence.insert(modality, confidence);
        self
    }
    
    /// Get the primary embedding (from base node or first multimodal embedding).
    pub fn get_primary_embedding(&self) -> Option<&Vec<f32>> {
        // First try the base node embedding
        if let Some(ref embedding) = self.base.embedding {
            return Some(embedding);
        }
        
        // Then try multimodal embeddings
        self.multimodal_embeddings.values().next()
    }
    
    /// Get embedding by key.
    pub fn get_embedding(&self, key: &str) -> Option<&Vec<f32>> {
        self.multimodal_embeddings.get(key)
    }
    
    /// Get all available embedding keys.
    pub fn embedding_keys(&self) -> Vec<&String> {
        self.multimodal_embeddings.keys().collect()
    }
    
    /// Check if this node has cross-modal relationships.
    pub fn has_cross_modal_relations(&self) -> bool {
        !self.cross_modal_relations.is_empty()
    }
    
    /// Get related nodes for a specific modality.
    pub fn get_related_nodes(&self, modality: ModalityType) -> Option<&Vec<Uuid>> {
        self.cross_modal_relations.get(&modality)
    }
    
    /// Get all related modalities.
    pub fn get_related_modalities(&self) -> Vec<ModalityType> {
        self.cross_modal_relations.keys().cloned().collect()
    }
    
    /// Get the confidence score for the primary modality.
    pub fn get_primary_confidence(&self) -> Option<f32> {
        self.modality_confidence.get(&self.modality).copied()
    }
    
    /// Check if this node is multimodal (has content from multiple modalities).
    pub fn is_multimodal(&self) -> bool {
        self.modality_confidence.len() > 1 || !self.cross_modal_relations.is_empty()
    }
}

/// Conversion trait for backward compatibility with core Node.
impl From<MultimodalNode> for Node {
    fn from(multimodal_node: MultimodalNode) -> Self {
        multimodal_node.base
    }
}

impl From<Node> for MultimodalNode {
    fn from(node: Node) -> Self {
        Self::from_node(node)
    }
}

/// Conversion trait for backward compatibility with core Document.
impl From<MultimodalDocument> for Document {
    fn from(multimodal_doc: MultimodalDocument) -> Self {
        multimodal_doc.base
    }
}

impl From<Document> for MultimodalDocument {
    fn from(document: Document) -> Self {
        Self::from_document(document)
    }
}

/// Builder for creating multimodal nodes with a fluent API.
#[derive(Debug, Default)]
pub struct MultimodalNodeBuilder {
    base: Option<Node>,
    modality: Option<ModalityType>,
    media_content: Option<MediaContent>,
    multimodal_embeddings: HashMap<String, Vec<f32>>,
    cross_modal_relations: HashMap<ModalityType, Vec<Uuid>>,
    modality_features: HashMap<String, serde_json::Value>,
    modality_confidence: HashMap<ModalityType, f32>,
}

impl MultimodalNodeBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the base node.
    pub fn base(mut self, node: Node) -> Self {
        self.base = Some(node);
        self
    }
    
    /// Set the modality.
    pub fn modality(mut self, modality: ModalityType) -> Self {
        self.modality = Some(modality);
        self
    }
    
    /// Set the media content.
    pub fn media_content(mut self, media: MediaContent) -> Self {
        self.media_content = Some(media);
        self
    }
    
    /// Add an embedding.
    pub fn embedding<K: Into<String>>(mut self, key: K, embedding: Vec<f32>) -> Self {
        self.multimodal_embeddings.insert(key.into(), embedding);
        self
    }
    
    /// Add a cross-modal relation.
    pub fn cross_modal_relation(mut self, modality: ModalityType, node_id: Uuid) -> Self {
        self.cross_modal_relations
            .entry(modality)
            .or_insert_with(Vec::new)
            .push(node_id);
        self
    }
    
    /// Add a modality feature.
    pub fn modality_feature<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.modality_features.insert(key.into(), value.into());
        self
    }
    
    /// Set modality confidence.
    pub fn modality_confidence(mut self, modality: ModalityType, confidence: f32) -> Self {
        self.modality_confidence.insert(modality, confidence);
        self
    }
    
    /// Build the multimodal node.
    pub fn build(self) -> anyhow::Result<MultimodalNode> {
        let base = self.base.ok_or_else(|| anyhow::anyhow!("Base node is required"))?;
        let modality = self.modality.unwrap_or(ModalityType::Text);
        
        Ok(MultimodalNode {
            base,
            modality,
            media_content: self.media_content,
            multimodal_embeddings: self.multimodal_embeddings,
            cross_modal_relations: self.cross_modal_relations,
            modality_features: self.modality_features,
            modality_confidence: self.modality_confidence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::types::Document;
    
    #[test]
    fn test_multimodal_document_creation() {
        let doc = Document::new("Test content");
        let multimodal_doc = MultimodalDocument::from_document(doc);
        
        assert_eq!(multimodal_doc.primary_modality, ModalityType::Text);
        assert!(multimodal_doc.media_content.is_none());
        assert_eq!(multimodal_doc.get_modalities(), vec![ModalityType::Text]);
    }
    
    #[test]
    fn test_multimodal_node_builder() {
        let chunk_info = cheungfun_core::types::ChunkInfo::new(0, 12, 0);
        let base_node = Node::new("Test content", uuid::Uuid::new_v4(), chunk_info);
        let multimodal_node = MultimodalNodeBuilder::new()
            .base(base_node)
            .modality(ModalityType::Image)
            .embedding("clip", vec![0.1, 0.2, 0.3])
            .modality_confidence(ModalityType::Image, 0.95)
            .build()
            .unwrap();
        
        assert_eq!(multimodal_node.modality, ModalityType::Image);
        assert_eq!(multimodal_node.get_embedding("clip"), Some(&vec![0.1, 0.2, 0.3]));
        assert_eq!(multimodal_node.get_primary_confidence(), Some(0.95));
    }
    
    #[test]
    fn test_cross_modal_relations() {
        let chunk_info = cheungfun_core::types::ChunkInfo::new(0, 12, 0);
        let base_node = Node::new("Test content", uuid::Uuid::new_v4(), chunk_info);
        let related_id = uuid::Uuid::new_v4();
        
        let multimodal_node = MultimodalNode::from_node(base_node)
            .add_cross_modal_relation(ModalityType::Image, related_id);
        
        assert!(multimodal_node.has_cross_modal_relations());
        assert_eq!(
            multimodal_node.get_related_nodes(ModalityType::Image),
            Some(&vec![related_id])
        );
    }
}
