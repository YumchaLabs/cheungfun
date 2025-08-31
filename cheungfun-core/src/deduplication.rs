//! Document deduplication and hash management for the Cheungfun framework.
//!
//! This module provides functionality for detecting and handling duplicate documents
//! in ingestion pipelines, following LlamaIndex's document management patterns.

use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

use crate::{Document, Node};

/// Strategy for handling duplicate documents during ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocstoreStrategy {
    /// Only check for duplicates and skip them.
    DuplicatesOnly,
    /// Update existing documents if content has changed (upsert).
    Upserts,
    /// Update existing documents and delete documents that are no longer present.
    UpsertsAndDelete,
}

impl Default for DocstoreStrategy {
    fn default() -> Self {
        Self::Upserts
    }
}

/// Document hash calculator for content-based deduplication.
#[derive(Debug, Clone)]
pub struct DocumentHasher {
    /// Whether to include metadata in hash calculation.
    include_metadata: bool,
    /// Whether to normalize whitespace before hashing.
    normalize_whitespace: bool,
}

impl Default for DocumentHasher {
    fn default() -> Self {
        Self {
            include_metadata: true,
            normalize_whitespace: true,
        }
    }
}

impl DocumentHasher {
    /// Create a new document hasher with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new document hasher with custom settings.
    #[must_use]
    pub fn with_options(include_metadata: bool, normalize_whitespace: bool) -> Self {
        Self {
            include_metadata,
            normalize_whitespace,
        }
    }

    /// Calculate hash for a document.
    ///
    /// The hash is calculated based on the document content and optionally metadata.
    /// This provides a stable way to detect content changes.
    ///
    /// # Arguments
    ///
    /// * `document` - The document to hash
    ///
    /// # Returns
    ///
    /// A SHA-256 hash string of the document content.
    pub fn calculate_hash(&self, document: &Document) -> String {
        let mut hasher = Sha256::new();

        // Hash the main content
        let content = if self.normalize_whitespace {
            document.content.trim().replace(char::is_whitespace, " ")
        } else {
            document.content.to_string()
        };
        hasher.update(content.as_bytes());

        // Optionally include metadata in hash
        if self.include_metadata {
            let mut metadata_keys: Vec<_> = document.metadata.keys().collect();
            metadata_keys.sort(); // Ensure consistent ordering

            for key in metadata_keys {
                if let Some(value) = document.metadata.get(key) {
                    hasher.update(key.as_bytes());
                    hasher.update(value.to_string().as_bytes());
                }
            }
        }

        format!("{:x}", hasher.finalize())
    }

    /// Calculate hash for a node.
    ///
    /// Similar to document hashing but for processed nodes.
    ///
    /// # Arguments
    ///
    /// * `node` - The node to hash
    ///
    /// # Returns
    ///
    /// A SHA-256 hash string of the node content.
    pub fn calculate_node_hash(&self, node: &Node) -> String {
        let mut hasher = Sha256::new();

        // Hash the main content
        let content = if self.normalize_whitespace {
            node.content.trim().replace(char::is_whitespace, " ")
        } else {
            node.content.to_string()
        };
        hasher.update(content.as_bytes());

        // Include node-specific information
        hasher.update(node.id.to_string().as_bytes());

        // Include source document ID
        hasher.update(node.source_document_id.to_string().as_bytes());

        // Optionally include metadata in hash
        if self.include_metadata {
            let mut metadata_keys: Vec<_> = node.metadata.keys().collect();
            metadata_keys.sort(); // Ensure consistent ordering

            for key in metadata_keys {
                if let Some(value) = node.metadata.get(key) {
                    hasher.update(key.as_bytes());
                    hasher.update(value.to_string().as_bytes());
                }
            }
        }

        format!("{:x}", hasher.finalize())
    }
}

/// Document deduplication manager.
///
/// This struct provides high-level functionality for managing document deduplication
/// during ingestion pipelines, following LlamaIndex's patterns.
#[derive(Debug)]
pub struct DocumentDeduplicator {
    /// Hash calculator for documents.
    hasher: DocumentHasher,
    /// Strategy for handling duplicates.
    strategy: DocstoreStrategy,
}

impl Default for DocumentDeduplicator {
    fn default() -> Self {
        Self {
            hasher: DocumentHasher::default(),
            strategy: DocstoreStrategy::default(),
        }
    }
}

impl DocumentDeduplicator {
    /// Create a new document deduplicator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new document deduplicator with custom strategy.
    #[must_use]
    pub fn with_strategy(strategy: DocstoreStrategy) -> Self {
        Self {
            hasher: DocumentHasher::default(),
            strategy,
        }
    }

    /// Create a new document deduplicator with custom hasher and strategy.
    #[must_use]
    pub fn with_hasher_and_strategy(hasher: DocumentHasher, strategy: DocstoreStrategy) -> Self {
        Self { hasher, strategy }
    }

    /// Filter documents based on existing hashes to identify duplicates and changes.
    ///
    /// # Arguments
    ///
    /// * `documents` - Input documents to process
    /// * `existing_hashes` - Map of existing document ID to hash
    ///
    /// # Returns
    ///
    /// A tuple of (documents_to_process, documents_to_skip, documents_to_update)
    pub fn filter_documents(
        &self,
        documents: Vec<Document>,
        existing_hashes: &HashMap<String, String>,
    ) -> (Vec<Document>, Vec<Document>, Vec<Document>) {
        let mut to_process = Vec::new();
        let mut to_skip = Vec::new();
        let mut to_update = Vec::new();

        for document in documents {
            let doc_id = document.id.to_string();
            let current_hash = self.hasher.calculate_hash(&document);

            match existing_hashes.get(&doc_id) {
                Some(existing_hash) => {
                    if existing_hash == &current_hash {
                        // Document unchanged, skip it
                        debug!("Skipping unchanged document: {}", doc_id);
                        to_skip.push(document);
                    } else {
                        // Document changed, needs update
                        debug!("Document changed, will update: {}", doc_id);
                        match self.strategy {
                            DocstoreStrategy::DuplicatesOnly => to_skip.push(document),
                            DocstoreStrategy::Upserts | DocstoreStrategy::UpsertsAndDelete => {
                                to_update.push(document);
                            }
                        }
                    }
                }
                None => {
                    // New document, process it
                    debug!("New document, will process: {}", doc_id);
                    to_process.push(document);
                }
            }
        }

        info!(
            "Document filtering complete: {} to process, {} to skip, {} to update",
            to_process.len(),
            to_skip.len(),
            to_update.len()
        );

        (to_process, to_skip, to_update)
    }

    /// Filter nodes based on existing hashes to identify duplicates.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Input nodes to process
    /// * `existing_hashes` - Set of existing node hashes
    ///
    /// # Returns
    ///
    /// A tuple of (nodes_to_process, nodes_to_skip)
    pub fn filter_nodes(
        &self,
        nodes: Vec<Node>,
        existing_hashes: &HashSet<String>,
    ) -> (Vec<Node>, Vec<Node>) {
        let mut to_process = Vec::new();
        let mut to_skip = Vec::new();

        for node in nodes {
            let node_hash = self.hasher.calculate_node_hash(&node);

            if existing_hashes.contains(&node_hash) {
                debug!("Skipping duplicate node: {}", node.id);
                to_skip.push(node);
            } else {
                debug!("Processing new node: {}", node.id);
                to_process.push(node);
            }
        }

        info!(
            "Node filtering complete: {} to process, {} to skip",
            to_process.len(),
            to_skip.len()
        );

        (to_process, to_skip)
    }

    /// Get the current deduplication strategy.
    #[must_use]
    pub fn strategy(&self) -> DocstoreStrategy {
        self.strategy
    }

    /// Get a reference to the document hasher.
    #[must_use]
    pub fn hasher(&self) -> &DocumentHasher {
        &self.hasher
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;

    #[test]
    fn test_document_hash_consistency() {
        let hasher = DocumentHasher::new();
        let doc = Document::new("test content");

        let hash1 = hasher.calculate_hash(&doc);
        let hash2 = hasher.calculate_hash(&doc);

        assert_eq!(hash1, hash2, "Hash should be consistent for same document");
    }

    #[test]
    fn test_document_hash_different_content() {
        let hasher = DocumentHasher::new();
        let doc1 = Document::new("content 1");
        let doc2 = Document::new("content 2");

        let hash1 = hasher.calculate_hash(&doc1);
        let hash2 = hasher.calculate_hash(&doc2);

        assert_ne!(
            hash1, hash2,
            "Hash should be different for different content"
        );
    }

    #[test]
    fn test_deduplication_filter() {
        let deduplicator = DocumentDeduplicator::new();
        let doc1 = Document::new("content 1");
        let doc2 = Document::new("content 2");

        let documents = vec![doc1.clone(), doc2.clone()];
        let existing_hashes = HashMap::new();

        let (to_process, to_skip, to_update) =
            deduplicator.filter_documents(documents, &existing_hashes);

        assert_eq!(to_process.len(), 2);
        assert_eq!(to_skip.len(), 0);
        assert_eq!(to_update.len(), 0);
    }
}
