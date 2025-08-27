//! Utility functions for node parsing and construction.
//!
//! This module provides essential utility functions for building nodes from text splits,
//! managing relationships, generating IDs, and handling metadata. It closely follows
//! the patterns established in LlamaIndex while leveraging Rust's type system.

use crate::utils::metadata::add_chunk_metadata;
use cheungfun_core::{ChunkInfo, Document, Node, Result as CoreResult};
use std::collections::HashMap;
use tracing::{debug, warn};
use uuid::Uuid;

/// Trait for ID generation functions.
///
/// This trait allows for customizable ID generation strategies,
/// enabling deterministic or random ID generation based on needs.
pub trait IdFunction: Send + Sync {
    /// Generate an ID for a node.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the chunk within the document
    /// * `document` - The source document
    ///
    /// # Returns
    ///
    /// A unique identifier for the node.
    fn generate_id(&self, index: usize, document: &Document) -> CoreResult<Uuid>;
}

/// Default ID generation function.
///
/// This generates deterministic UUIDs based on the document ID and chunk index,
/// ensuring that the same document will always produce the same node IDs.
#[derive(Debug, Clone)]
pub struct DefaultIdFunction;

impl IdFunction for DefaultIdFunction {
    fn generate_id(&self, index: usize, document: &Document) -> CoreResult<Uuid> {
        // Create deterministic UUID based on document ID and index
        let input = format!("{}-{}", document.id, index);
        Ok(Uuid::new_v5(&Uuid::NAMESPACE_OID, input.as_bytes()))
    }
}

/// Random ID generation function.
///
/// This generates completely random UUIDs for each node, useful when
/// deterministic IDs are not required.
#[derive(Debug, Clone)]
pub struct RandomIdFunction;

impl IdFunction for RandomIdFunction {
    fn generate_id(&self, _index: usize, _document: &Document) -> CoreResult<Uuid> {
        Ok(Uuid::new_v4())
    }
}

/// Build nodes from text splits.
///
/// This is the core function for converting text chunks into Node objects,
/// closely following the LlamaIndex `build_nodes_from_splits` implementation.
///
/// # Arguments
///
/// * `text_splits` - The text chunks to convert into nodes
/// * `document` - The source document
/// * `id_func` - Optional custom ID generation function
/// * `include_prev_next_rel` - Whether to include previous/next relationships
///
/// # Returns
///
/// A vector of nodes created from the text splits.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::node_parser::utils::build_nodes_from_splits;
/// use cheungfun_core::Document;
///
/// let document = Document::new("Sample content");
/// let splits = vec!["First chunk".to_string(), "Second chunk".to_string()];
///
/// let nodes = build_nodes_from_splits(splits, &document, None, true)?;
/// assert_eq!(nodes.len(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn build_nodes_from_splits(
    text_splits: Vec<String>,
    document: &Document,
    id_func: Option<&dyn IdFunction>,
    include_prev_next_rel: bool,
) -> CoreResult<Vec<Node>> {
    let id_func = id_func.unwrap_or(&DefaultIdFunction);
    let mut nodes = Vec::new();

    debug!("Building {} nodes from splits", text_splits.len());

    // Create base relationships (source document)
    let mut base_relationships = HashMap::new();
    base_relationships.insert("source".to_string(), document.id);

    // Calculate character offsets for each chunk
    let offsets = calculate_chunk_offsets(&document.content, &text_splits);

    // Create nodes from splits
    for (i, text_chunk) in text_splits.into_iter().enumerate() {
        if text_chunk.trim().is_empty() {
            warn!("Skipping empty chunk at index {}", i);
            continue;
        }

        let node_id = id_func.generate_id(i, document)?;
        let (start_offset, end_offset) = offsets.get(i).copied().unwrap_or((0, 0));

        // Clone base relationships for this node
        let mut relationships = base_relationships.clone();

        // Create enhanced metadata
        let mut metadata = document.metadata.clone();
        add_chunk_metadata(&mut metadata, i, &text_chunk, start_offset, end_offset);

        let node = Node {
            id: node_id,
            content: text_chunk,
            metadata,
            embedding: None,
            sparse_embedding: None,
            relationships,
            source_document_id: document.id,
            chunk_info: ChunkInfo {
                start_offset,
                end_offset,
                chunk_index: i,
            },
        };

        nodes.push(node);
    }

    // Add previous/next relationships if requested
    if include_prev_next_rel && nodes.len() > 1 {
        add_prev_next_relationships(&mut nodes)?;
    }

    debug!("Successfully built {} nodes", nodes.len());
    Ok(nodes)
}

/// Add previous/next relationships between nodes.
///
/// This function creates bidirectional relationships between consecutive nodes,
/// allowing for easy navigation through the document structure.
fn add_prev_next_relationships(nodes: &mut [Node]) -> CoreResult<()> {
    for i in 0..nodes.len() {
        // Add previous relationship
        if i > 0 {
            nodes[i]
                .relationships
                .insert("previous".to_string(), nodes[i - 1].id);
        }

        // Add next relationship
        if i < nodes.len() - 1 {
            nodes[i]
                .relationships
                .insert("next".to_string(), nodes[i + 1].id);
        }
    }
    Ok(())
}

/// Calculate character offsets for text chunks within the original document.
///
/// This function attempts to find the position of each chunk within the
/// original document text, providing accurate start and end offsets.
fn calculate_chunk_offsets(original_text: &str, chunks: &[String]) -> Vec<(usize, usize)> {
    let mut offsets = Vec::new();
    let mut current_pos = 0;

    for chunk in chunks {
        let chunk_trimmed = chunk.trim();

        // Try to find the chunk in the original text starting from current position
        if let Some(found_pos) = original_text[current_pos..].find(chunk_trimmed) {
            let start_offset = current_pos + found_pos;
            let end_offset = start_offset + chunk_trimmed.len();
            offsets.push((start_offset, end_offset));

            // Update position for next search, accounting for potential overlap
            current_pos = start_offset + chunk_trimmed.len();
        } else {
            // Fallback: estimate position based on current progress
            let end_offset = current_pos + chunk_trimmed.len();
            offsets.push((current_pos, end_offset));
            current_pos = end_offset;
        }
    }

    offsets
}

/// Extract metadata string from a document for metadata-aware splitting.
///
/// This function creates a string representation of the document's metadata
/// that can be used to calculate token limits when splitting text.
pub fn get_metadata_str(document: &Document) -> String {
    let mut metadata_parts = Vec::new();

    // Add common metadata fields
    if let Some(title) = document.metadata.get("title") {
        if let Some(title_str) = title.as_str() {
            metadata_parts.push(format!("Title: {}", title_str));
        }
    }

    if let Some(author) = document.metadata.get("author") {
        if let Some(author_str) = author.as_str() {
            metadata_parts.push(format!("Author: {}", author_str));
        }
    }

    if let Some(source) = document.metadata.get("source") {
        if let Some(source_str) = source.as_str() {
            metadata_parts.push(format!("Source: {}", source_str));
        }
    }

    // Add any other string metadata
    for (key, value) in &document.metadata {
        if !["title", "author", "source"].contains(&key.as_str()) {
            if let Some(value_str) = value.as_str() {
                metadata_parts.push(format!("{}: {}", key, value_str));
            }
        }
    }

    metadata_parts.join("\n")
}

/// Truncate text for logging purposes.
///
/// This utility function truncates long text to a specified length
/// for clean logging output.
pub fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len])
    }
}

/// Get ID function by name.
///
/// This function returns an ID function implementation based on a string identifier,
/// allowing for configurable ID generation strategies.
pub fn get_id_function(name: &str) -> Box<dyn IdFunction> {
    match name {
        "default" => Box::new(DefaultIdFunction),
        "random" => Box::new(RandomIdFunction),
        _ => {
            warn!("Unknown ID function '{}', using default", name);
            Box::new(DefaultIdFunction)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::Document;

    #[test]
    fn test_build_nodes_from_splits() {
        let document = Document::new("This is a test document with some content.");
        let splits = vec![
            "This is a test".to_string(),
            "document with some content.".to_string(),
        ];

        let nodes = build_nodes_from_splits(splits, &document, None, true).unwrap();

        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].content, "This is a test");
        assert_eq!(nodes[1].content, "document with some content.");

        // Check relationships
        assert!(nodes[0].relationships.contains_key("next"));
        assert!(nodes[1].relationships.contains_key("previous"));
        assert_eq!(nodes[0].relationships["next"], nodes[1].id);
        assert_eq!(nodes[1].relationships["previous"], nodes[0].id);
    }

    #[test]
    fn test_default_id_function() {
        let document = Document::new("Test content");
        let id_func = DefaultIdFunction;

        let id1 = id_func.generate_id(0, &document).unwrap();
        let id2 = id_func.generate_id(0, &document).unwrap();

        // Should be deterministic
        assert_eq!(id1, id2);

        let id3 = id_func.generate_id(1, &document).unwrap();
        // Different index should produce different ID
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_get_metadata_str() {
        let mut document = Document::new("Test content");
        document.metadata.insert(
            "title".to_string(),
            serde_json::Value::String("Test Title".to_string()),
        );
        document.metadata.insert(
            "author".to_string(),
            serde_json::Value::String("Test Author".to_string()),
        );

        let metadata_str = get_metadata_str(&document);

        assert!(metadata_str.contains("Title: Test Title"));
        assert!(metadata_str.contains("Author: Test Author"));
    }
}
