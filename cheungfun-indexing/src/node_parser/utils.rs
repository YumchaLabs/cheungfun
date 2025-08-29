//! Utility functions for node parsing and construction.
//!
//! This module provides essential utility functions for building nodes from text splits,
//! managing relationships, generating IDs, and handling metadata. It closely follows
//! the patterns established in LlamaIndex while leveraging Rust's type system.

use crate::utils::metadata::add_chunk_metadata;
use cheungfun_core::{
    relationships::{NodeRelationship, RelatedNodeInfo},
    ChunkInfo, Document, Node, Result as CoreResult,
};
use std::collections::HashMap;
use tracing::{debug, warn};
use unicode_segmentation::UnicodeSegmentation;
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

        // Create enhanced metadata
        let mut metadata = document.metadata.clone();
        add_chunk_metadata(&mut metadata, i, &text_chunk, start_offset, end_offset);

        let chunk_info = ChunkInfo::with_char_indices(start_offset, end_offset, i);
        let mut node = Node::new(text_chunk, document.id, chunk_info);
        node.id = node_id;
        node.metadata = metadata;

        // Add source document relationship
        let source_info = RelatedNodeInfo::with_type(document.id, "Document".to_string());
        node.relationships
            .set_single(NodeRelationship::Source, source_info);

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
            let prev_info = RelatedNodeInfo::with_type(nodes[i - 1].id, "TextNode".to_string());
            nodes[i]
                .relationships
                .set_single(NodeRelationship::Previous, prev_info);
        }

        // Add next relationship
        if i < nodes.len() - 1 {
            let next_info = RelatedNodeInfo::with_type(nodes[i + 1].id, "TextNode".to_string());
            nodes[i]
                .relationships
                .set_single(NodeRelationship::Next, next_info);
        }
    }
    Ok(())
}

/// Calculate character offsets for text chunks within the original document.
///
/// This function attempts to find the position of each chunk within the
/// original document text, providing accurate start and end offsets.
///
/// This implementation is Unicode-safe and handles multi-byte characters correctly
/// using the unicode-segmentation crate for proper grapheme cluster handling.
fn calculate_chunk_offsets(original_text: &str, chunks: &[String]) -> Vec<(usize, usize)> {
    let mut offsets = Vec::new();
    let mut current_byte_pos = 0;

    // Pre-compute grapheme cluster boundaries for efficient lookup
    let grapheme_indices: Vec<(usize, &str)> = original_text.grapheme_indices(true).collect();

    for chunk in chunks {
        let chunk_trimmed = chunk.trim();

        // Ensure current_pos doesn't exceed text length
        if current_byte_pos >= original_text.len() {
            // If we've reached the end, estimate remaining positions
            let estimated_end = current_byte_pos + chunk_trimmed.len();
            offsets.push((current_byte_pos, estimated_end));
            current_byte_pos = estimated_end;
            continue;
        }

        // Find a safe grapheme boundary at or after current_pos
        let safe_start_pos =
            find_safe_grapheme_boundary(original_text, current_byte_pos, &grapheme_indices);

        // Try to find the chunk in the original text starting from safe position
        if let Some(found_pos) = original_text[safe_start_pos..].find(chunk_trimmed) {
            let start_offset = safe_start_pos + found_pos;
            let end_offset = (start_offset + chunk_trimmed.len()).min(original_text.len());

            // Ensure end offset is also on a safe boundary
            let safe_end_offset =
                find_safe_grapheme_boundary(original_text, end_offset, &grapheme_indices);

            offsets.push((start_offset, safe_end_offset));
            current_byte_pos = safe_end_offset;
        } else {
            // Fallback: estimate position based on current progress
            let estimated_end = current_byte_pos + chunk_trimmed.len();
            let safe_end =
                find_safe_grapheme_boundary(original_text, estimated_end, &grapheme_indices);
            offsets.push((current_byte_pos, safe_end));
            current_byte_pos = safe_end;
        }
    }

    offsets
}

/// Find the nearest safe grapheme cluster boundary at or after the given byte position.
///
/// This function ensures that we never try to slice a string in the middle
/// of a grapheme cluster (which could be a multi-byte Unicode character or
/// a combining character sequence), preventing panics.
///
/// Since we have pre-computed grapheme boundaries, this is now a simple lookup operation.
fn find_safe_grapheme_boundary(
    text: &str,
    target_pos: usize,
    grapheme_indices: &[(usize, &str)],
) -> usize {
    if target_pos >= text.len() {
        return text.len();
    }

    // Find the first grapheme cluster boundary at or after target_pos
    // Since grapheme_indices is ordered by position, we can use binary search for efficiency
    match grapheme_indices.binary_search_by_key(&target_pos, |&(pos, _)| pos) {
        // Exact match - we're already at a grapheme boundary
        Ok(index) => grapheme_indices[index].0,
        // Not found - get the next boundary
        Err(index) => {
            if index < grapheme_indices.len() {
                grapheme_indices[index].0
            } else {
                // No more boundaries, return end of string
                text.len()
            }
        }
    }
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
        assert!(nodes[0]
            .relationships
            .get_single(NodeRelationship::Next)
            .is_some());
        assert!(nodes[1]
            .relationships
            .get_single(NodeRelationship::Previous)
            .is_some());
        assert_eq!(
            nodes[0]
                .relationships
                .get_single(NodeRelationship::Next)
                .unwrap()
                .node_id,
            nodes[1].id
        );
        assert_eq!(
            nodes[1]
                .relationships
                .get_single(NodeRelationship::Previous)
                .unwrap()
                .node_id,
            nodes[0].id
        );
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
