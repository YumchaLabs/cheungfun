//! Text splitting transformer for chunking documents.

use async_trait::async_trait;
use cheungfun_core::traits::Transformer;
use cheungfun_core::{ChunkInfo, Document, Node, Result as CoreResult};
use std::collections::HashMap;
use tracing::{debug, warn};
use uuid::Uuid;

use super::{utils, SplitterConfig};
use crate::error::Result;

/// Text splitter that divides documents into smaller chunks.
///
/// This transformer takes documents and splits them into smaller nodes that are
/// suitable for embedding and retrieval. It supports various splitting strategies
/// and can respect text boundaries like sentences and paragraphs.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::transformers::{TextSplitter, SplitterConfig};
/// use cheungfun_core::{Document, traits::Transformer};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = SplitterConfig::new(1000, 200)
///         .with_respect_sentence_boundaries(true);
///         
///     let splitter = TextSplitter::with_config(config);
///     let document = Document::new("This is a long document that needs to be split...");
///     
///     let nodes = splitter.transform(document).await?;
///     println!("Split document into {} chunks", nodes.len());
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct TextSplitter {
    /// Configuration for splitting behavior.
    config: SplitterConfig,
}

impl TextSplitter {
    /// Create a new text splitter with default configuration.
    #[must_use]
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            config: SplitterConfig::new(chunk_size, chunk_overlap),
        }
    }

    /// Create a new text splitter with custom configuration.
    #[must_use]
    pub fn with_config(config: SplitterConfig) -> Self {
        Self { config }
    }

    /// Get the splitter configuration.
    #[must_use]
    pub fn config(&self) -> &SplitterConfig {
        &self.config
    }

    /// Split text into chunks using the configured strategy.
    fn split_text(&self, text: &str) -> Result<Vec<String>> {
        debug!("Splitting text of {} characters", text.len());

        if text.len() <= self.config.chunk_size {
            return Ok(vec![text.to_string()]);
        }

        let mut chunks: Vec<String> = Vec::new();
        let mut current_pos = 0;

        while current_pos < text.len() {
            let chunk_end = std::cmp::min(current_pos + self.config.chunk_size, text.len());
            let mut chunk = &text[current_pos..chunk_end];

            // Try to find a good split point using separators
            if chunk_end < text.len() {
                chunk = self.find_split_point(chunk, &text[chunk_end..]);
            }

            // Clean the chunk
            let cleaned_chunk = utils::clean_text(chunk);

            // Check minimum chunk size
            if let Some(min_size) = self.config.min_chunk_size {
                if cleaned_chunk.len() < min_size && !chunks.is_empty() {
                    // Merge with previous chunk if too small
                    if let Some(last_chunk) = chunks.last_mut() {
                        last_chunk.push(' ');
                        last_chunk.push_str(&cleaned_chunk);
                    }
                } else {
                    chunks.push(cleaned_chunk);
                }
            } else {
                chunks.push(cleaned_chunk);
            }

            // Calculate next position with overlap
            let chunk_len = chunk.len();
            if chunk_len <= self.config.chunk_overlap {
                current_pos += chunk_len;
            } else {
                current_pos += chunk_len - self.config.chunk_overlap;
            }

            // Prevent infinite loop
            if current_pos <= chunk_end - chunk_len {
                current_pos = chunk_end;
            }
        }

        // Filter out empty chunks
        let filtered_chunks: Vec<String> = chunks
            .into_iter()
            .filter(|chunk| !chunk.trim().is_empty())
            .collect();

        debug!("Split text into {} chunks", filtered_chunks.len());
        Ok(filtered_chunks)
    }

    /// Find the best split point using configured separators.
    fn find_split_point<'a>(&self, chunk: &'a str, _remaining: &str) -> &'a str {
        // Try each separator in order of preference
        for separator in &self.config.separators {
            if let Some(split_pos) = chunk.rfind(separator) {
                let split_point = if self.config.keep_separators {
                    split_pos + separator.len()
                } else {
                    split_pos
                };

                // Make sure we don't create too small chunks
                if let Some(min_size) = self.config.min_chunk_size {
                    if split_point >= min_size {
                        return &chunk[..split_point];
                    }
                } else {
                    return &chunk[..split_point];
                }
            }
        }

        // If no good split point found, return the original chunk
        chunk
    }

    /// Create a node from a text chunk.
    fn create_node(
        &self,
        chunk: String,
        chunk_index: usize,
        start_offset: usize,
        end_offset: usize,
        source_document: &Document,
    ) -> Node {
        let mut metadata = source_document.metadata.clone();

        // Add chunk-specific metadata
        metadata.insert(
            "chunk_index".to_string(),
            serde_json::Value::Number(chunk_index.into()),
        );
        metadata.insert(
            "chunk_size".to_string(),
            serde_json::Value::Number(chunk.len().into()),
        );
        metadata.insert(
            "splitter_config".to_string(),
            serde_json::json!({
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "respect_sentence_boundaries": self.config.respect_sentence_boundaries,
                "respect_paragraph_boundaries": self.config.respect_paragraph_boundaries,
            }),
        );

        // Extract title from chunk if it's the first chunk
        if chunk_index == 0 {
            if let Some(title) = utils::extract_title(&chunk) {
                metadata.insert(
                    "extracted_title".to_string(),
                    serde_json::Value::String(title),
                );
            }
        }

        // Add text statistics
        let stats = utils::calculate_statistics(&chunk);
        for (key, value) in stats {
            metadata.insert(format!("chunk_{key}"), value);
        }

        Node {
            id: Uuid::new_v4(),
            content: chunk,
            metadata,
            embedding: None,
            sparse_embedding: None,
            relationships: HashMap::new(),
            source_document_id: source_document.id,
            chunk_info: ChunkInfo {
                start_offset,
                end_offset,
                chunk_index,
            },
        }
    }

    /// Calculate character offsets for chunks in the original text.
    fn calculate_offsets(&self, original_text: &str, chunks: &[String]) -> Vec<(usize, usize)> {
        let mut offsets = Vec::new();
        let mut current_pos = 0;

        for chunk in chunks {
            // Find the chunk in the original text starting from current position
            let search_text = if current_pos < original_text.len() {
                &original_text[current_pos..]
            } else {
                ""
            };

            if let Some(found_pos) = search_text.find(chunk.trim()) {
                let start_offset = current_pos + found_pos;
                let end_offset =
                    std::cmp::min(start_offset + chunk.trim().len(), original_text.len());
                offsets.push((start_offset, end_offset));

                // Update position for next search, accounting for overlap
                current_pos = if end_offset > self.config.chunk_overlap {
                    std::cmp::min(end_offset - self.config.chunk_overlap, original_text.len())
                } else {
                    end_offset
                };
            } else {
                // Fallback: estimate position
                let start_offset = current_pos;
                let end_offset = std::cmp::min(current_pos + chunk.len(), original_text.len());
                offsets.push((start_offset, end_offset));
                current_pos = end_offset;
            }
        }

        offsets
    }
}

#[async_trait]
impl Transformer for TextSplitter {
    async fn transform(&self, document: Document) -> CoreResult<Vec<Node>> {
        debug!("Transforming document {} with TextSplitter", document.id);

        if document.content.is_empty() {
            warn!("Document {} has empty content", document.id);
            return Ok(vec![]);
        }

        // Split the text into chunks
        let chunks = match self.split_text(&document.content) {
            Ok(chunks) => chunks,
            Err(e) => {
                return Err(cheungfun_core::error::CheungfunError::Pipeline {
                    message: format!("Text splitting failed: {e}"),
                });
            }
        };

        if chunks.is_empty() {
            warn!("No chunks created from document {}", document.id);
            return Ok(vec![]);
        }

        // Calculate offsets for each chunk
        let offsets = self.calculate_offsets(&document.content, &chunks);

        // Create nodes from chunks
        let mut nodes = Vec::new();
        for (chunk_index, (chunk, (start_offset, end_offset))) in
            chunks.into_iter().zip(offsets.into_iter()).enumerate()
        {
            let node = self.create_node(chunk, chunk_index, start_offset, end_offset, &document);

            nodes.push(node);
        }

        debug!(
            "Created {} nodes from document {}",
            nodes.len(),
            document.id
        );
        Ok(nodes)
    }

    async fn transform_batch(&self, documents: Vec<Document>) -> CoreResult<Vec<Node>> {
        debug!("Batch transforming {} documents", documents.len());

        let mut all_nodes = Vec::new();
        for document in documents {
            let nodes = self.transform(document).await?;
            all_nodes.extend(nodes);
        }

        debug!(
            "Batch transformation created {} total nodes",
            all_nodes.len()
        );
        Ok(all_nodes)
    }

    fn name(&self) -> &'static str {
        "TextSplitter"
    }
}
