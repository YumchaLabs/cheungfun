//! Sentence-based text splitter implementation.
//!
//! This module provides the SentenceSplitter, which is the primary text splitting
//! implementation that closely follows LlamaIndex's SentenceSplitter behavior.
//! It prioritizes keeping complete sentences and paragraphs together while
//! respecting token limits.

use super::{
    create_sentence_split_functions, Split, SplitFunction, Tokenizer, WhitespaceTokenizer,
};
use crate::node_parser::{
    callbacks::{CallbackManager, EventPayload},
    config::SentenceSplitterConfig,
    utils::{build_nodes_from_splits, get_id_function},
    MetadataAwareTextSplitter, NodeParser, TextSplitter,
};
use async_trait::async_trait;
use cheungfun_core::{
    traits::{DocumentState, NodeState, TypedData, TypedTransform},
    Document, Node, Result as CoreResult,
};
use std::sync::Arc;
use tracing::debug;

/// Sentence splitter that prioritizes complete sentences and paragraphs.
///
/// This splitter implements a hierarchical splitting strategy:
/// 1. Split by paragraph separators
/// 2. Split by sentence tokenizer (or regex)
/// 3. Split by secondary chunking regex
/// 4. Split by word separators
/// 5. Split by characters (as last resort)
///
/// The splitter then merges these splits back together while respecting
/// chunk size limits and maintaining sentence boundaries when possible.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::node_parser::{text::SentenceSplitter, TextSplitter};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let splitter = SentenceSplitter::from_defaults(1000, 200)?;
///     
///     let text = "This is the first sentence. This is the second sentence. This is a longer sentence that might need to be handled carefully.";
///     let chunks = splitter.split_text(text)?;
///     
///     println!("Split into {} chunks", chunks.len());
///     for (i, chunk) in chunks.iter().enumerate() {
///         println!("Chunk {}: {}", i + 1, chunk);
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct SentenceSplitter {
    config: SentenceSplitterConfig,
    tokenizer: Arc<dyn Tokenizer>,
    split_functions: Vec<Box<dyn SplitFunction>>,
    callback_manager: Option<CallbackManager>,
}

impl SentenceSplitter {
    /// Create a new sentence splitter with the given configuration.
    pub fn new(config: SentenceSplitterConfig) -> CoreResult<Self> {
        let tokenizer = Arc::new(WhitespaceTokenizer); // Default tokenizer
        let split_functions = create_sentence_split_functions()?;

        Ok(Self {
            config,
            tokenizer,
            split_functions,
            callback_manager: None,
        })
    }

    /// Create a sentence splitter with default configuration.
    pub fn from_defaults(chunk_size: usize, chunk_overlap: usize) -> CoreResult<Self> {
        let config = SentenceSplitterConfig::new(chunk_size, chunk_overlap);
        Self::new(config)
    }

    /// Set a custom tokenizer.
    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Set callback manager.
    pub fn with_callback_manager(mut self, callback_manager: CallbackManager) -> Self {
        self.callback_manager = Some(callback_manager);
        self
    }

    /// Split text with a specific chunk size.
    fn split_text_with_chunk_size(&self, text: &str, chunk_size: usize) -> CoreResult<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![text.to_string()]);
        }

        debug!(
            "Splitting text of {} characters with chunk size {}",
            text.len(),
            chunk_size
        );

        // Step 1: Split text into atomic units
        let splits = self.split_into_splits(text, chunk_size)?;

        // Step 2: Merge splits back together respecting chunk size
        let chunks = self.merge_splits(splits, chunk_size)?;

        debug!("Created {} chunks from text", chunks.len());
        Ok(chunks)
    }

    /// Split text into atomic Split units using hierarchical splitting.
    fn split_into_splits(&self, text: &str, chunk_size: usize) -> CoreResult<Vec<Split>> {
        let token_size = self.tokenizer.count_tokens(text)?;
        if token_size <= chunk_size {
            return Ok(vec![Split::new(text.to_string(), true, token_size)]);
        }

        // Try each split function in order until we get splits that fit
        let text_splits = self.get_splits_by_functions(text)?;
        let mut splits = Vec::new();

        for text_split in text_splits {
            let token_size = self.tokenizer.count_tokens(&text_split)?;
            if token_size <= chunk_size {
                splits.push(Split::new(text_split, true, token_size));
            } else {
                // Recursively split this piece
                let recursive_splits = self.split_into_splits(&text_split, chunk_size)?;
                splits.extend(recursive_splits);
            }
        }

        Ok(splits)
    }

    /// Apply split functions hierarchically to get text splits.
    fn get_splits_by_functions(&self, text: &str) -> CoreResult<Vec<String>> {
        let mut current_splits = vec![text.to_string()];

        for split_fn in &self.split_functions {
            let mut new_splits = Vec::new();

            for split in current_splits {
                let sub_splits = split_fn.split(&split)?;
                new_splits.extend(sub_splits);
            }

            // If we got more than one split, we found a good separator
            if new_splits.len() > 1 {
                return Ok(new_splits
                    .into_iter()
                    .filter(|s| !s.trim().is_empty())
                    .collect());
            }

            current_splits = new_splits;
        }

        // If no split function worked, return the original text
        Ok(vec![text.to_string()])
    }

    /// Merge splits back together while respecting chunk size limits.
    fn merge_splits(&self, splits: Vec<Split>, chunk_size: usize) -> CoreResult<Vec<String>> {
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_chunk_len = 0;
        let mut last_chunk = Vec::new();

        for split in splits {
            if split.token_size > chunk_size {
                return Err(cheungfun_core::error::CheungfunError::Pipeline {
                    message: format!(
                        "Single split exceeded chunk size: {} > {}",
                        split.token_size, chunk_size
                    ),
                });
            }

            // Check if adding this split would exceed the chunk size
            if current_chunk_len + split.token_size > chunk_size && !current_chunk.is_empty() {
                // Close current chunk
                self.close_chunk(
                    &mut chunks,
                    &mut current_chunk,
                    &mut current_chunk_len,
                    &mut last_chunk,
                )?;

                // Add overlap from previous chunk
                self.add_overlap(&mut current_chunk, &mut current_chunk_len, &last_chunk)?;
            }

            // Add split to current chunk
            current_chunk_len += split.token_size;
            current_chunk.push(split);
        }

        // Add the final chunk if it has content
        if !current_chunk.is_empty() {
            let chunk_text = current_chunk
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join("");
            chunks.push(chunk_text);
        }

        Ok(chunks)
    }

    /// Close the current chunk and prepare for the next one.
    fn close_chunk(
        &self,
        chunks: &mut Vec<String>,
        current_chunk: &mut Vec<Split>,
        current_chunk_len: &mut usize,
        last_chunk: &mut Vec<Split>,
    ) -> CoreResult<()> {
        let chunk_text = current_chunk
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        chunks.push(chunk_text);

        *last_chunk = current_chunk.clone();
        current_chunk.clear();
        *current_chunk_len = 0;

        Ok(())
    }

    /// Add overlap from the previous chunk to the current chunk.
    fn add_overlap(
        &self,
        current_chunk: &mut Vec<Split>,
        current_chunk_len: &mut usize,
        last_chunk: &[Split],
    ) -> CoreResult<()> {
        if last_chunk.is_empty() {
            return Ok(());
        }

        let overlap_limit = self.config.base.chunk_overlap;
        let mut overlap_len = 0;

        // Add splits from the end of the last chunk until we reach the overlap limit
        for split in last_chunk.iter().rev() {
            if overlap_len + split.token_size <= overlap_limit {
                current_chunk.insert(0, split.clone());
                *current_chunk_len += split.token_size;
                overlap_len += split.token_size;
            } else {
                break;
            }
        }

        Ok(())
    }
}

#[async_trait]
impl NodeParser for SentenceSplitter {
    async fn parse_nodes(
        &self,
        documents: &[Document],
        _show_progress: bool,
    ) -> CoreResult<Vec<Node>> {
        let mut all_nodes = Vec::new();

        // Emit start event
        if let Some(ref callback_manager) = self.callback_manager {
            let payload = EventPayload::node_parsing_start(documents.len());
            callback_manager.emit_event(payload).await?;
        }

        for document in documents {
            let chunks = self.split_text(&document.content)?;

            let id_func = self
                .config
                .base
                .base
                .id_func
                .as_ref()
                .map(|name| get_id_function(name));

            let nodes = build_nodes_from_splits(
                chunks,
                document,
                id_func.as_deref(),
                self.config.base.base.include_prev_next_rel,
            )?;

            all_nodes.extend(nodes);
        }

        // Emit end event
        if let Some(ref callback_manager) = self.callback_manager {
            let payload = EventPayload::node_parsing_end(&all_nodes);
            callback_manager.emit_event(payload).await?;
        }

        Ok(all_nodes)
    }
}

#[async_trait]
impl TextSplitter for SentenceSplitter {
    fn split_text(&self, text: &str) -> CoreResult<Vec<String>> {
        self.split_text_with_chunk_size(text, self.config.base.chunk_size)
    }
}

#[async_trait]
impl MetadataAwareTextSplitter for SentenceSplitter {
    fn split_text_metadata_aware(&self, text: &str, metadata_str: &str) -> CoreResult<Vec<String>> {
        // Calculate effective chunk size considering metadata
        let metadata_token_count = self.tokenizer.count_tokens(metadata_str)?;
        let effective_chunk_size = if self.config.base.chunk_size > metadata_token_count {
            self.config.base.chunk_size - metadata_token_count
        } else {
            // If metadata is larger than chunk size, use minimum chunk size
            self.config.base.chunk_size / 4
        };

        debug!(
            "Splitting with metadata awareness: original chunk size {}, metadata tokens {}, effective chunk size {}",
            self.config.base.chunk_size, metadata_token_count, effective_chunk_size
        );

        self.split_text_with_chunk_size(text, effective_chunk_size)
    }
}

// ============================================================================
// Type-Safe Transform Implementation
// ============================================================================

#[async_trait]
impl TypedTransform<DocumentState, NodeState> for SentenceSplitter {
    async fn transform(&self, input: TypedData<DocumentState>) -> CoreResult<TypedData<NodeState>> {
        let documents = input.documents();
        let nodes = NodeParser::parse_nodes(self, documents, false).await?;
        Ok(TypedData::from_nodes(nodes))
    }

    fn name(&self) -> &'static str {
        "SentenceSplitter"
    }

    fn description(&self) -> &'static str {
        "Splits documents into sentence-based chunks with configurable size and overlap"
    }
}

// Legacy Transform implementation has been removed.
// SentenceSplitter now only uses the type-safe TypedTransform system.

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sentence_splitter_basic() {
        let splitter = SentenceSplitter::from_defaults(50, 10).unwrap();

        let text =
            "This is the first sentence. This is the second sentence. This is the third sentence.";
        let chunks = splitter.split_text(text).unwrap();

        assert!(!chunks.is_empty());
        // Each chunk should be within token limits
        for chunk in &chunks {
            let token_count = splitter.tokenizer.count_tokens(chunk).unwrap();
            assert!(
                token_count <= 50,
                "Chunk exceeded token limit: {} tokens",
                token_count
            );
        }
    }

    #[tokio::test]
    async fn test_sentence_splitter_empty_text() {
        let splitter = SentenceSplitter::from_defaults(100, 20).unwrap();

        let chunks = splitter.split_text("").unwrap();
        assert_eq!(chunks, vec![""]);
    }

    #[tokio::test]
    async fn test_sentence_splitter_small_text() {
        let splitter = SentenceSplitter::from_defaults(100, 20).unwrap();

        let text = "Short text.";
        let chunks = splitter.split_text(text).unwrap();
        assert_eq!(chunks, vec![text]);
    }

    #[tokio::test]
    async fn test_metadata_aware_splitting() {
        let splitter = SentenceSplitter::from_defaults(20, 5).unwrap();

        let text = "This is a test sentence that should be split.";
        let metadata = "Title: Test Document\nAuthor: Test Author";

        let chunks = splitter.split_text_metadata_aware(text, metadata).unwrap();

        // Should create more chunks due to metadata overhead
        assert!(!chunks.is_empty());

        // Verify that text + metadata would fit in original chunk size
        for chunk in &chunks {
            let combined_tokens = splitter
                .tokenizer
                .count_tokens(&format!("{}\n{}", metadata, chunk))
                .unwrap();
            assert!(
                combined_tokens <= 20,
                "Combined chunk + metadata exceeded limit"
            );
        }
    }
}
