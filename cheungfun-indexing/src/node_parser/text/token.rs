//! Token-based text splitter implementation.
//!
//! This module provides the TokenTextSplitter, which splits text based on
//! raw token counts rather than semantic boundaries. It's useful when you
//! need precise control over token limits.

use super::{create_token_split_functions, SplitFunction, Tokenizer, WhitespaceTokenizer};
use crate::node_parser::{
    callbacks::{CallbackManager, EventPayload},
    config::TokenTextSplitterConfig,
    utils::{build_nodes_from_splits, get_id_function},
    MetadataAwareTextSplitter, NodeParser, TextSplitter,
};
use async_trait::async_trait;
use cheungfun_core::{
    traits::{Transform, TransformInput},
    CheungfunError, Document, Node, Result as CoreResult,
};
use std::sync::Arc;
use tracing::debug;

/// Token-based text splitter.
///
/// This splitter focuses on raw token counts and uses a simpler splitting
/// strategy compared to SentenceSplitter. It's useful when you need precise
/// control over token limits without concern for semantic boundaries.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::node_parser::{text::TokenTextSplitter, TextSplitter};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let splitter = TokenTextSplitter::from_defaults(1000, 200)?;
///     
///     let text = "This is a sample text that will be split based on token counts.";
///     let chunks = splitter.split_text(text)?;
///     
///     println!("Split into {} chunks", chunks.len());
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct TokenTextSplitter {
    config: TokenTextSplitterConfig,
    tokenizer: Arc<dyn Tokenizer>,
    split_functions: Vec<Box<dyn SplitFunction>>,
    callback_manager: Option<CallbackManager>,
}

impl TokenTextSplitter {
    /// Create a new token text splitter with the given configuration.
    pub fn new(config: TokenTextSplitterConfig) -> CoreResult<Self> {
        let tokenizer = Arc::new(WhitespaceTokenizer); // Default tokenizer
        let split_functions = create_token_split_functions()?;

        Ok(Self {
            config,
            tokenizer,
            split_functions,
            callback_manager: None,
        })
    }

    /// Create a token text splitter with default configuration.
    pub fn from_defaults(chunk_size: usize, chunk_overlap: usize) -> CoreResult<Self> {
        let config = TokenTextSplitterConfig::new(chunk_size, chunk_overlap);
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

    /// Split text using token-based strategy.
    fn split_text_internal(&self, text: &str, chunk_size: usize) -> CoreResult<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![text.to_string()]);
        }

        let token_count = self.tokenizer.count_tokens(text)?;
        if token_count <= chunk_size {
            return Ok(vec![text.to_string()]);
        }

        debug!(
            "Splitting text of {} tokens with chunk size {}",
            token_count, chunk_size
        );

        // Use split functions to break down the text
        let splits = self.get_splits_by_functions(text)?;
        let chunks = self.merge_splits_by_tokens(splits, chunk_size)?;

        debug!("Created {} chunks from text", chunks.len());
        Ok(chunks)
    }

    /// Apply split functions to get text splits.
    fn get_splits_by_functions(&self, text: &str) -> CoreResult<Vec<String>> {
        for split_fn in &self.split_functions {
            let splits = split_fn.split(text)?;
            if splits.len() > 1 {
                return Ok(splits
                    .into_iter()
                    .filter(|s| !s.trim().is_empty())
                    .collect());
            }
        }

        // If no split function worked, return the original text
        Ok(vec![text.to_string()])
    }

    /// Merge splits based on token counts.
    fn merge_splits_by_tokens(
        &self,
        splits: Vec<String>,
        chunk_size: usize,
    ) -> CoreResult<Vec<String>> {
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_token_count = 0;

        for split in splits {
            let split_token_count = self.tokenizer.count_tokens(&split)?;

            if split_token_count > chunk_size {
                // Split is too large, need to split it further
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.join(""));
                    current_chunk.clear();
                    current_token_count = 0;
                }

                // Recursively split this large piece
                let sub_chunks = self.split_text_internal(&split, chunk_size)?;
                chunks.extend(sub_chunks);
            } else if current_token_count + split_token_count > chunk_size {
                // Adding this split would exceed chunk size
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.join(""));

                    // Start new chunk with overlap
                    current_chunk = self.create_overlap_chunk(&current_chunk)?;
                    current_token_count = self.calculate_chunk_tokens(&current_chunk)?;
                }

                current_chunk.push(split);
                current_token_count += split_token_count;
            } else {
                // Add split to current chunk
                current_chunk.push(split);
                current_token_count += split_token_count;
            }
        }

        // Add final chunk if it has content
        if !current_chunk.is_empty() {
            chunks.push(current_chunk.join(""));
        }

        Ok(chunks)
    }

    /// Create overlap chunk from the end of the previous chunk.
    fn create_overlap_chunk(&self, previous_chunk: &[String]) -> CoreResult<Vec<String>> {
        let overlap_limit = self.config.base.chunk_overlap;
        let mut overlap_chunk = Vec::new();
        let mut overlap_tokens = 0;

        // Add splits from the end until we reach overlap limit
        for split in previous_chunk.iter().rev() {
            let split_tokens = self.tokenizer.count_tokens(split)?;
            if overlap_tokens + split_tokens <= overlap_limit {
                overlap_chunk.insert(0, split.clone());
                overlap_tokens += split_tokens;
            } else {
                break;
            }
        }

        Ok(overlap_chunk)
    }

    /// Calculate total token count for a chunk.
    fn calculate_chunk_tokens(&self, chunk: &[String]) -> CoreResult<usize> {
        let combined_text = chunk.join("");
        self.tokenizer.count_tokens(&combined_text)
    }
}

#[async_trait]
impl NodeParser for TokenTextSplitter {
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
impl TextSplitter for TokenTextSplitter {
    fn split_text(&self, text: &str) -> CoreResult<Vec<String>> {
        self.split_text_internal(text, self.config.base.chunk_size)
    }
}

#[async_trait]
impl MetadataAwareTextSplitter for TokenTextSplitter {
    fn split_text_metadata_aware(&self, text: &str, metadata_str: &str) -> CoreResult<Vec<String>> {
        // Calculate effective chunk size considering metadata
        let metadata_token_count = self.tokenizer.count_tokens(metadata_str)?;
        let metadata_format_len = self.config.metadata_format_len;
        let total_metadata_tokens = metadata_token_count + metadata_format_len;

        let effective_chunk_size = if self.config.base.chunk_size > total_metadata_tokens {
            self.config.base.chunk_size - total_metadata_tokens
        } else {
            // If metadata is larger than chunk size, use minimum chunk size
            self.config.base.chunk_size / 4
        };

        debug!(
            "Token splitting with metadata awareness: original chunk size {}, metadata tokens {}, effective chunk size {}",
            self.config.base.chunk_size, total_metadata_tokens, effective_chunk_size
        );

        self.split_text_internal(text, effective_chunk_size)
    }
}

#[async_trait]
impl Transform for TokenTextSplitter {
    async fn transform(&self, input: TransformInput) -> CoreResult<Vec<Node>> {
        match input {
            TransformInput::Document(document) => {
                // Use the existing NodeParser implementation
                NodeParser::parse_nodes(self, &[document], false).await
            }
            TransformInput::Documents(documents) => {
                // Use the existing NodeParser implementation for batch processing
                NodeParser::parse_nodes(self, &documents, false).await
            }
            TransformInput::Node(_) | TransformInput::Nodes(_) => {
                // TokenTextSplitter only processes documents, not nodes
                Err(CheungfunError::Validation {
                    message: "TokenTextSplitter only accepts documents as input".into(),
                })
            }
        }
    }

    fn name(&self) -> &'static str {
        "TokenTextSplitter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_text_splitter_basic() {
        let splitter = TokenTextSplitter::from_defaults(10, 2).unwrap();

        let text = "This is a test sentence with many words that should be split.";
        let chunks = splitter.split_text(text).unwrap();

        assert!(!chunks.is_empty());

        // Each chunk should be within token limits
        for chunk in &chunks {
            let token_count = splitter.tokenizer.count_tokens(chunk).unwrap();
            assert!(
                token_count <= 10,
                "Chunk exceeded token limit: {} tokens",
                token_count
            );
        }
    }

    #[tokio::test]
    async fn test_token_text_splitter_empty() {
        let splitter = TokenTextSplitter::from_defaults(100, 20).unwrap();

        let chunks = splitter.split_text("").unwrap();
        assert_eq!(chunks, vec![""]);
    }

    #[tokio::test]
    async fn test_token_text_splitter_small() {
        let splitter = TokenTextSplitter::from_defaults(100, 20).unwrap();

        let text = "Small text.";
        let chunks = splitter.split_text(text).unwrap();
        assert_eq!(chunks, vec![text]);
    }
}
