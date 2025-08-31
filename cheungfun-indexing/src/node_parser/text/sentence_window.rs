//! Sentence window node parser implementation.
//!
//! This module provides sentence window-based text parsing that creates nodes
//! for individual sentences while preserving surrounding context in metadata,
//! following the LlamaIndex SentenceWindowNodeParser pattern.

use crate::node_parser::{
    config::SentenceWindowConfig,
    text::{create_sentence_split_functions, SplitFunction},
    NodeParser, TextSplitter,
};
use async_trait::async_trait;
use cheungfun_core::{
    traits::{DocumentState, NodeState, TypedData, TypedTransform}, Document, Node, Result as CoreResult,
};
use tracing::{debug, warn};

/// Sentence window node parser that creates nodes for individual sentences.
///
/// This parser splits text into individual sentences and creates a node for each sentence.
/// Each node contains the sentence as its main content, and stores a window of surrounding
/// sentences in the metadata for context during generation.
///
/// This approach is particularly useful for:
/// - Precise retrieval at the sentence level
/// - Maintaining context through metadata windows
/// - Question-answering systems that need exact sentence matching
/// - Fine-grained document analysis
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::node_parser::text::SentenceWindowNodeParser;
/// use cheungfun_core::Document;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let parser = SentenceWindowNodeParser::new()
///         .with_window_size(2)
///         .with_window_metadata_key("context_window");
///     
///     let document = Document::new("First sentence. Second sentence. Third sentence.");
///     let nodes = parser.parse_nodes(&[document], false).await?;
///     
///     println!("Generated {} nodes", nodes.len());
///     for (i, node) in nodes.iter().enumerate() {
///         println!("Node {}: {}", i + 1, node.content);
///         if let Some(window) = node.metadata.get("context_window") {
///             println!("  Context: {}", window);
///         }
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct SentenceWindowNodeParser {
    /// Configuration for the sentence window parser.
    config: SentenceWindowConfig,
    /// Split functions for sentence splitting.
    split_functions: Vec<Box<dyn SplitFunction>>,
}

impl SentenceWindowNodeParser {
    /// Create a new sentence window node parser with default configuration.
    pub fn new() -> Self {
        let config = SentenceWindowConfig::default();
        let split_functions = create_sentence_split_functions().unwrap_or_else(|e| {
            warn!("Failed to create split functions: {}, using empty list", e);
            Vec::new()
        });

        Self {
            config,
            split_functions,
        }
    }

    /// Create a sentence window parser from configuration.
    pub fn from_config(config: SentenceWindowConfig) -> CoreResult<Self> {
        let split_functions = create_sentence_split_functions()?;

        Ok(Self {
            config,
            split_functions,
        })
    }

    /// Create a sentence window parser with default settings.
    pub fn from_defaults(
        window_size: usize,
        window_metadata_key: Option<String>,
        original_text_metadata_key: Option<String>,
    ) -> CoreResult<Self> {
        let mut config = SentenceWindowConfig::new().with_window_size(window_size);

        if let Some(key) = window_metadata_key {
            config = config.with_window_metadata_key(key);
        }

        if let Some(key) = original_text_metadata_key {
            config = config.with_original_text_metadata_key(key);
        }

        Self::from_config(config)
    }

    /// Set window size.
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.config.window_size = window_size;
        self
    }

    /// Set window metadata key.
    pub fn with_window_metadata_key<S: Into<String>>(mut self, key: S) -> Self {
        self.config.window_metadata_key = key.into();
        self
    }

    /// Set original text metadata key.
    pub fn with_original_text_metadata_key<S: Into<String>>(mut self, key: S) -> Self {
        self.config.original_text_metadata_key = key.into();
        self
    }

    /// Set sentence splitter configuration.
    pub fn with_sentence_splitter<S: Into<String>>(mut self, splitter: S) -> Self {
        self.config.sentence_splitter = Some(splitter.into());
        self
    }

    /// Split text into sentences using a simple sentence boundary detection.
    fn split_into_sentences(&self, text: &str) -> CoreResult<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Simple sentence splitting based on sentence-ending punctuation
        // This is a basic implementation - for production use, consider using
        // a proper NLP library like spacy or nltk
        let sentence_endings = regex::Regex::new(r"[.!?]+\s+").unwrap();

        let mut sentences = Vec::new();
        let mut last_end = 0;

        for mat in sentence_endings.find_iter(text) {
            let sentence_end = mat.end();
            let sentence = text[last_end..sentence_end].trim();
            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
            }
            last_end = sentence_end;
        }

        // Add the remaining text as the last sentence
        if last_end < text.len() {
            let remaining = text[last_end..].trim();
            if !remaining.is_empty() {
                sentences.push(remaining.to_string());
            }
        }

        // If no sentences were found, return the original text
        if sentences.is_empty() {
            sentences.push(text.to_string());
        }

        Ok(sentences)
    }

    /// Build window nodes from documents.
    fn build_window_nodes_from_documents(&self, documents: &[Document]) -> CoreResult<Vec<Node>> {
        let mut all_nodes = Vec::new();

        for document in documents {
            let text = &document.content;
            let sentences = self.split_into_sentences(text)?;

            debug!(
                "Split document {} into {} sentences",
                document.id,
                sentences.len()
            );

            // Create nodes for each sentence
            for (i, sentence) in sentences.iter().enumerate() {
                // Calculate window boundaries
                let window_start = i.saturating_sub(self.config.window_size);
                let window_end = (i + self.config.window_size + 1).min(sentences.len());

                // Create window content from surrounding sentences
                let window_sentences = &sentences[window_start..window_end];
                let window_content = window_sentences.join(" ");

                // Calculate approximate character positions
                let start_offset = sentences[..i].iter().map(|s| s.len() + 1).sum::<usize>();
                let end_offset = start_offset + sentence.len();

                let chunk_info = cheungfun_core::types::ChunkInfo::with_char_indices(
                    start_offset,
                    end_offset,
                    i,
                );

                let mut node = Node::new(sentence.clone(), document.id, chunk_info);

                // Copy metadata from document
                node.metadata = document.metadata.clone();

                // Add sentence window specific metadata
                node.metadata.insert(
                    self.config.window_metadata_key.clone(),
                    window_content.into(),
                );
                node.metadata.insert(
                    self.config.original_text_metadata_key.clone(),
                    sentence.clone().into(),
                );
                node.metadata.insert("sentence_index".to_string(), i.into());
                node.metadata
                    .insert("total_sentences".to_string(), sentences.len().into());
                node.metadata
                    .insert("window_size".to_string(), self.config.window_size.into());
                node.metadata
                    .insert("splitter_type".to_string(), "sentence_window".into());

                // Set relationships if configured
                if self.config.base.include_prev_next_rel {
                    if i > 0 {
                        // Previous relationship would be set here
                        // This requires coordination with other nodes
                    }
                    if i < sentences.len() - 1 {
                        // Next relationship would be set here
                        // This requires coordination with other nodes
                    }
                }

                all_nodes.push(node);
            }
        }

        debug!(
            "Generated {} window nodes from {} documents",
            all_nodes.len(),
            documents.len()
        );
        Ok(all_nodes)
    }
}

impl Default for SentenceWindowNodeParser {
    fn default() -> Self {
        Self::new()
    }
}

impl TextSplitter for SentenceWindowNodeParser {
    fn split_text(&self, text: &str) -> CoreResult<Vec<String>> {
        // For TextSplitter interface, just return the sentences
        self.split_into_sentences(text)
    }
}

#[async_trait]
impl NodeParser for SentenceWindowNodeParser {
    async fn parse_nodes(
        &self,
        documents: &[Document],
        _show_progress: bool,
    ) -> CoreResult<Vec<Node>> {
        self.build_window_nodes_from_documents(documents)
    }
}

// ============================================================================
// Type-Safe Transform Implementation
// ============================================================================

#[async_trait]
impl TypedTransform<DocumentState, NodeState> for SentenceWindowNodeParser {
    async fn transform(&self, input: TypedData<DocumentState>) -> CoreResult<TypedData<NodeState>> {
        let documents = input.documents();
        let nodes = NodeParser::parse_nodes(self, documents, false).await?;
        Ok(TypedData::from_nodes(nodes))
    }

    fn name(&self) -> &'static str {
        "SentenceWindowNodeParser"
    }

    fn description(&self) -> &'static str {
        "Creates nodes for individual sentences while preserving surrounding context in metadata"
    }
}

// ============================================================================
// Legacy Transform Implementation (Backward Compatibility)
// ============================================================================

// Legacy Transform implementation has been removed.
// SentenceWindowNodeParser now only uses the type-safe TypedTransform system.

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::Document;

    #[tokio::test]
    async fn test_sentence_window_basic() {
        let parser = SentenceWindowNodeParser::new().with_window_size(1);

        let document = Document::new("First sentence. Second sentence. Third sentence.");
        let nodes = <SentenceWindowNodeParser as crate::node_parser::NodeParser>::parse_nodes(
            &parser,
            &[document],
            false,
        )
        .await
        .unwrap();

        assert_eq!(nodes.len(), 3);

        // Check first node
        assert_eq!(nodes[0].content, "First sentence.");
        assert!(nodes[0].metadata.contains_key("window"));
        assert!(nodes[0].metadata.contains_key("original_text"));
        assert_eq!(
            nodes[0]
                .metadata
                .get("sentence_index")
                .unwrap()
                .as_u64()
                .unwrap(),
            0
        );

        // Check middle node
        assert_eq!(nodes[1].content, "Second sentence.");
        assert_eq!(
            nodes[1]
                .metadata
                .get("sentence_index")
                .unwrap()
                .as_u64()
                .unwrap(),
            1
        );

        // Check last node
        assert_eq!(nodes[2].content, "Third sentence.");
        assert_eq!(
            nodes[2]
                .metadata
                .get("sentence_index")
                .unwrap()
                .as_u64()
                .unwrap(),
            2
        );
    }

    #[tokio::test]
    async fn test_sentence_window_with_larger_window() {
        let parser = SentenceWindowNodeParser::new().with_window_size(2);

        let document = Document::new("One. Two. Three. Four. Five.");
        let nodes = <SentenceWindowNodeParser as crate::node_parser::NodeParser>::parse_nodes(
            &parser,
            &[document],
            false,
        )
        .await
        .unwrap();

        assert_eq!(nodes.len(), 5);

        // Check middle node (index 2) - should have full window
        let middle_node = &nodes[2];
        assert_eq!(middle_node.content, "Three.");

        let window_content = middle_node
            .metadata
            .get("window")
            .unwrap()
            .as_str()
            .unwrap();
        // Window should include: One. Two. Three. Four. Five. (all sentences due to window_size=2)
        assert!(window_content.contains("One."));
        assert!(window_content.contains("Two."));
        assert!(window_content.contains("Three."));
        assert!(window_content.contains("Four."));
        assert!(window_content.contains("Five."));
    }

    #[tokio::test]
    async fn test_sentence_window_empty_text() {
        let parser = SentenceWindowNodeParser::new();

        let document = Document::new("");
        let nodes = <SentenceWindowNodeParser as crate::node_parser::NodeParser>::parse_nodes(
            &parser,
            &[document],
            false,
        )
        .await
        .unwrap();

        assert!(nodes.is_empty());
    }

    #[tokio::test]
    async fn test_sentence_window_single_sentence() {
        let parser = SentenceWindowNodeParser::new().with_window_size(3);

        let document = Document::new("Only one sentence.");
        let nodes = <SentenceWindowNodeParser as crate::node_parser::NodeParser>::parse_nodes(
            &parser,
            &[document],
            false,
        )
        .await
        .unwrap();

        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].content, "Only one sentence.");

        let window_content = nodes[0].metadata.get("window").unwrap().as_str().unwrap();
        assert_eq!(window_content, "Only one sentence.");
    }

    #[tokio::test]
    async fn test_text_splitter_interface() {
        let parser = SentenceWindowNodeParser::new();

        let text = "First sentence. Second sentence. Third sentence.";
        let sentences = parser.split_text(text).unwrap();

        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "First sentence.");
        assert_eq!(sentences[1], "Second sentence.");
        assert_eq!(sentences[2], "Third sentence.");
    }

    #[tokio::test]
    async fn test_custom_metadata_keys() {
        let parser = SentenceWindowNodeParser::new()
            .with_window_size(1)
            .with_window_metadata_key("context")
            .with_original_text_metadata_key("original");

        let document = Document::new("First sentence. Second sentence.");
        let nodes = <SentenceWindowNodeParser as crate::node_parser::NodeParser>::parse_nodes(
            &parser,
            &[document],
            false,
        )
        .await
        .unwrap();

        assert_eq!(nodes.len(), 2);
        assert!(nodes[0].metadata.contains_key("context"));
        assert!(nodes[0].metadata.contains_key("original"));
        assert!(!nodes[0].metadata.contains_key("window"));
        assert!(!nodes[0].metadata.contains_key("original_text"));
    }
}
