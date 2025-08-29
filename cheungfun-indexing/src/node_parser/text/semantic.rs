//! Semantic splitter implementation.
//!
//! This module provides semantic-based text splitting that uses embeddings
//! to group semantically related sentences together, following the LlamaIndex
//! SemanticSplitterNodeParser pattern.

use crate::node_parser::{
    config::SemanticSplitterConfig,
    text::{create_sentence_split_functions, SplitFunction},
    NodeParser, TextSplitter,
};
use async_trait::async_trait;
use cheungfun_core::{
    traits::{Embedder, Transform, TransformInput},
    CheungfunError, Document, Node, Result as CoreResult,
};
use std::sync::Arc;
use tracing::{debug, warn};

/// Sentence combination for semantic analysis.
#[derive(Debug, Clone)]
struct SentenceCombination {
    /// Original sentence text.
    sentence: String,
    /// Index in the original sentence list.
    index: usize,
    /// Combined sentence with buffer context.
    combined_sentence: String,
    /// Embedding of the combined sentence.
    combined_sentence_embedding: Vec<f32>,
}

/// Semantic splitter that groups semantically related sentences.
///
/// This splitter uses embeddings to calculate semantic similarity between
/// sentence groups and creates chunks based on semantic boundaries rather
/// than simple token or character limits.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::node_parser::{text::SemanticSplitter, TextSplitter};
/// use cheungfun_integrations::embedders::fastembed::FastEmbedder;
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let embedder = Arc::new(FastEmbedder::new().await?);
///     let splitter = SemanticSplitter::new(embedder)
///         .with_buffer_size(2)
///         .with_breakpoint_percentile_threshold(90.0);
///     
///     let text = "This is the first sentence. This is related to the first. Now we change topics. This is about something completely different.";
///     let chunks = splitter.split_text(text).await?;
///     
///     println!("Split into {} semantic chunks", chunks.len());
///     for (i, chunk) in chunks.iter().enumerate() {
///         println!("Chunk {}: {}", i + 1, chunk);
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct SemanticSplitter {
    /// Configuration for the semantic splitter.
    config: SemanticSplitterConfig,
    /// Embedder for calculating semantic similarity.
    embedder: Arc<dyn Embedder>,
    /// Split functions for sentence splitting.
    split_functions: Vec<Box<dyn SplitFunction>>,
}

impl SemanticSplitter {
    /// Create a new semantic splitter with the given embedder.
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        let config = SemanticSplitterConfig::default();
        let split_functions = create_sentence_split_functions().unwrap_or_else(|e| {
            warn!("Failed to create split functions: {}, using empty list", e);
            Vec::new()
        });

        Self {
            config,
            embedder,
            split_functions,
        }
    }

    /// Create a semantic splitter from configuration.
    pub fn from_config(
        config: SemanticSplitterConfig,
        embedder: Arc<dyn Embedder>,
    ) -> CoreResult<Self> {
        let split_functions = create_sentence_split_functions()?;

        Ok(Self {
            config,
            embedder,
            split_functions,
        })
    }

    /// Set buffer size for sentence grouping.
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.config.buffer_size = buffer_size;
        self
    }

    /// Set breakpoint percentile threshold.
    pub fn with_breakpoint_percentile_threshold(mut self, threshold: f32) -> Self {
        self.config.breakpoint_percentile_threshold = threshold;
        self
    }

    /// Set sentence splitter configuration.
    pub fn with_sentence_splitter<S: Into<String>>(mut self, splitter: S) -> Self {
        self.config.sentence_splitter = Some(splitter.into());
        self
    }

    /// Split text into sentences using configured split functions.
    fn split_into_sentences(&self, text: &str) -> CoreResult<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Use the first split function that produces multiple splits
        for split_fn in &self.split_functions {
            let splits = split_fn.split(text)?;
            if splits.len() > 1 {
                return Ok(splits.into_iter().map(|s| s.trim().to_string()).collect());
            }
        }

        // If no split function works, return the original text as a single sentence
        Ok(vec![text.to_string()])
    }

    /// Build sentence groups with buffer context.
    fn build_sentence_groups(&self, sentences: Vec<String>) -> Vec<SentenceCombination> {
        let mut combinations = Vec::new();

        for (i, sentence) in sentences.iter().enumerate() {
            let mut combined_sentence = String::new();

            // Add buffer sentences before current sentence
            for j in (i.saturating_sub(self.config.buffer_size))..i {
                if j < sentences.len() {
                    combined_sentence.push_str(&sentences[j]);
                    combined_sentence.push(' ');
                }
            }

            // Add current sentence
            combined_sentence.push_str(sentence);

            // Add buffer sentences after current sentence
            for j in (i + 1)..=(i + self.config.buffer_size).min(sentences.len() - 1) {
                combined_sentence.push(' ');
                combined_sentence.push_str(&sentences[j]);
            }

            combinations.push(SentenceCombination {
                sentence: sentence.clone(),
                index: i,
                combined_sentence: combined_sentence.trim().to_string(),
                combined_sentence_embedding: Vec::new(), // Will be filled later
            });
        }

        combinations
    }

    /// Calculate distances between consecutive sentence groups.
    fn calculate_distances(&self, combinations: &[SentenceCombination]) -> Vec<f32> {
        let mut distances = Vec::new();

        for i in 0..combinations.len().saturating_sub(1) {
            let embedding1 = &combinations[i].combined_sentence_embedding;
            let embedding2 = &combinations[i + 1].combined_sentence_embedding;

            let similarity = self.calculate_cosine_similarity(embedding1, embedding2);
            let distance = 1.0 - similarity;
            distances.push(distance);
        }

        distances
    }

    /// Calculate cosine similarity between two embeddings.
    fn calculate_cosine_similarity(&self, embedding1: &[f32], embedding2: &[f32]) -> f32 {
        if embedding1.len() != embedding2.len() {
            warn!(
                "Embedding dimensions don't match: {} vs {}",
                embedding1.len(),
                embedding2.len()
            );
            return 0.0;
        }

        let dot_product: f32 = embedding1
            .iter()
            .zip(embedding2.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    /// Build semantic chunks based on distance breakpoints.
    fn build_semantic_chunks(
        &self,
        combinations: &[SentenceCombination],
        distances: &[f32],
    ) -> Vec<String> {
        if distances.is_empty() {
            // Single sentence or no distances calculated
            return combinations.iter().map(|c| c.sentence.clone()).collect();
        }

        // Calculate breakpoint threshold using percentile
        let mut sorted_distances = distances.to_vec();
        sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let percentile_index = ((self.config.breakpoint_percentile_threshold / 100.0)
            * sorted_distances.len() as f32) as usize;
        let breakpoint_threshold = sorted_distances
            .get(percentile_index.min(sorted_distances.len() - 1))
            .copied()
            .unwrap_or(0.5);

        debug!("Breakpoint threshold: {}", breakpoint_threshold);

        // Find breakpoint indices
        let breakpoint_indices: Vec<usize> = distances
            .iter()
            .enumerate()
            .filter_map(|(i, &distance)| {
                if distance > breakpoint_threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        debug!(
            "Found {} breakpoints at indices: {:?}",
            breakpoint_indices.len(),
            breakpoint_indices
        );

        // Build chunks based on breakpoints
        let mut chunks = Vec::new();
        let mut start_index = 0;

        for &breakpoint_index in &breakpoint_indices {
            let end_index = breakpoint_index + 1;
            let chunk_sentences: Vec<String> = combinations[start_index..end_index]
                .iter()
                .map(|c| c.sentence.clone())
                .collect();

            if !chunk_sentences.is_empty() {
                chunks.push(chunk_sentences.join(" "));
            }
            start_index = end_index;
        }

        // Add remaining sentences as the last chunk
        if start_index < combinations.len() {
            let chunk_sentences: Vec<String> = combinations[start_index..]
                .iter()
                .map(|c| c.sentence.clone())
                .collect();

            if !chunk_sentences.is_empty() {
                chunks.push(chunk_sentences.join(" "));
            }
        }

        // If no breakpoints found, return all sentences as one chunk
        if chunks.is_empty() {
            chunks.push(
                combinations
                    .iter()
                    .map(|c| c.sentence.clone())
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }

        chunks
    }
}

impl SemanticSplitter {
    /// Split text using semantic analysis (async version).
    pub async fn split_text_async(&self, text: &str) -> CoreResult<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        debug!(
            "Starting semantic splitting for text of length {}",
            text.len()
        );

        // Step 1: Split text into sentences
        let sentences = self.split_into_sentences(text)?;
        debug!("Split into {} sentences", sentences.len());

        if sentences.len() <= 1 {
            return Ok(sentences);
        }

        // Step 2: Build sentence groups with buffer context
        let mut combinations = self.build_sentence_groups(sentences);
        debug!("Built {} sentence combinations", combinations.len());

        // Step 3: Generate embeddings for combined sentences
        let combined_texts: Vec<&str> = combinations
            .iter()
            .map(|c| c.combined_sentence.as_str())
            .collect();

        let embeddings = self.embedder.embed_batch(combined_texts).await?;
        debug!("Generated {} embeddings", embeddings.len());

        // Step 4: Assign embeddings to combinations
        for (i, embedding) in embeddings.into_iter().enumerate() {
            if let Some(combination) = combinations.get_mut(i) {
                combination.combined_sentence_embedding = embedding;
            }
        }

        // Step 5: Calculate distances between consecutive sentence groups
        let distances = self.calculate_distances(&combinations);
        debug!("Calculated {} distances", distances.len());

        // Step 6: Build semantic chunks based on distance breakpoints
        let chunks = self.build_semantic_chunks(&combinations, &distances);
        debug!("Created {} semantic chunks", chunks.len());

        Ok(chunks)
    }
}

impl TextSplitter for SemanticSplitter {
    fn split_text(&self, text: &str) -> CoreResult<Vec<String>> {
        // For the sync version, we need to use a blocking approach
        // This is a limitation - semantic splitting really needs async for embeddings
        // For now, return a simple sentence split as fallback
        warn!("Using sync split_text for SemanticSplitter - falling back to sentence splitting");
        self.split_into_sentences(text)
    }
}

#[async_trait]
impl NodeParser for SemanticSplitter {
    async fn parse_nodes(
        &self,
        documents: &[Document],
        _show_progress: bool,
    ) -> CoreResult<Vec<Node>> {
        let mut all_nodes = Vec::new();

        for document in documents {
            let text = document.content.clone();
            let chunks = self.split_text_async(&text).await?;

            for (i, chunk) in chunks.into_iter().enumerate() {
                let chunk_info = cheungfun_core::types::ChunkInfo::with_char_indices(
                    0,           // start_offset - would need to calculate actual positions
                    chunk.len(), // end_offset
                    i,           // chunk_index
                );

                let mut node = Node::new(chunk, document.id, chunk_info);

                // Copy metadata from document
                node.metadata = document.metadata.clone();

                // Add chunk-specific metadata
                node.metadata.insert("chunk_index".to_string(), i.into());
                node.metadata.insert(
                    "source_document_id".to_string(),
                    document.id.to_string().into(),
                );
                node.metadata
                    .insert("splitter_type".to_string(), "semantic".into());
                node.metadata
                    .insert("buffer_size".to_string(), self.config.buffer_size.into());
                node.metadata.insert(
                    "breakpoint_threshold".to_string(),
                    self.config.breakpoint_percentile_threshold.into(),
                );

                // Set relationships if configured
                if self.config.base.include_prev_next_rel && i > 0 {
                    // Previous relationship would be set here
                    // This requires a more complex node relationship system
                }

                all_nodes.push(node);
            }
        }

        debug!(
            "Parsed {} documents into {} nodes",
            documents.len(),
            all_nodes.len()
        );
        Ok(all_nodes)
    }
}

#[async_trait]
impl Transform for SemanticSplitter {
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
                // SemanticSplitter only processes documents, not nodes
                Err(CheungfunError::Validation {
                    message: "SemanticSplitter only accepts documents as input".into(),
                })
            }
        }
    }

    fn name(&self) -> &'static str {
        "SemanticSplitter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::traits::Embedder;
    use std::sync::Arc;

    // Mock embedder for testing
    #[derive(Debug)]
    struct MockEmbedder {
        dimension: usize,
    }

    impl MockEmbedder {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, text: &str) -> CoreResult<Vec<f32>> {
            // Create a simple hash-based embedding for testing
            let mut embedding = vec![0.0; self.dimension];
            let text_hash = text
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));

            for (i, value) in embedding.iter_mut().enumerate() {
                let seed = (text_hash.wrapping_add(i as u64)) as f32;
                *value = (seed * 0.001).sin();
            }

            // Normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for value in embedding.iter_mut() {
                    *value /= norm;
                }
            }

            Ok(embedding)
        }

        async fn embed_batch(&self, texts: Vec<&str>) -> CoreResult<Vec<Vec<f32>>> {
            let mut embeddings = Vec::new();
            for text in texts {
                embeddings.push(self.embed(text).await?);
            }
            Ok(embeddings)
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn model_name(&self) -> &str {
            "mock-embedder"
        }
    }

    #[tokio::test]
    async fn test_semantic_splitter_basic() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let splitter = SemanticSplitter::new(embedder);

        let text =
            "This is the first sentence. This is the second sentence. This is the third sentence.";
        let chunks = splitter.split_text_async(text).await.unwrap();

        assert!(!chunks.is_empty());
        assert!(chunks.len() <= 3); // Should not exceed number of sentences
    }

    #[tokio::test]
    async fn test_semantic_splitter_empty_text() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let splitter = SemanticSplitter::new(embedder);

        let chunks = splitter.split_text_async("").await.unwrap();
        assert!(chunks.is_empty());
    }

    #[tokio::test]
    async fn test_semantic_splitter_single_sentence() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let splitter = SemanticSplitter::new(embedder);

        let text = "This is a single sentence.";
        let chunks = splitter.split_text_async(text).await.unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[tokio::test]
    async fn test_semantic_splitter_with_buffer() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let splitter = SemanticSplitter::new(embedder).with_buffer_size(2);

        let text = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        let chunks = splitter.split_text_async(text).await.unwrap();

        assert!(!chunks.is_empty());
        // With buffer size 2, each sentence gets context from 2 sentences before and after
    }

    #[tokio::test]
    async fn test_node_parser_implementation() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let splitter = SemanticSplitter::new(embedder);

        let document = Document::new("First sentence. Second sentence. Third sentence.");
        let nodes = <SemanticSplitter as crate::node_parser::NodeParser>::parse_nodes(
            &splitter,
            &[document],
            false,
        )
        .await
        .unwrap();

        assert!(!nodes.is_empty());
        for (i, node) in nodes.iter().enumerate() {
            assert!(node.metadata.contains_key("chunk_index"));
            assert!(node.metadata.contains_key("splitter_type"));
            assert_eq!(
                node.metadata.get("chunk_index").unwrap().as_u64().unwrap(),
                i as u64
            );
        }
    }
}
