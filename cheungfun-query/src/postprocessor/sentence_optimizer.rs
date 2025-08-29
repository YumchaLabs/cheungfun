//! Sentence-level embedding optimization for contextual compression.
//!
//! This module implements LlamaIndex's SentenceEmbeddingOptimizer approach,
//! which compresses node content by selecting the most relevant sentences
//! based on embedding similarity with the query.

use super::{CompressionMetrics, DocumentCompressor, NodePostprocessor, SentenceEmbeddingConfig};
use async_trait::async_trait;
use cheungfun_core::{Embedder, Result, ScoredNode};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, warn};

/// Sentence-level embedding optimizer for contextual compression.
///
/// Based on LlamaIndex's SentenceEmbeddingOptimizer, this compressor:
/// 1. Splits node content into sentences
/// 2. Generates embeddings for each sentence
/// 3. Calculates similarity with query embedding
/// 4. Keeps only the most relevant sentences
/// 5. Reconstructs compressed content with context
#[derive(Debug)]
pub struct SentenceEmbeddingOptimizer {
    /// Embedder for generating sentence embeddings.
    embedder: Arc<dyn Embedder>,

    /// Configuration for optimization.
    config: SentenceEmbeddingConfig,
}

impl SentenceEmbeddingOptimizer {
    /// Create a new sentence embedding optimizer.
    pub fn new(embedder: Arc<dyn Embedder>, config: SentenceEmbeddingConfig) -> Self {
        Self { embedder, config }
    }

    /// Create with default configuration.
    pub fn with_default_config(embedder: Arc<dyn Embedder>) -> Self {
        Self::new(embedder, SentenceEmbeddingConfig::default())
    }

    /// Split text into sentences using simple heuristics.
    ///
    /// In a production implementation, you might want to use a more
    /// sophisticated sentence tokenizer like NLTK's punkt tokenizer.
    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        text.split(&['.', '!', '?'])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() > 10) // Filter very short sentences
            .collect()
    }

    /// Calculate similarity between query and sentence embeddings.
    fn calculate_similarities(
        &self,
        query_embedding: &[f32],
        sentence_embeddings: &[Vec<f32>],
    ) -> Vec<f32> {
        sentence_embeddings
            .iter()
            .map(|sent_emb| self.cosine_similarity(query_embedding, sent_emb))
            .collect()
    }

    /// Calculate cosine similarity between two embeddings.
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Select top sentences based on similarity scores.
    fn select_top_sentences(
        &self,
        sentences: &[String],
        similarities: &[f32],
    ) -> Result<Vec<usize>> {
        let mut indexed_similarities: Vec<(usize, f32)> = similarities
            .iter()
            .enumerate()
            .map(|(i, &sim)| (i, sim))
            .collect();

        // Sort by similarity score (descending)
        indexed_similarities
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected_indices = Vec::new();

        // Apply percentile cutoff
        if let Some(percentile) = self.config.percentile_cutoff {
            let num_to_keep = ((sentences.len() as f32) * percentile).ceil() as usize;
            let top_indices: Vec<usize> = indexed_similarities
                .iter()
                .take(num_to_keep)
                .map(|(idx, _)| *idx)
                .collect();
            selected_indices.extend(top_indices);
        }

        // Apply threshold cutoff
        if let Some(threshold) = self.config.threshold_cutoff {
            let threshold_indices: Vec<usize> = indexed_similarities
                .iter()
                .filter(|(_, sim)| *sim >= threshold)
                .map(|(idx, _)| *idx)
                .collect();

            if selected_indices.is_empty() {
                selected_indices = threshold_indices;
            } else {
                // Intersection of percentile and threshold results
                selected_indices.retain(|idx| threshold_indices.contains(idx));
            }
        }

        // If no cutoffs specified, keep all sentences
        if self.config.percentile_cutoff.is_none() && self.config.threshold_cutoff.is_none() {
            selected_indices = (0..sentences.len()).collect();
        }

        // Ensure we have at least one sentence
        if selected_indices.is_empty() && !indexed_similarities.is_empty() {
            warn!("No sentences met the criteria, keeping the best one");
            selected_indices.push(indexed_similarities[0].0);
        }

        Ok(selected_indices)
    }

    /// Add context sentences around selected sentences.
    fn add_context_sentences(
        &self,
        selected_indices: &[usize],
        total_sentences: usize,
    ) -> Vec<usize> {
        let context_before = self.config.context_before.unwrap_or(1);
        let context_after = self.config.context_after.unwrap_or(1);

        let mut final_indices = std::collections::HashSet::new();

        for &idx in selected_indices {
            // Add the sentence itself
            final_indices.insert(idx);

            // Add context before
            for i in 1..=context_before {
                if idx >= i {
                    final_indices.insert(idx - i);
                }
            }

            // Add context after
            for i in 1..=context_after {
                if idx + i < total_sentences {
                    final_indices.insert(idx + i);
                }
            }
        }

        let mut result: Vec<usize> = final_indices.into_iter().collect();
        result.sort();
        result
    }

    /// Reconstruct text from selected sentence indices.
    fn reconstruct_text(&self, sentences: &[String], selected_indices: &[usize]) -> String {
        selected_indices
            .iter()
            .map(|&idx| sentences.get(idx).cloned().unwrap_or_default())
            .collect::<Vec<String>>()
            .join(". ")
            + "." // Add final period
    }

    /// Compress a single node's content.
    async fn compress_node_content(
        &self,
        content: &str,
        query_embedding: &[f32],
    ) -> Result<(String, CompressionMetrics)> {
        let start_time = Instant::now();
        let original_length = content.len();

        // Split into sentences
        let sentences = self.split_into_sentences(content);
        let sentences_processed = sentences.len();

        if sentences.is_empty() {
            let metrics = CompressionMetrics::new(
                original_length,
                original_length,
                1.0,
                start_time.elapsed().as_millis() as u64,
                0,
                0,
            );
            return Ok((content.to_string(), metrics));
        }

        // Limit sentences for performance
        let sentences = if let Some(max_sentences) = self.config.max_sentences_per_node {
            if sentences.len() > max_sentences {
                sentences.into_iter().take(max_sentences).collect()
            } else {
                sentences
            }
        } else {
            sentences
        };

        // Generate embeddings for sentences
        let sentence_refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
        let sentence_embeddings = self.embedder.embed_batch(sentence_refs).await?;

        // Calculate similarities
        let similarities = self.calculate_similarities(query_embedding, &sentence_embeddings);

        // Select top sentences
        let selected_indices = self.select_top_sentences(&sentences, &similarities)?;

        // Add context
        let final_indices = self.add_context_sentences(&selected_indices, sentences.len());

        // Reconstruct text
        let compressed_content = self.reconstruct_text(&sentences, &final_indices);
        let compressed_length = compressed_content.len();

        // Calculate average relevance score
        let avg_relevance = if !final_indices.is_empty() {
            final_indices
                .iter()
                .map(|&idx| similarities.get(idx).copied().unwrap_or(0.0))
                .sum::<f32>()
                / final_indices.len() as f32
        } else {
            0.0
        };

        let metrics = CompressionMetrics::new(
            original_length,
            compressed_length,
            avg_relevance,
            start_time.elapsed().as_millis() as u64,
            sentences_processed,
            final_indices.len(),
        );

        debug!(
            "Compressed node: {} -> {} chars ({:.1}% compression), relevance: {:.3}",
            original_length,
            compressed_length,
            metrics.compression_ratio * 100.0,
            avg_relevance
        );

        Ok((compressed_content, metrics))
    }
}

#[async_trait]
impl DocumentCompressor for SentenceEmbeddingOptimizer {
    async fn compress(&self, nodes: Vec<ScoredNode>, query: &str) -> Result<Vec<ScoredNode>> {
        if nodes.is_empty() {
            return Ok(nodes);
        }

        debug!(
            "Compressing {} nodes using sentence embedding optimization",
            nodes.len()
        );

        // Generate query embedding
        let query_embedding = self.embedder.embed(query).await?;

        let mut compressed_nodes = Vec::new();

        for mut node in nodes {
            let (compressed_content, metrics) = self
                .compress_node_content(&node.node.content, &query_embedding)
                .await?;

            // Update node content
            node.node.content = compressed_content;

            // Add compression metrics to metadata
            if let Ok(metrics_json) = serde_json::to_string(&metrics) {
                node.node.metadata.insert(
                    "compression_metrics".to_string(),
                    serde_json::Value::String(metrics_json),
                );
            }

            // Optionally adjust score based on relevance
            node.score = (node.score + metrics.relevance_score) / 2.0;

            compressed_nodes.push(node);
        }

        debug!("Compression completed for {} nodes", compressed_nodes.len());
        Ok(compressed_nodes)
    }

    fn name(&self) -> &'static str {
        "SentenceEmbeddingOptimizer"
    }
}

#[async_trait]
impl NodePostprocessor for SentenceEmbeddingOptimizer {
    async fn postprocess(&self, nodes: Vec<ScoredNode>, query: &str) -> Result<Vec<ScoredNode>> {
        self.compress(nodes, query).await
    }

    fn name(&self) -> &'static str {
        "SentenceEmbeddingOptimizer"
    }
}
