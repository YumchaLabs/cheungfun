// Rerankers Implementation

use super::*;
use anyhow::{Context, Result};
use async_trait::async_trait;
use cheungfun_core::{ResponseGenerator, ScoredNode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// LLM-based reranker.
///
/// Uses a large language model to rerank retrieval results, which can often provide
/// better semantic understanding.
#[derive(Debug)]
pub struct LLMReranker {
    /// The LLM client.
    pub llm_client: Arc<dyn ResponseGenerator>,
    /// The reranking prompt template.
    pub prompt_template: String,
    /// The maximum number of items to rerank.
    pub max_rerank_count: usize,
    /// The number of items to output.
    pub top_n: usize,
    /// The batch size for processing.
    pub batch_size: usize,
    /// Timeout setting for the operation.
    pub timeout: Duration,
}

impl LLMReranker {
    /// Creates a new LLM reranker.
    pub fn new(llm_client: Arc<dyn ResponseGenerator>) -> Self {
        Self {
            llm_client,
            prompt_template: DEFAULT_LLM_RERANK_PROMPT.to_string(),
            max_rerank_count: 20,
            top_n: 10,
            batch_size: 5,
            timeout: Duration::from_secs(30),
        }
    }

    /// Sets the prompt template.
    pub fn with_prompt_template(mut self, template: String) -> Self {
        self.prompt_template = template;
        self
    }

    /// Sets the number of items to output.
    pub fn with_top_n(mut self, top_n: usize) -> Self {
        self.top_n = top_n;
        self
    }

    /// Sets the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Executes LLM reranking for a batch.
    async fn llm_rerank_batch(
        &self,
        query: &str,
        nodes: &[ScoredNode],
    ) -> Result<Vec<(usize, f32)>> {
        debug!("LLM reranking {} nodes", nodes.len());

        // Build the reranking prompt.
        let documents_text = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                format!(
                    "Document {}: {}",
                    i + 1,
                    node.node.content.chars().take(500).collect::<String>()
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = self
            .prompt_template
            .replace("{query}", query)
            .replace("{documents}", &documents_text)
            .replace("{num_docs}", &nodes.len().to_string());

        // Call the LLM.
        let response = self
            .llm_client
            .generate_response(&prompt, vec![], &Default::default())
            .await
            .context("LLM reranking failed")?;

        // Parse the LLM response.
        self.parse_llm_ranking(&response.content, nodes.len())
    }

    /// Parses the ranking response from the LLM.
    fn parse_llm_ranking(&self, response: &str, num_docs: usize) -> Result<Vec<(usize, f32)>> {
        let mut rankings = Vec::new();

        // Try to parse different formats of ranking results.
        for line in response.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Try to parse "Document X: score" format.
            if let Some(parsed) = self.parse_document_score_format(line) {
                rankings.push(parsed);
                continue;
            }

            // Try to parse "X. Document Y" format.
            if let Some(parsed) = self.parse_ranked_list_format(line, rankings.len()) {
                rankings.push(parsed);
                continue;
            }
        }

        // If parsing fails, return the original order.
        if rankings.is_empty() {
            warn!("Failed to parse LLM ranking response, using original order");
            rankings = (0..num_docs).map(|i| (i, 1.0 - i as f32 * 0.1)).collect();
        }

        // Ensure all documents have a rank.
        let mut final_rankings = Vec::new();
        let mut used_indices = std::collections::HashSet::new();

        for (doc_idx, score) in rankings {
            if doc_idx < num_docs && !used_indices.contains(&doc_idx) {
                final_rankings.push((doc_idx, score));
                used_indices.insert(doc_idx);
            }
        }

        // Add unranked documents.
        for i in 0..num_docs {
            if !used_indices.contains(&i) {
                final_rankings.push((i, 0.1)); // Assign a low score.
            }
        }

        Ok(final_rankings)
    }

    /// Parses the "Document X: score" format.
    fn parse_document_score_format(&self, line: &str) -> Option<(usize, f32)> {
        if let Some(colon_pos) = line.find(':') {
            let doc_part = &line[..colon_pos];
            let score_part = &line[colon_pos + 1..].trim();

            // Extract the document number.
            if let Some(doc_num) = doc_part.split_whitespace().last() {
                if let Ok(doc_idx) = doc_num.parse::<usize>() {
                    if doc_idx > 0 {
                        // Try to parse the score.
                        if let Ok(score) = score_part.parse::<f32>() {
                            return Some((doc_idx - 1, score)); // Convert to 0-based index.
                        }
                    }
                }
            }
        }
        None
    }

    /// Parses the "X. Document Y" format.
    fn parse_ranked_list_format(&self, line: &str, rank: usize) -> Option<(usize, f32)> {
        // Find the document number.
        for word in line.split_whitespace() {
            if let Ok(doc_num) = word
                .trim_end_matches(&['.', ',', ':', ';'][..])
                .parse::<usize>()
            {
                if doc_num > 0 {
                    let score = 1.0 - rank as f32 * 0.1; // Score based on rank.
                    return Some((doc_num - 1, score));
                }
            }
        }
        None
    }
}

#[async_trait]
impl Reranker for LLMReranker {
    async fn rerank(
        &self,
        query: &AdvancedQuery,
        nodes: Vec<ScoredNode>,
    ) -> Result<Vec<ScoredNode>> {
        if nodes.is_empty() {
            return Ok(nodes);
        }

        debug!(
            "LLM reranking {} nodes for query: {}",
            nodes.len(),
            query.original_text
        );

        // Limit the number of nodes to rerank.
        let nodes_to_rerank = if nodes.len() > self.max_rerank_count {
            nodes.into_iter().take(self.max_rerank_count).collect()
        } else {
            nodes
        };

        // Process in batches.
        let mut all_rankings = Vec::new();
        let mut current_offset = 0;

        for chunk in nodes_to_rerank.chunks(self.batch_size) {
            let rankings = self.llm_rerank_batch(&query.original_text, chunk).await?;

            // Adjust the index offset.
            let adjusted_rankings: Vec<_> = rankings
                .into_iter()
                .map(|(idx, score)| (idx + current_offset, score))
                .collect();

            all_rankings.extend(adjusted_rankings);
            current_offset += chunk.len();
        }

        // Sort by score.
        all_rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Reconstruct the node list.
        let mut reranked_nodes = Vec::new();
        for (original_idx, new_score) in all_rankings.into_iter().take(self.top_n) {
            if let Some(mut node) = nodes_to_rerank.get(original_idx).cloned() {
                node.score = new_score;
                reranked_nodes.push(node);
            }
        }

        info!(
            "LLM reranking completed: from {} to {} nodes",
            nodes_to_rerank.len(),
            reranked_nodes.len()
        );
        Ok(reranked_nodes)
    }

    fn name(&self) -> &'static str {
        "LLMReranker"
    }

    fn max_nodes(&self) -> Option<usize> {
        Some(self.max_rerank_count)
    }

    fn estimated_rerank_time(&self, nodes_count: usize) -> Option<Duration> {
        let batches = (nodes_count + self.batch_size - 1) / self.batch_size;
        Some(Duration::from_secs(batches as u64 * 5)) // Estimate 5 seconds per batch.
    }
}

/// Model-based reranker.
///
/// Uses a specialized reranking model (like a Cross-Encoder) to rerank results.
#[derive(Debug)]
pub struct ModelReranker {
    /// The reranking model.
    pub model: Arc<dyn RerankModel>,
    /// The batch size for processing.
    pub batch_size: usize,
    /// The number of items to output.
    pub top_n: usize,
    /// The maximum input length for the model.
    pub max_input_length: usize,
}

impl ModelReranker {
    /// Creates a new model reranker.
    pub fn new(model: Arc<dyn RerankModel>) -> Self {
        Self {
            model,
            batch_size: 32,
            top_n: 10,
            max_input_length: 512,
        }
    }

    /// Sets the number of items to output.
    pub fn with_top_n(mut self, top_n: usize) -> Self {
        self.top_n = top_n;
        self
    }

    /// Sets the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Truncates text to the maximum length.
    fn truncate_text(&self, text: &str) -> String {
        if text.len() <= self.max_input_length {
            text.to_string()
        } else {
            let mut truncated = text.chars().take(self.max_input_length).collect::<String>();
            // Truncate at a word boundary.
            if let Some(last_space) = truncated.rfind(' ') {
                truncated.truncate(last_space);
            }
            truncated
        }
    }
}

#[async_trait]
impl Reranker for ModelReranker {
    async fn rerank(
        &self,
        query: &AdvancedQuery,
        nodes: Vec<ScoredNode>,
    ) -> Result<Vec<ScoredNode>> {
        if nodes.is_empty() {
            return Ok(nodes);
        }

        debug!("Model reranking {} nodes", nodes.len());

        // Prepare document texts.
        let documents: Vec<String> = nodes
            .iter()
            .map(|node| self.truncate_text(&node.node.content))
            .collect();

        let documents_ref: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();

        // Calculate relevance scores in batches.
        let mut all_scores = Vec::new();

        for chunk in documents_ref.chunks(self.batch_size) {
            let scores = self.model.score(&query.original_text, chunk).await?;
            all_scores.extend(scores);
        }

        // Create a list of nodes with their new scores.
        let mut scored_nodes: Vec<_> = nodes
            .into_iter()
            .zip(all_scores.into_iter())
            .map(|(mut node, score)| {
                node.score = score;
                node
            })
            .collect();

        // Sort by score.
        scored_nodes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return the top_n results.
        scored_nodes.truncate(self.top_n);

        info!("Model reranking completed: {} nodes", scored_nodes.len());
        Ok(scored_nodes)
    }

    fn name(&self) -> &'static str {
        "ModelReranker"
    }

    fn supports_batch(&self) -> bool {
        true
    }

    fn estimated_rerank_time(&self, nodes_count: usize) -> Option<Duration> {
        let batches = (nodes_count + self.batch_size - 1) / self.batch_size;
        Some(Duration::from_millis(batches as u64 * 100)) // Estimate 100ms per batch.
    }
}

/// Interface for a reranking model.
#[async_trait]
pub trait RerankModel: Send + Sync + std::fmt::Debug {
    /// Calculates the query-document relevance scores.
    async fn score(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>>;

    /// Gets the model name.
    fn model_name(&self) -> &str;

    /// Gets the maximum input length.
    fn max_input_length(&self) -> usize;

    /// Whether batch processing is supported.
    fn supports_batch(&self) -> bool {
        true
    }
}

/// Simple score-based reranker.
///
/// Reranks based on existing scores, allowing for different sorting strategies.
#[derive(Debug)]
pub struct ScoreReranker {
    /// The sorting strategy.
    pub strategy: ScoreRerankStrategy,
    /// The number of items to output.
    pub top_n: usize,
}

#[derive(Debug, Clone)]
pub enum ScoreRerankStrategy {
    /// By original score, descending.
    OriginalScore,
    /// By original score, ascending.
    OriginalScoreAsc,
    /// Random order.
    Random,
    /// Diversity-based ranking (avoids clustering of similar documents).
    Diversity { similarity_threshold: f32 },
}

impl ScoreReranker {
    pub fn new(strategy: ScoreRerankStrategy) -> Self {
        Self {
            strategy,
            top_n: 10,
        }
    }

    pub fn with_top_n(mut self, top_n: usize) -> Self {
        self.top_n = top_n;
        self
    }
}

#[async_trait]
impl Reranker for ScoreReranker {
    async fn rerank(
        &self,
        _query: &AdvancedQuery,
        mut nodes: Vec<ScoredNode>,
    ) -> Result<Vec<ScoredNode>> {
        debug!(
            "Score reranking {} nodes with strategy: {:?}",
            nodes.len(),
            self.strategy
        );

        match &self.strategy {
            ScoreRerankStrategy::OriginalScore => {
                nodes.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            ScoreRerankStrategy::OriginalScoreAsc => {
                nodes.sort_by(|a, b| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            ScoreRerankStrategy::Random => {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                nodes.shuffle(&mut rng);
            }
            ScoreRerankStrategy::Diversity {
                similarity_threshold,
            } => {
                nodes = self.diversity_rerank(nodes, *similarity_threshold)?;
            }
        }

        nodes.truncate(self.top_n);
        Ok(nodes)
    }

    fn name(&self) -> &'static str {
        "ScoreReranker"
    }

    fn estimated_rerank_time(&self, _nodes_count: usize) -> Option<Duration> {
        Some(Duration::from_millis(10)) // Very fast operation.
    }
}

impl ScoreReranker {
    /// Diversity reranking implementation.
    fn diversity_rerank(
        &self,
        mut nodes: Vec<ScoredNode>,
        similarity_threshold: f32,
    ) -> Result<Vec<ScoredNode>> {
        if nodes.len() <= 1 {
            return Ok(nodes);
        }

        // First, sort by score.
        nodes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut selected = Vec::new();
        // Since we sort descending, we should iterate from the beginning.
        let mut remaining = nodes;

        // Select the first one (highest score).
        if !remaining.is_empty() {
            selected.push(remaining.remove(0));
        }

        // Iterate and select remaining nodes, ensuring diversity.
        while !remaining.is_empty() && selected.len() < self.top_n {
            let mut best_idx = 0;
            let mut best_score = f32::NEG_INFINITY;

            for (i, candidate) in remaining.iter().enumerate() {
                // Calculate the minimum similarity with already selected nodes.
                let max_similarity_with_selected = selected
                    .iter()
                    .map(|selected_node| {
                        self.calculate_similarity(
                            &candidate.node.content,
                            &selected_node.node.content,
                        )
                    })
                    .fold(f32::NEG_INFINITY, f32::max);

                // If similarity is below the threshold, it's a good candidate.
                // We penalize similarity, so lower similarity is better.
                let diversity_penalty = if max_similarity_with_selected > similarity_threshold {
                    max_similarity_with_selected * 0.5
                } else {
                    0.0
                };
                let combined_score = candidate.score - diversity_penalty;

                if combined_score > best_score {
                    best_score = combined_score;
                    best_idx = i;
                }
            }

            selected.push(remaining.remove(best_idx));
        }

        Ok(selected)
    }

    /// Calculates the similarity between two texts (simple implementation).
    fn calculate_similarity(&self, text1: &str, text2: &str) -> f32 {
        // Simple Jaccard similarity.
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

// Default LLM reranking prompt template
const DEFAULT_LLM_RERANK_PROMPT: &str = r#"
Given a query and a list of documents, please rank the documents by their relevance to the query.
Provide a ranking from most relevant to least relevant.

Query: {query}

Documents:
{documents}

Please rank these {num_docs} documents by relevance to the query. 
Provide your ranking in the format:
1. Document X
2. Document Y
...

Ranking:
"#;
