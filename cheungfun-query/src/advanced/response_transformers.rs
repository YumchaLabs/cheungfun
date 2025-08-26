// Response Transformers Implementation

use super::{ResponseTransformer, RetrievalResponse};
use anyhow::{Context, Result};
use async_trait::async_trait;
use cheungfun_core::{types::GenerationOptions, ResponseGenerator, ScoredNode};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Summary Transformer
///
/// Generates summaries for retrieved results to reduce redundant information.
#[derive(Debug)]
pub struct SummaryTransformer {
    /// The LLM client.
    pub llm_client: Arc<dyn ResponseGenerator>,
    /// The prompt template for summarization.
    pub prompt_template: String,
    /// The maximum length of the summary.
    pub max_summary_length: usize,
    /// Whether to preserve the original content.
    pub preserve_original: bool,
    /// The batch size for processing.
    pub batch_size: usize,
}

impl SummaryTransformer {
    /// Creates a new summary transformer.
    pub fn new(llm_client: Arc<dyn ResponseGenerator>) -> Self {
        Self {
            llm_client,
            prompt_template: DEFAULT_SUMMARY_PROMPT.to_string(),
            max_summary_length: 200,
            preserve_original: true,
            batch_size: 5,
        }
    }

    /// Sets the prompt template.
    #[must_use]
    pub fn with_prompt_template(mut self, template: String) -> Self {
        self.prompt_template = template;
        self
    }

    /// Sets the maximum summary length.
    #[must_use]
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_summary_length = max_length;
        self
    }

    /// Sets whether to preserve the original content.
    #[must_use]
    pub fn with_preserve_original(mut self, preserve: bool) -> Self {
        self.preserve_original = preserve;
        self
    }

    /// Generates a summary for a single document.
    async fn generate_summary(&self, content: String, query: String) -> Result<String> {
        let prompt = self
            .prompt_template
            .replace("{content}", &content)
            .replace("{query}", &query)
            .replace("{max_length}", &self.max_summary_length.to_string());

        let response = self
            .llm_client
            .generate_response(&prompt, vec![], &GenerationOptions::default())
            .await
            .context("Failed to generate summary")?;

        let mut summary = response.content;

        // Ensure the summary does not exceed the maximum length.
        if summary.len() > self.max_summary_length {
            summary.truncate(self.max_summary_length);
            if let Some(last_space) = summary.rfind(' ') {
                summary.truncate(last_space);
            }
            summary.push_str("...");
        }

        Ok(summary)
    }
}

#[async_trait]
impl ResponseTransformer for SummaryTransformer {
    async fn transform(&self, response: &mut RetrievalResponse) -> Result<()> {
        if response.nodes.is_empty() {
            return Ok(());
        }

        debug!("Generating summaries for {} nodes", response.nodes.len());

        // Process nodes in batches.
        let query_text = response.query.original_text.clone();
        for chunk in response.nodes.chunks_mut(self.batch_size) {
            let mut futures = Vec::new();

            for node in chunk.iter() {
                let content = node.node.content.clone();
                let query = query_text.clone();
                futures.push(self.generate_summary(content, query));
            }

            let summaries = futures::future::try_join_all(futures).await?;

            // Update the node content.
            for (node, summary) in chunk.iter_mut().zip(summaries.into_iter()) {
                if self.preserve_original {
                    // Add the summary to the metadata.
                    node.node
                        .metadata
                        .insert("summary".to_string(), serde_json::Value::String(summary));
                } else {
                    // Replace the original content.
                    node.node.content = summary;
                }
            }
        }

        // Add transformation metadata.
        response
            .metadata
            .insert("summary_applied".to_string(), serde_json::Value::Bool(true));
        response.metadata.insert(
            "summary_preserve_original".to_string(),
            serde_json::Value::Bool(self.preserve_original),
        );

        info!(
            "Summary transformation completed for {} nodes",
            response.nodes.len()
        );
        Ok(())
    }

    fn name(&self) -> &'static str {
        "SummaryTransformer"
    }

    fn estimated_transform_time(&self, nodes_count: usize) -> Option<Duration> {
        let batches = nodes_count.div_ceil(self.batch_size);
        Some(Duration::from_secs(batches as u64 * 3)) // Estimate 3 seconds per batch.
    }
}

/// Deduplication Transformer
///
/// Removes duplicate or highly similar retrieval results.
#[derive(Debug)]
pub struct DeduplicationTransformer {
    /// The similarity threshold.
    pub similarity_threshold: f32,
    /// The deduplication method.
    pub method: DeduplicationMethod,
    /// Whether to keep the document with the highest score.
    pub keep_highest_score: bool,
}

#[derive(Debug, Clone)]
pub enum DeduplicationMethod {
    /// Based on content hash.
    ContentHash,
    /// Based on embedding similarity.
    EmbeddingSimilarity,
    /// Based on text similarity.
    TextSimilarity,
    /// Based on document ID.
    DocumentId,
}

impl DeduplicationTransformer {
    /// Creates a new deduplication transformer.
    #[must_use]
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            similarity_threshold,
            method: DeduplicationMethod::TextSimilarity,
            keep_highest_score: true,
        }
    }

    /// Sets the deduplication method.
    #[must_use]
    pub fn with_method(mut self, method: DeduplicationMethod) -> Self {
        self.method = method;
        self
    }

    /// Sets whether to keep the highest score.
    #[must_use]
    pub fn with_keep_highest_score(mut self, keep_highest: bool) -> Self {
        self.keep_highest_score = keep_highest;
        self
    }

    /// Calculates the content hash.
    fn calculate_content_hash(&self, content: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }

    /// Calculates text similarity (simple Jaccard similarity).
    fn calculate_text_similarity(&self, text1: &str, text2: &str) -> f32 {
        let words1: HashSet<&str> = text1.split_whitespace().collect();
        let words2: HashSet<&str> = text2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Calculates embedding similarity (cosine similarity).
    fn calculate_embedding_similarity(&self, emb1: &[f32], emb2: &[f32]) -> f32 {
        if emb1.len() != emb2.len() {
            return 0.0;
        }

        let dot_product: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    /// Checks if two nodes are duplicates.
    fn is_duplicate(&self, node1: &ScoredNode, node2: &ScoredNode) -> bool {
        match &self.method {
            DeduplicationMethod::ContentHash => {
                let hash1 = self.calculate_content_hash(&node1.node.content);
                let hash2 = self.calculate_content_hash(&node2.node.content);
                hash1 == hash2
            }
            DeduplicationMethod::TextSimilarity => {
                let similarity =
                    self.calculate_text_similarity(&node1.node.content, &node2.node.content);
                similarity >= self.similarity_threshold
            }
            DeduplicationMethod::EmbeddingSimilarity => {
                if let (Some(emb1), Some(emb2)) = (&node1.node.embedding, &node2.node.embedding) {
                    let similarity = self.calculate_embedding_similarity(emb1, emb2);
                    similarity >= self.similarity_threshold
                } else {
                    false
                }
            }
            DeduplicationMethod::DocumentId => {
                node1.node.source_document_id == node2.node.source_document_id
            }
        }
    }
}

#[async_trait]
impl ResponseTransformer for DeduplicationTransformer {
    async fn transform(&self, response: &mut RetrievalResponse) -> Result<()> {
        if response.nodes.len() <= 1 {
            return Ok(());
        }

        debug!(
            "Deduplicating {} nodes with method: {:?}",
            response.nodes.len(),
            self.method
        );

        let mut deduplicated: Vec<ScoredNode> = Vec::new();
        let mut to_be_replaced: Option<usize> = None;
        let original_count = response.nodes.len();

        // This is a naive O(n^2) implementation.
        for current_node in response.nodes.drain(..) {
            let mut is_duplicate_of_any = false;

            // Check if it's a duplicate of any already selected node.
            for (i, existing_node) in deduplicated.iter().enumerate() {
                if self.is_duplicate(&current_node, existing_node) {
                    is_duplicate_of_any = true;
                    // If the current node has a higher score, mark the existing node for replacement.
                    if self.keep_highest_score && current_node.score > existing_node.score {
                        to_be_replaced = Some(i);
                    }
                    break;
                }
            }

            if let Some(index) = to_be_replaced.take() {
                deduplicated[index] = current_node;
            } else if !is_duplicate_of_any {
                deduplicated.push(current_node);
            }
        }

        response.nodes = deduplicated;

        // Add transformation metadata.
        response.metadata.insert(
            "deduplication_applied".to_string(),
            serde_json::Value::Bool(true),
        );
        response.metadata.insert(
            "original_count".to_string(),
            serde_json::Value::Number(original_count.into()),
        );
        response.metadata.insert(
            "deduplicated_count".to_string(),
            serde_json::Value::Number(response.nodes.len().into()),
        );

        info!(
            "Deduplication completed: {} -> {} nodes",
            original_count,
            response.nodes.len()
        );
        Ok(())
    }

    fn name(&self) -> &'static str {
        "DeduplicationTransformer"
    }

    fn changes_node_count(&self) -> bool {
        true
    }

    fn estimated_transform_time(&self, nodes_count: usize) -> Option<Duration> {
        // O(n²) complexity deduplication algorithm
        let operations = nodes_count.saturating_mul(nodes_count) / 2;
        Some(Duration::from_millis(operations as u64 / 1000)) // Estimate
    }
}

/// Filter Transformer
///
/// Filters retrieval results based on various conditions.
#[derive(Debug)]
pub struct FilterTransformer {
    /// The filter conditions.
    pub filters: Vec<FilterCondition>,
}

#[derive(Debug, Clone)]
pub enum FilterCondition {
    /// Minimum score threshold.
    MinScore(f32),
    /// Maximum score threshold.
    MaxScore(f32),
    /// Content length range.
    ContentLength {
        min: Option<usize>,
        max: Option<usize>,
    },
    /// Metadata condition.
    Metadata {
        key: String,
        value: serde_json::Value,
    },
    /// Custom filter function.
    Custom(String), // Function name; actual implementation requires a registration mechanism.
}

impl Default for FilterTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl FilterTransformer {
    /// Creates a new filter transformer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Adds a filter condition.
    #[must_use]
    pub fn add_filter(mut self, condition: FilterCondition) -> Self {
        self.filters.push(condition);
        self
    }

    /// Adds a minimum score filter.
    #[must_use]
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.filters.push(FilterCondition::MinScore(min_score));
        self
    }

    /// Adds a content length filter.
    #[must_use]
    pub fn with_content_length(mut self, min: Option<usize>, max: Option<usize>) -> Self {
        self.filters
            .push(FilterCondition::ContentLength { min, max });
        self
    }

    /// Checks if a node meets a condition.
    fn matches_condition(&self, node: &ScoredNode, condition: &FilterCondition) -> bool {
        match condition {
            FilterCondition::MinScore(min_score) => node.score >= *min_score,
            FilterCondition::MaxScore(max_score) => node.score <= *max_score,
            FilterCondition::ContentLength { min, max } => {
                let content_len = node.node.content.len();
                let min_ok = min.is_none_or(|m| content_len >= m);
                let max_ok = max.is_none_or(|m| content_len <= m);
                min_ok && max_ok
            }
            FilterCondition::Metadata { key, value } => node.node.metadata.get(key) == Some(value),
            FilterCondition::Custom(_) => {
                // TODO: Implement custom filter function registration mechanism.
                warn!("Custom filter not implemented");
                true
            }
        }
    }
}

#[async_trait]
impl ResponseTransformer for FilterTransformer {
    async fn transform(&self, response: &mut RetrievalResponse) -> Result<()> {
        if self.filters.is_empty() || response.nodes.is_empty() {
            return Ok(());
        }

        debug!(
            "Filtering {} nodes with {} conditions",
            response.nodes.len(),
            self.filters.len()
        );

        let original_count = response.nodes.len();

        response.nodes.retain(|node| {
            self.filters
                .iter()
                .all(|condition| self.matches_condition(node, condition))
        });

        // Add transformation metadata.
        response
            .metadata
            .insert("filter_applied".to_string(), serde_json::Value::Bool(true));
        response.metadata.insert(
            "filter_original_count".to_string(),
            serde_json::Value::Number(original_count.into()),
        );
        response.metadata.insert(
            "filter_remaining_count".to_string(),
            serde_json::Value::Number(response.nodes.len().into()),
        );

        info!(
            "Filtering completed: {} -> {} nodes",
            original_count,
            response.nodes.len()
        );
        Ok(())
    }

    fn name(&self) -> &'static str {
        "FilterTransformer"
    }

    fn changes_node_count(&self) -> bool {
        true
    }

    fn estimated_transform_time(&self, nodes_count: usize) -> Option<Duration> {
        Some(Duration::from_millis(nodes_count as u64 / 100)) // Very fast operation.
    }
}

// Default summary prompt template
const DEFAULT_SUMMARY_PROMPT: &str = r"
Please provide a concise summary of the following content that is relevant to the query.
The summary should be no more than {max_length} characters and should capture the key information that answers or relates to the query.

Query: {query}

Content: {content}

Summary:
";
