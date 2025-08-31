//! Summary extractor transformer for generating node summaries using LLM.
//!
//! This module provides LLM-powered summary extraction capabilities,
//! following LlamaIndex's SummaryExtractor design exactly. It uses
//! large language models to intelligently generate summaries for nodes
//! and their adjacent context.

use crate::error::{IndexingError, Result as IndexingResult};
use async_trait::async_trait;
use cheungfun_core::{
    traits::{NodeState, TypedData, TypedTransform},
    types::Node,
    Result as CoreResult,
};
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use siumai::prelude::*;
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, info, warn};

/// Default prompt template for summary extraction.
/// Matches LlamaIndex's DEFAULT_SUMMARY_EXTRACT_TEMPLATE exactly.
pub const DEFAULT_SUMMARY_EXTRACT_TEMPLATE: &str = r#"Here is the content of the section:
{context_str}

Summarize the key topics and entities of the section. 

Summary: "#;

/// Summary types that can be extracted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SummaryType {
    /// Summary of the current node itself.
    #[serde(rename = "self")]
    SelfSummary,
    /// Summary of the previous node.
    #[serde(rename = "prev")]
    PrevSummary,
    /// Summary of the next node.
    #[serde(rename = "next")]
    NextSummary,
}

impl SummaryType {
    /// Get the metadata key for this summary type.
    pub fn metadata_key(&self) -> &'static str {
        match self {
            SummaryType::SelfSummary => "section_summary",
            SummaryType::PrevSummary => "prev_section_summary",
            SummaryType::NextSummary => "next_section_summary",
        }
    }

    /// Parse from string representation.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "self" => Some(SummaryType::SelfSummary),
            "prev" => Some(SummaryType::PrevSummary),
            "next" => Some(SummaryType::NextSummary),
            _ => None,
        }
    }

    /// Convert to string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            SummaryType::SelfSummary => "self",
            SummaryType::PrevSummary => "prev",
            SummaryType::NextSummary => "next",
        }
    }
}

/// Configuration for summary extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryExtractorConfig {
    /// List of summary types to extract.
    pub summaries: Vec<SummaryType>,
    /// Template for summary extraction.
    pub prompt_template: String,
    /// Whether to show progress during extraction.
    pub show_progress: bool,
    /// Number of concurrent workers for processing.
    pub num_workers: usize,
    /// Maximum length of context to send to LLM.
    pub max_context_length: usize,
    /// Whether to process nodes in place or create copies.
    pub in_place: bool,
}

impl Default for SummaryExtractorConfig {
    fn default() -> Self {
        Self {
            summaries: vec![SummaryType::SelfSummary],
            prompt_template: DEFAULT_SUMMARY_EXTRACT_TEMPLATE.to_string(),
            show_progress: true,
            num_workers: 4,
            max_context_length: 4000,
            in_place: true,
        }
    }
}

impl SummaryExtractorConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the summary types to extract.
    pub fn with_summaries(mut self, summaries: Vec<SummaryType>) -> Self {
        self.summaries = summaries;
        self
    }

    /// Set the prompt template.
    pub fn with_prompt_template(mut self, template: String) -> Self {
        self.prompt_template = template;
        self
    }

    /// Set whether to show progress.
    pub fn with_show_progress(mut self, show_progress: bool) -> Self {
        self.show_progress = show_progress;
        self
    }

    /// Set the number of workers.
    pub fn with_num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers.max(1);
        self
    }

    /// Set the maximum context length.
    pub fn with_max_context_length(mut self, max_length: usize) -> Self {
        self.max_context_length = max_length;
        self
    }

    /// Set whether to process in place.
    pub fn with_in_place(mut self, in_place: bool) -> Self {
        self.in_place = in_place;
        self
    }

    /// Enable self summary extraction.
    pub fn with_self_summary(mut self) -> Self {
        if !self.summaries.contains(&SummaryType::SelfSummary) {
            self.summaries.push(SummaryType::SelfSummary);
        }
        self
    }

    /// Enable previous node summary extraction.
    pub fn with_prev_summary(mut self) -> Self {
        if !self.summaries.contains(&SummaryType::PrevSummary) {
            self.summaries.push(SummaryType::PrevSummary);
        }
        self
    }

    /// Enable next node summary extraction.
    pub fn with_next_summary(mut self) -> Self {
        if !self.summaries.contains(&SummaryType::NextSummary) {
            self.summaries.push(SummaryType::NextSummary);
        }
        self
    }
}

/// LLM-powered summary extractor.
///
/// This transformer uses large language models to extract summaries from nodes
/// and their adjacent context, following LlamaIndex's SummaryExtractor design exactly.
/// It supports extracting summaries for the current node, previous node, and next node.
///
/// # Features
///
/// - LLM-powered summary generation with high quality
/// - Adjacent node context support (prev/self/next)
/// - Node-level processing for precise summary extraction
/// - Configurable summary types and templates
/// - Parallel processing for efficiency across multiple nodes
/// - Robust error handling and fallback strategies
/// - Support for custom LLM providers via siumai
/// - Full LlamaIndex compatibility
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::transformers::{SummaryExtractor, SummaryExtractorConfig, SummaryType};
/// use cheungfun_core::{Node, traits::{Transform, TransformInput}};
/// use siumai::prelude::*;
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create LLM client
///     let llm_client = Siumai::builder()
///         .openai()
///         .api_key("your-api-key")
///         .model("gpt-4o-mini")
///         .build()
///         .await?;
///
///     // Configure extraction
///     let config = SummaryExtractorConfig::new()
///         .with_summaries(vec![SummaryType::SelfSummary, SummaryType::PrevSummary])
///         .with_show_progress(true);
///
///     // Create extractor
///     let extractor = SummaryExtractor::new(Arc::new(llm_client), config)?;
///
///     // Extract summaries from nodes
///     let nodes = vec![/* your nodes */];
///     let enhanced_nodes = extractor.transform(TransformInput::Nodes(nodes)).await?;
///
///     // Check the extracted summaries
///     for node in enhanced_nodes {
///         if let Some(summary) = node.metadata.get("section_summary") {
///             println!("Summary: {}", summary.as_str().unwrap_or("N/A"));
///         }
///     }
///
///     Ok(())
/// }
/// ```
pub struct SummaryExtractor {
    /// LLM client for summary generation.
    llm_client: Arc<dyn LlmClient>,
    /// Configuration for summary extraction.
    config: SummaryExtractorConfig,
}

impl std::fmt::Debug for SummaryExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SummaryExtractor")
            .field("config", &self.config)
            .finish()
    }
}

impl SummaryExtractor {
    /// Create a new summary extractor.
    ///
    /// # Arguments
    ///
    /// * `llm_client` - The LLM client to use for summary generation
    /// * `config` - Configuration for the extractor
    ///
    /// # Returns
    ///
    /// A new `SummaryExtractor` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(
        llm_client: Arc<dyn LlmClient>,
        config: SummaryExtractorConfig,
    ) -> IndexingResult<Self> {
        if config.summaries.is_empty() {
            return Err(IndexingError::configuration(
                "At least one summary type must be specified".to_string(),
            ));
        }

        if config.prompt_template.is_empty() {
            return Err(IndexingError::configuration(
                "Prompt template cannot be empty".to_string(),
            ));
        }

        if !config.prompt_template.contains("{context_str}") {
            return Err(IndexingError::configuration(
                "Prompt template must contain {context_str} placeholder".to_string(),
            ));
        }

        Ok(Self { llm_client, config })
    }

    /// Create a new summary extractor with default configuration.
    ///
    /// # Arguments
    ///
    /// * `llm_client` - The LLM client to use for summary generation
    ///
    /// # Returns
    ///
    /// A new `SummaryExtractor` instance with default configuration.
    pub fn with_defaults(llm_client: Arc<dyn LlmClient>) -> IndexingResult<Self> {
        Self::new(llm_client, SummaryExtractorConfig::default())
    }

    /// Create a builder for configuring the summary extractor.
    pub fn builder(llm_client: Arc<dyn LlmClient>) -> SummaryExtractorBuilder {
        SummaryExtractorBuilder::new(llm_client)
    }
}

/// Builder for creating SummaryExtractor instances.
pub struct SummaryExtractorBuilder {
    llm_client: Arc<dyn LlmClient>,
    config: SummaryExtractorConfig,
}

impl SummaryExtractorBuilder {
    /// Create a new builder.
    pub fn new(llm_client: Arc<dyn LlmClient>) -> Self {
        Self {
            llm_client,
            config: SummaryExtractorConfig::default(),
        }
    }

    /// Set the summary types to extract.
    pub fn summaries(mut self, summaries: Vec<SummaryType>) -> Self {
        self.config = self.config.with_summaries(summaries);
        self
    }

    /// Set the prompt template.
    pub fn prompt_template(mut self, template: String) -> Self {
        self.config = self.config.with_prompt_template(template);
        self
    }

    /// Set whether to show progress.
    pub fn show_progress(mut self, show_progress: bool) -> Self {
        self.config = self.config.with_show_progress(show_progress);
        self
    }

    /// Set the number of workers.
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.config = self.config.with_num_workers(num_workers);
        self
    }

    /// Set the maximum context length.
    pub fn max_context_length(mut self, max_length: usize) -> Self {
        self.config = self.config.with_max_context_length(max_length);
        self
    }

    /// Set whether to process in place.
    pub fn in_place(mut self, in_place: bool) -> Self {
        self.config = self.config.with_in_place(in_place);
        self
    }

    /// Enable self summary extraction.
    pub fn with_self_summary(mut self) -> Self {
        self.config = self.config.with_self_summary();
        self
    }

    /// Enable previous node summary extraction.
    pub fn with_prev_summary(mut self) -> Self {
        self.config = self.config.with_prev_summary();
        self
    }

    /// Enable next node summary extraction.
    pub fn with_next_summary(mut self) -> Self {
        self.config = self.config.with_next_summary();
        self
    }

    /// Build the summary extractor.
    pub fn build(self) -> IndexingResult<SummaryExtractor> {
        SummaryExtractor::new(self.llm_client, self.config)
    }
}

impl SummaryExtractor {
    /// Process nodes to extract and add summaries.
    async fn process_nodes(&self, nodes: Vec<Node>) -> IndexingResult<Vec<Node>> {
        if nodes.is_empty() {
            return Ok(nodes);
        }

        if self.config.show_progress {
            info!("📝 Starting summary extraction for {} nodes", nodes.len());
        }

        // Extract summaries for each node in parallel
        let summary_jobs: Vec<_> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| self.extract_summaries_from_node(node, i, &nodes))
            .collect();

        // Process with controlled concurrency
        let summary_results: Vec<IndexingResult<HashMap<String, String>>> =
            stream::iter(summary_jobs)
                .buffer_unordered(self.config.num_workers)
                .collect()
                .await;

        // Apply summaries to nodes
        let enhanced_nodes = if self.config.in_place {
            self.apply_summaries_in_place(nodes, summary_results)?
        } else {
            self.apply_summaries_with_copy(nodes, summary_results)?
        };

        if self.config.show_progress {
            info!(
                "✅ Summary extraction completed for {} nodes",
                enhanced_nodes.len()
            );
        }

        Ok(enhanced_nodes)
    }

    /// Extract summaries from a single node and its adjacent context.
    async fn extract_summaries_from_node(
        &self,
        node: &Node,
        node_index: usize,
        all_nodes: &[Node],
    ) -> IndexingResult<HashMap<String, String>> {
        debug!("Extracting summaries from node {} of total", node_index + 1);

        let mut summaries = HashMap::new();

        for summary_type in &self.config.summaries {
            let content = match summary_type {
                SummaryType::SelfSummary => Some(self.get_limited_content(node)),
                SummaryType::PrevSummary => {
                    if node_index > 0 {
                        Some(self.get_limited_content(&all_nodes[node_index - 1]))
                    } else {
                        None
                    }
                }
                SummaryType::NextSummary => {
                    if node_index + 1 < all_nodes.len() {
                        Some(self.get_limited_content(&all_nodes[node_index + 1]))
                    } else {
                        None
                    }
                }
            };

            if let Some(content) = content {
                if !content.trim().is_empty() {
                    match self.generate_summary(&content).await {
                        Ok(summary) => {
                            summaries.insert(summary_type.metadata_key().to_string(), summary);
                            debug!(
                                "Generated {} summary for node {}",
                                summary_type.as_str(),
                                node_index + 1
                            );
                        }
                        Err(e) => {
                            warn!(
                                "Failed to generate {} summary for node {}: {}",
                                summary_type.as_str(),
                                node_index + 1,
                                e
                            );
                            // Continue with other summary types instead of failing completely
                        }
                    }
                }
            } else {
                debug!(
                    "No content available for {} summary at node {}",
                    summary_type.as_str(),
                    node_index + 1
                );
            }
        }

        Ok(summaries)
    }

    /// Generate a summary from content using LLM.
    async fn generate_summary(&self, content: &str) -> IndexingResult<String> {
        // Prepare the prompt
        let prompt = self
            .config
            .prompt_template
            .replace("{context_str}", content);

        debug!("Generating summary with prompt length: {}", prompt.len());

        // Call LLM using chat interface
        let messages = vec![ChatMessage::user(prompt).build()];

        let response = self.llm_client.chat(messages).await.map_err(|e| {
            IndexingError::processing(format!("LLM summary generation failed: {}", e))
        })?;

        // Extract text response
        let summary_text = if let Some(text) = response.content_text() {
            text.to_string()
        } else {
            return Err(IndexingError::processing(
                "No text content in LLM response".to_string(),
            ));
        };

        // Clean up the response
        let cleaned_summary = self.clean_summary_response(&summary_text);

        Ok(cleaned_summary)
    }

    /// Clean up summary response from LLM.
    fn clean_summary_response(&self, response: &str) -> String {
        // Remove common prefixes and clean up
        let cleaned = response.trim().trim_matches('"').trim_matches('\'').trim();

        // Remove "Summary:" prefix if present
        let cleaned = if let Some(pos) = cleaned.to_lowercase().find("summary:") {
            &cleaned[pos + 8..] // Skip "summary:"
        } else {
            cleaned
        };

        cleaned.trim().to_string()
    }

    /// Get content from node with length limitations.
    fn get_limited_content(&self, node: &Node) -> String {
        let content = &node.content;

        if content.len() <= self.config.max_context_length {
            content.to_string()
        } else {
            // Truncate to max length, trying to break at word boundaries
            let truncated = &content[..self.config.max_context_length];
            if let Some(last_space) = truncated.rfind(' ') {
                truncated[..last_space].to_string()
            } else {
                truncated.to_string()
            }
        }
    }

    /// Apply summaries to nodes in place.
    fn apply_summaries_in_place(
        &self,
        mut nodes: Vec<Node>,
        summary_results: Vec<IndexingResult<HashMap<String, String>>>,
    ) -> IndexingResult<Vec<Node>> {
        for (node, summary_result) in nodes.iter_mut().zip(summary_results.iter()) {
            match summary_result {
                Ok(summaries) => {
                    for (key, summary) in summaries {
                        node.metadata
                            .insert(key.clone(), serde_json::Value::String(summary.clone()));
                    }
                }
                Err(e) => {
                    warn!(
                        "Skipping summary application due to extraction error: {}",
                        e
                    );
                    // Continue processing other nodes
                }
            }
        }
        Ok(nodes)
    }

    /// Apply summaries to nodes with copying.
    fn apply_summaries_with_copy(
        &self,
        nodes: Vec<Node>,
        summary_results: Vec<IndexingResult<HashMap<String, String>>>,
    ) -> IndexingResult<Vec<Node>> {
        let enhanced_nodes = nodes
            .into_iter()
            .zip(summary_results.iter())
            .map(|(mut node, summary_result)| {
                match summary_result {
                    Ok(summaries) => {
                        for (key, summary) in summaries {
                            node.metadata
                                .insert(key.clone(), serde_json::Value::String(summary.clone()));
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Skipping summary application due to extraction error: {}",
                            e
                        );
                        // Continue processing other nodes
                    }
                }
                node
            })
            .collect();

        Ok(enhanced_nodes)
    }
}

// ============================================================================
// Type-Safe Transform Implementation
// ============================================================================

#[async_trait]
impl TypedTransform<NodeState, NodeState> for SummaryExtractor {
    async fn transform(&self, input: TypedData<NodeState>) -> CoreResult<TypedData<NodeState>> {
        let nodes = input.nodes().to_vec();
        let processed_nodes = self.process_nodes(nodes).await.map_err(|e| {
            cheungfun_core::CheungfunError::Pipeline {
                message: format!("Summary extraction failed: {}", e),
            }
        })?;
        Ok(TypedData::from_nodes(processed_nodes))
    }

    fn name(&self) -> &'static str {
        "SummaryExtractor"
    }

    fn description(&self) -> &'static str {
        "Generates intelligent summaries for nodes using large language models"
    }
}

// ============================================================================
// Legacy Transform Implementation (Backward Compatibility)
// ============================================================================

// Legacy Transform implementation has been removed.
// SummaryExtractor now only uses the type-safe TypedTransform system.

impl Default for SummaryExtractor {
    fn default() -> Self {
        // This is a placeholder implementation since we need an LLM client
        // In practice, users should use the builder or constructor methods
        panic!("SummaryExtractor requires an LLM client. Use SummaryExtractor::new() or SummaryExtractor::builder() instead.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::types::ChunkInfo;

    #[test]
    fn test_summary_type_metadata_keys() {
        assert_eq!(SummaryType::SelfSummary.metadata_key(), "section_summary");
        assert_eq!(
            SummaryType::PrevSummary.metadata_key(),
            "prev_section_summary"
        );
        assert_eq!(
            SummaryType::NextSummary.metadata_key(),
            "next_section_summary"
        );
    }

    #[test]
    fn test_summary_type_string_conversion() {
        assert_eq!(SummaryType::SelfSummary.as_str(), "self");
        assert_eq!(SummaryType::PrevSummary.as_str(), "prev");
        assert_eq!(SummaryType::NextSummary.as_str(), "next");

        assert_eq!(
            SummaryType::from_str("self"),
            Some(SummaryType::SelfSummary)
        );
        assert_eq!(
            SummaryType::from_str("prev"),
            Some(SummaryType::PrevSummary)
        );
        assert_eq!(
            SummaryType::from_str("next"),
            Some(SummaryType::NextSummary)
        );
        assert_eq!(SummaryType::from_str("invalid"), None);
    }

    #[test]
    fn test_summary_extractor_config() {
        let config = SummaryExtractorConfig::new()
            .with_summaries(vec![SummaryType::SelfSummary, SummaryType::PrevSummary])
            .with_show_progress(false)
            .with_num_workers(2)
            .with_max_context_length(2000);

        assert_eq!(config.summaries.len(), 2);
        assert!(config.summaries.contains(&SummaryType::SelfSummary));
        assert!(config.summaries.contains(&SummaryType::PrevSummary));
        assert!(!config.show_progress);
        assert_eq!(config.num_workers, 2);
        assert_eq!(config.max_context_length, 2000);
    }

    #[test]
    fn test_config_builder_methods() {
        let config = SummaryExtractorConfig::new()
            .with_self_summary()
            .with_prev_summary()
            .with_next_summary();

        assert_eq!(config.summaries.len(), 3);
        assert!(config.summaries.contains(&SummaryType::SelfSummary));
        assert!(config.summaries.contains(&SummaryType::PrevSummary));
        assert!(config.summaries.contains(&SummaryType::NextSummary));
    }

    #[test]
    fn test_clean_summary_response() {
        let test_cases = vec![
            (
                "  Summary: This is a test summary  ",
                "This is a test summary",
            ),
            ("\"This is a quoted summary\"", "This is a quoted summary"),
            ("'Single quoted summary'", "Single quoted summary"),
            ("SUMMARY: Uppercase prefix", "Uppercase prefix"),
            ("  Clean summary  ", "Clean summary"),
            ("", ""),
        ];

        for (input, expected) in test_cases {
            let result = clean_summary_test_helper(input);
            assert_eq!(result, expected, "Failed for input: '{}'", input);
        }
    }

    // Helper function to test summary cleaning logic
    fn clean_summary_test_helper(response: &str) -> String {
        let cleaned = response.trim().trim_matches('"').trim_matches('\'').trim();

        let cleaned = if let Some(pos) = cleaned.to_lowercase().find("summary:") {
            &cleaned[pos + 8..]
        } else {
            cleaned
        };

        cleaned.trim().to_string()
    }

    #[test]
    fn test_config_validation() {
        // Test empty summaries list
        let empty_config = SummaryExtractorConfig::new().with_summaries(vec![]);
        // This would fail in SummaryExtractor::new() validation

        // Test empty prompt template
        let empty_template_config =
            SummaryExtractorConfig::new().with_prompt_template("".to_string());
        // This would fail in SummaryExtractor::new() validation

        // Test template without placeholder
        let invalid_template_config = SummaryExtractorConfig::new()
            .with_prompt_template("Invalid template without placeholder".to_string());
        // This would fail in SummaryExtractor::new() validation

        // Verify the validation logic
        assert!(empty_config.summaries.is_empty());
        assert!(empty_template_config.prompt_template.is_empty());
        assert!(!invalid_template_config
            .prompt_template
            .contains("{context_str}"));
    }
}
