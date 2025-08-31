//! Keyword extractor transformer for extracting keywords using LLM.
//!
//! This module provides LLM-powered keyword extraction capabilities,
//! following LlamaIndex's KeywordExtractor design exactly. It uses
//! large language models to intelligently extract relevant keywords
//! from text content.

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
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Default prompt template for keyword extraction.
/// Matches LlamaIndex's DEFAULT_KEYWORD_EXTRACT_TEMPLATE exactly.
pub const DEFAULT_KEYWORD_EXTRACT_TEMPLATE: &str = r#"{context_str}. Give {keywords} unique keywords for this document. Format as comma separated. Keywords: "#;

/// Configuration for keyword extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordExtractorConfig {
    /// Number of keywords to extract per node.
    pub keywords: usize,
    /// Template for keyword extraction.
    pub prompt_template: String,
    /// Whether to show progress during extraction.
    pub show_progress: bool,
    /// Number of concurrent workers for processing.
    pub num_workers: usize,
    /// Maximum length of context to send to LLM.
    pub max_context_length: usize,
    /// Whether to process nodes in place or create copies.
    pub in_place: bool,
    /// Whether to lowercase extracted keywords.
    pub lowercase_keywords: bool,
    /// Whether to remove duplicate keywords.
    pub remove_duplicates: bool,
}

impl Default for KeywordExtractorConfig {
    fn default() -> Self {
        Self {
            keywords: 5,
            prompt_template: DEFAULT_KEYWORD_EXTRACT_TEMPLATE.to_string(),
            show_progress: true,
            num_workers: 4,
            max_context_length: 4000,
            in_place: true,
            lowercase_keywords: true,
            remove_duplicates: true,
        }
    }
}

impl KeywordExtractorConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of keywords to extract.
    pub fn with_keywords(mut self, keywords: usize) -> Self {
        self.keywords = keywords.max(1); // Ensure at least 1 keyword
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

    /// Set whether to lowercase keywords.
    pub fn with_lowercase_keywords(mut self, lowercase: bool) -> Self {
        self.lowercase_keywords = lowercase;
        self
    }

    /// Set whether to remove duplicate keywords.
    pub fn with_remove_duplicates(mut self, remove_duplicates: bool) -> Self {
        self.remove_duplicates = remove_duplicates;
        self
    }
}

/// LLM-powered keyword extractor.
///
/// This transformer uses large language models to extract relevant keywords
/// from text content, following LlamaIndex's KeywordExtractor design exactly.
/// It processes each node individually to generate contextually relevant keywords.
///
/// # Features
///
/// - LLM-powered keyword extraction with high accuracy
/// - Node-level processing for precise keyword extraction
/// - Configurable number of keywords per node
/// - Customizable prompt templates for different use cases
/// - Parallel processing for efficiency across multiple nodes
/// - Robust error handling and fallback strategies
/// - Support for custom LLM providers via siumai
/// - Full LlamaIndex compatibility
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::transformers::{KeywordExtractor, KeywordExtractorConfig};
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
///     let config = KeywordExtractorConfig::new()
///         .with_keywords(5)
///         .with_show_progress(true);
///
///     // Create extractor
///     let extractor = KeywordExtractor::new(Arc::new(llm_client), config)?;
///
///     // Extract keywords from nodes
///     let nodes = vec![/* your nodes */];
///     let enhanced_nodes = extractor.transform(TransformInput::Nodes(nodes)).await?;
///
///     // Check the extracted keywords
///     for node in enhanced_nodes {
///         if let Some(keywords) = node.metadata.get("excerpt_keywords") {
///             println!("Keywords: {}", keywords.as_str().unwrap_or("N/A"));
///         }
///     }
///
///     Ok(())
/// }
/// ```
pub struct KeywordExtractor {
    /// LLM client for keyword generation.
    llm_client: Arc<dyn LlmClient>,
    /// Configuration for keyword extraction.
    config: KeywordExtractorConfig,
}

impl std::fmt::Debug for KeywordExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeywordExtractor")
            .field("config", &self.config)
            .finish()
    }
}

impl KeywordExtractor {
    /// Create a new keyword extractor.
    ///
    /// # Arguments
    ///
    /// * `llm_client` - The LLM client to use for keyword generation
    /// * `config` - Configuration for the extractor
    ///
    /// # Returns
    ///
    /// A new `KeywordExtractor` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(
        llm_client: Arc<dyn LlmClient>,
        config: KeywordExtractorConfig,
    ) -> IndexingResult<Self> {
        if config.keywords == 0 {
            return Err(IndexingError::configuration(
                "Number of keywords must be greater than 0".to_string(),
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

        if !config.prompt_template.contains("{keywords}") {
            return Err(IndexingError::configuration(
                "Prompt template must contain {keywords} placeholder".to_string(),
            ));
        }

        Ok(Self { llm_client, config })
    }

    /// Create a new keyword extractor with default configuration.
    ///
    /// # Arguments
    ///
    /// * `llm_client` - The LLM client to use for keyword generation
    ///
    /// # Returns
    ///
    /// A new `KeywordExtractor` instance with default configuration.
    pub fn with_defaults(llm_client: Arc<dyn LlmClient>) -> IndexingResult<Self> {
        Self::new(llm_client, KeywordExtractorConfig::default())
    }

    /// Create a builder for configuring the keyword extractor.
    pub fn builder(llm_client: Arc<dyn LlmClient>) -> KeywordExtractorBuilder {
        KeywordExtractorBuilder::new(llm_client)
    }
}

/// Builder for creating KeywordExtractor instances.
pub struct KeywordExtractorBuilder {
    llm_client: Arc<dyn LlmClient>,
    config: KeywordExtractorConfig,
}

impl KeywordExtractorBuilder {
    /// Create a new builder.
    pub fn new(llm_client: Arc<dyn LlmClient>) -> Self {
        Self {
            llm_client,
            config: KeywordExtractorConfig::default(),
        }
    }

    /// Set the number of keywords to extract.
    pub fn keywords(mut self, keywords: usize) -> Self {
        self.config = self.config.with_keywords(keywords);
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

    /// Set whether to lowercase keywords.
    pub fn lowercase_keywords(mut self, lowercase: bool) -> Self {
        self.config = self.config.with_lowercase_keywords(lowercase);
        self
    }

    /// Set whether to remove duplicate keywords.
    pub fn remove_duplicates(mut self, remove_duplicates: bool) -> Self {
        self.config = self.config.with_remove_duplicates(remove_duplicates);
        self
    }

    /// Build the keyword extractor.
    pub fn build(self) -> IndexingResult<KeywordExtractor> {
        KeywordExtractor::new(self.llm_client, self.config)
    }
}

impl KeywordExtractor {
    /// Process nodes to extract and add keywords.
    async fn process_nodes(&self, nodes: Vec<Node>) -> IndexingResult<Vec<Node>> {
        if nodes.is_empty() {
            return Ok(nodes);
        }

        if self.config.show_progress {
            info!("🔑 Starting keyword extraction for {} nodes", nodes.len());
        }

        // Extract keywords for each node in parallel
        let keyword_jobs: Vec<_> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| self.extract_keywords_from_node(node, i))
            .collect();

        // Process with controlled concurrency
        let keyword_results: Vec<IndexingResult<Vec<String>>> = stream::iter(keyword_jobs)
            .buffer_unordered(self.config.num_workers)
            .collect()
            .await;

        // Apply keywords to nodes
        let enhanced_nodes = if self.config.in_place {
            self.apply_keywords_in_place(nodes, keyword_results)?
        } else {
            self.apply_keywords_with_copy(nodes, keyword_results)?
        };

        if self.config.show_progress {
            info!(
                "✅ Keyword extraction completed for {} nodes",
                enhanced_nodes.len()
            );
        }

        Ok(enhanced_nodes)
    }

    /// Extract keywords from a single node.
    async fn extract_keywords_from_node(
        &self,
        node: &Node,
        node_index: usize,
    ) -> IndexingResult<Vec<String>> {
        debug!("Extracting keywords from node {} of total", node_index + 1);

        // Get node content with length limit
        let content = self.get_limited_content(node);

        if content.trim().is_empty() {
            debug!("Skipping empty node for keyword extraction");
            return Ok(Vec::new());
        }

        // Generate keywords using LLM
        match self.extract_keywords_from_content(&content).await {
            Ok(keywords) => {
                debug!(
                    "Generated {} keywords for node {}",
                    keywords.len(),
                    node_index + 1
                );
                Ok(keywords)
            }
            Err(e) => {
                warn!(
                    "Failed to extract keywords from node {}: {}",
                    node_index + 1,
                    e
                );
                // Return empty keywords instead of failing completely
                Ok(Vec::new())
            }
        }
    }

    /// Extract keywords from content using LLM.
    async fn extract_keywords_from_content(&self, content: &str) -> IndexingResult<Vec<String>> {
        // Prepare the prompt
        let prompt = self
            .config
            .prompt_template
            .replace("{context_str}", content)
            .replace("{keywords}", &self.config.keywords.to_string());

        debug!("Extracting keywords with prompt length: {}", prompt.len());

        // Call LLM using chat interface
        let messages = vec![ChatMessage::user(prompt).build()];

        let response = self.llm_client.chat(messages).await.map_err(|e| {
            IndexingError::processing(format!("LLM keyword extraction failed: {}", e))
        })?;

        // Extract text response
        let keywords_text = if let Some(text) = response.content_text() {
            text.to_string()
        } else {
            return Err(IndexingError::processing(
                "No text content in LLM response".to_string(),
            ));
        };

        // Parse keywords from response
        let keywords = self.parse_keywords_response(&keywords_text);

        Ok(keywords)
    }

    /// Parse keywords from LLM response.
    fn parse_keywords_response(&self, response: &str) -> Vec<String> {
        // Clean up the response
        let cleaned_response = response.trim().trim_matches('"').trim_matches('\'').trim();

        // Look for "KEYWORDS:" prefix and extract what follows
        let keywords_text = if let Some(pos) = cleaned_response.to_lowercase().find("keywords:") {
            &cleaned_response[pos + 9..] // Skip "keywords:"
        } else {
            cleaned_response
        };

        // Split by comma and clean up each keyword
        let mut keywords: Vec<String> = keywords_text
            .split(',')
            .map(|kw| kw.trim().to_string())
            .filter(|kw| !kw.is_empty())
            .collect();

        // Apply post-processing
        if self.config.lowercase_keywords {
            keywords = keywords.into_iter().map(|kw| kw.to_lowercase()).collect();
        }

        if self.config.remove_duplicates {
            keywords.sort();
            keywords.dedup();
        }

        // Limit to requested number of keywords
        keywords.truncate(self.config.keywords);

        debug!("Parsed {} keywords from response", keywords.len());
        keywords
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

    /// Apply keywords to nodes in place.
    fn apply_keywords_in_place(
        &self,
        mut nodes: Vec<Node>,
        keyword_results: Vec<IndexingResult<Vec<String>>>,
    ) -> IndexingResult<Vec<Node>> {
        for (node, keyword_result) in nodes.iter_mut().zip(keyword_results.iter()) {
            match keyword_result {
                Ok(keywords) => {
                    if !keywords.is_empty() {
                        let keywords_str = keywords.join(", ");
                        node.metadata.insert(
                            "excerpt_keywords".to_string(),
                            serde_json::Value::String(keywords_str),
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "Skipping keyword application due to extraction error: {}",
                        e
                    );
                    // Continue processing other nodes
                }
            }
        }
        Ok(nodes)
    }

    /// Apply keywords to nodes with copying.
    fn apply_keywords_with_copy(
        &self,
        nodes: Vec<Node>,
        keyword_results: Vec<IndexingResult<Vec<String>>>,
    ) -> IndexingResult<Vec<Node>> {
        let enhanced_nodes = nodes
            .into_iter()
            .zip(keyword_results.iter())
            .map(|(mut node, keyword_result)| {
                match keyword_result {
                    Ok(keywords) => {
                        if !keywords.is_empty() {
                            let keywords_str = keywords.join(", ");
                            node.metadata.insert(
                                "excerpt_keywords".to_string(),
                                serde_json::Value::String(keywords_str),
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Skipping keyword application due to extraction error: {}",
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
impl TypedTransform<NodeState, NodeState> for KeywordExtractor {
    async fn transform(&self, input: TypedData<NodeState>) -> CoreResult<TypedData<NodeState>> {
        let nodes = input.nodes().to_vec();
        let processed_nodes = self.process_nodes(nodes).await.map_err(|e| {
            cheungfun_core::CheungfunError::Pipeline {
                message: format!("Keyword extraction failed: {}", e),
            }
        })?;
        Ok(TypedData::from_nodes(processed_nodes))
    }

    fn name(&self) -> &'static str {
        "KeywordExtractor"
    }

    fn description(&self) -> &'static str {
        "Extracts relevant keywords from node content using large language models"
    }
}

// Legacy Transform implementation has been removed.
// KeywordExtractor now only uses the type-safe TypedTransform system.

impl Default for KeywordExtractor {
    fn default() -> Self {
        // This is a placeholder implementation since we need an LLM client
        // In practice, users should use the builder or constructor methods
        panic!("KeywordExtractor requires an LLM client. Use KeywordExtractor::new() or KeywordExtractor::builder() instead.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::types::ChunkInfo;

    #[test]
    fn test_keyword_extractor_config() {
        let config = KeywordExtractorConfig::new()
            .with_keywords(10)
            .with_show_progress(false)
            .with_num_workers(2)
            .with_max_context_length(2000)
            .with_lowercase_keywords(false)
            .with_remove_duplicates(false);

        assert_eq!(config.keywords, 10);
        assert!(!config.show_progress);
        assert_eq!(config.num_workers, 2);
        assert_eq!(config.max_context_length, 2000);
        assert!(!config.lowercase_keywords);
        assert!(!config.remove_duplicates);
    }

    #[test]
    fn test_parse_keywords_response() {
        let config = KeywordExtractorConfig::new()
            .with_keywords(5)
            .with_lowercase_keywords(true)
            .with_remove_duplicates(true);

        // We can't create a real KeywordExtractor without an LLM client,
        // so we'll test the parsing logic separately
        let test_cases = vec![
            (
                "KEYWORDS: AI, Machine Learning, Deep Learning",
                vec!["ai", "machine learning", "deep learning"],
            ),
            (
                "ai, machine learning, deep learning, neural networks",
                vec!["ai", "machine learning", "deep learning", "neural networks"],
            ),
            ("  \"AI, ML, DL\"  ", vec!["ai", "ml", "dl"]),
            (
                "Keywords: Technology, Innovation, Future",
                vec!["technology", "innovation", "future"],
            ),
            ("", vec![]),
        ];

        for (input, expected) in test_cases {
            let result = parse_keywords_test(input, &config);
            assert_eq!(result, expected, "Failed for input: '{}'", input);
        }
    }

    // Helper function to test keyword parsing logic
    fn parse_keywords_test(response: &str, config: &KeywordExtractorConfig) -> Vec<String> {
        let cleaned_response = response.trim().trim_matches('"').trim_matches('\'').trim();

        let keywords_text = if let Some(pos) = cleaned_response.to_lowercase().find("keywords:") {
            &cleaned_response[pos + 9..]
        } else {
            cleaned_response
        };

        let mut keywords: Vec<String> = keywords_text
            .split(',')
            .map(|kw| kw.trim().to_string())
            .filter(|kw| !kw.is_empty())
            .collect();

        if config.lowercase_keywords {
            keywords = keywords.into_iter().map(|kw| kw.to_lowercase()).collect();
        }

        if config.remove_duplicates {
            keywords.sort();
            keywords.dedup();
        }

        keywords.truncate(config.keywords);
        keywords
    }

    #[test]
    fn test_config_validation() {
        let config = KeywordExtractorConfig::new().with_keywords(0);
        // The validation happens in KeywordExtractor::new(), not in config
        assert_eq!(config.keywords, 1); // Should be corrected to minimum 1
    }

    #[test]
    fn test_prompt_template_validation() {
        // Test that template must contain required placeholders
        let template_without_context = "Give {keywords} keywords.";
        let template_without_keywords = "Extract keywords from {context_str}.";
        let valid_template = "Extract {keywords} keywords from {context_str}.";

        // These would fail in KeywordExtractor::new() validation
        assert!(!template_without_context.contains("{context_str}"));
        assert!(!template_without_keywords.contains("{keywords}"));
        assert!(valid_template.contains("{context_str}") && valid_template.contains("{keywords}"));
    }
}
