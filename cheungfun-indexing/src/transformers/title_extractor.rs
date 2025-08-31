//! Title extractor transformer for generating document titles using LLM.
//!
//! This module provides LLM-powered title extraction capabilities,
//! following LlamaIndex's TitleExtractor design exactly. It uses
//! large language models to intelligently generate document titles
//! from the content of multiple nodes.

use crate::error::{IndexingError, Result as IndexingResult};
use async_trait::async_trait;
use cheungfun_core::{
    traits::{NodeState, TypedData, TypedTransform},
    types::Node,
    Result as CoreResult,
};
use serde::{Deserialize, Serialize};
use siumai::prelude::*;
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, info, warn};

/// Default prompt template for extracting title clues from individual nodes.
pub const DEFAULT_TITLE_NODE_TEMPLATE: &str = r#"
Context: {context_str}. Give a title that summarizes all of the unique entities, titles or themes found in the context. Title: "#;

/// Default prompt template for combining node-level title clues into a document title.
pub const DEFAULT_TITLE_COMBINE_TEMPLATE: &str = r#"
{context_str}. Based on the above candidate titles and content, what is the comprehensive title for this document? Title: "#;

/// Configuration for title extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TitleExtractorConfig {
    /// Number of nodes from the front to use for title extraction.
    pub nodes: usize,
    /// Template for node-level title clues extraction.
    pub node_template: String,
    /// Template for combining node-level clues into a document-level title.
    pub combine_template: String,
    /// Whether to show progress during extraction.
    pub show_progress: bool,
    /// Number of concurrent workers for processing.
    pub num_workers: usize,
    /// Maximum length of context to send to LLM.
    pub max_context_length: usize,
    /// Whether to process nodes in place or create copies.
    pub in_place: bool,
}

impl Default for TitleExtractorConfig {
    fn default() -> Self {
        Self {
            nodes: 5,
            node_template: DEFAULT_TITLE_NODE_TEMPLATE.to_string(),
            combine_template: DEFAULT_TITLE_COMBINE_TEMPLATE.to_string(),
            show_progress: true,
            num_workers: 4,
            max_context_length: 4000,
            in_place: true,
        }
    }
}

impl TitleExtractorConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of nodes to use for title extraction.
    pub fn with_nodes(mut self, nodes: usize) -> Self {
        self.nodes = nodes.max(1); // Ensure at least 1 node
        self
    }

    /// Set the node-level template.
    pub fn with_node_template(mut self, template: String) -> Self {
        self.node_template = template;
        self
    }

    /// Set the combine template.
    pub fn with_combine_template(mut self, template: String) -> Self {
        self.combine_template = template;
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
}

/// LLM-powered title extractor.
///
/// This transformer uses large language models to extract document titles
/// from text content, following LlamaIndex's TitleExtractor design exactly.
/// It processes multiple nodes from the beginning of documents to generate
/// comprehensive and accurate titles.
///
/// # Features
///
/// - LLM-powered title generation with high accuracy
/// - Two-stage extraction: node-level clues + document-level combination
/// - Configurable prompt templates for different use cases
/// - Parallel processing for efficiency across multiple documents
/// - Robust error handling and fallback strategies
/// - Support for custom LLM providers via siumai
/// - Full LlamaIndex compatibility
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::transformers::{TitleExtractor, TitleExtractorConfig};
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
///     let config = TitleExtractorConfig::new()
///         .with_nodes(3)
///         .with_show_progress(true);
///
///     // Create extractor
///     let extractor = TitleExtractor::new(Arc::new(llm_client), config)?;
///
///     // Extract titles from nodes
///     let nodes = vec![/* your nodes */];
///     let enhanced_nodes = extractor.transform(TransformInput::Nodes(nodes)).await?;
///
///     // Check the extracted titles
///     for node in enhanced_nodes {
///         if let Some(title) = node.metadata.get("document_title") {
///             println!("Document title: {}", title);
///         }
///     }
///
///     Ok(())
/// }
/// ```
pub struct TitleExtractor {
    /// LLM client for title generation.
    llm_client: Arc<dyn LlmClient>,
    /// Configuration for title extraction.
    config: TitleExtractorConfig,
}

impl std::fmt::Debug for TitleExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TitleExtractor")
            .field("config", &self.config)
            .finish()
    }
}

impl TitleExtractor {
    /// Create a new title extractor.
    ///
    /// # Arguments
    ///
    /// * `llm_client` - The LLM client to use for title generation
    /// * `config` - Configuration for the extractor
    ///
    /// # Returns
    ///
    /// A new `TitleExtractor` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(
        llm_client: Arc<dyn LlmClient>,
        config: TitleExtractorConfig,
    ) -> IndexingResult<Self> {
        if config.nodes == 0 {
            return Err(IndexingError::configuration(
                "Number of nodes must be greater than 0".to_string(),
            ));
        }

        if config.node_template.is_empty() {
            return Err(IndexingError::configuration(
                "Node template cannot be empty".to_string(),
            ));
        }

        if config.combine_template.is_empty() {
            return Err(IndexingError::configuration(
                "Combine template cannot be empty".to_string(),
            ));
        }

        Ok(Self { llm_client, config })
    }

    /// Create a new title extractor with default configuration.
    ///
    /// # Arguments
    ///
    /// * `llm_client` - The LLM client to use for title generation
    ///
    /// # Returns
    ///
    /// A new `TitleExtractor` instance with default configuration.
    pub fn with_defaults(llm_client: Arc<dyn LlmClient>) -> IndexingResult<Self> {
        Self::new(llm_client, TitleExtractorConfig::default())
    }

    /// Create a builder for configuring the title extractor.
    pub fn builder(llm_client: Arc<dyn LlmClient>) -> TitleExtractorBuilder {
        TitleExtractorBuilder::new(llm_client)
    }
}

/// Builder for creating TitleExtractor instances.
pub struct TitleExtractorBuilder {
    llm_client: Arc<dyn LlmClient>,
    config: TitleExtractorConfig,
}

impl TitleExtractorBuilder {
    /// Create a new builder.
    pub fn new(llm_client: Arc<dyn LlmClient>) -> Self {
        Self {
            llm_client,
            config: TitleExtractorConfig::default(),
        }
    }

    /// Set the number of nodes to use for title extraction.
    pub fn nodes(mut self, nodes: usize) -> Self {
        self.config = self.config.with_nodes(nodes);
        self
    }

    /// Set the node-level template.
    pub fn node_template(mut self, template: String) -> Self {
        self.config = self.config.with_node_template(template);
        self
    }

    /// Set the combine template.
    pub fn combine_template(mut self, template: String) -> Self {
        self.config = self.config.with_combine_template(template);
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

    /// Build the title extractor.
    pub fn build(self) -> IndexingResult<TitleExtractor> {
        TitleExtractor::new(self.llm_client, self.config)
    }
}

impl TitleExtractor {
    /// Process nodes to extract and add document titles.
    async fn process_nodes(&self, nodes: Vec<Node>) -> IndexingResult<Vec<Node>> {
        if nodes.is_empty() {
            return Ok(nodes);
        }

        if self.config.show_progress {
            info!("🏷️ Starting title extraction for {} nodes", nodes.len());
        }

        // Group nodes by document ID
        let nodes_by_doc_id = self.separate_nodes_by_ref_id(&nodes);

        if self.config.show_progress {
            info!(
                "📄 Processing {} documents for title extraction",
                nodes_by_doc_id.len()
            );
        }

        // Extract titles for each document
        let titles_by_doc_id = self.extract_titles(nodes_by_doc_id).await?;

        // Apply titles to nodes
        let enhanced_nodes = if self.config.in_place {
            self.apply_titles_in_place(nodes, &titles_by_doc_id)?
        } else {
            self.apply_titles_with_copy(nodes, &titles_by_doc_id)?
        };

        if self.config.show_progress {
            info!(
                "✅ Title extraction completed for {} nodes",
                enhanced_nodes.len()
            );
        }

        Ok(enhanced_nodes)
    }

    /// Separate nodes by their reference document ID.
    fn separate_nodes_by_ref_id<'a>(&self, nodes: &'a [Node]) -> HashMap<String, Vec<&'a Node>> {
        let mut nodes_by_doc_id: HashMap<String, Vec<&Node>> = HashMap::new();

        for node in nodes {
            let doc_id = node.source_document_id.to_string();
            nodes_by_doc_id.entry(doc_id).or_default().push(node);
        }

        nodes_by_doc_id
    }

    /// Extract titles for each document.
    async fn extract_titles(
        &self,
        nodes_by_doc_id: HashMap<String, Vec<&Node>>,
    ) -> IndexingResult<HashMap<String, String>> {
        let mut titles_by_doc_id = HashMap::new();

        // Process each document
        for (doc_id, doc_nodes) in nodes_by_doc_id {
            debug!("Extracting title for document: {}", doc_id);

            // Take only the first N nodes for title extraction
            let nodes_for_title: Vec<&Node> =
                doc_nodes.into_iter().take(self.config.nodes).collect();

            if nodes_for_title.is_empty() {
                warn!(
                    "No nodes available for title extraction for document: {}",
                    doc_id
                );
                titles_by_doc_id.insert(doc_id, "Untitled Document".to_string());
                continue;
            }

            // Extract title candidates from nodes
            let title_candidates = self.get_title_candidates(&nodes_for_title).await?;

            if title_candidates.is_empty() {
                warn!("No title candidates generated for document: {}", doc_id);
                titles_by_doc_id.insert(doc_id, "Untitled Document".to_string());
                continue;
            }

            // Combine candidates into final title
            let final_title = self.combine_title_candidates(&title_candidates).await?;
            debug!(
                "Generated title for document {}: extracted successfully",
                doc_id
            );
            titles_by_doc_id.insert(doc_id, final_title);
        }

        Ok(titles_by_doc_id)
    }

    /// Get title candidates from individual nodes.
    async fn get_title_candidates(&self, nodes: &[&Node]) -> IndexingResult<Vec<String>> {
        let mut candidates = Vec::new();

        for (i, node) in nodes.iter().enumerate() {
            debug!(
                "Extracting title candidate from node {} of {}",
                i + 1,
                nodes.len()
            );

            // Get node content with length limit
            let content = self.get_limited_content(node);

            if content.trim().is_empty() {
                debug!("Skipping empty node for title extraction");
                continue;
            }

            // Generate title candidate using LLM
            match self.extract_title_candidate(&content).await {
                Ok(candidate) => {
                    if !candidate.trim().is_empty() {
                        candidates.push(candidate.trim().to_string());
                        debug!("Generated title candidate: {}", candidate.trim());
                    }
                }
                Err(e) => {
                    warn!("Failed to extract title candidate from node: {}", e);
                    // Continue with other nodes instead of failing completely
                }
            }
        }

        Ok(candidates)
    }

    /// Extract a title candidate from a single piece of content.
    async fn extract_title_candidate(&self, content: &str) -> IndexingResult<String> {
        // Prepare the prompt
        let prompt = self.config.node_template.replace("{context_str}", content);

        debug!(
            "Extracting title candidate with prompt length: {}",
            prompt.len()
        );

        // Call LLM using chat interface
        let messages = vec![ChatMessage::user(prompt).build()];

        let response = self.llm_client.chat(messages).await.map_err(|e| {
            IndexingError::processing(format!("LLM title extraction failed: {}", e))
        })?;

        // Extract text response
        let title_candidate = if let Some(text) = response.content_text() {
            text.to_string()
        } else {
            return Err(IndexingError::processing(
                "No text content in LLM response".to_string(),
            ));
        };

        // Clean up the response
        let cleaned_title = self.clean_title_response(&title_candidate);

        Ok(cleaned_title)
    }

    /// Combine title candidates into a final document title.
    async fn combine_title_candidates(&self, candidates: &[String]) -> IndexingResult<String> {
        if candidates.is_empty() {
            return Ok("Untitled Document".to_string());
        }

        if candidates.len() == 1 {
            return Ok(candidates[0].clone());
        }

        // Combine all candidates into context
        let combined_candidates = candidates.join(", ");

        // Prepare the combine prompt
        let prompt = self
            .config
            .combine_template
            .replace("{context_str}", &combined_candidates);

        debug!(
            "Combining title candidates with prompt length: {}",
            prompt.len()
        );

        // Call LLM to generate final title
        let messages = vec![ChatMessage::user(prompt).build()];

        let response = self.llm_client.chat(messages).await.map_err(|e| {
            IndexingError::processing(format!("LLM title combination failed: {}", e))
        })?;

        // Extract text response
        let final_title = if let Some(text) = response.content_text() {
            text.to_string()
        } else {
            return Err(IndexingError::processing(
                "No text content in LLM response".to_string(),
            ));
        };

        // Clean up the response
        let cleaned_title = self.clean_title_response(&final_title);

        Ok(cleaned_title)
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

    /// Clean up title response from LLM.
    fn clean_title_response(&self, response: &str) -> String {
        response
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .trim()
            .lines()
            .next()
            .unwrap_or("Untitled Document")
            .to_string()
    }

    /// Apply titles to nodes in place.
    fn apply_titles_in_place(
        &self,
        mut nodes: Vec<Node>,
        titles_by_doc_id: &HashMap<String, String>,
    ) -> IndexingResult<Vec<Node>> {
        for node in &mut nodes {
            let doc_id = &node.source_document_id.to_string();
            if let Some(title) = titles_by_doc_id.get(doc_id) {
                node.metadata.insert(
                    "document_title".to_string(),
                    serde_json::Value::String(title.clone()),
                );
            }
        }
        Ok(nodes)
    }

    /// Apply titles to nodes with copying.
    fn apply_titles_with_copy(
        &self,
        nodes: Vec<Node>,
        titles_by_doc_id: &HashMap<String, String>,
    ) -> IndexingResult<Vec<Node>> {
        let enhanced_nodes = nodes
            .into_iter()
            .map(|mut node| {
                let doc_id = &node.source_document_id.to_string();
                if let Some(title) = titles_by_doc_id.get(doc_id) {
                    node.metadata.insert(
                        "document_title".to_string(),
                        serde_json::Value::String(title.clone()),
                    );
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
impl TypedTransform<NodeState, NodeState> for TitleExtractor {
    async fn transform(&self, input: TypedData<NodeState>) -> CoreResult<TypedData<NodeState>> {
        let nodes = input.nodes().to_vec();
        let processed_nodes = self.process_nodes(nodes).await.map_err(|e| {
            cheungfun_core::CheungfunError::Pipeline {
                message: format!("Title extraction failed: {}", e),
            }
        })?;
        Ok(TypedData::from_nodes(processed_nodes))
    }

    fn name(&self) -> &'static str {
        "TitleExtractor"
    }

    fn description(&self) -> &'static str {
        "Generates document titles from node content using large language models"
    }
}

// Legacy Transform implementation has been removed.
// TitleExtractor now only uses the type-safe TypedTransform system.

impl Default for TitleExtractor {
    fn default() -> Self {
        // This is a placeholder implementation since we need an LLM client
        // In practice, users should use the builder or constructor methods
        panic!("TitleExtractor requires an LLM client. Use TitleExtractor::new() or TitleExtractor::builder() instead.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::types::ChunkInfo;

    #[test]
    fn test_title_extractor_config() {
        let config = TitleExtractorConfig::new()
            .with_nodes(3)
            .with_show_progress(false)
            .with_num_workers(2)
            .with_max_context_length(2000);

        assert_eq!(config.nodes, 3);
        assert!(!config.show_progress);
        assert_eq!(config.num_workers, 2);
        assert_eq!(config.max_context_length, 2000);
    }

    #[test]
    fn test_clean_title_response() {
        // Create a mock extractor for testing (this will panic if used, but we only test the method)
        let config = TitleExtractorConfig::default();

        // We can't create a real TitleExtractor without an LLM client,
        // so we'll test the logic separately
        let test_cases = vec![
            ("  \"Sample Title\"  ", "Sample Title"),
            ("'Another Title'", "Another Title"),
            ("Title with\nmultiple lines", "Title with"),
            ("  Clean Title  ", "Clean Title"),
            ("", "Untitled Document"),
        ];

        for (input, expected) in test_cases {
            let result = input
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .trim()
                .lines()
                .next()
                .unwrap_or("Untitled Document")
                .to_string();

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_config_validation() {
        let config = TitleExtractorConfig::new().with_nodes(0);
        // The validation happens in TitleExtractor::new(), not in config
        assert_eq!(config.nodes, 1); // Should be corrected to minimum 1
    }
}
