//! LLM Chain Extractor - LLM-based content extraction and compression.
//!
//! This module implements an LLM-based postprocessor that extracts and compresses
//! relevant content from retrieved nodes, similar to LlamaIndex's LLMChainExtractor.
//!
//! **Reference**: LlamaIndex LLMChainExtractor
//! - Uses LLM to extract relevant information from nodes
//! - Supports configurable extraction prompts
//! - Provides content compression and summarization
//! - Maintains context relevance while reducing token count

use async_trait::async_trait;
use cheungfun_core::{
    types::ScoredNode,
    Result,
};
use crate::postprocessor::NodePostprocessor;
use serde::{Deserialize, Serialize};
use siumai::prelude::*;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

/// Configuration for LLM Chain Extractor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMChainExtractorConfig {
    /// Maximum number of tokens for extracted content
    pub max_tokens: usize,
    
    /// Whether to preserve original node metadata
    pub preserve_metadata: bool,
    
    /// Minimum relevance score threshold for extraction
    pub relevance_threshold: f32,
    
    /// Whether to use batch processing for multiple nodes
    pub use_batch_processing: bool,
    
    /// Maximum batch size for processing
    pub max_batch_size: usize,
    
    /// Custom extraction prompt template
    pub extraction_prompt: Option<String>,
    
    /// Whether to include node scores in extraction context
    pub include_scores: bool,
}

impl Default for LLMChainExtractorConfig {
    fn default() -> Self {
        Self {
            max_tokens: 500,
            preserve_metadata: true,
            relevance_threshold: 0.0,
            use_batch_processing: true,
            max_batch_size: 5,
            extraction_prompt: None,
            include_scores: false,
        }
    }
}

/// Default extraction prompt template.
const DEFAULT_EXTRACTION_PROMPT: &str = r#"
You are an expert content extractor. Your task is to extract the most relevant information from the given text that answers or relates to the user's query.

Query: {query}

Text to extract from:
{text}

Instructions:
1. Extract only the information that is directly relevant to the query
2. Maintain the original meaning and context
3. Remove redundant or irrelevant information
4. Keep the extracted content concise but complete
5. Preserve important details and facts

Extracted content:
"#;

/// Batch extraction prompt template.
const BATCH_EXTRACTION_PROMPT: &str = r#"
You are an expert content extractor. Your task is to extract the most relevant information from multiple text passages that answer or relate to the user's query.

Query: {query}

Text passages to extract from:

{texts}

Instructions:
1. Extract only the information that is directly relevant to the query from each passage
2. Maintain the original meaning and context
3. Remove redundant or irrelevant information
4. Keep the extracted content concise but complete
5. Preserve important details and facts
6. Format the output as a numbered list corresponding to the input passages

Extracted content:
"#;

/// An LLM-based postprocessor that extracts relevant content from nodes.
///
/// This postprocessor uses a language model to intelligently extract and compress
/// content from retrieved nodes, similar to LlamaIndex's LLMChainExtractor.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_query::postprocessor::{LLMChainExtractor, LLMChainExtractorConfig};
/// use siumai::prelude::*;
/// use std::sync::Arc;
///
/// # async fn example() -> cheungfun_core::Result<()> {
/// let llm_client = Siumai::builder()
///     .openai()
///     .api_key("your-api-key")
///     .model("gpt-3.5-turbo")
///     .build()
///     .await?;
///
/// let config = LLMChainExtractorConfig {
///     max_tokens: 300,
///     use_batch_processing: true,
///     ..Default::default()
/// };
///
/// let extractor = LLMChainExtractor::new(Arc::new(llm_client), config);
/// let processed_nodes = extractor.postprocess(nodes, &query).await?;
/// # Ok(())
/// # }
/// ```
pub struct LLMChainExtractor {
    /// The LLM client for content extraction
    llm_client: Arc<Siumai>,
    
    /// Configuration for extraction behavior
    config: LLMChainExtractorConfig,
}

impl LLMChainExtractor {
    /// Create a new LLMChainExtractor.
    ///
    /// # Arguments
    ///
    /// * `llm_client` - The LLM client to use for extraction
    /// * `config` - Configuration for extraction behavior
    pub fn new(llm_client: Arc<Siumai>, config: LLMChainExtractorConfig) -> Self {
        Self { llm_client, config }
    }

    /// Create an LLMChainExtractor with default configuration.
    pub fn with_defaults(llm_client: Arc<Siumai>) -> Self {
        Self::new(llm_client, LLMChainExtractorConfig::default())
    }

    /// Create an LLMChainExtractor from an LLM client.
    pub fn from_llm(llm_client: Arc<Siumai>) -> Self {
        Self::with_defaults(llm_client)
    }

    /// Extract content from a single node.
    async fn extract_single_node(&self, node: &ScoredNode, query: &str) -> Result<String> {
        let prompt_template = self.config.extraction_prompt
            .as_deref()
            .unwrap_or(DEFAULT_EXTRACTION_PROMPT);

        let mut context = node.node.content.clone();
        
        // Add score information if requested
        if self.config.include_scores {
            context = format!("Relevance Score: {:.3}\n\n{}", node.score, context);
        }

        let prompt = prompt_template
            .replace("{query}", query)
            .replace("{text}", &context);

        debug!("Extracting content from node {} with LLM", node.node.id);

        let response = self.llm_client
            .chat(vec![ChatMessage::user(&prompt).build()])
            .await
            .map_err(|e| cheungfun_core::CheungfunError::external(anyhow::anyhow!("LLM extraction failed: {}", e)))?;

        let extracted_content = response.content_text()
            .unwrap_or_default()
            .trim()
            .to_string();

        if extracted_content.is_empty() {
            warn!("LLM returned empty extraction for node {}", node.node.id);
            return Ok(node.node.content.clone()); // Fallback to original content
        }

        Ok(extracted_content)
    }

    /// Extract content from multiple nodes in batch.
    async fn extract_batch_nodes(&self, nodes: &[ScoredNode], query: &str) -> Result<Vec<String>> {
        let prompt_template = self.config.extraction_prompt
            .as_deref()
            .unwrap_or(BATCH_EXTRACTION_PROMPT);

        // Prepare batch text
        let texts = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let mut content = format!("Passage {}:\n{}", i + 1, node.node.content);
                if self.config.include_scores {
                    content = format!("Passage {} (Score: {:.3}):\n{}", i + 1, node.score, node.node.content);
                }
                content
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = prompt_template
            .replace("{query}", query)
            .replace("{texts}", &texts);

        debug!("Extracting content from {} nodes in batch with LLM", nodes.len());

        let response = self.llm_client
            .chat(vec![ChatMessage::user(&prompt).build()])
            .await
            .map_err(|e| cheungfun_core::CheungfunError::external(anyhow::anyhow!("LLM batch extraction failed: {}", e)))?;

        let batch_response = response.content_text()
            .unwrap_or_default()
            .trim();
        
        if batch_response.is_empty() {
            warn!("LLM returned empty batch extraction");
            return Ok(nodes.iter().map(|n| n.node.content.clone()).collect());
        }

        // Parse batch response - expect numbered list format
        let extracted_contents = self.parse_batch_response(batch_response, nodes.len());
        
        // Fallback to original content if parsing fails
        if extracted_contents.len() != nodes.len() {
            warn!("Batch extraction parsing failed, using original content");
            return Ok(nodes.iter().map(|n| n.node.content.clone()).collect());
        }

        Ok(extracted_contents)
    }

    /// Parse batch extraction response into individual extractions.
    fn parse_batch_response(&self, response: &str, expected_count: usize) -> Vec<String> {
        let lines: Vec<&str> = response.lines().collect();
        let mut extractions = Vec::new();
        let mut current_extraction = String::new();
        let mut in_extraction = false;

        for line in lines {
            let trimmed = line.trim();
            
            // Check if this is a numbered item (1., 2., etc.)
            if trimmed.starts_with(char::is_numeric) && trimmed.contains('.') {
                // Save previous extraction if we have one
                if in_extraction && !current_extraction.trim().is_empty() {
                    extractions.push(current_extraction.trim().to_string());
                }
                
                // Start new extraction
                current_extraction = trimmed
                    .splitn(2, '.')
                    .nth(1)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                in_extraction = true;
            } else if in_extraction {
                // Continue current extraction
                if !current_extraction.is_empty() {
                    current_extraction.push('\n');
                }
                current_extraction.push_str(trimmed);
            }
        }

        // Add the last extraction
        if in_extraction && !current_extraction.trim().is_empty() {
            extractions.push(current_extraction.trim().to_string());
        }

        // If parsing failed, try to split by double newlines
        if extractions.len() != expected_count {
            extractions = response
                .split("\n\n")
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        extractions
    }

    /// Filter nodes by relevance threshold.
    fn filter_by_relevance(&self, nodes: Vec<ScoredNode>) -> Vec<ScoredNode> {
        if self.config.relevance_threshold <= 0.0 {
            return nodes;
        }

        let filtered: Vec<_> = nodes
            .into_iter()
            .filter(|node| node.score >= self.config.relevance_threshold)
            .collect();

        debug!(
            "Filtered nodes by relevance threshold {}: {} remaining",
            self.config.relevance_threshold,
            filtered.len()
        );

        filtered
    }
}

#[async_trait]
impl NodePostprocessor for LLMChainExtractor {
    #[instrument(skip(self, nodes), fields(node_count = nodes.len()))]
    async fn postprocess(&self, nodes: Vec<ScoredNode>, query: &str) -> Result<Vec<ScoredNode>> {
        info!(
            "Starting LLM chain extraction for {} nodes with query: {}",
            nodes.len(),
            query
        );

        if nodes.is_empty() {
            return Ok(nodes);
        }

        // Filter by relevance threshold
        let filtered_nodes = self.filter_by_relevance(nodes);
        
        if filtered_nodes.is_empty() {
            info!("No nodes passed relevance threshold");
            return Ok(Vec::new());
        }

        let mut processed_nodes = Vec::new();

        if self.config.use_batch_processing && filtered_nodes.len() > 1 {
            // Process in batches
            for chunk in filtered_nodes.chunks(self.config.max_batch_size) {
                let extracted_contents = self.extract_batch_nodes(chunk, query).await?;
                
                for (node, extracted_content) in chunk.iter().zip(extracted_contents.iter()) {
                    let mut new_node = node.node.clone();
                    new_node.content = extracted_content.clone();
                    
                    // Preserve metadata if requested
                    if !self.config.preserve_metadata {
                        new_node.metadata.clear();
                    }
                    
                    processed_nodes.push(ScoredNode::new(new_node, node.score));
                }
            }
        } else {
            // Process individually
            for node in filtered_nodes {
                let extracted_content = self.extract_single_node(&node, query).await?;
                
                let mut new_node = node.node.clone();
                new_node.content = extracted_content;
                
                // Preserve metadata if requested
                if !self.config.preserve_metadata {
                    new_node.metadata.clear();
                }
                
                processed_nodes.push(ScoredNode::new(new_node, node.score));
            }
        }

        info!(
            "LLM chain extraction completed: {} nodes processed",
            processed_nodes.len()
        );

        Ok(processed_nodes)
    }

    fn name(&self) -> &'static str {
        "LLMChainExtractor"
    }
}

impl std::fmt::Debug for LLMChainExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LLMChainExtractor")
            .field("config", &self.config)
            .field("llm_client", &"<LLM Client>")
            .finish()
    }
}

/// Builder for LLMChainExtractor.
pub struct LLMChainExtractorBuilder {
    llm_client: Option<Arc<Siumai>>,
    config: LLMChainExtractorConfig,
}

impl Default for LLMChainExtractorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LLMChainExtractorBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            llm_client: None,
            config: LLMChainExtractorConfig::default(),
        }
    }

    /// Set the LLM client.
    pub fn llm_client(mut self, client: Arc<Siumai>) -> Self {
        self.llm_client = Some(client);
        self
    }

    /// Set maximum tokens for extraction.
    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.config.max_tokens = max_tokens;
        self
    }

    /// Set whether to preserve metadata.
    pub fn preserve_metadata(mut self, preserve: bool) -> Self {
        self.config.preserve_metadata = preserve;
        self
    }

    /// Set relevance threshold.
    pub fn relevance_threshold(mut self, threshold: f32) -> Self {
        self.config.relevance_threshold = threshold;
        self
    }

    /// Set whether to use batch processing.
    pub fn use_batch_processing(mut self, use_batch: bool) -> Self {
        self.config.use_batch_processing = use_batch;
        self
    }

    /// Set maximum batch size.
    pub fn max_batch_size(mut self, batch_size: usize) -> Self {
        self.config.max_batch_size = batch_size;
        self
    }

    /// Set custom extraction prompt.
    pub fn extraction_prompt(mut self, prompt: String) -> Self {
        self.config.extraction_prompt = Some(prompt);
        self
    }

    /// Set whether to include scores in extraction context.
    pub fn include_scores(mut self, include: bool) -> Self {
        self.config.include_scores = include;
        self
    }

    /// Build the LLMChainExtractor.
    pub fn build(self) -> Result<LLMChainExtractor> {
        let llm_client = self.llm_client
            .ok_or_else(|| cheungfun_core::CheungfunError::configuration("LLM client is required".to_string()))?;
        
        Ok(LLMChainExtractor::new(llm_client, self.config))
    }
}
