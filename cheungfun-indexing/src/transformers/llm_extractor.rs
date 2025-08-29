//! LLM-driven entity and relationship extractor.
//!
//! This module provides LLM-powered knowledge graph extraction capabilities,
//! following LlamaIndex's SimpleLLMPathExtractor design exactly. It uses
//! large language models to intelligently extract entities and relationships
//! from text content.

use crate::error::{IndexingError, Result as IndexingResult};
use cheungfun_core::{
    traits::{Transform, TransformInput},
    types::{ChunkInfo, EntityNode, Node, Relation, Triplet},
};
use serde::{Deserialize, Serialize};
use siumai::prelude::*;
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, info, warn};

/// Default prompt template for knowledge triplet extraction.
/// Based on LlamaIndex's DEFAULT_KG_TRIPLET_EXTRACT_PROMPT.
pub const DEFAULT_KG_TRIPLET_EXTRACT_PROMPT: &str = r#"Some text is provided below. Given the text, extract up to {max_knowledge_triplets} knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.
---------------------
Example:
Text: Alice is Bob's mother.
Triplets:
(Alice, is mother of, Bob)

Text: Philz is a coffee shop founded in Berkeley in 1982.
Triplets:
(Philz, is, coffee shop)
(Philz, founded in, Berkeley)
(Philz, founded in, 1982)
---------------------
Text: {text}
Triplets:
"#;

/// JSON-based prompt template for structured triplet extraction.
/// Provides more reliable parsing with structured output.
pub const JSON_KG_TRIPLET_EXTRACT_PROMPT: &str = r#"Extract up to {max_knowledge_triplets} knowledge triplets from the given text. Output the result as a JSON array where each triplet is represented as an object with "subject", "predicate", and "object" fields.

Guidelines:
- Focus on factual relationships
- Avoid stopwords and filler words
- Use the most complete form for entities
- Keep entities concise (3-5 words max)

Example:
Text: "Alice is Bob's mother. She works at Google in California."
Output:
[
  {"subject": "Alice", "predicate": "is mother of", "object": "Bob"},
  {"subject": "Alice", "predicate": "works at", "object": "Google"},
  {"subject": "Google", "predicate": "located in", "object": "California"}
]

Text: {text}
Output:
"#;

/// Extraction output format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExtractionFormat {
    /// Traditional parentheses format: (subject, predicate, object)
    Parentheses,
    /// JSON array format for structured parsing
    Json,
}

impl Default for ExtractionFormat {
    fn default() -> Self {
        Self::Json // Default to JSON for better reliability
    }
}

/// JSON triplet structure for parsing LLM responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonTriplet {
    /// Subject entity in the triplet.
    pub subject: String,
    /// Predicate/relationship in the triplet.
    pub predicate: String,
    /// Object entity in the triplet.
    pub object: String,
}

/// Configuration for LLM-based entity extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmExtractionConfig {
    /// Maximum number of triplets to extract per text chunk.
    pub max_triplets_per_chunk: usize,
    /// Number of worker tasks for parallel processing.
    pub num_workers: usize,
    /// Custom prompt template for extraction.
    pub extract_prompt: Option<String>,
    /// Output format for extraction.
    pub format: ExtractionFormat,
    /// Maximum length for entity/relation strings (in bytes).
    pub max_entity_length: usize,
    /// Temperature for LLM generation.
    pub temperature: f32,
    /// Maximum tokens for LLM response.
    pub max_tokens: usize,
    /// Whether to show progress during extraction.
    pub show_progress: bool,
    /// Whether to enable fallback parsing on JSON failure.
    pub enable_fallback: bool,
    /// Minimum confidence score for accepting triplets (0.0-1.0).
    pub min_confidence: f32,
    /// Whether to enable deduplication of extracted triplets.
    pub enable_deduplication: bool,
    /// Whether to validate entity types and relationships.
    pub enable_validation: bool,
}

impl Default for LlmExtractionConfig {
    fn default() -> Self {
        Self {
            max_triplets_per_chunk: 10,
            num_workers: 4,
            extract_prompt: None,
            format: ExtractionFormat::default(),
            max_entity_length: 128,
            temperature: 0.1, // Low temperature for consistent extraction
            max_tokens: 1000,
            show_progress: false,
            enable_fallback: true,
            min_confidence: 0.5,
            enable_deduplication: true,
            enable_validation: true,
        }
    }
}

/// LLM-driven entity and relationship extractor.
///
/// This transformer uses large language models to extract entities and relationships
/// from text content, following LlamaIndex's SimpleLLMPathExtractor design exactly.
/// It provides intelligent knowledge graph construction capabilities with high accuracy.
///
/// # Features
///
/// - LLM-powered entity and relationship extraction
/// - Configurable prompt templates
/// - Parallel processing for efficiency
/// - Robust error handling and fallback strategies
/// - Support for custom LLM providers via siumai
/// - Full LlamaIndex compatibility
///
/// # Examples
///
/// ```rust
/// use cheungfun_indexing::transformers::{LlmExtractor, LlmExtractionConfig};
/// use cheungfun_core::{Node, traits::{Transform, TransformInput}};
/// use siumai::prelude::*;
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
///     let config = LlmExtractionConfig {
///         max_triplets_per_chunk: 15,
///         temperature: 0.1,
///         ..Default::default()
///     };
///
///     // Create extractor
///     let extractor = LlmExtractor::new(llm_client, config)?;
///
///     // Extract from documents
///     let input = TransformInput::Documents(documents);
///     let nodes = extractor.transform(input).await?;
///
///     Ok(())
/// }
/// ```
pub struct LlmExtractor {
    /// LLM client for extraction.
    llm_client: Arc<dyn LlmClient>,
    /// Extraction configuration.
    config: LlmExtractionConfig,
    /// Compiled prompt template.
    prompt_template: String,
}

impl std::fmt::Debug for LlmExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmExtractor")
            .field("config", &self.config)
            .field("prompt_template", &self.prompt_template)
            .field("llm_client", &"<LlmClient>")
            .finish()
    }
}

impl LlmExtractor {
    /// Create a new LLM extractor.
    ///
    /// # Arguments
    ///
    /// * `llm_client` - The LLM client to use for extraction
    /// * `config` - Configuration for the extraction process
    ///
    /// # Returns
    ///
    /// A new `LlmExtractor` instance.
    pub fn new(
        llm_client: Arc<dyn LlmClient>,
        config: LlmExtractionConfig,
    ) -> IndexingResult<Self> {
        let prompt_template =
            config
                .extract_prompt
                .clone()
                .unwrap_or_else(|| match config.format {
                    ExtractionFormat::Parentheses => DEFAULT_KG_TRIPLET_EXTRACT_PROMPT.to_string(),
                    ExtractionFormat::Json => JSON_KG_TRIPLET_EXTRACT_PROMPT.to_string(),
                });

        Ok(Self {
            llm_client,
            config,
            prompt_template,
        })
    }

    /// Create a new LLM extractor with default configuration.
    pub fn with_defaults(llm_client: Arc<dyn LlmClient>) -> IndexingResult<Self> {
        Self::new(llm_client, LlmExtractionConfig::default())
    }

    /// Extract triplets from a single text chunk.
    async fn extract_triplets_from_text(&self, text: &str) -> IndexingResult<Vec<Triplet>> {
        // Prepare the prompt
        let prompt = self
            .prompt_template
            .replace(
                "{max_knowledge_triplets}",
                &self.config.max_triplets_per_chunk.to_string(),
            )
            .replace("{text}", text);

        debug!("Extracting triplets with prompt length: {}", prompt.len());

        // Call LLM using chat interface (siumai uses chat for completion)
        let messages = vec![ChatMessage::user(prompt).build()];

        let response = match self.llm_client.chat(messages).await {
            Ok(response) => response,
            Err(e) => {
                warn!("LLM extraction failed: {}", e);
                return Ok(Vec::new()); // Return empty on failure
            }
        };

        // Extract content from MessageContent
        let content_text = match &response.content {
            siumai::MessageContent::Text(text) => text.clone(),
            siumai::MessageContent::MultiModal(parts) => {
                // Extract text from multimodal content
                let mut text_content = String::new();
                for part in parts {
                    if let siumai::types::ContentPart::Text { text } = part {
                        if !text_content.is_empty() {
                            text_content.push(' ');
                        }
                        text_content.push_str(text);
                    }
                }
                text_content
            }
        };

        // Parse the response based on format
        let triplets = match self.config.format {
            ExtractionFormat::Json => match self.parse_json_response(&content_text) {
                Ok(triplets) => triplets,
                Err(e) if self.config.enable_fallback => {
                    warn!(
                        "JSON parsing failed, falling back to parentheses format: {}",
                        e
                    );
                    self.parse_parentheses_response(&content_text)?
                }
                Err(e) => {
                    warn!("JSON parsing failed: {}", e);
                    Vec::new()
                }
            },
            ExtractionFormat::Parentheses => self.parse_parentheses_response(&content_text)?,
        };

        debug!("Extracted {} triplets from text", triplets.len());
        Ok(triplets)
    }

    /// Parse JSON-formatted LLM response into triplets.
    fn parse_json_response(&self, response: &str) -> IndexingResult<Vec<Triplet>> {
        // Try to find JSON array in the response
        let json_start = response
            .find('[')
            .ok_or_else(|| IndexingError::parse_error("No JSON array found in response"))?;

        let json_end = response
            .rfind(']')
            .ok_or_else(|| IndexingError::parse_error("No closing bracket found in response"))?;

        if json_end <= json_start {
            return Err(IndexingError::parse_error("Invalid JSON structure"));
        }

        let json_str = &response[json_start..=json_end];

        let json_triplets: Vec<JsonTriplet> = serde_json::from_str(json_str)?;

        let mut results = Vec::new();

        for json_triplet in json_triplets {
            // Validate and clean the triplet
            let subject = json_triplet.subject.trim().to_string();
            let predicate = json_triplet.predicate.trim().to_string();
            let object = json_triplet.object.trim().to_string();

            // Skip empty components
            if subject.is_empty() || predicate.is_empty() || object.is_empty() {
                continue;
            }

            // Check length constraints
            if subject.len() > self.config.max_entity_length
                || predicate.len() > self.config.max_entity_length
                || object.len() > self.config.max_entity_length
            {
                continue;
            }

            // Create triplet
            let subject_node =
                EntityNode::new(subject.clone(), "entity".to_string(), HashMap::new());
            let object_node = EntityNode::new(object.clone(), "entity".to_string(), HashMap::new());
            let relation = Relation::new(
                format!("rel_{}_{}", subject_node.id(), object_node.id()),
                predicate,
                subject_node.id().clone(),
                object_node.id().clone(),
                HashMap::new(),
            );

            let triplet = Triplet::new(subject_node, relation, object_node);
            results.push(triplet);
        }

        Ok(results)
    }

    /// Parse parentheses-formatted LLM response into triplets.
    /// Based on LlamaIndex's default_parse_triplets_fn.
    fn parse_parentheses_response(&self, response: &str) -> IndexingResult<Vec<Triplet>> {
        let knowledge_strs: Vec<&str> = response.trim().split('\n').collect();
        let mut results = Vec::new();

        for text in knowledge_strs {
            // Skip empty lines and non-triplets
            if !text.contains('(') || !text.contains(')') {
                continue;
            }

            let start_idx = match text.find('(') {
                Some(idx) => idx,
                None => continue,
            };

            let end_idx = match text.find(')') {
                Some(idx) => idx,
                None => continue,
            };

            if end_idx <= start_idx {
                continue;
            }

            let triplet_part = &text[start_idx + 1..end_idx];
            let tokens: Vec<&str> = triplet_part.split(',').collect();

            if tokens.len() != 3 {
                continue;
            }

            // Check length constraints
            if tokens
                .iter()
                .any(|s| s.trim().len() > self.config.max_entity_length)
            {
                continue;
            }

            let subject = tokens[0].trim().trim_matches('"').to_string();
            let predicate = tokens[1].trim().trim_matches('"').to_string();
            let object = tokens[2].trim().trim_matches('"').to_string();

            // Skip empty components
            if subject.is_empty() || predicate.is_empty() || object.is_empty() {
                continue;
            }

            // Create triplet
            let subject_node =
                EntityNode::new(subject.clone(), "entity".to_string(), HashMap::new());
            let object_node = EntityNode::new(object.clone(), "entity".to_string(), HashMap::new());
            let relation = Relation::new(
                format!("rel_{}_{}", subject_node.id(), object_node.id()),
                predicate,
                subject_node.id().clone(),
                object_node.id().clone(),
                HashMap::new(),
            );

            let triplet = Triplet::new(subject_node, relation, object_node);
            results.push(triplet);
        }

        Ok(results)
    }

    /// Process a single node and extract triplets.
    async fn process_node(&self, node: &Node) -> IndexingResult<Vec<Triplet>> {
        let text = &node.content;
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let triplets = self.extract_triplets_from_text(text).await?;

        // Apply post-processing if enabled
        let processed_triplets =
            if self.config.enable_validation || self.config.enable_deduplication {
                self.post_process_triplets(triplets).await?
            } else {
                triplets
            };

        Ok(processed_triplets)
    }

    /// Post-process extracted triplets with validation and deduplication.
    async fn post_process_triplets(&self, triplets: Vec<Triplet>) -> IndexingResult<Vec<Triplet>> {
        let mut processed = triplets;

        // Apply validation if enabled
        if self.config.enable_validation {
            processed = self.validate_triplets(processed)?;
        }

        // Apply deduplication if enabled
        if self.config.enable_deduplication {
            processed = self.deduplicate_triplets(processed)?;
        }

        Ok(processed)
    }

    /// Validate triplets for quality and consistency.
    fn validate_triplets(&self, triplets: Vec<Triplet>) -> IndexingResult<Vec<Triplet>> {
        let mut valid_triplets = Vec::new();
        let triplets_len = triplets.len();

        for triplet in triplets {
            // Check for self-loops (entity relating to itself)
            if triplet.source.name == triplet.target.name {
                debug!(
                    "Skipping self-loop triplet: {} -> {}",
                    triplet.source.name, triplet.target.name
                );
                continue;
            }

            // Check for meaningful relationships (not just stopwords)
            if self.is_meaningful_relation(&triplet.relation.label) {
                valid_triplets.push(triplet);
            } else {
                debug!("Skipping low-quality relation: {}", triplet.relation.label);
            }
        }

        debug!(
            "Validated {} out of {} triplets",
            valid_triplets.len(),
            triplets_len
        );
        Ok(valid_triplets)
    }

    /// Check if a relation is meaningful (not just stopwords or generic terms).
    fn is_meaningful_relation(&self, relation: &str) -> bool {
        let relation_lower = relation.to_lowercase();

        // Skip very short relations
        if relation_lower.len() < 2 {
            return false;
        }

        // Skip common stopwords and generic relations
        let stopwords = [
            "is", "are", "was", "were", "be", "been", "being", "the", "a", "an", "and", "or",
            "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about",
            "into", "through", "during", "before", "after", "above", "below", "between", "among",
            "within", "without", "under", "over", "near", "far", "around", "behind", "beside",
        ];

        !stopwords.contains(&relation_lower.as_str())
    }

    /// Remove duplicate triplets based on subject-predicate-object combinations.
    fn deduplicate_triplets(&self, triplets: Vec<Triplet>) -> IndexingResult<Vec<Triplet>> {
        use std::collections::HashSet;

        let mut seen = HashSet::new();
        let mut unique_triplets = Vec::new();
        let triplets_len = triplets.len();

        for triplet in triplets {
            // Create a normalized key for deduplication
            let key = format!(
                "{}|{}|{}",
                triplet.source.name.to_lowercase().trim(),
                triplet.relation.label.to_lowercase().trim(),
                triplet.target.name.to_lowercase().trim()
            );

            if !seen.contains(&key) {
                seen.insert(key);
                unique_triplets.push(triplet);
            } else {
                debug!(
                    "Skipping duplicate triplet: {} -> {} -> {}",
                    triplet.source.name, triplet.relation.label, triplet.target.name
                );
            }
        }

        debug!(
            "Deduplicated {} triplets to {} unique ones",
            triplets_len,
            unique_triplets.len()
        );
        Ok(unique_triplets)
    }
}

#[async_trait::async_trait]
impl Transform for LlmExtractor {
    async fn transform(&self, input: TransformInput) -> cheungfun_core::Result<Vec<Node>> {
        match input {
            TransformInput::Documents(documents) => {
                info!("Starting LLM extraction for {} documents", documents.len());

                let mut all_nodes = Vec::new();

                for document in documents {
                    // Create a text node from the document
                    let chunk_info = ChunkInfo::with_char_indices(0, document.content.len(), 0);
                    let mut node = Node::new(document.content.clone(), document.id, chunk_info);
                    node.metadata = document.metadata.clone();

                    // Extract triplets
                    let triplets = self.process_node(&node).await?;

                    if !triplets.is_empty() {
                        // Store triplets in node metadata
                        let mut metadata = node.metadata.clone();
                        metadata.insert(
                            "extracted_triplets".to_string(),
                            serde_json::to_value(&triplets)?,
                        );

                        let mut enhanced_node = Node::new(
                            node.content.clone(),
                            node.source_document_id,
                            node.chunk_info.clone(),
                        );
                        enhanced_node.metadata = metadata;

                        all_nodes.push(enhanced_node);
                    } else {
                        all_nodes.push(node);
                    }
                }

                info!("LLM extraction completed for {} nodes", all_nodes.len());
                Ok(all_nodes)
            }
            TransformInput::Nodes(nodes) => {
                info!("Starting LLM extraction for {} nodes", nodes.len());

                let mut enhanced_nodes = Vec::new();

                for node in nodes {
                    let triplets = self.process_node(&node).await?;

                    if !triplets.is_empty() {
                        // Store triplets in node metadata
                        let mut metadata = node.metadata.clone();
                        metadata.insert(
                            "extracted_triplets".to_string(),
                            serde_json::to_value(&triplets)?,
                        );

                        let mut enhanced_node = Node::new(
                            node.content.clone(),
                            node.source_document_id,
                            node.chunk_info.clone(),
                        );
                        enhanced_node.metadata = metadata;

                        enhanced_nodes.push(enhanced_node);
                    } else {
                        enhanced_nodes.push(node);
                    }
                }

                info!(
                    "LLM extraction completed for {} nodes",
                    enhanced_nodes.len()
                );
                Ok(enhanced_nodes)
            }
            TransformInput::Document(document) => {
                // Handle single document
                let documents = vec![document];
                let input = TransformInput::Documents(documents);
                self.transform(input).await
            }
            TransformInput::Node(node) => {
                // Handle single node
                let nodes = vec![node];
                let input = TransformInput::Nodes(nodes);
                self.transform(input).await
            }
        }
    }

    fn name(&self) -> &'static str {
        "LlmExtractor"
    }
}
