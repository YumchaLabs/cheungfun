//! Metadata extraction transformer for enriching nodes.

use async_trait::async_trait;
use cheungfun_core::traits::NodeTransformer;
use cheungfun_core::{Node, Result as CoreResult};
use regex::Regex;
use std::collections::HashMap;
use tracing::{debug, warn};

use super::{MetadataConfig, utils};

/// Metadata extractor that enriches nodes with additional metadata.
///
/// This transformer analyzes node content and extracts various types of metadata
/// such as titles, statistics, language information, and more.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::transformers::{MetadataExtractor, MetadataConfig};
/// use cheungfun_core::{Node, ChunkInfo, traits::NodeTransformer};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = MetadataConfig::new()
///         .with_title_extraction(true)
///         .with_statistics(true);
///
///     let extractor = MetadataExtractor::with_config(config);
///     let content = "# Title\n\nThis is some content...";
///     let mut node = Node::new(
///         content.to_string(),
///         uuid::Uuid::new_v4(),
///         ChunkInfo {
///             start_offset: 0,
///             end_offset: content.len(),
///             chunk_index: 0,
///         }
///     );
///
///     let enriched_node = extractor.transform_node(node).await?;
///     println!("Extracted metadata: {:?}", enriched_node.metadata);
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MetadataExtractor {
    /// Configuration for metadata extraction.
    config: MetadataConfig,
    /// Compiled regex patterns for efficiency.
    patterns: MetadataPatterns,
}

/// Pre-compiled regex patterns for metadata extraction.
#[derive(Debug, Clone)]
struct MetadataPatterns {
    /// Pattern for detecting email addresses.
    email: Regex,
    /// Pattern for detecting URLs.
    url: Regex,
    /// Pattern for detecting phone numbers.
    phone: Regex,
    /// Pattern for detecting dates.
    date: Regex,
    /// Pattern for detecting markdown headers.
    markdown_header: Regex,
    /// Pattern for detecting code blocks.
    code_block: Regex,
}

impl Default for MetadataPatterns {
    fn default() -> Self {
        Self {
            email: Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap(),
            url: Regex::new(r"https?://[^\s]+").unwrap(),
            phone: Regex::new(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b").unwrap(),
            date: Regex::new(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
                .unwrap(),
            markdown_header: Regex::new(r"^#{1,6}\s+(.+)$").unwrap(),
            code_block: Regex::new(r"```[\s\S]*?```|`[^`]+`").unwrap(),
        }
    }
}

impl MetadataExtractor {
    /// Create a new metadata extractor with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: MetadataConfig::default(),
            patterns: MetadataPatterns::default(),
        }
    }

    /// Create a new metadata extractor with custom configuration.
    #[must_use]
    pub fn with_config(config: MetadataConfig) -> Self {
        Self {
            config,
            patterns: MetadataPatterns::default(),
        }
    }

    /// Get the extractor configuration.
    #[must_use]
    pub fn config(&self) -> &MetadataConfig {
        &self.config
    }

    /// Extract title information from content.
    fn extract_title_metadata(&self, content: &str) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();

        if !self.config.extract_title {
            return metadata;
        }

        // Try to extract title using utility function
        if let Some(title) = utils::extract_title(content) {
            metadata.insert(
                "extracted_title".to_string(),
                serde_json::Value::String(title),
            );
        }

        // Extract markdown headers
        let headers: Vec<String> = self
            .patterns
            .markdown_header
            .captures_iter(content)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect();

        if !headers.is_empty() {
            metadata.insert(
                "markdown_headers".to_string(),
                serde_json::Value::Array(
                    headers.into_iter().map(serde_json::Value::String).collect(),
                ),
            );
        }

        metadata
    }

    /// Extract statistical information from content.
    fn extract_statistics_metadata(&self, content: &str) -> HashMap<String, serde_json::Value> {
        if !self.config.extract_statistics {
            return HashMap::new();
        }

        utils::calculate_statistics(content)
    }

    /// Extract language information from content.
    fn extract_language_metadata(&self, content: &str) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();

        if !self.config.extract_language {
            return metadata;
        }

        if let Some(language) = utils::detect_language_simple(content) {
            metadata.insert(
                "detected_language".to_string(),
                serde_json::Value::String(language),
            );
        }

        metadata
    }

    /// Extract structural information from content.
    fn extract_structural_metadata(&self, content: &str) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();

        // Count different types of content
        let email_count = self.patterns.email.find_iter(content).count();
        let url_count = self.patterns.url.find_iter(content).count();
        let phone_count = self.patterns.phone.find_iter(content).count();
        let date_count = self.patterns.date.find_iter(content).count();
        let code_block_count = self.patterns.code_block.find_iter(content).count();

        if email_count > 0 {
            metadata.insert(
                "email_count".to_string(),
                serde_json::Value::Number(email_count.into()),
            );
        }

        if url_count > 0 {
            metadata.insert(
                "url_count".to_string(),
                serde_json::Value::Number(url_count.into()),
            );
        }

        if phone_count > 0 {
            metadata.insert(
                "phone_count".to_string(),
                serde_json::Value::Number(phone_count.into()),
            );
        }

        if date_count > 0 {
            metadata.insert(
                "date_count".to_string(),
                serde_json::Value::Number(date_count.into()),
            );
        }

        if code_block_count > 0 {
            metadata.insert(
                "code_block_count".to_string(),
                serde_json::Value::Number(code_block_count.into()),
            );
        }

        // Detect content type based on structure
        let content_type = self.detect_content_type(content);
        if !content_type.is_empty() {
            metadata.insert(
                "detected_content_type".to_string(),
                serde_json::Value::String(content_type),
            );
        }

        metadata
    }

    /// Detect the type of content based on structural patterns.
    fn detect_content_type(&self, content: &str) -> String {
        let content_lower = content.to_lowercase();

        // Check for code content
        if self.patterns.code_block.is_match(content)
            || content.lines().any(|line| {
                line.trim_start().starts_with("def ")
                    || line.trim_start().starts_with("function ")
                    || line.trim_start().starts_with("class ")
            })
        {
            return "code".to_string();
        }

        // Check for markdown
        if content.contains("# ")
            || content.contains("## ")
            || content.contains("**")
            || content.contains('*')
        {
            return "markdown".to_string();
        }

        // Check for structured data
        if content_lower.contains("json") && (content.contains('{') || content.contains('[')) {
            return "json".to_string();
        }

        if content.contains('<') && content.contains('>') {
            return "html".to_string();
        }

        // Check for academic/technical content
        if content_lower.contains("abstract")
            || content_lower.contains("introduction")
            || content_lower.contains("methodology")
            || content_lower.contains("conclusion")
        {
            return "academic".to_string();
        }

        // Check for documentation
        if content_lower.contains("api")
            || content_lower.contains("documentation")
            || content_lower.contains("readme")
        {
            return "documentation".to_string();
        }

        "text".to_string()
    }

    /// Extract keyword-like information (simple implementation).
    fn extract_keyword_metadata(&self, content: &str) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();

        if !self.config.extract_keywords {
            return metadata;
        }

        // Simple keyword extraction based on word frequency
        let words: Vec<&str> = content
            .split_whitespace()
            .filter(|word| word.len() > 3 && word.chars().all(char::is_alphabetic))
            .collect();

        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for word in words {
            let word_lower = word.to_lowercase();
            *word_counts.entry(word_lower).or_insert(0) += 1;
        }

        // Get top keywords
        let mut sorted_words: Vec<(String, usize)> = word_counts.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));

        let keywords: Vec<String> = sorted_words
            .into_iter()
            .take(self.config.max_keywords)
            .filter(|(_, count)| *count > 1) // Only include words that appear more than once
            .map(|(word, _)| word)
            .collect();

        if !keywords.is_empty() {
            metadata.insert(
                "extracted_keywords".to_string(),
                serde_json::Value::Array(
                    keywords
                        .into_iter()
                        .map(serde_json::Value::String)
                        .collect(),
                ),
            );
        }

        metadata
    }

    /// Extract all configured metadata from content.
    fn extract_all_metadata(&self, content: &str) -> HashMap<String, serde_json::Value> {
        let mut all_metadata = HashMap::new();

        // Extract different types of metadata
        let title_metadata = self.extract_title_metadata(content);
        let stats_metadata = self.extract_statistics_metadata(content);
        let language_metadata = self.extract_language_metadata(content);
        let structural_metadata = self.extract_structural_metadata(content);
        let keyword_metadata = self.extract_keyword_metadata(content);

        // Merge all metadata
        all_metadata.extend(title_metadata);
        all_metadata.extend(stats_metadata);
        all_metadata.extend(language_metadata);
        all_metadata.extend(structural_metadata);
        all_metadata.extend(keyword_metadata);

        // Add extraction timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        all_metadata.insert(
            "metadata_extracted_at".to_string(),
            serde_json::Value::Number(now.into()),
        );

        all_metadata
    }
}

impl Default for MetadataExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl NodeTransformer for MetadataExtractor {
    async fn transform_node(&self, mut node: Node) -> CoreResult<Node> {
        debug!("Extracting metadata from node {}", node.id);

        if node.content.is_empty() {
            warn!(
                "Node {} has empty content, skipping metadata extraction",
                node.id
            );
            return Ok(node);
        }

        // Extract metadata from content
        let extracted_metadata = self.extract_all_metadata(&node.content);

        // Merge with existing metadata (extracted metadata takes precedence)
        for (key, value) in extracted_metadata {
            node.metadata.insert(key, value);
        }

        debug!(
            "Extracted {} metadata fields for node {}",
            node.metadata.len(),
            node.id
        );

        Ok(node)
    }

    async fn transform_batch(&self, nodes: Vec<Node>) -> CoreResult<Vec<Node>> {
        debug!("Batch extracting metadata from {} nodes", nodes.len());

        let mut results = Vec::new();
        for node in nodes {
            let transformed_node = self.transform_node(node).await?;
            results.push(transformed_node);
        }

        debug!(
            "Batch metadata extraction completed for {} nodes",
            results.len()
        );
        Ok(results)
    }

    fn name(&self) -> &'static str {
        "MetadataExtractor"
    }
}
