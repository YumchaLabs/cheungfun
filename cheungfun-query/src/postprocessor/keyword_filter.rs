//! Keyword-based filtering postprocessor.
//!
//! This module implements LlamaIndex's KeywordNodePostprocessor approach,
//! which filters nodes based on keyword presence or absence.

use super::{KeywordFilterConfig, NodePostprocessor};
use async_trait::async_trait;
use cheungfun_core::{CheungfunError, Result, ScoredNode};
use regex::Regex;
use tracing::debug;

/// Keyword-based node filter.
///
/// Based on LlamaIndex's KeywordNodePostprocessor, this filter:
/// 1. Filters nodes based on required keywords
/// 2. Excludes nodes containing excluded keywords
/// 3. Supports case-sensitive and case-insensitive matching
/// 4. Allows configurable minimum required keyword matches
#[derive(Debug)]
pub struct KeywordFilter {
    /// Configuration for keyword filtering.
    config: KeywordFilterConfig,

    /// Compiled regex patterns for required keywords (for performance).
    required_patterns: Vec<Regex>,

    /// Compiled regex patterns for excluded keywords (for performance).
    excluded_patterns: Vec<Regex>,
}

impl KeywordFilter {
    /// Create a new keyword filter.
    pub fn new(config: KeywordFilterConfig) -> Result<Self> {
        let required_patterns =
            Self::compile_patterns(&config.required_keywords, config.case_sensitive)?;
        let excluded_patterns =
            Self::compile_patterns(&config.exclude_keywords, config.case_sensitive)?;

        Ok(Self {
            config,
            required_patterns,
            excluded_patterns,
        })
    }

    /// Create with required keywords only.
    pub fn with_required_keywords(keywords: Vec<String>) -> Result<Self> {
        Self::new(KeywordFilterConfig {
            required_keywords: keywords,
            ..Default::default()
        })
    }

    /// Create with excluded keywords only.
    pub fn with_excluded_keywords(keywords: Vec<String>) -> Result<Self> {
        Self::new(KeywordFilterConfig {
            exclude_keywords: keywords,
            ..Default::default()
        })
    }

    /// Create with both required and excluded keywords.
    pub fn with_keywords(required: Vec<String>, excluded: Vec<String>) -> Result<Self> {
        Self::new(KeywordFilterConfig {
            required_keywords: required,
            exclude_keywords: excluded,
            ..Default::default()
        })
    }

    /// Compile keyword patterns into regex for efficient matching.
    fn compile_patterns(keywords: &[String], case_sensitive: bool) -> Result<Vec<Regex>> {
        let mut patterns = Vec::new();

        for keyword in keywords {
            let pattern = if case_sensitive {
                format!(r"\b{}\b", regex::escape(keyword))
            } else {
                format!(r"(?i)\b{}\b", regex::escape(keyword))
            };

            let regex = Regex::new(&pattern).map_err(|e| CheungfunError::Configuration {
                message: format!("Invalid regex pattern for keyword '{}': {}", keyword, e),
            })?;
            patterns.push(regex);
        }

        Ok(patterns)
    }

    /// Check if content matches required keywords.
    fn matches_required_keywords(&self, content: &str) -> bool {
        if self.required_patterns.is_empty() {
            return true; // No required keywords means all content passes
        }

        let matches = self
            .required_patterns
            .iter()
            .filter(|pattern| pattern.is_match(content))
            .count();

        matches >= self.config.min_required_matches
    }

    /// Check if content contains excluded keywords.
    fn contains_excluded_keywords(&self, content: &str) -> bool {
        self.excluded_patterns
            .iter()
            .any(|pattern| pattern.is_match(content))
    }

    /// Filter a single node based on keyword criteria.
    fn should_keep_node(&self, node: &ScoredNode) -> bool {
        let content = &node.node.content;

        // Check required keywords
        if !self.matches_required_keywords(content) {
            debug!("Node filtered out: insufficient required keyword matches");
            return false;
        }

        // Check excluded keywords
        if self.contains_excluded_keywords(content) {
            debug!("Node filtered out: contains excluded keywords");
            return false;
        }

        true
    }
}

#[async_trait]
impl NodePostprocessor for KeywordFilter {
    async fn postprocess(&self, nodes: Vec<ScoredNode>, _query: &str) -> Result<Vec<ScoredNode>> {
        if nodes.is_empty() {
            return Ok(nodes);
        }

        debug!(
            "Filtering {} nodes with {} required keywords and {} excluded keywords",
            nodes.len(),
            self.config.required_keywords.len(),
            self.config.exclude_keywords.len()
        );

        let filtered_nodes: Vec<ScoredNode> = nodes
            .into_iter()
            .filter(|node| self.should_keep_node(node))
            .collect();

        debug!(
            "Keyword filtering completed: {} nodes remaining",
            filtered_nodes.len()
        );

        Ok(filtered_nodes)
    }

    fn name(&self) -> &'static str {
        "KeywordFilter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::{ChunkInfo, Node};
    use uuid::Uuid;

    fn create_test_node(content: &str) -> ScoredNode {
        let chunk_info = ChunkInfo {
            start_char_idx: Some(0),
            end_char_idx: Some(content.len()),
            chunk_index: 0,
        };
        ScoredNode {
            node: Node::new(content.to_string(), Uuid::new_v4(), chunk_info),
            score: 0.8,
        }
    }

    #[tokio::test]
    async fn test_required_keywords_filter() {
        let config = KeywordFilterConfig {
            required_keywords: vec!["climate".to_string(), "change".to_string()],
            min_required_matches: 1,
            ..Default::default()
        };

        let filter = KeywordFilter::new(config).unwrap();

        let nodes = vec![
            create_test_node("Climate change is a serious issue"),
            create_test_node("Weather patterns are changing"),
            create_test_node("This is about something else entirely"),
        ];

        let filtered = filter.postprocess(nodes, "test query").await.unwrap();

        // Should keep first two nodes (contain "climate" or "change")
        assert_eq!(filtered.len(), 2);
    }

    #[tokio::test]
    async fn test_excluded_keywords_filter() {
        let config = KeywordFilterConfig {
            exclude_keywords: vec!["spam".to_string(), "advertisement".to_string()],
            ..Default::default()
        };

        let filter = KeywordFilter::new(config).unwrap();

        let nodes = vec![
            create_test_node("This is useful content about climate"),
            create_test_node("This is spam content you should ignore"),
            create_test_node("Another useful piece of information"),
        ];

        let filtered = filter.postprocess(nodes, "test query").await.unwrap();

        // Should exclude the spam content
        assert_eq!(filtered.len(), 2);
        assert!(!filtered.iter().any(|n| n.node.content.contains("spam")));
    }

    #[tokio::test]
    async fn test_case_sensitivity() {
        let config = KeywordFilterConfig {
            required_keywords: vec!["Climate".to_string()],
            case_sensitive: true,
            ..Default::default()
        };

        let filter = KeywordFilter::new(config).unwrap();

        let nodes = vec![
            create_test_node("Climate change is serious"),
            create_test_node("climate change is serious"), // lowercase
        ];

        let filtered = filter.postprocess(nodes, "test query").await.unwrap();

        // Should only keep the first node (exact case match)
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].node.content, "Climate change is serious");
    }

    #[tokio::test]
    async fn test_minimum_required_matches() {
        let config = KeywordFilterConfig {
            required_keywords: vec![
                "climate".to_string(),
                "change".to_string(),
                "global".to_string(),
            ],
            min_required_matches: 2,
            ..Default::default()
        };

        let filter = KeywordFilter::new(config).unwrap();

        let nodes = vec![
            create_test_node("Climate change is a global issue"), // 3 matches
            create_test_node("Climate change affects everyone"),  // 2 matches
            create_test_node("Climate is important"),             // 1 match
        ];

        let filtered = filter.postprocess(nodes, "test query").await.unwrap();

        // Should keep first two nodes (>= 2 matches)
        assert_eq!(filtered.len(), 2);
    }
}
