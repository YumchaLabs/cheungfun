//! Keyword storage traits for keyword-based indexing.
//!
//! This module defines the traits and types for storing and retrieving
//! keyword-to-document mappings, supporting keyword table indices.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::Result;

/// A keyword entry that maps keywords to document nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordEntry {
    /// The keyword
    pub keyword: String,

    /// Node IDs that contain this keyword
    pub node_ids: Vec<Uuid>,

    /// Frequency of the keyword across all nodes
    pub total_frequency: usize,

    /// Per-node frequency information
    pub node_frequencies: HashMap<Uuid, usize>,
}

impl KeywordEntry {
    /// Create a new keyword entry.
    pub fn new(keyword: String) -> Self {
        Self {
            keyword,
            node_ids: Vec::new(),
            total_frequency: 0,
            node_frequencies: HashMap::new(),
        }
    }

    /// Add a node with keyword frequency.
    pub fn add_node(&mut self, node_id: Uuid, frequency: usize) {
        if !self.node_ids.contains(&node_id) {
            self.node_ids.push(node_id);
        }

        self.node_frequencies.insert(node_id, frequency);
        self.total_frequency = self.node_frequencies.values().sum();
    }

    /// Remove a node.
    pub fn remove_node(&mut self, node_id: &Uuid) {
        self.node_ids.retain(|id| id != node_id);
        if let Some(freq) = self.node_frequencies.remove(node_id) {
            self.total_frequency = self.total_frequency.saturating_sub(freq);
        }
    }

    /// Get frequency for a specific node.
    pub fn get_node_frequency(&self, node_id: &Uuid) -> usize {
        self.node_frequencies.get(node_id).copied().unwrap_or(0)
    }

    /// Check if the entry contains a specific node.
    pub fn contains_node(&self, node_id: &Uuid) -> bool {
        self.node_ids.contains(node_id)
    }
}

/// Statistics for keyword storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordStoreStats {
    /// Total number of unique keywords
    pub total_keywords: usize,

    /// Total number of keyword-node mappings
    pub total_mappings: usize,

    /// Average keywords per node
    pub avg_keywords_per_node: f64,

    /// Most frequent keywords (top 10)
    pub top_keywords: Vec<(String, usize)>,

    /// Storage size estimation in bytes
    pub estimated_size_bytes: usize,
}

impl Default for KeywordStoreStats {
    fn default() -> Self {
        Self {
            total_keywords: 0,
            total_mappings: 0,
            avg_keywords_per_node: 0.0,
            top_keywords: Vec::new(),
            estimated_size_bytes: 0,
        }
    }
}

/// Trait for storing and retrieving keyword-to-document mappings.
///
/// This trait provides the storage interface for keyword table indices,
/// supporting efficient keyword-based document retrieval.
#[async_trait]
pub trait KeywordStore: Send + Sync + std::fmt::Debug {
    /// Add keywords for a specific node.
    ///
    /// # Arguments
    ///
    /// * `node_id` - The ID of the node
    /// * `keywords` - Keywords extracted from the node with their frequencies
    async fn add_keywords(&self, node_id: Uuid, keywords: HashMap<String, usize>) -> Result<()>;

    /// Update keywords for a specific node.
    ///
    /// This replaces all existing keywords for the node.
    ///
    /// # Arguments
    ///
    /// * `node_id` - The ID of the node
    /// * `keywords` - New keywords with their frequencies
    async fn update_keywords(&self, node_id: Uuid, keywords: HashMap<String, usize>) -> Result<()>;

    /// Remove all keywords for a specific node.
    ///
    /// # Arguments
    ///
    /// * `node_id` - The ID of the node to remove
    async fn remove_node(&self, node_id: &Uuid) -> Result<()>;

    /// Search for nodes containing any of the specified keywords.
    ///
    /// # Arguments
    ///
    /// * `keywords` - Keywords to search for
    ///
    /// # Returns
    ///
    /// A list of node IDs that contain any of the keywords, with relevance scores
    async fn search_keywords(&self, keywords: &[String]) -> Result<Vec<(Uuid, f32)>>;

    /// Search for nodes containing all of the specified keywords.
    ///
    /// # Arguments
    ///
    /// * `keywords` - Keywords that must all be present
    ///
    /// # Returns
    ///
    /// A list of node IDs that contain all keywords, with relevance scores
    async fn search_keywords_all(&self, keywords: &[String]) -> Result<Vec<(Uuid, f32)>>;

    /// Get all keywords for a specific node.
    ///
    /// # Arguments
    ///
    /// * `node_id` - The ID of the node
    ///
    /// # Returns
    ///
    /// Keywords associated with the node and their frequencies
    async fn get_node_keywords(&self, node_id: &Uuid) -> Result<HashMap<String, usize>>;

    /// Get all nodes that contain a specific keyword.
    ///
    /// # Arguments
    ///
    /// * `keyword` - The keyword to search for
    ///
    /// # Returns
    ///
    /// Node IDs and their keyword frequencies
    async fn get_keyword_nodes(&self, keyword: &str) -> Result<Vec<(Uuid, usize)>>;

    /// Get information about a specific keyword.
    ///
    /// # Arguments
    ///
    /// * `keyword` - The keyword to get information for
    ///
    /// # Returns
    ///
    /// Keyword entry with all associated information
    async fn get_keyword_entry(&self, keyword: &str) -> Result<Option<KeywordEntry>>;

    /// Get all keywords in the store.
    ///
    /// # Returns
    ///
    /// List of all keywords with their total frequencies
    async fn list_keywords(&self) -> Result<Vec<(String, usize)>>;

    /// Get statistics about the keyword store.
    async fn stats(&self) -> Result<KeywordStoreStats>;

    /// Clear all data from the keyword store.
    ///
    /// # Warning
    ///
    /// This operation is destructive and cannot be undone.
    async fn clear(&self) -> Result<()>;

    /// Check if the keyword store is healthy and accessible.
    async fn health_check(&self) -> Result<()> {
        Ok(())
    }

    /// Get the name of this keyword store implementation.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Configuration for keyword extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordExtractionConfig {
    /// Minimum keyword length
    pub min_keyword_length: usize,

    /// Maximum keyword length
    pub max_keyword_length: usize,

    /// Whether to convert keywords to lowercase
    pub lowercase: bool,

    /// Whether to remove stop words
    pub remove_stop_words: bool,

    /// Custom stop words (in addition to default ones)
    pub custom_stop_words: Vec<String>,

    /// Minimum frequency for a keyword to be included
    pub min_frequency: usize,

    /// Maximum number of keywords per node
    pub max_keywords_per_node: Option<usize>,
}

impl Default for KeywordExtractionConfig {
    fn default() -> Self {
        Self {
            min_keyword_length: 2,
            max_keyword_length: 50,
            lowercase: true,
            remove_stop_words: true,
            custom_stop_words: Vec::new(),
            min_frequency: 1,
            max_keywords_per_node: Some(100),
        }
    }
}

/// Trait for extracting keywords from text content.
pub trait KeywordExtractor: Send + Sync + std::fmt::Debug {
    /// Extract keywords from text content.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content to extract keywords from
    ///
    /// # Returns
    ///
    /// A map of keywords to their frequencies in the content
    fn extract_keywords(&self, content: &str) -> Result<HashMap<String, usize>>;

    /// Get the configuration used by this extractor.
    fn config(&self) -> &KeywordExtractionConfig;
}

/// Simple regex-based keyword extractor.
#[derive(Debug, Clone)]
pub struct SimpleKeywordExtractor {
    config: KeywordExtractionConfig,
    stop_words: std::collections::HashSet<String>,
}

impl SimpleKeywordExtractor {
    /// Create a new simple keyword extractor.
    pub fn new(config: KeywordExtractionConfig) -> Self {
        let mut stop_words = Self::default_stop_words();

        // Add custom stop words
        for word in &config.custom_stop_words {
            stop_words.insert(word.to_lowercase());
        }

        Self { config, stop_words }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(KeywordExtractionConfig::default())
    }

    /// Get default English stop words.
    fn default_stop_words() -> std::collections::HashSet<String> {
        [
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
            "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with", "would",
            "could", "should", "may", "might", "can", "must", "shall", "this", "these", "those",
            "they", "them", "their", "there", "where", "when", "what", "who", "why", "how",
            "which", "while", "during", "before", "after", "above", "below", "up", "down", "out",
            "off", "over", "under", "again", "further", "then", "once", "here", "now", "any",
            "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very",
        ]
        .iter()
        .map(|&s| s.to_string())
        .collect()
    }
}

impl KeywordExtractor for SimpleKeywordExtractor {
    fn extract_keywords(&self, content: &str) -> Result<HashMap<String, usize>> {
        let mut keyword_counts = HashMap::new();

        // Simple tokenization - split by whitespace and punctuation
        let words: Vec<&str> = content
            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|word| !word.is_empty())
            .collect();

        for word in words {
            let processed_word = if self.config.lowercase {
                word.to_lowercase()
            } else {
                word.to_string()
            };

            // Apply length filters
            if processed_word.len() < self.config.min_keyword_length
                || processed_word.len() > self.config.max_keyword_length
            {
                continue;
            }

            // Remove stop words if enabled
            if self.config.remove_stop_words && self.stop_words.contains(&processed_word) {
                continue;
            }

            // Count the keyword
            *keyword_counts.entry(processed_word).or_insert(0) += 1;
        }

        // Apply frequency filter
        keyword_counts.retain(|_, &mut count| count >= self.config.min_frequency);

        // Apply max keywords limit
        if let Some(max_keywords) = self.config.max_keywords_per_node {
            if keyword_counts.len() > max_keywords {
                // Keep the most frequent keywords
                let mut sorted_keywords: Vec<_> = keyword_counts.into_iter().collect();
                sorted_keywords.sort_by(|a, b| b.1.cmp(&a.1));
                sorted_keywords.truncate(max_keywords);
                keyword_counts = sorted_keywords.into_iter().collect();
            }
        }

        Ok(keyword_counts)
    }

    fn config(&self) -> &KeywordExtractionConfig {
        &self.config
    }
}
