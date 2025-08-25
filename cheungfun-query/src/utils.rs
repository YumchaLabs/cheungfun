//! Utility functions and helper components for query processing.
//!
//! This module provides various utility functions and helper components
//! that support the main query processing functionality.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info};

use cheungfun_core::{
    types::{GeneratedResponse, Query, ScoredNode},
    Result,
};

/// Query optimizer for improving query performance and relevance.
///
/// This component provides various query optimization techniques such as
/// query expansion, spell correction, and semantic enhancement.
#[derive(Debug, Default)]
pub struct QueryOptimizer {
    /// Configuration for query optimization.
    config: QueryOptimizerConfig,
}

/// Configuration for query optimizer.
#[derive(Debug, Clone)]
pub struct QueryOptimizerConfig {
    /// Whether to enable spell correction.
    pub enable_spell_correction: bool,

    /// Whether to enable query expansion.
    pub enable_query_expansion: bool,

    /// Whether to enable stop word removal.
    pub enable_stop_word_removal: bool,

    /// Whether to enable stemming.
    pub enable_stemming: bool,

    /// Maximum number of expanded terms to add.
    pub max_expanded_terms: usize,
}

impl Default for QueryOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_spell_correction: false,
            enable_query_expansion: false,
            enable_stop_word_removal: false,
            enable_stemming: false,
            max_expanded_terms: 3,
        }
    }
}

impl QueryOptimizer {
    /// Create a new query optimizer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new query optimizer with custom configuration.
    #[must_use]
    pub fn with_config(config: QueryOptimizerConfig) -> Self {
        Self { config }
    }

    /// Optimize a query for better retrieval performance.
    pub fn optimize_query(&self, query: &Query) -> Result<Query> {
        let mut optimized_query = query.clone();

        if self.config.enable_spell_correction {
            optimized_query.text = self.correct_spelling(&optimized_query.text)?;
        }

        if self.config.enable_query_expansion {
            optimized_query.text = self.expand_query(&optimized_query.text)?;
        }

        if self.config.enable_stop_word_removal {
            optimized_query.text = self.remove_stop_words(&optimized_query.text)?;
        }

        if self.config.enable_stemming {
            optimized_query.text = self.apply_stemming(&optimized_query.text)?;
        }

        debug!(
            "Optimized query: '{}' -> '{}'",
            query.text, optimized_query.text
        );
        Ok(optimized_query)
    }

    /// Correct spelling errors in the query text.
    fn correct_spelling(&self, text: &str) -> Result<String> {
        // TODO: Implement spell correction
        // For now, just return the original text
        Ok(text.to_string())
    }

    /// Expand the query with related terms.
    fn expand_query(&self, text: &str) -> Result<String> {
        // TODO: Implement query expansion using synonyms, related terms, etc.
        // For now, just return the original text
        Ok(text.to_string())
    }

    /// Remove stop words from the query.
    fn remove_stop_words(&self, text: &str) -> Result<String> {
        // TODO: Implement stop word removal
        // For now, just return the original text
        Ok(text.to_string())
    }

    /// Apply stemming to the query terms.
    fn apply_stemming(&self, text: &str) -> Result<String> {
        // TODO: Implement stemming
        // For now, just return the original text
        Ok(text.to_string())
    }
}

/// Response post-processor for improving response quality and formatting.
///
/// This component provides various post-processing techniques such as
/// fact checking, citation formatting, and response enhancement.
#[derive(Debug, Default)]
pub struct ResponsePostProcessor {
    /// Configuration for response post-processing.
    config: ResponsePostProcessorConfig,
}

/// Configuration for response post-processor.
#[derive(Debug, Clone)]
pub struct ResponsePostProcessorConfig {
    /// Whether to enable citation formatting.
    pub enable_citation_formatting: bool,

    /// Whether to enable fact checking.
    pub enable_fact_checking: bool,

    /// Whether to enable response formatting.
    pub enable_response_formatting: bool,

    /// Whether to enable confidence scoring.
    pub enable_confidence_scoring: bool,

    /// Citation format to use.
    pub citation_format: CitationFormat,
}

/// Different citation formats supported.
#[derive(Debug, Clone)]
pub enum CitationFormat {
    /// Simple numbered citations [1], [2], etc.
    Numbered,

    /// Author-year format (Smith, 2023)
    AuthorYear,

    /// Footnote style
    Footnote,

    /// Custom format
    Custom(String),
}

impl Default for ResponsePostProcessorConfig {
    fn default() -> Self {
        Self {
            enable_citation_formatting: true,
            enable_fact_checking: false,
            enable_response_formatting: true,
            enable_confidence_scoring: false,
            citation_format: CitationFormat::Numbered,
        }
    }
}

impl ResponsePostProcessor {
    /// Create a new response post-processor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new response post-processor with custom configuration.
    #[must_use]
    pub fn with_config(config: ResponsePostProcessorConfig) -> Self {
        Self { config }
    }

    /// Post-process a generated response.
    pub fn process_response(
        &self,
        response: &GeneratedResponse,
        source_nodes: &[ScoredNode],
    ) -> Result<GeneratedResponse> {
        let mut processed_response = response.clone();

        if self.config.enable_citation_formatting {
            processed_response.content =
                self.format_citations(&processed_response.content, source_nodes)?;
        }

        if self.config.enable_response_formatting {
            processed_response.content = self.format_response(&processed_response.content)?;
        }

        if self.config.enable_fact_checking {
            // TODO: Implement fact checking
            debug!("Fact checking enabled but not yet implemented");
        }

        if self.config.enable_confidence_scoring {
            let confidence = self.calculate_confidence_score(response, source_nodes)?;
            processed_response.metadata.insert(
                "confidence_score".to_string(),
                serde_json::json!(confidence),
            );
        }

        info!(
            "Post-processed response with {} characters",
            processed_response.content.len()
        );
        Ok(processed_response)
    }

    /// Format citations in the response.
    fn format_citations(&self, content: &str, source_nodes: &[ScoredNode]) -> Result<String> {
        if source_nodes.is_empty() {
            return Ok(content.to_string());
        }

        let mut formatted_content = content.to_string();

        match &self.config.citation_format {
            CitationFormat::Numbered => {
                // Add numbered citations at the end
                formatted_content.push_str("\n\nSources:\n");
                for (i, node) in source_nodes.iter().enumerate() {
                    if let Some(source) = node.node.metadata.get("source") {
                        formatted_content.push_str(&format!("[{}] {}\n", i + 1, source));
                    }
                }
            }
            CitationFormat::AuthorYear => {
                // TODO: Implement author-year citations
                debug!("Author-year citations not yet implemented");
            }
            CitationFormat::Footnote => {
                // TODO: Implement footnote citations
                debug!("Footnote citations not yet implemented");
            }
            CitationFormat::Custom(_format) => {
                // TODO: Implement custom citation format
                debug!("Custom citation format not yet implemented");
            }
        }

        Ok(formatted_content)
    }

    /// Format the response text for better readability.
    fn format_response(&self, content: &str) -> Result<String> {
        // Basic formatting: ensure proper spacing and line breaks
        let formatted = content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");

        Ok(formatted)
    }

    /// Calculate a confidence score for the response.
    fn calculate_confidence_score(
        &self,
        _response: &GeneratedResponse,
        source_nodes: &[ScoredNode],
    ) -> Result<f64> {
        if source_nodes.is_empty() {
            return Ok(0.0);
        }

        // Simple confidence calculation based on average source scores
        let avg_score = source_nodes
            .iter()
            .map(|node| f64::from(node.score))
            .sum::<f64>()
            / source_nodes.len() as f64;

        // Normalize to 0-1 range (assuming scores are typically 0-1)
        let confidence = avg_score.min(1.0).max(0.0);

        Ok(confidence)
    }
}

/// Utility functions for query processing.
pub mod query_utils {

    /// Extract keywords from a query text.
    #[must_use]
    pub fn extract_keywords(text: &str) -> Vec<String> {
        // Simple keyword extraction: split by whitespace and remove punctuation
        text.split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty() && word.len() > 2)
            .collect()
    }

    /// Calculate text similarity between two strings.
    #[must_use]
    pub fn calculate_text_similarity(text1: &str, text2: &str) -> f32 {
        // Simple Jaccard similarity based on word overlap
        let words1: std::collections::HashSet<_> = extract_keywords(text1).into_iter().collect();
        let words2: std::collections::HashSet<_> = extract_keywords(text2).into_iter().collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Truncate text to a maximum length while preserving word boundaries.
    #[must_use]
    pub fn truncate_text(text: &str, max_length: usize) -> String {
        if text.len() <= max_length {
            return text.to_string();
        }

        let truncated = &text[..max_length];
        if let Some(last_space) = truncated.rfind(' ') {
            format!("{}...", &truncated[..last_space])
        } else {
            format!("{truncated}...")
        }
    }
}

/// Utility functions for response processing.
pub mod response_utils {
    use super::{GeneratedResponse, ScoredNode};

    /// Extract the main points from a response.
    #[must_use]
    pub fn extract_main_points(content: &str) -> Vec<String> {
        // Simple extraction: split by sentences and filter by length
        content
            .split('.')
            .map(|sentence| sentence.trim().to_string())
            .filter(|sentence| sentence.len() > 20 && sentence.len() < 200)
            .collect()
    }

    /// Calculate response quality score based on various factors.
    #[must_use]
    pub fn calculate_response_quality(
        response: &GeneratedResponse,
        source_nodes: &[ScoredNode],
    ) -> f32 {
        let mut score = 0.0;
        let mut factors = 0;

        // Factor 1: Response length (optimal range: 100-1000 characters)
        let length_score = if response.content.len() < 50 {
            0.2
        } else if response.content.len() > 2000 {
            0.6
        } else {
            1.0
        };
        score += length_score;
        factors += 1;

        // Factor 2: Number of source nodes used
        let source_score = if source_nodes.is_empty() {
            0.0
        } else if source_nodes.len() > 5 {
            1.0
        } else {
            source_nodes.len() as f32 / 5.0
        };
        score += source_score;
        factors += 1;

        // Factor 3: Average source relevance
        if !source_nodes.is_empty() {
            let avg_relevance =
                source_nodes.iter().map(|node| node.score).sum::<f32>() / source_nodes.len() as f32;
            score += avg_relevance;
            factors += 1;
        }

        if factors > 0 {
            score / factors as f32
        } else {
            0.0
        }
    }
}

/// Simple in-memory cache for query responses.
///
/// This cache provides basic caching functionality with TTL support.
/// For production use, consider using a more sophisticated caching solution.
pub struct QueryCache {
    /// Internal cache storage.
    cache: std::sync::RwLock<HashMap<String, CacheEntry>>,

    /// Default TTL for cache entries.
    default_ttl: Duration,
}

/// A cache entry with TTL support.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The cached response.
    response: GeneratedResponse,

    /// When this entry was created.
    created_at: Instant,

    /// Time-to-live for this entry.
    ttl: Duration,
}

impl QueryCache {
    /// Create a new query cache with default TTL.
    #[must_use]
    pub fn new(default_ttl: Duration) -> Self {
        Self {
            cache: std::sync::RwLock::new(HashMap::new()),
            default_ttl,
        }
    }

    /// Get a cached response for a query.
    pub fn get(&self, query: &str) -> Option<GeneratedResponse> {
        let cache = self.cache.read().ok()?;
        let entry = cache.get(query)?;

        // Check if entry has expired
        if entry.created_at.elapsed() > entry.ttl {
            drop(cache);
            // Remove expired entry
            if let Ok(mut cache) = self.cache.write() {
                cache.remove(query);
            }
            return None;
        }

        Some(entry.response.clone())
    }

    /// Store a response in the cache.
    pub fn put(&self, query: &str, response: GeneratedResponse) {
        self.put_with_ttl(query, response, self.default_ttl);
    }

    /// Store a response in the cache with custom TTL.
    pub fn put_with_ttl(&self, query: &str, response: GeneratedResponse, ttl: Duration) {
        if let Ok(mut cache) = self.cache.write() {
            let entry = CacheEntry {
                response,
                created_at: Instant::now(),
                ttl,
            };
            cache.insert(query.to_string(), entry);
        }
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Remove expired entries from the cache.
    pub fn cleanup_expired(&self) {
        if let Ok(mut cache) = self.cache.write() {
            let now = Instant::now();
            cache.retain(|_, entry| now.duration_since(entry.created_at) <= entry.ttl);
        }
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        if let Ok(cache) = self.cache.read() {
            let total_entries = cache.len();
            let expired_entries = cache
                .values()
                .filter(|entry| entry.created_at.elapsed() > entry.ttl)
                .count();

            CacheStats {
                total_entries,
                expired_entries,
                active_entries: total_entries - expired_entries,
            }
        } else {
            CacheStats::default()
        }
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of entries in cache.
    pub total_entries: usize,

    /// Number of expired entries.
    pub expired_entries: usize,

    /// Number of active (non-expired) entries.
    pub active_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_keywords() {
        let keywords = query_utils::extract_keywords("What is machine learning?");
        assert!(keywords.contains(&"machine".to_string()));
        assert!(keywords.contains(&"learning".to_string()));
        assert!(!keywords.contains(&"is".to_string())); // Too short
    }

    #[test]
    fn test_text_similarity() {
        let similarity = query_utils::calculate_text_similarity(
            "machine learning algorithms",
            "learning machine algorithms",
        );
        assert!(similarity > 0.8); // Should be high similarity
    }

    #[test]
    fn test_truncate_text() {
        let text = "This is a long sentence that should be truncated properly";
        let truncated = query_utils::truncate_text(text, 20);
        assert!(truncated.len() <= 23); // 20 + "..."
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_query_cache() {
        let cache = QueryCache::new(Duration::from_secs(1));
        let response = GeneratedResponse {
            content: "Test response".to_string(),
            source_nodes: vec![],
            metadata: HashMap::new(),
            usage: None,
        };

        // Test put and get
        cache.put("test query", response.clone());
        let cached = cache.get("test query");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().content, "Test response");

        // Test cache miss
        let missing = cache.get("missing query");
        assert!(missing.is_none());
    }
}
