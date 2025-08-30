//! In-memory keyword store implementation.
//!
//! This module provides a simple in-memory implementation of the KeywordStore trait,
//! suitable for development, testing, and small-scale applications.

use async_trait::async_trait;
use cheungfun_core::{
    traits::{KeywordEntry, KeywordStore, KeywordStoreStats},
    Result,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, info};
use uuid::Uuid;

/// In-memory keyword store implementation.
///
/// This store keeps all keyword-to-node mappings in memory using HashMap
/// for fast access. It's suitable for development and small-scale applications.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_integrations::keyword_stores::InMemoryKeywordStore;
/// use cheungfun_core::traits::KeywordStore;
/// use std::collections::HashMap;
/// use uuid::Uuid;
///
/// # async fn example() -> cheungfun_core::Result<()> {
/// let store = InMemoryKeywordStore::new();
///
/// let node_id = Uuid::new_v4();
/// let mut keywords = HashMap::new();
/// keywords.insert("rust".to_string(), 5);
/// keywords.insert("programming".to_string(), 3);
///
/// store.add_keywords(node_id, keywords).await?;
///
/// let results = store.search_keywords(&["rust".to_string()]).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct InMemoryKeywordStore {
    /// Keyword to entry mapping
    keywords: Arc<RwLock<HashMap<String, KeywordEntry>>>,

    /// Node to keywords mapping (for efficient node removal)
    node_keywords: Arc<RwLock<HashMap<Uuid, HashMap<String, usize>>>>,
}

impl Default for InMemoryKeywordStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryKeywordStore {
    /// Create a new in-memory keyword store.
    pub fn new() -> Self {
        Self {
            keywords: Arc::new(RwLock::new(HashMap::new())),
            node_keywords: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the number of unique keywords.
    pub fn keyword_count(&self) -> usize {
        self.keywords.read().unwrap().len()
    }

    /// Get the number of nodes with keywords.
    pub fn node_count(&self) -> usize {
        self.node_keywords.read().unwrap().len()
    }

    /// Calculate relevance score for a node based on keyword matches.
    fn calculate_relevance_score(
        &self,
        node_keywords: &HashMap<String, usize>,
        search_keywords: &[String],
    ) -> f32 {
        let mut score = 0.0;
        let total_node_keywords: usize = node_keywords.values().sum();

        if total_node_keywords == 0 {
            return 0.0;
        }

        for search_keyword in search_keywords {
            if let Some(&frequency) = node_keywords.get(search_keyword) {
                // TF-IDF-like scoring: frequency / total_keywords
                score += frequency as f32 / total_node_keywords as f32;
            }
        }

        // Normalize by number of search keywords
        if !search_keywords.is_empty() {
            score /= search_keywords.len() as f32;
        }

        score
    }
}

#[async_trait]
impl KeywordStore for InMemoryKeywordStore {
    async fn add_keywords(&self, node_id: Uuid, keywords: HashMap<String, usize>) -> Result<()> {
        debug!("Adding {} keywords for node {}", keywords.len(), node_id);

        let mut keyword_map = self.keywords.write().unwrap();
        let mut node_map = self.node_keywords.write().unwrap();

        // Store node keywords for efficient lookup
        node_map.insert(node_id, keywords.clone());

        // Update keyword entries
        for (keyword, frequency) in keywords {
            let entry = keyword_map
                .entry(keyword.clone())
                .or_insert_with(|| KeywordEntry::new(keyword));

            entry.add_node(node_id, frequency);
        }

        Ok(())
    }

    async fn update_keywords(&self, node_id: Uuid, keywords: HashMap<String, usize>) -> Result<()> {
        debug!("Updating keywords for node {}", node_id);

        // Remove existing keywords first
        self.remove_node(&node_id).await?;

        // Add new keywords
        self.add_keywords(node_id, keywords).await
    }

    async fn remove_node(&self, node_id: &Uuid) -> Result<()> {
        debug!("Removing keywords for node {}", node_id);

        let mut keyword_map = self.keywords.write().unwrap();
        let mut node_map = self.node_keywords.write().unwrap();

        // Get keywords for this node
        if let Some(node_keywords) = node_map.remove(node_id) {
            // Remove node from each keyword entry
            for keyword in node_keywords.keys() {
                if let Some(entry) = keyword_map.get_mut(keyword) {
                    entry.remove_node(node_id);

                    // Remove empty keyword entries
                    if entry.node_ids.is_empty() {
                        keyword_map.remove(keyword);
                    }
                }
            }
        }

        Ok(())
    }

    async fn search_keywords(&self, keywords: &[String]) -> Result<Vec<(Uuid, f32)>> {
        debug!("Searching for keywords: {:?}", keywords);

        if keywords.is_empty() {
            return Ok(Vec::new());
        }

        let keyword_map = self.keywords.read().unwrap();
        let node_map = self.node_keywords.read().unwrap();
        let mut node_scores: HashMap<Uuid, f32> = HashMap::new();

        // Find all nodes that contain any of the keywords
        for keyword in keywords {
            if let Some(entry) = keyword_map.get(keyword) {
                for &node_id in &entry.node_ids {
                    if let Some(node_keywords) = node_map.get(&node_id) {
                        let score = self.calculate_relevance_score(node_keywords, keywords);
                        node_scores.insert(node_id, score);
                    }
                }
            }
        }

        // Convert to sorted vector
        let mut results: Vec<(Uuid, f32)> = node_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        info!("Found {} nodes matching keywords", results.len());
        Ok(results)
    }

    async fn search_keywords_all(&self, keywords: &[String]) -> Result<Vec<(Uuid, f32)>> {
        debug!(
            "Searching for nodes containing ALL keywords: {:?}",
            keywords
        );

        if keywords.is_empty() {
            return Ok(Vec::new());
        }

        let keyword_map = self.keywords.read().unwrap();
        let node_map = self.node_keywords.read().unwrap();

        // Find nodes that contain ALL keywords
        let mut candidate_nodes: Option<std::collections::HashSet<Uuid>> = None;

        for keyword in keywords {
            if let Some(entry) = keyword_map.get(keyword) {
                let keyword_nodes: std::collections::HashSet<Uuid> =
                    entry.node_ids.iter().copied().collect();

                candidate_nodes = match candidate_nodes {
                    None => Some(keyword_nodes),
                    Some(existing) => {
                        Some(existing.intersection(&keyword_nodes).copied().collect())
                    }
                };
            } else {
                // If any keyword is not found, no nodes can match all keywords
                return Ok(Vec::new());
            }
        }

        let results = if let Some(nodes) = candidate_nodes {
            let mut scored_results = Vec::new();

            for node_id in nodes {
                if let Some(node_keywords) = node_map.get(&node_id) {
                    let score = self.calculate_relevance_score(node_keywords, keywords);
                    scored_results.push((node_id, score));
                }
            }

            // Sort by score
            scored_results
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored_results
        } else {
            Vec::new()
        };

        info!("Found {} nodes containing all keywords", results.len());
        Ok(results)
    }

    async fn get_node_keywords(&self, node_id: &Uuid) -> Result<HashMap<String, usize>> {
        let node_map = self.node_keywords.read().unwrap();
        Ok(node_map.get(node_id).cloned().unwrap_or_default())
    }

    async fn get_keyword_nodes(&self, keyword: &str) -> Result<Vec<(Uuid, usize)>> {
        let keyword_map = self.keywords.read().unwrap();

        if let Some(entry) = keyword_map.get(keyword) {
            let results = entry
                .node_ids
                .iter()
                .map(|&node_id| (node_id, entry.get_node_frequency(&node_id)))
                .collect();
            Ok(results)
        } else {
            Ok(Vec::new())
        }
    }

    async fn get_keyword_entry(&self, keyword: &str) -> Result<Option<KeywordEntry>> {
        let keyword_map = self.keywords.read().unwrap();
        Ok(keyword_map.get(keyword).cloned())
    }

    async fn list_keywords(&self) -> Result<Vec<(String, usize)>> {
        let keyword_map = self.keywords.read().unwrap();
        let results = keyword_map
            .iter()
            .map(|(keyword, entry)| (keyword.clone(), entry.total_frequency))
            .collect();
        Ok(results)
    }

    async fn stats(&self) -> Result<KeywordStoreStats> {
        let keyword_map = self.keywords.read().unwrap();
        let node_map = self.node_keywords.read().unwrap();

        let total_keywords = keyword_map.len();
        let total_mappings: usize = keyword_map.values().map(|entry| entry.node_ids.len()).sum();

        let avg_keywords_per_node = if node_map.is_empty() {
            0.0
        } else {
            let total_node_keywords: usize = node_map.values().map(|keywords| keywords.len()).sum();
            total_node_keywords as f64 / node_map.len() as f64
        };

        // Get top 10 most frequent keywords
        let mut keyword_frequencies: Vec<(String, usize)> = keyword_map
            .iter()
            .map(|(keyword, entry)| (keyword.clone(), entry.total_frequency))
            .collect();
        keyword_frequencies.sort_by(|a, b| b.1.cmp(&a.1));
        keyword_frequencies.truncate(10);

        // Rough size estimation
        let estimated_size_bytes = std::mem::size_of::<HashMap<String, KeywordEntry>>()
            + keyword_map.len()
                * (std::mem::size_of::<String>() + std::mem::size_of::<KeywordEntry>())
            + node_map.len()
                * (std::mem::size_of::<Uuid>() + std::mem::size_of::<HashMap<String, usize>>());

        Ok(KeywordStoreStats {
            total_keywords,
            total_mappings,
            avg_keywords_per_node,
            top_keywords: keyword_frequencies,
            estimated_size_bytes,
        })
    }

    async fn clear(&self) -> Result<()> {
        info!("Clearing all keyword data");

        let mut keyword_map = self.keywords.write().unwrap();
        let mut node_map = self.node_keywords.write().unwrap();

        keyword_map.clear();
        node_map.clear();

        Ok(())
    }

    async fn health_check(&self) -> Result<()> {
        // Simple health check - verify we can read from both maps
        let _keyword_count = self.keywords.read().unwrap().len();
        let _node_count = self.node_keywords.read().unwrap().len();
        Ok(())
    }

    fn name(&self) -> &'static str {
        "InMemoryKeywordStore"
    }
}
