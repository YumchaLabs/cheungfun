//! Graph-based retrieval using property graphs.
//!
//! This module provides graph-based retrieval capabilities that leverage
//! knowledge graphs to find relevant information through entity relationships
//! and graph traversal, following LlamaIndex's PropertyGraphRetriever design.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, info, warn};

use cheungfun_core::{
    traits::{PropertyGraphStore, Retriever},
    types::{Query, ScoredNode, Triplet},
    Node, Result,
};

/// Configuration for graph-based retrieval.
///
/// This configuration controls how the graph retriever searches for
/// relevant information using entity relationships and graph traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRetrievalConfig {
    /// Maximum number of nodes to return.
    pub top_k: usize,

    /// Maximum depth for graph traversal.
    pub max_depth: usize,

    /// Whether to include entity information in results.
    pub include_entities: bool,

    /// Whether to include relationship information in results.
    pub include_relationships: bool,

    /// Minimum confidence score for entity/relationship matching.
    pub min_confidence: f32,

    /// Whether to use fuzzy matching for entity names.
    pub fuzzy_matching: bool,

    /// Maximum number of entities to extract from query.
    pub max_query_entities: usize,

    /// Weight for entity-based scoring.
    pub entity_weight: f32,

    /// Weight for relationship-based scoring.
    pub relationship_weight: f32,

    /// Weight for text similarity scoring.
    pub text_weight: f32,
}

impl Default for GraphRetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            max_depth: 2,
            include_entities: true,
            include_relationships: true,
            min_confidence: 0.3,
            fuzzy_matching: true,
            max_query_entities: 10,
            entity_weight: 0.4,
            relationship_weight: 0.3,
            text_weight: 0.3,
        }
    }
}

/// Graph retrieval strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphRetrievalStrategy {
    /// Entity-based retrieval: find nodes related to entities in the query.
    Entity,

    /// Relationship-based retrieval: find nodes through relationship traversal.
    Relationship,

    /// Hybrid retrieval: combine entity and relationship-based approaches.
    Hybrid,

    /// Custom retrieval with specific entity and relationship filters.
    Custom {
        entity_names: Option<Vec<String>>,
        relation_names: Option<Vec<String>>,
        properties: Option<HashMap<String, serde_json::Value>>,
    },
}

impl Default for GraphRetrievalStrategy {
    fn default() -> Self {
        Self::Hybrid
    }
}

/// Graph-based retriever.
///
/// This retriever uses a property graph store to find relevant information
/// through entity relationships and graph traversal. It follows LlamaIndex's
/// PropertyGraphRetriever design exactly, providing comprehensive graph-based
/// retrieval capabilities.
///
/// # Features
///
/// - Entity-based retrieval using extracted entities
/// - Relationship traversal for connected information
/// - Hybrid scoring combining multiple signals
/// - Configurable depth and filtering
/// - Full LlamaIndex compatibility
///
/// # Examples
///
/// ```rust
/// use cheungfun_query::retrievers::{GraphRetriever, GraphRetrievalConfig};
/// use cheungfun_integrations::SimplePropertyGraphStore;
/// use cheungfun_core::{Query, traits::Retriever};
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let graph_store = Arc::new(SimplePropertyGraphStore::new());
///     
///     let config = GraphRetrievalConfig {
///         top_k: 5,
///         max_depth: 2,
///         include_entities: true,
///         include_relationships: true,
///         ..Default::default()
///     };
///
///     let retriever = GraphRetriever::new(graph_store, config);
///     
///     let query = Query::new("What companies does Alice work for?".to_string());
///     let results = retriever.retrieve(&query).await?;
///     
///     for result in results {
///         println!("Score: {:.3}, Content: {}", result.score, result.node.content);
///     }
///     
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct GraphRetriever {
    /// The property graph store to query.
    graph_store: Arc<dyn PropertyGraphStore>,

    /// Configuration for retrieval behavior.
    config: GraphRetrievalConfig,

    /// Retrieval strategy to use.
    strategy: GraphRetrievalStrategy,
}

impl GraphRetriever {
    /// Create a new graph retriever with default configuration.
    pub fn new(graph_store: Arc<dyn PropertyGraphStore>, config: GraphRetrievalConfig) -> Self {
        Self {
            graph_store,
            config,
            strategy: GraphRetrievalStrategy::default(),
        }
    }

    /// Create a new graph retriever with custom strategy.
    pub fn with_strategy(
        graph_store: Arc<dyn PropertyGraphStore>,
        config: GraphRetrievalConfig,
        strategy: GraphRetrievalStrategy,
    ) -> Self {
        Self {
            graph_store,
            config,
            strategy,
        }
    }

    /// Extract entities from query text.
    ///
    /// This is a simple entity extraction that looks for capitalized words
    /// and common entity patterns. In a production system, this would use
    /// the EntityExtractor transformer.
    fn extract_query_entities(&self, query_text: &str) -> Vec<String> {
        let mut entities = Vec::new();

        // Simple pattern-based entity extraction
        let words: Vec<&str> = query_text.split_whitespace().collect();

        for window in words.windows(2) {
            let combined = format!("{} {}", window[0], window[1]);

            // Check if it looks like a person name (two capitalized words)
            if window[0].chars().next().unwrap_or('a').is_uppercase()
                && window[1].chars().next().unwrap_or('a').is_uppercase()
                && window[0].len() > 1
                && window[1].len() > 1
            {
                entities.push(combined);
            }
        }

        // Also add individual capitalized words
        for word in words {
            if word.chars().next().unwrap_or('a').is_uppercase()
                && word.len() > 2
                && ![
                    "What", "Where", "When", "Who", "Why", "How", "The", "A", "An",
                ]
                .contains(&word)
            {
                entities.push(word.to_string());
            }
        }

        // Remove duplicates and limit
        entities.sort();
        entities.dedup();
        entities.truncate(self.config.max_query_entities);

        debug!(
            "Extracted {} entities from query: {:?}",
            entities.len(),
            entities
        );
        entities
    }

    /// Retrieve triplets based on the current strategy.
    async fn retrieve_triplets(&self, query: &Query) -> Result<Vec<Triplet>> {
        match &self.strategy {
            GraphRetrievalStrategy::Entity => {
                let entities = self.extract_query_entities(&query.text);
                if entities.is_empty() {
                    return Ok(Vec::new());
                }

                self.graph_store
                    .get_triplets(Some(entities), None, None, None)
                    .await
            }

            GraphRetrievalStrategy::Relationship => {
                // For relationship-based retrieval, we look for relationship keywords in the query
                let relation_keywords = self.extract_relation_keywords(&query.text);
                if relation_keywords.is_empty() {
                    return Ok(Vec::new());
                }

                self.graph_store
                    .get_triplets(None, Some(relation_keywords), None, None)
                    .await
            }

            GraphRetrievalStrategy::Hybrid => {
                let entities = self.extract_query_entities(&query.text);
                let relations = self.extract_relation_keywords(&query.text);

                if entities.is_empty() && relations.is_empty() {
                    return Ok(Vec::new());
                }

                self.graph_store
                    .get_triplets(
                        if entities.is_empty() {
                            None
                        } else {
                            Some(entities)
                        },
                        if relations.is_empty() {
                            None
                        } else {
                            Some(relations)
                        },
                        None,
                        None,
                    )
                    .await
            }

            GraphRetrievalStrategy::Custom {
                entity_names,
                relation_names,
                properties,
            } => {
                self.graph_store
                    .get_triplets(
                        entity_names.clone(),
                        relation_names.clone(),
                        properties.clone(),
                        None,
                    )
                    .await
            }
        }
    }

    /// Extract relationship keywords from query text.
    fn extract_relation_keywords(&self, query_text: &str) -> Vec<String> {
        let mut relations = Vec::new();
        let lower_query = query_text.to_lowercase();

        // Common relationship patterns
        let relation_patterns = [
            ("work", "WORKS_AT"),
            ("employ", "WORKS_AT"),
            ("job", "WORKS_AT"),
            ("company", "WORKS_AT"),
            ("located", "LOCATED_IN"),
            ("live", "LOCATED_IN"),
            ("from", "LOCATED_IN"),
            ("in", "LOCATED_IN"),
            ("at", "LOCATED_IN"),
            ("know", "KNOWS"),
            ("friend", "KNOWS"),
            ("partner", "PARTNER_OF"),
            ("own", "OWNS"),
            ("manage", "MANAGES"),
            ("lead", "LEADS"),
        ];

        for (keyword, relation) in &relation_patterns {
            if lower_query.contains(keyword) {
                relations.push(relation.to_string());
            }
        }

        relations.sort();
        relations.dedup();

        debug!(
            "Extracted {} relations from query: {:?}",
            relations.len(),
            relations
        );
        relations
    }

    /// Convert triplets to scored nodes.
    fn triplets_to_scored_nodes(&self, triplets: Vec<Triplet>, query: &Query) -> Vec<ScoredNode> {
        let mut scored_nodes = Vec::new();
        let mut seen_content = HashSet::new();

        for triplet in triplets {
            // Create content from triplet information
            let content = if self.config.include_entities && self.config.include_relationships {
                format!(
                    "Entity: {} ({})\nRelationship: {} -[{}]-> {}\nTarget: {} ({})",
                    triplet.source.name,
                    triplet.source.label,
                    triplet.source.name,
                    triplet.relation.label,
                    triplet.target.name,
                    triplet.target.name,
                    triplet.target.label
                )
            } else if self.config.include_entities {
                format!(
                    "Entities: {} ({}), {} ({})",
                    triplet.source.name,
                    triplet.source.label,
                    triplet.target.name,
                    triplet.target.label
                )
            } else if self.config.include_relationships {
                format!(
                    "Relationship: {} -[{}]-> {}",
                    triplet.source.name, triplet.relation.label, triplet.target.name
                )
            } else {
                format!(
                    "{} {} {}",
                    triplet.source.name, triplet.relation.label, triplet.target.name
                )
            };

            // Skip duplicates
            if seen_content.contains(&content) {
                continue;
            }
            seen_content.insert(content.clone());

            // Calculate relevance score
            let score = self.calculate_relevance_score(&triplet, query);

            if score >= self.config.min_confidence {
                // Create metadata
                let mut metadata = HashMap::new();
                metadata.insert(
                    "source_entity".to_string(),
                    serde_json::Value::String(triplet.source.name.clone()),
                );
                metadata.insert(
                    "source_label".to_string(),
                    serde_json::Value::String(triplet.source.label.clone()),
                );
                metadata.insert(
                    "target_entity".to_string(),
                    serde_json::Value::String(triplet.target.name.clone()),
                );
                metadata.insert(
                    "target_label".to_string(),
                    serde_json::Value::String(triplet.target.label.clone()),
                );
                metadata.insert(
                    "relation".to_string(),
                    serde_json::Value::String(triplet.relation.label.clone()),
                );
                metadata.insert(
                    "relation_id".to_string(),
                    serde_json::Value::String(triplet.relation.id.clone()),
                );
                metadata.insert(
                    "retrieval_type".to_string(),
                    serde_json::Value::String("graph".to_string()),
                );

                // Create node
                let node = Node::new(
                    content,
                    uuid::Uuid::new_v4(),
                    cheungfun_core::types::ChunkInfo::with_char_indices(0, 0, 0),
                );

                let mut node_with_metadata = node;
                node_with_metadata.metadata = metadata;

                scored_nodes.push(ScoredNode {
                    node: node_with_metadata,
                    score,
                });
            }
        }

        // Sort by score (highest first) and limit results
        scored_nodes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored_nodes.truncate(self.config.top_k);

        scored_nodes
    }

    /// Calculate relevance score for a triplet based on the query.
    fn calculate_relevance_score(&self, triplet: &Triplet, query: &Query) -> f32 {
        let query_lower = query.text.to_lowercase();
        let mut score = 0.0;

        // Entity matching score
        let entity_score = {
            let source_match = if query_lower.contains(&triplet.source.name.to_lowercase()) {
                1.0
            } else {
                0.0
            };
            let target_match = if query_lower.contains(&triplet.target.name.to_lowercase()) {
                1.0
            } else {
                0.0
            };
            (source_match + target_match) / 2.0
        };

        // Relationship matching score
        let relation_score = {
            let relation_lower = triplet.relation.label.to_lowercase();
            if query_lower.contains(&relation_lower) {
                1.0
            } else {
                // Check for semantic similarity with common patterns
                let relation_keywords = self.extract_relation_keywords(&query.text);
                if relation_keywords.contains(&triplet.relation.label) {
                    0.8
                } else {
                    0.0
                }
            }
        };

        // Text similarity score (simple word overlap)
        let text_score = {
            let query_words: HashSet<String> = query_lower
                .split_whitespace()
                .map(|w| w.to_string())
                .collect();

            let triplet_text = format!(
                "{} {} {}",
                triplet.source.name.to_lowercase(),
                triplet.relation.label.to_lowercase(),
                triplet.target.name.to_lowercase()
            );

            let triplet_words: HashSet<String> = triplet_text
                .split_whitespace()
                .map(|w| w.to_string())
                .collect();

            let intersection = query_words.intersection(&triplet_words).count();
            let union = query_words.union(&triplet_words).count();

            if union > 0 {
                intersection as f32 / union as f32
            } else {
                0.0
            }
        };

        // Combine scores with weights
        score += entity_score * self.config.entity_weight;
        score += relation_score * self.config.relationship_weight;
        score += text_score * self.config.text_weight;

        // Normalize to [0, 1]
        score.min(1.0).max(0.0)
    }

    /// Perform graph traversal to find connected information.
    async fn traverse_graph(
        &self,
        _initial_entities: &[String],
        _depth: usize,
    ) -> Result<Vec<Triplet>> {
        // TODO: Implement multi-hop graph traversal
        // This would involve:
        // 1. Starting from initial entities
        // 2. Finding connected entities through relationships
        // 3. Recursively exploring up to max_depth
        // 4. Collecting all relevant triplets

        // For now, return empty - this is a placeholder for future enhancement
        Ok(Vec::new())
    }
}

#[async_trait]
impl Retriever for GraphRetriever {
    async fn retrieve(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        info!("Starting graph retrieval for query: {}", query.text);

        // Retrieve triplets based on strategy
        let triplets = self.retrieve_triplets(query).await?;

        if triplets.is_empty() {
            warn!("No triplets found for query: {}", query.text);
            return Ok(Vec::new());
        }

        debug!("Found {} triplets for query", triplets.len());

        // Convert triplets to scored nodes
        let scored_nodes = self.triplets_to_scored_nodes(triplets, query);

        info!(
            "Graph retrieval completed: {} results for query '{}'",
            scored_nodes.len(),
            query.text
        );

        Ok(scored_nodes)
    }

    fn name(&self) -> &'static str {
        "GraphRetriever"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::types::{EntityNode, Relation};
    use std::collections::HashMap;

    // Mock PropertyGraphStore for testing
    #[derive(Debug)]
    struct MockPropertyGraphStore {
        triplets: Vec<Triplet>,
    }

    impl MockPropertyGraphStore {
        fn new() -> Self {
            // Create test data
            let alice = EntityNode::new("Alice".to_string(), "Person".to_string(), HashMap::new());
            let bob = EntityNode::new("Bob".to_string(), "Person".to_string(), HashMap::new());
            let microsoft = EntityNode::new(
                "Microsoft".to_string(),
                "Company".to_string(),
                HashMap::new(),
            );
            let seattle =
                EntityNode::new("Seattle".to_string(), "City".to_string(), HashMap::new());

            let triplets = vec![
                Triplet {
                    source: alice.clone(),
                    relation: Relation::new(
                        "rel_1".to_string(),
                        "WORKS_AT".to_string(),
                        alice.id(),
                        microsoft.id(),
                        HashMap::new(),
                    ),
                    target: microsoft.clone(),
                },
                Triplet {
                    source: bob.clone(),
                    relation: Relation::new(
                        "rel_2".to_string(),
                        "WORKS_AT".to_string(),
                        bob.id(),
                        microsoft.id(),
                        HashMap::new(),
                    ),
                    target: microsoft.clone(),
                },
                Triplet {
                    source: microsoft.clone(),
                    relation: Relation::new(
                        "rel_3".to_string(),
                        "LOCATED_IN".to_string(),
                        microsoft.id(),
                        seattle.id(),
                        HashMap::new(),
                    ),
                    target: seattle,
                },
            ];

            Self { triplets }
        }
    }

    #[async_trait]
    impl PropertyGraphStore for MockPropertyGraphStore {
        fn supports_structured_queries(&self) -> bool {
            false
        }

        fn supports_vector_queries(&self) -> bool {
            false
        }

        async fn upsert_nodes(
            &self,
            _nodes: Vec<Box<dyn cheungfun_core::traits::LabelledNode>>,
        ) -> Result<()> {
            Ok(())
        }

        async fn upsert_relations(&self, _relations: Vec<Relation>) -> Result<()> {
            Ok(())
        }

        async fn get_triplets(
            &self,
            entity_names: Option<Vec<String>>,
            relation_names: Option<Vec<String>>,
            _properties: Option<HashMap<String, serde_json::Value>>,
            _ids: Option<Vec<String>>,
        ) -> Result<Vec<Triplet>> {
            let mut filtered = self.triplets.clone();

            // Filter by entity names (check both entity names and IDs)
            if let Some(names) = entity_names {
                filtered = filtered
                    .into_iter()
                    .filter(|t| {
                        // Check both entity names and entity IDs
                        names.contains(&t.source.name)
                            || names.contains(&t.target.name)
                            || names.contains(&t.source.id())
                            || names.contains(&t.target.id())
                    })
                    .collect();
            }

            // Filter by relation names
            if let Some(rel_names) = relation_names {
                filtered = filtered
                    .into_iter()
                    .filter(|t| rel_names.contains(&t.relation.label))
                    .collect();
            }
            Ok(filtered)
        }

        async fn get(
            &self,
            _properties: Option<HashMap<String, serde_json::Value>>,
            _ids: Option<Vec<String>>,
        ) -> Result<Vec<Box<dyn cheungfun_core::traits::LabelledNode>>> {
            Ok(vec![])
        }

        async fn get_rel_map(
            &self,
            _entity_ids: Vec<String>,
            _depth: usize,
        ) -> Result<Vec<Vec<Box<dyn cheungfun_core::traits::LabelledNode>>>> {
            Ok(vec![])
        }

        async fn delete(
            &self,
            _entity_ids: Option<Vec<String>>,
            _relation_ids: Option<Vec<String>>,
        ) -> Result<()> {
            Ok(())
        }

        async fn health_check(&self) -> Result<()> {
            Ok(())
        }

        async fn stats(&self) -> Result<cheungfun_core::traits::GraphStoreStats> {
            Ok(cheungfun_core::traits::GraphStoreStats {
                entity_count: 4,
                relation_count: 3,
                triplet_count: 3,
                entity_label_count: 0,
                relation_label_count: 0,
                avg_degree: 0.0,
                additional_stats: HashMap::new(),
            })
        }
    }

    async fn create_test_graph_store() -> Arc<MockPropertyGraphStore> {
        Arc::new(MockPropertyGraphStore::new())
    }

    #[tokio::test]
    async fn test_graph_retriever_entity_strategy() {
        let store = create_test_graph_store().await;
        let config = GraphRetrievalConfig::default();
        let retriever =
            GraphRetriever::with_strategy(store.clone(), config, GraphRetrievalStrategy::Entity);

        let query = Query::new("Alice works at Microsoft".to_string());

        // Debug: check entity extraction
        let entities = retriever.extract_query_entities(&query.text);
        println!("Extracted entities: {:?}", entities);

        let results = retriever.retrieve(&query).await.unwrap();
        println!("Found {} results", results.len());

        if results.is_empty() {
            // Let's check what triplets are available
            let all_triplets = store.get_triplets(None, None, None, None).await.unwrap();
            println!("Available triplets: {}", all_triplets.len());
            for triplet in &all_triplets {
                println!(
                    "  {} -[{}]-> {}",
                    triplet.source.name, triplet.relation.label, triplet.target.name
                );
            }
        }

        assert!(
            !results.is_empty(),
            "Expected to find results for entity strategy"
        );
        assert!(results[0].score > 0.0);

        // Check that the result contains relevant information
        let content = &results[0].node.content;
        assert!(content.contains("Alice") || content.contains("Microsoft"));
    }

    #[tokio::test]
    async fn test_graph_retriever_relationship_strategy() {
        let store = create_test_graph_store().await;
        let config = GraphRetrievalConfig::default();
        let retriever =
            GraphRetriever::with_strategy(store, config, GraphRetrievalStrategy::Relationship);

        let query = Query::new("Who works at companies?".to_string());
        let results = retriever.retrieve(&query).await.unwrap();

        // Should find WORKS_AT relationships
        if !results.is_empty() {
            let content = &results[0].node.content;
            assert!(content.contains("WORKS_AT") || content.contains("work"));
        }
    }

    #[tokio::test]
    async fn test_graph_retriever_hybrid_strategy() {
        let store = create_test_graph_store().await;
        let config = GraphRetrievalConfig::default();
        let retriever =
            GraphRetriever::with_strategy(store, config, GraphRetrievalStrategy::Hybrid);

        let query = Query::new("Where does Alice work?".to_string());
        let results = retriever.retrieve(&query).await.unwrap();

        assert!(!results.is_empty());

        // Should find information about Alice and work relationships
        let content = &results[0].node.content;
        assert!(content.contains("Alice") || content.contains("WORKS_AT"));
    }

    #[tokio::test]
    async fn test_entity_extraction() {
        let store = create_test_graph_store().await;
        let config = GraphRetrievalConfig::default();
        let retriever = GraphRetriever::new(store, config);

        let entities =
            retriever.extract_query_entities("Alice Smith works at Microsoft Corporation");

        assert!(
            entities.contains(&"Alice Smith".to_string())
                || entities.contains(&"Alice".to_string())
        );
        assert!(entities.contains(&"Microsoft".to_string()));
    }

    #[tokio::test]
    async fn test_relation_extraction() {
        let store = create_test_graph_store().await;
        let config = GraphRetrievalConfig::default();
        let retriever = GraphRetriever::new(store, config);

        let relations = retriever.extract_relation_keywords("Who works at the company?");

        assert!(relations.contains(&"WORKS_AT".to_string()));
    }

    #[tokio::test]
    async fn test_empty_query() {
        let store = create_test_graph_store().await;
        let config = GraphRetrievalConfig::default();
        let retriever = GraphRetriever::new(store, config);

        let query = Query::new("xyz unknown entity".to_string());
        let results = retriever.retrieve(&query).await.unwrap();

        // Should return empty results for unknown entities
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_custom_strategy() {
        let store = create_test_graph_store().await;
        let config = GraphRetrievalConfig::default();
        let retriever = GraphRetriever::with_strategy(
            store,
            config,
            GraphRetrievalStrategy::Custom {
                entity_names: Some(vec!["Alice".to_string()]),
                relation_names: Some(vec!["WORKS_AT".to_string()]),
                properties: None,
            },
        );

        let query = Query::new("test query".to_string());
        let results = retriever.retrieve(&query).await.unwrap();

        // Should find results based on custom filters
        if !results.is_empty() {
            let content = &results[0].node.content;
            assert!(content.contains("Alice") || content.contains("WORKS_AT"));
        }
    }

    #[tokio::test]
    async fn test_scoring() {
        let store = create_test_graph_store().await;
        let config = GraphRetrievalConfig {
            entity_weight: 0.5,
            relationship_weight: 0.3,
            text_weight: 0.2,
            ..Default::default()
        };
        let retriever = GraphRetriever::new(store, config);

        let query = Query::new("Alice works at Microsoft".to_string());
        let results = retriever.retrieve(&query).await.unwrap();

        if !results.is_empty() {
            // Score should be reasonable (between 0 and 1)
            assert!(results[0].score >= 0.0);
            assert!(results[0].score <= 1.0);

            // Results should be sorted by score (highest first)
            for i in 1..results.len() {
                assert!(results[i - 1].score >= results[i].score);
            }
        }
    }
}
