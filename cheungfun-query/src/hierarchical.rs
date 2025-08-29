//! Hierarchical retrieval system integration.
//!
//! This module provides a unified interface for creating and using hierarchical
//! retrieval systems that combine multiple levels of indexing with intelligent
//! query routing and auto-merging capabilities.
//!
//! **Reference**: LlamaIndex hierarchical retrieval patterns
//! - File: `llama-index-core/llama_index/core/query_engine/retriever_query_engine.py`
//! - Lines: L62-L80 (factory pattern)

use std::collections::HashMap;
use std::sync::Arc;

use tracing::{info, instrument};

use crate::{
    engine::{
        PerformanceProfile, QueryEngineBuilder, QueryEngineMetadata, QueryType, RouterQueryEngine,
        RouterQueryEngineBuilder,
    },
    retriever::VectorRetriever,
    retrievers::hierarchical::{HierarchicalRetriever, StorageContext},
};
use async_trait::async_trait;
use cheungfun_core::{
    traits::{Embedder, ResponseGenerator, VectorStore},
    types::Node,
    Document, Result,
};
use cheungfun_indexing::{node_parser::relational::HierarchicalNodeParser, NodeParser};

/// Layered indices structure for hierarchical retrieval.
///
/// **Reference**: LlamaIndex hierarchical index organization
#[derive(Debug)]
pub struct LayeredIndices {
    /// Indices by level (0 = summary, 1 = detailed, etc.)
    level_indices: HashMap<usize, Arc<dyn VectorStore>>,
    /// Storage context for accessing nodes
    storage_context: Arc<dyn StorageContext>,
    /// Embedder for query processing
    embedder: Arc<dyn Embedder>,
}

impl LayeredIndices {
    /// Create new layered indices.
    pub fn new(
        level_indices: HashMap<usize, Arc<dyn VectorStore>>,
        storage_context: Arc<dyn StorageContext>,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        Self {
            level_indices,
            storage_context,
            embedder,
        }
    }

    /// Get retriever for leaf nodes (following LlamaIndex pattern).
    ///
    /// **Reference**: base.py L45 - base_retriever = self.base_index.as_retriever(similarity_top_k=6)
    pub fn get_leaf_retriever(&self) -> Arc<VectorRetriever> {
        let vector_store = self
            .level_indices
            .get(&0) // Level 0 contains leaf nodes
            .expect("Leaf node index should exist")
            .clone();

        Arc::new(VectorRetriever::new(vector_store, self.embedder.clone()))
    }

    /// Get retriever for summary level (same as leaf retriever in our implementation).
    /// This maintains API compatibility while following LlamaIndex's single-index approach.
    pub fn get_summary_retriever(&self) -> Arc<VectorRetriever> {
        self.get_leaf_retriever()
    }

    /// Get storage context for accessing all nodes (including parent nodes).
    ///
    /// **Reference**: base.py L43 - storage_context = StorageContext.from_defaults(docstore=docstore)
    pub fn storage_context(&self) -> Arc<dyn StorageContext> {
        self.storage_context.clone()
    }

    /// Get retriever for specific level.
    pub fn get_level_retriever(&self, level: usize) -> Option<Arc<VectorRetriever>> {
        self.level_indices.get(&level).map(|vector_store| {
            Arc::new(VectorRetriever::new(
                vector_store.clone(),
                self.embedder.clone(),
            ))
        })
    }
}

/// Builder for hierarchical retrieval systems.
///
/// This builder creates a complete hierarchical retrieval system with
/// multiple query engines and intelligent routing.
///
/// **Reference**: LlamaIndex RetrieverQueryEngine.from_args pattern
/// - File: `llama-index-core/llama_index/core/query_engine/retriever_query_engine.py`
/// - Lines: L62-L80
pub struct HierarchicalSystemBuilder {
    documents: Option<Vec<Document>>,
    embedder: Option<Arc<dyn Embedder>>,
    vector_store_factory: Option<Box<dyn Fn() -> Arc<dyn VectorStore> + Send + Sync>>,
    generator: Option<Arc<dyn ResponseGenerator>>,
    chunk_sizes: Vec<usize>,
    merge_threshold: f32,
    verbose: bool,
}

impl std::fmt::Debug for HierarchicalSystemBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HierarchicalSystemBuilder")
            .field("documents", &self.documents.as_ref().map(|d| d.len()))
            .field("embedder", &self.embedder.is_some())
            .field("vector_store_factory", &self.vector_store_factory.is_some())
            .field("generator", &self.generator.is_some())
            .field("chunk_sizes", &self.chunk_sizes)
            .field("merge_threshold", &self.merge_threshold)
            .field("verbose", &self.verbose)
            .finish()
    }
}

impl HierarchicalSystemBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            documents: None,
            embedder: None,
            vector_store_factory: None,
            generator: None,
            chunk_sizes: vec![2048, 512, 128], // Reference: hierarchical.py L127
            merge_threshold: 0.5,              // Reference: auto_merging_retriever.py L39
            verbose: false,
        }
    }

    /// Set documents to index.
    pub fn documents(mut self, documents: Vec<Document>) -> Self {
        self.documents = Some(documents);
        self
    }

    /// Set embedder for vector generation.
    pub fn embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Set vector store factory.
    pub fn vector_store_factory<F>(mut self, factory: F) -> Self
    where
        F: Fn() -> Arc<dyn VectorStore> + Send + Sync + 'static,
    {
        self.vector_store_factory = Some(Box::new(factory));
        self
    }

    /// Set response generator.
    pub fn generator(mut self, generator: Arc<dyn ResponseGenerator>) -> Self {
        self.generator = Some(generator);
        self
    }

    /// Set chunk sizes for hierarchical levels.
    ///
    /// **Reference**: hierarchical.py L127 default chunk sizes
    pub fn chunk_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.chunk_sizes = sizes;
        self
    }

    /// Set merge threshold for auto-merging.
    ///
    /// **Reference**: auto_merging_retriever.py L39
    pub fn merge_threshold(mut self, threshold: f32) -> Self {
        self.merge_threshold = threshold;
        self
    }

    /// Enable verbose logging.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build the complete hierarchical retrieval system.
    ///
    /// **Reference**: router_query_engine.py L112-L118 composition pattern
    #[instrument(skip(self))]
    pub async fn build(self) -> Result<RouterQueryEngine> {
        let documents =
            self.documents
                .ok_or_else(|| cheungfun_core::CheungfunError::Configuration {
                    message: "Documents are required".to_string(),
                })?;

        let embedder =
            self.embedder
                .ok_or_else(|| cheungfun_core::CheungfunError::Configuration {
                    message: "Embedder is required".to_string(),
                })?;

        // Extract all required fields first to avoid partial moves
        let vector_store_factory = self.vector_store_factory.ok_or_else(|| {
            cheungfun_core::CheungfunError::Configuration {
                message: "Vector store factory is required".to_string(),
            }
        })?;

        let generator = self.generator.clone().ok_or_else(|| {
            cheungfun_core::CheungfunError::Configuration {
                message: "Generator is required".to_string(),
            }
        })?;

        let chunk_sizes = self.chunk_sizes.clone();
        let merge_threshold = self.merge_threshold;
        let verbose = self.verbose;

        info!(
            "Building hierarchical retrieval system with {} levels",
            chunk_sizes.len()
        );

        // 1. Create hierarchical parser and indices
        let hierarchical_parser = HierarchicalNodeParser::from_defaults(chunk_sizes)?;
        let layered_indices = create_layered_indices_impl(
            &hierarchical_parser,
            documents,
            embedder.clone(),
            &vector_store_factory,
        )
        .await?;

        // 2. Create hierarchical retriever with auto-merging
        let hierarchical_retriever = HierarchicalRetriever::builder()
            .leaf_retriever(layered_indices.get_leaf_retriever())
            .storage_context(layered_indices.storage_context())
            .merge_threshold(merge_threshold)
            .verbose(verbose)
            .build()?;

        // 3. Create query engines for different strategies
        let summary_engine = QueryEngineBuilder::new()
            .retriever(layered_indices.get_summary_retriever())
            .generator(generator.clone())
            .build()?;

        let detail_engine = QueryEngineBuilder::new()
            .retriever(Arc::new(hierarchical_retriever))
            .generator(generator.clone())
            .build()?;

        // 4. Create LLM-based selector
        // Try to extract SiumaiGenerator to get the LLM client for intelligent selection
        let selector: Arc<dyn crate::engine::QuerySelector> = {
            // For now, use rule-based selector as it's more reliable
            // TODO: Implement proper type checking for SiumaiGenerator
            Arc::new(crate::engine::RuleBasedQuerySelector::new().with_verbose(verbose))
        };

        // 5. Build router query engine
        let router_engine = RouterQueryEngineBuilder::new()
            .add_engine_with_metadata(
                summary_engine,
                QueryEngineMetadata {
                    name: "summary".to_string(),
                    description: "High-level overview and summary questions".to_string(),
                    suitable_for: vec![QueryType::Summary],
                    performance_profile: PerformanceProfile::Fast,
                },
            )
            .add_engine_with_metadata(
                detail_engine,
                QueryEngineMetadata {
                    name: "detailed".to_string(),
                    description: "Detailed implementation and specific questions with auto-merging"
                        .to_string(),
                    suitable_for: vec![
                        QueryType::Detailed,
                        QueryType::CodeSpecific,
                        QueryType::Hybrid,
                    ],
                    performance_profile: PerformanceProfile::Thorough,
                },
            )
            .selector(selector)
            .verbose(verbose)
            .build()?;

        info!("Hierarchical retrieval system built successfully");
        Ok(router_engine)
    }
}

/// Create layered indices from documents following LlamaIndex AutoMergingRetriever pattern.
///
/// **Reference**:
/// - auto_merging_retriever.py L34-L48 (pack initialization)
/// - base.py L27-L48 (document processing)
async fn create_layered_indices_impl(
    parser: &HierarchicalNodeParser,
    documents: Vec<Document>,
    embedder: Arc<dyn Embedder>,
    vector_store_factory: &dyn Fn() -> Arc<dyn VectorStore>,
) -> Result<LayeredIndices> {
    info!(
        "Creating layered indices from {} documents following LlamaIndex pattern",
        documents.len()
    );

    // 1. Parse documents into hierarchical nodes (Reference: base.py L35)
    let all_nodes = parser.parse_nodes(&documents, true).await.map_err(|e| {
        cheungfun_core::CheungfunError::Pipeline {
            message: format!("Failed to parse documents hierarchically: {}", e),
        }
    })?;

    info!("Created {} hierarchical nodes", all_nodes.len());

    // 2. Extract leaf nodes only for vector indexing (Reference: base.py L36)
    let leaf_nodes = extract_leaf_nodes(&all_nodes);
    info!(
        "Extracted {} leaf nodes for vector indexing",
        leaf_nodes.len()
    );

    // 3. Create single vector store with only leaf nodes (Reference: base.py L44)
    let vector_store = vector_store_factory();
    let mut leaf_nodes_with_embeddings = leaf_nodes.clone();

    // Embed leaf nodes
    for node in &mut leaf_nodes_with_embeddings {
        let embedding = embedder.embed(&node.content).await.map_err(|e| {
            cheungfun_core::CheungfunError::Embedding {
                message: format!("Failed to embed node content: {}", e),
            }
        })?;
        node.embedding = Some(embedding);
    }

    // Add leaf nodes to vector store
    vector_store
        .add(leaf_nodes_with_embeddings)
        .await
        .map_err(|e| cheungfun_core::CheungfunError::VectorStore {
            message: format!("Failed to add leaf nodes to vector store: {}", e),
        })?;

    info!("Indexed {} leaf nodes to vector store", leaf_nodes.len());

    // 4. Create storage context with ALL nodes (Reference: base.py L40-L43)
    let storage_context = Arc::new(BasicStorageContext::with_nodes(all_nodes));

    // 5. Create layered indices structure (only one level for leaf nodes)
    let mut level_indices = HashMap::new();
    level_indices.insert(0, vector_store); // Level 0 = leaf nodes

    Ok(LayeredIndices::new(
        level_indices,
        storage_context,
        embedder,
    ))
}

/// Extract leaf nodes from hierarchical node structure.
///
/// **Reference**: LlamaIndex get_leaf_nodes() function (hierarchical.py L25-L31)
/// Leaf nodes are nodes that have no children in the hierarchy.
fn extract_leaf_nodes(all_nodes: &[Node]) -> Vec<Node> {
    use cheungfun_core::relationships::NodeRelationship;

    // Following LlamaIndex pattern: nodes without CHILD relationships are leaf nodes
    all_nodes
        .iter()
        .filter(|node| {
            // Check if node has any CHILD relationships
            node.relationships.get(&NodeRelationship::Child).is_none()
        })
        .cloned()
        .collect()
}

impl Default for HierarchicalSystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Basic storage context implementation.
///
/// **Note**: This is a simplified implementation for demonstration.
/// A full implementation would integrate with the actual storage system.
#[derive(Debug)]
struct BasicStorageContext {
    /// All nodes indexed by ID for quick lookup
    nodes: HashMap<String, Node>,
}

impl BasicStorageContext {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    fn with_nodes(nodes: Vec<Node>) -> Self {
        let mut node_map = HashMap::new();
        for node in nodes {
            node_map.insert(node.id.to_string(), node);
        }
        Self { nodes: node_map }
    }
}

#[async_trait]
impl StorageContext for BasicStorageContext {
    async fn get_node(&self, node_id: &str) -> Result<Node> {
        self.nodes
            .get(node_id)
            .cloned()
            .ok_or_else(|| cheungfun_core::CheungfunError::NotFound {
                resource: format!("Node with ID {} not found", node_id),
            })
    }

    async fn get_nodes(&self, node_ids: &[String]) -> Result<Vec<Node>> {
        let mut result = Vec::new();
        for node_id in node_ids {
            if let Some(node) = self.nodes.get(node_id) {
                result.push(node.clone());
            }
        }
        Ok(result)
    }
}
