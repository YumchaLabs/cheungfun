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
use cheungfun_core::{
    traits::{Embedder, ResponseGenerator, VectorStore},
    Document, Result,
};
use cheungfun_indexing::node_parser::relational::HierarchicalNodeParser;

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

    /// Get retriever for summary level (level 0).
    pub fn get_summary_retriever(&self) -> Arc<VectorRetriever> {
        let vector_store = self
            .level_indices
            .get(&0)
            .or_else(|| self.level_indices.values().next())
            .expect("At least one index level should exist")
            .clone();

        Arc::new(VectorRetriever::new(vector_store, self.embedder.clone()))
    }

    /// Get retriever for leaf level (highest level number).
    pub fn get_leaf_retriever(&self) -> Arc<VectorRetriever> {
        let max_level = self.level_indices.keys().max().copied().unwrap_or(0);
        let vector_store = self
            .level_indices
            .get(&max_level)
            .expect("Max level should exist")
            .clone();

        Arc::new(VectorRetriever::new(vector_store, self.embedder.clone()))
    }

    /// Get storage context.
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

/// Create layered indices from documents.
///
/// **Reference**: hierarchical.py L160-L205 recursive node creation
async fn create_layered_indices_impl(
    _parser: &HierarchicalNodeParser,
    _documents: Vec<Document>,
    embedder: Arc<dyn Embedder>,
    vector_store_factory: &dyn Fn() -> Arc<dyn VectorStore>,
) -> Result<LayeredIndices> {
    // This is a simplified implementation
    // In a full implementation, you would:
    // 1. Use the hierarchical parser to create multi-level nodes
    // 2. Group nodes by level
    // 3. Create vector stores for each level
    // 4. Create a storage context implementation

    // For now, create a basic two-level structure
    let mut level_indices = HashMap::new();

    // Summary level (level 0)
    level_indices.insert(0, vector_store_factory());

    // Detail level (level 1)
    level_indices.insert(1, vector_store_factory());

    // Create a basic storage context (would need proper implementation)
    let storage_context = Arc::new(BasicStorageContext::new());

    Ok(LayeredIndices::new(
        level_indices,
        storage_context,
        embedder,
    ))
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
struct BasicStorageContext;

impl BasicStorageContext {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl StorageContext for BasicStorageContext {
    async fn get_node(&self, _node_id: &str) -> Result<cheungfun_core::Node> {
        // Placeholder implementation
        Err(cheungfun_core::CheungfunError::NotFound {
            resource: "Node not found in basic storage context".to_string(),
        })
    }

    async fn get_nodes(&self, _node_ids: &[String]) -> Result<Vec<cheungfun_core::Node>> {
        // Placeholder implementation
        Ok(vec![])
    }
}
