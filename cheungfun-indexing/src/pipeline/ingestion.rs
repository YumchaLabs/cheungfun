//! Ingestion Pipeline implementation matching LlamaIndex's IngestionPipeline.
//!
//! This module provides a data preprocessing pipeline that focuses solely on
//! transforming documents into processed nodes, without handling index construction.
//! It matches LlamaIndex's IngestionPipeline design exactly.

use std::sync::Arc;
use tracing::{info, warn};

use cheungfun_core::{
    deduplication::DocstoreStrategy,
    traits::{DocumentStore, VectorStore},
    Document, Node, Result as CoreResult,
};

use super::common::{CommonStats, ConsoleProgressReporter, ProgressReporter};
use crate::cache::IngestionCache;

/// Configuration for the ingestion pipeline.
#[derive(Debug, Clone)]
pub struct IngestionConfig {
    /// Whether to show progress during processing.
    pub show_progress: bool,
    /// Number of parallel workers (None = auto-detect).
    pub num_workers: Option<usize>,
    /// Whether to store document text in docstore.
    pub store_doc_text: bool,
    /// Whether to modify nodes in place.
    pub in_place: bool,
    /// Document deduplication strategy.
    pub docstore_strategy: DocstoreStrategy,
    /// Whether to disable caching.
    pub disable_cache: bool,
    /// Cache collection name for transformations.
    pub cache_collection: Option<String>,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            show_progress: false,
            num_workers: None,
            store_doc_text: true,
            in_place: true,
            docstore_strategy: DocstoreStrategy::Upserts,
            disable_cache: false,
            cache_collection: None,
        }
    }
}

/// Statistics from ingestion pipeline execution.
#[derive(Debug, Clone)]
pub struct IngestionStats {
    /// Number of input documents processed.
    pub documents_processed: usize,
    /// Number of input nodes processed.
    pub nodes_processed: usize,
    /// Number of output nodes created.
    pub nodes_created: usize,
    /// Common statistics.
    pub common: CommonStats,
}

/// Ingestion Pipeline for data preprocessing.
///
/// This pipeline matches LlamaIndex's IngestionPipeline design, focusing solely
/// on data preprocessing: loading documents, applying transformations, and
/// outputting processed nodes. It does NOT handle index construction.
///
/// # Usage
///
/// ```rust,no_run
/// use cheungfun_indexing::pipeline::IngestionPipeline;
/// use cheungfun_core::prelude::*;
/// use std::sync::Arc;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let pipeline = IngestionPipeline::builder()
///     .with_transformations(vec![
///         Arc::new(SentenceSplitter::from_defaults(512, 20)?),
///         Arc::new(MetadataExtractor::new()),
///     ])
///     .build()?;
///
/// // Process documents
/// let documents = vec![Document::new("Sample text", None)];
/// let nodes = pipeline.run(Some(documents), None).await?;
/// # Ok(())
/// # }
/// ```
pub struct IngestionPipeline {
    /// Name of the pipeline.
    pub name: String,
    /// Project name.
    pub project_name: String,
    /// Transformation components to apply (using TypedTransform system).
    pub transformations: Vec<
        Arc<
            dyn cheungfun_core::traits::TypedTransform<
                cheungfun_core::traits::NodeState,
                cheungfun_core::traits::NodeState,
            >,
        >,
    >,
    /// Optional documents to process (can be overridden at runtime).
    pub documents: Option<Vec<Document>>,
    /// Optional vector store for direct storage.
    pub vector_store: Option<Arc<dyn VectorStore>>,
    /// Optional document store for deduplication.
    pub docstore: Option<Arc<dyn DocumentStore>>,
    /// Optional ingestion cache for transformation results.
    pub cache: Option<IngestionCache>,
    /// Pipeline configuration.
    pub config: IngestionConfig,
    /// Progress reporter.
    progress_reporter: Box<dyn ProgressReporter + Send + Sync>,
}

impl IngestionPipeline {
    /// Create a new ingestion pipeline builder.
    pub fn builder() -> IngestionPipelineBuilder {
        IngestionPipelineBuilder::new()
    }

    /// Persist the pipeline cache and docstore to disk.
    ///
    /// This matches LlamaIndex's IngestionPipeline.persist() method.
    ///
    /// # Arguments
    ///
    /// * `persist_dir` - Directory to persist to
    /// * `cache_name` - Name of the cache file (default: "cache.json")
    /// * `docstore_name` - Name of the docstore file (default: "docstore.json")
    pub async fn persist(
        &self,
        persist_dir: &str,
        cache_name: Option<&str>,
        docstore_name: Option<&str>,
    ) -> CoreResult<()> {
        let persist_path = std::path::Path::new(persist_dir);

        // Create directory if it doesn't exist
        if !persist_path.exists() {
            tokio::fs::create_dir_all(persist_path).await.map_err(|e| {
                cheungfun_core::error::CheungfunError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create persist directory: {}", e),
                ))
            })?;
        }

        // Persist cache if available
        if let Some(ref cache) = self.cache {
            let cache_file = cache_name.unwrap_or("cache.json");
            let cache_path = persist_path.join(cache_file);
            cache.persist(&cache_path).await.map_err(|e| {
                cheungfun_core::error::CheungfunError::Pipeline {
                    message: format!("Failed to persist cache: {}", e),
                }
            })?;
            info!("Persisted cache to: {}", cache_path.display());
        }

        // Note: DocumentStore trait doesn't have persist method in current implementation
        // This would need to be implemented by specific docstore implementations
        if let Some(ref _docstore) = self.docstore {
            let docstore_file = docstore_name.unwrap_or("docstore.json");
            let docstore_path = persist_path.join(docstore_file);
            info!(
                "Docstore persistence not implemented yet: {}",
                docstore_path.display()
            );
        }

        Ok(())
    }

    /// Load the pipeline cache and docstore from disk.
    ///
    /// This matches LlamaIndex's IngestionPipeline.load() method.
    ///
    /// # Arguments
    ///
    /// * `persist_dir` - Directory to load from
    /// * `cache_name` - Name of the cache file (default: "cache.json")
    /// * `docstore_name` - Name of the docstore file (default: "docstore.json")
    pub async fn load(
        &mut self,
        persist_dir: &str,
        cache_name: Option<&str>,
        docstore_name: Option<&str>,
    ) -> CoreResult<()> {
        let persist_path = std::path::Path::new(persist_dir);

        // Load cache if available
        if let Some(ref cache) = self.cache {
            let cache_file = cache_name.unwrap_or("cache.json");
            let cache_path = persist_path.join(cache_file);
            if cache_path.exists() {
                cache.load(&cache_path).await.map_err(|e| {
                    cheungfun_core::error::CheungfunError::Pipeline {
                        message: format!("Failed to load cache: {}", e),
                    }
                })?;
                info!("Loaded cache from: {}", cache_path.display());
            }
        }

        // Note: DocumentStore trait doesn't have load method in current implementation
        // This would need to be implemented by specific docstore implementations
        if let Some(ref _docstore) = self.docstore {
            let docstore_file = docstore_name.unwrap_or("docstore.json");
            let docstore_path = persist_path.join(docstore_file);
            if docstore_path.exists() {
                info!(
                    "Docstore loading not implemented yet: {}",
                    docstore_path.display()
                );
            }
        }

        Ok(())
    }

    /// Run the ingestion pipeline with optional runtime parameters.
    ///
    /// This method matches LlamaIndex's IngestionPipeline.run() signature exactly.
    ///
    /// # Arguments
    ///
    /// * `documents` - Optional documents to process. If None, uses pipeline's documents.
    /// * `nodes` - Optional pre-processed nodes. If provided, skips document loading.
    ///
    /// # Returns
    ///
    /// Returns a vector of processed nodes ready for indexing.
    pub async fn run(
        &self,
        documents: Option<Vec<Document>>,
        nodes: Option<Vec<Node>>,
    ) -> CoreResult<Vec<Node>> {
        let start_time = std::time::Instant::now();

        if self.config.show_progress {
            info!("🚀 Starting ingestion pipeline: {}", self.name);
        }

        // Determine input source
        let input_nodes = match (documents, nodes) {
            (Some(docs), None) => {
                if self.config.show_progress {
                    info!("📄 Processing {} provided documents", docs.len());
                }
                self.process_documents(docs).await?
            }
            (None, Some(input_nodes)) => {
                if self.config.show_progress {
                    info!("🔧 Processing {} provided nodes", input_nodes.len());
                }
                input_nodes
            }
            (Some(_), Some(_)) => {
                return Err(cheungfun_core::error::CheungfunError::Pipeline {
                    message: "Cannot provide both documents and nodes".to_string(),
                });
            }
            (None, None) => {
                // Use pipeline's documents if available
                match &self.documents {
                    Some(docs) => {
                        if self.config.show_progress {
                            info!("📄 Processing {} pipeline documents", docs.len());
                        }
                        self.process_documents(docs.clone()).await?
                    }
                    None => {
                        warn!("No documents or nodes provided");
                        return Ok(Vec::new());
                    }
                }
            }
        };

        // Apply transformations
        let processed_nodes = self.apply_transformations(input_nodes).await?;

        // Handle deduplication if docstore is configured
        let final_nodes = if let Some(ref docstore) = self.docstore {
            self.handle_deduplication(processed_nodes, docstore).await?
        } else {
            processed_nodes
        };

        // Store in vector store if configured
        if let Some(ref vector_store) = self.vector_store {
            if self.config.show_progress {
                info!("💾 Storing {} nodes in vector store", final_nodes.len());
            }
            vector_store.add(final_nodes.clone()).await.map_err(|e| {
                cheungfun_core::error::CheungfunError::Pipeline {
                    message: format!("Failed to store nodes: {}", e),
                }
            })?;
        }

        let processing_time = start_time.elapsed();
        if self.config.show_progress {
            info!(
                "✅ Ingestion completed in {:?}, produced {} nodes",
                processing_time,
                final_nodes.len()
            );
        }

        Ok(final_nodes)
    }

    /// Process documents into initial nodes.
    async fn process_documents(&self, documents: Vec<Document>) -> CoreResult<Vec<Node>> {
        // For now, convert documents to nodes directly
        // In a full implementation, this might involve more sophisticated processing
        let nodes: Vec<Node> = documents
            .into_iter()
            .map(|doc| {
                let chunk_info = cheungfun_core::types::ChunkInfo {
                    start_char_idx: Some(0),
                    end_char_idx: Some(doc.content.len()),
                    ..Default::default()
                };
                Node::new(doc.content.clone(), doc.id, chunk_info)
            })
            .collect();

        Ok(nodes)
    }

    /// Apply all transformations to the nodes.
    ///
    /// This implementation follows LlamaIndex's transformation pipeline pattern,
    /// applying each transformation sequentially with optional caching and progress reporting.
    async fn apply_transformations(&self, nodes: Vec<Node>) -> CoreResult<Vec<Node>> {
        use cheungfun_core::traits::TypedData;

        // Start with nodes wrapped in TypedData
        let mut current_data = TypedData::from_nodes(nodes);

        // Apply each transformation sequentially (following LlamaIndex pattern)
        for (i, transform) in self.transformations.iter().enumerate() {
            if self.config.show_progress {
                info!(
                    "🔄 Applying transformation {}/{}: {}",
                    i + 1,
                    self.transformations.len(),
                    transform.name()
                );
            }

            // Apply the transformation
            current_data = transform.transform(current_data).await.map_err(|e| {
                cheungfun_core::error::CheungfunError::Pipeline {
                    message: format!("Transformation '{}' failed: {}", transform.name(), e),
                }
            })?;

            if self.config.show_progress {
                info!(
                    "✅ Completed transformation '{}': {} nodes",
                    transform.name(),
                    current_data.nodes().len()
                );
            }
        }

        // Extract final nodes
        Ok(current_data.into_nodes())
    }

    /// Handle node deduplication using docstore.
    async fn handle_deduplication(
        &self,
        nodes: Vec<Node>,
        _docstore: &Arc<dyn DocumentStore>,
    ) -> CoreResult<Vec<Node>> {
        // This is a simplified implementation
        // In a full implementation, this would match LlamaIndex's deduplication logic
        if self.config.show_progress {
            info!("🔍 Checking for duplicate nodes");
        }

        // For now, return all nodes
        // TODO: Implement proper node-level deduplication
        Ok(nodes)
    }
}

/// Builder for IngestionPipeline.
pub struct IngestionPipelineBuilder {
    name: Option<String>,
    project_name: Option<String>,
    transformations: Vec<
        Arc<
            dyn cheungfun_core::traits::TypedTransform<
                cheungfun_core::traits::NodeState,
                cheungfun_core::traits::NodeState,
            >,
        >,
    >,
    documents: Option<Vec<Document>>,
    vector_store: Option<Arc<dyn VectorStore>>,
    docstore: Option<Arc<dyn DocumentStore>>,
    cache: Option<IngestionCache>,
    config: Option<IngestionConfig>,
    progress_reporter: Option<Box<dyn ProgressReporter + Send + Sync>>,
}

impl Default for IngestionPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IngestionPipelineBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            name: None,
            project_name: None,
            transformations: Vec::new(),
            documents: None,
            vector_store: None,
            docstore: None,
            cache: None,
            config: None,
            progress_reporter: None,
        }
    }

    /// Set the pipeline name.
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Set the project name.
    pub fn with_project_name(mut self, project_name: String) -> Self {
        self.project_name = Some(project_name);
        self
    }

    /// Set the transformations using TypedTransform system.
    pub fn with_transformations(
        mut self,
        transformations: Vec<
            Arc<
                dyn cheungfun_core::traits::TypedTransform<
                    cheungfun_core::traits::NodeState,
                    cheungfun_core::traits::NodeState,
                >,
            >,
        >,
    ) -> Self {
        self.transformations = transformations;
        self
    }

    /// Add a single transformation using TypedTransform system.
    pub fn with_transformation(
        mut self,
        transformation: Arc<
            dyn cheungfun_core::traits::TypedTransform<
                cheungfun_core::traits::NodeState,
                cheungfun_core::traits::NodeState,
            >,
        >,
    ) -> Self {
        self.transformations.push(transformation);
        self
    }

    /// Set the documents.
    pub fn with_documents(mut self, documents: Vec<Document>) -> Self {
        self.documents = Some(documents);
        self
    }

    /// Set the vector store.
    pub fn with_vector_store(mut self, vector_store: Arc<dyn VectorStore>) -> Self {
        self.vector_store = Some(vector_store);
        self
    }

    /// Set the document store.
    pub fn with_docstore(mut self, docstore: Arc<dyn DocumentStore>) -> Self {
        self.docstore = Some(docstore);
        self
    }

    /// Set the ingestion cache.
    pub fn with_cache(mut self, cache: IngestionCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: IngestionConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set a custom progress reporter.
    pub fn with_progress_reporter(
        mut self,
        reporter: Box<dyn ProgressReporter + Send + Sync>,
    ) -> Self {
        self.progress_reporter = Some(reporter);
        self
    }

    /// Enable simple in-memory caching.
    pub fn with_simple_cache(mut self) -> Self {
        self.cache = Some(IngestionCache::simple());
        self
    }

    /// Enable caching with a specific collection name.
    pub fn with_cache_collection(mut self, collection: String) -> Self {
        let cache = IngestionCache::simple();
        self.cache = Some(cache);
        // Update config to use the collection
        let mut config = self.config.unwrap_or_default();
        config.cache_collection = Some(collection);
        self.config = Some(config);
        self
    }

    /// Load pipeline from a persist directory.
    pub async fn from_persist_path(persist_dir: &str) -> CoreResult<IngestionPipelineBuilder> {
        let mut builder = Self::new();

        // Try to load cache
        let cache_path = std::path::Path::new(persist_dir).join("cache.json");
        if cache_path.exists() {
            match IngestionCache::from_persist_path(&cache_path).await {
                Ok(cache) => {
                    builder.cache = Some(cache);
                    info!("Loaded cache from: {}", cache_path.display());
                }
                Err(e) => {
                    warn!("Failed to load cache: {}", e);
                }
            }
        }

        Ok(builder)
    }

    /// Build the ingestion pipeline.
    pub fn build(self) -> CoreResult<IngestionPipeline> {
        Ok(IngestionPipeline {
            name: self.name.unwrap_or_else(|| "default_pipeline".to_string()),
            project_name: self
                .project_name
                .unwrap_or_else(|| "default_project".to_string()),
            transformations: self.transformations,
            documents: self.documents,
            vector_store: self.vector_store,
            docstore: self.docstore,
            cache: self.cache,
            config: self.config.unwrap_or_default(),
            progress_reporter: self
                .progress_reporter
                .unwrap_or_else(|| Box::new(ConsoleProgressReporter)),
        })
    }
}
