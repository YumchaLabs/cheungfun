//! Complete indexing pipeline implementation.
//!
//! This module provides a full indexing pipeline that combines data preprocessing
//! with index construction. It's designed for scenarios where you want a complete
//! end-to-end indexing solution.

use async_trait::async_trait;
use cheungfun_core::{
    cache::UnifiedCache,
    deduplication::{DocstoreStrategy, DocumentDeduplicator},
    traits::{
        DocumentStore, Embedder, IndexingPipeline, Loader, StorageContext,
        VectorStore,
    },
    Document, IndexingProgress, IndexingStats, Node, Result as CoreResult,
};
use futures::stream::{self, StreamExt};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::time::timeout;
use tracing::{debug, info, warn};

use crate::error::{IndexingError, Result};

/// Configuration for the indexing pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum number of concurrent operations.
    pub max_concurrency: usize,
    /// Batch size for processing documents.
    pub batch_size: usize,
    /// Whether to continue processing on errors.
    pub continue_on_error: bool,
    /// Timeout for individual operations (in seconds).
    pub operation_timeout_seconds: Option<u64>,
    /// Whether to enable progress reporting.
    pub enable_progress_reporting: bool,
    /// Whether to enable caching.
    pub enable_caching: bool,
    /// Default TTL for cache entries (in seconds).
    pub cache_ttl_seconds: u64,
    /// Document deduplication strategy.
    pub docstore_strategy: DocstoreStrategy,
    /// Whether to enable document deduplication.
    pub enable_deduplication: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 4,
            batch_size: 10,
            continue_on_error: true,
            operation_timeout_seconds: Some(300), // 5 minutes
            enable_progress_reporting: true,
            enable_caching: true,
            cache_ttl_seconds: 3600, // 1 hour
            docstore_strategy: DocstoreStrategy::Upserts,
            enable_deduplication: true,
        }
    }
}

/// A complete indexing pipeline that processes documents through multiple stages.
///
/// The pipeline consists of the following stages:
/// 1. **Loading**: Load documents from various sources
/// 2. **Transformation**: Apply document transformations (splitting, metadata extraction, etc.)
/// 3. **Embedding**: Generate vector embeddings for text content
/// 4. **Storage**: Store processed nodes in vector databases and document stores
///
/// # Example
///
/// ```rust,no_run
/// use cheungfun_indexing::pipeline::indexing::DefaultIndexingPipeline;
/// use cheungfun_core::prelude::*;
/// use std::sync::Arc;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let pipeline = DefaultIndexingPipeline::builder()
///     .with_loader(Arc::new(FileLoader::new("documents.txt")?))
///     .with_transformer(Arc::new(SentenceSplitter::from_defaults(512, 20)?))
///     .with_embedder(Arc::new(FastEmbedEmbedder::try_new_default()?))
///     .build()?;
///
/// let (nodes, stats) = pipeline.run(None, None, false, true, None, true).await?;
/// println!("Processed {} nodes in {:?}", nodes.len(), stats.processing_time);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct DefaultIndexingPipeline {
    /// Document loader.
    loader: Arc<dyn Loader>,
    /// Transformation pipeline using TypedTransform system.
    document_processors: Vec<Arc<dyn cheungfun_core::traits::TypedTransform<cheungfun_core::traits::DocumentState, cheungfun_core::traits::NodeState>>>,
    node_processors: Vec<Arc<dyn cheungfun_core::traits::TypedTransform<cheungfun_core::traits::NodeState, cheungfun_core::traits::NodeState>>>,
    /// Optional embedder for generating vector representations.
    embedder: Option<Arc<dyn Embedder>>,
    /// Optional vector store for persistence.
    vector_store: Option<Arc<dyn VectorStore>>,
    /// Optional storage context for document and index management.
    storage_context: Option<Arc<StorageContext>>,
    /// Cache for embeddings and processed nodes.
    cache: Option<UnifiedCache>,
    /// Document deduplicator for handling duplicates.
    deduplicator: Option<DocumentDeduplicator>,
    /// Pipeline configuration.
    config: PipelineConfig,
}

#[async_trait]
impl IndexingPipeline for DefaultIndexingPipeline {
    async fn run(
        &self,
        documents: Option<Vec<Document>>,
        nodes: Option<Vec<Node>>,
        show_progress: bool,
        store_doc_text: bool,
        num_workers: Option<usize>,
        in_place: bool,
    ) -> CoreResult<(Vec<Node>, IndexingStats)> {
        info!("Starting indexing pipeline");

        // Determine input source: provided documents, provided nodes, or loader
        let input_documents = match (documents, nodes) {
            (Some(docs), None) => {
                info!("Using {} provided documents", docs.len());
                docs
            }
            (None, Some(nodes)) => {
                info!(
                    "Using {} provided nodes, skipping document loading",
                    nodes.len()
                );
                // If nodes are provided, process them directly
                return self
                    .process_nodes_directly(
                        nodes,
                        show_progress,
                        store_doc_text,
                        num_workers,
                        in_place,
                    )
                    .await;
            }
            (Some(_), Some(_)) => {
                return Err(cheungfun_core::error::CheungfunError::Pipeline {
                    message: "Cannot provide both documents and nodes".to_string(),
                });
            }
            (None, None) => {
                // Use configured loader
                let documents = self.loader.load().await?;
                info!(
                    "Loaded {} documents from configured loader",
                    documents.len()
                );
                documents
            }
        };

        if input_documents.is_empty() {
            warn!("No documents to process");
            return Ok((
                Vec::new(),
                IndexingStats {
                    documents_processed: 0,
                    nodes_created: 0,
                    nodes_stored: 0,
                    processing_time: Duration::from_secs(0),
                    errors: vec![],
                    additional_stats: std::collections::HashMap::new(),
                },
            ));
        }

        // Handle document deduplication
        let filtered_documents = self
            .handle_deduplication(input_documents)
            .await
            .map_err(|e| cheungfun_core::error::CheungfunError::Pipeline {
                message: e.to_string(),
            })?;

        info!(
            "After deduplication: {} documents to process",
            filtered_documents.len()
        );

        if filtered_documents.is_empty() {
            info!("No documents to process after deduplication");
            return Ok((
                Vec::new(),
                IndexingStats {
                    documents_processed: 0,
                    nodes_created: 0,
                    nodes_stored: 0,
                    processing_time: Duration::from_secs(0),
                    errors: vec![],
                    additional_stats: std::collections::HashMap::new(),
                },
            ));
        }

        // Process documents with the specified parameters
        let (processed_nodes, stats) = self
            .process_in_batches_with_options(
                filtered_documents,
                show_progress,
                store_doc_text,
                num_workers,
                in_place,
            )
            .await
            .map_err(|e| cheungfun_core::error::CheungfunError::Pipeline {
                message: e.to_string(),
            })?;

        Ok((processed_nodes, stats))
    }

    async fn run_with_progress(
        &self,
        documents: Option<Vec<Document>>,
        nodes: Option<Vec<Node>>,
        store_doc_text: bool,
        num_workers: Option<usize>,
        in_place: bool,
        _progress_callback: Box<dyn Fn(IndexingProgress) + Send + Sync>,
    ) -> CoreResult<(Vec<Node>, IndexingStats)> {
        // For now, just call run with show_progress = true
        // TODO: Implement proper progress callback integration
        self.run(
            documents,
            nodes,
            true,
            store_doc_text,
            num_workers,
            in_place,
        )
        .await
    }

    fn validate(&self) -> CoreResult<()> {
        // Validate that required components are present
        // At least one processor (document or node) is required for meaningful processing
        if self.document_processors.is_empty() && self.node_processors.is_empty() {
            return Err(cheungfun_core::error::CheungfunError::Configuration {
                message: "At least one processor (document or node) is required".to_string(),
            });
        }

        // Validate embedder if vector store is configured
        if self.vector_store.is_some() && self.embedder.is_none() {
            return Err(cheungfun_core::error::CheungfunError::Configuration {
                message: "Embedder is required when vector store is configured".to_string(),
            });
        }

        // Validate that we have a way to convert documents to nodes
        // Either through document processors or by having nodes as input
        if !self.document_processors.is_empty() {
            debug!("✅ Pipeline has {} document processors", self.document_processors.len());
        }

        if !self.node_processors.is_empty() {
            debug!("✅ Pipeline has {} node processors", self.node_processors.len());
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "DefaultIndexingPipeline"
    }

    fn config(&self) -> HashMap<String, serde_json::Value> {
        let mut config = HashMap::new();
        config.insert(
            "max_concurrency".to_string(),
            self.config.max_concurrency.into(),
        );
        config.insert("batch_size".to_string(), self.config.batch_size.into());
        config.insert(
            "continue_on_error".to_string(),
            self.config.continue_on_error.into(),
        );
        config.insert(
            "enable_caching".to_string(),
            self.config.enable_caching.into(),
        );
        config.insert(
            "enable_deduplication".to_string(),
            self.config.enable_deduplication.into(),
        );
        config
    }
}

impl DefaultIndexingPipeline {
    /// Create a new pipeline builder.
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    /// Process nodes directly without document loading.
    async fn process_nodes_directly(
        &self,
        nodes: Vec<Node>,
        _show_progress: bool,
        _store_doc_text: bool,
        _num_workers: Option<usize>,
        _in_place: bool,
    ) -> CoreResult<(Vec<Node>, IndexingStats)> {
        let start_time = std::time::Instant::now();
        let mut stats = IndexingStats {
            documents_processed: 0, // No documents processed, only nodes
            nodes_created: nodes.len(),
            nodes_stored: 0,
            processing_time: Duration::from_secs(0),
            errors: vec![],
            additional_stats: std::collections::HashMap::new(),
        };

        // Apply transformations to nodes
        // TODO: Reimplement using TypedTransform system
        let mut processed_nodes = nodes;

        // Generate embeddings if embedder is configured
        processed_nodes = self.apply_embeddings(processed_nodes, &mut stats).await?;

        // Store nodes if vector store is configured
        if let Some(ref vector_store) = self.vector_store {
            match vector_store.add(processed_nodes.clone()).await {
                Ok(stored_ids) => {
                    stats.nodes_stored = stored_ids.len();
                }
                Err(e) => {
                    if self.config.continue_on_error {
                        stats.errors.push(format!("Storage error: {}", e));
                    } else {
                        return Err(cheungfun_core::error::CheungfunError::Pipeline {
                            message: format!("Storage failed: {}", e),
                        });
                    }
                }
            }
        }

        stats.processing_time = start_time.elapsed();
        Ok((processed_nodes, stats))
    }

    /// Handle document deduplication if enabled.
    async fn handle_deduplication(&self, documents: Vec<Document>) -> Result<Vec<Document>> {
        // If deduplication is disabled or no storage context, return all documents
        if !self.config.enable_deduplication {
            debug!("Document deduplication is disabled");
            return Ok(documents);
        }

        let storage_context = match &self.storage_context {
            Some(ctx) => ctx,
            None => {
                debug!("No storage context available, skipping deduplication");
                return Ok(documents);
            }
        };

        let deduplicator = match &self.deduplicator {
            Some(dedup) => dedup,
            None => {
                debug!("No deduplicator configured, skipping deduplication");
                return Ok(documents);
            }
        };

        // Get existing document hashes
        let existing_hashes = storage_context
            .doc_store
            .get_all_document_hashes()
            .await
            .map_err(|e| {
                IndexingError::processing(format!("Failed to get document hashes: {}", e))
            })?;

        info!("Found {} existing document hashes", existing_hashes.len());

        // Filter documents based on deduplication strategy
        let (to_process, to_skip, to_update) =
            deduplicator.filter_documents(documents, &existing_hashes);

        info!(
            "Deduplication results: {} to process, {} to skip, {} to update",
            to_process.len(),
            to_skip.len(),
            to_update.len()
        );

        // Combine documents that need processing
        let mut final_documents = to_process;
        final_documents.extend(to_update);

        Ok(final_documents)
    }

    /// Process documents in batches with full options support.
    async fn process_in_batches_with_options(
        &self,
        documents: Vec<Document>,
        show_progress: bool,
        _store_doc_text: bool,
        _num_workers: Option<usize>,
        _in_place: bool,
    ) -> Result<(Vec<Node>, IndexingStats)> {
        let progress_callback = if show_progress {
            Some(Box::new(|progress: IndexingProgress| {
                let total_str = progress.total.map_or("?".to_string(), |t| t.to_string());
                info!(
                    "Progress: {}/{} documents processed",
                    progress.processed, total_str
                );
            })
                as Box<dyn Fn(IndexingProgress) + Send + Sync>)
        } else {
            None
        };

        // Use the existing batch processing logic but return nodes too
        let stats = self
            .process_in_batches(documents.clone(), progress_callback)
            .await?;

        // For now, we need to reconstruct the nodes from the processing
        // This is a temporary solution until we refactor the batch processing
        let nodes = self.process_documents(documents).await?;

        Ok((nodes, stats))
    }

    /// Process documents through all transformation stages.
    ///
    /// This implementation follows LlamaIndex's transformation pipeline pattern,
    /// applying document processors first, then node processors.
    async fn process_documents(&self, documents: Vec<Document>) -> Result<Vec<Node>> {
        use cheungfun_core::traits::TypedData;

        // Start with documents wrapped in TypedData
        let current_document_data = TypedData::from_documents(documents);

        // Apply document processors first (Documents -> Nodes)
        // We need to handle the transition from DocumentState to NodeState
        let mut current_node_data = if !self.document_processors.is_empty() {
            // Apply the first document processor (Documents -> Nodes)
            let processor = &self.document_processors[0];
            debug!(
                "🔄 Applying document processor 1/{}: {}",
                self.document_processors.len(),
                processor.name()
            );

            // Document processors convert DocumentState to NodeState
            let node_data = processor.transform(current_document_data).await
                .map_err(|e| cheungfun_core::error::CheungfunError::Pipeline {
                    message: format!("Document processor '{}' failed: {}", processor.name(), e),
                })?;

            debug!(
                "✅ Completed document processor '{}': {} nodes",
                processor.name(),
                node_data.nodes().len()
            );

            // Warn about additional document processors (they can't be applied after conversion to nodes)
            if self.document_processors.len() > 1 {
                debug!("⚠️ Skipping {} additional document processors as we already have nodes",
                       self.document_processors.len() - 1);
            }

            node_data
        } else {
            // If no document processors, convert documents to nodes directly
            let nodes: Vec<Node> = current_document_data.into_documents()
                .into_iter()
                .enumerate()
                .map(|(idx, doc)| {
                    let chunk_info = cheungfun_core::types::ChunkInfo::new(None, None, idx);
                    Node::new(doc.content, doc.id, chunk_info)
                })
                .collect();
            TypedData::from_nodes(nodes)
        };

        // Now apply node processors (Nodes -> Nodes)
        for (i, processor) in self.node_processors.iter().enumerate() {
            debug!(
                "🔄 Applying node processor {}/{}: {}",
                i + 1,
                self.node_processors.len(),
                processor.name()
            );

            current_node_data = processor.transform(current_node_data).await
                .map_err(|e| cheungfun_core::error::CheungfunError::Pipeline {
                    message: format!("Node processor '{}' failed: {}", processor.name(), e),
                })?;

            debug!(
                "✅ Completed node processor '{}': {} nodes",
                processor.name(),
                current_node_data.nodes().len()
            );
        }

        // Extract final nodes
        Ok(current_node_data.into_nodes())
    }

    /// Process documents in batches with concurrency control.
    async fn process_in_batches(
        &self,
        documents: Vec<Document>,
        progress_callback: Option<Box<dyn Fn(IndexingProgress) + Send + Sync>>,
    ) -> Result<IndexingStats> {
        let start_time = Instant::now();
        let total_docs = documents.len();
        let processed_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));
        let mut errors = Vec::new();
        let mut total_nodes_created = 0;
        let mut total_nodes_stored = 0;

        info!(
            "Processing {} documents in batches of {} with concurrency {}",
            total_docs, self.config.batch_size, self.config.max_concurrency
        );

        // Split documents into batches
        let batches: Vec<Vec<Document>> = documents
            .chunks(self.config.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Process batches with controlled concurrency
        let batch_results = stream::iter(batches)
            .map(|batch| {
                let processed_count = Arc::clone(&processed_count);
                let error_count = Arc::clone(&error_count);
                let progress_callback = progress_callback.as_ref();

                async move {
                    let batch_result =
                        if let Some(timeout_secs) = self.config.operation_timeout_seconds {
                            timeout(
                                Duration::from_secs(timeout_secs),
                                self.process_batch(batch.clone()),
                            )
                            .await
                            .map_err(|_| {
                                IndexingError::timeout("Batch processing timed out".to_string())
                            })?
                        } else {
                            self.process_batch(batch.clone()).await
                        };

                    match batch_result {
                        Ok((nodes_created, nodes_stored)) => {
                            let current_processed = processed_count
                                .fetch_add(batch.len(), Ordering::SeqCst)
                                + batch.len();

                            if let Some(callback) = progress_callback {
                                callback(IndexingProgress {
                                    stage: "Processing documents".to_string(),
                                    processed: current_processed,
                                    total: Some(total_docs),
                                    current_item: None,
                                    estimated_remaining: None,
                                    metadata: HashMap::new(),
                                });
                            }

                            Ok((nodes_created, nodes_stored))
                        }
                        Err(e) => {
                            error_count.fetch_add(1, Ordering::SeqCst);
                            if self.config.continue_on_error {
                                warn!("Batch processing error (continuing): {}", e);
                                Ok((0, 0)) // Return zero counts for failed batch
                            } else {
                                Err(e)
                            }
                        }
                    }
                }
            })
            .buffer_unordered(self.config.max_concurrency)
            .collect::<Vec<_>>()
            .await;

        // Collect results and errors
        for result in batch_results {
            match result {
                Ok((nodes_created, nodes_stored)) => {
                    total_nodes_created += nodes_created;
                    total_nodes_stored += nodes_stored;
                }
                Err(e) => {
                    errors.push(e.to_string());
                    if !self.config.continue_on_error {
                        return Err(e);
                    }
                }
            }
        }

        let processing_time = start_time.elapsed();
        let final_processed = processed_count.load(Ordering::SeqCst);
        let final_errors = error_count.load(Ordering::SeqCst);

        info!(
            "Batch processing completed: {}/{} documents processed, {} errors, {} nodes created, {} nodes stored in {:?}",
            final_processed, total_docs, final_errors, total_nodes_created, total_nodes_stored, processing_time
        );

        Ok(IndexingStats {
            documents_processed: final_processed,
            nodes_created: total_nodes_created,
            nodes_stored: total_nodes_stored,
            processing_time,
            errors,
            additional_stats: HashMap::new(),
        })
    }

    /// Process a single batch of documents.
    async fn process_batch(&self, documents: Vec<Document>) -> Result<(usize, usize)> {
        debug!("Processing batch of {} documents", documents.len());

        // Process documents through transformations
        let nodes = self.process_documents(documents).await?;
        let nodes_created = nodes.len();

        // Store nodes if vector store is configured
        let nodes_stored = if let Some(ref vector_store) = self.vector_store {
            let stored_ids = vector_store
                .add(nodes.clone())
                .await
                .map_err(|e| IndexingError::storage(format!("Failed to store nodes: {}", e)))?;
            stored_ids.len()
        } else {
            0
        };

        // Store in document store if storage context is configured
        if let Some(ref storage_context) = self.storage_context {
            // Convert nodes back to documents for document store
            // This is a simplified approach - in practice, you might want to store the original documents
            let docs_for_storage: Vec<Document> = nodes
                .iter()
                .map(|node| {
                    let mut doc = Document::new(&node.content);
                    doc.id = node.source_document_id;
                    doc
                })
                .collect();

            if !docs_for_storage.is_empty() {
                storage_context
                    .doc_store
                    .add_documents(docs_for_storage)
                    .await
                    .map_err(|e| {
                        IndexingError::storage(format!("Failed to store documents: {}", e))
                    })?;
            }
        }

        debug!(
            "Batch processed: {} nodes created, {} nodes stored",
            nodes_created, nodes_stored
        );
        Ok((nodes_created, nodes_stored))
    }
}

/// Builder for creating DefaultIndexingPipeline instances.
#[derive(Debug, Default)]
pub struct PipelineBuilder {
    loader: Option<Arc<dyn Loader>>,
    document_processors: Vec<Arc<dyn cheungfun_core::traits::TypedTransform<cheungfun_core::traits::DocumentState, cheungfun_core::traits::NodeState>>>,
    node_processors: Vec<Arc<dyn cheungfun_core::traits::TypedTransform<cheungfun_core::traits::NodeState, cheungfun_core::traits::NodeState>>>,
    embedder: Option<Arc<dyn Embedder>>,
    vector_store: Option<Arc<dyn VectorStore>>,
    storage_context: Option<Arc<StorageContext>>,
    cache: Option<UnifiedCache>,
    deduplicator: Option<DocumentDeduplicator>,
    config: Option<PipelineConfig>,
}

impl PipelineBuilder {
    /// Create a new pipeline builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the document loader.
    pub fn with_loader(mut self, loader: Arc<dyn Loader>) -> Self {
        self.loader = Some(loader);
        self
    }

    /// Add a document processor (Documents -> Nodes) to the pipeline.
    pub fn with_document_processor(
        mut self,
        processor: Arc<dyn cheungfun_core::traits::TypedTransform<cheungfun_core::traits::DocumentState, cheungfun_core::traits::NodeState>>
    ) -> Self {
        self.document_processors.push(processor);
        self
    }

    /// Add a node processor (Nodes -> Nodes) to the pipeline.
    pub fn with_node_processor(
        mut self,
        processor: Arc<dyn cheungfun_core::traits::TypedTransform<cheungfun_core::traits::NodeState, cheungfun_core::traits::NodeState>>
    ) -> Self {
        self.node_processors.push(processor);
        self
    }

    /// Legacy method for backward compatibility - for document processors.
    /// This method is deprecated. Use `with_document_processor` instead.
    pub fn with_transformer<T>(mut self, transformer: Arc<T>) -> Self
    where
        T: cheungfun_core::traits::TypedTransform<cheungfun_core::traits::DocumentState, cheungfun_core::traits::NodeState> + 'static
    {
        self.document_processors.push(transformer);
        self
    }

    /// Legacy method for backward compatibility - for node processors.
    /// This method is deprecated. Use `with_node_processor` instead.
    pub fn with_node_transformer<T>(mut self, transformer: Arc<T>) -> Self
    where
        T: cheungfun_core::traits::TypedTransform<cheungfun_core::traits::NodeState, cheungfun_core::traits::NodeState> + 'static
    {
        self.node_processors.push(transformer);
        self
    }

    /// Set the embedder.
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Set the vector store.
    pub fn with_vector_store(mut self, vector_store: Arc<dyn VectorStore>) -> Self {
        self.vector_store = Some(vector_store);
        self
    }

    /// Set the storage context.
    pub fn with_storage_context(mut self, storage_context: Arc<StorageContext>) -> Self {
        self.storage_context = Some(storage_context);
        self
    }

    /// Set the cache for embeddings and processed nodes.
    pub fn with_cache(mut self, cache: UnifiedCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Set the document deduplicator.
    pub fn with_deduplicator(mut self, deduplicator: DocumentDeduplicator) -> Self {
        self.deduplicator = Some(deduplicator);
        self
    }

    /// Enable document deduplication with default settings.
    pub fn with_deduplication(mut self) -> Self {
        self.deduplicator = Some(DocumentDeduplicator::new());
        self
    }

    /// Enable document deduplication with custom strategy.
    pub fn with_deduplication_strategy(mut self, strategy: DocstoreStrategy) -> Self {
        self.deduplicator = Some(DocumentDeduplicator::with_strategy(strategy));
        self
    }

    /// Set the pipeline configuration.
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the pipeline.
    pub fn build(self) -> CoreResult<DefaultIndexingPipeline> {
        let loader =
            self.loader
                .ok_or_else(|| cheungfun_core::error::CheungfunError::Configuration {
                    message: "Loader is required".to_string(),
                })?;

        // Validate that at least one processor is configured
        if self.document_processors.is_empty() && self.node_processors.is_empty() {
            return Err(cheungfun_core::error::CheungfunError::Configuration {
                message: "At least one processor (document or node) is required".to_string(),
            });
        }

        let config = self.config.unwrap_or_default();

        // Create default deduplicator if deduplication is enabled but no deduplicator provided
        let deduplicator = if config.enable_deduplication && self.deduplicator.is_none() {
            Some(DocumentDeduplicator::with_strategy(
                config.docstore_strategy,
            ))
        } else {
            self.deduplicator
        };

        Ok(DefaultIndexingPipeline {
            loader,
            document_processors: self.document_processors,
            node_processors: self.node_processors,
            embedder: self.embedder,
            vector_store: self.vector_store,
            storage_context: self.storage_context,
            cache: self.cache,
            deduplicator,
            config,
        })
    }
}

impl DefaultIndexingPipeline {
    /// Apply embeddings to processed nodes.
    async fn apply_embeddings(
        &self,
        mut processed_nodes: Vec<Node>,
        stats: &mut IndexingStats,
    ) -> CoreResult<Vec<Node>> {
        if let Some(ref embedder) = self.embedder {
            // Extract texts for embedding
            let texts: Vec<String> = processed_nodes.iter().map(|n| n.content.clone()).collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

            // Generate embeddings
            match embedder.embed_batch(text_refs).await {
                Ok(embeddings) => {
                    // Apply embeddings to nodes
                    for (node, embedding) in processed_nodes.iter_mut().zip(embeddings.iter()) {
                        node.embedding = Some(embedding.clone());
                    }
                }
                Err(e) => {
                    if self.config.continue_on_error {
                        stats.errors.push(format!("Embedding error: {}", e));
                    } else {
                        return Err(cheungfun_core::error::CheungfunError::Pipeline {
                            message: format!("Embedding failed: {}", e),
                        });
                    }
                }
            }
        }
        Ok(processed_nodes)
    }
}
