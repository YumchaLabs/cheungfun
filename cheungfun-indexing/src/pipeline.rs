//! Indexing pipeline implementation.

use async_trait::async_trait;
use cheungfun_core::{
    cache::UnifiedCache,
    traits::{
        DocumentStore, Embedder, IndexStore, IndexingPipeline, Loader, NodeTransformer,
        PipelineCache, StorageContext, Transformer, VectorStore,
    },
    Document, IndexingProgress, IndexingStats, Node, Result as CoreResult,
};
use futures::future::join_all;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::{debug, error, info, warn};

use crate::error::{IndexingError, Result};

/// Configuration for the indexing pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum number of concurrent operations.
    pub max_concurrency: usize,
    /// Batch size for processing documents.
    pub batch_size: usize,
    /// Whether to continue processing if some operations fail.
    pub continue_on_error: bool,
    /// Timeout for individual operations (in seconds).
    pub operation_timeout_seconds: Option<u64>,
    /// Whether to enable progress reporting.
    pub enable_progress_reporting: bool,
    /// Whether to enable caching for embeddings and nodes.
    pub enable_caching: bool,
    /// Default TTL for cache entries (in seconds).
    pub cache_ttl_seconds: u64,
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
        }
    }
}

/// A complete indexing pipeline that processes documents through multiple stages.
///
/// The pipeline consists of the following stages:
/// 1. **Loading**: Load documents from data sources
/// 2. **Transformation**: Convert documents to nodes (chunking, etc.)
/// 3. **Node Transformation**: Enrich nodes with metadata
/// 4. **Embedding**: Generate vector embeddings for nodes
/// 5. **Storage**: Store nodes in vector database
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::pipeline::{DefaultIndexingPipeline, PipelineConfig};
/// use cheungfun_indexing::loaders::FileLoader;
/// use cheungfun_indexing::transformers::TextSplitter;
/// use cheungfun_core::traits::IndexingPipeline;
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let loader = Arc::new(FileLoader::new("document.txt")?);
///     let transformer = Arc::new(TextSplitter::new(1000, 200));
///     
///     let pipeline = DefaultIndexingPipeline::builder()
///         .with_loader(loader)
///         .with_transformer(transformer)
///         .build()?;
///     
///     let stats = pipeline.run().await?;
///     println!("Processed {} documents", stats.documents_processed);
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct DefaultIndexingPipeline {
    /// Document loader.
    loader: Arc<dyn Loader>,
    /// Document transformers (applied in order).
    transformers: Vec<Arc<dyn Transformer>>,
    /// Node transformers (applied in order).
    node_transformers: Vec<Arc<dyn NodeTransformer>>,
    /// Embedder for generating vector representations.
    embedder: Option<Arc<dyn Embedder>>,
    /// Vector store for persisting nodes (legacy, use storage_context instead).
    vector_store: Option<Arc<dyn VectorStore>>,
    /// Storage context for unified storage management.
    storage_context: Option<Arc<StorageContext>>,
    /// Cache for embeddings and processed nodes.
    cache: Option<UnifiedCache>,
    /// Pipeline configuration.
    config: PipelineConfig,
}

impl DefaultIndexingPipeline {
    /// Create a new pipeline builder.
    #[must_use]
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    /// Process documents through all transformation stages.
    async fn process_documents(&self, documents: Vec<Document>) -> Result<Vec<Node>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        info!(
            "Processing {} documents through transformation pipeline",
            documents.len()
        );

        // Stage 1: Apply document transformers
        let mut all_nodes = Vec::new();
        for transformer in &self.transformers {
            debug!("Applying transformer: {}", transformer.name());

            let nodes = transformer
                .transform_batch(documents.clone())
                .await
                .map_err(|e| {
                    IndexingError::pipeline(format!(
                        "Transformer {} failed: {}",
                        transformer.name(),
                        e
                    ))
                })?;

            all_nodes.extend(nodes);
        }

        if all_nodes.is_empty() {
            warn!("No nodes created from {} documents", documents.len());
            return Ok(vec![]);
        }

        info!(
            "Created {} nodes from {} documents",
            all_nodes.len(),
            documents.len()
        );

        // Stage 2: Apply node transformers
        for node_transformer in &self.node_transformers {
            debug!("Applying node transformer: {}", node_transformer.name());

            all_nodes = node_transformer
                .transform_batch(all_nodes)
                .await
                .map_err(|e| {
                    IndexingError::pipeline(format!(
                        "Node transformer {} failed: {}",
                        node_transformer.name(),
                        e
                    ))
                })?;
        }

        Ok(all_nodes)
    }

    /// Generate embeddings for nodes with caching support.
    async fn generate_embeddings(&self, mut nodes: Vec<Node>) -> Result<Vec<Node>> {
        if let Some(embedder) = &self.embedder {
            info!("Generating embeddings for {} nodes", nodes.len());

            if self.config.enable_caching && self.cache.is_some() {
                // Use cache-aware embedding generation
                self.generate_embeddings_with_cache(&mut nodes, embedder)
                    .await?;
            } else {
                // Direct embedding generation without cache
                self.generate_embeddings_direct(&mut nodes, embedder)
                    .await?;
            }

            info!("Generated embeddings for {} nodes", nodes.len());
        } else {
            debug!("No embedder configured, skipping embedding generation");
        }

        Ok(nodes)
    }

    /// Generate embeddings with cache support.
    async fn generate_embeddings_with_cache(
        &self,
        nodes: &mut [Node],
        embedder: &Arc<dyn Embedder>,
    ) -> Result<()> {
        let cache = self.cache.as_ref().unwrap();
        let ttl = Duration::from_secs(self.config.cache_ttl_seconds);

        let mut cache_hits = 0;
        let mut cache_misses = 0;
        let mut texts_to_embed = Vec::new();
        let mut indices_to_embed = Vec::new();
        let mut cached_embeddings = Vec::new();

        // First pass: check cache and collect data
        for (index, node) in nodes.iter().enumerate() {
            let cache_key = cheungfun_core::traits::CacheKeyGenerator::embedding_key(
                &node.content,
                "default", // TODO: Use actual model name
                None,
            );

            match cache.get_embedding(&cache_key).await {
                Ok(Some(cached_embedding)) => {
                    cached_embeddings.push((index, cached_embedding));
                    cache_hits += 1;
                    debug!("Cache hit for node {}: {}", index, &cache_key[..16]);
                }
                Ok(None) => {
                    texts_to_embed.push(node.content.clone());
                    indices_to_embed.push(index);
                    cache_misses += 1;
                    debug!("Cache miss for node {}: {}", index, &cache_key[..16]);
                }
                Err(e) => {
                    warn!("Cache error for node {}: {}", index, e);
                    texts_to_embed.push(node.content.clone());
                    indices_to_embed.push(index);
                    cache_misses += 1;
                }
            }
        }

        // Second pass: set cached embeddings
        for (index, embedding) in cached_embeddings {
            nodes[index].embedding = Some(embedding);
        }

        info!(
            "Cache statistics: {} hits, {} misses ({:.1}% hit rate)",
            cache_hits,
            cache_misses,
            if cache_hits + cache_misses > 0 {
                (f64::from(cache_hits) / f64::from(cache_hits + cache_misses)) * 100.0
            } else {
                0.0
            }
        );

        // Generate embeddings for cache misses
        if !texts_to_embed.is_empty() {
            debug!("Generating {} new embeddings", texts_to_embed.len());

            let text_refs: Vec<&str> = texts_to_embed
                .iter()
                .map(std::string::String::as_str)
                .collect();
            let new_embeddings = embedder.embed_batch(text_refs).await.map_err(|e| {
                IndexingError::pipeline(format!("Embedding generation failed: {e}"))
            })?;

            // Store new embeddings in cache and assign to nodes
            for (node_index, embedding) in indices_to_embed.iter().zip(new_embeddings.iter()) {
                // Assign to node
                nodes[*node_index].embedding = Some(embedding.clone());

                // Cache the embedding
                let cache_key = cheungfun_core::traits::CacheKeyGenerator::embedding_key(
                    &nodes[*node_index].content,
                    "default", // TODO: Use actual model name
                    None,
                );

                if let Err(e) = cache
                    .put_embedding(&cache_key, embedding.clone(), ttl)
                    .await
                {
                    warn!("Failed to cache embedding for node {}: {}", node_index, e);
                }
            }

            debug!("Cached {} new embeddings", new_embeddings.len());
        }

        Ok(())
    }

    /// Generate embeddings directly without cache.
    async fn generate_embeddings_direct(
        &self,
        nodes: &mut [Node],
        embedder: &Arc<dyn Embedder>,
    ) -> Result<()> {
        // Extract text content for embedding
        let texts: Vec<&str> = nodes.iter().map(|node| node.content.as_str()).collect();

        // Generate embeddings in batches
        let embeddings = embedder
            .embed_batch(texts)
            .await
            .map_err(|e| IndexingError::pipeline(format!("Embedding generation failed: {e}")))?;

        // Assign embeddings to nodes
        for (node, embedding) in nodes.iter_mut().zip(embeddings.into_iter()) {
            node.embedding = Some(embedding);
        }

        Ok(())
    }

    /// Store nodes in vector database and document store.
    async fn store_nodes(&self, nodes: Vec<Node>) -> Result<Vec<uuid::Uuid>> {
        if nodes.is_empty() {
            return Ok(vec![]);
        }

        // Prefer storage context over direct vector store
        if let Some(storage_context) = &self.storage_context {
            info!("Storing {} nodes using storage context", nodes.len());

            // Store nodes in vector store
            let node_ids = storage_context
                .vector_store()
                .add(nodes.clone())
                .await
                .map_err(|e| {
                    IndexingError::pipeline(format!("Vector store operation failed: {e}"))
                })?;

            // TODO: Store documents in document store if needed
            // This would require converting nodes back to documents or storing original documents

            info!("Stored {} nodes using storage context", node_ids.len());
            Ok(node_ids)
        } else if let Some(vector_store) = &self.vector_store {
            info!(
                "Storing {} nodes in vector store (legacy mode)",
                nodes.len()
            );

            let node_ids = vector_store.add(nodes).await.map_err(|e| {
                IndexingError::pipeline(format!("Vector store operation failed: {e}"))
            })?;

            info!("Stored {} nodes in vector store", node_ids.len());
            Ok(node_ids)
        } else {
            debug!("No storage configured, skipping storage");
            Ok(vec![])
        }
    }

    /// Process documents in batches with concurrency control.
    async fn process_in_batches(
        &self,
        documents: Vec<Document>,
        progress_callback: Option<&(dyn Fn(IndexingProgress) + Send + Sync)>,
    ) -> Result<IndexingStats> {
        let start_time = Instant::now();
        let total_documents = documents.len();
        let mut total_nodes_created = 0;
        let mut total_nodes_stored = 0;
        let mut errors = Vec::new();

        // Create semaphore for concurrency control
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrency));

        // Process documents in batches
        let batches: Vec<Vec<Document>> = documents
            .chunks(self.config.batch_size)
            .map(<[cheungfun_core::Document]>::to_vec)
            .collect();

        info!(
            "Processing {} documents in {} batches",
            total_documents,
            batches.len()
        );

        let mut batch_futures = Vec::new();

        for (batch_index, batch) in batches.into_iter().enumerate() {
            let semaphore = semaphore.clone();
            let batch_size = batch.len();

            let future = async move {
                let _permit = semaphore.acquire().await.unwrap();

                // Report progress
                if let Some(callback) = progress_callback {
                    callback(IndexingProgress {
                        stage: format!("Processing batch {}", batch_index + 1),
                        processed: batch_index * self.config.batch_size,
                        total: Some(total_documents),
                        current_item: Some(format!(
                            "Batch {} ({} documents)",
                            batch_index + 1,
                            batch_size
                        )),
                        estimated_remaining: None,
                        metadata: std::collections::HashMap::new(),
                    });
                }

                // Process the batch
                let result = self.process_batch(batch).await;
                (batch_index, result)
            };

            batch_futures.push(future);
        }

        // Execute all batches
        let batch_results = join_all(batch_futures).await;

        // Collect results
        for (batch_index, result) in batch_results {
            match result {
                Ok((nodes_created, nodes_stored)) => {
                    total_nodes_created += nodes_created;
                    total_nodes_stored += nodes_stored;
                    debug!(
                        "Batch {} completed: {} nodes created, {} stored",
                        batch_index + 1,
                        nodes_created,
                        nodes_stored
                    );
                }
                Err(e) => {
                    let error_msg = format!("Batch {} failed: {}", batch_index + 1, e);
                    error!("{}", error_msg);
                    errors.push(error_msg);

                    if !self.config.continue_on_error {
                        return Err(e);
                    }
                }
            }
        }

        let processing_time = start_time.elapsed();

        // Final progress report
        if let Some(callback) = progress_callback {
            callback(IndexingProgress {
                stage: "Completed".to_string(),
                processed: total_documents,
                total: Some(total_documents),
                current_item: None,
                estimated_remaining: None,
                metadata: std::collections::HashMap::new(),
            });
        }

        info!(
            "Pipeline completed in {:?}: {} documents processed, {} nodes created, {} nodes stored",
            processing_time, total_documents, total_nodes_created, total_nodes_stored
        );

        Ok(IndexingStats {
            documents_processed: total_documents,
            nodes_created: total_nodes_created,
            nodes_stored: total_nodes_stored,
            processing_time,
            errors,
            additional_stats: std::collections::HashMap::new(),
        })
    }

    /// Process a single batch of documents.
    async fn process_batch(&self, documents: Vec<Document>) -> Result<(usize, usize)> {
        // Transform documents to nodes
        let nodes = self.process_documents(documents).await?;
        let nodes_created = nodes.len();

        // Generate embeddings
        let nodes_with_embeddings = self.generate_embeddings(nodes).await?;

        // Store nodes
        let stored_ids = self.store_nodes(nodes_with_embeddings).await?;
        let nodes_stored = stored_ids.len();

        Ok((nodes_created, nodes_stored))
    }
}

#[async_trait]
impl IndexingPipeline for DefaultIndexingPipeline {
    async fn run(&self) -> CoreResult<IndexingStats> {
        info!("Starting indexing pipeline");

        // Load documents
        let documents = self.loader.load().await?;
        info!("Loaded {} documents", documents.len());

        if documents.is_empty() {
            warn!("No documents to process");
            return Ok(IndexingStats {
                documents_processed: 0,
                nodes_created: 0,
                nodes_stored: 0,
                processing_time: Duration::from_secs(0),
                errors: vec![],
                additional_stats: std::collections::HashMap::new(),
            });
        }

        // Process documents
        let stats = self
            .process_in_batches(documents, None)
            .await
            .map_err(|e| cheungfun_core::error::CheungfunError::Pipeline {
                message: e.to_string(),
            })?;

        Ok(stats)
    }

    async fn run_with_progress(
        &self,
        progress_callback: Box<dyn Fn(IndexingProgress) + Send + Sync>,
    ) -> CoreResult<IndexingStats> {
        info!("Starting indexing pipeline with progress reporting");

        // Load documents
        progress_callback(IndexingProgress {
            stage: "Loading documents".to_string(),
            processed: 0,
            total: None,
            current_item: Some("Loading...".to_string()),
            estimated_remaining: None,
            metadata: std::collections::HashMap::new(),
        });

        let documents = self.loader.load().await?;
        info!("Loaded {} documents", documents.len());

        if documents.is_empty() {
            warn!("No documents to process");
            return Ok(IndexingStats {
                documents_processed: 0,
                nodes_created: 0,
                nodes_stored: 0,
                processing_time: Duration::from_secs(0),
                errors: vec![],
                additional_stats: std::collections::HashMap::new(),
            });
        }

        // Process documents with progress reporting
        let stats = self
            .process_in_batches(documents, Some(progress_callback.as_ref()))
            .await
            .map_err(|e| cheungfun_core::error::CheungfunError::Pipeline {
                message: e.to_string(),
            })?;

        Ok(stats)
    }

    fn validate(&self) -> CoreResult<()> {
        // Check that required components are present
        if self.transformers.is_empty() {
            return Err(cheungfun_core::error::CheungfunError::Configuration {
                message: "At least one transformer is required".to_string(),
            });
        }

        // Validate loader
        // Note: We can't easily validate the loader without calling it,
        // so we'll skip this for now

        Ok(())
    }
}

/// Builder for constructing indexing pipelines.
#[derive(Debug, Default)]
pub struct PipelineBuilder {
    loader: Option<Arc<dyn Loader>>,
    transformers: Vec<Arc<dyn Transformer>>,
    node_transformers: Vec<Arc<dyn NodeTransformer>>,
    embedder: Option<Arc<dyn Embedder>>,
    vector_store: Option<Arc<dyn VectorStore>>,
    storage_context: Option<Arc<StorageContext>>,
    cache: Option<UnifiedCache>,
    config: Option<PipelineConfig>,
}

impl PipelineBuilder {
    /// Create a new pipeline builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the document loader.
    pub fn with_loader(mut self, loader: Arc<dyn Loader>) -> Self {
        self.loader = Some(loader);
        self
    }

    /// Add a document transformer.
    pub fn with_transformer(mut self, transformer: Arc<dyn Transformer>) -> Self {
        self.transformers.push(transformer);
        self
    }

    /// Add a node transformer.
    pub fn with_node_transformer(mut self, transformer: Arc<dyn NodeTransformer>) -> Self {
        self.node_transformers.push(transformer);
        self
    }

    /// Set the embedder.
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Set the vector store.
    pub fn with_vector_store(mut self, store: Arc<dyn VectorStore>) -> Self {
        self.vector_store = Some(store);
        self
    }

    /// Set the storage context (preferred over individual stores).
    pub fn with_storage_context(mut self, storage_context: Arc<StorageContext>) -> Self {
        self.storage_context = Some(storage_context);
        self
    }

    /// Set the cache for embeddings and processed nodes.
    #[must_use]
    pub fn with_cache(mut self, cache: UnifiedCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Set the pipeline configuration.
    #[must_use]
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the indexing pipeline.
    pub fn build(self) -> Result<DefaultIndexingPipeline> {
        let loader = self
            .loader
            .ok_or_else(|| IndexingError::configuration("Loader is required"))?;

        if self.transformers.is_empty() {
            return Err(IndexingError::configuration(
                "At least one transformer is required",
            ));
        }

        let config = self.config.unwrap_or_default();

        Ok(DefaultIndexingPipeline {
            loader,
            transformers: self.transformers,
            node_transformers: self.node_transformers,
            embedder: self.embedder,
            vector_store: self.vector_store,
            storage_context: self.storage_context,
            cache: self.cache,
            config,
        })
    }
}
