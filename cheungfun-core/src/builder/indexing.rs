//! Builder for constructing indexing pipelines.
//!
//! This module provides a fluent API for building indexing pipelines
//! with proper validation and error handling.

use std::sync::Arc;

use crate::{
    config::{EmbedderConfig, IndexingPipelineConfig, VectorStoreConfig},
    factory::{EmbedderFactoryRegistry, VectorStoreFactoryRegistry},
    traits::{Embedder, IndexingPipeline, Loader, Transformer, VectorStore},
    Result,
};

/// Builder for creating indexing pipelines.
///
/// This builder provides a fluent API for constructing indexing pipelines
/// with various components. It supports both direct component instances
/// and factory-based creation from configuration.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::builder::IndexingPipelineBuilder;
/// use cheungfun_core::config::{EmbedderConfig, VectorStoreConfig};
/// use std::sync::Arc;
///
/// let builder = IndexingPipelineBuilder::new()
///     .with_embedder_config(EmbedderConfig::candle("test-model", "cpu"))
///     .with_vector_store_config(VectorStoreConfig::memory(768))
///     .with_batch_size(32)
///     .with_chunk_size(1000);
///
/// // let pipeline = builder.build().await?;
/// ```
#[derive(Debug)]
pub struct IndexingPipelineBuilder {
    /// Loader component (optional).
    loader: Option<Arc<dyn Loader>>,

    /// Transformer component (optional).
    transformer: Option<Arc<dyn Transformer>>,

    /// Embedder component or configuration.
    embedder: Option<EmbedderComponent>,

    /// Vector store component or configuration.
    vector_store: Option<VectorStoreComponent>,

    /// Embedder factory registry.
    embedder_factory: Option<Arc<EmbedderFactoryRegistry>>,

    /// Vector store factory registry.
    vector_store_factory: Option<Arc<VectorStoreFactoryRegistry>>,

    /// Pipeline configuration.
    config: IndexingPipelineConfig,
}

/// Embedder component that can be either an instance or configuration.
#[derive(Debug)]
enum EmbedderComponent {
    Instance(Arc<dyn Embedder>),
    Config(EmbedderConfig),
}

/// Vector store component that can be either an instance or configuration.
#[derive(Debug)]
enum VectorStoreComponent {
    Instance(Arc<dyn VectorStore>),
    Config(VectorStoreConfig),
}

impl IndexingPipelineBuilder {
    /// Create a new indexing pipeline builder.
    #[must_use]
    pub fn new() -> Self {
        // Create default configuration
        let default_embedder = EmbedderConfig::default();
        let default_vector_store = VectorStoreConfig::default();
        let config =
            IndexingPipelineConfig::new(default_embedder.clone(), default_vector_store.clone());

        Self {
            loader: None,
            transformer: None,
            embedder: Some(EmbedderComponent::Config(default_embedder)),
            vector_store: Some(VectorStoreComponent::Config(default_vector_store)),
            embedder_factory: None,
            vector_store_factory: None,
            config,
        }
    }

    /// Set the loader component.
    pub fn with_loader(mut self, loader: Arc<dyn Loader>) -> Self {
        self.loader = Some(loader);
        self
    }

    /// Set the transformer component.
    pub fn with_transformer(mut self, transformer: Arc<dyn Transformer>) -> Self {
        self.transformer = Some(transformer);
        self
    }

    /// Set the embedder instance directly.
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(EmbedderComponent::Instance(embedder));
        self
    }

    /// Set the embedder configuration.
    pub fn with_embedder_config(mut self, config: EmbedderConfig) -> Self {
        self.config.embedder = config.clone();
        self.embedder = Some(EmbedderComponent::Config(config));
        self
    }

    /// Set the vector store instance directly.
    pub fn with_vector_store(mut self, vector_store: Arc<dyn VectorStore>) -> Self {
        self.vector_store = Some(VectorStoreComponent::Instance(vector_store));
        self
    }

    /// Set the vector store configuration.
    pub fn with_vector_store_config(mut self, config: VectorStoreConfig) -> Self {
        self.config.vector_store = config.clone();
        self.vector_store = Some(VectorStoreComponent::Config(config));
        self
    }

    /// Set the embedder factory registry.
    pub fn with_embedder_factory(mut self, factory: Arc<EmbedderFactoryRegistry>) -> Self {
        self.embedder_factory = Some(factory);
        self
    }

    /// Set the vector store factory registry.
    pub fn with_vector_store_factory(mut self, factory: Arc<VectorStoreFactoryRegistry>) -> Self {
        self.vector_store_factory = Some(factory);
        self
    }

    /// Set the batch size for processing.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.config = self.config.with_batch_size(batch_size);
        self
    }

    /// Set the chunk size for text splitting.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.config = self.config.with_chunk_size(chunk_size);
        self
    }

    /// Set the chunk overlap.
    pub fn with_chunk_overlap(mut self, chunk_overlap: usize) -> Self {
        self.config = self.config.with_chunk_overlap(chunk_overlap);
        self
    }

    /// Set the concurrency level.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.config = self.config.with_concurrency(concurrency);
        self
    }

    /// Set whether to continue on error.
    pub fn with_continue_on_error(mut self, continue_on_error: bool) -> Self {
        self.config = self.config.with_continue_on_error(continue_on_error);
        self
    }

    /// Add additional configuration parameter.
    pub fn with_config<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.config = self.config.with_config(key, value);
        self
    }

    /// Validate the builder configuration.
    ///
    /// This method checks that all required components are set and
    /// that the configuration is valid.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or if
    /// required components are missing.
    pub fn validate(&self) -> Result<()> {
        // Validate the pipeline configuration
        self.config.validate()?;

        // Check that embedder is set
        if self.embedder.is_none() {
            return Err(crate::CheungfunError::configuration("Embedder is required"));
        }

        // Check that vector store is set
        if self.vector_store.is_none() {
            return Err(crate::CheungfunError::configuration(
                "Vector store is required",
            ));
        }

        // If using configuration-based components, check that factories are available
        if let Some(EmbedderComponent::Config(_)) = &self.embedder {
            if self.embedder_factory.is_none() {
                return Err(crate::CheungfunError::configuration(
                    "Embedder factory is required when using embedder configuration",
                ));
            }
        }

        if let Some(VectorStoreComponent::Config(_)) = &self.vector_store {
            if self.vector_store_factory.is_none() {
                return Err(crate::CheungfunError::configuration(
                    "Vector store factory is required when using vector store configuration",
                ));
            }
        }

        Ok(())
    }

    /// Build the indexing pipeline.
    ///
    /// This method creates the actual pipeline instance with all
    /// configured components.
    ///
    /// # Returns
    ///
    /// An Arc-wrapped indexing pipeline instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline cannot be built due to
    /// invalid configuration or component creation failures.
    pub async fn build(self) -> Result<Arc<dyn IndexingPipeline>> {
        // Validate the configuration first
        self.validate()?;

        // Create embedder instance
        let _embedder = match self.embedder.unwrap() {
            EmbedderComponent::Instance(embedder) => embedder,
            EmbedderComponent::Config(config) => {
                let factory = self.embedder_factory.unwrap();
                factory.create_embedder(&config).await?
            }
        };

        // Create vector store instance
        let _vector_store = match self.vector_store.unwrap() {
            VectorStoreComponent::Instance(vector_store) => vector_store,
            VectorStoreComponent::Config(config) => {
                let factory = self.vector_store_factory.unwrap();
                factory.create_vector_store(&config).await?
            }
        };

        // Create the pipeline implementation
        // This would be implemented in the cheungfun-indexing crate
        // For now, we'll return an error indicating this needs to be implemented
        Err(crate::CheungfunError::internal(
            "IndexingPipeline implementation not yet available. This will be implemented in cheungfun-indexing crate."
        ))
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &IndexingPipelineConfig {
        &self.config
    }

    /// Check if all required components are set.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.embedder.is_some() && self.vector_store.is_some()
    }

    /// Get information about the current builder state.
    #[must_use]
    pub fn info(&self) -> BuilderInfo {
        BuilderInfo {
            has_loader: self.loader.is_some(),
            has_transformer: self.transformer.is_some(),
            has_embedder: self.embedder.is_some(),
            has_vector_store: self.vector_store.is_some(),
            has_embedder_factory: self.embedder_factory.is_some(),
            has_vector_store_factory: self.vector_store_factory.is_some(),
            is_complete: self.is_complete(),
        }
    }
}

impl Default for IndexingPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about the builder state.
#[derive(Debug, Clone)]
pub struct BuilderInfo {
    /// Whether a loader is set.
    pub has_loader: bool,

    /// Whether a transformer is set.
    pub has_transformer: bool,

    /// Whether an embedder is set.
    pub has_embedder: bool,

    /// Whether a vector store is set.
    pub has_vector_store: bool,

    /// Whether an embedder factory is set.
    pub has_embedder_factory: bool,

    /// Whether a vector store factory is set.
    pub has_vector_store_factory: bool,

    /// Whether all required components are set.
    pub is_complete: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = IndexingPipelineBuilder::new();
        let info = builder.info();

        assert!(!info.has_loader);
        assert!(!info.has_transformer);
        assert!(info.has_embedder);
        assert!(info.has_vector_store);
        assert!(!info.has_embedder_factory);
        assert!(!info.has_vector_store_factory);
        assert!(info.is_complete);
    }

    #[test]
    fn test_builder_configuration() {
        let builder = IndexingPipelineBuilder::new()
            .with_batch_size(64)
            .with_chunk_size(2000)
            .with_chunk_overlap(100)
            .with_concurrency(8)
            .with_continue_on_error(false);

        let config = builder.config();
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.chunk_size, 2000);
        assert_eq!(config.chunk_overlap, 100);
        assert_eq!(config.concurrency, 8);
        assert!(!config.continue_on_error);
    }

    #[test]
    fn test_builder_validation() {
        let builder = IndexingPipelineBuilder::new();

        // Should fail validation because no factories are set for config-based components
        assert!(builder.validate().is_err());
    }

    #[test]
    fn test_embedder_config() {
        let config = EmbedderConfig::candle("test-model", "cpu");
        let builder = IndexingPipelineBuilder::new().with_embedder_config(config.clone());

        assert_eq!(builder.config().embedder, config);
    }

    #[test]
    fn test_vector_store_config() {
        let config = VectorStoreConfig::memory(512);
        let builder = IndexingPipelineBuilder::new().with_vector_store_config(config.clone());

        assert_eq!(builder.config().vector_store, config);
    }
}
