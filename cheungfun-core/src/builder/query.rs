//! Builder for constructing query pipelines.
//!
//! This module provides a fluent API for building query pipelines
//! with proper validation and error handling.

use std::sync::Arc;

use crate::{
    config::{EmbedderConfig, LlmConfig, QueryPipelineConfig, VectorStoreConfig},
    factory::{EmbedderFactoryRegistry, LlmFactoryRegistry, VectorStoreFactoryRegistry},
    traits::{Embedder, QueryPipeline, ResponseGenerator, Retriever, VectorStore},
    Result,
};

/// Builder for creating query pipelines.
///
/// This builder provides a fluent API for constructing query pipelines
/// with various components. It supports both direct component instances
/// and factory-based creation from configuration.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::builder::QueryPipelineBuilder;
/// use cheungfun_core::config::{EmbedderConfig, VectorStoreConfig, LlmConfig};
/// use std::sync::Arc;
///
/// let builder = QueryPipelineBuilder::new()
///     .with_embedder_config(EmbedderConfig::candle("test-model", "cpu"))
///     .with_vector_store_config(VectorStoreConfig::memory(768))
///     .with_llm_config(LlmConfig::openai("gpt-3.5-turbo", "your-api-key"))
///     .with_top_k(10)
///     .with_temperature(0.7);
///
/// // let pipeline = builder.build().await?;
/// ```
#[derive(Debug)]
pub struct QueryPipelineBuilder {
    /// Embedder component or configuration.
    embedder: Option<EmbedderComponent>,

    /// Vector store component or configuration.
    vector_store: Option<VectorStoreComponent>,

    /// Retriever component (optional, will use default if not set).
    retriever: Option<Arc<dyn Retriever>>,

    /// Response generator component or configuration.
    response_generator: Option<ResponseGeneratorComponent>,

    /// Embedder factory registry.
    embedder_factory: Option<Arc<EmbedderFactoryRegistry>>,

    /// Vector store factory registry.
    vector_store_factory: Option<Arc<VectorStoreFactoryRegistry>>,

    /// LLM factory registry.
    llm_factory: Option<Arc<LlmFactoryRegistry>>,

    /// Pipeline configuration.
    config: QueryPipelineConfig,
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

/// Response generator component that can be either an instance or configuration.
#[derive(Debug)]
enum ResponseGeneratorComponent {
    Instance(Arc<dyn ResponseGenerator>),
    Config(Box<LlmConfig>),
}

impl QueryPipelineBuilder {
    /// Create a new query pipeline builder.
    #[must_use]
    pub fn new() -> Self {
        // Create default configuration
        let default_embedder = EmbedderConfig::default();
        let default_vector_store = VectorStoreConfig::default();
        let default_llm = LlmConfig::default();
        let config = QueryPipelineConfig::new(
            default_embedder.clone(),
            default_vector_store.clone(),
            default_llm.clone(),
        );

        Self {
            embedder: Some(EmbedderComponent::Config(default_embedder)),
            vector_store: Some(VectorStoreComponent::Config(default_vector_store)),
            retriever: None,
            response_generator: Some(ResponseGeneratorComponent::Config(Box::new(default_llm))),
            embedder_factory: None,
            vector_store_factory: None,
            llm_factory: None,
            config,
        }
    }

    /// Set the embedder instance directly.
    #[must_use]
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(EmbedderComponent::Instance(embedder));
        self
    }

    /// Set the embedder configuration.
    #[must_use]
    pub fn with_embedder_config(mut self, config: EmbedderConfig) -> Self {
        self.config.embedder = config.clone();
        self.embedder = Some(EmbedderComponent::Config(config));
        self
    }

    /// Set the vector store instance directly.
    #[must_use]
    pub fn with_vector_store(mut self, vector_store: Arc<dyn VectorStore>) -> Self {
        self.vector_store = Some(VectorStoreComponent::Instance(vector_store));
        self
    }

    /// Set the vector store configuration.
    #[must_use]
    pub fn with_vector_store_config(mut self, config: VectorStoreConfig) -> Self {
        self.config.vector_store = config.clone();
        self.vector_store = Some(VectorStoreComponent::Config(config));
        self
    }

    /// Set the retriever component.
    #[must_use]
    pub fn with_retriever(mut self, retriever: Arc<dyn Retriever>) -> Self {
        self.retriever = Some(retriever);
        self
    }

    /// Set the response generator instance directly.
    pub fn with_response_generator(mut self, generator: Arc<dyn ResponseGenerator>) -> Self {
        self.response_generator = Some(ResponseGeneratorComponent::Instance(generator));
        self
    }

    /// Set the LLM configuration.
    #[must_use]
    pub fn with_llm_config(mut self, config: LlmConfig) -> Self {
        self.config.llm = config.clone();
        self.response_generator = Some(ResponseGeneratorComponent::Config(Box::new(config)));
        self
    }

    /// Set the embedder factory registry.
    #[must_use]
    pub fn with_embedder_factory(mut self, factory: Arc<EmbedderFactoryRegistry>) -> Self {
        self.embedder_factory = Some(factory);
        self
    }

    /// Set the vector store factory registry.
    #[must_use]
    pub fn with_vector_store_factory(mut self, factory: Arc<VectorStoreFactoryRegistry>) -> Self {
        self.vector_store_factory = Some(factory);
        self
    }

    /// Set the LLM factory registry.
    #[must_use]
    pub fn with_llm_factory(mut self, factory: Arc<LlmFactoryRegistry>) -> Self {
        self.llm_factory = Some(factory);
        self
    }

    /// Set the default top-k value.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.config = self.config.with_top_k(top_k);
        self
    }

    /// Set the similarity threshold.
    #[must_use]
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.config = self.config.with_similarity_threshold(threshold);
        self
    }

    /// Set the maximum tokens for generation.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.config = self.config.with_max_tokens(max_tokens);
        self
    }

    /// Set the temperature for generation.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config = self.config.with_temperature(temperature);
        self
    }

    /// Set the system prompt template.
    pub fn with_system_prompt<S: Into<String>>(mut self, system_prompt: S) -> Self {
        self.config = self.config.with_system_prompt(system_prompt);
        self
    }

    /// Set whether to include citations.
    #[must_use]
    pub fn with_citations(mut self, include_citations: bool) -> Self {
        self.config = self.config.with_citations(include_citations);
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

        // Check that response generator is set
        if self.response_generator.is_none() {
            return Err(crate::CheungfunError::configuration(
                "Response generator is required",
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

        if let Some(ResponseGeneratorComponent::Config(_)) = &self.response_generator {
            if self.llm_factory.is_none() {
                return Err(crate::CheungfunError::configuration(
                    "LLM factory is required when using LLM configuration",
                ));
            }
        }

        Ok(())
    }

    /// Build the query pipeline.
    ///
    /// This method creates the actual pipeline instance with all
    /// configured components.
    ///
    /// # Returns
    ///
    /// An Arc-wrapped query pipeline instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline cannot be built due to
    /// invalid configuration or component creation failures.
    pub async fn build(self) -> Result<Arc<dyn QueryPipeline>> {
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

        // Create response generator instance
        let _response_generator = match self.response_generator.unwrap() {
            ResponseGeneratorComponent::Instance(generator) => generator,
            ResponseGeneratorComponent::Config(config) => {
                let factory = self.llm_factory.unwrap();
                factory.create_llm(&config).await?
            }
        };

        // Create the pipeline implementation
        // This would be implemented in the cheungfun-query crate
        // For now, we'll return an error indicating this needs to be implemented
        Err(crate::CheungfunError::internal(
            "QueryPipeline implementation not yet available. This will be implemented in cheungfun-query crate.",
        ))
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &QueryPipelineConfig {
        &self.config
    }

    /// Check if all required components are set.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.embedder.is_some() && self.vector_store.is_some() && self.response_generator.is_some()
    }

    /// Get information about the current builder state.
    #[must_use]
    pub fn info(&self) -> QueryBuilderInfo {
        QueryBuilderInfo {
            has_embedder: self.embedder.is_some(),
            has_vector_store: self.vector_store.is_some(),
            has_retriever: self.retriever.is_some(),
            has_response_generator: self.response_generator.is_some(),
            has_embedder_factory: self.embedder_factory.is_some(),
            has_vector_store_factory: self.vector_store_factory.is_some(),
            has_llm_factory: self.llm_factory.is_some(),
            is_complete: self.is_complete(),
        }
    }
}

impl Default for QueryPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about the query builder state.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct QueryBuilderInfo {
    /// Whether an embedder is set.
    pub has_embedder: bool,

    /// Whether a vector store is set.
    pub has_vector_store: bool,

    /// Whether a retriever is set.
    pub has_retriever: bool,

    /// Whether a response generator is set.
    pub has_response_generator: bool,

    /// Whether an embedder factory is set.
    pub has_embedder_factory: bool,

    /// Whether a vector store factory is set.
    pub has_vector_store_factory: bool,

    /// Whether an LLM factory is set.
    pub has_llm_factory: bool,

    /// Whether all required components are set.
    pub is_complete: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = QueryPipelineBuilder::new();
        let info = builder.info();

        assert!(info.has_embedder);
        assert!(info.has_vector_store);
        assert!(!info.has_retriever);
        assert!(info.has_response_generator);
        assert!(!info.has_embedder_factory);
        assert!(!info.has_vector_store_factory);
        assert!(!info.has_llm_factory);
        assert!(info.is_complete);
    }

    #[test]
    fn test_builder_configuration() {
        let builder = QueryPipelineBuilder::new()
            .with_top_k(15)
            .with_similarity_threshold(0.8)
            .with_max_tokens(1500)
            .with_temperature(0.9)
            .with_citations(true);

        let config = builder.config();
        assert_eq!(config.top_k, 15);
        assert_eq!(config.similarity_threshold, Some(0.8));
        assert_eq!(config.effective_max_tokens(), 1500);
        assert_eq!(config.effective_temperature(), 0.9);
        assert!(config.include_citations);
    }

    #[test]
    fn test_builder_validation() {
        let builder = QueryPipelineBuilder::new();

        // Should fail validation because no factories are set for config-based components
        assert!(builder.validate().is_err());
    }

    #[test]
    fn test_embedder_config() {
        let config = EmbedderConfig::candle("test-model", "cpu");
        let builder = QueryPipelineBuilder::new().with_embedder_config(config.clone());

        assert_eq!(builder.config().embedder, config);
    }

    #[test]
    fn test_llm_config() {
        let config = LlmConfig::openai("gpt-4", "test-key");
        let builder = QueryPipelineBuilder::new().with_llm_config(config.clone());

        assert_eq!(builder.config().llm, config);
    }
}
