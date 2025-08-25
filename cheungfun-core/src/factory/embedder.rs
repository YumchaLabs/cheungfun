//! Factory for creating embedders from configuration.
//!
//! This module provides factory traits and implementations for creating
//! embedder instances from configuration objects.

use async_trait::async_trait;
use std::sync::Arc;

use crate::{config::EmbedderConfig, traits::Embedder, Result};

/// Factory for creating embedders from configuration.
///
/// This trait provides a unified interface for creating different types
/// of embedders based on configuration. Implementations handle the
/// specifics of each embedder type.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::factory::EmbedderFactory;
/// use cheungfun_core::config::EmbedderConfig;
/// use cheungfun_core::traits::Embedder;
/// use cheungfun_core::Result;
/// use async_trait::async_trait;
/// use std::sync::Arc;
///
/// struct MyEmbedderFactory;
///
/// #[async_trait]
/// impl EmbedderFactory for MyEmbedderFactory {
///     async fn create_embedder(&self, config: &EmbedderConfig) -> Result<Arc<dyn Embedder>> {
///         // Implementation would create embedder based on config
///         todo!("Implement embedder creation")
///     }
/// }
/// ```
#[async_trait]
pub trait EmbedderFactory: Send + Sync + std::fmt::Debug {
    /// Create an embedder from configuration.
    ///
    /// This method takes an embedder configuration and returns a concrete
    /// embedder implementation wrapped in an Arc for shared ownership.
    ///
    /// # Arguments
    ///
    /// * `config` - The embedder configuration
    ///
    /// # Returns
    ///
    /// An Arc-wrapped embedder instance that implements the Embedder trait.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedder cannot be created due to invalid
    /// configuration, missing dependencies, or initialization failures.
    async fn create_embedder(&self, config: &EmbedderConfig) -> Result<Arc<dyn Embedder>>;

    /// Check if this factory can create an embedder for the given configuration.
    ///
    /// This method can be used to validate configuration before attempting
    /// to create an embedder.
    ///
    /// # Arguments
    ///
    /// * `config` - The embedder configuration to check
    ///
    /// # Returns
    ///
    /// `true` if this factory can create an embedder for the configuration,
    /// `false` otherwise.
    fn can_create(&self, config: &EmbedderConfig) -> bool;

    /// Get a human-readable name for this factory.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Get the supported embedder types.
    ///
    /// Returns a list of embedder type identifiers that this factory
    /// can create. This is useful for discovery and validation.
    fn supported_types(&self) -> Vec<&'static str>;

    /// Validate the configuration without creating the embedder.
    ///
    /// This method performs validation checks on the configuration
    /// to ensure it's valid for creating an embedder.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to validate
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    async fn validate_config(&self, config: &EmbedderConfig) -> Result<()> {
        // Default implementation uses the config's validate method
        config.validate()
    }

    /// Get metadata about this factory.
    ///
    /// Returns information about the factory's capabilities,
    /// supported features, etc.
    fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("name".to_string(), self.name().into());
        metadata.insert("supported_types".to_string(), self.supported_types().into());
        metadata
    }
}

/// Registry for embedder factories.
///
/// This struct manages multiple embedder factories and provides
/// a unified interface for creating embedders from any supported
/// configuration.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::factory::{EmbedderFactoryRegistry, EmbedderFactory};
/// use cheungfun_core::config::EmbedderConfig;
/// use std::sync::Arc;
///
/// let mut registry = EmbedderFactoryRegistry::new();
/// // registry.register("candle", Arc::new(CandleEmbedderFactory::new()));
/// // registry.register("api", Arc::new(ApiEmbedderFactory::new()));
///
/// let config = EmbedderConfig::candle("test-model", "cpu");
/// // let embedder = registry.create_embedder(&config).await?;
/// ```
#[derive(Debug, Default)]
pub struct EmbedderFactoryRegistry {
    factories: std::collections::HashMap<String, Arc<dyn EmbedderFactory>>,
}

impl EmbedderFactoryRegistry {
    /// Create a new embedder factory registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            factories: std::collections::HashMap::new(),
        }
    }

    /// Register a factory for a specific embedder type.
    ///
    /// # Arguments
    ///
    /// * `embedder_type` - The type identifier for the embedder
    /// * `factory` - The factory implementation
    pub fn register<S: Into<String>>(
        &mut self,
        embedder_type: S,
        factory: Arc<dyn EmbedderFactory>,
    ) {
        self.factories.insert(embedder_type.into(), factory);
    }

    /// Create an embedder from configuration.
    ///
    /// This method finds the appropriate factory for the configuration
    /// and uses it to create the embedder.
    ///
    /// # Arguments
    ///
    /// * `config` - The embedder configuration
    ///
    /// # Returns
    ///
    /// An Arc-wrapped embedder instance.
    ///
    /// # Errors
    ///
    /// Returns an error if no suitable factory is found or if the
    /// factory fails to create the embedder.
    pub async fn create_embedder(&self, config: &EmbedderConfig) -> Result<Arc<dyn Embedder>> {
        let embedder_type = config.provider();

        let factory = self.factories.get(embedder_type).ok_or_else(|| {
            crate::CheungfunError::configuration(format!(
                "No factory registered for embedder type: {embedder_type}"
            ))
        })?;

        if !factory.can_create(config) {
            return Err(crate::CheungfunError::configuration(format!(
                "Factory {} cannot create embedder for the given configuration",
                factory.name()
            )));
        }

        factory.create_embedder(config).await
    }

    /// Check if an embedder can be created for the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to check
    ///
    /// # Returns
    ///
    /// `true` if an embedder can be created, `false` otherwise.
    #[must_use]
    pub fn can_create(&self, config: &EmbedderConfig) -> bool {
        let embedder_type = config.provider();

        self.factories
            .get(embedder_type)
            .is_some_and(|factory| factory.can_create(config))
    }

    /// Get all registered embedder types.
    #[must_use]
    pub fn registered_types(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }

    /// Get metadata about all registered factories.
    #[must_use]
    pub fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        let mut metadata = std::collections::HashMap::new();

        for (embedder_type, factory) in &self.factories {
            metadata.insert(
                embedder_type.clone(),
                serde_json::to_value(factory.metadata()).unwrap_or_default(),
            );
        }

        metadata
    }

    /// Validate configuration for all registered factories.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to validate
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub async fn validate_config(&self, config: &EmbedderConfig) -> Result<()> {
        let embedder_type = config.provider();

        let factory = self.factories.get(embedder_type).ok_or_else(|| {
            crate::CheungfunError::configuration(format!(
                "No factory registered for embedder type: {embedder_type}"
            ))
        })?;

        factory.validate_config(config).await
    }

    /// Remove a factory from the registry.
    ///
    /// # Arguments
    ///
    /// * `embedder_type` - The type identifier to remove
    ///
    /// # Returns
    ///
    /// The removed factory, if it existed.
    pub fn unregister(&mut self, embedder_type: &str) -> Option<Arc<dyn EmbedderFactory>> {
        self.factories.remove(embedder_type)
    }

    /// Check if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }

    /// Get the number of registered factories.
    #[must_use]
    pub fn len(&self) -> usize {
        self.factories.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EmbedderConfig;

    // Mock factory for testing
    #[derive(Debug)]
    struct MockEmbedderFactory {
        supported_types: Vec<&'static str>,
    }

    impl MockEmbedderFactory {
        fn new(supported_types: Vec<&'static str>) -> Self {
            Self { supported_types }
        }
    }

    #[async_trait]
    impl EmbedderFactory for MockEmbedderFactory {
        async fn create_embedder(&self, _config: &EmbedderConfig) -> Result<Arc<dyn Embedder>> {
            // This is a mock implementation for testing
            Err(crate::CheungfunError::internal(
                "Mock factory - not implemented",
            ))
        }

        fn can_create(&self, config: &EmbedderConfig) -> bool {
            self.supported_types.contains(&config.provider())
        }

        fn supported_types(&self) -> Vec<&'static str> {
            self.supported_types.clone()
        }
    }

    #[test]
    fn test_registry_creation() {
        let registry = EmbedderFactoryRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_registration() {
        let mut registry = EmbedderFactoryRegistry::new();
        let factory = Arc::new(MockEmbedderFactory::new(vec!["candle"]));

        registry.register("candle", factory);

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
        assert!(registry.registered_types().contains(&"candle".to_string()));
    }

    #[test]
    fn test_can_create() {
        let mut registry = EmbedderFactoryRegistry::new();
        let factory = Arc::new(MockEmbedderFactory::new(vec!["candle"]));
        registry.register("candle", factory);

        let candle_config = EmbedderConfig::candle("test-model", "cpu");
        let api_config = EmbedderConfig::api("openai", "test-model", "test-key");

        assert!(registry.can_create(&candle_config));
        assert!(!registry.can_create(&api_config));
    }

    #[test]
    fn test_unregister() {
        let mut registry = EmbedderFactoryRegistry::new();
        let factory = Arc::new(MockEmbedderFactory::new(vec!["candle"]));
        registry.register("candle", factory);

        assert_eq!(registry.len(), 1);

        let removed = registry.unregister("candle");
        assert!(removed.is_some());
        assert_eq!(registry.len(), 0);

        let not_found = registry.unregister("nonexistent");
        assert!(not_found.is_none());
    }
}
