//! Factory for creating vector stores from configuration.
//!
//! This module provides factory traits and implementations for creating
//! vector store instances from configuration objects.

use async_trait::async_trait;
use std::sync::Arc;

use crate::{config::VectorStoreConfig, traits::VectorStore, Result};

/// Factory for creating vector stores from configuration.
///
/// This trait provides a unified interface for creating different types
/// of vector stores based on configuration. Implementations handle the
/// specifics of each vector store type.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::factory::VectorStoreFactory;
/// use cheungfun_core::config::VectorStoreConfig;
/// use cheungfun_core::traits::VectorStore;
/// use cheungfun_core::Result;
/// use async_trait::async_trait;
/// use std::sync::Arc;
///
/// #[derive(Debug)]
/// struct MyVectorStoreFactory;
///
/// #[async_trait]
/// impl VectorStoreFactory for MyVectorStoreFactory {
///     async fn create_vector_store(&self, config: &VectorStoreConfig) -> Result<Arc<dyn VectorStore>> {
///         // Implementation would create vector store based on config
///         todo!("Implement vector store creation")
///     }
///
///     fn can_create(&self, config: &VectorStoreConfig) -> bool {
///         // Check if this factory can create the requested vector store type
///         true
///     }
///
///     fn supported_types(&self) -> Vec<&'static str> {
///         vec!["in-memory", "qdrant"]
///     }
/// }
/// ```
#[async_trait]
pub trait VectorStoreFactory: Send + Sync + std::fmt::Debug {
    /// Create a vector store from configuration.
    ///
    /// This method takes a vector store configuration and returns a concrete
    /// vector store implementation wrapped in an Arc for shared ownership.
    ///
    /// # Arguments
    ///
    /// * `config` - The vector store configuration
    ///
    /// # Returns
    ///
    /// An Arc-wrapped vector store instance that implements the VectorStore trait.
    ///
    /// # Errors
    ///
    /// Returns an error if the vector store cannot be created due to invalid
    /// configuration, missing dependencies, or initialization failures.
    async fn create_vector_store(&self, config: &VectorStoreConfig)
        -> Result<Arc<dyn VectorStore>>;

    /// Check if this factory can create a vector store for the given configuration.
    ///
    /// This method can be used to validate configuration before attempting
    /// to create a vector store.
    ///
    /// # Arguments
    ///
    /// * `config` - The vector store configuration to check
    ///
    /// # Returns
    ///
    /// `true` if this factory can create a vector store for the configuration,
    /// `false` otherwise.
    fn can_create(&self, config: &VectorStoreConfig) -> bool;

    /// Get a human-readable name for this factory.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Get the supported vector store types.
    ///
    /// Returns a list of vector store type identifiers that this factory
    /// can create. This is useful for discovery and validation.
    fn supported_types(&self) -> Vec<&'static str>;

    /// Validate the configuration without creating the vector store.
    ///
    /// This method performs validation checks on the configuration
    /// to ensure it's valid for creating a vector store.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to validate
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    async fn validate_config(&self, config: &VectorStoreConfig) -> Result<()> {
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

/// Registry for vector store factories.
///
/// This struct manages multiple vector store factories and provides
/// a unified interface for creating vector stores from any supported
/// configuration.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::factory::{VectorStoreFactoryRegistry, VectorStoreFactory};
/// use cheungfun_core::config::VectorStoreConfig;
/// use std::sync::Arc;
///
/// let mut registry = VectorStoreFactoryRegistry::new();
/// // registry.register("memory", Arc::new(MemoryVectorStoreFactory::new()));
/// // registry.register("qdrant", Arc::new(QdrantVectorStoreFactory::new()));
///
/// let config = VectorStoreConfig::memory(768);
/// // let vector_store = registry.create_vector_store(&config).await?;
/// ```
#[derive(Debug, Default)]
pub struct VectorStoreFactoryRegistry {
    factories: std::collections::HashMap<String, Arc<dyn VectorStoreFactory>>,
}

impl VectorStoreFactoryRegistry {
    /// Create a new vector store factory registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            factories: std::collections::HashMap::new(),
        }
    }

    /// Register a factory for a specific vector store type.
    ///
    /// # Arguments
    ///
    /// * `store_type` - The type identifier for the vector store
    /// * `factory` - The factory implementation
    pub fn register<S: Into<String>>(
        &mut self,
        store_type: S,
        factory: Arc<dyn VectorStoreFactory>,
    ) {
        self.factories.insert(store_type.into(), factory);
    }

    /// Create a vector store from configuration.
    ///
    /// This method finds the appropriate factory for the configuration
    /// and uses it to create the vector store.
    ///
    /// # Arguments
    ///
    /// * `config` - The vector store configuration
    ///
    /// # Returns
    ///
    /// An Arc-wrapped vector store instance.
    ///
    /// # Errors
    ///
    /// Returns an error if no suitable factory is found or if the
    /// factory fails to create the vector store.
    pub async fn create_vector_store(
        &self,
        config: &VectorStoreConfig,
    ) -> Result<Arc<dyn VectorStore>> {
        let store_type = config.store_type();

        let factory = self.factories.get(store_type).ok_or_else(|| {
            crate::CheungfunError::configuration(format!(
                "No factory registered for vector store type: {store_type}"
            ))
        })?;

        if !factory.can_create(config) {
            return Err(crate::CheungfunError::configuration(format!(
                "Factory {} cannot create vector store for the given configuration",
                factory.name()
            )));
        }

        factory.create_vector_store(config).await
    }

    /// Check if a vector store can be created for the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to check
    ///
    /// # Returns
    ///
    /// `true` if a vector store can be created, `false` otherwise.
    #[must_use]
    pub fn can_create(&self, config: &VectorStoreConfig) -> bool {
        let store_type = config.store_type();

        self.factories
            .get(store_type)
            .is_some_and(|factory| factory.can_create(config))
    }

    /// Get all registered vector store types.
    #[must_use]
    pub fn registered_types(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }

    /// Get metadata about all registered factories.
    #[must_use]
    pub fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        let mut metadata = std::collections::HashMap::new();

        for (store_type, factory) in &self.factories {
            metadata.insert(
                store_type.clone(),
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
    pub async fn validate_config(&self, config: &VectorStoreConfig) -> Result<()> {
        let store_type = config.store_type();

        let factory = self.factories.get(store_type).ok_or_else(|| {
            crate::CheungfunError::configuration(format!(
                "No factory registered for vector store type: {store_type}"
            ))
        })?;

        factory.validate_config(config).await
    }

    /// Remove a factory from the registry.
    ///
    /// # Arguments
    ///
    /// * `store_type` - The type identifier to remove
    ///
    /// # Returns
    ///
    /// The removed factory, if it existed.
    pub fn unregister(&mut self, store_type: &str) -> Option<Arc<dyn VectorStoreFactory>> {
        self.factories.remove(store_type)
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
    use crate::config::VectorStoreConfig;

    // Mock factory for testing
    #[derive(Debug)]
    struct MockVectorStoreFactory {
        supported_types: Vec<&'static str>,
    }

    impl MockVectorStoreFactory {
        fn new(supported_types: Vec<&'static str>) -> Self {
            Self { supported_types }
        }
    }

    #[async_trait]
    impl VectorStoreFactory for MockVectorStoreFactory {
        async fn create_vector_store(
            &self,
            _config: &VectorStoreConfig,
        ) -> Result<Arc<dyn VectorStore>> {
            // This is a mock implementation for testing
            Err(crate::CheungfunError::internal(
                "Mock factory - not implemented",
            ))
        }

        fn can_create(&self, config: &VectorStoreConfig) -> bool {
            self.supported_types.contains(&config.store_type())
        }

        fn supported_types(&self) -> Vec<&'static str> {
            self.supported_types.clone()
        }
    }

    #[test]
    fn test_registry_creation() {
        let registry = VectorStoreFactoryRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_registration() {
        let mut registry = VectorStoreFactoryRegistry::new();
        let factory = Arc::new(MockVectorStoreFactory::new(vec!["memory"]));

        registry.register("memory", factory);

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
        assert!(registry.registered_types().contains(&"memory".to_string()));
    }

    #[test]
    fn test_can_create() {
        let mut registry = VectorStoreFactoryRegistry::new();
        let factory = Arc::new(MockVectorStoreFactory::new(vec!["memory"]));
        registry.register("memory", factory);

        let memory_config = VectorStoreConfig::memory(768);
        let qdrant_config = VectorStoreConfig::qdrant("http://localhost:6333", "test", 768);

        assert!(registry.can_create(&memory_config));
        assert!(!registry.can_create(&qdrant_config));
    }

    #[test]
    fn test_unregister() {
        let mut registry = VectorStoreFactoryRegistry::new();
        let factory = Arc::new(MockVectorStoreFactory::new(vec!["memory"]));
        registry.register("memory", factory);

        assert_eq!(registry.len(), 1);

        let removed = registry.unregister("memory");
        assert!(removed.is_some());
        assert_eq!(registry.len(), 0);

        let not_found = registry.unregister("nonexistent");
        assert!(not_found.is_none());
    }
}
