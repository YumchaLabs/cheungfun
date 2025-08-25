//! Factory for creating LLM clients from configuration.
//!
//! This module provides factory traits and implementations for creating
//! LLM client instances from configuration objects, with a focus on
//! using the siumai crate for LLM integration.

use async_trait::async_trait;
use std::sync::Arc;

use crate::{config::LlmConfig, traits::ResponseGenerator, Result};

/// Factory for creating LLM clients from configuration.
///
/// This trait provides a unified interface for creating different types
/// of LLM clients based on configuration. Implementations handle the
/// specifics of each LLM provider.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::factory::LlmFactory;
/// use cheungfun_core::config::LlmConfig;
/// use cheungfun_core::traits::ResponseGenerator;
/// use cheungfun_core::Result;
/// use async_trait::async_trait;
/// use std::sync::Arc;
///
/// struct MyLlmFactory;
///
/// #[async_trait]
/// impl LlmFactory for MyLlmFactory {
///     async fn create_llm(&self, config: &LlmConfig) -> Result<Arc<dyn ResponseGenerator>> {
///         // Implementation would create LLM client based on config
///         todo!("Implement LLM client creation")
///     }
/// }
/// ```
#[async_trait]
pub trait LlmFactory: Send + Sync + std::fmt::Debug {
    /// Create an LLM client from configuration.
    ///
    /// This method takes an LLM configuration and returns a concrete
    /// response generator implementation wrapped in an Arc for shared ownership.
    ///
    /// # Arguments
    ///
    /// * `config` - The LLM configuration
    ///
    /// # Returns
    ///
    /// An Arc-wrapped response generator instance that implements the ResponseGenerator trait.
    ///
    /// # Errors
    ///
    /// Returns an error if the LLM client cannot be created due to invalid
    /// configuration, missing dependencies, or initialization failures.
    async fn create_llm(&self, config: &LlmConfig) -> Result<Arc<dyn ResponseGenerator>>;

    /// Check if this factory can create an LLM client for the given configuration.
    ///
    /// This method can be used to validate configuration before attempting
    /// to create an LLM client.
    ///
    /// # Arguments
    ///
    /// * `config` - The LLM configuration to check
    ///
    /// # Returns
    ///
    /// `true` if this factory can create an LLM client for the configuration,
    /// `false` otherwise.
    fn can_create(&self, config: &LlmConfig) -> bool;

    /// Get a human-readable name for this factory.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Get the supported LLM providers.
    ///
    /// Returns a list of LLM provider identifiers that this factory
    /// can create. This is useful for discovery and validation.
    fn supported_providers(&self) -> Vec<&'static str>;

    /// Validate the configuration without creating the LLM client.
    ///
    /// This method performs validation checks on the configuration
    /// to ensure it's valid for creating an LLM client.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to validate
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    async fn validate_config(&self, config: &LlmConfig) -> Result<()> {
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
        metadata.insert(
            "supported_providers".to_string(),
            self.supported_providers().into(),
        );
        metadata
    }
}

/// Registry for LLM factories.
///
/// This struct manages multiple LLM factories and provides
/// a unified interface for creating LLM clients from any supported
/// configuration.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::factory::{LlmFactoryRegistry, LlmFactory};
/// use cheungfun_core::config::LlmConfig;
/// use std::sync::Arc;
///
/// let mut registry = LlmFactoryRegistry::new();
/// // registry.register("openai", Arc::new(OpenAiLlmFactory::new()));
/// // registry.register("anthropic", Arc::new(AnthropicLlmFactory::new()));
///
/// let config = LlmConfig::openai("gpt-4", "your-api-key");
/// // let llm = registry.create_llm(&config).await?;
/// ```
#[derive(Debug, Default)]
pub struct LlmFactoryRegistry {
    factories: std::collections::HashMap<String, Arc<dyn LlmFactory>>,
}

impl LlmFactoryRegistry {
    /// Create a new LLM factory registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            factories: std::collections::HashMap::new(),
        }
    }

    /// Register a factory for a specific LLM provider.
    ///
    /// # Arguments
    ///
    /// * `provider` - The provider identifier for the LLM
    /// * `factory` - The factory implementation
    pub fn register<S: Into<String>>(&mut self, provider: S, factory: Arc<dyn LlmFactory>) {
        self.factories.insert(provider.into(), factory);
    }

    /// Create an LLM client from configuration.
    ///
    /// This method finds the appropriate factory for the configuration
    /// and uses it to create the LLM client.
    ///
    /// # Arguments
    ///
    /// * `config` - The LLM configuration
    ///
    /// # Returns
    ///
    /// An Arc-wrapped response generator instance.
    ///
    /// # Errors
    ///
    /// Returns an error if no suitable factory is found or if the
    /// factory fails to create the LLM client.
    pub async fn create_llm(&self, config: &LlmConfig) -> Result<Arc<dyn ResponseGenerator>> {
        let provider = &config.provider;

        let factory = self.factories.get(provider).ok_or_else(|| {
            crate::CheungfunError::configuration(format!(
                "No factory registered for LLM provider: {provider}"
            ))
        })?;

        if !factory.can_create(config) {
            return Err(crate::CheungfunError::configuration(format!(
                "Factory {} cannot create LLM client for the given configuration",
                factory.name()
            )));
        }

        factory.create_llm(config).await
    }

    /// Check if an LLM client can be created for the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to check
    ///
    /// # Returns
    ///
    /// `true` if an LLM client can be created, `false` otherwise.
    #[must_use]
    pub fn can_create(&self, config: &LlmConfig) -> bool {
        let provider = &config.provider;

        self.factories
            .get(provider)
            .is_some_and(|factory| factory.can_create(config))
    }

    /// Get all registered LLM providers.
    #[must_use]
    pub fn registered_providers(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }

    /// Get metadata about all registered factories.
    #[must_use]
    pub fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        let mut metadata = std::collections::HashMap::new();

        for (provider, factory) in &self.factories {
            metadata.insert(
                provider.clone(),
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
    pub async fn validate_config(&self, config: &LlmConfig) -> Result<()> {
        let provider = &config.provider;

        let factory = self.factories.get(provider).ok_or_else(|| {
            crate::CheungfunError::configuration(format!(
                "No factory registered for LLM provider: {provider}"
            ))
        })?;

        factory.validate_config(config).await
    }

    /// Remove a factory from the registry.
    ///
    /// # Arguments
    ///
    /// * `provider` - The provider identifier to remove
    ///
    /// # Returns
    ///
    /// The removed factory, if it existed.
    pub fn unregister(&mut self, provider: &str) -> Option<Arc<dyn LlmFactory>> {
        self.factories.remove(provider)
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

/// Siumai-based LLM factory implementation.
///
/// This factory creates LLM clients using the siumai crate, which provides
/// a unified interface for multiple LLM providers including `OpenAI`, Anthropic,
/// Google Gemini, Ollama, and others.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::factory::{SiumaiLlmFactory, LlmFactory};
/// use cheungfun_core::config::LlmConfig;
/// use std::sync::Arc;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let factory = SiumaiLlmFactory::new();
///
/// // Create OpenAI client
/// let openai_config = LlmConfig::openai("gpt-4", "your-api-key");
/// let openai_client = factory.create_llm(&openai_config).await?;
///
/// // Create Anthropic client
/// let anthropic_config = LlmConfig::anthropic("claude-3-sonnet", "your-api-key");
/// let anthropic_client = factory.create_llm(&anthropic_config).await?;
///
/// // Create Ollama client
/// let ollama_config = LlmConfig::ollama("llama2");
/// let ollama_client = factory.create_llm(&ollama_config).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct SiumaiLlmFactory;

impl SiumaiLlmFactory {
    /// Create a new Siumai LLM factory.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl LlmFactory for SiumaiLlmFactory {
    async fn create_llm(&self, _config: &LlmConfig) -> Result<Arc<dyn ResponseGenerator>> {
        // This is a placeholder implementation. The actual implementation
        // should be done in higher-level crates that have access to
        // concrete ResponseGenerator implementations.
        Err(crate::CheungfunError::configuration(
            "SiumaiLlmFactory is a base factory. Use concrete implementations in higher-level crates like cheungfun-query.".to_string()
        ))
    }

    fn can_create(&self, config: &LlmConfig) -> bool {
        matches!(
            config.provider.as_str(),
            "openai" | "anthropic" | "google" | "gemini" | "ollama" | "local"
        )
    }

    fn supported_providers(&self) -> Vec<&'static str> {
        vec!["openai", "anthropic", "google", "gemini", "ollama", "local"]
    }

    async fn validate_config(&self, config: &LlmConfig) -> Result<()> {
        // First, use the config's built-in validation
        config.validate()?;

        // Additional siumai-specific validation
        if !self.can_create(config) {
            return Err(crate::CheungfunError::configuration(format!(
                "Unsupported provider for SiumaiLlmFactory: {}",
                config.provider
            )));
        }

        // Check provider-specific requirements
        match config.provider.as_str() {
            "openai" | "anthropic" | "google" | "gemini" => {
                if config.api_key.is_none() {
                    return Err(crate::CheungfunError::configuration(format!(
                        "API key is required for provider: {}",
                        config.provider
                    )));
                }
            }
            "local" => {
                if config.base_url.is_none() {
                    return Err(crate::CheungfunError::configuration(
                        "base_url is required for local provider".to_string(),
                    ));
                }
            }
            "ollama" => {
                // Ollama is optional for base_url, defaults to localhost:11434
            }
            _ => {
                return Err(crate::CheungfunError::configuration(format!(
                    "Unknown provider: {}",
                    config.provider
                )));
            }
        }

        Ok(())
    }

    fn metadata(&self) -> std::collections::HashMap<String, serde_json::Value> {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("name".to_string(), "SiumaiLlmFactory".into());
        metadata.insert(
            "description".to_string(),
            "Factory for creating LLM clients using the siumai crate".into(),
        );
        metadata.insert(
            "supported_providers".to_string(),
            self.supported_providers().into(),
        );
        metadata.insert("version".to_string(), env!("CARGO_PKG_VERSION").into());
        metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LlmConfig;

    // Mock factory for testing
    #[derive(Debug)]
    struct MockLlmFactory {
        supported_providers: Vec<&'static str>,
    }

    impl MockLlmFactory {
        fn new(supported_providers: Vec<&'static str>) -> Self {
            Self {
                supported_providers,
            }
        }
    }

    #[async_trait]
    impl LlmFactory for MockLlmFactory {
        async fn create_llm(&self, _config: &LlmConfig) -> Result<Arc<dyn ResponseGenerator>> {
            // This is a mock implementation for testing
            Err(crate::CheungfunError::internal(
                "Mock factory - not implemented",
            ))
        }

        fn can_create(&self, config: &LlmConfig) -> bool {
            self.supported_providers.contains(&config.provider.as_str())
        }

        fn supported_providers(&self) -> Vec<&'static str> {
            self.supported_providers.clone()
        }
    }

    #[test]
    fn test_registry_creation() {
        let registry = LlmFactoryRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_registration() {
        let mut registry = LlmFactoryRegistry::new();
        let factory = Arc::new(MockLlmFactory::new(vec!["openai"]));

        registry.register("openai", factory);

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
        assert!(registry
            .registered_providers()
            .contains(&"openai".to_string()));
    }

    #[test]
    fn test_can_create() {
        let mut registry = LlmFactoryRegistry::new();
        let factory = Arc::new(MockLlmFactory::new(vec!["openai"]));
        registry.register("openai", factory);

        let openai_config = LlmConfig::openai("gpt-4", "test-key");
        let anthropic_config = LlmConfig::anthropic("claude-3", "test-key");

        assert!(registry.can_create(&openai_config));
        assert!(!registry.can_create(&anthropic_config));
    }

    #[test]
    fn test_unregister() {
        let mut registry = LlmFactoryRegistry::new();
        let factory = Arc::new(MockLlmFactory::new(vec!["openai"]));
        registry.register("openai", factory);

        assert_eq!(registry.len(), 1);

        let removed = registry.unregister("openai");
        assert!(removed.is_some());
        assert_eq!(registry.len(), 0);

        let not_found = registry.unregister("nonexistent");
        assert!(not_found.is_none());
    }
}
