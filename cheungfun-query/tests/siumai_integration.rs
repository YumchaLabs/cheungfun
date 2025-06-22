//! Integration tests for siumai LLM integration.
//!
//! These tests verify that the siumai integration works correctly with
//! different LLM providers and configurations.

use cheungfun_core::{
    config::LlmConfig,
    factory::LlmFactory,
};
use cheungfun_query::{
    factory::SiumaiLlmFactory,
    generator::SiumaiGeneratorConfig,
};





#[tokio::test]
async fn test_siumai_generator_config_creation() {
    let config = SiumaiGeneratorConfig {
        default_model: Some("gpt-4".to_string()),
        default_temperature: 0.8,
        default_max_tokens: 2000,
        default_system_prompt: "You are a helpful assistant.".to_string(),
        include_citations: true,
        max_context_length: 10000,
        timeout_seconds: 120,
    };

    assert_eq!(config.default_model, Some("gpt-4".to_string()));
    assert_eq!(config.default_temperature, 0.8);
    assert_eq!(config.default_max_tokens, 2000);
    assert!(config.include_citations);
}

#[tokio::test]
async fn test_siumai_generator_config_default() {
    let config = SiumaiGeneratorConfig::default();

    assert!(config.default_model.is_none());
    assert_eq!(config.default_temperature, 0.7);
    assert_eq!(config.default_max_tokens, 1000);
    assert!(config.include_citations); // Default is true
    assert_eq!(config.max_context_length, 8000);
    assert_eq!(config.timeout_seconds, 60);
}

#[tokio::test]
async fn test_siumai_llm_factory_supported_providers() {
    let factory = SiumaiLlmFactory::new();
    let providers = factory.supported_providers();
    
    assert!(providers.contains(&"openai"));
    assert!(providers.contains(&"anthropic"));
    assert!(providers.contains(&"ollama"));
    assert!(providers.contains(&"local"));
}

#[tokio::test]
async fn test_siumai_llm_factory_can_create() {
    let factory = SiumaiLlmFactory::new();
    
    let openai_config = LlmConfig::openai("gpt-4", "test-key");
    assert!(factory.can_create(&openai_config));
    
    let anthropic_config = LlmConfig::anthropic("claude-3", "test-key");
    assert!(factory.can_create(&anthropic_config));
    
    let unsupported_config = LlmConfig::new("unsupported", "model");
    assert!(!factory.can_create(&unsupported_config));
}

#[tokio::test]
async fn test_siumai_llm_factory_validation() {
    let factory = SiumaiLlmFactory::new();
    
    // Valid OpenAI config
    let valid_config = LlmConfig::openai("gpt-4", "test-key");
    assert!(factory.validate_config(&valid_config).await.is_ok());
    
    // Invalid config - missing API key
    let invalid_config = LlmConfig::new("openai", "gpt-4");
    assert!(factory.validate_config(&invalid_config).await.is_err());
    
    // Invalid config - unsupported provider
    let unsupported_config = LlmConfig::new("unsupported", "model");
    assert!(factory.validate_config(&unsupported_config).await.is_err());
}

#[tokio::test]
async fn test_siumai_llm_factory_metadata() {
    let factory = SiumaiLlmFactory::new();
    let metadata = factory.metadata();
    
    assert_eq!(metadata.get("name").unwrap(), "SiumaiLlmFactory");
    assert!(metadata.contains_key("description"));
    assert!(metadata.contains_key("supported_providers"));
    assert!(metadata.contains_key("version"));
}
