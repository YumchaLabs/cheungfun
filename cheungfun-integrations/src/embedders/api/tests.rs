//! Tests for API embedders.

use super::embedder::ApiEmbedderBuilder;
use super::*;
use cheungfun_core::traits::Embedder;
use std::time::Duration;

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn test_openai_config_creation() {
        let config = ApiEmbedderConfig::openai("test-key", "text-embedding-3-small");

        assert_eq!(config.provider, ApiProvider::OpenAI);
        assert_eq!(config.model, "text-embedding-3-small");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.max_retries, 3);
        assert!(config.enable_cache);
    }

    #[test]
    fn test_anthropic_config_creation() {
        let config = ApiEmbedderConfig::anthropic("test-key", "claude-embedding-v1");

        assert_eq!(config.provider, ApiProvider::Anthropic);
        assert_eq!(config.model, "claude-embedding-v1");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.batch_size, 50);
    }

    #[test]
    fn test_custom_config_creation() {
        let config = ApiEmbedderConfig::custom(
            "custom-provider",
            "https://api.example.com",
            "test-key",
            "custom-model",
        );

        match config.provider {
            ApiProvider::Custom { name, base_url } => {
                assert_eq!(name, "custom-provider");
                assert_eq!(base_url, "https://api.example.com");
            }
            _ => panic!("Expected custom provider"),
        }
        assert_eq!(config.model, "custom-model");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, Some("https://api.example.com".to_string()));
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        let valid_config = ApiEmbedderConfig::openai("test-key", "text-embedding-3-small");
        assert!(valid_config.validate().is_ok());

        // Empty API key
        let mut invalid_config = valid_config.clone();
        invalid_config.api_key = String::new();
        assert!(invalid_config.validate().is_err());

        // Empty model
        let mut invalid_config = valid_config.clone();
        invalid_config.model = String::new();
        assert!(invalid_config.validate().is_err());

        // Unsupported model for OpenAI
        let mut invalid_config = valid_config.clone();
        invalid_config.model = "unsupported-model".to_string();
        assert!(invalid_config.validate().is_err());

        // Zero batch size
        let mut invalid_config = valid_config.clone();
        invalid_config.batch_size = 0;
        assert!(invalid_config.validate().is_err());

        // Zero timeout
        let mut invalid_config = valid_config;
        invalid_config.timeout = Duration::from_secs(0);
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_config_builder_methods() {
        let config = ApiEmbedderConfig::openai("test-key", "text-embedding-3-small")
            .with_batch_size(50)
            .with_max_retries(5)
            .with_timeout(Duration::from_secs(60))
            .with_cache(false)
            .with_cache_ttl(Duration::from_secs(7200))
            .with_base_url("https://custom.openai.com")
            .with_config("temperature", serde_json::json!(0.7));

        assert_eq!(config.batch_size, 50);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(!config.enable_cache);
        assert_eq!(config.cache_ttl, Duration::from_secs(7200));
        assert_eq!(
            config.base_url,
            Some("https://custom.openai.com".to_string())
        );
        assert_eq!(
            config.additional_config.get("temperature"),
            Some(&serde_json::json!(0.7))
        );
    }
}

#[cfg(test)]
mod provider_tests {
    use super::*;

    #[test]
    fn test_provider_names() {
        assert_eq!(ApiProvider::OpenAI.name(), "openai");
        assert_eq!(ApiProvider::Anthropic.name(), "anthropic");

        let custom = ApiProvider::Custom {
            name: "test-provider".to_string(),
            base_url: "https://api.test.com".to_string(),
        };
        assert_eq!(custom.name(), "test-provider");
    }

    #[test]
    fn test_openai_model_support() {
        let provider = ApiProvider::OpenAI;

        // Supported models
        assert!(provider.supports_model("text-embedding-ada-002"));
        assert!(provider.supports_model("text-embedding-3-small"));
        assert!(provider.supports_model("text-embedding-3-large"));
        assert!(provider.supports_model("text-embedding-custom"));

        // Unsupported models
        assert!(!provider.supports_model("gpt-4"));
        assert!(!provider.supports_model("claude-v1"));
        assert!(!provider.supports_model("random-model"));
    }

    #[test]
    fn test_anthropic_model_support() {
        let provider = ApiProvider::Anthropic;

        // Currently no supported models
        assert!(!provider.supports_model("claude-embedding-v1"));
        assert!(!provider.supports_model("text-embedding-3-small"));
    }

    #[test]
    fn test_custom_provider_model_support() {
        let provider = ApiProvider::Custom {
            name: "test".to_string(),
            base_url: "https://api.test.com".to_string(),
        };

        // Custom providers support any model
        assert!(provider.supports_model("any-model"));
        assert!(provider.supports_model("custom-embedding-v1"));
        assert!(provider.supports_model("test-model"));
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let config_error = ApiEmbedderError::configuration("Invalid config");
        assert!(matches!(
            config_error,
            ApiEmbedderError::Configuration { .. }
        ));

        let auth_error = ApiEmbedderError::authentication("Invalid API key");
        assert!(matches!(
            auth_error,
            ApiEmbedderError::Authentication { .. }
        ));

        let rate_limit_error = ApiEmbedderError::rate_limit("Too many requests");
        assert!(matches!(
            rate_limit_error,
            ApiEmbedderError::RateLimit { .. }
        ));

        let server_error = ApiEmbedderError::server(500, "Internal server error");
        assert!(matches!(
            server_error,
            ApiEmbedderError::Server {
                status_code: 500,
                ..
            }
        ));

        let timeout_error = ApiEmbedderError::timeout(30);
        assert!(matches!(
            timeout_error,
            ApiEmbedderError::Timeout { seconds: 30 }
        ));
    }

    #[test]
    fn test_error_retryability() {
        // Retryable errors
        assert!(ApiEmbedderError::network("Connection failed").is_retryable());
        assert!(ApiEmbedderError::server(500, "Internal error").is_retryable());
        assert!(ApiEmbedderError::server(502, "Bad gateway").is_retryable());
        assert!(ApiEmbedderError::timeout(30).is_retryable());
        assert!(ApiEmbedderError::rate_limit("Rate limited").is_retryable());

        // Non-retryable errors
        assert!(!ApiEmbedderError::configuration("Invalid config").is_retryable());
        assert!(!ApiEmbedderError::authentication("Invalid key").is_retryable());
        assert!(!ApiEmbedderError::server(400, "Bad request").is_retryable());
        assert!(!ApiEmbedderError::server(404, "Not found").is_retryable());
        assert!(!ApiEmbedderError::invalid_model("unknown-model").is_retryable());
    }

    #[test]
    fn test_error_retry_delays() {
        assert_eq!(
            ApiEmbedderError::rate_limit("Rate limited").retry_delay(),
            Some(60)
        );
        assert_eq!(
            ApiEmbedderError::network("Connection failed").retry_delay(),
            Some(5)
        );
        assert_eq!(
            ApiEmbedderError::server(500, "Internal error").retry_delay(),
            Some(5)
        );
        assert_eq!(ApiEmbedderError::timeout(30).retry_delay(), Some(5));

        assert_eq!(
            ApiEmbedderError::configuration("Invalid config").retry_delay(),
            None
        );
        assert_eq!(
            ApiEmbedderError::authentication("Invalid key").retry_delay(),
            None
        );
    }

    #[test]
    fn test_error_display() {
        let config_error = ApiEmbedderError::configuration("Test message");
        assert_eq!(
            config_error.to_string(),
            "Configuration error: Test message"
        );

        let server_error = ApiEmbedderError::server(500, "Internal error");
        assert_eq!(
            server_error.to_string(),
            "Server error: 500 - Internal error"
        );

        let timeout_error = ApiEmbedderError::timeout(30);
        assert_eq!(
            timeout_error.to_string(),
            "Request timeout after 30 seconds"
        );
    }
}

#[cfg(test)]
mod builder_tests {
    use super::*;

    #[tokio::test]
    async fn test_builder_openai() {
        // Test that builder can be created and configured
        let builder = ApiEmbedderBuilder::new().openai("test-key");

        // We can't access private fields, so we test by trying to build
        // This will fail due to network/auth issues, but that's expected
        let result = builder.build().await;
        // The result could be Ok or Err depending on network conditions
        // We just test that the builder doesn't panic
        let _ = result;
    }

    #[tokio::test]
    async fn test_builder_anthropic() {
        let builder = ApiEmbedderBuilder::new().anthropic("test-key");

        // Anthropic is not yet supported, so this should fail
        let result = builder.build().await;
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        println!("Anthropic error message: {}", error_msg);
        // Just check that it fails - the exact error message may vary
        assert!(error_msg.len() > 0);
    }

    #[tokio::test]
    async fn test_builder_chaining() {
        let builder = ApiEmbedderBuilder::new()
            .openai("test-key")
            .model("text-embedding-3-large")
            .batch_size(50)
            .enable_cache(false)
            .max_retries(5)
            .timeout(Duration::from_secs(60));

        // Test that chaining works by attempting to build
        let result = builder.build().await;
        // The result could be Ok or Err depending on network conditions
        // We just test that the builder doesn't panic
        let _ = result;
    }

    #[tokio::test]
    async fn test_builder_build_without_config() {
        let builder = ApiEmbedderBuilder::new();
        let result = builder.build().await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ApiEmbedderError::Configuration { .. }
        ));
    }
}

// Integration tests would require actual API keys, so they're marked as ignored
// Run with: cargo test -- --ignored
#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::env;

    #[tokio::test]
    #[ignore = "Requires OpenAI API key"]
    async fn test_openai_integration() {
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

        let embedder = ApiEmbedder::builder()
            .openai(api_key)
            .model("text-embedding-3-small")
            .build()
            .await
            .expect("Failed to create embedder");

        // Test single embedding
        let embedding = embedder
            .embed("Hello, world!")
            .await
            .expect("Failed to embed text");
        assert_eq!(embedding.len(), 1536);

        // Test batch embedding
        let texts = vec!["Hello", "World", "Rust is amazing!"];
        let embeddings = embedder
            .embed_batch(texts)
            .await
            .expect("Failed to embed batch");
        assert_eq!(embeddings.len(), 3);
        assert!(embeddings.iter().all(|e| e.len() == 1536));

        // Test health check
        embedder.health_check().await.expect("Health check failed");

        // Test model info
        assert_eq!(embedder.model_name(), "text-embedding-3-small");
        assert_eq!(embedder.dimension(), 1536);
    }

    #[tokio::test]
    #[ignore = "Requires OpenAI API key"]
    async fn test_caching_behavior() {
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

        let embedder = ApiEmbedder::builder()
            .openai(api_key)
            .model("text-embedding-3-small")
            .enable_cache(true)
            .build()
            .await
            .expect("Failed to create embedder");

        let text = "This is a test for caching";

        // First call - should hit API
        let start = std::time::Instant::now();
        let embedding1 = embedder.embed(text).await.expect("Failed to embed text");
        let first_duration = start.elapsed();

        // Second call - should hit cache
        let start = std::time::Instant::now();
        let embedding2 = embedder.embed(text).await.expect("Failed to embed text");
        let second_duration = start.elapsed();

        // Results should be identical
        assert_eq!(embedding1, embedding2);

        // Second call should be faster (cached)
        assert!(second_duration < first_duration);

        // Check cache stats
        if let Some(cache_stats) = embedder.cache_stats().await {
            assert!(cache_stats.hits > 0);
            assert!(cache_stats.total_entries > 0);
        }
    }
}
