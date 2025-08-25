//! Integration tests for FastEmbed embedder.

#[cfg(feature = "fastembed")]
mod tests {
    use cheungfun_core::traits::Embedder;
    use cheungfun_integrations::embedders::fastembed::{
        FastEmbedConfig, FastEmbedder, ModelPreset,
    };

    #[tokio::test]
    #[ignore] // Ignore by default since it requires network access
    async fn test_fastembed_creation() {
        let embedder = FastEmbedder::new().await;
        assert!(
            embedder.is_ok(),
            "Failed to create FastEmbedder: {:?}",
            embedder.err()
        );
    }

    #[tokio::test]
    #[ignore] // Ignore by default since it requires network access
    async fn test_fastembed_with_model() {
        let embedder = FastEmbedder::with_model("BAAI/bge-small-en-v1.5").await;
        assert!(
            embedder.is_ok(),
            "Failed to create FastEmbedder with model: {:?}",
            embedder.err()
        );
    }

    #[tokio::test]
    #[ignore] // Ignore by default since it requires network access
    async fn test_fastembed_presets() {
        // Test high quality preset
        let embedder = FastEmbedder::high_quality().await;
        assert!(
            embedder.is_ok(),
            "Failed to create high quality FastEmbedder: {:?}",
            embedder.err()
        );

        // Test fast preset
        let embedder = FastEmbedder::fast().await;
        assert!(
            embedder.is_ok(),
            "Failed to create fast FastEmbedder: {:?}",
            embedder.err()
        );
    }

    #[tokio::test]
    #[ignore] // Ignore by default since it requires network access
    async fn test_single_embedding() {
        let embedder = FastEmbedder::new()
            .await
            .expect("Failed to create embedder");

        let text = "Hello, world! This is a test sentence.";
        let embedding = embedder.embed(text).await;

        assert!(
            embedding.is_ok(),
            "Failed to generate embedding: {:?}",
            embedding.err()
        );

        let embedding = embedding.unwrap();
        assert_eq!(embedding.len(), 384, "Expected 384-dimensional embedding");

        // Check that embedding values are reasonable
        assert!(
            embedding.iter().any(|&x| x != 0.0),
            "Embedding should not be all zeros"
        );
    }

    #[tokio::test]
    #[ignore] // Ignore by default since it requires network access
    async fn test_batch_embeddings() {
        let embedder = FastEmbedder::new()
            .await
            .expect("Failed to create embedder");

        let texts = vec![
            "Hello, world!",
            "This is a test.",
            "Rust is a great programming language.",
            "FastEmbed makes embeddings easy.",
        ];

        let embeddings = embedder.embed_batch(texts.clone()).await;
        assert!(
            embeddings.is_ok(),
            "Failed to generate batch embeddings: {:?}",
            embeddings.err()
        );

        let embeddings = embeddings.unwrap();
        assert_eq!(
            embeddings.len(),
            texts.len(),
            "Should have one embedding per text"
        );

        for embedding in &embeddings {
            assert_eq!(
                embedding.len(),
                384,
                "Each embedding should be 384-dimensional"
            );
            assert!(
                embedding.iter().any(|&x| x != 0.0),
                "Embedding should not be all zeros"
            );
        }
    }

    #[tokio::test]
    #[ignore] // Ignore by default since it requires network access
    async fn test_empty_batch() {
        let embedder = FastEmbedder::new()
            .await
            .expect("Failed to create embedder");

        let embeddings = embedder.embed_batch(vec![]).await;
        assert!(embeddings.is_ok(), "Empty batch should succeed");
        assert_eq!(
            embeddings.unwrap().len(),
            0,
            "Empty batch should return empty result"
        );
    }

    #[tokio::test]
    #[ignore] // Ignore by default since it requires network access
    async fn test_health_check() {
        let embedder = FastEmbedder::new()
            .await
            .expect("Failed to create embedder");

        let health = embedder.health_check().await;
        assert!(
            health.is_ok(),
            "Health check should pass: {:?}",
            health.err()
        );
    }

    #[tokio::test]
    #[ignore] // Ignore by default since it requires network access
    async fn test_stats() {
        let embedder = FastEmbedder::new()
            .await
            .expect("Failed to create embedder");

        // Initial stats should be empty
        let stats = embedder.stats().await;
        assert_eq!(stats.texts_embedded, 0);
        assert_eq!(stats.embeddings_failed, 0);

        // Generate some embeddings
        let _ = embedder.embed("test").await.unwrap();
        let _ = embedder.embed_batch(vec!["test1", "test2"]).await.unwrap();

        // Check updated stats
        let stats = embedder.stats().await;
        assert_eq!(stats.texts_embedded, 3); // 1 + 2
        assert_eq!(stats.embeddings_failed, 0);
    }

    #[test]
    fn test_config_builder() {
        let config = FastEmbedConfig::new("test-model")
            .with_max_length(256)
            .with_batch_size(16)
            .without_progress();

        assert_eq!(config.model_name, "test-model");
        assert_eq!(config.max_length, 256);
        assert_eq!(config.batch_size, 16);
        assert!(!config.show_progress);
    }

    #[test]
    fn test_model_presets() {
        assert_eq!(ModelPreset::Default.model_name(), "BAAI/bge-small-en-v1.5");
        assert_eq!(ModelPreset::Default.dimension(), 384);

        assert_eq!(
            ModelPreset::HighQuality.model_name(),
            "BAAI/bge-large-en-v1.5"
        );
        assert_eq!(ModelPreset::HighQuality.dimension(), 1024);

        assert_eq!(
            ModelPreset::Multilingual.model_name(),
            "intfloat/multilingual-e5-base"
        );
        assert_eq!(ModelPreset::Multilingual.dimension(), 768);

        assert_eq!(
            ModelPreset::Fast.model_name(),
            "sentence-transformers/all-MiniLM-L6-v2"
        );
        assert_eq!(ModelPreset::Fast.dimension(), 384);

        assert_eq!(
            ModelPreset::Code.model_name(),
            "jinaai/jina-embeddings-v2-base-code"
        );
        assert_eq!(ModelPreset::Code.dimension(), 768);
    }

    #[test]
    fn test_config_validation() {
        // Valid config should pass
        let valid_config = FastEmbedConfig::default();
        assert!(valid_config.validate().is_ok());

        // Invalid configs should fail
        let invalid_config = FastEmbedConfig {
            model_name: "".to_string(),
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());

        let invalid_config = FastEmbedConfig {
            max_length: 0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());

        let invalid_config = FastEmbedConfig {
            batch_size: 0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }
}
