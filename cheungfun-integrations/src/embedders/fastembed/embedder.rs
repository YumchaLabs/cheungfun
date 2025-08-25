//! `FastEmbed` embedder implementation.

use async_trait::async_trait;
use cheungfun_core::{
    traits::{Embedder, EmbeddingStats},
    Result as CoreResult,
};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use super::{
    config::{FastEmbedConfig, ModelPreset},
    error::{FastEmbedError, Result},
};

/// FastEmbed-based embedder with a focus on simplicity and performance.
///
/// This embedder provides a clean, ergonomic interface for generating embeddings
/// using the `FastEmbed` library. It handles model initialization, caching, and
/// provides convenient preset configurations for common use cases.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_integrations::embedders::fastembed::FastEmbedder;
/// use cheungfun_core::traits::Embedder;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Quick start with defaults
/// let embedder = FastEmbedder::new().await?;
/// let embedding = embedder.embed("Hello, world!").await?;
///
/// // Use a specific model
/// let embedder = FastEmbedder::with_model("BAAI/bge-large-en-v1.5").await?;
///
/// // Use a preset for common scenarios
/// let embedder = FastEmbedder::multilingual().await?;
/// let embedder = FastEmbedder::high_quality().await?;
/// let embedder = FastEmbedder::fast().await?;
/// # Ok(())
/// # }
/// ```
pub struct FastEmbedder {
    /// Configuration
    config: FastEmbedConfig,
    /// `FastEmbed` model
    model: TextEmbedding,
    /// Statistics tracking
    stats: Arc<Mutex<EmbeddingStats>>,
}

impl std::fmt::Debug for FastEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedder")
            .field("config", &self.config)
            .field("stats", &"<stats>")
            .finish()
    }
}

impl FastEmbedder {
    /// Create a new embedder with default configuration.
    ///
    /// Uses the default model (BAAI/bge-small-en-v1.5) which provides a good
    /// balance of speed and quality for English text.
    pub async fn new() -> Result<Self> {
        Self::from_config(FastEmbedConfig::default()).await
    }

    /// Create an embedder with a specific model name.
    pub async fn with_model<S: Into<String>>(model_name: S) -> Result<Self> {
        let config = FastEmbedConfig::new(model_name);
        Self::from_config(config).await
    }

    /// Create an embedder optimized for high quality English text.
    /// Uses BAAI/bge-large-en-v1.5 (1024 dimensions).
    pub async fn high_quality() -> Result<Self> {
        let config = FastEmbedConfig::from_preset(ModelPreset::HighQuality);
        Self::from_config(config).await
    }

    /// Create an embedder optimized for multilingual text.
    /// Uses BAAI/bge-m3 (1024 dimensions).
    pub async fn multilingual() -> Result<Self> {
        let config = FastEmbedConfig::from_preset(ModelPreset::Multilingual);
        Self::from_config(config).await
    }

    /// Create an embedder optimized for speed.
    /// Uses sentence-transformers/all-MiniLM-L6-v2 (384 dimensions).
    pub async fn fast() -> Result<Self> {
        let config = FastEmbedConfig::from_preset(ModelPreset::Fast);
        Self::from_config(config).await
    }

    /// Create an embedder optimized for source code.
    /// Uses microsoft/codebert-base (768 dimensions).
    pub async fn for_code() -> Result<Self> {
        let config = FastEmbedConfig::from_preset(ModelPreset::Code);
        Self::from_config(config).await
    }

    /// Create an embedder from a custom configuration.
    pub async fn from_config(config: FastEmbedConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        info!("Initializing FastEmbed model: {}", config.model_name);

        // Create FastEmbed initialization options
        // First, convert model name string to EmbeddingModel enum
        let embedding_model = Self::parse_model_name(&config.model_name)?;

        let mut init_options = InitOptions::new(embedding_model)
            .with_max_length(config.max_length)
            .with_show_download_progress(config.show_progress);

        if let Some(cache_dir) = &config.cache_dir {
            init_options = init_options.with_cache_dir(cache_dir.clone());
        }

        // Initialize the model
        let model =
            TextEmbedding::try_new(init_options).map_err(|e| FastEmbedError::ModelInit {
                model: config.model_name.clone(),
                reason: e.to_string(),
            })?;

        info!("FastEmbed model initialized successfully");

        Ok(Self {
            config,
            model,
            stats: Arc::new(Mutex::new(EmbeddingStats::new())),
        })
    }

    /// Get the configuration used by this embedder.
    pub fn config(&self) -> &FastEmbedConfig {
        &self.config
    }

    /// Get embedding statistics.
    pub async fn stats(&self) -> EmbeddingStats {
        let stats = self.stats.lock().await;
        stats.clone()
    }

    /// Reset statistics.
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.lock().await;
        *stats = EmbeddingStats::new();
    }

    /// Generate embeddings with automatic retry and error handling.
    async fn generate_embeddings_with_retry(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        const MAX_RETRIES: usize = 3;
        let mut last_error = None;

        for attempt in 1..=MAX_RETRIES {
            match self
                .model
                .embed(texts.clone(), Some(self.config.batch_size))
            {
                Ok(embeddings) => {
                    debug!("Generated {} embeddings successfully", embeddings.len());
                    return Ok(embeddings);
                }
                Err(e) => {
                    warn!("Embedding attempt {} failed: {}", attempt, e);
                    last_error = Some(e);

                    if attempt < MAX_RETRIES {
                        // Exponential backoff
                        let delay = std::time::Duration::from_millis(100 * (1 << attempt));
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(FastEmbedError::Embedding {
            reason: format!(
                "Failed after {} attempts: {}",
                MAX_RETRIES,
                last_error.map_or_else(|| "Unknown error".to_string(), |e| e.to_string())
            ),
        })
    }

    /// Update statistics after an embedding operation.
    async fn update_stats(&self, texts_count: usize, duration: std::time::Duration, success: bool) {
        let mut stats = self.stats.lock().await;

        if success {
            stats.texts_embedded += texts_count;
        } else {
            stats.embeddings_failed += texts_count;
        }

        stats.duration += duration;
        stats.update_avg_time();
    }
}

#[async_trait]
impl Embedder for FastEmbedder {
    async fn embed(&self, text: &str) -> CoreResult<Vec<f32>> {
        let start_time = std::time::Instant::now();

        let result = self
            .generate_embeddings_with_retry(vec![text.to_string()])
            .await;
        let duration = start_time.elapsed();

        match result {
            Ok(mut embeddings) => {
                self.update_stats(1, duration, true).await;
                Ok(embeddings.pop().unwrap_or_default())
            }
            Err(e) => {
                self.update_stats(1, duration, false).await;
                Err(e.into())
            }
        }
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> CoreResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let start_time = std::time::Instant::now();
        let texts_owned: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();
        let texts_count = texts_owned.len();

        let result = self.generate_embeddings_with_retry(texts_owned).await;
        let duration = start_time.elapsed();

        match result {
            Ok(embeddings) => {
                self.update_stats(texts_count, duration, true).await;
                Ok(embeddings)
            }
            Err(e) => {
                self.update_stats(texts_count, duration, false).await;
                Err(e.into())
            }
        }
    }

    fn dimension(&self) -> usize {
        // Try to determine dimension from model name or use a reasonable default
        if let Some(preset) = self.get_preset_for_model(&self.config.model_name) {
            preset.dimension()
        } else {
            // Default dimension for unknown models
            384
        }
    }

    fn model_name(&self) -> &str {
        &self.config.model_name
    }

    async fn health_check(&self) -> CoreResult<()> {
        // Try a simple embedding to verify the model is working
        match self.embed("health check").await {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }
}

impl FastEmbedder {
    /// Parse model name string to `EmbeddingModel` enum.
    fn parse_model_name(model_name: &str) -> Result<EmbeddingModel> {
        match model_name {
            "BAAI/bge-small-en-v1.5" => Ok(EmbeddingModel::BGESmallENV15),
            "BAAI/bge-large-en-v1.5" => Ok(EmbeddingModel::BGELargeENV15),
            "BAAI/bge-base-en-v1.5" => Ok(EmbeddingModel::BGEBaseENV15),
            "sentence-transformers/all-MiniLM-L6-v2" => Ok(EmbeddingModel::AllMiniLML6V2),
            "intfloat/multilingual-e5-base" => Ok(EmbeddingModel::MultilingualE5Base),
            "intfloat/multilingual-e5-large" => Ok(EmbeddingModel::MultilingualE5Large),
            "jinaai/jina-embeddings-v2-base-code" => Ok(EmbeddingModel::JinaEmbeddingsV2BaseCode),
            // Add more model mappings as needed
            _ => Err(FastEmbedError::ModelInit {
                model: model_name.to_string(),
                reason: format!(
                    "Unsupported model: {model_name}. Please use a supported model name."
                ),
            }),
        }
    }

    /// Helper to determine if a model name matches a known preset.
    fn get_preset_for_model(&self, model_name: &str) -> Option<ModelPreset> {
        match model_name {
            "BAAI/bge-small-en-v1.5" => Some(ModelPreset::Default),
            "BAAI/bge-large-en-v1.5" => Some(ModelPreset::HighQuality),
            "intfloat/multilingual-e5-base" => Some(ModelPreset::Multilingual),
            "sentence-transformers/all-MiniLM-L6-v2" => Some(ModelPreset::Fast),
            "jinaai/jina-embeddings-v2-base-code" => Some(ModelPreset::Code),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_config_creation() {
        let config = FastEmbedConfig::default();
        assert_eq!(config.model_name, "BAAI/bge-small-en-v1.5");
        assert_eq!(config.max_length, 512);
        assert_eq!(config.batch_size, 32);
    }

    #[tokio::test]
    async fn test_preset_dimensions() {
        assert_eq!(ModelPreset::Default.dimension(), 384);
        assert_eq!(ModelPreset::HighQuality.dimension(), 1024);
        assert_eq!(ModelPreset::Multilingual.dimension(), 768); // Fixed: should be 768, not 1024
        assert_eq!(ModelPreset::Fast.dimension(), 384);
        assert_eq!(ModelPreset::Code.dimension(), 768);
    }

    // Note: These tests require actual model downloads and are disabled by default
    #[tokio::test]
    #[ignore]
    async fn test_embedder_creation() {
        let embedder = FastEmbedder::new().await;
        assert!(embedder.is_ok());
    }

    #[tokio::test]
    #[ignore]
    async fn test_embedding_generation() {
        let embedder = FastEmbedder::new().await.unwrap();
        let embedding = embedder.embed("Hello, world!").await;
        assert!(embedding.is_ok());

        let embedding = embedding.unwrap();
        assert_eq!(embedding.len(), 384); // Default model dimension
    }

    #[tokio::test]
    #[ignore]
    async fn test_batch_embedding() {
        let embedder = FastEmbedder::new().await.unwrap();
        let texts = vec!["Hello", "World", "Test"];
        let embeddings = embedder.embed_batch(texts).await;
        assert!(embeddings.is_ok());

        let embeddings = embeddings.unwrap();
        assert_eq!(embeddings.len(), 3);
        for embedding in embeddings {
            assert_eq!(embedding.len(), 384);
        }
    }
}
