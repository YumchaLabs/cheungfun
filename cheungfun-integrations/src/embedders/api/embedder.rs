//! API-based embedder implementation using siumai.

use async_trait::async_trait;
use cheungfun_core::{
    traits::{Embedder, EmbeddingStats},
    Result as CoreResult,
};
use siumai::{
    providers::openai::{OpenAiConfig, OpenAiEmbeddings},
    traits::EmbeddingCapability,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use super::{
    cache::{CacheKey, EmbeddingCache, InMemoryCache},
    config::{ApiEmbedderConfig, ApiProvider},
    error::{ApiEmbedderError, Result},
};

/// API-based embedder using siumai for cloud embedding services.
///
/// This embedder provides access to various cloud embedding APIs through
/// the siumai library, with built-in caching, retry mechanisms, and cost tracking.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_integrations::embedders::api::ApiEmbedder;
/// use cheungfun_core::traits::Embedder;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Simple usage with OpenAI
/// let embedder = ApiEmbedder::builder()
///     .openai("your-api-key")
///     .model("text-embedding-3-small")
///     .build()
///     .await?;
///
/// let embedding = embedder.embed("Hello, world!").await?;
/// println!("Embedding dimension: {}", embedding.len());
///
/// // Batch processing
/// let texts = vec!["Hello", "World", "Rust is amazing!"];
/// let embeddings = embedder.embed_batch(texts).await?;
/// println!("Generated {} embeddings", embeddings.len());
/// # Ok(())
/// # }
/// ```
pub struct ApiEmbedder {
    /// Configuration
    config: ApiEmbedderConfig,
    /// Siumai embedding client
    client: Box<dyn EmbeddingCapability + Send + Sync>,
    /// Embedding cache
    cache: Option<Box<dyn EmbeddingCache>>,
    /// Statistics tracking
    stats: Arc<Mutex<EmbeddingStats>>,
    /// HTTP client for requests
    http_client: reqwest::Client,
}

impl std::fmt::Debug for ApiEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiEmbedder")
            .field("config", &self.config)
            .field("cache_enabled", &self.cache.is_some())
            .field("stats", &"<stats>")
            .finish()
    }
}

impl ApiEmbedder {
    /// Create a new API embedder from configuration.
    pub async fn from_config(config: ApiEmbedderConfig) -> Result<Self> {
        // Validate configuration
        config.validate().map_err(ApiEmbedderError::configuration)?;

        // Create HTTP client with timeout
        let http_client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| {
                ApiEmbedderError::network(format!("Failed to create HTTP client: {}", e))
            })?;

        // Create siumai client based on provider
        let client: Box<dyn EmbeddingCapability + Send + Sync> = match &config.provider {
            ApiProvider::OpenAI => {
                let mut openai_config = OpenAiConfig::new(&config.api_key);

                // Set custom base URL if provided
                if let Some(base_url) = &config.base_url {
                    openai_config = openai_config.with_base_url(base_url);
                }

                Box::new(OpenAiEmbeddings::new(openai_config, http_client.clone()))
            }
            ApiProvider::Anthropic => {
                return Err(ApiEmbedderError::configuration(
                    "Anthropic embedding API is not yet supported",
                ));
            }
            ApiProvider::Custom { name, base_url } => {
                return Err(ApiEmbedderError::configuration(format!(
                    "Custom provider '{}' with base URL '{}' is not yet supported",
                    name, base_url
                )));
            }
        };

        // Create cache if enabled
        let cache = if config.enable_cache {
            Some(Box::new(InMemoryCache::new()) as Box<dyn EmbeddingCache>)
        } else {
            None
        };

        Ok(Self {
            config,
            client,
            cache,
            stats: Arc::new(Mutex::new(EmbeddingStats::default())),
            http_client,
        })
    }

    /// Create a builder for configuring the API embedder.
    pub fn builder() -> ApiEmbedderBuilder {
        ApiEmbedderBuilder::new()
    }

    /// Generate embeddings with caching and retry logic.
    async fn generate_embeddings_with_cache(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let start_time = Instant::now();

        // Check cache first if enabled
        let mut cache_results = Vec::new();
        let mut uncached_texts = Vec::new();
        let mut uncached_indices = Vec::new();

        if let Some(cache) = &self.cache {
            for (i, text) in texts.iter().enumerate() {
                let cache_key = CacheKey::new(&self.config.model, text);
                match cache.get(&cache_key).await {
                    Ok(Some(embedding)) => {
                        cache_results.push((i, embedding));
                        debug!("Cache hit for text index {}", i);
                    }
                    Ok(None) => {
                        uncached_texts.push(text.clone());
                        uncached_indices.push(i);
                        debug!("Cache miss for text index {}", i);
                    }
                    Err(e) => {
                        warn!("Cache error for text index {}: {}", i, e);
                        uncached_texts.push(text.clone());
                        uncached_indices.push(i);
                    }
                }
            }
        } else {
            // No cache, process all texts
            uncached_texts = texts.clone();
            uncached_indices = (0..uncached_texts.len()).collect();
        }

        // Generate embeddings for uncached texts
        let mut api_results = Vec::new();
        if !uncached_texts.is_empty() {
            info!("Generating {} embeddings via API", uncached_texts.len());

            let embeddings = self
                .generate_embeddings_via_api(uncached_texts.clone())
                .await?;

            // Cache the results if caching is enabled
            if let Some(cache) = &self.cache {
                let cache_entries: Vec<_> = uncached_texts
                    .iter()
                    .zip(embeddings.iter())
                    .map(|(text, embedding)| {
                        (CacheKey::new(&self.config.model, text), embedding.clone())
                    })
                    .collect();

                if let Err(e) = cache.put_batch(cache_entries, self.config.cache_ttl).await {
                    warn!("Failed to cache embeddings: {}", e);
                }
            }

            api_results = uncached_indices.into_iter().zip(embeddings).collect();
        }

        // Combine cached and API results
        let mut final_results = vec![Vec::new(); texts.len()];

        for (index, embedding) in cache_results {
            final_results[index] = embedding;
        }

        for (index, embedding) in api_results {
            final_results[index] = embedding;
        }

        let duration = start_time.elapsed();
        self.update_stats(texts.len(), duration, true).await;

        Ok(final_results)
    }

    /// Generate embeddings via API with retry logic.
    async fn generate_embeddings_via_api(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match self.client.embed(texts.clone()).await {
                Ok(response) => {
                    debug!(
                        "Successfully generated {} embeddings on attempt {}",
                        response.embeddings.len(),
                        attempt + 1
                    );
                    return Ok(response.embeddings);
                }
                Err(e) => {
                    last_error = Some(e);

                    if attempt < self.config.max_retries {
                        let delay = Duration::from_secs(2_u64.pow(attempt as u32)); // Exponential backoff
                        warn!(
                            "API request failed on attempt {}, retrying in {:?}: {}",
                            attempt + 1,
                            delay,
                            last_error.as_ref().unwrap()
                        );
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(ApiEmbedderError::Siumai(last_error.unwrap().to_string()))
    }

    /// Update statistics.
    async fn update_stats(&self, count: usize, duration: Duration, success: bool) {
        let mut stats = self.stats.lock().await;

        if success {
            stats.texts_embedded += count;
        } else {
            stats.embeddings_failed += count;
        }

        stats.duration += duration;
        stats.update_avg_time();
    }

    /// Get embedding statistics.
    pub async fn stats(&self) -> EmbeddingStats {
        self.stats.lock().await.clone()
    }

    /// Get cache statistics if caching is enabled.
    pub async fn cache_stats(&self) -> Option<super::cache::CacheStats> {
        if let Some(cache) = &self.cache {
            Some(cache.stats().await)
        } else {
            None
        }
    }

    /// Clear the cache if caching is enabled.
    pub async fn clear_cache(&self) -> Result<()> {
        if let Some(cache) = &self.cache {
            cache.clear().await?;
        }
        Ok(())
    }

    /// Cleanup expired cache entries.
    pub async fn cleanup_cache(&self) -> Result<usize> {
        if let Some(cache) = &self.cache {
            cache.cleanup().await
        } else {
            Ok(0)
        }
    }
}

#[async_trait]
impl Embedder for ApiEmbedder {
    async fn embed(&self, text: &str) -> CoreResult<Vec<f32>> {
        let embeddings = self
            .generate_embeddings_with_cache(vec![text.to_string()])
            .await
            .map_err(|e| cheungfun_core::CheungfunError::from(e))?;

        Ok(embeddings.into_iter().next().unwrap_or_default())
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> CoreResult<Vec<Vec<f32>>> {
        let text_strings: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        self.generate_embeddings_with_cache(text_strings)
            .await
            .map_err(|e| cheungfun_core::CheungfunError::from(e))
    }

    fn dimension(&self) -> usize {
        // Return dimension based on model
        match self.config.model.as_str() {
            "text-embedding-ada-002" => 1536,
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            _ => 1536, // Default dimension
        }
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    async fn health_check(&self) -> CoreResult<()> {
        // Try to embed a simple test text
        match self.embed("health check").await {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }
}

/// Builder for creating API embedders.
pub struct ApiEmbedderBuilder {
    config: Option<ApiEmbedderConfig>,
}

impl ApiEmbedderBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self { config: None }
    }

    /// Configure for OpenAI provider.
    pub fn openai<S: Into<String>>(mut self, api_key: S) -> Self {
        self.config = Some(ApiEmbedderConfig::openai(
            api_key.into(),
            "text-embedding-3-small".to_string(),
        ));
        self
    }

    /// Configure for Anthropic provider.
    pub fn anthropic<S: Into<String>>(mut self, api_key: S) -> Self {
        self.config = Some(ApiEmbedderConfig::anthropic(
            api_key.into(),
            "claude-embedding-v1".to_string(),
        ));
        self
    }

    /// Set the model name.
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        if let Some(ref mut config) = self.config {
            config.model = model.into();
        }
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        if let Some(ref mut config) = self.config {
            config.batch_size = batch_size;
        }
        self
    }

    /// Enable or disable caching.
    pub fn enable_cache(mut self, enable: bool) -> Self {
        if let Some(ref mut config) = self.config {
            config.enable_cache = enable;
        }
        self
    }

    /// Set maximum retries.
    pub fn max_retries(mut self, max_retries: usize) -> Self {
        if let Some(ref mut config) = self.config {
            config.max_retries = max_retries;
        }
        self
    }

    /// Set timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        if let Some(ref mut config) = self.config {
            config.timeout = timeout;
        }
        self
    }

    /// Build the API embedder.
    pub async fn build(self) -> Result<ApiEmbedder> {
        let config = self
            .config
            .ok_or_else(|| ApiEmbedderError::configuration("No provider configured"))?;

        ApiEmbedder::from_config(config).await
    }
}

impl Default for ApiEmbedderBuilder {
    fn default() -> Self {
        Self::new()
    }
}
