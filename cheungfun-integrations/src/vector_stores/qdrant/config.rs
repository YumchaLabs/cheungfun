//! Qdrant configuration management.
//!
//! This module provides configuration structures and builder patterns for
//! configuring QdrantVectorStore instances.

use cheungfun_core::traits::DistanceMetric;
use std::time::Duration;

/// Configuration for QdrantVectorStore.
///
/// This struct contains all the necessary configuration parameters for
/// connecting to and using a Qdrant vector database instance.
///
/// # Examples
///
/// ```rust
/// use cheungfun_integrations::vector_stores::qdrant::QdrantConfig;
/// use cheungfun_core::traits::DistanceMetric;
/// use std::time::Duration;
///
/// let config = QdrantConfig::new("http://localhost:6334", "my_collection", 384)
///     .with_api_key("your_api_key")
///     .with_distance_metric(DistanceMetric::Cosine)
///     .with_timeout(Duration::from_secs(30))
///     .with_max_retries(3)
///     .with_create_collection_if_missing(true);
/// ```
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    /// Qdrant server URL (e.g., "http://localhost:6334")
    pub url: String,
    /// Optional API key for authentication
    pub api_key: Option<String>,
    /// Collection name to use
    pub collection_name: String,
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric for similarity calculation
    pub distance_metric: DistanceMetric,
    /// Request timeout
    pub timeout: Duration,
    /// Maximum number of retries for failed requests
    pub max_retries: usize,
    /// Whether to create collection if it doesn't exist
    pub create_collection_if_missing: bool,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_string(),
            api_key: None,
            collection_name: "cheungfun_vectors".to_string(),
            dimension: 384,
            distance_metric: DistanceMetric::Cosine,
            timeout: Duration::from_secs(30),
            max_retries: 3,
            create_collection_if_missing: true,
        }
    }
}

impl QdrantConfig {
    /// Create a new Qdrant configuration.
    ///
    /// # Arguments
    ///
    /// * `url` - Qdrant server URL (e.g., "http://localhost:6334")
    /// * `collection_name` - Name of the collection to use
    /// * `dimension` - Vector dimension for embeddings
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cheungfun_integrations::vector_stores::qdrant::QdrantConfig;
    ///
    /// let config = QdrantConfig::new("http://localhost:6334", "my_collection", 384);
    /// ```
    pub fn new(
        url: impl Into<String>,
        collection_name: impl Into<String>,
        dimension: usize,
    ) -> Self {
        Self {
            url: url.into(),
            collection_name: collection_name.into(),
            dimension,
            ..Default::default()
        }
    }

    /// Set the API key for authentication.
    ///
    /// # Arguments
    ///
    /// * `api_key` - API key for Qdrant authentication
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cheungfun_integrations::vector_stores::qdrant::QdrantConfig;
    ///
    /// let config = QdrantConfig::new("http://localhost:6334", "my_collection", 384)
    ///     .with_api_key("your_api_key");
    /// ```
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the distance metric.
    ///
    /// # Arguments
    ///
    /// * `metric` - Distance metric to use for similarity calculations
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cheungfun_integrations::vector_stores::qdrant::QdrantConfig;
    /// use cheungfun_core::traits::DistanceMetric;
    ///
    /// let config = QdrantConfig::new("http://localhost:6334", "my_collection", 384)
    ///     .with_distance_metric(DistanceMetric::Euclidean);
    /// ```
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set the request timeout.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait for requests
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cheungfun_integrations::vector_stores::qdrant::QdrantConfig;
    /// use std::time::Duration;
    ///
    /// let config = QdrantConfig::new("http://localhost:6334", "my_collection", 384)
    ///     .with_timeout(Duration::from_secs(60));
    /// ```
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the maximum number of retries.
    ///
    /// # Arguments
    ///
    /// * `max_retries` - Maximum number of retry attempts for failed requests
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cheungfun_integrations::vector_stores::qdrant::QdrantConfig;
    ///
    /// let config = QdrantConfig::new("http://localhost:6334", "my_collection", 384)
    ///     .with_max_retries(5);
    /// ```
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set whether to create collection if missing.
    ///
    /// # Arguments
    ///
    /// * `create` - Whether to automatically create the collection if it doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cheungfun_integrations::vector_stores::qdrant::QdrantConfig;
    ///
    /// let config = QdrantConfig::new("http://localhost:6334", "my_collection", 384)
    ///     .with_create_collection_if_missing(false);
    /// ```
    pub fn with_create_collection_if_missing(mut self, create: bool) -> Self {
        self.create_collection_if_missing = create;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qdrant_config_creation() {
        let config = QdrantConfig::new("http://localhost:6334", "test_collection", 384);

        assert_eq!(config.url, "http://localhost:6334");
        assert_eq!(config.collection_name, "test_collection");
        assert_eq!(config.dimension, 384);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert!(config.create_collection_if_missing);
    }

    #[test]
    fn test_qdrant_config_builder() {
        let config = QdrantConfig::new("http://localhost:6334", "test", 512)
            .with_api_key("test_key")
            .with_distance_metric(DistanceMetric::Euclidean)
            .with_timeout(Duration::from_secs(60))
            .with_max_retries(5)
            .with_create_collection_if_missing(false);

        assert_eq!(config.api_key, Some("test_key".to_string()));
        assert_eq!(config.distance_metric, DistanceMetric::Euclidean);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.max_retries, 5);
        assert!(!config.create_collection_if_missing);
    }

    #[test]
    fn test_qdrant_config_default() {
        let config = QdrantConfig::default();

        assert_eq!(config.url, "http://localhost:6334");
        assert_eq!(config.collection_name, "cheungfun_vectors");
        assert_eq!(config.dimension, 384);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_retries, 3);
        assert!(config.create_collection_if_missing);
    }
}
