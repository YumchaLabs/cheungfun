//! Configuration for vector storage backends.
//!
//! This module provides configuration structures for different vector
//! database implementations including in-memory stores, Qdrant, and
//! other vector databases.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{Result, traits::DistanceMetric};

/// Configuration for vector storage backends.
///
/// This enum supports different types of vector stores including
/// in-memory stores for development, Qdrant for production, and
/// custom implementations.
///
/// # Examples
///
/// ```rust
/// use cheungfun_core::config::VectorStoreConfig;
/// use cheungfun_core::traits::DistanceMetric;
/// use std::collections::HashMap;
///
/// // In-memory store for development
/// let memory_config = VectorStoreConfig::Memory {
///     dimension: 768,
///     distance_metric: DistanceMetric::Cosine,
///     capacity: Some(10000),
/// };
///
/// // Qdrant configuration
/// let qdrant_config = VectorStoreConfig::Qdrant {
///     url: "http://localhost:6333".to_string(),
///     collection_name: "documents".to_string(),
///     dimension: 768,
///     distance_metric: DistanceMetric::Cosine,
///     api_key: None,
///     additional_config: HashMap::new(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VectorStoreConfig {
    /// In-memory vector store for development and testing.
    Memory {
        /// Dimension of the vectors.
        dimension: usize,

        /// Distance metric to use for similarity.
        distance_metric: DistanceMetric,

        /// Maximum number of vectors to store (optional).
        capacity: Option<usize>,
    },

    /// Qdrant vector database.
    Qdrant {
        /// Qdrant server URL.
        url: String,

        /// Collection name to use.
        collection_name: String,

        /// Dimension of the vectors.
        dimension: usize,

        /// Distance metric to use.
        distance_metric: DistanceMetric,

        /// API key for authentication (optional).
        api_key: Option<String>,

        /// Additional Qdrant-specific configuration.
        additional_config: HashMap<String, serde_json::Value>,
    },

    /// Custom vector store implementation.
    Custom {
        /// Implementation identifier.
        implementation: String,

        /// Custom configuration parameters.
        config: HashMap<String, serde_json::Value>,
    },
}

impl VectorStoreConfig {
    /// Create a new in-memory vector store configuration.
    #[must_use]
    pub fn memory(dimension: usize) -> Self {
        Self::Memory {
            dimension,
            distance_metric: DistanceMetric::Cosine,
            capacity: None,
        }
    }

    /// Create a new Qdrant vector store configuration.
    pub fn qdrant<S: Into<String>>(url: S, collection_name: S, dimension: usize) -> Self {
        Self::Qdrant {
            url: url.into(),
            collection_name: collection_name.into(),
            dimension,
            distance_metric: DistanceMetric::Cosine,
            api_key: None,
            additional_config: HashMap::new(),
        }
    }

    /// Create a new custom vector store configuration.
    pub fn custom<S: Into<String>>(implementation: S) -> Self {
        Self::Custom {
            implementation: implementation.into(),
            config: HashMap::new(),
        }
    }

    /// Set the distance metric.
    #[must_use]
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        match &mut self {
            Self::Memory {
                distance_metric, ..
            } => *distance_metric = metric,
            Self::Qdrant {
                distance_metric, ..
            } => *distance_metric = metric,
            Self::Custom { config, .. } => {
                config.insert(
                    "distance_metric".to_string(),
                    serde_json::to_value(&metric).unwrap_or_default(),
                );
            }
        }
        self
    }

    /// Set capacity for memory store.
    #[must_use]
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        match &mut self {
            Self::Memory { capacity: cap, .. } => *cap = Some(capacity),
            Self::Custom { config, .. } => {
                config.insert("capacity".to_string(), capacity.into());
            }
            _ => {} // Ignore for other types
        }
        self
    }

    /// Set API key for Qdrant.
    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        if let Self::Qdrant { api_key: key, .. } = &mut self {
            *key = Some(api_key.into());
        }
        self
    }

    /// Add additional configuration parameter.
    pub fn with_config<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        let config = match &mut self {
            Self::Qdrant {
                additional_config, ..
            } => additional_config,
            Self::Custom { config, .. } => config,
            _ => return self, // Memory store doesn't support additional config
        };
        config.insert(key.into(), value.into());
        self
    }

    /// Get the vector dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        match self {
            Self::Memory { dimension, .. } => *dimension,
            Self::Qdrant { dimension, .. } => *dimension,
            Self::Custom { config, .. } => config
                .get("dimension")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(768) as usize,
        }
    }

    /// Get the distance metric.
    #[must_use]
    pub fn distance_metric(&self) -> &DistanceMetric {
        match self {
            Self::Memory {
                distance_metric, ..
            } => distance_metric,
            Self::Qdrant {
                distance_metric, ..
            } => distance_metric,
            Self::Custom { .. } => &DistanceMetric::Cosine, // Default for custom
        }
    }

    /// Get the store type name.
    #[must_use]
    pub fn store_type(&self) -> &str {
        match self {
            Self::Memory { .. } => "memory",
            Self::Qdrant { .. } => "qdrant",
            Self::Custom { .. } => "custom",
        }
    }

    /// Check if this is a local store.
    #[must_use]
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Memory { .. } | Self::Custom { .. })
    }

    /// Check if this is a remote store.
    #[must_use]
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Qdrant { .. })
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Memory {
                dimension,
                capacity,
                ..
            } => {
                if *dimension == 0 {
                    return Err(crate::CheungfunError::configuration(
                        "Dimension must be greater than 0",
                    ));
                }
                if let Some(cap) = capacity {
                    if *cap == 0 {
                        return Err(crate::CheungfunError::configuration(
                            "Capacity must be greater than 0",
                        ));
                    }
                }
            }
            Self::Qdrant {
                url,
                collection_name,
                dimension,
                ..
            } => {
                if url.is_empty() {
                    return Err(crate::CheungfunError::configuration("URL cannot be empty"));
                }
                if collection_name.is_empty() {
                    return Err(crate::CheungfunError::configuration(
                        "Collection name cannot be empty",
                    ));
                }
                if *dimension == 0 {
                    return Err(crate::CheungfunError::configuration(
                        "Dimension must be greater than 0",
                    ));
                }
                // Basic URL validation
                if !url.starts_with("http://") && !url.starts_with("https://") {
                    return Err(crate::CheungfunError::configuration(
                        "URL must start with http:// or https://",
                    ));
                }
            }
            Self::Custom { implementation, .. } => {
                if implementation.is_empty() {
                    return Err(crate::CheungfunError::configuration(
                        "Implementation cannot be empty",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Get connection information for logging (without sensitive data).
    #[must_use]
    pub fn connection_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        match self {
            Self::Memory {
                dimension,
                capacity,
                ..
            } => {
                info.insert("type".to_string(), "memory".to_string());
                info.insert("dimension".to_string(), dimension.to_string());
                if let Some(cap) = capacity {
                    info.insert("capacity".to_string(), cap.to_string());
                }
            }
            Self::Qdrant {
                url,
                collection_name,
                dimension,
                ..
            } => {
                info.insert("type".to_string(), "qdrant".to_string());
                info.insert("url".to_string(), url.clone());
                info.insert("collection".to_string(), collection_name.clone());
                info.insert("dimension".to_string(), dimension.to_string());
                // Don't include API key in connection info
            }
            Self::Custom { implementation, .. } => {
                info.insert("type".to_string(), "custom".to_string());
                info.insert("implementation".to_string(), implementation.clone());
            }
        }
        info
    }
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self::memory(768)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_config() {
        let config = VectorStoreConfig::memory(512)
            .with_distance_metric(DistanceMetric::Euclidean)
            .with_capacity(5000);

        assert_eq!(config.dimension(), 512);
        assert_eq!(config.distance_metric(), &DistanceMetric::Euclidean);
        assert_eq!(config.store_type(), "memory");
        assert!(config.is_local());
        assert!(!config.is_remote());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_qdrant_config() {
        let config = VectorStoreConfig::qdrant("http://localhost:6333", "test_collection", 768)
            .with_distance_metric(DistanceMetric::DotProduct)
            .with_api_key("test-key")
            .with_config("timeout", 30);

        assert_eq!(config.dimension(), 768);
        assert_eq!(config.distance_metric(), &DistanceMetric::DotProduct);
        assert_eq!(config.store_type(), "qdrant");
        assert!(!config.is_local());
        assert!(config.is_remote());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_custom_config() {
        let config = VectorStoreConfig::custom("my-vector-store")
            .with_config("dimension", 1024)
            .with_config("endpoint", "http://localhost:8080");

        assert_eq!(config.dimension(), 1024);
        assert_eq!(config.store_type(), "custom");
        assert!(config.is_local());
        assert!(!config.is_remote());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_errors() {
        let invalid_memory = VectorStoreConfig::memory(0);
        assert!(invalid_memory.validate().is_err());

        let invalid_qdrant = VectorStoreConfig::qdrant("", "collection", 768);
        assert!(invalid_qdrant.validate().is_err());

        let invalid_url = VectorStoreConfig::qdrant("localhost:6333", "collection", 768);
        assert!(invalid_url.validate().is_err());

        let invalid_custom = VectorStoreConfig::custom("");
        assert!(invalid_custom.validate().is_err());
    }

    #[test]
    fn test_connection_info() {
        let config = VectorStoreConfig::qdrant("http://localhost:6333", "test", 768)
            .with_api_key("secret-key");

        let info = config.connection_info();
        assert_eq!(info.get("type"), Some(&"qdrant".to_string()));
        assert_eq!(info.get("url"), Some(&"http://localhost:6333".to_string()));
        assert_eq!(info.get("collection"), Some(&"test".to_string()));
        assert!(!info.contains_key("api_key")); // Should not include sensitive data
    }

    #[test]
    fn test_serialization() {
        let config = VectorStoreConfig::qdrant("http://localhost:6333", "test", 768);
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: VectorStoreConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }
}
