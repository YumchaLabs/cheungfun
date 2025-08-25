//! Qdrant client management and connection handling.
//!
//! This module provides utilities for creating and managing Qdrant client connections,
//! including collection management and health checks.

use cheungfun_core::{traits::DistanceMetric, Result};
use qdrant_client::{
    qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder},
    Qdrant, QdrantError,
};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use super::{config::QdrantConfig, error::map_qdrant_error};

/// Qdrant client wrapper with connection management.
///
/// This struct provides a higher-level interface for managing Qdrant connections
/// and performing common operations like collection management.
pub struct QdrantClient {
    /// The underlying Qdrant client
    client: Arc<Qdrant>,
    /// Configuration used for this client
    config: QdrantConfig,
}

impl std::fmt::Debug for QdrantClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantClient")
            .field("config", &self.config)
            .finish()
    }
}

impl QdrantClient {
    /// Create a new QdrantClient.
    ///
    /// This will establish a connection to the Qdrant server using the provided configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Qdrant configuration
    ///
    /// # Returns
    ///
    /// A Result containing the QdrantClient or an error
    ///
    /// # Errors
    ///
    /// Returns an error if the connection to Qdrant fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cheungfun_integrations::vector_stores::qdrant::{QdrantClient, QdrantConfig};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = QdrantConfig::new("http://localhost:6334", "my_collection", 384);
    /// let client = QdrantClient::new(config).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(config: QdrantConfig) -> Result<Self> {
        info!(
            "Creating QdrantClient with URL: {}, collection: {}",
            config.url, config.collection_name
        );

        // Build Qdrant client
        let mut client_builder = Qdrant::from_url(&config.url);

        if let Some(api_key) = &config.api_key {
            client_builder = client_builder.api_key(api_key.clone());
        }

        let client = client_builder
            .timeout(config.timeout)
            .build()
            .map_err(|e| {
                error!("Failed to create Qdrant client: {}", e);
                map_qdrant_error(e)
            })?;

        let client = Arc::new(client);

        let qdrant_client = Self { client, config };

        // Test the connection
        qdrant_client.health_check().await?;

        info!("QdrantClient created successfully");
        Ok(qdrant_client)
    }

    /// Get the underlying Qdrant client.
    ///
    /// This provides access to the raw Qdrant client for advanced operations
    /// not covered by the wrapper methods.
    pub fn client(&self) -> &Qdrant {
        &self.client
    }

    /// Get the configuration.
    pub fn config(&self) -> &QdrantConfig {
        &self.config
    }

    /// Perform a health check on the Qdrant connection.
    ///
    /// # Returns
    ///
    /// A Result indicating whether the health check passed
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use cheungfun_integrations::vector_stores::qdrant::{QdrantClient, QdrantConfig};
    /// # async fn example(client: QdrantClient) -> Result<(), Box<dyn std::error::Error>> {
    /// client.health_check().await?;
    /// println!("Qdrant is healthy!");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn health_check(&self) -> Result<()> {
        debug!("Performing health check for Qdrant");

        self.client.health_check().await.map_err(|e| {
            error!("Qdrant health check failed: {}", e);
            map_qdrant_error(e)
        })?;

        debug!("Qdrant health check passed");
        Ok(())
    }

    /// Ensure the collection exists, creating it if necessary.
    ///
    /// This method checks if the configured collection exists and creates it
    /// if it doesn't exist and the configuration allows it.
    ///
    /// # Returns
    ///
    /// A Result indicating whether the collection exists or was created successfully
    pub async fn ensure_collection_exists(&self) -> Result<()> {
        debug!(
            "Checking if collection '{}' exists",
            self.config.collection_name
        );

        // Try to get collection info
        match self
            .client
            .collection_info(&self.config.collection_name)
            .await
        {
            Ok(_) => {
                debug!(
                    "Collection '{}' already exists",
                    self.config.collection_name
                );
                Ok(())
            }
            Err(QdrantError::ResponseError { status })
                if status.message().contains("not found") =>
            {
                if self.config.create_collection_if_missing {
                    info!(
                        "Collection '{}' not found, creating it",
                        self.config.collection_name
                    );
                    self.create_collection().await
                } else {
                    Err(cheungfun_core::CheungfunError::NotFound {
                        resource: format!("Qdrant collection '{}'", self.config.collection_name),
                    })
                }
            }
            Err(e) => {
                error!("Failed to check collection existence: {}", e);
                Err(map_qdrant_error(e))
            }
        }
    }

    /// Create a new collection with the configured parameters.
    ///
    /// This method creates a new collection using the vector dimension and
    /// distance metric specified in the configuration.
    ///
    /// # Returns
    ///
    /// A Result indicating whether the collection was created successfully
    pub async fn create_collection(&self) -> Result<()> {
        let distance = match self.config.distance_metric {
            DistanceMetric::Cosine => Distance::Cosine,
            DistanceMetric::Euclidean => Distance::Euclid,
            DistanceMetric::DotProduct => Distance::Dot,
            DistanceMetric::Manhattan => Distance::Manhattan,
            DistanceMetric::Custom(_) => {
                warn!("Custom distance metric not supported by Qdrant, using Cosine");
                Distance::Cosine
            }
        };

        let create_collection =
            CreateCollectionBuilder::new(&self.config.collection_name).vectors_config(
                VectorParamsBuilder::new(self.config.dimension as u64, distance),
            );

        self.client
            .create_collection(create_collection)
            .await
            .map_err(|e| {
                error!("Failed to create collection: {}", e);
                map_qdrant_error(e)
            })?;

        info!(
            "Collection '{}' created successfully",
            self.config.collection_name
        );
        Ok(())
    }

    /// Delete the configured collection.
    ///
    /// This method deletes the entire collection and all its data.
    /// Use with caution as this operation is irreversible.
    ///
    /// # Returns
    ///
    /// A Result indicating whether the collection was deleted successfully
    pub async fn delete_collection(&self) -> Result<()> {
        debug!("Deleting collection '{}'", self.config.collection_name);

        self.client
            .delete_collection(&self.config.collection_name)
            .await
            .map_err(|e| {
                error!("Failed to delete collection: {}", e);
                map_qdrant_error(e)
            })?;

        info!(
            "Collection '{}' deleted successfully",
            self.config.collection_name
        );
        Ok(())
    }

    /// Get information about the configured collection.
    ///
    /// # Returns
    ///
    /// A Result containing the collection information
    pub async fn collection_info(&self) -> Result<qdrant_client::qdrant::CollectionInfo> {
        self.client
            .collection_info(&self.config.collection_name)
            .await
            .map(|response| response.result.unwrap())
            .map_err(|e| {
                error!("Failed to get collection info: {}", e);
                map_qdrant_error(e)
            })
    }

    /// Check if the configured collection exists.
    ///
    /// # Returns
    ///
    /// A Result containing a boolean indicating whether the collection exists
    pub async fn collection_exists(&self) -> Result<bool> {
        match self
            .client
            .collection_info(&self.config.collection_name)
            .await
        {
            Ok(_) => Ok(true),
            Err(QdrantError::ResponseError { status })
                if status.message().contains("not found") =>
            {
                Ok(false)
            }
            Err(e) => {
                error!("Failed to check collection existence: {}", e);
                Err(map_qdrant_error(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qdrant_client_debug() {
        let config = QdrantConfig::new("http://localhost:6334", "test", 384);
        let client = Qdrant::from_url("http://localhost:6334").build().unwrap();
        let qdrant_client = QdrantClient {
            client: Arc::new(client),
            config,
        };

        let debug_str = format!("{:?}", qdrant_client);
        assert!(debug_str.contains("QdrantClient"));
    }

    // Integration tests would require a running Qdrant instance
    // These are commented out but show the structure for integration testing

    /*
    #[tokio::test]
    #[ignore] // Requires running Qdrant instance
    async fn test_client_creation_and_health_check() {
        let config = QdrantConfig::new("http://localhost:6334", "test_client", 384);
        let client = QdrantClient::new(config).await.unwrap();

        // Health check should pass
        client.health_check().await.unwrap();
    }

    #[tokio::test]
    #[ignore] // Requires running Qdrant instance
    async fn test_collection_management() {
        let config = QdrantConfig::new("http://localhost:6334", "test_collection_mgmt", 384);
        let client = QdrantClient::new(config).await.unwrap();

        // Collection should not exist initially
        assert!(!client.collection_exists().await.unwrap());

        // Create collection
        client.create_collection().await.unwrap();
        assert!(client.collection_exists().await.unwrap());

        // Get collection info
        let info = client.collection_info().await.unwrap();
        assert_eq!(info.collection_name, "test_collection_mgmt");

        // Clean up
        client.delete_collection().await.unwrap();
        assert!(!client.collection_exists().await.unwrap());
    }
    */
}
