//! Configuration extensions for enhanced functionality.
//!
//! This module provides extensions to existing configuration structures
//! to support JSON loading, validation, and backward compatibility.

use crate::{CheungfunError, Result};
#[cfg(feature = "config-manager")]
use crate::config::ConfigManager;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Trait for configuration structures that can be loaded from JSON.
pub trait JsonConfigurable: Sized + for<'de> Deserialize<'de> + Serialize {
    /// Load configuration from a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the JSON configuration file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or contains invalid JSON.
    async fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = tokio::fs::read_to_string(path.as_ref())
            .await
            .map_err(|e| CheungfunError::Configuration {
                message: format!(
                    "Failed to read configuration file {}: {}",
                    path.as_ref().display(),
                    e
                ),
            })?;

        Self::from_json_str(&content)
    }

    /// Load configuration from a JSON string.
    ///
    /// # Arguments
    ///
    /// * `json` - JSON string containing the configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is invalid or cannot be deserialized.
    fn from_json_str(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| CheungfunError::Configuration {
            message: format!("Failed to parse JSON configuration: {}", e),
        })
    }

    /// Save configuration to a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where to save the configuration file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    async fn to_json_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = self.to_json_string()?;
        tokio::fs::write(path.as_ref(), json)
            .await
            .map_err(|e| CheungfunError::Configuration {
                message: format!(
                    "Failed to write configuration file {}: {}",
                    path.as_ref().display(),
                    e
                ),
            })
    }

    /// Convert configuration to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| CheungfunError::Configuration {
            message: format!("Failed to serialize configuration to JSON: {}", e),
        })
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    fn validate(&self) -> Result<()> {
        // Default implementation does no validation
        Ok(())
    }

    /// Merge with another configuration of the same type.
    ///
    /// The other configuration takes precedence for conflicting values.
    fn merge(&mut self, other: Self) -> Result<()>;
}

/// Trait for configuration structures that can be loaded from a ConfigManager.
#[cfg(feature = "config-manager")]
pub trait ManagedConfigurable: Sized {
    /// Load configuration from a ConfigManager namespace.
    ///
    /// # Arguments
    ///
    /// * `manager` - The configuration manager
    /// * `namespace` - The configuration namespace to load from
    ///
    /// # Errors
    ///
    /// Returns an error if the namespace is not found or deserialization fails.
    fn from_manager(manager: &ConfigManager, namespace: &str) -> Result<Self>;

    /// Update configuration in a ConfigManager.
    ///
    /// # Arguments
    ///
    /// * `manager` - The configuration manager
    /// * `namespace` - The configuration namespace to update
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration cannot be serialized or stored.
    fn update_manager(&self, manager: &ConfigManager, namespace: &str) -> Result<()>;
}

// Note: Implementations for specific config types would go here
// but are currently disabled due to compilation issues

/// Helper function to load any JsonConfigurable from a file with validation.
pub async fn load_and_validate_config<T, P>(path: P) -> Result<T>
where
    T: JsonConfigurable,
    P: AsRef<Path>,
{
    let config = T::from_json_file(path).await?;
    config.validate()?;
    Ok(config)
}
