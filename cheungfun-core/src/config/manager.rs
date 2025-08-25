//! Configuration management with hot reloading and JSON support.
//!
//! This module provides a unified configuration manager that supports:
//! - JSON configuration file loading
//! - Hot reloading with file system watching
//! - Environment variable substitution
//! - Configuration validation
//! - Backward compatibility with existing config APIs

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::{CheungfunError, Result};

/// Unified configuration manager with hot reloading support.
///
/// This manager provides centralized configuration management for all
/// Cheungfun components, supporting JSON files, environment variables,
/// and automatic reloading when configuration files change.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::config::ConfigManager;
/// use std::path::Path;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut manager = ConfigManager::new();
///     
///     // Load configuration from directory
///     manager.load_from_directory(Path::new("./config")).await?;
///     
///     // Enable hot reloading
///     manager.enable_hot_reload().await?;
///     
///     // Get configuration values
///     let db_url = manager.get_string("database.url")?;
///     let max_connections = manager.get_u32("database.max_connections")?;
///     
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct ConfigManager {
    /// Current configuration state.
    config: Arc<RwLock<ConfigState>>,

    /// File system watcher for hot reloading.
    #[cfg(feature = "hot-reload")]
    watcher: Option<notify::RecommendedWatcher>,

    /// Configuration directory being watched.
    config_dir: Option<PathBuf>,

    /// Hot reload enabled flag.
    hot_reload_enabled: bool,
}

/// Internal configuration state.
#[derive(Debug, Clone, Default)]
struct ConfigState {
    /// Raw configuration data from all sources.
    data: HashMap<String, serde_json::Value>,

    /// Configuration file metadata.
    file_metadata: HashMap<PathBuf, FileMetadata>,

    /// Environment variable overrides.
    env_overrides: HashMap<String, String>,
}

/// Metadata about a configuration file.
#[derive(Debug, Clone)]
struct FileMetadata {
    /// Last modification time.
    last_modified: std::time::SystemTime,

    /// File size in bytes.
    size: u64,

    /// Configuration keys loaded from this file.
    keys: Vec<String>,
}

/// Configuration change event.
#[derive(Debug, Clone)]
pub struct ConfigChangeEvent {
    /// The configuration key that changed.
    pub key: String,

    /// The old value (if any).
    pub old_value: Option<serde_json::Value>,

    /// The new value.
    pub new_value: serde_json::Value,

    /// The source file that triggered the change.
    pub source_file: Option<PathBuf>,
}

/// Configuration validation error.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// The configuration key that failed validation.
    pub key: String,

    /// The validation error message.
    pub message: String,

    /// The invalid value.
    pub value: serde_json::Value,
}

impl ConfigManager {
    /// Create a new configuration manager.
    pub fn new() -> Self {
        Self {
            config: Arc::new(RwLock::new(ConfigState::default())),
            #[cfg(feature = "hot-reload")]
            watcher: None,
            config_dir: None,
            hot_reload_enabled: false,
        }
    }

    /// Load configuration from a directory containing JSON files.
    ///
    /// This method scans the directory for `.json` files and loads them
    /// into the configuration state. File names become configuration namespaces.
    ///
    /// # Arguments
    ///
    /// * `config_dir` - Directory containing configuration files
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read or if any JSON files
    /// contain invalid syntax.
    pub async fn load_from_directory<P: AsRef<Path>>(&mut self, config_dir: P) -> Result<()> {
        let config_dir = config_dir.as_ref();
        info!(
            "Loading configuration from directory: {}",
            config_dir.display()
        );

        if !config_dir.exists() {
            return Err(CheungfunError::Configuration {
                message: format!(
                    "Configuration directory does not exist: {}",
                    config_dir.display()
                ),
            });
        }

        let mut entries =
            tokio::fs::read_dir(config_dir)
                .await
                .map_err(|e| CheungfunError::Configuration {
                    message: format!("Failed to read configuration directory: {}", e),
                })?;

        let mut loaded_files = 0;
        while let Some(entry) =
            entries
                .next_entry()
                .await
                .map_err(|e| CheungfunError::Configuration {
                    message: format!("Failed to read directory entry: {}", e),
                })?
        {
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                self.load_json_file(&path).await?;
                loaded_files += 1;
            }
        }

        self.config_dir = Some(config_dir.to_path_buf());
        info!(
            "Loaded {} configuration files from {}",
            loaded_files,
            config_dir.display()
        );

        Ok(())
    }

    /// Load a single JSON configuration file.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the JSON configuration file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or contains invalid JSON.
    pub async fn load_json_file<P: AsRef<Path>>(&self, file_path: P) -> Result<()> {
        let file_path = file_path.as_ref();
        debug!("Loading configuration file: {}", file_path.display());

        let content = tokio::fs::read_to_string(file_path).await.map_err(|e| {
            CheungfunError::Configuration {
                message: format!(
                    "Failed to read configuration file {}: {}",
                    file_path.display(),
                    e
                ),
            }
        })?;

        // Substitute environment variables
        let content = self.substitute_env_variables(&content)?;

        let json_value: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| CheungfunError::Configuration {
                message: format!(
                    "Invalid JSON in configuration file {}: {}",
                    file_path.display(),
                    e
                ),
            })?;

        // Get file metadata
        let metadata =
            tokio::fs::metadata(file_path)
                .await
                .map_err(|e| CheungfunError::Configuration {
                    message: format!("Failed to get metadata for {}: {}", file_path.display(), e),
                })?;

        let file_metadata = FileMetadata {
            last_modified: metadata.modified().unwrap_or(std::time::UNIX_EPOCH),
            size: metadata.len(),
            keys: Vec::new(), // Will be populated when we flatten the JSON
        };

        // Use filename (without extension) as namespace
        let namespace = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("default");

        // Update configuration state
        {
            let mut config = self.config.write().unwrap();
            config.data.insert(namespace.to_string(), json_value);
            config
                .file_metadata
                .insert(file_path.to_path_buf(), file_metadata);
        }

        debug!(
            "Loaded configuration namespace '{}' from {}",
            namespace,
            file_path.display()
        );
        Ok(())
    }

    /// Substitute environment variables in configuration content.
    ///
    /// Supports the format `${VAR_NAME}` and `${VAR_NAME:default_value}`.
    fn substitute_env_variables(&self, content: &str) -> Result<String> {
        let mut result = content.to_string();

        // Simple regex-based substitution for ${VAR_NAME} and ${VAR_NAME:default}
        let env_var_regex = regex::Regex::new(r"\$\{([^}:]+)(?::([^}]*))?\}").unwrap();

        for captures in env_var_regex.captures_iter(content) {
            let full_match = captures.get(0).unwrap().as_str();
            let var_name = captures.get(1).unwrap().as_str();
            let default_value = captures.get(2).map(|m| m.as_str()).unwrap_or("");

            let value = std::env::var(var_name).unwrap_or_else(|_| default_value.to_string());
            result = result.replace(full_match, &value);
        }

        Ok(result)
    }

    /// Get a configuration value as a string.
    ///
    /// # Arguments
    ///
    /// * `key` - Configuration key in dot notation (e.g., "database.url")
    ///
    /// # Errors
    ///
    /// Returns an error if the key is not found or cannot be converted to a string.
    pub fn get_string(&self, key: &str) -> Result<String> {
        let value = self.get_value(key)?;

        match value {
            serde_json::Value::String(s) => Ok(s),
            serde_json::Value::Number(n) => Ok(n.to_string()),
            serde_json::Value::Bool(b) => Ok(b.to_string()),
            _ => Err(CheungfunError::Configuration {
                message: format!("Configuration key '{}' cannot be converted to string", key),
            }),
        }
    }

    /// Get a configuration value as a u32.
    pub fn get_u32(&self, key: &str) -> Result<u32> {
        let value = self.get_value(key)?;

        match value {
            serde_json::Value::Number(n) => n
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| CheungfunError::Configuration {
                    message: format!("Configuration key '{}' cannot be converted to u32", key),
                }),
            _ => Err(CheungfunError::Configuration {
                message: format!("Configuration key '{}' is not a number", key),
            }),
        }
    }

    /// Get a configuration value as a boolean.
    pub fn get_bool(&self, key: &str) -> Result<bool> {
        let value = self.get_value(key)?;

        match value {
            serde_json::Value::Bool(b) => Ok(b),
            serde_json::Value::String(s) => match s.to_lowercase().as_str() {
                "true" | "yes" | "1" | "on" => Ok(true),
                "false" | "no" | "0" | "off" => Ok(false),
                _ => Err(CheungfunError::Configuration {
                    message: format!("Configuration key '{}' cannot be converted to boolean", key),
                }),
            },
            _ => Err(CheungfunError::Configuration {
                message: format!("Configuration key '{}' is not a boolean", key),
            }),
        }
    }

    /// Get a raw configuration value.
    fn get_value(&self, key: &str) -> Result<serde_json::Value> {
        let config = self.config.read().unwrap();

        // Check environment variable overrides first
        if let Some(env_value) = config.env_overrides.get(key) {
            return Ok(serde_json::Value::String(env_value.clone()));
        }

        // Parse dot notation key
        let parts: Vec<&str> = key.split('.').collect();
        if parts.is_empty() {
            return Err(CheungfunError::Configuration {
                message: "Empty configuration key".to_string(),
            });
        }

        // First part is the namespace
        let namespace = parts[0];
        let namespace_data =
            config
                .data
                .get(namespace)
                .ok_or_else(|| CheungfunError::Configuration {
                    message: format!("Configuration namespace '{}' not found", namespace),
                })?;

        // Navigate through the remaining parts
        let mut current = namespace_data;
        for part in &parts[1..] {
            current = current
                .get(part)
                .ok_or_else(|| CheungfunError::Configuration {
                    message: format!("Configuration key '{}' not found", key),
                })?;
        }

        Ok(current.clone())
    }

    /// Enable hot reloading of configuration files.
    ///
    /// This method sets up file system watching to automatically reload
    /// configuration when files change.
    ///
    /// # Errors
    ///
    /// Returns an error if file watching cannot be set up.
    #[cfg(feature = "hot-reload")]
    pub async fn enable_hot_reload(&mut self) -> Result<()> {
        use notify::{Event, EventKind, RecursiveMode, Watcher};

        let config_dir = self
            .config_dir
            .as_ref()
            .ok_or_else(|| CheungfunError::Configuration {
                message: "No configuration directory set for hot reload".to_string(),
            })?;

        info!(
            "Enabling hot reload for configuration directory: {}",
            config_dir.display()
        );

        let (tx, mut rx) = mpsc::channel(100);
        let config_clone = Arc::clone(&self.config);
        let config_dir_clone = config_dir.clone();

        // Create file watcher
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            if let Ok(event) = res {
                if let Err(e) = tx.blocking_send(event) {
                    error!("Failed to send file watch event: {}", e);
                }
            }
        })
        .map_err(|e| CheungfunError::Configuration {
            message: format!("Failed to create file watcher: {}", e),
        })?;

        watcher
            .watch(config_dir, RecursiveMode::NonRecursive)
            .map_err(|e| CheungfunError::Configuration {
                message: format!("Failed to watch configuration directory: {}", e),
            })?;

        // Spawn task to handle file change events
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event.kind {
                    EventKind::Modify(_) | EventKind::Create(_) => {
                        for path in event.paths {
                            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                                info!("Configuration file changed: {}", path.display());

                                // Reload the file with a small delay to ensure write is complete
                                tokio::time::sleep(Duration::from_millis(100)).await;

                                if let Err(e) = Self::reload_file(&config_clone, &path).await {
                                    error!(
                                        "Failed to reload configuration file {}: {}",
                                        path.display(),
                                        e
                                    );
                                }
                            }
                        }
                    }
                    _ => {} // Ignore other event types
                }
            }
        });

        self.watcher = Some(watcher);
        self.hot_reload_enabled = true;

        info!("Hot reload enabled for configuration files");
        Ok(())
    }

    /// Reload a specific configuration file.
    #[cfg(feature = "hot-reload")]
    async fn reload_file(config: &Arc<RwLock<ConfigState>>, file_path: &Path) -> Result<()> {
        debug!("Reloading configuration file: {}", file_path.display());

        let content = tokio::fs::read_to_string(file_path).await.map_err(|e| {
            CheungfunError::Configuration {
                message: format!(
                    "Failed to read configuration file {}: {}",
                    file_path.display(),
                    e
                ),
            }
        })?;

        // Simple environment variable substitution
        let content = Self::substitute_env_vars_static(&content);

        let json_value: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| CheungfunError::Configuration {
                message: format!(
                    "Invalid JSON in configuration file {}: {}",
                    file_path.display(),
                    e
                ),
            })?;

        let namespace = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("default");

        // Update configuration state
        {
            let mut config_state = config.write().unwrap();
            config_state.data.insert(namespace.to_string(), json_value);
        }

        info!(
            "Reloaded configuration namespace '{}' from {}",
            namespace,
            file_path.display()
        );
        Ok(())
    }

    /// Static version of environment variable substitution for hot reload.
    #[cfg(feature = "hot-reload")]
    fn substitute_env_vars_static(content: &str) -> String {
        let mut result = content.to_string();

        let env_var_regex = regex::Regex::new(r"\$\{([^}:]+)(?::([^}]*))?\}").unwrap();

        for captures in env_var_regex.captures_iter(content) {
            let full_match = captures.get(0).unwrap().as_str();
            let var_name = captures.get(1).unwrap().as_str();
            let default_value = captures.get(2).map(|m| m.as_str()).unwrap_or("");

            let value = std::env::var(var_name).unwrap_or_else(|_| default_value.to_string());
            result = result.replace(full_match, &value);
        }

        result
    }

    /// Get all configuration keys in a namespace.
    pub fn get_namespace_keys(&self, namespace: &str) -> Result<Vec<String>> {
        let config = self.config.read().unwrap();

        let namespace_data =
            config
                .data
                .get(namespace)
                .ok_or_else(|| CheungfunError::Configuration {
                    message: format!("Configuration namespace '{}' not found", namespace),
                })?;

        let mut keys = Vec::new();
        self.collect_keys(namespace_data, namespace, &mut keys);
        Ok(keys)
    }

    /// Recursively collect all keys from a JSON value.
    fn collect_keys(&self, value: &serde_json::Value, prefix: &str, keys: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, val) in map {
                    let full_key = format!("{}.{}", prefix, key);
                    keys.push(full_key.clone());
                    self.collect_keys(val, &full_key, keys);
                }
            }
            _ => {} // Leaf values are already added
        }
    }

    /// Set an environment variable override.
    ///
    /// Environment overrides take precedence over configuration file values.
    pub fn set_env_override(&self, key: &str, value: &str) {
        let mut config = self.config.write().unwrap();
        config
            .env_overrides
            .insert(key.to_string(), value.to_string());
        debug!("Set environment override: {} = {}", key, value);
    }

    /// Remove an environment variable override.
    pub fn remove_env_override(&self, key: &str) {
        let mut config = self.config.write().unwrap();
        config.env_overrides.remove(key);
        debug!("Removed environment override: {}", key);
    }

    /// Get configuration as a typed struct.
    ///
    /// # Arguments
    ///
    /// * `namespace` - Configuration namespace to deserialize
    ///
    /// # Errors
    ///
    /// Returns an error if the namespace is not found or deserialization fails.
    pub fn get_typed<T>(&self, namespace: &str) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let config = self.config.read().unwrap();

        let namespace_data =
            config
                .data
                .get(namespace)
                .ok_or_else(|| CheungfunError::Configuration {
                    message: format!("Configuration namespace '{}' not found", namespace),
                })?;

        serde_json::from_value(namespace_data.clone()).map_err(|e| CheungfunError::Configuration {
            message: format!(
                "Failed to deserialize configuration namespace '{}': {}",
                namespace, e
            ),
        })
    }

    /// Check if hot reload is enabled.
    pub fn is_hot_reload_enabled(&self) -> bool {
        self.hot_reload_enabled
    }

    /// Get configuration statistics.
    pub fn get_stats(&self) -> ConfigStats {
        let config = self.config.read().unwrap();

        ConfigStats {
            total_namespaces: config.data.len(),
            total_files: config.file_metadata.len(),
            env_overrides: config.env_overrides.len(),
            hot_reload_enabled: self.hot_reload_enabled,
        }
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigStats {
    /// Total number of configuration namespaces.
    pub total_namespaces: usize,

    /// Total number of configuration files loaded.
    pub total_files: usize,

    /// Number of environment variable overrides.
    pub env_overrides: usize,

    /// Whether hot reload is enabled.
    pub hot_reload_enabled: bool,
}
