//! Ingestion cache for caching transformation results.
//!
//! This module provides LlamaIndex-compatible ingestion caching capabilities,
//! allowing transformation results to be cached and reused across pipeline runs.
//! This significantly improves performance for repeated processing of the same content.

use crate::error::{IndexingError, Result as IndexingResult};
use async_trait::async_trait;
use cheungfun_core::types::Node;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    path::Path,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{fs, sync::RwLock};
use tracing::{debug, info, warn};

/// Default cache collection name.
pub const DEFAULT_CACHE_COLLECTION: &str = "llama_cache";

/// Default nodes key in cache entries.
pub const DEFAULT_NODES_KEY: &str = "nodes";

/// Cache entry with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The cached nodes.
    pub nodes: Vec<Node>,
    /// Timestamp when the entry was created.
    pub created_at: u64,
    /// Optional TTL for the entry.
    pub ttl: Option<Duration>,
    /// Metadata about the transformation.
    pub metadata: HashMap<String, String>,
}

impl CacheEntry {
    /// Create a new cache entry.
    pub fn new(nodes: Vec<Node>) -> Self {
        Self {
            nodes,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            ttl: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new cache entry with TTL.
    pub fn with_ttl(nodes: Vec<Node>, ttl: Duration) -> Self {
        Self {
            nodes,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            ttl: Some(ttl),
            metadata: HashMap::new(),
        }
    }

    /// Create a new cache entry with metadata.
    pub fn with_metadata(nodes: Vec<Node>, metadata: HashMap<String, String>) -> Self {
        Self {
            nodes,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            ttl: None,
            metadata,
        }
    }

    /// Check if the entry has expired.
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            now > self.created_at + ttl.as_secs()
        } else {
            false
        }
    }
}

/// Base trait for cache backends.
#[async_trait]
pub trait CacheBackend: Send + Sync + std::fmt::Debug {
    /// Put a value into the cache.
    async fn put(
        &self,
        key: &str,
        entry: CacheEntry,
        collection: Option<&str>,
    ) -> IndexingResult<()>;

    /// Get a value from the cache.
    async fn get(&self, key: &str, collection: Option<&str>) -> IndexingResult<Option<CacheEntry>>;

    /// Delete a value from the cache.
    async fn delete(&self, key: &str, collection: Option<&str>) -> IndexingResult<bool>;

    /// Get all keys in a collection.
    async fn get_all_keys(&self, collection: Option<&str>) -> IndexingResult<Vec<String>>;

    /// Clear all entries in a collection.
    async fn clear(&self, collection: Option<&str>) -> IndexingResult<()>;

    /// Persist the cache to storage (if supported).
    async fn persist(&self, path: &Path) -> IndexingResult<()>;

    /// Load the cache from storage (if supported).
    async fn load(&self, path: &Path) -> IndexingResult<()>;
}

/// Simple in-memory cache backend.
#[derive(Debug, Default)]
pub struct SimpleCacheBackend {
    /// Storage for cache entries organized by collection.
    storage: Arc<RwLock<HashMap<String, HashMap<String, CacheEntry>>>>,
}

impl SimpleCacheBackend {
    /// Create a new simple cache backend.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the effective collection name.
    fn collection_name(&self, collection: Option<&str>) -> String {
        collection.unwrap_or(DEFAULT_CACHE_COLLECTION).to_string()
    }
}

#[async_trait]
impl CacheBackend for SimpleCacheBackend {
    async fn put(
        &self,
        key: &str,
        entry: CacheEntry,
        collection: Option<&str>,
    ) -> IndexingResult<()> {
        let collection_name = self.collection_name(collection);
        let mut storage = self.storage.write().await;

        storage
            .entry(collection_name)
            .or_default()
            .insert(key.to_string(), entry);

        debug!("Cached entry with key: {}", key);
        Ok(())
    }

    async fn get(&self, key: &str, collection: Option<&str>) -> IndexingResult<Option<CacheEntry>> {
        let collection_name = self.collection_name(collection);
        let storage = self.storage.read().await;

        if let Some(collection_data) = storage.get(&collection_name) {
            if let Some(entry) = collection_data.get(key) {
                if entry.is_expired() {
                    debug!("Cache entry expired for key: {}", key);
                    return Ok(None);
                }
                debug!("Cache hit for key: {}", key);
                return Ok(Some(entry.clone()));
            }
        }

        debug!("Cache miss for key: {}", key);
        Ok(None)
    }

    async fn delete(&self, key: &str, collection: Option<&str>) -> IndexingResult<bool> {
        let collection_name = self.collection_name(collection);
        let mut storage = self.storage.write().await;

        if let Some(collection_data) = storage.get_mut(&collection_name) {
            let removed = collection_data.remove(key).is_some();
            debug!("Deleted cache entry for key: {}, existed: {}", key, removed);
            Ok(removed)
        } else {
            Ok(false)
        }
    }

    async fn get_all_keys(&self, collection: Option<&str>) -> IndexingResult<Vec<String>> {
        let collection_name = self.collection_name(collection);
        let storage = self.storage.read().await;

        if let Some(collection_data) = storage.get(&collection_name) {
            Ok(collection_data.keys().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }

    async fn clear(&self, collection: Option<&str>) -> IndexingResult<()> {
        let collection_name = self.collection_name(collection);
        let mut storage = self.storage.write().await;

        if let Some(collection_data) = storage.get_mut(&collection_name) {
            let count = collection_data.len();
            collection_data.clear();
            info!(
                "Cleared {} entries from collection: {}",
                count, collection_name
            );
        }

        Ok(())
    }

    async fn persist(&self, path: &Path) -> IndexingResult<()> {
        let storage = self.storage.read().await;
        let serialized = serde_json::to_string_pretty(&*storage)
            .map_err(|e| IndexingError::processing(format!("Failed to serialize cache: {}", e)))?;

        fs::write(path, serialized)
            .await
            .map_err(|e| IndexingError::io(format!("Failed to write cache file: {}", e)))?;

        info!("Persisted cache to: {}", path.display());
        Ok(())
    }

    async fn load(&self, path: &Path) -> IndexingResult<()> {
        if !path.exists() {
            warn!("Cache file does not exist: {}", path.display());
            return Ok(());
        }

        let content = fs::read_to_string(path)
            .await
            .map_err(|e| IndexingError::io(format!("Failed to read cache file: {}", e)))?;

        let loaded_storage: HashMap<String, HashMap<String, CacheEntry>> =
            serde_json::from_str(&content).map_err(|e| {
                IndexingError::processing(format!("Failed to deserialize cache: {}", e))
            })?;

        let mut storage = self.storage.write().await;
        *storage = loaded_storage;

        info!("Loaded cache from: {}", path.display());
        Ok(())
    }
}

/// Ingestion cache for caching transformation results.
///
/// This cache follows LlamaIndex's IngestionCache design exactly, providing
/// transformation-level caching to improve pipeline performance.
///
/// # Features
///
/// - Transformation-level caching with automatic hash generation
/// - Support for multiple cache collections
/// - Pluggable cache backends (memory, file, Redis, etc.)
/// - Automatic expiration and cleanup
/// - Persistence support for durable caching
/// - Full LlamaIndex compatibility
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::cache::{IngestionCache, SimpleCacheBackend};
/// use cheungfun_core::{Node, traits::{Transform, TransformInput}};
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create cache with simple backend
///     let backend = Arc::new(SimpleCacheBackend::new());
///     let cache = IngestionCache::new(backend);
///
///     // Cache transformation results
///     let nodes = vec![/* your nodes */];
///     let transform_hash = "transformation_hash_123";
///     cache.put(transform_hash, nodes.clone(), None).await?;
///
///     // Retrieve cached results
///     let cached_nodes = cache.get(transform_hash, None).await?;
///     assert_eq!(cached_nodes.unwrap().len(), nodes.len());
///
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct IngestionCache {
    /// The cache backend.
    backend: Arc<dyn CacheBackend>,
    /// Default collection name.
    default_collection: String,
    /// Default TTL for cache entries.
    default_ttl: Option<Duration>,
}

impl IngestionCache {
    /// Create a new ingestion cache.
    pub fn new(backend: Arc<dyn CacheBackend>) -> Self {
        Self {
            backend,
            default_collection: DEFAULT_CACHE_COLLECTION.to_string(),
            default_ttl: None,
        }
    }

    /// Create a new ingestion cache with custom collection.
    pub fn with_collection(backend: Arc<dyn CacheBackend>, collection: String) -> Self {
        Self {
            backend,
            default_collection: collection,
            default_ttl: None,
        }
    }

    /// Create a new ingestion cache with TTL.
    pub fn with_ttl(backend: Arc<dyn CacheBackend>, ttl: Duration) -> Self {
        Self {
            backend,
            default_collection: DEFAULT_CACHE_COLLECTION.to_string(),
            default_ttl: Some(ttl),
        }
    }

    /// Create a simple in-memory cache.
    pub fn simple() -> Self {
        Self::new(Arc::new(SimpleCacheBackend::new()))
    }

    /// Create a simple cache from a persist path.
    pub async fn from_persist_path<P: AsRef<Path>>(path: P) -> IndexingResult<Self> {
        let backend = Arc::new(SimpleCacheBackend::new());
        backend.load(path.as_ref()).await?;
        Ok(Self::new(backend))
    }

    /// Put nodes into the cache.
    ///
    /// # Arguments
    ///
    /// * `key` - The cache key
    /// * `nodes` - The nodes to cache
    /// * `collection` - Optional collection name (uses default if None)
    pub async fn put(
        &self,
        key: &str,
        nodes: Vec<Node>,
        collection: Option<&str>,
    ) -> IndexingResult<()> {
        let collection = collection.unwrap_or(&self.default_collection);

        let entry = if let Some(ttl) = self.default_ttl {
            CacheEntry::with_ttl(nodes, ttl)
        } else {
            CacheEntry::new(nodes)
        };

        self.backend.put(key, entry, Some(collection)).await
    }

    /// Get nodes from the cache.
    ///
    /// # Arguments
    ///
    /// * `key` - The cache key
    /// * `collection` - Optional collection name (uses default if None)
    ///
    /// # Returns
    ///
    /// The cached nodes if found and not expired, None otherwise.
    pub async fn get(
        &self,
        key: &str,
        collection: Option<&str>,
    ) -> IndexingResult<Option<Vec<Node>>> {
        let collection = collection.unwrap_or(&self.default_collection);

        match self.backend.get(key, Some(collection)).await? {
            Some(entry) => {
                if entry.is_expired() {
                    // Clean up expired entry
                    let _ = self.backend.delete(key, Some(collection)).await;
                    Ok(None)
                } else {
                    Ok(Some(entry.nodes))
                }
            }
            None => Ok(None),
        }
    }

    /// Delete an entry from the cache.
    pub async fn delete(&self, key: &str, collection: Option<&str>) -> IndexingResult<bool> {
        let collection = collection.unwrap_or(&self.default_collection);
        self.backend.delete(key, Some(collection)).await
    }

    /// Clear all entries in a collection.
    pub async fn clear(&self, collection: Option<&str>) -> IndexingResult<()> {
        let collection = collection.unwrap_or(&self.default_collection);
        self.backend.clear(Some(collection)).await
    }

    /// Get all keys in a collection.
    pub async fn get_all_keys(&self, collection: Option<&str>) -> IndexingResult<Vec<String>> {
        let collection = collection.unwrap_or(&self.default_collection);
        self.backend.get_all_keys(Some(collection)).await
    }

    /// Persist the cache to a file.
    pub async fn persist<P: AsRef<Path>>(&self, path: P) -> IndexingResult<()> {
        self.backend.persist(path.as_ref()).await
    }

    /// Load the cache from a file.
    pub async fn load<P: AsRef<Path>>(&self, path: P) -> IndexingResult<()> {
        self.backend.load(path.as_ref()).await
    }

    /// Get cache statistics.
    pub async fn stats(&self, collection: Option<&str>) -> IndexingResult<CacheStats> {
        let keys = self.get_all_keys(collection).await?;
        let total_entries = keys.len();

        // Count expired entries
        let mut expired_entries = 0;
        let collection_name = collection.unwrap_or(&self.default_collection);

        for key in &keys {
            if let Ok(Some(entry)) = self.backend.get(key, Some(collection_name)).await {
                if entry.is_expired() {
                    expired_entries += 1;
                }
            }
        }

        Ok(CacheStats {
            total_entries,
            expired_entries,
            active_entries: total_entries - expired_entries,
        })
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total number of entries.
    pub total_entries: usize,
    /// Number of expired entries.
    pub expired_entries: usize,
    /// Number of active (non-expired) entries.
    pub active_entries: usize,
}

impl CacheStats {
    /// Calculate the cache efficiency (active / total).
    pub fn efficiency(&self) -> f64 {
        if self.total_entries == 0 {
            1.0
        } else {
            self.active_entries as f64 / self.total_entries as f64
        }
    }
}

/// Transformation hash generator.
///
/// This follows LlamaIndex's hash generation algorithm exactly.
pub struct TransformationHasher;

impl TransformationHasher {
    /// Generate a hash for a transformation applied to nodes.
    ///
    /// This follows LlamaIndex's `get_transformation_hash` function exactly.
    ///
    /// # Arguments
    ///
    /// * `nodes` - The input nodes
    /// * `transform` - The transformation being applied
    ///
    /// # Returns
    ///
    /// A SHA-256 hash string representing the unique combination of nodes and transformation.
    pub fn hash(nodes: &[Node], transform_name: &str) -> String {
        // Concatenate all node content (following LlamaIndex's approach)
        let nodes_str: String = nodes
            .iter()
            .map(|node| node.content.as_str())
            .collect::<Vec<_>>()
            .join("");

        // Combine and hash
        let combined = format!("{}{}", nodes_str, transform_name);
        let mut hasher = Sha256::new();
        hasher.update(combined.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Generate a hash for a transformation with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `nodes` - The input nodes
    /// * `transform_name` - The name of the transformation
    /// * `config` - The transformation configuration
    ///
    /// # Returns
    ///
    /// A SHA-256 hash string.
    pub fn hash_with_config(
        nodes: &[Node],
        transform_name: &str,
        config: &HashMap<String, serde_json::Value>,
    ) -> String {
        // Concatenate all node content
        let nodes_str: String = nodes
            .iter()
            .map(|node| node.content.as_str())
            .collect::<Vec<_>>()
            .join("");

        // Create transformation string
        let transform_string = format!("{}:{:?}", transform_name, config);

        // Combine and hash
        let combined = format!("{}{}", nodes_str, transform_string);
        let mut hasher = Sha256::new();
        hasher.update(combined.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Generate a simple content-based hash.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to hash
    ///
    /// # Returns
    ///
    /// A SHA-256 hash string.
    pub fn content_hash(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}
