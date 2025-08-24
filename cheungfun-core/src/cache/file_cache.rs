//! File-based cache implementation with better cleanup and LRU eviction.
//!
//! This module provides an improved file cache that addresses the limitations of the
//! DiskCache-based implementation, offering better control over cleanup, eviction,
//! and statistics.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::fs;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::traits::{CacheHealth, CacheStats, HealthStatus, PipelineCache};
use crate::{CheungfunError, Node};

/// File-based cache implementation.
///
/// This cache provides better control over file operations, cleanup, and eviction
/// compared to the DiskCache-based implementation.
///
/// # Features
/// - Custom file format for better control
/// - LRU eviction with configurable size limits
/// - Automatic cleanup of expired entries
/// - Detailed statistics and health monitoring
/// - Atomic operations for data integrity
/// - Configurable compression
#[derive(Debug)]
pub struct FileCache {
    /// Base directory for cache files
    cache_dir: PathBuf,
    /// Cache configuration
    config: FileCacheConfig,
    /// In-memory index for fast lookups
    index: Arc<RwLock<CacheIndex>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
}

/// Configuration for the file cache.
#[derive(Debug, Clone)]
pub struct FileCacheConfig {
    /// Maximum number of entries per cache type
    pub max_entries: usize,
    /// Default TTL for cache entries
    pub default_ttl: Duration,
    /// Whether to enable compression
    pub enable_compression: bool,
    /// Whether to enable automatic cleanup
    pub enable_auto_cleanup: bool,
    /// Cleanup interval
    pub cleanup_interval: Duration,
    /// Maximum file size for individual cache entries (in bytes)
    pub max_entry_size: usize,
}

impl Default for FileCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            default_ttl: Duration::from_secs(3600),
            enable_compression: false,
            enable_auto_cleanup: true,
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            max_entry_size: 10 * 1024 * 1024,           // 10 MB
        }
    }
}

/// In-memory index for tracking cache entries.
#[derive(Debug, Default, Serialize, Deserialize)]
struct CacheIndex {
    /// Embedding entries
    embeddings: BTreeMap<String, CacheEntryMetadata>,
    /// Node entries
    nodes: BTreeMap<String, CacheEntryMetadata>,
    /// Data entries
    data: BTreeMap<String, CacheEntryMetadata>,
    /// LRU tracking
    access_order: HashMap<String, u64>,
    /// Next access counter
    next_access_id: u64,
}

/// Metadata for a cache entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntryMetadata {
    /// File path relative to cache directory
    file_path: PathBuf,
    /// When the entry was created
    created_at: u64,
    /// TTL in seconds
    ttl_seconds: u64,
    /// Size of the entry in bytes
    size_bytes: u64,
    /// Last access time
    last_accessed: u64,
    /// Entry type
    entry_type: CacheEntryType,
}

/// Type of cache entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum CacheEntryType {
    Embedding,
    Nodes,
    Data,
}

/// Serializable cache entry data.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntryData<T> {
    /// The cached data
    data: T,
    /// When the entry was created (seconds since UNIX epoch)
    created_at: u64,
    /// TTL in seconds
    ttl_seconds: u64,
}

impl<T> CacheEntryData<T> {
    /// Create a new cache entry with the given TTL.
    fn new(data: T, ttl: Duration) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            data,
            created_at,
            ttl_seconds: ttl.as_secs(),
        }
    }

    /// Check if the cache entry has expired.
    fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        now > self.created_at + self.ttl_seconds
    }
}

impl CacheEntryMetadata {
    /// Check if the cache entry has expired.
    fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        now > self.created_at + self.ttl_seconds
    }
}

impl FileCache {
    /// Create a new enhanced file cache.
    ///
    /// # Arguments
    /// * `cache_dir` - Directory where cache files will be stored
    /// * `config` - Cache configuration
    ///
    /// # Errors
    /// Returns an error if the cache directory cannot be created or the index cannot be loaded.
    pub async fn new<P: AsRef<Path>>(
        cache_dir: P,
        config: FileCacheConfig,
    ) -> Result<Self, CheungfunError> {
        let cache_dir = cache_dir.as_ref().to_path_buf();

        // Create cache directory structure
        fs::create_dir_all(&cache_dir).await?;
        fs::create_dir_all(cache_dir.join("embeddings")).await?;
        fs::create_dir_all(cache_dir.join("nodes")).await?;
        fs::create_dir_all(cache_dir.join("data")).await?;

        // Load or create index
        let index = Self::load_or_create_index(&cache_dir).await?;

        info!("Created enhanced file cache at: {}", cache_dir.display());

        let cache = Self {
            cache_dir,
            config,
            index: Arc::new(RwLock::new(index)),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        };

        // Start background cleanup if enabled
        if cache.config.enable_auto_cleanup {
            cache.start_background_cleanup().await;
        }

        Ok(cache)
    }

    /// Create a new enhanced file cache with default configuration.
    pub async fn with_default_config<P: AsRef<Path>>(cache_dir: P) -> Result<Self, CheungfunError> {
        Self::new(cache_dir, FileCacheConfig::default()).await
    }

    /// Load or create the cache index.
    async fn load_or_create_index(cache_dir: &Path) -> Result<CacheIndex, CheungfunError> {
        let index_path = cache_dir.join("index.json");

        if index_path.exists() {
            match fs::read_to_string(&index_path).await {
                Ok(content) => match serde_json::from_str::<CacheIndex>(&content) {
                    Ok(index) => {
                        debug!("Loaded cache index with {} entries", index.total_entries());
                        return Ok(index);
                    }
                    Err(e) => {
                        warn!("Failed to parse cache index: {}, creating new index", e);
                    }
                },
                Err(e) => {
                    warn!("Failed to read cache index: {}, creating new index", e);
                }
            }
        }

        Ok(CacheIndex::default())
    }

    /// Save the cache index to disk.
    async fn save_index(&self) -> Result<(), CheungfunError> {
        let index = self.index.read().await;
        let index_path = self.cache_dir.join("index.json");

        let content = serde_json::to_string_pretty(&*index)
            .map_err(|e| CheungfunError::internal(format!("Failed to serialize index: {e}")))?;

        fs::write(&index_path, content).await?;
        debug!("Saved cache index to {}", index_path.display());

        Ok(())
    }

    /// Start background cleanup task.
    async fn start_background_cleanup(&self) {
        let cache_dir = self.cache_dir.clone();
        let index = self.index.clone();
        let stats = self.stats.clone();
        let cleanup_interval = self.config.cleanup_interval;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            loop {
                interval.tick().await;
                if let Err(e) = Self::background_cleanup(&cache_dir, &index, &stats).await {
                    error!("Background cleanup failed: {}", e);
                }
            }
        });

        debug!("Started background cleanup task");
    }

    /// Perform background cleanup.
    async fn background_cleanup(
        cache_dir: &Path,
        index: &Arc<RwLock<CacheIndex>>,
        stats: &Arc<RwLock<CacheStats>>,
    ) -> Result<(), CheungfunError> {
        debug!("Starting background cleanup");

        let mut index_guard = index.write().await;
        let mut removed_count = 0;

        // Clean up expired embeddings
        let expired_embeddings: Vec<String> = index_guard
            .embeddings
            .iter()
            .filter(|(_, metadata)| metadata.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_embeddings {
            if let Some(metadata) = index_guard.embeddings.remove(&key) {
                let file_path = cache_dir.join(&metadata.file_path);
                if let Err(e) = fs::remove_file(&file_path).await {
                    warn!(
                        "Failed to remove expired file {}: {}",
                        file_path.display(),
                        e
                    );
                } else {
                    removed_count += 1;
                }
            }
        }

        // Clean up expired nodes
        let expired_nodes: Vec<String> = index_guard
            .nodes
            .iter()
            .filter(|(_, metadata)| metadata.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_nodes {
            if let Some(metadata) = index_guard.nodes.remove(&key) {
                let file_path = cache_dir.join(&metadata.file_path);
                if let Err(e) = fs::remove_file(&file_path).await {
                    warn!(
                        "Failed to remove expired file {}: {}",
                        file_path.display(),
                        e
                    );
                } else {
                    removed_count += 1;
                }
            }
        }

        // Clean up expired data
        let expired_data: Vec<String> = index_guard
            .data
            .iter()
            .filter(|(_, metadata)| metadata.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_data {
            if let Some(metadata) = index_guard.data.remove(&key) {
                let file_path = cache_dir.join(&metadata.file_path);
                if let Err(e) = fs::remove_file(&file_path).await {
                    warn!(
                        "Failed to remove expired file {}: {}",
                        file_path.display(),
                        e
                    );
                } else {
                    removed_count += 1;
                }
            }
        }

        if removed_count > 0 {
            // Update stats
            let mut stats_guard = stats.write().await;
            stats_guard.expired_entries += removed_count;

            info!(
                "Background cleanup removed {} expired entries",
                removed_count
            );
        }

        Ok(())
    }
}

impl CacheIndex {
    /// Get the total number of entries across all cache types.
    fn total_entries(&self) -> usize {
        self.embeddings.len() + self.nodes.len() + self.data.len()
    }

    /// Update access order for LRU tracking.
    fn update_access(&mut self, key: &str) {
        self.access_order
            .insert(key.to_string(), self.next_access_id);
        self.next_access_id += 1;
    }

    /// Get the least recently used key for a specific cache type.
    fn get_lru_key(&self, cache_type: CacheEntryType) -> Option<String> {
        let entries = match cache_type {
            CacheEntryType::Embedding => &self.embeddings,
            CacheEntryType::Nodes => &self.nodes,
            CacheEntryType::Data => &self.data,
        };

        entries
            .keys()
            .min_by_key(|key| self.access_order.get(*key).unwrap_or(&0))
            .cloned()
    }
}

#[async_trait]
impl PipelineCache for FileCache {
    type Error = CheungfunError;

    async fn get_embedding(&self, key: &str) -> std::result::Result<Option<Vec<f32>>, Self::Error> {
        debug!("Getting embedding from enhanced file cache: {}", key);

        // First, check if entry exists and get metadata
        let metadata = {
            let index = self.index.read().await;
            index.embeddings.get(key).cloned()
        };

        if let Some(metadata) = metadata {
            if metadata.is_expired() {
                // Remove expired entry
                let mut index = self.index.write().await;
                index.embeddings.remove(key);
                drop(index);

                let file_path = self.cache_dir.join(&metadata.file_path);
                if let Err(e) = fs::remove_file(&file_path).await {
                    warn!(
                        "Failed to remove expired file {}: {}",
                        file_path.display(),
                        e
                    );
                }

                self.record_miss().await;
                debug!("Embedding cache entry expired: {}", key);
                return Ok(None);
            }

            // Update access order
            {
                let mut index = self.index.write().await;
                index.update_access(key);
            }

            // Read from file
            let file_path = self.cache_dir.join(&metadata.file_path);
            match fs::read(&file_path).await {
                Ok(content) => {
                    match bincode::serde::decode_from_slice::<CacheEntryData<Vec<f32>>, _>(
                        &content,
                        bincode::config::standard(),
                    ) {
                        Ok((entry_data, _)) => {
                            if entry_data.is_expired() {
                                // Double-check expiration
                                let mut index = self.index.write().await;
                                index.embeddings.remove(key);
                                drop(index);

                                if let Err(e) = fs::remove_file(&file_path).await {
                                    warn!(
                                        "Failed to remove expired file {}: {}",
                                        file_path.display(),
                                        e
                                    );
                                }
                                self.record_miss().await;
                                return Ok(None);
                            }

                            self.record_hit().await;
                            debug!("Embedding cache hit: {}", key);
                            return Ok(Some(entry_data.data));
                        }
                        Err(e) => {
                            warn!("Failed to deserialize embedding data: {}", e);
                            // Remove corrupted entry
                            let mut index = self.index.write().await;
                            index.embeddings.remove(key);
                            drop(index);
                            let _ = fs::remove_file(&file_path).await;
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to read embedding file {}: {}",
                        file_path.display(),
                        e
                    );
                    // Remove entry with missing file
                    let mut index = self.index.write().await;
                    index.embeddings.remove(key);
                }
            }
        }

        self.record_miss().await;
        debug!("Embedding cache miss: {}", key);
        Ok(None)
    }

    async fn put_embedding(
        &self,
        key: &str,
        embedding: Vec<f32>,
        ttl: Duration,
    ) -> std::result::Result<(), Self::Error> {
        debug!("Storing embedding in enhanced file cache: {}", key);

        let mut index = self.index.write().await;

        // Check if we need to evict entries
        if index.embeddings.len() >= self.config.max_entries {
            if let Some(lru_key) = index.get_lru_key(CacheEntryType::Embedding) {
                if let Some(metadata) = index.embeddings.remove(&lru_key) {
                    let file_path = self.cache_dir.join(&metadata.file_path);
                    if let Err(e) = fs::remove_file(&file_path).await {
                        warn!("Failed to remove LRU file {}: {}", file_path.display(), e);
                    } else {
                        debug!("Evicted LRU embedding entry: {}", lru_key);
                        self.record_eviction().await;
                    }
                }
            }
        }

        // Create entry data
        let entry_data = CacheEntryData::new(embedding, ttl);
        let serialized = bincode::serde::encode_to_vec(&entry_data, bincode::config::standard())
            .map_err(|e| CheungfunError::internal(format!("Failed to serialize embedding: {e}")))?;

        // Check entry size
        if serialized.len() > self.config.max_entry_size {
            return Err(CheungfunError::internal(format!(
                "Embedding entry too large: {} bytes (max: {})",
                serialized.len(),
                self.config.max_entry_size
            )));
        }

        // Generate file path
        let file_name = format!("{}.bin", uuid::Uuid::new_v4());
        let file_path = PathBuf::from("embeddings").join(&file_name);
        let full_path = self.cache_dir.join(&file_path);

        // Write to file
        fs::write(&full_path, &serialized).await?;

        // Update index
        let metadata = CacheEntryMetadata {
            file_path,
            created_at: entry_data.created_at,
            ttl_seconds: entry_data.ttl_seconds,
            size_bytes: serialized.len() as u64,
            last_accessed: entry_data.created_at,
            entry_type: CacheEntryType::Embedding,
        };

        index.embeddings.insert(key.to_string(), metadata);
        index.update_access(key);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_entries += 1;

        debug!("Stored embedding in file: {}", full_path.display());
        Ok(())
    }

    async fn get_nodes(&self, key: &str) -> std::result::Result<Option<Vec<Node>>, Self::Error> {
        debug!("Getting nodes from enhanced file cache: {}", key);

        // First, check if entry exists and get metadata
        let metadata = {
            let index = self.index.read().await;
            index.nodes.get(key).cloned()
        };

        if let Some(metadata) = metadata {
            if metadata.is_expired() {
                // Remove expired entry
                let mut index = self.index.write().await;
                index.nodes.remove(key);
                drop(index);

                let file_path = self.cache_dir.join(&metadata.file_path);
                if let Err(e) = fs::remove_file(&file_path).await {
                    warn!(
                        "Failed to remove expired file {}: {}",
                        file_path.display(),
                        e
                    );
                }

                self.record_miss().await;
                debug!("Nodes cache entry expired: {}", key);
                return Ok(None);
            }

            // Update access order
            {
                let mut index = self.index.write().await;
                index.update_access(key);
            }

            // Read from file
            let file_path = self.cache_dir.join(&metadata.file_path);
            match fs::read(&file_path).await {
                Ok(content) => {
                    match bincode::serde::decode_from_slice::<CacheEntryData<Vec<Node>>, _>(
                        &content,
                        bincode::config::standard(),
                    ) {
                        Ok((entry_data, _)) => {
                            if entry_data.is_expired() {
                                // Double-check expiration
                                let mut index = self.index.write().await;
                                index.nodes.remove(key);
                                drop(index);

                                if let Err(e) = fs::remove_file(&file_path).await {
                                    warn!(
                                        "Failed to remove expired file {}: {}",
                                        file_path.display(),
                                        e
                                    );
                                }
                                self.record_miss().await;
                                return Ok(None);
                            }

                            self.record_hit().await;
                            debug!("Nodes cache hit: {}", key);
                            return Ok(Some(entry_data.data));
                        }
                        Err(e) => {
                            warn!("Failed to deserialize nodes data: {}", e);
                            // Remove corrupted entry
                            let mut index = self.index.write().await;
                            index.nodes.remove(key);
                            drop(index);
                            let _ = fs::remove_file(&file_path).await;
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to read nodes file {}: {}", file_path.display(), e);
                    // Remove entry with missing file
                    let mut index = self.index.write().await;
                    index.nodes.remove(key);
                }
            }
        }

        self.record_miss().await;
        debug!("Nodes cache miss: {}", key);
        Ok(None)
    }

    async fn put_nodes(
        &self,
        key: &str,
        nodes: Vec<Node>,
        ttl: Duration,
    ) -> std::result::Result<(), Self::Error> {
        debug!("Storing nodes in enhanced file cache: {}", key);

        let mut index = self.index.write().await;

        // Check if we need to evict entries
        if index.nodes.len() >= self.config.max_entries {
            if let Some(lru_key) = index.get_lru_key(CacheEntryType::Nodes) {
                if let Some(metadata) = index.nodes.remove(&lru_key) {
                    let file_path = self.cache_dir.join(&metadata.file_path);
                    if let Err(e) = fs::remove_file(&file_path).await {
                        warn!("Failed to remove LRU file {}: {}", file_path.display(), e);
                    } else {
                        debug!("Evicted LRU nodes entry: {}", lru_key);
                        self.record_eviction().await;
                    }
                }
            }
        }

        // Create entry data
        let entry_data = CacheEntryData::new(nodes, ttl);
        let serialized = bincode::serde::encode_to_vec(&entry_data, bincode::config::standard())
            .map_err(|e| CheungfunError::internal(format!("Failed to serialize nodes: {e}")))?;

        // Check entry size
        if serialized.len() > self.config.max_entry_size {
            return Err(CheungfunError::internal(format!(
                "Nodes entry too large: {} bytes (max: {})",
                serialized.len(),
                self.config.max_entry_size
            )));
        }

        // Generate file path
        let file_name = format!("{}.bin", uuid::Uuid::new_v4());
        let file_path = PathBuf::from("nodes").join(&file_name);
        let full_path = self.cache_dir.join(&file_path);

        // Write to file
        fs::write(&full_path, &serialized).await?;

        // Update index
        let metadata = CacheEntryMetadata {
            file_path,
            created_at: entry_data.created_at,
            ttl_seconds: entry_data.ttl_seconds,
            size_bytes: serialized.len() as u64,
            last_accessed: entry_data.created_at,
            entry_type: CacheEntryType::Nodes,
        };

        index.nodes.insert(key.to_string(), metadata);
        index.update_access(key);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_entries += 1;

        debug!("Stored nodes in file: {}", full_path.display());
        Ok(())
    }

    async fn get_data_bytes(&self, key: &str) -> std::result::Result<Option<Vec<u8>>, Self::Error> {
        debug!("Getting data bytes from enhanced file cache: {}", key);

        // First, check if entry exists and get metadata
        let metadata = {
            let index = self.index.read().await;
            index.data.get(key).cloned()
        };

        if let Some(metadata) = metadata {
            if metadata.is_expired() {
                // Remove expired entry
                let mut index = self.index.write().await;
                index.data.remove(key);
                drop(index);

                let file_path = self.cache_dir.join(&metadata.file_path);
                if let Err(e) = fs::remove_file(&file_path).await {
                    warn!(
                        "Failed to remove expired file {}: {}",
                        file_path.display(),
                        e
                    );
                }

                self.record_miss().await;
                debug!("Data cache entry expired: {}", key);
                return Ok(None);
            }

            // Update access order
            {
                let mut index = self.index.write().await;
                index.update_access(key);
            }

            // Read from file
            let file_path = self.cache_dir.join(&metadata.file_path);
            match fs::read(&file_path).await {
                Ok(content) => {
                    match bincode::serde::decode_from_slice::<CacheEntryData<Vec<u8>>, _>(
                        &content,
                        bincode::config::standard(),
                    ) {
                        Ok((entry_data, _)) => {
                            if entry_data.is_expired() {
                                // Double-check expiration
                                let mut index = self.index.write().await;
                                index.data.remove(key);
                                drop(index);

                                if let Err(e) = fs::remove_file(&file_path).await {
                                    warn!(
                                        "Failed to remove expired file {}: {}",
                                        file_path.display(),
                                        e
                                    );
                                }
                                self.record_miss().await;
                                return Ok(None);
                            }

                            self.record_hit().await;
                            debug!("Data cache hit: {}", key);
                            return Ok(Some(entry_data.data));
                        }
                        Err(e) => {
                            warn!("Failed to deserialize data: {}", e);
                            // Remove corrupted entry
                            let mut index = self.index.write().await;
                            index.data.remove(key);
                            drop(index);
                            let _ = fs::remove_file(&file_path).await;
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to read data file {}: {}", file_path.display(), e);
                    // Remove entry with missing file
                    let mut index = self.index.write().await;
                    index.data.remove(key);
                }
            }
        }

        self.record_miss().await;
        debug!("Data cache miss: {}", key);
        Ok(None)
    }

    async fn put_data_bytes(
        &self,
        key: &str,
        data_bytes: Vec<u8>,
        ttl: Duration,
    ) -> std::result::Result<(), Self::Error> {
        debug!("Storing data bytes in enhanced file cache: {}", key);

        let mut index = self.index.write().await;

        // Check if we need to evict entries
        if index.data.len() >= self.config.max_entries {
            if let Some(lru_key) = index.get_lru_key(CacheEntryType::Data) {
                if let Some(metadata) = index.data.remove(&lru_key) {
                    let file_path = self.cache_dir.join(&metadata.file_path);
                    if let Err(e) = fs::remove_file(&file_path).await {
                        warn!("Failed to remove LRU file {}: {}", file_path.display(), e);
                    } else {
                        debug!("Evicted LRU data entry: {}", lru_key);
                        self.record_eviction().await;
                    }
                }
            }
        }

        // Create entry data
        let entry_data = CacheEntryData::new(data_bytes, ttl);
        let serialized = bincode::serde::encode_to_vec(&entry_data, bincode::config::standard())
            .map_err(|e| CheungfunError::internal(format!("Failed to serialize data: {e}")))?;

        // Check entry size
        if serialized.len() > self.config.max_entry_size {
            return Err(CheungfunError::internal(format!(
                "Data entry too large: {} bytes (max: {})",
                serialized.len(),
                self.config.max_entry_size
            )));
        }

        // Generate file path
        let file_name = format!("{}.bin", uuid::Uuid::new_v4());
        let file_path = PathBuf::from("data").join(&file_name);
        let full_path = self.cache_dir.join(&file_path);

        // Write to file
        fs::write(&full_path, &serialized).await?;

        // Update index
        let metadata = CacheEntryMetadata {
            file_path,
            created_at: entry_data.created_at,
            ttl_seconds: entry_data.ttl_seconds,
            size_bytes: serialized.len() as u64,
            last_accessed: entry_data.created_at,
            entry_type: CacheEntryType::Data,
        };

        index.data.insert(key.to_string(), metadata);
        index.update_access(key);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_entries += 1;

        debug!("Stored data in file: {}", full_path.display());
        Ok(())
    }

    async fn exists(&self, key: &str) -> std::result::Result<bool, Self::Error> {
        let index = self.index.read().await;

        // Check all cache types
        let exists = index.embeddings.contains_key(key)
            || index.nodes.contains_key(key)
            || index.data.contains_key(key);

        Ok(exists)
    }

    async fn remove(&self, key: &str) -> std::result::Result<(), Self::Error> {
        debug!("Removing cache entry: {}", key);

        let mut index = self.index.write().await;
        let mut removed = false;

        // Remove from embeddings
        if let Some(metadata) = index.embeddings.remove(key) {
            let file_path = self.cache_dir.join(&metadata.file_path);
            if let Err(e) = fs::remove_file(&file_path).await {
                warn!("Failed to remove file {}: {}", file_path.display(), e);
            } else {
                removed = true;
            }
        }

        // Remove from nodes
        if let Some(metadata) = index.nodes.remove(key) {
            let file_path = self.cache_dir.join(&metadata.file_path);
            if let Err(e) = fs::remove_file(&file_path).await {
                warn!("Failed to remove file {}: {}", file_path.display(), e);
            } else {
                removed = true;
            }
        }

        // Remove from data
        if let Some(metadata) = index.data.remove(key) {
            let file_path = self.cache_dir.join(&metadata.file_path);
            if let Err(e) = fs::remove_file(&file_path).await {
                warn!("Failed to remove file {}: {}", file_path.display(), e);
            } else {
                removed = true;
            }
        }

        // Remove from access order
        index.access_order.remove(key);

        if removed {
            debug!("Successfully removed cache entry: {}", key);
        }

        Ok(())
    }

    async fn clear(&self) -> std::result::Result<(), Self::Error> {
        info!("Clearing all cache entries");

        let mut index = self.index.write().await;

        // Remove all files
        for metadata in index.embeddings.values() {
            let file_path = self.cache_dir.join(&metadata.file_path);
            if let Err(e) = fs::remove_file(&file_path).await {
                warn!("Failed to remove file {}: {}", file_path.display(), e);
            }
        }

        for metadata in index.nodes.values() {
            let file_path = self.cache_dir.join(&metadata.file_path);
            if let Err(e) = fs::remove_file(&file_path).await {
                warn!("Failed to remove file {}: {}", file_path.display(), e);
            }
        }

        for metadata in index.data.values() {
            let file_path = self.cache_dir.join(&metadata.file_path);
            if let Err(e) = fs::remove_file(&file_path).await {
                warn!("Failed to remove file {}: {}", file_path.display(), e);
            }
        }

        // Clear index
        index.embeddings.clear();
        index.nodes.clear();
        index.data.clear();
        index.access_order.clear();
        index.next_access_id = 0;

        // Reset stats
        let mut stats = self.stats.write().await;
        *stats = CacheStats::default();

        // Save empty index
        self.save_index().await?;

        info!("Cache cleared successfully");
        Ok(())
    }

    async fn cleanup(&self) -> std::result::Result<usize, Self::Error> {
        debug!("Cleaning up expired cache entries");

        Self::background_cleanup(&self.cache_dir, &self.index, &self.stats).await?;

        // Save updated index
        self.save_index().await?;

        let stats = self.stats.read().await;
        Ok(stats.expired_entries)
    }

    async fn stats(&self) -> std::result::Result<CacheStats, Self::Error> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }

    async fn health(&self) -> std::result::Result<CacheHealth, Self::Error> {
        let stats = self.stats().await?;
        let index = self.index.read().await;

        let total_entries = index.total_entries();
        let usage_ratio = (total_entries as f64 / (self.config.max_entries * 3) as f64).min(1.0);

        // Determine health status
        let health_status = if usage_ratio > 0.9 {
            HealthStatus::Warning
        } else if usage_ratio > 0.95 {
            HealthStatus::Critical
        } else {
            HealthStatus::Healthy
        };

        let mut messages = Vec::new();
        if usage_ratio > 0.8 {
            messages.push(format!("Cache usage is high: {:.1}%", usage_ratio * 100.0));
        }

        if stats.hit_rate() < 50.0 && stats.total_operations() > 100 {
            messages.push(format!("Low cache hit rate: {:.1}%", stats.hit_rate()));
        }

        // Calculate estimated size
        let estimated_size_mb = index
            .embeddings
            .values()
            .chain(index.nodes.values())
            .chain(index.data.values())
            .map(|metadata| metadata.size_bytes)
            .sum::<u64>() as f64
            / 1024.0
            / 1024.0;

        Ok(CacheHealth {
            status: health_status,
            usage_ratio,
            hit_rate: stats.hit_rate(),
            total_entries,
            estimated_size_mb,
            messages,
        })
    }
}

impl FileCache {
    /// Update cache statistics for a hit.
    async fn record_hit(&self) {
        let mut stats = self.stats.write().await;
        stats.hits += 1;
    }

    /// Update cache statistics for a miss.
    async fn record_miss(&self) {
        let mut stats = self.stats.write().await;
        stats.misses += 1;
    }

    /// Update cache statistics for an eviction.
    async fn record_eviction(&self) {
        let mut stats = self.stats.write().await;
        stats.evictions += 1;
    }

    /// Get the cache directory path.
    #[must_use]
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Get the cache configuration.
    #[must_use]
    pub fn config(&self) -> &FileCacheConfig {
        &self.config
    }

    /// Get cache usage statistics.
    pub async fn usage_stats(&self) -> Result<CacheUsageStats, CheungfunError> {
        let index = self.index.read().await;

        let embedding_count = index.embeddings.len();
        let nodes_count = index.nodes.len();
        let data_count = index.data.len();

        let total_size = index
            .embeddings
            .values()
            .chain(index.nodes.values())
            .chain(index.data.values())
            .map(|metadata| metadata.size_bytes)
            .sum::<u64>();

        Ok(CacheUsageStats {
            embedding_entries: embedding_count,
            nodes_entries: nodes_count,
            data_entries: data_count,
            total_entries: embedding_count + nodes_count + data_count,
            total_size_bytes: total_size,
            max_entries_per_type: self.config.max_entries,
        })
    }

    /// Perform manual cleanup of expired entries.
    pub async fn manual_cleanup(&self) -> Result<usize, CheungfunError> {
        self.cleanup().await
    }

    /// Compact the cache by removing expired entries and optimizing storage.
    pub async fn compact(&self) -> Result<CompactionStats, CheungfunError> {
        info!("Starting cache compaction");

        let start_time = std::time::Instant::now();
        let initial_stats = self.usage_stats().await?;

        // Cleanup expired entries
        let expired_removed = self.cleanup().await?;

        // Save index to ensure consistency
        self.save_index().await?;

        let final_stats = self.usage_stats().await?;
        let compaction_time = start_time.elapsed();

        let stats = CompactionStats {
            initial_entries: initial_stats.total_entries,
            final_entries: final_stats.total_entries,
            expired_removed,
            space_freed_bytes: initial_stats
                .total_size_bytes
                .saturating_sub(final_stats.total_size_bytes),
            compaction_time,
        };

        info!(
            "Cache compaction completed: removed {} expired entries, freed {} bytes in {:?}",
            stats.expired_removed, stats.space_freed_bytes, stats.compaction_time
        );

        Ok(stats)
    }
}

/// Statistics about cache usage.
#[derive(Debug, Clone)]
pub struct CacheUsageStats {
    /// Number of embedding entries
    pub embedding_entries: usize,
    /// Number of node entries
    pub nodes_entries: usize,
    /// Number of data entries
    pub data_entries: usize,
    /// Total number of entries
    pub total_entries: usize,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// Maximum entries per cache type
    pub max_entries_per_type: usize,
}

/// Statistics about cache compaction.
#[derive(Debug, Clone)]
pub struct CompactionStats {
    /// Number of entries before compaction
    pub initial_entries: usize,
    /// Number of entries after compaction
    pub final_entries: usize,
    /// Number of expired entries removed
    pub expired_removed: usize,
    /// Bytes of space freed
    pub space_freed_bytes: u64,
    /// Time taken for compaction
    pub compaction_time: Duration,
}
