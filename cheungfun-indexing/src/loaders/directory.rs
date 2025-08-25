//! Directory-based document loader.

use async_trait::async_trait;
use cheungfun_core::traits::Loader;
use cheungfun_core::{Document, Result as CoreResult};
use futures::future::join_all;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, error, info, warn};

use super::{file::FileLoader, FileFilter, LoaderConfig};
use crate::error::{IndexingError, Result};

/// Loads documents from all files in a directory.
///
/// This loader recursively traverses a directory and loads all supported files.
/// It can be configured to control recursion depth, file filtering, and error handling.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::loaders::{DirectoryLoader, LoaderConfig};
/// use cheungfun_core::traits::Loader;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = LoaderConfig::new()
///         .with_max_depth(3)
///         .with_include_extensions(vec!["txt".to_string(), "md".to_string()]);
///         
///     let loader = DirectoryLoader::with_config("./docs", config)?;
///     let documents = loader.load().await?;
///     
///     println!("Loaded {} documents", documents.len());
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct DirectoryLoader {
    /// Path to the directory to load from.
    path: PathBuf,
    /// Configuration for the loader.
    config: LoaderConfig,
    /// Enhanced file filter (if enabled).
    file_filter: Option<FileFilter>,
}

impl DirectoryLoader {
    /// Create a new directory loader for the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the directory to load from
    ///
    /// # Errors
    ///
    /// Returns an error if the path does not exist or is not a directory.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(IndexingError::directory_not_found(
                path.display().to_string(),
            ));
        }

        if !path.is_dir() {
            return Err(IndexingError::configuration(format!(
                "Path is not a directory: {}",
                path.display()
            )));
        }

        Ok(Self {
            path,
            config: LoaderConfig::default(),
            file_filter: None,
        })
    }

    /// Create a new directory loader with custom configuration.
    pub fn with_config<P: AsRef<Path>>(path: P, config: LoaderConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Create file filter if enhanced filtering is enabled
        let file_filter = if let Some(ref filter_config) = config.filter_config {
            match FileFilter::new(&path, filter_config.clone()) {
                Ok(filter) => Some(filter),
                Err(e) => {
                    warn!("Failed to create file filter: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let mut loader = Self::new(&path)?;
        loader.config = config;
        loader.file_filter = file_filter;
        Ok(loader)
    }

    /// Get the directory path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the loader configuration.
    #[must_use]
    pub fn config(&self) -> &LoaderConfig {
        &self.config
    }

    /// Recursively find all files in the directory.
    async fn find_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        self.find_files_recursive(&self.path, 0, &mut files).await?;
        Ok(files)
    }

    /// Recursively traverse directory and collect file paths.
    fn find_files_recursive<'a>(
        &'a self,
        dir: &'a Path,
        current_depth: usize,
        files: &'a mut Vec<PathBuf>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // Check depth limit
            if let Some(max_depth) = self.config.max_depth {
                if current_depth >= max_depth {
                    debug!("Reached maximum depth {} at {}", max_depth, dir.display());
                    return Ok(());
                }
            }

            debug!(
                "Scanning directory: {} (depth: {})",
                dir.display(),
                current_depth
            );

            let mut entries = fs::read_dir(dir).await.map_err(IndexingError::Io)?;

            while let Some(entry) = entries.next_entry().await.map_err(IndexingError::Io)? {
                let path = entry.path();

                // Handle symbolic links
                let metadata = if self.config.follow_symlinks {
                    fs::metadata(&path).await
                } else {
                    fs::symlink_metadata(&path).await
                };

                let metadata = match metadata {
                    Ok(metadata) => metadata,
                    Err(e) => {
                        warn!("Failed to read metadata for {}: {}", path.display(), e);
                        if !self.config.continue_on_error {
                            return Err(IndexingError::Io(e));
                        }
                        continue;
                    }
                };

                if metadata.is_file() {
                    // Check if file should be included
                    if self.should_include_file(&path) {
                        files.push(path);
                    }
                } else if metadata.is_dir() {
                    // Check if directory should be traversed
                    if self.should_traverse_directory(&path) {
                        // Recursively process subdirectory
                        if let Err(e) = self
                            .find_files_recursive(&path, current_depth + 1, files)
                            .await
                        {
                            error!("Failed to process directory {}: {}", path.display(), e);
                            if !self.config.continue_on_error {
                                return Err(e);
                            }
                        }
                    } else {
                        debug!("Skipping directory: {}", path.display());
                    }
                }
                // Skip other file types (symlinks, devices, etc.)
            }

            Ok(())
        })
    }

    /// Check if a file should be included based on configuration.
    fn should_include_file(&self, path: &Path) -> bool {
        // Use enhanced filtering if available
        if let Some(ref filter) = self.file_filter {
            return filter.should_include_file(path);
        }

        // Fall back to legacy filtering
        self.legacy_should_include_file(path)
    }

    /// Check if a directory should be traversed.
    fn should_traverse_directory(&self, path: &Path) -> bool {
        // Use enhanced filtering if available
        if let Some(ref filter) = self.file_filter {
            return filter.should_traverse_directory(path);
        }

        // Legacy behavior: traverse all directories
        true
    }

    /// Legacy file filtering logic (for backward compatibility).
    fn legacy_should_include_file(&self, path: &Path) -> bool {
        // Check file extension
        if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
            let ext = extension.to_lowercase();

            // Check exclude list
            if self.config.exclude_extensions.contains(&ext) {
                debug!("Excluding file due to extension: {}", path.display());
                return false;
            }

            // Check include list if specified
            if let Some(ref include_exts) = self.config.include_extensions {
                if !include_exts.contains(&ext) {
                    debug!("Excluding file not in include list: {}", path.display());
                    return false;
                }
            }
        } else {
            // No extension - check if we have an include list
            if self.config.include_extensions.is_some() {
                debug!("Excluding file without extension: {}", path.display());
                return false;
            }
        }

        true
    }

    /// Load documents from all files in parallel.
    async fn load_files_parallel(&self, file_paths: Vec<PathBuf>) -> CoreResult<Vec<Document>> {
        info!(
            "Loading {} files from directory: {}",
            file_paths.len(),
            self.path.display()
        );

        // Create file loaders for each file
        let mut loaders = Vec::new();
        for path in file_paths {
            match FileLoader::with_config(&path, self.config.clone()) {
                Ok(loader) => loaders.push(loader),
                Err(e) => {
                    error!("Failed to create loader for {}: {}", path.display(), e);
                    if !self.config.continue_on_error {
                        return Err(e.into());
                    }
                }
            }
        }

        // Load all files in parallel
        let load_futures = loaders.into_iter().map(|loader| async move {
            match loader.load().await {
                Ok(docs) => docs,
                Err(e) => {
                    error!("Failed to load file: {}", e);
                    vec![] // Return empty vec on error if continue_on_error is true
                }
            }
        });

        let results = join_all(load_futures).await;

        // Flatten results
        let mut all_documents = Vec::new();
        for docs in results {
            all_documents.extend(docs);
        }

        info!(
            "Successfully loaded {} documents from directory",
            all_documents.len()
        );
        Ok(all_documents)
    }
}

#[async_trait]
impl Loader for DirectoryLoader {
    async fn load(&self) -> CoreResult<Vec<Document>> {
        info!("Loading documents from directory: {}", self.path.display());

        // Find all files in the directory
        let file_paths = match self.find_files().await {
            Ok(paths) => paths,
            Err(e) => {
                error!("Failed to scan directory {}: {}", self.path.display(), e);
                return Err(e.into());
            }
        };

        if file_paths.is_empty() {
            warn!("No files found in directory: {}", self.path.display());
            return Ok(vec![]);
        }

        // Load all files
        self.load_files_parallel(file_paths).await
    }

    fn name(&self) -> &'static str {
        "DirectoryLoader"
    }

    async fn health_check(&self) -> CoreResult<()> {
        if !self.path.exists() {
            return Err(IndexingError::directory_not_found(self.path.display().to_string()).into());
        }

        if !self.path.is_dir() {
            return Err(IndexingError::configuration(format!(
                "Path is not a directory: {}",
                self.path.display()
            ))
            .into());
        }

        // Try to read directory
        let _ = fs::read_dir(&self.path).await.map_err(IndexingError::Io)?;

        Ok(())
    }

    async fn metadata(&self) -> CoreResult<std::collections::HashMap<String, serde_json::Value>> {
        let mut metadata = std::collections::HashMap::new();

        metadata.insert(
            "loader_type".to_string(),
            serde_json::Value::String("directory".to_string()),
        );
        metadata.insert(
            "directory_path".to_string(),
            serde_json::Value::String(self.path.display().to_string()),
        );

        // Count files (this is expensive but useful for metadata)
        if let Ok(file_paths) = self.find_files().await {
            metadata.insert(
                "file_count".to_string(),
                serde_json::Value::Number(file_paths.len().into()),
            );
        }

        Ok(metadata)
    }
}
