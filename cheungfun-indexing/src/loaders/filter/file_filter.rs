//! Unified file filtering implementation.

use super::{Filter, FilterConfig, FilterResult, GitignoreMatcher, GlobMatcher};
use std::path::Path;
use tracing::{debug, warn};

/// A comprehensive file filter that combines multiple filtering strategies.
#[derive(Debug)]
pub struct FileFilter {
    /// Configuration for the filter.
    config: FilterConfig,

    /// Gitignore pattern matcher.
    gitignore_matcher: Option<GitignoreMatcher>,

    /// Custom glob pattern matcher.
    glob_matcher: Option<GlobMatcher>,
}

impl FileFilter {
    /// Create a new FileFilter with the given configuration.
    pub fn new<P: AsRef<Path>>(base_dir: P, config: FilterConfig) -> FilterResult<Self> {
        let base_dir = base_dir.as_ref();

        debug!("Creating FileFilter for directory: {}", base_dir.display());

        // Initialize gitignore matcher if enabled
        let gitignore_matcher = if config.respect_gitignore {
            match GitignoreMatcher::new(base_dir) {
                Ok(matcher) => Some(matcher),
                Err(e) => {
                    warn!("Failed to create gitignore matcher: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Initialize glob matcher if there are patterns
        let glob_matcher =
            if !config.exclude_patterns.is_empty() || !config.include_patterns.is_empty() {
                match GlobMatcher::new(
                    &config.exclude_patterns,
                    &config.include_patterns,
                    config.case_sensitive,
                ) {
                    Ok(matcher) => Some(matcher),
                    Err(e) => {
                        warn!("Failed to create glob matcher: {}", e);
                        return Err(e);
                    }
                }
            } else {
                None
            };

        Ok(Self {
            config,
            gitignore_matcher,
            glob_matcher,
        })
    }

    /// Create a FileFilter with additional ignore files.
    pub fn with_ignore_files<P: AsRef<Path>>(
        base_dir: P,
        config: FilterConfig,
    ) -> FilterResult<Self> {
        let base_dir = base_dir.as_ref();

        debug!(
            "Creating FileFilter with ignore files for: {}",
            base_dir.display()
        );

        // Initialize gitignore matcher with custom ignore files
        let gitignore_matcher = if config.respect_gitignore || !config.ignore_files.is_empty() {
            match GitignoreMatcher::with_ignore_files(base_dir, &config.ignore_files) {
                Ok(matcher) => Some(matcher),
                Err(e) => {
                    warn!(
                        "Failed to create gitignore matcher with ignore files: {}",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        // Initialize glob matcher
        let glob_matcher =
            if !config.exclude_patterns.is_empty() || !config.include_patterns.is_empty() {
                match GlobMatcher::new(
                    &config.exclude_patterns,
                    &config.include_patterns,
                    config.case_sensitive,
                ) {
                    Ok(matcher) => Some(matcher),
                    Err(e) => {
                        warn!("Failed to create glob matcher: {}", e);
                        return Err(e);
                    }
                }
            } else {
                None
            };

        Ok(Self {
            config,
            gitignore_matcher,
            glob_matcher,
        })
    }

    /// Check if a file should be included based on all filtering rules.
    pub fn should_include_file(&self, path: &Path) -> bool {
        // Check if it's a file (not a directory)
        if path.is_dir() {
            return false;
        }

        // Apply all filters in order
        self.apply_all_filters(path)
    }

    /// Check if a directory should be traversed.
    pub fn should_traverse_directory(&self, path: &Path) -> bool {
        // Check if it's a directory
        if !path.is_dir() {
            return false;
        }

        // Apply directory-specific filtering
        self.apply_directory_filters(path)
    }

    /// Apply all filtering rules to a path.
    fn apply_all_filters(&self, path: &Path) -> bool {
        // 1. Check hidden files
        if self.config.exclude_hidden && self.is_hidden(path) {
            debug!("Excluding hidden file: {}", path.display());
            return false;
        }

        // 2. Check file size (if it's a file)
        if path.is_file() {
            if let Ok(metadata) = std::fs::metadata(path) {
                let size = metadata.len();

                if let Some(max_size) = self.config.max_file_size {
                    if size > max_size {
                        debug!(
                            "Excluding file due to size {} > {}: {}",
                            size,
                            max_size,
                            path.display()
                        );
                        return false;
                    }
                }

                if let Some(min_size) = self.config.min_file_size {
                    if size < min_size {
                        debug!(
                            "Excluding file due to size {} < {}: {}",
                            size,
                            min_size,
                            path.display()
                        );
                        return false;
                    }
                }

                if self.config.exclude_empty && size == 0 {
                    debug!("Excluding empty file: {}", path.display());
                    return false;
                }
            }
        }

        // 3. Check file extensions
        if !self.check_file_extension(path) {
            return false;
        }

        // 4. Check gitignore patterns
        if let Some(ref gitignore) = self.gitignore_matcher {
            if !gitignore.should_include(path) {
                debug!("Excluding file due to gitignore: {}", path.display());
                return false;
            }
        }

        // 5. Check custom glob patterns
        if let Some(ref glob) = self.glob_matcher {
            if !glob.should_include(path) {
                debug!("Excluding file due to glob patterns: {}", path.display());
                return false;
            }
        }

        true
    }

    /// Apply directory-specific filtering rules.
    fn apply_directory_filters(&self, path: &Path) -> bool {
        // 1. Check hidden directories
        if self.config.exclude_hidden && self.is_hidden(path) {
            debug!("Excluding hidden directory: {}", path.display());
            return false;
        }

        // 2. Check gitignore patterns for directories
        if let Some(ref gitignore) = self.gitignore_matcher {
            if !gitignore.should_traverse_dir(path) {
                debug!("Excluding directory due to gitignore: {}", path.display());
                return false;
            }
        }

        // 3. Check custom glob patterns for directories
        if let Some(ref glob) = self.glob_matcher {
            if !glob.should_traverse_dir(path) {
                debug!(
                    "Excluding directory due to glob patterns: {}",
                    path.display()
                );
                return false;
            }
        }

        true
    }

    /// Check if a file matches the extension filters.
    fn check_file_extension(&self, path: &Path) -> bool {
        let extension = match path.extension().and_then(|e| e.to_str()) {
            Some(ext) => ext.to_lowercase(),
            None => {
                // No extension - check if we have an include list
                if self.config.include_extensions.is_some() {
                    debug!("Excluding file without extension: {}", path.display());
                    return false;
                }
                return true;
            }
        };

        // Check exclude list first
        if self.config.exclude_extensions.contains(&extension) {
            debug!(
                "Excluding file due to extension '{}': {}",
                extension,
                path.display()
            );
            return false;
        }

        // Check include list if specified
        if let Some(ref include_exts) = self.config.include_extensions {
            if !include_exts.contains(&extension) {
                debug!(
                    "Excluding file not in include list '{}': {}",
                    extension,
                    path.display()
                );
                return false;
            }
        }

        true
    }

    /// Check if a path represents a hidden file or directory.
    fn is_hidden(&self, path: &Path) -> bool {
        path.file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with('.'))
            .unwrap_or(false)
    }

    /// Get the filter configuration.
    pub fn config(&self) -> &FilterConfig {
        &self.config
    }

    /// Check if gitignore filtering is enabled.
    pub fn has_gitignore(&self) -> bool {
        self.gitignore_matcher.is_some()
    }

    /// Check if glob pattern filtering is enabled.
    pub fn has_glob_patterns(&self) -> bool {
        self.glob_matcher.is_some()
    }
}

impl Filter for FileFilter {
    fn should_include(&self, path: &Path) -> bool {
        if path.is_dir() {
            self.should_traverse_directory(path)
        } else {
            self.should_include_file(path)
        }
    }

    fn should_traverse_dir(&self, path: &Path) -> bool {
        self.should_traverse_directory(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_file_filter_basic() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        let config = FilterConfig::new()
            .with_exclude_extensions(vec!["log".to_string()])
            .with_exclude_hidden(true);

        let filter = FileFilter::new(base_path, config).unwrap();

        // Create test files
        fs::write(base_path.join("main.rs"), "fn main() {}").unwrap();
        fs::write(base_path.join("debug.log"), "log content").unwrap();
        fs::write(base_path.join(".hidden"), "hidden content").unwrap();

        // Test filtering
        assert!(filter.should_include_file(&base_path.join("main.rs")));
        assert!(!filter.should_include_file(&base_path.join("debug.log")));
        assert!(!filter.should_include_file(&base_path.join(".hidden")));
    }

    #[test]
    fn test_file_filter_with_gitignore() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create .gitignore
        fs::write(base_path.join(".gitignore"), "*.tmp\ntarget/\n").unwrap();

        let config = FilterConfig::new().with_respect_gitignore(true);
        let filter = FileFilter::new(base_path, config).unwrap();

        // Create test files
        fs::write(base_path.join("main.rs"), "fn main() {}").unwrap();
        fs::write(base_path.join("temp.tmp"), "temp content").unwrap();
        fs::create_dir(base_path.join("target")).unwrap();

        // Test filtering
        assert!(filter.should_include_file(&base_path.join("main.rs")));
        assert!(!filter.should_include_file(&base_path.join("temp.tmp")));
        assert!(!filter.should_traverse_directory(&base_path.join("target")));
    }

    #[test]
    fn test_file_filter_size_limits() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        let config = FilterConfig::new()
            .with_max_file_size(10)
            .with_min_file_size(2);

        let filter = FileFilter::new(base_path, config).unwrap();

        // Create test files with different sizes
        fs::write(base_path.join("empty.txt"), "").unwrap();
        fs::write(base_path.join("small.txt"), "a").unwrap();
        fs::write(base_path.join("good.txt"), "hello").unwrap();
        fs::write(
            base_path.join("large.txt"),
            "this is a very long file content",
        )
        .unwrap();

        // Test filtering
        assert!(!filter.should_include_file(&base_path.join("empty.txt"))); // too small
        assert!(!filter.should_include_file(&base_path.join("small.txt"))); // too small
        assert!(filter.should_include_file(&base_path.join("good.txt"))); // just right
        assert!(!filter.should_include_file(&base_path.join("large.txt"))); // too large
    }
}
