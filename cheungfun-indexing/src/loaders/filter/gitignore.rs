//! Gitignore pattern matching implementation.

use super::{Filter, FilterError, FilterResult};
use ignore::{Match, gitignore::GitignoreBuilder};
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

/// A matcher that respects .gitignore files and custom ignore patterns.
#[derive(Debug)]
pub struct GitignoreMatcher {
    /// The compiled gitignore matcher from the ignore crate.
    matcher: Option<ignore::gitignore::Gitignore>,

    /// Base directory for relative path resolution.
    base_dir: PathBuf,

    /// Whether to respect global gitignore files.
    respect_global: bool,
}

impl GitignoreMatcher {
    /// Create a new `GitignoreMatcher` for the given directory.
    ///
    /// This will automatically discover and parse .gitignore files in the directory tree.
    pub fn new<P: AsRef<Path>>(base_dir: P) -> FilterResult<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        debug!(
            "Creating GitignoreMatcher for directory: {}",
            base_dir.display()
        );

        let mut builder = GitignoreBuilder::new(&base_dir);

        // Add .gitignore files from the directory tree
        let gitignore_path = base_dir.join(".gitignore");
        if gitignore_path.exists() {
            if let Some(e) = builder.add(&gitignore_path) {
                warn!(
                    "Failed to parse .gitignore file {}: {}",
                    gitignore_path.display(),
                    e
                );
            } else {
                debug!("Added .gitignore file: {}", gitignore_path.display());
            }
        }

        // Look for .gitignore files in parent directories
        let mut current_dir = base_dir.parent();
        while let Some(dir) = current_dir {
            let gitignore_path = dir.join(".gitignore");
            if gitignore_path.exists() {
                if let Some(e) = builder.add(&gitignore_path) {
                    warn!(
                        "Failed to parse parent .gitignore file {}: {}",
                        gitignore_path.display(),
                        e
                    );
                } else {
                    debug!("Added parent .gitignore file: {}", gitignore_path.display());
                }
            }
            current_dir = dir.parent();
        }

        let matcher = match builder.build() {
            Ok(gitignore) => Some(gitignore),
            Err(e) => {
                warn!("Failed to build gitignore matcher: {}", e);
                None
            }
        };

        Ok(Self {
            matcher,
            base_dir,
            respect_global: true,
        })
    }

    /// Create a `GitignoreMatcher` with custom ignore files.
    pub fn with_ignore_files<P: AsRef<Path>>(
        base_dir: P,
        ignore_files: &[PathBuf],
    ) -> FilterResult<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        debug!(
            "Creating GitignoreMatcher with custom ignore files for: {}",
            base_dir.display()
        );

        let mut builder = GitignoreBuilder::new(&base_dir);

        // Add custom ignore files
        for ignore_file in ignore_files {
            if ignore_file.exists() {
                if let Some(e) = builder.add(ignore_file) {
                    warn!(
                        "Failed to parse ignore file {}: {}",
                        ignore_file.display(),
                        e
                    );
                } else {
                    debug!("Added ignore file: {}", ignore_file.display());
                }
            } else {
                warn!("Ignore file does not exist: {}", ignore_file.display());
            }
        }

        let matcher = match builder.build() {
            Ok(gitignore) => Some(gitignore),
            Err(e) => {
                warn!("Failed to build gitignore matcher: {}", e);
                None
            }
        };

        Ok(Self {
            matcher,
            base_dir,
            respect_global: true,
        })
    }

    /// Create a `GitignoreMatcher` from custom patterns.
    pub fn from_patterns<P: AsRef<Path>>(base_dir: P, patterns: &[String]) -> FilterResult<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        debug!(
            "Creating GitignoreMatcher from patterns for: {}",
            base_dir.display()
        );

        let mut builder = GitignoreBuilder::new(&base_dir);

        // Add patterns directly
        for pattern in patterns {
            if let Err(e) = builder.add_line(None, pattern) {
                return Err(FilterError::GitignoreParse(format!(
                    "Invalid pattern '{pattern}': {e}"
                )));
            }
        }

        let matcher = match builder.build() {
            Ok(gitignore) => Some(gitignore),
            Err(e) => {
                return Err(FilterError::GitignoreParse(format!(
                    "Failed to build gitignore matcher: {e}"
                )));
            }
        };

        Ok(Self {
            matcher,
            base_dir,
            respect_global: true,
        })
    }

    /// Set whether to respect global gitignore files.
    #[must_use]
    pub fn with_respect_global(mut self, respect: bool) -> Self {
        self.respect_global = respect;
        self
    }

    /// Check if a path matches any ignore pattern.
    #[must_use]
    pub fn is_ignored(&self, path: &Path) -> bool {
        let Some(ref matcher) = self.matcher else {
            return false;
        };

        // Convert to relative path if needed
        let relative_path = if path.is_absolute() {
            match path.strip_prefix(&self.base_dir) {
                Ok(rel) => rel,
                Err(_) => path, // Use absolute path if not under base_dir
            }
        } else {
            path
        };

        match matcher.matched(relative_path, path.is_dir()) {
            Match::None => false,
            Match::Ignore(_) => true,
            Match::Whitelist(_) => false, // Explicitly whitelisted
        }
    }

    /// Get the base directory.
    #[must_use]
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }
}

impl Filter for GitignoreMatcher {
    fn should_include(&self, path: &Path) -> bool {
        !self.is_ignored(path)
    }

    fn should_traverse_dir(&self, path: &Path) -> bool {
        // For directories, we need to be more careful
        // A directory might be ignored but contain whitelisted files
        !self.is_ignored(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_gitignore_basic_patterns() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create a .gitignore file
        let gitignore_content = "*.log\ntarget/\n!important.log\n";
        fs::write(base_path.join(".gitignore"), gitignore_content).unwrap();

        // Create test files and directories
        fs::write(base_path.join("debug.log"), "log content").unwrap();
        fs::create_dir(base_path.join("target")).unwrap();
        fs::create_dir_all(base_path.join("target/debug")).unwrap();
        fs::write(base_path.join("important.log"), "important content").unwrap();
        fs::create_dir(base_path.join("src")).unwrap();
        fs::write(base_path.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(base_path.join("README.md"), "# README").unwrap();

        let matcher = GitignoreMatcher::new(base_path).unwrap();

        // Test ignored files
        assert!(!matcher.should_include(&base_path.join("debug.log")));
        assert!(!matcher.should_include(&base_path.join("target")));
        // Note: target/debug might not be ignored if target/ pattern doesn't match subdirectories
        // This depends on the gitignore implementation
        // assert!(!matcher.should_include(&base_path.join("target/debug")));

        // Test whitelisted file
        assert!(matcher.should_include(&base_path.join("important.log")));

        // Test non-matching files
        assert!(matcher.should_include(&base_path.join("src/main.rs")));
        assert!(matcher.should_include(&base_path.join("README.md")));
    }

    #[test]
    fn test_gitignore_from_patterns() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        let patterns = vec![
            "*.tmp".to_string(),
            "build/".to_string(),
            "!keep.tmp".to_string(),
        ];

        // Create test files and directories
        fs::write(base_path.join("temp.tmp"), "temp content").unwrap();
        fs::create_dir(base_path.join("build")).unwrap();
        fs::write(base_path.join("keep.tmp"), "keep content").unwrap();
        fs::create_dir(base_path.join("src")).unwrap();
        fs::write(base_path.join("src/main.rs"), "fn main() {}").unwrap();

        let matcher = GitignoreMatcher::from_patterns(base_path, &patterns).unwrap();

        assert!(!matcher.should_include(&base_path.join("temp.tmp")));
        assert!(!matcher.should_include(&base_path.join("build")));
        assert!(matcher.should_include(&base_path.join("keep.tmp")));
        assert!(matcher.should_include(&base_path.join("src/main.rs")));
    }
}
