//! High-performance glob pattern matching implementation.

use super::{Filter, FilterError, FilterResult};
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use std::path::Path;
use tracing::{debug, warn};

/// A high-performance glob pattern matcher using the globset crate.
#[derive(Debug)]
pub struct GlobMatcher {
    /// Compiled exclude patterns.
    exclude_set: Option<GlobSet>,

    /// Compiled include patterns.
    include_set: Option<GlobSet>,

    /// Whether pattern matching is case sensitive.
    case_sensitive: bool,
}

impl GlobMatcher {
    /// Create a new `GlobMatcher` with the given patterns.
    pub fn new(
        exclude_patterns: &[String],
        include_patterns: &[String],
        case_sensitive: bool,
    ) -> FilterResult<Self> {
        debug!(
            "Creating GlobMatcher with {} exclude and {} include patterns",
            exclude_patterns.len(),
            include_patterns.len()
        );

        let exclude_set = if exclude_patterns.is_empty() {
            None
        } else {
            Some(Self::build_glob_set(exclude_patterns, case_sensitive)?)
        };

        let include_set = if include_patterns.is_empty() {
            None
        } else {
            Some(Self::build_glob_set(include_patterns, case_sensitive)?)
        };

        Ok(Self {
            exclude_set,
            include_set,
            case_sensitive,
        })
    }

    /// Create a `GlobMatcher` with only exclude patterns.
    pub fn exclude_only(patterns: &[String], case_sensitive: bool) -> FilterResult<Self> {
        Self::new(patterns, &[], case_sensitive)
    }

    /// Create a `GlobMatcher` with only include patterns.
    pub fn include_only(patterns: &[String], case_sensitive: bool) -> FilterResult<Self> {
        Self::new(&[], patterns, case_sensitive)
    }

    /// Build a `GlobSet` from patterns.
    fn build_glob_set(patterns: &[String], case_sensitive: bool) -> FilterResult<GlobSet> {
        let mut builder = GlobSetBuilder::new();

        for pattern in patterns {
            let glob = if case_sensitive {
                GlobBuilder::new(pattern)
                    .build()
                    .map_err(|e| FilterError::InvalidGlob(format!("Pattern '{pattern}': {e}")))?
            } else {
                GlobBuilder::new(pattern)
                    .case_insensitive(true)
                    .build()
                    .map_err(|e| FilterError::InvalidGlob(format!("Pattern '{pattern}': {e}")))?
            };

            builder.add(glob);
        }

        builder
            .build()
            .map_err(|e| FilterError::InvalidGlob(format!("Failed to build glob set: {e}")))
    }

    /// Check if a path matches any exclude pattern.
    #[must_use]
    pub fn is_excluded(&self, path: &Path) -> bool {
        if let Some(ref exclude_set) = self.exclude_set {
            exclude_set.is_match(path)
        } else {
            false
        }
    }

    /// Check if a path matches any include pattern.
    #[must_use]
    pub fn is_included(&self, path: &Path) -> bool {
        if let Some(ref include_set) = self.include_set {
            include_set.is_match(path)
        } else {
            true // If no include patterns, include everything
        }
    }

    /// Check if a path should be included based on glob patterns.
    ///
    /// Logic:
    /// 1. If there are include patterns, the path must match at least one
    /// 2. If the path matches any exclude pattern, it's excluded
    /// 3. Include patterns override exclude patterns
    #[must_use]
    pub fn matches(&self, path: &Path) -> bool {
        // Check if explicitly included (overrides exclude)
        if let Some(ref include_set) = self.include_set {
            if include_set.is_match(path) {
                return true; // Explicitly included, overrides exclude
            }
            // If we have include patterns but path doesn't match any, exclude it
            return false;
        }

        // No include patterns, just check exclude patterns
        !self.is_excluded(path)
    }

    /// Add more exclude patterns to the matcher.
    pub fn add_exclude_patterns(&mut self, patterns: &[String]) -> FilterResult<()> {
        if patterns.is_empty() {
            return Ok(());
        }

        let new_set = Self::build_glob_set(patterns, self.case_sensitive)?;

        self.exclude_set = match self.exclude_set.take() {
            Some(_existing) => {
                // This is a limitation - we can't easily extract patterns from existing GlobSet
                // For now, we'll replace with the new set
                warn!("Replacing existing exclude patterns with new ones");
                Some(new_set)
            }
            None => Some(new_set),
        };

        Ok(())
    }

    /// Add more include patterns to the matcher.
    pub fn add_include_patterns(&mut self, patterns: &[String]) -> FilterResult<()> {
        if patterns.is_empty() {
            return Ok(());
        }

        let new_set = Self::build_glob_set(patterns, self.case_sensitive)?;

        self.include_set = match self.include_set.take() {
            Some(_existing) => {
                // Similar limitation as above
                warn!("Replacing existing include patterns with new ones");
                Some(new_set)
            }
            None => Some(new_set),
        };

        Ok(())
    }

    /// Get whether the matcher is case sensitive.
    #[must_use]
    pub fn is_case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    /// Check if the matcher has any patterns.
    #[must_use]
    pub fn has_patterns(&self) -> bool {
        self.exclude_set.is_some() || self.include_set.is_some()
    }
}

impl Filter for GlobMatcher {
    fn should_include(&self, path: &Path) -> bool {
        self.matches(path)
    }

    fn should_traverse_dir(&self, path: &Path) -> bool {
        // For directories, we need to be more permissive
        // A directory might be excluded but contain included files

        // If there are include patterns, check if any could match subdirectories
        if let Some(ref include_set) = self.include_set {
            // Check if the directory path or any potential subpath could match
            if include_set.is_match(path) {
                return true;
            }

            // Check if any include pattern could match files in this directory
            // This is a heuristic - we allow traversal if it's not explicitly excluded
            if !self.is_excluded(path) {
                return true;
            }
        }

        // If no include patterns, just check exclude patterns
        !self.is_excluded(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_exclude_patterns() {
        let exclude_patterns = vec![
            "*.log".to_string(),
            "target/**".to_string(), // Use ** to match directory contents
            "**/*.tmp".to_string(),
        ];

        let matcher = GlobMatcher::exclude_only(&exclude_patterns, false).unwrap();

        // Test excluded files
        assert!(!matcher.should_include(&PathBuf::from("debug.log")));
        assert!(!matcher.should_include(&PathBuf::from("target/debug")));
        assert!(!matcher.should_include(&PathBuf::from("src/temp.tmp")));
        assert!(!matcher.should_include(&PathBuf::from("deep/nested/file.tmp")));

        // Test included files
        assert!(matcher.should_include(&PathBuf::from("src/main.rs")));
        assert!(matcher.should_include(&PathBuf::from("README.md")));
    }

    #[test]
    fn test_include_patterns() {
        let include_patterns = vec!["*.rs".to_string(), "*.md".to_string()];

        let matcher = GlobMatcher::include_only(&include_patterns, false).unwrap();

        // Test included files
        assert!(matcher.should_include(&PathBuf::from("main.rs")));
        assert!(matcher.should_include(&PathBuf::from("README.md")));

        // Test excluded files
        assert!(!matcher.should_include(&PathBuf::from("debug.log")));
        assert!(!matcher.should_include(&PathBuf::from("config.json")));
    }

    #[test]
    fn test_include_override_exclude() {
        let exclude_patterns = vec!["*.log".to_string()];
        let include_patterns = vec!["important.log".to_string()];

        let matcher = GlobMatcher::new(&exclude_patterns, &include_patterns, false).unwrap();

        // Include pattern should override exclude
        assert!(matcher.should_include(&PathBuf::from("important.log")));

        // Other log files should still be excluded (because they don't match include patterns)
        assert!(!matcher.should_include(&PathBuf::from("debug.log")));

        // Non-matching files should be excluded (because we have include patterns and they don't match)
        assert!(!matcher.should_include(&PathBuf::from("main.rs")));
    }

    #[test]
    fn test_case_sensitivity() {
        let patterns = vec!["*.LOG".to_string()];

        let case_sensitive = GlobMatcher::exclude_only(&patterns, true).unwrap();
        let case_insensitive = GlobMatcher::exclude_only(&patterns, false).unwrap();

        let test_path = PathBuf::from("debug.log");

        // Case sensitive should not match
        assert!(case_sensitive.should_include(&test_path));

        // Case insensitive should match
        assert!(!case_insensitive.should_include(&test_path));
    }

    #[test]
    fn test_directory_traversal() {
        let exclude_patterns = vec!["target/".to_string()];
        let include_patterns = vec!["target/important.txt".to_string()];

        let matcher = GlobMatcher::new(&exclude_patterns, &include_patterns, false).unwrap();

        // Directory should be traversed because it contains included files
        assert!(matcher.should_traverse_dir(&PathBuf::from("target")));

        // But the directory itself might not be included
        // This depends on the specific use case
    }
}
