//! Configuration for file filtering.

use std::path::PathBuf;

/// Configuration for enhanced file filtering.
#[derive(Debug, Clone)]
pub struct FilterConfig {
    /// Whether to respect .gitignore files in the directory tree.
    pub respect_gitignore: bool,

    /// Additional ignore files to consider (e.g., .dockerignore, .eslintignore).
    pub ignore_files: Vec<PathBuf>,

    /// Custom glob patterns to exclude files/directories.
    /// Examples: "*.log", "temp/", "**/*.tmp"
    pub exclude_patterns: Vec<String>,

    /// Custom glob patterns to include files/directories.
    /// These patterns override exclude patterns.
    /// Examples: "!important.log", "src/**/*.rs"
    pub include_patterns: Vec<String>,

    /// File extensions to include (if None, all supported types are included).
    pub include_extensions: Option<Vec<String>>,

    /// File extensions to exclude.
    pub exclude_extensions: Vec<String>,

    /// Whether to exclude hidden files and directories (starting with '.').
    pub exclude_hidden: bool,

    /// Whether to exclude empty files.
    pub exclude_empty: bool,

    /// Maximum file size to process (in bytes).
    pub max_file_size: Option<u64>,

    /// Minimum file size to process (in bytes).
    pub min_file_size: Option<u64>,

    /// Whether to follow symbolic links.
    pub follow_symlinks: bool,

    /// Case sensitivity for pattern matching.
    pub case_sensitive: bool,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            respect_gitignore: true,
            ignore_files: vec![],
            exclude_patterns: vec![
                // Common build artifacts
                "target/".to_string(),
                "build/".to_string(),
                "dist/".to_string(),
                "out/".to_string(),
                // Dependencies
                "node_modules/".to_string(),
                "vendor/".to_string(),
                // Cache directories
                ".cache/".to_string(),
                "**/.fastembed_cache/".to_string(),
                // IDE files
                ".vscode/".to_string(),
                ".idea/".to_string(),
                // OS files
                ".DS_Store".to_string(),
                "Thumbs.db".to_string(),
                // Temporary files
                "*.tmp".to_string(),
                "*.temp".to_string(),
                "*.swp".to_string(),
                "*.swo".to_string(),
                "*~".to_string(),
            ],
            include_patterns: vec![],
            include_extensions: None,
            exclude_extensions: vec![
                // Binary executables
                "exe".to_string(),
                "bin".to_string(),
                "dll".to_string(),
                "so".to_string(),
                "dylib".to_string(),
                // Archives
                "zip".to_string(),
                "tar".to_string(),
                "gz".to_string(),
                "rar".to_string(),
                "7z".to_string(),
                // Images (unless specifically needed)
                "png".to_string(),
                "jpg".to_string(),
                "jpeg".to_string(),
                "gif".to_string(),
                "bmp".to_string(),
                "ico".to_string(),
                // Videos
                "mp4".to_string(),
                "avi".to_string(),
                "mov".to_string(),
                "wmv".to_string(),
                // Audio
                "mp3".to_string(),
                "wav".to_string(),
                "flac".to_string(),
                "ogg".to_string(),
            ],
            exclude_hidden: true,
            exclude_empty: false,
            max_file_size: Some(100 * 1024 * 1024), // 100MB
            min_file_size: None,
            follow_symlinks: false,
            case_sensitive: false,
        }
    }
}

impl FilterConfig {
    /// Create a new filter configuration with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable gitignore respect.
    #[must_use]
    pub fn with_respect_gitignore(mut self, respect: bool) -> Self {
        self.respect_gitignore = respect;
        self
    }

    /// Add additional ignore files.
    #[must_use]
    pub fn with_ignore_files(mut self, files: Vec<PathBuf>) -> Self {
        self.ignore_files = files;
        self
    }

    /// Add exclude patterns.
    #[must_use]
    pub fn with_exclude_patterns(mut self, patterns: Vec<String>) -> Self {
        self.exclude_patterns.extend(patterns);
        self
    }

    /// Add include patterns.
    #[must_use]
    pub fn with_include_patterns(mut self, patterns: Vec<String>) -> Self {
        self.include_patterns.extend(patterns);
        self
    }

    /// Set file extensions to include.
    #[must_use]
    pub fn with_include_extensions(mut self, extensions: Vec<String>) -> Self {
        self.include_extensions = Some(extensions);
        self
    }

    /// Set file extensions to exclude.
    #[must_use]
    pub fn with_exclude_extensions(mut self, extensions: Vec<String>) -> Self {
        self.exclude_extensions = extensions;
        self
    }

    /// Enable or disable hidden file exclusion.
    #[must_use]
    pub fn with_exclude_hidden(mut self, exclude: bool) -> Self {
        self.exclude_hidden = exclude;
        self
    }

    /// Enable or disable empty file exclusion.
    #[must_use]
    pub fn with_exclude_empty(mut self, exclude: bool) -> Self {
        self.exclude_empty = exclude;
        self
    }

    /// Set maximum file size.
    #[must_use]
    pub fn with_max_file_size(mut self, size: u64) -> Self {
        self.max_file_size = Some(size);
        self
    }

    /// Set minimum file size.
    #[must_use]
    pub fn with_min_file_size(mut self, size: u64) -> Self {
        self.min_file_size = Some(size);
        self
    }

    /// Enable or disable symlink following.
    #[must_use]
    pub fn with_follow_symlinks(mut self, follow: bool) -> Self {
        self.follow_symlinks = follow;
        self
    }

    /// Set case sensitivity for pattern matching.
    #[must_use]
    pub fn with_case_sensitive(mut self, sensitive: bool) -> Self {
        self.case_sensitive = sensitive;
        self
    }

    /// Create a minimal configuration for text files only.
    #[must_use]
    pub fn text_files_only() -> Self {
        Self {
            include_extensions: Some(vec![
                "txt".to_string(),
                "md".to_string(),
                "rst".to_string(),
                "org".to_string(),
                "tex".to_string(),
            ]),
            ..Self::default()
        }
    }

    /// Create a configuration for source code files.
    #[must_use]
    pub fn source_code_only() -> Self {
        Self {
            include_extensions: Some(vec![
                // Rust
                "rs".to_string(),
                // Python
                "py".to_string(),
                "pyi".to_string(),
                // JavaScript/TypeScript
                "js".to_string(),
                "ts".to_string(),
                "jsx".to_string(),
                "tsx".to_string(),
                // C/C++
                "c".to_string(),
                "cpp".to_string(),
                "cc".to_string(),
                "cxx".to_string(),
                "h".to_string(),
                "hpp".to_string(),
                // Java
                "java".to_string(),
                // Go
                "go".to_string(),
                // Other
                "rb".to_string(),
                "php".to_string(),
                "cs".to_string(),
                "swift".to_string(),
                "kt".to_string(),
                "scala".to_string(),
            ]),
            ..Self::default()
        }
    }
}
