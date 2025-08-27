//! File-based document loader.

use async_trait::async_trait;
use cheungfun_core::traits::Loader;
use cheungfun_core::{Document, Result as CoreResult};
use std::path::{Path, PathBuf};
use tracing::{debug, error, info, warn};

use super::{utils, LoaderConfig};
use crate::error::{IndexingError, Result};

/// Loads documents from individual files.
///
/// This loader can handle various file formats including plain text, PDF, Word documents,
/// and more. It extracts text content and adds relevant metadata.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::loaders::FileLoader;
/// use cheungfun_core::traits::Loader;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let loader = FileLoader::new("document.txt")?;
///     let documents = loader.load().await?;
///     
///     for doc in documents {
///         println!("Loaded document: {} characters", doc.content.len());
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct FileLoader {
    /// Path to the file to load.
    path: PathBuf,
    /// Configuration for the loader.
    config: LoaderConfig,
}

impl FileLoader {
    /// Create a new file loader for the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to load
    ///
    /// # Errors
    ///
    /// Returns an error if the path does not exist or is not a file.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(IndexingError::file_not_found(path.display().to_string()));
        }

        if !path.is_file() {
            return Err(IndexingError::configuration(format!(
                "Path is not a file: {}",
                path.display()
            )));
        }

        Ok(Self {
            path,
            config: LoaderConfig::default(),
        })
    }

    /// Create a new file loader with custom configuration.
    pub fn with_config<P: AsRef<Path>>(path: P, config: LoaderConfig) -> Result<Self> {
        let mut loader = Self::new(path)?;
        loader.config = config;
        Ok(loader)
    }

    /// Get the file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the loader configuration.
    #[must_use]
    pub fn config(&self) -> &LoaderConfig {
        &self.config
    }

    /// Check if the file should be processed based on configuration.
    fn should_process_file(&self) -> bool {
        // Check file extension
        if let Some(extension) = self.path.extension().and_then(|e| e.to_str()) {
            let ext = extension.to_lowercase();

            // Check exclude list
            if self.config.exclude_extensions.contains(&ext) {
                debug!(
                    "Skipping file due to excluded extension: {}",
                    self.path.display()
                );
                return false;
            }

            // Check include list if specified
            if let Some(ref include_exts) = self.config.include_extensions {
                if !include_exts.contains(&ext) {
                    debug!("Skipping file not in include list: {}", self.path.display());
                    return false;
                }
            }
        }

        true
    }

    /// Extract text content from the file based on its type.
    async fn extract_text_content(&self) -> Result<String> {
        let content_type = utils::detect_content_type(&self.path);

        match content_type.as_deref() {
            Some(
                "text/plain" | "text/markdown" | "text/html" | "text/csv" | "application/json"
                | "application/xml" | "text/x-rust" | "text/x-python" | "text/javascript"
                | "text/typescript" | "text/x-java" | "text/x-csharp" | "text/x-c++" | "text/x-c"
                | "text/x-go" | "text/x-ruby" | "text/x-php" | "text/x-swift" | "text/x-kotlin"
                | "text/x-scala" | "text/x-haskell" | "text/x-clojure" | "text/x-erlang"
                | "text/x-elixir" | "text/x-lua" | "text/x-shellscript" | "text/x-sql" | "text/css"
                | "text/x-yaml" | "text/x-toml",
            ) => self.extract_text_file().await,
            Some("application/pdf") => self.extract_pdf_content().await,
            Some(
                "application/msword"
                | "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ) => self.extract_word_content().await,
            _ => {
                // Try to read as text file as fallback
                warn!(
                    "Unknown file type for {}, attempting to read as text",
                    self.path.display()
                );
                self.extract_text_file().await
            }
        }
    }

    /// Extract content from plain text files.
    async fn extract_text_file(&self) -> Result<String> {
        debug!("Reading text file: {}", self.path.display());
        utils::read_text_file(&self.path).await
    }

    /// Extract text content from PDF files.
    async fn extract_pdf_content(&self) -> Result<String> {
        debug!("Extracting text from PDF: {}", self.path.display());

        let bytes = tokio::fs::read(&self.path)
            .await
            .map_err(IndexingError::Io)?;

        // Use pdf-extract crate to extract text
        let text = pdf_extract::extract_text_from_mem(&bytes)
            .map_err(|e| IndexingError::text_extraction(format!("PDF extraction failed: {e}")))?;

        Ok(text)
    }

    /// Extract text content from Word documents.
    async fn extract_word_content(&self) -> Result<String> {
        debug!(
            "Extracting text from Word document: {}",
            self.path.display()
        );

        let bytes = tokio::fs::read(&self.path)
            .await
            .map_err(IndexingError::Io)?;

        // Use docx-rs crate for .docx files
        if self.path.extension().and_then(|e| e.to_str()) == Some("docx") {
            let docx = docx_rs::read_docx(&bytes)
                .map_err(|e| IndexingError::text_extraction(format!("DOCX parsing failed: {e}")))?;

            // Extract text from all paragraphs
            let mut text = String::new();
            for paragraph in docx.document.children {
                if let docx_rs::DocumentChild::Paragraph(para) = paragraph {
                    for run in para.children {
                        if let docx_rs::ParagraphChild::Run(run) = run {
                            for child in run.children {
                                if let docx_rs::RunChild::Text(text_elem) = child {
                                    text.push_str(&text_elem.text);
                                }
                            }
                        }
                    }
                    text.push('\n');
                }
            }
            Ok(text)
        } else {
            // For .doc files, we'd need a different approach
            // For now, return an error
            Err(IndexingError::unsupported_format(
                "Legacy .doc files not supported yet",
            ))
        }
    }
}

#[async_trait]
impl Loader for FileLoader {
    async fn load(&self) -> CoreResult<Vec<Document>> {
        info!("Loading file: {}", self.path.display());

        // Check if file should be processed
        if !self.should_process_file() {
            return Ok(vec![]);
        }

        // Check file size if configured
        if let Some(max_size) = self.config.max_file_size {
            match utils::get_file_size(&self.path).await {
                Ok(size) if size > max_size => {
                    warn!(
                        "Skipping file {} (size {} > max {})",
                        self.path.display(),
                        size,
                        max_size
                    );
                    return Ok(vec![]);
                }
                Err(e) => {
                    error!("Failed to get file size for {}: {}", self.path.display(), e);
                    if !self.config.continue_on_error {
                        return Err(e.into());
                    }
                    return Ok(vec![]);
                }
                _ => {}
            }
        }

        // Extract text content
        let content = match self.extract_text_content().await {
            Ok(content) => content,
            Err(e) => {
                error!(
                    "Failed to extract content from {}: {}",
                    self.path.display(),
                    e
                );
                if !self.config.continue_on_error {
                    return Err(e.into());
                }
                return Ok(vec![]);
            }
        };

        // Create document with metadata
        let content_type = utils::detect_content_type(&self.path);
        let file_size = utils::get_file_size(&self.path).await.ok();
        let document =
            utils::create_document_from_file(content, &self.path, content_type, file_size);

        debug!("Successfully loaded document from {}", self.path.display());
        Ok(vec![document])
    }

    fn name(&self) -> &'static str {
        "FileLoader"
    }

    async fn health_check(&self) -> CoreResult<()> {
        if !self.path.exists() {
            return Err(IndexingError::file_not_found(self.path.display().to_string()).into());
        }

        if !self.path.is_file() {
            return Err(IndexingError::configuration(format!(
                "Path is not a file: {}",
                self.path.display()
            ))
            .into());
        }

        // Try to read file metadata
        tokio::fs::metadata(&self.path)
            .await
            .map_err(IndexingError::Io)?;

        Ok(())
    }

    async fn metadata(&self) -> CoreResult<std::collections::HashMap<String, serde_json::Value>> {
        let mut metadata = std::collections::HashMap::new();

        metadata.insert(
            "loader_type".to_string(),
            serde_json::Value::String("file".to_string()),
        );
        metadata.insert(
            "file_path".to_string(),
            serde_json::Value::String(self.path.display().to_string()),
        );

        if let Some(content_type) = utils::detect_content_type(&self.path) {
            metadata.insert(
                "content_type".to_string(),
                serde_json::Value::String(content_type),
            );
        }

        if let Ok(size) = utils::get_file_size(&self.path).await {
            metadata.insert(
                "file_size".to_string(),
                serde_json::Value::Number(size.into()),
            );
        }

        Ok(metadata)
    }
}
