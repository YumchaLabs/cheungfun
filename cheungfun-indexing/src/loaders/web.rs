//! Web-based document loader.

use async_trait::async_trait;
use cheungfun_core::traits::Loader;
use cheungfun_core::{Document, Result as CoreResult};
use reqwest::Client;
use scraper::{Html, Selector};
use std::time::Duration;
use tracing::{debug, error, info, warn};

use crate::error::{IndexingError, Result};

/// Loads documents from web URLs.
///
/// This loader can fetch content from web pages and extract text content.
/// It supports HTML parsing and can extract clean text from web pages.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::loaders::WebLoader;
/// use cheungfun_core::traits::Loader;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let loader = WebLoader::new("https://example.com")?;
///     let documents = loader.load().await?;
///     
///     for doc in documents {
///         println!("Loaded web document: {} characters", doc.content.len());
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct WebLoader {
    /// URL to load from.
    url: String,
    /// HTTP client for making requests.
    client: Client,
    /// Configuration for web loading.
    config: WebLoaderConfig,
}

/// Configuration for web loading operations.
#[derive(Debug, Clone)]
pub struct WebLoaderConfig {
    /// Request timeout in seconds.
    pub timeout_seconds: u64,
    /// User agent string to use.
    pub user_agent: String,
    /// Whether to extract only text content (vs. including HTML).
    pub extract_text_only: bool,
    /// CSS selectors to extract content from (if empty, extracts from body).
    pub content_selectors: Vec<String>,
    /// CSS selectors to exclude from content extraction.
    pub exclude_selectors: Vec<String>,
    /// Maximum content length to process.
    pub max_content_length: Option<usize>,
    /// Whether to follow redirects.
    pub follow_redirects: bool,
}

impl Default for WebLoaderConfig {
    fn default() -> Self {
        Self {
            timeout_seconds: 30,
            user_agent: "Cheungfun-Indexing/1.0".to_string(),
            extract_text_only: true,
            content_selectors: vec!["body".to_string()],
            exclude_selectors: vec![
                "script".to_string(),
                "style".to_string(),
                "nav".to_string(),
                "header".to_string(),
                "footer".to_string(),
                ".advertisement".to_string(),
                ".ads".to_string(),
            ],
            max_content_length: Some(1024 * 1024), // 1MB
            follow_redirects: true,
        }
    }
}

impl WebLoader {
    /// Create a new web loader for the specified URL.
    ///
    /// # Arguments
    ///
    /// * `url` - URL to load content from
    ///
    /// # Errors
    ///
    /// Returns an error if the URL is invalid.
    pub fn new<S: Into<String>>(url: S) -> Result<Self> {
        let url = url.into();

        // Validate URL
        if url::Url::parse(&url).is_err() {
            return Err(IndexingError::configuration(format!(
                "Invalid URL: {}",
                url
            )));
        }

        let config = WebLoaderConfig::default();
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .user_agent(&config.user_agent)
            .redirect(if config.follow_redirects {
                reqwest::redirect::Policy::limited(10)
            } else {
                reqwest::redirect::Policy::none()
            })
            .build()
            .map_err(|e| {
                IndexingError::configuration(format!("Failed to create HTTP client: {}", e))
            })?;

        Ok(Self {
            url,
            client,
            config,
        })
    }

    /// Create a new web loader with custom configuration.
    pub fn with_config<S: Into<String>>(url: S, config: WebLoaderConfig) -> Result<Self> {
        let url = url.into();

        // Validate URL
        if url::Url::parse(&url).is_err() {
            return Err(IndexingError::configuration(format!(
                "Invalid URL: {}",
                url
            )));
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .user_agent(&config.user_agent)
            .redirect(if config.follow_redirects {
                reqwest::redirect::Policy::limited(10)
            } else {
                reqwest::redirect::Policy::none()
            })
            .build()
            .map_err(|e| {
                IndexingError::configuration(format!("Failed to create HTTP client: {}", e))
            })?;

        Ok(Self {
            url,
            client,
            config,
        })
    }

    /// Get the URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Get the loader configuration.
    pub fn config(&self) -> &WebLoaderConfig {
        &self.config
    }

    /// Fetch content from the URL.
    async fn fetch_content(&self) -> Result<String> {
        debug!("Fetching content from URL: {}", self.url);

        let response = self
            .client
            .get(&self.url)
            .send()
            .await
            .map_err(IndexingError::Network)?;

        if !response.status().is_success() {
            return Err(IndexingError::generic(format!(
                "HTTP error {}: {}",
                response.status(),
                self.url
            )));
        }

        let content = response.text().await.map_err(IndexingError::Network)?;

        // Check content length
        if let Some(max_len) = self.config.max_content_length {
            if content.len() > max_len {
                warn!(
                    "Content length {} exceeds maximum {}, truncating",
                    content.len(),
                    max_len
                );
                return Ok(content.chars().take(max_len).collect());
            }
        }

        Ok(content)
    }

    /// Extract text content from HTML.
    fn extract_text_from_html(&self, html: &str) -> Result<String> {
        debug!("Extracting text from HTML content");

        let document = Html::parse_document(html);
        let mut extracted_text = String::new();

        // Create selectors for content extraction
        let content_selectors: Vec<Selector> = self
            .config
            .content_selectors
            .iter()
            .filter_map(|s| Selector::parse(s).ok())
            .collect();

        let exclude_selectors: Vec<Selector> = self
            .config
            .exclude_selectors
            .iter()
            .filter_map(|s| Selector::parse(s).ok())
            .collect();

        // If no valid content selectors, use body as default
        let default_selectors = if content_selectors.is_empty() {
            vec![Selector::parse("body").unwrap()]
        } else {
            content_selectors
        };

        for selector in default_selectors {
            for element in document.select(&selector) {
                // Check if this element should be excluded
                let should_exclude = exclude_selectors
                    .iter()
                    .any(|exclude_sel| element.select(exclude_sel).next().is_some());

                if !should_exclude {
                    let text = element.text().collect::<Vec<_>>().join(" ");
                    if !text.trim().is_empty() {
                        extracted_text.push_str(&text);
                        extracted_text.push('\n');
                    }
                }
            }
        }

        // Clean up the text
        let cleaned_text = extracted_text
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        Ok(cleaned_text)
    }

    /// Create a document from web content.
    fn create_document(&self, content: String, content_type: Option<String>) -> Document {
        let mut doc = Document::new(content);

        // Add web-specific metadata
        doc.metadata.insert(
            "source".to_string(),
            serde_json::Value::String(self.url.clone()),
        );
        doc.metadata.insert(
            "source_type".to_string(),
            serde_json::Value::String("web".to_string()),
        );
        doc.metadata.insert(
            "loader".to_string(),
            serde_json::Value::String("WebLoader".to_string()),
        );

        if let Some(content_type) = content_type {
            doc.metadata.insert(
                "content_type".to_string(),
                serde_json::Value::String(content_type),
            );
        }

        // Add timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        doc.metadata.insert(
            "fetched_at".to_string(),
            serde_json::Value::Number(now.into()),
        );

        // Try to extract domain from URL
        if let Ok(parsed_url) = url::Url::parse(&self.url) {
            if let Some(domain) = parsed_url.domain() {
                doc.metadata.insert(
                    "domain".to_string(),
                    serde_json::Value::String(domain.to_string()),
                );
            }
            doc.metadata.insert(
                "scheme".to_string(),
                serde_json::Value::String(parsed_url.scheme().to_string()),
            );
        }

        doc
    }
}

#[async_trait]
impl Loader for WebLoader {
    async fn load(&self) -> CoreResult<Vec<Document>> {
        info!("Loading content from URL: {}", self.url);

        // Fetch content
        let content = match self.fetch_content().await {
            Ok(content) => content,
            Err(e) => {
                error!("Failed to fetch content from {}: {}", self.url, e);
                return Err(e.into());
            }
        };

        // Process content based on type
        let processed_content = if self.config.extract_text_only {
            // Try to extract text from HTML
            match self.extract_text_from_html(&content) {
                Ok(text) => text,
                Err(e) => {
                    warn!("Failed to extract text from HTML, using raw content: {}", e);
                    content
                }
            }
        } else {
            content
        };

        // Create document
        let content_type = if self.config.extract_text_only {
            Some("text/plain".to_string())
        } else {
            Some("text/html".to_string())
        };

        let document = self.create_document(processed_content, content_type);

        debug!("Successfully loaded document from {}", self.url);
        Ok(vec![document])
    }

    fn name(&self) -> &'static str {
        "WebLoader"
    }

    async fn health_check(&self) -> CoreResult<()> {
        // Try to make a HEAD request to check if URL is accessible
        let response = self
            .client
            .head(&self.url)
            .send()
            .await
            .map_err(IndexingError::Network)?;

        if !response.status().is_success() {
            return Err(IndexingError::generic(format!(
                "URL not accessible: {} (status: {})",
                self.url,
                response.status()
            ))
            .into());
        }

        Ok(())
    }

    async fn metadata(&self) -> CoreResult<std::collections::HashMap<String, serde_json::Value>> {
        let mut metadata = std::collections::HashMap::new();

        metadata.insert(
            "loader_type".to_string(),
            serde_json::Value::String("web".to_string()),
        );
        metadata.insert(
            "url".to_string(),
            serde_json::Value::String(self.url.clone()),
        );
        metadata.insert(
            "extract_text_only".to_string(),
            serde_json::Value::Bool(self.config.extract_text_only),
        );

        // Try to extract domain
        if let Ok(parsed_url) = url::Url::parse(&self.url) {
            if let Some(domain) = parsed_url.domain() {
                metadata.insert(
                    "domain".to_string(),
                    serde_json::Value::String(domain.to_string()),
                );
            }
        }

        Ok(metadata)
    }
}
