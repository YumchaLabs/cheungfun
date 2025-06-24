//! HTTP client tool for making web requests.

use crate::{
    error::{AgentError, Result},
    tool::{Tool, ToolContext, ToolResult, create_simple_schema, string_param},
    types::ToolSchema,
};
use async_trait::async_trait;
use reqwest;
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Duration;

/// HTTP tool for making web requests
#[derive(Debug, Clone)]
pub struct HttpTool {
    name: String,
    /// HTTP client with configured timeouts
    client: reqwest::Client,
    /// Whether to allow requests to localhost/private IPs
    allow_local: bool,
    /// Maximum response size in bytes
    max_response_size: usize,
}

impl HttpTool {
    /// Create a new HTTP tool with default settings
    #[must_use]
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("cheungfun-agents/0.1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            name: "http".to_string(),
            client,
            allow_local: false,
            max_response_size: 1024 * 1024, // 1MB
        }
    }

    /// Create an HTTP tool that allows requests to localhost
    #[must_use]
    pub fn with_local_access() -> Self {
        let mut tool = Self::new();
        tool.allow_local = true;
        tool
    }

    /// Create an HTTP tool with custom timeout
    #[must_use]
    pub fn with_timeout(timeout_secs: u64) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .user_agent("cheungfun-agents/0.1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            name: "http".to_string(),
            client,
            allow_local: false,
            max_response_size: 1024 * 1024,
        }
    }

    /// Set maximum response size
    #[must_use]
    pub fn with_max_response_size(mut self, size: usize) -> Self {
        self.max_response_size = size;
        self
    }

    /// Validate URL for security
    fn validate_url(&self, url: &str) -> Result<reqwest::Url> {
        let parsed = reqwest::Url::parse(url)
            .map_err(|e| AgentError::tool(&self.name, format!("Invalid URL: {e}")))?;

        // Check scheme
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(AgentError::tool(
                &self.name,
                "Only HTTP and HTTPS URLs are allowed",
            ));
        }

        // Check for localhost/private IPs if not allowed
        if !self.allow_local {
            if let Some(host) = parsed.host_str() {
                if host == "localhost"
                    || host == "127.0.0.1"
                    || host == "::1"
                    || host.starts_with("192.168.")
                    || host.starts_with("10.")
                    || host.starts_with("172.16.")
                    || host.starts_with("172.17.")
                    || host.starts_with("172.18.")
                    || host.starts_with("172.19.")
                    || host.starts_with("172.2")
                    || host.starts_with("172.30.")
                    || host.starts_with("172.31.")
                {
                    return Err(AgentError::tool(
                        &self.name,
                        "Requests to localhost/private IPs are not allowed",
                    ));
                }
            }
        }

        Ok(parsed)
    }
}

impl Default for HttpTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for HttpTool {
    fn schema(&self) -> ToolSchema {
        let mut properties = HashMap::new();

        properties.insert(
            "method".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "HTTP method",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
                "default": "GET"
            }),
        );

        let (url_schema, _) = string_param("URL to request", true);
        properties.insert("url".to_string(), url_schema);

        properties.insert(
            "headers".to_string(),
            serde_json::json!({
                "type": "object",
                "description": "HTTP headers to include",
                "additionalProperties": {
                    "type": "string"
                }
            }),
        );

        properties.insert(
            "body".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Request body (for POST, PUT, PATCH)"
            }),
        );

        properties.insert(
            "follow_redirects".to_string(),
            serde_json::json!({
                "type": "boolean",
                "description": "Whether to follow redirects",
                "default": true
            }),
        );

        ToolSchema {
            name: self.name.clone(),
            description: format!(
                "Make HTTP requests to web services. {}",
                if self.allow_local {
                    "Allows requests to localhost and private IPs."
                } else {
                    "Blocks requests to localhost and private IPs for security."
                }
            ),
            input_schema: create_simple_schema(properties, vec!["url".to_string()]),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "status": {
                        "type": "integer",
                        "description": "HTTP status code"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Response headers"
                    },
                    "body": {
                        "type": "string",
                        "description": "Response body"
                    },
                    "url": {
                        "type": "string",
                        "description": "Final URL (after redirects)"
                    }
                }
            })),
            dangerous: false,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert(
                    "allow_local".to_string(),
                    serde_json::json!(self.allow_local),
                );
                meta.insert(
                    "max_response_size".to_string(),
                    serde_json::json!(self.max_response_size),
                );
                meta
            },
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult> {
        #[derive(Deserialize)]
        struct HttpArgs {
            #[serde(default = "default_method")]
            method: String,
            url: String,
            #[serde(default)]
            headers: HashMap<String, String>,
            body: Option<String>,
            #[serde(default = "default_follow_redirects")]
            follow_redirects: bool,
        }

        fn default_method() -> String {
            "GET".to_string()
        }

        fn default_follow_redirects() -> bool {
            true
        }

        let args: HttpArgs = serde_json::from_value(arguments)
            .map_err(|e| AgentError::tool(&self.name, format!("Invalid arguments: {e}")))?;

        // Validate URL
        let url = self.validate_url(&args.url)?;

        // Build request
        let method = args.method.to_uppercase();
        let mut request_builder = match method.as_str() {
            "GET" => self.client.get(url.clone()),
            "POST" => self.client.post(url.clone()),
            "PUT" => self.client.put(url.clone()),
            "DELETE" => self.client.delete(url.clone()),
            "PATCH" => self.client.patch(url.clone()),
            "HEAD" => self.client.head(url.clone()),
            _ => {
                return Ok(ToolResult::error(format!(
                    "Unsupported HTTP method: {method}"
                )));
            }
        };

        // Add headers
        for (key, value) in args.headers {
            request_builder = request_builder.header(&key, &value);
        }

        // Add body if provided
        if let Some(body) = args.body {
            request_builder = request_builder.body(body);
        }

        // Configure redirects - Note: redirect policy is set on client, not request
        // For now, we'll handle this at the client level in the constructor

        // Execute request
        match request_builder.send().await {
            Ok(response) => {
                let status = response.status().as_u16();
                let final_url = response.url().to_string();

                // Get headers
                let mut response_headers = HashMap::new();
                for (key, value) in response.headers() {
                    if let Ok(value_str) = value.to_str() {
                        response_headers.insert(key.to_string(), value_str.to_string());
                    }
                }

                // Get body with size limit
                match response.text().await {
                    Ok(body) => {
                        if body.len() > self.max_response_size {
                            return Ok(ToolResult::error(format!(
                                "Response too large: {} bytes (max: {})",
                                body.len(),
                                self.max_response_size
                            )));
                        }

                        let content = format!(
                            "HTTP {} {}\n\nBody:\n{}",
                            status,
                            reqwest::StatusCode::from_u16(status)
                                .map(|s| s.canonical_reason().unwrap_or("Unknown"))
                                .unwrap_or("Unknown"),
                            body
                        );

                        Ok(ToolResult::success(content)
                            .with_metadata("status".to_string(), serde_json::json!(status))
                            .with_metadata(
                                "headers".to_string(),
                                serde_json::json!(response_headers),
                            )
                            .with_metadata("body".to_string(), serde_json::json!(body))
                            .with_metadata("url".to_string(), serde_json::json!(final_url))
                            .with_metadata("method".to_string(), serde_json::json!(method)))
                    }
                    Err(e) => Ok(ToolResult::error(format!(
                        "Failed to read response body: {e}"
                    ))),
                }
            }
            Err(e) => {
                if e.is_timeout() {
                    Ok(ToolResult::error("Request timed out"))
                } else if e.is_connect() {
                    Ok(ToolResult::error("Failed to connect to server"))
                } else {
                    Ok(ToolResult::error(format!("HTTP request failed: {e}")))
                }
            }
        }
    }

    fn capabilities(&self) -> Vec<String> {
        vec!["http".to_string(), "web".to_string(), "api".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_validation() {
        let tool = HttpTool::new();

        // Valid URLs
        assert!(tool.validate_url("https://example.com").is_ok());
        assert!(tool.validate_url("http://api.example.com/data").is_ok());

        // Invalid schemes
        assert!(tool.validate_url("ftp://example.com").is_err());
        assert!(tool.validate_url("file:///etc/passwd").is_err());

        // Localhost (should be blocked by default)
        assert!(tool.validate_url("http://localhost:8080").is_err());
        assert!(tool.validate_url("http://127.0.0.1").is_err());
        assert!(tool.validate_url("http://192.168.1.1").is_err());
    }

    #[test]
    fn test_url_validation_with_local_access() {
        let tool = HttpTool::with_local_access();

        // Localhost should be allowed
        assert!(tool.validate_url("http://localhost:8080").is_ok());
        assert!(tool.validate_url("http://127.0.0.1").is_ok());
    }

    #[tokio::test]
    async fn test_http_tool_schema() {
        let tool = HttpTool::new();
        let schema = tool.schema();

        assert_eq!(schema.name, "http");
        assert!(!schema.description.is_empty());
        assert!(!schema.dangerous);
    }

    // Note: Integration tests with real HTTP requests would require a test server
    // or mocking, which is beyond the scope of this basic implementation
}
