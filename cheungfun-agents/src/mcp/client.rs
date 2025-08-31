//! MCP client implementation using rmcp with HTTP transport.

use crate::{
    error::{AgentError, Result},
    mcp::McpToolExecutionResult,
};
use rmcp::model::{Implementation, Tool as RmcpTool};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, info, warn};

/// JSON-RPC request structure for MCP communication
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    method: String,
    params: Option<serde_json::Value>,
    id: u64,
}

/// JSON-RPC response structure from MCP server
#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
    #[allow(dead_code)]
    id: Option<serde_json::Value>,
}

/// JSON-RPC error structure
#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[allow(dead_code)]
    data: Option<serde_json::Value>,
}

/// HTTP MCP client for connecting to MCP servers via HTTP
#[derive(Debug, Clone)]
pub struct McpClient {
    /// HTTP client for communication
    http_client: reqwest::Client,
    /// Server URL
    server_url: Option<String>,
    /// Client information
    client_info: Implementation,
    /// Connection status
    connected: bool,
    /// Request ID counter
    request_id: Arc<std::sync::atomic::AtomicU64>,
    /// Tools cache
    tools_cache: Option<Vec<RmcpTool>>,
}

/// MCP client handler implementation
#[derive(Debug, Default)]
pub struct McpClientHandler {
    /// Available tools cache
    #[allow(dead_code)]
    tools_cache: Option<Vec<RmcpTool>>,
}

impl McpClient {
    /// Create a new HTTP MCP client
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        let client_info = Implementation {
            name: name.into(),
            version: version.into(),
        };

        Self {
            http_client: reqwest::Client::new(),
            server_url: None,
            client_info,
            connected: false,
            request_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            tools_cache: None,
        }
    }

    /// Generate next request ID
    fn next_id(&self) -> u64 {
        self.request_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    /// Send JSON-RPC request to MCP server
    async fn send_request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let server_url = self
            .server_url
            .as_ref()
            .ok_or_else(|| AgentError::mcp("Client not connected to server"))?;

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
            id: self.next_id(),
        };

        debug!("📤 Sending MCP request: {}", method);

        let response = self
            .http_client
            .post(server_url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AgentError::mcp(format!("HTTP request failed: {e}")))?;

        let json_response: JsonRpcResponse = response
            .json()
            .await
            .map_err(|e| AgentError::mcp(format!("Failed to parse JSON response: {e}")))?;

        if let Some(error) = json_response.error {
            return Err(AgentError::mcp(format!(
                "MCP server error: {} (code: {})",
                error.message, error.code
            )));
        }

        json_response
            .result
            .ok_or_else(|| AgentError::mcp("No result in MCP response"))
    }

    /// Connect to an MCP server
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The client is already connected to a server
    /// - The server URL is invalid or unreachable
    /// - The MCP handshake fails
    /// - Authentication with the server fails
    pub async fn connect(&mut self, server_url: &str) -> Result<()> {
        info!("Connecting to MCP server: {}", server_url);

        self.server_url = Some(server_url.to_string());

        // Initialize connection with MCP server
        let params = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": self.client_info.name,
                "version": self.client_info.version
            }
        });

        self.send_request("initialize", Some(params)).await?;
        self.connected = true;

        info!("✅ Connected to MCP server: {}", server_url);
        Ok(())
    }

    /// Disconnect from the MCP server
    ///
    /// # Errors
    /// Returns an error if disconnection fails
    pub fn disconnect(&mut self) -> Result<()> {
        if self.connected {
            info!("Disconnecting from MCP server");
            self.connected = false;
            self.server_url = None;
            self.tools_cache = None;
        }
        Ok(())
    }

    /// Check if client is connected
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// List available tools from the server
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The client is not connected to the server
    /// - Communication with the server fails
    /// - The server returns an invalid response
    ///
    /// # Panics
    ///
    /// May panic if the internal tool list becomes corrupted during processing
    pub async fn list_tools(&mut self) -> Result<Vec<RmcpTool>> {
        if !self.connected {
            return Err(AgentError::mcp("Client not connected to server"));
        }

        debug!("📋 Listing tools from MCP server");

        // Check cache first
        if let Some(cached_tools) = &self.tools_cache {
            debug!("Returning cached tools ({})", cached_tools.len());
            return Ok(cached_tools.clone());
        }

        // Send tools/list request to MCP server
        let result = self.send_request("tools/list", None).await?;

        let tools_array = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| AgentError::mcp("Invalid tools response format"))?;

        let mut tools = Vec::new();
        for tool_value in tools_array {
            let name = tool_value
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| AgentError::mcp("Tool missing name"))?
                .to_string();

            let description = tool_value
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("No description")
                .to_string();

            let input_schema = tool_value
                .get("inputSchema")
                .ok_or_else(|| AgentError::mcp("Tool missing inputSchema"))?;

            let tool = RmcpTool {
                name: name.into(),
                description: Some(description.into()),
                input_schema: Arc::new(input_schema.as_object().unwrap().clone()),
                output_schema: None,
                annotations: None,
            };

            tools.push(tool);
        }

        // Cache the tools
        self.tools_cache = Some(tools.clone());

        info!("📋 Retrieved {} tools from MCP server", tools.len());
        Ok(tools)
    }

    /// Call a tool on the server
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolExecutionResult> {
        if !self.connected {
            return Err(AgentError::mcp("Client not connected to server"));
        }

        debug!(
            "🔧 Calling MCP tool: {} with args: {:?}",
            tool_name, arguments
        );

        let params = json!({
            "name": tool_name,
            "arguments": arguments
        });

        let result = self.send_request("tools/call", Some(params)).await?;

        // Extract content from MCP response
        let content = result
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("text"))
            .and_then(|text| text.as_str())
            .unwrap_or("");

        let is_error = result
            .get("isError")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        let execution_result = if is_error {
            McpToolExecutionResult {
                content: String::new(),
                success: false,
                error: Some(content.to_string()),
                metadata: HashMap::new(),
            }
        } else {
            McpToolExecutionResult {
                content: content.to_string(),
                success: true,
                error: None,
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("tool".to_string(), serde_json::json!(tool_name));
                    meta.insert("mcp_result".to_string(), result.clone());
                    meta
                },
            }
        };

        if execution_result.success {
            debug!("✅ MCP tool {} executed successfully", tool_name);
        } else {
            warn!(
                "❌ MCP tool {} execution failed: {:?}",
                tool_name, execution_result.error
            );
        }

        Ok(execution_result)
    }

    /// Get client information
    #[must_use]
    pub fn client_info(&self) -> &Implementation {
        &self.client_info
    }

    /// Clear tools cache
    pub fn clear_tools_cache(&mut self) {
        self.tools_cache = None;
        debug!("Cleared tools cache");
    }

    /// Get connection status and statistics
    #[must_use]
    pub fn status(&self) -> McpClientStatus {
        let tools_count = self.tools_cache.as_ref().map_or(0, std::vec::Vec::len);

        McpClientStatus {
            connected: self.connected,
            client_name: self.client_info.name.clone(),
            client_version: self.client_info.version.clone(),
            cached_tools_count: tools_count,
            server_url: self.server_url.clone(),
        }
    }
}

/// MCP client status information
#[derive(Debug, Clone)]
pub struct McpClientStatus {
    /// Whether client is connected
    pub connected: bool,
    /// Client name
    pub client_name: String,
    /// Client version
    pub client_version: String,
    /// Number of cached tools
    pub cached_tools_count: usize,
    /// Server URL if connected
    pub server_url: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mcp_client_creation() {
        let client = McpClient::new("test_client", "1.0.0");
        assert_eq!(client.client_info().name, "test_client");
        assert_eq!(client.client_info().version, "1.0.0");
        assert!(!client.is_connected());
    }

    #[tokio::test]
    async fn test_mcp_client_connection() {
        let mut client = McpClient::new("test_client", "1.0.0");

        // Note: These tests would require a real MCP server running
        // For now, we test the basic structure
        assert!(!client.is_connected());
        assert_eq!(client.status().client_name, "test_client");

        // Test disconnection
        client.disconnect().unwrap();
        assert!(!client.is_connected());
    }

    #[tokio::test]
    async fn test_mcp_client_status() {
        let client = McpClient::new("test_client", "1.0.0");
        let status = client.status();

        assert!(!status.connected);
        assert_eq!(status.client_name, "test_client");
        assert_eq!(status.cached_tools_count, 0);
        assert!(status.server_url.is_none());
    }
}
