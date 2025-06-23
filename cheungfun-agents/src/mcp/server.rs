//! MCP server implementation using rmcp.

use crate::{
    error::{AgentError, Result},
    tool::{Tool, ToolContext, ToolRegistry},
};
use rmcp::{
    ServerHandler,
    model::{
        CallToolRequestParam, CallToolResult, Content, Implementation, ListToolsResult, ServerInfo,
        Tool as RmcpTool,
    },
    service::{RequestContext, RoleServer},
};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// MCP server for exposing Cheungfun tools via MCP protocol
#[derive(Debug)]
pub struct McpServer {
    /// Server handler
    handler: McpServerHandler,
    /// Server information
    server_info: ServerInfo,
    /// Whether server is running
    running: bool,
}

/// MCP server handler implementation
#[derive(Debug, Clone)]
pub struct McpServerHandler {
    /// Tool registry for available tools
    tool_registry: Arc<ToolRegistry>,
    /// Server statistics
    stats: ServerStats,
}

/// Server statistics
#[derive(Debug, Default, Clone)]
pub struct ServerStats {
    /// Total requests handled
    pub total_requests: usize,
    /// Total tool calls
    pub total_tool_calls: usize,
    /// Successful tool calls
    pub successful_tool_calls: usize,
    /// Failed tool calls
    pub failed_tool_calls: usize,
    /// Connected clients count
    pub connected_clients: usize,
}

impl McpServer {
    /// Create a new MCP server
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        tool_registry: Arc<ToolRegistry>,
    ) -> Self {
        let server_info = ServerInfo {
            protocol_version: rmcp::model::ProtocolVersion::default(),
            capabilities: rmcp::model::ServerCapabilities::default(),
            server_info: Implementation {
                name: name.into(),
                version: version.into(),
            },
            instructions: Some("Cheungfun MCP Server providing AI agent tools".to_string()),
        };

        let handler = McpServerHandler {
            tool_registry,
            stats: ServerStats::default(),
        };

        Self {
            handler,
            server_info,
            running: false,
        }
    }

    /// Start the MCP server
    pub async fn start(&mut self, bind_address: &str) -> Result<()> {
        if self.running {
            return Err(AgentError::mcp("Server is already running"));
        }

        info!("Starting MCP server on: {}", bind_address);

        // In a real implementation, this would start the actual server
        // For now, we'll simulate starting the server
        self.running = true;

        info!("MCP server started successfully");
        Ok(())
    }

    /// Stop the MCP server
    pub async fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Err(AgentError::mcp("Server is not running"));
        }

        info!("Stopping MCP server");
        self.running = false;

        info!("MCP server stopped");
        Ok(())
    }

    /// Check if server is running
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Get server information
    pub fn server_info(&self) -> &ServerInfo {
        &self.server_info
    }

    /// Get server statistics
    pub fn stats(&self) -> ServerStats {
        self.handler.stats.clone()
    }

    /// Get available tools
    pub fn available_tools(&self) -> Vec<String> {
        self.handler.tool_registry.tool_names()
    }

    /// Add a tool to the server
    pub fn add_tool(&mut self, tool: Arc<dyn Tool>) -> Result<()> {
        Arc::get_mut(&mut self.handler.tool_registry)
            .ok_or_else(|| AgentError::mcp("Cannot get mutable reference to tool registry"))?
            .register(tool)
            .map_err(|e| AgentError::mcp(format!("Failed to add tool: {e}")))?;

        debug!("Added tool to MCP server");
        Ok(())
    }

    /// Remove a tool from the server
    pub fn remove_tool(&mut self, tool_name: &str) -> Result<()> {
        Arc::get_mut(&mut self.handler.tool_registry)
            .ok_or_else(|| AgentError::mcp("Cannot get mutable reference to tool registry"))?
            .unregister(tool_name)
            .map_err(|e| AgentError::mcp(format!("Failed to remove tool: {e}")))?;

        debug!("Removed tool '{}' from MCP server", tool_name);
        Ok(())
    }

    /// Get server status
    pub fn status(&self) -> McpServerStatus {
        let stats = self.stats();
        let tools = self.available_tools();

        McpServerStatus {
            running: self.running,
            server_name: self.server_info.server_info.name.clone(),
            server_version: self.server_info.server_info.version.clone(),
            available_tools_count: tools.len(),
            stats,
        }
    }
}

/// MCP server status information
#[derive(Debug, Clone)]
pub struct McpServerStatus {
    /// Whether server is running
    pub running: bool,
    /// Server name
    pub server_name: String,
    /// Server version
    pub server_version: String,
    /// Number of available tools
    pub available_tools_count: usize,
    /// Server statistics
    pub stats: ServerStats,
}

impl ServerHandler for McpServerHandler {
    fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParamInner>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<
        Output = std::result::Result<ListToolsResult, rmcp::model::ErrorData>,
    > + Send
    + '_ {
        async move {
            debug!("Handling list_tools request");

            let schemas = self.tool_registry.schemas();
            let tools: Vec<RmcpTool> = schemas
                .into_iter()
                .map(|schema| RmcpTool {
                    name: schema.name.into(),
                    description: schema.description.into(),
                    input_schema: Arc::new(schema.input_schema.as_object().unwrap().clone()),
                })
                .collect();

            debug!("Returning {} tools", tools.len());

            Ok(ListToolsResult {
                tools,
                next_cursor: None,
            })
        }
    }

    fn call_tool(
        &self,
        request: CallToolRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<
        Output = std::result::Result<CallToolResult, rmcp::model::ErrorData>,
    > + Send
    + '_ {
        async move {
            debug!("Handling call_tool request for: {}", request.name);

            // Create tool context
            let context =
                ToolContext::new().with_data("mcp_request".to_string(), serde_json::json!(true));

            // Execute the tool
            match self
                .tool_registry
                .execute(
                    &request.name,
                    serde_json::Value::Object(request.arguments.unwrap_or_default()),
                    &context,
                )
                .await
            {
                Ok(result) => {
                    if result.success {
                        debug!("Tool '{}' executed successfully", request.name);

                        Ok(CallToolResult {
                            content: vec![Content::text(result.content)],
                            is_error: Some(false),
                        })
                    } else {
                        warn!(
                            "Tool '{}' execution failed: {:?}",
                            request.name, result.error
                        );

                        Ok(CallToolResult {
                            content: vec![Content::text(
                                result.error.unwrap_or_else(|| "Unknown error".to_string()),
                            )],
                            is_error: Some(true),
                        })
                    }
                }
                Err(e) => {
                    error!("Tool '{}' execution error: {}", request.name, e);

                    // Convert AgentError to ErrorData
                    Err(rmcp::model::ErrorData {
                        code: rmcp::model::ErrorCode::INTERNAL_ERROR,
                        message: e.to_string().into(),
                        data: None,
                    })
                }
            }
        }
    }

    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: rmcp::model::ProtocolVersion::default(),
            capabilities: rmcp::model::ServerCapabilities::default(),
            server_info: Implementation {
                name: "cheungfun-mcp-server".to_string(),
                version: "1.0.0".to_string(),
            },
            instructions: Some("Cheungfun MCP Server providing AI agent tools".to_string()),
        }
    }
}

impl McpServerHandler {
    /// Get mutable reference to tool registry
    pub fn tool_registry_mut(&mut self) -> &mut ToolRegistry {
        Arc::get_mut(&mut self.tool_registry).expect("Tool registry should be exclusively owned")
    }

    /// Update client connection count
    pub fn set_connected_clients(&mut self, count: usize) {
        self.stats.connected_clients = count;
    }
}

/// Builder for MCP server configuration
#[derive(Debug)]
pub struct McpServerBuilder {
    name: Option<String>,
    version: Option<String>,
    tool_registry: Option<Arc<ToolRegistry>>,
    tools: Vec<Arc<dyn Tool>>,
}

impl Default for McpServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl McpServerBuilder {
    /// Create a new MCP server builder
    pub fn new() -> Self {
        Self {
            name: None,
            version: None,
            tool_registry: None,
            tools: Vec::new(),
        }
    }

    /// Set server name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set server version
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Use existing tool registry
    pub fn tool_registry(mut self, registry: Arc<ToolRegistry>) -> Self {
        self.tool_registry = Some(registry);
        self
    }

    /// Add a tool
    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools
    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Build the MCP server
    pub fn build(self) -> Result<McpServer> {
        let name = self
            .name
            .unwrap_or_else(|| "cheungfun-mcp-server".to_string());
        let version = self.version.unwrap_or_else(|| "1.0.0".to_string());

        let tool_registry = if let Some(registry) = self.tool_registry {
            registry
        } else {
            let mut registry = ToolRegistry::new();
            for tool in self.tools {
                registry.register(tool).map_err(|e| {
                    AgentError::configuration(format!("Failed to register tool: {e}"))
                })?;
            }
            Arc::new(registry)
        };

        Ok(McpServer::new(name, version, tool_registry))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::builtin::EchoTool;

    #[tokio::test]
    async fn test_mcp_server_creation() {
        let registry = Arc::new(ToolRegistry::new());
        let server = McpServer::new("test_server", "1.0.0", registry);

        assert_eq!(server.server_info().server_info.name, "test_server");
        assert_eq!(server.server_info().server_info.version, "1.0.0");
        assert!(!server.is_running());
    }

    #[tokio::test]
    async fn test_mcp_server_lifecycle() {
        let registry = Arc::new(ToolRegistry::new());
        let mut server = McpServer::new("test_server", "1.0.0", registry);

        // Test starting server
        server.start("localhost:8080").await.unwrap();
        assert!(server.is_running());

        // Test stopping server
        server.stop().await.unwrap();
        assert!(!server.is_running());
    }

    #[test]
    fn test_mcp_server_builder() {
        let echo_tool = Arc::new(EchoTool::new());

        let server = McpServerBuilder::new()
            .name("test_server")
            .version("1.0.0")
            .tool(echo_tool)
            .build()
            .unwrap();

        assert_eq!(server.server_info().server_info.name, "test_server");
        assert!(server.available_tools().contains(&"echo".to_string()));
    }

    #[test]
    fn test_server_stats() {
        let registry = Arc::new(ToolRegistry::new());
        let server = McpServer::new("test_server", "1.0.0", registry);
        let stats = server.stats();

        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.total_tool_calls, 0);
        assert_eq!(stats.connected_clients, 0);
    }
}
