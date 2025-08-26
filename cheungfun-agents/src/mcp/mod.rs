//! Model Context Protocol (MCP) integration using rmcp.

use crate::{
    error::{AgentError, Result},
    tool::{Tool, ToolContext, ToolResult},
    types::ToolSchema,
};
use async_trait::async_trait;
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, error, info};

pub mod client;
pub mod server;
pub mod service;

pub use client::McpClient;
pub use server::McpServer;
pub use service::McpService;

/// MCP tool adapter that wraps rmcp tools for use in Cheungfun agents
#[derive(Debug)]
pub struct McpTool {
    /// Tool name
    name: String,
    /// Tool description
    description: String,
    /// MCP client for tool execution
    client: Arc<McpClient>,
    /// Tool schema cache
    schema_cache: Option<ToolSchema>,
}

impl McpTool {
    /// Create a new MCP tool adapter
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        client: Arc<McpClient>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            client,
            schema_cache: None,
        }
    }

    /// Create MCP tool with cached schema
    pub fn with_schema(
        name: impl Into<String>,
        description: impl Into<String>,
        client: Arc<McpClient>,
        schema: ToolSchema,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            client,
            schema_cache: Some(schema),
        }
    }

    /// Get the MCP client
    #[must_use]
    pub fn client(&self) -> &Arc<McpClient> {
        &self.client
    }
}

#[async_trait]
impl Tool for McpTool {
    fn schema(&self) -> ToolSchema {
        if let Some(cached) = &self.schema_cache {
            return cached.clone();
        }

        // Default schema if not cached
        ToolSchema {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "arguments": {
                        "type": "object",
                        "description": "Tool arguments"
                    }
                }
            }),
            output_schema: None,
            dangerous: false,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("mcp_tool".to_string(), serde_json::json!(true));
                meta
            },
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult> {
        debug!("Executing MCP tool: {}", self.name);

        match self.client.call_tool(&self.name, arguments).await {
            Ok(result) => {
                debug!("MCP tool {} executed successfully", self.name);
                let content = result.content.clone();
                Ok(ToolResult::success(content)
                    .with_metadata("mcp_result".to_string(), serde_json::json!(result)))
            }
            Err(e) => {
                error!("MCP tool {} execution failed: {}", self.name, e);
                Ok(ToolResult::error(format!("MCP tool execution failed: {e}")))
            }
        }
    }

    fn capabilities(&self) -> Vec<String> {
        vec!["mcp".to_string(), "remote".to_string()]
    }
}

/// MCP tool registry for managing MCP-based tools
#[derive(Debug)]
pub struct McpToolRegistry {
    /// MCP clients by name
    clients: HashMap<String, Arc<McpClient>>,
    /// Available tools by client
    tools_by_client: HashMap<String, Vec<McpToolInfo>>,
}

/// Information about an MCP tool
#[derive(Debug, Clone)]
pub struct McpToolInfo {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Tool schema
    pub schema: ToolSchema,
    /// Client name that provides this tool
    pub client_name: String,
}

impl McpToolRegistry {
    /// Create a new MCP tool registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            tools_by_client: HashMap::new(),
        }
    }

    /// Register an MCP client
    pub async fn register_client(
        &mut self,
        name: impl Into<String>,
        mut client: McpClient,
    ) -> Result<()> {
        let name = name.into();
        info!("Registering MCP client: {}", name);

        // Get available tools from the client
        let tools = client.list_tools().await.map_err(|e| {
            AgentError::mcp(format!("Failed to list tools from client '{name}': {e}"))
        })?;

        let mut tool_infos = Vec::new();
        for tool in tools {
            let tool_info = McpToolInfo {
                name: tool.name.to_string(),
                description: tool
                    .description
                    .as_ref()
                    .map(std::string::ToString::to_string)
                    .unwrap_or_default(),
                schema: ToolSchema {
                    name: tool.name.to_string(),
                    description: tool
                        .description
                        .as_ref()
                        .map(std::string::ToString::to_string)
                        .unwrap_or_default(),
                    input_schema: serde_json::Value::Object(tool.input_schema.as_ref().clone()),
                    output_schema: None,
                    dangerous: false,
                    metadata: HashMap::new(),
                },
                client_name: name.clone(),
            };
            tool_infos.push(tool_info);
        }

        self.clients.insert(name.clone(), Arc::new(client));
        self.tools_by_client.insert(name.clone(), tool_infos);

        info!(
            "Registered MCP client '{}' with {} tools",
            name,
            self.tools_by_client[&name].len()
        );
        Ok(())
    }

    /// Get all available tools
    #[must_use]
    pub fn list_tools(&self) -> Vec<McpToolInfo> {
        self.tools_by_client
            .values()
            .flat_map(|tools| tools.iter().cloned())
            .collect()
    }

    /// Get tools from a specific client
    #[must_use]
    pub fn list_tools_by_client(&self, client_name: &str) -> Vec<McpToolInfo> {
        self.tools_by_client
            .get(client_name)
            .cloned()
            .unwrap_or_default()
    }

    /// Create a tool adapter for an MCP tool
    pub fn create_tool(&self, tool_name: &str) -> Result<Arc<dyn Tool>> {
        // Find the tool and its client
        for (client_name, tools) in &self.tools_by_client {
            if let Some(tool_info) = tools.iter().find(|t| t.name == tool_name) {
                let client = self
                    .clients
                    .get(client_name)
                    .ok_or_else(|| AgentError::mcp(format!("Client '{client_name}' not found")))?;

                let mcp_tool = McpTool::with_schema(
                    tool_info.name.clone(),
                    tool_info.description.clone(),
                    Arc::clone(client),
                    tool_info.schema.clone(),
                );

                return Ok(Arc::new(mcp_tool));
            }
        }

        Err(AgentError::mcp(format!("Tool '{tool_name}' not found")))
    }

    /// Get all registered client names
    #[must_use]
    pub fn client_names(&self) -> Vec<String> {
        self.clients.keys().cloned().collect()
    }

    /// Get a client by name
    #[must_use]
    pub fn get_client(&self, name: &str) -> Option<&Arc<McpClient>> {
        self.clients.get(name)
    }

    /// Remove a client and its tools
    pub fn remove_client(&mut self, name: &str) -> Result<()> {
        if self.clients.remove(name).is_some() {
            self.tools_by_client.remove(name);
            info!("Removed MCP client: {}", name);
            Ok(())
        } else {
            Err(AgentError::mcp(format!("Client '{name}' not found")))
        }
    }

    /// Refresh tools for a specific client
    pub async fn refresh_client_tools(&mut self, client_name: &str) -> Result<()> {
        let client = self
            .clients
            .get_mut(client_name)
            .ok_or_else(|| AgentError::mcp(format!("Client '{client_name}' not found")))?;

        let tools = Arc::get_mut(client)
            .ok_or_else(|| AgentError::mcp("Cannot get mutable reference to client".to_string()))?
            .list_tools()
            .await
            .map_err(|e| {
                AgentError::mcp(format!(
                    "Failed to refresh tools for client '{client_name}': {e}"
                ))
            })?;

        let mut tool_infos = Vec::new();
        for tool in tools {
            let tool_info = McpToolInfo {
                name: tool.name.to_string(),
                description: tool
                    .description
                    .as_ref()
                    .map(std::string::ToString::to_string)
                    .unwrap_or_default(),
                schema: ToolSchema {
                    name: tool.name.to_string(),
                    description: tool
                        .description
                        .as_ref()
                        .map(std::string::ToString::to_string)
                        .unwrap_or_default(),
                    input_schema: serde_json::Value::Object(tool.input_schema.as_ref().clone()),
                    output_schema: None,
                    dangerous: false,
                    metadata: HashMap::new(),
                },
                client_name: client_name.to_string(),
            };
            tool_infos.push(tool_info);
        }

        self.tools_by_client
            .insert(client_name.to_string(), tool_infos);
        info!("Refreshed tools for MCP client '{}'", client_name);
        Ok(())
    }

    /// Get registry statistics
    #[must_use]
    pub fn stats(&self) -> McpRegistryStats {
        let total_tools = self.tools_by_client.values().map(std::vec::Vec::len).sum();
        let tools_by_client = self
            .tools_by_client
            .iter()
            .map(|(name, tools)| (name.clone(), tools.len()))
            .collect();

        McpRegistryStats {
            total_clients: self.clients.len(),
            total_tools,
            tools_by_client,
        }
    }
}

impl Default for McpToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// MCP registry statistics
#[derive(Debug, Clone)]
pub struct McpRegistryStats {
    /// Total number of registered clients
    pub total_clients: usize,
    /// Total number of available tools
    pub total_tools: usize,
    /// Tools count by client
    pub tools_by_client: HashMap<String, usize>,
}

/// MCP tool execution result
#[derive(Debug, Clone, serde::Serialize)]
pub struct McpToolExecutionResult {
    /// Result content
    pub content: String,
    /// Whether execution was successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_tool_registry_creation() {
        let registry = McpToolRegistry::new();
        assert_eq!(registry.client_names().len(), 0);
        assert_eq!(registry.list_tools().len(), 0);
    }

    #[test]
    fn test_mcp_tool_schema() {
        // This would require a mock MCP client for proper testing
        // For now, we'll test the basic structure
        let registry = McpToolRegistry::new();
        let stats = registry.stats();
        assert_eq!(stats.total_clients, 0);
        assert_eq!(stats.total_tools, 0);
    }
}
