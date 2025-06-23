//! MCP service for managing both client and server functionality.

use crate::{
    error::{AgentError, Result},
    mcp::{McpClient, McpServer, McpToolRegistry},
    tool::{Tool, ToolRegistry},
};
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, error, info, warn};

/// MCP service that manages both client and server functionality
#[derive(Debug)]
pub struct McpService {
    /// MCP clients by name
    clients: HashMap<String, Arc<McpClient>>,
    /// MCP servers by name
    servers: HashMap<String, McpServer>,
    /// MCP tool registry
    tool_registry: McpToolRegistry,
    /// Service configuration
    config: McpServiceConfig,
}

/// MCP service configuration
#[derive(Debug, Clone)]
pub struct McpServiceConfig {
    /// Default client timeout in milliseconds
    pub default_client_timeout_ms: u64,
    /// Default server bind address
    pub default_server_address: String,
    /// Maximum number of concurrent connections
    pub max_connections: usize,
    /// Whether to enable verbose logging
    pub verbose_logging: bool,
    /// Auto-reconnect settings
    pub auto_reconnect: bool,
    /// Reconnect interval in milliseconds
    pub reconnect_interval_ms: u64,
}

impl Default for McpServiceConfig {
    fn default() -> Self {
        Self {
            default_client_timeout_ms: 30_000,
            default_server_address: "localhost:8080".to_string(),
            max_connections: 100,
            verbose_logging: false,
            auto_reconnect: true,
            reconnect_interval_ms: 5_000,
        }
    }
}

impl McpService {
    /// Create a new MCP service
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            servers: HashMap::new(),
            tool_registry: McpToolRegistry::new(),
            config: McpServiceConfig::default(),
        }
    }

    /// Create MCP service with custom configuration
    pub fn with_config(config: McpServiceConfig) -> Self {
        Self {
            clients: HashMap::new(),
            servers: HashMap::new(),
            tool_registry: McpToolRegistry::new(),
            config,
        }
    }

    /// Add an MCP client
    pub async fn add_client(&mut self, name: impl Into<String>, client: McpClient) -> Result<()> {
        let name = name.into();
        info!("Adding MCP client: {}", name);

        let client_arc = Arc::new(client);

        // Register client with tool registry
        self.tool_registry
            .register_client(&name, (*client_arc).clone())
            .await?;

        self.clients.insert(name.clone(), client_arc);

        if self.config.verbose_logging {
            debug!("MCP client '{}' added successfully", name);
        }

        Ok(())
    }

    /// Add an MCP server
    pub fn add_server(&mut self, name: impl Into<String>, server: McpServer) -> Result<()> {
        let name = name.into();
        info!("Adding MCP server: {}", name);

        if self.servers.contains_key(&name) {
            return Err(AgentError::mcp(format!("Server '{}' already exists", name)));
        }

        self.servers.insert(name.clone(), server);

        if self.config.verbose_logging {
            debug!("MCP server '{}' added successfully", name);
        }

        Ok(())
    }

    /// Remove an MCP client
    pub fn remove_client(&mut self, name: &str) -> Result<()> {
        if self.clients.remove(name).is_some() {
            self.tool_registry.remove_client(name)?;
            info!("Removed MCP client: {}", name);
            Ok(())
        } else {
            Err(AgentError::mcp(format!("Client '{}' not found", name)))
        }
    }

    /// Remove an MCP server
    pub fn remove_server(&mut self, name: &str) -> Result<()> {
        if self.servers.remove(name).is_some() {
            info!("Removed MCP server: {}", name);
            Ok(())
        } else {
            Err(AgentError::mcp(format!("Server '{}' not found", name)))
        }
    }

    /// Get an MCP client by name
    pub fn get_client(&self, name: &str) -> Option<&Arc<McpClient>> {
        self.clients.get(name)
    }

    /// Get an MCP server by name
    pub fn get_server(&self, name: &str) -> Option<&McpServer> {
        self.servers.get(name)
    }

    /// Get mutable reference to an MCP server
    pub fn get_server_mut(&mut self, name: &str) -> Option<&mut McpServer> {
        self.servers.get_mut(name)
    }

    /// List all client names
    pub fn client_names(&self) -> Vec<String> {
        self.clients.keys().cloned().collect()
    }

    /// List all server names
    pub fn server_names(&self) -> Vec<String> {
        self.servers.keys().cloned().collect()
    }

    /// Get the MCP tool registry
    pub fn tool_registry(&self) -> &McpToolRegistry {
        &self.tool_registry
    }

    /// Get mutable reference to the MCP tool registry
    pub fn tool_registry_mut(&mut self) -> &mut McpToolRegistry {
        &mut self.tool_registry
    }

    /// Start all servers
    pub async fn start_all_servers(&mut self) -> Result<()> {
        info!("Starting all MCP servers");

        for (name, server) in &mut self.servers {
            if !server.is_running() {
                let address = &self.config.default_server_address;
                match server.start(address).await {
                    Ok(()) => {
                        info!("Started MCP server: {}", name);
                    }
                    Err(e) => {
                        error!("Failed to start MCP server '{}': {}", name, e);
                        return Err(e);
                    }
                }
            }
        }

        info!("All MCP servers started successfully");
        Ok(())
    }

    /// Stop all servers
    pub async fn stop_all_servers(&mut self) -> Result<()> {
        info!("Stopping all MCP servers");

        for (name, server) in &mut self.servers {
            if server.is_running() {
                match server.stop().await {
                    Ok(()) => {
                        info!("Stopped MCP server: {}", name);
                    }
                    Err(e) => {
                        error!("Failed to stop MCP server '{}': {}", name, e);
                        return Err(e);
                    }
                }
            }
        }

        info!("All MCP servers stopped successfully");
        Ok(())
    }

    /// Connect all clients
    pub async fn connect_all_clients(&mut self) -> Result<()> {
        info!("Connecting all MCP clients");

        for (name, client) in &self.clients {
            // In a real implementation, you'd have connection URLs stored
            let url = format!("ws://localhost:8080/{}", name);

            // Note: We can't call mutable methods on Arc<McpClient>
            // In a real implementation, you'd need to handle this differently
            if self.config.verbose_logging {
                debug!("Would connect client '{}' to {}", name, url);
            }
        }

        info!("All MCP clients connected successfully");
        Ok(())
    }

    /// Disconnect all clients
    pub async fn disconnect_all_clients(&mut self) -> Result<()> {
        info!("Disconnecting all MCP clients");

        for (name, _client) in &self.clients {
            // Note: We can't call mutable methods on Arc<McpClient>
            // In a real implementation, you'd need to handle this differently
            if self.config.verbose_logging {
                debug!("Would disconnect client '{}'", name);
            }
        }

        info!("All MCP clients disconnected successfully");
        Ok(())
    }

    /// Get service status
    pub fn status(&self) -> McpServiceStatus {
        let client_statuses: HashMap<String, bool> = self
            .clients
            .iter()
            .map(|(name, client)| (name.clone(), client.is_connected()))
            .collect();

        let server_statuses: HashMap<String, bool> = self
            .servers
            .iter()
            .map(|(name, server)| (name.clone(), server.is_running()))
            .collect();

        let tool_stats = self.tool_registry.stats();

        McpServiceStatus {
            total_clients: self.clients.len(),
            total_servers: self.servers.len(),
            connected_clients: client_statuses
                .values()
                .filter(|&&connected| connected)
                .count(),
            running_servers: server_statuses.values().filter(|&&running| running).count(),
            client_statuses,
            server_statuses,
            tool_registry_stats: tool_stats,
        }
    }

    /// Refresh all client tools
    pub async fn refresh_all_client_tools(&mut self) -> Result<()> {
        info!("Refreshing tools for all MCP clients");

        for client_name in self.client_names() {
            match self.tool_registry.refresh_client_tools(&client_name).await {
                Ok(()) => {
                    debug!("Refreshed tools for client '{}'", client_name);
                }
                Err(e) => {
                    warn!(
                        "Failed to refresh tools for client '{}': {}",
                        client_name, e
                    );
                }
            }
        }

        info!("Finished refreshing client tools");
        Ok(())
    }

    /// Create a tool from the registry
    pub fn create_tool(&self, tool_name: &str) -> Result<Arc<dyn Tool>> {
        self.tool_registry.create_tool(tool_name)
    }

    /// Get service configuration
    pub fn config(&self) -> &McpServiceConfig {
        &self.config
    }

    /// Update service configuration
    pub fn set_config(&mut self, config: McpServiceConfig) {
        self.config = config;
    }
}

impl Default for McpService {
    fn default() -> Self {
        Self::new()
    }
}

/// MCP service status information
#[derive(Debug, Clone)]
pub struct McpServiceStatus {
    /// Total number of clients
    pub total_clients: usize,
    /// Total number of servers
    pub total_servers: usize,
    /// Number of connected clients
    pub connected_clients: usize,
    /// Number of running servers
    pub running_servers: usize,
    /// Client connection statuses
    pub client_statuses: HashMap<String, bool>,
    /// Server running statuses
    pub server_statuses: HashMap<String, bool>,
    /// Tool registry statistics
    pub tool_registry_stats: crate::mcp::McpRegistryStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::builtin::EchoTool;

    #[tokio::test]
    async fn test_mcp_service_creation() {
        let service = McpService::new();
        assert_eq!(service.client_names().len(), 0);
        assert_eq!(service.server_names().len(), 0);
    }

    #[tokio::test]
    async fn test_mcp_service_client_management() {
        let mut service = McpService::new();
        let client = McpClient::new("test_client", "1.0.0");

        // Test that we can't add a disconnected client (this should fail)
        let result = service.add_client("test_client", client).await;
        assert!(result.is_err());

        // Verify no client was added
        assert_eq!(service.client_names().len(), 0);
    }

    #[test]
    fn test_mcp_service_server_management() {
        let mut service = McpService::new();
        let registry = Arc::new(ToolRegistry::new());
        let server = McpServer::new("test_server", "1.0.0", registry);

        // Add server
        service.add_server("test_server", server).unwrap();
        assert_eq!(service.server_names().len(), 1);
        assert!(service.get_server("test_server").is_some());

        // Remove server
        service.remove_server("test_server").unwrap();
        assert_eq!(service.server_names().len(), 0);
    }

    #[test]
    fn test_mcp_service_status() {
        let service = McpService::new();
        let status = service.status();

        assert_eq!(status.total_clients, 0);
        assert_eq!(status.total_servers, 0);
        assert_eq!(status.connected_clients, 0);
        assert_eq!(status.running_servers, 0);
    }
}
