//! Example demonstrating MCP (Model Context Protocol) integration in Cheungfun
//!
//! This example shows how to:
//! 1. Create MCP clients and servers
//! 2. Register tools with MCP servers
//! 3. Use MCP service to manage multiple clients and servers
//! 4. Execute tools through MCP protocol

use cheungfun_agents::{
    error::Result,
    mcp::{McpClient, McpServer, McpService},
    tool::{ToolRegistry, builtin::EchoTool},
};
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Cheungfun MCP Integration Example");
    println!("=====================================");

    // Example 1: Create and use MCP Server
    println!("\n📡 Example 1: MCP Server");
    example_mcp_server().await?;

    // Example 2: Create and use MCP Client
    println!("\n📱 Example 2: MCP Client");
    example_mcp_client().await?;

    // Example 3: Use MCP Service to manage multiple clients and servers
    println!("\n🔧 Example 3: MCP Service");
    example_mcp_service().await?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Example demonstrating MCP Server usage
async fn example_mcp_server() -> Result<()> {
    // Create a tool registry and add some tools
    let mut tool_registry = ToolRegistry::new();
    let echo_tool = Arc::new(EchoTool::new());
    tool_registry.register(echo_tool as Arc<dyn cheungfun_agents::tool::Tool>)?;

    // Create MCP server
    let mut server = McpServer::new("example-server", "1.0.0", Arc::new(tool_registry));

    println!(
        "  📋 Server info: {:?}",
        server.server_info().server_info.name
    );
    println!("  🔧 Available tools: {:?}", server.available_tools());
    println!("  📊 Server stats: {:?}", server.stats());

    // Start server (in a real scenario, this would bind to a port)
    server.start("localhost:8080").await?;
    println!("  ✅ Server started on localhost:8080");

    // Stop server
    server.stop().await?;
    println!("  🛑 Server stopped");

    Ok(())
}

/// Example demonstrating MCP Client usage
async fn example_mcp_client() -> Result<()> {
    // Create MCP client
    let client = McpClient::new("example-client", "1.0.0");

    println!("  📋 Client info: {:?}", client.client_info().name);
    println!("  🔗 Connection status: {}", client.is_connected());
    println!("  📊 Client status: {:?}", client.status());

    // Note: In a real scenario, you would connect to an actual MCP server
    // client.connect("ws://localhost:8080").await?;
    // let tools = client.list_tools().await?;
    // println!("  🔧 Available tools: {:?}", tools);

    Ok(())
}

/// Example demonstrating MCP Service usage
async fn example_mcp_service() -> Result<()> {
    let mut service = McpService::new();

    // Create and add a server
    let tool_registry = Arc::new(ToolRegistry::new());
    let server = McpServer::new("service-server", "1.0.0", tool_registry);
    service.add_server("main-server", server)?;

    // Create a client (note: this will fail to add since it's not connected)
    let client = McpClient::new("service-client", "1.0.0");
    let result = service.add_client("main-client", client).await;
    println!(
        "  ⚠️  Adding disconnected client (expected to fail): {:?}",
        result.is_err()
    );

    // Check service status
    let status = service.status();
    println!("  📊 Service status:");
    println!("    - Total clients: {}", status.total_clients);
    println!("    - Total servers: {}", status.total_servers);
    println!("    - Connected clients: {}", status.connected_clients);
    println!("    - Running servers: {}", status.running_servers);

    // List servers
    println!("  📋 Server names: {:?}", service.server_names());

    Ok(())
}
