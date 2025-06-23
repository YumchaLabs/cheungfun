//! MCP (Model Context Protocol) integration example.
//!
//! This example demonstrates the complete MCP integration workflow:
//! 1. HTTP MCP client connecting to external MCP servers
//! 2. Tool discovery and registration
//! 3. Agent creation with MCP tools
//! 4. Task execution using MCP tools
//!
//! To run this example:
//! 1. Start the HTTP MCP server: `cargo run --example http_mcp_server`
//! 2. Run this integration example: `cargo run --example mcp_integration`

use cheungfun_agents::prelude::*;
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();

    println!("🔌 Cheungfun Agents - MCP Integration Example");
    println!("==============================================");

    // Create MCP service
    let mut mcp_service = McpService::new();
    println!("✅ Created MCP service");

    // Create HTTP MCP clients that connect to real MCP servers
    let mut http_client = McpClient::new("http_mcp_client", "1.0.0");

    println!("📱 Created HTTP MCP client");

    // Connect to HTTP MCP server
    match http_client.connect("http://127.0.0.1:3000/mcp").await {
        Ok(_) => {
            println!("✅ Connected to HTTP MCP server");

            // Add the connected client to service
            mcp_service.add_client("http_server", http_client).await?;
        }
        Err(e) => {
            println!("❌ Failed to connect to HTTP MCP server: {}", e);
            println!("💡 Make sure to start the server first:");
            println!("   cargo run --example http_mcp_server");
            return Ok(());
        }
    }

    println!("🔗 Added HTTP MCP client to service");

    // Note: In this example, we focus on the HTTP MCP client integration
    // The server part is handled by the separate http_mcp_server example

    // Show service status
    let status = mcp_service.status();
    println!("\n📊 MCP Service Status:");
    println!("   Total clients: {}", status.total_clients);
    println!("   Connected clients: {}", status.connected_clients);

    // List available tools from MCP registry
    let tool_registry = mcp_service.tool_registry();
    let available_tools = tool_registry.list_tools();

    println!("\n🛠️  Available MCP Tools:");
    for tool_info in &available_tools {
        println!(
            "   - {} (from {}): {}",
            tool_info.name, tool_info.client_name, tool_info.description
        );
    }

    // Create tools from MCP registry
    if !available_tools.is_empty() {
        println!("\n🔧 Testing MCP tools:");

        for tool_info in available_tools.iter().take(2) {
            // Test first 2 tools
            match tool_registry.create_tool(&tool_info.name) {
                Ok(tool) => {
                    println!("   ✅ Created tool: {}", tool.name());

                    // Test the tool with sample arguments
                    let context = ToolContext::new();
                    let test_args = match tool_info.name.as_str() {
                        "add" => serde_json::json!({"a": 15, "b": 27}),
                        "get_time" => serde_json::json!({"timezone": "UTC"}),
                        _ => serde_json::json!({}),
                    };

                    match tool.execute(test_args, &context).await {
                        Ok(result) => {
                            println!("      Result: {}", result.content);
                            println!("      Success: {}", result.success);
                        }
                        Err(e) => {
                            println!("      ❌ Execution failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("   ❌ Failed to create tool {}: {}", tool_info.name, e);
                }
            }
        }
    }

    // Create an agent that uses MCP tools
    let mut agent_builder = AgentBuilder::assistant()
        .name("mcp_agent")
        .description("An agent that uses MCP tools");

    // Add MCP tools to the agent
    for tool_info in available_tools.iter().take(3) {
        if let Ok(tool) = tool_registry.create_tool(&tool_info.name) {
            agent_builder = agent_builder.tool(tool);
        }
    }

    let agent = agent_builder.build()?;
    println!("\n🤖 Created agent with MCP tools");
    println!("   Agent tools: {:?}", agent.tools());

    // Test the agent with MCP tools
    let test_tasks = vec![
        Task::builder()
            .name("Addition Test")
            .input("add 15 and 27")
            .build()?,
        Task::builder()
            .name("Time Test")
            .input("get current time in UTC")
            .build()?,
    ];

    println!("\n🧪 Testing agent with MCP tools:");
    for task in test_tasks {
        println!("   📝 Task: {}", task.name);
        match agent.execute(&task).await {
            Ok(response) => {
                println!("      ✅ Response: {}", response.content);
                println!("      Tool calls: {}", response.stats.tool_calls_count);
            }
            Err(e) => {
                println!("      ❌ Error: {}", e);
            }
        }
    }

    // Show final statistics
    println!("\n📈 Final MCP Statistics:");
    let final_status = mcp_service.status();
    println!("   Total clients: {}", final_status.total_clients);
    println!("   Connected clients: {}", final_status.connected_clients);

    // Cleanup - disconnect from MCP servers
    println!("\n🛑 Disconnecting from MCP servers...");

    println!("\n🎉 MCP integration example completed!");
    Ok(())
}
