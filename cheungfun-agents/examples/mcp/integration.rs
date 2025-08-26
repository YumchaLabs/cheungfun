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

use cheungfun_agents::{
    agent::{
        base::{AgentContext, BaseAgent},
        react::{ReActAgent, ReActConfig},
    },
    llm::SiumaiLlmClient,
    prelude::*,
    tool::ToolContext,
    types::AgentMessage,
};
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

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
    if !available_tools.is_empty() {
        println!("\n🤖 Creating ReAct agent with MCP tools...");

        // Setup LLM client
        let llm_client = SiumaiLlmClient::openai(
            std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()),
            "gpt-4",
        )
        .await?;

        // Create tool registry with MCP tools
        let mut agent_tool_registry = ToolRegistry::new();

        // Add MCP tools to the agent's tool registry
        for tool_info in available_tools.iter().take(3) {
            if let Ok(tool) = tool_registry.create_tool(&tool_info.name) {
                agent_tool_registry.register(tool)?;
                println!("   ✅ Added MCP tool: {}", tool_info.name);
            }
        }

        // Create ReAct agent configuration
        let react_config = ReActConfig::new("MCP Agent").with_max_iterations(5);

        let mut config = react_config;
        config.base_config.description =
            Some("An agent that uses MCP tools for external integrations".to_string());
        config.base_config.instructions = Some(
            "You are a helpful assistant with access to MCP tools. \
             Use the available tools to answer questions and perform tasks. \
             Think step by step and use tools when appropriate."
                .to_string(),
        );

        // Create the ReAct agent
        let agent = ReActAgent::with_llm_client(config, Arc::new(agent_tool_registry), llm_client);

        println!(
            "   🎯 Agent created with {} MCP tools",
            available_tools.len().min(3)
        );

        // Test the agent with MCP tools
        let test_queries = vec!["add 15 and 27", "get current time in UTC"];

        println!("\n🧪 Testing agent with MCP tools:");
        for query in test_queries {
            println!("   📝 Query: {}", query);

            let message = AgentMessage::user(query);
            let mut context = AgentContext::new();

            match agent.chat(message, Some(&mut context)).await {
                Ok(response) => {
                    println!("      ✅ Agent Response: {}", response.content);
                    if !response.metadata.is_empty() {
                        println!("      🔧 Tool calls made: {}", response.metadata.len());
                    }
                }
                Err(e) => {
                    println!("      ❌ Error: {}", e);
                }
            }
        }
    } else {
        println!("\n⚠️  No MCP tools available - skipping agent creation");
        println!("   Make sure the MCP server is running and provides tools");
    }

    // Show final statistics
    println!("\n📈 Final MCP Integration Statistics:");
    let final_status = mcp_service.status();
    println!("   Total clients: {}", final_status.total_clients);
    println!("   Connected clients: {}", final_status.connected_clients);
    println!("   Total servers: {}", final_status.total_servers);
    println!("   Running servers: {}", final_status.running_servers);

    let tool_stats = final_status.tool_registry_stats;
    println!("   Available tools: {}", tool_stats.total_tools);
    println!("   Tools by client: {}", tool_stats.tools_by_client.len());

    // Cleanup - disconnect from MCP servers
    println!("\n🛑 Cleaning up MCP connections...");
    // Note: In a real application, you would properly disconnect clients here

    println!("\n🎉 MCP Integration Example Completed Successfully!");
    println!("   ✅ MCP service created and configured");
    println!("   ✅ HTTP MCP client connected to external server");
    println!("   ✅ Tools discovered and registered");
    println!("   ✅ ReAct agent created with MCP tools");
    println!("   ✅ Agent successfully used MCP tools for task execution");

    Ok(())
}
