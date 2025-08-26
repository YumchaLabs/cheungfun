//! HTTP MCP Client Example with Siumai LLM Integration
//!
//! This example demonstrates how to create an HTTP MCP client that connects to
//! an MCP server, retrieves available tools, and integrates them with siumai
//! for LLM tool calling functionality.
//!
//! Features:
//! - HTTP MCP client implementation
//! - Tool discovery and conversion
//! - LLM integration with tool calling
//! - Complete request-response flow
//! - Real HTTP communication with MCP server
//!
//! To run this example:
//! 1. First start the MCP server: `cargo run --example http_mcp_server`
//! 2. Then run this client: `cargo run --example http_mcp_client`

use serde::{Deserialize, Serialize};
use serde_json::json;
use siumai::prelude::*;
use std::collections::HashMap;

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

/// HTTP MCP Client that communicates with MCP server via HTTP
#[derive(Clone)]
pub struct HttpMcpClient {
    http_client: reqwest::Client,
    server_url: String,
    request_id: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl HttpMcpClient {
    /// Create a new HTTP MCP client
    pub fn new(server_url: String) -> Self {
        Self {
            http_client: reqwest::Client::new(),
            server_url,
            request_id: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(1)),
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
    ) -> Result<serde_json::Value, LlmError> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
            id: self.next_id(),
        };

        println!("📤 Sending MCP request: {}", method);

        let response = self
            .http_client
            .post(&self.server_url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::provider_error("MCP", format!("HTTP request failed: {}", e)))?;

        let json_response: JsonRpcResponse = response.json().await.map_err(|e| {
            LlmError::provider_error("MCP", format!("Failed to parse JSON response: {}", e))
        })?;

        if let Some(error) = json_response.error {
            return Err(LlmError::provider_error(
                "MCP",
                format!("MCP server error: {} (code: {})", error.message, error.code),
            ));
        }

        json_response
            .result
            .ok_or_else(|| LlmError::provider_error("MCP", "No result in MCP response"))
    }

    /// Initialize connection with MCP server
    pub async fn initialize(&self) -> Result<(), LlmError> {
        let params = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "cheungfun-http-mcp-client",
                "version": "1.0.0"
            }
        });

        let _result = self.send_request("initialize", Some(params)).await?;
        println!("✅ MCP client initialized successfully");
        Ok(())
    }

    /// Get available tools from MCP server
    pub async fn get_tools(&self) -> Result<Vec<Tool>, LlmError> {
        let result = self.send_request("tools/list", None).await?;

        let tools_array = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| LlmError::provider_error("MCP", "Invalid tools response format"))?;

        let mut llm_tools = Vec::new();

        for tool_value in tools_array {
            let name = tool_value
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| LlmError::provider_error("MCP", "Tool missing name"))?;

            let description = tool_value
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("No description");

            let input_schema = tool_value
                .get("inputSchema")
                .ok_or_else(|| LlmError::provider_error("MCP", "Tool missing inputSchema"))?;

            // Convert MCP tool schema to siumai Tool
            let llm_tool = Tool::function(
                name.to_string(),
                description.to_string(),
                input_schema.clone(),
            );

            llm_tools.push(llm_tool);
        }

        println!("📋 Retrieved {} tools from MCP server", llm_tools.len());
        Ok(llm_tools)
    }

    /// Execute a tool call via MCP server
    pub async fn call_tool(&self, tool_call: &ToolCall) -> Result<String, LlmError> {
        let function = tool_call
            .function
            .as_ref()
            .ok_or_else(|| LlmError::provider_error("MCP", "Tool call missing function"))?;

        let arguments: serde_json::Value =
            serde_json::from_str(&function.arguments).map_err(|e| {
                LlmError::provider_error("MCP", format!("Invalid tool arguments: {}", e))
            })?;

        let params = json!({
            "name": function.name,
            "arguments": arguments
        });

        println!(
            "🔧 Calling tool: {} with args: {}",
            function.name, arguments
        );

        let result = self.send_request("tools/call", Some(params)).await?;

        // Extract text content from MCP response
        let content = result
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("text"))
            .and_then(|text| text.as_str())
            .ok_or_else(|| LlmError::provider_error("MCP", "Invalid tool call response format"))?;

        println!("✅ Tool call result: {}", content);
        Ok(content.to_string())
    }
}

/// Demonstration of HTTP MCP + LLM integration
pub struct HttpMcpLlmDemo {
    mcp_client: HttpMcpClient,
    llm_client: Option<Box<dyn ChatCapability>>,
}

impl HttpMcpLlmDemo {
    pub async fn new(server_url: String) -> Result<Self, LlmError> {
        // Create HTTP MCP client
        let mcp_client = HttpMcpClient::new(server_url);

        // Initialize MCP connection
        mcp_client.initialize().await?;

        // Try to create LLM client (optional for demo)
        let llm_client = if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            if api_key != "demo-key" && !api_key.is_empty() {
                Some(Box::new(
                    LlmBuilder::new()
                        .openai()
                        .api_key(api_key)
                        .model("gpt-3.5-turbo")
                        .build()
                        .await?,
                ) as Box<dyn ChatCapability>)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            mcp_client,
            llm_client,
        })
    }

    /// Run a complete demo showing HTTP MCP tool integration with LLM
    pub async fn run_demo(&self) -> Result<(), LlmError> {
        println!("🚀 HTTP MCP + LLM Integration Demo");
        println!("===================================");
        println!();

        // Step 1: Get available tools from HTTP MCP server
        println!("📋 Step 1: Discovering tools from HTTP MCP server...");
        let tools = self.mcp_client.get_tools().await?;
        println!("✅ Found {} tools:", tools.len());
        for tool in &tools {
            println!(
                "   • {} - {}",
                tool.function.name, tool.function.description
            );
        }
        println!();

        // Step 2: Test tool calls directly (simulating LLM behavior)
        println!("🔧 Step 2: Testing tool calls via HTTP MCP...");
        let test_tool_calls = vec![
            ToolCall {
                id: "call_1".to_string(),
                r#type: "function".to_string(),
                function: Some(FunctionCall {
                    name: "add".to_string(),
                    arguments: json!({"a": 15, "b": 27}).to_string(),
                }),
            },
            ToolCall {
                id: "call_2".to_string(),
                r#type: "function".to_string(),
                function: Some(FunctionCall {
                    name: "get_time".to_string(),
                    arguments: json!({"timezone": "UTC"}).to_string(),
                }),
            },
        ];

        let mut tool_results = Vec::new();
        for tool_call in &test_tool_calls {
            let result = self.mcp_client.call_tool(tool_call).await?;
            tool_results.push(result);
        }
        println!();

        // Step 3: Show integration with LLM (if available)
        if let Some(ref llm_client) = self.llm_client {
            println!("🤖 Step 3: Integrating with real LLM...");

            // Create messages for the LLM
            let user_message =
                "Please add the numbers 15 and 27, then tell me the current time in UTC.";
            println!("👤 User: {}", user_message);

            let mut messages = vec![ChatMessage::user(user_message).build()];

            // Send request to LLM with tools
            match llm_client
                .chat_with_tools(messages.clone(), Some(tools.clone()))
                .await
            {
                Ok(response) => {
                    if let Some(tool_calls) = response.get_tool_calls() {
                        println!("🔧 LLM requested {} tool calls:", tool_calls.len());

                        // Collect tool results
                        let mut tool_results = Vec::new();

                        for tool_call in tool_calls {
                            if let Some(function) = &tool_call.function {
                                println!("   📞 Calling: {}", function.name);
                                let result = self.mcp_client.call_tool(tool_call).await?;
                                println!("   ✅ Result: {}", result);

                                // Add tool result to messages
                                tool_results.push(
                                    ChatMessage::tool(result.clone(), tool_call.id.clone()).build(),
                                );
                            }
                        }

                        // Add assistant message with tool calls
                        messages.push(
                            ChatMessage::assistant("")
                                .with_tool_calls(tool_calls.clone())
                                .build(),
                        );

                        // Add tool results
                        messages.extend(tool_results);

                        // Get final response from LLM
                        println!("🔄 Getting final response from LLM...");
                        match llm_client
                            .chat_with_tools(messages, Some(tools.clone()))
                            .await
                        {
                            Ok(final_response) => {
                                println!(
                                    "🤖 LLM Final Response: {}",
                                    final_response
                                        .text()
                                        .unwrap_or_else(|| "No response".to_string())
                                );
                            }
                            Err(e) => {
                                println!("⚠️ Error getting final LLM response: {}", e);
                            }
                        }

                        println!("✅ Complete LLM tool calling workflow finished!");
                    } else {
                        println!(
                            "🤖 LLM Response: {}",
                            response.text().unwrap_or_else(|| "No response".to_string())
                        );
                    }
                }
                Err(e) => {
                    println!("⚠️ Error with LLM integration: {}", e);
                    println!("💡 Continuing with tool demonstration...");
                }
            }
        } else {
            println!(
                "💡 Step 3: No LLM client available (set OPENAI_API_KEY for full integration)"
            );
        }
        println!();

        // Step 4: Summary
        println!("📊 Demo Summary:");
        println!("   ✅ HTTP MCP server communication: Working");
        println!("   ✅ Tool discovery: {} tools found", tools.len());
        println!(
            "   ✅ Tool execution: {} calls successful",
            tool_results.len()
        );
        if self.llm_client.is_some() {
            println!("   ✅ LLM integration: Available");
        } else {
            println!("   ⚠️ LLM integration: Not configured");
        }
        println!();

        println!("✨ HTTP MCP integration demo completed!");
        println!("🔄 This demonstrates:");
        println!("   1. HTTP communication with MCP server");
        println!("   2. Real-time tool discovery via JSON-RPC");
        println!("   3. Tool execution with parameter passing");
        println!("   4. Integration with siumai for AI tool calling");

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🌐 HTTP MCP Client - LLM Integration Demo");
    println!("==========================================");
    println!();

    // Check if MCP server is running
    println!("💡 Make sure the HTTP MCP server is running:");
    println!("   cargo run --example http_mcp_server");
    println!("   Server should be available at: http://127.0.0.1:3000/mcp");
    println!();

    let server_url = "http://127.0.0.1:3000/mcp".to_string();

    // Create and run the demo
    match HttpMcpLlmDemo::new(server_url).await {
        Ok(demo) => {
            if let Err(e) = demo.run_demo().await {
                eprintln!("❌ Demo failed: {}", e);
                eprintln!("💡 Make sure the HTTP MCP server is running on port 3000");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to initialize HTTP MCP client: {}", e);
            eprintln!("💡 Make sure the HTTP MCP server is running:");
            eprintln!("   cargo run --example http_mcp_server");
            std::process::exit(1);
        }
    }

    Ok(())
}
