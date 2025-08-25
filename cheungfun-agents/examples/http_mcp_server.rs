//! HTTP MCP Server Example
//!
//! This example demonstrates how to create an HTTP MCP server using rmcp crate
//! that provides simple tools like add and get_time. The server can be used
//! with the HTTP MCP client to demonstrate tool calling integration with LLMs.
//!
//! Features:
//! - HTTP transport with Axum web server
//! - Simple mathematical operations (add)
//! - Time utilities (get_time)
//! - Real MCP protocol implementation
//! - JSON-RPC over HTTP
//!
//! To run this server:
//! ```bash
//! cargo run --example http_mcp_server
//! ```

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::Json,
    routing::post,
    Router,
};
use rmcp::{model::*, tool, Error as McpError, ServerHandler};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::net::TcpListener;

#[derive(Debug, Deserialize)]
pub struct AddRequest {
    pub a: f64,
    pub b: f64,
}

#[derive(Debug, Deserialize)]
pub struct GetTimeRequest {
    pub timezone: Option<String>,
}

/// MCP Server implementation with tools
#[derive(Clone)]
pub struct McpServer;

impl McpServer {
    /// Add two numbers together
    pub fn add(&self, a: f64, b: f64) -> String {
        let result = a + b;
        format!("{} + {} = {}", a, b, result)
    }

    /// Get current time
    pub fn get_time(&self, timezone: Option<String>) -> String {
        let now = std::time::SystemTime::now();
        match timezone.as_deref() {
            Some("local") => {
                format!("Current local time: {:?}", now)
            }
            Some(tz_name) => {
                format!("Current time in {}: {:?}", tz_name, now)
            }
            None => {
                format!("Current UTC time: {:?}", now)
            }
        }
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

impl McpServer {
    pub fn new() -> Self {
        Self
    }
}

impl ServerHandler for McpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation {
                name: "cheungfun-http-mcp-server".into(),
                version: "1.0.0".into(),
            },
            instructions: Some("HTTP MCP Server providing simple tools like add and get_time for cheungfun agents integration examples.".to_string()),
        }
    }
}

/// JSON-RPC request structure
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    method: String,
    params: Option<Value>,
    id: Option<Value>,
}

/// JSON-RPC response structure
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
    id: Option<Value>,
}

/// JSON-RPC error structure
#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

/// HTTP MCP handler
async fn handle_mcp_request(
    State(server): State<Arc<McpServer>>,
    _headers: HeaderMap,
    Json(request): Json<JsonRpcRequest>,
) -> Result<Json<JsonRpcResponse>, StatusCode> {
    println!(
        "📨 Received MCP request: method={}, id={:?}",
        request.method, request.id
    );

    // Handle different MCP methods
    let result = match request.method.as_str() {
        "initialize" => {
            // Return server capabilities
            let server_info = server.get_info();
            json!({
                "protocolVersion": server_info.protocol_version,
                "capabilities": server_info.capabilities,
                "serverInfo": server_info.server_info,
                "instructions": server_info.instructions
            })
        }
        "tools/list" => {
            // Return available tools
            let tools = vec![
                json!({
                    "name": "add",
                    "description": "Add two numbers together",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "number",
                                "description": "First number to add"
                            },
                            "b": {
                                "type": "number",
                                "description": "Second number to add"
                            }
                        },
                        "required": ["a", "b"]
                    }
                }),
                json!({
                    "name": "get_time",
                    "description": "Get current date and time",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "Timezone (optional, defaults to UTC)"
                            }
                        },
                        "required": []
                    }
                }),
            ];
            json!({ "tools": tools })
        }
        "tools/call" => {
            // Execute tool call
            if let Some(params) = request.params {
                if let Some(tool_name) = params.get("name").and_then(|n| n.as_str()) {
                    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

                    println!("🔧 Executing tool: {}", tool_name);
                    println!(
                        "📥 Arguments: {}",
                        serde_json::to_string_pretty(&arguments)
                            .unwrap_or_else(|_| "{}".to_string())
                    );

                    match tool_name {
                        "add" => {
                            let a = arguments.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let b = arguments.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let result_text = server.add(a, b);

                            println!("📤 Result: {}", result_text);

                            json!({
                                "content": [{
                                    "type": "text",
                                    "text": result_text
                                }],
                                "isError": false
                            })
                        }
                        "get_time" => {
                            let timezone = arguments
                                .get("timezone")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string());

                            let time_str = server.get_time(timezone);

                            println!("📤 Result: {}", time_str);

                            json!({
                                "content": [{
                                    "type": "text",
                                    "text": time_str
                                }],
                                "isError": false
                            })
                        }
                        _ => {
                            return Ok(Json(JsonRpcResponse {
                                jsonrpc: "2.0".to_string(),
                                result: None,
                                error: Some(JsonRpcError {
                                    code: -32601,
                                    message: format!("Unknown tool: {}", tool_name),
                                    data: None,
                                }),
                                id: request.id,
                            }));
                        }
                    }
                } else {
                    return Ok(Json(JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32602,
                            message: "Missing tool name".to_string(),
                            data: None,
                        }),
                        id: request.id,
                    }));
                }
            } else {
                return Ok(Json(JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32602,
                        message: "Missing parameters".to_string(),
                        data: None,
                    }),
                    id: request.id,
                }));
            }
        }
        _ => {
            return Ok(Json(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("Unknown method: {}", request.method),
                    data: None,
                }),
                id: request.id,
            }));
        }
    };

    println!("✅ Sending response for method: {}", request.method);

    Ok(Json(JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        result: Some(result),
        error: None,
        id: request.id,
    }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🌐 HTTP MCP Server - Starting server with simple tools");
    println!("📋 Available tools:");
    println!("   • add - Add two numbers together");
    println!("   • get_time - Get current date and time");
    println!();

    // Create MCP server instance
    let server = Arc::new(McpServer::new());

    // Create Axum router
    let app = Router::new()
        .route("/mcp", post(handle_mcp_request))
        .with_state(server);

    // Start HTTP server
    let listener = TcpListener::bind("127.0.0.1:3000").await?;
    println!("🌐 HTTP MCP Server started on http://127.0.0.1:3000");
    println!("📡 MCP endpoint: http://127.0.0.1:3000/mcp");
    println!("⏹️  Press Ctrl+C to stop");
    println!();

    axum::serve(listener, app).await?;

    Ok(())
}
