# Getting Started with Cheungfun Agents

Welcome to Cheungfun Agents! This guide will help you get up and running with the powerful agent framework.

## 🚀 Installation

Add Cheungfun Agents to your `Cargo.toml`:

```toml
[dependencies]
cheungfun-agents = { path = "../cheungfun-agents" }
tokio = { version = "1.0", features = ["full"] }
```

## 📚 Core Concepts

### Agents
Agents are intelligent entities that can execute tasks, use tools, and participate in workflows. They implement the `Agent` trait and can be customized for specific use cases.

### Tools
Tools extend agent capabilities by providing specific functions like file operations, HTTP requests, or mathematical calculations. All tools implement the `Tool` trait.

### Workflows
Workflows orchestrate multiple agents to complete complex tasks through a series of coordinated steps with dependency management.

### MCP Integration
The Model Context Protocol (MCP) enables agents to connect to external services and share tools across different systems.

## 🎯 Your First Agent

Let's create a simple agent:

```rust
use cheungfun_agents::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Create an agent with built-in tools
    let agent = AgentBuilder::assistant()
        .name("my_first_agent")
        .description("My first Cheungfun agent")
        .tool(Arc::new(EchoTool::new()))
        .build()?;

    // Create and execute a task
    let task = Task::new("Echo 'Hello, Cheungfun!'");
    let response = agent.execute(&task).await?;
    
    println!("Agent response: {}", response.content);
    Ok(())
}
```

## 🛠️ Adding Tools

Agents become more powerful with tools:

```rust
use cheungfun_agents::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let agent = AgentBuilder::assistant()
        .name("tool_agent")
        .tool(Arc::new(EchoTool::new()))
        .tool(Arc::new(HttpTool::new()))
        .tool(Arc::new(FileTool::new()))
        .build()?;

    // The agent can now use echo, HTTP, and file tools
    let task = Task::new("List the files in the current directory");
    let response = agent.execute(&task).await?;
    
    println!("Response: {}", response.content);
    Ok(())
}
```

## 🎭 Multi-Agent Workflows

Create complex workflows with multiple agents:

```rust
use cheungfun_agents::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Create specialized agents
    let researcher = AgentBuilder::researcher()
        .name("researcher")
        .build()?;
    
    let writer = AgentBuilder::assistant()
        .name("writer")
        .build()?;

    // Create orchestrator and register agents
    let mut orchestrator = AgentOrchestrator::new();
    orchestrator.register_agent(Arc::new(researcher))?;
    orchestrator.register_agent(Arc::new(writer))?;

    // Create workflow
    let workflow = Workflow::builder()
        .name("Research and Write")
        .step(create_step("research", "Research Topic", researcher.id()))
        .step(create_step_with_deps(
            "write", 
            "Write Article", 
            writer.id(), 
            vec!["research".to_string()]
        ))
        .variable("topic", serde_json::json!("AI agents"))
        .build()?;

    // Execute workflow
    let result = orchestrator.execute_workflow(workflow).await?;
    println!("Workflow status: {:?}", result.status);
    
    Ok(())
}
```

## 🔌 MCP Integration

Connect to external services using MCP:

```rust
use cheungfun_agents::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create MCP service
    let mut mcp_service = McpService::new();
    
    // Add MCP client (connects to external MCP server)
    let client = McpClient::new("external_tools", "1.0.0");
    mcp_service.add_client("external", client).await?;
    
    // Create MCP server (exposes your tools)
    let server = McpServerBuilder::new()
        .name("my_server")
        .tool(Arc::new(EchoTool::new()))
        .build()?;
    mcp_service.add_server("my_server", server)?;
    
    // Use MCP tools in agents
    let tool_registry = mcp_service.tool_registry();
    let available_tools = tool_registry.list_tools();
    
    println!("Available MCP tools: {}", available_tools.len());
    Ok(())
}
```

## 🧠 RAG-Enhanced Agents

Create knowledge-enhanced agents with RAG:

```rust
use cheungfun_agents::prelude::*;
use cheungfun_query::engine::QueryEngine;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Assuming you have a QueryEngine from cheungfun-query
    // let query_engine = Arc::new(your_query_engine);
    // let tool_registry = Arc::new(ToolRegistry::new());
    
    // let rag_agent = RagAgent::new(
    //     AgentConfig::default(),
    //     query_engine,
    //     tool_registry,
    // );
    
    // The RAG agent can now answer questions using your knowledge base
    println!("RAG agent would use your knowledge base here!");
    Ok(())
}
```

## 🔧 Configuration

### Agent Configuration

```rust
let config = AgentConfig {
    name: "custom_agent".to_string(),
    description: Some("A customized agent".to_string()),
    capabilities: AgentCapabilities {
        supports_tools: true,
        supports_streaming: true,
        max_context_length: Some(8192),
        ..Default::default()
    },
    max_execution_time_ms: Some(30_000),
    verbose: true,
    ..Default::default()
};
```

### Tool Security

```rust
// Read-only file tool
let safe_file_tool = FileTool::new();

// File tool with write access (dangerous)
let write_file_tool = FileTool::with_write_access();

// HTTP tool that blocks localhost
let safe_http_tool = HttpTool::new();

// HTTP tool that allows localhost
let local_http_tool = HttpTool::with_local_access();
```

## 📊 Monitoring and Health

```rust
// Check agent health
let health = agent.health_check().await?;
println!("Agent status: {:?}", health.status);
println!("Metrics: {:?}", health.metrics);

// Monitor orchestrator
let stats = orchestrator.stats();
println!("Workflows executed: {}", stats.total_workflows);
println!("Success rate: {:.2}%", 
    stats.successful_workflows as f64 / stats.total_workflows as f64 * 100.0
);
```

## 🚨 Error Handling

```rust
match agent.execute(&task).await {
    Ok(response) => {
        println!("Success: {}", response.content);
    }
    Err(AgentError::Timeout { operation, timeout_ms }) => {
        println!("Operation '{}' timed out after {}ms", operation, timeout_ms);
    }
    Err(AgentError::Tool { tool_name, message }) => {
        println!("Tool '{}' failed: {}", tool_name, message);
    }
    Err(e) => {
        println!("Other error: {}", e);
    }
}
```

## 🎯 Best Practices

1. **Start Simple**: Begin with basic agents and gradually add complexity
2. **Use Presets**: Leverage built-in agent presets for common use cases
3. **Tool Security**: Be careful with dangerous tools like file write operations
4. **Error Handling**: Always handle errors appropriately
5. **Monitoring**: Use health checks and statistics for production deployments
6. **Timeouts**: Set appropriate timeouts for long-running operations
7. **Testing**: Test your agents thoroughly with various inputs

## 📖 Next Steps

- Explore the [examples](../examples/) directory for more complex scenarios
- Read the [API documentation](api.md) for detailed interface information
- Check out the integration with [cheungfun-query](../../cheungfun-query) for RAG capabilities
- Learn about [MCP protocol](https://modelcontextprotocol.io/) for external integrations

## 🆘 Getting Help

- Check the [README](../README.md) for overview and features
- Look at [examples](../examples/) for practical usage patterns
- Review [API docs](api.md) for detailed interface documentation
- Open an issue on GitHub for bugs or feature requests

Happy coding with Cheungfun Agents! 🤖✨
