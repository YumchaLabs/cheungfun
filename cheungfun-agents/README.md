# Cheungfun Agents

🤖 **Advanced Agent Framework with MCP Integration for Cheungfun**

Cheungfun Agents is a powerful, production-ready agent framework that provides intelligent agents with tool calling capabilities, workflow orchestration, and deep integration with the Model Context Protocol (MCP). Built on top of the Cheungfun RAG framework, it enables the creation of sophisticated AI applications with multi-agent coordination.

## ✨ Features

### 🎯 Core Agent Framework
- **Flexible Agent Architecture**: Trait-based design for easy customization and extension
- **Built-in Agent Types**: Assistant, Researcher, File Manager, Code Assistant, and Web Agent presets
- **Tool System**: Extensible tool registry with built-in tools for common operations
- **Health Monitoring**: Comprehensive health checks and performance metrics

### 🛠️ Built-in Tools
- **Echo Tool**: Testing and debugging utility
- **File Tool**: Secure file operations with path validation
- **HTTP Tool**: Web requests with security controls
- **Search Tool**: Integration with Cheungfun's RAG system
- **Math Tool**: Basic mathematical calculations

### 🔌 MCP Integration
- **Full MCP Support**: Complete Model Context Protocol implementation using rmcp
- **MCP Client**: Connect to external MCP servers and use their tools
- **MCP Server**: Expose Cheungfun tools via MCP protocol
- **Tool Registry**: Unified management of local and remote tools

### 🎭 Workflow Orchestration
- **Multi-Agent Coordination**: Orchestrate complex workflows across multiple agents
- **Dependency Management**: Define step dependencies with automatic execution ordering
- **Parallel Execution**: Concurrent task execution with configurable limits
- **Error Handling**: Retry logic and graceful failure handling

### 🧠 RAG Integration
- **Knowledge-Enhanced Agents**: Deep integration with cheungfun-query for RAG capabilities
- **Intelligent Retrieval**: Automatic context retrieval based on query analysis
- **Source Attribution**: Include source information in agent responses
- **Configurable RAG**: Flexible configuration for different use cases

### 🤖 LLM Integration
- **Multi-Provider Support**: OpenAI, Anthropic, Ollama via siumai library
- **ReAct Pattern**: Reasoning and Acting pattern implementation
- **Flexible Configuration**: Temperature, max tokens, system prompts
- **Error Handling**: Robust error handling and fallback mechanisms

## 🚀 Quick Start

### Basic Agent

```rust
use cheungfun_agents::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Create an agent with built-in tools
    let agent = AgentBuilder::assistant()
        .name("my_assistant")
        .description("A helpful AI assistant")
        .tool(Arc::new(EchoTool::new()))
        .tool(Arc::new(HttpTool::new()))
        .build()?;

    // Execute a task
    let task = Task::new("Echo 'Hello, World!'");
    let response = agent.execute(&task).await?;
    
    println!("Response: {}", response.content);
    Ok(())
}
```

### Workflow Orchestration

```rust
use cheungfun_agents::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create agents
    let researcher = AgentBuilder::researcher().build()?;
    let writer = AgentBuilder::assistant().name("writer").build()?;
    
    // Create orchestrator
    let mut orchestrator = AgentOrchestrator::new();
    orchestrator.register_agent(Arc::new(researcher))?;
    orchestrator.register_agent(Arc::new(writer))?;
    
    // Create workflow
    let workflow = Workflow::builder()
        .name("Content Creation")
        .step(create_step("research", "Research", researcher.id()))
        .step(create_step_with_deps("write", "Write", writer.id(), vec!["research"]))
        .build()?;
    
    // Execute workflow
    let result = orchestrator.execute_workflow(workflow).await?;
    println!("Workflow completed: {:?}", result.status);
    
    Ok(())
}
```

### MCP Integration

```rust
use cheungfun_agents::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create MCP service
    let mut mcp_service = McpService::new();
    
    // Add MCP client
    let client = McpClient::new("external_tools", "1.0.0");
    mcp_service.add_client("external", client).await?;
    
    // Create tools from MCP registry
    let tool_registry = mcp_service.tool_registry();
    let tools = tool_registry.list_tools();
    
    // Create agent with MCP tools
    let mut builder = AgentBuilder::assistant();
    for tool_info in tools {
        if let Ok(tool) = tool_registry.create_tool(&tool_info.name) {
            builder = builder.tool(tool);
        }
    }
    
    let agent = builder.build()?;
    Ok(())
}
```

### LLM Integration

```rust
use cheungfun_agents::{
    agent::react::{ReActAgent, ReActConfig},
    llm::SiumaiLlmClient,
    tool::ToolRegistry,
    types::{AgentConfig, AgentCapabilities, AgentMessage, MessageRole},
};
use std::{collections::HashMap, sync::Arc};

#[tokio::main]
async fn main() -> Result<()> {
    // Create LLM client (supports OpenAI, Anthropic, Ollama)
    let llm_client = SiumaiLlmClient::ollama(
        "http://localhost:11434",
        "llama3.2"
    ).await?;

    // Create ReAct agent with LLM
    let config = ReActConfig::default();
    let tools = Arc::new(ToolRegistry::new());
    let agent = ReActAgent::with_llm_client(config, tools, llm_client);

    // Chat with the agent
    let message = AgentMessage {
        id: uuid::Uuid::new_v4(),
        content: "Explain quantum computing in simple terms".to_string(),
        role: MessageRole::User,
        metadata: HashMap::new(),
        timestamp: chrono::Utc::now(),
        tool_calls: Vec::new(),
    };

    let response = agent.chat(message, None).await?;
    println!("Agent: {}", response.content);
    Ok(())
}
```

## 📚 Examples

The `examples/` directory contains comprehensive examples:

- **`basic_agent.rs`**: Basic agent usage with built-in tools
- **`workflow_orchestration.rs`**: Multi-agent workflow coordination
- **`mcp_integration.rs`**: MCP client/server integration
- **`rag_agent.rs`**: RAG-enhanced intelligent agents
- **`react_with_llm.rs`**: ReAct agent with LLM integration

Run examples with:
```bash
cargo run --example basic_agent
cargo run --example workflow_orchestration
cargo run --example mcp_integration
cargo run --example react_with_llm -- --provider ollama --model llama3.2
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Agent Layer   │    │  Orchestration  │    │  MCP Protocol   │
│                 │    │     Layer       │    │     Layer       │
│ • AgentBuilder  │    │ • Orchestrator  │    │ • MCP Client    │
│ • Agent Traits  │    │ • Workflows     │    │ • MCP Server    │
│ • Executors     │    │ • Dependencies  │    │ • Tool Registry │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                    Tool System Layer                              │
│                                                                   │
│ • Tool Registry     • Built-in Tools      • Custom Tools         │
│ • Tool Execution    • Security Controls   • Schema Validation    │
└───────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                 Cheungfun Core Integration                        │
│                                                                   │
│ • RAG System        • Vector Stores       • LLM Integration      │
│ • Query Engine      • Retrievers          • Error Handling       │
└───────────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

### Agent Configuration

```rust
let config = AgentConfig {
    name: "my_agent".to_string(),
    description: Some("Custom agent".to_string()),
    capabilities: AgentCapabilities {
        supports_tools: true,
        supports_streaming: true,
        supports_conversation: true,
        max_context_length: Some(8192),
        ..Default::default()
    },
    max_execution_time_ms: Some(30_000),
    max_tool_calls: Some(10),
    verbose: true,
    ..Default::default()
};
```

### Orchestrator Configuration

```rust
let config = OrchestratorConfig {
    max_concurrent_workflows: 5,
    max_concurrent_tasks_per_workflow: 3,
    default_workflow_timeout_ms: 300_000,
    auto_retry_failed_tasks: true,
    max_retry_attempts: 3,
    ..Default::default()
};

let orchestrator = AgentOrchestrator::with_config(config);
```

## 🛡️ Security

- **Path Validation**: File tools include path traversal protection
- **Network Controls**: HTTP tools can restrict localhost/private IP access
- **Tool Permissions**: Dangerous tools require explicit enablement
- **Execution Limits**: Configurable timeouts and resource limits

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🔗 Related Projects

- **[cheungfun-core](../cheungfun-core)**: Core types and traits
- **[cheungfun-query](../cheungfun-query)**: RAG query processing
- **[rmcp](https://github.com/modelcontextprotocol/rmcp)**: Rust MCP implementation

---

Built with ❤️ by the Cheungfun team
