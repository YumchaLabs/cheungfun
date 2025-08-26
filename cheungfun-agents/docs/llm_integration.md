# 🤖 LLM Integration Guide

This guide explains how to integrate Large Language Models (LLMs) with Cheungfun Agents using the siumai library.

## 🚀 Quick Start

### 1. Setup API Keys

Before using LLM providers, set up your API keys:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For Ollama (optional, defaults to localhost:11434)
export OLLAMA_BASE_URL="http://localhost:11434"
```

### 2. Basic Usage

```rust
use cheungfun_agents::{
    agent::react::{ReActAgent, ReActConfig},
    llm::{SiumaiLlmClient, LlmClientConfig},
    tool::ToolRegistry,
    types::{AgentConfig, AgentCapabilities, AgentMessage, MessageRole},
};
use std::{collections::HashMap, sync::Arc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create LLM client
    let llm_client = SiumaiLlmClient::openai(
        std::env::var("OPENAI_API_KEY")?,
        "gpt-4o-mini"
    ).await?;

    // Create tool registry
    let tool_registry = Arc::new(ToolRegistry::new());

    // Create agent configuration
    let config = ReActConfig {
        base_config: AgentConfig {
            name: "My ReAct Agent".to_string(),
            description: Some("A ReAct agent with LLM integration".to_string()),
            instructions: Some("You are a helpful assistant.".to_string()),
            capabilities: AgentCapabilities::default(),
            max_execution_time_ms: Some(30_000),
            max_tool_calls: Some(5),
            verbose: true,
            custom_config: HashMap::new(),
        },
        max_iterations: 5,
        max_thinking_time_ms: 10_000,
        include_trace: true,
        custom_settings: HashMap::new(),
    };

    // Create ReAct agent with LLM
    let agent = ReActAgent::with_llm_client(config, tool_registry, llm_client);

    // Use the agent
    let message = AgentMessage {
        id: uuid::Uuid::new_v4(),
        content: "Hello! Can you help me?".to_string(),
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

## 🔧 Supported Providers

### OpenAI

```rust
let llm_client = SiumaiLlmClient::openai("your-api-key", "gpt-4o-mini").await?;
```

**Supported Models:**
- `gpt-4o` - Latest GPT-4 Omni model
- `gpt-4o-mini` - Faster, cheaper GPT-4 Omni
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-3.5-turbo` - GPT-3.5 Turbo

### Anthropic

```rust
let llm_client = SiumaiLlmClient::anthropic("your-api-key", "claude-3-haiku-20240307").await?;
```

**Supported Models:**
- `claude-3-opus-20240229` - Most capable Claude 3 model
- `claude-3-sonnet-20240229` - Balanced performance and speed
- `claude-3-haiku-20240307` - Fastest Claude 3 model

### Ollama (Local)

```rust
let llm_client = SiumaiLlmClient::ollama("http://localhost:11434", "llama3.2").await?;
```

**Popular Models:**
- `llama3.2` - Meta's Llama 3.2
- `mistral` - Mistral 7B
- `codellama` - Code Llama
- `phi3` - Microsoft Phi-3

## ⚙️ Configuration

### LLM Client Configuration

```rust
let config = LlmClientConfig::new("openai", "gpt-4o-mini")
    .with_temperature(0.7)
    .with_max_tokens(2048)
    .with_system_prompt("You are a helpful ReAct agent.")
    .with_tools(true);

let llm_client = SiumaiLlmClient::from_config(config).await?;
```

### ReAct Agent Configuration

```rust
let react_config = ReActConfig {
    base_config: AgentConfig {
        name: "Advanced ReAct Agent".to_string(),
        description: Some("An advanced ReAct agent with custom settings".to_string()),
        instructions: Some(
            "You are a ReAct agent. Think step by step, use tools when needed."
        ),
        capabilities: AgentCapabilities {
            supports_tools: true,
            supports_streaming: false,
            supports_conversation: true,
            supports_files: false,
            supports_web: false,
            supports_code_execution: false,
            max_context_length: Some(4096),
            supported_input_formats: vec!["text".to_string()],
            supported_output_formats: vec!["text".to_string()],
            custom_capabilities: HashMap::new(),
        },
        max_execution_time_ms: Some(60_000),
        max_tool_calls: Some(10),
        verbose: true,
        custom_config: HashMap::new(),
    },
    max_iterations: 10,
    max_thinking_time_ms: 15_000,
    include_trace: true,
    custom_settings: HashMap::new(),
};
```

## 🛠️ Adding Tools

```rust
use cheungfun_agents::tool::builtin::{EchoTool, HttpTool, SearchTool};

let mut tool_registry = ToolRegistry::new();

// Add built-in tools
tool_registry.register(Arc::new(EchoTool::new()))?;
tool_registry.register(Arc::new(HttpTool::new()))?;
tool_registry.register(Arc::new(SearchTool::new()))?;

let tool_registry = Arc::new(tool_registry);
```

## 🎯 ReAct Pattern

The ReAct (Reasoning and Acting) pattern follows this structure:

1. **Thought**: The agent reasons about the task
2. **Action**: The agent decides to use a tool (optional)
3. **Observation**: The agent observes the tool's result
4. **Final Answer**: The agent provides the final response

Example ReAct trace:
```
Thought: I need to search for information about Rust programming.
Action: search("Rust programming language features")
Observation: Rust is a systems programming language focused on safety, speed, and concurrency...
Final Answer: Rust is a systems programming language that focuses on memory safety...
```

## 🔍 Running Examples

### Basic Example

```bash
cargo run --example react_with_llm -- --provider ollama --model llama3.2
```

### With OpenAI

```bash
export OPENAI_API_KEY="your-key"
cargo run --example react_with_llm -- --provider openai --model gpt-4o-mini
```

### With Anthropic

```bash
export ANTHROPIC_API_KEY="your-key"
cargo run --example react_with_llm -- --provider anthropic --model claude-3-haiku-20240307
```

## 🚨 Error Handling

```rust
match agent.chat(message, None).await {
    Ok(response) => {
        println!("Success: {}", response.content);
        println!("Execution time: {}ms", response.stats.execution_time_ms);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
        match e {
            AgentError::InvalidConfiguration(msg) => {
                eprintln!("Configuration error: {}", msg);
            }
            AgentError::LlmError(msg) => {
                eprintln!("LLM error: {}", msg);
            }
            _ => {
                eprintln!("Other error: {}", e);
            }
        }
    }
}
```

## 📊 Performance Tips

1. **Choose the right model**: Use smaller models for simple tasks
2. **Set appropriate timeouts**: Configure `max_execution_time_ms`
3. **Limit tool calls**: Set `max_tool_calls` to prevent infinite loops
4. **Use streaming**: Enable streaming for long responses (when supported)
5. **Cache responses**: Implement caching for repeated queries

## 🔗 Next Steps

- [Tool Development Guide](./tool_development.md)
- [MCP Integration](./mcp_integration.md)
- [Advanced ReAct Patterns](./advanced_react.md)
- [Performance Optimization](./performance.md)
