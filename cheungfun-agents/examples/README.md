# ChengFun Agents Examples

This directory contains comprehensive examples demonstrating how to use the ChengFun Agents framework. The examples are organized by complexity and use case to help you learn and implement AI agents effectively.

## 🗂️ Directory Structure

```
examples/
├── basic/              # Entry-level examples for beginners
│   ├── simple_agent.rs        # Basic agent setup and usage
│   └── tool_usage.rs          # Comprehensive tool usage patterns
├── advanced/           # Complex examples for experienced users
│   ├── comprehensive_agent.rs # Multi-agent collaboration
│   ├── custom_tools.rs        # Creating custom tools
│   └── memory_management.rs   # Context and memory patterns
├── integration/        # Real-world integration examples
│   ├── openai_integration.rs  # Complete OpenAI API integration
│   └── rag_integration.rs     # RAG system integration
├── mcp/               # Model Context Protocol examples
│   ├── client.rs              # MCP client implementation
│   ├── server.rs              # MCP server implementation
│   ├── integration.rs         # MCP integration patterns
│   └── simple_test.rs         # Basic MCP testing
└── README.md          # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Rust**: Install from [rustup.rs](https://rustup.rs/)
2. **OpenAI API Key** (for real examples):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### Running Examples

From the project root directory:

```bash
# Basic examples
cargo run --example simple_agent
cargo run --example tool_usage

# Advanced examples  
cargo run --example comprehensive_agent
cargo run --example custom_tools
cargo run --example memory_management

# Integration examples
cargo run --example openai_integration
cargo run --example rag_integration
```

## 📚 Example Categories

### 🌱 Basic Examples

Perfect for beginners learning the fundamentals:

#### `basic/simple_agent.rs`
- **What it teaches**: Agent setup, tool registration, basic conversations
- **Key concepts**: ReActAgent, ToolRegistry, AgentContext
- **Tools used**: Calculator, Weather
- **Complexity**: ⭐ Beginner

**Sample usage:**
```rust
let agent = ReActAgent::with_llm_client(config, tools, llm_client);
let response = agent.chat(message, &mut context).await?;
```

#### `basic/tool_usage.rs`  
- **What it teaches**: Different tool usage patterns and best practices
- **Key concepts**: Tool selection, tool chaining, specialized agents
- **Tools used**: All built-in tools
- **Complexity**: ⭐⭐ Beginner-Intermediate

**Patterns demonstrated:**
- Basic tool registration
- Selective tool usage for specialized agents
- Tool chaining for complex tasks
- Custom tool configuration

### 🎯 Advanced Examples

For experienced users building complex systems:

#### `advanced/comprehensive_agent.rs`
- **What it teaches**: Multi-agent coordination and complex workflows
- **Key concepts**: Agent roles, handoff strategies, pipeline orchestration
- **Tools used**: Research, analysis, and writing tools
- **Complexity**: ⭐⭐⭐⭐ Advanced

**Features demonstrated:**
- Single agent with complex reasoning
- Multi-tool research assistant
- Multi-agent research pipeline (Research → Analysis → Writing)

#### `advanced/custom_tools.rs`
- **What it teaches**: Creating your own custom tools
- **Key concepts**: Tool trait implementation, validation, state management
- **Custom tools**: Greeting, Counter, Email Validator
- **Complexity**: ⭐⭐⭐ Intermediate-Advanced

**Tool patterns:**
- Simple stateless tools
- Stateful tools with internal state
- Validation tools with error handling

#### `advanced/memory_management.rs`
- **What it teaches**: Managing conversation context and agent memory
- **Key concepts**: Context serialization, state persistence, context sharing
- **Use cases**: Session management, multi-agent collaboration
- **Complexity**: ⭐⭐⭐ Intermediate-Advanced

### 🔗 Integration Examples

Real-world integration with external services:

#### `integration/openai_integration.rs`
- **What it teaches**: Complete OpenAI API integration
- **Key concepts**: Model comparison, configuration, error handling
- **Models**: GPT-3.5-turbo, GPT-4, custom configurations
- **Complexity**: ⭐⭐⭐ Intermediate-Advanced

**Features:**
- Basic OpenAI integration
- Model performance comparison
- Advanced configuration options
- Real-world assistant patterns

#### `integration/rag_integration.rs`
- **What it teaches**: Integrating with Retrieval-Augmented Generation systems
- **Key concepts**: RAG tools, knowledge bases, context enhancement
- **Dependencies**: ChengFun Query system
- **Complexity**: ⭐⭐⭐⭐ Advanced

### 🔌 MCP (Model Context Protocol) Examples

Protocol-level integration examples:

#### `mcp/client.rs` & `mcp/server.rs`
- **What they teach**: MCP protocol implementation
- **Key concepts**: Client-server communication, protocol handling
- **Use cases**: External system integration
- **Complexity**: ⭐⭐⭐⭐ Advanced

## 💡 Learning Path

### For Beginners:
1. Start with `basic/simple_agent.rs`
2. Explore `basic/tool_usage.rs`
3. Try `integration/openai_integration.rs`

### For Intermediate Users:
1. Study `advanced/custom_tools.rs`
2. Experiment with `advanced/memory_management.rs`
3. Build with `advanced/comprehensive_agent.rs`

### For Advanced Users:
1. Implement `integration/rag_integration.rs`
2. Explore MCP examples in `mcp/`
3. Create your own custom examples

## 🔧 Common Patterns

### Agent Setup Pattern
```rust
// 1. Create LLM client
let llm_client = SiumaiLlmClient::openai(api_key, "gpt-4").await?;

// 2. Register tools
let mut tools = ToolRegistry::new();
tools.register(Arc::new(CalculatorTool::new()))?;

// 3. Configure agent
let config = ReActConfig::new("Agent Name").with_max_iterations(5);

// 4. Create agent
let agent = ReActAgent::with_llm_client(config, Arc::new(tools), llm_client);

// 5. Use agent
let response = agent.chat(message, Some(&mut context)).await?;
```

### Custom Tool Pattern
```rust
#[derive(Debug, Clone)]
pub struct MyCustomTool { /* fields */ }

#[async_trait]
impl Tool for MyCustomTool {
    fn schema(&self) -> ToolSchema { /* define schema */ }
    
    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolResult> {
        // Tool implementation
    }
}
```

### Multi-Agent Pattern
```rust
let orchestrator = MultiAgentOrchestratorBuilder::new()
    .add_agent(research_agent, research_role)
    .add_agent(analysis_agent, analysis_role)
    .handoff_strategy(HandoffStrategy::Sequential)
    .build().await?;

let result = orchestrator.execute(initial_message).await?;
```

## 🛠️ Development Tools

### Environment Variables
```bash
# Required for real OpenAI integration
export OPENAI_API_KEY="your-key-here"

# Optional for debugging
export RUST_LOG=debug
```

### Useful Commands
```bash
# Run with logging
RUST_LOG=info cargo run --example simple_agent

# Check example compilation
cargo check --example custom_tools

# Test examples
cargo test --examples

# Build all examples
cargo build --examples
```

## 📖 Architecture Overview

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LLM Client    │────│   ReAct Agent   │────│ Tool Registry   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ OpenAI/Other    │    │ Agent Context   │    │ Built-in Tools  │
    │   Models        │    │    Memory       │    │ Custom Tools    │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **User Input** → Agent receives message
2. **Reasoning** → Agent analyzes and plans
3. **Tool Selection** → Agent chooses appropriate tools  
4. **Tool Execution** → Tools perform operations
5. **Response Generation** → Agent formulates response
6. **Context Update** → Memory and context updated

## 🎓 Best Practices

### 1. Error Handling
- Always handle `Result` types properly
- Implement graceful degradation for API failures
- Provide meaningful error messages

### 2. Resource Management
- Use `Arc` for sharing tools between agents
- Implement proper cleanup for stateful resources
- Monitor memory usage in long-running applications

### 3. Security
- Never commit API keys to version control
- Validate tool inputs thoroughly
- Be cautious with shell and file system access

### 4. Performance
- Reuse LLM clients when possible
- Cache expensive operations
- Use appropriate model sizes for your needs

### 5. Testing
- Write unit tests for custom tools
- Test agent responses with various inputs
- Mock external dependencies in tests

## 🐛 Troubleshooting

### Common Issues

**Example won't compile:**
```bash
# Check dependencies
cargo check --example example_name

# Update dependencies
cargo update
```

**OpenAI API errors:**
```bash
# Verify API key
echo $OPENAI_API_KEY

# Check API quota and billing
# Visit https://platform.openai.com/usage
```

**Tool execution fails:**
```bash
# Enable debug logging
RUST_LOG=debug cargo run --example tool_usage

# Check tool input validation
# Review tool schema definitions
```

### Getting Help

1. **Check the logs**: Enable debug logging with `RUST_LOG=debug`
2. **Review the source**: Examples include detailed comments
3. **Test components**: Isolate issues by testing individual components
4. **Community support**: Refer to the main project documentation

## 📝 Contributing

Want to add more examples? Here's how:

### Adding New Examples

1. **Choose the right directory**: 
   - `basic/` for simple demonstrations
   - `advanced/` for complex patterns
   - `integration/` for external service integration
   - `mcp/` for protocol-level examples

2. **Follow naming conventions**:
   - Use descriptive snake_case names
   - Include `.rs` extension
   - Add to this README

3. **Example template**:
   ```rust
   //! Brief description of what this example demonstrates
   
   use cheungfun_agents::*;
   
   #[tokio::main]
   async fn main() -> Result<()> {
       // Example implementation
       Ok(())
   }
   ```

4. **Documentation requirements**:
   - Clear doc comments explaining the purpose
   - Step-by-step comments in the code
   - Usage examples in comments
   - Add to README with complexity rating

### Example Quality Guidelines

- ✅ **Complete**: Examples should run without additional setup (except API keys)
- ✅ **Educational**: Focus on teaching concepts, not just showing features
- ✅ **Well-commented**: Explain the why, not just the what
- ✅ **Error handling**: Show proper error handling patterns
- ✅ **Best practices**: Demonstrate recommended approaches

## 📚 Additional Resources

- **Main Documentation**: See the project root README
- **API Reference**: Generated docs with `cargo doc --open`
- **Tool Development**: Check `src/tool/` for built-in tool implementations
- **Agent Patterns**: Study `src/agent/` for agent architecture

---

**Happy coding with ChengFun Agents!** 🚀

For questions, issues, or contributions, please refer to the main project repository.