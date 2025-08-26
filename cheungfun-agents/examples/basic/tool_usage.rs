//! Tool Usage Patterns - Comprehensive Guide to Using Tools with Agents
//!
//! This example focuses specifically on different tool usage patterns and best practices.
//! Learn how to register, configure, and effectively use tools in your agent applications.

use cheungfun_agents::{
    agent::{
        base::{AgentContext, BaseAgent},
        react::{ReActAgent, ReActConfig},
    },
    error::Result,
    llm::SiumaiLlmClient,
    tool::{
        builtin::{CalculatorTool, EchoTool, FileTool, HttpTool, WeatherTool, WikipediaTool},
        ToolRegistry,
    },
    types::AgentMessage,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🔧 Tool Usage Patterns Example");
    println!("{}", "=".repeat(40));

    // Demo different tool usage patterns
    basic_tool_usage().await?;
    println!();

    selective_tool_usage().await?;
    println!();

    tool_chaining_example().await?;
    println!();

    println!("✅ All tool usage patterns demonstrated!");
    Ok(())
}

/// Basic tool usage - Register and use all available tools
async fn basic_tool_usage() -> Result<()> {
    println!("📚 Pattern 1: Basic Tool Registration and Usage");
    println!("{}", "-".repeat(30));

    // Setup LLM
    let llm_client = SiumaiLlmClient::openai(
        std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()),
        "gpt-4",
    )
    .await?;

    // Register multiple tools
    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(Arc::new(EchoTool::new()))?;
    tool_registry.register(Arc::new(CalculatorTool::new()))?;
    tool_registry.register(Arc::new(WeatherTool::new()))?;

    // Create agent with all tools
    let agent_config = ReActConfig::new("Tool Demo Agent").with_max_iterations(3);

    let mut config = agent_config;
    config.base_config.instructions = Some(
        "You have access to multiple tools: echo, calculator, and weather. \
         Use the most appropriate tool for each request."
            .to_string(),
    );

    let agent = ReActAgent::with_llm_client(config, Arc::new(tool_registry), llm_client);

    // Test basic tool usage
    let message = AgentMessage::user("Calculate 25 * 4 and then echo the result");

    println!("User: {}", message.content);
    let mut context = AgentContext::new();
    let response = agent.chat(message, Some(&mut context)).await?;
    println!("Agent: {}", response.content);

    Ok(())
}

/// Selective tool usage - Only register specific tools for focused tasks
async fn selective_tool_usage() -> Result<()> {
    println!("🎯 Pattern 2: Selective Tool Usage");
    println!("{}", "-".repeat(30));

    let llm_client = SiumaiLlmClient::openai(
        std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()),
        "gpt-4",
    )
    .await?;

    // Only register math-related tools for a math-focused agent
    let mut math_tools = ToolRegistry::new();
    math_tools.register(Arc::new(CalculatorTool::new()))?;

    let math_agent = ReActAgent::with_llm_client(
        ReActConfig::new("Math Specialist").with_max_iterations(3),
        Arc::new(math_tools),
        llm_client.clone(),
    );

    // Only register information tools for a research agent
    let mut research_tools = ToolRegistry::new();
    research_tools.register(Arc::new(WikipediaTool::new()))?;

    let research_agent = ReActAgent::with_llm_client(
        ReActConfig::new("Research Specialist").with_max_iterations(3),
        Arc::new(research_tools),
        llm_client,
    );

    // Demonstrate specialized usage
    println!("Math Agent:");
    let math_query = AgentMessage::user("What's the compound interest on $1000 at 5% for 2 years?");
    println!("User: {}", math_query.content);
    let mut math_context = AgentContext::new();
    let math_response = math_agent.chat(math_query, Some(&mut math_context)).await?;
    println!("Math Agent: {}", math_response.content);

    println!("\nResearch Agent:");
    let research_query = AgentMessage::user("Tell me about artificial intelligence");
    println!("User: {}", research_query.content);
    let mut research_context = AgentContext::new();
    let research_response = research_agent
        .chat(research_query, Some(&mut research_context))
        .await?;
    println!("Research Agent: {}", research_response.content);

    Ok(())
}

/// Tool chaining - Using multiple tools in sequence for complex tasks
async fn tool_chaining_example() -> Result<()> {
    println!("🔗 Pattern 3: Tool Chaining for Complex Tasks");
    println!("{}", "-".repeat(30));

    let llm_client = SiumaiLlmClient::openai(
        std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()),
        "gpt-4",
    )
    .await?;

    // Register tools that work well together
    let mut chaining_tools = ToolRegistry::new();
    chaining_tools.register(Arc::new(WikipediaTool::new()))?;
    chaining_tools.register(Arc::new(CalculatorTool::new()))?;
    chaining_tools.register(Arc::new(WeatherTool::new()))?;

    let chaining_agent_config = ReActConfig::new("Multi-Tool Agent").with_max_iterations(6); // More iterations for complex chaining

    let mut config = chaining_agent_config;
    config.base_config.instructions = Some(
        "You are an expert at using multiple tools in sequence to solve complex problems. \
         Break down complex requests into steps and use the appropriate tools for each step. \
         For example: research information, then perform calculations, then provide analysis."
            .to_string(),
    );

    let chaining_agent = ReActAgent::with_llm_client(config, Arc::new(chaining_tools), llm_client);

    // Complex multi-tool task
    let complex_query = AgentMessage::user(
        "I want to plan a trip to Tokyo. Research Tokyo's basic info, \
         check the current weather, and calculate how much I'd need \
         if hotel costs $100/night for a 7-day stay with 20% tip included.",
    );

    println!("User: {}", complex_query.content);
    let mut chain_context = AgentContext::new();
    let chain_response = chaining_agent
        .chat(complex_query, Some(&mut chain_context))
        .await?;
    println!("Multi-Tool Agent: {}", chain_response.content);

    Ok(())
}

/// Example of creating a custom tool configuration
#[allow(dead_code)]
async fn custom_tool_configuration_example() -> Result<()> {
    println!("⚙️ Pattern 4: Custom Tool Configuration");
    println!("{}", "-".repeat(30));

    let llm_client = SiumaiLlmClient::openai(
        std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()),
        "gpt-4",
    )
    .await?;

    // Create tools with custom configurations
    let mut configured_tools = ToolRegistry::new();

    // Weather tool with specific configuration
    let weather_tool = WeatherTool::new(); // In real usage, you might configure API keys here
    configured_tools.register(Arc::new(weather_tool))?;

    // File tool with read-only access (safer for demos)
    let file_tool = FileTool::new();
    configured_tools.register(Arc::new(file_tool))?;

    // HTTP tool for web requests
    let http_tool = HttpTool::new();
    configured_tools.register(Arc::new(http_tool))?;

    let configured_agent = ReActAgent::with_llm_client(
        ReActConfig::new("Configured Agent").with_max_iterations(4),
        Arc::new(configured_tools),
        llm_client,
    );

    // Test configured tools
    let config_query = AgentMessage::user("Check if you can access system information safely");
    println!("User: {}", config_query.content);
    let mut config_context = AgentContext::new();
    let config_response = configured_agent
        .chat(config_query, Some(&mut config_context))
        .await?;
    println!("Configured Agent: {}", config_response.content);

    Ok(())
}

/// Best practices and tips for tool usage
#[allow(dead_code)]
fn tool_usage_best_practices() {
    println!("💡 Tool Usage Best Practices:");
    println!("1. **Selective Registration**: Only register tools that your agent actually needs");
    println!("2. **Tool Naming**: Use clear, descriptive names for custom tools");
    println!("3. **Error Handling**: Tools should provide clear error messages");
    println!("4. **Security**: Be cautious with tools that access external systems");
    println!("5. **Performance**: Consider caching for expensive tool operations");
    println!("6. **Documentation**: Always document what each tool does and its limitations");
    println!("7. **Testing**: Test tools individually before using in agent workflows");
}
