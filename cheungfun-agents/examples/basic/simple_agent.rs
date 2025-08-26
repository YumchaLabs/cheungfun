//! Simple Agent Example - Getting Started Guide
//!
//! This example shows the basics of using agents with tools,
//! perfect for getting started quickly.

use cheungfun_agents::{
    agent::{
        base::{AgentContext, BaseAgent},
        react::{ReActAgent, ReActConfig},
    },
    error::Result,
    llm::SiumaiLlmClient,
    tool::{
        builtin::{CalculatorTool, WeatherTool},
        ToolRegistry,
    },
    types::AgentMessage,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging for debugging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🤖 Simple Agent Example");
    println!("{}", "=".repeat(40));

    // Step 1: Configure OpenAI client
    println!("1️⃣  Setting up OpenAI client...");
    let llm_client = SiumaiLlmClient::openai(
        std::env::var("OPENAI_API_KEY").expect("Please set OPENAI_API_KEY environment variable"),
        "gpt-4",
    )
    .await?;

    // Step 2: Setup tools
    println!("2️⃣  Registering tools...");
    let mut tool_registry = ToolRegistry::new();

    // Add calculator tool
    tool_registry.register(Arc::new(CalculatorTool::new()))?;

    // Add weather tool
    tool_registry.register(Arc::new(WeatherTool::new()))?;

    // Step 3: Create the ReAct agent
    println!("3️⃣  Building ReAct agent...");
    let react_config = ReActConfig::new("Helpful Assistant").with_max_iterations(5);

    let mut config = react_config;
    config.base_config.description =
        Some("A helpful assistant with calculator and weather tools".to_string());
    config.base_config.instructions = Some(
        "You are a helpful assistant with access to tools. \
         Use the calculator for math problems and weather tool for weather queries. \
         Think step by step and use tools when needed."
            .to_string(),
    );

    let agent = ReActAgent::with_llm_client(config, Arc::new(tool_registry), llm_client);

    // Step 4: Run some example conversations
    println!("4️⃣  Starting conversations...");

    // Example 1: Math calculation
    println!("\n📊 Math Example:");
    println!("{}", "-".repeat(20));

    let math_query = AgentMessage::user(
        "What's the result of (15 + 25) * 3.14159, and what percentage is that of 500?",
    );

    println!("Human: {}", math_query.content);

    let mut context = AgentContext::new();
    let response = agent.chat(math_query, Some(&mut context)).await?;

    println!("Agent: {}", response.content);

    // Example 2: Weather query
    println!("\n🌤️  Weather Example:");
    println!("{}", "-".repeat(20));

    let weather_query = AgentMessage::user(
        "What's the weather like in Tokyo? Please give me the temperature in both Celsius and Fahrenheit."
    );

    println!("Human: {}", weather_query.content);

    let mut context2 = AgentContext::new();
    let response2 = agent.chat(weather_query, Some(&mut context2)).await?;

    println!("Agent: {}", response2.content);

    // Example 3: Combined query
    println!("\n🔄 Combined Example:");
    println!("{}", "-".repeat(20));

    let combined_query = AgentMessage::user(
        "If the temperature in London is 12°C, convert it to Fahrenheit. \
         Then calculate what the temperature would be if it increased by 15%.",
    );

    println!("Human: {}", combined_query.content);

    let mut context3 = AgentContext::new();
    let response3 = agent.chat(combined_query, Some(&mut context3)).await?;

    println!("Agent: {}", response3.content);

    println!("\n✅ Example completed! The agent successfully used tools to answer questions.");

    // Show some metadata
    println!("\n📋 Agent capabilities:");
    println!("- Tools available: calculator, weather");
    println!("- Max reasoning iterations: 5");
    println!("- Model: gpt-4");

    Ok(())
}

/// Example of running interactive mode
#[allow(dead_code)]
async fn interactive_mode() -> Result<()> {
    use std::io::{self, Write};

    // Setup agent (same as above)
    let llm_client = SiumaiLlmClient::openai(
        std::env::var("OPENAI_API_KEY").expect("Please set OPENAI_API_KEY environment variable"),
        "gpt-4",
    )
    .await?;

    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(Arc::new(CalculatorTool::new()))?;
    tool_registry.register(Arc::new(WeatherTool::new()))?;

    let react_config = ReActConfig::new("Interactive Assistant").with_max_iterations(5);

    let mut config = react_config;
    config.base_config.instructions =
        Some("You are a helpful assistant. Use tools when appropriate.".to_string());

    let agent = ReActAgent::with_llm_client(config, Arc::new(tool_registry), llm_client);

    println!("🤖 Interactive Agent Mode (type 'quit' to exit)");

    let mut context = AgentContext::new();

    loop {
        print!("\nHuman: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "quit" {
            break;
        }

        if input.is_empty() {
            continue;
        }

        let message = AgentMessage::user(input);
        match agent.chat(message, Some(&mut context)).await {
            Ok(response) => println!("Agent: {}", response.content),
            Err(e) => println!("Error: {}", e),
        }
    }

    Ok(())
}
