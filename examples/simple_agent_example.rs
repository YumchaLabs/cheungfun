//! Simple Agent Example - Getting Started Guide
//! 
//! This example shows the basics of using agents with tools,
//! perfect for getting started quickly.

use cheungfun_agents::{
    agent::{base::AgentContext, react::ReActAgent},
    error::Result,
    llm::SiumaiLlmClient,
    tool::{
        builtin::{CalculatorTool, WeatherTool},
        ToolRegistry,
    },
    types::AgentMessage,
};
use cheungfun_core::{
    llm::{LlmClientConfig, LlmProvider},
    memory::chat::ChatMemoryBuffer,
};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging for debugging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🤖 Simple Agent Example");
    println!("=" .repeat(40));

    // Step 1: Configure OpenAI client
    println!("1️⃣  Setting up OpenAI client...");
    let llm_config = LlmClientConfig::builder()
        .provider(LlmProvider::OpenAi)
        .model("gpt-4".to_string())
        .api_key(std::env::var("OPENAI_API_KEY")
            .expect("Please set OPENAI_API_KEY environment variable"))
        .temperature(0.3)
        .max_tokens(Some(1000))
        .build();

    let llm_client = Arc::new(SiumaiLlmClient::new(llm_config)?);

    // Step 2: Create memory for conversation history
    println!("2️⃣  Creating conversation memory...");
    let memory = Arc::new(Mutex::new(ChatMemoryBuffer::new(10)));

    // Step 3: Setup tools
    println!("3️⃣  Registering tools...");
    let mut tool_registry = ToolRegistry::new();
    
    // Add calculator tool
    tool_registry.register_tool(
        "calculator".to_string(), 
        Arc::new(CalculatorTool::new())
    ).await?;
    
    // Add weather tool
    tool_registry.register_tool(
        "weather".to_string(), 
        Arc::new(WeatherTool::new())
    ).await?;

    // Step 4: Create the ReAct agent
    println!("4️⃣  Building ReAct agent...");
    let agent = ReActAgent::builder()
        .llm_client(llm_client)
        .memory(memory)
        .tool_registry(Arc::new(tool_registry))
        .system_prompt(
            "You are a helpful assistant with access to tools. \
             Use the calculator for math problems and weather tool for weather queries. \
             Think step by step and use tools when needed.".to_string()
        )
        .max_iterations(5)
        .build()?;

    // Step 5: Run some example conversations
    println!("5️⃣  Starting conversations...");
    
    // Example 1: Math calculation
    println!("\n📊 Math Example:");
    println!("-".repeat(20));
    
    let math_query = AgentMessage::user(
        "What's the result of (15 + 25) * 3.14159, and what percentage is that of 500?"
    );
    
    println!("Human: {}", math_query.content);
    
    let mut context = AgentContext::new();
    let response = agent.chat(math_query, Some(&mut context)).await?;
    
    println!("Agent: {}", response.content);
    
    // Example 2: Weather query
    println!("\n🌤️  Weather Example:");
    println!("-".repeat(20));
    
    let weather_query = AgentMessage::user(
        "What's the weather like in Tokyo? Please give me the temperature in both Celsius and Fahrenheit."
    );
    
    println!("Human: {}", weather_query.content);
    
    let mut context2 = AgentContext::new();
    let response2 = agent.chat(weather_query, Some(&mut context2)).await?;
    
    println!("Agent: {}", response2.content);
    
    // Example 3: Combined query
    println!("\n🔄 Combined Example:");
    println!("-".repeat(20));
    
    let combined_query = AgentMessage::user(
        "If the temperature in London is 12°C, convert it to Fahrenheit. \
         Then calculate what the temperature would be if it increased by 15%."
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
    println!("- Memory buffer: 10 messages");
    println!("- Model: gpt-4");

    Ok(())
}

/// Example of running interactive mode
#[allow(dead_code)]
async fn interactive_mode() -> Result<()> {
    use std::io::{self, Write};
    
    // Setup agent (same as above)
    let llm_config = LlmClientConfig::builder()
        .provider(LlmProvider::OpenAi)
        .model("gpt-4".to_string())
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .temperature(0.3)
        .build();

    let llm_client = Arc::new(SiumaiLlmClient::new(llm_config)?);
    let memory = Arc::new(Mutex::new(ChatMemoryBuffer::new(10)));
    
    let mut tool_registry = ToolRegistry::new();
    tool_registry.register_tool("calculator".to_string(), Arc::new(CalculatorTool::new())).await?;
    tool_registry.register_tool("weather".to_string(), Arc::new(WeatherTool::new())).await?;

    let agent = ReActAgent::builder()
        .llm_client(llm_client)
        .memory(memory)
        .tool_registry(Arc::new(tool_registry))
        .system_prompt("You are a helpful assistant. Use tools when appropriate.".to_string())
        .build()?;

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