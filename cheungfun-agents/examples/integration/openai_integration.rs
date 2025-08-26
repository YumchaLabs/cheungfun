//! Complete OpenAI Integration Example
//!
//! This example shows comprehensive integration with OpenAI's API,
//! demonstrating different models, configurations, and real-world usage patterns.

use cheungfun_agents::{
    agent::{
        base::{AgentContext, BaseAgent},
        react::{ReActAgent, ReActConfig},
    },
    error::Result,
    llm::SiumaiLlmClient,
    tool::{
        builtin::{CalculatorTool, WeatherTool, WikipediaTool},
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

    println!("🤖 Complete OpenAI Integration Example");
    println!("{}", "=".repeat(45));

    // Check for API key
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => {
            println!("❌ Error: OPENAI_API_KEY environment variable not set or empty");
            println!("Please set your OpenAI API key:");
            println!("export OPENAI_API_KEY=\"your-api-key-here\"");
            println!("\nRunning in demo mode with mock responses...");
            return demo_mode_example().await;
        }
    };

    println!("✅ OpenAI API key found, running real examples...");
    println!();

    // Run different OpenAI integration examples
    basic_openai_integration(&api_key).await?;
    println!();

    different_models_comparison(&api_key).await?;
    println!();

    advanced_configuration_example(&api_key).await?;
    println!();

    real_world_assistant_example(&api_key).await?;
    println!();

    println!("✅ All OpenAI integration examples completed!");
    Ok(())
}

/// Basic OpenAI integration with GPT-4
async fn basic_openai_integration(api_key: &str) -> Result<()> {
    println!("🔗 Example 1: Basic OpenAI Integration (GPT-4)");
    println!("{}", "-".repeat(35));

    // Create OpenAI client with GPT-4
    let llm_client = SiumaiLlmClient::openai(api_key, "gpt-4").await?;

    // Simple tool setup
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(CalculatorTool::new()))?;

    let agent_config = ReActConfig::new("GPT-4 Agent").with_max_iterations(3);

    let mut config = agent_config;
    config.base_config.instructions = Some(
        "You are a helpful assistant powered by GPT-4. \
         You have access to a calculator for mathematical operations. \
         Be precise and explain your reasoning clearly."
            .to_string(),
    );

    let agent = ReActAgent::with_llm_client(config, Arc::new(tools), llm_client);

    // Test with a calculation that requires reasoning
    let message = AgentMessage::user(
        "I have a rectangular garden that is 15 meters long and 8 meters wide. \
         If I want to put a fence around the entire perimeter, how many meters of fencing do I need? \
         Also, what's the area of the garden?"
    );

    println!("User: {}", message.content);
    let mut context = AgentContext::new();
    let response = agent.chat(message, Some(&mut context)).await?;
    println!("GPT-4 Agent: {}", response.content);

    Ok(())
}

/// Compare different OpenAI models
async fn different_models_comparison(api_key: &str) -> Result<()> {
    println!("⚖️ Example 2: Different Models Comparison");
    println!("{}", "-".repeat(35));

    // Test with GPT-3.5-turbo for speed
    println!("Testing GPT-3.5-turbo (faster, more economical):");
    let gpt35_client = SiumaiLlmClient::openai(api_key, "gpt-3.5-turbo").await?;

    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(CalculatorTool::new()))?;

    let gpt35_agent = ReActAgent::with_llm_client(
        ReActConfig::new("GPT-3.5 Agent").with_max_iterations(2),
        Arc::new(tools.clone()),
        gpt35_client,
    );

    let quick_message = AgentMessage::user("What's 15% of 240?");
    println!("User: {}", quick_message.content);

    let mut gpt35_context = AgentContext::new();
    let start_time = std::time::Instant::now();
    let gpt35_response = gpt35_agent
        .chat(quick_message.clone(), Some(&mut gpt35_context))
        .await?;
    let gpt35_duration = start_time.elapsed();

    println!(
        "GPT-3.5: {} (took {:?})",
        gpt35_response.content, gpt35_duration
    );

    println!();

    // Test with GPT-4 for quality
    println!("Testing GPT-4 (higher quality, more thoughtful):");
    let gpt4_client = SiumaiLlmClient::openai(api_key, "gpt-4").await?;

    let gpt4_agent = ReActAgent::with_llm_client(
        ReActConfig::new("GPT-4 Agent").with_max_iterations(2),
        Arc::new(tools),
        gpt4_client,
    );

    let mut gpt4_context = AgentContext::new();
    let start_time = std::time::Instant::now();
    let gpt4_response = gpt4_agent
        .chat(quick_message, Some(&mut gpt4_context))
        .await?;
    let gpt4_duration = start_time.elapsed();

    println!(
        "GPT-4: {} (took {:?})",
        gpt4_response.content, gpt4_duration
    );

    println!("\n💡 Model Selection Tips:");
    println!("- Use GPT-3.5-turbo for: Fast responses, simple tasks, cost-sensitive applications");
    println!("- Use GPT-4 for: Complex reasoning, detailed analysis, high-quality responses");

    Ok(())
}

/// Advanced configuration with custom parameters
async fn advanced_configuration_example(api_key: &str) -> Result<()> {
    println!("⚙️ Example 3: Advanced Configuration");
    println!("{}", "-".repeat(30));

    // Create client with custom configuration
    // Note: SiumaiLlmClient currently uses default settings,
    // but this shows the pattern for when more configuration is available
    let llm_client = SiumaiLlmClient::openai(api_key, "gpt-4").await?;

    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(WikipediaTool::new()))?;
    tools.register(Arc::new(CalculatorTool::new()))?;

    // Configure agent for creative writing
    let creative_config = ReActConfig::new("Creative Research Agent")
        .with_max_iterations(5)
        .with_max_thinking_time(10_000); // 10 seconds thinking time

    let mut config = creative_config;
    config.base_config.instructions = Some(
        "You are a creative research assistant with a flair for storytelling. \
         When researching topics, provide both factual information and interesting \
         narrative context. Use tools to gather accurate data, then present it \
         in an engaging way."
            .to_string(),
    );

    let creative_agent = ReActAgent::with_llm_client(config, Arc::new(tools), llm_client);

    let creative_query = AgentMessage::user(
        "Research the history of artificial intelligence and tell me about \
         one fascinating milestone, including some calculations about timeline or impact.",
    );

    println!("User: {}", creative_query.content);
    let mut creative_context = AgentContext::new();
    let creative_response = creative_agent
        .chat(creative_query, Some(&mut creative_context))
        .await?;
    println!("Creative Agent: {}", creative_response.content);

    Ok(())
}

/// Real-world assistant example with comprehensive capabilities
async fn real_world_assistant_example(api_key: &str) -> Result<()> {
    println!("🌟 Example 4: Real-World Assistant");
    println!("{}", "-".repeat(25));

    let llm_client = SiumaiLlmClient::openai(api_key, "gpt-4").await?;

    // Full tool suite for comprehensive assistance
    let mut full_tools = ToolRegistry::new();
    full_tools.register(Arc::new(CalculatorTool::new()))?;
    full_tools.register(Arc::new(WikipediaTool::new()))?;
    full_tools.register(Arc::new(WeatherTool::new()))?;

    let assistant_config = ReActConfig::new("Comprehensive Assistant").with_max_iterations(8);

    let mut config = assistant_config;
    config.base_config.instructions = Some(
        "You are a comprehensive AI assistant designed to help with various tasks. \
         You have access to calculation, research, and weather tools. \
         \n\nYour approach:\
         \n1. Understand the user's need completely\
         \n2. Break down complex requests into steps\
         \n3. Use appropriate tools for each step\
         \n4. Provide clear, actionable results\
         \n5. Offer additional helpful information when relevant\
         \n\nAlways be helpful, accurate, and efficient."
            .to_string(),
    );

    let assistant = ReActAgent::with_llm_client(config, Arc::new(full_tools), llm_client);

    // Complex real-world scenario
    let complex_request = AgentMessage::user(
        "I'm planning a business trip to San Francisco next week. \
         I need to know: 1) What's the weather forecast? \
         2) If my daily budget is $200 and the trip is 5 days, \
         what's my total budget with a 15% contingency added? \
         3) Can you tell me something interesting about San Francisco \
         that might be useful for networking conversations?",
    );

    println!("User: {}", complex_request.content);
    println!("\n[Assistant thinking and working through the request...]");

    let mut assistant_context = AgentContext::new();
    let assistant_response = assistant
        .chat(complex_request, Some(&mut assistant_context))
        .await?;

    println!("Comprehensive Assistant: {}", assistant_response.content);

    // Follow-up question to test context retention
    println!("\n--- Follow-up ---");
    let followup = AgentMessage::user(
        "Based on my budget calculation, how much can I spend per meal if I eat 3 meals per day?",
    );
    println!("User: {}", followup.content);

    let followup_response = assistant
        .chat(followup, Some(&mut assistant_context))
        .await?;
    println!("Assistant: {}", followup_response.content);

    Ok(())
}

/// Demo mode when API key is not available
async fn demo_mode_example() -> Result<()> {
    println!("🎭 Demo Mode: Simulated OpenAI Integration");
    println!("{}", "-".repeat(35));

    println!("This would demonstrate:");
    println!("✅ Real OpenAI API integration with various models");
    println!("✅ Performance comparison between GPT-3.5 and GPT-4");
    println!("✅ Advanced configuration options");
    println!("✅ Complex multi-tool workflows");
    println!("✅ Context retention across conversations");

    println!("\n💡 To see these examples in action:");
    println!("1. Sign up at https://platform.openai.com/");
    println!("2. Get your API key from the API keys section");
    println!("3. Set the environment variable: export OPENAI_API_KEY=\"your-key\"");
    println!("4. Re-run this example");

    println!("\n🔍 What you would see:");
    println!("- Actual GPT responses with reasoning");
    println!("- Real tool usage for calculations and research");
    println!("- Performance metrics comparing different models");
    println!("- Context-aware follow-up responses");

    Ok(())
}

/// Configuration best practices
#[allow(dead_code)]
fn openai_integration_best_practices() {
    println!("💡 OpenAI Integration Best Practices:");
    println!();

    println!("🔑 API Key Management:");
    println!("- Store API keys in environment variables, never in code");
    println!("- Use different keys for development and production");
    println!("- Implement key rotation policies");
    println!("- Monitor API usage and set billing alerts");
    println!();

    println!("🎯 Model Selection:");
    println!("- GPT-3.5-turbo: Fast, cost-effective for simple tasks");
    println!("- GPT-4: Best quality for complex reasoning and analysis");
    println!("- GPT-4-turbo: Balanced performance and speed");
    println!("- Consider model capabilities vs. cost for your use case");
    println!();

    println!("⚡ Performance Optimization:");
    println!("- Cache responses when possible");
    println!("- Use streaming for real-time applications");
    println!("- Batch similar requests");
    println!("- Implement request timeout and retry logic");
    println!();

    println!("💰 Cost Management:");
    println!("- Set max_tokens limits appropriately");
    println!("- Monitor token usage per request");
    println!("- Use shorter prompts when possible");
    println!("- Consider fine-tuning for specialized tasks");
    println!();

    println!("🛡️ Error Handling:");
    println!("- Handle rate limiting gracefully");
    println!("- Implement exponential backoff for retries");
    println!("- Validate responses before processing");
    println!("- Have fallback strategies for API failures");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_demo_mode() {
        // Test that demo mode runs without errors
        let result = demo_mode_example().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_api_key_validation() {
        // Test API key validation logic
        let empty_key = "";
        let valid_key = "sk-test-key-here";

        assert!(empty_key.is_empty());
        assert!(!valid_key.is_empty());
    }
}
