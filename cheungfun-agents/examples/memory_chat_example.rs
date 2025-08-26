//! Memory-enabled chat example demonstrating the new agent architecture.

use cheungfun_agents::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🤖 Cheungfun Agents - Memory-Enabled Chat Example");
    println!("==================================================");

    // Create memory for conversation history
    let memory_config = ChatMemoryConfig::with_token_limit(2000);
    let mut memory = ChatMemoryBuffer::new(memory_config);

    println!("📝 Created chat memory with 2000 token limit");

    // Create some tools
    let echo_tool = Arc::new(EchoTool::new());
    let http_tool = Arc::new(HttpTool::new());

    // Create an agent with function calling strategy
    let agent = AgentBuilder::new()
        .name("memory_chat_agent")
        .description("A chat agent with memory and tool capabilities")
        .tool(echo_tool)
        .tool(http_tool)
        .function_calling_strategy() // Use function calling strategy
        .verbose()
        .build()?;

    println!("🤖 Created agent: {}", agent.name());
    println!("   Strategy: {}", agent.strategy().name());
    println!("   Tools: {:?}", agent.tools());

    // Simulate a conversation
    let conversations = vec![
        "Hello! What's your name?",
        "Can you echo 'Hello World' for me?",
        "What did I ask you to echo earlier?",
        "Thanks! That's all for now.",
    ];

    for (i, message) in conversations.iter().enumerate() {
        println!("\n--- Turn {} ---", i + 1);
        println!("👤 User: {}", message);

        // Chat with the agent using memory
        let response = agent.chat(message, &mut memory).await?;

        println!("🤖 Agent: {}", response.content);

        // Show execution stats
        println!(
            "📊 Stats: {}ms, {} tool calls",
            response.stats.execution_time_ms, response.stats.tool_calls_count
        );

        // Show memory stats
        let memory_stats = memory.stats();
        println!(
            "💭 Memory: {} messages, ~{} tokens",
            memory_stats.message_count, memory_stats.estimated_tokens
        );
    }

    // Demonstrate memory persistence
    println!("\n--- Memory Contents ---");
    let messages = memory.get_messages().await?;
    for (i, msg) in messages.iter().enumerate() {
        println!(
            "{}: {:?} - {}",
            i + 1,
            msg.role,
            if msg.content.len() > 50 {
                format!("{}...", &msg.content[..50])
            } else {
                msg.content.clone()
            }
        );
    }

    // Test different strategies
    println!("\n--- Testing Direct Strategy ---");
    let direct_agent = AgentBuilder::new()
        .name("direct_agent")
        .description("Agent with direct strategy")
        .direct_strategy() // Use direct strategy (no tools)
        .build()?;

    let response = direct_agent
        .chat("Hello, can you help me?", &mut memory)
        .await?;
    println!("🤖 Direct Agent: {}", response.content);
    println!("📊 Tool calls: {}", response.stats.tool_calls_count);

    println!("\n✨ Example completed successfully!");
    Ok(())
}
