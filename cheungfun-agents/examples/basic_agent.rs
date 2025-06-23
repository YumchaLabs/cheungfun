//! Basic agent example demonstrating core functionality.

use cheungfun_agents::prelude::*;
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🤖 Cheungfun Agents - Basic Agent Example");
    println!("==========================================");

    // Create some built-in tools
    let echo_tool = Arc::new(EchoTool::new());
    let http_tool = Arc::new(HttpTool::new());
    let file_tool = Arc::new(FileTool::new()); // Read-only by default

    // Create an agent with tools
    let agent = AgentBuilder::assistant()
        .name("demo_assistant")
        .description("A demonstration assistant with basic tools")
        .tool(echo_tool)
        .tool(http_tool)
        .tool(file_tool)
        .max_execution_time_ms(30_000)
        .verbose()
        .build()?;

    println!("✅ Created agent: {}", agent.name());
    println!("📋 Available tools: {:?}", agent.tools());
    println!("🔧 Capabilities: {:?}", agent.capabilities());

    // Create some example tasks
    let tasks = vec![
        Task::builder()
            .name("Echo Test")
            .input("echo Hello, World!")
            .build()?,
        Task::builder()
            .name("Tool Listing")
            .input("What tools do you have available?")
            .build()?,
        Task::builder()
            .name("Simple Question")
            .input("What is the capital of France?")
            .build()?,
    ];

    // Execute tasks
    for (i, task) in tasks.iter().enumerate() {
        println!("\n📝 Task {}: {}", i + 1, task.name);
        println!("   Input: {}", task.input);

        match agent.execute(task).await {
            Ok(response) => {
                println!("   ✅ Response: {}", response.content);
                println!(
                    "   ⏱️  Execution time: {}ms",
                    response.stats.execution_time_ms
                );
                println!("   🔧 Tool calls: {}", response.stats.tool_calls_count);

                if !response.tool_outputs.is_empty() {
                    println!("   🛠️  Tool outputs:");
                    for output in &response.tool_outputs {
                        println!(
                            "      - {}: {}",
                            output.tool_name,
                            if output.success {
                                "✅ Success"
                            } else {
                                "❌ Failed"
                            }
                        );
                    }
                }
            }
            Err(e) => {
                println!("   ❌ Error: {}", e);
            }
        }
    }

    // Demonstrate agent health check
    println!("\n🏥 Agent Health Check");
    match agent.health_check().await {
        Ok(health) => {
            println!("   Status: {:?}", health.status);
            println!("   Message: {}", health.message);
            println!("   Metrics: {:?}", health.metrics);
        }
        Err(e) => {
            println!("   ❌ Health check failed: {}", e);
        }
    }

    println!("\n🎉 Basic agent example completed!");
    Ok(())
}
