//! Comprehensive Agent Example with Real OpenAI Integration
//!
//! This example demonstrates practical agent usage patterns similar to LlamaIndex,
//! including tool integration, multi-step reasoning, and real API calls.

use cheungfun_agents::{
    agent::{
        base::{AgentContext, BaseAgent},
        multi_agent::{
            AgentRole, HandoffStrategy, MultiAgentOrchestrator, MultiAgentOrchestratorBuilder,
        },
        react::ReActAgent,
    },
    error::Result,
    llm::SiumaiLlmClient,
    tool::{
        builtin::{CalculatorTool, ShellTool, WeatherTool, WikipediaTool},
        ToolRegistry,
    },
    types::{AgentMessage, AgentResponse},
};
use cheungfun_core::{
    llm::{LlmClientConfig, LlmProvider},
    memory::chat::ChatMemoryBuffer,
    traits::BaseMemory,
};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

/// Comprehensive example showcasing different agent patterns
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 Starting Comprehensive Agent Example");
    println!("=".repeat(60));

    // Run different example scenarios
    single_agent_calculator_example().await?;
    println!();

    research_assistant_example().await?;
    println!();

    multi_agent_research_pipeline().await?;
    println!();

    println!("✅ All examples completed successfully!");
    Ok(())
}

/// Example 1: Single Agent with Calculator Tool (Similar to LlamaIndex basic example)
async fn single_agent_calculator_example() -> Result<()> {
    println!("📊 Example 1: Single Agent Calculator");
    println!("{}", "-".repeat(40));

    // Setup LLM client with OpenAI
    let llm_config = LlmClientConfig::builder()
        .provider(LlmProvider::OpenAi)
        .model("gpt-4".to_string())
        .api_key(std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set"))
        .temperature(0.1)
        .max_tokens(Some(1000))
        .build();

    let llm_client = Arc::new(SiumaiLlmClient::new(llm_config)?);

    // Create memory
    let memory = Arc::new(Mutex::new(ChatMemoryBuffer::new(10)));

    // Setup tools
    let mut tool_registry = ToolRegistry::new();
    tool_registry
        .register_tool("calculator".to_string(), Arc::new(CalculatorTool::new()))
        .await?;

    // Create ReAct agent
    let agent = ReActAgent::builder()
        .llm_client(llm_client)
        .memory(memory)
        .tool_registry(Arc::new(tool_registry))
        .system_prompt("You are a helpful math assistant. Use the calculator tool for any mathematical calculations.".to_string())
        .max_iterations(5)
        .build()?;

    // Test complex calculation
    let message = AgentMessage::user(
        "Calculate the compound interest for $10,000 invested at 5% annual rate for 10 years, \
         compounded quarterly. Use the formula A = P(1 + r/n)^(nt)",
    );

    println!("User: {}", message.content);

    let mut context = AgentContext::new();
    let response = agent.chat(message, Some(&mut context)).await?;

    println!("Agent: {}", response.content);
    println!("Tool calls made: {}", response.metadata.len());

    Ok(())
}

/// Example 2: Multi-Tool Research Assistant (Similar to LlamaIndex retrieval agent)
async fn research_assistant_example() -> Result<()> {
    println!("🔍 Example 2: Multi-Tool Research Assistant");
    println!("{}", "-".repeat(40));

    // Setup LLM client
    let llm_config = LlmClientConfig::builder()
        .provider(LlmProvider::OpenAi)
        .model("gpt-4".to_string())
        .api_key(std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set"))
        .temperature(0.3)
        .max_tokens(Some(1500))
        .build();

    let llm_client = Arc::new(SiumaiLlmClient::new(llm_config)?);
    let memory = Arc::new(Mutex::new(ChatMemoryBuffer::new(15)));

    // Setup multiple tools
    let mut tool_registry = ToolRegistry::new();
    tool_registry
        .register_tool("wikipedia".to_string(), Arc::new(WikipediaTool::new()))
        .await?;
    tool_registry
        .register_tool("calculator".to_string(), Arc::new(CalculatorTool::new()))
        .await?;
    tool_registry
        .register_tool("weather".to_string(), Arc::new(WeatherTool::new()))
        .await?;

    // Safe shell commands for demo
    let safe_shell = ShellTool::with_allowed_commands(vec![
        "echo".to_string(),
        "date".to_string(),
        "pwd".to_string(),
        "ls".to_string(),
    ]);
    tool_registry
        .register_tool("shell".to_string(), Arc::new(safe_shell))
        .await?;

    // Create research assistant agent
    let agent = ReActAgent::builder()
        .llm_client(llm_client)
        .memory(memory)
        .tool_registry(Arc::new(tool_registry))
        .system_prompt(
            "You are an expert research assistant with access to multiple tools:
            - Wikipedia: For factual information and research
            - Calculator: For mathematical calculations
            - Weather: For current weather information
            - Shell: For basic system commands (limited for security)
            
            Always use the most appropriate tool for each task. For complex research questions,
            break them down into steps and use multiple tools as needed."
                .to_string(),
        )
        .max_iterations(8)
        .build()?;

    // Complex research query
    let message = AgentMessage::user(
        "I'm planning a trip to Tokyo next month. Can you research:
        1. Basic information about Tokyo from Wikipedia
        2. Current weather in Tokyo
        3. Calculate the time difference if I'm in New York (Tokyo is UTC+9, New York is UTC-5)
        4. Show me the current date and time on this system
        
        Please provide a comprehensive summary for my trip planning.",
    );

    println!("User: {}", message.content);

    let mut context = AgentContext::new();
    let response = agent.chat(message, Some(&mut context)).await?;

    println!("Agent: {}", response.content);
    println!(
        "Research completed with {} tool interactions",
        response.metadata.len()
    );

    Ok(())
}

/// Example 3: Multi-Agent Research Pipeline (Similar to LlamaIndex multi-agent workflow)
async fn multi_agent_research_pipeline() -> Result<()> {
    println!("👥 Example 3: Multi-Agent Research Pipeline");
    println!("{}", "-".repeat(40));

    // Setup base LLM config
    let base_config = LlmClientConfig::builder()
        .provider(LlmProvider::OpenAi)
        .model("gpt-4".to_string())
        .api_key(std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set"))
        .max_tokens(Some(1200))
        .build();

    // Create specialized agents

    // 1. Research Agent - Gathers information
    let research_llm = Arc::new(SiumaiLlmClient::new(
        base_config.clone_with_temperature(0.2),
    )?);
    let research_memory = Arc::new(Mutex::new(ChatMemoryBuffer::new(10)));
    let mut research_tools = ToolRegistry::new();
    research_tools
        .register_tool("wikipedia".to_string(), Arc::new(WikipediaTool::new()))
        .await?;
    research_tools
        .register_tool("calculator".to_string(), Arc::new(CalculatorTool::new()))
        .await?;

    let research_agent = Arc::new(ReActAgent::builder()
        .llm_client(research_llm)
        .memory(research_memory)
        .tool_registry(Arc::new(research_tools))
        .system_prompt(
            "You are a specialized research agent. Your role is to gather comprehensive, \
             accurate information using available tools. Focus on facts, data, and reliable sources. \
             When you have completed your research, indicate this by saying 'RESEARCH_COMPLETE:'".to_string()
        )
        .max_iterations(6)
        .build()?) as Arc<dyn BaseAgent>;

    // 2. Analysis Agent - Analyzes and processes information
    let analysis_llm = Arc::new(SiumaiLlmClient::new(
        base_config.clone_with_temperature(0.4),
    )?);
    let analysis_memory = Arc::new(Mutex::new(ChatMemoryBuffer::new(10)));
    let mut analysis_tools = ToolRegistry::new();
    analysis_tools
        .register_tool("calculator".to_string(), Arc::new(CalculatorTool::new()))
        .await?;

    let analysis_agent = Arc::new(
        ReActAgent::builder()
            .llm_client(analysis_llm)
            .memory(analysis_memory)
            .tool_registry(Arc::new(analysis_tools))
            .system_prompt(
                "You are a specialized analysis agent. Your role is to analyze research data, \
             identify patterns, draw insights, and perform calculations. You receive information \
             from the research agent and provide analytical conclusions. \
             When analysis is complete, indicate with 'ANALYSIS_COMPLETE:'"
                    .to_string(),
            )
            .max_iterations(5)
            .build()?,
    ) as Arc<dyn BaseAgent>;

    // 3. Writer Agent - Creates final summary
    let writer_llm = Arc::new(SiumaiLlmClient::new(
        base_config.clone_with_temperature(0.6),
    )?);
    let writer_memory = Arc::new(Mutex::new(ChatMemoryBuffer::new(10)));
    let writer_tools = ToolRegistry::new();

    let writer_agent = Arc::new(ReActAgent::builder()
        .llm_client(writer_llm)
        .memory(writer_memory)
        .tool_registry(Arc::new(writer_tools))
        .system_prompt(
            "You are a specialized writing agent. Your role is to take research data and analysis \
             and create clear, well-structured, comprehensive reports. Focus on clarity, organization, \
             and actionable insights. Create a final summary that would be useful for decision-making.".to_string()
        )
        .max_iterations(3)
        .build()?) as Arc<dyn BaseAgent>;

    // Create agent roles
    let research_role = AgentRole {
        agent_id: Uuid::new_v4(),
        role: "Researcher".to_string(),
        description: "Gathers comprehensive information using tools".to_string(),
        capabilities: vec![
            "information_gathering".to_string(),
            "fact_checking".to_string(),
        ],
        preferred_tasks: vec!["research".to_string(), "data_collection".to_string()],
    };

    let analysis_role = AgentRole {
        agent_id: Uuid::new_v4(),
        role: "Analyst".to_string(),
        description: "Analyzes data and identifies insights".to_string(),
        capabilities: vec![
            "data_analysis".to_string(),
            "pattern_recognition".to_string(),
        ],
        preferred_tasks: vec!["analysis".to_string(), "calculations".to_string()],
    };

    let writer_role = AgentRole {
        agent_id: Uuid::new_v4(),
        role: "Writer".to_string(),
        description: "Creates clear, structured reports".to_string(),
        capabilities: vec!["content_creation".to_string(), "summarization".to_string()],
        preferred_tasks: vec!["writing".to_string(), "reporting".to_string()],
    };

    // Build multi-agent orchestrator
    let orchestrator = MultiAgentOrchestratorBuilder::new()
        .name("Research Pipeline".to_string())
        .handoff_strategy(HandoffStrategy::Sequential)
        .max_handoffs(3)
        .add_agent(research_agent, research_role)
        .add_agent(analysis_agent, analysis_role)
        .add_agent(writer_agent, writer_role)
        .build()
        .await?;

    // Execute complex research task
    let initial_message = AgentMessage::user(
        "Please research and analyze Rust programming language adoption in 2024:
        1. Find information about Rust's current market position and usage trends
        2. Calculate growth metrics if any numerical data is available  
        3. Analyze the factors driving adoption
        4. Provide a comprehensive report with actionable insights for a development team considering Rust adoption"
    );

    println!("Research Pipeline Query: {}", initial_message.content);
    println!("Starting multi-agent collaboration...");

    let result = orchestrator.execute(initial_message).await?;

    println!("Pipeline Results:");
    println!("- Total agents involved: {}", result.responses.len());
    println!("- Handoffs completed: {}", result.handoff_count);
    println!("- Execution time: {}ms", result.execution_time_ms);

    // Display each agent's contribution
    for (i, response) in result.responses.iter().enumerate() {
        println!("\nAgent {} Response:", i + 1);
        println!("{}", response.content);
        println!("Tool calls: {}", response.metadata.len());
    }

    // Show handoff history
    if !result.handoff_history.is_empty() {
        println!("\nHandoff History:");
        for handoff in &result.handoff_history {
            println!(
                "- {} → {}: {}",
                handoff.from_agent, handoff.to_agent, handoff.reason
            );
        }
    }

    Ok(())
}

/// Extension trait for LlmClientConfig to easily create variants
trait ConfigExtensions {
    fn clone_with_temperature(&self, temperature: f32) -> Self;
}

impl ConfigExtensions for LlmClientConfig {
    fn clone_with_temperature(&self, temperature: f32) -> Self {
        LlmClientConfig::builder()
            .provider(self.provider.clone())
            .model(self.model.clone())
            .api_key(self.api_key.clone())
            .temperature(temperature)
            .max_tokens(self.max_tokens)
            .build()
    }
}
