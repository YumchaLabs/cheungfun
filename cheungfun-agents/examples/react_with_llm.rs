//! ReAct Agent with LLM Integration Example
//!
//! This example demonstrates how to use the ReAct agent with siumai LLM integration.
//! It shows how to create an agent that can reason and act using various LLM providers.
//!
//! ## Setup
//!
//! Before running this example, set up your API keys:
//!
//! ```bash
//! # For OpenAI
//! export OPENAI_API_KEY="your-openai-api-key"
//!
//! # For Anthropic
//! export ANTHROPIC_API_KEY="your-anthropic-api-key"
//!
//! # For Ollama (optional, defaults to localhost:11434)
//! export OLLAMA_BASE_URL="http://localhost:11434"
//! ```
//!
//! ## Usage
//!
//! ```bash
//! # Run with OpenAI
//! cargo run --example react_with_llm -- --provider openai --model gpt-4o-mini
//!
//! # Run with Anthropic
//! cargo run --example react_with_llm -- --provider anthropic --model claude-3-haiku-20240307
//!
//! # Run with Ollama
//! cargo run --example react_with_llm -- --provider ollama --model llama3.2
//! ```

use cheungfun_agents::{
    agent::{
        base::{AgentContext, BaseAgent},
        react::{ReActAgent, ReActConfig},
    },
    error::Result,
    llm::{LlmClientConfig, SiumaiLlmClient},
    tool::{builtin::EchoTool, ToolRegistry},
    types::{AgentCapabilities, AgentConfig, AgentMessage, MessageRole},
};
use clap::{Arg, Command};
use std::{collections::HashMap, sync::Arc};
use tracing::{info, warn, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    // Parse command line arguments
    let matches = Command::new("ReAct Agent with LLM")
        .version("1.0")
        .about("Demonstrates ReAct agent with LLM integration")
        .arg(
            Arg::new("provider")
                .long("provider")
                .value_name("PROVIDER")
                .help("LLM provider (openai, anthropic, ollama)")
                .default_value("ollama"),
        )
        .arg(
            Arg::new("model")
                .long("model")
                .value_name("MODEL")
                .help("Model name")
                .default_value("llama3.2"),
        )
        .arg(
            Arg::new("temperature")
                .long("temperature")
                .value_name("TEMP")
                .help("Temperature for response generation")
                .default_value("0.7"),
        )
        .get_matches();

    let provider = matches.get_one::<String>("provider").unwrap();
    let model = matches.get_one::<String>("model").unwrap();
    let temperature: f32 = matches
        .get_one::<String>("temperature")
        .unwrap()
        .parse()
        .unwrap_or(0.7);

    info!("🚀 Starting ReAct Agent with LLM Integration");
    info!(
        "Provider: {}, Model: {}, Temperature: {}",
        provider, model, temperature
    );

    // Create LLM client
    let llm_client = create_llm_client(provider, model, temperature).await?;
    info!("✅ LLM client created successfully");

    // Create tool registry with some basic tools
    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(Arc::new(EchoTool::new()))?;
    let tool_registry = Arc::new(tool_registry);
    info!(
        "🔧 Tool registry created with {} tools",
        tool_registry.tool_names().len()
    );

    // Create ReAct agent configuration
    let base_config = AgentConfig {
        name: "ReAct-LLM-Agent".to_string(),
        description: Some("A ReAct agent with LLM integration for reasoning and acting".to_string()),
        instructions: Some(
            "You are a helpful ReAct agent. Think step by step, use tools when needed, and provide clear answers.".to_string()
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
        max_execution_time_ms: Some(30_000),
        max_tool_calls: Some(5),
        verbose: true,
        custom_config: HashMap::new(),
    };

    let react_config = ReActConfig {
        base_config,
        max_iterations: 5,
        max_thinking_time_ms: 10_000,
        include_trace: true,
        custom_settings: HashMap::new(),
    };

    // Create ReAct agent with LLM client
    let mut agent = ReActAgent::with_llm_client(react_config, tool_registry, llm_client);
    info!("🤖 ReAct agent created successfully");

    // Test the agent with some example queries
    let test_queries = vec![
        "Hello! Can you introduce yourself?",
        "What is the capital of France? Please think step by step.",
        "Can you echo the message 'Hello from ReAct agent'?",
        "Explain the concept of artificial intelligence in simple terms.",
    ];

    for (i, query) in test_queries.iter().enumerate() {
        info!("\n📝 Test Query {}: {}", i + 1, query);

        let message = AgentMessage {
            id: uuid::Uuid::new_v4(),
            content: query.to_string(),
            role: MessageRole::User,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
            tool_calls: Vec::new(),
        };

        let mut context = AgentContext::new();

        match agent.chat(message, Some(&mut context)).await {
            Ok(response) => {
                info!("🤖 Agent Response:");
                println!("{}", response.content);

                if !response.tool_calls.is_empty() {
                    info!("🔧 Tool calls made: {}", response.tool_calls.len());
                }

                info!("⏱️  Execution time: {}ms", response.stats.execution_time_ms);
            }
            Err(e) => {
                warn!("❌ Error processing query: {}", e);
            }
        }

        // Add a small delay between queries
        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
    }

    info!("\n✅ ReAct Agent with LLM Integration demo completed!");
    Ok(())
}

/// Create LLM client based on provider
async fn create_llm_client(
    provider: &str,
    model: &str,
    temperature: f32,
) -> Result<SiumaiLlmClient> {
    let config = LlmClientConfig::new(provider, model)
        .with_temperature(temperature)
        .with_max_tokens(2048)
        .with_system_prompt(
            "You are a helpful ReAct (Reasoning and Acting) agent. \
             Follow the ReAct format: Thought -> Action -> Observation -> Final Answer. \
             Think step by step and use tools when appropriate.",
        );

    match provider {
        "openai" => {
            let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
                warn!("OPENAI_API_KEY not found in environment variables");
                cheungfun_agents::error::AgentError::invalid_configuration(
                    "OPENAI_API_KEY environment variable is required for OpenAI provider",
                )
            })?;

            info!("🔑 Using OpenAI with model: {}", model);
            SiumaiLlmClient::openai(api_key, model).await
        }
        "anthropic" => {
            let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
                warn!("ANTHROPIC_API_KEY not found in environment variables");
                cheungfun_agents::error::AgentError::invalid_configuration(
                    "ANTHROPIC_API_KEY environment variable is required for Anthropic provider",
                )
            })?;

            info!("🔑 Using Anthropic with model: {}", model);
            SiumaiLlmClient::anthropic(api_key, model).await
        }
        "ollama" => {
            let base_url = std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());

            info!("🦙 Using Ollama at {} with model: {}", base_url, model);
            SiumaiLlmClient::ollama(base_url, model).await
        }
        _ => {
            warn!("Unsupported provider: {}", provider);
            Err(cheungfun_agents::error::AgentError::invalid_configuration(
                format!(
                    "Unsupported provider: {}. Use 'openai', 'anthropic', or 'ollama'",
                    provider
                ),
            ))
        }
    }
}
