//! Demonstration of the new workflow-based agent system
//!
//! This example shows how to use the new ReAct agent with the workflow system.

use async_trait::async_trait;
use cheungfun_agents::{
    error::Result,
    llm::LlmClientManager,
    tool::{Tool, ToolContext, ToolResult},
    types::ToolSchema,
    workflow::{
        react::{ReActAgent, ReActConfig},
        WorkflowContext,
    },
};
use cheungfun_core::{traits::BaseMemory, ChatMessage, MessageRole};
use std::sync::Arc;
use tokio;

/// Simple memory implementation for demonstration
#[derive(Debug)]
struct SimpleMemory {
    messages: Vec<ChatMessage>,
}

impl SimpleMemory {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }
}

#[async_trait]
impl BaseMemory for SimpleMemory {
    async fn get_messages(&self) -> cheungfun_core::Result<Vec<ChatMessage>> {
        Ok(self.messages.clone())
    }

    async fn add_message(&mut self, message: ChatMessage) -> cheungfun_core::Result<()> {
        self.messages.push(message);
        Ok(())
    }

    async fn clear(&mut self) -> cheungfun_core::Result<()> {
        self.messages.clear();
        Ok(())
    }

    async fn get_memory_variables(
        &self,
    ) -> cheungfun_core::Result<std::collections::HashMap<String, String>> {
        let mut vars = std::collections::HashMap::new();
        vars.insert("message_count".to_string(), self.messages.len().to_string());
        vars.insert(
            "total_chars".to_string(),
            self.messages
                .iter()
                .map(|msg| msg.content.len())
                .sum::<usize>()
                .to_string(),
        );
        Ok(vars)
    }
}

/// Simple calculator tool for demonstration
#[derive(Debug)]
struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "calculator".to_string(),
            description: "Perform basic arithmetic calculations".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2')"
                    }
                },
                "required": ["expression"]
            }),
            output_schema: None,
            dangerous: false,
            metadata: std::collections::HashMap::new(),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult> {
        let expression = arguments
            .get("expression")
            .and_then(|v| v.as_str())
            .unwrap_or("0");

        // Simple expression evaluation (just for demo)
        let result = match expression {
            "2 + 2" => "4",
            "10 - 3" => "7",
            "5 * 6" => "30",
            "15 / 3" => "5",
            _ => "Unable to calculate this expression",
        };

        Ok(ToolResult::success(format!("The result is: {}", result)))
    }
}

/// Simple search tool for demonstration
#[derive(Debug)]
struct SearchTool;

#[async_trait]
impl Tool for SearchTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "search".to_string(),
            description: "Search for information on the web".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }),
            output_schema: None,
            dangerous: false,
            metadata: std::collections::HashMap::new(),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult> {
        let query = arguments
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Mock search results
        let result = format!("Found information about '{}': This is a mock search result for demonstration purposes.", query);

        Ok(ToolResult::success(result))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Starting Cheungfun Workflow Demo");

    // Create tools
    let calculator: Arc<dyn Tool> = Arc::new(CalculatorTool);
    let search: Arc<dyn Tool> = Arc::new(SearchTool);
    let tools = vec![calculator, search];

    // Create LLM client manager (mock for now)
    let llm_manager = Arc::new(LlmClientManager::new());

    // Create ReAct configuration
    let config = ReActConfig::default()
        .with_max_steps(5)
        .with_verbose(true)
        .with_temperature(0.1);

    // Build ReAct agent
    let agent = ReActAgent::builder()
        .name("Demo ReAct Agent")
        .description("A demonstration ReAct agent with calculator and search tools")
        .tools(tools)
        .config(config)
        .llm_manager(llm_manager)
        .build()?;

    println!("✅ ReAct agent created successfully!");
    println!("📊 Agent stats: {:?}", agent.stats());

    // Create workflow context
    let mut context = WorkflowContext::new();
    context.set_variable("demo_mode", serde_json::json!(true));

    // Create memory
    let mut memory = SimpleMemory::new();

    // Create test input
    let input_message = ChatMessage {
        role: MessageRole::User,
        content: "What is 2 + 2?".to_string(),
        metadata: None,
        timestamp: chrono::Utc::now(),
    };

    println!("💬 User input: {}", input_message.content);

    // Test the agent (simplified version since LLM integration is not complete)
    println!("🤖 Agent would process this input using ReAct reasoning...");
    println!("📝 Available tools: calculator, search");
    println!("🔄 ReAct process would be:");
    println!("   1. Thought: I need to calculate 2 + 2");
    println!("   2. Action: calculator");
    println!("   3. Action Input: {{\"expression\": \"2 + 2\"}}");
    println!("   4. Observation: The result is: 4");
    println!("   5. Thought: I now know the final answer");
    println!("   6. Final Answer: 2 + 2 equals 4");

    // Test tool execution directly
    println!("\n🔧 Testing tool execution directly:");
    let calc_tool = CalculatorTool;
    let calc_args = serde_json::json!({"expression": "2 + 2"});
    let calc_context = ToolContext::new();

    match calc_tool.execute(calc_args, &calc_context).await {
        Ok(result) => println!("✅ Calculator result: {}", result.content),
        Err(e) => println!("❌ Calculator error: {}", e),
    }

    // Test search tool
    let search_tool = SearchTool;
    let search_args = serde_json::json!({"query": "Rust programming"});
    let search_context = ToolContext::new();

    match search_tool.execute(search_args, &search_context).await {
        Ok(result) => println!("✅ Search result: {}", result.content),
        Err(e) => println!("❌ Search error: {}", e),
    }

    println!("\n🎉 Workflow demo completed successfully!");
    println!("📈 New workflow system features:");
    println!("   ✅ Type-safe reasoning steps");
    println!("   ✅ Workflow context management");
    println!("   ✅ Tool execution system");
    println!("   ✅ Event-driven architecture");
    println!("   ✅ Memory integration");
    println!("   ✅ Streaming support (planned)");

    Ok(())
}
