//! Custom Tools Development Guide
//!
//! This example shows how to create your own custom tools for agents,
//! demonstrating different patterns and best practices for tool development.

use async_trait::async_trait;
use cheungfun_agents::{
    agent::{
        base::{AgentContext, BaseAgent},
        react::{ReActAgent, ReActConfig},
    },
    error::{AgentError, Result},
    llm::SiumaiLlmClient,
    tool::{create_simple_schema, Tool, ToolContext, ToolRegistry, ToolResult},
    types::{AgentMessage, ToolSchema},
};
use serde::Deserialize;
use std::{collections::HashMap, sync::Arc};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🛠️ Custom Tools Development Guide");
    println!("{}", "=".repeat(40));

    // Demonstrate different custom tool patterns
    simple_custom_tool_example().await?;
    println!();

    stateful_tool_example().await?;
    println!();

    validation_tool_example().await?;
    println!();

    println!("✅ All custom tool patterns demonstrated!");
    Ok(())
}

/// Example 1: Simple Custom Tool
/// A basic tool that performs a specific operation with minimal complexity
#[derive(Debug, Clone)]
pub struct GreetingTool {
    name: String,
    default_language: String,
}

impl GreetingTool {
    pub fn new() -> Self {
        Self {
            name: "greeting".to_string(),
            default_language: "English".to_string(),
        }
    }

    pub fn with_language(language: impl Into<String>) -> Self {
        Self {
            name: "greeting".to_string(),
            default_language: language.into(),
        }
    }
}

#[async_trait]
impl Tool for GreetingTool {
    fn schema(&self) -> ToolSchema {
        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Name of the person to greet"
            }),
        );
        properties.insert(
            "language".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Language for greeting (optional)",
                "default": self.default_language
            }),
        );

        ToolSchema {
            name: self.name.clone(),
            description: "Generate personalized greetings in different languages".to_string(),
            input_schema: create_simple_schema(properties, vec!["name".to_string()]),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "greeting": {
                        "type": "string",
                        "description": "The generated greeting message"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language used for the greeting"
                    }
                }
            })),
            dangerous: false,
            metadata: HashMap::new(),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult> {
        #[derive(Deserialize)]
        struct GreetingArgs {
            name: String,
            #[serde(default)]
            language: Option<String>,
        }

        let args: GreetingArgs = serde_json::from_value(arguments)
            .map_err(|e| AgentError::tool(&self.name, format!("Invalid arguments: {e}")))?;

        let language = args
            .language
            .unwrap_or_else(|| self.default_language.clone());

        let greeting = match language.to_lowercase().as_str() {
            "spanish" | "español" => format!("¡Hola, {}! ¿Cómo estás?", args.name),
            "french" | "français" => format!("Bonjour, {} ! Comment allez-vous ?", args.name),
            "german" | "deutsch" => format!("Hallo, {}! Wie geht es Ihnen?", args.name),
            "japanese" | "日本語" => format!("こんにちは、{}さん！元気ですか？", args.name),
            "chinese" | "中文" => format!("你好，{}！你好吗？", args.name),
            _ => format!("Hello, {}! How are you today?", args.name), // Default to English
        };

        let result = ToolResult::success(greeting.clone())
            .with_metadata("greeting".to_string(), serde_json::json!(greeting))
            .with_metadata("language".to_string(), serde_json::json!(language));

        Ok(result)
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "greeting".to_string(),
            "multilingual".to_string(),
            "personalization".to_string(),
        ]
    }
}

/// Example 2: Stateful Tool
/// A tool that maintains internal state between calls
#[derive(Debug)]
pub struct CounterTool {
    name: String,
    count: std::sync::Mutex<i64>,
    step: i64,
}

impl CounterTool {
    pub fn new() -> Self {
        Self {
            name: "counter".to_string(),
            count: std::sync::Mutex::new(0),
            step: 1,
        }
    }

    pub fn with_step(step: i64) -> Self {
        Self {
            name: "counter".to_string(),
            count: std::sync::Mutex::new(0),
            step,
        }
    }
}

impl Clone for CounterTool {
    fn clone(&self) -> Self {
        let current_count = *self.count.lock().unwrap();
        Self {
            name: self.name.clone(),
            count: std::sync::Mutex::new(current_count),
            step: self.step,
        }
    }
}

#[async_trait]
impl Tool for CounterTool {
    fn schema(&self) -> ToolSchema {
        let mut properties = HashMap::new();
        properties.insert(
            "action".to_string(),
            serde_json::json!({
                "type": "string",
                "enum": ["increment", "decrement", "reset", "get"],
                "description": "Action to perform on the counter"
            }),
        );
        properties.insert(
            "amount".to_string(),
            serde_json::json!({
                "type": "number",
                "description": "Amount to change counter by (optional, uses default step)",
                "default": self.step
            }),
        );

        ToolSchema {
            name: self.name.clone(),
            description: "A stateful counter tool that can increment, decrement, reset, or get the current value".to_string(),
            input_schema: create_simple_schema(properties, vec!["action".to_string()]),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "current_value": {
                        "type": "number",
                        "description": "Current counter value"
                    },
                    "action_performed": {
                        "type": "string",
                        "description": "Action that was performed"
                    }
                }
            })),
            dangerous: false,
            metadata: HashMap::new(),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult> {
        #[derive(Deserialize)]
        struct CounterArgs {
            action: String,
            amount: Option<i64>,
        }

        let args: CounterArgs = serde_json::from_value(arguments)
            .map_err(|e| AgentError::tool(&self.name, format!("Invalid arguments: {e}")))?;

        let mut count = self.count.lock().unwrap();
        let amount = args.amount.unwrap_or(self.step);

        match args.action.to_lowercase().as_str() {
            "increment" => *count += amount,
            "decrement" => *count -= amount,
            "reset" => *count = 0,
            "get" => {} // Just return current value
            _ => {
                return Ok(ToolResult::error(format!(
                    "Unknown action: {}",
                    args.action
                )))
            }
        }

        let current_value = *count;
        let message = format!(
            "Counter action '{}' performed. Current value: {}",
            args.action, current_value
        );

        let result = ToolResult::success(message)
            .with_metadata(
                "current_value".to_string(),
                serde_json::json!(current_value),
            )
            .with_metadata(
                "action_performed".to_string(),
                serde_json::json!(args.action),
            );

        Ok(result)
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "counter".to_string(),
            "stateful".to_string(),
            "arithmetic".to_string(),
        ]
    }
}

/// Example 3: Validation Tool
/// A tool that demonstrates input validation and error handling
#[derive(Debug, Clone)]
pub struct EmailValidatorTool {
    name: String,
}

impl EmailValidatorTool {
    pub fn new() -> Self {
        Self {
            name: "email_validator".to_string(),
        }
    }
}

#[async_trait]
impl Tool for EmailValidatorTool {
    fn schema(&self) -> ToolSchema {
        let mut properties = HashMap::new();
        properties.insert(
            "email".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Email address to validate",
                "format": "email"
            }),
        );
        properties.insert(
            "strict".to_string(),
            serde_json::json!({
                "type": "boolean",
                "description": "Whether to use strict validation rules",
                "default": false
            }),
        );

        ToolSchema {
            name: self.name.clone(),
            description: "Validate email addresses and provide detailed feedback".to_string(),
            input_schema: create_simple_schema(properties, vec!["email".to_string()]),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "is_valid": {
                        "type": "boolean",
                        "description": "Whether the email is valid"
                    },
                    "email": {
                        "type": "string",
                        "description": "The email that was validated"
                    },
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of validation issues found"
                    },
                    "suggestions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Suggestions for fixing the email"
                    }
                }
            })),
            dangerous: false,
            metadata: HashMap::new(),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult> {
        #[derive(Deserialize)]
        struct EmailArgs {
            email: String,
            #[serde(default)]
            strict: bool,
        }

        let args: EmailArgs = serde_json::from_value(arguments)
            .map_err(|e| AgentError::tool(&self.name, format!("Invalid arguments: {e}")))?;

        let (is_valid, issues, suggestions) = self.validate_email(&args.email, args.strict);

        let status = if is_valid { "valid" } else { "invalid" };
        let message = format!(
            "Email '{}' is {}. {} issue(s) found.",
            args.email,
            status,
            issues.len()
        );

        let result = ToolResult::success(message)
            .with_metadata("is_valid".to_string(), serde_json::json!(is_valid))
            .with_metadata("email".to_string(), serde_json::json!(args.email))
            .with_metadata("issues".to_string(), serde_json::json!(issues))
            .with_metadata("suggestions".to_string(), serde_json::json!(suggestions));

        Ok(result)
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "validation".to_string(),
            "email".to_string(),
            "analysis".to_string(),
        ]
    }
}

impl EmailValidatorTool {
    fn validate_email(&self, email: &str, strict: bool) -> (bool, Vec<String>, Vec<String>) {
        let mut issues = Vec::new();
        let mut suggestions = Vec::new();

        // Basic validation
        if email.is_empty() {
            issues.push("Email is empty".to_string());
            suggestions.push("Please provide an email address".to_string());
            return (false, issues, suggestions);
        }

        // Check for @ symbol
        if !email.contains('@') {
            issues.push("Missing @ symbol".to_string());
            suggestions.push("Add @ symbol between local and domain parts".to_string());
        }

        // Check for multiple @ symbols
        if email.matches('@').count() > 1 {
            issues.push("Multiple @ symbols found".to_string());
            suggestions.push("Use only one @ symbol".to_string());
        }

        // Check for domain
        if let Some(at_pos) = email.rfind('@') {
            let domain = &email[at_pos + 1..];
            if domain.is_empty() {
                issues.push("Missing domain".to_string());
                suggestions.push("Add domain after @ symbol (e.g., @example.com)".to_string());
            } else if !domain.contains('.') {
                issues.push("Domain missing top-level domain".to_string());
                suggestions.push("Add top-level domain (e.g., .com, .org)".to_string());
            }

            let local = &email[..at_pos];
            if local.is_empty() {
                issues.push("Missing local part".to_string());
                suggestions.push("Add local part before @ symbol".to_string());
            }
        }

        // Strict validation
        if strict {
            if email.starts_with('.') || email.ends_with('.') {
                issues.push("Email cannot start or end with period".to_string());
            }

            if email.contains("..") {
                issues.push("Consecutive periods not allowed".to_string());
            }
        }

        let is_valid = issues.is_empty();
        (is_valid, issues, suggestions)
    }
}

/// Demonstrate simple custom tool usage
async fn simple_custom_tool_example() -> Result<()> {
    println!("👋 Example 1: Simple Custom Tool (Greeting Tool)");
    println!("{}", "-".repeat(30));

    let llm_client = SiumaiLlmClient::openai(
        std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()),
        "gpt-4",
    )
    .await?;

    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(GreetingTool::new()))?;

    let agent = ReActAgent::with_llm_client(
        ReActConfig::new("Greeting Agent").with_max_iterations(2),
        Arc::new(tools),
        llm_client,
    );

    let message = AgentMessage::user("Greet me in Spanish and French");
    println!("User: {}", message.content);
    let mut context = AgentContext::new();
    let response = agent.chat(message, Some(&mut context)).await?;
    println!("Agent: {}", response.content);

    Ok(())
}

/// Demonstrate stateful tool usage
async fn stateful_tool_example() -> Result<()> {
    println!("🔢 Example 2: Stateful Tool (Counter Tool)");
    println!("{}", "-".repeat(30));

    let llm_client = SiumaiLlmClient::openai(
        std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()),
        "gpt-4",
    )
    .await?;

    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(CounterTool::new()))?;

    let agent = ReActAgent::with_llm_client(
        ReActConfig::new("Counter Agent").with_max_iterations(4),
        Arc::new(tools),
        llm_client,
    );

    let message = AgentMessage::user("Increment the counter 3 times, then get the current value");
    println!("User: {}", message.content);
    let mut context = AgentContext::new();
    let response = agent.chat(message, Some(&mut context)).await?;
    println!("Agent: {}", response.content);

    Ok(())
}

/// Demonstrate validation tool usage
async fn validation_tool_example() -> Result<()> {
    println!("✅ Example 3: Validation Tool (Email Validator)");
    println!("{}", "-".repeat(30));

    let llm_client = SiumaiLlmClient::openai(
        std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()),
        "gpt-4",
    )
    .await?;

    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(EmailValidatorTool::new()))?;

    let agent = ReActAgent::with_llm_client(
        ReActConfig::new("Validation Agent").with_max_iterations(3),
        Arc::new(tools),
        llm_client,
    );

    let message = AgentMessage::user(
        "Validate these emails: user@example.com, invalid-email, test@incomplete",
    );
    println!("User: {}", message.content);
    let mut context = AgentContext::new();
    let response = agent.chat(message, Some(&mut context)).await?;
    println!("Agent: {}", response.content);

    Ok(())
}

/// Best practices for custom tool development
#[allow(dead_code)]
fn custom_tool_best_practices() {
    println!("💡 Custom Tool Development Best Practices:");
    println!("1. **Clear Schemas**: Define clear input/output schemas with good descriptions");
    println!("2. **Error Handling**: Always validate inputs and provide helpful error messages");
    println!("3. **Documentation**: Use doc comments to explain tool purpose and usage");
    println!("4. **Capabilities**: Define clear capabilities that agents can understand");
    println!("5. **Thread Safety**: Use appropriate synchronization for stateful tools");
    println!("6. **Async Operations**: Use async/await for I/O operations");
    println!("7. **Resource Management**: Clean up resources properly");
    println!("8. **Testing**: Write unit tests for your custom tools");
}
