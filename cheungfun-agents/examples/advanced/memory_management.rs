//! Memory and Context Management Examples
//!
//! This example demonstrates different approaches to managing agent memory,
//! conversation context, and state persistence across agent interactions.

use cheungfun_agents::{
    agent::{
        base::{AgentContext, BaseAgent},
        react::{ReActAgent, ReActConfig},
    },
    error::Result,
    llm::SiumaiLlmClient,
    tool::{builtin::EchoTool, ToolRegistry},
    types::AgentMessage,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🧠 Memory and Context Management Examples");
    println!("{}", "=".repeat(50));

    // Demonstrate different memory management patterns
    basic_context_usage().await?;
    println!();

    persistent_context_example().await?;
    println!();

    context_sharing_example().await?;
    println!();

    println!("✅ All memory management patterns demonstrated!");
    Ok(())
}

/// Example 1: Basic Context Usage
/// Shows how to use AgentContext for single conversations
async fn basic_context_usage() -> Result<()> {
    println!("📝 Example 1: Basic Context Usage");
    println!("{}", "-".repeat(30));

    let llm_client = SiumaiLlmClient::openai(
        std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()),
        "gpt-4",
    )
    .await?;

    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(EchoTool::new()))?;

    let agent_config = ReActConfig::new("Context Demo Agent").with_max_iterations(3);

    let mut config = agent_config;
    config.base_config.instructions = Some(
        "You are a helpful assistant. Remember information from our conversation \
         and refer back to it when relevant."
            .to_string(),
    );

    let agent = ReActAgent::with_llm_client(config, Arc::new(tools), llm_client);

    // Create a context for this conversation
    let mut context = AgentContext::new();

    // Add some initial context data
    context.set_variable("user_name".to_string(), serde_json::json!("Alice"));
    context.set_variable(
        "conversation_topic".to_string(),
        serde_json::json!("memory management"),
    );

    // First interaction - establishing context
    let message1 = AgentMessage::user("Hi! My name is Alice and I'm learning about agent memory.");
    println!("User: {}", message1.content);
    let response1 = agent.chat(message1, Some(&mut context)).await?;
    println!("Agent: {}", response1.content);

    println!();

    // Second interaction - referencing previous context
    let message2 = AgentMessage::user("Can you remind me what we were discussing?");
    println!("User: {}", message2.content);
    let response2 = agent.chat(message2, Some(&mut context)).await?;
    println!("Agent: {}", response2.content);

    // Show context data
    println!("\nContext Data:");
    if let Some(name) = context.get_variable("user_name") {
        println!("- User Name: {}", name);
    }
    if let Some(topic) = context.get_variable("conversation_topic") {
        println!("- Topic: {}", topic);
    }

    Ok(())
}

/// Example 2: Persistent Context
/// Shows how to serialize and restore context across sessions
async fn persistent_context_example() -> Result<()> {
    println!("💾 Example 2: Persistent Context (Serialization)");
    println!("{}", "-".repeat(30));

    let llm_client = SiumaiLlmClient::openai(
        std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()),
        "gpt-4",
    )
    .await?;

    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(EchoTool::new()))?;

    let agent = ReActAgent::with_llm_client(
        ReActConfig::new("Persistent Agent").with_max_iterations(2),
        Arc::new(tools),
        llm_client,
    );

    // === Session 1: Create and save context ===
    println!("Session 1: Creating context...");
    let mut session1_context = AgentContext::new();
    session1_context.set_variable(
        "user_preferences".to_string(),
        serde_json::json!({
            "name": "Bob",
            "favorite_color": "blue",
            "interests": ["programming", "AI", "music"]
        }),
    );

    let message1 = AgentMessage::user(
        "Hi, I'm Bob. I love programming and music, and my favorite color is blue.",
    );
    println!("User: {}", message1.content);
    let response1 = agent.chat(message1, Some(&mut session1_context)).await?;
    println!("Agent: {}", response1.content);

    // Serialize context (in real app, you'd save this to a file or database)
    let context_json = serialize_context(&session1_context)?;
    println!("Context serialized: {} bytes", context_json.len());

    // === Session 2: Restore context ===
    println!("\nSession 2: Restoring context...");
    let mut session2_context = deserialize_context(&context_json)?;

    let message2 = AgentMessage::user("What do you remember about my preferences?");
    println!("User: {}", message2.content);
    let response2 = agent.chat(message2, Some(&mut session2_context)).await?;
    println!("Agent: {}", response2.content);

    Ok(())
}

/// Example 3: Context Sharing Between Agents
/// Shows how to share context data between different agents
async fn context_sharing_example() -> Result<()> {
    println!("🤝 Example 3: Context Sharing Between Agents");
    println!("{}", "-".repeat(30));

    let llm_client = SiumaiLlmClient::openai(
        std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()),
        "gpt-4",
    )
    .await?;

    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(EchoTool::new()))?;

    // Create two specialized agents
    let information_agent = ReActAgent::with_llm_client(
        ReActConfig::new("Information Collector").with_max_iterations(2),
        Arc::new(tools.clone()),
        llm_client.clone(),
    );

    let analysis_agent = ReActAgent::with_llm_client(
        ReActConfig::new("Information Analyzer").with_max_iterations(2),
        Arc::new(tools),
        llm_client,
    );

    // Shared context that both agents can access
    let mut shared_context = AgentContext::new();

    // === Phase 1: Information Collection ===
    println!("Phase 1: Information Collection Agent");
    shared_context.set_variable("phase".to_string(), serde_json::json!("collection"));

    let collect_message = AgentMessage::user("Please collect information: The user likes coffee, works in tech, and lives in San Francisco.");
    println!("User: {}", collect_message.content);
    let collect_response = information_agent
        .chat(collect_message, Some(&mut shared_context))
        .await?;
    println!("Information Agent: {}", collect_response.content);

    // Information agent adds data to shared context
    shared_context.set_variable(
        "user_profile".to_string(),
        serde_json::json!({
            "preferences": ["coffee"],
            "profession": "tech",
            "location": "San Francisco",
            "collected_at": chrono::Utc::now().to_rfc3339()
        }),
    );

    println!();

    // === Phase 2: Analysis ===
    println!("Phase 2: Analysis Agent (using shared context)");
    shared_context.set_variable("phase".to_string(), serde_json::json!("analysis"));

    let analyze_message =
        AgentMessage::user("Based on the collected information, what insights can you provide?");
    println!("User: {}", analyze_message.content);
    let analyze_response = analysis_agent
        .chat(analyze_message, Some(&mut shared_context))
        .await?;
    println!("Analysis Agent: {}", analyze_response.content);

    // Show shared context data
    println!("\nShared Context Data:");
    if let Some(profile) = shared_context.get_variable("user_profile") {
        println!(
            "- User Profile: {}",
            serde_json::to_string_pretty(&profile).unwrap_or_default()
        );
    }
    if let Some(phase) = shared_context.get_variable("phase") {
        println!("- Current Phase: {}", phase);
    }

    Ok(())
}

/// Helper function to serialize context for persistence
fn serialize_context(context: &AgentContext) -> Result<String> {
    // In a real implementation, you might want to create a proper serializable
    // representation of the context. For demo purposes, we'll serialize the data.
    #[derive(Serialize)]
    struct SerializableContext {
        data: std::collections::HashMap<String, serde_json::Value>,
        // In a real implementation, you might also serialize conversation history
        timestamp: String,
    }

    let serializable = SerializableContext {
        data: context.get_all_data().clone(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    serde_json::to_string(&serializable)
        .map_err(|e| cheungfun_agents::error::AgentError::tool("serialization", e.to_string()))
}

/// Helper function to deserialize context from persistence
fn deserialize_context(context_json: &str) -> Result<AgentContext> {
    #[derive(Deserialize)]
    struct SerializableContext {
        data: std::collections::HashMap<String, serde_json::Value>,
        timestamp: String,
    }

    let serializable: SerializableContext = serde_json::from_str(context_json)
        .map_err(|e| cheungfun_agents::error::AgentError::tool("deserialization", e.to_string()))?;

    let mut context = AgentContext::new();

    // Restore data
    for (key, value) in serializable.data {
        context.set_variable(key, value);
    }

    // Add restoration metadata
    context.set_variable(
        "restored_at".to_string(),
        serde_json::json!(chrono::Utc::now().to_rfc3339()),
    );
    context.set_variable(
        "original_timestamp".to_string(),
        serde_json::json!(serializable.timestamp),
    );

    Ok(context)
}

/// Context management utilities and best practices
#[allow(dead_code)]
mod context_utils {
    use super::*;

    /// Utility to create a context with user session data
    pub fn create_user_session_context(user_id: &str, session_id: &str) -> AgentContext {
        let mut context = AgentContext::new();
        context.set_variable("user_id".to_string(), serde_json::json!(user_id));
        context.set_variable("session_id".to_string(), serde_json::json!(session_id));
        context.set_variable(
            "created_at".to_string(),
            serde_json::json!(chrono::Utc::now().to_rfc3339()),
        );
        context
    }

    /// Utility to merge contexts (useful for context sharing)
    pub fn merge_contexts(base: &mut AgentContext, other: &AgentContext) {
        for (key, value) in other.get_all_data() {
            // Only merge if key doesn't exist in base (don't overwrite)
            if base.get_variable(key).is_none() {
                base.set_variable(key.clone(), value.clone());
            }
        }
    }

    /// Utility to clean up old context data
    pub fn cleanup_context(context: &mut AgentContext, max_age_hours: i64) {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(max_age_hours);

        // Remove entries with timestamps older than cutoff
        // This is a simplified example - in practice you'd need to track timestamps per entry
        if let Some(created_at) = context.get_variable("created_at") {
            if let Ok(timestamp) = created_at
                .as_str()
                .unwrap_or("")
                .parse::<chrono::DateTime<chrono::Utc>>()
            {
                if timestamp < cutoff {
                    // In a real implementation, you might clear specific old entries
                    println!(
                        "Context is older than {} hours and should be cleaned up",
                        max_age_hours
                    );
                }
            }
        }
    }

    /// Best practices documentation
    pub fn memory_management_best_practices() {
        println!("💡 Memory Management Best Practices:");
        println!(
            "1. **Context Size**: Keep context data reasonably small to avoid performance issues"
        );
        println!(
            "2. **Data Structure**: Use structured data (JSON) for complex context information"
        );
        println!("3. **Cleanup**: Regularly clean up old or unnecessary context data");
        println!("4. **Persistence**: Serialize important context for long-term storage");
        println!(
            "5. **Security**: Be careful with sensitive data in context - consider encryption"
        );
        println!(
            "6. **Versioning**: Consider versioning your context schema for backward compatibility"
        );
        println!("7. **Sharing**: Design context sharing patterns carefully to avoid conflicts");
        println!("8. **Memory Limits**: Set appropriate limits to prevent memory leaks");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_serialization() {
        let mut context = AgentContext::new();
        context.set_variable("test_key".to_string(), serde_json::json!("test_value"));

        // Test serialization
        let serialized = serialize_context(&context).unwrap();
        assert!(!serialized.is_empty());

        // Test deserialization
        let deserialized = deserialize_context(&serialized).unwrap();
        let value = deserialized.get_variable("test_key").unwrap();
        assert_eq!(value.as_str().unwrap(), "test_value");
    }
}
