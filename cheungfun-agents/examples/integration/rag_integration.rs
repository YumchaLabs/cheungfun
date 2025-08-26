//! RAG-Agent Integration Demo
//!
//! This example demonstrates the deep integration between our RAG system
//! and agent framework, showcasing how to properly leverage existing architecture.

use cheungfun_agents::{
    prelude::*,
    tool::rag_query_tool::{RagQueryTool, RagQueryToolBuilder},
};
use cheungfun_core::{
    llm::MockLLM,
    prelude::*,
    traits::{BaseLLM, BaseMemory, ResponseGenerator, Retriever},
};
use cheungfun_query::prelude::*;
use std::{collections::HashMap, sync::Arc};
use tokio;

/// Mock retriever for demonstration
#[derive(Debug)]
struct MockRetriever {
    name: String,
}

impl MockRetriever {
    fn new() -> Self {
        Self {
            name: "MockRetriever".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl Retriever for MockRetriever {
    fn name(&self) -> &str {
        &self.name
    }

    async fn retrieve(&self, query: &Query) -> cheungfun_core::Result<Vec<ScoredNode>> {
        // Mock retrieval - return some example nodes
        let nodes = vec![
            ScoredNode {
                node: Node {
                    id: "node1".to_string(),
                    content: format!("This is relevant information about: {}", query.text),
                    metadata: HashMap::new(),
                },
                score: 0.95,
            },
            ScoredNode {
                node: Node {
                    id: "node2".to_string(),
                    content: "Additional context from knowledge base...".to_string(),
                    metadata: {
                        let mut m = HashMap::new();
                        m.insert(
                            "source".to_string(),
                            serde_json::Value::String("knowledge_base".to_string()),
                        );
                        m
                    },
                },
                score: 0.87,
            },
        ];

        Ok(nodes)
    }

    async fn health_check(&self) -> cheungfun_core::Result<()> {
        Ok(())
    }
}

/// Mock generator for demonstration
#[derive(Debug)]
struct MockGenerator {
    name: String,
}

impl MockGenerator {
    fn new() -> Self {
        Self {
            name: "MockGenerator".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl ResponseGenerator for MockGenerator {
    fn name(&self) -> &str {
        &self.name
    }

    async fn generate_response(
        &self,
        query: &str,
        context_nodes: Vec<ScoredNode>,
        options: &GenerationOptions,
    ) -> cheungfun_core::Result<GeneratedResponse> {
        let context_summary = context_nodes
            .iter()
            .map(|node| &node.node.content)
            .collect::<Vec<_>>()
            .join("\n");

        let response = GeneratedResponse {
            content: format!(
                "Based on the retrieved context, here's the answer to '{}': \n\nContext: {}\n\nAnswer: This is a synthesized response using the retrieved knowledge.",
                query, context_summary
            ),
            metadata: HashMap::new(),
        };

        Ok(response)
    }

    async fn health_check(&self) -> cheungfun_core::Result<()> {
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🧠 RAG-Agent Integration Demo");
    println!("=============================\n");

    // 1. Create RAG components
    let retriever = Arc::new(MockRetriever::new()) as Arc<dyn Retriever>;
    let generator = Arc::new(MockGenerator::new()) as Arc<dyn ResponseGenerator>;
    let memory = Arc::new(tokio::sync::Mutex::new(ChatMemoryBuffer::new()))
        as Arc<tokio::sync::Mutex<dyn BaseMemory>>;

    // 2. Create QueryEngine
    let query_engine = Arc::new(
        QueryEngineBuilder::new()
            .retriever(retriever.clone())
            .generator(generator.clone())
            .memory(memory.clone())
            .build()
            .map_err(|e| {
                AgentError::tool_execution(format!("Failed to create query engine: {}", e))
            })?,
    );

    println!("✅ Created QueryEngine with:");
    println!("   - Retriever: {}", retriever.name());
    println!("   - Generator: {}", generator.name());
    println!("   - Memory: Enabled");
    println!();

    // 3. Create RAG Query Tool using our existing QueryEngine
    let rag_tool = Arc::new(
        RagQueryToolBuilder::new()
            .query_engine(query_engine)
            .name("knowledge_search".to_string())
            .description("Search the knowledge base for relevant information".to_string())
            .enable_deep_research(true)
            .enable_query_rewriting(true)
            .build()?,
    );

    println!("✅ Created RAG Query Tool: {}", rag_tool.name());
    println!("   Description: {}", rag_tool.description());
    println!("   Deep Research: Enabled");
    println!("   Query Rewriting: Enabled");
    println!();

    // 4. Test the RAG tool directly
    let test_args = serde_json::json!({
        "query": "What are the benefits of using RAG in AI applications?",
        "search_mode": "hybrid",
        "top_k": 3
    });

    let tool_context = crate::tool::ToolContext::new();
    let result = rag_tool.execute(test_args, &tool_context).await?;

    println!("🔍 RAG Tool Test Results:");
    println!("Success: {}", result.success);
    if result.success {
        println!("Content: {}", result.content);
        println!(
            "Metadata: {}",
            serde_json::to_string_pretty(&result.metadata)?
        );
    } else {
        println!(
            "Error: {}",
            result.error_message().unwrap_or("Unknown error")
        );
    }
    println!();

    // 5. Create an agent that uses the RAG tool
    let llm = Arc::new(MockLLM::new()) as Arc<dyn BaseLLM>;
    let mut agent = ReActAgent::builder()
        .name("RAG-Enhanced Assistant")
        .description("An assistant that can search the knowledge base")
        .llm(llm)
        .tool(rag_tool.clone() as Arc<dyn Tool>)
        .max_iterations(5)
        .build()?;

    println!("🤖 Created RAG-Enhanced ReAct Agent:");
    println!("   Name: {}", agent.name());
    println!("   Description: {}", agent.description().unwrap_or("N/A"));
    println!("   Tools: {} (including RAG)", agent.get_tool_names().len());
    println!();

    // 6. Test the agent with a knowledge-seeking query
    let user_query =
        "Can you search for information about machine learning and explain the key concepts?";
    println!("🎯 Testing Agent with Query: '{}'", user_query);
    println!();

    let agent_message = AgentMessage::user(user_query);
    let mut context = crate::agent::base::AgentContext::new();

    let response = agent.chat(agent_message, Some(&mut context)).await?;

    println!("📋 Agent Response:");
    println!("Content: {}", response.content);
    println!("Tool Calls Made: {}", response.tool_calls.len());
    println!("Tool Outputs: {}", response.tool_outputs.len());
    println!("Execution Time: {}ms", response.stats.execution_time_ms);
    println!();

    // 7. Show RAG tool statistics
    match rag_tool.get_stats() {
        Ok(stats) => {
            println!("📊 RAG Tool Statistics:");
            println!("   Total Queries: {}", stats.total_queries);
            println!("   Deep Research Queries: {}", stats.deep_research_queries);
            println!("   Rewritten Queries: {}", stats.rewritten_queries);
            println!("   Avg Response Time: {:.2}ms", stats.avg_response_time_ms);
            println!("   Total Nodes Retrieved: {}", stats.total_nodes_retrieved);
        }
        Err(e) => {
            println!("❌ Failed to get stats: {}", e);
        }
    }
    println!();

    // 8. Demonstrate workflow integration
    println!("🔄 Workflow Integration Demo");
    println!("============================\n");

    // Create a simple workflow that uses RAG
    let workflow = SimpleWorkflowBuilder::new("Knowledge Research Workflow")
        .description("A workflow that performs knowledge research and analysis")
        .simple_step("knowledge_search", "Search Knowledge Base", agent.id())
        .build()?;

    let mut workflow_executor = SimpleWorkflowExecutor::new();
    workflow_executor.add_agent(Arc::new(agent));

    let workflow_message =
        AgentMessage::user("Research the latest trends in AI and provide a comprehensive summary");
    let workflow_result = workflow_executor
        .execute(workflow, workflow_message)
        .await?;

    println!("✅ Workflow Execution Result:");
    println!("   Status: {:?}", workflow_result.status);
    println!(
        "   Steps Completed: {}",
        workflow_result.execution_details.steps_completed
    );
    println!(
        "   Steps Failed: {}",
        workflow_result.execution_details.steps_failed
    );
    println!("   Total Time: {}ms", workflow_result.execution_time_ms);
    println!();

    // 9. Summary and recommendations
    println!("💡 Key Integration Benefits Demonstrated:");
    println!("=========================================");
    println!("✅ Deep RAG Integration: QueryEngine seamlessly becomes an agent tool");
    println!("✅ Existing Architecture Reuse: Full utilization of cheungfun-query capabilities");
    println!("✅ Flexible Configuration: Deep research, query rewriting, multiple search modes");
    println!("✅ Statistical Monitoring: Built-in performance tracking and analytics");
    println!("✅ Workflow Compatible: Works within both simple and complex workflow systems");
    println!("✅ Memory Integration: Conversation context enhances RAG retrieval");
    println!();

    println!("🚀 This demonstrates how to properly leverage our existing RAG library");
    println!("   within the agent system, following LlamaIndex's integration patterns!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rag_tool_creation() {
        let retriever = Arc::new(MockRetriever::new()) as Arc<dyn Retriever>;
        let generator = Arc::new(MockGenerator::new()) as Arc<dyn ResponseGenerator>;

        let result = RagQueryToolBuilder::new()
            .retriever(retriever)
            .generator(generator)
            .build();

        assert!(result.is_ok());
        let tool = result.unwrap();
        assert_eq!(tool.name(), "rag_query");
    }

    #[tokio::test]
    async fn test_rag_tool_execution() {
        let retriever = Arc::new(MockRetriever::new()) as Arc<dyn Retriever>;
        let generator = Arc::new(MockGenerator::new()) as Arc<dyn ResponseGenerator>;

        let tool = RagQueryToolBuilder::new()
            .retriever(retriever)
            .generator(generator)
            .build()
            .unwrap();

        let args = serde_json::json!({
            "query": "test query",
            "top_k": 2
        });

        let context = crate::tool::ToolContext::new();
        let result = tool.execute(args, &context).await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.success);
        assert!(!tool_result.content.is_empty());
    }
}
