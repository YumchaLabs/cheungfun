//! Workflow orchestration example demonstrating multi-agent coordination.

use cheungfun_agents::prelude::*;
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🎭 Cheungfun Agents - Workflow Orchestration Example");
    println!("===================================================");

    // Create different types of agents
    let researcher = AgentBuilder::researcher()
        .name("research_agent")
        .description("Specialized in research and information gathering")
        .build()?;

    let analyst = AgentBuilder::assistant()
        .name("analysis_agent")
        .description("Specialized in data analysis and insights")
        .build()?;

    let writer = AgentBuilder::assistant()
        .name("writing_agent")
        .description("Specialized in content creation and writing")
        .build()?;

    println!("✅ Created {} agents", 3);

    // Create an orchestrator
    let mut orchestrator = AgentOrchestrator::new();

    // Register agents
    orchestrator.register_agent(Arc::clone(&researcher))?;
    orchestrator.register_agent(Arc::clone(&analyst))?;
    orchestrator.register_agent(Arc::clone(&writer))?;

    println!("📋 Registered agents in orchestrator");

    // Create a workflow for content creation
    let workflow = Workflow::builder()
        .name("Content Creation Pipeline")
        .description("A multi-step workflow for creating content")
        .variable("topic", serde_json::json!("artificial intelligence"))
        .variable(
            "target_audience",
            serde_json::json!("technical professionals"),
        )
        // Step 1: Research
        .step(WorkflowStep {
            id: "research".to_string(),
            name: "Research Phase".to_string(),
            description: Some("Gather information about the topic".to_string()),
            agent_id: researcher.id(),
            dependencies: vec![],
            config: {
                let mut config = std::collections::HashMap::new();
                config.insert("depth".to_string(), serde_json::json!("comprehensive"));
                config
            },
            retryable: true,
            max_retries: Some(2),
        })
        // Step 2: Analysis (depends on research)
        .step(WorkflowStep {
            id: "analysis".to_string(),
            name: "Analysis Phase".to_string(),
            description: Some("Analyze the research findings".to_string()),
            agent_id: analyst.id(),
            dependencies: vec!["research".to_string()],
            config: {
                let mut config = std::collections::HashMap::new();
                config.insert(
                    "analysis_type".to_string(),
                    serde_json::json!("trend_analysis"),
                );
                config
            },
            retryable: true,
            max_retries: Some(2),
        })
        // Step 3: Writing (depends on both research and analysis)
        .step(WorkflowStep {
            id: "writing".to_string(),
            name: "Writing Phase".to_string(),
            description: Some("Create content based on research and analysis".to_string()),
            agent_id: writer.id(),
            dependencies: vec!["research".to_string(), "analysis".to_string()],
            config: {
                let mut config = std::collections::HashMap::new();
                config.insert("format".to_string(), serde_json::json!("article"));
                config.insert("length".to_string(), serde_json::json!("medium"));
                config
            },
            retryable: true,
            max_retries: Some(1),
        })
        .timeout_ms(120_000) // 2 minutes total timeout
        .build()?;

    println!("🔄 Created workflow: {}", workflow.name());
    println!("   Steps: {}", workflow.steps().len());

    // Show execution order
    match workflow.execution_order() {
        Ok(order) => {
            println!("   Execution order: {:?}", order);
        }
        Err(e) => {
            println!("   ❌ Invalid workflow: {}", e);
            return Ok(());
        }
    }

    // Execute the workflow
    println!("\n🚀 Executing workflow...");
    let start_time = std::time::Instant::now();

    match orchestrator.execute_workflow(workflow).await {
        Ok(result) => {
            let duration = start_time.elapsed();
            println!("✅ Workflow completed successfully!");
            println!("   Status: {:?}", result.status);
            println!("   Duration: {:?}", duration);
            println!("   Steps executed: {}", result.task_results.len());

            // Show results for each step
            for (step_id, task_result) in &result.task_results {
                println!("\n   📋 Step '{}': {:?}", step_id, task_result.status);
                if task_result.status == TaskStatus::Completed {
                    println!("      Result: {}", task_result.content);
                    if let Some(duration) = task_result.duration_ms {
                        println!("      Duration: {}ms", duration);
                    }
                } else if let Some(error) = &task_result.error {
                    println!("      Error: {}", error);
                }
            }

            // Show final context
            if !result.context.variables.is_empty() {
                println!("\n   📊 Final context variables:");
                for (key, value) in &result.context.variables {
                    println!("      {}: {}", key, value);
                }
            }
        }
        Err(e) => {
            println!("❌ Workflow execution failed: {}", e);
        }
    }

    // Show orchestrator statistics
    println!("\n📈 Orchestrator Statistics");
    let stats = orchestrator.stats();
    println!("   Total workflows: {}", stats.total_workflows);
    println!("   Successful workflows: {}", stats.successful_workflows);
    println!("   Failed workflows: {}", stats.failed_workflows);
    println!("   Total tasks: {}", stats.total_tasks);
    println!("   Successful tasks: {}", stats.successful_tasks);
    println!("   Failed tasks: {}", stats.failed_tasks);
    println!(
        "   Average workflow time: {:.2}ms",
        stats.avg_workflow_time_ms
    );
    println!("   Active workflows: {}", stats.active_workflows);

    println!("\n🎉 Workflow orchestration example completed!");
    Ok(())
}
