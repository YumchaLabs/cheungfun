//! Test utilities and helper functions for examples.

use cheungfun_core::prelude::*;
use std::time::Instant;
use tokio::time::Duration;

/// Performance metrics for benchmarking
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub indexing_time: Duration,
    pub query_time: Duration,
    pub total_documents: usize,
    pub total_nodes: usize,
    pub queries_processed: usize,
    pub average_query_time: Duration,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            indexing_time: Duration::from_secs(0),
            query_time: Duration::from_secs(0),
            total_documents: 0,
            total_nodes: 0,
            queries_processed: 0,
            average_query_time: Duration::from_secs(0),
        }
    }

    pub fn record_indexing_time(&mut self, duration: Duration) {
        self.indexing_time = duration;
    }

    pub fn record_query(&mut self, duration: Duration) {
        self.query_time += duration;
        self.queries_processed += 1;
        self.average_query_time = self.query_time / self.queries_processed as u32;
    }

    pub fn print_summary(&self) {
        println!("\n📊 Performance Summary");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📚 Documents processed: {}", self.total_documents);
        println!("🔗 Nodes created: {}", self.total_nodes);
        println!("⏱️  Indexing time: {:.2?}", self.indexing_time);
        println!("🔍 Queries processed: {}", self.queries_processed);
        println!("⚡ Average query time: {:.2?}", self.average_query_time);
        println!("🎯 Total query time: {:.2?}", self.query_time);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer utility for measuring execution time
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        println!("⏳ Starting: {}", name);
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed().into()
    }

    pub fn finish(self) -> Duration {
        let duration = self.elapsed();
        println!("✅ Completed: {} in {:.2?}", self.name, duration);
        duration
    }
}

/// Pretty print query results
pub fn print_query_results(query: &str, response: &cheungfun_core::QueryResponse) {
    println!("\n🔍 Query: {}", query);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📝 Response: {}", response.response.content);

    if !response.retrieved_nodes.is_empty() {
        println!(
            "\n📚 Retrieved Context ({} nodes):",
            response.retrieved_nodes.len()
        );
        for (i, node) in response.retrieved_nodes.iter().enumerate() {
            println!(
                "  {}. Score: {:.4} | Content: {}...",
                i + 1,
                node.score,
                node.node.content.chars().take(100).collect::<String>()
            );
        }
    }

    if let Some(usage) = &response.response.usage {
        println!("\n💰 Token Usage:");
        println!(
            "  Input: {} | Output: {} | Total: {}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

/// Print retrieved context in a formatted way
pub fn print_context(nodes: &[cheungfun_core::ScoredNode]) {
    println!("\n📚 Retrieved Context ({} nodes):", nodes.len());
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    for (i, scored_node) in nodes.iter().enumerate() {
        println!("{}. 📄 Score: {:.4}", i + 1, scored_node.score);

        // Print metadata if available
        if !scored_node.node.metadata.is_empty() {
            println!("   📋 Metadata:");
            for (key, value) in &scored_node.node.metadata {
                println!("      {}: {}", key, value);
            }
        }

        // Print content preview
        let content_preview = if scored_node.node.content.len() > 200 {
            format!("{}...", &scored_node.node.content[..200])
        } else {
            scored_node.node.content.clone()
        };

        println!("   📝 Content: {}", content_preview);
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

/// Common test queries for climate change document
pub fn get_climate_test_queries() -> Vec<&'static str> {
    vec![
        "What is the main cause of climate change?",
        "How do greenhouse gases affect the atmosphere?",
        "What are the impacts of climate change on sea levels?",
        "What mitigation strategies can help address climate change?",
        "How do ice core samples help scientists understand climate?",
    ]
}

/// Common test queries for customer data
pub fn get_customer_test_queries() -> Vec<&'static str> {
    vec![
        "How many customers are from California?",
        "What is the average age of customers?",
        "Which customers have the highest purchase amounts?",
        "What are the most common job titles?",
        "How are customers distributed by state?",
        "Who are the customers with email addresses ending in .com?",
        "Which customers are from New York?",
        "Show me customers who work in technology",
    ]
}

/// Validate that required environment variables are set
pub fn check_env_vars(vars: &[&str]) -> crate::ExampleResult<()> {
    for var in vars {
        if std::env::var(var).is_err() {
            return Err(super::ExampleError::Config(format!(
                "Environment variable {} is not set",
                var
            )));
        }
    }
    Ok(())
}

/// Setup logging for examples
pub fn setup_logging() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
}
