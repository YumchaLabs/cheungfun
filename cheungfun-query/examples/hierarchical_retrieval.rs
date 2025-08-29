//! Hierarchical Retrieval Example
//!
//! This example demonstrates how to use the hierarchical retrieval system
//! with auto-merging and intelligent query routing.
//!
//! **Reference**: LlamaIndex AutoMergingRetriever examples
//! - File: `llama-index-core/docs/docs/examples/retrievers/auto_merging_retriever.ipynb`

use std::sync::Arc;

use cheungfun_core::{Document, Result};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{generator::SiumaiGenerator, hierarchical::HierarchicalSystemBuilder};
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Hierarchical Retrieval Example");
    println!("==================================");

    // 1. Create sample documents
    let documents = create_sample_documents();
    println!("📚 Created {} sample documents", documents.len());

    // 2. Set up embedder
    let embedder = Arc::new(
        FastEmbedder::with_model("BAAI/bge-small-en-v1.5")
            .await
            .map_err(|e| cheungfun_core::CheungfunError::Embedding {
                message: format!("Failed to create embedder: {}", e),
            })?,
    );
    println!("🔤 Embedder initialized");

    // 3. Set up LLM generator
    let siumai_client = Siumai::builder()
        .openai()
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string()))
        .model("gpt-3.5-turbo")
        .build()
        .await
        .map_err(|e| cheungfun_core::CheungfunError::Llm {
            message: format!("Failed to create Siumai client: {}", e),
        })?;

    let generator = Arc::new(SiumaiGenerator::new(siumai_client));
    println!("🤖 LLM generator initialized");

    // 4. Build hierarchical retrieval system
    println!("🏗️  Building hierarchical retrieval system...");

    let hierarchical_system = HierarchicalSystemBuilder::new()
        .documents(documents)
        .embedder(embedder)
        .vector_store_factory(|| {
            Arc::new(InMemoryVectorStore::new(
                384,
                cheungfun_core::DistanceMetric::Cosine,
            ))
        })
        .generator(generator)
        .chunk_sizes(vec![1024, 256, 64]) // Three levels: summary, medium, detailed
        .merge_threshold(0.6) // Merge when 60% of children are retrieved
        .verbose(true)
        .build()
        .await?;

    println!("✅ Hierarchical retrieval system built successfully!");

    // 5. Test different types of queries
    println!("\n🔍 Testing Query Routing and Hierarchical Retrieval");
    println!("====================================================");

    let test_queries = vec![
        ("What is this document about?", "Summary query"),
        (
            "How does the authentication system work in detail?",
            "Detailed query",
        ),
        (
            "Explain the database connection process",
            "Implementation query",
        ),
        (
            "Give me an overview of the security features",
            "Overview query",
        ),
    ];

    for (query, query_type) in test_queries {
        println!("\n📝 Query Type: {}", query_type);
        println!("❓ Query: {}", query);
        println!("⏳ Processing...");

        match hierarchical_system.query(query).await {
            Ok(response) => {
                println!("✅ Response:");
                println!("   Content: {}", response.response.content);

                if let Some(selected_engine) = response.query_metadata.get("selected_engine") {
                    println!("   🎯 Selected Engine: {}", selected_engine);
                }
                if let Some(reason) = response.query_metadata.get("selection_reason") {
                    println!("   💭 Selection Reason: {}", reason);
                }

                println!("   📊 Sources: {} nodes", response.retrieved_nodes.len());
            }
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }
    }

    println!("\n🎉 Hierarchical retrieval example completed!");
    Ok(())
}

/// Create sample documents for testing.
///
/// **Reference**: LlamaIndex example document creation patterns
fn create_sample_documents() -> Vec<Document> {
    vec![
        Document::new(
            "Authentication System Overview\n\
             Our application uses a multi-layered authentication system with JWT tokens, \
             OAuth2 integration, and role-based access control. The system supports \
             multiple authentication providers including Google, GitHub, and local accounts.",
        )
        .with_metadata("title", "Authentication Overview")
        .with_metadata("category", "security")
        .with_metadata("level", "overview"),
        Document::new(
            "JWT Token Implementation Details\n\
             The JWT token implementation uses RS256 algorithm for signing. Tokens are \
             generated with a 24-hour expiration time and include user ID, roles, and \
             permissions in the payload. The private key is stored securely in environment \
             variables and rotated monthly. Token validation includes signature verification, \
             expiration check, and issuer validation.",
        )
        .with_metadata("title", "JWT Implementation")
        .with_metadata("category", "security")
        .with_metadata("level", "detailed")
        .with_metadata("parent_id", "auth_overview"),
        Document::new(
            "Database Connection Management\n\
             The application uses a connection pool with a maximum of 20 connections. \
             Connection timeout is set to 30 seconds, and idle connections are closed \
             after 10 minutes. The system supports both read and write replicas with \
             automatic failover capabilities.",
        )
        .with_metadata("title", "Database Connections")
        .with_metadata("category", "infrastructure")
        .with_metadata("level", "overview"),
        Document::new(
            "Connection Pool Configuration\n\
             Connection pool is configured using HikariCP with the following settings: \
             maximum-pool-size=20, minimum-idle=5, connection-timeout=30000ms, \
             idle-timeout=600000ms, max-lifetime=1800000ms. Health checks are performed \
             every 30 seconds using SELECT 1 query. Failed connections trigger automatic \
             retry with exponential backoff starting at 1 second.",
        )
        .with_metadata("title", "Connection Pool Details")
        .with_metadata("category", "infrastructure")
        .with_metadata("level", "detailed")
        .with_metadata("parent_id", "db_overview"),
        Document::new(
            "Security Features Summary\n\
             The application implements comprehensive security measures including: \
             - End-to-end encryption for data in transit\n\
             - AES-256 encryption for data at rest\n\
             - Rate limiting and DDoS protection\n\
             - Input validation and SQL injection prevention\n\
             - Regular security audits and penetration testing\n\
             - GDPR compliance for data protection",
        )
        .with_metadata("title", "Security Features")
        .with_metadata("category", "security")
        .with_metadata("level", "overview"),
        Document::new(
            "Rate Limiting Implementation\n\
             Rate limiting is implemented using Redis with sliding window algorithm. \
             Default limits are 100 requests per minute for authenticated users and \
             20 requests per minute for anonymous users. The system tracks requests \
             by IP address and user ID. When limits are exceeded, HTTP 429 status \
             is returned with Retry-After header indicating when to retry.",
        )
        .with_metadata("title", "Rate Limiting Details")
        .with_metadata("category", "security")
        .with_metadata("level", "detailed")
        .with_metadata("parent_id", "security_overview"),
    ]
}
