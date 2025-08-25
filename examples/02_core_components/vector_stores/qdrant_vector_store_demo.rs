//! QdrantVectorStore demonstration.
//!
//! This example shows how to use QdrantVectorStore for production-grade
//! vector storage and retrieval with Qdrant as the backend.
//!
//! # Prerequisites
//!
//! 1. Start Qdrant server:
//!    ```bash
//!    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
//!    ```
//!
//! 2. Run this example:
//!    ```bash
//!    cargo run --example qdrant_vector_store_demo
//!    ```

use anyhow::Result;
use cheungfun_core::{
    traits::{DistanceMetric, VectorStore},
    types::{ChunkInfo, Node, Query, SearchMode},
};
use cheungfun_integrations::vector_stores::qdrant::{QdrantConfig, QdrantVectorStore};
use serde_json::Value;
use tracing::{info, Level};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🚀 Starting QdrantVectorStore demonstration");

    // Step 1: Configure QdrantVectorStore
    info!("\n📋 Step 1: Configuring QdrantVectorStore");

    let config = QdrantConfig::new("http://localhost:6334", "cheungfun_demo", 384)
        .with_distance_metric(DistanceMetric::Cosine)
        .with_create_collection_if_missing(true);

    info!("✅ Configuration created:");
    info!("   URL: {}", config.url);
    info!("   Collection: {}", config.collection_name);
    info!("   Dimension: {}", config.dimension);
    info!("   Distance Metric: {:?}", config.distance_metric);

    // Step 2: Create QdrantVectorStore
    info!("\n🔗 Step 2: Connecting to Qdrant and creating store");

    let store = QdrantVectorStore::new(config).await?;
    info!("✅ QdrantVectorStore created successfully");

    // Step 3: Health check
    info!("\n🏥 Step 3: Performing health check");
    store.health_check().await?;
    info!("✅ Qdrant is healthy and accessible");

    // Step 4: Create sample nodes with embeddings
    info!("\n📝 Step 4: Creating sample nodes with embeddings");

    let nodes = create_sample_nodes();
    info!("✅ Created {} sample nodes", nodes.len());

    // Step 5: Add nodes to the store
    info!("\n💾 Step 5: Adding nodes to QdrantVectorStore");

    let start_time = std::time::Instant::now();
    let added_ids = store.add(nodes.clone()).await?;
    let add_duration = start_time.elapsed();

    info!(
        "✅ Successfully added {} nodes in {:?}",
        added_ids.len(),
        add_duration
    );

    // Step 6: Get collection statistics
    info!("\n📊 Step 6: Getting collection statistics");

    let count = store.count().await?;
    let metadata = store.metadata().await?;
    let stats = store.stats().await?;

    info!("✅ Collection statistics:");
    info!("   Total nodes: {}", count);
    info!("   Insert operations: {}", stats.insert_operations);
    info!("   Collection type: {:?}", metadata.get("type"));

    // Step 7: Perform vector search
    info!("\n🔍 Step 7: Performing vector similarity search");

    let query_embedding = vec![0.1; 384]; // Simple query embedding
    let query = Query::builder()
        .text("What is machine learning?")
        .embedding(query_embedding)
        .top_k(3)
        .search_mode(SearchMode::Vector)
        .build();

    let search_start = std::time::Instant::now();
    let search_results = store.search(&query).await?;
    let search_duration = search_start.elapsed();

    info!("✅ Search completed in {:?}", search_duration);
    info!("   Found {} results:", search_results.len());

    for (i, result) in search_results.iter().enumerate() {
        info!(
            "   {}. Score: {:.4}, Content: '{}'",
            i + 1,
            result.score,
            result.node.content.chars().take(50).collect::<String>()
        );
    }

    // Step 8: Retrieve specific nodes
    info!("\n📖 Step 8: Retrieving specific nodes by ID");

    let first_two_ids = added_ids.iter().take(2).cloned().collect::<Vec<_>>();
    let retrieved_nodes = store.get(first_two_ids.clone()).await?;

    info!("✅ Retrieved {} nodes:", retrieved_nodes.len());
    for (i, node_opt) in retrieved_nodes.iter().enumerate() {
        if let Some(node) = node_opt {
            info!(
                "   {}. ID: {}, Content: '{}'",
                i + 1,
                node.id,
                node.content.chars().take(30).collect::<String>()
            );
        }
    }

    // Step 9: Update a node
    info!("\n✏️ Step 9: Updating a node");

    if let Some(Some(mut node)) = retrieved_nodes.into_iter().next() {
        node.content = "Updated content: This node has been modified".to_string();
        node.metadata
            .insert("updated".to_string(), Value::Bool(true));

        store.update(vec![node.clone()]).await?;
        info!("✅ Updated node: {}", node.id);
    }

    // Step 10: Advanced search with Qdrant-specific features
    info!("\n🔬 Step 10: Advanced search with filtering");

    let advanced = store.advanced();
    let advanced_results = advanced
        .search_with_filter(
            vec![0.2; 384],
            5,
            None,      // No filter for this demo
            Some(0.1), // Score threshold
        )
        .await?;

    info!(
        "✅ Advanced search found {} results",
        advanced_results.len()
    );

    // Step 11: Collection information
    info!("\n📋 Step 11: Getting detailed collection information");

    let _collection_info = store.client().collection_info().await?;
    let collection_stats = advanced.collection_stats().await?;

    info!("✅ Collection details:");
    info!(
        "   Points count: {:?}",
        collection_stats.get("points_count")
    );
    info!(
        "   Segments count: {:?}",
        collection_stats.get("segments_count")
    );
    info!("   Status: {:?}", collection_stats.get("status"));

    // Step 12: Batch operations
    info!("\n📦 Step 12: Demonstrating batch operations");

    let batch_nodes = create_batch_nodes(10);
    let batch_ids = advanced.batch_upsert(batch_nodes, 5).await?;

    info!("✅ Batch upserted {} nodes", batch_ids.len());

    // Step 13: Performance summary
    info!("\n📈 Step 13: Performance Summary");

    let final_count = store.count().await?;
    let final_stats = store.stats().await?;

    info!("✅ Final statistics:");
    info!("   Total nodes in collection: {}", final_count);
    info!(
        "   Total insert operations: {}",
        final_stats.insert_operations
    );
    info!(
        "   Total search operations: {}",
        final_stats.search_operations
    );
    info!(
        "   Total update operations: {}",
        final_stats.update_operations
    );

    // Step 14: Cleanup (optional)
    info!("\n🧹 Step 14: Cleanup (optional)");
    info!("💡 To clean up the collection, uncomment the following line:");
    info!("   // store.clear().await?;");

    // Uncomment to clean up:
    // store.clear().await?;
    // info!("✅ Collection cleared");

    info!("\n🎉 QdrantVectorStore demonstration completed successfully!");
    info!(
        "💡 The collection '{}' remains in Qdrant for further exploration",
        store.config().collection_name
    );

    Ok(())
}

/// Create sample nodes for demonstration.
fn create_sample_nodes() -> Vec<Node> {
    let documents = vec![
        (
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "technology",
        ),
        (
            "Rust is a systems programming language focused on safety and performance.",
            "programming",
        ),
        (
            "Vector databases are specialized for storing and querying high-dimensional vectors.",
            "database",
        ),
        (
            "Natural language processing enables computers to understand human language.",
            "ai",
        ),
        (
            "Qdrant is a vector similarity search engine written in Rust.",
            "database",
        ),
    ];

    documents
        .into_iter()
        .enumerate()
        .map(|(i, (content, category))| {
            let source_doc_id = Uuid::new_v4();
            let chunk_info = ChunkInfo::new(0, content.len(), i);

            // Create simple embeddings (in real use, these would come from an embedding model)
            let embedding = create_simple_embedding(content, 384);

            let mut node = Node::new(content, source_doc_id, chunk_info);
            node.embedding = Some(embedding);
            node.metadata
                .insert("category".to_string(), Value::String(category.to_string()));
            node.metadata
                .insert("index".to_string(), Value::Number(i.into()));

            node
        })
        .collect()
}

/// Create batch nodes for demonstration.
fn create_batch_nodes(count: usize) -> Vec<Node> {
    (0..count)
        .map(|i| {
            let content = format!("Batch node {} with some sample content for testing", i);
            let source_doc_id = Uuid::new_v4();
            let chunk_info = ChunkInfo::new(0, content.len(), i);

            let embedding = create_simple_embedding(&content, 384);

            let mut node = Node::new(content, source_doc_id, chunk_info);
            node.embedding = Some(embedding);
            node.metadata.insert("batch".to_string(), Value::Bool(true));
            node.metadata
                .insert("batch_index".to_string(), Value::Number(i.into()));

            node
        })
        .collect()
}

/// Create a simple embedding based on content (for demonstration only).
/// In production, use a proper embedding model.
fn create_simple_embedding(content: &str, dimension: usize) -> Vec<f32> {
    let mut embedding = vec![0.0; dimension];

    // Simple hash-based embedding (not suitable for production)
    let hash = content.chars().map(|c| c as u32).sum::<u32>();
    let base_value = (hash % 1000) as f32 / 1000.0;

    for (i, value) in embedding.iter_mut().enumerate() {
        *value = base_value + (i as f32 * 0.001) % 1.0;
    }

    // Normalize the vector
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }

    embedding
}
