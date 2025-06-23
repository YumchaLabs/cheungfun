//! Feature configuration demonstration.
//!
//! This example shows how to use different embedding backends through Cargo features.

use cheungfun_core::{
    traits::{
        embedder::Embedder,
        storage::{DistanceMetric, VectorStore},
    },
    types::{ChunkInfo, Node, Query},
};
use cheungfun_integrations::InMemoryVectorStore;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Cheungfun Feature Configuration Demo");
    println!("=====================================");

    // Show which features are enabled
    #[cfg(feature = "fastembed")]
    println!("✅ FastEmbed feature enabled");

    #[cfg(feature = "candle")]
    println!("✅ Candle feature enabled");

    #[cfg(not(any(feature = "fastembed", feature = "candle")))]
    println!("⚠️  No embedding features enabled");

    // Test vector store (always available)
    test_vector_store().await?;

    // Test FastEmbed (if enabled)
    #[cfg(feature = "fastembed")]
    test_fastembed().await?;

    // Test Candle (if enabled)
    #[cfg(feature = "candle")]
    test_candle().await?;

    println!("\n✅ All enabled features working correctly!");
    Ok(())
}

async fn test_vector_store() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📦 Testing InMemoryVectorStore...");

    let store = InMemoryVectorStore::new(3, DistanceMetric::Cosine);

    // Create test nodes with embeddings
    let source_doc_id = Uuid::new_v4();
    let test_nodes = vec![
        Node::new(
            "First test document",
            source_doc_id,
            ChunkInfo::new(0, 20, 0),
        )
        .with_embedding(vec![0.1, 0.2, 0.3]),
        Node::new(
            "Second test document",
            source_doc_id,
            ChunkInfo::new(21, 42, 1),
        )
        .with_embedding(vec![0.4, 0.5, 0.6]),
        Node::new(
            "Third test document",
            source_doc_id,
            ChunkInfo::new(43, 63, 2),
        )
        .with_embedding(vec![0.7, 0.8, 0.9]),
    ];

    // Add nodes to store
    store.add(test_nodes).await?;

    // Test search
    let query = Query::new("test query")
        .with_embedding(vec![0.2, 0.3, 0.4])
        .with_top_k(2);
    let results = store.search(&query).await?;

    println!(
        "   ✅ Added 3 nodes, found {} similar results",
        results.len()
    );
    Ok(())
}

#[cfg(feature = "fastembed")]
async fn test_fastembed() -> Result<(), Box<dyn std::error::Error>> {
    use cheungfun_integrations::FastEmbedder;

    println!("\n🚀 Testing FastEmbedder (default feature)...");

    // Test with default configuration
    match FastEmbedder::new().await {
        Ok(embedder) => {
            let text = "Hello, FastEmbed world!";
            match embedder.embed(text).await {
                Ok(embedding) => {
                    println!(
                        "   ✅ FastEmbed embedding generated: {} dimensions",
                        embedding.len()
                    );
                }
                Err(e) => {
                    println!("   ⚠️  FastEmbed embedding failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("   ⚠️  FastEmbed initialization failed: {}", e);
            println!("      This is normal if models aren't downloaded yet.");
        }
    }

    Ok(())
}

#[cfg(feature = "candle")]
async fn test_candle() -> Result<(), Box<dyn std::error::Error>> {
    use cheungfun_integrations::{CandleEmbedder, CandleEmbedderConfig};

    println!("\n🔧 Testing CandleEmbedder (advanced feature)...");

    // Test with mock configuration (since real models aren't integrated yet)
    let config = CandleEmbedderConfig::new("sentence-transformers/all-MiniLM-L6-v2");

    match CandleEmbedder::from_config(config).await {
        Ok(embedder) => {
            let text = "Hello, Candle world!";
            match embedder.embed(text).await {
                Ok(embedding) => {
                    println!(
                        "   ✅ Candle embedding generated: {} dimensions",
                        embedding.len()
                    );
                }
                Err(e) => {
                    println!("   ⚠️  Candle embedding failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("   ⚠️  Candle initialization failed: {}", e);
            println!("      This is expected since real models aren't integrated yet.");
            println!("      TODO: Complete real model integration (see embedder.rs)");
        }
    }

    Ok(())
}

// This function is only compiled when no embedding features are enabled
#[cfg(not(any(feature = "fastembed", feature = "candle")))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Cheungfun Feature Configuration Demo");
    println!("=====================================");
    println!("⚠️  No embedding features enabled!");
    println!("   Enable features with:");
    println!("   cargo run --example feature_demo --features fastembed");
    println!("   cargo run --example feature_demo --features candle");
    println!("   cargo run --example feature_demo --features all-embedders");
    Ok(())
}
