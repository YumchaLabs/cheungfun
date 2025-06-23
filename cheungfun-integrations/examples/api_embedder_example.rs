//! Example usage of ApiEmbedder with different providers and configurations.
//!
//! This example demonstrates how to use the ApiEmbedder with various cloud
//! embedding services through the siumai library.
//!
//! To run this example:
//! ```bash
//! # Set your API key
//! export OPENAI_API_KEY="your-openai-api-key-here"
//!
//! # Run the example
//! cargo run --example api_embedder_example --features api
//! ```

use cheungfun_core::traits::Embedder;
use cheungfun_integrations::embedders::api::{ApiEmbedder, ApiEmbedderConfig};
use std::env;
use std::time::{Duration, Instant};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging (optional)
    tracing_subscriber::fmt::init();

    println!("🚀 ApiEmbedder Example");
    println!("======================");

    // Check for API key
    let api_key = match env::var("OPENAI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            println!("❌ Please set OPENAI_API_KEY environment variable");
            println!("   export OPENAI_API_KEY=\"your-api-key-here\"");
            return Ok(());
        }
    };

    // Example 1: Simple usage with builder pattern
    println!("\n📝 Example 1: Simple OpenAI Embedding");
    println!("-------------------------------------");

    let embedder = ApiEmbedder::builder()
        .openai(&api_key)
        .model("text-embedding-3-small")
        .build()
        .await?;

    let text = "Hello, world! This is a test embedding.";
    let start = Instant::now();
    let embedding = embedder.embed(text).await?;
    let duration = start.elapsed();

    println!("✅ Text: \"{}\"", text);
    println!("✅ Embedding dimension: {}", embedding.len());
    println!(
        "✅ First 5 values: {:?}",
        &embedding[..5.min(embedding.len())]
    );
    println!("✅ Time taken: {:?}", duration);

    // Example 2: Batch processing
    println!("\n📦 Example 2: Batch Processing");
    println!("------------------------------");

    let texts = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Rust is a systems programming language.",
        "Machine learning is transforming technology.",
        "Embeddings capture semantic meaning in vectors.",
        "Natural language processing enables AI understanding.",
    ];

    let start = Instant::now();
    let embeddings = embedder
        .embed_batch(texts.iter().map(|s| *s).collect())
        .await?;
    let duration = start.elapsed();

    println!("✅ Processed {} texts", texts.len());
    println!("✅ Generated {} embeddings", embeddings.len());
    println!("✅ Each embedding has {} dimensions", embeddings[0].len());
    println!("✅ Batch processing time: {:?}", duration);
    println!(
        "✅ Average time per text: {:?}",
        duration / texts.len() as u32
    );

    // Example 3: Advanced configuration
    println!("\n⚙️  Example 3: Advanced Configuration");
    println!("------------------------------------");

    let advanced_embedder = ApiEmbedder::builder()
        .openai(&api_key)
        .model("text-embedding-3-large") // Larger model for better quality
        .batch_size(50) // Smaller batches
        .max_retries(5) // More retries
        .timeout(Duration::from_secs(60)) // Longer timeout
        .enable_cache(true) // Enable caching
        .build()
        .await?;

    println!("✅ Model: {}", advanced_embedder.model_name());
    println!("✅ Dimension: {}", advanced_embedder.dimension());

    // Test the same text twice to demonstrate caching
    let test_text = "This text will be cached after the first embedding.";

    let start = Instant::now();
    let _embedding1 = advanced_embedder.embed(test_text).await?;
    let first_duration = start.elapsed();

    let start = Instant::now();
    let _embedding2 = advanced_embedder.embed(test_text).await?;
    let second_duration = start.elapsed();

    println!("✅ First embedding time: {:?}", first_duration);
    println!("✅ Second embedding time: {:?} (cached)", second_duration);
    println!(
        "✅ Speed improvement: {:.2}x",
        first_duration.as_nanos() as f64 / second_duration.as_nanos() as f64
    );

    // Example 4: Configuration from struct
    println!("\n🔧 Example 4: Configuration from Struct");
    println!("---------------------------------------");

    let config = ApiEmbedderConfig::openai(api_key.clone(), "text-embedding-3-small".to_string())
        .with_batch_size(25)
        .with_cache(true)
        .with_cache_ttl(Duration::from_secs(1800)) // 30 minutes
        .with_timeout(Duration::from_secs(45));

    let config_embedder = ApiEmbedder::from_config(config).await?;

    let embedding = config_embedder
        .embed("Configuration-based embedder test")
        .await?;
    println!(
        "✅ Config-based embedder works! Dimension: {}",
        embedding.len()
    );

    // Example 5: Error handling and health check
    println!("\n🏥 Example 5: Health Check and Error Handling");
    println!("---------------------------------------------");

    match embedder.health_check().await {
        Ok(()) => println!("✅ Health check passed - API is accessible"),
        Err(e) => println!("❌ Health check failed: {}", e),
    }

    // Example 6: Statistics and monitoring
    println!("\n📊 Example 6: Statistics and Monitoring");
    println!("--------------------------------------");

    let stats = embedder.stats().await;
    println!("✅ Embedding Statistics:");
    println!("   - Texts embedded: {}", stats.texts_embedded);
    println!("   - Embeddings failed: {}", stats.embeddings_failed);
    println!("   - Total duration: {:?}", stats.duration);
    println!(
        "   - Average time per embedding: {:?}",
        stats.avg_time_per_embedding
    );

    if let Some(cache_stats) = embedder.cache_stats().await {
        println!("✅ Cache Statistics:");
        println!("   - Total entries: {}", cache_stats.total_entries);
        println!("   - Cache hits: {}", cache_stats.hits);
        println!("   - Cache misses: {}", cache_stats.misses);
        println!("   - Hit rate: {:.2}%", cache_stats.hit_rate());
        println!("   - Expired entries: {}", cache_stats.expired_entries);
    }

    // Example 7: Semantic similarity demonstration
    println!("\n🔍 Example 7: Semantic Similarity");
    println!("---------------------------------");

    let sentences = vec![
        "The cat sits on the mat.",
        "A feline rests on the rug.",
        "Dogs are loyal animals.",
        "Canines are faithful pets.",
    ];

    let sentence_embeddings = embedder
        .embed_batch(sentences.iter().map(|s| *s).collect())
        .await?;

    // Calculate cosine similarity between first two (similar) and first and third (different)
    let similarity_similar = cosine_similarity(&sentence_embeddings[0], &sentence_embeddings[1]);
    let similarity_different = cosine_similarity(&sentence_embeddings[0], &sentence_embeddings[2]);

    println!("✅ Sentences:");
    for (i, sentence) in sentences.iter().enumerate() {
        println!("   {}. \"{}\"", i + 1, sentence);
    }
    println!(
        "✅ Similarity between sentences 1 and 2 (similar): {:.4}",
        similarity_similar
    );
    println!(
        "✅ Similarity between sentences 1 and 3 (different): {:.4}",
        similarity_different
    );
    println!(
        "✅ Similar sentences have higher similarity: {}",
        similarity_similar > similarity_different
    );

    println!("\n🎉 All examples completed successfully!");
    Ok(())
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
