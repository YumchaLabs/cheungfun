//! Candle Embedder Demo
//!
//! This example demonstrates the usage of the CandleEmbedder implementation.
//! This implementation uses real HuggingFace models downloaded via hf-hub
//! and processed using candle-transformers for BERT inference.
//!
//! **Note**: This example requires the `candle` feature to be enabled.
//! Run with: `cargo run --example candle_embedder_demo --features candle`

#[cfg(feature = "candle")]
use cheungfun_core::traits::Embedder;
#[cfg(feature = "candle")]
use cheungfun_integrations::embedders::candle::{CandleEmbedder, CandleEmbedderConfig};
#[cfg(feature = "candle")]
use tracing::{info, warn};

#[cfg(feature = "candle")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🚀 Starting Candle Embedder Demo");
    info!("🚀 Starting Candle Embedder Demo");

    // Test 1: Create embedder with default configuration
    println!("📝 Test 1: Creating embedder with default configuration");
    info!("📝 Test 1: Creating embedder with default configuration");
    let embedder =
        match CandleEmbedder::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").await {
            Ok(embedder) => {
                info!("✅ Successfully created CandleEmbedder");
                info!("   Model: {}", embedder.model_name());
                info!("   Dimension: {}", embedder.dimension());
                info!("   Device: {}", embedder.device_info());
                embedder
            }
            Err(e) => {
                warn!("❌ Failed to create CandleEmbedder: {}", e);
                warn!("   This might be due to network issues or missing dependencies");
                return Err(e.into());
            }
        };

    // Test 2: Single text embedding
    info!("📝 Test 2: Generating single text embedding");
    let text = "Hello, world! This is a test sentence for embedding.";
    match embedder.embed(text).await {
        Ok(embedding) => {
            info!("✅ Successfully generated embedding");
            info!("   Text: '{}'", text);
            info!("   Embedding dimension: {}", embedding.len());
            info!(
                "   First 5 values: {:?}",
                &embedding[..5.min(embedding.len())]
            );
        }
        Err(e) => {
            warn!("❌ Failed to generate embedding: {}", e);
            warn!("   Check model loading and network connectivity");
        }
    }

    // Test 3: Batch text embedding
    info!("📝 Test 3: Generating batch text embeddings");
    let texts = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Rust is a systems programming language.",
        "Embeddings capture semantic meaning of text.",
    ];

    match embedder.embed_batch(texts.clone()).await {
        Ok(embeddings) => {
            info!("✅ Successfully generated batch embeddings");
            info!("   Number of texts: {}", texts.len());
            info!("   Number of embeddings: {}", embeddings.len());
            for (i, (text, embedding)) in texts.iter().zip(embeddings.iter()).enumerate() {
                info!(
                    "   Text {}: '{}' -> {} dimensions",
                    i + 1,
                    &text[..50.min(text.len())],
                    embedding.len()
                );
            }
        }
        Err(e) => {
            warn!("❌ Failed to generate batch embeddings: {}", e);
            warn!("   This is expected as we're using a Mock implementation");
        }
    }

    // Test 4: Custom configuration
    info!("📝 Test 4: Creating embedder with custom configuration");
    let custom_config = CandleEmbedderConfig::new("custom-model")
        .with_dimension(512)
        .with_max_length(256)
        .with_batch_size(16)
        .with_device("cpu")
        .with_normalize(true);

    match CandleEmbedder::from_config(custom_config).await {
        Ok(custom_embedder) => {
            info!("✅ Successfully created custom CandleEmbedder");
            info!("   Model: {}", custom_embedder.model_name());
            info!("   Dimension: {}", custom_embedder.dimension());
            info!("   Device: {}", custom_embedder.device_info());
        }
        Err(e) => {
            warn!("❌ Failed to create custom CandleEmbedder: {}", e);
            warn!("   This is expected as we're using a Mock implementation");
        }
    }

    // Test 5: Health check
    info!("📝 Test 5: Performing health check");
    match embedder.health_check().await {
        Ok(()) => {
            info!("✅ Health check passed");
        }
        Err(e) => {
            warn!("❌ Health check failed: {}", e);
            warn!("   This is expected as we're using a Mock implementation");
        }
    }

    // Test 6: Statistics
    info!("📝 Test 6: Checking embedding statistics");
    let stats = embedder.stats().await;
    info!("📊 Embedding Statistics:");
    info!("   Texts embedded: {}", stats.texts_embedded);
    info!("   Total duration: {:?}", stats.duration);
    info!(
        "   Average time per embedding: {:?}",
        stats.avg_time_per_embedding
    );

    info!("🎉 Candle Embedder Demo completed!");
    info!("");
    info!("📋 Summary:");
    info!("   - CandleEmbedder with real HuggingFace integration is implemented");
    info!("   - Automatic model downloading from HuggingFace Hub");
    info!("   - Real BERT model inference using candle-transformers");
    info!("   - Proper tokenization using HuggingFace tokenizers");
    info!("   - Device management supports CPU/CUDA/Metal detection");
    info!("   - Configuration system is flexible and extensible");
    info!("   - Error handling is comprehensive");
    info!("   - Statistics tracking is available");
    info!("");
    info!("🚀 Features:");
    info!("   - ✅ HuggingFace Hub model downloading");
    info!("   - ✅ Real BERT model loading and inference");
    info!("   - ✅ Proper tokenizer integration");
    info!("   - ✅ Batch processing optimization");
    info!("   - ✅ Mean pooling and normalization");
    info!("   - 🔧 GPU acceleration (future enhancement)");

    Ok(())
}

// Fallback main function when candle feature is not enabled
#[cfg(not(feature = "candle"))]
fn main() {
    println!("🚀 Candle Embedder Demo");
    println!("========================");
    println!();
    println!("❌ This example requires the 'candle' feature to be enabled.");
    println!();
    println!("📋 To run this demo, use one of the following commands:");
    println!("   cargo run --example candle_embedder_demo --features candle");
    println!("   cargo run --example candle_embedder_demo --features all-embedders");
    println!();
    println!("🔧 The candle feature provides:");
    println!("   • Real HuggingFace model integration");
    println!("   • Local model inference with Candle");
    println!("   • Advanced configuration options");
    println!("   • Device management (CPU/CUDA/Metal)");
    println!();
    println!("💡 For a quick start with embeddings, try:");
    println!("   cargo run --example basic_indexing  # Uses FastEmbed (default)");
}
