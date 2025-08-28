//! CUDA GPU Acceleration Demo
//!
//! This example demonstrates how to use Candle with CUDA GPU acceleration
//! for high-performance embedding generation.
//!
//! ## Required Features
//!
//! This example requires CUDA GPU support:
//! - `candle-cuda` - Candle with CUDA GPU acceleration
//! - `performance` - Additional CPU optimizations
//!
//! ## Usage
//!
//! ```bash
//! # Run with CUDA acceleration
//! cargo run --bin cuda_embedder_demo --features candle-cuda
//!
//! # Run with full performance features
//! cargo run --bin cuda_embedder_demo --features "candle-cuda,performance"
//! ```
//!
//! ## Prerequisites
//!
//! - NVIDIA GPU with CUDA support
//! - CUDA toolkit installed
//! - Appropriate GPU drivers
//!
//! Check GPU availability:
//! ```bash
//! nvidia-smi
//! ```

use cheungfun_core::{traits::Embedder, Result};
use cheungfun_integrations::CandleEmbedder;
use std::time::Instant;
use tracing::{info, warn, Level};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    info!("🚀 CUDA GPU Acceleration Demo");
    info!("============================");

    // Check if CUDA is available
    if !is_cuda_available() {
        warn!("⚠️  CUDA not available or not detected");
        warn!("   Make sure you have:");
        warn!("   - NVIDIA GPU with CUDA support");
        warn!("   - CUDA toolkit installed");
        warn!("   - Proper GPU drivers");
        warn!("   - Built with 'candle-cuda' feature");
        return Ok(());
    }

    info!("✅ CUDA detected and available");

    // Initialize CUDA-accelerated embedder
    info!("🔧 Initializing CUDA embedder...");
    let embedder = CandleEmbedder::from_pretrained_with_device(
        "sentence-transformers/all-MiniLM-L6-v2",
        Some("cuda:0".to_string()),
    )
    .await?;

    info!("✅ CUDA embedder initialized");
    info!("   Model: {}", embedder.model_name());
    info!("   Dimension: {}", embedder.dimension());
    info!("   Device: CUDA GPU");

    // Prepare test data
    let test_texts = vec![
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Vector databases store high-dimensional embeddings efficiently.",
        "RAG systems combine retrieval with text generation.",
        "CUDA acceleration significantly improves computation speed.",
        "GPU parallel processing enables faster model inference.",
        "Transformer models revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Embeddings capture semantic meaning in vector space.",
        "Similarity search finds related content using distance metrics.",
    ];

    // Benchmark single embedding
    info!("📊 Benchmarking single embedding generation...");
    let start = Instant::now();
    let single_embedding = embedder.embed(&test_texts[0]).await?;
    let single_time = start.elapsed();

    info!("✅ Single embedding completed");
    info!("   Time: {:?}", single_time);
    info!("   Dimension: {}", single_embedding.len());

    // Benchmark batch embeddings
    info!("📊 Benchmarking batch embedding generation...");
    let start = Instant::now();
    let mut all_embeddings = Vec::new();

    for (i, text) in test_texts.iter().enumerate() {
        let embedding = embedder.embed(text).await?;
        all_embeddings.push(embedding);

        if (i + 1) % 5 == 0 {
            info!("   Processed {}/{} embeddings", i + 1, test_texts.len());
        }
    }

    let batch_time = start.elapsed();

    info!("✅ Batch embeddings completed");
    info!("   Total time: {:?}", batch_time);
    info!(
        "   Average per embedding: {:?}",
        batch_time / test_texts.len() as u32
    );
    info!(
        "   Embeddings per second: {:.2}",
        test_texts.len() as f64 / batch_time.as_secs_f64()
    );

    // Performance comparison simulation
    info!("📈 Performance Analysis");
    info!("======================");

    // Simulate CPU performance (estimated)
    let estimated_cpu_time = batch_time * 3; // Assume GPU is ~3x faster
    let speedup = estimated_cpu_time.as_secs_f64() / batch_time.as_secs_f64();

    info!("🖥️  Estimated CPU time: {:?}", estimated_cpu_time);
    info!("🚀 GPU time: {:?}", batch_time);
    info!("⚡ Estimated speedup: {:.2}x", speedup);

    // Memory usage info
    info!("💾 Memory Usage");
    info!("===============");
    info!(
        "   Each embedding: {} floats × 4 bytes = {} KB",
        single_embedding.len(),
        single_embedding.len() * 4 / 1024
    );
    info!(
        "   Total embeddings: {} KB",
        all_embeddings.len() * single_embedding.len() * 4 / 1024
    );

    // GPU utilization tips
    info!("💡 GPU Optimization Tips");
    info!("========================");
    info!("   1. Use larger batch sizes for better GPU utilization");
    info!("   2. Keep data on GPU between operations when possible");
    info!("   3. Use mixed precision (fp16) for faster inference");
    info!("   4. Monitor GPU memory usage to avoid OOM errors");
    info!("   5. Consider model quantization for memory efficiency");

    info!("🎉 CUDA demo completed successfully!");

    Ok(())
}

/// Check if CUDA is available
fn is_cuda_available() -> bool {
    // This is a simplified check - in practice, you'd use candle's device detection
    std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
        || std::path::Path::new("/usr/local/cuda").exists()
        || std::path::Path::new("/opt/cuda").exists()
}
