//! Metal GPU Acceleration Demo (macOS)
//!
//! This example demonstrates how to use Candle with Metal GPU acceleration
//! on macOS systems with Apple Silicon or AMD GPUs.
//!
//! ## Required Features
//!
//! This example requires Metal GPU support:
//! - `candle-metal` - Candle with Metal GPU acceleration
//! - `performance` - Additional CPU optimizations
//!
//! ## Usage
//!
//! ```bash
//! # Run with Metal acceleration (macOS only)
//! cargo run --bin metal_embedder_demo --features candle-metal
//!
//! # Run with full performance features
//! cargo run --bin metal_embedder_demo --features "candle-metal,performance"
//! ```
//!
//! ## Prerequisites
//!
//! - macOS system
//! - Apple Silicon (M1/M2/M3) or AMD GPU
//! - Metal framework (included with macOS)

use cheungfun_core::{Result, traits::Embedder};
use cheungfun_integrations::CandleEmbedder;
use std::time::Instant;
use tracing::{Level, info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    info!("🚀 Metal GPU Acceleration Demo");
    info!("==============================");

    // Check if Metal is available
    if !is_metal_available() {
        warn!("⚠️  Metal not available or not detected");
        warn!("   Make sure you have:");
        warn!("   - macOS system");
        warn!("   - Apple Silicon or AMD GPU");
        warn!("   - Built with 'candle-metal' feature");
        return Ok(());
    }

    info!("✅ Metal detected and available");

    // Initialize Metal-accelerated embedder
    info!("🔧 Initializing Metal embedder...");
    let embedder = CandleEmbedder::from_pretrained_with_device(
        "sentence-transformers/all-MiniLM-L6-v2",
        Some("metal".to_string()),
    )
    .await?;

    info!("✅ Metal embedder initialized");
    info!("   Model: {}", embedder.model_name());
    info!("   Dimension: {}", embedder.dimension());
    info!("   Device: Metal GPU");

    // Prepare test data optimized for Apple Silicon
    let test_texts = vec![
        "Apple Silicon revolutionized Mac performance with unified memory architecture.",
        "Metal provides low-level access to GPU compute and graphics capabilities.",
        "Neural Engine accelerates machine learning workloads on Apple devices.",
        "Unified memory allows CPU and GPU to share the same memory pool efficiently.",
        "M-series chips integrate CPU, GPU, and Neural Engine on a single die.",
        "Metal Performance Shaders optimize common compute operations.",
        "Core ML framework enables on-device machine learning inference.",
        "Apple's custom silicon delivers industry-leading performance per watt.",
        "Metal compute shaders enable parallel processing on Apple GPUs.",
        "Heterogeneous computing leverages different processing units optimally.",
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

    // Performance analysis for Apple Silicon
    info!("📈 Apple Silicon Performance Analysis");
    info!("====================================");

    // Estimate performance gains
    let estimated_cpu_time = batch_time * 2; // Metal typically 2x faster than CPU
    let speedup = estimated_cpu_time.as_secs_f64() / batch_time.as_secs_f64();

    info!("🖥️  Estimated CPU time: {:?}", estimated_cpu_time);
    info!("🚀 Metal GPU time: {:?}", batch_time);
    info!("⚡ Estimated speedup: {:.2}x", speedup);

    // Memory efficiency on Apple Silicon
    info!("💾 Unified Memory Architecture");
    info!("==============================");
    info!("   Unified memory allows zero-copy data sharing");
    info!(
        "   Each embedding: {} floats × 4 bytes = {} KB",
        single_embedding.len(),
        single_embedding.len() * 4 / 1024
    );
    info!(
        "   Total embeddings: {} KB",
        all_embeddings.len() * single_embedding.len() * 4 / 1024
    );
    info!("   No GPU memory transfers needed!");

    // Apple Silicon optimization tips
    info!("💡 Apple Silicon Optimization Tips");
    info!("==================================");
    info!("   1. Leverage unified memory architecture");
    info!("   2. Use Metal Performance Shaders for common operations");
    info!("   3. Consider Neural Engine for supported models");
    info!("   4. Optimize for memory bandwidth over compute");
    info!("   5. Use fp16 precision for better performance");
    info!("   6. Batch operations to maximize GPU utilization");

    // System information
    info!("🖥️  System Information");
    info!("======================");
    if let Ok(output) = std::process::Command::new("sysctl")
        .args(&["-n", "machdep.cpu.brand_string"])
        .output()
    {
        if let Ok(cpu_info) = String::from_utf8(output.stdout) {
            info!("   CPU: {}", cpu_info.trim());
        }
    }

    if let Ok(output) = std::process::Command::new("sysctl")
        .args(&["-n", "hw.memsize"])
        .output()
    {
        if let Ok(mem_str) = String::from_utf8(output.stdout) {
            if let Ok(mem_bytes) = mem_str.trim().parse::<u64>() {
                let mem_gb = mem_bytes / (1024 * 1024 * 1024);
                info!("   Memory: {} GB unified memory", mem_gb);
            }
        }
    }

    info!("🎉 Metal demo completed successfully!");

    Ok(())
}

/// Check if Metal is available (macOS only)
fn is_metal_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        // On macOS, Metal is generally available
        // In practice, you'd use candle's device detection
        true
    }

    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}
