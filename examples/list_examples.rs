//! List Available Examples
//!
//! This utility helps users discover available examples and their required features.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin list_examples
//! ```

use std::collections::HashMap;

fn main() {
    println!("🚀 Cheungfun Examples");
    println!("====================");
    println!();

    let examples = get_examples();

    // Group examples by category
    let mut categories: HashMap<&str, Vec<&ExampleInfo>> = HashMap::new();
    for example in &examples {
        categories
            .entry(example.category)
            .or_default()
            .push(example);
    }

    // Display examples by category
    for (category, examples) in categories {
        println!("📁 {}", category);
        println!("{}", "=".repeat(category.len() + 3));

        for example in examples {
            println!("  📄 {}", example.name);
            println!("     Description: {}", example.description);
            println!(
                "     Command: cargo run --bin {} --features \"{}\"",
                example.binary_name,
                example.features.join(",")
            );
            println!();
        }
    }

    println!("💡 Feature Bundles");
    println!("==================");
    println!("  🎯 basic-examples  - Basic functionality, no special requirements");
    println!("  ⚡ performance      - CPU performance optimizations (SIMD, memory, HNSW)");
    println!("  🚀 candle-cuda     - NVIDIA GPU acceleration");
    println!("  🍎 candle-metal    - Apple Metal GPU acceleration");
    println!("  🏭 production      - Production-ready features");
    println!("  🌟 full            - All features enabled");
    println!();

    println!("🔧 Quick Commands");
    println!("=================");
    println!("  # Start with basics");
    println!("  cargo run --bin hello_world");
    println!();
    println!("  # Try performance features");
    println!("  cargo run --bin feature_comparison --features \"benchmarks,candle\"");
    println!();
    println!("  # Production example");
    println!("  cargo run --bin complete_rag_system --features production-examples");
    println!();
    println!("  # GPU acceleration (if available)");
    println!("  cargo run --bin cuda_embedder_demo --features candle-cuda");
    println!("  cargo run --bin metal_embedder_demo --features candle-metal");
}

struct ExampleInfo {
    name: &'static str,
    binary_name: &'static str,
    description: &'static str,
    category: &'static str,
    features: Vec<&'static str>,
}

fn get_examples() -> Vec<ExampleInfo> {
    vec![
        // Getting Started
        ExampleInfo {
            name: "Hello World",
            binary_name: "hello_world",
            description: "Basic introduction to Cheungfun",
            category: "Getting Started",
            features: vec!["basic-examples"],
        },
        ExampleInfo {
            name: "Basic Indexing",
            binary_name: "basic_indexing",
            description: "Document loading and indexing",
            category: "Getting Started",
            features: vec!["basic-examples"],
        },
        ExampleInfo {
            name: "Basic Querying",
            binary_name: "basic_querying",
            description: "Simple query processing",
            category: "Getting Started",
            features: vec!["basic-examples"],
        },
        // Core Components
        ExampleInfo {
            name: "Candle Embedder Demo",
            binary_name: "candle_embedder_demo",
            description: "Candle ML framework for embeddings",
            category: "Core Components",
            features: vec!["candle"],
        },
        ExampleInfo {
            name: "Candle Performance Test",
            binary_name: "candle_embedder_performance",
            description: "Performance testing with Candle",
            category: "Core Components",
            features: vec!["candle", "benchmarks"],
        },
        ExampleInfo {
            name: "Qdrant Vector Store",
            binary_name: "qdrant_store_demo",
            description: "Qdrant vector database integration",
            category: "Core Components",
            features: vec!["qdrant"],
        },
        // Performance
        ExampleInfo {
            name: "Feature Comparison",
            binary_name: "feature_comparison",
            description: "Compare performance of different features",
            category: "Performance",
            features: vec!["benchmarks", "candle"],
        },
        ExampleInfo {
            name: "Embedder Benchmark",
            binary_name: "embedder_benchmark",
            description: "Benchmark different embedding providers",
            category: "Performance",
            features: vec!["benchmarks", "all-embedders"],
        },
        ExampleInfo {
            name: "End-to-End Benchmark",
            binary_name: "end_to_end_benchmark",
            description: "Complete system performance test",
            category: "Performance",
            features: vec!["benchmarks", "production"],
        },
        // GPU Acceleration
        ExampleInfo {
            name: "CUDA GPU Demo",
            binary_name: "cuda_embedder_demo",
            description: "NVIDIA CUDA GPU acceleration",
            category: "GPU Acceleration",
            features: vec!["candle-cuda"],
        },
        ExampleInfo {
            name: "Metal GPU Demo",
            binary_name: "metal_embedder_demo",
            description: "Apple Metal GPU acceleration",
            category: "GPU Acceleration",
            features: vec!["candle-metal"],
        },
        // Production
        ExampleInfo {
            name: "Complete RAG System",
            binary_name: "complete_rag_system",
            description: "Full production-ready RAG demo",
            category: "Production",
            features: vec!["production-examples"],
        },
        ExampleInfo {
            name: "End-to-End Indexing",
            binary_name: "end_to_end_indexing",
            description: "Production indexing pipeline",
            category: "Production",
            features: vec!["production-examples"],
        },
        ExampleInfo {
            name: "End-to-End Query",
            binary_name: "end_to_end_query",
            description: "Production query pipeline",
            category: "Production",
            features: vec!["production-examples"],
        },
        // Integrations
        ExampleInfo {
            name: "MCP Integration",
            binary_name: "mcp_integration_example",
            description: "Model Context Protocol integration",
            category: "Integrations",
            features: vec!["mcp"],
        },
    ]
}
