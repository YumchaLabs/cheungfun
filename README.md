# Cheungfun

[![Crates.io](https://img.shields.io/crates/v/cheungfun.svg)](https://crates.io/crates/cheungfun)
[![Documentation](https://docs.rs/cheungfun/badge.svg)](https://docs.rs/cheungfun)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Build Status](https://github.com/YumchaLabs/cheungfun/workflows/CI/badge.svg)](https://github.com/YumchaLabs/cheungfun/actions)

**Fast, streaming indexing, query, and agentic LLM applications in Rust**

Cheungfun is a high-performance RAG (Retrieval-Augmented Generation) and AI application development framework built in Rust, inspired by LlamaIndex. It features modular design, streaming processing architecture, and blazing-fast performance optimizations.

## ✨ Key Features

- **🚀 High Performance**: Built with Rust's zero-cost abstractions and memory safety
  - SIMD-accelerated vector operations (54x speedup)
  - HNSW approximate nearest neighbor search
  - Optimized memory management
- **🔧 Modular Architecture**: Clean separation of concerns with extensible design
- **🌊 Streaming Processing**: Handle large-scale data with streaming indexing and querying
- **💻 Advanced Code Indexing**: Tree-sitter AST parsing for 9+ programming languages
  - Extract functions, classes, imports, comments, and complexity metrics
  - Code-aware splitting that preserves syntactic boundaries
  - Support for Rust, Python, JavaScript, TypeScript, Java, C#, C/C++, Go
- **🛡️ Type Safety**: Leverage Rust's type system for runtime safety guarantees
- **🔌 Unified LLM Interface**: Seamless integration with multiple LLM providers via [siumai](https://crates.io/crates/siumai)
- **⚡ Async-First**: Built on tokio for high-performance async operations
- **🎯 Production Ready**: Comprehensive testing, benchmarking, and optimization

## 📊 Performance Benchmarks

Cheungfun delivers exceptional performance across all components:

| Feature | Performance | Comparison |
|---------|-------------|------------|
| **SIMD Vector Operations** | 54.5x speedup | Industry-leading |
| **Vector Search (HNSW)** | 110+ queries/sec | Competitive with Qdrant |
| **Memory Optimization** | 4.9x improvement | Above average |
| **Indexing Throughput** | Streaming capable | Production-ready |

## 📦 Architecture

```text
cheungfun/
├── cheungfun-core/          # Core traits and data structures
├── cheungfun-indexing/      # Data loading and index building
├── cheungfun-query/         # Query processing and response generation
├── cheungfun-agents/        # Intelligent agents and tool calling
├── cheungfun-integrations/  # External service integrations
└── examples/               # Usage examples
```

## 🚀 Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
cheungfun = "0.1.0"
siumai = "0.4.0"
tokio = { version = "1.0", features = ["full"] }
```

### Feature Flags

Choose the right features for your use case:

```toml
# Default: stable and safe
cheungfun = "0.1.0"

# Performance optimized (recommended for production)
cheungfun = { version = "0.1.0", features = ["performance"] }

# Full feature set
cheungfun = { version = "0.1.0", features = ["full"] }
```

### Basic Usage

```rust
use cheungfun::prelude::*;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure embedding model
    let embedder = SiumaiEmbedder::new("openai", "text-embedding-3-small", "your-api-key").await?;

    // 2. Set up vector store
    let vector_store = InMemoryVectorStore::new(384, DistanceMetric::Cosine);

    // 3. Build indexing pipeline
    let indexing_pipeline = DefaultIndexingPipeline::builder()
        .with_loader(Arc::new(DirectoryLoader::new("./docs")?))
        .with_transformer(Arc::new(TextSplitter::new(1000, 200)))
        .with_node_transformer(Arc::new(MetadataExtractor::new()))
        .build()?;

    // 4. Run indexing
    let documents = indexing_pipeline.load().await?;
    let nodes = indexing_pipeline.transform_documents(documents).await?;
    vector_store.add(nodes).await?;

    // 5. Configure LLM client
    let llm_client = Siumai::builder()
        .openai()
        .api_key("your-api-key")
        .model("gpt-4")
        .build()
        .await?;

    // 6. Build query engine
    let query_engine = DefaultQueryPipeline::builder()
        .with_retriever(Arc::new(VectorRetriever::new(vector_store, embedder)))
        .with_synthesizer(Arc::new(SimpleResponseSynthesizer::new(llm_client)))
        .build()?;

    // 7. Execute query
    let response = query_engine.query("What is the main content of the documents?").await?;
    println!("Answer: {}", response.content);

    Ok(())
}
```

### Advanced Usage with Performance Features

```rust
use cheungfun::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use HNSW for high-performance vector search
    let vector_store = HnswVectorStore::new(384, DistanceMetric::Cosine);
    vector_store.initialize_index(10000)?; // Pre-allocate for 10k vectors

    // Use optimized memory store for better performance
    let optimized_store = OptimizedInMemoryVectorStore::new(384, DistanceMetric::Cosine);

    // SIMD-accelerated vector operations
    #[cfg(feature = "simd")]
    {
        let simd_ops = SimdVectorOps::new();
        if simd_ops.is_simd_available() {
            println!("SIMD acceleration enabled: {}", simd_ops.get_capabilities());
        }
    }

    Ok(())
}
```

## 🎯 Feature Flags

Cheungfun provides granular control over features and dependencies:

| Feature | Description | Use Case |
|---------|-------------|----------|
| `default` | Stable, optimized memory operations | Development, testing |
| `simd` | SIMD-accelerated vector operations | High-performance computing |
| `hnsw` | HNSW approximate nearest neighbor | Large-scale vector search |
| `performance` | All performance optimizations | Production deployments |
| `candle` | Candle ML framework integration | Local embeddings |
| `qdrant` | Qdrant vector database | Distributed vector storage |
| `fastembed` | FastEmbed integration | Quick embedding setup |
| `full` | All features enabled | Maximum functionality |

## 📚 Documentation

- [Architecture Guide](docs/architecture.md) - System design and development guide
- [Performance Report](PERFORMANCE_REPORT.md) - Detailed benchmarks and optimizations
- [API Documentation](https://docs.rs/cheungfun) - Complete API reference
- [Examples](examples/) - Practical usage examples

## 🏗️ Roadmap

### ✅ Phase 1 - Core Foundation

- [x] Project architecture and module design
- [x] Core traits and data structures
- [x] SIMD-accelerated vector operations
- [x] HNSW approximate nearest neighbor search
- [x] Memory-optimized vector stores
- [x] Comprehensive performance benchmarks

### 🚧 Phase 2 - Advanced Features

- [ ] MCP (Model Context Protocol) agent framework
- [ ] Qdrant vector database integration
- [ ] Advanced query processing pipeline
- [ ] Multi-modal document processing
- [ ] Distributed indexing capabilities

### 🔮 Phase 3 - Enterprise Features

- [ ] Workflow orchestration engine
- [ ] Model fine-tuning and training
- [ ] Advanced evaluation metrics
- [ ] Cloud-native deployment
- [ ] Enterprise security features

## 🤝 Contributing

We welcome contributions of all kinds! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/YumchaLabs/cheungfun.git
cd cheungfun

# Build with default features
cargo build

# Build with performance features
cargo build --features performance

# Run tests
cargo test

# Run performance benchmarks
cargo test --features performance --test performance_integration_test

# Run examples
cargo run --example basic_usage
```

### Performance Testing

```bash
# Test SIMD acceleration
cargo test --features simd test_simd_performance -- --nocapture

# Test vector store performance
cargo test --features "hnsw,simd" test_vector_store_performance -- --nocapture

# Full performance suite
cargo test --features performance --test performance_integration_test -- --nocapture
```

## 📄 License

This project is dual-licensed under:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

## 🙏 Acknowledgments

- [LlamaIndex](https://github.com/run-llama/llama_index) - Inspiration for the design philosophy
- [Swiftide](https://github.com/bosun-ai/swiftide) - Reference implementation in Rust ecosystem
- [Siumai](https://crates.io/crates/siumai) - Unified LLM interface library
- [SimSIMD](https://github.com/ashvardanian/SimSIMD) - High-performance SIMD operations
- [HNSW-RS](https://github.com/jean-pierreBoth/hnswlib-rs) - Rust HNSW implementation

## 📞 Community

- **GitHub Issues**: [Bug reports and feature requests](https://github.com/YumchaLabs/cheungfun/issues)
- **Discussions**: [Community discussions](https://github.com/YumchaLabs/cheungfun/discussions)
- **Documentation**: [API docs and guides](https://docs.rs/cheungfun)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=YumchaLabs/cheungfun&type=Date)](https://star-history.com/#YumchaLabs/cheungfun&Date)

---

**Made with ❤️ by the YumchaLabs team**

*Cheungfun - Where performance meets elegance in Rust RAG applications*
