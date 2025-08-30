# Cheungfun

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

**🎓 A Learning-Focused RAG Framework in Rust**

> **⚠️ Learning Project Disclaimer**: Cheungfun is a **personal learning project** designed to explore and practice RAG (Retrieval-Augmented Generation) architecture design in Rust. While functionally complete, it's still in development and **not recommended for production use**.
>
> **📚 Learning Goals**:
> - Deep dive into Rust's advanced features and best practices
> - Explore RAG system architecture design and implementation patterns
> - Practice LlamaIndex design philosophy and interface patterns
> - Provide learning and reference code examples

Cheungfun is a high-performance RAG (Retrieval-Augmented Generation) framework built in Rust, inspired by LlamaIndex. It features modular design, streaming processing architecture, and performance optimization capabilities, primarily designed for learning and exploring modern RAG system implementations.

## ✨ Learning Features

- **🚀 Performance Exploration**: Exploring Rust's zero-cost abstractions and memory safety features
  - SIMD-accelerated vector operations experiments
  - HNSW approximate nearest neighbor search implementation
  - Memory management optimization practices
- **🔧 Modular Design**: Learning separation of concerns and scalable architecture design
- **🌊 Streaming Processing**: Experimenting with large-scale streaming indexing and querying
- **💻 Advanced Code Indexing**: Tree-sitter AST-based code processing
  - Extract functions, classes, imports, comments, and complexity metrics
  - Code-aware splitting that maintains syntax boundaries
  - Support for Rust, Python, JavaScript, TypeScript, Java, C#, C/C++, Go
- **🛡️ Type Safety**: Leveraging Rust's type system for runtime safety guarantees
- **🔌 Unified Interface**: Adopting LlamaIndex's Transform interface design pattern
- **⚡ Async-First**: High-performance async operations built on tokio
- **🎓 Learning-Oriented**: Complete examples and documentation for learning reference

## 📊 Performance Experiment Results

Performance optimization achievements during the learning process:

| Feature | Performance | Learning Outcome |
|---------|-------------|------------------|
| **SIMD Vector Operations** | 30.17x speedup | Learned SIMD optimization techniques |
| **Vector Search (HNSW)** | 378+ QPS | Understanding of approximate nearest neighbor algorithms |
| **Memory Optimization** | Significant improvement | Mastered Rust memory management |
| **Indexing Throughput** | Streaming processing | Practiced async programming patterns |

> **Note**: These data come from learning experiments and do not represent production environment performance guarantees.

## 📦 Learning Architecture

```text
cheungfun/
├── cheungfun-core/          # Core traits and data structures
├── cheungfun-indexing/      # Data loading and indexing with unified Transform interface
├── cheungfun-query/         # Query processing and response generation
├── cheungfun-agents/        # Intelligent agents and tool calling (MCP integration)
├── cheungfun-integrations/  # External service integrations (FastEmbed, Qdrant, etc.)
├── cheungfun-multimodal/    # Multi-modal processing (text, images, audio, video)
└── examples/               # Learning examples and usage demonstrations
```

### 🔄 Unified Interface Refactoring

Recently completed a major architectural refactoring, adopting a unified Transform interface consistent with LlamaIndex:

- **Unified Interface**: All processing components implement the same `Transform` trait
- **Type Safety**: Using `TransformInput` enum for compile-time type checking
- **Pipeline Simplification**: Unified processing flow, easier to compose and extend

## 🚀 Getting Started

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
cheungfun = "0.1.0"
siumai = "0.9.1"  # LLM integration
tokio = { version = "1.0", features = ["full"] }
```

### Feature Flags

Choose features suitable for learning:

```toml
# Default: stable and secure
cheungfun = "0.1.0"

# Learning experiments (includes all features)
cheungfun = { version = "0.1.0", features = ["performance"] }

# Full feature set
cheungfun = { version = "0.1.0", features = ["full"] }
```

### Basic Usage (Unified Interface)

```rust
use cheungfun::prelude::*;
use cheungfun_core::traits::{Transform, TransformInput};
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure embedding model
    let embedder = SiumaiEmbedder::new("openai", "text-embedding-3-small", "your-api-key").await?;

    // 2. Set up vector storage
    let vector_store = InMemoryVectorStore::new(384, DistanceMetric::Cosine);

    // 3. Build indexing pipeline with unified interface
    let indexing_pipeline = DefaultIndexingPipeline::builder()
        .with_loader(Arc::new(DirectoryLoader::new("./docs")?))
        .with_transformer(Arc::new(SentenceSplitter::from_defaults(1000, 200)?))  // Unified interface
        .with_transformer(Arc::new(MetadataExtractor::new()))                     // Unified interface
        .build()?;

    // 4. Run indexing
    let stats = indexing_pipeline.run().await?;
    println!("Indexing complete: {} documents, {} nodes", stats.documents_processed, stats.nodes_created);

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

### Unified Transform Interface Example

```rust
use cheungfun_core::traits::{Transform, TransformInput};
use cheungfun_indexing::node_parser::text::SentenceSplitter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create document splitter
    let splitter = SentenceSplitter::from_defaults(300, 75)?;

    // Use unified interface to process documents
    let input = TransformInput::Documents(documents);
    let nodes = splitter.transform(input).await?;

    // Polymorphic processing example
    let transforms: Vec<Box<dyn Transform>> = vec![
        Box::new(SentenceSplitter::from_defaults(200, 40)?),
        Box::new(TokenTextSplitter::from_defaults(180, 35)?),
    ];

    for transform in transforms {
        let nodes = transform.transform(input.clone()).await?;
        println!("Transform {}: {} nodes", transform.name(), nodes.len());
    }

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

## ✅ Current Status

### 🎯 Completed Features

- ✅ **Core Foundation**: Complete modular architecture with 6 main crates
- ✅ **Performance Optimization**: SIMD acceleration, HNSW search, memory optimization
- ✅ **Advanced Indexing**: Tree-sitter AST parsing for 9+ programming languages
- ✅ **Query Processing**: Hybrid search, query transformations, reranking algorithms
- ✅ **Agent Framework**: Complete MCP integration with workflow orchestration
- ✅ **Multimodal Support**: Text, image, audio, video processing capabilities
- ✅ **External Integrations**: FastEmbed, Qdrant, API embedders with caching
- ✅ **Unified Interface**: LlamaIndex-style Transform interface for all components

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

## 📚 Learning Statement

**Cheungfun** is a personal learning project, primarily used for:

- 🦀 **Learning Rust**: Exploring Rust's advanced features and best practices
- 🏗️ **Architecture Design**: Practicing modern RAG system architectural patterns
- 📖 **Knowledge Sharing**: Providing learning and reference code examples
- 🔬 **Technical Experimentation**: Trying new algorithms and optimization techniques

While functionally complete, it's **not recommended for production use**. If you're interested in RAG systems and Rust development, welcome to learn and reference!

---

*Made with ❤️ for learning and exploration*
