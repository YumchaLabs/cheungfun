# Cheungfun

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

**🎓 A Learning-Focused RAG Framework in Rust**

> **⚠️ 学习项目声明**: Cheungfun 是一个**个人学习项目**，用于探索和实践 Rust 中的 RAG (Retrieval-Augmented Generation) 架构设计。虽然功能相对完整，但仍在开发中，**不建议用于生产环境**。
>
> **📚 学习目标**:
> - 深入学习 Rust 语言的高级特性和最佳实践
> - 探索 RAG 系统的架构设计和实现模式
> - 实践 LlamaIndex 的设计理念和接口模式
> - 提供学习和参考的代码示例

Cheungfun 是一个高性能的 RAG 框架，采用 Rust 构建，灵感来源于 LlamaIndex。它具有模块化设计、流式处理架构和性能优化特性，主要用于学习和探索现代 RAG 系统的实现。

## ✨ 学习特性

- **🚀 性能探索**: 探索 Rust 的零成本抽象和内存安全特性
  - SIMD 加速向量操作实验
  - HNSW 近似最近邻搜索实现
  - 内存管理优化实践
- **🔧 模块化设计**: 学习关注点分离和可扩展架构设计
- **🌊 流式处理**: 实验大规模数据的流式索引和查询
- **💻 高级代码索引**: 基于 Tree-sitter AST 解析的代码处理
  - 提取函数、类、导入、注释和复杂度指标
  - 保持语法边界的代码感知分割
  - 支持 Rust、Python、JavaScript、TypeScript、Java、C#、C/C++、Go
- **🛡️ 类型安全**: 利用 Rust 类型系统保证运行时安全
- **🔌 统一接口**: 采用 LlamaIndex 的 Transform 接口设计模式
- **⚡ 异步优先**: 基于 tokio 的高性能异步操作
- **🎓 学习导向**: 提供完整的示例和文档用于学习参考

## 📊 性能实验结果

在学习过程中实现的性能优化效果：

| 特性 | 性能表现 | 学习收获 |
|------|----------|----------|
| **SIMD 向量操作** | 30.17x 加速 | 学习了 SIMD 优化技术 |
| **向量搜索 (HNSW)** | 378+ QPS | 理解了近似最近邻算法 |
| **内存优化** | 显著改善 | 掌握了 Rust 内存管理 |
| **索引吞吐量** | 流式处理 | 实践了异步编程模式 |

> **注意**: 这些数据来自学习实验，不代表生产环境性能保证。

## 📦 学习架构

```text
cheungfun/
├── cheungfun-core/          # 核心 trait 和数据结构
├── cheungfun-indexing/      # 统一 Transform 接口的数据加载和索引构建
├── cheungfun-query/         # 查询处理和响应生成
├── cheungfun-agents/        # 智能代理和工具调用 (MCP 集成)
├── cheungfun-integrations/  # 外部服务集成 (FastEmbed, Qdrant 等)
├── cheungfun-multimodal/    # 多模态处理 (文本、图像、音频、视频)
└── examples/               # 学习示例和用法演示
```

### 🔄 统一接口重构

最近完成了重大架构重构，采用了与 LlamaIndex 一致的统一 Transform 接口：

- **统一接口**: 所有处理组件都实现同一个 `Transform` trait
- **类型安全**: 使用 `TransformInput` 枚举提供编译时类型检查
- **管道简化**: 统一的处理流程，更易于组合和扩展

## 🚀 学习开始

### 安装

添加到你的 `Cargo.toml`:

```toml
[dependencies]
cheungfun = "0.1.0"
siumai = "0.4.0"  # LLM 集成
tokio = { version = "1.0", features = ["full"] }
```

### 特性标志

选择适合学习的特性:

```toml
# 默认: 稳定和安全
cheungfun = "0.1.0"

# 学习实验 (包含所有特性)
cheungfun = { version = "0.1.0", features = ["performance"] }

# Full feature set
cheungfun = { version = "0.1.0", features = ["full"] }
```

### 基本使用 (统一接口)

```rust
use cheungfun::prelude::*;
use cheungfun_core::traits::{Transform, TransformInput};
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 配置嵌入模型
    let embedder = SiumaiEmbedder::new("openai", "text-embedding-3-small", "your-api-key").await?;

    // 2. 设置向量存储
    let vector_store = InMemoryVectorStore::new(384, DistanceMetric::Cosine);

    // 3. 构建统一接口的索引管道
    let indexing_pipeline = DefaultIndexingPipeline::builder()
        .with_loader(Arc::new(DirectoryLoader::new("./docs")?))
        .with_transformer(Arc::new(SentenceSplitter::from_defaults(1000, 200)?))  // 统一接口
        .with_transformer(Arc::new(MetadataExtractor::new()))                     // 统一接口
        .build()?;

    // 4. 运行索引
    let stats = indexing_pipeline.run().await?;
    println!("索引完成: {} 个文档, {} 个节点", stats.documents_processed, stats.nodes_created);

    // 5. 配置 LLM 客户端
    let llm_client = Siumai::builder()
        .openai()
        .api_key("your-api-key")
        .model("gpt-4")
        .build()
        .await?;

    // 6. 构建查询引擎
    let query_engine = DefaultQueryPipeline::builder()
        .with_retriever(Arc::new(VectorRetriever::new(vector_store, embedder)))
        .with_synthesizer(Arc::new(SimpleResponseSynthesizer::new(llm_client)))
        .build()?;

    // 7. 执行查询
    let response = query_engine.query("文档的主要内容是什么？").await?;
    println!("回答: {}", response.content);

    Ok(())
}
```

### 统一 Transform 接口示例

```rust
use cheungfun_core::traits::{Transform, TransformInput};
use cheungfun_indexing::node_parser::text::SentenceSplitter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建文档分割器
    let splitter = SentenceSplitter::from_defaults(300, 75)?;

    // 使用统一接口处理文档
    let input = TransformInput::Documents(documents);
    let nodes = splitter.transform(input).await?;

    // 多态处理示例
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

## 📚 学习声明

**Cheungfun** 是一个个人学习项目，主要用于：

- 🦀 **学习 Rust**: 探索 Rust 语言的高级特性和最佳实践
- 🏗️ **架构设计**: 实践现代 RAG 系统的架构模式
- 📖 **知识分享**: 提供学习和参考的代码示例
- 🔬 **技术实验**: 尝试新的算法和优化技术

虽然功能相对完整，但**不建议用于生产环境**。如果你对 RAG 系统和 Rust 开发感兴趣，欢迎学习和参考！

---

*Made with ❤️ for learning and exploration*
