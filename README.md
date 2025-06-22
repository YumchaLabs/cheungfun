# Cheungfun - Rust版LlamaIndex

[![Crates.io](https://img.shields.io/crates/v/cheungfun.svg)](https://crates.io/crates/cheungfun)
[![Documentation](https://docs.rs/cheungfun/badge.svg)](https://docs.rs/cheungfun)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

Cheungfun是一个基于Rust的高性能RAG（检索增强生成）和AI应用开发框架，灵感来源于LlamaIndex，采用模块化设计和流式处理架构。

## 🎯 项目特色

- **🚀 高性能**: 基于Rust的零成本抽象和内存安全
- **🔧 模块化**: 清晰的模块划分，易于扩展和维护
- **🌊 流式处理**: 支持大规模数据的流式索引和查询
- **🛡️ 类型安全**: 充分利用Rust类型系统确保运行时安全
- **🔌 统一接口**: 通过siumai提供统一的LLM访问接口
- **⚡ 异步优先**: 基于tokio的高性能异步编程

## 📦 模块架构

```
cheungfun/
├── cheungfun-core/          # 核心trait和数据结构
├── cheungfun-indexing/      # 数据加载和索引构建
├── cheungfun-query/         # 查询处理和响应生成
├── cheungfun-agents/        # 智能代理和工具调用
├── cheungfun-integrations/  # 外部服务集成
├── cheungfun-evaluation/    # 性能评估和指标
└── examples/               # 使用示例
```

## 🚀 快速开始

### 安装

```toml
[dependencies]
cheungfun = "0.1.0"
siumai = "0.4.0"
tokio = { version = "1.0", features = ["full"] }
```

### 基本使用

```rust
use cheungfun::prelude::*;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 配置嵌入模型
    let embedder = SiumaiEmbedder::new("openai", "text-embedding-3-small", "your-api-key").await?;
    
    // 2. 配置向量存储
    let vector_store = InMemoryVectorStore::new();
    
    // 3. 构建索引管道
    let indexing_pipeline = IndexingPipeline::builder()
        .loader(FileLoader::new("./docs").recursive(true))
        .transformer(TextSplitter::new(1000))
        .transformer(EmbeddingTransformer::new(embedder.clone()))
        .storage(vector_store.clone())
        .build()?;
    
    // 4. 运行索引
    let stats = indexing_pipeline.run().await?;
    println!("索引完成: {} 文档, {} 节点", stats.documents_processed, stats.nodes_created);
    
    // 5. 配置LLM客户端
    let llm_client = Siumai::builder()
        .openai()
        .api_key("your-api-key")
        .model("gpt-4")
        .build()
        .await?;
    
    // 6. 构建查询引擎
    let query_engine = QueryEngine::builder()
        .retriever(VectorRetriever::new(vector_store, embedder))
        .synthesizer(SimpleResponseSynthesizer::new(llm_client))
        .build()?;
    
    // 7. 执行查询
    let response = query_engine.query("文档的主要内容是什么？").await?;
    println!("回答: {}", response.content);
    
    Ok(())
}
```

## 📚 文档

- [架构设计](docs/architecture.md) - 整体架构和开发指南
- [Siumai文档](docs/siumai.md) - LLM统一接口说明

## 🏗️ 开发路线图

### 🎯 第一阶段 - 核心功能
- [x] 项目架构设计
- [ ] 基础数据结构和trait
- [ ] Candle嵌入生成器
- [ ] 文件加载和文本处理
- [ ] 基础查询引擎

### 🚀 第二阶段 - 扩展功能
- [ ] MCP代理框架
- [ ] Qdrant向量数据库集成
- [ ] 更多数据加载器
- [ ] 高级查询功能

### ⭐ 第三阶段 - 高级功能
- [ ] 多模态支持
- [ ] 工作流引擎
- [ ] 模型训练
- [ ] 性能优化

## 🤝 贡献指南

我们欢迎所有形式的贡献！请查看[贡献指南](CONTRIBUTING.md)了解详情。

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/YumchaLabs/cheungfun.git
cd cheungfun

# 安装依赖
cargo build

# 运行测试
cargo test

# 运行示例
cargo run --example basic_usage
```

## 📄 许可证

本项目采用双许可证：

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

## 🙏 致谢

- [LlamaIndex](https://github.com/run-llama/llama_index) - 提供了优秀的设计理念
- [Swiftide](https://github.com/bosun-ai/swiftide) - Rust生态的RAG框架参考
- [Siumai](https://crates.io/crates/siumai) - 统一的LLM接口库

## 📞 联系我们

- GitHub Issues: [问题反馈](https://github.com/YumchaLabs/cheungfun/issues)
- 讨论区: [GitHub Discussions](https://github.com/YumchaLabs/cheungfun/discussions)

---

Made with ❤️ by the YumchaLabs team
