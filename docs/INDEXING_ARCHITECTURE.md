# Cheungfun Indexing 架构文档

## 📋 概述

Cheungfun-indexing 是 Cheungfun RAG 框架的核心索引模块，负责文档加载、处理和转换。

> **⚠️ 学习项目声明**: 这是一个个人学习项目，用于探索和实践 Rust 中的 RAG 架构设计。虽然功能相对完整，但仍在开发中，不建议用于生产环境。

## 🏗️ 统一架构设计

经过重构，Cheungfun-indexing 现在采用**统一的 Transform 接口**，遵循 LlamaIndex 的 TransformComponent 设计模式：

- **统一接口**：所有处理组件都实现同一个 `Transform` trait
- **类型安全**：使用 `TransformInput` 枚举提供类型安全的输入处理
- **灵活组合**：支持文档分割器和节点处理器的无缝组合

## 🏗️ 模块结构

```text
cheungfun-indexing/src/
├── error.rs                    # 错误类型定义
├── lib.rs                      # 模块入口和预导入
├── loaders/                    # 文档加载器
│   ├── mod.rs                  # 加载器接口和配置
│   ├── file.rs                 # 单文件加载器
│   ├── directory.rs            # 目录加载器
│   ├── code.rs                 # 代码文件加载器
│   ├── web.rs                  # Web 内容加载器
│   └── filter/                 # 文件过滤器
├── parsers/                    # 内容解析器
│   ├── mod.rs                  # 解析器接口
│   └── ast_parser.rs           # AST 代码解析器
├── transformers/               # 节点处理器
│   ├── mod.rs                  # 处理器接口
│   └── metadata_extractor.rs   # 元数据提取器
├── node_parser/                # 文档分割器
│   ├── mod.rs                  # 核心接口定义
│   ├── config.rs               # 统一配置系统
│   ├── utils.rs                # 工具函数
│   ├── callbacks.rs            # 回调和事件系统
│   └── text/                   # 文本分割器实现
│       ├── mod.rs              # 文本分割器接口
│       ├── sentence.rs         # 句子分割器
│       ├── token.rs            # Token 分割器
│       ├── code.rs             # AST 增强代码分割器
│       └── utils.rs            # 文本处理工具
└── pipeline.rs                 # 统一索引管道
```

## 🔄 统一 Transform 架构

### 核心设计理念

**遵循 LlamaIndex 模式**：采用统一的 `Transform` 接口，所有处理组件都实现相同的接口，消除了原来 `Transformer` 和 `NodeTransformer` 的区别。

### 核心接口

```rust
#[async_trait]
pub trait Transform: Send + Sync + std::fmt::Debug {
    /// 统一的转换方法
    async fn transform(&self, input: TransformInput) -> Result<Vec<Node>>;

    /// 批处理方法
    async fn transform_batch(&self, inputs: Vec<TransformInput>) -> Result<Vec<Node>>;

    /// 组件名称
    fn name(&self) -> &'static str;
}
```

### 类型安全输入

```rust
#[derive(Debug, Clone)]
pub enum TransformInput {
    Document(Document),      // 单个文档
    Node(Node),             // 单个节点
    Documents(Vec<Document>), // 文档批次
    Nodes(Vec<Node>),       // 节点批次
}
```

### 组件分类

#### 文档分割器 (Document → Nodes)

- **SentenceSplitter**: 句子优先分割器（对标 LlamaIndex）
- **TokenTextSplitter**: 基于 token 的分割器
- **CodeSplitter**: AST 增强的代码分割器

#### 节点处理器 (Nodes → Nodes)

- **MetadataExtractor**: 元数据提取器

### 配置系统

**统一配置基类**：

- `NodeParserConfig`: 统一配置基类
- `TextSplitterConfig`: 文本分割配置基类
- `SentenceSplitterConfig`: 句子分割器专用配置
- `TokenTextSplitterConfig`: Token 分割器专用配置
- `CodeSplitterConfig`: 代码分割器专用配置

## 📊 架构优势

### 统一接口的优势

| 特性 | 描述 | 优势 |
|------|------|------|
| **单一接口** | 所有组件实现同一个 Transform trait | 简化 API，减少学习成本 |
| **类型安全** | TransformInput 枚举提供编译时检查 | 避免运行时错误 |
| **多态处理** | 支持运行时组件替换和组合 | 提高灵活性 |
| **管道集成** | 统一的管道处理逻辑 | 简化管道实现 |
| **批处理** | 内置批处理支持 | 提升性能 |
| **LlamaIndex 兼容** | 遵循 LlamaIndex 设计模式 | 与主流框架一致 |

## 🔧 使用示例

### 基本使用

```rust
use cheungfun_core::traits::{Transform, TransformInput};
use cheungfun_indexing::node_parser::text::SentenceSplitter;

// 创建分割器
let splitter = SentenceSplitter::from_defaults(300, 75)?;

// 使用统一接口
let input = TransformInput::Documents(documents);
let nodes = splitter.transform(input).await?;
```

### 管道使用

```rust
use cheungfun_indexing::pipeline::DefaultIndexingPipeline;

let pipeline = DefaultIndexingPipeline::builder()
    .with_loader(Arc::new(DirectoryLoader::new(path)?))
    .with_transformer(Arc::new(SentenceSplitter::from_defaults(300, 75)?))
    .with_transformer(Arc::new(MetadataExtractor::new()))
    .build()?;

let stats = pipeline.run().await?;
```

### 多态处理

```rust
// 统一接口的多态优势
let transforms: Vec<Box<dyn Transform>> = vec![
    Box::new(SentenceSplitter::from_defaults(200, 40)?),
    Box::new(TokenTextSplitter::from_defaults(180, 35)?),
];

for transform in transforms {
    let nodes = transform.transform(input.clone()).await?;
    println!("Transform {}: {} nodes", transform.name(), nodes.len());
}
```

## 🚀 重构成果

### 已完成

- ✅ **统一接口**: 完全移除了 `Transformer` 和 `NodeTransformer` 的区别
- ✅ **类型安全**: `TransformInput` 枚举提供编译时类型检查
- ✅ **管道重构**: `DefaultIndexingPipeline` 使用统一的处理流程
- ✅ **组件更新**: 所有核心组件都实现了新的 `Transform` 接口
- ✅ **示例更新**: 提供了完整的使用示例和文档

### 架构优势

1. **简化性**: 单一接口，减少学习成本
2. **一致性**: 所有组件遵循相同的模式
3. **灵活性**: 支持多态处理和动态组合
4. **可扩展性**: 新组件更容易添加和集成
5. **性能**: 零成本抽象，无额外运行时开销

## 📚 学习项目声明

> **⚠️ 重要提醒**: Cheungfun 是一个**个人学习项目**，用于探索和实践 Rust 中的 RAG 架构设计。虽然功能相对完整，但仍在开发中，**不建议用于生产环境**。
>
> 这个项目的主要目的是：
> - 学习 Rust 语言的高级特性
> - 探索 RAG 系统的架构设计
> - 实践 LlamaIndex 的设计模式
> - 提供学习和参考的代码示例

---

**总结**: Cheungfun-indexing 现在采用了与 LlamaIndex 一致的统一 Transform 接口架构，提供了更简洁、更灵活、更易扩展的文档处理能力。这个重构为整个 Cheungfun 框架的现代化奠定了坚实的基础。
