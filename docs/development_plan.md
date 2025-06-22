# Cheungfun 开发计划

## 📋 项目概述

基于当前实现状态分析，Cheungfun 项目的核心架构已经完成约 70%，主要缺少具体的实现类。本文档制定了详细的开发计划，按优先级推进剩余功能的实现。

## 🎯 当前状态

### ✅ 已完成 (70%)
- **cheungfun-core**: 95% - 核心架构、trait 定义、配置系统完整
- **cheungfun-indexing**: 90% - 文档加载、文本处理完整  
- **cheungfun-query**: 80% - 查询框架、Siumai 集成完整
- **cheungfun-integrations**: 5% - 几乎空白，需要重点开发

### ❌ 缺少的关键实现
1. **具体的 Embedder 实现** (CandleEmbedder, ApiEmbedder)
2. **具体的 VectorStore 实现** (InMemoryVectorStore, QdrantVectorStore)
3. **完整的端到端工作流**
4. **cheungfun-integrations 模块的所有内容**

## 🚀 开发计划

### 第一阶段：基础功能实现 (2-3周)

#### 1.1 内存向量存储 (优先级: 🔴 最高)
**目标**: 实现基础的内存向量存储，支持基本的 CRUD 和相似度搜索

**任务清单**:
- [ ] 实现 `InMemoryVectorStore` 结构体
- [ ] 实现向量相似度计算 (cosine, euclidean, dot_product)
- [ ] 实现基础的 CRUD 操作 (add, get, delete, update)
- [ ] 实现向量搜索 (top-k 相似度搜索)
- [ ] 添加元数据过滤功能
- [ ] 编写单元测试和集成测试

**预期产出**:
```rust
// cheungfun-integrations/src/vector_stores/memory.rs
pub struct InMemoryVectorStore {
    vectors: HashMap<Uuid, Vec<f32>>,
    metadata: HashMap<Uuid, HashMap<String, Value>>,
    dimension: usize,
}
```

#### 1.2 Candle 嵌入生成器 (优先级: 🔴 最高)
**目标**: 实现基于 Candle 的本地嵌入生成器

**任务清单**:
- [ ] 设计 `CandleEmbedder` 结构体
- [ ] 实现模型加载 (从 HuggingFace Hub)
- [ ] 实现文本 tokenization
- [ ] 实现批量嵌入生成
- [ ] 添加设备选择 (CPU/CUDA)
- [ ] 实现嵌入归一化
- [ ] 编写测试用例

**预期产出**:
```rust
// cheungfun-integrations/src/embedders/candle.rs
pub struct CandleEmbedder {
    model: Box<dyn candle_nn::Module>,
    tokenizer: tokenizers::Tokenizer,
    device: candle_core::Device,
    config: CandleEmbedderConfig,
}
```

#### 1.3 基础 VectorRetriever 实现 (优先级: 🟡 高)
**目标**: 实现具体的向量检索器

**任务清单**:
- [ ] 完善 `VectorRetriever` 的具体实现
- [ ] 集成 InMemoryVectorStore
- [ ] 实现查询嵌入生成
- [ ] 实现相似度搜索和排序
- [ ] 添加结果过滤和重排序
- [ ] 编写测试用例

### 第二阶段：端到端工作流 (2-3周)

#### 2.1 完整索引管道 (优先级: 🟡 高)
**目标**: 实现从文档到向量存储的完整流程

**任务清单**:
- [ ] 创建端到端索引示例
- [ ] 集成 FileLoader + TextSplitter + CandleEmbedder + InMemoryVectorStore
- [ ] 实现批量处理和进度跟踪
- [ ] 添加错误处理和重试机制
- [ ] 性能优化和内存管理
- [ ] 编写集成测试

**预期产出**:
```rust
// examples/src/end_to_end_indexing.rs
async fn main() -> Result<()> {
    let embedder = CandleEmbedder::from_pretrained("all-MiniLM-L6-v2").await?;
    let vector_store = InMemoryVectorStore::new(384);
    
    let pipeline = IndexingPipeline::builder()
        .loader(FileLoader::new("./docs"))
        .transformer(TextSplitter::new(1000, 200))
        .embedder(embedder)
        .storage(vector_store)
        .build()?;
    
    pipeline.run().await?;
}
```

#### 2.2 完整查询管道 (优先级: 🟡 高)
**目标**: 实现从查询到响应的完整流程

**任务清单**:
- [ ] 创建端到端查询示例
- [ ] 集成 VectorRetriever + SiumaiGenerator
- [ ] 实现上下文组装和提示工程
- [ ] 添加响应后处理
- [ ] 实现流式响应支持
- [ ] 编写集成测试

### 第三阶段：生产级功能 (3-4周)

#### 3.1 Qdrant 集成 (优先级: 🟠 中)
**目标**: 实现生产级向量数据库集成

**任务清单**:
- [ ] 实现 `QdrantVectorStore`
- [ ] 支持集合创建和管理
- [ ] 实现批量向量操作
- [ ] 添加连接池和重试机制
- [ ] 支持混合搜索 (dense + sparse)
- [ ] 编写性能测试

#### 3.2 API 嵌入生成器 (优先级: 🟠 中)
**目标**: 支持外部 API 嵌入服务

**任务清单**:
- [ ] 实现 `ApiEmbedder` 基础结构
- [ ] 支持 OpenAI Embeddings API
- [ ] 支持 Cohere Embeddings API
- [ ] 实现请求限流和重试
- [ ] 添加缓存机制
- [ ] 编写集成测试

#### 3.3 高级检索功能 (优先级: 🟠 中)
**目标**: 实现高级检索策略

**任务清单**:
- [ ] 实现混合搜索 (向量 + 关键词)
- [ ] 添加重排序算法
- [ ] 实现查询扩展
- [ ] 支持多模态检索
- [ ] 添加检索评估指标

### 第四阶段：优化和扩展 (2-3周)

#### 4.1 性能优化 (优先级: 🔵 低)
- [ ] 向量计算优化 (SIMD, 并行化)
- [ ] 内存使用优化
- [ ] 批处理优化
- [ ] 缓存策略优化

#### 4.2 监控和可观测性 (优先级: 🔵 低)
- [ ] 添加详细的日志记录
- [ ] 实现性能指标收集
- [ ] 添加健康检查端点
- [ ] 实现分布式追踪

## 📅 时间线

| 阶段 | 时间 | 主要交付物 |
|------|------|-----------|
| 第一阶段 | 第1-3周 | InMemoryVectorStore, CandleEmbedder, VectorRetriever |
| 第二阶段 | 第4-6周 | 端到端索引和查询管道 |
| 第三阶段 | 第7-10周 | Qdrant集成, API嵌入器, 高级检索 |
| 第四阶段 | 第11-13周 | 性能优化, 监控系统 |

## 🎯 里程碑

### 里程碑 1: 基础可用 (第3周末)
- ✅ 可以本地运行完整的 RAG 流程
- ✅ 支持文档索引和基础查询
- ✅ 内存向量存储和 Candle 嵌入

### 里程碑 2: 端到端完整 (第6周末)  
- ✅ 完整的示例应用
- ✅ 稳定的 API 接口
- ✅ 完善的错误处理

### 里程碑 3: 生产就绪 (第10周末)
- ✅ Qdrant 生产级存储
- ✅ 多种嵌入选项
- ✅ 高级检索功能

### 里程碑 4: 企业级 (第13周末)
- ✅ 性能优化完成
- ✅ 监控和可观测性
- ✅ 完整的文档和示例

## 🔧 开发环境设置

### 必需依赖
```toml
# 添加到 cheungfun-integrations/Cargo.toml
[dependencies]
candle-core = "0.9"
candle-nn = "0.9"
candle-transformers = "0.9"
tokenizers = "0.20"
hf-hub = "0.3"
qdrant-client = "1.14"
```

### 开发工具
- Rust 1.75+
- Docker (用于 Qdrant 测试)
- Python 3.8+ (用于模型验证)

## 📊 成功指标

### 功能指标
- [ ] 支持至少 3 种文档格式 (txt, pdf, docx)
- [ ] 支持至少 2 种嵌入模型 (Candle 本地 + API)
- [ ] 支持至少 2 种向量存储 (内存 + Qdrant)
- [ ] 端到端延迟 < 2秒 (1000 token 查询)

### 质量指标  
- [ ] 单元测试覆盖率 > 80%
- [ ] 集成测试覆盖率 > 60%
- [ ] 文档完整性 > 90%
- [ ] 零 unsafe 代码

### 性能指标
- [ ] 索引速度 > 100 文档/秒
- [ ] 查询延迟 < 100ms (P95)
- [ ] 内存使用 < 1GB (10万文档)
- [ ] 并发支持 > 100 QPS

## 🤝 贡献指南

1. **选择任务**: 从上述任务清单中选择一个任务
2. **创建分支**: `git checkout -b feature/task-name`
3. **实现功能**: 遵循现有的代码风格和架构
4. **编写测试**: 确保新功能有充分的测试覆盖
5. **更新文档**: 更新相关的文档和示例
6. **提交 PR**: 详细描述变更内容和测试结果

## 📚 参考资料

- [Candle 文档](https://github.com/huggingface/candle)
- [Qdrant 文档](https://qdrant.tech/documentation/)
- [Siumai 文档](https://github.com/siumai/siumai)
- [RAG 最佳实践](https://docs.llamaindex.ai/en/stable/)
