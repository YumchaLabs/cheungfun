# Cheungfun Graph RAG 开发 TODO

## 📊 当前实现状态

### ✅ 已完成的核心组件

#### 1. 图数据结构 (cheungfun-core)
- ✅ `LabelledPropertyGraph` - 完整的属性图数据结构
- ✅ `EntityNode` 和 `ChunkNode` - 实体和文本块节点
- ✅ `Relation` 和 `Triplet` - 关系和三元组
- ✅ `PropertyGraphStore` trait - 图存储接口
- ✅ 测试覆盖：完整

#### 2. 图存储 (cheungfun-integrations)
- ✅ `SimplePropertyGraphStore` - 内存图存储实现
- ✅ 完整的 CRUD 操作支持
- ✅ 高级过滤和查询功能
- ✅ 线程安全的并发访问
- ✅ 测试覆盖：5/5 通过

#### 3. 图检索 (cheungfun-query)
- ✅ `GraphRetriever` - 图检索器
- ✅ 多种检索策略（实体、关系、混合、自定义）
- ✅ 智能评分和排序
- ✅ 完整的查询处理流程
- ✅ 测试覆盖：8/8 通过

#### 4. 图索引 (cheungfun-query)
- ✅ `PropertyGraphIndex` - 统一的图索引接口
- ✅ LlamaIndex 兼容的 API 设计
- ✅ `from_documents()`, `from_existing()`, `as_retriever()` 方法
- ✅ 双存储支持（PropertyGraphStore + VectorStore）
- ✅ 测试覆盖：3/3 通过

#### 5. 基础实体抽取 (cheungfun-indexing)
- ✅ `EntityExtractor` - 基于规则的实体抽取器
- ✅ 支持多种实体类型（人名、组织、地点、日期等）
- ✅ 关系抽取（工作关系、位置关系等）
- ✅ 测试覆盖：8/8 通过

### 📈 测试结果总结
- **总计测试**：24/24 通过 ✅
- **编译状态**：所有组件编译成功 ✅
- **API 兼容性**：完全兼容 LlamaIndex 设计 ✅

## 🚧 待实现功能

### 🔥 高优先级

#### 1. LLM 驱动的实体抽取
**目标**：替换当前基于规则的实体抽取，使用 LLM 进行智能抽取

**LlamaIndex 参考源码**：
- `llama_index/core/indices/property_graph/extractors/`
  - `simple.py` - SimpleLLMPathExtractor
  - `schema_llm.py` - SchemaLLMPathExtractor  
  - `implicit.py` - ImplicitPathExtractor
  - `dynamic.py` - DynamicLLMPathExtractor
- `llama_index/core/indices/property_graph/base.py` - PropertyGraphIndex 中的抽取集成

**实现要点**：
- 集成 siumai LLM 客户端
- 设计 Prompt 模板用于实体和关系抽取
- 实现结构化输出解析（JSON 格式的三元组）
- 添加抽取结果验证和后处理
- 支持批量文档处理和进度显示

**关键技术细节**：
- **Prompt 工程**: 参考 LlamaIndex 的 `DEFAULT_KG_TRIPLET_EXTRACT_PROMPT`
- **输出格式**: JSON 数组，每个元素包含 `(subject, predicate, object)` 三元组
- **错误处理**: LLM 输出解析失败时的降级策略
- **批处理**: 支持多文档并行处理，控制 API 调用频率
- **缓存机制**: 避免重复抽取相同内容

**预期文件**：
- `cheungfun-indexing/src/extractors/llm_extractor.rs`
- `cheungfun-indexing/src/extractors/schema_extractor.rs`
- `cheungfun-query/src/indices/property_graph_index.rs` (更新)

#### 2. 生产级图存储后端
**目标**：支持 Neo4j、Qdrant 等生产级图数据库

**LlamaIndex 参考源码**：
- `llama_index/graph_stores/neo4j.py` - Neo4j 图存储
- `llama_index/graph_stores/simple.py` - 简单图存储参考
- `llama_index/core/graph_stores/types.py` - 图存储接口定义

**实现要点**：
- Neo4j 驱动集成
- Cypher 查询构建器
- 连接池和事务管理
- 数据迁移和备份功能

**预期文件**：
- `cheungfun-integrations/src/graph_store/neo4j_store.rs`
- `cheungfun-integrations/src/graph_store/qdrant_graph_store.rs`

#### 3. 高级图检索策略
**目标**：实现更复杂的图遍历和检索算法

**LlamaIndex 参考源码**：
- `llama_index/core/indices/property_graph/retrievers/`
  - `llm_synonym.py` - LLMSynonymRetriever
  - `vector_context.py` - VectorContextRetriever
  - `custom.py` - CustomPGRetriever
  - `text_to_cypher.py` - TextToCypherRetriever

**实现要点**：
- 同义词扩展检索
- 向量上下文检索
- 自然语言到 Cypher 查询转换
- 多跳图遍历算法

**预期文件**：
- `cheungfun-query/src/retrievers/llm_synonym_retriever.rs`
- `cheungfun-query/src/retrievers/vector_context_retriever.rs`
- `cheungfun-query/src/retrievers/text_to_cypher_retriever.rs`

### 🔶 中优先级

#### 4. 图可视化
**目标**：提供图结构的可视化展示

**参考**：
- D3.js 图可视化
- Graphviz DOT 格式导出
- Web 界面集成

**预期文件**：
- `cheungfun-query/src/visualization/graph_viz.rs`
- `cheungfun-query/src/visualization/web_interface.rs`

#### 5. 图算法库
**目标**：实现常用的图算法

**功能**：
- 社区发现算法
- 中心性分析（PageRank、Betweenness）
- 最短路径算法
- 图聚类算法

**预期文件**：
- `cheungfun-query/src/algorithms/community_detection.rs`
- `cheungfun-query/src/algorithms/centrality.rs`
- `cheungfun-query/src/algorithms/clustering.rs`

#### 6. 性能优化
**目标**：支持大规模图数据处理

**优化点**：
- 并行图构建
- 增量索引更新
- 内存优化和缓存策略
- 查询性能优化

### 🔷 低优先级

#### 7. 多模态图支持
**目标**：支持图像、音频等多模态实体

**LlamaIndex 参考**：
- `llama_index/multi_modal/` 相关实现

#### 8. 图数据导入导出
**目标**：支持多种图数据格式

**格式支持**：
- GraphML
- GEXF
- JSON-LD
- RDF/Turtle

#### 9. 分布式图处理
**目标**：支持分布式图计算

**技术栈**：
- Apache Arrow
- 分布式图分区
- 并行查询执行

## 🎯 下一步行动计划

### 第一阶段：LLM 驱动实体抽取 (1-2 周)
1. **研究 LlamaIndex 实现**
   - 分析 `SimpleLLMPathExtractor` 的 Prompt 设计
   - 理解 `SchemaLLMPathExtractor` 的模式定义
   - 学习结构化输出解析逻辑

2. **设计 Rust 接口**
   - 定义 `LLMExtractor` trait
   - 设计 Prompt 模板系统
   - 实现输出解析器

3. **集成 siumai 客户端**
   - 添加 LLM 调用封装
   - 实现批量处理和重试机制
   - 添加成本控制和限流

4. **实现核心功能**
   - 实体识别和分类
   - 关系抽取和验证
   - 三元组生成和去重

5. **测试和优化**
   - 单元测试和集成测试
   - 性能基准测试
   - 准确率评估

### 第二阶段：生产级存储 (2-3 周)
1. **Neo4j 集成**
   - 添加 neo4j 依赖
   - 实现连接管理
   - Cypher 查询构建

2. **事务和性能**
   - 批量操作优化
   - 连接池配置
   - 错误恢复机制

3. **数据迁移**
   - SimplePropertyGraphStore → Neo4j 迁移工具
   - 数据一致性验证
   - 备份和恢复功能

### 第三阶段：高级检索 (2-3 周)
1. **同义词检索**
   - LLM 驱动的同义词扩展
   - 语义相似度计算
   - 查询重写机制

2. **向量上下文检索**
   - 图节点向量化
   - 混合检索策略
   - 上下文窗口优化

3. **自然语言查询**
   - 文本到 Cypher 转换
   - 查询意图理解
   - 结果解释生成

## 🔄 架构对比：Cheungfun vs LlamaIndex

### 核心组件映射

| LlamaIndex 组件 | Cheungfun 对应组件 | 实现状态 | 文件路径 |
|----------------|-------------------|----------|----------|
| `PropertyGraphIndex` | `PropertyGraphIndex` | ✅ 完成 | `cheungfun-query/src/indices/property_graph_index.rs` |
| `GraphStore` (trait) | `PropertyGraphStore` (trait) | ✅ 完成 | `cheungfun-core/src/traits/graph_store.rs` |
| `SimpleGraphStore` | `SimplePropertyGraphStore` | ✅ 完成 | `cheungfun-integrations/src/graph_store/simple_property_graph_store.rs` |
| `Neo4jGraphStore` | `Neo4jGraphStore` | 🚧 待实现 | `cheungfun-integrations/src/graph_store/neo4j_store.rs` |
| `SimpleLLMPathExtractor` | `LLMExtractor` | 🚧 待实现 | `cheungfun-indexing/src/extractors/llm_extractor.rs` |
| `SchemaLLMPathExtractor` | `SchemaExtractor` | 🚧 待实现 | `cheungfun-indexing/src/extractors/schema_extractor.rs` |
| `PropertyGraphRetriever` | `GraphRetriever` | ✅ 完成 | `cheungfun-query/src/retrievers/graph_retriever.rs` |
| `LLMSynonymRetriever` | `LLMSynonymRetriever` | 🚧 待实现 | `cheungfun-query/src/retrievers/llm_synonym_retriever.rs` |
| `VectorContextRetriever` | `VectorContextRetriever` | 🚧 待实现 | `cheungfun-query/src/retrievers/vector_context_retriever.rs` |

### 关键差异和优势

#### Cheungfun 优势
- **类型安全**: Rust 的编译时类型检查，避免运行时错误
- **内存安全**: 无 GC 的零成本抽象，更高的性能
- **并发安全**: 内置的并发安全保证，无需额外的锁机制
- **模块化设计**: 更清晰的模块边界和依赖关系

#### LlamaIndex 优势
- **生态成熟**: 丰富的预训练模型和集成
- **社区活跃**: 大量的示例和最佳实践
- **快速迭代**: Python 的灵活性支持快速原型开发

## 📚 参考资源

### LlamaIndex 核心源码

#### 主要模块
- **PropertyGraphIndex**: `llama_index/core/indices/property_graph/base.py`
- **图存储接口**: `llama_index/core/graph_stores/types.py`
- **实体抽取器**: `llama_index/core/indices/property_graph/extractors/`
- **图检索器**: `llama_index/core/indices/property_graph/retrievers/`

#### 具体实现文件
- **SimpleLLMPathExtractor**: `llama_index/core/indices/property_graph/extractors/simple.py`
- **SchemaLLMPathExtractor**: `llama_index/core/indices/property_graph/extractors/schema_llm.py`
- **ImplicitPathExtractor**: `llama_index/core/indices/property_graph/extractors/implicit.py`
- **DynamicLLMPathExtractor**: `llama_index/core/indices/property_graph/extractors/dynamic.py`
- **LLMSynonymRetriever**: `llama_index/core/indices/property_graph/retrievers/llm_synonym.py`
- **VectorContextRetriever**: `llama_index/core/indices/property_graph/retrievers/vector_context.py`
- **TextToCypherRetriever**: `llama_index/core/indices/property_graph/retrievers/text_to_cypher.py`

### 技术文档
- Neo4j Rust Driver 文档
- Cypher 查询语言参考
- 图算法理论基础
- 知识图谱构建最佳实践

### 测试数据集
- 准备多领域测试文档
- 构建标准评估基准
- 性能测试用例

## 🏆 成功指标

### 功能完整性
- [ ] LLM 抽取准确率 > 85%
- [ ] 支持 3+ 种图存储后端
- [ ] 检索延迟 < 100ms (P95)
- [ ] 支持 10K+ 实体的图

### 代码质量
- [ ] 测试覆盖率 > 90%
- [ ] 文档覆盖率 100%
- [ ] 零 unsafe 代码
- [ ] 完整的错误处理

### 性能指标
- [ ] 索引构建速度 > 1000 docs/min
- [ ] 内存使用 < 1GB (10K 实体)
- [ ] 并发查询支持 > 100 QPS
- [ ] 图遍历深度支持 > 5 跳

---

**最后更新**: 2024-12-19
**维护者**: Cheungfun 开发团队
