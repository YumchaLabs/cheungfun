# Cheungfun Examples Architecture - Development-Focused

## 设计原则

1. **各crate独立演示**：每个crate展示自己的核心功能
2. **根目录综合应用**：完整的端到端应用示例
3. **快速开发友好**：不追求教程完整性，专注功能验证
4. **实用导向**：直接展示如何使用，减少解释性内容

## 推荐架构

### 各Crate示例结构

```
cheungfun-core/
├── examples/
│   ├── basic_types.rs              # 展示核心数据结构使用
│   ├── pipeline_builder.rs         # 展示Pipeline构建模式
│   └── config_management.rs        # 配置系统演示

cheungfun-indexing/
├── examples/
│   ├── file_loaders.rs             # 文件加载器演示
│   ├── code_parsing.rs             # AST代码解析演示
│   ├── text_splitters.rs           # 文本分割器对比
│   ├── metadata_extraction.rs      # 元数据提取演示
│   └── transform_pipeline.rs       # Transform链演示

cheungfun-query/
├── examples/
│   ├── basic_retrieval.rs          # 基础检索功能
│   ├── response_generation.rs      # LLM响应生成
│   ├── memory_management.rs        # 对话记忆管理
│   └── advanced_search.rs          # 高级搜索策略

cheungfun-agents/
├── examples/
│   ├── react_agent.rs              # ReAct推理代理
│   ├── tool_integration.rs         # 工具集成演示
│   ├── mcp_client.rs               # MCP客户端使用
│   ├── mcp_server.rs               # MCP服务端实现
│   └── multi_agent.rs              # 多代理协作

cheungfun-integrations/
├── examples/
│   ├── embedders/
│   │   ├── candle_embedder.rs      # Candle本地嵌入
│   │   ├── api_embedder.rs         # API嵌入服务
│   │   └── fastembed_demo.rs       # FastEmbed演示
│   ├── vector_stores/
│   │   ├── memory_store.rs         # 内存向量存储
│   │   ├── qdrant_integration.rs   # Qdrant集成
│   │   └── hnsw_performance.rs     # HNSW性能测试
│   ├── performance/
│   │   ├── simd_operations.rs      # SIMD加速演示
│   │   ├── gpu_acceleration.rs     # GPU加速演示
│   │   └── benchmark_suite.rs      # 性能基准测试
│   └── storage/
│       ├── kvstore_backends.rs     # KV存储后端
│       └── database_integration.rs # 数据库集成

cheungfun-multimodal/
├── examples/
│   ├── image_processing.rs         # 图像处理演示
│   ├── audio_analysis.rs           # 音频分析演示
│   └── multimodal_rag.rs           # 多模态RAG演示
```

### 根目录综合示例结构

```
examples/
├── applications/                   # 完整应用示例
│   ├── code_qa_system/            # 代码问答系统
│   │   ├── main.rs                # 主程序入口
│   │   ├── indexer.rs             # 代码索引器
│   │   ├── query_engine.rs        # 查询引擎
│   │   └── config.toml            # 配置文件
│   │
│   ├── document_analysis/         # 文档分析系统
│   │   ├── main.rs
│   │   ├── pipeline.rs
│   │   └── README.md
│   │
│   ├── intelligent_assistant/     # 智能助手
│   │   ├── main.rs                # 带Agent的RAG系统
│   │   ├── agents/
│   │   └── tools/
│   │
│   └── knowledge_base/            # 知识库系统
│       ├── indexing_service.rs    # 索引服务
│       ├── query_service.rs       # 查询服务
│       └── web_interface.rs       # Web界面
│
├── integrations/                  # 集成演示
│   ├── openai_rag.rs             # OpenAI集成RAG
│   ├── claude_rag.rs             # Claude集成RAG
│   ├── local_llm_rag.rs          # 本地LLM RAG
│   ├── qdrant_production.rs      # Qdrant生产环境
│   └── database_backed_rag.rs    # 数据库支持的RAG
│
├── performance/                   # 性能演示
│   ├── end_to_end_benchmark.rs   # 端到端性能测试
│   ├── component_comparison.rs   # 组件性能对比
│   ├── memory_usage_analysis.rs  # 内存使用分析
│   └── scaling_tests.rs          # 扩展性测试
│
├── use_cases/                     # 实际用例
│   ├── unity_code_assistant.rs   # Unity代码助手
│   ├── rust_documentation.rs     # Rust文档问答
│   ├── research_papers.rs        # 学术论文分析
│   └── api_documentation.rs      # API文档问答
│
└── utilities/                     # 实用工具
    ├── benchmark_runner.rs        # 基准测试运行器
    ├── example_validator.rs       # 示例验证器
    ├── config_generator.rs        # 配置生成器
    └── data_importer.rs           # 数据导入工具
```

## Cargo.toml配置策略

### 各Crate的examples配置
```toml
# 每个crate保持简单，只演示本crate功能
[[bin]]
name = "file_loaders"
path = "examples/file_loaders.rs"
required-features = []  # 尽量减少依赖

[[bin]]
name = "code_parsing"
path = "examples/code_parsing.rs"
required-features = ["tree-sitter"]
```

### 根目录examples配置
```toml
# 根目录可以有复杂的feature组合
[features]
# 应用级feature bundles
code-assistant = ["cheungfun-indexing/code-parsing", "fastembed", "qdrant"]
document-analysis = ["candle", "performance"]
production-ready = ["all-embedders", "qdrant", "performance"]

# 示例分类
[[bin]]
name = "code_qa_system"
path = "applications/code_qa_system/main.rs"
required-features = ["code-assistant"]

[[bin]]
name = "openai_rag"
path = "integrations/openai_rag.rs"
required-features = ["api-embedders"]
```

## 开发工作流程

### 新功能开发
1. 在对应crate的`examples/`下创建功能演示
2. 确保示例能独立运行和测试
3. 如果需要，在根目录创建综合应用

### 集成测试
1. 各crate示例用于单元功能验证
2. 根目录示例用于集成测试
3. 性能测试统一在根目录进行

### 用户使用
1. **学习单个组件**：查看对应crate的examples
2. **构建完整应用**：参考根目录的applications
3. **性能优化**：参考performance示例

## 实施建议

### 即时行动（本周）
1. **清理现有示例**：将相似功能的示例合并
2. **重新分类**：按照新架构移动示例文件
3. **简化依赖**：减少复杂的feature要求

### 持续维护
1. **新功能必须有示例**：每个新功能在对应crate添加演示
2. **定期验证示例**：确保所有示例都能正常运行
3. **性能回归检测**：重要示例加入CI/CD

### 文档策略
- **README简洁明了**：只说明如何运行，不解释原理
- **代码注释实用**：重点注释配置和关键步骤
- **避免过度文档**：专注代码演示而非教学

这个架构更适合快速开发阶段，既保持了各组件的独立性，又提供了完整的应用演示，同时避免了过度的教程化内容。