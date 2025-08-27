# Examples Quick Start Guide

## 新的架构说明

我们已经重新组织了示例架构，现在分为以下几个主要类别：

### 🏗️ 各Crate独立示例

每个crate现在有自己的核心功能演示：

```bash
# cheungfun-indexing - 文档处理和索引
cd cheungfun-indexing
cargo run --example text_splitters      # 文本分割器对比
cargo run --example code_parsing        # AST代码解析演示

# cheungfun-query - 查询处理
cd cheungfun-query  
cargo run --example query_pipeline      # 查询管道演示

# cheungfun-integrations - 性能对比
cd cheungfun-integrations
cargo run --example vector_store_performance  # 向量存储性能对比
```

### 📱 根目录综合应用

根目录现在专注于完整的端到端应用：

#### 应用程序 (applications/)
```bash
# 代码问答系统
cargo run --bin code_qa_system --features code-analysis -- /path/to/your/project

# Unity C# 专用RAG
cargo run --bin unity_csharp_rag --features code-analysis -- /path/to/unity/project

# 文档分析系统
cargo run --bin markdown_rag_system --features document-analysis -- /path/to/docs

# 智能助手
cargo run --bin comprehensive_assistant --features intelligent-assistant
cargo run --bin simple_assistant --features intelligent-assistant
```

#### 集成演示 (integrations/)
```bash
# KV存储集成
cargo run --bin kvstore_integration
```

#### 性能测试 (performance/)
```bash
# 综合性能测试
cargo run --bin comprehensive_performance --features benchmarks

# 混合性能演示  
cargo run --bin hybrid_performance --features performance

# 性能测试运行器
cargo run --bin performance_runner --features benchmarks
```

#### 实际用例 (use_cases/)
```bash
# Unity C# CLI工具
cargo run --bin unity_csharp_cli --features code-analysis -- /path/to/unity/project

# 高级RAG示例
cargo run --bin advanced_rag --features document-analysis

# RAG Agent示例
cargo run --bin rag_agent --features intelligent-assistant
```

#### 实用工具 (utilities/)
```bash
# AST分割器测试
cargo run --bin ast_splitter_test

# 调试代码分块
cargo run --bin debug_code_chunking --features code-analysis

# 增强文件过滤
cargo run --bin enhanced_file_filtering

# 列出所有示例
cargo run --bin list_examples
```

## 🚀 快速开始

### 1. 基础功能测试
```bash
# 最简单的示例 - 无需外部依赖
cargo run --bin hello_world

# 基本索引功能
cargo run --bin basic_indexing

# 基本查询功能  
cargo run --bin basic_querying
```

### 2. 代码分析应用
```bash
# 分析你的Rust项目
cargo run --bin code_qa_system --features code-analysis -- /path/to/your/rust/project

# 分析Unity C#项目
cargo run --bin unity_csharp_rag --features code-analysis -- /path/to/unity/project
```

### 3. 文档问答系统
```bash
# Markdown文档RAG
cargo run --bin markdown_rag_system --features document-analysis -- /path/to/markdown/docs
```

### 4. 性能测试
```bash
# 运行性能基准测试
cargo run --bin performance_runner --features benchmarks

# SIMD性能测试
cargo run --bin simple_simd_test --features performance

# HNSW性能测试  
cargo run --bin simple_hnsw_test --features performance
```

## 📋 Feature说明

新的feature组织更加清晰：

- **basic**: 基础功能，无外部依赖
- **code-analysis**: 代码分析应用bundle
- **document-analysis**: 文档分析应用bundle  
- **intelligent-assistant**: 智能助手应用bundle
- **knowledge-base**: 知识库应用bundle
- **performance**: 性能优化功能
- **benchmarks**: 基准测试工具
- **full**: 所有功能

## 💡 开发工作流

### 测试单个组件功能
```bash
cd cheungfun-indexing
cargo run --example text_splitters
```

### 构建完整应用
```bash
cargo run --bin code_qa_system --features code-analysis -- ./my-project
```

### 性能分析和优化
```bash
cargo run --bin performance_runner --features benchmarks
```

### 调试和实用工具
```bash
cargo run --bin debug_code_chunking --features code-analysis
cargo run --bin list_examples
```

这个新架构让开发更高效：各组件独立测试，应用级别集成，性能分析独立，实用工具齐全！