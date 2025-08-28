# RAG+Agent 智能问答系统

这个文件夹包含了完整的RAG+Agent智能问答系统示例，展示如何结合检索增强生成(RAG)和智能Agent构建下一代问答系统。

## 🎯 核心特性

### 智能化问答流程
- **智能问题分类**: 自动识别问题类型并选择最佳处理策略
- **ReAct推理**: 使用思考-行动-观察的循环进行复杂推理
- **长期记忆**: 维护对话历史和用户偏好
- **工具增强**: 集成专业化工具提升回答质量

### 多层次处理策略
1. **简单事实查询** → 直接RAG检索
2. **复杂分析问题** → ReAct推理Agent
3. **多文档对比** → 多Agent协作
4. **计算类问题** → 工具增强Agent
5. **对话式问题** → 上下文感知处理

## 📁 文件结构

```
08_rag_agent/
├── README.md                 # 本文件
├── main.rs                   # 主程序入口
├── memory/                   # 记忆管理模块
│   ├── mod.rs               # 记忆模块导出
│   ├── conversation.rs      # 对话历史管理
│   └── long_term.rs         # 长期记忆存储
├── agents/                   # Agent实现
│   ├── mod.rs               # Agent模块导出
│   ├── classifier.rs        # 问题分类器
│   └── rag_react.rs         # RAG增强的ReAct Agent
├── tools/                    # 专用工具
│   ├── mod.rs               # 工具模块导出
│   ├── rag_search.rs        # RAG搜索工具
│   ├── summarizer.rs        # 文档摘要工具
│   └── fact_checker.rs      # 事实核查工具
└── system.rs                # 系统核心逻辑
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 设置OpenAI API密钥（可选，会回退到Ollama）
export OPENAI_API_KEY="your-api-key-here"

# 或者启动本地Ollama
ollama serve
ollama pull llama3.2
```

### 2. 运行示例

```bash
# 进入示例目录
cd examples

# 运行RAG+Agent系统
cargo run --bin rag_agent_main --features "fastembed,agents"
```

### 3. 交互式使用

```
🤔 您的问题: 什么是RAG？
🔍 正在智能分析和处理...
📋 问题类型: SimpleFactual
🔍 使用直接RAG检索

🤖 AI回答:
RAG（Retrieval-Augmented Generation）是一种结合信息检索和文本生成的AI技术...
```

## 💡 示例问题

### 简单事实查询
- "什么是RAG？"
- "Rust的主要特性是什么？"

### 复杂分析问题
- "比较RAG和传统搜索的优缺点"
- "分析AI开发的主要挑战"

### 多文档对比
- "总结所有文档中关于AI的观点"
- "对比不同文档中的技术方案"

### 计算类问题
- "计算2+3*4的结果"
- "帮我分析这些数据的趋势"

### 对话式问题
- "继续上一个话题"
- "能详细解释一下吗？"

## 🧠 记忆系统

### 对话历史
- 自动保存每轮对话
- 支持上下文引用
- 智能摘要长对话

### 长期记忆
- 用户偏好学习
- 知识点关联
- 个性化回答风格

## 🛠️ 工具生态

### RAG专用工具
- **向量搜索**: 精确的语义检索
- **文档摘要**: 智能内容提取
- **事实核查**: 基于证据的验证

### 通用工具
- **数学计算**: 支持复杂计算
- **网络搜索**: 实时信息获取
- **文件操作**: 文档读写处理

## 📊 性能特性

- **智能路由**: 根据问题类型选择最优策略
- **并行处理**: 多Agent协作提升效率
- **缓存优化**: 减少重复计算
- **流式输出**: 实时响应用户

## 🔧 配置选项

系统支持多种配置选项：

```rust
// 记忆配置
MemoryConfig {
    max_conversation_length: 50,    // 最大对话轮数
    summary_threshold: 20,          // 摘要触发阈值
    enable_long_term_memory: true,  // 启用长期记忆
}

// Agent配置
AgentConfig {
    max_iterations: 5,              // 最大推理轮数
    enable_tool_use: true,          // 启用工具使用
    verbose_reasoning: true,        // 显示推理过程
}
```

## 🎯 技术亮点

1. **智能分类**: 使用LLM自动识别问题类型
2. **策略选择**: 根据问题复杂度选择处理方案
3. **记忆增强**: 维护对话上下文和长期记忆
4. **工具集成**: 专业化工具提升回答质量
5. **可扩展性**: 模块化设计便于功能扩展

## 📈 未来发展

- [ ] 多模态支持（图片、音频）
- [ ] 实时学习能力
- [ ] 个性化定制
- [ ] 协作网络
- [ ] 性能优化

---

这个RAG+Agent系统代表了下一代智能问答的发展方向，通过Agent的推理能力将传统RAG升级为智能化的问答体验！🚀
