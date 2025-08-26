# Cheungfun Agents 下一步开发计划

## 📋 当前状态

### ✅ 已完成
- **基础架构重构** - Agent trait 重新设计，支持内存驱动对话
- **推理策略系统** - DirectStrategy 和 FunctionCallingStrategy 实现
- **内存管理集成** - 完全集成 cheungfun-core 的 BaseMemory 系统
- **AgentBuilder 增强** - 支持策略选择和流畅构建
- **示例验证** - memory_chat_example.rs 成功运行

### 🚧 进行中
- **开发路线图制定** - 详细的五阶段开发计划
- **Siumai 集成规划** - LLM 连接库的深度集成方案

## 🎯 立即开始的任务

### 1. ReActAgent 实现 (优先级：🔥🔥🔥)

#### 目标
实现观察-思考-行动循环的推理模式，这是用户第二常用的 Agent 类型。

#### 具体任务
```rust
// 在 cheungfun-agents/src/agent/strategy.rs 中添加
pub struct ReActStrategy {
    max_iterations: usize,
    thought_template: String,
    action_template: String,
    observation_template: String,
    stop_sequences: Vec<String>,
}

impl ReActStrategy {
    pub fn new() -> Self { ... }
    
    async fn reasoning_loop(
        &self,
        input: &str,
        context: &ReasoningContext,
        tools: &[Arc<dyn Tool>],
    ) -> Result<ReasoningResult> {
        // 实现 ReAct 循环逻辑
        // 1. Thought: 分析问题
        // 2. Action: 选择和执行工具
        // 3. Observation: 观察结果
        // 4. 重复直到得出最终答案
    }
}
```

#### 实现步骤
1. **创建 ReActStrategy 结构** - 定义配置和模板
2. **实现推理循环** - 观察-思考-行动逻辑
3. **集成 Siumai** - LLM 调用和响应处理
4. **添加停止条件** - 避免无限循环
5. **错误处理** - 工具调用失败的恢复机制

#### 验证方式
- 运行 `react_agent_example.rs`
- 测试多步推理场景
- 验证工具调用链

### 2. Siumai 深度集成 (优先级：🔥🔥)

#### 目标
将 siumai 库深度集成到 Agent 系统中，提供统一的 LLM 接口。

#### 具体任务
```rust
// 在 cheungfun-agents/src/llm/ 中创建
pub struct LlmClientManager {
    clients: HashMap<String, Arc<dyn ChatCapability>>,
    factory: LlmClientFactory,
}

pub struct LlmClientFactory;

impl LlmClientFactory {
    pub async fn create_client(
        provider: &str,
        config: &LlmConfig,
    ) -> Result<Arc<dyn ChatCapability>> {
        // 支持 OpenAI, Anthropic, Ollama 等
    }
}
```

#### 实现步骤
1. **创建 LLM 模块** - `cheungfun-agents/src/llm/mod.rs`
2. **实现客户端工厂** - 支持多提供商
3. **添加配置管理** - LLM 参数配置
4. **集成到 Agent** - 替换现有的模拟实现
5. **添加错误处理** - 网络和 API 错误处理

#### 参考资源
- `repo-ref/siumai/examples/01_getting_started/quick_start.rs`
- `repo-ref/siumai/examples/02_core_features/streaming_chat.rs`
- `cheungfun-query/src/generator.rs` (已有集成示例)

### 3. QueryEngineTool 集成 (优先级：🔥🔥)

#### 目标
将 cheungfun-query 的查询引擎包装为工具，实现 RAG 功能。

#### 具体任务
```rust
// 在 cheungfun-agents/src/tool/builtin/ 中添加
pub struct QueryEngineTool {
    query_engine: Arc<dyn QueryEngine>,
    metadata: ToolMetadata,
    timeout: Duration,
}

impl QueryEngineTool {
    pub fn from_defaults(
        query_engine: Arc<dyn QueryEngine>,
        name: &str,
        description: &str,
    ) -> Self { ... }
}
```

#### 实现步骤
1. **创建 QueryEngineTool** - 包装查询引擎
2. **实现 Tool trait** - 标准工具接口
3. **添加参数验证** - 查询参数检查
4. **结果格式化** - 统一输出格式
5. **集成测试** - 与现有查询引擎测试

## 📅 开发时间表

### 第1周 (立即开始)
- **周一-周二**: ReActStrategy 基础实现
- **周三-周四**: Siumai 客户端管理器
- **周五**: QueryEngineTool 基础框架

### 第2周
- **周一-周二**: ReActAgent 完整实现
- **周三-周四**: 集成测试和调试
- **周五**: 文档和示例更新

### 第3周
- **周一-周二**: FunctionTool 增强
- **周三-周四**: ToolSpec 支持
- **周五**: 性能优化和缓存

## 🧪 测试策略

### 单元测试
```rust
#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_react_strategy_basic_reasoning() {
        // 测试基础推理循环
    }

    #[tokio::test]
    async fn test_llm_client_factory() {
        // 测试客户端创建
    }

    #[tokio::test]
    async fn test_query_engine_tool() {
        // 测试查询引擎工具
    }
}
```

### 集成测试
```rust
#[cfg(test)]
mod integration_tests {
    #[tokio::test]
    #[ignore] // 需要 API key
    async fn test_react_agent_with_real_llm() {
        // 使用真实 LLM 测试
    }
}
```

### 示例验证
- `react_agent_example.rs` - ReAct 推理示例
- `rag_agent_example.rs` - RAG 集成示例
- `multi_tool_example.rs` - 多工具协作示例

## 📚 文档计划

### API 文档
- **ReActAgent** - 使用指南和最佳实践
- **LLM 集成** - Siumai 配置和使用
- **工具开发** - 自定义工具开发指南

### 示例文档
- **快速开始** - 5分钟上手指南
- **高级用法** - 复杂场景示例
- **最佳实践** - 性能和安全建议

## 🔧 开发环境设置

### 环境变量
```bash
# OpenAI (可选)
export OPENAI_API_KEY="your-key"

# Anthropic (可选)
export ANTHROPIC_API_KEY="your-key"

# Ollama (本地，推荐用于开发)
# 确保 Ollama 在 http://localhost:11434 运行
```

### 依赖更新
```toml
# 在 cheungfun-agents/Cargo.toml 中
[dependencies]
siumai = { path = "../repo-ref/siumai" }
cheungfun-core = { path = "../cheungfun-core" }
cheungfun-query = { path = "../cheungfun-query" }
```

## 🎯 成功标准

### 功能目标
- ✅ ReActAgent 支持多步推理
- ✅ 支持 3+ LLM 提供商 (OpenAI, Anthropic, Ollama)
- ✅ QueryEngineTool 无缝集成
- ✅ 示例运行成功率 > 95%

### 性能目标
- 🎯 简单查询响应时间 < 2秒
- 🎯 复杂推理响应时间 < 10秒
- 🎯 内存使用 < 100MB
- 🎯 并发支持 > 50 请求/秒

### 开发体验目标
- 🎯 新功能学习成本 < 15分钟
- 🎯 API 一致性 > 90%
- 🎯 错误信息清晰度 > 85%

## 🚀 开始开发

### 立即行动
1. **克隆并设置环境**
2. **运行现有示例** - 验证基础功能
3. **开始 ReActStrategy 实现**
4. **设置 Siumai 集成**

### 开发流程
1. **功能分支** - 每个功能独立分支
2. **单元测试** - 先写测试，后写实现
3. **集成测试** - 验证端到端功能
4. **文档更新** - 同步更新文档和示例

### 协作方式
- **代码审查** - 确保代码质量
- **定期同步** - 每周进度回顾
- **问题跟踪** - 及时记录和解决问题

---

**准备好开始了吗？让我们从 ReActAgent 开始，打造一个真正强大的 Agent 系统！** 🚀
