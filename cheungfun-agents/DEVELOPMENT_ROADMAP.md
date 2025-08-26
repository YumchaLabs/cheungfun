# Cheungfun Agents 开发路线图

## 🎯 项目目标

基于对 LlamaIndex Agent 实现的深入分析，制定 cheungfun-agents 向主流用法看齐的开发计划。目标是支撑用户主流的用法，避免过于企业化或不常用的功能。

## 🔧 技术栈说明

### LLM 连接库
我们使用自研的 **siumai** 库进行 LLM 连接：
- **位置**: `e:\Rust\cheungfun/repo-ref\siumai/`
- **特性**: 统一的多提供商 LLM 接口
- **支持**: OpenAI, Anthropic, Ollama, Google 等
- **架构**: 基于能力分离的 trait 设计

### Siumai 核心用法
```rust
use siumai::prelude::*;

// 统一接口创建客户端
let client = Siumai::builder()
    .openai()
    .api_key("your-api-key")
    .model("gpt-4")
    .temperature(0.7)
    .build()
    .await?;

// 发送聊天请求
let messages = vec![user!("Hello, world!")];
let response = client.chat(messages).await?;

// 流式响应
let stream = client.chat_stream(messages).await?;
```

### 参考资源
- **源码**: `repo-ref/siumai/src/`
- **示例**: `repo-ref/siumai/examples/`
- **文档**: `repo-ref/siumai/README.md`

## 📊 主流用法分析

基于 LlamaIndex 用户行为分析：

### 🔥 最常用模式 (90%+ 用例)
1. **FunctionAgent + 简单工具** - 基础工具调用
2. **QueryEngineTool + RAG集成** - 知识库问答
3. **ChatMemoryBuffer + token限制** - 对话历史管理
4. **异步执行 + Context管理** - 状态管理

### 📈 典型使用场景
- **RAG问答**: `QueryEngineTool` + `FunctionAgent`
- **API集成**: `ToolSpec` + `FunctionAgent`  
- **多轮对话**: `Memory` + `Context`
- **复杂推理**: `ReActAgent` + 多工具

## 🏗️ 五阶段开发计划

### 第一阶段：核心 Agent 类型 (优先级：🔥🔥🔥)

#### 1.1 FunctionAgent 增强 ✅ (已完成)
- ✅ 基础 FunctionAgent 实现
- ✅ 工具调用策略集成
- ✅ 内存管理集成
- ✅ Siumai LLM 连接

#### 1.2 ReActAgent 实现 🚧 (进行中)
**目标**: 实现观察-思考-行动循环的推理模式

**核心组件**:
```rust
/// ReAct (Reasoning + Acting) Agent 实现
pub struct ReActAgent {
    config: AgentConfig,
    tool_registry: Arc<ToolRegistry>,
    strategy: ReActStrategy,
    llm_client: Arc<dyn ChatCapability>,
    max_iterations: usize,
}

/// ReAct 推理策略
pub struct ReActStrategy {
    thought_template: String,
    action_template: String,
    observation_template: String,
    max_iterations: usize,
    stop_sequences: Vec<String>,
}
```

**实现要点**:
- 观察-思考-行动循环
- 中间推理步骤可见
- 支持复杂多步推理
- 错误恢复机制
- 与 siumai 深度集成

#### 1.3 QueryEngineTool 集成
**目标**: 将 cheungfun-query 集成为工具

```rust
/// 查询引擎工具，集成 cheungfun-query
pub struct QueryEngineTool {
    query_engine: Arc<dyn QueryEngine>,
    metadata: ToolMetadata,
    description: String,
    timeout: Duration,
}

impl QueryEngineTool {
    pub fn from_defaults(
        query_engine: Arc<dyn QueryEngine>,
        name: &str,
        description: &str,
    ) -> Self { ... }
    
    pub fn with_timeout(mut self, timeout: Duration) -> Self { ... }
}
```

**集成要点**:
- 无缝集成 cheungfun-query
- 支持向量检索和混合搜索
- 自动结果格式化
- 错误处理和超时控制

### 第二阶段：工具生态完善 (优先级：🔥🔥)

#### 2.1 FunctionTool 增强
**目标**: 提升工具易用性和安全性

```rust
/// 增强的函数工具
pub struct FunctionTool {
    function: Box<dyn AsyncFn>,
    metadata: ToolMetadata,
    validator: Option<Box<dyn ParameterValidator>>,
    retry_config: Option<RetryConfig>,
    timeout: Duration,
}

/// 参数验证器
pub trait ParameterValidator {
    fn validate(&self, args: &serde_json::Value) -> Result<()>;
    fn sanitize(&self, args: &mut serde_json::Value) -> Result<()>;
}
```

#### 2.2 ToolSpec 支持
**目标**: 快速集成第三方 API

```rust
/// 工具规范，类似 LlamaIndex 的 ToolSpec
pub trait ToolSpec {
    fn to_tool_list(&self) -> Vec<Arc<dyn Tool>>;
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn category(&self) -> &str;
}

/// HTTP API 工具规范示例
pub struct HttpToolSpec {
    base_url: String,
    api_key: Option<String>,
    endpoints: Vec<EndpointConfig>,
    rate_limiter: Option<RateLimiter>,
}
```

#### 2.3 工具安全和验证
- 参数类型验证和清理
- 权限检查机制
- 危险操作确认
- 执行超时控制
- 速率限制

### 第三阶段：内存和上下文管理 (优先级：🔥🔥)

#### 3.1 Memory 系统增强
**目标**: 更智能的对话历史管理

```rust
/// 扩展内存配置
pub struct AdvancedMemoryConfig {
    pub token_limit: usize,
    pub message_limit: Option<usize>,
    pub summarization_enabled: bool,
    pub compression_ratio: f32,
    pub preserve_system_messages: bool,
    pub auto_cleanup: bool,
}

/// 会话内存管理器
pub struct SessionMemoryManager {
    sessions: HashMap<String, SessionMemory>,
    cleanup_interval: Duration,
    max_sessions: usize,
}
```

#### 3.2 Context 管理
**目标**: 统一的执行上下文管理

```rust
/// Agent 执行上下文
pub struct AgentContext {
    session_id: Option<String>,
    memory: Box<dyn BaseMemory>,
    metadata: HashMap<String, serde_json::Value>,
    tool_results: Vec<ToolResult>,
    execution_stats: ExecutionStats,
}

impl AgentContext {
    pub fn new(session_id: Option<String>) -> Self { ... }
    pub fn with_memory(mut self, memory: Box<dyn BaseMemory>) -> Self { ... }
    pub fn fork(&self) -> Self { ... } // 创建子上下文
}
```

#### 3.3 Session 持久化
- 内存状态持久化到数据库
- 会话恢复机制
- 跨会话数据共享
- 自动清理过期会话

### 第四阶段：流式和异步优化 (优先级：🔥)

#### 4.1 流式响应完善
**目标**: 真正的流式 Agent 体验

```rust
/// 流式 Agent 响应
pub struct StreamingAgent {
    inner: Box<dyn Agent>,
    buffer_size: usize,
    flush_interval: Duration,
}

/// 流式事件类型
pub enum AgentStreamEvent {
    Thinking(String),
    ToolCall(ToolCall),
    ToolResult(ToolOutput),
    PartialResponse(String),
    Complete(AgentResponse),
    Error(AgentError),
}
```

#### 4.2 事件系统
**目标**: 可观测的 Agent 执行

```rust
/// Agent 事件处理器
pub trait AgentEventHandler: Send + Sync {
    async fn on_tool_call(&self, call: &ToolCall);
    async fn on_tool_result(&self, result: &ToolOutput);
    async fn on_thinking(&self, thought: &str);
    async fn on_error(&self, error: &AgentError);
}

/// 事件驱动的 Agent
pub struct EventDrivenAgent {
    inner: Box<dyn Agent>,
    handlers: Vec<Box<dyn AgentEventHandler>>,
    event_buffer: VecDeque<AgentEvent>,
}
```

### 第五阶段：高级特性 (优先级：⚠️)

#### 5.1 多 Agent 协作
**目标**: 简单的 Agent 间协作

```rust
/// Agent 协作工作流
pub struct AgentWorkflow {
    agents: HashMap<String, Arc<dyn Agent>>,
    handoff_rules: Vec<HandoffRule>,
    coordinator: SimpleCoordinator,
}

/// 简单协调器
pub struct SimpleCoordinator {
    routing_rules: Vec<RoutingRule>,
    max_handoffs: usize,
}
```

#### 5.2 错误处理和重试
**目标**: 生产级错误处理

```rust
/// 重试配置
pub struct RetryConfig {
    max_attempts: usize,
    backoff_strategy: BackoffStrategy,
    retry_on: Vec<ErrorType>,
    timeout: Duration,
}

/// 错误恢复策略
pub trait ErrorRecoveryStrategy {
    async fn recover(
        &self, 
        error: &AgentError, 
        context: &AgentContext
    ) -> Result<RecoveryAction>;
}
```

#### 5.3 性能监控
**目标**: 基础性能监控

```rust
/// Agent 性能指标
pub struct AgentMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub average_response_time: Duration,
    pub tool_usage_stats: HashMap<String, ToolUsageStats>,
    pub memory_usage: MemoryUsageStats,
}
```

## 🎯 实现优先级

### 🔥 立即实现 (第1-2阶段) - 4周
1. **ReActAgent** - 满足复杂推理需求
2. **QueryEngineTool** - RAG 集成的核心
3. **FunctionTool 增强** - 提升工具易用性
4. **ToolSpec 支持** - 快速集成第三方 API

### 🔥 近期实现 (第3阶段) - 3周
1. **Memory 增强** - 更好的对话体验
2. **Context 管理** - 状态管理优化
3. **Session 持久化** - 生产环境必需

### ⚠️ 可选实现 (第4-5阶段) - 按需
1. **流式响应** - 用户体验提升
2. **多 Agent 协作** - 复杂场景支持
3. **高级监控** - 企业级特性

## 📋 Siumai 集成指南

### 基础集成模式
```rust
use siumai::prelude::*;

// 在 Agent 中集成 siumai
pub struct SiumaiAgent {
    client: Arc<dyn ChatCapability>,
    config: AgentConfig,
    // ... 其他字段
}

impl SiumaiAgent {
    pub async fn new(
        provider: &str,
        api_key: &str,
        model: &str,
    ) -> Result<Self> {
        let client = match provider {
            "openai" => Siumai::builder()
                .openai()
                .api_key(api_key)
                .model(model)
                .build()
                .await?,
            "anthropic" => Siumai::builder()
                .anthropic()
                .api_key(api_key)
                .model(model)
                .build()
                .await?,
            "ollama" => Siumai::builder()
                .ollama()
                .base_url("http://localhost:11434")
                .model(model)
                .build()
                .await?,
            _ => return Err(AgentError::UnsupportedProvider(provider.to_string())),
        };
        
        Ok(Self {
            client: Arc::new(client),
            config: AgentConfig::default(),
        })
    }
}
```

### 参考示例
- **基础用法**: `repo-ref/siumai/examples/01_getting_started/quick_start.rs`
- **流式响应**: `repo-ref/siumai/examples/02_core_features/streaming_chat.rs`
- **错误处理**: `repo-ref/siumai/examples/02_core_features/error_handling.rs`
- **多提供商**: `repo-ref/siumai/examples/01_getting_started/provider_comparison.rs`

## 📝 实现细节和最佳实践

### Siumai 集成最佳实践

#### 1. 统一的 LLM 客户端管理
```rust
/// LLM 客户端工厂
pub struct LlmClientFactory;

impl LlmClientFactory {
    pub async fn create_client(
        provider: &str,
        config: &LlmConfig,
    ) -> Result<Arc<dyn ChatCapability>> {
        match provider.to_lowercase().as_str() {
            "openai" => {
                let client = Siumai::builder()
                    .openai()
                    .api_key(&config.api_key)
                    .model(&config.model)
                    .temperature(config.temperature)
                    .max_tokens(config.max_tokens)
                    .build()
                    .await?;
                Ok(Arc::new(client))
            }
            "anthropic" => {
                let client = Siumai::builder()
                    .anthropic()
                    .api_key(&config.api_key)
                    .model(&config.model)
                    .temperature(config.temperature)
                    .max_tokens(config.max_tokens)
                    .build()
                    .await?;
                Ok(Arc::new(client))
            }
            "ollama" => {
                let client = Siumai::builder()
                    .ollama()
                    .base_url(&config.base_url.unwrap_or_else(|| "http://localhost:11434".to_string()))
                    .model(&config.model)
                    .temperature(config.temperature)
                    .build()
                    .await?;
                Ok(Arc::new(client))
            }
            _ => Err(AgentError::UnsupportedProvider(provider.to_string())),
        }
    }
}
```

#### 2. 消息格式转换
```rust
/// 将 cheungfun ChatMessage 转换为 siumai 格式
pub fn convert_to_siumai_messages(
    messages: &[cheungfun_core::ChatMessage],
) -> Vec<siumai::types::ChatMessage> {
    messages
        .iter()
        .map(|msg| {
            match msg.role {
                cheungfun_core::MessageRole::User => {
                    siumai::types::ChatMessage::user(&msg.content).build()
                }
                cheungfun_core::MessageRole::Assistant => {
                    siumai::types::ChatMessage::assistant(&msg.content).build()
                }
                cheungfun_core::MessageRole::System => {
                    siumai::types::ChatMessage::system(&msg.content).build()
                }
            }
        })
        .collect()
}
```

### 测试策略

#### 1. 单元测试
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_react_agent_basic_reasoning() {
        let mock_client = create_mock_llm_client();
        let agent = ReActAgent::new(mock_client, default_config()).await.unwrap();

        let response = agent.chat("What is 2+2?", &mut mock_memory()).await.unwrap();
        assert!(response.content.contains("4"));
    }

    #[tokio::test]
    async fn test_query_engine_tool_integration() {
        let mock_query_engine = create_mock_query_engine();
        let tool = QueryEngineTool::from_defaults(
            Arc::new(mock_query_engine),
            "test_tool",
            "Test query engine tool"
        );

        let result = tool.execute(
            serde_json::json!({"query": "test query"}),
            &default_tool_context()
        ).await.unwrap();

        assert!(result.success);
    }
}
```

#### 2. 集成测试
```rust
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // 需要真实的 API key
    async fn test_openai_integration() {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap();
        let agent = AgentBuilder::new()
            .name("test_agent")
            .llm_provider("openai")
            .api_key(&api_key)
            .model("gpt-4o-mini")
            .function_calling_strategy()
            .build()
            .await
            .unwrap();

        let mut memory = ChatMemoryBuffer::new(ChatMemoryConfig::default());
        let response = agent.chat("Hello!", &mut memory).await.unwrap();

        assert!(!response.content.is_empty());
    }
}
```

### 性能优化建议

#### 1. 连接池管理
```rust
/// LLM 客户端池
pub struct LlmClientPool {
    clients: Arc<RwLock<HashMap<String, Arc<dyn ChatCapability>>>>,
    max_clients: usize,
}

impl LlmClientPool {
    pub async fn get_or_create_client(
        &self,
        provider: &str,
        config: &LlmConfig,
    ) -> Result<Arc<dyn ChatCapability>> {
        let key = format!("{}:{}", provider, config.model);

        // 先尝试从池中获取
        {
            let clients = self.clients.read().await;
            if let Some(client) = clients.get(&key) {
                return Ok(Arc::clone(client));
            }
        }

        // 创建新客户端
        let client = LlmClientFactory::create_client(provider, config).await?;

        // 添加到池中
        {
            let mut clients = self.clients.write().await;
            if clients.len() < self.max_clients {
                clients.insert(key, Arc::clone(&client));
            }
        }

        Ok(client)
    }
}
```

#### 2. 缓存策略
```rust
/// 响应缓存
pub struct ResponseCache {
    cache: Arc<RwLock<LruCache<String, CachedResponse>>>,
    ttl: Duration,
}

impl ResponseCache {
    pub async fn get_or_compute<F, Fut>(
        &self,
        key: &str,
        compute: F,
    ) -> Result<String>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<String>>,
    {
        // 检查缓存
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(key) {
                if cached.created_at.elapsed() < self.ttl {
                    return Ok(cached.response.clone());
                }
            }
        }

        // 计算新值
        let response = compute().await?;

        // 更新缓存
        {
            let mut cache = self.cache.write().await;
            cache.put(key.to_string(), CachedResponse {
                response: response.clone(),
                created_at: Instant::now(),
            });
        }

        Ok(response)
    }
}
```

## 🚀 下一步行动

### 立即开始 (本周)
1. **ReActAgent 实现** - 创建基础 ReAct 推理循环
2. **Siumai 深度集成** - 完善 LLM 客户端管理
3. **单元测试框架** - 建立测试基础设施

### 短期目标 (2周内)
1. **QueryEngineTool 集成** - 连接 cheungfun-query
2. **FunctionTool 增强** - 添加验证和安全机制
3. **示例和文档** - 创建使用示例

### 中期目标 (1个月内)
1. **ToolSpec 支持** - 快速 API 集成
2. **Memory 系统增强** - 智能对话管理
3. **性能优化** - 连接池和缓存

### 长期目标 (按需实现)
1. **流式响应** - 实时用户体验
2. **多 Agent 协作** - 复杂工作流
3. **企业级特性** - 监控和管理

## 📊 成功指标

### 功能完整性
- ✅ 支持 90% 的 LlamaIndex 主流用例
- ✅ 与 cheungfun 生态无缝集成
- ✅ 支持主流 LLM 提供商

### 性能指标
- 🎯 响应时间 < 2秒 (简单查询)
- 🎯 内存使用 < 100MB (单 Agent)
- 🎯 并发支持 > 100 请求/秒

### 开发体验
- 🎯 API 学习成本 < 30分钟
- 🎯 示例覆盖率 > 80%
- 🎯 文档完整性 > 90%

这个路线图确保我们：
- ✅ **支撑主流用法** - 覆盖 90% 的用户场景
- ✅ **避免过度工程** - 不实现企业级复杂功能
- ✅ **渐进式开发** - 每个阶段都有可用的功能
- ✅ **深度集成 siumai** - 充分利用自研 LLM 库的优势
- ✅ **性能优先** - 注重实际使用中的性能表现
