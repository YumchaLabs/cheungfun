//! ReActAgent 实现示例
//!
//! 这个示例展示了如何实现和使用 ReActAgent，它使用观察-思考-行动循环
//! 进行复杂的多步推理。
//!
//! 运行示例：
//! ```bash
//! cargo run --example react_agent_example
//! ```

use cheungfun_agents::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志
    tracing_subscriber::fmt::init();

    println!("🤖 Cheungfun Agents - ReActAgent 示例");
    println!("=====================================");

    // 创建内存
    let memory_config = ChatMemoryConfig::with_token_limit(3000);
    let mut memory = ChatMemoryBuffer::new(memory_config);

    // 创建工具
    let calculator_tool = Arc::new(CalculatorTool::new());
    let search_tool = Arc::new(MockSearchTool::new());

    // 创建 ReActAgent (模拟实现)
    let agent = create_react_agent(vec![calculator_tool, search_tool]).await?;

    println!("🧠 创建了 ReActAgent，支持复杂推理");
    println!("🔧 可用工具: calculator, search");

    // 测试复杂推理场景
    let test_cases = vec![
        "我需要计算 15 * 23，然后搜索这个结果相关的信息",
        "帮我找到关于机器学习的信息，然后计算如果我每天学习2小时，一个月能学习多少小时",
        "计算 100 除以 4 的结果，然后告诉我这个数字的含义",
    ];

    for (i, query) in test_cases.iter().enumerate() {
        println!("\n--- 测试案例 {} ---", i + 1);
        println!("👤 用户: {}", query);

        // 使用 ReActAgent 进行推理
        let response = agent.chat(query, &mut memory).await?;

        println!("🤖 Agent: {}", response.content);
        println!("📊 推理步骤: {}", response.stats.tool_calls_count);
        println!("⏱️  执行时间: {}ms", response.stats.execution_time_ms);

        // 显示推理过程（如果有的话）
        if let Some(reasoning_steps) = response.stats.custom_metrics.get("reasoning_steps") {
            println!("🧠 推理过程:");
            if let Some(steps) = reasoning_steps.as_array() {
                for (step_idx, step) in steps.iter().enumerate() {
                    println!("   {}. {}", step_idx + 1, step.as_str().unwrap_or(""));
                }
            }
        }
    }

    // 显示内存统计
    let memory_stats = memory.stats();
    println!("\n--- 内存统计 ---");
    println!("💭 总消息数: {}", memory_stats.message_count);
    println!("🔢 估计 tokens: {}", memory_stats.estimated_tokens);

    println!("\n✨ ReActAgent 示例完成！");
    Ok(())
}

/// 创建 ReActAgent
async fn create_react_agent(tools: Vec<Arc<dyn Tool>>) -> Result<Arc<dyn Agent>> {
    let mut tool_registry = ToolRegistry::new();
    for tool in tools {
        tool_registry.register_tool(tool)?;
    }

    // 创建 Agent 配置
    let agent_config = AgentConfig {
        name: "react_agent".to_string(),
        description: Some("ReAct 推理 Agent".to_string()),
        max_execution_time_ms: Some(30000),
        max_tool_calls: Some(10),
        verbose: true,
        ..Default::default()
    };

    // 使用 AgentBuilder 创建 ReActAgent
    let agent = AgentBuilder::new()
        .name("react_agent")
        .description("ReAct reasoning agent with observation-thought-action loops")
        .config(agent_config)
        .tools(tool_registry)
        .react_strategy() // 使用 ReAct 策略
        .build()
        .await?;

    Ok(agent)
}

/// 创建 LLM 客户端
async fn create_llm_client() -> Result<Arc<dyn siumai::traits::ChatCapability>> {
    // 尝试不同的提供商
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let client = Siumai::builder()
            .openai()
            .api_key(&api_key)
            .model("gpt-4o-mini")
            .temperature(0.7)
            .max_tokens(1000)
            .build()
            .await?;
        return Ok(Arc::new(client));
    }

    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        let client = Siumai::builder()
            .anthropic()
            .api_key(&api_key)
            .model("claude-3-sonnet-20240229")
            .temperature(0.7)
            .max_tokens(1000)
            .build()
            .await?;
        return Ok(Arc::new(client));
    }

    // 默认使用 Ollama（假设本地运行）
    let client = Siumai::builder()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .temperature(0.7)
        .build()
        .await?;
    Ok(Arc::new(client))
}

/// 计算器工具示例
#[derive(Debug)]
pub struct CalculatorTool;

impl CalculatorTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "执行基本数学计算，支持加减乘除运算"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，如 '15 * 23' 或 '100 / 4'"
                    }
                },
                "required": ["expression"]
            }),
        }
    }

    async fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> Result<ToolResult> {
        let expression = args["expression"]
            .as_str()
            .ok_or_else(|| AgentError::InvalidToolArgs("Missing expression".to_string()))?;

        // 简单的计算器实现
        let result = match expression {
            expr if expr.contains(" * ") => {
                let parts: Vec<&str> = expr.split(" * ").collect();
                if parts.len() == 2 {
                    let a: f64 = parts[0].parse().map_err(|_| {
                        AgentError::ToolExecutionFailed("Invalid number".to_string())
                    })?;
                    let b: f64 = parts[1].parse().map_err(|_| {
                        AgentError::ToolExecutionFailed("Invalid number".to_string())
                    })?;
                    a * b
                } else {
                    return Err(AgentError::ToolExecutionFailed(
                        "Invalid multiplication expression".to_string(),
                    ));
                }
            }
            expr if expr.contains(" / ") => {
                let parts: Vec<&str> = expr.split(" / ").collect();
                if parts.len() == 2 {
                    let a: f64 = parts[0].parse().map_err(|_| {
                        AgentError::ToolExecutionFailed("Invalid number".to_string())
                    })?;
                    let b: f64 = parts[1].parse().map_err(|_| {
                        AgentError::ToolExecutionFailed("Invalid number".to_string())
                    })?;
                    if b == 0.0 {
                        return Err(AgentError::ToolExecutionFailed(
                            "Division by zero".to_string(),
                        ));
                    }
                    a / b
                } else {
                    return Err(AgentError::ToolExecutionFailed(
                        "Invalid division expression".to_string(),
                    ));
                }
            }
            expr if expr.contains(" + ") => {
                let parts: Vec<&str> = expr.split(" + ").collect();
                if parts.len() == 2 {
                    let a: f64 = parts[0].parse().map_err(|_| {
                        AgentError::ToolExecutionFailed("Invalid number".to_string())
                    })?;
                    let b: f64 = parts[1].parse().map_err(|_| {
                        AgentError::ToolExecutionFailed("Invalid number".to_string())
                    })?;
                    a + b
                } else {
                    return Err(AgentError::ToolExecutionFailed(
                        "Invalid addition expression".to_string(),
                    ));
                }
            }
            _ => {
                return Err(AgentError::ToolExecutionFailed(
                    "Unsupported expression".to_string(),
                ))
            }
        };

        Ok(ToolResult {
            success: true,
            content: format!("{} = {}", expression, result),
            metadata: std::collections::HashMap::new(),
        })
    }
}

/// 模拟搜索工具
#[derive(Debug)]
pub struct MockSearchTool;

impl MockSearchTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Tool for MockSearchTool {
    fn name(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "搜索相关信息和知识"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询词"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> Result<ToolResult> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| AgentError::InvalidToolArgs("Missing query".to_string()))?;

        // 模拟搜索结果
        let mock_results = match query.to_lowercase().as_str() {
            q if q.contains("机器学习") => {
                "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。主要包括监督学习、无监督学习和强化学习三大类。"
            }
            q if q.contains("345") => {
                "345是一个三位数，等于15×23的乘积。在数学中，它是一个合数，有多个因数。"
            }
            q if q.contains("25") => {
                "25是5的平方，也是100除以4的结果。在数学中，25是一个完全平方数。"
            }
            _ => {
                "找到了一些相关信息，但需要更具体的查询词来获得更准确的结果。"
            }
        };

        Ok(ToolResult {
            success: true,
            content: format!("搜索 '{}' 的结果: {}", query, mock_results),
            metadata: std::collections::HashMap::new(),
        })
    }
}
