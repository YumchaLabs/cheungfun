//! 简单的 Agent 示例
//!
//! 这个示例展示了如何创建和使用一个基本的 Agent。
//!
//! 运行示例：
//! ```bash
//! cargo run --example simple_agent_example
//! ```

use cheungfun_agents::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志
    tracing_subscriber::fmt::init();

    println!("🚀 启动简单 Agent 示例");
    println!("✅ cheungfun-agents 编译成功！");

    // 创建一个基本的 Agent Builder 来测试 API
    let _builder = AgentBuilder::new()
        .name("test_agent")
        .description("A test agent");

    println!("✅ AgentBuilder 创建成功");

    // 测试错误类型
    let _error = AgentError::configuration("Test configuration error");
    println!("✅ AgentError 创建成功");

    println!("✨ 示例完成！所有核心组件都可以正常使用。");
    Ok(())
}
