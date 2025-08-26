# Cheungfun Agents 架构重构方案

基于 LlamaIndex 最佳实践的代码重组方案。

## 🎯 目标架构

```
cheungfun-agents/src/
├── lib.rs                    # 统一导出
├── error.rs                  # 错误处理
├── types.rs                  # 核心类型
├── agent/                    # Agent 系统（新增）
│   ├── mod.rs               # Agent 统一导出
│   ├── base.rs              # BaseAgent trait
│   ├── react/               # ReAct 实现
│   │   ├── mod.rs
│   │   ├── agent.rs         # ReActAgent 实现
│   │   ├── formatter.rs     # 从 workflow/react/ 移动
│   │   ├── output_parser.rs # 从 workflow/react/ 移动
│   │   ├── tool_executor.rs # 从 workflow/react/ 移动
│   │   ├── tool_selector.rs # 从 workflow/react/ 移动
│   │   └── reasoning.rs     # 从 workflow/react/ 移动
│   └── workflow/            # Workflow Agent 实现
│       ├── mod.rs
│       ├── base_agent.rs    # 从 workflow/ 移动
│       ├── context.rs       # 从 workflow/ 移动
│       └── events.rs        # 从 workflow/ 移动
├── tools/                   # 工具系统（保持）
│   ├── mod.rs
│   ├── base.rs              # BaseTool trait
│   ├── registry.rs
│   ├── calling.rs           # 工具调用逻辑（新增）
│   └── builtin/
├── llm/                     # LLM 集成（保持）
├── mcp/                     # MCP 协议（保持）
└── workflow/                # 纯工作流引擎（简化）
    ├── mod.rs
    ├── workflow.rs          # 核心工作流引擎
    ├── context.rs           # 工作流上下文
    └── events.rs            # 工作流事件
```

## 🔄 重构步骤

### 1. 创建新的 agent 模块
- 创建 `src/agent/` 目录
- 实现 `BaseAgent` trait
- 移动 ReAct 相关代码到 `agent/react/`

### 2. 重组工具系统
- 在 `tools/` 中添加 `calling.rs`
- 统一工具调用接口
- 简化工具选择和执行逻辑

### 3. 简化 workflow 模块
- 移除 agent 相关代码
- 专注于纯工作流引擎功能
- 保持事件系统和上下文管理

### 4. 更新导出结构
- 更新 `lib.rs` 的导出
- 确保向后兼容性
- 优化 prelude 模块

## 🎯 预期效果

1. **清晰的职责分离**：
   - Agent 系统独立管理
   - 工具系统统一接口
   - 工作流引擎专注核心功能

2. **更好的可维护性**：
   - 模块边界清晰
   - 依赖关系简单
   - 易于扩展新功能

3. **符合业界标准**：
   - 参考 LlamaIndex 架构
   - 遵循 Rust 最佳实践
   - 便于社区贡献

## 📋 实施计划

1. **Phase 1**: 创建新的 agent 模块结构
2. **Phase 2**: 移动和重构 ReAct 相关代码
3. **Phase 3**: 重组工具系统
4. **Phase 4**: 简化 workflow 模块
5. **Phase 5**: 更新导出和测试
