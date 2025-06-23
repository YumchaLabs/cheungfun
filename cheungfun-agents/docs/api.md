# Cheungfun Agents API Documentation

This document provides comprehensive API documentation for the Cheungfun Agents framework.

## Core Traits

### Agent Trait

The `Agent` trait is the core interface that all agents must implement.

```rust
#[async_trait]
pub trait Agent: Send + Sync + std::fmt::Debug {
    fn id(&self) -> AgentId;
    fn name(&self) -> &str;
    fn description(&self) -> Option<&str>;
    fn capabilities(&self) -> &AgentCapabilities;
    fn config(&self) -> &AgentConfig;
    
    async fn execute(&self, task: &Task) -> Result<AgentResponse>;
    async fn process_message(&self, message: &AgentMessage) -> Result<AgentResponse>;
    
    fn tools(&self) -> Vec<String>;
    fn supports_capability(&self, capability: &str) -> bool;
    fn can_handle_task(&self, task: &Task) -> Result<()>;
    async fn health_check(&self) -> Result<AgentHealthStatus>;
}
```

### Tool Trait

The `Tool` trait defines the interface for agent tools.

```rust
#[async_trait]
pub trait Tool: Send + Sync + std::fmt::Debug {
    fn schema(&self) -> ToolSchema;
    async fn execute(&self, arguments: serde_json::Value, context: &ToolContext) -> Result<ToolResult>;
    
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn is_dangerous(&self) -> bool;
    fn validate_arguments(&self, arguments: &serde_json::Value) -> Result<()>;
    fn capabilities(&self) -> Vec<String>;
}
```

## Agent Builder

### AgentBuilder

The `AgentBuilder` provides a fluent API for creating agents.

```rust
impl AgentBuilder {
    pub fn new() -> Self;
    pub fn name(self, name: impl Into<String>) -> Self;
    pub fn description(self, description: impl Into<String>) -> Self;
    pub fn instructions(self, instructions: impl Into<String>) -> Self;
    pub fn capabilities(self, capabilities: AgentCapabilities) -> Self;
    
    // Capability shortcuts
    pub fn with_tools(self) -> Self;
    pub fn with_streaming(self) -> Self;
    pub fn with_conversation(self) -> Self;
    pub fn with_files(self) -> Self;
    pub fn with_web(self) -> Self;
    pub fn with_code_execution(self) -> Self;
    
    // Configuration
    pub fn max_execution_time_ms(self, ms: u64) -> Self;
    pub fn max_tool_calls(self, count: usize) -> Self;
    pub fn verbose(self) -> Self;
    
    // Tools
    pub fn tool(self, tool: Arc<dyn Tool>) -> Self;
    pub fn tools(self, tools: Vec<Arc<dyn Tool>>) -> Self;
    pub fn tool_registry(self, registry: Arc<ToolRegistry>) -> Self;
    
    pub fn build(self) -> Result<Box<dyn Agent>>;
}
```

### Preset Agents

```rust
impl AgentBuilder {
    pub fn assistant() -> Self;        // Basic assistant
    pub fn researcher() -> Self;       // Research specialist
    pub fn file_manager() -> Self;     // File operations
    pub fn code_assistant() -> Self;   // Code analysis
    pub fn web_agent() -> Self;        // Web interactions
}
```

## Task Management

### Task

```rust
pub struct Task {
    pub id: TaskId,
    pub name: String,
    pub description: Option<String>,
    pub input: String,
    pub priority: TaskPriority,
    pub status: TaskStatus,
    pub agent_id: Option<AgentId>,
    pub context: TaskContext,
    pub metadata: HashMap<String, serde_json::Value>,
    // ... other fields
}

impl Task {
    pub fn new(input: impl Into<String>) -> Self;
    pub fn builder() -> TaskBuilder;
    pub fn set_status(&mut self, status: TaskStatus);
    pub fn assign_agent(&mut self, agent_id: AgentId);
    pub fn add_context_variable(&mut self, key: impl Into<String>, value: serde_json::Value);
    pub fn is_ready(&self, completed_tasks: &[TaskId]) -> bool;
}
```

### TaskBuilder

```rust
impl TaskBuilder {
    pub fn name(self, name: impl Into<String>) -> Self;
    pub fn description(self, description: impl Into<String>) -> Self;
    pub fn input(self, input: impl Into<String>) -> Self;
    pub fn priority(self, priority: TaskPriority) -> Self;
    pub fn agent_id(self, agent_id: AgentId) -> Self;
    pub fn context_variable(self, key: impl Into<String>, value: serde_json::Value) -> Self;
    pub fn deadline(self, deadline: DateTime<Utc>) -> Self;
    pub fn max_execution_time_ms(self, ms: u64) -> Self;
    pub fn dependency(self, task_id: TaskId) -> Self;
    pub fn build(self) -> Result<Task>;
}
```

## Workflow Orchestration

### AgentOrchestrator

```rust
impl AgentOrchestrator {
    pub fn new() -> Self;
    pub fn with_config(config: OrchestratorConfig) -> Self;
    
    pub fn register_agent(&mut self, agent: Arc<dyn Agent>) -> Result<()>;
    pub fn unregister_agent(&mut self, agent_id: &AgentId) -> Result<()>;
    pub fn get_agent(&self, agent_id: &AgentId) -> Option<&Arc<dyn Agent>>;
    pub fn list_agents(&self) -> Vec<&Arc<dyn Agent>>;
    
    pub async fn execute_workflow(&mut self, workflow: Workflow) -> Result<WorkflowResult>;
    pub fn cancel_workflow(&mut self, workflow_id: &WorkflowId) -> Result<()>;
    pub fn active_workflows(&self) -> Vec<&Workflow>;
    pub fn stats(&self) -> &OrchestratorStats;
}
```

### Workflow

```rust
impl Workflow {
    pub fn new(name: impl Into<String>) -> Self;
    pub fn builder() -> WorkflowBuilder;
    
    pub fn id(&self) -> WorkflowId;
    pub fn name(&self) -> &str;
    pub fn steps(&self) -> &[WorkflowStep];
    pub fn context(&self) -> &WorkflowContext;
    pub fn status(&self) -> &WorkflowStatus;
    
    pub fn add_step(&mut self, step: WorkflowStep);
    pub fn get_step(&self, step_id: &str) -> Option<&WorkflowStep>;
    pub fn remove_step(&mut self, step_id: &str) -> Result<()>;
    pub fn get_ready_steps(&self, completed_steps: &[String]) -> Option<Vec<&WorkflowStep>>;
    pub fn execution_order(&self) -> Result<Vec<String>>;
    pub fn validate(&self) -> Result<()>;
}
```

### WorkflowBuilder

```rust
impl WorkflowBuilder {
    pub fn new() -> Self;
    pub fn name(self, name: impl Into<String>) -> Self;
    pub fn description(self, description: impl Into<String>) -> Self;
    pub fn step(self, step: WorkflowStep) -> Self;
    pub fn steps(self, steps: Vec<WorkflowStep>) -> Self;
    pub fn variable(self, key: impl Into<String>, value: serde_json::Value) -> Self;
    pub fn timeout_ms(self, timeout_ms: u64) -> Self;
    pub fn build(self) -> Result<Workflow>;
}
```

## MCP Integration

### McpService

```rust
impl McpService {
    pub fn new() -> Self;
    pub fn with_config(config: McpServiceConfig) -> Self;
    
    pub async fn add_client(&mut self, name: impl Into<String>, client: McpClient) -> Result<()>;
    pub fn add_server(&mut self, name: impl Into<String>, server: McpServer) -> Result<()>;
    pub fn remove_client(&mut self, name: &str) -> Result<()>;
    pub fn remove_server(&mut self, name: &str) -> Result<()>;
    
    pub fn get_client(&self, name: &str) -> Option<&Arc<McpClient>>;
    pub fn get_server(&self, name: &str) -> Option<&McpServer>;
    pub fn tool_registry(&self) -> &McpToolRegistry;
    
    pub async fn start_all_servers(&mut self) -> Result<()>;
    pub async fn stop_all_servers(&mut self) -> Result<()>;
    pub fn status(&self) -> McpServiceStatus;
}
```

### McpClient

```rust
impl McpClient {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self;
    
    pub async fn connect(&mut self, server_url: &str) -> Result<()>;
    pub async fn disconnect(&mut self) -> Result<()>;
    pub fn is_connected(&self) -> bool;
    
    pub async fn list_tools(&self) -> Result<Vec<RmcpTool>>;
    pub async fn call_tool(&self, tool_name: &str, arguments: serde_json::Value) -> Result<McpToolExecutionResult>;
    
    pub fn client_info(&self) -> &ClientInfo;
    pub fn status(&self) -> McpClientStatus;
}
```

### McpServer

```rust
impl McpServer {
    pub fn new(name: impl Into<String>, version: impl Into<String>, tool_registry: Arc<ToolRegistry>) -> Self;
    
    pub async fn start(&mut self, bind_address: &str) -> Result<()>;
    pub async fn stop(&mut self) -> Result<()>;
    pub fn is_running(&self) -> bool;
    
    pub fn add_tool(&mut self, tool: Arc<dyn Tool>) -> Result<()>;
    pub fn remove_tool(&mut self, tool_name: &str) -> Result<()>;
    pub fn available_tools(&self) -> Vec<String>;
    pub fn status(&self) -> McpServerStatus;
}
```

## Built-in Tools

### EchoTool

```rust
impl EchoTool {
    pub fn new() -> Self;
}
```

### FileTool

```rust
impl FileTool {
    pub fn new() -> Self;                                    // Read-only
    pub fn with_write_access() -> Self;                      // Read/write
    pub fn with_base_directory(base_dir: impl Into<String>) -> Self;  // Restricted
    pub fn with_write_access_and_base_directory(base_dir: impl Into<String>) -> Self;
}
```

### HttpTool

```rust
impl HttpTool {
    pub fn new() -> Self;                           // Default settings
    pub fn with_local_access() -> Self;             // Allow localhost
    pub fn with_timeout(timeout_secs: u64) -> Self; // Custom timeout
    pub fn with_max_response_size(self, size: usize) -> Self;
}
```

### SearchTool

```rust
impl SearchTool {
    pub fn new() -> Self;
    pub fn with_limits(default_top_k: usize, max_top_k: usize) -> Self;
    pub fn with_retriever(self, retriever: Box<dyn std::any::Any + Send + Sync>) -> Self;
}
```

## Error Handling

### AgentError

```rust
pub enum AgentError {
    Core(CheungfunError),
    Configuration { message: String },
    Execution { message: String },
    Tool { tool_name: String, message: String },
    Mcp { message: String },
    Task { task_id: String, message: String },
    Orchestration { message: String },
    Workflow { workflow_id: String, message: String },
    Communication { message: String },
    ResourceAccess { resource: String, message: String },
    Timeout { operation: String, timeout_ms: u64 },
    Authentication { message: String },
    Validation { field: String, message: String },
    // ... other variants
}

impl AgentError {
    pub fn is_retryable(&self) -> bool;
    pub fn category(&self) -> &'static str;
    
    // Convenience constructors
    pub fn configuration(message: impl Into<String>) -> Self;
    pub fn execution(message: impl Into<String>) -> Self;
    pub fn tool(tool_name: impl Into<String>, message: impl Into<String>) -> Self;
    // ... other constructors
}
```

## Configuration Types

### AgentConfig

```rust
pub struct AgentConfig {
    pub name: String,
    pub description: Option<String>,
    pub instructions: Option<String>,
    pub capabilities: AgentCapabilities,
    pub max_execution_time_ms: Option<u64>,
    pub max_tool_calls: Option<usize>,
    pub verbose: bool,
    pub custom_config: HashMap<String, serde_json::Value>,
}
```

### AgentCapabilities

```rust
pub struct AgentCapabilities {
    pub supports_tools: bool,
    pub supports_streaming: bool,
    pub supports_conversation: bool,
    pub supports_files: bool,
    pub supports_web: bool,
    pub supports_code_execution: bool,
    pub max_context_length: Option<usize>,
    pub supported_input_formats: Vec<String>,
    pub supported_output_formats: Vec<String>,
    pub custom_capabilities: HashMap<String, serde_json::Value>,
}
```

This API documentation covers the main interfaces and types in the Cheungfun Agents framework. For more detailed examples and usage patterns, see the examples directory and README.
