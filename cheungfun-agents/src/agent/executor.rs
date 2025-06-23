//! Agent execution engine with advanced features like retries, timeouts, and monitoring.

use crate::{
    agent::Agent,
    error::{AgentError, Result},
    task::{Task, TaskResult, TaskStatus},
    types::{AgentId, AgentResponse},
};
use std::{sync::Arc, time::Duration};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

/// Agent executor with advanced execution features
#[derive(Debug)]
pub struct AgentExecutor {
    /// The agent to execute
    agent: Arc<dyn Agent>,
    /// Execution configuration
    config: ExecutorConfig,
}

/// Executor configuration
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Initial retry delay in milliseconds
    pub initial_retry_delay_ms: u64,
    /// Retry delay multiplier (exponential backoff)
    pub retry_delay_multiplier: f64,
    /// Maximum retry delay in milliseconds
    pub max_retry_delay_ms: u64,
    /// Global execution timeout in milliseconds
    pub execution_timeout_ms: Option<u64>,
    /// Whether to enable detailed logging
    pub verbose_logging: bool,
    /// Whether to collect execution metrics
    pub collect_metrics: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_retry_delay_ms: 1000,
            retry_delay_multiplier: 2.0,
            max_retry_delay_ms: 30_000,
            execution_timeout_ms: Some(300_000), // 5 minutes
            verbose_logging: false,
            collect_metrics: true,
        }
    }
}

/// Execution context for tracking execution state
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Current attempt number (1-based)
    pub attempt: usize,
    /// Total attempts allowed
    pub max_attempts: usize,
    /// Execution start time
    pub start_time: std::time::Instant,
    /// Last error encountered
    pub last_error: Option<String>,
    /// Execution metrics
    pub metrics: ExecutionMetrics,
}

/// Execution metrics
#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    /// Total execution time in milliseconds
    pub total_time_ms: u64,
    /// Time spent in retries in milliseconds
    pub retry_time_ms: u64,
    /// Number of retries performed
    pub retries_performed: usize,
    /// Whether execution was successful
    pub success: bool,
    /// Final error message if failed
    pub error_message: Option<String>,
}

impl AgentExecutor {
    /// Create a new agent executor
    pub fn new(agent: Arc<dyn Agent>) -> Self {
        Self {
            agent,
            config: ExecutorConfig::default(),
        }
    }

    /// Create an agent executor with custom configuration
    pub fn with_config(agent: Arc<dyn Agent>, config: ExecutorConfig) -> Self {
        Self { agent, config }
    }

    /// Execute a task with retry logic and timeout
    pub async fn execute_task(&self, task: &Task) -> Result<TaskResult> {
        let start_time = std::time::Instant::now();
        let mut context = ExecutionContext {
            attempt: 1,
            max_attempts: self.config.max_retries + 1,
            start_time,
            last_error: None,
            metrics: ExecutionMetrics::default(),
        };

        if self.config.verbose_logging {
            info!(
                "Starting task execution: {} (agent: {})",
                task.id,
                self.agent.name()
            );
        }

        // Check if agent can handle the task
        if let Err(e) = self.agent.can_handle_task(task) {
            return Ok(TaskResult::failure(
                task.id,
                e.to_string(),
                chrono::Utc::now(),
                Some(self.agent.id()),
            ));
        }

        let mut last_error = None;

        for attempt in 1..=context.max_attempts {
            context.attempt = attempt;

            if self.config.verbose_logging && attempt > 1 {
                info!("Retry attempt {} for task {}", attempt, task.id);
            }

            // Execute with timeout
            let execution_result = if let Some(timeout_ms) = self.config.execution_timeout_ms {
                let timeout_duration = Duration::from_millis(timeout_ms);
                match timeout(timeout_duration, self.agent.execute(task)).await {
                    Ok(result) => result,
                    Err(_) => {
                        let error_msg = format!("Task execution timed out after {}ms", timeout_ms);
                        warn!("{}", error_msg);
                        Err(AgentError::timeout("task_execution", timeout_ms))
                    }
                }
            } else {
                self.agent.execute(task).await
            };

            match execution_result {
                Ok(response) => {
                    // Success!
                    let total_time = start_time.elapsed();
                    context.metrics.total_time_ms = total_time.as_millis() as u64;
                    context.metrics.retries_performed = attempt - 1;
                    context.metrics.success = true;

                    if self.config.verbose_logging {
                        info!(
                            "Task {} completed successfully in {}ms (attempt {})",
                            task.id, context.metrics.total_time_ms, attempt
                        );
                    }

                    return Ok(TaskResult::success(
                        task.id,
                        response.content,
                        chrono::Utc::now()
                            - chrono::Duration::milliseconds(context.metrics.total_time_ms as i64),
                        Some(self.agent.id()),
                    ));
                }
                Err(e) => {
                    let error_msg = e.to_string();
                    let is_retryable = e.is_retryable();

                    if self.config.verbose_logging {
                        warn!("Task {} failed on attempt {}: {}", task.id, attempt, e);
                    }

                    last_error = Some(e);
                    context.last_error = Some(error_msg);

                    // Check if error is retryable
                    if !is_retryable || attempt >= context.max_attempts {
                        break;
                    }

                    // Calculate retry delay
                    let delay_ms = self.calculate_retry_delay(attempt - 1);
                    if delay_ms > 0 {
                        debug!("Waiting {}ms before retry", delay_ms);
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        context.metrics.retry_time_ms += delay_ms;
                    }
                }
            }
        }

        // All attempts failed
        let total_time = start_time.elapsed();
        context.metrics.total_time_ms = total_time.as_millis() as u64;
        context.metrics.retries_performed = context.max_attempts - 1;
        context.metrics.success = false;
        context.metrics.error_message = context.last_error.clone();

        let error_msg = last_error
            .map(|e| e.to_string())
            .unwrap_or_else(|| "Unknown error".to_string());

        if self.config.verbose_logging {
            error!(
                "Task {} failed after {} attempts in {}ms: {}",
                task.id, context.max_attempts, context.metrics.total_time_ms, error_msg
            );
        }

        Ok(TaskResult::failure(
            task.id,
            error_msg,
            chrono::Utc::now()
                - chrono::Duration::milliseconds(context.metrics.total_time_ms as i64),
            Some(self.agent.id()),
        ))
    }

    /// Execute a task with streaming response
    pub async fn execute_task_streaming(
        &self,
        task: &Task,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String>> + Send + Unpin>> {
        // Check if agent supports streaming
        if !self.agent.supports_capability("streaming") {
            return Err(AgentError::execution(
                "Agent does not support streaming execution",
            ));
        }

        // Check if agent can handle the task
        self.agent.can_handle_task(task)?;

        if self.config.verbose_logging {
            info!(
                "Starting streaming task execution: {} (agent: {})",
                task.id,
                self.agent.name()
            );
        }

        // Execute with timeout if configured
        if let Some(timeout_ms) = self.config.execution_timeout_ms {
            let timeout_duration = Duration::from_millis(timeout_ms);
            match timeout(timeout_duration, self.agent.execute_streaming(task)).await {
                Ok(result) => result,
                Err(_) => Err(AgentError::timeout("streaming_execution", timeout_ms)),
            }
        } else {
            self.agent.execute_streaming(task).await
        }
    }

    /// Get the agent being executed
    pub fn agent(&self) -> &Arc<dyn Agent> {
        &self.agent
    }

    /// Get the executor configuration
    pub fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    /// Update executor configuration
    pub fn set_config(&mut self, config: ExecutorConfig) {
        self.config = config;
    }

    /// Calculate retry delay with exponential backoff
    fn calculate_retry_delay(&self, retry_count: usize) -> u64 {
        if retry_count == 0 {
            return 0;
        }

        let delay = self.config.initial_retry_delay_ms as f64
            * self
                .config
                .retry_delay_multiplier
                .powi(retry_count as i32 - 1);

        (delay as u64).min(self.config.max_retry_delay_ms)
    }

    /// Check agent health
    pub async fn health_check(&self) -> Result<crate::agent::AgentHealthStatus> {
        self.agent.health_check().await
    }
}

/// Builder for executor configuration
#[derive(Debug, Default)]
pub struct ExecutorConfigBuilder {
    config: ExecutorConfig,
}

impl ExecutorConfigBuilder {
    /// Create a new executor config builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum retries
    pub fn max_retries(mut self, retries: usize) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// Set initial retry delay
    pub fn initial_retry_delay_ms(mut self, delay_ms: u64) -> Self {
        self.config.initial_retry_delay_ms = delay_ms;
        self
    }

    /// Set retry delay multiplier
    pub fn retry_delay_multiplier(mut self, multiplier: f64) -> Self {
        self.config.retry_delay_multiplier = multiplier;
        self
    }

    /// Set maximum retry delay
    pub fn max_retry_delay_ms(mut self, delay_ms: u64) -> Self {
        self.config.max_retry_delay_ms = delay_ms;
        self
    }

    /// Set execution timeout
    pub fn execution_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.config.execution_timeout_ms = Some(timeout_ms);
        self
    }

    /// Disable execution timeout
    pub fn no_timeout(mut self) -> Self {
        self.config.execution_timeout_ms = None;
        self
    }

    /// Enable verbose logging
    pub fn verbose_logging(mut self) -> Self {
        self.config.verbose_logging = true;
        self
    }

    /// Enable metrics collection
    pub fn collect_metrics(mut self) -> Self {
        self.config.collect_metrics = true;
        self
    }

    /// Build the configuration
    pub fn build(self) -> ExecutorConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{agent::AgentBuilder, tool::builtin::EchoTool};

    #[tokio::test]
    async fn test_executor_basic() {
        let echo_tool = Arc::new(EchoTool::new());
        let agent = AgentBuilder::new()
            .name("test_agent")
            .tool(echo_tool)
            .build()
            .unwrap();

        let executor = AgentExecutor::new(agent);
        let task = Task::new("Test task");

        let result = executor.execute_task(&task).await.unwrap();
        assert_eq!(result.status, TaskStatus::Completed);
    }

    #[test]
    fn test_retry_delay_calculation() {
        let config = ExecutorConfig {
            initial_retry_delay_ms: 1000,
            retry_delay_multiplier: 2.0,
            max_retry_delay_ms: 10000,
            ..Default::default()
        };

        let executor = AgentExecutor::with_config(
            Arc::new(AgentBuilder::new().name("test").build().unwrap()),
            config,
        );

        assert_eq!(executor.calculate_retry_delay(0), 0);
        assert_eq!(executor.calculate_retry_delay(1), 1000);
        assert_eq!(executor.calculate_retry_delay(2), 2000);
        assert_eq!(executor.calculate_retry_delay(3), 4000);
        assert_eq!(executor.calculate_retry_delay(4), 8000);
        assert_eq!(executor.calculate_retry_delay(5), 10000); // Capped at max
    }

    #[test]
    fn test_executor_config_builder() {
        let config = ExecutorConfigBuilder::new()
            .max_retries(5)
            .initial_retry_delay_ms(500)
            .execution_timeout_ms(60000)
            .verbose_logging()
            .build();

        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_retry_delay_ms, 500);
        assert_eq!(config.execution_timeout_ms, Some(60000));
        assert!(config.verbose_logging);
    }
}
