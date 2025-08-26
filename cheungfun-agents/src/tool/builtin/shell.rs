//! Shell Command Tool - Execute shell/system commands
//!
//! This tool allows agents to execute shell commands and system operations.
//! IMPORTANT: This is a powerful tool that should be used with caution.

use crate::{
    error::{AgentError, Result},
    tool::{create_simple_schema, Tool, ToolContext, ToolResult},
    types::ToolSchema,
};
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;
use std::process::Stdio;
use tokio::process::Command;

/// Shell command execution tool
#[derive(Debug, Clone)]
pub struct ShellTool {
    name: String,
    allowed_commands: Option<Vec<String>>,
    timeout_seconds: u64,
}

impl ShellTool {
    /// Create a new shell tool
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "shell".to_string(),
            allowed_commands: None,
            timeout_seconds: 30,
        }
    }

    /// Create shell tool with allowed commands whitelist
    #[must_use]
    pub fn with_allowed_commands(commands: Vec<String>) -> Self {
        Self {
            name: "shell".to_string(),
            allowed_commands: Some(commands),
            timeout_seconds: 30,
        }
    }

    /// Set timeout for command execution
    #[must_use]
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }
}

impl Default for ShellTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn schema(&self) -> ToolSchema {
        let mut properties = HashMap::new();
        properties.insert(
            "command".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Shell command to execute"
            }),
        );
        properties.insert(
            "args".to_string(),
            serde_json::json!({
                "type": "array",
                "items": {"type": "string"},
                "description": "Command arguments (optional)",
                "default": []
            }),
        );
        properties.insert(
            "working_directory".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Working directory for command execution (optional)"
            }),
        );
        properties.insert(
            "timeout".to_string(),
            serde_json::json!({
                "type": "number",
                "description": "Timeout in seconds (optional)",
                "default": 30
            }),
        );

        ToolSchema {
            name: self.name.clone(),
            description: "Execute shell commands and system operations. Use with caution as this can modify the system.".to_string(),
            input_schema: create_simple_schema(properties, vec!["command".to_string()]),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "stdout": {
                        "type": "string",
                        "description": "Standard output from the command"
                    },
                    "stderr": {
                        "type": "string",
                        "description": "Standard error from the command"
                    },
                    "exit_code": {
                        "type": "number",
                        "description": "Command exit code"
                    },
                    "command": {
                        "type": "string",
                        "description": "The executed command"
                    },
                    "execution_time_ms": {
                        "type": "number",
                        "description": "Command execution time in milliseconds"
                    }
                }
            })),
            dangerous: true, // Mark as dangerous since it executes system commands
            metadata: HashMap::new(),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult> {
        #[derive(Deserialize)]
        struct ShellArgs {
            command: String,
            #[serde(default)]
            args: Vec<String>,
            working_directory: Option<String>,
            #[serde(default = "default_timeout")]
            timeout: u64,
        }

        fn default_timeout() -> u64 {
            30
        }

        let args: ShellArgs = serde_json::from_value(arguments)
            .map_err(|e| AgentError::tool(&self.name, format!("Invalid arguments: {e}")))?;

        // Check if command is allowed (if whitelist is configured)
        if let Some(allowed) = &self.allowed_commands {
            if !allowed.contains(&args.command) {
                return Ok(ToolResult::error(format!(
                    "Command '{}' is not allowed. Allowed commands: {:?}",
                    args.command, allowed
                )));
            }
        }

        // Security check - prevent dangerous commands
        if self.is_dangerous_command(&args.command) {
            return Ok(ToolResult::error(format!(
                "Command '{}' is potentially dangerous and blocked for safety",
                args.command
            )));
        }

        let start_time = std::time::Instant::now();

        // Execute command
        let result = self
            .execute_command(
                &args.command,
                &args.args,
                args.working_directory.as_deref(),
                args.timeout,
            )
            .await;

        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        match result {
            Ok((stdout, stderr, exit_code)) => {
                let success = exit_code == 0;
                let content = if success {
                    if stdout.trim().is_empty() {
                        "Command executed successfully (no output)".to_string()
                    } else {
                        format!("Command output:\n{stdout}")
                    }
                } else {
                    format!("Command failed (exit code: {exit_code}):\nstdout: {stdout}\nstderr: {stderr}")
                };

                let full_command = if args.args.is_empty() {
                    args.command.clone()
                } else {
                    format!("{} {}", args.command, args.args.join(" "))
                };

                let result = if success {
                    ToolResult::success(content)
                } else {
                    ToolResult::error(content)
                }
                .with_metadata("stdout".to_string(), serde_json::json!(stdout))
                .with_metadata("stderr".to_string(), serde_json::json!(stderr))
                .with_metadata("exit_code".to_string(), serde_json::json!(exit_code))
                .with_metadata("command".to_string(), serde_json::json!(full_command))
                .with_metadata(
                    "execution_time_ms".to_string(),
                    serde_json::json!(execution_time_ms),
                );

                Ok(result)
            }
            Err(e) => Ok(ToolResult::error(format!("Command execution failed: {e}"))
                .with_metadata("command".to_string(), serde_json::json!(args.command))
                .with_metadata(
                    "execution_time_ms".to_string(),
                    serde_json::json!(execution_time_ms),
                )),
        }
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "shell".to_string(),
            "system".to_string(),
            "command".to_string(),
            "execution".to_string(),
        ]
    }
}

impl ShellTool {
    /// Execute shell command with timeout
    async fn execute_command(
        &self,
        command: &str,
        args: &[String],
        working_directory: Option<&str>,
        timeout: u64,
    ) -> Result<(String, String, i32)> {
        let mut cmd = Command::new(command);
        cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());

        if let Some(dir) = working_directory {
            cmd.current_dir(dir);
        }

        let child = cmd
            .spawn()
            .map_err(|e| AgentError::tool("shell", format!("Failed to spawn command: {e}")))?;

        // Wait for command with timeout
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(timeout),
            child.wait_with_output(),
        )
        .await
        .map_err(|_| AgentError::tool("shell", "Command timed out"))?
        .map_err(|e| AgentError::tool("shell", format!("Command execution failed: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        Ok((stdout, stderr, exit_code))
    }

    /// Check if command is potentially dangerous
    fn is_dangerous_command(&self, command: &str) -> bool {
        let dangerous_commands = [
            "rm",
            "del",
            "format",
            "mkfs",
            "dd",
            "fdisk",
            "parted",
            "sudo",
            "su",
            "passwd",
            "chmod 777",
            "chown",
            "shutdown",
            "reboot",
            "halt",
            "kill",
            "killall",
            "pkill",
            "> /dev/",
            "cat /dev/zero",
            "curl",
            "wget",
            "nc",
            "netcat",
            "python -c",
            "perl -e",
            "ruby -e",
            "eval",
            "exec",
        ];

        dangerous_commands.iter().any(|&dangerous| {
            command.to_lowercase().contains(dangerous) || command.starts_with(dangerous)
        })
    }

    /// Add allowed command to whitelist
    pub fn add_allowed_command(&mut self, command: String) {
        if let Some(ref mut allowed) = self.allowed_commands {
            if !allowed.contains(&command) {
                allowed.push(command);
            }
        } else {
            self.allowed_commands = Some(vec![command]);
        }
    }

    /// Set allowed commands whitelist
    pub fn set_allowed_commands(&mut self, commands: Vec<String>) {
        self.allowed_commands = Some(commands);
    }

    /// Clear allowed commands (allow all commands)
    pub fn clear_allowed_commands(&mut self) {
        self.allowed_commands = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_shell_tool_safe_command() {
        let tool = ShellTool::with_allowed_commands(vec!["echo".to_string(), "ls".to_string()]);
        let context = ToolContext::new();

        // Test safe command
        let args = serde_json::json!({
            "command": "echo",
            "args": ["Hello, World!"]
        });

        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);
        assert!(result.content.contains("Hello, World!"));

        // Check metadata
        assert!(result.metadata.contains_key("stdout"));
        assert!(result.metadata.contains_key("exit_code"));
        assert_eq!(
            result.metadata.get("exit_code").unwrap().as_i64().unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn test_shell_tool_dangerous_command() {
        let tool = ShellTool::new();
        let context = ToolContext::new();

        // Test dangerous command
        let args = serde_json::json!({
            "command": "rm",
            "args": ["-rf", "/"]
        });

        let result = tool.execute(args, &context).await.unwrap();
        assert!(!result.success);
        assert!(result.content.contains("dangerous"));
    }

    #[tokio::test]
    async fn test_shell_tool_not_allowed_command() {
        let tool = ShellTool::with_allowed_commands(vec!["echo".to_string()]);
        let context = ToolContext::new();

        // Test command not in whitelist
        let args = serde_json::json!({
            "command": "ls",
            "args": []
        });

        let result = tool.execute(args, &context).await.unwrap();
        assert!(!result.success);
        assert!(result.content.contains("not allowed"));
    }

    #[test]
    fn test_dangerous_command_detection() {
        let tool = ShellTool::new();

        assert!(tool.is_dangerous_command("rm -rf"));
        assert!(tool.is_dangerous_command("sudo something"));
        assert!(tool.is_dangerous_command("shutdown now"));
        assert!(!tool.is_dangerous_command("echo hello"));
        assert!(!tool.is_dangerous_command("ls -la"));
    }
}
