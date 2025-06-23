//! File operations tool for reading and writing files.

use crate::{
    error::{AgentError, Result},
    tool::{Tool, ToolContext, ToolResult, create_simple_schema, string_param},
    types::ToolSchema,
};
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;
use tokio::fs;

/// File tool for file system operations
#[derive(Debug, Clone)]
pub struct FileTool {
    name: String,
    /// Whether to allow dangerous operations (write, delete)
    allow_dangerous: bool,
    /// Base directory to restrict operations to (for security)
    base_directory: Option<String>,
}

impl FileTool {
    /// Create a new file tool with default settings (read-only)
    pub fn new() -> Self {
        Self {
            name: "file".to_string(),
            allow_dangerous: false,
            base_directory: None,
        }
    }

    /// Create a file tool that allows write operations
    pub fn with_write_access() -> Self {
        Self {
            name: "file".to_string(),
            allow_dangerous: true,
            base_directory: None,
        }
    }

    /// Create a file tool restricted to a base directory
    pub fn with_base_directory(base_dir: impl Into<String>) -> Self {
        Self {
            name: "file".to_string(),
            allow_dangerous: false,
            base_directory: Some(base_dir.into()),
        }
    }

    /// Create a file tool with write access and base directory restriction
    pub fn with_write_access_and_base_directory(base_dir: impl Into<String>) -> Self {
        Self {
            name: "file".to_string(),
            allow_dangerous: true,
            base_directory: Some(base_dir.into()),
        }
    }

    /// Validate and resolve file path
    fn resolve_path(&self, path: &str) -> Result<std::path::PathBuf> {
        let path = std::path::Path::new(path);

        // Check for path traversal attempts
        if path
            .components()
            .any(|comp| matches!(comp, std::path::Component::ParentDir))
        {
            return Err(AgentError::tool(
                &self.name,
                "Path traversal not allowed (..)",
            ));
        }

        let resolved = if let Some(base) = &self.base_directory {
            std::path::Path::new(base).join(path)
        } else {
            path.to_path_buf()
        };

        // Canonicalize to prevent symlink attacks
        match resolved.canonicalize() {
            Ok(canonical) => {
                // If we have a base directory, ensure the canonical path is within it
                if let Some(base) = &self.base_directory {
                    let base_canonical =
                        std::path::Path::new(base).canonicalize().map_err(|e| {
                            AgentError::tool(&self.name, format!("Invalid base directory: {e}"))
                        })?;

                    if !canonical.starts_with(base_canonical) {
                        return Err(AgentError::tool(
                            &self.name,
                            "Path outside allowed directory",
                        ));
                    }
                }
                Ok(canonical)
            }
            Err(_) => {
                // File doesn't exist, but we can still validate the parent directory
                if let Some(parent) = resolved.parent() {
                    if parent.exists() {
                        Ok(resolved)
                    } else {
                        Err(AgentError::tool(
                            &self.name,
                            "Parent directory does not exist",
                        ))
                    }
                } else {
                    Ok(resolved)
                }
            }
        }
    }
}

impl Default for FileTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for FileTool {
    fn schema(&self) -> ToolSchema {
        let mut properties = HashMap::new();

        properties.insert(
            "operation".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "File operation to perform",
                "enum": if self.allow_dangerous {
                    vec!["read", "write", "append", "delete", "list", "exists"]
                } else {
                    vec!["read", "list", "exists"]
                }
            }),
        );

        let (path_schema, _) = string_param("File or directory path", true);
        properties.insert("path".to_string(), path_schema);

        properties.insert(
            "content".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Content to write (for write/append operations)"
            }),
        );

        ToolSchema {
            name: self.name.clone(),
            description: format!(
                "Perform file system operations. {}",
                if self.allow_dangerous {
                    "Supports read, write, append, delete, list, and exists operations."
                } else {
                    "Read-only mode: supports read, list, and exists operations only."
                }
            ),
            input_schema: create_simple_schema(
                properties,
                vec!["operation".to_string(), "path".to_string()],
            ),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "Operation result"
                    },
                    "operation": {
                        "type": "string",
                        "description": "The operation performed"
                    },
                    "path": {
                        "type": "string",
                        "description": "The file path"
                    }
                }
            })),
            dangerous: self.allow_dangerous,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert(
                    "allow_dangerous".to_string(),
                    serde_json::json!(self.allow_dangerous),
                );
                if let Some(base) = &self.base_directory {
                    meta.insert("base_directory".to_string(), serde_json::json!(base));
                }
                meta
            },
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult> {
        #[derive(Deserialize)]
        struct FileArgs {
            operation: String,
            path: String,
            content: Option<String>,
        }

        let args: FileArgs = serde_json::from_value(arguments)
            .map_err(|e| AgentError::tool(&self.name, format!("Invalid arguments: {e}")))?;

        let resolved_path = self.resolve_path(&args.path)?;

        match args.operation.as_str() {
            "read" => self.read_file(&resolved_path).await,
            "write" => {
                if !self.allow_dangerous {
                    return Ok(ToolResult::error(
                        "Write operations not allowed in read-only mode",
                    ));
                }
                let content = args.content.unwrap_or_default();
                self.write_file(&resolved_path, &content).await
            }
            "append" => {
                if !self.allow_dangerous {
                    return Ok(ToolResult::error(
                        "Append operations not allowed in read-only mode",
                    ));
                }
                let content = args.content.unwrap_or_default();
                self.append_file(&resolved_path, &content).await
            }
            "delete" => {
                if !self.allow_dangerous {
                    return Ok(ToolResult::error(
                        "Delete operations not allowed in read-only mode",
                    ));
                }
                self.delete_file(&resolved_path).await
            }
            "list" => self.list_directory(&resolved_path).await,
            "exists" => self.check_exists(&resolved_path).await,
            _ => Ok(ToolResult::error(format!(
                "Unknown operation: {}",
                args.operation
            ))),
        }
    }

    fn capabilities(&self) -> Vec<String> {
        let mut caps = vec!["file_system".to_string(), "read".to_string()];
        if self.allow_dangerous {
            caps.extend(vec!["write".to_string(), "delete".to_string()]);
        }
        caps
    }

    fn is_dangerous(&self) -> bool {
        self.allow_dangerous
    }
}

impl FileTool {
    async fn read_file(&self, path: &std::path::Path) -> Result<ToolResult> {
        match fs::read_to_string(path).await {
            Ok(content) => Ok(ToolResult::success(content)
                .with_metadata("operation".to_string(), serde_json::json!("read"))
                .with_metadata(
                    "path".to_string(),
                    serde_json::json!(path.display().to_string()),
                )),
            Err(e) => Ok(ToolResult::error(format!("Failed to read file: {e}"))),
        }
    }

    async fn write_file(&self, path: &std::path::Path, content: &str) -> Result<ToolResult> {
        match fs::write(path, content).await {
            Ok(()) => Ok(ToolResult::success(format!(
                "Successfully wrote {} bytes to file",
                content.len()
            ))
            .with_metadata("operation".to_string(), serde_json::json!("write"))
            .with_metadata(
                "path".to_string(),
                serde_json::json!(path.display().to_string()),
            )
            .with_metadata(
                "bytes_written".to_string(),
                serde_json::json!(content.len()),
            )),
            Err(e) => Ok(ToolResult::error(format!("Failed to write file: {e}"))),
        }
    }

    async fn append_file(&self, path: &std::path::Path, content: &str) -> Result<ToolResult> {
        use tokio::io::AsyncWriteExt;

        match fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
        {
            Ok(mut file) => match file.write_all(content.as_bytes()).await {
                Ok(()) => Ok(ToolResult::success(format!(
                    "Successfully appended {} bytes to file",
                    content.len()
                ))
                .with_metadata("operation".to_string(), serde_json::json!("append"))
                .with_metadata(
                    "path".to_string(),
                    serde_json::json!(path.display().to_string()),
                )
                .with_metadata(
                    "bytes_appended".to_string(),
                    serde_json::json!(content.len()),
                )),
                Err(e) => Ok(ToolResult::error(format!("Failed to append to file: {e}"))),
            },
            Err(e) => Ok(ToolResult::error(format!(
                "Failed to open file for append: {e}"
            ))),
        }
    }

    async fn delete_file(&self, path: &std::path::Path) -> Result<ToolResult> {
        if path.is_dir() {
            match fs::remove_dir_all(path).await {
                Ok(()) => Ok(ToolResult::success("Directory deleted successfully")
                    .with_metadata("operation".to_string(), serde_json::json!("delete"))
                    .with_metadata(
                        "path".to_string(),
                        serde_json::json!(path.display().to_string()),
                    )
                    .with_metadata("type".to_string(), serde_json::json!("directory"))),
                Err(e) => Ok(ToolResult::error(format!(
                    "Failed to delete directory: {e}"
                ))),
            }
        } else {
            match fs::remove_file(path).await {
                Ok(()) => Ok(ToolResult::success("File deleted successfully")
                    .with_metadata("operation".to_string(), serde_json::json!("delete"))
                    .with_metadata(
                        "path".to_string(),
                        serde_json::json!(path.display().to_string()),
                    )
                    .with_metadata("type".to_string(), serde_json::json!("file"))),
                Err(e) => Ok(ToolResult::error(format!("Failed to delete file: {e}"))),
            }
        }
    }

    async fn list_directory(&self, path: &std::path::Path) -> Result<ToolResult> {
        if !path.is_dir() {
            return Ok(ToolResult::error("Path is not a directory"));
        }

        match fs::read_dir(path).await {
            Ok(mut entries) => {
                let mut files = Vec::new();
                let mut dirs = Vec::new();

                while let Ok(Some(entry)) = entries.next_entry().await {
                    let entry_path = entry.path();
                    let name = entry_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("?")
                        .to_string();

                    if entry_path.is_dir() {
                        dirs.push(name);
                    } else {
                        files.push(name);
                    }
                }

                dirs.sort();
                files.sort();

                let content = format!(
                    "Directories ({}): {}\nFiles ({}): {}",
                    dirs.len(),
                    if dirs.is_empty() {
                        "none".to_string()
                    } else {
                        dirs.join(", ")
                    },
                    files.len(),
                    if files.is_empty() {
                        "none".to_string()
                    } else {
                        files.join(", ")
                    }
                );

                Ok(ToolResult::success(content)
                    .with_metadata("operation".to_string(), serde_json::json!("list"))
                    .with_metadata(
                        "path".to_string(),
                        serde_json::json!(path.display().to_string()),
                    )
                    .with_metadata("directories".to_string(), serde_json::json!(dirs))
                    .with_metadata("files".to_string(), serde_json::json!(files)))
            }
            Err(e) => Ok(ToolResult::error(format!("Failed to list directory: {e}"))),
        }
    }

    async fn check_exists(&self, path: &std::path::Path) -> Result<ToolResult> {
        let exists = path.exists();
        let file_type = if exists {
            if path.is_dir() {
                "directory"
            } else if path.is_file() {
                "file"
            } else {
                "other"
            }
        } else {
            "none"
        };

        let content = format!("Path exists: {}, type: {}", exists, file_type);

        Ok(ToolResult::success(content)
            .with_metadata("operation".to_string(), serde_json::json!("exists"))
            .with_metadata(
                "path".to_string(),
                serde_json::json!(path.display().to_string()),
            )
            .with_metadata("exists".to_string(), serde_json::json!(exists))
            .with_metadata("type".to_string(), serde_json::json!(file_type)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_file_tool_read_only() {
        let tool = FileTool::new();
        let context = ToolContext::new();

        // Test that write operations are rejected
        let args = serde_json::json!({
            "operation": "write",
            "path": "/tmp/test.txt",
            "content": "test"
        });

        let result = tool.execute(args, &context).await.unwrap();
        assert!(!result.success);
        assert!(result.error_message().unwrap().contains("not allowed"));
    }

    #[tokio::test]
    async fn test_file_tool_with_write_access() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");

        let tool = FileTool::with_write_access();
        let context = ToolContext::new();

        // Test write
        let args = serde_json::json!({
            "operation": "write",
            "path": file_path.display().to_string(),
            "content": "Hello, World!"
        });

        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);

        // Test read
        let args = serde_json::json!({
            "operation": "read",
            "path": file_path.display().to_string()
        });

        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);
        assert_eq!(result.content, "Hello, World!");
    }

    #[test]
    fn test_path_traversal_protection() {
        let tool = FileTool::new();
        let result = tool.resolve_path("../../../etc/passwd");
        assert!(result.is_err());
    }
}
