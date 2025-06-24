//! Tool registry for managing and discovering tools.

use crate::{
    error::{AgentError, Result},
    tool::{Tool, ToolContext, ToolResult},
    types::ToolSchema,
};
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, info, warn};

/// Tool registry for managing available tools
#[derive(Debug, Default)]
pub struct ToolRegistry {
    /// Registered tools by name
    tools: HashMap<String, Arc<dyn Tool>>,
    /// Tool categories for organization
    categories: HashMap<String, Vec<String>>,
    /// Tool aliases for alternative names
    aliases: HashMap<String, String>,
}

impl ToolRegistry {
    /// Create a new empty tool registry
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool in the registry
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<()> {
        let name = tool.name().to_string();

        if self.tools.contains_key(&name) {
            return Err(AgentError::configuration(format!(
                "Tool '{name}' is already registered"
            )));
        }

        info!("Registering tool: {}", name);
        self.tools.insert(name.clone(), tool);
        debug!("Tool '{}' registered successfully", name);

        Ok(())
    }

    /// Register a tool with a specific category
    pub fn register_with_category(
        &mut self,
        tool: Arc<dyn Tool>,
        category: impl Into<String>,
    ) -> Result<()> {
        let name = tool.name().to_string();
        let category = category.into();

        self.register(tool)?;

        self.categories
            .entry(category.clone())
            .or_default()
            .push(name.clone());

        debug!("Tool '{}' added to category '{}'", name, category);
        Ok(())
    }

    /// Register a tool alias
    pub fn register_alias(
        &mut self,
        alias: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> Result<()> {
        let alias = alias.into();
        let tool_name = tool_name.into();

        if !self.tools.contains_key(&tool_name) {
            return Err(AgentError::configuration(format!(
                "Cannot create alias '{alias}' for non-existent tool '{tool_name}'"
            )));
        }

        if self.aliases.contains_key(&alias) {
            return Err(AgentError::configuration(format!(
                "Alias '{alias}' is already registered"
            )));
        }

        self.aliases.insert(alias.clone(), tool_name.clone());
        debug!("Alias '{}' registered for tool '{}'", alias, tool_name);
        Ok(())
    }

    /// Get a tool by name or alias
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        // Try direct lookup first
        if let Some(tool) = self.tools.get(name) {
            return Some(Arc::clone(tool));
        }

        // Try alias lookup
        if let Some(real_name) = self.aliases.get(name) {
            return self.tools.get(real_name).map(Arc::clone);
        }

        None
    }

    /// Check if a tool exists (by name or alias)
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name) || self.aliases.contains_key(name)
    }

    /// Get all registered tool names
    #[must_use]
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Get all tool schemas
    #[must_use]
    pub fn schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|tool| tool.schema()).collect()
    }

    /// Get tools by category
    #[must_use]
    pub fn tools_by_category(&self, category: &str) -> Vec<Arc<dyn Tool>> {
        if let Some(tool_names) = self.categories.get(category) {
            tool_names
                .iter()
                .filter_map(|name| self.tools.get(name).map(Arc::clone))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get all categories
    #[must_use]
    pub fn categories(&self) -> Vec<String> {
        self.categories.keys().cloned().collect()
    }

    /// Execute a tool by name
    pub async fn execute(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
        context: &ToolContext,
    ) -> Result<ToolResult> {
        let tool = self
            .get(tool_name)
            .ok_or_else(|| AgentError::tool(tool_name, "Tool not found in registry"))?;

        debug!(
            "Executing tool '{}' with arguments: {:?}",
            tool_name, arguments
        );

        // Validate arguments
        tool.validate_arguments(&arguments)?;

        // Execute the tool
        let result = tool.execute(arguments, context).await;

        match &result {
            Ok(tool_result) => {
                if tool_result.success {
                    debug!("Tool '{}' executed successfully", tool_name);
                } else {
                    warn!(
                        "Tool '{}' execution failed: {:?}",
                        tool_name, tool_result.error
                    );
                }
            }
            Err(e) => {
                warn!("Tool '{}' execution error: {}", tool_name, e);
            }
        }

        result
    }

    /// Remove a tool from the registry
    pub fn unregister(&mut self, tool_name: &str) -> Result<()> {
        if !self.tools.contains_key(tool_name) {
            return Err(AgentError::configuration(format!(
                "Tool '{tool_name}' is not registered"
            )));
        }

        self.tools.remove(tool_name);

        // Remove from categories
        for tools in self.categories.values_mut() {
            tools.retain(|name| name != tool_name);
        }

        // Remove aliases
        self.aliases.retain(|_, target| target != tool_name);

        info!("Tool '{}' unregistered", tool_name);
        Ok(())
    }

    /// Clear all tools from the registry
    pub fn clear(&mut self) {
        let count = self.tools.len();
        self.tools.clear();
        self.categories.clear();
        self.aliases.clear();
        info!("Cleared {} tools from registry", count);
    }

    /// Get registry statistics
    #[must_use]
    pub fn stats(&self) -> RegistryStats {
        RegistryStats {
            total_tools: self.tools.len(),
            total_categories: self.categories.len(),
            total_aliases: self.aliases.len(),
            tools_by_category: self
                .categories
                .iter()
                .map(|(cat, tools)| (cat.clone(), tools.len()))
                .collect(),
        }
    }

    /// Find tools by capability
    pub fn find_by_capability(&self, capability: &str) -> Vec<Arc<dyn Tool>> {
        self.tools
            .values()
            .filter(|tool| tool.capabilities().contains(&capability.to_string()))
            .map(Arc::clone)
            .collect()
    }

    /// Find dangerous tools
    pub fn dangerous_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools
            .values()
            .filter(|tool| tool.is_dangerous())
            .map(Arc::clone)
            .collect()
    }

    /// Validate all tools in the registry
    pub fn validate_all(&self) -> Result<()> {
        for (name, tool) in &self.tools {
            let schema = tool.schema();

            // Basic schema validation
            if schema.name.is_empty() {
                return Err(AgentError::validation(
                    "tool_schema",
                    format!("Tool '{name}' has empty name in schema"),
                ));
            }

            if schema.description.is_empty() {
                return Err(AgentError::validation(
                    "tool_schema",
                    format!("Tool '{name}' has empty description in schema"),
                ));
            }

            if schema.name != *name {
                return Err(AgentError::validation(
                    "tool_schema",
                    format!(
                        "Tool '{}' schema name '{}' doesn't match registry name",
                        name, schema.name
                    ),
                ));
            }
        }

        debug!("All {} tools validated successfully", self.tools.len());
        Ok(())
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStats {
    /// Total number of registered tools
    pub total_tools: usize,
    /// Total number of categories
    pub total_categories: usize,
    /// Total number of aliases
    pub total_aliases: usize,
    /// Tools count by category
    pub tools_by_category: HashMap<String, usize>,
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
            categories: self.categories.clone(),
            aliases: self.aliases.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::builtin::EchoTool;

    #[tokio::test]
    async fn test_registry_basic_operations() {
        let mut registry = ToolRegistry::new();
        let echo_tool = Arc::new(EchoTool::new());

        // Register tool
        registry.register(echo_tool as Arc<dyn Tool>).unwrap();
        assert!(registry.contains("echo"));
        assert_eq!(registry.tool_names().len(), 1);

        // Get tool
        let retrieved = registry.get("echo").unwrap();
        assert_eq!(retrieved.name(), "echo");

        // Execute tool
        let context = ToolContext::new();
        let result = registry
            .execute("echo", serde_json::json!({"message": "test"}), &context)
            .await
            .unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_registry_categories() {
        let mut registry = ToolRegistry::new();
        let echo_tool = Arc::new(EchoTool::new());

        registry
            .register_with_category(echo_tool, "utility")
            .unwrap();

        let utility_tools = registry.tools_by_category("utility");
        assert_eq!(utility_tools.len(), 1);
        assert_eq!(utility_tools[0].name(), "echo");
    }

    #[test]
    fn test_registry_aliases() {
        let mut registry = ToolRegistry::new();
        let echo_tool = Arc::new(EchoTool::new());

        registry.register(echo_tool).unwrap();
        registry.register_alias("repeat", "echo").unwrap();

        assert!(registry.contains("repeat"));
        let tool = registry.get("repeat").unwrap();
        assert_eq!(tool.name(), "echo");
    }
}
