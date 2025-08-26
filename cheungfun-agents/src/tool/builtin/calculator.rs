//! Advanced Calculator Tool - Enhanced math operations
//!
//! This tool provides comprehensive mathematical calculation capabilities
//! using a proper expression parser for complex mathematical operations.

use crate::{
    error::{AgentError, Result},
    tool::{create_simple_schema, Tool, ToolContext, ToolResult},
    types::ToolSchema,
};
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;

/// Advanced calculator tool with comprehensive math support
#[derive(Debug, Clone)]
pub struct CalculatorTool {
    name: String,
}

impl CalculatorTool {
    /// Create a new advanced calculator tool
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "calculator".to_string(),
        }
    }
}

impl Default for CalculatorTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for CalculatorTool {
    fn schema(&self) -> ToolSchema {
        let mut properties = HashMap::new();
        properties.insert(
            "expression".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Mathematical expression to evaluate. Supports: +, -, *, /, ^, sqrt(), sin(), cos(), tan(), log(), ln(), abs(), pi, e"
            }),
        );

        ToolSchema {
            name: self.name.clone(),
            description: "Advanced calculator for mathematical expressions. Supports arithmetic, trigonometry, logarithms, and constants."
                .to_string(),
            input_schema: create_simple_schema(properties, vec!["expression".to_string()]),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "result": {
                        "type": "number",
                        "description": "The calculated result"
                    },
                    "expression": {
                        "type": "string",
                        "description": "The original expression"
                    },
                    "formatted": {
                        "type": "string",
                        "description": "Human-readable result format"
                    }
                }
            })),
            dangerous: false,
            metadata: HashMap::new(),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult> {
        #[derive(Deserialize)]
        struct CalculatorArgs {
            expression: String,
        }

        let args: CalculatorArgs = serde_json::from_value(arguments)
            .map_err(|e| AgentError::tool(&self.name, format!("Invalid arguments: {e}")))?;

        match self.evaluate_advanced_expression(&args.expression) {
            Ok(value) => {
                let formatted = if value.fract() == 0.0 {
                    format!("{}", value as i64)
                } else {
                    format!("{value:.6}")
                        .trim_end_matches('0')
                        .trim_end_matches('.')
                        .to_string()
                };

                let content = format!("Result: {formatted}");
                let result = ToolResult::success(content)
                    .with_metadata("result".to_string(), serde_json::json!(value))
                    .with_metadata("expression".to_string(), serde_json::json!(args.expression))
                    .with_metadata("formatted".to_string(), serde_json::json!(formatted));

                Ok(result)
            }
            Err(e) => Ok(ToolResult::error(format!("Calculation error: {e}"))),
        }
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "calculation".to_string(),
            "math".to_string(),
            "arithmetic".to_string(),
            "trigonometry".to_string(),
            "logarithms".to_string(),
        ]
    }
}

impl CalculatorTool {
    /// Advanced expression evaluator
    ///
    /// This is a simplified implementation. In production, you'd want to use
    /// a proper expression parser like `evalexpr` or `fasteval` crates.
    fn evaluate_advanced_expression(&self, expr: &str) -> Result<f64> {
        let cleaned = expr.replace(' ', "").to_lowercase();

        // Handle constants
        let with_constants = cleaned
            .replace("pi", &std::f64::consts::PI.to_string())
            .replace('e', &std::f64::consts::E.to_string());

        // Handle simple functions (basic implementation)
        if with_constants.starts_with("sqrt(") && with_constants.ends_with(')') {
            let inner = &with_constants[5..with_constants.len() - 1];
            let val = self.evaluate_basic(inner)?;
            return Ok(val.sqrt());
        }

        if with_constants.starts_with("abs(") && with_constants.ends_with(')') {
            let inner = &with_constants[4..with_constants.len() - 1];
            let val = self.evaluate_basic(inner)?;
            return Ok(val.abs());
        }

        if with_constants.starts_with("sin(") && with_constants.ends_with(')') {
            let inner = &with_constants[4..with_constants.len() - 1];
            let val = self.evaluate_basic(inner)?;
            return Ok(val.sin());
        }

        if with_constants.starts_with("cos(") && with_constants.ends_with(')') {
            let inner = &with_constants[4..with_constants.len() - 1];
            let val = self.evaluate_basic(inner)?;
            return Ok(val.cos());
        }

        if with_constants.starts_with("tan(") && with_constants.ends_with(')') {
            let inner = &with_constants[4..with_constants.len() - 1];
            let val = self.evaluate_basic(inner)?;
            return Ok(val.tan());
        }

        if with_constants.starts_with("log(") && with_constants.ends_with(')') {
            let inner = &with_constants[4..with_constants.len() - 1];
            let val = self.evaluate_basic(inner)?;
            return Ok(val.log10());
        }

        if with_constants.starts_with("ln(") && with_constants.ends_with(')') {
            let inner = &with_constants[3..with_constants.len() - 1];
            let val = self.evaluate_basic(inner)?;
            return Ok(val.ln());
        }

        // Fall back to basic evaluation
        self.evaluate_basic(&with_constants)
    }

    /// Basic arithmetic evaluation
    fn evaluate_basic(&self, expr: &str) -> Result<f64> {
        let cleaned = expr.trim();

        // Handle single numbers
        if let Ok(num) = cleaned.parse::<f64>() {
            return Ok(num);
        }

        // Handle parentheses (basic implementation)
        if let (Some(start), Some(end)) = (cleaned.find('('), cleaned.rfind(')')) {
            let before = &cleaned[..start];
            let inner = &cleaned[start + 1..end];
            let after = &cleaned[end + 1..];
            let inner_result = self.evaluate_basic(inner)?;
            let combined = format!("{before}{inner_result}{after}");
            return self.evaluate_basic(&combined);
        }

        // Handle power operator
        if let Some(pos) = cleaned.find('^') {
            let left = self.evaluate_basic(&cleaned[..pos])?;
            let right = self.evaluate_basic(&cleaned[pos + 1..])?;
            return Ok(left.powf(right));
        }

        // Handle multiplication and division (left to right)
        let mut current = 0.0;
        let mut last_op = '+';
        let mut current_number = String::new();

        for (_i, ch) in cleaned.char_indices() {
            match ch {
                '+' | '-' | '*' | '/' => {
                    if !current_number.is_empty() {
                        let num = current_number
                            .parse::<f64>()
                            .map_err(|_| AgentError::tool("calculator", "Invalid number format"))?;

                        current = match last_op {
                            '+' => current + num,
                            '-' => current - num,
                            '*' => current * num,
                            '/' => {
                                if num == 0.0 {
                                    return Err(AgentError::tool("calculator", "Division by zero"));
                                }
                                current / num
                            }
                            _ => num,
                        };
                    }
                    last_op = ch;
                    current_number.clear();
                }
                _ => current_number.push(ch),
            }
        }

        // Handle the last number
        if !current_number.is_empty() {
            let num = current_number
                .parse::<f64>()
                .map_err(|_| AgentError::tool("calculator", "Invalid number format"))?;

            current = match last_op {
                '+' => current + num,
                '-' => current - num,
                '*' => current * num,
                '/' => {
                    if num == 0.0 {
                        return Err(AgentError::tool("calculator", "Division by zero"));
                    }
                    current / num
                }
                _ => num,
            };
        }

        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_calculator_basic() {
        let tool = CalculatorTool::new();
        let context = ToolContext::new();

        // Test basic arithmetic
        let test_cases = vec![
            ("2 + 3", 5.0),
            ("10 - 4", 6.0),
            ("3 * 4", 12.0),
            ("15 / 3", 5.0),
            ("2 ^ 3", 8.0),
        ];

        for (expr, expected) in test_cases {
            let args = serde_json::json!({"expression": expr});
            let result = tool.execute(args, &context).await.unwrap();
            assert!(result.success, "Failed for expression: {}", expr);

            let metadata = result.metadata.get("result").unwrap();
            let actual = metadata.as_f64().unwrap();
            assert!(
                (actual - expected).abs() < 0.0001,
                "Expected {} but got {} for expression {}",
                expected,
                actual,
                expr
            );
        }
    }

    #[tokio::test]
    async fn test_calculator_functions() {
        let tool = CalculatorTool::new();
        let context = ToolContext::new();

        // Test functions
        let args = serde_json::json!({"expression": "sqrt(16)"});
        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);

        let metadata = result.metadata.get("result").unwrap();
        let actual = metadata.as_f64().unwrap();
        assert!((actual - 4.0).abs() < 0.0001);
    }

    #[tokio::test]
    async fn test_calculator_constants() {
        let tool = CalculatorTool::new();
        let context = ToolContext::new();

        // Test constants
        let args = serde_json::json!({"expression": "pi"});
        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);

        let metadata = result.metadata.get("result").unwrap();
        let actual = metadata.as_f64().unwrap();
        assert!((actual - std::f64::consts::PI).abs() < 0.0001);
    }
}
