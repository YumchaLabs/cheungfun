//! Async-First Design Demo for CodeSplitter
//!
//! This example demonstrates the correct way to use CodeSplitter in different contexts.
//! Cheungfun follows an async-first design pattern where all parsing is asynchronous.

use cheungfun_core::Document;
use cheungfun_indexing::{
    loaders::ProgrammingLanguage,
    node_parser::{text::CodeSplitter, NodeParser},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 CodeSplitter Async-First Design Demo\n");

    // Demo 1: Preferred usage in async context
    demo_async_usage().await?;

    // Demo 2: Usage in sync context (requires manual runtime creation)
    demo_sync_usage()?;

    // Demo 3: Advanced async patterns
    demo_advanced_async_patterns().await?;

    Ok(())
}

/// ✅ CORRECT: Use parse_nodes() in async context
async fn demo_async_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("✅ Demo 1: Correct Async Usage");
    println!("==============================");

    let splitter = CodeSplitter::from_defaults(
        ProgrammingLanguage::Rust,
        40,   // lines per chunk
        10,   // overlap
        1500, // max chars
    )?;

    let rust_code = r#"
        use std::collections::HashMap;

        /// A simple calculator struct
        pub struct Calculator {
            memory: HashMap<String, f64>,
        }

        impl Calculator {
            /// Create a new calculator
            pub fn new() -> Self {
                Self {
                    memory: HashMap::new(),
                }
            }

            /// Add two numbers
            pub fn add(&self, a: f64, b: f64) -> f64 {
                a + b
            }

            /// Store a value in memory
            pub fn store(&mut self, key: String, value: f64) {
                self.memory.insert(key, value);
            }

            /// Recall a value from memory
            pub fn recall(&self, key: &str) -> Option<f64> {
                self.memory.get(key).copied()
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn test_add() {
                let calc = Calculator::new();
                assert_eq!(calc.add(2.0, 3.0), 5.0);
            }

            #[test]
            fn test_memory() {
                let mut calc = Calculator::new();
                calc.store("result".to_string(), 42.0);
                assert_eq!(calc.recall("result"), Some(42.0));
            }
        }
    "#;

    let document = Document::new(rust_code);

    // ✅ CORRECT: Use parse_nodes() in async context
    let nodes = splitter.parse_nodes(&[document], false).await?;

    println!("Successfully parsed {} nodes", nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        println!("  Node {}: {} chars", i + 1, node.content.len());
    }
    println!();

    Ok(())
}

/// ✅ CORRECT: Use parse_nodes() in sync context with runtime
fn demo_sync_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("✅ Demo 2: Correct Sync Usage");
    println!("=============================");

    let splitter = CodeSplitter::from_defaults(
        ProgrammingLanguage::Python,
        30,   // lines per chunk
        5,    // overlap
        1200, // max chars
    )?;

    let python_code = r#"
import math
from typing import List, Optional

class MathUtils:
    """Utility class for mathematical operations."""

    @staticmethod
    def factorial(n: int) -> int:
        """Calculate factorial of n."""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0 or n == 1:
            return 1
        return n * MathUtils.factorial(n - 1)

    @staticmethod
    def fibonacci(n: int) -> List[int]:
        """Generate Fibonacci sequence up to n terms."""
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        elif n == 2:
            return [0, 1]

        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib

    @staticmethod
    def is_prime(n: int) -> bool:
        """Check if a number is prime."""
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

def main():
    """Main function to demonstrate the utilities."""
    print("Factorial of 5:", MathUtils.factorial(5))
    print("Fibonacci sequence (10 terms):", MathUtils.fibonacci(10))
    print("Is 17 prime?", MathUtils.is_prime(17))

if __name__ == "__main__":
    main()
    "#;

    let document = Document::new(python_code);

    // ✅ CORRECT: Create runtime and use parse_nodes() in sync context
    let rt = tokio::runtime::Runtime::new()?;
    let nodes = rt.block_on(splitter.parse_nodes(&[document], false))?;

    println!("Successfully parsed {} nodes", nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        println!("  Node {}: {} chars", i + 1, node.content.len());
    }
    println!();

    Ok(())
}

/// ✅ ADVANCED: Demonstrate concurrent parsing and batch processing
async fn demo_advanced_async_patterns() -> Result<(), Box<dyn std::error::Error>> {
    println!("✅ Demo 3: Advanced Async Patterns");
    println!("==================================");

    let splitter = CodeSplitter::from_defaults(
        ProgrammingLanguage::JavaScript,
        25,   // lines per chunk
        5,    // overlap
        1000, // max chars
    )?;

    let js_code = r#"
        function fibonacci(n) {
            if (n <= 1) return n;
            return fibonacci(n - 1) + fibonacci(n - 2);
        }

        class Calculator {
            constructor() {
                this.memory = {};
            }

            add(a, b) {
                return a + b;
            }

            store(key, value) {
                this.memory[key] = value;
            }

            recall(key) {
                return this.memory[key];
            }
        }

        // Usage example
        const calc = new Calculator();
        console.log(calc.add(5, 3));
        calc.store('result', 42);
        console.log(calc.recall('result'));
    "#;

    // Create multiple documents for batch processing
    let documents = vec![
        Document::new(js_code),
        Document::new("console.log('Hello, world!');"),
        Document::new("const x = 42; const y = x * 2;"),
    ];

    // ✅ Batch processing with single call
    let nodes = splitter.parse_nodes(&documents, false).await?;

    println!(
        "Successfully parsed {} nodes from {} documents",
        nodes.len(),
        documents.len()
    );

    // ✅ Concurrent processing of multiple splitters
    use cheungfun_indexing::node_parser::text::SentenceSplitter;
    let sentence_splitter = SentenceSplitter::from_defaults(200, 50)?;

    let (code_nodes, sentence_nodes) = tokio::join!(
        splitter.parse_nodes(&documents, false),
        sentence_splitter.parse_nodes(&documents, false)
    );

    println!(
        "Concurrent results: {} code nodes, {} sentence nodes",
        code_nodes?.len(),
        sentence_nodes?.len()
    );
    println!();

    Ok(())
}

/// Helper function to demonstrate proper usage patterns
#[allow(dead_code)]
async fn demonstrate_best_practices() {
    let splitter = CodeSplitter::from_defaults(ProgrammingLanguage::Rust, 40, 10, 1500).unwrap();

    let document = Document::new("fn main() { println!(\"Hello, world!\"); }");

    // ✅ CORRECT: Use parse_nodes() in async context
    match splitter.parse_nodes(&[document], false).await {
        Ok(nodes) => println!("Successfully parsed {} nodes", nodes.len()),
        Err(e) => println!("Error: {}", e),
    }
}
