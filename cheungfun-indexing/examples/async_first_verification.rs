//! Async-First Design Verification
//!
//! This example verifies that the async-first design works correctly
//! and that the runtime conflict issue has been resolved.

use cheungfun_core::Document;
use cheungfun_indexing::{
    loaders::ProgrammingLanguage,
    node_parser::{text::CodeSplitter, NodeParser},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Async-First Design Verification\n");

    // Test 1: Basic async usage
    test_basic_async_usage().await?;
    
    // Test 2: Nested async calls (this would previously fail)
    test_nested_async_calls().await?;
    
    // Test 3: Concurrent processing
    test_concurrent_processing().await?;

    println!("✅ All tests passed! Runtime conflict issue resolved.");
    Ok(())
}

/// Test basic async usage
async fn test_basic_async_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("Test 1: Basic Async Usage");
    println!("=========================");

    let splitter = CodeSplitter::from_defaults(
        ProgrammingLanguage::Rust,
        20, 5, 800,
    )?;

    let rust_code = r#"
        fn fibonacci(n: u32) -> u32 {
            match n {
                0 => 0,
                1 => 1,
                _ => fibonacci(n - 1) + fibonacci(n - 2),
            }
        }

        fn main() {
            for i in 0..10 {
                println!("fib({}) = {}", i, fibonacci(i));
            }
        }
    "#;

    let document = Document::new(rust_code);
    let nodes = splitter.parse_nodes(&[document], false).await?;
    
    println!("✅ Successfully parsed {} nodes", nodes.len());
    println!();
    Ok(())
}

/// Test nested async calls (this would previously cause runtime conflicts)
async fn test_nested_async_calls() -> Result<(), Box<dyn std::error::Error>> {
    println!("Test 2: Nested Async Calls");
    println!("===========================");

    // This simulates the scenario where user code calls our parser
    // from within an async context, which previously caused issues
    async fn user_async_function() -> Result<usize, Box<dyn std::error::Error>> {
        let splitter = CodeSplitter::from_defaults(
            ProgrammingLanguage::Python,
            15, 3, 600,
        )?;

        let python_code = r#"
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)

# Example usage
numbers = [3, 6, 8, 10, 1, 2, 1]
sorted_numbers = quicksort(numbers)
print(sorted_numbers)
        "#;

        let document = Document::new(python_code);
        
        // This call would previously fail with "Cannot start a runtime from within a runtime"
        let nodes = splitter.parse_nodes(&[document], false).await?;
        Ok(nodes.len())
    }

    let node_count = user_async_function().await?;
    println!("✅ Successfully handled nested async call: {} nodes", node_count);
    println!();
    Ok(())
}

/// Test concurrent processing
async fn test_concurrent_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("Test 3: Concurrent Processing");
    println!("==============================");

    let splitter = CodeSplitter::from_defaults(
        ProgrammingLanguage::JavaScript,
        25, 5, 1000,
    )?;

    let js_codes = vec![
        r#"
        function bubbleSort(arr) {
            let n = arr.length;
            for (let i = 0; i < n - 1; i++) {
                for (let j = 0; j < n - i - 1; j++) {
                    if (arr[j] > arr[j + 1]) {
                        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
                    }
                }
            }
            return arr;
        }
        "#,
        r#"
        class Stack {
            constructor() {
                this.items = [];
            }
            
            push(item) {
                this.items.push(item);
            }
            
            pop() {
                return this.items.pop();
            }
            
            peek() {
                return this.items[this.items.length - 1];
            }
            
            isEmpty() {
                return this.items.length === 0;
            }
        }
        "#,
        r#"
        async function fetchData(url) {
            try {
                const response = await fetch(url);
                const data = await response.json();
                return data;
            } catch (error) {
                console.error('Error fetching data:', error);
                throw error;
            }
        }
        "#,
    ];

    // Process all documents concurrently
    let tasks: Vec<_> = js_codes
        .into_iter()
        .enumerate()
        .map(|(i, code)| {
            let splitter = splitter.clone();
            async move {
                let document = Document::new(code);
                let nodes = splitter.parse_nodes(&[document], false).await?;
                Ok::<_, Box<dyn std::error::Error>>((i, nodes.len()))
            }
        })
        .collect();

    let results = futures::future::try_join_all(tasks).await?;
    
    let total_nodes: usize = results.iter().map(|(_, count)| count).sum();
    println!("✅ Concurrent processing completed:");
    for (i, count) in results {
        println!("  Document {}: {} nodes", i + 1, count);
    }
    println!("  Total: {} nodes", total_nodes);
    println!();
    
    Ok(())
}

// Helper trait to make CodeSplitter cloneable for the test
trait CloneableSplitter {
    fn clone(&self) -> Self;
}

impl CloneableSplitter for CodeSplitter {
    fn clone(&self) -> Self {
        // Create a new splitter with the same configuration
        CodeSplitter::from_defaults(
            self.config.language,
            self.config.chunk_lines,
            self.config.chunk_lines_overlap,
            self.config.max_chars,
        ).expect("Failed to clone CodeSplitter")
    }
}
