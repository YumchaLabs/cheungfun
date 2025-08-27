//! Simple modern API example demonstrating the unified Transform interface.

use cheungfun_core::{
    traits::{Transform, TransformInput},
    Document,
};
use cheungfun_indexing::loaders::ProgrammingLanguage;
use cheungfun_indexing::node_parser::{
    text::{CodeSplitter, SentenceSplitter, TokenTextSplitter},
    NodeParser,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Unified Transform Interface Example");
    println!("=======================================\n");

    // Create a sample document
    let sample_text = "This is a comprehensive test document. It contains multiple sentences \
                      with varying lengths and complexity. Some sentences are short. Others are \
                      much longer and contain multiple clauses that demonstrate how the text \
                      splitter handles different types of content. The goal is to show how \
                      different splitting strategies work with the same input text.";

    let mut metadata = HashMap::new();
    metadata.insert(
        "type".to_string(),
        serde_json::Value::String("test".to_string()),
    );

    let document = Document {
        id: uuid::Uuid::new_v4(),
        content: sample_text.to_string(),
        metadata,
        embedding: None,
    };

    let documents = vec![document];

    // Example 1: SentenceSplitter using Transform interface
    println!("📝 Example 1: SentenceSplitter (Transform Interface)");
    println!("----------------------------------------------------");

    let sentence_splitter = SentenceSplitter::from_defaults(200, 40)?;
    let input = TransformInput::Documents(documents.clone());
    let nodes = sentence_splitter.transform(input).await?;

    println!("   ✅ Created {} nodes", nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        println!("   📄 Node {}: {} chars", i + 1, node.content.len());
        println!(
            "      Content: {}...",
            node.content.chars().take(50).collect::<String>()
        );
    }

    // Example 2: TokenTextSplitter using Transform interface
    println!("\n🔤 Example 2: TokenTextSplitter (Transform Interface)");
    println!("-----------------------------------------------------");

    let token_splitter = TokenTextSplitter::from_defaults(150, 30)?;
    let input = TransformInput::Documents(documents.clone());
    let nodes = token_splitter.transform(input).await?;

    println!("   ✅ Created {} nodes", nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        println!("   📄 Node {}: {} chars", i + 1, node.content.len());
        println!(
            "      Content: {}...",
            node.content.chars().take(50).collect::<String>()
        );
    }

    // Example 3: CodeSplitter with Rust code
    println!("\n💻 Example 3: CodeSplitter");
    println!("--------------------------");

    let rust_code = r#"
use std::collections::HashMap;

/// A simple user struct
pub struct User {
    pub id: u64,
    pub name: String,
}

impl User {
    /// Create a new user
    pub fn new(id: u64, name: String) -> Self {
        Self { id, name }
    }

    /// Get display name
    pub fn display_name(&self) -> String {
        format!("User: {}", self.name)
    }
}

fn main() {
    let user = User::new(1, "Alice".to_string());
    println!("{}", user.display_name());
}
"#;

    let mut code_metadata = HashMap::new();
    code_metadata.insert(
        "language".to_string(),
        serde_json::Value::String("rust".to_string()),
    );
    code_metadata.insert(
        "filename".to_string(),
        serde_json::Value::String("user.rs".to_string()),
    );

    let code_document = Document {
        id: uuid::Uuid::new_v4(),
        content: rust_code.to_string(),
        metadata: code_metadata,
        embedding: None,
    };

    let code_splitter = CodeSplitter::from_defaults(ProgrammingLanguage::Rust, 15, 3, 500)?;

    let input = TransformInput::Document(code_document);
    let nodes = code_splitter.transform(input).await?;

    println!("   ✅ Created {} nodes from Rust code", nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        println!("   📄 Node {}: {} chars", i + 1, node.content.len());
        let preview = node.content.lines().take(2).collect::<Vec<_>>().join(" ");
        println!(
            "      Content: {}...",
            preview.chars().take(60).collect::<String>()
        );
    }

    // Example 4: Comparison
    println!("\n📊 Example 4: Splitter Comparison");
    println!("----------------------------------");

    let test_text = "The quick brown fox jumps over the lazy dog. This is a simple sentence. \
                    Here's another one with more complexity and multiple clauses that should \
                    demonstrate different splitting behaviors across various algorithms.";

    let mut test_metadata = HashMap::new();
    test_metadata.insert(
        "test".to_string(),
        serde_json::Value::String("comparison".to_string()),
    );

    let test_document = Document {
        id: uuid::Uuid::new_v4(),
        content: test_text.to_string(),
        metadata: test_metadata,
        embedding: None,
    };

    let test_docs = vec![test_document];

    // Test different chunk sizes
    let configs = vec![
        ("Small chunks (100 chars)", 100, 20),
        ("Medium chunks (200 chars)", 200, 40),
        ("Large chunks (300 chars)", 300, 60),
    ];

    for (name, chunk_size, overlap) in configs {
        let splitter = SentenceSplitter::from_defaults(chunk_size, overlap)?;
        let nodes = splitter.parse_nodes(&test_docs, false).await?;

        let avg_size = if !nodes.is_empty() {
            nodes.iter().map(|n| n.content.len()).sum::<usize>() / nodes.len()
        } else {
            0
        };

        println!(
            "   📈 {}: {} nodes, avg {} chars",
            name,
            nodes.len(),
            avg_size
        );
    }

    // Example 5: Unified Transform Interface Benefits
    println!("\n🔄 Example 5: Unified Transform Interface Benefits");
    println!("--------------------------------------------------");

    // Create a vector of different transform components
    let transforms: Vec<Box<dyn Transform>> = vec![
        Box::new(SentenceSplitter::from_defaults(200, 40)?),
        Box::new(TokenTextSplitter::from_defaults(180, 35)?),
    ];

    // Process the same input with different transforms
    let input = TransformInput::Documents(test_docs.clone());
    for (i, transform) in transforms.iter().enumerate() {
        let nodes = transform.transform(input.clone()).await?;
        println!(
            "   🔧 Transform {}: {} ({} nodes)",
            i + 1,
            transform.name(),
            nodes.len()
        );
    }

    println!("\n🎉 Unified Transform Interface Example Complete!");
    println!("=======================================================");
    println!("✅ Demonstrated unified Transform interface");
    println!("✅ Showed SentenceSplitter with Transform");
    println!("✅ Showed TokenTextSplitter with Transform");
    println!("✅ Showed CodeSplitter with Transform");
    println!("✅ Compared different chunk sizes");
    println!("✅ Demonstrated polymorphic transform usage");
    println!("\n📚 Key Benefits of Unified Interface:");
    println!("   • Single interface for all transformations");
    println!("   • Type-safe TransformInput enum");
    println!("   • Polymorphic transform processing");
    println!("   • Consistent async API across all components");
    println!("   • Better composability and pipeline integration");

    Ok(())
}
