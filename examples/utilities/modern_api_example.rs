//! Modern API example demonstrating the unified Transform interface architecture.

use cheungfun_core::{
    traits::{DocumentState, NodeState, TypedData, TypedTransform},
    Document,
};
use cheungfun_indexing::prelude::*;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Cheungfun Unified Transform Interface Example");
    println!("===============================================\n");

    // Example 1: Direct Transform interface usage
    println!("📝 Example 1: Direct Transform Interface Usage");
    println!("----------------------------------------------");

    let documents = create_sample_documents();

    // Create different types of splitters
    let sentence_splitter = SentenceSplitter::from_defaults(500, 100)?;
    let token_splitter = TokenTextSplitter::from_defaults(400, 80)?;
    let code_splitter = CodeSplitter::from_defaults(ProgrammingLanguage::Rust, 20, 5, 800)?;

    // Test sentence splitter using unified TypedTransform interface
    let input = TypedData::from_documents(documents.clone());
    let result = sentence_splitter.transform(input).await?;
    let nodes = result.into_nodes();
    println!("   ✅ SentenceSplitter created {} nodes", nodes.len());

    // Test token splitter using unified TypedTransform interface
    let input = TypedData::from_documents(documents.clone());
    let result = token_splitter.transform(input).await?;
    let nodes = result.into_nodes();
    println!("   ✅ TokenTextSplitter created {} nodes", nodes.len());

    // Test code splitter with Rust code using unified TypedTransform interface
    let rust_doc = create_rust_code_document();
    let input = TypedData::from_documents(vec![rust_doc]);
    let result = code_splitter.transform(input).await?;
    let nodes = result.into_nodes();
    println!(
        "   ✅ CodeSplitter created {} nodes from Rust code",
        nodes.len()
    );

    // Example 2: Unified Transform Interface Demonstration
    println!("\n🔄 Example 2: Unified Transform Interface");
    println!("------------------------------------------");

    // All components now implement the same TypedTransform trait
    let transforms: Vec<Box<dyn TypedTransform<DocumentState, NodeState>>> = vec![
        Box::new(SentenceSplitter::from_defaults(400, 80)?),
        Box::new(TokenTextSplitter::from_defaults(300, 60)?),
        Box::new(CodeSplitter::from_defaults(
            ProgrammingLanguage::Rust,
            20,
            5,
            800,
        )?),
    ];

    for transform in &transforms {
        println!("   ✅ Transform component: {}", transform.name());
    }

    // Example 3: Polymorphic Processing
    println!("\n🔧 Example 3: Polymorphic Processing");
    println!("------------------------------------");

    let test_input = TypedData::from_documents(documents.clone());
    for transform in transforms {
        let result = transform.transform(test_input.clone()).await?;
        let nodes = result.into_nodes();
        println!("   📊 {}: {} nodes", transform.name(), nodes.len());
    }

    // Example 4: Transform comparison with detailed analysis
    println!("\n📊 Example 4: Transform Comparison");
    println!("-----------------------------------");

    let test_doc = create_test_document();
    let test_input = TypedData::from_documents(vec![test_doc]);

    // Compare different transforms using unified interface
    let transforms: Vec<(&str, Box<dyn TypedTransform<DocumentState, NodeState>>)> = vec![
        (
            "SentenceSplitter",
            Box::new(SentenceSplitter::from_defaults(200, 40)?),
        ),
        (
            "TokenTextSplitter",
            Box::new(TokenTextSplitter::from_defaults(200, 40)?),
        ),
    ];

    for (name, transform) in transforms {
        let result = transform.transform(test_input.clone()).await?;
        let nodes = result.into_nodes();
        println!("   📈 {}: {} nodes", name, nodes.len());

        if !nodes.is_empty() {
            let avg_length = nodes.iter().map(|n| n.content.len()).sum::<usize>() / nodes.len();
            println!("      Average node length: {} characters", avg_length);
        }
    }

    println!("\n🎉 Unified Transform Interface Example Complete!");
    println!("===============================================");
    println!("✅ Demonstrated unified Transform interface usage");
    println!("✅ Showed polymorphic processing capabilities");
    println!("✅ Used type-safe TransformInput enum");
    println!("✅ Compared different transform types");
    println!("\n📚 Key Benefits of Unified Architecture:");
    println!("   • Single interface for all transformations");
    println!("   • Type-safe input handling with TransformInput");
    println!("   • Polymorphic processing support");
    println!("   • LlamaIndex-compatible design");
    println!("   • Simplified pipeline integration");

    Ok(())
}

/// Create sample documents for testing.
fn create_sample_documents() -> Vec<Document> {
    let mut documents = Vec::new();

    // Create a text document
    let mut metadata1 = HashMap::new();
    metadata1.insert(
        "filename".to_string(),
        serde_json::Value::String("sample.txt".to_string()),
    );
    metadata1.insert(
        "type".to_string(),
        serde_json::Value::String("text".to_string()),
    );

    documents.push(Document {
        id: uuid::Uuid::new_v4(),
        content: "This is a sample document. It contains multiple sentences for testing. \
                 The text splitter should handle this content properly. Each sentence \
                 provides meaningful content for indexing and retrieval."
            .to_string(),
        metadata: metadata1,
        embedding: None,
    });

    // Create a markdown document
    let mut metadata2 = HashMap::new();
    metadata2.insert(
        "filename".to_string(),
        serde_json::Value::String("readme.md".to_string()),
    );
    metadata2.insert(
        "type".to_string(),
        serde_json::Value::String("markdown".to_string()),
    );

    documents.push(Document {
        id: uuid::Uuid::new_v4(),
        content: "# Sample Markdown\n\n\
                 This is a **markdown** document with various elements.\n\n\
                 ## Features\n\n\
                 - Lists and formatting\n\
                 - Code blocks\n\
                 - Headers and emphasis\n\n\
                 The content should be processed correctly by the indexing system."
            .to_string(),
        metadata: metadata2,
        embedding: None,
    });

    documents
}

/// Create a Rust code document for testing code splitter.
fn create_rust_code_document() -> Document {
    let rust_code = r#"
use std::collections::HashMap;

/// A simple user struct
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
}

impl User {
    /// Create a new user
    pub fn new(id: u64, name: String, email: String) -> Self {
        Self { id, name, email }
    }

    /// Get user display name
    pub fn display_name(&self) -> String {
        format!("{} <{}>", self.name, self.email)
    }
}

/// User manager for handling multiple users
pub struct UserManager {
    users: HashMap<u64, User>,
    next_id: u64,
}

impl UserManager {
    /// Create a new user manager
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
            next_id: 1,
        }
    }

    /// Add a new user
    pub fn add_user(&mut self, name: String, email: String) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        
        let user = User::new(id, name, email);
        self.users.insert(id, user);
        
        id
    }
}
"#;

    let mut metadata = HashMap::new();
    metadata.insert(
        "filename".to_string(),
        serde_json::Value::String("user.rs".to_string()),
    );
    metadata.insert(
        "language".to_string(),
        serde_json::Value::String("rust".to_string()),
    );

    Document {
        id: uuid::Uuid::new_v4(),
        content: rust_code.to_string(),
        metadata,
        embedding: None,
    }
}

/// Create a test document with varied content.
fn create_test_document() -> Document {
    let content = "This is a comprehensive test document designed to evaluate different text splitting strategies. \
                   It contains sentences of varying lengths and complexity. Some sentences are short. \
                   Others are much longer and contain multiple clauses, subclauses, and various punctuation marks \
                   that might affect how the text splitter processes the content. The document also includes \
                   technical terms, numbers like 123 and 456, and special characters such as @, #, and $. \
                   This variety helps ensure that the splitting algorithms work correctly across different \
                   types of content and maintain semantic coherence in the resulting chunks.";

    let mut metadata = HashMap::new();
    metadata.insert(
        "type".to_string(),
        serde_json::Value::String("test".to_string()),
    );
    metadata.insert(
        "purpose".to_string(),
        serde_json::Value::String("comparison".to_string()),
    );

    Document {
        id: uuid::Uuid::new_v4(),
        content: content.to_string(),
        metadata,
        embedding: None,
    }
}
