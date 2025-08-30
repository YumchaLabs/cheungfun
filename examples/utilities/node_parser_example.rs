//! Example demonstrating the new node parser system.
//!
//! This example shows how to use the new LlamaIndex-inspired node parser
//! architecture with different splitting strategies.

use cheungfun_core::Document;
use cheungfun_indexing::loaders::ProgrammingLanguage;
use cheungfun_indexing::node_parser::{
    text::{CodeSplitter, SentenceSplitter, TokenTextSplitter},
    NodeParser, TextSplitter,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🚀 Cheungfun Node Parser Example");
    println!("=================================\n");

    // Create sample documents
    let text_document = create_text_document();
    let code_document = create_code_document();

    // Test SentenceSplitter
    println!("📝 Testing SentenceSplitter");
    println!("---------------------------");
    test_sentence_splitter(&text_document).await?;

    // Test TokenTextSplitter
    println!("\n🔤 Testing TokenTextSplitter");
    println!("-----------------------------");
    test_token_splitter(&text_document).await?;

    // Test CodeSplitter
    println!("\n💻 Testing CodeSplitter");
    println!("------------------------");
    test_code_splitter(&code_document).await?;

    println!("\n✅ All tests completed successfully!");
    Ok(())
}

fn create_text_document() -> Document {
    let mut metadata = HashMap::new();
    metadata.insert(
        "title".to_string(),
        serde_json::Value::String("Sample Article".to_string()),
    );
    metadata.insert(
        "author".to_string(),
        serde_json::Value::String("AI Assistant".to_string()),
    );

    let content = r#"
Artificial Intelligence and Machine Learning

Artificial intelligence (AI) is a rapidly evolving field that encompasses various technologies and methodologies. Machine learning, a subset of AI, focuses on algorithms that can learn and improve from experience without being explicitly programmed.

Deep learning, which uses neural networks with multiple layers, has revolutionized many areas of AI. These networks can process vast amounts of data and identify complex patterns that were previously difficult to detect.

Natural language processing (NLP) is another crucial area of AI that deals with the interaction between computers and human language. NLP enables machines to understand, interpret, and generate human language in a valuable way.

Computer vision allows machines to interpret and understand visual information from the world. This technology is used in applications ranging from medical imaging to autonomous vehicles.

The future of AI holds immense potential for transforming various industries, including healthcare, finance, transportation, and education. However, it also raises important ethical considerations that must be carefully addressed.
"#.trim();

    Document {
        id: uuid::Uuid::new_v4(),
        content: content.to_string(),
        metadata,
        embedding: None,
    }
}

fn create_code_document() -> Document {
    let mut metadata = HashMap::new();
    metadata.insert(
        "filename".to_string(),
        serde_json::Value::String("example.rs".to_string()),
    );
    metadata.insert(
        "language".to_string(),
        serde_json::Value::String("rust".to_string()),
    );

    let content = r#"
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// A simple data structure for storing user information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
    pub age: Option<u32>,
}

impl User {
    /// Create a new user with the given information.
    pub fn new(id: u64, name: String, email: String) -> Self {
        Self {
            id,
            name,
            email,
            age: None,
        }
    }

    /// Set the user's age.
    pub fn set_age(&mut self, age: u32) {
        self.age = Some(age);
    }

    /// Get the user's display name.
    pub fn display_name(&self) -> String {
        format!("{} ({})", self.name, self.email)
    }
}

/// A user manager for handling multiple users.
pub struct UserManager {
    users: HashMap<u64, User>,
    next_id: u64,
}

impl UserManager {
    /// Create a new user manager.
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
            next_id: 1,
        }
    }

    /// Add a new user to the manager.
    pub fn add_user(&mut self, name: String, email: String) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        
        let user = User::new(id, name, email);
        self.users.insert(id, user);
        
        id
    }

    /// Get a user by ID.
    pub fn get_user(&self, id: u64) -> Option<&User> {
        self.users.get(&id)
    }

    /// Get all users.
    pub fn get_all_users(&self) -> Vec<&User> {
        self.users.values().collect()
    }
}
"#
    .trim();

    Document {
        id: uuid::Uuid::new_v4(),
        content: content.to_string(),
        metadata,
        embedding: None,
    }
}

async fn test_sentence_splitter(document: &Document) -> Result<(), Box<dyn std::error::Error>> {
    let splitter = SentenceSplitter::from_defaults(200, 50)?;

    // Test text splitting
    let chunks = splitter.split_text(&document.content)?;
    println!("Split text into {} chunks", chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        println!("  Chunk {}: {} characters", i + 1, chunk.len());
        if chunk.len() < 100 {
            println!("    Content: {}", chunk.trim());
        } else {
            println!("    Content: {}...", &chunk.trim()[..97]);
        }
    }

    // Test node parsing
    let nodes = NodeParser::parse_nodes(&splitter, &[document.clone()], false).await?;
    println!("Created {} nodes from document", nodes.len());

    // Check relationships
    let mut has_relationships = 0;
    for node in &nodes {
        if !node.relationships.relationship_types().is_empty() {
            has_relationships += 1;
        }
    }
    println!("Nodes with relationships: {}", has_relationships);

    Ok(())
}

async fn test_token_splitter(document: &Document) -> Result<(), Box<dyn std::error::Error>> {
    let splitter = TokenTextSplitter::from_defaults(150, 30)?;

    // Test text splitting
    let chunks = splitter.split_text(&document.content)?;
    println!("Split text into {} chunks", chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        println!("  Chunk {}: {} characters", i + 1, chunk.len());
    }

    // Test node parsing
    let nodes = NodeParser::parse_nodes(&splitter, &[document.clone()], false).await?;
    println!("Created {} nodes from document", nodes.len());

    Ok(())
}

async fn test_code_splitter(document: &Document) -> Result<(), Box<dyn std::error::Error>> {
    let splitter = CodeSplitter::from_defaults(
        ProgrammingLanguage::Rust,
        20,  // chunk_lines
        5,   // chunk_lines_overlap
        800, // max_chars
    )?;

    // Test text splitting
    let chunks = splitter.split_text(&document.content)?;
    println!("Split code into {} chunks", chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        let line_count = chunk.lines().count();
        println!(
            "  Chunk {}: {} lines, {} characters",
            i + 1,
            line_count,
            chunk.len()
        );

        // Show first few lines of each chunk
        let first_lines: Vec<&str> = chunk.lines().take(3).collect();
        for (j, line) in first_lines.iter().enumerate() {
            if j == 0 {
                println!("    {}", line.trim());
            } else {
                println!("    {}", line.trim());
            }
        }
        if chunk.lines().count() > 3 {
            println!("    ...");
        }
    }

    // Test node parsing
    let nodes = NodeParser::parse_nodes(&splitter, &[document.clone()], false).await?;
    println!("Created {} nodes from code document", nodes.len());

    Ok(())
}
