//! Simple test for the new node parser system.

use cheungfun_core::Document;
use cheungfun_indexing::node_parser::text::SentenceSplitter;
use cheungfun_indexing::node_parser::TextSplitter;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Simple Node Parser Test");
    println!("==========================\n");

    // Create a simple document
    let mut metadata = HashMap::new();
    metadata.insert(
        "title".to_string(),
        serde_json::Value::String("Test Document".to_string()),
    );

    let content = "This is the first sentence. This is the second sentence. This is the third sentence with more content to test splitting.";

    let document = Document {
        id: uuid::Uuid::new_v4(),
        content: content.to_string(),
        metadata,
        embedding: None,
    };

    // Test SentenceSplitter
    println!("📝 Testing SentenceSplitter");
    println!("---------------------------");

    let splitter = SentenceSplitter::from_defaults(50, 10)?;

    // Test text splitting
    let chunks = splitter.split_text(&document.content)?;
    println!("Split text into {} chunks", chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        println!("  Chunk {}: {}", i + 1, chunk);
    }

    println!("\n✅ Test completed successfully!");
    Ok(())
}
