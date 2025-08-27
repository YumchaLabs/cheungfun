//! Transform Trait Verification Example
//!
//! This example verifies that all NodeParser implementations correctly
//! implement the Transform trait and can be used polymorphically.

use cheungfun_core::{
    traits::{Transform, TransformInput},
    Document, Result as CoreResult,
};
use cheungfun_indexing::{
    loaders::ProgrammingLanguage,
    node_parser::text::{
        CodeSplitter, MarkdownNodeParser, SemanticSplitter, SentenceSplitter,
        SentenceWindowNodeParser, TokenTextSplitter,
    },
};
use std::sync::Arc;
use tracing::{info, Level};

// Mock embedder for SemanticSplitter
#[derive(Debug)]
struct MockEmbedder {
    dimension: usize,
}

impl MockEmbedder {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait::async_trait]
impl cheungfun_core::traits::Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> CoreResult<Vec<f32>> {
        let _ = text;
        Ok(vec![0.1; self.dimension])
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> CoreResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.1; self.dimension]).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        "mock-embedder"
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    println!("🔍 Transform Trait Verification");
    println!("===============================");
    println!();

    // Create test document
    let test_document = Document::new(
        r#"# Introduction
This is the introduction section with some content.

## Getting Started
Here's how to get started with the system.

### Installation
Run the following command to install:
```bash
cargo install example
```

## Configuration
This section covers configuration options.

The system supports various configuration parameters.
You can set them in the config file.
"#,
    );

    // Create all parser instances
    let transforms: Vec<(String, Box<dyn Transform>)> = vec![
        (
            "SentenceSplitter".to_string(),
            Box::new(SentenceSplitter::from_defaults(300, 75)?),
        ),
        (
            "TokenTextSplitter".to_string(),
            Box::new(TokenTextSplitter::from_defaults(250, 50)?),
        ),
        (
            "MarkdownNodeParser".to_string(),
            Box::new(MarkdownNodeParser::new()),
        ),
        (
            "CodeSplitter".to_string(),
            Box::new(CodeSplitter::from_defaults(
                ProgrammingLanguage::Rust,
                40,
                10,
                1500,
            )?),
        ),
        (
            "SemanticSplitter".to_string(),
            Box::new(SemanticSplitter::new(Arc::new(MockEmbedder::new(384)))),
        ),
        (
            "SentenceWindowNodeParser".to_string(),
            Box::new(SentenceWindowNodeParser::new()),
        ),
    ];

    println!(
        "📊 Testing {} parsers with Transform trait",
        transforms.len()
    );
    println!();

    // Test each parser
    for (name, transform) in transforms {
        info!("Testing {}", name);

        // Test single document
        let input = TransformInput::Document(test_document.clone());
        match transform.transform(input).await {
            Ok(nodes) => {
                println!("✅ {}: {} nodes generated", name, nodes.len());

                // Show first node preview
                if let Some(first_node) = nodes.first() {
                    let preview = first_node.content.chars().take(100).collect::<String>();
                    println!("   Preview: {}...", preview.replace('\n', " "));
                }
            }
            Err(e) => {
                println!("❌ {}: Error - {}", name, e);
            }
        }

        // Test batch processing
        let batch_input = TransformInput::Documents(vec![
            test_document.clone(),
            Document::new("Another test document with different content."),
        ]);

        match transform.transform(batch_input).await {
            Ok(nodes) => {
                println!("✅ {} (batch): {} nodes generated", name, nodes.len());
            }
            Err(e) => {
                println!("❌ {} (batch): Error - {}", name, e);
            }
        }

        println!();
    }

    println!("🎉 Transform trait verification completed!");
    println!();
    println!(
        "All parsers successfully implement the Transform trait and can be used polymorphically."
    );

    Ok(())
}
