//! Markdown Node Parser Demo
//!
//! This example demonstrates the MarkdownNodeParser functionality,
//! showing how it parses Markdown documents based on header structure
//! while preserving hierarchical context and metadata.

use cheungfun_core::{Document, Result as CoreResult};
use cheungfun_indexing::node_parser::{
    config::MarkdownConfig, text::MarkdownNodeParser, NodeParser, TextSplitter,
};
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> CoreResult<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("📝 Markdown Node Parser Demo");
    info!("============================");

    // Create different markdown parsers with various configurations
    info!("Creating markdown parsers with different configurations...");

    let parser_default = MarkdownNodeParser::new();

    let parser_shallow = MarkdownNodeParser::new()
        .with_max_header_depth(2)
        .with_min_section_length(50);

    let parser_custom_separator = MarkdownNodeParser::new()
        .with_header_path_separator(" → ")
        .with_include_header_in_content(false);

    let parser_documentation =
        MarkdownNodeParser::from_config(MarkdownConfig::for_documentation())?;

    let parser_blog = MarkdownNodeParser::from_config(MarkdownConfig::for_blog_posts())?;

    // Sample markdown content with various structures
    let sample_markdown = r#"# Cheungfun RAG Framework

Cheungfun is a high-performance Retrieval-Augmented Generation (RAG) framework written in Rust.

## Overview

This framework provides comprehensive tools for building RAG applications with excellent performance and flexibility.

### Key Features

- **High Performance**: Built with Rust for maximum speed and safety
- **Modular Architecture**: Clean separation of concerns with well-defined interfaces
- **Multiple Embedders**: Support for FastEmbed, OpenAI, and local models

### Supported Vector Stores

The framework supports multiple vector storage backends:

1. **In-Memory Store**: Fast, SIMD-optimized storage for development
2. **Qdrant**: Production-ready vector database with clustering support

## Getting Started

Follow these steps to get started with Cheungfun:

### Installation

Add Cheungfun to your `Cargo.toml`:

```toml
[dependencies]
cheungfun = "0.1.0"
```

### Basic Usage

Here's a simple example:

```rust
use cheungfun::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let embedder = FastEmbedEmbedder::new()?;
    let vector_store = InMemoryVectorStore::new();
    
    // Build your RAG pipeline
    let pipeline = Pipeline::builder()
        .with_embedder(embedder)
        .with_vector_store(vector_store)
        .build()?;
    
    Ok(())
}
```

## Advanced Features

### Query Transformations

Cheungfun supports various query transformation techniques:

#### HyDE (Hypothetical Document Embeddings)

Generate hypothetical documents to improve retrieval quality.

#### Multi-Query Generation

Automatically generate multiple query variations for better coverage.

### Reranking

Multiple reranking strategies are available:

- **LLM-based Reranking**: Use language models for semantic reranking
- **Cross-encoder Models**: Specialized models for relevance scoring

## Performance

Cheungfun achieves exceptional performance metrics:

- **30.17x** SIMD acceleration
- **12.61x** parallel processing speedup  
- **378+ QPS** query throughput
- **P95 latency** under 100ms

## Contributing

We welcome contributions! Please see our contributing guidelines.

### Development Setup

1. Clone the repository
2. Install Rust toolchain
3. Run tests: `cargo test`

## License

This project is licensed under the MIT License.
"#;

    info!("\n📄 Sample Markdown Content:");
    info!("Document length: {} characters", sample_markdown.len());
    info!("Preview: {}...", &sample_markdown[..200]);

    // Test default parser
    info!("\n🔍 Default Parser:");
    let document = Document::new(sample_markdown);
    let nodes_default = <MarkdownNodeParser as NodeParser>::parse_nodes(
        &parser_default,
        &[document.clone()],
        false,
    )
    .await?;

    info!("Generated {} nodes:", nodes_default.len());
    for (i, node) in nodes_default.iter().take(5).enumerate() {
        let header = node
            .metadata
            .get("header")
            .map(|h| h.as_str().unwrap_or(""))
            .unwrap_or("(no header)");
        let level = node
            .metadata
            .get("header_level")
            .map(|l| l.as_u64().unwrap_or(0))
            .unwrap_or(0);
        let path = node
            .metadata
            .get("header_path")
            .map(|p| p.as_str().unwrap_or(""))
            .unwrap_or("");

        info!("Node {}: Level {} - '{}'", i + 1, level, header);
        info!("  Path: '{}'", path);
        info!("  Content length: {} chars", node.content.len());
        info!(
            "  Preview: {}...",
            node.content
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(60)
                .collect::<String>()
        );
        info!("");
    }
    if nodes_default.len() > 5 {
        info!("... and {} more nodes", nodes_default.len() - 5);
    }

    // Test shallow parser (max depth 2)
    info!("\n🔍 Shallow Parser (max_depth=2, min_length=50):");
    let nodes_shallow = <MarkdownNodeParser as NodeParser>::parse_nodes(
        &parser_shallow,
        &[document.clone()],
        false,
    )
    .await?;

    info!("Generated {} nodes:", nodes_shallow.len());
    for (i, node) in nodes_shallow.iter().take(3).enumerate() {
        let header = node
            .metadata
            .get("header")
            .map(|h| h.as_str().unwrap_or(""))
            .unwrap_or("(no header)");
        let level = node
            .metadata
            .get("header_level")
            .map(|l| l.as_u64().unwrap_or(0))
            .unwrap_or(0);

        info!("Node {}: Level {} - '{}'", i + 1, level, header);
        info!("  Content length: {} chars", node.content.len());
        info!("");
    }

    // Test custom separator parser
    info!("\n🔍 Custom Separator Parser (→, no header in content):");
    let nodes_custom = <MarkdownNodeParser as NodeParser>::parse_nodes(
        &parser_custom_separator,
        &[document.clone()],
        false,
    )
    .await?;

    info!("Generated {} nodes:", nodes_custom.len());
    for (i, node) in nodes_custom.iter().take(3).enumerate() {
        let header = node
            .metadata
            .get("header")
            .map(|h| h.as_str().unwrap_or(""))
            .unwrap_or("(no header)");
        let path = node
            .metadata
            .get("header_path")
            .map(|p| p.as_str().unwrap_or(""))
            .unwrap_or("");

        info!("Node {}: '{}'", i + 1, header);
        info!("  Path: '{}'", path);
        info!("  Starts with header: {}", node.content.starts_with('#'));
        info!("");
    }

    // Test preset configurations
    info!("\n🔍 Preset Configurations:");

    // Documentation config
    let nodes_doc = <MarkdownNodeParser as NodeParser>::parse_nodes(
        &parser_documentation,
        &[document.clone()],
        false,
    )
    .await?;
    info!(
        "Documentation config: {} nodes (max_depth=4, min_length=50)",
        nodes_doc.len()
    );

    // Blog config
    let nodes_blog =
        <MarkdownNodeParser as NodeParser>::parse_nodes(&parser_blog, &[document.clone()], false)
            .await?;
    info!(
        "Blog config: {} nodes (max_depth=3, separator=' > ')",
        nodes_blog.len()
    );
    if let Some(node) = nodes_blog.first() {
        if let Some(path) = node.metadata.get("header_path") {
            info!("  Sample path: '{}'", path.as_str().unwrap_or(""));
        }
    }

    // Test TextSplitter interface
    info!("\n📄 Testing TextSplitter Interface:");
    let sections = parser_default.split_text(sample_markdown)?;
    info!("Split into {} sections:", sections.len());
    for (i, section) in sections.iter().take(3).enumerate() {
        let preview = section
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(50)
            .collect::<String>();
        info!("  Section {}: {}...", i + 1, preview);
    }

    // Performance comparison
    info!("\n⏱️  Performance Comparison:");
    let start = std::time::Instant::now();
    let _ = <MarkdownNodeParser as NodeParser>::parse_nodes(
        &parser_default,
        &[document.clone()],
        false,
    )
    .await?;
    let markdown_time = start.elapsed();

    let sentence_splitter =
        cheungfun_indexing::node_parser::text::SentenceSplitter::from_defaults(1000, 200)?;
    let start = std::time::Instant::now();
    let _ = <cheungfun_indexing::node_parser::text::SentenceSplitter as NodeParser>::parse_nodes(
        &sentence_splitter,
        &[document],
        false,
    )
    .await?;
    let sentence_time = start.elapsed();

    info!("Markdown Parser: {:?}", markdown_time);
    info!("Sentence Parser: {:?}", sentence_time);
    info!(
        "Markdown parser is {:.2}x {} than sentence parser",
        if markdown_time > sentence_time {
            markdown_time.as_secs_f64() / sentence_time.as_secs_f64()
        } else {
            sentence_time.as_secs_f64() / markdown_time.as_secs_f64()
        },
        if markdown_time > sentence_time {
            "slower"
        } else {
            "faster"
        }
    );

    // Test edge cases
    info!("\n🧪 Testing Edge Cases:");

    // Empty markdown
    let empty_doc = Document::new("");
    let empty_nodes =
        <MarkdownNodeParser as NodeParser>::parse_nodes(&parser_default, &[empty_doc], false)
            .await?;
    info!("Empty markdown: {} nodes", empty_nodes.len());

    // Markdown without headers
    let no_headers_doc = Document::new("This is just plain text without any headers.\n\nIt has multiple paragraphs but no markdown structure.");
    let no_headers_nodes =
        <MarkdownNodeParser as NodeParser>::parse_nodes(&parser_default, &[no_headers_doc], false)
            .await?;
    info!("No headers markdown: {} nodes", no_headers_nodes.len());

    // Only headers
    let headers_only_doc = Document::new("# Header 1\n## Header 2\n### Header 3");
    let headers_only_nodes = <MarkdownNodeParser as NodeParser>::parse_nodes(
        &parser_default,
        &[headers_only_doc],
        false,
    )
    .await?;
    info!("Headers only markdown: {} nodes", headers_only_nodes.len());

    // Deep nesting
    let deep_nesting = "# L1\n## L2\n### L3\n#### L4\n##### L5\n###### L6\nContent at level 6";
    let deep_doc = Document::new(deep_nesting);
    let deep_nodes =
        <MarkdownNodeParser as NodeParser>::parse_nodes(&parser_default, &[deep_doc], false)
            .await?;
    info!("Deep nesting markdown: {} nodes", deep_nodes.len());

    info!("\n✅ Markdown Node Parser Demo completed successfully!");
    info!("\n💡 Key Features Demonstrated:");
    info!("   • Header-based document structure parsing");
    info!("   • Hierarchical metadata with configurable separators");
    info!("   • Flexible header depth and section length filtering");
    info!("   • Multiple preset configurations for different use cases");
    info!("   • Seamless TextSplitter interface integration");

    info!("\n🎯 Use Cases:");
    info!("   • Technical documentation processing");
    info!("   • Blog post and article analysis");
    info!("   • README and wiki content indexing");
    info!("   • Structured knowledge base construction");
    info!("   • Hierarchical content navigation systems");

    Ok(())
}

/// Demonstrate advanced markdown parsing features
#[allow(dead_code)]
async fn demonstrate_advanced_features() -> CoreResult<()> {
    info!("\n🚀 Advanced Markdown Features:");

    // This would demonstrate:
    // - Code block extraction and handling
    // - Table parsing and metadata extraction
    // - Link and reference processing
    // - Custom markdown extensions

    Ok(())
}

/// Compare different markdown parsing strategies
#[allow(dead_code)]
async fn compare_parsing_strategies(_content: &str) -> CoreResult<()> {
    info!("\n📈 Markdown Parsing Strategy Comparison:");

    // This would compare:
    // - Header-based splitting vs. content-based splitting
    // - Different header depth strategies
    // - Section length optimization
    // - Metadata extraction approaches

    Ok(())
}
