# Cheungfun Indexing

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Cheungfun-indexing** is the core indexing module of the Cheungfun RAG framework, providing document loading, parsing, splitting, and transformation functionalities. It is designed with a unified `Transform` interface, following the architectural patterns of LlamaIndex.

> **⚠️ Learning Project Disclaimer**: This is a personal learning project for exploring RAG architecture design in Rust. Although it is relatively feature-complete, it is still under active development and is **not recommended for production use**.

## 🚀 Quick Start

```rust
use cheungfun_core::{Document, traits::Transform};
use cheungfun_indexing::node_parser::text::SentenceSplitter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a sentence splitter
    let splitter = SentenceSplitter::from_defaults(300, 75)?;

    // Create a document
    let document = Document::new("This is the first sentence. This is the second sentence. This is the third sentence.");

    // Use parse_nodes() for direct parsing
    let nodes = splitter.parse_nodes(&[document], false).await?;

    // Alternative: Use the unified Transform interface
    let input = cheungfun_core::traits::TransformInput::Documents(vec![document]);
    let nodes = splitter.transform(input).await?;

    println!("Generated {} nodes", nodes.len());
    Ok(())
}
```

### 🎯 Async-First Design

Cheungfun follows an **async-first design pattern** where all parsing operations are asynchronous. This provides better performance and composability:

```rust
// ✅ Async context (recommended)
let nodes = splitter.parse_nodes(&documents, false).await?;

// ✅ Sync context (requires manual runtime)
let rt = tokio::runtime::Runtime::new()?;
let nodes = rt.block_on(splitter.parse_nodes(&documents, false))?;
````

## 📚 Supported Parser Types

### 🔤 Text Parsers

#### 1. SentenceSplitter

**Usage**: Intelligent text splitting based on sentence boundaries, prioritizing sentence integrity.

```rust
use cheungfun_indexing::node_parser::text::SentenceSplitter;

// Basic usage
let splitter = SentenceSplitter::from_defaults(300, 75)?;

// Advanced configuration
let splitter = SentenceSplitter::new()
    .with_chunk_size(500)
    .with_chunk_overlap(100)
    .with_separator("\n\n")
    .with_backup_separators(vec!["\n", " ", ""]);
```

#### 2. TokenTextSplitter

**Usage**: Precise splitting based on the number of tokens, ideal for scenarios with LLM token limits.

```rust
use cheungfun_indexing::node_parser::text::TokenTextSplitter;

// Basic usage
let splitter = TokenTextSplitter::from_defaults(250, 50)?;

// Custom encoder
let splitter = TokenTextSplitter::new()
    .with_chunk_size(512)
    .with_chunk_overlap(64)
    .with_encoding_name("cl100k_base"); // GPT-4 encoding
```

#### 3. MarkdownNodeParser

**Usage**: Hierarchical document splitting based on Markdown header structure.

#### 4. HierarchicalNodeParser

**Usage**: Multi-level hierarchical chunking with parent-child relationships for advanced retrieval patterns.

```rust
use cheungfun_indexing::node_parser::relational::HierarchicalNodeParser;

// Create 3-level hierarchy: 2048 -> 512 -> 128 tokens
let parser = HierarchicalNodeParser::from_defaults(vec![2048, 512, 128])?;

// Advanced configuration
let parser = HierarchicalNodeParser::new(
    HierarchicalConfig::new(vec![1024, 256, 64])
        .with_chunk_overlap(50)
        .with_hierarchy_metadata(true)
)?;
```

## 🗂️ File Format Parsers

### HTMLNodeParser

**Usage**: Extract structured text from HTML documents while preserving tag information.

```rust
use cheungfun_indexing::node_parser::file::HTMLNodeParser;

// Basic usage with default tags
let parser = HTMLNodeParser::new();

// Custom tags
let parser = HTMLNodeParser::with_tags(vec![
    "h1".to_string(), "h2".to_string(), "p".to_string()
]);
```

### JSONNodeParser

**Usage**: Parse JSON documents into structured nodes with hierarchical metadata.

```rust
use cheungfun_indexing::node_parser::file::JSONNodeParser;

// Basic usage
let parser = JSONNodeParser::new();

// Custom configuration
let parser = JSONNodeParser::with_config(
    JSONConfig::default()
        .with_create_object_nodes(true)
        .with_include_paths(true)
        .with_max_depth(5)
);
```

```rust
use cheungfun_indexing::node_parser::text::MarkdownNodeParser;

// Basic usage
let parser = MarkdownNodeParser::new();

// Advanced configuration
let parser = MarkdownNodeParser::new()
    .with_max_header_depth(3)           // Max header depth
    .with_header_path_separator(" > ")  // Path separator
    .with_min_section_length(50)        // Min section length
    .with_include_header_in_content(true);

// Preset configurations
let doc_parser = MarkdownNodeParser::from_config(
    MarkdownConfig::for_documentation()
)?;
let blog_parser = MarkdownNodeParser::from_config(
    MarkdownConfig::for_blog_posts()
)?;
```

#### 4. SentenceWindowNodeParser

**Usage**: Creates a node for each sentence while preserving the context of surrounding sentences in the metadata.

```rust
use cheungfun_indexing::node_parser::text::SentenceWindowNodeParser;

// Basic usage
let parser = SentenceWindowNodeParser::new()
    .with_window_size(2)                    // 2 sentences before and after
    .with_window_metadata_key("context")    // Custom metadata key
    .with_original_text_metadata_key("sentence");
```

#### 5. SemanticSplitter

**Usage**: Intelligent splitting based on semantic similarity, grouping semantically related sentences.

```rust
use cheungfun_indexing::node_parser::text::SemanticSplitter;

// Requires an embedder
let embedder = Arc::new(your_embedder);
let splitter = SemanticSplitter::new(embedder)
    .with_buffer_size(1)
    .with_breakpoint_percentile_threshold(95.0)
    .with_number_of_chunks(None);
```

### 💻 Code Parsers

#### CodeSplitter - AST-Enhanced Code Splitter with SweepAI Optimization

**Usage**: Intelligent code splitting based on Abstract Syntax Trees (AST) to maintain code structural integrity. Uses SweepAI-enhanced chunking algorithm for optimal code analysis.

```rust
use cheungfun_indexing::node_parser::text::CodeSplitter;
use cheungfun_indexing::loaders::ProgrammingLanguage;

// Basic usage
let splitter = CodeSplitter::from_defaults(
    ProgrammingLanguage::Rust,
    500,  // chunk_size
    100   // chunk_overlap
)?;

// Advanced configuration
let splitter = CodeSplitter::new()
    .with_language(ProgrammingLanguage::Python)
    .with_chunk_size(800)
    .with_chunk_overlap(150)
    .with_respect_function_boundaries(true)
    .with_respect_class_boundaries(true)
    .with_preserve_indentation(true);
```

**Supported Programming Languages**:

  - **Systems**: Rust, C, C++, Go
  - **Application**: Python, JavaScript, TypeScript, Java, C\#
  - **Functional**: Haskell, Clojure, Erlang, Elixir, Scala
  - **Scripting**: PHP, Ruby, Lua
  - **Mobile**: Swift, Kotlin
  - **Markup**: HTML, CSS, SQL

## 🔧 Loaders

### FileLoader

```rust
use cheungfun_indexing::loaders::FileLoader;

let loader = FileLoader::new("document.txt")?;
let documents = loader.load().await?;
```

### DirectoryLoader

```rust
use cheungfun_indexing::loaders::DirectoryLoader;

let loader = DirectoryLoader::new("./docs")?
    .with_recursive(true)
    .with_file_filter(|path| path.extension() == Some("md".as_ref()));
```

### CodeLoader

```rust
use cheungfun_indexing::loaders::{CodeLoader, ProgrammingLanguage};

let loader = CodeLoader::new("./src", ProgrammingLanguage::Rust)?
    .with_extract_functions(true)
    .with_extract_classes(true)
    .with_extract_imports(true);
```

### WebLoader

```rust
use cheungfun_indexing::loaders::WebLoader;

let loader = WebLoader::new("[https://example.com](https://example.com)")?
    .with_max_depth(2)
    .with_follow_links(true);
```

## 🔄 Transformers

### MetadataExtractor

```rust
use cheungfun_indexing::transformers::MetadataExtractor;

let extractor = MetadataExtractor::new()
    .with_extract_keywords(true)
    .with_extract_summary(true)
    .with_extract_entities(true);
```

## 🏗️ Pipeline System

### DefaultIndexingPipeline

```rust
use cheungfun_indexing::pipeline::DefaultIndexingPipeline;
use std::sync::Arc;

let pipeline = DefaultIndexingPipeline::builder()
    .with_loader(Arc::new(DirectoryLoader::new("./docs")?))
    .with_transformer(Arc::new(SentenceSplitter::from_defaults(300, 75)?))
    .with_transformer(Arc::new(MetadataExtractor::new()))
    .build()?;

let stats = pipeline.run().await?;
println!("Processed {} documents, created {} nodes", 
         stats.documents_processed, stats.nodes_created);
```

## 📊 Configuration System

### Unified Configuration Base

All parsers are based on a unified configuration system:

```rust
use cheungfun_indexing::node_parser::config::*;

// Text splitter configuration
let config = TextSplitterConfig::new()
    .with_chunk_size(400)
    .with_chunk_overlap(80)
    .with_include_metadata(true);

// Markdown configuration
let md_config = MarkdownConfig::new()
    .with_max_header_depth(4)
    .with_preserve_header_hierarchy(true);

// Code splitter configuration
let code_config = CodeSplitterConfig::new()
    .with_language(ProgrammingLanguage::Rust)
    .with_respect_function_boundaries(true);
```

## 🎯 Use Cases

### 📖 Document Processing

  - **Technical Docs**: Use `MarkdownNodeParser` for READMEs and API documentation.
  - **Long-form Text**: Use `SentenceSplitter` to maintain semantic integrity.
  - **Q\&A Systems**: Use `SentenceWindowNodeParser` to provide precise context.

### 💻 Code Analysis

  - **Codebase Indexing**: Use `CodeSplitter` + `CodeLoader`.
  - **Function-level Retrieval**: Enable AST parsing to extract functions and classes.
  - **Multi-language Support**: Supports 20+ programming languages.

### 🌐 Web Content

  - **Website Crawling**: Use `WebLoader` + `SentenceSplitter`.
  - **Content Cleaning**: Automatically filters HTML tags and noise.

## 📈 Performance Features

  - **Zero-copy**: Efficient memory management via Rust's ownership system.
  - **Parallel Processing**: Supports batch processing and parallel transformations.
  - **Streaming**: Memory-friendly handling of large files.
  - **Cache Optimization**: Smart caching to reduce redundant computations.

## 🔗 Related Documents

  - [Architecture Document](./docs/INDEXING_ARCHITECTURE.md) - Detailed architectural design.
  - [Examples](/examples/) - Complete usage examples.
  - [API Documentation](https://docs.rs/cheungfun-indexing) - Full API reference.

## 🎨 Advanced Usage

### Polymorphic Handling

```rust
use cheungfun_core::traits::{Transform, TransformInput};

// Polymorphic advantage of the unified interface
let transforms: Vec<Box<dyn Transform>> = vec![
    Box::new(SentenceSplitter::from_defaults(200, 40)?),
    Box::new(TokenTextSplitter::from_defaults(180, 35)?),
    Box::new(MarkdownNodeParser::new()),
];

let input = TransformInput::Documents(documents);
for transform in transforms {
    let nodes = transform.transform(input.clone()).await?;
    println!("Transform {}: {} nodes", transform.name(), nodes.len());
}
```

### Batch Processing Optimization

```rust
// Batch process multiple documents
let inputs = documents.into_iter()
    .map(TransformInput::Document)
    .collect();

let all_nodes = splitter.transform_batch(inputs).await?;
```

### Custom Filters

```rust
use cheungfun_indexing::loaders::filter::{FileFilter, FilterConfig};

let filter = FileFilter::new()
    .with_extensions(&["rs", "py", "js"])
    .with_max_file_size(1024 * 1024) // 1MB
    .with_gitignore_support(true);

let loader = DirectoryLoader::new("./src")?
    .with_filter(filter);
```

## 🛠️ Best Practices

### 1. Choosing the Right Splitter

| Scenario | Recommended Splitter | Reason |
|---|---|---|
| General Documents | `SentenceSplitter` | Maintains semantic integrity. |
| LLM Applications | `TokenTextSplitter` | Precisely controls token count. |
| Markdown Docs | `MarkdownNodeParser` | Preserves document structure. |
| Q\&A Systems | `SentenceWindowNodeParser` | Provides precise context. |
| Code Analysis | `CodeSplitter` | Preserves code structure. |
| Semantic Search | `SemanticSplitter` | Groups by semantic relevance. |

### 2. Parameter Tuning Guide

#### Chunk Size Selection

```rust
// Choose the appropriate chunk size based on the use case
let qa_splitter = SentenceSplitter::from_defaults(300, 50)?;     // Q&A system
let summary_splitter = SentenceSplitter::from_defaults(800, 100)?; // Summary generation
let search_splitter = SentenceSplitter::from_defaults(500, 75)?;   // Semantic search
```

#### Overlap Strategy

```rust
// High overlap: better contextual continuity, but more redundancy
let high_overlap = SentenceSplitter::from_defaults(400, 120)?; // 30% overlap

// Low overlap: less redundancy, but may lose context
let low_overlap = SentenceSplitter::from_defaults(400, 40)?;   // 10% overlap
```

### 3. Error Handling Patterns

```rust
use cheungfun_indexing::error::IndexingError;

match splitter.transform(input).await {
    Ok(nodes) => println!("Successfully generated {} nodes", nodes.len()),
    Err(IndexingError::InvalidInput { message }) => {
        eprintln!("Input error: {}", message);
    },
    Err(IndexingError::ProcessingError { source, context }) => {
        eprintln!("Processing error: {} (context: {})", source, context);
    },
    Err(e) => eprintln!("Other error: {}", e),
}
```

### 4. Memory Optimization

```rust
// For large files, use streaming
let loader = DirectoryLoader::new("./large_docs")?
    .with_batch_size(10)  // Batch processing
    .with_memory_limit(512 * 1024 * 1024); // 512MB memory limit

// Release unnecessary data promptly
drop(documents); // Explicitly free document memory
```

## 🔍 Debugging and Monitoring

### Enable Verbose Logging

```rust
use tracing::{info, debug, Level};
use tracing_subscriber;

// Initialize logger
tracing_subscriber::fmt()
    .with_max_level(Level::DEBUG)
    .init();

// Use in code
debug!("Starting to process document: {}", document.id);
info!("Generated {} nodes", nodes.len());
```

### Performance Monitoring

```rust
use std::time::Instant;

let start = Instant::now();
let nodes = splitter.transform(input).await?;
let duration = start.elapsed();

println!("Processing time: {:?}, Nodes created: {}, Speed: {:.2} nodes/sec",
         duration, nodes.len(),
         nodes.len() as f64 / duration.as_secs_f64());
```

## 🧪 Testing Support

### Unit Test Example

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::Document;

    #[tokio::test]
    async fn test_sentence_splitter() {
        let splitter = SentenceSplitter::from_defaults(100, 20).unwrap();
        let doc = Document::new("First sentence. Second sentence. Third sentence.");

        let input = TransformInput::Document(doc);
        let nodes = splitter.transform(input).await.unwrap();

        assert!(!nodes.is_empty());
        assert!(nodes.iter().all(|n| !n.content.is_empty()));
    }
}
```

## 🤝 Contribution Guide

1.  **Fork** this repository
2.  Create your feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m 'Add amazing feature'`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Open a **Pull Request**

### Development Environment Setup

```bash
# Clone the repository
git clone [https://github.com/YumchaLabs/cheungfun.git](https://github.com/YumchaLabs/cheungfun.git)
cd cheungfun

# Install dependencies
cargo build

# Run tests
cargo test -p cheungfun-indexing

# Run examples
cargo run --example basic_integration_demo
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
