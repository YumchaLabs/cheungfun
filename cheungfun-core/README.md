# Cheungfun Core

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Core traits, types, and interfaces for the Cheungfun RAG framework**

Cheungfun-core provides the foundational building blocks for building RAG (Retrieval-Augmented Generation) applications in Rust. It defines the core traits, data structures, and interfaces that all other Cheungfun modules build upon.

> **⚠️ Learning Project Disclaimer**: This is a personal learning project for exploring RAG architecture design in Rust. While functionally complete, it is still under development and **not recommended for production use**.

## 🚀 Features

### 🏗️ Core Architecture
- **Unified Transform Interface**: All processing components implement the same `Transform` trait
- **Type-Safe Pipeline**: Compile-time type checking with `TransformInput` enum
- **Async-First Design**: Built on tokio for high-performance async operations
- **Builder Patterns**: Fluent APIs for constructing documents, nodes, and queries

### 📊 Data Structures
- **Document**: Rich document representation with metadata and relationships
- **Node**: Processed document chunks with embeddings and context
- **Query**: Flexible query structure with multiple search modes
- **Response**: Comprehensive response with generation metadata and sources

### 🔧 Core Traits
- **Transform**: Unified interface for all document/node processing
- **Embedder**: Generate vector representations of text
- **VectorStore**: Persist and search vector embeddings
- **Retriever**: Find relevant content for queries
- **ResponseGenerator**: Generate responses using LLMs
- **Loader**: Load documents from various sources

### ⚙️ Configuration & Factory
- **Type-Safe Configuration**: Structured configuration for all components
- **Factory Pattern**: Dynamic component creation and registration
- **Builder Support**: Fluent APIs for complex object construction

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
cheungfun-core = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
```

## 🚀 Quick Start

### Basic Document and Node Creation

```rust
use cheungfun_core::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create a document
    let doc = Document::builder()
        .content("This is a sample document about machine learning.")
        .metadata("source", "example.txt")
        .metadata("category", "AI")
        .build();

    // Create a node from the document
    let node = Node::builder()
        .content("Machine learning is a subset of artificial intelligence.")
        .document_id(doc.id())
        .metadata("chunk_index", "0")
        .build();

    println!("Document ID: {}", doc.id());
    println!("Node content: {}", node.content());
    
    Ok(())
}
```

### Query Construction

```rust
use cheungfun_core::prelude::*;

// Basic text query
let query = Query::new("What is machine learning?");

// Query with search mode
let query = Query::builder()
    .text("What is machine learning?")
    .search_mode(SearchMode::hybrid(0.7)) // 70% vector, 30% keyword
    .top_k(10)
    .build();

// Query with metadata filters
let query = Query::builder()
    .text("AI applications")
    .metadata_filter("category", "AI")
    .metadata_filter("language", "en")
    .build();
```

### Transform Interface Usage

```rust
use cheungfun_core::traits::{Transform, TransformInput};
use async_trait::async_trait;

// Example transformer implementation
struct SentenceSplitter {
    chunk_size: usize,
    overlap: usize,
}

#[async_trait]
impl Transform for SentenceSplitter {
    async fn transform(&self, input: TransformInput) -> Result<Vec<Node>> {
        match input {
            TransformInput::Documents(docs) => {
                // Split documents into nodes
                let mut nodes = Vec::new();
                for doc in docs {
                    // Split logic here
                    let node = Node::from_document(&doc, 0, doc.content().len());
                    nodes.push(node);
                }
                Ok(nodes)
            }
            TransformInput::Nodes(nodes) => {
                // Pass through nodes unchanged
                Ok(nodes)
            }
        }
    }
}

// Usage
let splitter = SentenceSplitter { chunk_size: 512, overlap: 50 };
let documents = vec![doc];
let nodes = splitter.transform(TransformInput::Documents(documents)).await?;
```

## 🏗️ Architecture

### Core Components

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Document     │    │      Node       │    │     Query       │
│                 │    │                 │    │                 │
│ - content       │    │ - content       │    │ - text          │
│ - metadata      │    │ - embedding     │    │ - filters       │
│ - relationships │    │ - metadata      │    │ - search_mode   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Transform     │
                    │                 │
                    │ Documents  →    │
                    │   Nodes         │
                    └─────────────────┘
```

### Trait Hierarchy

- **Transform**: Unified processing interface
- **Embedder**: Vector generation
- **VectorStore**: Vector persistence and search
- **Retriever**: Content retrieval
- **ResponseGenerator**: LLM-based generation
- **Loader**: Document loading

## 📚 Core Modules

### Types (`types/`)
- `document.rs`: Document data structure and builder
- `node.rs`: Node data structure and operations
- `query.rs`: Query types and search modes
- `response.rs`: Response types and metadata

### Traits (`traits/`)
- `transformer.rs`: Unified Transform interface
- `embedder.rs`: Embedding generation
- `retriever.rs`: Content retrieval
- `generator.rs`: Response generation
- `loader.rs`: Document loading

### Configuration (`config/`)
- Type-safe configuration structures
- JSON/TOML serialization support
- Environment variable integration

### Factory (`factory/`)
- Dynamic component creation
- Registry pattern implementation
- Plugin architecture support

## 🔧 Configuration

```rust
use cheungfun_core::config::*;

// Embedder configuration
let embedder_config = EmbedderConfig {
    model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
    batch_size: 32,
    max_length: 512,
    normalize: true,
};

// Vector store configuration
let vector_config = VectorStoreConfig {
    dimension: 384,
    metric: DistanceMetric::Cosine,
    index_type: "hnsw".to_string(),
    ef_construction: 200,
    m: 16,
};
```

## 🧪 Testing

Run the test suite:

```bash
cargo test -p cheungfun-core
```

Run with logging:

```bash
RUST_LOG=debug cargo test -p cheungfun-core
```

## 📖 Examples

See the `examples/` directory for complete working examples:

- Basic document and node creation
- Transform interface implementation
- Configuration management
- Builder pattern usage

## 🔗 Related Crates

- **[cheungfun-indexing](../cheungfun-indexing)**: Document loading and processing
- **[cheungfun-query](../cheungfun-query)**: Query processing and retrieval
- **[cheungfun-agents](../cheungfun-agents)**: Agent framework and MCP integration
- **[cheungfun-integrations](../cheungfun-integrations)**: External service integrations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
