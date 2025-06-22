# Cheungfun Query

Query processing and retrieval engine for the Cheungfun RAG framework.

## Overview

The `cheungfun-query` crate provides comprehensive query processing capabilities for Retrieval-Augmented Generation (RAG) applications. It includes components for retrieving relevant content, generating responses using LLMs, and orchestrating complete query workflows.

## Features

- **Multiple Search Modes**: Vector, keyword, and hybrid search capabilities
- **Streaming Responses**: Real-time response generation with streaming support
- **Flexible Architecture**: Modular design with pluggable components
- **LLM Integration**: Built-in support for multiple LLM providers via Siumai
- **Context Management**: Conversation history and user context handling
- **Caching**: Built-in query response caching with TTL support
- **Query Optimization**: Query preprocessing and enhancement
- **Response Processing**: Post-processing with citation formatting and quality scoring

## Architecture

The query module follows a modular architecture with the following components:

```
Query → Retriever → Vector Store
  ↓
Generator → LLM → Response
```

### Core Components

- **Retrievers**: Find relevant nodes based on queries
  - `VectorRetriever`: Vector similarity search with hybrid capabilities
- **Generators**: LLM-based response generation
  - `SiumaiGenerator`: Multi-provider LLM integration
- **Query Engines**: High-level query processing
  - `QueryEngine`: Combines retrieval and generation
- **Pipelines**: Complete workflow orchestration
  - `DefaultQueryPipeline`: Full RAG pipeline with context management

## Quick Start

### Basic Usage

```rust
use cheungfun_query::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Set up components
    let embedder = Arc::new(your_embedder);
    let vector_store = Arc::new(your_vector_store);
    let siumai_client = Siumai::builder().openai().build().await?;

    // Create retriever and generator
    let retriever = Arc::new(VectorRetriever::new(vector_store, embedder));
    let generator = Arc::new(SiumaiGenerator::new(siumai_client));

    // Create query engine
    let query_engine = QueryEngine::new(retriever, generator);

    // Execute query
    let response = query_engine.query("What is machine learning?").await?;
    println!("Answer: {}", response.response.content);

    Ok(())
}
```

### Advanced Configuration

```rust
use cheungfun_query::prelude::*;

// Configure retriever
let retriever_config = VectorRetrieverConfig {
    default_top_k: 10,
    max_top_k: 50,
    enable_reranking: true,
    ..Default::default()
};

let retriever = VectorRetriever::builder()
    .vector_store(vector_store)
    .embedder(embedder)
    .config(retriever_config)
    .build()?;

// Configure generator
let generator_config = SiumaiGeneratorConfig {
    default_temperature: 0.7,
    default_max_tokens: 1000,
    include_citations: true,
    ..Default::default()
};

let generator = SiumaiGenerator::builder()
    .client(siumai_client)
    .config(generator_config)
    .build()?;

// Configure query engine
let engine_config = QueryEngineConfig {
    default_top_k: 5,
    min_context_nodes: 2,
    enable_query_preprocessing: true,
    ..Default::default()
};

let query_engine = QueryEngine::builder()
    .retriever(retriever)
    .generator(generator)
    .config(engine_config)
    .build()?;
```

### Streaming Responses

```rust
use futures::StreamExt;

let pipeline = DefaultQueryPipeline::new(retriever, generator);
let options = QueryOptions::default();

let mut stream = pipeline.query_stream("Explain quantum computing", &options).await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(text) => print!("{}", text),
        Err(e) => eprintln!("Stream error: {}", e),
    }
}
```

### Search Modes

```rust
// Vector search (default)
let query = Query::new("machine learning")
    .with_search_mode(SearchMode::Vector);

// Keyword search
let query = Query::new("machine learning")
    .with_search_mode(SearchMode::Keyword);

// Hybrid search (70% vector, 30% keyword)
let query = Query::new("machine learning")
    .with_search_mode(SearchMode::hybrid(0.7));
```

### Caching

```rust
use std::time::Duration;

// Create cache with 1-hour TTL
let cache = QueryCache::new(Duration::from_secs(3600));

// Check cache before querying
if let Some(cached_response) = cache.get("What is AI?") {
    println!("Cached: {}", cached_response.content);
} else {
    let response = query_engine.query("What is AI?").await?;
    cache.put("What is AI?", response.response.clone());
    println!("Fresh: {}", response.response.content);
}
```

## Configuration

### Retriever Configuration

```rust
VectorRetrieverConfig {
    default_top_k: 10,              // Default number of results
    max_top_k: 100,                 // Maximum allowed results
    default_similarity_threshold: Some(0.7), // Similarity threshold
    enable_query_expansion: false,   // Query expansion
    enable_reranking: false,        // Result reranking
    timeout_seconds: Some(30),      // Operation timeout
}
```

### Generator Configuration

```rust
SiumaiGeneratorConfig {
    default_model: Some("gpt-4".to_string()), // Default model
    default_temperature: 0.7,       // Generation temperature
    default_max_tokens: 1000,       // Max response tokens
    include_citations: true,        // Include source citations
    max_context_length: 8000,       // Max context length
    timeout_seconds: 60,            // Generation timeout
}
```

### Pipeline Configuration

```rust
QueryPipelineConfig {
    enable_conversation_history: true,    // Track conversation
    max_conversation_turns: 10,           // Max turns to keep
    enable_query_rewriting: false,        // Query rewriting
    enable_context_compression: false,    // Context compression
    max_total_context_length: 16000,      // Max total context
    enable_response_caching: false,       // Response caching
    cache_ttl_seconds: 3600,              // Cache TTL
}
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_query.rs`: Basic query engine setup and usage
- More examples coming soon...

## Testing

Run the test suite:

```bash
cargo test -p cheungfun-query
```

Run with logging:

```bash
RUST_LOG=debug cargo test -p cheungfun-query
```

## Dependencies

- `cheungfun-core`: Core types and traits
- `siumai`: LLM integration
- `tokio`: Async runtime
- `tracing`: Logging and instrumentation
- `futures`: Async utilities
- `uuid`: Unique identifiers
- `serde_json`: JSON serialization

## Contributing

Contributions are welcome! Please see the main project README for contribution guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
