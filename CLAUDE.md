# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cheungfun is a high-performance RAG (Retrieval-Augmented Generation) framework built in Rust, inspired by LlamaIndex and Swiftide. It's designed as a learning project exploring modern RAG architecture patterns with modular design, type safety, and streaming capabilities.

**⚠️ Learning Project**: This is primarily an educational project for exploring Rust and RAG system design. It's functional but not recommended for production use.

## Architecture

The project follows a modular workspace architecture with the following crates:

- **cheungfun-core**: Core traits, types, error handling, and configuration
- **cheungfun-indexing**: Document loading, parsing, and indexing pipelines
- **cheungfun-query**: Query engines, retrievers, and response generation
- **cheungfun-agents**: Intelligent agents with ReAct reasoning and MCP integration
- **cheungfun-integrations**: External service integrations (Qdrant, FastEmbed, Candle)
- **cheungfun-multimodal**: Multi-modal processing (text, images, audio, video)

### Core Design Principles

1. **Unified Transform Interface**: All processing components implement the same `Transform` trait from cheungfun-core
2. **Type Safety**: Heavy use of Rust's type system for compile-time correctness
3. **Async-First**: Built on tokio for high-performance async operations
4. **Modular Components**: Clear separation of concerns with trait-based abstractions

## Common Development Commands

### Building and Testing
```bash
# Build all workspace members
cargo build

# Build with performance features
cargo build --features performance

# Run all tests
cargo test

# Run specific test suites
cargo run --package cheungfun-tools --bin run_tests unit
cargo run --package cheungfun-tools --bin run_tests integration
cargo run --package cheungfun-tools --bin run_tests performance

# Run performance benchmarks
cargo test --features performance --test performance_integration_test

# Test coverage (requires cargo-llvm-cov)
./scripts/test_coverage.sh        # On Unix
./scripts/test_coverage.bat       # On Windows
```

### Feature Testing
```bash
# Test SIMD acceleration
cargo test --features simd test_simd_performance -- --nocapture

# Test vector store performance  
cargo test --features "hnsw,simd" test_vector_store_performance -- --nocapture

# Full performance suite
cargo test --features performance --test performance_integration_test -- --nocapture
```

### Examples
```bash
# Basic usage examples
cargo run --example basic_usage
cargo run --example hello_world

# Feature-specific examples
cargo run --features fastembed --example fastembed_demo
cargo run --features candle --example candle_embedder_demo

# Performance testing
cargo run --example performance_benchmark
```

## Key Features and Components

### Feature Flags
The project uses granular feature flags for optional functionality:

- `default`: Stable, optimized memory operations
- `simd`: SIMD-accelerated vector operations (30.17x speedup achieved)
- `hnsw`: HNSW approximate nearest neighbor search (378+ QPS)
- `performance`: All performance optimizations combined
- `candle`: Local ML model support via Candle framework
- `qdrant`: Qdrant vector database integration
- `fastembed`: FastEmbed embedding generation
- `full`: All features enabled

### Core Abstractions

#### Transform Interface
All document processing uses a unified `Transform` trait:
```rust
#[async_trait]
pub trait Transform: Send + Sync + std::fmt::Debug {
    async fn transform(&self, input: TransformInput) -> Result<Vec<Node>>;
    // ...
}
```

This includes:
- Text splitters (sentence, token, semantic)
- Metadata extractors
- Code parsers (AST-based parsing for 9+ languages)
- Document transformers

#### Pipeline Architecture
- **Indexing Pipeline**: Document loading → Transform chain → Embedding → Vector storage
- **Query Pipeline**: Query → Retrieval → Response generation

### Advanced Code Indexing

The system includes sophisticated code analysis capabilities:
- **AST Parsing**: Uses tree-sitter for syntax-aware parsing
- **Multi-language Support**: Rust, Python, JavaScript/TypeScript, Java, C#, C/C++, Go
- **Intelligent Extraction**: Functions, classes, imports, comments, complexity metrics
- **Code-aware Splitting**: Maintains syntax boundaries when chunking code

### File Filtering System
Enhanced file processing with:
- **Gitignore Support**: Automatic `.gitignore` rule application
- **Glob Patterns**: Complex pattern matching (`*.rs`, `**/*.tmp`, `src/**`)
- **Size Filtering**: File size-based filtering
- **Hidden File Handling**: Configurable hidden file exclusion

## Performance Characteristics

Recent benchmarking has revealed both strengths and optimization opportunities:

### SIMD Acceleration
- **30.17x speedup** for vector operations when SIMD is enabled
- Automatic fallback to scalar operations on unsupported hardware

### Vector Search Performance  
- **HNSW Implementation**: 378+ QPS for approximate nearest neighbor search
- **Memory Optimization**: Significant improvements in memory usage patterns

### Current Limitations (Areas for Improvement)
- Vector insert operations: ~40 ops/sec (target: >1000 ops/sec)
- Memory usage can be high for large datasets
- Some algorithms may use brute-force approaches that could be optimized

## External Dependencies

### Core Dependencies
- `siumai = "0.9.0"` - Unified LLM interface supporting multiple providers
- `tokio = "1.47"` - Async runtime with multi-thread support
- `candle-core = "0.9"` - Rust-native ML framework for local models
- `qdrant-client = "1.15"` - Vector database client
- `rmcp = "0.6"` - Model Context Protocol implementation

### Key Libraries
- `text-splitter = "0.27"` - Text chunking utilities  
- `tiktoken-rs = "0.7"` - Token counting and text encoding
- `tree-sitter` libraries - AST parsing for multiple languages
- `rayon = "1.11"` - Data parallelism
- `cached = "0.56"` - Caching with disk persistence

## Development Guidelines

### Code Organization
- Each crate has its own `lib.rs` with clear public API exports
- Common utilities are shared through `prelude` modules
- Error handling uses the `thiserror` crate with context-rich error types
- All public APIs are documented with rustdoc

### Testing Strategy
- Unit tests in individual crates
- Integration tests in `tests/` directories
- Performance benchmarks in `05_performance/benchmarks/`
- Custom test runner in `tools/src/test_runner.rs`

### Performance Testing
The project includes comprehensive performance testing:
- Embedder benchmarks
- Vector store performance tests
- End-to-end pipeline benchmarks
- Memory usage analysis
- SIMD optimization verification

### Error Handling
All components use the unified `CheungfunError` type from cheungfun-core:
```rust
pub enum CheungfunError {
    Io(#[from] std::io::Error),
    Embedding { message: String },
    VectorStore { message: String },
    // ... comprehensive error types
}
```

## Important Notes

1. **Learning Focus**: This is an educational project exploring RAG architecture patterns
2. **Not Production Ready**: While functional, it's designed for learning rather than production use
3. **Performance Experimental**: Performance optimizations are learning exercises
4. **External Services**: Some examples require external services (Qdrant, API keys)
5. **Feature Dependencies**: Many advanced features require optional cargo features to be enabled

## Useful File Locations

- **Core traits**: `cheungfun-core/src/traits/`
- **Pipeline implementations**: `cheungfun-indexing/src/pipeline.rs`, `cheungfun-query/src/pipeline.rs`
- **Examples**: `examples/` directory with comprehensive usage patterns
- **Performance tests**: `examples/05_performance/benchmarks/`
- **Architecture docs**: `docs/architecture.md`, `docs/core_interfaces.md`
- **Test utilities**: `tools/src/test_runner.rs`

## LLM Integration

The project uses the `siumai` crate for unified LLM access, supporting providers like:
- OpenAI (GPT models)
- Anthropic (Claude models)  
- Local models via various backends
- Custom API endpoints

This abstraction allows easy switching between different LLM providers without changing application code.