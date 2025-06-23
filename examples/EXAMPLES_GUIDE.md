# Cheungfun Examples Guide

Welcome to the Cheungfun examples! This guide will help you navigate through different types of examples and understand how to use the Cheungfun RAG framework effectively.

## 📁 Directory Structure

```
examples/
├── 01_getting_started/          # 🚀 Basic usage examples
│   ├── hello_world.rs           # Simplest possible example
│   ├── basic_indexing.rs        # Basic document indexing
│   ├── basic_querying.rs        # Basic query processing
│   └── README.md                # Getting started guide
│
├── 02_core_components/          # 🔧 Individual component examples
│   ├── embedders/               # Embedding examples
│   │   ├── fastembed_demo.rs
│   │   ├── api_embedder_demo.rs
│   │   └── candle_embedder_demo.rs
│   ├── vector_stores/           # Vector storage examples
│   │   ├── memory_store_demo.rs
│   │   └── qdrant_store_demo.rs
│   ├── loaders/                 # Document loading examples
│   │   ├── text_loader_demo.rs
│   │   ├── pdf_loader_demo.rs
│   │   └── web_loader_demo.rs
│   └── transformers/            # Text processing examples
│       ├── text_splitter_demo.rs
│       └── custom_transformer_demo.rs
│
├── 03_advanced_features/        # 🎯 Advanced functionality
│   ├── hybrid_search.rs         # Vector + keyword search
│   ├── reranking.rs             # Result reranking
│   ├── caching.rs               # Query caching
│   ├── custom_retriever.rs      # Custom retrieval logic
│   └── multi_modal.rs           # Multi-modal RAG
│
├── 04_integrations/             # 🔌 External integrations
│   ├── mcp_integration.rs       # Model Context Protocol
│   ├── api_integration.rs       # REST API integration
│   ├── database_integration.rs  # Database connections
│   └── streaming_demo.rs        # Streaming responses
│
├── 05_performance/              # ⚡ Performance & optimization
│   ├── benchmarks/              # Performance benchmarking
│   │   ├── embedder_benchmark.rs
│   │   ├── vector_store_benchmark.rs
│   │   ├── end_to_end_benchmark.rs
│   │   └── run_benchmarks.sh
│   ├── optimization/            # Performance optimization
│   │   ├── batch_processing.rs
│   │   ├── parallel_indexing.rs
│   │   └── memory_optimization.rs
│   └── monitoring/              # Performance monitoring
│       ├── metrics_collection.rs
│       └── health_checks.rs
│
├── 06_production/               # 🏭 Production-ready examples
│   ├── complete_rag_system.rs   # Full RAG implementation
│   ├── error_handling.rs        # Robust error handling
│   ├── configuration.rs         # Configuration management
│   └── deployment.rs            # Deployment examples
│
├── 07_use_cases/                # 💼 Real-world use cases
│   ├── document_qa.rs           # Document Q&A system
│   ├── code_search.rs           # Code search engine
│   ├── knowledge_base.rs        # Knowledge base system
│   └── chatbot.rs               # RAG-powered chatbot
│
└── shared/                      # 📚 Shared utilities
    ├── test_data/               # Sample data for examples
    ├── common.rs                # Common utilities
    ├── benchmark_framework.rs   # Benchmarking framework
    └── report_generator.rs      # Report generation
```

## 🚀 Quick Start

### 1. Hello World (5 minutes)
```bash
cargo run --bin hello_world
```
The simplest possible Cheungfun example - index a document and ask a question.

### 2. Basic Indexing (10 minutes)
```bash
cargo run --bin basic_indexing
```
Learn how to load, process, and index documents.

### 3. Basic Querying (10 minutes)
```bash
cargo run --bin basic_querying
```
Learn how to query your indexed documents and get responses.

## 📖 Learning Path

### Beginner (New to RAG)
1. `01_getting_started/hello_world.rs` - Understand the basics
2. `01_getting_started/basic_indexing.rs` - Learn document processing
3. `01_getting_started/basic_querying.rs` - Learn querying
4. `02_core_components/embedders/` - Understand embeddings
5. `02_core_components/vector_stores/` - Understand vector storage

### Intermediate (Familiar with RAG concepts)
1. `03_advanced_features/hybrid_search.rs` - Advanced search
2. `03_advanced_features/reranking.rs` - Improve result quality
3. `04_integrations/` - Connect to external systems
4. `05_performance/optimization/` - Optimize performance
5. `06_production/complete_rag_system.rs` - Production setup

### Advanced (Building production systems)
1. `05_performance/benchmarks/` - Performance testing
2. `06_production/` - Production considerations
3. `07_use_cases/` - Real-world implementations
4. Custom implementations based on your needs

## 🎯 Examples by Feature

### Embeddings
- **Local Models**: `fastembed_demo.rs`, `candle_embedder_demo.rs`
- **Cloud APIs**: `api_embedder_demo.rs`
- **Performance**: `embedder_benchmark.rs`

### Vector Storage
- **In-Memory**: `memory_store_demo.rs`
- **Persistent**: `qdrant_store_demo.rs`
- **Performance**: `vector_store_benchmark.rs`

### Search & Retrieval
- **Basic Search**: `basic_querying.rs`
- **Hybrid Search**: `hybrid_search.rs`
- **Custom Retrieval**: `custom_retriever.rs`

### Performance
- **Benchmarking**: `05_performance/benchmarks/`
- **Optimization**: `05_performance/optimization/`
- **Monitoring**: `05_performance/monitoring/`

## 🔧 Running Examples

### Prerequisites
```bash
# Basic examples (always work)
cargo run --bin hello_world

# Examples with optional features
cargo run --features fastembed --bin fastembed_demo
cargo run --features candle --bin candle_embedder_demo

# Examples requiring external services
docker run -p 6334:6334 qdrant/qdrant  # For Qdrant examples
export OPENAI_API_KEY="your-key"       # For API examples
```

### Environment Setup
```bash
# Install optional dependencies
cargo add fastembed --features onnx     # For local embeddings
cargo add candle-core candle-nn         # For Candle support

# Start external services
docker-compose up -d                    # Start all services
```

## 📊 Performance Analysis

Based on recent benchmarks, here's what to expect:

### Current Performance (Baseline)
- **Vector Insert**: ~40 ops/sec ⚠️ (Target: >1000 ops/sec)
- **Vector Search**: ~30 ops/sec ⚠️ (Target: >500 ops/sec)
- **Memory Usage**: ~36GB ❌ (Target: <1GB for 10K vectors)

### Performance Issues Identified
1. **Memory Usage**: Extremely high memory consumption
2. **Search Algorithm**: Likely using brute-force search
3. **No SIMD Optimization**: Vector operations not optimized
4. **Batch Processing**: Limited batch processing benefits

### Optimization Examples
- `05_performance/optimization/batch_processing.rs` - Improve throughput
- `05_performance/optimization/memory_optimization.rs` - Reduce memory usage
- `05_performance/optimization/parallel_indexing.rs` - Parallel processing

## 🆘 Troubleshooting

### Common Issues

**"Feature not available"**
```bash
# Add the required feature
cargo add fastembed --features onnx
cargo run --features fastembed --bin example_name
```

**"Service not available"**
```bash
# Start required services
docker run -p 6334:6334 qdrant/qdrant
```

**"API key not found"**
```bash
# Set environment variables
export OPENAI_API_KEY="your-api-key"
export SIUMAI_API_KEY="your-siumai-key"
```

**Poor Performance**
- Check `05_performance/` examples for optimization techniques
- Run benchmarks to identify bottlenecks
- Consider hardware acceleration (GPU)

## 🤝 Contributing

To add new examples:

1. Choose the appropriate directory based on complexity/purpose
2. Follow the naming convention: `feature_name.rs`
3. Include comprehensive comments and error handling
4. Add to the appropriate README.md
5. Test with different configurations

## 📞 Support

- **Documentation**: Check individual README files in each directory
- **Performance Issues**: Run benchmarks in `05_performance/benchmarks/`
- **Integration Help**: See examples in `04_integrations/`
- **Production Guidance**: Check `06_production/` examples

Happy coding with Cheungfun! 🎉
