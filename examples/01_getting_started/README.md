# Getting Started with Cheungfun

Welcome to Cheungfun! This directory contains the essential examples to get you started with the Cheungfun RAG framework.

## 🚀 Quick Start (5 minutes)

### 1. Hello World
```bash
cargo run --bin hello_world
```

The simplest possible Cheungfun example. This shows you how to:
- Create a document
- Index it in memory
- Ask questions and get answers

**Perfect for**: First-time users who want to see Cheungfun in action immediately.

### 2. Basic Indexing (10 minutes)
```bash
cargo run --bin basic_indexing
```

Learn the fundamentals of document processing:
- Loading documents from files
- Text splitting strategies
- Embedding generation
- Vector storage
- Performance monitoring

**Perfect for**: Understanding how documents become searchable.

### 3. Basic Querying (10 minutes)
```bash
cargo run --bin basic_querying
```

Master the art of information retrieval:
- Different query types
- Search parameters (top_k, similarity threshold)
- Result analysis
- Performance optimization

**Perfect for**: Learning how to get the best search results.

## 📚 What You'll Learn

### Core Concepts

1. **RAG Pipeline**: Document → Chunks → Embeddings → Vector Store → Search → Response
2. **Components**: Loaders, Transformers, Embedders, Vector Stores, Retrievers, Generators
3. **Configuration**: How to tune parameters for your use case

### Key Skills

- **Document Processing**: How to prepare your data for RAG
- **Search Optimization**: Getting relevant results quickly
- **Performance Tuning**: Making your system fast and efficient

## 🎯 Learning Path

### Complete Beginner (New to RAG)
1. `hello_world.rs` - See the magic happen
2. `basic_indexing.rs` - Understand document processing
3. `basic_querying.rs` - Master information retrieval
4. Move to `../02_core_components/` for deeper understanding

### Familiar with RAG Concepts
1. `hello_world.rs` - See Cheungfun's approach
2. `basic_indexing.rs` - Learn Cheungfun's indexing pipeline
3. Skip to `../03_advanced_features/` for sophisticated features

### Experienced Developer
1. Skim `hello_world.rs` for API overview
2. Jump to `../05_performance/` for optimization
3. Check `../06_production/` for deployment patterns

## 🔧 Prerequisites

### Required
- Rust 1.75+
- Basic understanding of async/await

### Optional (for advanced examples)
- Docker (for external services)
- API keys (for cloud embeddings)

## 📊 Performance Expectations

Based on recent benchmarks with the current implementation:

### Current Performance (Baseline)
- **Indexing**: ~40 documents/sec
- **Search**: ~30 queries/sec
- **Memory**: High usage (optimization needed)

### What's Normal
- **Small datasets** (<1K docs): Sub-second responses
- **Medium datasets** (1K-10K docs): 1-3 second responses
- **Large datasets** (>10K docs): May need optimization

### Performance Issues
If you see very slow performance or high memory usage, check:
1. `../05_performance/benchmarks/` - Run performance tests
2. `../05_performance/optimization/` - Apply optimizations
3. Consider using persistent storage (Qdrant) instead of in-memory

## 🚨 Common Issues

### "No such file or directory"
Make sure you're in the examples directory:
```bash
cd examples
cargo run --bin hello_world
```

### "Feature not enabled"
Some examples require optional features:
```bash
# For FastEmbed examples
cargo run --features fastembed --bin example_name

# For Candle examples  
cargo run --features candle --bin example_name
```

### "Service unavailable"
External services need to be running:
```bash
# For Qdrant examples
docker run -p 6334:6334 qdrant/qdrant

# For API examples
export OPENAI_API_KEY="your-key"
```

### Poor Performance
This is expected with the current implementation. See:
- `../05_performance/` for optimization techniques
- Performance analysis in the examples shows current bottlenecks

## 🎓 Next Steps

After completing these examples:

### Learn More Components
- `../02_core_components/embedders/` - Different embedding options
- `../02_core_components/vector_stores/` - Storage alternatives
- `../02_core_components/loaders/` - Document loading strategies

### Advanced Features
- `../03_advanced_features/hybrid_search.rs` - Combine vector + keyword search
- `../03_advanced_features/reranking.rs` - Improve result quality
- `../03_advanced_features/caching.rs` - Speed up repeated queries

### Production Ready
- `../06_production/complete_rag_system.rs` - Full implementation
- `../06_production/error_handling.rs` - Robust error handling
- `../06_production/configuration.rs` - Configuration management

### Real-World Use Cases
- `../07_use_cases/document_qa.rs` - Document Q&A system
- `../07_use_cases/knowledge_base.rs` - Knowledge base search
- `../07_use_cases/chatbot.rs` - RAG-powered chatbot

## 💡 Tips for Success

1. **Start Simple**: Begin with `hello_world.rs` even if you're experienced
2. **Read the Code**: The examples are heavily commented - read them!
3. **Experiment**: Modify parameters and see what happens
4. **Measure Performance**: Use the benchmarking tools to understand your system
5. **Ask Questions**: Check the main documentation for detailed explanations

## 🤝 Getting Help

- **Examples not working?** Check the troubleshooting section above
- **Performance issues?** Run the benchmarks in `../05_performance/`
- **Need advanced features?** Explore `../03_advanced_features/`
- **Production deployment?** See `../06_production/`

Happy learning with Cheungfun! 🎉
