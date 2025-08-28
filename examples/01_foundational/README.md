# 🌱 Foundational RAG Techniques

This directory contains foundational RAG (Retrieval-Augmented Generation) techniques that form the building blocks of more advanced systems. These examples are perfect for beginners and provide a solid understanding of core RAG concepts.

## 📚 Examples Overview

### 1. Simple RAG (`simple_rag.rs`)
**Concept**: Basic RAG implementation with document loading, chunking, embedding, and retrieval.

**Key Learning Points**:
- Document loading and preprocessing
- Text chunking strategies
- Vector embeddings and storage
- Basic similarity search
- Response generation with context

**Use Cases**: 
- Getting started with RAG
- Understanding the basic RAG pipeline
- Prototyping simple Q&A systems

### 2. CSV RAG (`csv_rag.rs`)
**Concept**: RAG system specifically designed for structured CSV data.

**Key Learning Points**:
- Handling structured data in RAG
- CSV parsing and processing
- Metadata extraction from structured data
- Query-to-data mapping

**Use Cases**:
- Customer data analysis
- Business intelligence Q&A
- Structured data exploration

### 3. Reliable RAG (`reliable_rag.rs`)
**Concept**: Enhanced RAG with validation and quality checks.

**Key Learning Points**:
- Document relevance validation
- Response quality assessment
- Error handling and fallbacks
- Confidence scoring

**Use Cases**:
- Production RAG systems
- High-accuracy requirements
- Quality-assured responses

### 4. Chunk Size Optimization (`chunk_size_optimization.rs`)
**Concept**: Experimenting with different chunk sizes to find optimal performance.

**Key Learning Points**:
- Impact of chunk size on retrieval quality
- Performance vs. accuracy trade-offs
- Benchmarking different configurations
- Optimal chunk size selection

**Use Cases**:
- System optimization
- Performance tuning
- Domain-specific adaptation

### 5. Proposition Chunking (`proposition_chunking.rs`)
**Concept**: Breaking text into meaningful, self-contained propositions.

**Key Learning Points**:
- Semantic chunking strategies
- Proposition extraction
- Context preservation
- Quality-based chunking

**Use Cases**:
- Knowledge extraction
- Fact-based Q&A systems
- High-precision retrieval

## 🚀 Getting Started

### Prerequisites

1. **Environment Setup**:
   ```bash
   # Set your API keys (choose one or more)
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```

2. **Dependencies**: All examples use the Cheungfun framework with these key components:
   - `cheungfun-core`: Core traits and types
   - `cheungfun-indexing`: Document loading and processing
   - `cheungfun-query`: Query processing and generation
   - `cheungfun-integrations`: FastEmbed, Qdrant, and other integrations

### Running Examples

Each example can be run independently:

```bash
# Run simple RAG example
cargo run --bin simple_rag --features fastembed

# Run CSV RAG example
cargo run --bin csv_rag --features fastembed

# Run reliable RAG example
cargo run --bin reliable_rag --features fastembed

# Run chunk size optimization
cargo run --bin chunk_size_optimization --features fastembed

# Run proposition chunking
cargo run --bin proposition_chunking --features fastembed
```

### Configuration

Examples support multiple configuration options:

```bash
# Use FastEmbed (default, no API key required)
cargo run --bin simple_rag --features fastembed

# Use OpenAI embeddings (requires OPENAI_API_KEY)
cargo run --bin simple_rag --features fastembed -- --embedding-provider openai

# Use custom chunk size
cargo run --bin simple_rag --features fastembed -- --chunk-size 500 --chunk-overlap 100

# Enable interactive mode
cargo run --bin simple_rag --features fastembed -- --interactive

# Proposition chunking with comparison
cargo run --bin proposition_chunking --features fastembed -- --compare-traditional --verbose

# Enable verbose logging
RUST_LOG=info cargo run --bin simple_rag --features fastembed
```

## 📊 Performance Expectations

Based on our testing with the climate change document:

| Example | Indexing Time | Query Time | Accuracy | Memory Usage |
|---------|---------------|------------|----------|--------------|
| Simple RAG | ~2-5s | ~100-300ms | Good | Low |
| CSV RAG | ~1-3s | ~50-150ms | High | Low |
| Reliable RAG | ~3-7s | ~200-500ms | Very High | Medium |
| Chunk Optimization | ~5-15s | ~100-400ms | Variable | Medium |
| Proposition Chunking | ~10-30s | ~150-400ms | Very High | High |

*Note: Times vary based on document size, hardware, and embedding provider.*

## 🎯 Learning Path

We recommend following this order:

1. **Start with Simple RAG** - Understand the basic pipeline
2. **Try CSV RAG** - Learn structured data handling
3. **Explore Reliable RAG** - Add quality controls
4. **Experiment with Chunk Optimization** - Tune performance
5. **Advanced: Proposition Chunking** - Semantic understanding

## 🔧 Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure environment variables are set
2. **Memory Issues**: Reduce chunk size or use streaming
3. **Slow Performance**: Try FastEmbed instead of OpenAI embeddings
4. **Poor Results**: Experiment with different chunk sizes

### Getting Help

- Check the logs with `RUST_LOG=debug`
- Review the shared utilities in `../shared/`
- Refer to the main Cheungfun documentation

## 🎓 Next Steps

After mastering these foundational techniques, explore:

- **Query Enhancement** (`../02_query_enhancement/`) - Improve query processing
- **Context Enrichment** (`../03_context_enrichment/`) - Enhance retrieved context
- **Advanced Retrieval** (`../04_advanced_retrieval/`) - Sophisticated retrieval methods

---

*These examples are designed for learning and experimentation. For production use, consider additional optimizations and error handling.*
