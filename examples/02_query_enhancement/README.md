# 🔍 Query Enhancement Examples

This directory contains examples demonstrating various query enhancement techniques for RAG systems.

## Available Examples

### 1. **HyDE (Hypothetical Document Embedding)** - `hyde.rs`
- Generates hypothetical documents to improve query-document matching
- Supports multiple generation strategies and comparison modes
- Demonstrates the power of query-time document generation

### 2. **HyPE (Hypothetical Prompt Embedding)** - `hype.rs`
- Precomputes hypothetical questions during indexing for better query-document alignment
- Transforms retrieval into question-question matching problem
- Eliminates query-time overhead with offline question generation
- Provides significant performance improvements over traditional embedding

### 3. **Query Transformations** - `query_transformations.rs`
- Implements 5 different query transformation techniques
- Includes query rewriting, step-back prompting, sub-question decomposition, etc.
- Comprehensive comparison of different transformation strategies

## Quick Start

### Running Examples

```bash
# Run HyDE example
cargo run --bin hyde --features fastembed

# Run HyPE example
cargo run --bin hype --features fastembed

# Run Query Transformations example
cargo run --bin query_transformations --features fastembed
```

### Interactive Mode

```bash
# Interactive HyDE
cargo run --bin hyde --features fastembed -- --interactive

# Interactive HyPE with comparison
cargo run --bin hype --features fastembed -- --interactive --compare-traditional

# Interactive Query Transformations
cargo run --bin query_transformations --features fastembed -- --interactive
```

### Configuration Options

```bash
# Custom chunk size and questions per chunk for HyPE
cargo run --bin hype --features fastembed -- --chunk-size 800 --questions-per-chunk 5

# Enable verbose output for detailed analysis
cargo run --bin hype --features fastembed -- --verbose --compare-traditional

# Use OpenAI embeddings (requires OPENAI_API_KEY)
cargo run --bin hype --features fastembed -- --embedding-provider openai
```

## Key Features

### HyDE Features
- **Hypothetical Document Generation**: Creates synthetic documents that match query intent
- **Multiple Generation Strategies**: Different approaches for various query types
- **Performance Comparison**: Compare with traditional retrieval methods
- **Flexible Configuration**: Customizable generation parameters

### HyPE Features
- **Offline Question Generation**: Precompute hypothetical questions during indexing
- **Question-Question Matching**: Transform retrieval into Q-Q matching problem
- **Multi-Vector Storage**: Store multiple question embeddings per chunk
- **Zero Query-Time Overhead**: All processing done during indexing
- **Significant Performance Gains**: Up to 42% improvement in retrieval precision

### Query Transformations Features
- **5 Transformation Techniques**: Query rewriting, step-back, sub-questions, expansion, multi-perspective
- **Intelligent JSON Parsing**: Automatic parsing with fallback mechanisms
- **Confidence Scoring**: Each transformation includes confidence scores
- **Comprehensive Analysis**: Detailed performance metrics and comparisons

## Performance Expectations

| Technique | Indexing Time | Query Time | Precision Improvement | Best Use Case |
|-----------|---------------|------------|----------------------|---------------|
| HyDE | Standard | +2-3s | 15-25% | Complex queries |
| HyPE | +50-100% | Standard | 25-42% | All query types |
| Query Transformations | Standard | +15-25s | 5-15% | Ambiguous queries |

## Technical Details

### HyPE Architecture
1. **Document Chunking**: Split documents into manageable chunks
2. **Question Generation**: Use LLM to generate 3-5 questions per chunk
3. **Question Embedding**: Create embeddings for generated questions
4. **Multi-Vector Storage**: Store multiple embeddings per original chunk
5. **Query Matching**: Match user queries against question embeddings
6. **Content Retrieval**: Return original chunk content for matched questions

### Benefits of HyPE
- **Better Semantic Alignment**: Questions are more similar to user queries than document text
- **Improved Recall**: Multiple questions per chunk increase chances of matching
- **Scalable Performance**: No additional query-time computation
- **Language Flexibility**: Works well across different domains and languages

## Dependencies

- `cheungfun-core`: Core RAG functionality
- `cheungfun-indexing`: Document loading and processing
- `cheungfun-integrations`: Embedders and generators
- `cheungfun-query`: Query engines and retrievers
- `siumai`: LLM integration for question/document generation
- `clap`: Command-line argument parsing
- `serde`: Serialization for configuration

## Environment Setup

```bash
# Optional: Set OpenAI API key for better LLM performance
export OPENAI_API_KEY="your-api-key-here"

# Optional: Configure Ollama for local LLM
# Make sure Ollama is running on localhost:11434
```

## Troubleshooting

### Common Issues

1. **LLM Connection Errors**
   - Ensure OpenAI API key is valid or Ollama is running
   - Check network connectivity

2. **Memory Issues with Large Documents**
   - Reduce chunk size: `--chunk-size 256`
   - Reduce questions per chunk: `--questions-per-chunk 2`

3. **Slow Performance**
   - Use FastEmbed instead of OpenAI embeddings
   - Reduce document size or number of chunks

### Performance Tips

- Use `--verbose` flag to understand processing bottlenecks
- Start with smaller documents to test configuration
- Monitor memory usage with large document sets
- Consider using GPU acceleration for FastEmbed if available

## Next Steps

After exploring these query enhancement techniques, consider:

1. **Combining Techniques**: Use HyPE with query transformations
2. **Custom Question Generation**: Modify prompts for domain-specific questions
3. **Advanced Retrieval**: Implement fusion retrieval or reranking
4. **Production Deployment**: Scale to larger document collections

For more advanced examples, see the `03_retrieval_optimization` directory.
