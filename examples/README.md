# Cheungfun Examples

Examples demonstrating the Cheungfun RAG framework capabilities, organized by feature requirements.

## Quick Start

Basic examples (no special features required):

```bash
cargo run --bin hello_world
cargo run --bin basic_indexing
cargo run --bin basic_querying

# Agent examples (requires OpenAI API key)
cargo run --example simple_agent_example
cargo run --example comprehensive_agent_example
```

## Agent Examples

AI agents with tool integration and real API calls:

### Simple Agent Example
Perfect for beginners - demonstrates basic agent setup with OpenAI integration:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run basic agent example
cargo run --example simple_agent_example
```

Features demonstrated:
- ReAct agent with reasoning capabilities
- Calculator and weather tools
- Memory management
- Step-by-step problem solving

### Comprehensive Agent Example  
Advanced patterns inspired by LlamaIndex:

```bash
cargo run --example comprehensive_agent_example
```

Features demonstrated:
- Multi-tool research assistant
- Multi-agent workflow pipeline (Research → Analysis → Writing)
- Agent coordination and handoffs
- Complex reasoning chains

See `examples/README.md` in the agent examples directory for detailed documentation.


### Candle ML Framework

CPU-only Candle examples:

```bash
cargo run --bin candle_embedder_demo --features candle
cargo run --bin candle_embedder_performance --features "candle,benchmarks"
```

GPU-accelerated Candle examples:

```bash
# NVIDIA CUDA (Linux/Windows)
cargo run --bin cuda_embedder_demo --features candle-cuda

# Apple Metal (macOS)
cargo run --bin metal_embedder_demo --features candle-metal
```

### Performance Optimization

CPU performance features:

```bash
# SIMD acceleration
cargo run --bin embedder_benchmark --features "simd,candle"

# Memory optimization
cargo run --bin vector_store_benchmark --features "optimized-memory,qdrant"

# HNSW approximate search
cargo run --bin advanced_retrieval_demo --features "hnsw,candle"

# All CPU performance features
cargo run --bin end_to_end_benchmark --features performance

# Compare different feature combinations
cargo run --bin feature_comparison --features "benchmarks,candle"
```

### Production Examples

Complete RAG systems:

```bash
# Production-ready RAG demo
cargo run --bin complete_rag_system --features production-examples

# End-to-end pipelines
cargo run --bin end_to_end_indexing --features production-examples
cargo run --bin end_to_end_query --features production-examples
```

## Available Features

### Core Features
- `basic-examples` - Basic examples that work everywhere
- `candle` - Candle ML framework (CPU)
- `candle-cuda` - Candle with CUDA GPU acceleration
- `candle-metal` - Candle with Metal GPU acceleration (macOS)

### Performance Features
- `simd` - SIMD acceleration for vector operations
- `optimized-memory` - Optimized memory management
- `hnsw` - HNSW approximate nearest neighbor search

### Integration Features
- `qdrant` - Qdrant vector database
- `mcp` - Model Context Protocol tools

### Feature Bundles
- `performance` - All CPU performance optimizations
- `production` - Recommended for production use
- `full` - Everything enabled

## Environment Setup

API keys for LLM examples:

```bash
export OPENAI_API_KEY="your-api-key"
```

Local LLM setup:

```bash
ollama serve
ollama pull llama2
```

GPU setup:

```bash
# Check CUDA (Linux/Windows)
nvidia-smi

# Run with GPU features
cargo run --bin <example> --features candle-cuda  # NVIDIA
cargo run --bin <example> --features candle-metal # Apple
```

### Performance Examples

#### CPU Performance Optimizations
```bash
# SIMD acceleration
cargo run --bin embedder_benchmark --features "simd,candle"

# Memory optimization
cargo run --bin vector_store_benchmark --features "optimized-memory,qdrant"

# HNSW approximate nearest neighbor
cargo run --bin advanced_retrieval_demo --features "hnsw,candle"

# All CPU performance features
cargo run --bin end_to_end_benchmark --features performance
```

#### Comprehensive Benchmarks
```bash
# Run all performance tests
cargo run --bin run_performance_tests --features benchmarks

# Compare different embedders
cargo run --bin embedder_benchmark --features "benchmarks,all-embedders"

# End-to-end system benchmark
cargo run --bin end_to_end_benchmark --features "benchmarks,production"

# Compare performance of different feature combinations
cargo run --bin feature_comparison --features "benchmarks,candle"
cargo run --bin feature_comparison --features "benchmarks,performance,candle"
```

### GPU Acceleration Examples

#### CUDA GPU (Linux/Windows with NVIDIA)

```bash
# Basic CUDA demo
cargo run --bin cuda_embedder_demo --features candle-cuda

# CUDA with performance optimizations
cargo run --bin cuda_embedder_demo --features "candle-cuda,performance"
```

#### Metal GPU (macOS with Apple Silicon)

```bash
# Basic Metal demo
cargo run --bin metal_embedder_demo --features candle-metal

# Metal with performance optimizations
cargo run --bin metal_embedder_demo --features "candle-metal,performance"
```

### Integration Examples

#### Vector Databases
```bash
# Qdrant vector store
cargo run --bin qdrant_store_demo --features qdrant
```

#### MCP (Model Context Protocol)
```bash
# MCP integration
cargo run --bin mcp_integration_example --features mcp
```

### Production Examples

#### Complete RAG Systems
```bash
# Production-ready RAG demo
cargo run --bin complete_rag_system --features production-examples

# End-to-end indexing pipeline
cargo run --bin end_to_end_indexing --features production-examples

# Query processing pipeline
cargo run --bin end_to_end_query --features production-examples
```

## Feature Bundles

### Recommended Combinations

#### Development Setup
```bash
# Basic development with Candle
cargo run --bin <example> --features "candle,basic-examples"
```

#### Performance Testing
```bash
# CPU performance testing
cargo run --bin <example> --features "performance,candle"

# GPU performance testing (CUDA)
cargo run --bin <example> --features "performance,candle-cuda"

# GPU performance testing (Metal)
cargo run --bin <example> --features "performance,candle-metal"
```

#### Production Deployment
```bash
# Production-ready setup
cargo run --bin <example> --features production

# Full feature set
cargo run --bin <example> --features full
```

## Available Features

### Core Features
- `basic-examples` - Basic examples that work everywhere
- `candle` - Candle ML framework (CPU)
- `candle-cuda` - Candle with CUDA GPU acceleration
- `candle-metal` - Candle with Metal GPU acceleration (macOS)
- `fastembed` - FastEmbed embeddings

### Performance Features
- `simd` - SIMD acceleration for vector operations
- `optimized-memory` - Optimized memory management
- `hnsw` - HNSW approximate nearest neighbor search

### Integration Features
- `qdrant` - Qdrant vector database
- `mcp` - Model Context Protocol tools

### Feature Bundles
- `performance` - All CPU performance optimizations
- `gpu` - GPU acceleration features
- `production` - Recommended for production use
- `full` - Everything enabled

## Environment Setup

### API Keys
Some examples require API keys:
```bash
# For OpenAI examples
export OPENAI_API_KEY="your-api-key"

# For other LLM providers
export ANTHROPIC_API_KEY="your-api-key"
```

### Local LLM Setup
For examples using local LLMs:
```bash
# Start Ollama (if using local models)
ollama serve
ollama pull llama2
```

### GPU Setup

#### CUDA (Linux/Windows)
```bash
# Ensure CUDA toolkit is installed
nvidia-smi

# Run with CUDA features
cargo run --bin <example> --features candle-cuda
```

#### Metal (macOS)
```bash
# Metal is available by default on macOS
cargo run --bin <example> --features candle-metal
```

## Troubleshooting

### Common Issues

1. **Missing features error**: Make sure to enable required features
2. **GPU not detected**: Check GPU drivers and CUDA/Metal installation
3. **API key errors**: Verify environment variables are set
4. **Performance issues**: Try enabling performance features

### Getting Help

- Check the main project documentation
- Review individual example source code
- Open an issue on GitHub for bugs or questions
