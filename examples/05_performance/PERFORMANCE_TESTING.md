# Cheungfun Performance Testing Guide

This guide explains how to use the comprehensive performance testing suite for the Cheungfun RAG framework.

## 🎯 Overview

The performance testing suite provides comprehensive benchmarks for all major components of Cheungfun with feature-aware testing.

## 🚀 Quick Start

### Unified Performance Testing (Recommended)

Use the unified test runner for the best experience:

```bash
# Run all tests with recommended features
cargo run --bin run_performance_tests

# Run with specific features
cargo run --bin run_performance_tests --features performance

# Run specific test categories
cargo run --bin run_performance_tests -- --embedders
cargo run --bin run_performance_tests -- --vector-stores
cargo run --bin run_performance_tests -- --comprehensive
```

### Individual Component Tests

```bash
# Comprehensive performance comparison
cargo run --bin performance_benchmark --features performance

# Embedder performance
cargo run --bin embedder_benchmark

# Vector store performance
cargo run --bin vector_store_benchmark

# End-to-end RAG performance
cargo run --bin end_to_end_benchmark
```

## Available Features

### Performance Features

- `simd` - SIMD vector operations (2-4x speedup)
- `optimized-memory` - Memory-optimized data structures
- `hnsw` - Approximate nearest neighbor search
- `performance` - All performance features combined

### ML Framework Features

- `candle` - Local ML models with Candle framework
- `fastembed` - Fast embedding models (default)
- `gpu-cuda` - NVIDIA GPU acceleration
- `gpu-metal` - Apple Silicon GPU acceleration

## 📊 Benchmark Components

### 1. Embedder Benchmarks

Tests the performance of different embedding implementations:

- **Single Text Embedding**: Latency and throughput for individual texts
- **Batch Embedding**: Performance with multiple texts processed together
- **Large Batch Processing**: Scalability with 100+ texts
- **Memory and CPU Usage**: Resource consumption analysis

**Supported Embedders:**
- `FastEmbedder` (requires `fastembed` feature)
- `ApiEmbedder` (requires API key)
- `CandleEmbedder` (requires `candle` feature)

### 2. Vector Store Benchmarks

Tests vector storage and retrieval performance:

- **Insert Performance**: Single and batch vector insertion
- **Search Performance**: Vector similarity search with various dataset sizes
- **Memory Usage**: RAM consumption for different dataset sizes
- **Scalability**: Performance with 1K, 10K+ vectors

**Supported Stores:**
- `InMemoryVectorStore` (always available)
- `QdrantVectorStore` (requires Qdrant server)

### 3. End-to-End RAG Benchmarks

Tests complete RAG pipeline performance:

- **Document Indexing**: Loading, chunking, embedding, and storing documents
- **Query Processing**: End-to-end query response time
- **Scalability**: Performance with different document counts
- **Resource Usage**: Memory and CPU consumption for full pipelines

## 📈 Understanding Results

### Performance Metrics

Each benchmark collects comprehensive metrics:

```
📊 Benchmark Results: FastEmbedder Single Text
=====================================
🕐 Duration: 2.5s
🔢 Operations: 100
⚡ Ops/sec: 40.0
📈 Latency:
  • Average: 25ms
  • Min: 20ms
  • Max: 35ms
  • P95: 30ms
  • P99: 33ms
💾 Memory:
  • Peak: 256.3 MB
  • Average: 245.1 MB
🖥️  CPU:
  • Average: 45.2%
  • Peak: 78.5%
```

### Key Performance Indicators

- **Operations/Second**: Higher is better for throughput
- **Average Latency**: Lower is better for responsiveness
- **P95/P99 Latency**: Consistency indicators (lower is better)
- **Peak Memory**: Resource usage (lower is better for efficiency)
- **CPU Usage**: Processing intensity

### Interpreting Results

**Good Performance Indicators:**
- Embedder: >10 ops/sec for single text, >50 ops/sec for batches
- Vector Store: >100 ops/sec for inserts, >500 ops/sec for searches
- End-to-End: <2s for document indexing, <500ms for queries

**Performance Bottlenecks:**
- High latency (>1s): Consider faster models or hardware acceleration
- High memory usage (>1GB): Reduce batch sizes or use streaming
- Low throughput: Enable batching or parallel processing

## 🔧 Configuration Options

### Benchmark Configuration

Customize benchmark parameters in the code:

```rust
let config = BenchmarkConfig {
    name: "Custom Benchmark".to_string(),
    warmup_iterations: 5,      // Warmup runs
    measurement_iterations: 50, // Measurement runs
    collect_system_metrics: true,
    metrics_sample_rate: Duration::from_millis(100),
    ..Default::default()
};
```

### Environment Variables

- `QDRANT_URL`: Qdrant server URL (default: `http://localhost:6334`)
- `OPENAI_API_KEY`: OpenAI API key for cloud embeddings
- `SIUMAI_API_KEY`: Siumai API key for cloud embeddings

## 📁 Output Files

Benchmarks generate comprehensive reports:

```
benchmark_results/
├── benchmark_summary.md          # Executive summary
├── performance_report.html       # Interactive HTML report
├── charts/                       # Performance visualizations
│   ├── ops_per_second.png
│   ├── latency_distribution.png
│   ├── memory_usage.png
│   └── performance_matrix.png
├── raw_metrics/                  # Raw JSON data
│   ├── all_metrics.json
│   ├── embedder_metrics.json
│   └── vector_store_metrics.json
└── *.log                        # Detailed benchmark logs
```

## 🎯 Performance Optimization Tips

### For Embedders

1. **Use Batch Processing**: Always prefer `embed_batch()` over multiple `embed()` calls
2. **Optimize Batch Size**: Test different batch sizes (8-32 typically optimal)
3. **Choose Right Model**: Balance accuracy vs speed based on your needs
4. **Enable Caching**: Use ApiEmbedder caching for repeated texts

### For Vector Stores

1. **Batch Operations**: Use batch inserts for better throughput
2. **Appropriate Indexing**: Configure HNSW parameters for your use case
3. **Memory Management**: Monitor memory usage with large datasets
4. **Connection Pooling**: Use connection pools for production deployments

### For End-to-End Performance

1. **Pipeline Optimization**: Tune chunk sizes and overlap
2. **Parallel Processing**: Enable concurrent processing where possible
3. **Resource Monitoring**: Monitor memory and CPU usage
4. **Caching Strategies**: Implement caching at multiple levels

## 🔍 Troubleshooting

### Common Issues

**"FastEmbed feature not available"**
```bash
# Add FastEmbed dependency
cargo add fastembed --features onnx
```

**"Qdrant server not available"**
```bash
# Start Qdrant with Docker
docker run -p 6334:6334 qdrant/qdrant
```

**"API key not found"**
```bash
# Set environment variable
export OPENAI_API_KEY="your-api-key"
```

**High memory usage**
- Reduce batch sizes
- Use streaming processing for large datasets
- Monitor system resources

**Poor performance**
- Check hardware specifications
- Verify network connectivity for cloud services
- Consider GPU acceleration for compute-intensive workloads

## 📚 Advanced Usage

### Custom Benchmarks

Create custom benchmarks by extending the framework:

```rust
use examples::benchmark_framework::{BenchmarkConfig, run_benchmark};

let config = BenchmarkConfig {
    name: "My Custom Benchmark".to_string(),
    measurement_iterations: 100,
    ..Default::default()
};

let metrics = run_benchmark(config, || async {
    // Your custom operation here
    my_operation().await
}).await?;
```

### Continuous Performance Monitoring

Set up automated benchmarking:

1. Run benchmarks on CI/CD pipelines
2. Compare results with baseline metrics
3. Set up alerts for performance regressions
4. Track performance trends over time

## 🤝 Contributing

To add new benchmarks:

1. Extend the benchmark framework in `src/benchmark_framework.rs`
2. Add new benchmark binaries in the examples directory
3. Update the runner scripts to include new benchmarks
4. Add documentation for new metrics and interpretations

## 📞 Support

For questions or issues with performance testing:

1. Check the troubleshooting section above
2. Review the generated log files for detailed error information
3. Open an issue in the Cheungfun repository with benchmark results
4. Include system specifications and configuration details
