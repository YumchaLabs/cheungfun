# 🚀 Advanced Retrieval Optimization Examples

This directory contains examples demonstrating advanced retrieval optimization techniques in RAG systems using the Cheungfun framework.

## 📋 Available Examples

### ✅ Completed Examples

1. **[Fusion Retrieval](fusion_retrieval.rs)** - Hybrid search combining vector and keyword-based retrieval
2. **[Intelligent Reranking](intelligent_reranking.rs)** - Multiple reranking strategies for improving result relevance
3. **[Multi-faceted Filtering](multi_faceted_filtering.rs)** - Advanced filtering techniques for improving retrieval quality

### 🚧 Planned Examples

1. **Hierarchical Indices** - Multi-tiered indexing for efficient information navigation
2. **Ensemble Retrieval** - Combining multiple retrieval models
3. **Dartboard Retrieval** - Optimizing relevant information gain
4. **Multi-modal RAG** - Cross-modal retrieval and generation

## 🎯 Multi-faceted Filtering Example

### Overview

The Multi-faceted Filtering example demonstrates various filtering strategies for improving RAG retrieval quality by applying multiple filtering techniques to retrieved documents:

- **Metadata Filtering**: Filter by document source, type, or other metadata attributes
- **Score-based Filtering**: Apply similarity score thresholds to remove low-relevance results
- **Content Quality Filtering**: Filter based on content length, completeness, and readability
- **Contextual Compression**: Extract only the most relevant sentences from documents
- **Relevant Segment Extraction**: Extract contiguous relevant segments while preserving context

### Key Features

- 🏷️ **Multiple Filtering Strategies**: Metadata, score, quality, compression, and segment extraction
- 📊 **Strategy Comparison**: Side-by-side comparison of filtering effectiveness
- ⚡ **Configurable Parameters**: Adjustable thresholds, result counts, and filtering criteria
- 🎯 **Interactive Mode**: Real-time testing with custom queries
- 📈 **Performance Metrics**: Detailed timing and quality measurements

### Usage

#### Basic Usage

```bash
# Run with all filtering strategies comparison
cargo run --bin multi_faceted_filtering

# Test a specific strategy
cargo run --bin multi_faceted_filtering -- --strategy metadata

# Single query test
cargo run --bin multi_faceted_filtering -- --query "What causes climate change?"
```

#### Advanced Options

```bash
# Custom configuration
cargo run --bin multi_faceted_filtering -- \
  --data "data/Understanding_Climate_Change.pdf" \
  --strategy combined \
  --top-n 10 \
  --initial-count 50 \
  --embedding-provider fastembed
```

#### Available Filtering Strategies

- `metadata` - Filter by document metadata (source, type, etc.)
- `score` - Filter by similarity score threshold (default: 0.7)
- `quality` - Filter by content quality metrics (length, completeness)
- `compression` - Apply contextual compression to extract key content
- `segments` - Extract relevant contiguous segments
- `combined` - Compare all strategies (default)

## 🎯 Intelligent Reranking Example

### Overview

The Intelligent Reranking example demonstrates various strategies for reordering retrieved documents to improve relevance:

- **LLM-based Reranking**: Uses language models to assess document relevance
- **Score-based Reranking**: Reorders based on similarity scores with different strategies
- **Diversity-based Reranking**: Ensures result diversity by filtering similar documents
- **Combined Reranking**: Merges multiple reranking approaches

### Key Features

- 🧠 **Multiple Reranking Strategies**: LLM, score-based, diversity, and combined approaches
- 📊 **Performance Comparison**: Side-by-side comparison of different methods
- ⚡ **Configurable Parameters**: Adjustable top-N, batch sizes, and thresholds
- 🎯 **Interactive Mode**: Real-time testing with custom queries
- 📈 **Performance Metrics**: Detailed timing and effectiveness measurements

### Usage

#### Basic Usage
```bash
# Run with all reranking strategies comparison
cargo run --bin intelligent_reranking

# Test a specific strategy
cargo run --bin intelligent_reranking -- --strategy llm

# Single query test
cargo run --bin intelligent_reranking -- --query "What are the impacts of climate change on biodiversity?"
```

#### Advanced Options
```bash
# Custom configuration
cargo run --bin intelligent_reranking -- \
  --data "data/nike_2023_annual_report.txt" \
  --strategy diversity \
  --top-n 8 \
  --initial-count 20 \
  --llm-provider openai \
  --embedding-provider openai
```

#### Available Strategies
- `llm` - LLM-based semantic reranking
- `score` - Original similarity score reranking
- `diversity` - Diversity-based reranking (similarity threshold: 0.8)
- `random` - Random reordering (for baseline comparison)
- `combined` - Weighted combination of LLM and score methods
- `all` - Compare all strategies side-by-side

### Example Output

```
🎯 Intelligent Reranking Example
═══════════════════════════════════════════════════════════════
📊 Strategy: all
📁 Data: data/Understanding_Climate_Change.pdf
🔢 Top N: 5, Initial Count: 15
🤖 LLM: ollama, Embeddings: fastembed

🔍 Comparing Reranking Strategies
═══════════════════════════════════════════════════════════════

📊 1. Baseline: Simple Similarity Search
📋 Baseline Results:
1. [Score: 0.8234] Climate change affects biodiversity through habitat loss, temperature changes...
2. [Score: 0.7891] Species migration patterns are disrupted by changing environmental conditions...
3. [Score: 0.7654] Ocean acidification impacts marine ecosystems and coral reef systems...

🧠 2. LLM-based Reranking
⏱️  Reranking time: 2.34s
📋 LLM Reranking Results:
1. [Score: 0.9500] Climate change affects biodiversity through habitat loss, temperature changes...
2. [Score: 0.9200] Ecosystem disruption leads to species extinction and reduced genetic diversity...
3. [Score: 0.8800] Conservation efforts must adapt to climate change impacts on wildlife...
```

### Performance Characteristics

| Strategy | Speed | Accuracy | Diversity | Use Case |
|----------|-------|----------|-----------|----------|
| LLM | Slow (2-5s) | High | Medium | High-quality results, semantic understanding |
| Score | Fast (<0.1s) | Medium | Low | Quick reordering, baseline comparison |
| Diversity | Fast (<0.2s) | Medium | High | Avoiding result clustering |
| Combined | Medium (1-3s) | High | High | Best of both worlds |

### Interactive Mode

The interactive mode allows real-time testing:

```
🔍 Query: What causes ocean acidification?
🧠 LLM-based Reranking
⏱️  Reranking time: 1.87s
📋 LLM Reranking Results:
1. [Score: 0.9400] Ocean acidification is caused by increased CO2 absorption...
2. [Score: 0.9100] Marine chemistry changes affect shell-forming organisms...

🔍 Query: help
📖 Available Commands:
  help  - Show this help message
  stats - Show performance statistics
  quit  - Exit the program
```

## 🔧 Configuration

### Environment Variables

```bash
# For OpenAI integration
export OPENAI_API_KEY="your-api-key-here"

# For custom Ollama endpoint
export OLLAMA_BASE_URL="http://localhost:11434"
```

### Embedding Providers

- **FastEmbed** (default): Local embedding model, no API key required
- **OpenAI**: Cloud-based embeddings, requires API key

### LLM Providers

- **Ollama** (default): Local LLM server, supports various models
- **OpenAI**: Cloud-based LLMs, requires API key

## 📊 Performance Optimization

### Tips for Better Performance

1. **Batch Size Tuning**: Adjust LLM reranking batch size based on your hardware
2. **Initial Count**: Balance between recall and processing time
3. **Strategy Selection**: Choose based on your accuracy vs. speed requirements
4. **Caching**: Results are automatically cached for repeated queries

### Benchmarking

Run performance tests with different configurations:

```bash
# Test different batch sizes
for batch in 3 5 10; do
  echo "Testing batch size: $batch"
  cargo run --bin intelligent_reranking -- \
    --strategy llm \
    --query "climate change impacts" \
    --verbose
done
```

## 🤝 Contributing

When adding new reranking strategies:

1. Implement the `Reranker` trait
2. Add configuration options to `RerankingStrategy` enum
3. Update the CLI parser and help text
4. Add performance benchmarks
5. Update this README with usage examples

## 📚 References

- [RAG Techniques Repository](https://github.com/NirDiamant/RAG_Techniques)
- [Reranking in Information Retrieval](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/reranking.ipynb)
- [LlamaIndex Reranking Guide](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors.html)

## 🔄 Next Steps

After mastering reranking techniques, explore:

1. **Context Enrichment** - Enhance retrieved content with additional context
2. **Semantic Chunking** - Improve document segmentation strategies
3. **Graph RAG** - Leverage knowledge graphs for enhanced retrieval
4. **Multi-modal Integration** - Combine text, image, and other modalities

---

**Note**: This example is part of the comprehensive RAG techniques tutorial series. Each technique builds upon previous concepts while introducing new optimization strategies.
