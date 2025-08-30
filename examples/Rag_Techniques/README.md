# RAG Techniques Examples

This directory contains Rust implementations of advanced RAG techniques based on the [RAG_Techniques repository](https://github.com/NirDiamant/RAG_Techniques). Each example demonstrates a specific technique with practical implementation using the Cheungfun RAG framework.

## Organization

The examples are organized following the same structure as the original RAG_Techniques repository:

### 🌱 Foundational RAG Techniques

| # | Technique | File | Description |
|---|-----------|------|-------------|
| 01 | Simple RAG | `01_simple_rag.rs` | Basic RAG implementation |
| 02 | RAG with CSV Files | `02_csv_rag.rs` | RAG using CSV data sources |
| 03 | Reliable RAG | `03_reliable_rag.rs` | Enhanced RAG with validation |
| 04 | Optimizing Chunk Sizes | `04_chunk_size_optimization.rs` | Finding optimal chunk sizes |
| 05 | Proposition Chunking | `05_proposition_chunking.rs` | Semantic proposition-based chunking |

### 🔍 Query Enhancement

| # | Technique | File | Description |
|---|-----------|------|-------------|
| 06 | Query Transformations | `06_query_transformations.rs` | Query rewriting and enhancement |
| 07 | HyDE | `07_hyde.rs` | Hypothetical Document Embedding |
| 08 | HyPE | `08_hype.rs` | Hypothetical Prompt Embedding |

### 📚 Context Enrichment

| # | Technique | File | Description |
|---|-----------|------|-------------|
| 09 | Contextual Chunk Headers | `09_contextual_chunk_headers.rs` | Adding context headers to chunks |
| 10 | Relevant Segment Extraction | `10_relevant_segment_extraction.rs` | Dynamic segment construction |
| 11 | Context Window Enhancement | `11_context_window_enhancement.rs` | Expanding context around chunks |
| 12 | Semantic Chunking | `12_semantic_chunking.rs` | Semantic boundary-based chunking |
| 13 | Contextual Compression | `13_contextual_compression.rs` | Compressing retrieved content |
| 14 | Document Augmentation | `14_document_augmentation.rs` | Question generation for documents |

### 🚀 Advanced Retrieval Methods

| # | Technique | File | Description |
|---|-----------|------|-------------|
| 15 | Fusion Retrieval | `15_fusion_retrieval.rs` | Combining multiple retrieval methods |
| 16 | Intelligent Reranking | `16_intelligent_reranking.rs` | Advanced result reranking |
| 17 | Multi-faceted Filtering | `17_multi_faceted_filtering.rs` | Multiple filtering strategies |
| 18 | Hierarchical Indices | `18_hierarchical_indices.rs` | Multi-tiered indexing |
| 19 | Ensemble Retrieval | `19_ensemble_retrieval.rs` | Multiple model ensemble |

### 🔁 Iterative and Adaptive Techniques

| # | Technique | File | Description |
|---|-----------|------|-------------|
| 20 | Retrieval with Feedback Loop | `20_retrieval_feedback_loop.rs` | Learning from user feedback |
| 21 | Adaptive Retrieval | `21_adaptive_retrieval.rs` | Dynamic strategy adjustment |
| 22 | Iterative Retrieval | `22_iterative_retrieval.rs` | Multi-round retrieval |

### 🏗️ Advanced Architectures

| # | Technique | File | Description |
|---|-----------|------|-------------|
| 23 | Self-RAG | `23_self_rag.rs` | Self-reflective RAG |
| 24 | Corrective RAG (CRAG) | `24_corrective_rag.rs` | Error correction in RAG |

## Usage

Each example can be run independently:

```bash
# Run a specific technique
cargo run --bin 01_simple_rag --features fastembed

# With custom parameters
cargo run --bin 14_document_augmentation --features fastembed -- --questions-per-chunk 5 --verbose

# Interactive mode
cargo run --bin 06_query_transformations --features fastembed -- --interactive
```

## Features

- **Comprehensive Coverage**: Implements most techniques from the original repository
- **Rust Performance**: Leverages Rust's performance and safety
- **Modular Design**: Each technique is self-contained
- **Practical Examples**: Real-world applicable implementations
- **Detailed Documentation**: Each file includes comprehensive documentation

## Data

The examples use the same test data as the original repository:
- `Understanding_Climate_Change.pdf` - Climate change document
- `customers-100.csv` - Customer data
- `nike_2023_annual_report.txt` - Nike annual report
- `q_a.json` - Question-answer pairs

## Dependencies

All examples use the Cheungfun RAG framework with:
- FastEmbed for embeddings
- Siumai for LLM integration
- InMemoryVectorStore or QdrantVectorStore for vector storage
- Various text processing and chunking strategies
