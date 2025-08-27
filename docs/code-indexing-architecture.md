# Code Indexing Architecture for RAG Applications

## Overview

This document outlines the architecture design for code indexing capabilities in the Cheungfun RAG framework, inspired by llama_index's approach and designed to support deepwiki-like applications.

## Core Components

### 1. Code Loader (`CodeLoader`)

**Purpose**: Specialized loader for source code files with language-aware parsing and metadata extraction.

**Key Features**:
- Multi-language support (Rust, Python, JavaScript, TypeScript, Java, C#, Go, etc.)
- Automatic language detection from file extensions and content
- Code-specific metadata extraction (functions, classes, imports, comments)
- Lines of code (LOC) calculation
- Recursive directory traversal with smart filtering

**Usage Example**:
```rust
let code_loader = CodeLoader::new("./src")?;
let documents = code_loader.load().await?;

// Each document contains:
// - Original source code content
// - Language metadata
// - Extracted functions, classes, imports
// - Code statistics (LOC, total lines)
```

### 2. Code Splitter (`CodeSplitter`)

**Purpose**: Syntax-aware text splitter that preserves code structure when creating chunks.

**Key Features**:
- Language-specific break point detection
- Function and class boundary preservation
- Configurable chunk sizes and overlap
- Indentation context preservation
- Smart splitting at natural code boundaries

**Usage Example**:
```rust
let code_splitter = CodeSplitter::with_config(CodeSplitterConfig {
    max_chunk_lines: 40,
    chunk_overlap_lines: 15,
    respect_function_boundaries: true,
    preserve_indentation: true,
    ..Default::default()
});

let nodes = code_splitter.split_document(document).await?;
```

### 3. Programming Language Detection

**Purpose**: Accurate language identification for proper parsing and processing.

**Supported Languages**:
- **Systems**: Rust, C, C++, Go
- **Web**: JavaScript, TypeScript, HTML, CSS
- **Enterprise**: Java, C#, Scala, Kotlin
- **Scripting**: Python, Ruby, PHP, Lua, Shell
- **Functional**: Haskell, Clojure, Erlang, Elixir
- **Data**: SQL, JSON, YAML, TOML, XML
- **Documentation**: Markdown

### 4. Metadata Extraction

**Extracted Information**:
- **Functions/Methods**: Signatures and names
- **Classes/Structs**: Definitions and inheritance
- **Imports/Includes**: Dependencies and modules
- **Comments/Docstrings**: Documentation content
- **Code Statistics**: LOC, complexity metrics
- **File Information**: Size, timestamps, paths

## Architecture Comparison

### llama_index Approach
```python
from llama_index.core.node_parser import CodeSplitter

splitter = CodeSplitter(
    language="python",
    chunk_lines=40,
    chunk_lines_overlap=15,
    max_chars=1500,
)
nodes = splitter.get_nodes_from_documents(documents)
```

### Cheungfun Approach
```rust
use cheungfun_indexing::prelude::*;

let code_loader = CodeLoader::new("./src")?;
let documents = code_loader.load().await?;

let code_splitter = CodeSplitter::new();
let mut all_nodes = Vec::new();
for doc in documents {
    let nodes = code_splitter.split_document(doc).await?;
    all_nodes.extend(nodes);
}
```

## Integration with RAG Pipeline

### 1. Document Loading Phase
```rust
// Load code files with metadata extraction
let code_loader = CodeLoader::with_config(path, CodeLoaderConfig {
    extract_functions: true,
    extract_classes: true,
    extract_imports: true,
    extract_comments: true,
    ..Default::default()
});

let documents = code_loader.load().await?;
```

### 2. Text Splitting Phase
```rust
// Split code into semantically meaningful chunks
let code_splitter = CodeSplitter::with_config(CodeSplitterConfig {
    max_chunk_lines: 40,
    respect_function_boundaries: true,
    preserve_indentation: true,
    ..Default::default()
});

let nodes = code_splitter.split_document(document).await?;
```

### 3. Embedding and Indexing Phase
```rust
// Create embeddings for code chunks
let embedding_model = EmbeddingModel::new()?;
for node in &mut nodes {
    let embedding = embedding_model.embed(&node.content).await?;
    node.embedding = Some(embedding);
}

// Build vector index
let vector_store = VectorStore::new()?;
vector_store.add_nodes(nodes).await?;
```

### 4. Retrieval Phase
```rust
// Code-aware retrieval with metadata filtering
let query = "How to create a new user?";
let query_embedding = embedding_model.embed(query).await?;

let results = vector_store
    .similarity_search(query_embedding, 5)
    .with_metadata_filter("language", "rust")
    .with_metadata_filter("functions", "*new*")
    .execute()
    .await?;
```

## DeepWiki Integration Strategy

### 1. Repository Analysis
- **Code Structure Discovery**: Identify main modules, entry points, and dependencies
- **Architecture Mapping**: Extract relationships between components
- **Documentation Generation**: Create comprehensive code documentation

### 2. Intelligent Q&A
- **Context-Aware Retrieval**: Find relevant code snippets based on semantic similarity
- **Cross-Reference Resolution**: Link related functions, classes, and modules
- **Code Example Generation**: Provide usage examples and patterns

### 3. Visual Documentation
- **Dependency Graphs**: Show import/include relationships
- **Class Hierarchies**: Visualize inheritance and composition
- **Call Graphs**: Map function call relationships
- **Module Structure**: Display project organization

## Advanced Features

### 1. AST-Based Parsing (Future Enhancement)
```rust
// TODO: Implement tree-sitter integration for precise AST parsing
use tree_sitter::{Language, Parser};

let mut parser = Parser::new();
parser.set_language(tree_sitter_rust::language())?;
let tree = parser.parse(&code_content, None)?;
```

### 2. Code Complexity Analysis
```rust
// Calculate cyclomatic complexity and other metrics
pub struct CodeComplexity {
    pub cyclomatic: u32,
    pub cognitive: u32,
    pub nesting_depth: u32,
    pub function_length: u32,
}
```

### 3. Cross-Language Support
```rust
// Handle polyglot repositories with multiple languages
let multi_lang_loader = CodeLoader::with_config(path, CodeLoaderConfig {
    supported_languages: vec![
        ProgrammingLanguage::Rust,
        ProgrammingLanguage::Python,
        ProgrammingLanguage::JavaScript,
    ],
    ..Default::default()
});
```

## Performance Considerations

### 1. Parallel Processing
- **Concurrent File Loading**: Process multiple files simultaneously
- **Batch Metadata Extraction**: Extract metadata in parallel
- **Streaming Processing**: Handle large codebases efficiently

### 2. Caching Strategy
- **Metadata Caching**: Cache extracted metadata to avoid reprocessing
- **Incremental Updates**: Only process changed files
- **Smart Invalidation**: Invalidate cache when dependencies change

### 3. Memory Management
- **Lazy Loading**: Load file content on demand
- **Chunk Streaming**: Process large files in chunks
- **Resource Cleanup**: Proper cleanup of temporary resources

## Configuration Examples

### Basic Code Indexing
```rust
let config = CodeLoaderConfig::default();
let loader = CodeLoader::with_config("./src", config)?;
```

### Advanced Configuration
```rust
let config = CodeLoaderConfig {
    base: LoaderConfig::new()
        .with_max_file_size(10 * 1024 * 1024) // 10MB limit
        .with_source_code_filtering(),
    extract_functions: true,
    extract_classes: true,
    extract_imports: true,
    extract_comments: true,
    preserve_structure: true,
    max_chunk_lines: 50,
    chunk_overlap_lines: 10,
    max_chunk_chars: 2000,
};
```

## Next Steps

1. **AST Integration**: Implement tree-sitter for precise syntax parsing
2. **Semantic Analysis**: Add code understanding capabilities
3. **Cross-Reference Resolution**: Build symbol tables and reference graphs
4. **Documentation Generation**: Automatic API documentation creation
5. **Code Search**: Implement semantic code search capabilities
6. **Visualization**: Generate interactive code exploration interfaces

This architecture provides a solid foundation for building deepwiki-like applications with comprehensive code understanding and intelligent retrieval capabilities.
