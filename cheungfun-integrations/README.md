# Cheungfun Integrations

External service integrations for Cheungfun RAG system, providing embedding models and vector storage implementations.

## Features

This crate provides flexible embedding solutions through Cargo features:

### 🚀 FastEmbed Integration (Default, Recommended)

**Default feature**: Provides the best user experience with minimal configuration.

```toml
[dependencies]
cheungfun-integrations = "0.1.0"  # FastEmbed enabled by default
```

```rust
use cheungfun_integrations::FastEmbedder;

// One-line setup with smart defaults
let embedder = FastEmbedder::new().await?;
let embedding = embedder.embed("Hello, world!").await?;

// Or use intelligent presets
let embedder = FastEmbedder::high_quality().await?;  // Best quality
let embedder = FastEmbedder::multilingual().await?;  // Multi-language
let embedder = FastEmbedder::fast().await?;          // Fastest speed
```

**Benefits**:
- ✅ **Zero configuration** - Works out of the box
- ✅ **Intelligent presets** - Optimized for common use cases  
- ✅ **Automatic optimization** - Built-in retry, caching, batching
- ✅ **Production ready** - Used by thousands of projects
- ✅ **Minimal dependencies** - Smaller binary size

### 🔧 Candle Integration (Advanced Users)

**Optional feature**: For users who need deep customization and local control.

```toml
[dependencies]
cheungfun-integrations = { version = "0.1.0", features = ["candle"] }
```

```rust
use cheungfun_integrations::CandleEmbedder;

// Advanced configuration with full control
let embedder = CandleEmbedder::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").await?;

// Or with custom configuration
let config = CandleEmbedderConfig::new("custom-model")
    .with_device_preference(DevicePreference::Cuda)
    .with_batch_size(32)
    .with_max_length(512);
let embedder = CandleEmbedder::from_config(config).await?;
```

**Benefits**:
- ✅ **Full local control** - No external API dependencies
- ✅ **Device optimization** - CPU/CUDA/Metal support
- ✅ **Custom models** - Load any HuggingFace sentence-transformers model
- ✅ **Advanced configuration** - Fine-tune every parameter

**Benefits**:
- ✅ **Full local control** - No external API dependencies
- ✅ **Device optimization** - CPU/CUDA/Metal support
- ✅ **Custom models** - Load any HuggingFace sentence-transformers model
- ✅ **Advanced configuration** - Fine-tune every parameter

### 🎯 Feature Combinations

```toml
# Default: FastEmbed only (recommended for most users)
cheungfun-integrations = "0.1.0"

# Advanced: Candle only
cheungfun-integrations = { version = "0.1.0", default-features = false, features = ["candle"] }

# Power user: Both FastEmbed and Candle
cheungfun-integrations = { version = "0.1.0", features = ["all-embedders"] }
```

## Vector Storage

All configurations include the high-performance `InMemoryVectorStore`:

```rust
use cheungfun_integrations::InMemoryVectorStore;

let store = InMemoryVectorStore::new();
store.add_vector("doc1", vec![0.1, 0.2, 0.3], metadata).await?;
let results = store.search(&query_vector, 5).await?;
```

## Quick Start Examples

### Basic RAG with FastEmbed (Recommended)

```rust
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize embedder (one line!)
    let embedder = FastEmbedder::new().await?;
    
    // Initialize vector store
    let mut store = InMemoryVectorStore::new();
    
    // Add documents
    let docs = vec!["Rust is fast", "Python is easy", "Go is simple"];
    for (i, doc) in docs.iter().enumerate() {
        let embedding = embedder.embed(doc).await?;
        store.add_vector(&format!("doc_{}", i), embedding, None).await?;
    }
    
    // Search
    let query_embedding = embedder.embed("programming languages").await?;
    let results = store.search(&query_embedding, 2).await?;
    
    println!("Found {} similar documents", results.len());
    Ok(())
}
```

### Advanced RAG with Candle

```rust
use cheungfun_integrations::{CandleEmbedder, CandleEmbedderConfig, InMemoryVectorStore};

#[tokio::main] 
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Advanced configuration
    let config = CandleEmbedderConfig::new("sentence-transformers/all-mpnet-base-v2")
        .with_batch_size(16)
        .with_normalize(true);
    
    let embedder = CandleEmbedder::from_config(config).await?;
    
    // Rest of the code is the same...
    Ok(())
}
```

## Architecture

```
cheungfun-integrations/
├── embedders/
│   ├── fastembed/     # FastEmbed integration (default)
│   └── candle/        # Candle integration (optional)
│       ├── embedder.rs    # Main embedder implementation  
│       ├── model.rs       # Model loading & HuggingFace Hub
│       ├── tokenizer.rs   # Tokenization pipeline
│       ├── device.rs      # Device management (CPU/CUDA/Metal)
│       ├── config.rs      # Configuration system
│       └── error.rs       # Error handling
└── vector_stores/
    └── memory.rs      # In-memory vector storage
```

## Performance Comparison

| Aspect | FastEmbed | Candle | Winner |
|--------|-----------|--------|---------|
| Setup Time | 1 line | 5+ lines | **FastEmbed** |
| Binary Size | Smaller | Larger | **FastEmbed** |
| Memory Usage | Optimized | Configurable | **FastEmbed** |
| Customization | Limited | Full | **Candle** |
| Local Control | Limited | Complete | **Candle** |
| Maintenance | Zero | Manual | **FastEmbed** |

## Recommendations

- 🥇 **For most users**: Use default FastEmbed - it's optimized, maintained, and just works
- 🥈 **For advanced users**: Use Candle if you need specific models or full local control  
- 🥉 **For enterprises**: Consider both options based on your specific requirements

## Contributing

See the main [Cheungfun repository](https://github.com/your-org/cheungfun) for contribution guidelines.

## License

Licensed under the same terms as the main Cheungfun project.
