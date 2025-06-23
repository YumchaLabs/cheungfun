# FastEmbed Embedder

A simple, fast, and ergonomic embedding solution using FastEmbed.

## Features

- **Simple API**: Just `FastEmbedder::new().await?` to get started
- **Smart Presets**: Pre-configured models for common use cases
- **Automatic Retry**: Built-in retry logic with exponential backoff
- **Performance Tracking**: Built-in statistics and monitoring
- **Type Safety**: Strong typing with helpful error messages

## Quick Start

```rust
use cheungfun_integrations::embedders::fastembed::FastEmbedder;
use cheungfun_core::traits::Embedder;

// Simple usage with defaults
let embedder = FastEmbedder::new().await?;
let embedding = embedder.embed("Hello, world!").await?;

// Batch processing (more efficient)
let texts = vec!["Hello", "World", "Rust is amazing!"];
let embeddings = embedder.embed_batch(texts).await?;
```

## Presets for Common Use Cases

Instead of remembering model names and configurations, use presets:

```rust
// For high-quality English text
let embedder = FastEmbedder::high_quality().await?;

// For multilingual content
let embedder = FastEmbedder::multilingual().await?;

// For maximum speed
let embedder = FastEmbedder::fast().await?;

// For source code
let embedder = FastEmbedder::for_code().await?;
```

## Custom Configuration

```rust
use cheungfun_integrations::embedders::fastembed::{FastEmbedder, FastEmbedConfig};

let config = FastEmbedConfig::new("BAAI/bge-large-en-v1.5")
    .with_max_length(512)
    .with_batch_size(64)
    .with_cache_dir("./my_cache")
    .without_progress();

let embedder = FastEmbedder::from_config(config).await?;
```

## Error Handling

Simplified error types that are easy to handle:

```rust
match FastEmbedder::new().await {
    Ok(embedder) => {
        // Use embedder
    }
    Err(FastEmbedError::ModelInit { model, reason }) => {
        eprintln!("Failed to initialize model {}: {}", model, reason);
    }
    Err(FastEmbedError::Embedding { reason }) => {
        eprintln!("Embedding failed: {}", reason);
    }
    Err(e) => {
        eprintln!("Other error: {}", e);
    }
}
```

## Performance Monitoring

```rust
let embedder = FastEmbedder::new().await?;

// Generate some embeddings
embedder.embed("test").await?;
embedder.embed_batch(vec!["test1", "test2"]).await?;

// Check statistics
let stats = embedder.stats().await;
println!("Embedded {} texts in {:?}", 
         stats.texts_embedded, 
         stats.duration);
println!("Success rate: {:.1}%", stats.success_rate());
```

## Available Models

| Preset | Model | Dimensions | Use Case |
|--------|-------|------------|----------|
| `Default` | BAAI/bge-small-en-v1.5 | 384 | Balanced performance for English |
| `HighQuality` | BAAI/bge-large-en-v1.5 | 1024 | Best quality for English |
| `Multilingual` | BAAI/bge-m3 | 1024 | Multiple languages |
| `Fast` | sentence-transformers/all-MiniLM-L6-v2 | 384 | Maximum speed |
| `Code` | microsoft/codebert-base | 768 | Source code and technical text |

## Comparison with Previous Implementation

### Before (Complex)
```rust
// Old candle implementation
let config = CandleEmbedderConfig::new("model-name")
    .with_device("auto")
    .with_cache_dir("./cache")
    .with_batch_size(32)
    .with_normalize(true)
    .with_max_length(512);

config.validate()?;
let embedder = CandleEmbedder::from_config(config).await?;
```

### After (Simple)
```rust
// New FastEmbed implementation
let embedder = FastEmbedder::new().await?;
// or
let embedder = FastEmbedder::high_quality().await?;
```

## Benefits

1. **80% less code** for common use cases
2. **No configuration required** for getting started
3. **Built-in best practices** (retry, error handling, monitoring)
4. **Type-safe presets** prevent configuration mistakes
5. **Automatic optimization** handled by FastEmbed library
6. **Better error messages** that actually help users

## Migration Guide

If you're migrating from the old Candle implementation:

1. Replace `CandleEmbedder::from_pretrained()` with `FastEmbedder::new()`
2. Use presets instead of manual configuration
3. Remove manual error handling - it's built-in now
4. Enjoy the simplified API!
