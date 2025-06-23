# ApiEmbedder - Cloud Embedding Services

ApiEmbedder provides seamless access to cloud-based embedding services through the [siumai](https://github.com/siumai/siumai) library. It offers a unified interface for multiple embedding providers with built-in caching, retry mechanisms, and comprehensive error handling.

## Features

- 🌐 **Multi-provider support**: OpenAI, Anthropic (planned), and custom providers
- 🚀 **High performance**: Intelligent caching and batch processing
- 🔄 **Robust reliability**: Automatic retries with exponential backoff
- 📊 **Comprehensive monitoring**: Built-in statistics and health checks
- 🛡️ **Type safety**: Full Rust type safety with comprehensive error handling
- ⚡ **Async/await**: Non-blocking operations with tokio

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
cheungfun-integrations = { version = "0.1", features = ["api"] }
cheungfun-core = "0.1"
tokio = { version = "1.0", features = ["full"] }
```

### Basic Usage

```rust
use cheungfun_integrations::embedders::api::ApiEmbedder;
use cheungfun_core::traits::Embedder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create embedder with OpenAI
    let embedder = ApiEmbedder::builder()
        .openai("your-openai-api-key")
        .model("text-embedding-3-small")
        .build()
        .await?;

    // Single embedding
    let embedding = embedder.embed("Hello, world!").await?;
    println!("Embedding dimension: {}", embedding.len());

    // Batch processing
    let texts = vec!["Hello", "World", "Rust is amazing!"];
    let embeddings = embedder.embed_batch(texts).await?;
    println!("Generated {} embeddings", embeddings.len());

    Ok(())
}
```

## Supported Providers

### OpenAI

Supports all OpenAI embedding models:

- `text-embedding-ada-002` (1536 dimensions)
- `text-embedding-3-small` (1536 dimensions) - **Recommended**
- `text-embedding-3-large` (3072 dimensions)

```rust
let embedder = ApiEmbedder::builder()
    .openai("your-api-key")
    .model("text-embedding-3-small")
    .build()
    .await?;
```

### Anthropic (Planned)

Support for Anthropic embedding models is planned for future releases.

### Custom Providers

You can integrate with custom embedding APIs:

```rust
let config = ApiEmbedderConfig::custom(
    "my-provider",
    "https://api.myprovider.com",
    "api-key",
    "custom-model"
);
let embedder = ApiEmbedder::from_config(config).await?;
```

## Configuration

### Builder Pattern

The builder pattern provides a fluent API for configuration:

```rust
let embedder = ApiEmbedder::builder()
    .openai("your-api-key")
    .model("text-embedding-3-large")
    .batch_size(50)                    // Process 50 texts per batch
    .max_retries(5)                    // Retry failed requests 5 times
    .timeout(Duration::from_secs(60))  // 60-second timeout
    .enable_cache(true)                // Enable caching
    .build()
    .await?;
```

### Configuration Struct

For more complex scenarios, use the configuration struct:

```rust
let config = ApiEmbedderConfig::openai("your-api-key", "text-embedding-3-small")
    .with_batch_size(100)
    .with_cache_ttl(Duration::from_secs(3600))  // 1-hour cache TTL
    .with_base_url("https://custom.openai.com") // Custom endpoint
    .with_config("temperature", serde_json::json!(0.7)); // Additional config

let embedder = ApiEmbedder::from_config(config).await?;
```

## Caching

ApiEmbedder includes intelligent caching to reduce API costs and improve performance:

```rust
let embedder = ApiEmbedder::builder()
    .openai("your-api-key")
    .enable_cache(true)                        // Enable caching
    .build()
    .await?;

// First call hits the API
let embedding1 = embedder.embed("Hello, world!").await?;

// Second call uses cache (much faster!)
let embedding2 = embedder.embed("Hello, world!").await?;

// Check cache statistics
if let Some(stats) = embedder.cache_stats().await {
    println!("Cache hit rate: {:.2}%", stats.hit_rate());
    println!("Total entries: {}", stats.total_entries);
}
```

### Cache Management

```rust
// Clear cache
embedder.clear_cache().await?;

// Cleanup expired entries
let removed = embedder.cleanup_cache().await?;
println!("Removed {} expired entries", removed);
```

## Error Handling

ApiEmbedder provides comprehensive error handling with automatic retry logic:

```rust
use cheungfun_integrations::embedders::api::ApiEmbedderError;

match embedder.embed("test").await {
    Ok(embedding) => println!("Success: {} dimensions", embedding.len()),
    Err(e) => match e.downcast_ref::<ApiEmbedderError>() {
        Some(ApiEmbedderError::RateLimit { .. }) => {
            println!("Rate limited - will retry automatically");
        }
        Some(ApiEmbedderError::Authentication { .. }) => {
            println!("Invalid API key");
        }
        Some(ApiEmbedderError::Network { .. }) => {
            println!("Network error - will retry automatically");
        }
        _ => println!("Other error: {}", e),
    }
}
```

### Retryable Errors

The following errors are automatically retried with exponential backoff:

- Network errors
- Server errors (5xx status codes)
- Timeout errors
- Rate limit errors (with longer delay)

## Monitoring and Statistics

Track usage and performance with built-in statistics:

```rust
// Embedding statistics
let stats = embedder.stats().await;
println!("Texts embedded: {}", stats.texts_embedded);
println!("Failed embeddings: {}", stats.embeddings_failed);
println!("Average time: {:?}", stats.avg_time);

// Cache statistics (if caching enabled)
if let Some(cache_stats) = embedder.cache_stats().await {
    println!("Hit rate: {:.2}%", cache_stats.hit_rate());
    println!("Cache size: {}", cache_stats.total_entries);
}

// Health check
match embedder.health_check().await {
    Ok(()) => println!("API is healthy"),
    Err(e) => println!("API health check failed: {}", e),
}
```

## Best Practices

### 1. Use Appropriate Models

- **text-embedding-3-small**: Best balance of cost and performance
- **text-embedding-3-large**: Higher quality for critical applications
- **text-embedding-ada-002**: Legacy model, use newer models when possible

### 2. Optimize Batch Sizes

```rust
let embedder = ApiEmbedder::builder()
    .openai("your-api-key")
    .batch_size(100)  // Optimize based on your use case
    .build()
    .await?;
```

### 3. Enable Caching for Repeated Texts

```rust
let embedder = ApiEmbedder::builder()
    .openai("your-api-key")
    .enable_cache(true)
    .build()
    .await?;
```

### 4. Handle Errors Gracefully

```rust
async fn embed_with_fallback(embedder: &ApiEmbedder, text: &str) -> Vec<f32> {
    match embedder.embed(text).await {
        Ok(embedding) => embedding,
        Err(e) => {
            eprintln!("Embedding failed: {}", e);
            vec![0.0; embedder.dimension()] // Return zero vector as fallback
        }
    }
}
```

### 5. Monitor Usage

```rust
// Periodically check statistics
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    loop {
        interval.tick().await;
        let stats = embedder.stats().await;
        println!("Embedded {} texts, {} failed", 
                 stats.texts_embedded, stats.embeddings_failed);
    }
});
```

## Examples

See the [examples directory](../examples/) for complete working examples:

- `api_embedder_example.rs` - Comprehensive usage examples
- Integration with vector databases
- Semantic search implementations
- Batch processing workflows

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify your API key is correct
   - Check if the API key has necessary permissions

2. **Rate Limiting**
   - ApiEmbedder automatically handles rate limits with retries
   - Consider reducing batch size if you hit limits frequently

3. **Network Timeouts**
   - Increase timeout duration for slow networks
   - Check your internet connection

4. **Model Not Found**
   - Verify the model name is correct
   - Check if the model is available for your API key

### Debug Logging

Enable debug logging to troubleshoot issues:

```rust
use tracing_subscriber;

// Initialize logging
tracing_subscriber::init();

// Now ApiEmbedder will log debug information
let embedder = ApiEmbedder::builder()
    .openai("your-api-key")
    .build()
    .await?;
```

## Contributing

Contributions are welcome! Please see the main project's contributing guidelines.

## License

This project is licensed under the same license as the main Cheungfun project.
