//! Basic Indexing - Learn document processing and indexing
//!
//! This example demonstrates:
//! 1. Loading documents from various sources
//! 2. Text splitting and chunking strategies
//! 3. Embedding generation
//! 4. Vector storage
//! 5. Monitoring the indexing process

use anyhow::Result;
use cheungfun_core::{
    DistanceMetric,
    types::{ChunkInfo, Document},
};
use cheungfun_indexing::{
    loaders::text::TextLoader, pipeline::IndexingPipelineBuilder,
    transformers::text_splitter::TextSplitter,
};
use cheungfun_integrations::{
    embedders::api::{ApiEmbedder, ApiEmbedderConfig},
    vector_stores::memory::InMemoryVectorStore,
};
use std::sync::Arc;
use tempfile::TempDir;
use tokio;
use tracing::{Level, info};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    println!("📚 Cheungfun Basic Indexing Example");
    println!("===================================");
    println!();

    // Step 1: Create sample documents
    println!("📝 Step 1: Creating sample documents...");
    let temp_dir = create_sample_documents().await?;
    println!("✅ Created sample documents in temporary directory");
    println!();

    // Step 2: Configure text splitting
    println!("✂️  Step 2: Configuring text splitting...");

    // Different chunking strategies for different use cases
    let strategies = vec![
        ("Small Chunks (200 chars)", TextSplitter::new(200, 20)),
        ("Medium Chunks (500 chars)", TextSplitter::new(500, 50)),
        ("Large Chunks (1000 chars)", TextSplitter::new(1000, 100)),
    ];

    for (name, splitter) in &strategies {
        println!(
            "  📏 {}: chunk_size={}, overlap={}",
            name,
            splitter.chunk_size(),
            splitter.overlap()
        );
    }
    println!();

    // Step 3: Set up embedder
    println!("🧠 Step 3: Setting up embedder...");
    let embedder = Arc::new(create_embedder().await?);
    println!("✅ Embedder ready: {}", embedder.name());
    println!();

    // Step 4: Index with different strategies
    for (strategy_name, text_splitter) in strategies {
        println!("🔄 Processing with {}...", strategy_name);

        // Create vector store for this strategy
        let vector_store = Arc::new(InMemoryVectorStore::new(384, DistanceMetric::Cosine));

        // Create indexing pipeline
        let pipeline = IndexingPipelineBuilder::new()
            .with_loader(TextLoader::new(get_document_paths(&temp_dir)?))
            .with_transformer_arc(Arc::new(text_splitter))
            .with_embedder_arc(embedder.clone())
            .with_vector_store_arc(vector_store.clone())
            .build()?;

        // Run indexing and measure time
        let start_time = std::time::Instant::now();
        pipeline.run().await?;
        let duration = start_time.elapsed();

        // Get statistics (mock implementation)
        let chunk_count = get_chunk_count(&vector_store).await?;

        println!("  ✅ Completed in {:?}", duration);
        println!("  📊 Generated {} chunks", chunk_count);
        println!(
            "  ⚡ Processing rate: {:.1} chunks/sec",
            chunk_count as f64 / duration.as_secs_f64()
        );
        println!();
    }

    // Step 5: Demonstrate advanced indexing features
    println!("🎯 Step 5: Advanced indexing features...");
    demonstrate_advanced_features().await?;

    println!("🎉 Basic indexing example completed!");
    println!();
    println!("🚀 Next Steps:");
    println!("  1. Try basic_querying.rs to learn about searching your indexed documents");
    println!("  2. Explore 02_core_components/embedders/ for different embedding options");
    println!("  3. Check 02_core_components/vector_stores/ for storage alternatives");
    println!("  4. See 05_performance/optimization/ for indexing performance tips");

    Ok(())
}

/// Create sample documents for indexing
async fn create_sample_documents() -> Result<TempDir> {
    let temp_dir = TempDir::new()?;

    let documents = vec![
        (
            "rust_programming.txt",
            r#"
# Rust Programming Language

Rust is a systems programming language that runs blazingly fast, prevents segfaults, 
and guarantees thread safety. It accomplishes these goals by being memory safe without 
using garbage collection.

## Key Features

- Zero-cost abstractions
- Move semantics
- Guaranteed memory safety
- Threads without data races
- Trait-based generics
- Pattern matching
- Type inference
- Minimal runtime
- Efficient C bindings

## Use Cases

Rust is perfect for:
- System programming
- Web backends
- Network services
- Embedded systems
- Blockchain applications
- Game engines
"#,
        ),
        (
            "machine_learning.txt",
            r#"
# Machine Learning with Rust

Rust is increasingly being used for machine learning applications due to its 
performance characteristics and safety guarantees.

## ML Libraries in Rust

- **Candle**: A minimalist ML framework for Rust
- **tch**: PyTorch bindings for Rust
- **SmartCore**: Comprehensive ML library
- **Linfa**: A toolkit for classical ML

## Advantages

- Memory safety prevents common ML bugs
- Zero-cost abstractions for performance
- Excellent parallelization capabilities
- Growing ecosystem of ML tools

## Applications

- High-performance inference engines
- Real-time ML systems
- Edge computing applications
- Scientific computing
"#,
        ),
        (
            "web_development.txt",
            r#"
# Web Development with Rust

Rust has a thriving web development ecosystem with frameworks that prioritize 
performance and safety.

## Popular Frameworks

- **Axum**: Modern async web framework
- **Actix-web**: High-performance web framework
- **Warp**: Lightweight web framework
- **Rocket**: Type-safe web framework

## Benefits

- Excellent performance
- Memory safety
- Strong type system
- Great async support
- Growing ecosystem

## Use Cases

- REST APIs
- GraphQL servers
- Microservices
- Real-time applications
- High-traffic web services
"#,
        ),
    ];

    for (filename, content) in documents {
        let file_path = temp_dir.path().join(filename);
        tokio::fs::write(file_path, content).await?;
    }

    Ok(temp_dir)
}

/// Get paths of all documents in the temp directory
fn get_document_paths(temp_dir: &TempDir) -> Result<Vec<std::path::PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(temp_dir.path())? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            paths.push(entry.path());
        }
    }
    Ok(paths)
}

/// Create embedder (mock for this example)
async fn create_embedder() -> Result<MockEmbedder> {
    Ok(MockEmbedder::new(384))
}

/// Get chunk count from vector store (mock implementation)
async fn get_chunk_count(vector_store: &InMemoryVectorStore) -> Result<usize> {
    // In a real implementation, you'd query the vector store
    // For this example, we'll return a mock count
    Ok(42) // Mock value
}

/// Demonstrate advanced indexing features
async fn demonstrate_advanced_features() -> Result<()> {
    println!("🔧 Advanced Features:");
    println!("  📊 Batch Processing: Process multiple documents efficiently");
    println!("  🔄 Incremental Updates: Add new documents without full reindex");
    println!("  📈 Progress Monitoring: Track indexing progress in real-time");
    println!("  🎯 Custom Transformers: Apply domain-specific text processing");
    println!("  💾 Persistent Storage: Save indexes for later use");
    println!("  🔍 Metadata Extraction: Extract and index document metadata");

    Ok(())
}

/// Mock embedder for demonstration
struct MockEmbedder {
    dimension: usize,
}

impl MockEmbedder {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait::async_trait]
impl cheungfun_core::traits::Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, cheungfun_core::error::CheungfunError> {
        // Simple hash-based embedding for demonstration
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        let mut embedding = vec![0.0; self.dimension];
        for i in 0..self.dimension {
            embedding[i] = ((hash.wrapping_add(i as u64) % 1000) as f32 - 500.0) / 500.0;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(embedding)
    }

    async fn embed_batch(
        &self,
        texts: Vec<&str>,
    ) -> Result<Vec<Vec<f32>>, cheungfun_core::error::CheungfunError> {
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.embed(text).await?);
        }
        Ok(embeddings)
    }

    fn name(&self) -> &'static str {
        "MockEmbedder"
    }
}
