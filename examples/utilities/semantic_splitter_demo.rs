//! Semantic Splitter Demo
//!
//! This example demonstrates the SemanticSplitter functionality,
//! showing how it groups semantically related sentences together.

use async_trait::async_trait;
use cheungfun_core::{traits::Embedder, Document, Result as CoreResult};
use cheungfun_indexing::node_parser::{text::SemanticSplitter, NodeParser, TextSplitter};
use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber;

// Mock embedder for demonstration purposes
#[derive(Debug)]
struct MockEmbedder {
    dimension: usize,
}

impl MockEmbedder {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> CoreResult<Vec<f32>> {
        // Create a simple hash-based embedding for testing
        let mut embedding = vec![0.0; self.dimension];
        let text_hash = text
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));

        for (i, value) in embedding.iter_mut().enumerate() {
            let seed = (text_hash.wrapping_add(i as u64)) as f32;
            *value = (seed * 0.001).sin();
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in embedding.iter_mut() {
                *value /= norm;
            }
        }

        Ok(embedding)
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> CoreResult<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.embed(text).await?);
        }
        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        "mock-embedder"
    }
}

#[tokio::main]
async fn main() -> CoreResult<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🧠 Semantic Splitter Demo");
    info!("========================");

    // Create embedder - using a mock embedder for demo purposes
    info!("Creating mock embedder for demonstration...");
    let embedder = Arc::new(MockEmbedder::new(384));

    // Create semantic splitter with different configurations
    info!("Creating semantic splitters with different configurations...");

    let splitter_basic = SemanticSplitter::new(embedder.clone());

    let splitter_with_buffer = SemanticSplitter::new(embedder.clone())
        .with_buffer_size(2)
        .with_breakpoint_percentile_threshold(90.0);

    let splitter_strict = SemanticSplitter::new(embedder.clone())
        .with_buffer_size(1)
        .with_breakpoint_percentile_threshold(80.0);

    // Test text with clear semantic boundaries
    let test_text = r#"
        Machine learning is a subset of artificial intelligence. It focuses on algorithms that can learn from data.
        Deep learning uses neural networks with multiple layers. These networks can recognize complex patterns in data.
        
        The weather today is sunny and warm. It's a perfect day for outdoor activities.
        Many people enjoy hiking in good weather. The fresh air and exercise are beneficial for health.
        
        Quantum computing represents a new paradigm in computation. It uses quantum mechanical phenomena like superposition.
        Quantum computers could solve certain problems exponentially faster than classical computers.
        This technology is still in early development but shows great promise.
        
        Cooking is both an art and a science. Different cultures have developed unique culinary traditions.
        Spices and herbs play crucial roles in flavor development. The combination of ingredients creates complex tastes.
    "#;

    info!("\n📝 Test Text:");
    info!("{}", test_text.trim());

    // Test basic semantic splitter
    info!("\n🔍 Basic Semantic Splitter (buffer_size=1, threshold=95%):");
    let chunks_basic = splitter_basic.split_text_async(test_text).await?;
    for (i, chunk) in chunks_basic.iter().enumerate() {
        info!("Chunk {}: {}", i + 1, chunk.trim());
    }

    // Test splitter with buffer
    info!("\n🔍 Semantic Splitter with Buffer (buffer_size=2, threshold=90%):");
    let chunks_buffer = splitter_with_buffer.split_text_async(test_text).await?;
    for (i, chunk) in chunks_buffer.iter().enumerate() {
        info!("Chunk {}: {}", i + 1, chunk.trim());
    }

    // Test strict splitter
    info!("\n🔍 Strict Semantic Splitter (buffer_size=1, threshold=80%):");
    let chunks_strict = splitter_strict.split_text_async(test_text).await?;
    for (i, chunk) in chunks_strict.iter().enumerate() {
        info!("Chunk {}: {}", i + 1, chunk.trim());
    }

    // Test with NodeParser interface
    info!("\n📄 Testing NodeParser Interface:");
    let document = Document::new(test_text);
    let nodes =
        <SemanticSplitter as NodeParser>::parse_nodes(&splitter_basic, &[document], false).await?;

    info!("Generated {} nodes:", nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        info!("Node {}: {}", i + 1, node.content.trim());
        info!(
            "  Metadata: chunk_index={}, splitter_type={}",
            node.metadata
                .get("chunk_index")
                .map(|v| v.to_string())
                .unwrap_or_default(),
            node.metadata
                .get("splitter_type")
                .map(|v| v.to_string())
                .unwrap_or_default()
        );
    }

    // Compare with sentence splitter
    info!("\n📊 Comparison with SentenceSplitter:");
    let sentence_splitter =
        cheungfun_indexing::node_parser::text::SentenceSplitter::from_defaults(1000, 200)?;
    let sentence_chunks = sentence_splitter.split_text(test_text)?;

    info!("Semantic Splitter: {} chunks", chunks_basic.len());
    info!("Sentence Splitter: {} chunks", sentence_chunks.len());

    // Performance comparison
    info!("\n⏱️  Performance Comparison:");
    let start = std::time::Instant::now();
    let _ = splitter_basic.split_text_async(test_text).await?;
    let semantic_time = start.elapsed();

    let start = std::time::Instant::now();
    let _ = sentence_splitter.split_text(test_text)?;
    let sentence_time = start.elapsed();

    info!("Semantic Splitter: {:?}", semantic_time);
    info!("Sentence Splitter: {:?}", sentence_time);
    info!(
        "Semantic splitting is {:.2}x slower (due to embedding computation)",
        semantic_time.as_secs_f64() / sentence_time.as_secs_f64()
    );

    // Test edge cases
    info!("\n🧪 Testing Edge Cases:");

    // Empty text
    let empty_chunks = splitter_basic.split_text_async("").await?;
    info!("Empty text: {} chunks", empty_chunks.len());

    // Single sentence
    let single_chunks = splitter_basic
        .split_text_async("This is a single sentence.")
        .await?;
    info!("Single sentence: {} chunks", single_chunks.len());

    // Very short text
    let short_chunks = splitter_basic.split_text_async("Short. Text.").await?;
    info!("Very short text: {} chunks", short_chunks.len());

    info!("\n✅ Semantic Splitter Demo completed successfully!");
    info!("\n💡 Key Observations:");
    info!("   • Semantic splitter groups related sentences together");
    info!("   • Buffer size affects context consideration");
    info!("   • Lower threshold creates more, smaller chunks");
    info!("   • Embedding computation adds processing time but improves semantic coherence");

    Ok(())
}

/// Demonstrate advanced semantic splitting features
#[allow(dead_code)]
async fn demonstrate_advanced_features() -> CoreResult<()> {
    info!("\n🚀 Advanced Semantic Splitting Features:");

    // This would require additional implementation
    // - Custom sentence splitters
    // - Different embedding models
    // - Similarity threshold tuning
    // - Chunk size constraints

    Ok(())
}

/// Compare different splitter strategies
#[allow(dead_code)]
async fn compare_splitter_strategies(_text: &str) -> CoreResult<()> {
    info!("\n📈 Splitter Strategy Comparison:");

    // This would compare:
    // - Token-based splitting
    // - Sentence-based splitting
    // - Semantic splitting
    // - Code-aware splitting (for code content)

    // Metrics to compare:
    // - Number of chunks
    // - Average chunk size
    // - Processing time
    // - Semantic coherence (if measurable)

    Ok(())
}
