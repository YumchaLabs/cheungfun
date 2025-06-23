//! Performance demonstration for CandleEmbedder
//!
//! This example shows how to use CandleEmbedder efficiently with batch processing
//! and demonstrates performance characteristics.

use cheungfun_core::traits::Embedder;
use cheungfun_integrations::embedders::candle::{
    CandleEmbedder, CandleEmbedderConfig,
};
use std::time::Instant;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 CandleEmbedder Performance Demo");
    println!("==================================");

    // Configuration for a fast, lightweight model
    let config = CandleEmbedderConfig {
        model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        revision: "main".to_string(),
        dimension: Some(384),
        normalize: true,
        max_length: 128,
        batch_size: 16, // Optimized batch size
        cache_dir: Some("./model_cache".into()),
        device: None, // Use CPU for this demo
    };

    println!("📥 Loading model: {}", config.model_name);
    let start_time = Instant::now();
    
    let embedder = CandleEmbedder::from_pretrained(config).await?;
    let load_time = start_time.elapsed();
    
    println!("✅ Model loaded in {:?}", load_time);
    println!("📊 Embedding dimension: {}", embedder.dimension());
    println!();

    // Demo 1: Single text embedding
    println!("🔍 Demo 1: Single Text Embedding");
    println!("---------------------------------");
    
    let single_text = "This is a sample sentence for embedding generation.";
    let start_time = Instant::now();
    
    let embedding = embedder.embed(single_text).await?;
    let single_time = start_time.elapsed();
    
    println!("Text: \"{}\"", single_text);
    println!("Embedding dimension: {}", embedding.len());
    println!("Time taken: {:?}", single_time);
    println!("First 5 values: {:?}", &embedding[0..5]);
    println!();

    // Demo 2: Batch processing comparison
    println!("⚡ Demo 2: Batch vs Individual Processing");
    println!("----------------------------------------");
    
    let test_texts: Vec<String> = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Natural language processing enables computers to understand text.",
        "Embeddings capture semantic meaning in vector space.",
        "Batch processing improves computational efficiency.",
        "Deep learning models require significant computational resources.",
        "Transformers have revolutionized natural language understanding.",
        "Vector databases enable semantic search capabilities.",
        "Artificial intelligence is advancing rapidly.",
        "Text embeddings are useful for many applications.",
    ];

    let text_refs: Vec<&str> = test_texts.iter().map(|s| s.as_str()).collect();

    // Individual processing
    println!("🐌 Processing {} texts individually...", test_texts.len());
    let start_time = Instant::now();
    
    let mut individual_embeddings = Vec::new();
    for text in &text_refs {
        let embedding = embedder.embed(text).await?;
        individual_embeddings.push(embedding);
    }
    
    let individual_time = start_time.elapsed();
    let individual_rate = test_texts.len() as f64 / individual_time.as_secs_f64();

    // Batch processing
    println!("🚀 Processing {} texts in batch...", test_texts.len());
    let start_time = Instant::now();
    
    let batch_embeddings = embedder.embed_batch(text_refs).await?;
    let batch_time = start_time.elapsed();
    let batch_rate = test_texts.len() as f64 / batch_time.as_secs_f64();

    println!();
    println!("📈 Performance Comparison:");
    println!("  Individual processing: {:?} ({:.2} texts/sec)", individual_time, individual_rate);
    println!("  Batch processing:      {:?} ({:.2} texts/sec)", batch_time, batch_rate);
    println!("  Speedup:               {:.2}x", batch_rate / individual_rate);
    println!();

    // Demo 3: Large batch processing
    println!("🏋️  Demo 3: Large Batch Processing");
    println!("----------------------------------");
    
    let large_batch_size = 100;
    let large_texts: Vec<String> = (0..large_batch_size)
        .map(|i| format!("This is test sentence number {} with various content to demonstrate large batch processing capabilities.", i))
        .collect();
    
    let large_text_refs: Vec<&str> = large_texts.iter().map(|s| s.as_str()).collect();
    
    println!("📊 Processing {} texts in large batch...", large_batch_size);
    let start_time = Instant::now();
    
    let large_embeddings = embedder.embed_batch(large_text_refs).await?;
    let large_time = start_time.elapsed();
    let large_rate = large_batch_size as f64 / large_time.as_secs_f64();
    
    println!("✅ Processed {} embeddings in {:?}", large_embeddings.len(), large_time);
    println!("📈 Processing rate: {:.2} texts/second", large_rate);
    println!();

    // Demo 4: Semantic similarity
    println!("🔗 Demo 4: Semantic Similarity");
    println!("------------------------------");
    
    let similarity_texts = vec![
        "The cat is sleeping on the couch.",
        "A feline is resting on the sofa.",
        "The dog is running in the park.",
    ];
    
    let similarity_refs: Vec<&str> = similarity_texts.iter().map(|s| s.as_str()).collect();
    let similarity_embeddings = embedder.embed_batch(similarity_refs).await?;
    
    // Calculate similarities
    let sim_1_2 = cosine_similarity(&similarity_embeddings[0], &similarity_embeddings[1]);
    let sim_1_3 = cosine_similarity(&similarity_embeddings[0], &similarity_embeddings[2]);
    
    println!("Text 1: \"{}\"", similarity_texts[0]);
    println!("Text 2: \"{}\"", similarity_texts[1]);
    println!("Text 3: \"{}\"", similarity_texts[2]);
    println!();
    println!("Similarity (Text 1 ↔ Text 2): {:.3}", sim_1_2);
    println!("Similarity (Text 1 ↔ Text 3): {:.3}", sim_1_3);
    println!("✅ Similar texts have higher similarity: {}", sim_1_2 > sim_1_3);
    println!();

    // Demo 5: Memory and performance statistics
    println!("📊 Demo 5: Performance Statistics");
    println!("---------------------------------");
    
    let stats = embedder.stats().await;
    println!("Total texts embedded: {}", stats.texts_embedded);
    println!("Total processing time: {:?}", stats.duration);
    println!("Average time per text: {:?}", stats.avg_time_per_text);
    println!();

    println!("🎉 Performance demo completed successfully!");
    println!();
    println!("💡 Tips for optimal performance:");
    println!("  • Use batch processing for multiple texts");
    println!("  • Choose appropriate batch sizes (8-32 typically work well)");
    println!("  • Consider using GPU acceleration for large workloads");
    println!("  • Cache models locally to avoid repeated downloads");
    println!("  • Use shorter max_length for faster processing if possible");

    Ok(())
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
