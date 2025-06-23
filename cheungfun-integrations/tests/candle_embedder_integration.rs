//! Integration tests for CandleEmbedder
//!
//! These tests verify the complete functionality of the CandleEmbedder,
//! including model downloading, loading, and embedding generation.
//!
//! **Note**: These tests require the `candle` feature to be enabled.
//! Run with: `cargo test --features candle`

#![cfg(feature = "candle")]

use cheungfun_core::traits::Embedder;
use cheungfun_integrations::embedders::candle::{CandleEmbedder, CandleEmbedderConfig};
use std::collections::HashMap;

/// Test configuration for a small BERT model suitable for testing
fn test_config() -> CandleEmbedderConfig {
    CandleEmbedderConfig {
        model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        revision: "main".to_string(),
        max_length: 128,
        normalize: true,
        batch_size: 8,
        dimension: Some(384),
        device: "cpu".to_string(), // Use CPU for testing
        use_half_precision: false,
        cache_dir: Some("./test_cache".to_string()),
        trust_remote_code: false,
        model_config: HashMap::new(),
    }
}

#[tokio::test]
#[ignore] // Ignore by default since it requires network access
async fn test_candle_embedder_initialization() {
    let config = test_config();
    let embedder = CandleEmbedder::from_config(config).await;

    assert!(
        embedder.is_ok(),
        "Failed to initialize CandleEmbedder: {:?}",
        embedder.err()
    );

    let embedder = embedder.unwrap();
    assert_eq!(embedder.dimension(), 384);
}

#[tokio::test]
#[ignore] // Ignore by default since it requires network access
async fn test_candle_embedder_from_pretrained() {
    let embedder = CandleEmbedder::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").await;

    assert!(
        embedder.is_ok(),
        "Failed to initialize CandleEmbedder: {:?}",
        embedder.err()
    );

    let embedder = embedder.unwrap();
    assert_eq!(embedder.dimension(), 384);
}

#[tokio::test]
#[ignore] // Ignore by default since it requires network access
async fn test_single_text_embedding() {
    let config = test_config();
    let embedder = CandleEmbedder::from_config(config)
        .await
        .expect("Failed to create embedder");

    let text = "Hello, world! This is a test sentence.";
    let embedding = embedder.embed(text).await;

    assert!(
        embedding.is_ok(),
        "Failed to generate embedding: {:?}",
        embedding.err()
    );

    let embedding = embedding.unwrap();
    assert_eq!(embedding.len(), 384, "Embedding dimension mismatch");

    // Check that embedding is normalized (approximately unit length)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "Embedding is not normalized: norm = {}",
        norm
    );
}

#[tokio::test]
#[ignore] // Ignore by default since it requires network access
async fn test_batch_embedding() {
    let config = test_config();
    let embedder = CandleEmbedder::from_config(config)
        .await
        .expect("Failed to create embedder");

    let texts = vec![
        "This is the first sentence.",
        "Here is another sentence.",
        "And this is the third one.",
    ];

    let embeddings = embedder.embed_batch(texts).await;

    assert!(
        embeddings.is_ok(),
        "Failed to generate batch embeddings: {:?}",
        embeddings.err()
    );

    let embeddings = embeddings.unwrap();
    assert_eq!(embeddings.len(), 3, "Wrong number of embeddings");

    for (i, embedding) in embeddings.iter().enumerate() {
        assert_eq!(embedding.len(), 384, "Embedding {} dimension mismatch", i);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embedding {} is not normalized: norm = {}",
            i,
            norm
        );
    }
}

#[tokio::test]
#[ignore] // Ignore by default since it requires network access
async fn test_embedding_similarity() {
    let config = test_config();
    let embedder = CandleEmbedder::from_config(config)
        .await
        .expect("Failed to create embedder");

    let similar_texts = vec!["The cat sits on the mat.", "A cat is sitting on a mat."];

    let different_texts = vec!["The cat sits on the mat.", "The weather is sunny today."];

    let similar_embeddings = embedder
        .embed_batch(similar_texts)
        .await
        .expect("Failed to generate similar embeddings");

    let different_embeddings = embedder
        .embed_batch(different_texts)
        .await
        .expect("Failed to generate different embeddings");

    // Calculate cosine similarity
    let similar_similarity = cosine_similarity(&similar_embeddings[0], &similar_embeddings[1]);
    let different_similarity =
        cosine_similarity(&different_embeddings[0], &different_embeddings[1]);

    // Similar texts should have higher similarity than different texts
    assert!(
        similar_similarity > different_similarity,
        "Similar texts similarity ({:.3}) should be higher than different texts similarity ({:.3})",
        similar_similarity,
        different_similarity
    );

    // Similar texts should have reasonably high similarity
    assert!(
        similar_similarity > 0.7,
        "Similar texts should have high similarity, got {:.3}",
        similar_similarity
    );
}

#[tokio::test]
#[ignore] // Ignore by default since it requires network access
async fn test_large_batch_processing() {
    let config = test_config();
    let embedder = CandleEmbedder::from_config(config)
        .await
        .expect("Failed to create embedder");

    // Create a large batch of texts
    let texts: Vec<String> = (0..20)
        .map(|i| format!("This is test sentence number {}.", i))
        .collect();

    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    let start_time = std::time::Instant::now();
    let embeddings = embedder.embed_batch(text_refs).await;
    let duration = start_time.elapsed();

    assert!(
        embeddings.is_ok(),
        "Failed to process large batch: {:?}",
        embeddings.err()
    );

    let embeddings = embeddings.unwrap();
    assert_eq!(
        embeddings.len(),
        20,
        "Wrong number of embeddings in large batch"
    );

    println!("Processed {} texts in {:?}", embeddings.len(), duration);

    // Verify all embeddings are valid
    for (i, embedding) in embeddings.iter().enumerate() {
        assert_eq!(embedding.len(), 384, "Embedding {} dimension mismatch", i);
        assert!(
            !embedding.iter().any(|&x| x.is_nan()),
            "Embedding {} contains NaN",
            i
        );
    }
}

#[tokio::test]
#[ignore] // Ignore by default since it requires network access
async fn test_empty_text_handling() {
    let config = test_config();
    let embedder = CandleEmbedder::from_config(config)
        .await
        .expect("Failed to create embedder");

    let empty_text = "";
    let embedding = embedder.embed(empty_text).await;

    // Should handle empty text gracefully
    assert!(
        embedding.is_ok(),
        "Failed to handle empty text: {:?}",
        embedding.err()
    );

    let embedding = embedding.unwrap();
    assert_eq!(
        embedding.len(),
        384,
        "Empty text embedding dimension mismatch"
    );
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

/// Benchmark test for performance measurement
#[tokio::test]
#[ignore] // Ignore by default since it's a benchmark
async fn benchmark_embedding_performance() {
    let config = test_config();
    let embedder = CandleEmbedder::from_config(config)
        .await
        .expect("Failed to create embedder");

    let test_texts: Vec<String> = (0..100)
        .map(|i| {
            format!(
                "This is a longer test sentence number {} with more words to process.",
                i
            )
        })
        .collect();

    let text_refs: Vec<&str> = test_texts.iter().map(|s| s.as_str()).collect();

    // Warm up
    let _ = embedder.embed_batch(text_refs[0..5].to_vec()).await;

    // Benchmark batch processing
    let start_time = std::time::Instant::now();
    let embeddings = embedder
        .embed_batch(text_refs)
        .await
        .expect("Benchmark failed");
    let duration = start_time.elapsed();

    let texts_per_second = embeddings.len() as f64 / duration.as_secs_f64();

    println!("Benchmark results:");
    println!("  Processed {} texts in {:?}", embeddings.len(), duration);
    println!("  Performance: {:.2} texts/second", texts_per_second);

    // Basic performance assertion (adjust based on your requirements)
    assert!(
        texts_per_second > 1.0,
        "Performance too slow: {:.2} texts/second",
        texts_per_second
    );
}
