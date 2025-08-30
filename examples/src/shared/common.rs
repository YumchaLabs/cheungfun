//! Common utilities for Cheungfun examples
//!
//! This module provides shared utilities, helpers, and mock implementations
//! that are used across multiple examples.

use anyhow::Result;
use cheungfun_core::{error::CheungfunError, traits::Embedder};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tempfile::TempDir;
use tokio;

/// Mock embedder for demonstration purposes
///
/// This embedder generates deterministic embeddings based on text hashing.
/// It's useful for examples and testing, but should not be used in production.
#[derive(Debug)]
pub struct MockEmbedder {
    dimension: usize,
}

impl MockEmbedder {
    /// Create a new mock embedder with the specified dimension
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait::async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, CheungfunError> {
        // Generate a deterministic embedding based on text hash
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        let mut embedding = vec![0.0; self.dimension];
        for (i, item) in embedding.iter_mut().enumerate().take(self.dimension) {
            *item = ((hash.wrapping_add(i as u64) % 1000) as f32 - 500.0) / 500.0;
        }

        // Normalize the embedding
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(embedding)
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, CheungfunError> {
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.embed(text).await?);
        }
        Ok(embeddings)
    }

    fn name(&self) -> &'static str {
        "MockEmbedder"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &'static str {
        "mock-embedder-v1"
    }
}

/// Create sample documents for testing and examples
pub async fn create_sample_documents() -> Result<TempDir> {
    let temp_dir = TempDir::new()?;

    let documents = vec![
        (
            "rust_programming.txt",
            include_str!("../../shared/test_data/rust_programming.txt"),
        ),
        (
            "machine_learning.txt",
            include_str!("../../shared/test_data/machine_learning.txt"),
        ),
        (
            "web_development.txt",
            include_str!("../../shared/test_data/web_development.txt"),
        ),
    ];

    for (filename, content) in documents {
        let file_path = temp_dir.path().join(filename);
        tokio::fs::write(file_path, content).await?;
    }

    Ok(temp_dir)
}

/// Get all document paths from a directory
pub fn get_document_paths(temp_dir: &TempDir) -> Result<Vec<std::path::PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(temp_dir.path())? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            paths.push(entry.path());
        }
    }
    Ok(paths)
}

/// Format duration in a human-readable way
#[must_use]
pub fn format_duration(duration: std::time::Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() > 0 {
        format!("{}ms", duration.as_millis())
    } else {
        format!("{}µs", duration.as_micros())
    }
}

/// Format bytes in a human-readable way
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

/// Print a section header
pub fn print_section(title: &str) {
    println!();
    println!("🔹 {title}");
    println!("{}", "=".repeat(title.len() + 3));
}

/// Print a subsection header
pub fn print_subsection(title: &str) {
    println!();
    println!("📌 {title}");
    println!("{}", "-".repeat(title.len() + 3));
}

/// Print a success message
pub fn print_success(message: &str) {
    println!("✅ {message}");
}

/// Print an info message
pub fn print_info(message: &str) {
    println!("ℹ️  {message}");
}

/// Print a warning message
pub fn print_warning(message: &str) {
    println!("⚠️  {message}");
}

/// Print an error message
pub fn print_error(message: &str) {
    println!("❌ {message}");
}

/// Initialize logging for examples
pub fn init_logging() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
}

/// Check if a feature is available
#[must_use]
pub fn check_feature(feature_name: &str) -> bool {
    match feature_name {
        "fastembed" => cfg!(feature = "fastembed"),
        "candle" => cfg!(feature = "candle"),
        "qdrant" => true, // Always available as it's a client
        _ => false,
    }
}

/// Print feature availability
pub fn print_feature_status() {
    println!("🔍 Feature Availability:");
    println!(
        "  • FastEmbed: {}",
        if check_feature("fastembed") {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "  • Candle: {}",
        if check_feature("candle") {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "  • Qdrant: {}",
        if check_feature("qdrant") {
            "✅"
        } else {
            "❌"
        }
    );

    // Check environment variables
    println!("🔑 Environment Variables:");
    println!(
        "  • OPENAI_API_KEY: {}",
        if std::env::var("OPENAI_API_KEY").is_ok() {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "  • SIUMAI_API_KEY: {}",
        if std::env::var("SIUMAI_API_KEY").is_ok() {
            "✅"
        } else {
            "❌"
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_embedder() {
        let embedder = MockEmbedder::new(384);

        let text = "Hello, world!";
        let embedding = embedder.embed(text).await.unwrap();

        assert_eq!(embedding.len(), 384);

        // Test that the same text produces the same embedding
        let embedding2 = embedder.embed(text).await.unwrap();
        assert_eq!(embedding, embedding2);

        // Test that different text produces different embeddings
        let embedding3 = embedder.embed("Different text").await.unwrap();
        assert_ne!(embedding, embedding3);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(std::time::Duration::from_secs(5)), "5.00s");
        assert_eq!(
            format_duration(std::time::Duration::from_millis(500)),
            "500ms"
        );
        assert_eq!(
            format_duration(std::time::Duration::from_micros(500)),
            "500µs"
        );
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}
