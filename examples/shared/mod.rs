//! Shared utilities and configurations for Cheungfun RAG examples.
//!
//! This module provides common functionality used across multiple examples,
//! including test utilities, configuration helpers, and data processing functions.

pub mod config;
pub mod test_utils;

// Re-export commonly used items
pub use config::*;
pub use test_utils::*;

use cheungfun_core::prelude::*;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Get the path to the examples data directory
pub fn get_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("data")
}

/// Get the path to a specific data file
pub fn get_data_file(filename: &str) -> PathBuf {
    get_data_dir().join(filename)
}

/// Common constants used across examples
pub mod constants {
    /// Default chunk size for text splitting
    pub const DEFAULT_CHUNK_SIZE: usize = 1000;

    /// Default chunk overlap
    pub const DEFAULT_CHUNK_OVERLAP: usize = 200;

    /// Default number of documents to retrieve
    pub const DEFAULT_TOP_K: usize = 5;

    /// Default embedding dimension for FastEmbed
    pub const DEFAULT_EMBEDDING_DIM: usize = 384;

    /// Climate change PDF file name
    pub const CLIMATE_CHANGE_PDF: &str = "Understanding_Climate_Change.pdf";

    /// Customer CSV file name
    pub const CUSTOMERS_CSV: &str = "customers-100.csv";

    /// Nike annual report text file name
    pub const NIKE_REPORT_TXT: &str = "nike_2023_annual_report.txt";

    /// Q&A JSON file name
    pub const QA_JSON: &str = "q_a.json";
}

/// Common error types for examples
#[derive(Debug, thiserror::Error)]
pub enum ExampleError {
    #[error("Cheungfun error: {0}")]
    Cheungfun(#[from] cheungfun_core::CheungfunError),

    #[error("Indexing error: {0}")]
    Indexing(#[from] cheungfun_indexing::error::IndexingError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Example configuration error: {0}")]
    Config(String),

    #[error("Data processing error: {0}")]
    DataProcessing(String),
}

pub type ExampleResult<T> = std::result::Result<T, ExampleError>;

/// Simple timer for measuring execution time
pub struct Timer {
    name: String,
    start: Instant,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start: Instant::now(),
        }
    }

    pub fn finish(self) -> Duration {
        let duration = self.start.elapsed();
        println!("⏱️  {}: {:.2?}", self.name, duration);
        duration
    }
}

/// Performance metrics collector
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    pub indexing_time: Option<Duration>,
    pub query_times: Vec<Duration>,
    pub total_documents: usize,
    pub total_nodes: usize,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_indexing_time(&mut self, duration: Duration) {
        self.indexing_time = Some(duration);
    }

    pub fn record_query(&mut self, duration: Duration) {
        self.query_times.push(duration);
    }

    pub fn print_summary(&self) {
        println!("\n📊 Performance Summary");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        if let Some(indexing_time) = self.indexing_time {
            println!(
                "📚 Indexing: {:.2?} ({} docs, {} nodes)",
                indexing_time, self.total_documents, self.total_nodes
            );
        }

        if !self.query_times.is_empty() {
            let total_query_time: Duration = self.query_times.iter().sum();
            let avg_query_time = total_query_time / self.query_times.len() as u32;
            println!(
                "🔍 Queries: {} total, avg {:.2?}, total {:.2?}",
                self.query_times.len(),
                avg_query_time,
                total_query_time
            );
        }

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}
