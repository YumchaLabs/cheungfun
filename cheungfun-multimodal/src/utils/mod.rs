//! Utility functions and helpers for multimodal operations.
//!
//! This module provides various utility functions for format detection,
//! content validation, and common operations across different modalities.

pub mod format_detection;
pub mod conversion;

// Re-export commonly used utilities
pub use format_detection::{detect_format, detect_modality_from_path};
#[cfg(feature = "format-detection")]
pub use format_detection::detect_format_from_bytes;
pub use conversion::{convert_to_base64, convert_from_base64, normalize_path};

use crate::types::{MediaContent, MediaFormat, ModalityType};
use crate::error::Result;
use std::path::Path;

/// Validate media content for processing.
pub fn validate_media_content(content: &MediaContent) -> Result<()> {
    // Check if content is not empty
    match &content.data {
        crate::types::MediaData::Embedded(data) => {
            if data.is_empty() {
                return Err(crate::error::MultimodalError::validation("Embedded data is empty"));
            }
        }
        crate::types::MediaData::FilePath(path) => {
            if !path.exists() {
                return Err(crate::error::MultimodalError::file_not_found(path.clone()));
            }
        }
        crate::types::MediaData::Url(url) => {
            if url.is_empty() {
                return Err(crate::error::MultimodalError::validation("URL is empty"));
            }
        }
        crate::types::MediaData::Base64(data) => {
            if data.is_empty() {
                return Err(crate::error::MultimodalError::validation("Base64 data is empty"));
            }
        }
    }

    // Validate format compatibility
    let expected_modality = content.format.modality_type();
    if !crate::types::utils::are_modalities_compatible(expected_modality, content.modality_type()) {
        return Err(crate::error::MultimodalError::invalid_format(
            expected_modality.to_string(),
            content.modality_type().to_string(),
        ));
    }

    Ok(())
}

/// Calculate a checksum for media content.
pub async fn calculate_checksum(content: &MediaContent) -> Result<String> {
    use sha2::{Sha256, Digest};
    
    let data = content.data.as_bytes().await?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

/// Get the estimated processing complexity for content.
pub fn estimate_processing_complexity(content: &MediaContent) -> f32 {
    let base_complexity = match content.modality_type() {
        ModalityType::Text => 1.0,
        ModalityType::Image => 3.0,
        ModalityType::Audio => 4.0,
        ModalityType::Video => 8.0,
        ModalityType::Document => 2.0,
        _ => 2.0,
    };

    // Adjust based on content size
    let size_factor = if let Some(size) = content.estimated_size() {
        (size as f32 / 1_000_000.0).max(1.0) // MB
    } else {
        1.0
    };

    base_complexity * size_factor.sqrt()
}

/// Check if content requires preprocessing.
pub fn requires_preprocessing(content: &MediaContent) -> bool {
    match content.modality_type() {
        ModalityType::Image | ModalityType::Audio | ModalityType::Video => true,
        ModalityType::Document => true,
        _ => false,
    }
}

/// Get the recommended batch size for a modality.
pub fn recommended_batch_size(modality: ModalityType) -> usize {
    crate::traits::utils::optimal_batch_size_for_modality(modality)
}

/// Create a unique identifier for content.
pub fn create_content_id(content: &MediaContent) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    
    // Hash the format
    content.format.hash(&mut hasher);
    
    // Hash metadata if available
    if let Some(ref filename) = content.metadata.filename {
        filename.hash(&mut hasher);
    }
    if let Some(size) = content.metadata.size {
        size.hash(&mut hasher);
    }
    
    // Hash a portion of the data for uniqueness
    match &content.data {
        crate::types::MediaData::Embedded(data) => {
            let sample_size = std::cmp::min(data.len(), 1024);
            data[..sample_size].hash(&mut hasher);
        }
        crate::types::MediaData::FilePath(path) => {
            path.hash(&mut hasher);
        }
        crate::types::MediaData::Url(url) => {
            url.hash(&mut hasher);
        }
        crate::types::MediaData::Base64(data) => {
            let sample_size = std::cmp::min(data.len(), 1024);
            data[..sample_size].hash(&mut hasher);
        }
    }

    format!("content_{:016x}", hasher.finish())
}

/// Sanitize filename for safe storage.
pub fn sanitize_filename(filename: &str) -> String {
    filename
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect()
}

/// Get MIME type from file extension.
pub fn mime_type_from_extension(extension: &str) -> Option<&'static str> {
    match extension.to_lowercase().as_str() {
        "jpg" | "jpeg" => Some("image/jpeg"),
        "png" => Some("image/png"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        "svg" => Some("image/svg+xml"),
        "bmp" => Some("image/bmp"),
        "tiff" | "tif" => Some("image/tiff"),
        
        "mp3" => Some("audio/mpeg"),
        "wav" => Some("audio/wav"),
        "flac" => Some("audio/flac"),
        "ogg" => Some("audio/ogg"),
        "m4a" => Some("audio/mp4"),
        "aac" => Some("audio/aac"),
        
        "mp4" => Some("video/mp4"),
        "avi" => Some("video/x-msvideo"),
        "mkv" => Some("video/x-matroska"),
        "webm" => Some("video/webm"),
        "mov" => Some("video/quicktime"),
        "wmv" => Some("video/x-ms-wmv"),
        
        "pdf" => Some("application/pdf"),
        "docx" => Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        "doc" => Some("application/msword"),
        "pptx" => Some("application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        "xlsx" => Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        
        "txt" => Some("text/plain"),
        "md" => Some("text/markdown"),
        "html" | "htm" => Some("text/html"),
        "json" => Some("application/json"),
        "xml" => Some("application/xml"),
        "yaml" | "yml" => Some("application/x-yaml"),
        "csv" => Some("text/csv"),
        
        _ => None,
    }
}

/// Performance monitoring utilities.
pub mod performance {
    use std::time::{Duration, Instant};
    use std::collections::HashMap;

    /// Simple performance timer.
    #[derive(Debug)]
    pub struct Timer {
        start: Instant,
        name: String,
    }

    impl Timer {
        /// Start a new timer.
        pub fn start<S: Into<String>>(name: S) -> Self {
            Self {
                start: Instant::now(),
                name: name.into(),
            }
        }

        /// Stop the timer and return the elapsed duration.
        pub fn stop(self) -> Duration {
            let elapsed = self.start.elapsed();
            tracing::debug!("Timer '{}' elapsed: {:?}", self.name, elapsed);
            elapsed
        }

        /// Get elapsed time without stopping the timer.
        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }
    }

    /// Performance metrics collector.
    #[derive(Debug, Default)]
    pub struct MetricsCollector {
        timings: HashMap<String, Vec<Duration>>,
        counters: HashMap<String, u64>,
    }

    impl MetricsCollector {
        /// Create a new metrics collector.
        pub fn new() -> Self {
            Self::default()
        }

        /// Record a timing measurement.
        pub fn record_timing<S: Into<String>>(&mut self, name: S, duration: Duration) {
            self.timings.entry(name.into()).or_default().push(duration);
        }

        /// Increment a counter.
        pub fn increment_counter<S: Into<String>>(&mut self, name: S) {
            *self.counters.entry(name.into()).or_default() += 1;
        }

        /// Get average timing for a metric.
        pub fn average_timing(&self, name: &str) -> Option<Duration> {
            self.timings.get(name).and_then(|timings| {
                if timings.is_empty() {
                    None
                } else {
                    let total: Duration = timings.iter().sum();
                    Some(total / timings.len() as u32)
                }
            })
        }

        /// Get counter value.
        pub fn counter_value(&self, name: &str) -> u64 {
            self.counters.get(name).copied().unwrap_or(0)
        }

        /// Get all metrics as a summary.
        pub fn summary(&self) -> HashMap<String, serde_json::Value> {
            let mut summary = HashMap::new();

            // Add timing metrics
            for (name, timings) in &self.timings {
                if !timings.is_empty() {
                    let total: Duration = timings.iter().sum();
                    let avg = total / timings.len() as u32;
                    let min = timings.iter().min().copied().unwrap_or_default();
                    let max = timings.iter().max().copied().unwrap_or_default();

                    let timing_data = serde_json::json!({
                        "count": timings.len(),
                        "total_ms": total.as_millis(),
                        "average_ms": avg.as_millis(),
                        "min_ms": min.as_millis(),
                        "max_ms": max.as_millis(),
                    });

                    summary.insert(format!("timing_{}", name), timing_data);
                }
            }

            // Add counter metrics
            for (name, value) in &self.counters {
                summary.insert(format!("counter_{}", name), (*value).into());
            }

            summary
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MediaContent, MediaFormat};

    #[test]
    fn test_validate_media_content() {
        let valid_content = MediaContent::from_bytes(vec![1, 2, 3], MediaFormat::Jpeg);
        assert!(validate_media_content(&valid_content).is_ok());

        let empty_content = MediaContent::from_bytes(vec![], MediaFormat::Jpeg);
        assert!(validate_media_content(&empty_content).is_err());
    }

    #[test]
    fn test_estimate_processing_complexity() {
        let text_content = MediaContent::from_bytes(vec![1; 1000], MediaFormat::PlainText);
        let image_content = MediaContent::from_bytes(vec![1; 1000], MediaFormat::Jpeg);
        let video_content = MediaContent::from_bytes(vec![1; 1000], MediaFormat::Mp4);

        let text_complexity = estimate_processing_complexity(&text_content);
        let image_complexity = estimate_processing_complexity(&image_content);
        let video_complexity = estimate_processing_complexity(&video_content);

        assert!(text_complexity < image_complexity);
        assert!(image_complexity < video_complexity);
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("normal_file.txt"), "normal_file.txt");
        assert_eq!(sanitize_filename("file/with\\bad:chars"), "file_with_bad_chars");
        assert_eq!(sanitize_filename("file<>|*?.txt"), "file______.txt");
    }

    #[test]
    fn test_mime_type_from_extension() {
        assert_eq!(mime_type_from_extension("jpg"), Some("image/jpeg"));
        assert_eq!(mime_type_from_extension("PNG"), Some("image/png"));
        assert_eq!(mime_type_from_extension("mp3"), Some("audio/mpeg"));
        assert_eq!(mime_type_from_extension("unknown"), None);
    }

    #[test]
    fn test_performance_timer() {
        let timer = performance::Timer::start("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed >= std::time::Duration::from_millis(10));
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = performance::MetricsCollector::new();
        
        collector.record_timing("test_op", std::time::Duration::from_millis(100));
        collector.record_timing("test_op", std::time::Duration::from_millis(200));
        collector.increment_counter("test_counter");
        collector.increment_counter("test_counter");

        assert_eq!(collector.average_timing("test_op"), Some(std::time::Duration::from_millis(150)));
        assert_eq!(collector.counter_value("test_counter"), 2);

        let summary = collector.summary();
        assert!(summary.contains_key("timing_test_op"));
        assert!(summary.contains_key("counter_test_counter"));
    }
}
