//! Conversion utilities for multimodal content.
//!
//! This module provides functions for converting between different data formats,
//! encodings, and representations used in multimodal processing.

use crate::error::Result;
use std::path::{Path, PathBuf};

/// Convert binary data to base64 string.
pub fn convert_to_base64(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// Convert base64 string to binary data.
pub fn convert_from_base64(base64_str: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(base64_str)
        .map_err(crate::error::MultimodalError::Base64Decode)
}

/// Normalize file path for cross-platform compatibility.
pub fn normalize_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let path = path.as_ref();

    // Convert to absolute path if possible
    if let Ok(absolute) = path.canonicalize() {
        absolute
    } else {
        // Fallback to the original path
        path.to_path_buf()
    }
}

/// Convert file size to human-readable format.
pub fn format_file_size(size: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    const THRESHOLD: u64 = 1024;

    if size == 0 {
        return "0 B".to_string();
    }

    let mut size = size as f64;
    let mut unit_index = 0;

    while size >= THRESHOLD as f64 && unit_index < UNITS.len() - 1 {
        size /= THRESHOLD as f64;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", size as u64, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Convert duration to human-readable format.
pub fn format_duration(duration: std::time::Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    let milliseconds = duration.subsec_millis();

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else if seconds > 0 {
        format!("{}.{:03}s", seconds, milliseconds)
    } else {
        format!("{}ms", milliseconds)
    }
}

/// Convert MIME type to media format.
pub fn mime_type_to_format(mime_type: &str) -> Option<crate::types::MediaFormat> {
    match mime_type {
        "image/jpeg" => Some(crate::types::MediaFormat::Jpeg),
        "image/png" => Some(crate::types::MediaFormat::Png),
        "image/gif" => Some(crate::types::MediaFormat::Gif),
        "image/webp" => Some(crate::types::MediaFormat::WebP),
        "image/svg+xml" => Some(crate::types::MediaFormat::Svg),
        "image/bmp" => Some(crate::types::MediaFormat::Bmp),
        "image/tiff" => Some(crate::types::MediaFormat::Tiff),
        "image/x-icon" => Some(crate::types::MediaFormat::Ico),

        "audio/mpeg" => Some(crate::types::MediaFormat::Mp3),
        "audio/wav" => Some(crate::types::MediaFormat::Wav),
        "audio/flac" => Some(crate::types::MediaFormat::Flac),
        "audio/ogg" => Some(crate::types::MediaFormat::Ogg),
        "audio/mp4" => Some(crate::types::MediaFormat::M4a),
        "audio/aac" => Some(crate::types::MediaFormat::Aac),
        "audio/x-ms-wma" => Some(crate::types::MediaFormat::Wma),

        "video/mp4" => Some(crate::types::MediaFormat::Mp4),
        "video/x-msvideo" => Some(crate::types::MediaFormat::Avi),
        "video/x-matroska" => Some(crate::types::MediaFormat::Mkv),
        "video/webm" => Some(crate::types::MediaFormat::WebM),
        "video/quicktime" => Some(crate::types::MediaFormat::Mov),
        "video/x-ms-wmv" => Some(crate::types::MediaFormat::Wmv),
        "video/x-flv" => Some(crate::types::MediaFormat::Flv),

        "application/pdf" => Some(crate::types::MediaFormat::Pdf),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
            Some(crate::types::MediaFormat::Docx)
        }
        "application/msword" => Some(crate::types::MediaFormat::Doc),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation" => {
            Some(crate::types::MediaFormat::Pptx)
        }
        "application/vnd.ms-powerpoint" => Some(crate::types::MediaFormat::Ppt),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => {
            Some(crate::types::MediaFormat::Xlsx)
        }
        "application/vnd.ms-excel" => Some(crate::types::MediaFormat::Xls),
        "application/vnd.oasis.opendocument.text" => Some(crate::types::MediaFormat::Odt),
        "application/rtf" => Some(crate::types::MediaFormat::Rtf),

        "text/plain" => Some(crate::types::MediaFormat::PlainText),
        "text/markdown" => Some(crate::types::MediaFormat::Markdown),
        "text/html" => Some(crate::types::MediaFormat::Html),
        "application/xml" | "text/xml" => Some(crate::types::MediaFormat::Xml),
        "application/json" => Some(crate::types::MediaFormat::Json),
        "application/x-yaml" | "text/yaml" => Some(crate::types::MediaFormat::Yaml),
        "text/csv" => Some(crate::types::MediaFormat::Csv),

        _ => None,
    }
}

/// Convert coordinates between different coordinate systems.
#[derive(Debug, Clone, Copy)]
pub enum CoordinateSystem {
    /// Normalized coordinates (0.0 to 1.0)
    Normalized,
    /// Pixel coordinates
    Pixel { width: u32, height: u32 },
    /// Percentage coordinates (0.0 to 100.0)
    Percentage,
}

/// Convert bounding box coordinates.
pub fn convert_bounding_box(
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    from: CoordinateSystem,
    to: CoordinateSystem,
) -> (f32, f32, f32, f32) {
    // First convert to normalized coordinates
    let (norm_x, norm_y, norm_w, norm_h) = match from {
        CoordinateSystem::Normalized => (x, y, width, height),
        CoordinateSystem::Pixel {
            width: img_w,
            height: img_h,
        } => (
            x / img_w as f32,
            y / img_h as f32,
            width / img_w as f32,
            height / img_h as f32,
        ),
        CoordinateSystem::Percentage => (x / 100.0, y / 100.0, width / 100.0, height / 100.0),
    };

    // Then convert from normalized to target
    match to {
        CoordinateSystem::Normalized => (norm_x, norm_y, norm_w, norm_h),
        CoordinateSystem::Pixel {
            width: img_w,
            height: img_h,
        } => (
            norm_x * img_w as f32,
            norm_y * img_h as f32,
            norm_w * img_w as f32,
            norm_h * img_h as f32,
        ),
        CoordinateSystem::Percentage => (
            norm_x * 100.0,
            norm_y * 100.0,
            norm_w * 100.0,
            norm_h * 100.0,
        ),
    }
}

/// Color space conversion utilities.
pub mod color {
    /// Convert RGB to HSV color space.
    pub fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        let h = if delta == 0.0 {
            0.0
        } else if max == r {
            60.0 * (((g - b) / delta) % 6.0)
        } else if max == g {
            60.0 * ((b - r) / delta + 2.0)
        } else {
            60.0 * ((r - g) / delta + 4.0)
        };

        let s = if max == 0.0 { 0.0 } else { delta / max };
        let v = max;

        (h, s, v)
    }

    /// Convert HSV to RGB color space.
    pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;

        let (r_prime, g_prime, b_prime) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        (r_prime + m, g_prime + m, b_prime + m)
    }

    /// Convert RGB to grayscale using luminance formula.
    pub fn rgb_to_grayscale(r: f32, g: f32, b: f32) -> f32 {
        0.299 * r + 0.587 * g + 0.114 * b
    }
}

/// Audio conversion utilities.
pub mod audio {
    /// Convert sample rate.
    pub fn convert_sample_rate(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        if from_rate == to_rate {
            return samples.to_vec();
        }

        let ratio = to_rate as f64 / from_rate as f64;
        let output_len = (samples.len() as f64 * ratio) as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_index = i as f64 / ratio;
            let src_index_floor = src_index.floor() as usize;
            let src_index_ceil = (src_index_floor + 1).min(samples.len() - 1);
            let fraction = src_index - src_index_floor as f64;

            if src_index_floor < samples.len() {
                let sample = if fraction == 0.0 {
                    samples[src_index_floor]
                } else {
                    // Linear interpolation
                    let sample1 = samples[src_index_floor];
                    let sample2 = samples[src_index_ceil];
                    sample1 + (sample2 - sample1) * fraction as f32
                };
                output.push(sample);
            }
        }

        output
    }

    /// Convert stereo to mono by averaging channels.
    pub fn stereo_to_mono(stereo_samples: &[f32]) -> Vec<f32> {
        stereo_samples
            .chunks_exact(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect()
    }

    /// Convert mono to stereo by duplicating the channel.
    pub fn mono_to_stereo(mono_samples: &[f32]) -> Vec<f32> {
        mono_samples
            .iter()
            .flat_map(|&sample| [sample, sample])
            .collect()
    }

    /// Normalize audio samples to [-1.0, 1.0] range.
    pub fn normalize_samples(samples: &mut [f32]) {
        let max_abs = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        if max_abs > 0.0 {
            for sample in samples {
                *sample /= max_abs;
            }
        }
    }
}

/// Text encoding conversion utilities.
pub mod text {
    use crate::error::Result;

    /// Detect text encoding (simplified implementation).
    pub fn detect_encoding(data: &[u8]) -> &'static str {
        // Check for BOM
        if data.starts_with(&[0xEF, 0xBB, 0xBF]) {
            return "utf-8";
        }
        if data.starts_with(&[0xFF, 0xFE]) {
            return "utf-16le";
        }
        if data.starts_with(&[0xFE, 0xFF]) {
            return "utf-16be";
        }

        // Try UTF-8
        if std::str::from_utf8(data).is_ok() {
            return "utf-8";
        }

        // Default to latin-1 for binary compatibility
        "latin-1"
    }

    /// Convert text to UTF-8.
    pub fn to_utf8(data: &[u8], encoding: &str) -> Result<String> {
        match encoding.to_lowercase().as_str() {
            "utf-8" => String::from_utf8(data.to_vec())
                .map_err(|_| crate::error::MultimodalError::validation("Invalid UTF-8 data")),
            "latin-1" | "iso-8859-1" => {
                // Convert Latin-1 to UTF-8
                Ok(data.iter().map(|&b| b as char).collect())
            }
            _ => Err(crate::error::MultimodalError::unsupported_format(
                encoding,
                "text encoding",
            )),
        }
    }

    /// Clean text by removing control characters and normalizing whitespace.
    pub fn clean_text(text: &str) -> String {
        text.chars()
            .filter(|c| !c.is_control() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Truncate text to a maximum length, preserving word boundaries.
    pub fn truncate_text(text: &str, max_length: usize) -> String {
        if text.len() <= max_length {
            return text.to_string();
        }

        let mut truncated = String::new();
        let mut current_length = 0;

        for word in text.split_whitespace() {
            let word_length = word.len();
            if current_length + word_length + 1 > max_length {
                break;
            }

            if !truncated.is_empty() {
                truncated.push(' ');
                current_length += 1;
            }

            truncated.push_str(word);
            current_length += word_length;
        }

        if truncated.len() < text.len() {
            truncated.push_str("...");
        }

        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_conversion() {
        let data = b"Hello, world!";
        let base64_str = convert_to_base64(data);
        let decoded = convert_from_base64(&base64_str).unwrap();
        assert_eq!(data, decoded.as_slice());
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(0), "0 B");
        assert_eq!(format_file_size(512), "512 B");
        assert_eq!(format_file_size(1024), "1.0 KB");
        assert_eq!(format_file_size(1536), "1.5 KB");
        assert_eq!(format_file_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_file_size(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(
            format_duration(std::time::Duration::from_millis(500)),
            "500ms"
        );
        assert_eq!(format_duration(std::time::Duration::from_secs(5)), "5.000s");
        assert_eq!(format_duration(std::time::Duration::from_secs(65)), "1m 5s");
        assert_eq!(
            format_duration(std::time::Duration::from_secs(3665)),
            "1h 1m 5s"
        );
    }

    #[test]
    fn test_coordinate_conversion() {
        let (x, y, w, h) = convert_bounding_box(
            0.5,
            0.5,
            0.2,
            0.3,
            CoordinateSystem::Normalized,
            CoordinateSystem::Pixel {
                width: 100,
                height: 200,
            },
        );
        // Use approximate equality for floating point comparison
        assert!((x - 50.0).abs() < 0.001);
        assert!((y - 100.0).abs() < 0.001);
        assert!((w - 20.0).abs() < 0.001);
        assert!((h - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_rgb_to_hsv() {
        let (h, s, v) = color::rgb_to_hsv(1.0, 0.0, 0.0); // Red
        assert!((h - 0.0).abs() < 0.001);
        assert!((s - 1.0).abs() < 0.001);
        assert!((v - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_audio_sample_rate_conversion() {
        let samples = vec![1.0, 0.5, -0.5, -1.0];
        let converted = audio::convert_sample_rate(&samples, 44100, 22050);
        assert_eq!(converted.len(), 2);
    }

    #[test]
    fn test_stereo_to_mono() {
        let stereo = vec![1.0, -1.0, 0.5, -0.5];
        let mono = audio::stereo_to_mono(&stereo);
        assert_eq!(mono, vec![0.0, 0.0]);
    }

    #[test]
    fn test_text_cleaning() {
        let dirty_text = "  Hello,\t\nworld!  \r\n  ";
        let clean = text::clean_text(dirty_text);
        assert_eq!(clean, "Hello, world!");
    }

    #[test]
    fn test_text_truncation() {
        let long_text = "This is a very long text that should be truncated";
        let truncated = text::truncate_text(long_text, 20);
        assert!(truncated.len() <= 23); // 20 + "..."
        assert!(truncated.ends_with("..."));
    }
}
