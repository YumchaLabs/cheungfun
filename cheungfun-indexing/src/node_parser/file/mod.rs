//! File format-specific node parsers.
//!
//! This module provides specialized parsers for different file formats,
//! including HTML, JSON, and other structured document types.

pub mod html;
pub mod json;

// Re-export commonly used types
pub use html::HTMLNodeParser;
pub use json::JSONNodeParser;
