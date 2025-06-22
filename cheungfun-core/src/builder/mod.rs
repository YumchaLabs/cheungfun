//! Builder patterns for constructing pipelines.
//!
//! This module provides fluent APIs for building indexing and query pipelines
//! with proper validation and error handling. Builders support both direct
//! component instances and Arc-wrapped shared components.

pub mod indexing;
pub mod query;

// Re-export all builder types for convenience
pub use indexing::*;
pub use query::*;
