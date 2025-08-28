//! Relational node parsers for hierarchical document processing.
//!
//! This module provides node parsers that create relationships between nodes,
//! particularly hierarchical structures that enable advanced retrieval patterns
//! like auto-merging and parent-child relationships.

pub mod hierarchical;

// Re-export commonly used types
pub use hierarchical::HierarchicalNodeParser;
