//! Core data types for the Cheungfun framework.
//!
//! This module contains the fundamental data structures used throughout
//! the RAG pipeline, including documents, nodes, queries, and responses.

pub mod document;
pub mod node;
pub mod query;
pub mod response;

// Re-export all types for convenience
pub use document::*;
pub use node::*;
pub use query::*;
pub use response::*;
