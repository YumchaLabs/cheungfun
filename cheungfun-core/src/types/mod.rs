//! Core data types for the Cheungfun framework.
//!
//! This module contains the fundamental data structures used throughout
//! the RAG pipeline, including documents, nodes, queries, and responses.

pub mod document;
pub mod graph;
pub mod labelled_property_graph;
pub mod node;
pub mod query;
pub mod response;

// Re-export all types for convenience
pub use document::*;
pub use graph::*;
pub use labelled_property_graph::*;
pub use node::*;
pub use query::*;
pub use response::*;
