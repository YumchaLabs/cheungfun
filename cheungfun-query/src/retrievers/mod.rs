//! Specialized retrievers for different retrieval strategies.
//!
//! This module provides various retriever implementations that extend
//! the basic retrieval capabilities with specialized strategies.

pub mod graph_retriever;

pub use graph_retriever::{
    GraphRetriever, GraphRetrievalConfig, GraphRetrievalStrategy,
};
