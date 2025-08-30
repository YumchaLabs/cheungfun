//! Specialized retrievers for different retrieval strategies.
//!
//! This module provides various retriever implementations that extend
//! the basic retrieval capabilities with specialized strategies.

pub mod bm25;
pub mod contextual_compression;
pub mod graph_retriever;
pub mod hierarchical;
pub mod query_fusion;

pub use bm25::{BM25Config, BM25Params, BM25Retriever, BM25RetrieverBuilder};
pub use contextual_compression::{
    ContextualCompressionRetriever, ContextualCompressionRetrieverBuilder,
};
pub use graph_retriever::{GraphRetrievalConfig, GraphRetrievalStrategy, GraphRetriever};
pub use hierarchical::{HierarchicalRetriever, HierarchicalRetrieverBuilder, StorageContext};
pub use query_fusion::{
    DistanceMetric, FusionMode, QueryFusionConfig, QueryFusionRetriever, QueryFusionRetrieverBuilder,
};
