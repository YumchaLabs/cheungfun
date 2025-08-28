//! Graph store implementations for Cheungfun.
//!
//! This module provides various graph store implementations that implement
//! the PropertyGraphStore trait, enabling graph-based retrieval and storage.

pub mod simple_property_graph_store;

pub use simple_property_graph_store::SimplePropertyGraphStore;
