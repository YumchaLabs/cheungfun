//! # Cheungfun - Rust RAG Framework
//!
//! Cheungfun is a high-performance RAG (Retrieval-Augmented Generation) framework
//! built in Rust, inspired by LlamaIndex and Swiftide. It provides type-safe,
//! async-first components for building AI applications.
//!
//! ## Quick Start
//!
//! ```rust
//! use cheungfun::prelude::*;
//!
//! // Create a simple document
//! let doc = Document::new("This is a sample document for indexing.");
//!
//! // Create a query
//! let query = Query::new("What is this document about?");
//!
//! println!("Document: {}", doc.content);
//! println!("Query: {}", query.text);
//! ```
//!
//! ## Architecture
//!
//! The framework is organized into several modules:
//!
//! - **cheungfun-core**: Core traits, types, and interfaces
//! - **cheungfun-indexing**: Document loading and processing
//! - **cheungfun-query**: Query processing and retrieval
//! - **cheungfun-agents**: Agent framework and MCP integration
//! - **cheungfun-integrations**: External service integrations

#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

// Re-export all public APIs from sub-crates
pub use cheungfun_agents as agents;
pub use cheungfun_core as core;
pub use cheungfun_indexing as indexing;
pub use cheungfun_integrations as integrations;
pub use cheungfun_query as query;

/// Prelude module for convenient imports.
///
/// This module re-exports the most commonly used types and traits
/// from all Cheungfun modules.
pub mod prelude {
    // Re-export core prelude
    pub use cheungfun_core::prelude::*;

    // Re-export other commonly used items
    // (will be added as we implement other modules)
}

/// Version information for the Cheungfun framework.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
