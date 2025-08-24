//! Document loading and indexing for the Cheungfun RAG framework.
//!
//! This crate provides components for loading documents from various sources
//! and transforming them into searchable nodes. It includes:
//!
//! - **Loaders**: File system, web, and database document loaders
//! - **Transformers**: Text splitters, metadata extractors, and content processors
//! - **Pipelines**: Complete indexing workflows that combine loaders and transformers
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use cheungfun_indexing::prelude::*;
//! use cheungfun_core::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Load documents from a directory
//!     let loader = FileLoader::new("./docs")?;
//!     let documents = loader.load().await?;
//!
//!     // Split documents into chunks
//!     let splitter = TextSplitter::new(1000, 200);
//!     let mut nodes = Vec::new();
//!     for doc in documents {
//!         let doc_nodes = splitter.transform(doc).await?;
//!         nodes.extend(doc_nodes);
//!     }
//!
//!     println!("Created {} nodes from documents", nodes.len());
//!     Ok(())
//! }
//! ```

#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod loaders;
pub mod parsers;
pub mod pipeline;
pub mod transformers;

/// Re-export commonly used types and traits.
pub mod prelude {
    pub use crate::error::*;
    pub use crate::loaders::*;
    pub use crate::parsers::*;
    pub use crate::pipeline::*;
    pub use crate::transformers::*;

    // Re-export core types
    pub use cheungfun_core::prelude::*;
}
