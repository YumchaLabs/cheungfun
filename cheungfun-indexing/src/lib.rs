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
pub mod node_parser;
pub mod parsers;
pub mod pipeline;
pub mod transformers;
pub mod utils;

// Re-export new node parser architecture for easy access
pub use crate::node_parser::{
    text::{CodeSplitter, SentenceSplitter, TokenTextSplitter},
    MetadataAwareTextSplitter, NodeParser, TextSplitter,
};

// Re-export useful transformers
pub use crate::transformers::MetadataExtractor;

/// Re-export commonly used types and traits.
pub mod prelude {
    // Re-export our own error types
    pub use crate::error::{IndexingError, Result as IndexingResult};

    // Re-export specific loaders to avoid conflicts
    pub use crate::loaders::{
        CodeLoader, CodeLoaderConfig, DirectoryLoader, FileFilter, FileLoader, Filter,
        FilterConfig, LoaderConfig, ProgrammingLanguage, WebLoader,
    };

    // Re-export new node parser architecture
    pub use crate::node_parser::{
        config::{
            CodeSplitterConfig, NodeParserConfig, SemanticSplitterConfig, SentenceSplitterConfig,
            TextSplitterConfig, TokenTextSplitterConfig,
        },
        text::{CodeSplitter, SentenceSplitter, TokenTextSplitter},
        MetadataAwareTextSplitter, NodeParser, TextSplitter,
    };

    // Re-export parsers
    pub use crate::parsers::{AstAnalysis, AstParser, AstParserConfig};

    // Re-export pipeline
    pub use crate::pipeline::{DefaultIndexingPipeline, PipelineConfig};

    // Re-export transformers
    pub use crate::transformers::{MetadataConfig, MetadataExtractor};

    // Re-export unified utilities
    pub use crate::utils::{file, metadata, text};

    // Re-export core types (avoid conflicts)
    pub use cheungfun_core::{
        traits::{Embedder, IndexingPipeline, Retriever, VectorStore},
        CheungfunError, ChunkInfo, Document, Node, Query, QueryResponse, Result as CoreResult,
        ScoredNode,
    };
}
