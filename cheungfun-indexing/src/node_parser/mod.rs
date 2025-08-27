//! Node parser module - Core interfaces for document parsing and text splitting.
//!
//! This module provides a comprehensive set of traits and implementations for parsing
//! documents into nodes, with support for various splitting strategies including
//! text-based, code-aware, and semantic splitting.
//!
//! # Architecture
//!
//! The module follows a hierarchical trait design inspired by LlamaIndex:
//!
//! ```
//! NodeParser (trait)
//! ├── TextSplitter (trait)
//! │   ├── MetadataAwareTextSplitter (trait)
//! │   │   ├── SentenceSplitter
//! │   │   ├── TokenTextSplitter
//! │   │   └── SemanticSplitter
//! │   └── CodeSplitter
//! ├── FileNodeParser (trait)
//! └── HierarchicalNodeParser
//! ```

pub mod callbacks;
pub mod config;
pub mod text;
pub mod utils;

use async_trait::async_trait;
use cheungfun_core::{Document, Node, Result as CoreResult};
use std::fmt::Debug;

/// Base trait for all node parsers.
///
/// This trait defines the core interface for parsing documents into nodes.
/// All node parsers must implement this trait to provide consistent behavior
/// across different parsing strategies.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::node_parser::{NodeParser, text::SentenceSplitter};
/// use cheungfun_core::Document;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let parser = SentenceSplitter::from_defaults(1000, 200)?;
///     let documents = vec![Document::new("Sample text content")];
///     
///     let nodes = parser.parse_nodes(&documents, false).await?;
///     println!("Created {} nodes", nodes.len());
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait NodeParser: Send + Sync + Debug {
    /// Parse documents into nodes.
    ///
    /// # Arguments
    ///
    /// * `documents` - The documents to parse
    /// * `show_progress` - Whether to show progress information
    ///
    /// # Returns
    ///
    /// A vector of nodes created from the input documents.
    async fn parse_nodes(
        &self,
        documents: &[Document],
        show_progress: bool,
    ) -> CoreResult<Vec<Node>>;

    /// Get a human-readable name for this node parser.
    ///
    /// This is used for logging, debugging, and identification purposes.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Asynchronously parse documents into nodes with custom options.
    ///
    /// This method provides additional flexibility for async processing
    /// with custom parameters.
    async fn aparse_nodes(
        &self,
        documents: &[Document],
        show_progress: bool,
        kwargs: std::collections::HashMap<String, serde_json::Value>,
    ) -> CoreResult<Vec<Node>> {
        // Default implementation delegates to parse_nodes
        // Custom implementations can use kwargs for additional configuration
        let _ = kwargs; // Suppress unused variable warning
        self.parse_nodes(documents, show_progress).await
    }

    /// Get nodes from documents (convenience method).
    ///
    /// This is a synchronous wrapper around `parse_nodes` for compatibility.
    fn get_nodes_from_documents(
        &self,
        documents: &[Document],
        show_progress: bool,
    ) -> CoreResult<Vec<Node>> {
        // Use tokio runtime to run async method
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(self.parse_nodes(documents, show_progress))
    }
}

/// Text splitter trait for splitting text into chunks.
///
/// This trait extends `NodeParser` with text-specific splitting capabilities.
/// It provides methods for splitting raw text into string chunks before
/// converting them into nodes.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::node_parser::{TextSplitter, text::SentenceSplitter};
/// use cheungfun_core::traits::{Transform, TransformInput};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let splitter = SentenceSplitter::from_defaults(1000, 200)?;
///
///     let text = "This is a sample text. It has multiple sentences.";
///     let chunks = splitter.split_text(text)?;
///
///     println!("Split into {} chunks", chunks.len());
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait TextSplitter: NodeParser {
    /// Split text into string chunks.
    ///
    /// This is the core method that defines how text should be split.
    /// Different implementations will use different strategies (sentences,
    /// tokens, semantic boundaries, etc.).
    ///
    /// # Arguments
    ///
    /// * `text` - The text to split
    ///
    /// # Returns
    ///
    /// A vector of text chunks.
    fn split_text(&self, text: &str) -> CoreResult<Vec<String>>;

    /// Split multiple texts into chunks.
    ///
    /// This method processes multiple texts and returns all chunks
    /// in a single flattened vector.
    ///
    /// # Arguments
    ///
    /// * `texts` - The texts to split
    ///
    /// # Returns
    ///
    /// A flattened vector of all text chunks.
    fn split_texts(&self, texts: &[String]) -> CoreResult<Vec<String>> {
        let mut all_chunks = Vec::new();
        for text in texts {
            let chunks = self.split_text(text)?;
            all_chunks.extend(chunks);
        }
        Ok(all_chunks)
    }

    /// Default implementation of parse_nodes for text splitters.
    ///
    /// This method implements the standard flow:
    /// 1. Extract text from documents
    /// 2. Split text into chunks
    /// 3. Build nodes from chunks
    async fn parse_nodes(
        &self,
        documents: &[Document],
        show_progress: bool,
    ) -> CoreResult<Vec<Node>> {
        let mut all_nodes = Vec::new();

        for document in documents {
            let text = &document.content;
            let chunks = self.split_text(text)?;

            let nodes = utils::build_nodes_from_splits(
                chunks, document, None, // Use default ID function
                true, // Include prev/next relationships
            )?;

            all_nodes.extend(nodes);
        }

        Ok(all_nodes)
    }
}

/// Metadata-aware text splitter trait.
///
/// This trait extends `TextSplitter` with the ability to consider metadata
/// when splitting text. This is useful for ensuring that the combined
/// text + metadata doesn't exceed token limits.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_indexing::node_parser::{MetadataAwareTextSplitter, text::SentenceSplitter};
/// use cheungfun_core::traits::{Transform, TransformInput};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let splitter = SentenceSplitter::from_defaults(1000, 200)?;
///
///     let text = "This is the main content.";
///     let metadata = "Title: Sample Document\nAuthor: John Doe";
///
///     let chunks = splitter.split_text_metadata_aware(text, metadata)?;
///     println!("Split into {} chunks considering metadata", chunks.len());
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait MetadataAwareTextSplitter: TextSplitter {
    /// Split text considering metadata constraints.
    ///
    /// This method splits text while ensuring that the combination of
    /// text chunk + metadata doesn't exceed the configured limits.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to split
    /// * `metadata_str` - The metadata string to consider
    ///
    /// # Returns
    ///
    /// A vector of text chunks that respect metadata constraints.
    fn split_text_metadata_aware(&self, text: &str, metadata_str: &str) -> CoreResult<Vec<String>>;

    /// Split multiple texts with their corresponding metadata.
    ///
    /// # Arguments
    ///
    /// * `texts` - The texts to split
    /// * `metadata_strs` - The corresponding metadata strings
    ///
    /// # Returns
    ///
    /// A flattened vector of all text chunks.
    fn split_texts_metadata_aware(
        &self,
        texts: &[String],
        metadata_strs: &[String],
    ) -> CoreResult<Vec<String>> {
        if texts.len() != metadata_strs.len() {
            return Err(cheungfun_core::error::CheungfunError::Pipeline {
                message: "Texts and metadata_strs must have the same length".to_string(),
            });
        }

        let mut all_chunks = Vec::new();
        for (text, metadata) in texts.iter().zip(metadata_strs.iter()) {
            let chunks = self.split_text_metadata_aware(text, metadata)?;
            all_chunks.extend(chunks);
        }
        Ok(all_chunks)
    }

    /// Enhanced parse_nodes implementation that considers metadata.
    async fn parse_nodes_metadata_aware(
        &self,
        documents: &[Document],
        show_progress: bool,
    ) -> CoreResult<Vec<Node>> {
        let mut all_nodes = Vec::new();

        for document in documents {
            let text = &document.content;
            let metadata_str = utils::get_metadata_str(document);

            let chunks = self.split_text_metadata_aware(text, &metadata_str)?;

            let nodes = utils::build_nodes_from_splits(
                chunks, document, None, // Use default ID function
                true, // Include prev/next relationships
            )?;

            all_nodes.extend(nodes);
        }

        Ok(all_nodes)
    }
}

/// File-specific node parser trait.
///
/// This trait is for parsers that handle specific file formats
/// (Markdown, HTML, JSON, etc.) and need format-specific processing.
#[async_trait]
pub trait FileNodeParser: NodeParser {
    /// Get the supported file extensions.
    fn supported_extensions(&self) -> Vec<&'static str>;

    /// Check if a file extension is supported.
    fn supports_extension(&self, extension: &str) -> bool {
        self.supported_extensions().contains(&extension)
    }
}

/// Hierarchical node parser trait.
///
/// This trait is for parsers that create hierarchical relationships
/// between nodes (parent-child relationships).
#[async_trait]
pub trait HierarchicalNodeParser: NodeParser {
    /// Parse documents into hierarchical nodes.
    async fn parse_hierarchical_nodes(
        &self,
        documents: &[Document],
        show_progress: bool,
    ) -> CoreResult<Vec<Node>>;
}

pub use callbacks::*;
/// Re-export commonly used types and traits.
pub use config::*;
pub use utils::*;
