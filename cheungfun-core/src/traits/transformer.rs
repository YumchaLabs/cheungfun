//! Unified transformation interface for document and node processing.
//!
//! This module defines a unified transformation interface following LlamaIndex's
//! TransformComponent design pattern. All processing components (document splitters,
//! metadata extractors, etc.) implement the same Transform trait for maximum
//! flexibility and composability.
//!
//! ## Type-Safe Pipeline System
//!
//! The new type-safe pipeline system provides compile-time guarantees for component
//! compatibility while maintaining full backward compatibility with the existing
//! Transform trait.

use async_trait::async_trait;
use std::marker::PhantomData;

use crate::{Document, Node, Result};

// ============================================================================
// Type-Safe Pipeline System
// ============================================================================

/// Marker trait for input types in the type-safe pipeline system.
pub trait InputType: Send + Sync + 'static {}

/// Marker trait for output types in the type-safe pipeline system.
pub trait OutputType: Send + Sync + 'static {}

/// Document state marker - indicates data contains documents.
#[derive(Debug, Clone, Copy)]
pub struct DocumentState;

/// Node state marker - indicates data contains nodes.
#[derive(Debug, Clone, Copy)]
pub struct NodeState;

impl InputType for DocumentState {}
impl InputType for NodeState {}
impl OutputType for DocumentState {}
impl OutputType for NodeState {}

/// Type-safe data container that carries compile-time type information.
///
/// This container ensures that data transformations are type-safe at compile time,
/// preventing invalid pipeline compositions while maintaining runtime efficiency.
#[derive(Debug, Clone)]
pub struct TypedData<T>
where
    T: InputType + OutputType,
{
    documents: Option<Vec<Document>>,
    nodes: Option<Vec<Node>>,
    _phantom: PhantomData<T>,
}

impl TypedData<DocumentState> {
    /// Create typed data from documents.
    pub fn from_documents(documents: Vec<Document>) -> Self {
        Self {
            documents: Some(documents),
            nodes: None,
            _phantom: PhantomData,
        }
    }

    /// Get reference to documents.
    pub fn documents(&self) -> &[Document] {
        self.documents
            .as_ref()
            .expect("DocumentState should contain documents")
    }

    /// Take ownership of documents.
    pub fn into_documents(self) -> Vec<Document> {
        self.documents
            .expect("DocumentState should contain documents")
    }
}

impl TypedData<NodeState> {
    /// Create typed data from nodes.
    pub fn from_nodes(nodes: Vec<Node>) -> Self {
        Self {
            documents: None,
            nodes: Some(nodes),
            _phantom: PhantomData,
        }
    }

    /// Get reference to nodes.
    pub fn nodes(&self) -> &[Node] {
        self.nodes.as_ref().expect("NodeState should contain nodes")
    }

    /// Take ownership of nodes.
    pub fn into_nodes(self) -> Vec<Node> {
        self.nodes.expect("NodeState should contain nodes")
    }
}

/// Type-safe transformation trait for compile-time pipeline validation.
///
/// This trait provides compile-time guarantees that components are compatible
/// with each other in a pipeline. The type parameters ensure that only valid
/// combinations can be constructed.
///
/// # Type Parameters
///
/// * `I` - Input type (DocumentState or NodeState)
/// * `O` - Output type (currently always NodeState)
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::{TypedTransform, TypedData, DocumentState, NodeState};
/// use cheungfun_core::{Document, Node, Result};
/// use async_trait::async_trait;
///
/// #[derive(Debug)]
/// struct SentenceSplitter;
///
/// #[async_trait]
/// impl TypedTransform<DocumentState, NodeState> for SentenceSplitter {
///     async fn transform(&self, input: TypedData<DocumentState>) -> Result<TypedData<NodeState>> {
///         let documents = input.documents();
///         // Split documents into nodes...
///         let nodes = vec![]; // Implementation here
///         Ok(TypedData::from_nodes(nodes))
///     }
///
///     fn name(&self) -> &'static str {
///         "SentenceSplitter"
///     }
/// }
/// ```
#[async_trait]
pub trait TypedTransform<I, O = NodeState>: Send + Sync + std::fmt::Debug
where
    I: InputType + OutputType,
    O: InputType + OutputType,
{
    /// Execute type-safe transformation.
    async fn transform(&self, input: TypedData<I>) -> Result<TypedData<O>>;

    /// Get component name.
    fn name(&self) -> &'static str;

    /// Get component description.
    fn description(&self) -> &'static str {
        "Transform component"
    }
}

// Note: Legacy Transform system has been removed in favor of the type-safe
// TypedTransform system. All components now use TypedTransform for compile-time
// type safety and better error prevention.

// Legacy Transform trait has been removed. All components now use the type-safe
// TypedTransform system for compile-time type safety and better error prevention.

// All legacy Transform system code has been removed.

/// Configuration for transformation operations.
#[derive(Debug, Clone)]
pub struct TransformConfig {
    /// Maximum number of nodes to process in a single batch.
    pub batch_size: Option<usize>,

    /// Maximum number of concurrent transformation operations.
    pub concurrency: Option<usize>,

    /// Whether to continue processing if some transformations fail.
    pub continue_on_error: bool,

    /// Timeout for transformation operations in seconds.
    pub timeout_seconds: Option<u64>,

    /// Additional transformer-specific configuration.
    pub additional_config: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            batch_size: Some(50),
            concurrency: Some(4),
            continue_on_error: true,
            timeout_seconds: Some(60),
            additional_config: std::collections::HashMap::new(),
        }
    }
}

/// Statistics about a transformation operation.
#[derive(Debug, Clone)]
pub struct TransformStats {
    /// Number of documents processed.
    pub documents_processed: usize,

    /// Number of nodes created.
    pub nodes_created: usize,

    /// Number of transformations that failed.
    pub transformations_failed: usize,

    /// Total time taken for transformation.
    pub duration: std::time::Duration,

    /// Average time per document.
    pub avg_time_per_document: std::time::Duration,

    /// List of errors encountered.
    pub errors: Vec<String>,

    /// Additional statistics.
    pub additional_stats: std::collections::HashMap<String, serde_json::Value>,
}

impl TransformStats {
    /// Create new transformation statistics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            documents_processed: 0,
            nodes_created: 0,
            transformations_failed: 0,
            duration: std::time::Duration::ZERO,
            avg_time_per_document: std::time::Duration::ZERO,
            errors: Vec::new(),
            additional_stats: std::collections::HashMap::new(),
        }
    }

    /// Calculate the success rate as a percentage.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.documents_processed == 0 {
            0.0
        } else {
            let successful = self.documents_processed - self.transformations_failed;
            (successful as f64 / self.documents_processed as f64) * 100.0
        }
    }

    /// Calculate the average number of nodes per document.
    #[must_use]
    pub fn avg_nodes_per_document(&self) -> f64 {
        if self.documents_processed == 0 {
            0.0
        } else {
            self.nodes_created as f64 / self.documents_processed as f64
        }
    }

    /// Update average time per document.
    pub fn update_avg_time(&mut self) {
        if self.documents_processed > 0 {
            self.avg_time_per_document = self.duration / self.documents_processed as u32;
        }
    }
}

impl Default for TransformStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_stats() {
        let mut stats = TransformStats::new();
        stats.documents_processed = 100;
        stats.nodes_created = 250;
        stats.transformations_failed = 5;
        stats.duration = std::time::Duration::from_secs(50);
        stats.update_avg_time();

        assert_eq!(stats.success_rate(), 95.0);
        assert_eq!(stats.avg_nodes_per_document(), 2.5);
        assert_eq!(
            stats.avg_time_per_document,
            std::time::Duration::from_millis(500)
        );
    }

    #[test]
    fn test_transform_config_default() {
        let config = TransformConfig::default();
        assert_eq!(config.batch_size, Some(50));
        assert_eq!(config.concurrency, Some(4));
        assert!(config.continue_on_error);
        assert_eq!(config.timeout_seconds, Some(60));
    }
}
