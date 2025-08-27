//! Unified transformation interface for document and node processing.
//!
//! This module defines a unified transformation interface following LlamaIndex's
//! TransformComponent design pattern. All processing components (document splitters,
//! metadata extractors, etc.) implement the same Transform trait for maximum
//! flexibility and composability.

use async_trait::async_trait;

use crate::{Document, Node, Result};

/// Unified input type for transformations.
///
/// This enum allows transformations to accept either documents or nodes,
/// providing flexibility for different types of processing components.
#[derive(Debug, Clone)]
pub enum TransformInput {
    /// Document input for document-to-nodes transformations (e.g., text splitters).
    Document(Document),
    /// Node input for node-to-node transformations (e.g., metadata extractors).
    Node(Node),
    /// Batch of documents for efficient batch processing.
    Documents(Vec<Document>),
    /// Batch of nodes for efficient batch processing.
    Nodes(Vec<Node>),
}

impl TransformInput {
    /// Create a document input.
    pub fn document(document: Document) -> Self {
        Self::Document(document)
    }

    /// Create a node input.
    pub fn node(node: Node) -> Self {
        Self::Node(node)
    }

    /// Create a batch of documents input.
    pub fn documents(documents: Vec<Document>) -> Self {
        Self::Documents(documents)
    }

    /// Create a batch of nodes input.
    pub fn nodes(nodes: Vec<Node>) -> Self {
        Self::Nodes(nodes)
    }
}

/// Unified transformation trait for all processing components.
///
/// This trait provides a single interface for all types of transformations,
/// whether they process documents into nodes or nodes into nodes. This design
/// follows LlamaIndex's TransformComponent pattern for maximum flexibility.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::{Transform, TransformInput};
/// use cheungfun_core::{Document, Node, Result, CheungfunError};
/// use async_trait::async_trait;
///
/// struct SentenceSplitter {
///     chunk_size: usize,
///     overlap: usize,
/// }
///
/// #[async_trait]
/// impl Transform for SentenceSplitter {
///     async fn transform(&self, input: TransformInput) -> Result<Vec<Node>> {
///         match input {
///             TransformInput::Document(doc) => {
///                 // Split document into sentence-based chunks
///                 // Implementation here...
///                 Ok(vec![])
///             }
///             _ => Err(CheungfunError::InvalidInput(
///                 "SentenceSplitter only accepts documents".into()
///             ))
///         }
///     }
/// }
/// ```
#[async_trait]
pub trait Transform: Send + Sync + std::fmt::Debug {
    /// Transform input into nodes.
    ///
    /// This is the core transformation method that all components must implement.
    /// The input can be documents, nodes, or batches thereof, allowing maximum
    /// flexibility for different types of processing components.
    ///
    /// # Arguments
    ///
    /// * `input` - The input to transform (documents or nodes)
    ///
    /// # Returns
    ///
    /// A vector of nodes created from the input. An empty vector
    /// indicates that the input should be skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if the transformation fails due to invalid
    /// input, processing errors, or unsupported input types.
    async fn transform(&self, input: TransformInput) -> Result<Vec<Node>>;

    /// Transform multiple inputs in batch for better performance.
    ///
    /// The default implementation processes inputs one by one, but
    /// implementations can override this for batch optimization.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Vector of inputs to transform
    ///
    /// # Returns
    ///
    /// A vector containing all nodes from all inputs.
    async fn transform_batch(&self, inputs: Vec<TransformInput>) -> Result<Vec<Node>> {
        let mut all_nodes = Vec::new();
        for input in inputs {
            let nodes = self.transform(input).await?;
            all_nodes.extend(nodes);
        }
        Ok(all_nodes)
    }

    /// Get a human-readable name for this transformer.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Validate that the transformer can process the given input.
    ///
    /// This method can be used to check if an input is compatible
    /// with this transformer before attempting transformation.
    async fn can_transform(&self, _input: &TransformInput) -> bool {
        // Default implementation accepts all inputs
        true
    }

    /// Get configuration information about this transformer.
    fn config(&self) -> std::collections::HashMap<String, serde_json::Value> {
        // Default implementation returns empty config
        std::collections::HashMap::new()
    }
}

// Convenience functions for creating TransformInput

impl From<Document> for TransformInput {
    fn from(document: Document) -> Self {
        Self::Document(document)
    }
}

impl From<Node> for TransformInput {
    fn from(node: Node) -> Self {
        Self::Node(node)
    }
}

impl From<Vec<Document>> for TransformInput {
    fn from(documents: Vec<Document>) -> Self {
        Self::Documents(documents)
    }
}

impl From<Vec<Node>> for TransformInput {
    fn from(nodes: Vec<Node>) -> Self {
        Self::Nodes(nodes)
    }
}

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
