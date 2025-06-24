//! Document and node transformation traits.
//!
//! This module defines traits for transforming documents into nodes and
//! for processing nodes to extract metadata, relationships, and other
//! enrichments.

use async_trait::async_trait;

use crate::{Document, Node, Result};

/// Transforms documents into nodes (chunking, preprocessing).
///
/// This trait handles the conversion of raw documents into searchable nodes.
/// Common implementations include text splitters, table extractors, and
/// other document processing components.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::Transformer;
/// use cheungfun_core::{Document, Node, ChunkInfo, Result};
/// use async_trait::async_trait;
///
/// struct TextSplitter {
///     chunk_size: usize,
///     overlap: usize,
/// }
///
/// #[async_trait]
/// impl Transformer for TextSplitter {
///     async fn transform(&self, document: Document) -> Result<Vec<Node>> {
///         // Implementation would split the document into chunks
///         let chunk_info = ChunkInfo::new(0, document.content.len(), 0);
///         Ok(vec![Node::new(document.content, document.id, chunk_info)])
///     }
/// }
/// ```
#[async_trait]
pub trait Transformer: Send + Sync + std::fmt::Debug {
    /// Transform a single document into multiple nodes.
    ///
    /// This method takes a document and converts it into one or more nodes
    /// that can be indexed and searched. The transformation might involve
    /// chunking, cleaning, or extracting specific content.
    ///
    /// # Arguments
    ///
    /// * `document` - The document to transform
    ///
    /// # Returns
    ///
    /// A vector of nodes created from the document. An empty vector
    /// indicates that the document should be skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if the transformation fails due to invalid
    /// content or processing errors.
    async fn transform(&self, document: Document) -> Result<Vec<Node>>;

    /// Transform multiple documents in batch for better performance.
    ///
    /// The default implementation processes documents one by one, but
    /// implementations can override this for batch optimization.
    ///
    /// # Arguments
    ///
    /// * `documents` - Vector of documents to transform
    ///
    /// # Returns
    ///
    /// A vector containing all nodes from all documents.
    async fn transform_batch(&self, documents: Vec<Document>) -> Result<Vec<Node>> {
        let mut all_nodes = Vec::new();
        for document in documents {
            let nodes = self.transform(document).await?;
            all_nodes.extend(nodes);
        }
        Ok(all_nodes)
    }

    /// Get a human-readable name for this transformer.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Validate that the transformer can process the given document.
    ///
    /// This method can be used to check if a document is compatible
    /// with this transformer before attempting transformation.
    async fn can_transform(&self, _document: &Document) -> bool {
        // Default implementation accepts all documents
        true
    }

    /// Get configuration information about this transformer.
    fn config(&self) -> std::collections::HashMap<String, serde_json::Value> {
        // Default implementation returns empty config
        std::collections::HashMap::new()
    }
}

/// Transforms nodes (metadata extraction, enrichment).
///
/// This trait is used for post-processing nodes to add metadata,
/// extract entities, create relationships, or perform other enrichments.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::NodeTransformer;
/// use cheungfun_core::{Node, Result};
/// use async_trait::async_trait;
///
/// struct MetadataExtractor;
///
/// #[async_trait]
/// impl NodeTransformer for MetadataExtractor {
///     async fn transform_node(&self, mut node: Node) -> Result<Node> {
///         // Extract metadata from node content
///         node.metadata.insert(
///             "word_count".to_string(),
///             serde_json::Value::Number(
///                 node.content.split_whitespace().count().into()
///             )
///         );
///         Ok(node)
///     }
/// }
/// ```
#[async_trait]
pub trait NodeTransformer: Send + Sync + std::fmt::Debug {
    /// Transform a single node.
    ///
    /// This method takes a node and enriches it with additional metadata,
    /// relationships, or other information. The node content may also be
    /// modified if needed.
    ///
    /// # Arguments
    ///
    /// * `node` - The node to transform
    ///
    /// # Returns
    ///
    /// The transformed node with additional metadata or modifications.
    ///
    /// # Errors
    ///
    /// Returns an error if the transformation fails.
    async fn transform_node(&self, node: Node) -> Result<Node>;

    /// Transform multiple nodes in batch for better performance.
    ///
    /// The default implementation processes nodes one by one, but
    /// implementations can override this for batch optimization.
    async fn transform_batch(&self, nodes: Vec<Node>) -> Result<Vec<Node>> {
        let mut results = Vec::new();
        for node in nodes {
            results.push(self.transform_node(node).await?);
        }
        Ok(results)
    }

    /// Get a human-readable name for this transformer.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Check if this transformer can process the given node.
    async fn can_transform(&self, _node: &Node) -> bool {
        // Default implementation accepts all nodes
        true
    }

    /// Get configuration information about this transformer.
    fn config(&self) -> std::collections::HashMap<String, serde_json::Value> {
        // Default implementation returns empty config
        std::collections::HashMap::new()
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
