//! Contextual compression retriever.
//!
//! This module implements a retriever that wraps a base retriever and applies
//! contextual compression to the retrieved results, similar to LangChain's
//! ContextualCompressionRetriever.

use crate::postprocessor::{DocumentCompressor, NodePostprocessor};
use async_trait::async_trait;
use cheungfun_core::{ChunkInfo, Query, Result, Retriever, ScoredNode};
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, info};
use uuid::Uuid;

/// Contextual compression retriever.
///
/// This retriever wraps a base retriever and applies compression/postprocessing
/// to the retrieved results. It's inspired by LangChain's ContextualCompressionRetriever
/// and supports multiple compression strategies.
#[derive(Debug)]
pub struct ContextualCompressionRetriever {
    /// Base retriever to get initial results.
    base_retriever: Arc<dyn Retriever>,

    /// Document compressor to apply compression.
    compressor: Option<Arc<dyn DocumentCompressor>>,

    /// Additional postprocessors to apply.
    postprocessors: Vec<Arc<dyn NodePostprocessor>>,

    /// Whether to apply compression before or after postprocessing.
    compress_first: bool,
}

impl ContextualCompressionRetriever {
    /// Create a new contextual compression retriever.
    pub fn new(base_retriever: Arc<dyn Retriever>) -> Self {
        Self {
            base_retriever,
            compressor: None,
            postprocessors: Vec::new(),
            compress_first: true,
        }
    }

    /// Add a document compressor.
    pub fn with_compressor(mut self, compressor: Arc<dyn DocumentCompressor>) -> Self {
        self.compressor = Some(compressor);
        self
    }

    /// Add a postprocessor.
    pub fn with_postprocessor(mut self, postprocessor: Arc<dyn NodePostprocessor>) -> Self {
        self.postprocessors.push(postprocessor);
        self
    }

    /// Add multiple postprocessors.
    pub fn with_postprocessors(mut self, postprocessors: Vec<Arc<dyn NodePostprocessor>>) -> Self {
        self.postprocessors.extend(postprocessors);
        self
    }

    /// Set whether to apply compression before postprocessing.
    ///
    /// If true (default), compression is applied first, then postprocessing.
    /// If false, postprocessing is applied first, then compression.
    pub fn with_compress_first(mut self, compress_first: bool) -> Self {
        self.compress_first = compress_first;
        self
    }

    /// Apply compression to nodes.
    async fn apply_compression(
        &self,
        nodes: Vec<ScoredNode>,
        query: &str,
    ) -> Result<Vec<ScoredNode>> {
        if let Some(compressor) = &self.compressor {
            debug!("Applying compression with: {}", compressor.name());
            compressor.compress(nodes, query).await
        } else {
            Ok(nodes)
        }
    }

    /// Apply postprocessors to nodes.
    async fn apply_postprocessors(
        &self,
        mut nodes: Vec<ScoredNode>,
        query: &str,
    ) -> Result<Vec<ScoredNode>> {
        for postprocessor in &self.postprocessors {
            debug!("Applying postprocessor: {}", postprocessor.name());
            nodes = postprocessor.postprocess(nodes, query).await?;
        }
        Ok(nodes)
    }
}

#[async_trait]
impl Retriever for ContextualCompressionRetriever {
    async fn retrieve(&self, query: &Query) -> Result<Vec<ScoredNode>> {
        info!(
            "Starting contextual compression retrieval for query: {}",
            query.text
        );

        // Step 1: Retrieve from base retriever
        debug!("Retrieving from base retriever: {:?}", self.base_retriever);
        let mut nodes = self.base_retriever.retrieve(query).await?;

        if nodes.is_empty() {
            debug!("No nodes retrieved from base retriever");
            return Ok(nodes);
        }

        debug!("Retrieved {} nodes from base retriever", nodes.len());

        // Step 2: Apply compression and postprocessing based on configuration
        if self.compress_first {
            // Compression first, then postprocessing
            nodes = self.apply_compression(nodes, &query.text).await?;
            nodes = self.apply_postprocessors(nodes, &query.text).await?;
        } else {
            // Postprocessing first, then compression
            nodes = self.apply_postprocessors(nodes, &query.text).await?;
            nodes = self.apply_compression(nodes, &query.text).await?;
        }

        info!(
            "Contextual compression completed. Final result: {} nodes",
            nodes.len()
        );

        Ok(nodes)
    }

    async fn retrieve_with_context(
        &self,
        query: &Query,
        context: &cheungfun_core::RetrievalContext,
    ) -> Result<Vec<ScoredNode>> {
        // Use base retriever's context-aware retrieval if available
        let mut nodes = self
            .base_retriever
            .retrieve_with_context(query, context)
            .await?;

        if nodes.is_empty() {
            return Ok(nodes);
        }

        // Apply the same compression and postprocessing pipeline
        if self.compress_first {
            nodes = self.apply_compression(nodes, &query.text).await?;
            nodes = self.apply_postprocessors(nodes, &query.text).await?;
        } else {
            nodes = self.apply_postprocessors(nodes, &query.text).await?;
            nodes = self.apply_compression(nodes, &query.text).await?;
        }

        Ok(nodes)
    }

    fn name(&self) -> &'static str {
        "ContextualCompressionRetriever"
    }
}

/// Builder for creating contextual compression retrievers.
pub struct ContextualCompressionRetrieverBuilder {
    base_retriever: Option<Arc<dyn Retriever>>,
    compressor: Option<Arc<dyn DocumentCompressor>>,
    postprocessors: Vec<Arc<dyn NodePostprocessor>>,
    compress_first: bool,
}

impl ContextualCompressionRetrieverBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            base_retriever: None,
            compressor: None,
            postprocessors: Vec::new(),
            compress_first: true,
        }
    }

    /// Set the base retriever.
    pub fn base_retriever(mut self, retriever: Arc<dyn Retriever>) -> Self {
        self.base_retriever = Some(retriever);
        self
    }

    /// Set the document compressor.
    pub fn compressor(mut self, compressor: Arc<dyn DocumentCompressor>) -> Self {
        self.compressor = Some(compressor);
        self
    }

    /// Add a postprocessor.
    pub fn postprocessor(mut self, postprocessor: Arc<dyn NodePostprocessor>) -> Self {
        self.postprocessors.push(postprocessor);
        self
    }

    /// Add multiple postprocessors.
    pub fn postprocessors(mut self, postprocessors: Vec<Arc<dyn NodePostprocessor>>) -> Self {
        self.postprocessors.extend(postprocessors);
        self
    }

    /// Set compression order.
    pub fn compress_first(mut self, compress_first: bool) -> Self {
        self.compress_first = compress_first;
        self
    }

    /// Build the contextual compression retriever.
    pub fn build(self) -> Result<ContextualCompressionRetriever> {
        let base_retriever = self.base_retriever.ok_or_else(|| {
            cheungfun_core::CheungfunError::configuration("Base retriever is required")
        })?;

        let mut retriever = ContextualCompressionRetriever::new(base_retriever);

        if let Some(compressor) = self.compressor {
            retriever = retriever.with_compressor(compressor);
        }

        if !self.postprocessors.is_empty() {
            retriever = retriever.with_postprocessors(self.postprocessors);
        }

        retriever = retriever.with_compress_first(self.compress_first);

        Ok(retriever)
    }
}

impl Default for ContextualCompressionRetrieverBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postprocessor::SimilarityFilter;
    use cheungfun_core::{Node, ScoredNode};
    use std::collections::HashMap;

    // Mock retriever for testing
    #[derive(Debug)]
    struct MockRetriever {
        nodes: Vec<ScoredNode>,
    }

    #[async_trait]
    impl Retriever for MockRetriever {
        async fn retrieve(&self, _query: &Query) -> Result<Vec<ScoredNode>> {
            Ok(self.nodes.clone())
        }
    }

    fn create_test_nodes() -> Vec<ScoredNode> {
        vec![
            ScoredNode {
                node: Node::new(
                    "High relevance content with many details".to_string(),
                    Uuid::new_v4(),
                    ChunkInfo::new(Some(0), Some(42), 0),
                ),
                score: 0.9,
            },
            ScoredNode {
                node: Node::new(
                    "Medium relevance content".to_string(),
                    Uuid::new_v4(),
                    ChunkInfo::new(Some(0), Some(24), 1),
                ),
                score: 0.7,
            },
            ScoredNode {
                node: Node::new(
                    "Low relevance content".to_string(),
                    Uuid::new_v4(),
                    ChunkInfo::new(Some(0), Some(21), 2),
                ),
                score: 0.3,
            },
        ]
    }

    #[tokio::test]
    async fn test_contextual_compression_retriever_with_filter() {
        let mock_retriever = Arc::new(MockRetriever {
            nodes: create_test_nodes(),
        });

        let similarity_filter = Arc::new(SimilarityFilter::with_cutoff(0.5));

        let compression_retriever = ContextualCompressionRetriever::new(mock_retriever)
            .with_postprocessor(similarity_filter);

        let query = Query::new("test query");
        let results = compression_retriever.retrieve(&query).await.unwrap();

        // Should filter out nodes with score < 0.5
        assert_eq!(results.len(), 2);
        // Note: We can't directly compare UUIDs with strings, so we check the content instead
        assert_eq!(
            results[0].node.content,
            "High relevance content with many details"
        );
        assert_eq!(results[1].node.content, "Medium relevance content");
    }

    #[tokio::test]
    async fn test_builder_pattern() {
        let mock_retriever = Arc::new(MockRetriever {
            nodes: create_test_nodes(),
        });

        let similarity_filter = Arc::new(SimilarityFilter::with_cutoff(0.6));

        let compression_retriever = ContextualCompressionRetrieverBuilder::new()
            .base_retriever(mock_retriever)
            .postprocessor(similarity_filter)
            .compress_first(false)
            .build()
            .unwrap();

        let query = Query::new("test query");
        let results = compression_retriever.retrieve(&query).await.unwrap();

        // Should filter out nodes with score < 0.6
        assert_eq!(results.len(), 2);
    }
}
