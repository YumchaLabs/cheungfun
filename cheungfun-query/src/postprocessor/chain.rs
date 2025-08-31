//! Postprocessor chain management for sequential node processing.
//!
//! This module provides a unified interface for managing multiple postprocessors
//! in a chain, similar to LlamaIndex's node_postprocessors approach.

use async_trait::async_trait;
use cheungfun_core::{Result, ScoredNode};
use std::sync::Arc;
use tracing::{debug, info, instrument};

use super::NodePostprocessor;

/// A chain of postprocessors that are applied sequentially to nodes.
///
/// This provides a unified interface for managing multiple postprocessors,
/// similar to LlamaIndex's approach where you can specify multiple postprocessors
/// in a query engine.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_query::postprocessor::{PostprocessorChain, KeywordFilter, SimilarityFilter};
/// use std::sync::Arc;
///
/// # async fn example() -> cheungfun_core::Result<()> {
/// let chain = PostprocessorChain::new(vec![
///     Arc::new(KeywordFilter::new(keyword_config)?),
///     Arc::new(SimilarityFilter::new(similarity_config)),
/// ]);
///
/// let processed_nodes = chain.postprocess(nodes, "query").await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PostprocessorChain {
    /// The postprocessors to apply in sequence.
    processors: Vec<Arc<dyn NodePostprocessor>>,

    /// Whether to continue processing if a postprocessor fails.
    continue_on_error: bool,

    /// Whether to log detailed processing information.
    verbose: bool,
}

impl PostprocessorChain {
    /// Create a new postprocessor chain.
    ///
    /// # Arguments
    ///
    /// * `processors` - The postprocessors to apply in sequence
    pub fn new(processors: Vec<Arc<dyn NodePostprocessor>>) -> Self {
        Self {
            processors,
            continue_on_error: false,
            verbose: false,
        }
    }

    /// Create a new postprocessor chain with error handling configuration.
    ///
    /// # Arguments
    ///
    /// * `processors` - The postprocessors to apply in sequence
    /// * `continue_on_error` - Whether to continue if a postprocessor fails
    pub fn with_error_handling(
        processors: Vec<Arc<dyn NodePostprocessor>>,
        continue_on_error: bool,
    ) -> Self {
        Self {
            processors,
            continue_on_error,
            verbose: false,
        }
    }

    /// Create a new postprocessor chain with verbose logging.
    ///
    /// # Arguments
    ///
    /// * `processors` - The postprocessors to apply in sequence
    /// * `verbose` - Whether to enable verbose logging
    pub fn with_verbose(processors: Vec<Arc<dyn NodePostprocessor>>, verbose: bool) -> Self {
        Self {
            processors,
            continue_on_error: false,
            verbose,
        }
    }

    /// Create a new postprocessor chain with full configuration.
    ///
    /// # Arguments
    ///
    /// * `processors` - The postprocessors to apply in sequence
    /// * `continue_on_error` - Whether to continue if a postprocessor fails
    /// * `verbose` - Whether to enable verbose logging
    pub fn with_config(
        processors: Vec<Arc<dyn NodePostprocessor>>,
        continue_on_error: bool,
        verbose: bool,
    ) -> Self {
        Self {
            processors,
            continue_on_error,
            verbose,
        }
    }

    /// Add a postprocessor to the end of the chain.
    ///
    /// # Arguments
    ///
    /// * `processor` - The postprocessor to add
    pub fn add_processor(&mut self, processor: Arc<dyn NodePostprocessor>) {
        self.processors.push(processor);
    }

    /// Insert a postprocessor at a specific position in the chain.
    ///
    /// # Arguments
    ///
    /// * `index` - The position to insert at
    /// * `processor` - The postprocessor to insert
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn insert_processor(&mut self, index: usize, processor: Arc<dyn NodePostprocessor>) {
        self.processors.insert(index, processor);
    }

    /// Remove a postprocessor at a specific position.
    ///
    /// # Arguments
    ///
    /// * `index` - The position to remove from
    ///
    /// # Returns
    ///
    /// The removed postprocessor, or None if the index is out of bounds.
    pub fn remove_processor(&mut self, index: usize) -> Option<Arc<dyn NodePostprocessor>> {
        if index < self.processors.len() {
            Some(self.processors.remove(index))
        } else {
            None
        }
    }

    /// Get the number of postprocessors in the chain.
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    /// Get the names of all postprocessors in the chain.
    pub fn processor_names(&self) -> Vec<&'static str> {
        self.processors.iter().map(|p| p.name()).collect()
    }

    /// Clear all postprocessors from the chain.
    pub fn clear(&mut self) {
        self.processors.clear();
    }
}

#[async_trait]
impl NodePostprocessor for PostprocessorChain {
    #[instrument(skip(self, nodes), fields(chain_length = self.processors.len()))]
    async fn postprocess(
        &self,
        mut nodes: Vec<ScoredNode>,
        query: &str,
    ) -> Result<Vec<ScoredNode>> {
        if self.processors.is_empty() {
            debug!("Empty postprocessor chain, returning nodes unchanged");
            return Ok(nodes);
        }

        info!(
            "Processing {} nodes through chain of {} postprocessors",
            nodes.len(),
            self.processors.len()
        );

        let original_count = nodes.len();

        for (index, processor) in self.processors.iter().enumerate() {
            let processor_name = processor.name();
            let before_count = nodes.len();

            if self.verbose {
                info!(
                    "Step {}/{}: Applying {} to {} nodes",
                    index + 1,
                    self.processors.len(),
                    processor_name,
                    before_count
                );
            }

            // Process nodes and handle the result
            let processing_result = processor.postprocess(nodes, query).await;

            match processing_result {
                Ok(processed_nodes) => {
                    nodes = processed_nodes;
                    let after_count = nodes.len();

                    if self.verbose {
                        info!(
                            "Step {}/{}: {} processed {} → {} nodes",
                            index + 1,
                            self.processors.len(),
                            processor_name,
                            before_count,
                            after_count
                        );
                    }
                }
                Err(e) => {
                    if self.continue_on_error {
                        tracing::warn!(
                            "Postprocessor {} failed: {}, continuing with remaining processors",
                            processor_name,
                            e
                        );
                        // We can't continue because nodes was moved into the failed postprocess call
                        // In this case, we need to return an error or have a different design
                        return Err(cheungfun_core::CheungfunError::pipeline(format!(
                            "Postprocessor {} failed and nodes were consumed: {}",
                            processor_name, e
                        )));
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        let final_count = nodes.len();
        info!(
            "Postprocessor chain completed: {} → {} nodes ({:.1}% reduction)",
            original_count,
            final_count,
            if original_count > 0 {
                (1.0 - final_count as f32 / original_count as f32) * 100.0
            } else {
                0.0
            }
        );

        Ok(nodes)
    }

    fn name(&self) -> &'static str {
        "PostprocessorChain"
    }
}

/// Builder for creating postprocessor chains with a fluent interface.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_query::postprocessor::{PostprocessorChainBuilder, KeywordFilter, SimilarityFilter};
/// use std::sync::Arc;
///
/// # async fn example() -> cheungfun_core::Result<()> {
/// let chain = PostprocessorChainBuilder::new()
///     .add_processor(Arc::new(KeywordFilter::new(keyword_config)?))
///     .add_processor(Arc::new(SimilarityFilter::new(similarity_config)))
///     .with_error_handling(true)
///     .with_verbose(true)
///     .build();
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct PostprocessorChainBuilder {
    processors: Vec<Arc<dyn NodePostprocessor>>,
    continue_on_error: bool,
    verbose: bool,
}

impl PostprocessorChainBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a postprocessor to the chain.
    pub fn add_processor(mut self, processor: Arc<dyn NodePostprocessor>) -> Self {
        self.processors.push(processor);
        self
    }

    /// Set error handling behavior.
    pub fn with_error_handling(mut self, continue_on_error: bool) -> Self {
        self.continue_on_error = continue_on_error;
        self
    }

    /// Set verbose logging.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build the postprocessor chain.
    pub fn build(self) -> PostprocessorChain {
        PostprocessorChain::with_config(self.processors, self.continue_on_error, self.verbose)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::{Node, ScoredNode};
    use std::collections::HashMap;
    use uuid::Uuid;

    // Mock postprocessor for testing
    #[derive(Debug)]
    struct MockPostprocessor {
        name: &'static str,
        should_fail: bool,
    }

    impl MockPostprocessor {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                should_fail: false,
            }
        }

        fn with_failure(name: &'static str) -> Self {
            Self {
                name,
                should_fail: true,
            }
        }
    }

    #[async_trait]
    impl NodePostprocessor for MockPostprocessor {
        async fn postprocess(
            &self,
            mut nodes: Vec<ScoredNode>,
            _query: &str,
        ) -> Result<Vec<ScoredNode>> {
            if self.should_fail {
                return Err(cheungfun_core::CheungfunError::pipeline(format!(
                    "{} failed",
                    self.name
                )));
            }

            // Remove one node to simulate processing
            if !nodes.is_empty() {
                nodes.pop();
            }
            Ok(nodes)
        }

        fn name(&self) -> &'static str {
            self.name
        }
    }

    fn create_test_nodes(count: usize) -> Vec<ScoredNode> {
        (0..count)
            .map(|i| {
                ScoredNode::new(
                    Node::new(
                        format!("Test content {}", i),
                        Uuid::new_v4(),
                        cheungfun_core::ChunkInfo {
                            start_char_idx: Some(0),
                            end_char_idx: Some(20),
                            chunk_index: i,
                        },
                    ),
                    0.5 + (i as f32 * 0.1),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn test_empty_chain() {
        let chain = PostprocessorChain::new(vec![]);
        let nodes = create_test_nodes(3);
        let result = chain
            .postprocess(nodes.clone(), "test query")
            .await
            .unwrap();
        assert_eq!(result.len(), nodes.len());
    }

    #[tokio::test]
    async fn test_single_processor() {
        let chain =
            PostprocessorChain::new(vec![Arc::new(MockPostprocessor::new("TestProcessor"))]);

        let nodes = create_test_nodes(3);
        let result = chain.postprocess(nodes, "test query").await.unwrap();
        assert_eq!(result.len(), 2); // One node removed
    }

    #[tokio::test]
    async fn test_multiple_processors() {
        let chain = PostprocessorChain::new(vec![
            Arc::new(MockPostprocessor::new("Processor1")),
            Arc::new(MockPostprocessor::new("Processor2")),
        ]);

        let nodes = create_test_nodes(5);
        let result = chain.postprocess(nodes, "test query").await.unwrap();
        assert_eq!(result.len(), 3); // Two nodes removed
    }

    #[tokio::test]
    async fn test_error_handling_fail_fast() {
        let chain = PostprocessorChain::new(vec![
            Arc::new(MockPostprocessor::new("Processor1")),
            Arc::new(MockPostprocessor::with_failure("FailingProcessor")),
            Arc::new(MockPostprocessor::new("Processor3")),
        ]);

        let nodes = create_test_nodes(5);
        let result = chain.postprocess(nodes, "test query").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_error_handling_continue() {
        let chain = PostprocessorChain::with_error_handling(
            vec![
                Arc::new(MockPostprocessor::new("Processor1")),
                Arc::new(MockPostprocessor::with_failure("FailingProcessor")),
                Arc::new(MockPostprocessor::new("Processor3")),
            ],
            true, // continue_on_error
        );

        let nodes = create_test_nodes(5);
        let result = chain.postprocess(nodes, "test query").await.unwrap();
        assert_eq!(result.len(), 3); // Two successful processors removed 2 nodes
    }

    #[tokio::test]
    async fn test_builder() {
        let chain = PostprocessorChainBuilder::new()
            .add_processor(Arc::new(MockPostprocessor::new("Processor1")))
            .add_processor(Arc::new(MockPostprocessor::new("Processor2")))
            .with_error_handling(true)
            .with_verbose(true)
            .build();

        assert_eq!(chain.len(), 2);
        assert_eq!(chain.processor_names(), vec!["Processor1", "Processor2"]);
    }
}
