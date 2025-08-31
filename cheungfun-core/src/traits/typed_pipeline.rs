//! Type-safe pipeline system for compile-time validation.
//!
//! This module provides a type-safe pipeline system that ensures component
//! compatibility at compile time, preventing invalid pipeline compositions
//! while maintaining excellent runtime performance.

use std::marker::PhantomData;

use super::transformer::{
    DocumentState, InputType, NodeState, OutputType, TypedData, TypedTransform,
};
use crate::{Document, Node, Result};

// ============================================================================
// Type-Safe Pipeline Builder
// ============================================================================

/// Type-safe pipeline builder that ensures component compatibility at compile time.
///
/// The builder uses Rust's type system to prevent invalid pipeline compositions.
/// For example, you cannot add a node processor before a document processor.
pub struct TypedPipelineBuilder<T: InputType> {
    document_processors: Vec<Box<dyn TypedTransform<DocumentState, NodeState>>>,
    node_processors: Vec<Box<dyn TypedTransform<NodeState, NodeState>>>,
    _phantom: PhantomData<T>,
}

impl TypedPipelineBuilder<DocumentState> {
    /// Create a new pipeline builder that starts with documents.
    pub fn new() -> Self {
        Self {
            document_processors: Vec::new(),
            node_processors: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Add a document processor (Documents -> Nodes).
    ///
    /// This transitions the pipeline from DocumentState to NodeState,
    /// enabling subsequent node processors to be added.
    pub fn add_document_processor<C>(mut self, component: C) -> TypedPipelineBuilder<NodeState>
    where
        C: TypedTransform<DocumentState, NodeState> + 'static,
    {
        self.document_processors.push(Box::new(component));
        TypedPipelineBuilder {
            document_processors: self.document_processors,
            node_processors: self.node_processors,
            _phantom: PhantomData,
        }
    }
}

impl TypedPipelineBuilder<NodeState> {
    /// Add a node processor (Nodes -> Nodes).
    ///
    /// Node processors can be chained indefinitely since they maintain
    /// the NodeState throughout the pipeline.
    pub fn add_node_processor<C>(mut self, component: C) -> Self
    where
        C: TypedTransform<NodeState, NodeState> + 'static,
    {
        self.node_processors.push(Box::new(component));
        self
    }

    /// Build the final type-safe pipeline.
    pub fn build(self) -> TypedPipeline<NodeState> {
        TypedPipeline {
            document_processors: self.document_processors,
            node_processors: self.node_processors,
            _phantom: PhantomData,
        }
    }
}

impl Default for TypedPipelineBuilder<DocumentState> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Type-Safe Pipeline
// ============================================================================

/// Type-safe pipeline that guarantees valid component composition.
///
/// This pipeline provides compile-time guarantees that all components
/// are compatible with each other, eliminating runtime type errors.
pub struct TypedPipeline<T: OutputType> {
    document_processors: Vec<Box<dyn TypedTransform<DocumentState, NodeState>>>,
    node_processors: Vec<Box<dyn TypedTransform<NodeState, NodeState>>>,
    _phantom: PhantomData<T>,
}

impl TypedPipeline<NodeState> {
    /// Execute the pipeline with documents as input.
    ///
    /// This method processes documents through all pipeline components
    /// and returns the final processed nodes.
    pub async fn run(&self, documents: Vec<Document>) -> Result<Vec<Node>> {
        if self.document_processors.is_empty() && self.node_processors.is_empty() {
            return Err(crate::CheungfunError::Validation {
                message: "Pipeline is empty - no components to execute".into(),
            });
        }

        // First, process documents through document processors
        let mut nodes = if self.document_processors.is_empty() {
            // If no document processors, convert documents to nodes directly
            documents
                .into_iter()
                .enumerate()
                .map(|(idx, doc)| {
                    let chunk_info = crate::types::ChunkInfo::new(None, None, idx);
                    Node::new(doc.content, doc.id, chunk_info)
                })
                .collect()
        } else {
            // Apply the single document processor (Documents -> Nodes)
            // In our design, there should be exactly one document processor
            if self.document_processors.len() != 1 {
                return Err(crate::CheungfunError::Validation {
                    message: format!(
                        "Expected exactly 1 document processor, found {}",
                        self.document_processors.len()
                    ),
                });
            }

            let processor = &self.document_processors[0];
            let doc_data = TypedData::from_documents(documents);
            match processor.transform(doc_data).await {
                Ok(node_data) => node_data.into_nodes(),
                Err(e) => {
                    return Err(crate::CheungfunError::Pipeline {
                        message: format!("Document processor ({}) failed: {}", processor.name(), e),
                    });
                }
            }
        };

        // Apply node processors (Nodes -> Nodes)
        for (index, processor) in self.node_processors.iter().enumerate() {
            let node_data = TypedData::from_nodes(nodes);
            match processor.transform(node_data).await {
                Ok(processed_data) => {
                    nodes = processed_data.into_nodes();
                }
                Err(e) => {
                    return Err(crate::CheungfunError::Pipeline {
                        message: format!(
                            "Node processor {} ({}) failed: {}",
                            index,
                            processor.name(),
                            e
                        ),
                    });
                }
            }
        }

        Ok(nodes)
    }

    /// Execute the pipeline with pre-processed nodes as input.
    ///
    /// This method is useful when you want to apply only the node processing
    /// components of the pipeline to already processed nodes.
    pub async fn run_with_nodes(&self, mut nodes: Vec<Node>) -> Result<Vec<Node>> {
        if self.node_processors.is_empty() {
            return Ok(nodes);
        }

        // Apply node processors (Nodes -> Nodes)
        for (index, processor) in self.node_processors.iter().enumerate() {
            let node_data = TypedData::from_nodes(nodes);
            match processor.transform(node_data).await {
                Ok(processed_data) => {
                    nodes = processed_data.into_nodes();
                }
                Err(e) => {
                    return Err(crate::CheungfunError::Pipeline {
                        message: format!(
                            "Node processor {} ({}) failed: {}",
                            index,
                            processor.name(),
                            e
                        ),
                    });
                }
            }
        }

        Ok(nodes)
    }

    /// Get the number of components in this pipeline.
    pub fn len(&self) -> usize {
        self.document_processors.len() + self.node_processors.len()
    }

    /// Check if the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.document_processors.is_empty() && self.node_processors.is_empty()
    }

    /// Get the names of all components in this pipeline.
    pub fn component_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        names.extend(self.document_processors.iter().map(|c| c.name()));
        names.extend(self.node_processors.iter().map(|c| c.name()));
        names
    }
}

impl std::fmt::Debug for TypedPipeline<NodeState> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypedPipeline")
            .field("document_processors", &self.document_processors.len())
            .field("node_processors", &self.node_processors.len())
            .field("component_names", &self.component_names())
            .finish()
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a new type-safe pipeline builder.
pub fn pipeline() -> TypedPipelineBuilder<DocumentState> {
    TypedPipelineBuilder::new()
}
