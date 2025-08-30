//! Integration tests for hierarchical retrieval system.

use std::sync::Arc;

use cheungfun_core::{Document, Result, Retriever};
use cheungfun_query::{
    engine::{
        PerformanceProfile, QueryEngineMetadata, QuerySelector, QueryType, RuleBasedQuerySelector,
    },
    hierarchical::HierarchicalSystemBuilder,
    retrievers::hierarchical::{HierarchicalRetriever, StorageContext},
};
use futures;

/// Mock storage context for testing.
#[derive(Debug)]
struct MockStorageContext;

#[async_trait::async_trait]
impl StorageContext for MockStorageContext {
    async fn get_node(&self, _node_id: &str) -> Result<cheungfun_core::Node> {
        Err(cheungfun_core::CheungfunError::NotFound {
            resource: "Mock node not found".to_string(),
        })
    }

    async fn get_nodes(&self, _node_ids: &[String]) -> Result<Vec<cheungfun_core::Node>> {
        Ok(vec![])
    }
}

/// Mock embedder for testing.
#[derive(Debug)]
struct MockEmbedder;

#[async_trait::async_trait]
impl cheungfun_core::traits::Embedder for MockEmbedder {
    fn name(&self) -> &'static str {
        "MockEmbedder"
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Ok(vec![0.1, 0.2, 0.3, 0.4])
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        for _ in texts {
            embeddings.push(vec![0.1, 0.2, 0.3, 0.4]);
        }
        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        4
    }

    fn model_name(&self) -> &str {
        "mock-model"
    }
}

/// Mock vector store for testing.
#[derive(Debug)]
struct MockVectorStore;

#[async_trait::async_trait]
impl cheungfun_core::traits::VectorStore for MockVectorStore {
    fn name(&self) -> &'static str {
        "MockVectorStore"
    }

    async fn add(&self, _nodes: Vec<cheungfun_core::Node>) -> Result<Vec<uuid::Uuid>> {
        Ok(vec![])
    }

    async fn update(&self, _nodes: Vec<cheungfun_core::Node>) -> Result<()> {
        Ok(())
    }

    async fn delete(&self, _node_ids: Vec<uuid::Uuid>) -> Result<()> {
        Ok(())
    }

    async fn search(
        &self,
        _query: &cheungfun_core::types::Query,
    ) -> Result<Vec<cheungfun_core::ScoredNode>> {
        Ok(vec![])
    }

    async fn get(&self, _node_ids: Vec<uuid::Uuid>) -> Result<Vec<Option<cheungfun_core::Node>>> {
        Ok(vec![])
    }

    async fn clear(&self) -> Result<()> {
        Ok(())
    }

    async fn health_check(&self) -> Result<()> {
        Ok(())
    }
}

/// Mock response generator for testing.
#[derive(Debug)]
struct MockGenerator;

#[async_trait::async_trait]
impl cheungfun_core::traits::ResponseGenerator for MockGenerator {
    fn name(&self) -> &'static str {
        "MockGenerator"
    }

    async fn generate_response(
        &self,
        _prompt: &str,
        _context_nodes: Vec<cheungfun_core::ScoredNode>,
        _options: &cheungfun_core::types::GenerationOptions,
    ) -> Result<cheungfun_core::types::GeneratedResponse> {
        Ok(cheungfun_core::types::GeneratedResponse::new(
            "Mock response".to_string(),
        ))
    }

    async fn generate_response_stream(
        &self,
        _prompt: &str,
        _context_nodes: Vec<cheungfun_core::ScoredNode>,
        _options: &cheungfun_core::types::GenerationOptions,
    ) -> Result<std::pin::Pin<Box<dyn futures::Stream<Item = Result<String>> + Send + 'static>>>
    {
        use futures::stream;
        let stream = stream::once(async { Ok("Mock response".to_string()) });
        Ok(Box::pin(stream))
    }
}

#[tokio::test]
async fn test_hierarchical_retriever_creation() -> Result<()> {
    // Test creating a hierarchical retriever
    let storage_context = Arc::new(MockStorageContext);
    let embedder = Arc::new(MockEmbedder);
    let vector_store = Arc::new(MockVectorStore);

    let leaf_retriever = Arc::new(cheungfun_query::retriever::VectorRetriever::new(
        vector_store,
        embedder,
    ));

    let hierarchical_retriever = HierarchicalRetriever::builder()
        .leaf_retriever(leaf_retriever)
        .storage_context(storage_context)
        .merge_threshold(0.5)
        .verbose(true)
        .build()?;

    // Test basic retrieval (should not fail)
    let query = cheungfun_core::types::Query::builder()
        .text("test query")
        .build();

    let results = hierarchical_retriever.retrieve(&query).await?;
    assert_eq!(results.len(), 0); // Mock returns empty results

    Ok(())
}

#[tokio::test]
async fn test_query_selector() -> Result<()> {
    // Test rule-based query selector
    let selector = RuleBasedQuerySelector::new().with_verbose(true);

    // Create mock query engines
    let mock_engines = vec![
        cheungfun_query::engine::QueryEngineWrapper {
            engine: Arc::new(
                cheungfun_query::engine::QueryEngine::builder()
                    .retriever(Arc::new(cheungfun_query::retriever::VectorRetriever::new(
                        Arc::new(MockVectorStore),
                        Arc::new(MockEmbedder),
                    )))
                    .generator(Arc::new(MockGenerator))
                    .build()?,
            ),
            metadata: QueryEngineMetadata {
                name: "summary".to_string(),
                description: "Summary engine".to_string(),
                suitable_for: vec![QueryType::Summary],
                performance_profile: PerformanceProfile::Fast,
            },
        },
        cheungfun_query::engine::QueryEngineWrapper {
            engine: Arc::new(
                cheungfun_query::engine::QueryEngine::builder()
                    .retriever(Arc::new(cheungfun_query::retriever::VectorRetriever::new(
                        Arc::new(MockVectorStore),
                        Arc::new(MockEmbedder),
                    )))
                    .generator(Arc::new(MockGenerator))
                    .build()?,
            ),
            metadata: QueryEngineMetadata {
                name: "detailed".to_string(),
                description: "Detailed engine".to_string(),
                suitable_for: vec![QueryType::Detailed],
                performance_profile: PerformanceProfile::Thorough,
            },
        },
    ];

    // Test selection
    let selection = selector
        .select(&mock_engines, "What is this about?")
        .await?;
    assert_eq!(selection.index, 0); // Should select first engine for summary query

    let selection = selector
        .select(&mock_engines, "How does this work in detail?")
        .await?;
    assert_eq!(selection.index, 1); // Should select second engine for detailed query

    Ok(())
}

#[test]
fn test_hierarchical_system_builder() {
    // Test builder pattern
    let builder = HierarchicalSystemBuilder::new()
        .documents(vec![Document::new("Test content")])
        .chunk_sizes(vec![1024, 256])
        .merge_threshold(0.6)
        .verbose(true);

    // Builder should be created successfully
    assert!(format!("{:?}", builder).contains("HierarchicalSystemBuilder"));
}

#[test]
fn test_query_engine_metadata() {
    let metadata = QueryEngineMetadata {
        name: "test_engine".to_string(),
        description: "Test engine description".to_string(),
        suitable_for: vec![QueryType::Summary, QueryType::Detailed],
        performance_profile: PerformanceProfile::Balanced,
    };

    assert_eq!(metadata.name, "test_engine");
    assert_eq!(metadata.suitable_for.len(), 2);
}
