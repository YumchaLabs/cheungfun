// Advanced Retrieval Demo
// 高级检索功能演示

use anyhow::Result;
use cheungfun::prelude::*;
use cheungfun_query::advanced::*;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, Level};

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("🚀 Starting Advanced Retrieval Demo");

    // 1. 设置基础组件
    let (embedder, vector_store, llm_client) = setup_components().await?;

    // 2. 演示基础混合搜索
    demo_basic_hybrid_search(&embedder, &vector_store, &llm_client).await?;

    // 3. 演示高级Pipeline
    demo_advanced_pipeline(&embedder, &vector_store, &llm_client).await?;

    // 4. 演示专门的代码搜索Pipeline
    demo_code_search_pipeline(&embedder, &vector_store, &llm_client).await?;

    // 5. 演示性能对比
    demo_performance_comparison(&embedder, &vector_store, &llm_client).await?;

    info!("✅ Advanced Retrieval Demo completed successfully!");
    Ok(())
}

/// 设置基础组件
async fn setup_components() -> Result<(
    Arc<dyn Embedder>,
    Arc<dyn VectorStore>,
    Arc<dyn ResponseGenerator>,
)> {
    info!("Setting up components...");

    // 创建嵌入器
    let embedder = Arc::new(FastEmbedder::new().await?);

    // 创建向量存储
    let vector_store = Arc::new(
        QdrantVectorStore::new(
            QdrantConfig::new("http://localhost:6334", "advanced_demo", 384)
                .with_create_collection_if_missing(true)
        ).await?
    );

    // 创建LLM客户端
    let llm_client = Arc::new(
        SiumaiGenerator::new(
            siumai::Siumai::builder()
                .openai()
                .build()
                .await?
        )
    );

    // 索引一些示例文档
    index_sample_documents(&embedder, &vector_store).await?;

    Ok((embedder, vector_store, llm_client))
}

/// 索引示例文档
async fn index_sample_documents(
    embedder: &Arc<dyn Embedder>,
    vector_store: &Arc<dyn VectorStore>,
) -> Result<()> {
    info!("Indexing sample documents...");

    let documents = vec![
        "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently.",
        "RAG (Retrieval-Augmented Generation) combines information retrieval with language generation for better AI responses.",
        "Hybrid search combines vector similarity search with traditional keyword search for improved relevance.",
        "The transformer architecture revolutionized natural language processing with its attention mechanism.",
        "Embeddings are dense vector representations of text that capture semantic meaning in high-dimensional space.",
        "Large language models like GPT have shown remarkable capabilities in understanding and generating human-like text.",
    ];

    let mut nodes = Vec::new();
    for (i, content) in documents.iter().enumerate() {
        let embedding = embedder.embed(content).await?;
        let node = Node {
            id: uuid::Uuid::new_v4(),
            content: content.to_string(),
            metadata: std::collections::HashMap::new(),
            embedding: Some(embedding),
            sparse_embedding: None,
            relationships: std::collections::HashMap::new(),
            source_document_id: uuid::Uuid::new_v4(),
            chunk_info: ChunkInfo {
                start_offset: 0,
                end_offset: content.len(),
                chunk_index: i,
            },
        };
        nodes.push(node);
    }

    vector_store.add(nodes).await?;
    info!("Indexed {} documents", documents.len());
    Ok(())
}

/// 演示基础混合搜索
async fn demo_basic_hybrid_search(
    embedder: &Arc<dyn Embedder>,
    vector_store: &Arc<dyn VectorStore>,
    llm_client: &Arc<dyn ResponseGenerator>,
) -> Result<()> {
    info!("\n🔍 Demo: Basic Hybrid Search");

    // 创建混合搜索策略
    let hybrid_strategy = HybridSearchStrategy::builder()
        .with_vector_weight(0.7)
        .with_keyword_weight(0.3)
        .with_keyword_fields(vec!["content"])
        .with_fusion_method(FusionMethod::ReciprocalRankFusion { k: 60.0 })
        .build();

    // 创建简单的Pipeline
    let pipeline = AdvancedRetrievalPipeline::builder()
        .with_search_strategy(hybrid_strategy)
        .with_vector_store(vector_store.clone())
        .build()?;

    // 执行搜索
    let query = "What is machine learning and AI?";
    let response = pipeline.retrieve(query).await?;

    println!("\n📊 Basic Hybrid Search Results:");
    println!("Query: {}", query);
    println!("Found {} results in {:?}", 
             response.nodes.len(), 
             response.stats.retrieval_time);

    for (i, node) in response.nodes.iter().take(3).enumerate() {
        println!("{}. [Score: {:.3}] {}", 
                 i + 1, 
                 node.score, 
                 node.node.content.chars().take(100).collect::<String>());
    }

    Ok(())
}

/// 演示高级Pipeline
async fn demo_advanced_pipeline(
    embedder: &Arc<dyn Embedder>,
    vector_store: &Arc<dyn VectorStore>,
    llm_client: &Arc<dyn ResponseGenerator>,
) -> Result<()> {
    info!("\n🚀 Demo: Advanced Pipeline with All Features");

    // 创建高级Pipeline
    let pipeline = AdvancedRetrievalPipeline::builder()
        // 查询转换器
        .add_query_transformer(
            HyDETransformer::new(llm_client.clone())
                .with_num_hypothetical_docs(2)
                .with_include_original(true)
        )
        .add_query_transformer(
            SubquestionTransformer::new(llm_client.clone())
                .with_num_subquestions(3)
        )
        // 混合搜索策略
        .with_search_strategy(
            HybridSearchStrategy::builder()
                .with_vector_weight(0.6)
                .with_keyword_weight(0.4)
                .with_fusion_method(FusionMethod::ReciprocalRankFusion { k: 60.0 })
                .build()
        )
        // 重排序器
        .add_reranker(
            LLMReranker::new(llm_client.clone())
                .with_top_n(10)
                .with_batch_size(5)
        )
        .add_reranker(
            ScoreReranker::new(ScoreRerankStrategy::Diversity { 
                similarity_threshold: 0.8 
            }).with_top_n(8)
        )
        // 响应转换器
        .add_response_transformer(
            DeduplicationTransformer::new(0.9)
                .with_method(DeduplicationMethod::TextSimilarity)
        )
        .add_response_transformer(
            FilterTransformer::new()
                .with_min_score(0.1)
                .with_content_length(Some(10), Some(500))
        )
        .add_response_transformer(
            SummaryTransformer::new(llm_client.clone())
                .with_max_length(150)
                .with_preserve_original(true)
        )
        .with_vector_store(vector_store.clone())
        .with_config(
            PipelineConfig {
                concurrency: 4,
                timeout: Duration::from_secs(60),
                enable_cache: true,
                enable_metrics: true,
                enable_concurrent_query_transform: true,
                enable_concurrent_rerank: false,
                retry_config: RetryConfig::default(),
            }
        )
        .build()?;

    // 执行复杂查询
    let query = "How do vector databases work with machine learning embeddings?";
    let response = pipeline.retrieve(query).await?;

    println!("\n🎯 Advanced Pipeline Results:");
    println!("Query: {}", query);
    println!("Processing stages:");
    for (stage, duration) in &response.stats.stage_times {
        println!("  - {}: {:?}", stage, duration);
    }
    println!("Total time: {:?}", response.stats.retrieval_time);
    println!("Query transformations: {}", response.stats.query_transformations);
    println!("Rerank operations: {}", response.stats.rerank_operations);
    println!("Response transformations: {}", response.stats.response_transformations);
    println!("Final results: {}", response.nodes.len());

    for (i, node) in response.nodes.iter().take(3).enumerate() {
        println!("\n{}. [Score: {:.3}]", i + 1, node.score);
        println!("   Content: {}", 
                 node.node.content.chars().take(80).collect::<String>());
        if let Some(summary) = node.node.metadata.get("summary") {
            println!("   Summary: {}", summary.as_str().unwrap_or(""));
        }
    }

    Ok(())
}

/// 演示专门的代码搜索Pipeline
async fn demo_code_search_pipeline(
    embedder: &Arc<dyn Embedder>,
    vector_store: &Arc<dyn VectorStore>,
    llm_client: &Arc<dyn ResponseGenerator>,
) -> Result<()> {
    info!("\n💻 Demo: Specialized Code Search Pipeline");

    // 创建代码搜索专用Pipeline
    let code_pipeline = AdvancedRetrievalPipeline::builder()
        // 代码相关的查询转换
        .add_query_transformer(
            SubquestionTransformer::new(llm_client.clone())
                .with_prompt_template(CODE_SUBQUESTION_TEMPLATE.to_string())
                .with_num_subquestions(2)
        )
        // 混合搜索，更重视关键词匹配
        .with_search_strategy(
            HybridSearchStrategy::builder()
                .with_vector_weight(0.4)
                .with_keyword_weight(0.6)
                .with_keyword_fields(vec!["content", "function_name", "class_name"])
                .build()
        )
        // 代码相关的重排序
        .add_reranker(
            LLMReranker::new(llm_client.clone())
                .with_prompt_template(CODE_RERANK_TEMPLATE.to_string())
                .with_top_n(8)
        )
        // 去重和过滤
        .add_response_transformer(
            DeduplicationTransformer::new(0.85)
                .with_method(DeduplicationMethod::EmbeddingSimilarity)
        )
        .add_response_transformer(
            FilterTransformer::new()
                .with_min_score(0.2)
        )
        .with_vector_store(vector_store.clone())
        .build()?;

    let query = "How to implement thread safety in Rust?";
    let response = code_pipeline.retrieve(query).await?;

    println!("\n🔧 Code Search Results:");
    println!("Query: {}", query);
    println!("Found {} code-relevant results", response.nodes.len());

    for (i, node) in response.nodes.iter().take(2).enumerate() {
        println!("{}. [Score: {:.3}] {}", 
                 i + 1, 
                 node.score, 
                 node.node.content.chars().take(120).collect::<String>());
    }

    Ok(())
}

/// 演示性能对比
async fn demo_performance_comparison(
    embedder: &Arc<dyn Embedder>,
    vector_store: &Arc<dyn VectorStore>,
    llm_client: &Arc<dyn ResponseGenerator>,
) -> Result<()> {
    info!("\n⚡ Demo: Performance Comparison");

    let query = "What are embeddings in machine learning?";

    // 1. 基础向量搜索
    let start = std::time::Instant::now();
    let basic_pipeline = AdvancedRetrievalPipeline::builder()
        .with_search_strategy(VectorSearchStrategy::new(VectorSearchConfig::default()))
        .with_vector_store(vector_store.clone())
        .build()?;
    let basic_response = basic_pipeline.retrieve(query).await?;
    let basic_time = start.elapsed();

    // 2. 混合搜索
    let start = std::time::Instant::now();
    let hybrid_pipeline = AdvancedRetrievalPipeline::builder()
        .with_search_strategy(HybridSearchStrategy::builder().build())
        .with_vector_store(vector_store.clone())
        .build()?;
    let hybrid_response = hybrid_pipeline.retrieve(query).await?;
    let hybrid_time = start.elapsed();

    // 3. 完整高级Pipeline
    let start = std::time::Instant::now();
    let advanced_pipeline = AdvancedRetrievalPipeline::builder()
        .add_query_transformer(HyDETransformer::new(llm_client.clone()))
        .with_search_strategy(HybridSearchStrategy::builder().build())
        .add_reranker(LLMReranker::new(llm_client.clone()).with_top_n(5))
        .add_response_transformer(DeduplicationTransformer::new(0.9))
        .with_vector_store(vector_store.clone())
        .build()?;
    let advanced_response = advanced_pipeline.retrieve(query).await?;
    let advanced_time = start.elapsed();

    println!("\n📈 Performance Comparison:");
    println!("Query: {}", query);
    println!();
    println!("1. Basic Vector Search:");
    println!("   Time: {:?}", basic_time);
    println!("   Results: {}", basic_response.nodes.len());
    println!();
    println!("2. Hybrid Search:");
    println!("   Time: {:?}", hybrid_time);
    println!("   Results: {}", hybrid_response.nodes.len());
    println!();
    println!("3. Advanced Pipeline:");
    println!("   Time: {:?}", advanced_time);
    println!("   Results: {}", advanced_response.nodes.len());
    println!("   Query transformations: {}", advanced_response.stats.query_transformations);
    println!("   Rerank operations: {}", advanced_response.stats.rerank_operations);

    Ok(())
}

// 代码搜索相关的提示模板
const CODE_SUBQUESTION_TEMPLATE: &str = r#"
Given a programming-related question, break it down into {num_questions} more specific technical sub-questions that would help find relevant code examples and documentation.

Original question: {query}

Please provide {num_questions} technical sub-questions:
"#;

const CODE_RERANK_TEMPLATE: &str = r#"
Given a programming query and code-related documents, rank them by technical relevance and code quality.
Consider factors like: code correctness, best practices, completeness, and direct relevance to the query.

Query: {query}

Documents:
{documents}

Rank these {num_docs} documents by their technical relevance:
"#;
