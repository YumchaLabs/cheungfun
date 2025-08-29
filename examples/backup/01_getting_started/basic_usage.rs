//! Basic usage example for the Cheungfun RAG framework.
//!
//! This example demonstrates how to use the core types and builders
//! to set up a basic RAG pipeline configuration.

use cheungfun::prelude::*;
use tracing_subscriber::fmt::init;

#[tokio::main]
#[allow(clippy::too_many_lines)]
async fn main() -> Result<()> {
    // Initialize logging
    init();

    println!("🚀 Cheungfun RAG Framework - Basic Usage Example");
    println!("================================================");

    // 1. Create a sample document
    println!("\n📄 Creating a sample document...");
    let doc = Document::builder()
        .content("Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.")
        .metadata("source", "rust_intro.txt")
        .metadata("author", "Rust Team")
        .metadata("category", "programming")
        .build();

    println!("✅ Document created:");
    println!("   ID: {}", doc.id);
    println!("   Content: {}", doc.content);
    println!(
        "   Source: {}",
        doc.get_metadata_string("source").unwrap_or_default()
    );

    // 2. Create nodes from the document
    println!("\n🔗 Creating nodes from document...");
    let chunk_info = ChunkInfo::with_char_indices(0, doc.content.len(), 0);
    let node = Node::builder()
        .content(doc.content.clone())
        .source_document_id(doc.id)
        .chunk_info(chunk_info)
        .metadata("word_count", doc.content.split_whitespace().count())
        .build()
        .expect("Failed to build node");

    println!("✅ Node created:");
    println!("   ID: {}", node.id);
    println!("   Source Document: {}", node.source_document_id);
    println!(
        "   Word Count: {}",
        node.get_metadata("word_count").unwrap()
    );

    // 3. Create a query
    println!("\n❓ Creating a sample query...");
    let query = Query::builder()
        .text("What is Rust programming language?")
        .top_k(5)
        .search_mode(SearchMode::Vector)
        .filter("category", "programming")
        .build();

    println!("✅ Query created:");
    println!("   Text: {}", query.text);
    println!("   Top-K: {}", query.top_k);
    println!("   Has filters: {}", query.has_filters());

    // 4. Create scored nodes (simulating search results)
    println!("\n🎯 Creating scored search results...");
    let scored_node = ScoredNode::new(node.expect("Failed to get node"), 0.95);
    println!(
        "✅ Scored node created with similarity: {}",
        scored_node.score
    );

    // 5. Create a generated response
    println!("\n🤖 Creating a sample response...");
    let response = GeneratedResponse::new(
        "Rust is a systems programming language focused on safety, speed, and concurrency. \
         It prevents segfaults and guarantees thread safety through its ownership system.",
    )
    .with_source_node(scored_node.node.id)
    .with_usage(TokenUsage::new(50, 30))
    .with_metadata("model", "example-llm");

    println!("✅ Response generated:");
    println!("   Content: {}", response.content);
    println!("   Source nodes: {}", response.source_nodes.len());
    println!("   Total tokens: {}", response.total_tokens().unwrap_or(0));

    // 6. Create a complete query response
    println!("\n📋 Creating complete query response...");
    let query_response = QueryResponse::new(response, vec![scored_node])
        .with_metadata("query_time", 150)
        .with_metadata("retrieval_method", "vector_search");

    println!("✅ Query response created:");
    println!("   Retrieved nodes: {}", query_response.num_retrieved());
    println!("   Response content: {}", query_response.content());

    // 7. Demonstrate configuration
    println!("\n⚙️  Creating pipeline configurations...");

    // Embedder configuration
    let embedder_config = EmbedderConfig::candle("sentence-transformers/all-MiniLM-L6-v2", "cpu")
        .with_batch_size(32)
        .with_normalize(true);

    println!(
        "✅ Embedder config: {} ({})",
        embedder_config.model_name(),
        embedder_config.provider()
    );

    // Vector store configuration
    let vector_store_config = VectorStoreConfig::memory(384)
        .with_distance_metric(DistanceMetric::Cosine)
        .with_capacity(10000);

    println!(
        "✅ Vector store config: {} (dimension: {})",
        vector_store_config.store_type(),
        vector_store_config.dimension()
    );

    // LLM configuration
    let llm_config = LlmConfig::new("local", "llama-2-7b")
        .with_base_url("http://localhost:8080")
        .with_temperature(0.7)
        .with_max_tokens(1000);

    println!(
        "✅ LLM config: {} ({})",
        llm_config.model, llm_config.provider
    );

    // 8. Create pipeline configurations
    println!("\n🔧 Creating pipeline configurations...");

    let indexing_config =
        IndexingPipelineConfig::new(embedder_config.clone(), vector_store_config.clone())
            .with_batch_size(50)
            .with_chunk_size(1000)
            .with_chunk_overlap(200);

    println!("✅ Indexing pipeline config:");
    println!("   Batch size: {}", indexing_config.batch_size);
    println!("   Chunk size: {}", indexing_config.chunk_size);
    println!("   Chunk overlap: {}", indexing_config.chunk_overlap);

    let query_config = QueryPipelineConfig::new(embedder_config, vector_store_config, llm_config)
        .with_top_k(10)
        .with_similarity_threshold(0.7)
        .with_citations(true);

    println!("✅ Query pipeline config:");
    println!("   Top-K: {}", query_config.top_k);
    println!(
        "   Similarity threshold: {:?}",
        query_config.similarity_threshold
    );
    println!("   Include citations: {}", query_config.include_citations);

    // 9. Validate configurations
    println!("\n✅ Validating configurations...");
    indexing_config.validate()?;
    query_config.validate()?;
    println!("✅ All configurations are valid!");

    // 10. Demonstrate builders
    println!("\n🏗️  Creating pipeline builders...");

    let indexing_builder = IndexingPipelineBuilder::new()
        .with_batch_size(64)
        .with_chunk_size(1500)
        .with_concurrency(4);

    let builder_info = indexing_builder.info();
    println!("✅ Indexing builder created:");
    println!("   Has embedder: {}", builder_info.has_embedder);
    println!("   Has vector store: {}", builder_info.has_vector_store);
    println!("   Is complete: {}", builder_info.is_complete);

    let query_builder = QueryPipelineBuilder::new()
        .with_top_k(15)
        .with_temperature(0.8)
        .with_citations(true);

    let query_builder_info = query_builder.info();
    println!("✅ Query builder created:");
    println!("   Has embedder: {}", query_builder_info.has_embedder);
    println!("   Has LLM: {}", query_builder_info.has_response_generator);
    println!("   Is complete: {}", query_builder_info.is_complete);

    println!("\n🎉 Basic usage example completed successfully!");
    println!("   This demonstrates the core types and configuration system.");
    println!("   Next steps: Implement concrete components in other crates.");

    Ok(())
}
