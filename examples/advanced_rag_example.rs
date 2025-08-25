//! Advanced RAG example showcasing all new features.
//!
//! This example demonstrates:
//! - Storage context with unified storage management
//! - Chat memory with conversation history
//! - Advanced RAG features (deep research, query rewriting, reranking)
//! - Enhanced configuration system with JSON support
//! - Hot reloading configuration

use std::sync::Arc;
use tokio::sync::Mutex;

use cheungfun_core::{
    config::{ConfigManager, JsonConfigurable, IndexingPipelineConfig},
    traits::{BaseMemory, StorageContext},
    ChatMessage, MessageRole,
};
use cheungfun_query::{
    engine::{QueryEngine, QueryEngineBuilder, QueryRewriteStrategy},
    memory::{ChatMemoryBuffer, ChatMemoryConfig},
};
use cheungfun_integrations::storage::{SqlxStorageConfig, SqlxDocumentStore, SqlxChatStore, SqlxIndexStore};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::init();

    println!("🚀 Advanced RAG Example with Cheungfun");
    println!("=====================================");

    // 1. Enhanced Configuration System
    println!("\n📋 1. Loading configuration from JSON files...");
    let mut config_manager = ConfigManager::new();
    
    // Load configuration from directory
    config_manager.load_from_directory("./config").await?;
    
    // Enable hot reloading (if feature is enabled)
    #[cfg(feature = "hot-reload")]
    {
        config_manager.enable_hot_reload().await?;
        println!("✅ Hot reload enabled for configuration files");
    }

    // Get configuration values
    let db_url = config_manager.get_string("database.url")?;
    let embedding_model = config_manager.get_string("embedding.model")?;
    let max_memory_tokens = config_manager.get_u32("memory.max_tokens")?;
    
    println!("✅ Configuration loaded:");
    println!("   - Database URL: {}", db_url);
    println!("   - Embedding Model: {}", embedding_model);
    println!("   - Max Memory Tokens: {}", max_memory_tokens);

    // 2. Storage Context Setup
    println!("\n🗄️  2. Setting up unified storage context...");
    
    // Create storage configuration
    let storage_config = SqlxStorageConfig::new(db_url)
        .with_auto_migrate(true)
        .with_table_prefix("cheungfun_");

    // Create storage components
    let doc_store = Arc::new(SqlxDocumentStore::new(storage_config.clone()).await?);
    let chat_store = Arc::new(SqlxChatStore::new(storage_config.clone()).await?);
    let index_store = Arc::new(SqlxIndexStore::new(storage_config.clone()).await?);
    
    // Create vector store (using in-memory for this example)
    let vector_store = Arc::new(
        cheungfun_integrations::InMemoryVectorStore::new(768)
    );

    // Create unified storage context
    let storage_context = Arc::new(StorageContext::new(
        doc_store,
        index_store,
        vector_store,
        Some(chat_store),
    ));

    println!("✅ Storage context created with unified storage management");

    // 3. Chat Memory Setup
    println!("\n🧠 3. Setting up conversation memory...");
    
    let memory_config = ChatMemoryConfig::with_token_limit(max_memory_tokens as usize)
        .with_summarization(Some("Summarize the previous conversation context.".to_string()));
    
    let chat_memory = Arc::new(Mutex::new(
        ChatMemoryBuffer::new(memory_config)
    ));

    println!("✅ Chat memory configured with {} token limit", max_memory_tokens);

    // 4. Query Engine with Advanced Features
    println!("\n🔍 4. Creating advanced query engine...");
    
    // Create embedder and retriever (simplified for example)
    let embedder = create_embedder(&embedding_model).await?;
    let retriever = create_retriever(storage_context.clone(), embedder).await?;
    let generator = create_generator().await?;

    // Build query engine with memory support
    let query_engine = QueryEngineBuilder::new()
        .retriever(retriever)
        .generator(generator)
        .memory(chat_memory.clone())
        .build()?;

    println!("✅ Query engine created with memory and advanced features");

    // 5. Demonstrate Advanced RAG Features
    println!("\n🎯 5. Demonstrating advanced RAG features...");

    // Basic chat interaction
    println!("\n💬 Basic Chat:");
    let response1 = query_engine.chat(
        "What is machine learning?",
        None
    ).await?;
    println!("User: What is machine learning?");
    println!("Assistant: {}", response1.response.content);

    // Follow-up with memory
    let response2 = query_engine.chat(
        "Can you give me some examples?",
        None
    ).await?;
    println!("\nUser: Can you give me some examples?");
    println!("Assistant: {}", response2.response.content);

    // Query rewriting
    println!("\n🔄 Query Rewriting:");
    let rewritten_response = query_engine.query_with_rewrite(
        "ML applications",
        QueryRewriteStrategy::Expansion,
        None
    ).await?;
    println!("Original: ML applications");
    println!("Rewritten query used for better retrieval");
    println!("Response: {}", rewritten_response.response.content);

    // Deep research
    println!("\n🔬 Deep Research:");
    let research_response = query_engine.deep_research(
        "What are the latest trends in artificial intelligence?",
        Some(3), // 3 research iterations
        None
    ).await?;
    println!("Research Question: What are the latest trends in artificial intelligence?");
    println!("Comprehensive Answer: {}", research_response.response.content);
    println!("Retrieved {} unique sources", research_response.retrieved_nodes.len());

    // 6. Memory Statistics
    println!("\n📊 6. Memory and performance statistics...");
    
    if let Some(memory_stats) = query_engine.get_memory_stats().await? {
        println!("Memory Statistics:");
        println!("  - Total messages: {}", memory_stats.message_count);
        println!("  - Estimated tokens: {}", memory_stats.estimated_tokens);
        println!("  - User messages: {}", memory_stats.user_message_count);
        println!("  - Assistant messages: {}", memory_stats.assistant_message_count);
        println!("  - At capacity: {}", memory_stats.at_capacity);
    }

    let config_stats = config_manager.get_stats();
    println!("Configuration Statistics:");
    println!("  - Total namespaces: {}", config_stats.total_namespaces);
    println!("  - Total files: {}", config_stats.total_files);
    println!("  - Environment overrides: {}", config_stats.env_overrides);
    println!("  - Hot reload enabled: {}", config_stats.hot_reload_enabled);

    // 7. Advanced Storage Operations
    println!("\n💾 7. Demonstrating storage context operations...");
    
    // Store conversation in chat store
    let conversation_messages = {
        let memory_guard = chat_memory.lock().await;
        memory_guard.get_messages().await?
    };

    for message in conversation_messages {
        storage_context.chat_store()
            .unwrap()
            .add_message("example_conversation", message)
            .await?;
    }

    println!("✅ Conversation stored in persistent chat store");

    // Get storage statistics
    let storage_stats = storage_context.get_stats().await?;
    println!("Storage Statistics:");
    println!("  - Documents: {}", storage_stats.document_count);
    println!("  - Indexes: {}", storage_stats.index_count);
    println!("  - Vector entries: {}", storage_stats.vector_count);
    println!("  - Chat messages: {}", storage_stats.chat_message_count);

    println!("\n🎉 Advanced RAG example completed successfully!");
    println!("All features demonstrated:");
    println!("  ✅ Enhanced configuration system with JSON support");
    println!("  ✅ Unified storage context");
    println!("  ✅ Conversation memory with token management");
    println!("  ✅ Advanced RAG features (rewriting, deep research)");
    println!("  ✅ Persistent storage integration");

    Ok(())
}

// Helper functions (simplified implementations)
async fn create_embedder(model_name: &str) -> Result<Arc<dyn cheungfun_core::traits::Embedder>, Box<dyn std::error::Error>> {
    // In a real implementation, you would create the appropriate embedder
    // based on the model name and configuration
    println!("Creating embedder for model: {}", model_name);
    
    #[cfg(feature = "fastembed")]
    {
        use cheungfun_integrations::FastEmbedder;
        let embedder = FastEmbedder::new(model_name).await?;
        Ok(Arc::new(embedder))
    }
    
    #[cfg(not(feature = "fastembed"))]
    {
        // Fallback to a mock embedder for this example
        use cheungfun_core::traits::Embedder;
        use async_trait::async_trait;
        
        #[derive(Debug)]
        struct MockEmbedder;
        
        #[async_trait]
        impl Embedder for MockEmbedder {
            async fn embed(&self, texts: Vec<String>) -> cheungfun_core::Result<Vec<Vec<f32>>> {
                // Return mock embeddings
                Ok(texts.into_iter().map(|_| vec![0.0; 768]).collect())
            }
            
            fn dimension(&self) -> usize { 768 }
            fn name(&self) -> &'static str { "MockEmbedder" }
        }
        
        Ok(Arc::new(MockEmbedder))
    }
}

async fn create_retriever(
    storage_context: Arc<StorageContext>,
    embedder: Arc<dyn cheungfun_core::traits::Embedder>
) -> Result<Arc<dyn cheungfun_core::traits::Retriever>, Box<dyn std::error::Error>> {
    use cheungfun_query::retriever::{VectorRetriever, VectorRetrieverConfig};
    
    let config = VectorRetrieverConfig::default();
    let retriever = VectorRetriever::new(
        storage_context.vector_store(),
        embedder,
        config
    ).await?;
    
    Ok(Arc::new(retriever))
}

async fn create_generator() -> Result<Arc<dyn cheungfun_core::traits::ResponseGenerator>, Box<dyn std::error::Error>> {
    // In a real implementation, you would create a proper generator
    // For this example, we'll use a mock generator
    use cheungfun_core::traits::ResponseGenerator;
    use async_trait::async_trait;
    
    #[derive(Debug)]
    struct MockGenerator;
    
    #[async_trait]
    impl ResponseGenerator for MockGenerator {
        async fn generate_response(
            &self,
            query: &str,
            context: Vec<cheungfun_core::ScoredNode>,
            _options: &cheungfun_core::types::GenerationOptions,
        ) -> cheungfun_core::Result<cheungfun_core::types::GeneratedResponse> {
            use cheungfun_core::types::GeneratedResponse;
            
            let response_content = if context.is_empty() {
                format!("I understand you're asking about: {}. However, I don't have specific context to provide a detailed answer.", query)
            } else {
                format!("Based on {} relevant sources, here's what I can tell you about {}: This is a mock response for demonstration purposes.", context.len(), query)
            };
            
            Ok(GeneratedResponse {
                content: response_content,
                metadata: std::collections::HashMap::new(),
                usage: None,
            })
        }
        
        fn name(&self) -> &'static str { "MockGenerator" }
    }
    
    Ok(Arc::new(MockGenerator))
}
