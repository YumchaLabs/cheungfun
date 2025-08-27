//! KVStore Architecture Demo
//!
//! This example demonstrates the new KVStore-based storage architecture,
//! showing how to use the unified storage abstraction with different backends.

use cheungfun_core::{
    traits::{DocumentStore, IndexStore, ChatStore, KVStore, IndexStruct, DEFAULT_COLLECTION},
    Document, ChatMessage, Result,
};
use cheungfun_integrations::storage::{
    InMemoryKVStore, KVDocumentStore, KVIndexStore, KVChatStore,
};
use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("🚀 Starting KVStore Architecture Demo");

    // Demo 1: Basic KVStore operations
    demo_basic_kvstore().await?;

    // Demo 2: Document store operations
    demo_document_store().await?;

    // Demo 3: Index store operations
    demo_index_store().await?;

    // Demo 4: Chat store operations
    demo_chat_store().await?;

    // Demo 5: Complete storage context
    demo_storage_context().await?;

    info!("✅ All demos completed successfully!");
    Ok(())
}

/// Demonstrate basic KVStore operations
async fn demo_basic_kvstore() -> Result<()> {
    info!("📦 Demo 1: Basic KVStore Operations");

    let kv_store = Arc::new(InMemoryKVStore::new());

    // Basic put/get operations
    let test_data = serde_json::json!({
        "name": "Cheungfun",
        "version": "0.1.0",
        "description": "High-performance RAG framework"
    });

    kv_store.put("project_info", test_data.clone(), DEFAULT_COLLECTION).await?;
    
    let retrieved = kv_store.get("project_info", DEFAULT_COLLECTION).await?;
    assert_eq!(retrieved, Some(test_data));
    info!("✓ Basic put/get operations work");

    // Collection operations
    kv_store.put("key1", serde_json::json!("value1"), "collection1").await?;
    kv_store.put("key2", serde_json::json!("value2"), "collection1").await?;
    kv_store.put("key1", serde_json::json!("different_value"), "collection2").await?;

    let collections = kv_store.list_collections().await?;
    info!("✓ Collections: {:?}", collections);

    let count1 = kv_store.count("collection1").await?;
    let count2 = kv_store.count("collection2").await?;
    info!("✓ Collection counts: collection1={}, collection2={}", count1, count2);

    // Batch operations
    let batch_data = vec![
        ("batch_key1".to_string(), serde_json::json!("batch_value1")),
        ("batch_key2".to_string(), serde_json::json!("batch_value2")),
        ("batch_key3".to_string(), serde_json::json!("batch_value3")),
    ];
    kv_store.put_all(batch_data, "batch_collection").await?;
    
    let batch_count = kv_store.count("batch_collection").await?;
    info!("✓ Batch operations: {} items inserted", batch_count);

    Ok(())
}

/// Demonstrate document store operations
async fn demo_document_store() -> Result<()> {
    info!("📄 Demo 2: Document Store Operations");

    let kv_store = Arc::new(InMemoryKVStore::new());
    let doc_store = KVDocumentStore::new(kv_store, Some("documents".to_string()));

    // Create test documents
    let mut doc1 = Document::new("This is the first document about Rust programming.");
    doc1.id = uuid::Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap();
    doc1.metadata.insert("category".to_string(), serde_json::Value::String("programming".to_string()));
    doc1.metadata.insert("language".to_string(), serde_json::Value::String("rust".to_string()));

    let mut doc2 = Document::new("This is the second document about machine learning.");
    doc2.id = uuid::Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap();
    doc2.metadata.insert("category".to_string(), serde_json::Value::String("ai".to_string()));
    doc2.metadata.insert("topic".to_string(), serde_json::Value::String("ml".to_string()));

    let mut doc3 = Document::new("This is the third document about Rust and AI.");
    doc3.id = uuid::Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap();
    doc3.metadata.insert("category".to_string(), serde_json::Value::String("programming".to_string()));
    doc3.metadata.insert("language".to_string(), serde_json::Value::String("rust".to_string()));
    doc3.metadata.insert("topic".to_string(), serde_json::Value::String("ai".to_string()));

    // Add documents
    let doc_ids = doc_store.add_documents(vec![doc1, doc2, doc3]).await?;
    info!("✓ Added {} documents", doc_ids.len());

    // Retrieve documents
    let doc_id = "00000000-0000-0000-0000-000000000001";
    let retrieved_doc = doc_store.get_document(doc_id).await?;
    assert!(retrieved_doc.is_some());
    info!("✓ Retrieved document: {}", retrieved_doc.unwrap().content);

    // Count documents
    let doc_count = doc_store.count_documents().await?;
    info!("✓ Total documents: {}", doc_count);

    // Filter by metadata
    let mut filter = std::collections::HashMap::new();
    filter.insert("category".to_string(), "programming".to_string());
    let programming_docs = doc_store.get_documents_by_metadata(filter).await?;
    info!("✓ Programming documents: {}", programming_docs.len());

    // Get all documents
    let all_docs = doc_store.get_all_documents().await?;
    info!("✓ All documents retrieved: {}", all_docs.len());

    Ok(())
}

/// Demonstrate index store operations
async fn demo_index_store() -> Result<()> {
    info!("🗂️ Demo 3: Index Store Operations");

    let kv_store = Arc::new(InMemoryKVStore::new());
    let index_store = KVIndexStore::new(kv_store, Some("indexes".to_string()));

    // Create test index
    let index = IndexStruct::new("test_index", "vector");

    // Add index
    let index_id = index.index_id.clone();
    index_store.add_index_struct(index).await?;
    info!("✓ Added index: {}", index_id);

    // Retrieve index
    let retrieved_index = index_store.get_index_struct("test_index").await?;
    assert!(retrieved_index.is_some());
    info!("✓ Retrieved index: {}", retrieved_index.unwrap().index_type);

    // Note: index_exists and count_indexes are not available in the current trait
    // These would require implementing custom existence checking logic

    Ok(())
}

/// Demonstrate chat store operations
async fn demo_chat_store() -> Result<()> {
    info!("💬 Demo 4: Chat Store Operations");

    let kv_store = Arc::new(InMemoryKVStore::new());
    let chat_store = KVChatStore::new(kv_store, Some("conversations".to_string()));

    let conversation_id = "demo_conversation";

    // Add messages
    let messages = vec![
        ChatMessage::user("Hello, I have a question about Rust."),
        ChatMessage::assistant("Hello! I'd be happy to help you with Rust. What would you like to know?"),
        ChatMessage::user("How do I implement a trait for a struct?"),
        ChatMessage::assistant("To implement a trait for a struct, you use the `impl` keyword. Here's the basic syntax:\n\n```rust\nimpl TraitName for StructName {\n    // implement trait methods\n}\n```"),
    ];

    for message in messages {
        chat_store.add_message(conversation_id, message).await?;
    }

    info!("✓ Added messages to conversation");

    // Retrieve messages
    let retrieved_messages = chat_store.get_messages(conversation_id).await?;
    info!("✓ Retrieved {} messages", retrieved_messages.len());

    // Get last messages
    // Get last messages
    let last_messages = chat_store.get_last_messages(conversation_id, 2).await?;
    info!("✓ Last 2 messages retrieved: {}", last_messages.len());

    // Get messages by role
    let user_messages = chat_store.get_messages_by_role(conversation_id, cheungfun_core::MessageRole::User).await?;
    let assistant_messages = chat_store.get_messages_by_role(conversation_id, cheungfun_core::MessageRole::Assistant).await?;
    info!("✓ User messages: {}, Assistant messages: {}", user_messages.len(), assistant_messages.len());

    // List conversations
    let conversations = chat_store.list_conversations().await?;
    info!("✓ Conversations: {:?}", conversations);

    Ok(())
}

/// Demonstrate complete storage context
async fn demo_storage_context() -> Result<()> {
    info!("🏗️ Demo 5: Complete Storage Context");

    // Create shared KV store
    let kv_store = Arc::new(InMemoryKVStore::new());

    // Create specialized stores
    let doc_store = Arc::new(KVDocumentStore::new(kv_store.clone(), Some("documents".to_string())));
    let index_store = Arc::new(KVIndexStore::new(kv_store.clone(), Some("indexes".to_string())));
    let chat_store = Some(Arc::new(KVChatStore::new(kv_store.clone(), Some("conversations".to_string()))));

    // For this demo, we'll use a placeholder vector store
    // In a real application, you'd use InMemoryVectorStore or QdrantVectorStore
    info!("⚠️ Note: Vector store integration requires additional setup");
    info!("✓ Storage context architecture demonstrated");

    // Show statistics
    let doc_stats = doc_store.get_stats().await?;
    let index_stats = index_store.get_stats().await?;
    let chat_stats = chat_store.as_ref().unwrap().get_stats().await?;

    info!("📊 Storage Statistics:");
    info!("  Documents: {} in collection '{}'", doc_stats.document_count, doc_stats.collection_name);
    info!("  Indexes: {} in collection '{}'", index_stats.index_count, index_stats.collection_name);
    info!("  Conversations: {} with {} total messages", chat_stats.conversation_count, chat_stats.total_messages);

    Ok(())
}
