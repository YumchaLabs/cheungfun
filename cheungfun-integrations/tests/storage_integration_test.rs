//! Integration tests for the storage system.

#[cfg(feature = "storage")]
mod storage_tests {
    use cheungfun_core::{
        traits::{ChatStore, DocumentStore, IndexStore, StorageContext},
        ChatMessage, Document, MessageRole,
    };
    use cheungfun_integrations::storage::{
        KVChatStore, KVDocumentStore, KVIndexStore, SqlxStorageConfig,
    };
    use chrono::Utc;
    use std::collections::HashMap;
    use std::sync::Arc;

    async fn setup_test_storage(
    ) -> Result<(KVDocumentStore, KVChatStore, KVIndexStore), Box<dyn std::error::Error>> {
        // Use in-memory SQLite for testing
        let config = SqlxStorageConfig::new("sqlite::memory:").with_table_prefix("test_");

        let kv_store = Arc::new(config.create_kv_store().await?);
        let doc_store = KVDocumentStore::new(kv_store.clone(), None);
        let chat_store = KVChatStore::new(kv_store.clone(), None);
        let index_store = KVIndexStore::new(kv_store, None);

        Ok((doc_store, chat_store, index_store))
    }

    #[tokio::test]
    async fn test_document_store_basic_operations() {
        let (doc_store, _, _) = setup_test_storage().await.unwrap();

        // Create test document
        let mut metadata = HashMap::new();
        metadata.insert(
            "source".to_string(),
            serde_json::Value::String("test".to_string()),
        );

        let doc = Document {
            id: uuid::Uuid::new_v4(),
            content: "This is a test document".to_string(),
            metadata,
            embedding: None,
        };

        // Test add document
        let doc_ids = doc_store.add_documents(vec![doc.clone()]).await.unwrap();
        assert!(!doc_ids.is_empty());
        let doc_id = &doc_ids[0];

        // Test get document
        let retrieved_doc = doc_store.get_document(&doc_id).await.unwrap();
        assert!(retrieved_doc.is_some());
        let retrieved_doc = retrieved_doc.unwrap();
        assert_eq!(retrieved_doc.content, doc.content);

        // Test document exists
        assert!(doc_store.document_exists(&doc_id).await.unwrap());

        // Test delete document
        doc_store.delete_document(&doc_id).await.unwrap();
        assert!(!doc_store.document_exists(&doc_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_chat_store_basic_operations() {
        let (_, chat_store, _) = setup_test_storage().await.unwrap();

        let conversation_key = "test_conversation";

        // Create test messages
        let user_message = ChatMessage {
            role: MessageRole::User,
            content: "Hello, how are you?".to_string(),
            timestamp: Utc::now(),
            metadata: None,
        };

        let assistant_message = ChatMessage {
            role: MessageRole::Assistant,
            content: "I'm doing well, thank you!".to_string(),
            timestamp: Utc::now(),
            metadata: None,
        };

        // Test add messages
        chat_store
            .add_message(conversation_key, user_message.clone())
            .await
            .unwrap();
        chat_store
            .add_message(conversation_key, assistant_message.clone())
            .await
            .unwrap();

        // Test get messages
        let messages = chat_store.get_messages(conversation_key).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, user_message.content);
        assert_eq!(messages[1].content, assistant_message.content);

        // Test get keys
        let keys = chat_store.get_keys().await.unwrap();
        assert!(keys.contains(&conversation_key.to_string()));

        // Test delete messages
        chat_store.delete_messages(conversation_key).await.unwrap();
        let messages = chat_store.get_messages(conversation_key).await.unwrap();
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn test_index_store_basic_operations() {
        let (_, _, index_store) = setup_test_storage().await.unwrap();

        // Create test index struct
        let mut config = std::collections::HashMap::new();
        config.insert("dimension".to_string(), serde_json::json!(768));

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("created_by".to_string(), serde_json::json!("test"));

        let node_id1 = uuid::Uuid::new_v4();
        let node_id2 = uuid::Uuid::new_v4();

        let index_struct = cheungfun_core::traits::IndexStruct {
            index_id: "test_index".to_string(),
            index_type: "vector_index".to_string(),
            config,
            node_ids: vec![node_id1, node_id2],
            metadata,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        // Test add index struct
        index_store
            .add_index_struct(index_struct.clone())
            .await
            .unwrap();

        // Test get index struct
        let retrieved_struct = index_store
            .get_index_struct(&index_struct.index_id)
            .await
            .unwrap();
        assert!(retrieved_struct.is_some());
        let retrieved_struct = retrieved_struct.unwrap();
        assert_eq!(retrieved_struct.index_id, index_struct.index_id);
        assert_eq!(retrieved_struct.index_type, index_struct.index_type);

        // Test list index structs
        let index_ids = index_store.list_index_structs().await.unwrap();
        assert!(index_ids.contains(&index_struct.index_id));

        // Test delete index struct
        index_store
            .delete_index_struct(&index_struct.index_id)
            .await
            .unwrap();
        let retrieved_struct = index_store
            .get_index_struct(&index_struct.index_id)
            .await
            .unwrap();
        assert!(retrieved_struct.is_none());
    }

    #[tokio::test]
    async fn test_storage_context_integration() {
        let (doc_store, chat_store, index_store) = setup_test_storage().await.unwrap();

        // Create a mock vector store for testing
        let vector_store = Arc::new(cheungfun_integrations::InMemoryVectorStore::new(
            768,
            cheungfun_core::DistanceMetric::Cosine,
        ));

        // Create storage context
        let storage_context = StorageContext::new(
            Arc::new(doc_store),
            Arc::new(index_store),
            vector_store,
            Some(Arc::new(chat_store)),
            None, // No graph store
        );

        // Test that all stores are accessible
        // StorageContext has direct field access, not methods
        // Just verify the stores exist (they're Arc<dyn Trait> so always valid)
        assert!(storage_context.chat_store.is_some());

        // Test storage statistics
        let stats = storage_context.get_stats().await.unwrap();
        assert_eq!(stats.doc_count, 0);
        assert_eq!(stats.index_count, 0);
        assert_eq!(stats.conversation_count, 0);
        // Vector stats are in vector_stats field
        assert_eq!(stats.vector_stats.total_nodes, 0);
    }

    #[tokio::test]
    async fn test_storage_error_handling() {
        // Test with invalid database URL
        let invalid_config = SqlxStorageConfig::new("invalid://url");

        let result = invalid_config.create_kv_store().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_concurrent_storage_operations() {
        let (doc_store, chat_store, _) = setup_test_storage().await.unwrap();
        let doc_store: Arc<dyn DocumentStore> = Arc::new(doc_store);
        let chat_store: Arc<dyn ChatStore> = Arc::new(chat_store);

        // Test concurrent document operations
        let mut handles = Vec::new();

        for i in 0..10 {
            let doc_store_clone = Arc::clone(&doc_store);
            let handle = tokio::spawn(async move {
                let mut metadata = HashMap::new();
                metadata.insert("index".to_string(), serde_json::Value::Number(i.into()));

                let doc = Document {
                    id: uuid::Uuid::new_v4(),
                    content: format!("Test document {}", i),
                    metadata,
                    embedding: None,
                };

                doc_store_clone
                    .add_documents(vec![doc])
                    .await
                    .map(|ids| ids[0].clone())
            });
            handles.push(handle);
        }

        // Wait for all operations to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }

        // Test concurrent chat operations
        let mut chat_handles = Vec::new();

        for i in 0..10 {
            let chat_store_clone = Arc::clone(&chat_store);
            let handle = tokio::spawn(async move {
                let message = ChatMessage {
                    role: MessageRole::User,
                    content: format!("Message {}", i),
                    timestamp: Utc::now(),
                    metadata: None,
                };

                chat_store_clone
                    .add_message(&format!("conversation_{}", i), message)
                    .await
            });
            chat_handles.push(handle);
        }

        // Wait for all chat operations to complete
        for handle in chat_handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_storage_performance() {
        let (doc_store, _, _) = setup_test_storage().await.unwrap();

        let start = std::time::Instant::now();

        // Add 100 documents
        for i in 0..100 {
            let mut metadata = HashMap::new();
            metadata.insert(
                "batch".to_string(),
                serde_json::Value::String("performance_test".to_string()),
            );

            let doc = Document {
                id: uuid::Uuid::new_v4(),
                content: format!("Performance test document {}", i),
                metadata,
                embedding: None,
            };

            doc_store.add_documents(vec![doc]).await.unwrap();
        }

        let duration = start.elapsed();
        println!("Added 100 documents in {:?}", duration);

        // Should be reasonably fast (less than 5 seconds for 100 documents)
        assert!(duration.as_secs() < 5);
    }
}

// Tests that require the storage feature
#[cfg(feature = "storage")]
#[tokio::test]
async fn test_storage_config_creation() {
    use cheungfun_integrations::SqlxStorageConfig;

    let config = SqlxStorageConfig::new("sqlite::memory:")
        .with_table_prefix("test_")
        .with_max_connections(5);

    // Test that config is created correctly
    assert_eq!(config.table_prefix, "test_");
    assert_eq!(config.max_connections, 5);
}
