//! Key-value based chat store implementation.

use async_trait::async_trait;
use cheungfun_core::{
    traits::{ChatStore, KVStore, ChatStoreStats},
    ChatMessage, MessageRole, Result, CheungfunError,
};
use serde_json::Value;

use std::sync::Arc;
use tracing::{debug, error, info};

/// Chat store implementation based on KVStore.
///
/// This implementation uses a KVStore backend to persist chat messages
/// and conversation history. Each conversation is stored as a collection
/// of messages with the conversation ID as the collection name.
///
/// # Examples
///
/// ```rust
/// use cheungfun_integrations::storage::{kvstore::InMemoryKVStore, chat_store::KVChatStore};
/// use cheungfun_core::{traits::ChatStore, ChatMessage};
/// use std::sync::Arc;
///
/// # tokio_test::block_on(async {
/// let kv_store = Arc::new(InMemoryKVStore::new());
/// let chat_store = KVChatStore::new(kv_store, Some("conversations".to_string()));
/// 
/// let message = ChatMessage::user("Hello, world!");
/// chat_store.add_message("conv1", message).await.unwrap();
/// # });
/// ```
#[derive(Debug)]
pub struct KVChatStore {
    /// Underlying KV store for persistence.
    kv_store: Arc<dyn KVStore>,
    /// Base collection name for chat storage.
    base_collection: String,
}

impl KVChatStore {
    /// Create a new KV-based chat store.
    ///
    /// # Arguments
    ///
    /// * `kv_store` - The underlying KV store implementation
    /// * `base_collection` - Optional base collection name (defaults to "conversations")
    pub fn new(kv_store: Arc<dyn KVStore>, base_collection: Option<String>) -> Self {
        let base_collection = base_collection.unwrap_or_else(|| "conversations".to_string());
        info!("Created KV chat store with base collection '{}'", base_collection);
        
        Self {
            kv_store,
            base_collection,
        }
    }

    /// Get the base collection name used by this store.
    pub fn base_collection(&self) -> &str {
        &self.base_collection
    }

    /// Get a reference to the underlying KV store.
    pub fn kv_store(&self) -> &Arc<dyn KVStore> {
        &self.kv_store
    }

    /// Get the collection name for a specific conversation.
    fn conversation_collection(&self, conversation_id: &str) -> String {
        format!("{}_{}", self.base_collection, conversation_id)
    }

    /// Convert a ChatMessage to a JSON value for storage.
    fn message_to_value(&self, message: &ChatMessage) -> Result<Value> {
        serde_json::to_value(message)
            .map_err(|e| CheungfunError::Serialization(e))
    }

    /// Convert a JSON value back to a ChatMessage.
    fn value_to_message(&self, value: Value) -> Result<ChatMessage> {
        serde_json::from_value(value)
            .map_err(|e| CheungfunError::Serialization(e))
    }

    /// Generate a message key based on timestamp and index.
    fn generate_message_key(&self, message: &ChatMessage, index: usize) -> String {
        format!("{}_{:06}", message.timestamp.timestamp_millis(), index)
    }

    /// Get storage statistics for this chat store.
    pub async fn get_stats(&self) -> Result<ChatStoreStats> {
        let collections = self.kv_store.list_collections().await?;
        let conversation_collections: Vec<_> = collections
            .iter()
            .filter(|c| c.starts_with(&format!("{}_", self.base_collection)))
            .collect();
        
        let mut total_messages = 0;
        for collection in &conversation_collections {
            total_messages += self.kv_store.count(collection).await?;
        }
        
        Ok(ChatStoreStats {
            conversation_count: conversation_collections.len(),
            total_messages,
            collection_name: self.base_collection.clone(),
        })
    }

    /// Check if the chat store is healthy.
    pub async fn health_check(&self) -> Result<bool> {
        match self.kv_store.list_collections().await {
            Ok(_) => Ok(true),
            Err(e) => {
                error!("Chat store health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

#[async_trait]
impl ChatStore for KVChatStore {
    async fn add_message(&self, conversation_id: &str, message: ChatMessage) -> Result<()> {
        let collection = self.conversation_collection(conversation_id);
        
        // Get current message count to generate unique key
        let current_count = self.kv_store.count(&collection).await?;
        let message_key = self.generate_message_key(&message, current_count);
        let message_value = self.message_to_value(&message)?;
        
        self.kv_store.put(&message_key, message_value, &collection).await?;
        
        debug!("Added message to conversation '{}' in collection '{}'", conversation_id, collection);
        Ok(())
    }

    async fn get_messages(&self, conversation_id: &str) -> Result<Vec<ChatMessage>> {
        let collection = self.conversation_collection(conversation_id);
        let all_data = self.kv_store.get_all(&collection).await?;
        
        let mut messages = Vec::with_capacity(all_data.len());
        let mut message_pairs: Vec<(String, Value)> = all_data.into_iter().collect();
        
        // Sort by key to maintain chronological order
        message_pairs.sort_by(|a, b| a.0.cmp(&b.0));
        
        for (message_key, value) in message_pairs {
            match self.value_to_message(value) {
                Ok(message) => messages.push(message),
                Err(e) => {
                    error!("Failed to deserialize message '{}' in conversation '{}': {}", 
                           message_key, conversation_id, e);
                    // Continue processing other messages
                }
            }
        }
        
        debug!("Retrieved {} messages from conversation '{}'", messages.len(), conversation_id);
        Ok(messages)
    }

    async fn get_keys(&self) -> Result<Vec<String>> {
        let collections = self.kv_store.list_collections().await?;
        let prefix = format!("{}_", self.base_collection);

        let conversation_ids: Vec<String> = collections
            .into_iter()
            .filter_map(|collection| {
                if collection.starts_with(&prefix) {
                    Some(collection[prefix.len()..].to_string())
                } else {
                    None
                }
            })
            .collect();

        debug!("Listed {} conversations", conversation_ids.len());
        Ok(conversation_ids)
    }

    async fn get_messages_by_role(&self, key: &str, role: MessageRole) -> Result<Vec<ChatMessage>> {
        let all_messages = self.get_messages(key).await?;
        let filtered_messages: Vec<ChatMessage> = all_messages
            .into_iter()
            .filter(|message| message.role == role)
            .collect();

        debug!("Retrieved {} messages with role '{:?}' from conversation '{}'",
               filtered_messages.len(), role, key);
        Ok(filtered_messages)
    }

    async fn get_stats(&self) -> Result<ChatStoreStats> {
        let conversation_keys = self.get_keys().await?;
        let conversation_count = conversation_keys.len();

        let mut total_messages = 0;
        for key in &conversation_keys {
            total_messages += self.count_messages(key).await?;
        }

        Ok(ChatStoreStats {
            conversation_count,
            total_messages,
            collection_name: self.base_collection.clone(),
        })
    }

    async fn set_messages(&self, conversation_id: &str, messages: Vec<ChatMessage>) -> Result<()> {
        // Clear existing messages first
        self.delete_messages(conversation_id).await?;

        // Add new messages
        for message in messages {
            self.add_message(conversation_id, message).await?;
        }

        Ok(())
    }

    async fn delete_messages(&self, conversation_id: &str) -> Result<()> {
        let collection = self.conversation_collection(conversation_id);
        self.kv_store.delete_collection(&collection).await?;
        debug!("Deleted all messages for conversation '{}'", conversation_id);
        Ok(())
    }




}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::kvstore::InMemoryKVStore;

    async fn create_test_store() -> KVChatStore {
        let kv_store = Arc::new(InMemoryKVStore::new());
        KVChatStore::new(kv_store, Some("test_conversations".to_string()))
    }

    #[tokio::test]
    async fn test_add_and_get_messages() {
        let store = create_test_store().await;
        let conversation_id = "conv1";

        let message1 = ChatMessage::user("Hello");
        let message2 = ChatMessage::assistant("Hi there!");

        // Add messages
        store.add_message(conversation_id, message1.clone()).await.unwrap();
        store.add_message(conversation_id, message2.clone()).await.unwrap();

        // Get messages
        let messages = store.get_messages(conversation_id).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "Hello");
        assert_eq!(messages[1].content, "Hi there!");
    }

    #[tokio::test]
    async fn test_conversation_operations() {
        let store = create_test_store().await;
        let conversation_id = "conv1";

        // Check non-existent conversation
        assert!(!store.conversation_exists(conversation_id).await.unwrap());

        // Add a message
        let message = ChatMessage::user("Test message");
        store.add_message(conversation_id, message).await.unwrap();

        // Check existence
        assert!(store.conversation_exists(conversation_id).await.unwrap());

        // Count messages
        let count = store.count_messages(conversation_id).await.unwrap();
        assert_eq!(count, 1);

        // Delete conversation
        let deleted = store.delete_conversation(conversation_id).await.unwrap();
        assert!(deleted);

        // Check it's gone
        assert!(!store.conversation_exists(conversation_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let store = create_test_store().await;
        let conversation_id = "conv1";

        let messages = vec![
            ChatMessage::user("Message 1"),
            ChatMessage::assistant("Response 1"),
            ChatMessage::user("Message 2"),
        ];

        // Add multiple messages
        store.add_messages(conversation_id, messages).await.unwrap();

        // Check count
        let count = store.count_messages(conversation_id).await.unwrap();
        assert_eq!(count, 3);

        // Get last messages
        let last_messages = store.get_last_messages(conversation_id, 2).await.unwrap();
        assert_eq!(last_messages.len(), 2);
        assert_eq!(last_messages[0].content, "Response 1");
        assert_eq!(last_messages[1].content, "Message 2");
    }

    #[tokio::test]
    async fn test_multiple_conversations() {
        let store = create_test_store().await;

        // Add messages to different conversations
        store.add_message("conv1", ChatMessage::user("Hello from conv1")).await.unwrap();
        store.add_message("conv2", ChatMessage::user("Hello from conv2")).await.unwrap();
        store.add_message("conv1", ChatMessage::assistant("Response to conv1")).await.unwrap();

        // List conversations
        let conversations = store.list_conversations().await.unwrap();
        assert_eq!(conversations.len(), 2);
        assert!(conversations.contains(&"conv1".to_string()));
        assert!(conversations.contains(&"conv2".to_string()));

        // Check message counts
        assert_eq!(store.count_messages("conv1").await.unwrap(), 2);
        assert_eq!(store.count_messages("conv2").await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_messages_by_role() {
        let store = create_test_store().await;
        let conversation_id = "conv1";

        let messages = vec![
            ChatMessage::user("User message 1"),
            ChatMessage::assistant("Assistant response 1"),
            ChatMessage::user("User message 2"),
            ChatMessage::assistant("Assistant response 2"),
        ];

        store.add_messages(conversation_id, messages).await.unwrap();

        // Get user messages
        let user_messages = store.get_messages_by_role(conversation_id, "user").await.unwrap();
        assert_eq!(user_messages.len(), 2);
        assert_eq!(user_messages[0].content, "User message 1");
        assert_eq!(user_messages[1].content, "User message 2");

        // Get assistant messages
        let assistant_messages = store.get_messages_by_role(conversation_id, "assistant").await.unwrap();
        assert_eq!(assistant_messages.len(), 2);
        assert_eq!(assistant_messages[0].content, "Assistant response 1");
        assert_eq!(assistant_messages[1].content, "Assistant response 2");
    }
}
