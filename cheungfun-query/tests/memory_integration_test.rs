//! Integration tests for the memory system.

use cheungfun_core::{
    traits::{ApproximateTokenCounter, BaseMemory, TokenCounter},
    ChatMessage, MessageRole,
};
use cheungfun_query::memory::{ChatMemoryBuffer, ChatMemoryConfig};
use chrono::Utc;

#[tokio::test]
async fn test_chat_memory_buffer_basic_operations() {
    let config = ChatMemoryConfig::with_token_limit(1000);
    let mut memory = ChatMemoryBuffer::new(config);

    // Test empty memory
    assert!(memory.is_empty().await.unwrap());
    assert_eq!(memory.message_count().await.unwrap(), 0);

    // Add a user message
    let user_message = ChatMessage {
        role: MessageRole::User,
        content: "Hello, how are you?".to_string(),
        timestamp: Utc::now(),
        metadata: None,
    };

    memory.add_message(user_message.clone()).await.unwrap();

    // Test memory state
    assert!(!memory.is_empty().await.unwrap());
    assert_eq!(memory.message_count().await.unwrap(), 1);

    // Get messages
    let messages = memory.get_messages().await.unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].content, user_message.content);
    assert_eq!(messages[0].role, MessageRole::User);

    // Add assistant message
    let assistant_message = ChatMessage {
        role: MessageRole::Assistant,
        content: "I'm doing well, thank you for asking!".to_string(),
        timestamp: Utc::now(),
        metadata: None,
    };

    memory.add_message(assistant_message.clone()).await.unwrap();

    // Test updated state
    assert_eq!(memory.message_count().await.unwrap(), 2);

    let messages = memory.get_messages().await.unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1].content, assistant_message.content);
}

#[tokio::test]
async fn test_chat_memory_token_limits() {
    // Create memory with very small token limit
    let config = ChatMemoryConfig::with_token_limit(50); // Very small limit
    let mut memory = ChatMemoryBuffer::new(config);

    // Add several messages that should exceed the limit
    for i in 0..10 {
        let message = ChatMessage {
            role: if i % 2 == 0 { MessageRole::User } else { MessageRole::Assistant },
            content: format!("This is a longer message number {} that contains multiple words and should consume several tokens when counted by the approximate token counter.", i),
            timestamp: Utc::now(),
            metadata: None,
        };

        memory.add_message(message).await.unwrap();
    }

    // Memory should have been truncated
    let messages = memory.get_messages().await.unwrap();
    assert!(
        messages.len() < 10,
        "Memory should have been truncated due to token limit"
    );

    // Check memory statistics
    let stats = memory.stats();
    println!("Token count: {}, limit: 50", stats.estimated_tokens);
    println!("Messages remaining: {}", messages.len());
    println!("Truncated count: {}", stats.last_truncated_count);

    // The token limit should be approximately respected (allow some tolerance)
    assert!(
        stats.estimated_tokens <= 100,
        "Token count should be reasonably close to limit (got {})",
        stats.estimated_tokens
    );
    assert!(
        stats.last_truncated_count > 0,
        "Some messages should have been truncated"
    );
}

#[tokio::test]
async fn test_chat_memory_message_limits() {
    let config = ChatMemoryConfig::with_message_limit(3);
    let mut memory = ChatMemoryBuffer::new(config);

    // Add more messages than the limit
    for i in 0..5 {
        let message = ChatMessage {
            role: MessageRole::User,
            content: format!("Message {}", i),
            timestamp: Utc::now(),
            metadata: None,
        };

        memory.add_message(message).await.unwrap();
    }

    // Should only keep the last 3 messages
    let messages = memory.get_messages().await.unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].content, "Message 2");
    assert_eq!(messages[1].content, "Message 3");
    assert_eq!(messages[2].content, "Message 4");
}

#[tokio::test]
async fn test_chat_memory_preservation_rules() {
    let config = ChatMemoryConfig::with_token_limit(100)
        .with_token_counter(cheungfun_core::traits::TokenCountingMethod::Approximate);
    let mut memory = ChatMemoryBuffer::new(config);

    // Add system message
    let system_message = ChatMessage {
        role: MessageRole::System,
        content: "You are a helpful assistant.".to_string(),
        timestamp: Utc::now(),
        metadata: None,
    };
    memory.add_message(system_message.clone()).await.unwrap();

    // Add many user/assistant pairs to trigger truncation
    for i in 0..10 {
        let user_message = ChatMessage {
            role: MessageRole::User,
            content: format!(
                "User message {} with some additional content to increase token count",
                i
            ),
            timestamp: Utc::now(),
            metadata: None,
        };

        let assistant_message = ChatMessage {
            role: MessageRole::Assistant,
            content: format!(
                "Assistant response {} with detailed explanation and additional content",
                i
            ),
            timestamp: Utc::now(),
            metadata: None,
        };

        memory.add_message(user_message).await.unwrap();
        memory.add_message(assistant_message).await.unwrap();
    }

    let messages = memory.get_messages().await.unwrap();

    // System message should be preserved if configured to do so
    let has_system_message = messages.iter().any(|m| m.role == MessageRole::System);
    assert!(has_system_message, "System message should be preserved");

    // Should have recent user-assistant pairs
    let user_count = messages
        .iter()
        .filter(|m| m.role == MessageRole::User)
        .count();
    let assistant_count = messages
        .iter()
        .filter(|m| m.role == MessageRole::Assistant)
        .count();
    assert!(user_count > 0, "Should have at least one user message");
    assert!(
        assistant_count > 0,
        "Should have at least one assistant message"
    );
}

#[tokio::test]
async fn test_chat_memory_variables() {
    let config = ChatMemoryConfig::with_token_limit(1000);
    let mut memory = ChatMemoryBuffer::new(config);

    // Add some messages
    let user_message = ChatMessage {
        role: MessageRole::User,
        content: "What is machine learning?".to_string(),
        timestamp: Utc::now(),
        metadata: None,
    };

    let assistant_message = ChatMessage {
        role: MessageRole::Assistant,
        content: "Machine learning is a subset of artificial intelligence...".to_string(),
        timestamp: Utc::now(),
        metadata: None,
    };

    memory.add_message(user_message.clone()).await.unwrap();
    memory.add_message(assistant_message.clone()).await.unwrap();

    // Get memory variables
    let variables = memory.get_memory_variables().await.unwrap();

    assert!(variables.contains_key("message_count"));
    assert!(variables.contains_key("estimated_tokens"));
    assert!(variables.contains_key("user_messages"));
    assert!(variables.contains_key("assistant_messages"));
    assert!(variables.contains_key("last_user_message"));
    assert!(variables.contains_key("last_assistant_message"));

    assert_eq!(variables.get("message_count").unwrap(), "2");
    assert_eq!(variables.get("user_messages").unwrap(), "1");
    assert_eq!(variables.get("assistant_messages").unwrap(), "1");
    assert_eq!(
        variables.get("last_user_message").unwrap(),
        &user_message.content
    );
    assert_eq!(
        variables.get("last_assistant_message").unwrap(),
        &assistant_message.content
    );
}

#[tokio::test]
async fn test_chat_memory_recent_messages() {
    let config = ChatMemoryConfig::with_token_limit(1000);
    let mut memory = ChatMemoryBuffer::new(config);

    // Add 10 messages
    for i in 0..10 {
        let message = ChatMessage {
            role: MessageRole::User,
            content: format!("Message {}", i),
            timestamp: Utc::now(),
            metadata: None,
        };
        memory.add_message(message).await.unwrap();
    }

    // Get recent messages
    let recent_3 = memory.get_recent_messages(3).await.unwrap();
    assert_eq!(recent_3.len(), 3);
    assert_eq!(recent_3[0].content, "Message 7");
    assert_eq!(recent_3[1].content, "Message 8");
    assert_eq!(recent_3[2].content, "Message 9");

    let recent_5 = memory.get_recent_messages(5).await.unwrap();
    assert_eq!(recent_5.len(), 5);
    assert_eq!(recent_5[0].content, "Message 5");
    assert_eq!(recent_5[4].content, "Message 9");
}

#[tokio::test]
async fn test_chat_memory_pop_message() {
    let config = ChatMemoryConfig::with_token_limit(1000);
    let mut memory = ChatMemoryBuffer::new(config);

    // Add messages
    let message1 = ChatMessage {
        role: MessageRole::User,
        content: "First message".to_string(),
        timestamp: Utc::now(),
        metadata: None,
    };

    let message2 = ChatMessage {
        role: MessageRole::User,
        content: "Second message".to_string(),
        timestamp: Utc::now(),
        metadata: None,
    };

    memory.add_message(message1).await.unwrap();
    memory.add_message(message2.clone()).await.unwrap();

    assert_eq!(memory.message_count().await.unwrap(), 2);

    // Pop the last message
    let popped = memory.pop_message().await.unwrap();
    assert!(popped.is_some());
    assert_eq!(popped.unwrap().content, message2.content);
    assert_eq!(memory.message_count().await.unwrap(), 1);

    // Pop from empty memory
    memory.clear().await.unwrap();
    let popped_empty = memory.pop_message().await.unwrap();
    assert!(popped_empty.is_none());
}

#[tokio::test]
async fn test_chat_memory_clear() {
    let config = ChatMemoryConfig::with_token_limit(1000);
    let mut memory = ChatMemoryBuffer::new(config);

    // Add messages
    for i in 0..5 {
        let message = ChatMessage {
            role: MessageRole::User,
            content: format!("Message {}", i),
            timestamp: Utc::now(),
            metadata: None,
        };
        memory.add_message(message).await.unwrap();
    }

    assert_eq!(memory.message_count().await.unwrap(), 5);

    // Clear memory
    memory.clear().await.unwrap();

    assert_eq!(memory.message_count().await.unwrap(), 0);
    assert!(memory.is_empty().await.unwrap());
    let messages = memory.get_messages().await.unwrap();
    assert!(messages.is_empty());
}

#[tokio::test]
async fn test_approximate_token_counter() {
    let counter = ApproximateTokenCounter::default();

    let message = ChatMessage {
        role: MessageRole::User,
        content: "This is a test message with multiple words".to_string(),
        timestamp: Utc::now(),
        metadata: None,
    };

    let token_count = counter.count_message_tokens(&message).await.unwrap();
    assert!(token_count > 0, "Token count should be greater than 0");

    // Test text token counting
    let text_tokens = counter.count_text_tokens("Hello world").await.unwrap();
    assert!(text_tokens > 0, "Text token count should be greater than 0");

    // Longer text should have more tokens
    let long_text_tokens = counter.count_text_tokens("This is a much longer text with many more words and should result in a higher token count").await.unwrap();
    assert!(
        long_text_tokens > text_tokens,
        "Longer text should have more tokens"
    );
}

#[tokio::test]
async fn test_chat_memory_conversation_patterns() {
    let config = ChatMemoryConfig::with_token_limit(1000);
    let mut memory = ChatMemoryBuffer::new(config);

    // Add alternating user-assistant messages
    let messages = vec![
        (MessageRole::System, "You are a helpful assistant."),
        (MessageRole::User, "Hello"),
        (MessageRole::Assistant, "Hi there!"),
        (MessageRole::User, "How are you?"),
        (MessageRole::Assistant, "I'm doing well, thanks!"),
    ];

    for (role, content) in messages {
        let message = ChatMessage {
            role,
            content: content.to_string(),
            timestamp: Utc::now(),
            metadata: None,
        };
        memory.add_message(message).await.unwrap();
    }

    // Test conversation pattern detection
    let has_pattern = memory.has_conversation_pattern().await.unwrap();
    assert!(
        has_pattern,
        "Should detect proper user-assistant conversation pattern"
    );

    // Test getting messages by role
    let user_messages = memory
        .get_messages_by_role(MessageRole::User)
        .await
        .unwrap();
    assert_eq!(user_messages.len(), 2);

    let assistant_messages = memory
        .get_messages_by_role(MessageRole::Assistant)
        .await
        .unwrap();
    assert_eq!(assistant_messages.len(), 2);

    let system_messages = memory
        .get_messages_by_role(MessageRole::System)
        .await
        .unwrap();
    assert_eq!(system_messages.len(), 1);

    // Test getting last message by role
    let last_user = memory
        .get_last_message_by_role(MessageRole::User)
        .await
        .unwrap();
    assert!(last_user.is_some());
    assert_eq!(last_user.unwrap().content, "How are you?");

    let last_assistant = memory
        .get_last_message_by_role(MessageRole::Assistant)
        .await
        .unwrap();
    assert!(last_assistant.is_some());
    assert_eq!(last_assistant.unwrap().content, "I'm doing well, thanks!");
}
