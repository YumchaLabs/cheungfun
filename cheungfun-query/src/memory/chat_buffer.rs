//! Chat memory buffer implementation with token limits and intelligent truncation.

use async_trait::async_trait;
use cheungfun_core::{
    traits::{ApproximateTokenCounter, BaseMemory, ChatMemoryConfig, MemoryStats, TokenCounter},
    ChatMessage, MessageRole, Result,
};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Chat memory buffer with token limits and intelligent truncation.
///
/// This implementation provides conversation memory management with configurable
/// token limits, message limits, and intelligent truncation strategies that
/// preserve important context while staying within model constraints.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_query::memory::{ChatMemoryBuffer, ChatMemoryConfig};
/// use cheungfun_core::{ChatMessage, MessageRole};
/// use chrono::Utc;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = ChatMemoryConfig::with_token_limit(2000);
///     let mut memory = ChatMemoryBuffer::new(config);
///
///     // Add messages to memory
///     let user_msg = ChatMessage {
///         role: MessageRole::User,
///         content: "Hello, how are you?".to_string(),
///         timestamp: Utc::now(),
///         metadata: None,
///     };
///     memory.add_message(user_msg).await?;
///
///     let assistant_msg = ChatMessage {
///         role: MessageRole::Assistant,
///         content: "I'm doing well, thank you for asking!".to_string(),
///         timestamp: Utc::now(),
///         metadata: None,
///     };
///     memory.add_message(assistant_msg).await?;
///
///     // Get conversation history
///     let messages = memory.get_messages().await?;
///     println!("Conversation has {} messages", messages.len());
///
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct ChatMemoryBuffer {
    /// Conversation messages in chronological order.
    messages: Vec<ChatMessage>,

    /// Memory configuration.
    config: ChatMemoryConfig,

    /// Token counter for estimating message sizes.
    token_counter: Arc<dyn TokenCounter>,

    /// Memory statistics.
    stats: MemoryStats,
}

impl ChatMemoryBuffer {
    /// Create a new chat memory buffer with default configuration.
    pub fn new(config: ChatMemoryConfig) -> Self {
        let token_counter: Arc<dyn TokenCounter> = match &config.token_counter {
            cheungfun_core::traits::TokenCountingMethod::Approximate => {
                Arc::new(ApproximateTokenCounter::default())
            }
            cheungfun_core::traits::TokenCountingMethod::Custom => {
                // For now, fall back to approximate
                Arc::new(ApproximateTokenCounter::default())
            }
        };

        Self {
            messages: Vec::new(),
            config,
            token_counter,
            stats: MemoryStats::empty(),
        }
    }

    /// Create a new chat memory buffer with custom token counter.
    pub fn with_token_counter(
        config: ChatMemoryConfig,
        token_counter: Arc<dyn TokenCounter>,
    ) -> Self {
        Self {
            messages: Vec::new(),
            config,
            token_counter,
            stats: MemoryStats::empty(),
        }
    }

    /// Get memory statistics.
    pub fn stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Update memory statistics.
    async fn update_stats(&mut self) -> Result<()> {
        self.stats =
            MemoryStats::from_messages(&self.messages, self.token_counter.as_ref()).await?;

        // Check if we're at capacity
        let at_token_capacity = if let Some(max_tokens) = self.config.max_tokens {
            self.stats.estimated_tokens >= max_tokens
        } else {
            false
        };

        let at_message_capacity = if let Some(max_messages) = self.config.max_messages {
            self.stats.message_count >= max_messages
        } else {
            false
        };

        self.stats.at_capacity = at_token_capacity || at_message_capacity;

        Ok(())
    }

    /// Truncate messages to fit within configured limits.
    async fn truncate_if_needed(&mut self) -> Result<()> {
        let mut truncated_count = 0;

        // Check token limit
        if let Some(max_tokens) = self.config.max_tokens {
            while self.stats.estimated_tokens > max_tokens && self.messages.len() > 1 {
                if let Some(removed) = self.remove_oldest_truncatable_message().await? {
                    truncated_count += 1;
                    debug!("Truncated message due to token limit: {:?}", removed.role);
                } else {
                    break; // No more messages can be safely removed
                }
                self.update_stats().await?;
            }
        }

        // Check message limit
        if let Some(max_messages) = self.config.max_messages {
            while self.messages.len() > max_messages {
                if let Some(removed) = self.remove_oldest_truncatable_message().await? {
                    truncated_count += 1;
                    debug!("Truncated message due to message limit: {:?}", removed.role);
                } else {
                    break; // No more messages can be safely removed
                }
            }
        }

        if truncated_count > 0 {
            info!("Truncated {} messages from memory", truncated_count);
            // Update stats first, then set the truncated count
            self.update_stats().await?;
            self.stats.last_truncated_count = truncated_count;
        }

        Ok(())
    }

    /// Remove the oldest message that can be safely truncated.
    async fn remove_oldest_truncatable_message(&mut self) -> Result<Option<ChatMessage>> {
        if self.messages.is_empty() {
            return Ok(None);
        }

        // Find the first message that can be removed based on configuration
        let mut remove_index = None;

        for (i, message) in self.messages.iter().enumerate() {
            let can_remove = match message.role {
                MessageRole::System => !self.config.preserve_system_messages,
                MessageRole::User | MessageRole::Assistant | MessageRole::Tool => {
                    if self.config.preserve_recent_pair && self.messages.len() >= 2 {
                        // Don't remove the last user-assistant pair
                        let is_in_recent_pair = i >= self.messages.len().saturating_sub(2);
                        !is_in_recent_pair
                    } else {
                        true
                    }
                }
            };

            if can_remove {
                remove_index = Some(i);
                break;
            }
        }

        if let Some(index) = remove_index {
            Ok(Some(self.messages.remove(index)))
        } else {
            warn!("Cannot truncate any more messages due to preservation rules");
            Ok(None)
        }
    }

    /// Get messages formatted for LLM consumption.
    pub async fn get_formatted_messages(&self) -> Result<Vec<ChatMessage>> {
        // For now, just return the messages as-is
        // In the future, this could apply formatting, add system prompts, etc.
        Ok(self.messages.clone())
    }

    /// Get conversation summary if available.
    pub async fn get_summary(&self) -> Result<Option<String>> {
        // TODO: Implement conversation summarization
        // This would use an LLM to summarize truncated messages
        Ok(None)
    }

    /// Reset memory to initial state.
    pub async fn reset(&mut self) -> Result<()> {
        self.messages.clear();
        self.stats = MemoryStats::empty();
        info!("Reset chat memory buffer");
        Ok(())
    }

    /// Get the last message of a specific role.
    pub async fn get_last_message_by_role(&self, role: MessageRole) -> Result<Option<ChatMessage>> {
        Ok(self
            .messages
            .iter()
            .rev()
            .find(|msg| msg.role == role)
            .cloned())
    }

    /// Get all messages of a specific role.
    pub async fn get_messages_by_role(&self, role: MessageRole) -> Result<Vec<ChatMessage>> {
        Ok(self
            .messages
            .iter()
            .filter(|msg| msg.role == role)
            .cloned()
            .collect())
    }

    /// Check if the conversation has a specific pattern (e.g., user followed by assistant).
    pub async fn has_conversation_pattern(&self) -> Result<bool> {
        if self.messages.len() < 2 {
            return Ok(false);
        }

        // Check if we have alternating user-assistant pattern
        let mut expecting_user = true;
        for message in &self.messages {
            match message.role {
                MessageRole::System | MessageRole::Tool => continue, // Skip system and tool messages
                MessageRole::User => {
                    if !expecting_user {
                        return Ok(false);
                    }
                    expecting_user = false;
                }
                MessageRole::Assistant => {
                    if expecting_user {
                        return Ok(false);
                    }
                    expecting_user = true;
                }
            }
        }

        Ok(true)
    }
}

#[async_trait]
impl BaseMemory for ChatMemoryBuffer {
    async fn get_messages(&self) -> Result<Vec<ChatMessage>> {
        Ok(self.messages.clone())
    }

    async fn add_message(&mut self, message: ChatMessage) -> Result<()> {
        debug!("Adding message to memory: {:?}", message.role);

        self.messages.push(message);
        self.update_stats().await?;
        self.truncate_if_needed().await?;

        Ok(())
    }

    async fn clear(&mut self) -> Result<()> {
        self.messages.clear();
        self.stats = MemoryStats::empty();
        info!("Cleared chat memory buffer");
        Ok(())
    }

    async fn get_memory_variables(&self) -> Result<HashMap<String, String>> {
        let mut vars = HashMap::new();

        vars.insert(
            "message_count".to_string(),
            self.stats.message_count.to_string(),
        );
        vars.insert(
            "estimated_tokens".to_string(),
            self.stats.estimated_tokens.to_string(),
        );
        vars.insert(
            "system_messages".to_string(),
            self.stats.system_message_count.to_string(),
        );
        vars.insert(
            "user_messages".to_string(),
            self.stats.user_message_count.to_string(),
        );
        vars.insert(
            "assistant_messages".to_string(),
            self.stats.assistant_message_count.to_string(),
        );
        vars.insert(
            "tool_messages".to_string(),
            self.stats.tool_message_count.to_string(),
        );
        vars.insert(
            "at_capacity".to_string(),
            self.stats.at_capacity.to_string(),
        );
        vars.insert(
            "last_truncated".to_string(),
            self.stats.last_truncated_count.to_string(),
        );

        // Add conversation context
        if let Some(last_user) = self.get_last_message_by_role(MessageRole::User).await? {
            vars.insert("last_user_message".to_string(), last_user.content);
        }

        if let Some(last_assistant) = self
            .get_last_message_by_role(MessageRole::Assistant)
            .await?
        {
            vars.insert("last_assistant_message".to_string(), last_assistant.content);
        }

        Ok(vars)
    }

    async fn pop_message(&mut self) -> Result<Option<ChatMessage>> {
        let last_message = self.messages.pop();
        if last_message.is_some() {
            self.update_stats().await?;
        }
        Ok(last_message)
    }
}
