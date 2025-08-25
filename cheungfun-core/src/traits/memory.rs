//! Memory management traits for conversation history.
//!
//! This module defines traits for managing conversation memory in RAG applications,
//! following LlamaIndex's memory management patterns. Memory components handle
//! conversation history, context window management, and intelligent truncation.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::{ChatMessage, Result};

/// Base trait for conversation memory management.
///
/// This trait provides a unified interface for managing conversation history,
/// following LlamaIndex's BaseMemory pattern. Memory implementations handle
/// context window limits, conversation truncation, and history persistence.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::BaseMemory;
/// use cheungfun_core::{ChatMessage, MessageRole, Result};
/// use async_trait::async_trait;
///
/// struct SimpleMemory {
///     messages: Vec<ChatMessage>,
///     max_tokens: usize,
/// }
///
/// #[async_trait]
/// impl BaseMemory for SimpleMemory {
///     async fn get_messages(&self) -> Result<Vec<ChatMessage>> {
///         Ok(self.messages.clone())
///     }
///
///     async fn add_message(&mut self, message: ChatMessage) -> Result<()> {
///         self.messages.push(message);
///         Ok(())
///     }
///
///     async fn clear(&mut self) -> Result<()> {
///         self.messages.clear();
///         Ok(())
///     }
///
///     async fn get_memory_variables(&self) -> Result<HashMap<String, String>> {
///         let mut vars = HashMap::new();
///         vars.insert("message_count".to_string(), self.messages.len().to_string());
///         Ok(vars)
///     }
/// }
/// ```
#[async_trait]
pub trait BaseMemory: Send + Sync + std::fmt::Debug {
    /// Get all messages in the conversation history.
    ///
    /// # Returns
    ///
    /// A vector of chat messages in chronological order.
    async fn get_messages(&self) -> Result<Vec<ChatMessage>>;

    /// Add a new message to the conversation history.
    ///
    /// # Arguments
    ///
    /// * `message` - The chat message to add
    async fn add_message(&mut self, message: ChatMessage) -> Result<()>;

    /// Add multiple messages to the conversation history.
    ///
    /// # Arguments
    ///
    /// * `messages` - Vector of chat messages to add
    async fn add_messages(&mut self, messages: Vec<ChatMessage>) -> Result<()> {
        for message in messages {
            self.add_message(message).await?;
        }
        Ok(())
    }

    /// Clear all messages from the conversation history.
    async fn clear(&mut self) -> Result<()>;

    /// Get memory variables for template substitution.
    ///
    /// # Returns
    ///
    /// A map of variable names to their string representations.
    async fn get_memory_variables(&self) -> Result<HashMap<String, String>>;

    /// Get the number of messages in memory.
    async fn message_count(&self) -> Result<usize> {
        Ok(self.get_messages().await?.len())
    }

    /// Check if the memory is empty.
    async fn is_empty(&self) -> Result<bool> {
        Ok(self.message_count().await? == 0)
    }

    /// Get recent messages up to a limit.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of recent messages to return
    async fn get_recent_messages(&self, limit: usize) -> Result<Vec<ChatMessage>> {
        let messages = self.get_messages().await?;
        let start = messages.len().saturating_sub(limit);
        Ok(messages[start..].to_vec())
    }

    /// Remove the last message from memory.
    ///
    /// # Returns
    ///
    /// The removed message, if any.
    async fn pop_message(&mut self) -> Result<Option<ChatMessage>> {
        let mut messages = self.get_messages().await?;
        let last_message = messages.pop();
        self.clear().await?;
        self.add_messages(messages).await?;
        Ok(last_message)
    }
}

/// Configuration for chat memory buffer.
#[derive(Debug, Clone)]
pub struct ChatMemoryConfig {
    /// Maximum number of tokens to keep in memory.
    pub max_tokens: Option<usize>,

    /// Maximum number of messages to keep in memory.
    pub max_messages: Option<usize>,

    /// Whether to preserve system messages during truncation.
    pub preserve_system_messages: bool,

    /// Whether to preserve the most recent user-assistant pair.
    pub preserve_recent_pair: bool,

    /// Token counting method.
    pub token_counter: TokenCountingMethod,

    /// Whether to summarize truncated messages.
    pub summarize_truncated: bool,

    /// Summary prompt template.
    pub summary_prompt: Option<String>,
}

impl Default for ChatMemoryConfig {
    fn default() -> Self {
        Self {
            max_tokens: Some(4000), // Conservative default for most models
            max_messages: Some(50),
            preserve_system_messages: true,
            preserve_recent_pair: true,
            token_counter: TokenCountingMethod::Approximate,
            summarize_truncated: false,
            summary_prompt: None,
        }
    }
}

impl ChatMemoryConfig {
    /// Create a new configuration with token limit.
    pub fn with_token_limit(max_tokens: usize) -> Self {
        Self {
            max_tokens: Some(max_tokens),
            ..Default::default()
        }
    }

    /// Create a new configuration with message limit.
    pub fn with_message_limit(max_messages: usize) -> Self {
        Self {
            max_messages: Some(max_messages),
            ..Default::default()
        }
    }

    /// Enable summarization of truncated messages.
    pub fn with_summarization(mut self, summary_prompt: Option<String>) -> Self {
        self.summarize_truncated = true;
        self.summary_prompt = summary_prompt;
        self
    }

    /// Set token counting method.
    pub fn with_token_counter(mut self, method: TokenCountingMethod) -> Self {
        self.token_counter = method;
        self
    }
}

/// Methods for counting tokens in messages.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenCountingMethod {
    /// Approximate token count (chars / 4).
    Approximate,
    /// Use tiktoken for accurate counting (requires tiktoken feature).
    #[cfg(feature = "tiktoken")]
    Tiktoken(String), // model name
    /// Custom token counter function.
    Custom,
}

/// Token counting trait for different implementations.
#[async_trait]
pub trait TokenCounter: Send + Sync + std::fmt::Debug {
    /// Count tokens in a message.
    ///
    /// # Arguments
    ///
    /// * `message` - The chat message to count tokens for
    ///
    /// # Returns
    ///
    /// The estimated number of tokens.
    async fn count_message_tokens(&self, message: &ChatMessage) -> Result<usize>;

    /// Count tokens in multiple messages.
    ///
    /// # Arguments
    ///
    /// * `messages` - Vector of chat messages
    ///
    /// # Returns
    ///
    /// The total estimated number of tokens.
    async fn count_messages_tokens(&self, messages: &[ChatMessage]) -> Result<usize> {
        let mut total = 0;
        for message in messages {
            total += self.count_message_tokens(message).await?;
        }
        Ok(total)
    }

    /// Count tokens in a text string.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to count tokens for
    ///
    /// # Returns
    ///
    /// The estimated number of tokens.
    async fn count_text_tokens(&self, text: &str) -> Result<usize>;
}

/// Simple approximate token counter.
#[derive(Debug, Clone)]
pub struct ApproximateTokenCounter {
    /// Characters per token ratio (default: 4).
    chars_per_token: f32,
}

impl Default for ApproximateTokenCounter {
    fn default() -> Self {
        Self {
            chars_per_token: 4.0,
        }
    }
}

impl ApproximateTokenCounter {
    /// Create a new approximate token counter.
    pub fn new(chars_per_token: f32) -> Self {
        Self { chars_per_token }
    }
}

#[async_trait]
impl TokenCounter for ApproximateTokenCounter {
    async fn count_message_tokens(&self, message: &ChatMessage) -> Result<usize> {
        // Count tokens in content plus some overhead for role and metadata
        let content_tokens = self.count_text_tokens(&message.content).await?;
        let role_tokens = 2; // Approximate overhead for role
        let metadata_tokens = if message.metadata.is_some() { 5 } else { 0 };

        Ok(content_tokens + role_tokens + metadata_tokens)
    }

    async fn count_text_tokens(&self, text: &str) -> Result<usize> {
        Ok((text.len() as f32 / self.chars_per_token).ceil() as usize)
    }
}

/// Memory statistics for monitoring and debugging.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total number of messages in memory.
    pub message_count: usize,

    /// Estimated total tokens in memory.
    pub estimated_tokens: usize,

    /// Number of system messages.
    pub system_message_count: usize,

    /// Number of user messages.
    pub user_message_count: usize,

    /// Number of assistant messages.
    pub assistant_message_count: usize,

    /// Number of tool messages.
    pub tool_message_count: usize,

    /// Number of messages truncated in the last operation.
    pub last_truncated_count: usize,

    /// Whether the memory is at capacity.
    pub at_capacity: bool,
}

impl MemoryStats {
    /// Create empty memory statistics.
    pub fn empty() -> Self {
        Self {
            message_count: 0,
            estimated_tokens: 0,
            system_message_count: 0,
            user_message_count: 0,
            assistant_message_count: 0,
            tool_message_count: 0,
            last_truncated_count: 0,
            at_capacity: false,
        }
    }

    /// Calculate statistics from messages.
    pub async fn from_messages(
        messages: &[ChatMessage],
        token_counter: &dyn TokenCounter,
    ) -> Result<Self> {
        let message_count = messages.len();
        let estimated_tokens = token_counter.count_messages_tokens(messages).await?;

        let mut system_count = 0;
        let mut user_count = 0;
        let mut assistant_count = 0;
        let mut tool_count = 0;

        for message in messages {
            match message.role {
                crate::MessageRole::System => system_count += 1,
                crate::MessageRole::User => user_count += 1,
                crate::MessageRole::Assistant => assistant_count += 1,
                crate::MessageRole::Tool => tool_count += 1,
            }
        }

        Ok(Self {
            message_count,
            estimated_tokens,
            system_message_count: system_count,
            user_message_count: user_count,
            assistant_message_count: assistant_count,
            tool_message_count: tool_count,
            last_truncated_count: 0,
            at_capacity: false,
        })
    }
}
