//! Memory management implementations for conversation history.
//!
//! This module provides concrete implementations of memory management traits,
//! including chat memory buffers with token limits and intelligent truncation.

pub mod chat_buffer;

pub use chat_buffer::*;

// Re-export memory traits from core for convenience
pub use cheungfun_core::traits::{
    ApproximateTokenCounter, BaseMemory, ChatMemoryConfig, MemoryStats, TokenCounter,
    TokenCountingMethod,
};
