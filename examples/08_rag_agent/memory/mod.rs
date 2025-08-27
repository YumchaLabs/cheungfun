//! 记忆管理模块
//!
//! 这个模块提供了完整的记忆管理功能，包括：
//! - 对话历史管理
//! - 长期记忆存储
//! - 智能摘要和压缩
//! - 上下文感知检索

pub mod conversation;
pub mod long_term;

pub use conversation::{ConversationMemory, ConversationTurn, MemoryConfig};
pub use long_term::{LongTermMemory, MemoryEntry, MemoryType};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 统一的记忆管理器
#[derive(Debug)]
pub struct MemoryManager {
    /// 对话记忆
    conversation: ConversationMemory,
    /// 长期记忆
    long_term: LongTermMemory,
    /// 配置
    config: MemoryConfig,
}

impl MemoryManager {
    /// 创建新的记忆管理器
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            conversation: ConversationMemory::new(config.clone()),
            long_term: LongTermMemory::new(),
            config,
        }
    }

    /// 添加对话轮次
    pub async fn add_conversation_turn(
        &mut self,
        user_input: String,
        assistant_response: String,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<(), MemoryError> {
        // 添加到对话记忆
        self.conversation
            .add_turn(
                user_input.clone(),
                assistant_response.clone(),
                metadata.clone(),
            )
            .await?;

        // 如果启用长期记忆，提取重要信息
        if self.config.enable_long_term_memory {
            self.extract_to_long_term(&user_input, &assistant_response)
                .await?;
        }

        // 检查是否需要摘要
        if self.conversation.should_summarize(&self.config) {
            self.conversation.summarize_if_needed(&self.config).await?;
        }

        Ok(())
    }

    /// 获取相关的对话上下文
    pub async fn get_relevant_context(
        &self,
        query: &str,
        max_turns: usize,
    ) -> Vec<ConversationTurn> {
        self.conversation
            .get_relevant_context(query, max_turns)
            .await
    }

    /// 获取长期记忆中的相关信息
    pub async fn get_long_term_context(&self, query: &str, max_entries: usize) -> Vec<MemoryEntry> {
        self.long_term.search_relevant(query, max_entries).await
    }

    /// 获取完整的上下文（对话+长期记忆）
    pub async fn get_full_context(&self, query: &str) -> MemoryContext {
        let conversation_context = self
            .get_relevant_context(query, self.config.max_context_turns)
            .await;
        let long_term_context = self
            .get_long_term_context(query, self.config.max_long_term_entries)
            .await;

        MemoryContext {
            conversation_turns: conversation_context,
            long_term_entries: long_term_context,
            query: query.to_string(),
            timestamp: Utc::now(),
        }
    }

    /// 获取对话统计信息
    pub fn get_conversation_stats(&self) -> conversation::ConversationStats {
        self.conversation.get_stats()
    }

    /// 清理旧的记忆
    pub async fn cleanup_old_memories(&mut self) -> Result<(), MemoryError> {
        self.conversation.cleanup_old_turns(&self.config).await?;
        self.long_term.cleanup_old_entries().await?;
        Ok(())
    }

    /// 提取信息到长期记忆
    async fn extract_to_long_term(
        &mut self,
        user_input: &str,
        assistant_response: &str,
    ) -> Result<(), MemoryError> {
        // 简化实现：如果对话包含重要关键词，则存储到长期记忆
        let important_keywords = ["定义", "是什么", "如何", "为什么", "原理", "方法", "步骤"];

        if important_keywords
            .iter()
            .any(|&keyword| user_input.contains(keyword))
        {
            let entry = MemoryEntry {
                id: uuid::Uuid::new_v4().to_string(),
                content: format!("Q: {}\nA: {}", user_input, assistant_response),
                memory_type: MemoryType::Knowledge,
                importance: 0.8,
                created_at: Utc::now(),
                last_accessed: Utc::now(),
                access_count: 1,
                tags: vec!["conversation".to_string()],
                metadata: HashMap::new(),
            };

            self.long_term.add_entry(entry).await?;
        }

        Ok(())
    }
}

/// 记忆上下文
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    /// 相关的对话轮次
    pub conversation_turns: Vec<ConversationTurn>,
    /// 相关的长期记忆条目
    pub long_term_entries: Vec<MemoryEntry>,
    /// 查询内容
    pub query: String,
    /// 时间戳
    pub timestamp: DateTime<Utc>,
}

impl MemoryContext {
    /// 格式化为提示文本
    pub fn format_for_prompt(&self) -> String {
        let mut context = String::new();

        // 添加长期记忆上下文
        if !self.long_term_entries.is_empty() {
            context.push_str("相关知识：\n");
            for entry in &self.long_term_entries {
                context.push_str(&format!("- {}\n", entry.content));
            }
            context.push('\n');
        }

        // 添加对话历史
        if !self.conversation_turns.is_empty() {
            context.push_str("对话历史：\n");
            for turn in &self.conversation_turns {
                context.push_str(&format!(
                    "用户: {}\n助手: {}\n\n",
                    turn.user_input, turn.assistant_response
                ));
            }
        }

        context
    }

    /// 检查是否有相关上下文
    pub fn has_context(&self) -> bool {
        !self.conversation_turns.is_empty() || !self.long_term_entries.is_empty()
    }

    /// 获取上下文摘要
    pub fn get_summary(&self) -> String {
        format!(
            "找到 {} 轮相关对话和 {} 条长期记忆",
            self.conversation_turns.len(),
            self.long_term_entries.len()
        )
    }
}

/// 记忆错误类型
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("序列化错误: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO错误: {0}")]
    Io(#[from] std::io::Error),

    #[error("记忆容量已满")]
    CapacityFull,

    #[error("记忆条目未找到: {0}")]
    EntryNotFound(String),

    #[error("配置错误: {0}")]
    Configuration(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_manager() {
        let config = MemoryConfig::default();
        let mut manager = MemoryManager::new(config);

        // 添加对话
        manager
            .add_conversation_turn(
                "什么是RAG？".to_string(),
                "RAG是检索增强生成技术...".to_string(),
                None,
            )
            .await
            .unwrap();

        // 获取上下文
        let context = manager.get_full_context("RAG").await;
        assert!(context.has_context());
    }

    #[tokio::test]
    async fn test_memory_context_formatting() {
        let context = MemoryContext {
            conversation_turns: vec![ConversationTurn {
                id: "1".to_string(),
                user_input: "测试问题".to_string(),
                assistant_response: "测试回答".to_string(),
                timestamp: Utc::now(),
                metadata: HashMap::new(),
            }],
            long_term_entries: vec![],
            query: "测试".to_string(),
            timestamp: Utc::now(),
        };

        let formatted = context.format_for_prompt();
        assert!(formatted.contains("对话历史"));
        assert!(formatted.contains("测试问题"));
    }
}
