//! 对话历史管理
//!
//! 这个模块负责管理对话历史，包括：
//! - 对话轮次的存储和检索
//! - 智能摘要和压缩
//! - 上下文相关性计算
//! - 对话历史的清理

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

/// 对话记忆配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// 最大对话轮数
    pub max_conversation_length: usize,
    /// 摘要触发阈值
    pub summary_threshold: usize,
    /// 启用长期记忆
    pub enable_long_term_memory: bool,
    /// 最大上下文轮数
    pub max_context_turns: usize,
    /// 最大长期记忆条目数
    pub max_long_term_entries: usize,
    /// 记忆保留天数
    pub retention_days: i64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_conversation_length: 50,
            summary_threshold: 20,
            enable_long_term_memory: true,
            max_context_turns: 5,
            max_long_term_entries: 10,
            retention_days: 30,
        }
    }
}

/// 对话轮次
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// 唯一标识
    pub id: String,
    /// 用户输入
    pub user_input: String,
    /// 助手回应
    pub assistant_response: String,
    /// 时间戳
    pub timestamp: DateTime<Utc>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl ConversationTurn {
    /// 创建新的对话轮次
    pub fn new(
        user_input: String,
        assistant_response: String,
        metadata: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            user_input,
            assistant_response,
            timestamp: Utc::now(),
            metadata: metadata.unwrap_or_default(),
        }
    }

    /// 计算与查询的相关性分数
    pub fn relevance_score(&self, query: &str) -> f64 {
        let query_lower = query.to_lowercase();
        let user_input_lower = self.user_input.to_lowercase();
        let response_lower = self.assistant_response.to_lowercase();

        let mut score = 0.0;

        // 简单的关键词匹配
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let total_words = query_words.len() as f64;

        if total_words == 0.0 {
            return 0.0;
        }

        for word in query_words {
            if user_input_lower.contains(word) {
                score += 1.0;
            }
            if response_lower.contains(word) {
                score += 0.5; // 回应中的匹配权重较低
            }
        }

        // 时间衰减：越新的对话相关性越高
        let hours_ago = (Utc::now() - self.timestamp).num_hours() as f64;
        let time_decay = 1.0 / (1.0 + hours_ago / 24.0); // 每天衰减一半

        (score / total_words) * time_decay
    }

    /// 获取对话摘要
    pub fn get_summary(&self) -> String {
        let user_preview = if self.user_input.len() > 50 {
            format!("{}...", &self.user_input[..50])
        } else {
            self.user_input.clone()
        };

        let response_preview = if self.assistant_response.len() > 100 {
            format!("{}...", &self.assistant_response[..100])
        } else {
            self.assistant_response.clone()
        };

        format!("Q: {} | A: {}", user_preview, response_preview)
    }
}

/// 对话记忆管理器
#[derive(Debug)]
pub struct ConversationMemory {
    /// 对话历史
    turns: VecDeque<ConversationTurn>,
    /// 摘要历史
    summaries: Vec<String>,
    /// 配置
    config: MemoryConfig,
}

impl ConversationMemory {
    /// 创建新的对话记忆
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            turns: VecDeque::with_capacity(config.max_conversation_length),
            summaries: Vec::new(),
            config,
        }
    }

    /// 添加对话轮次
    pub async fn add_turn(
        &mut self,
        user_input: String,
        assistant_response: String,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<(), super::MemoryError> {
        let turn = ConversationTurn::new(user_input, assistant_response, metadata);

        // 如果达到最大长度，移除最旧的对话
        if self.turns.len() >= self.config.max_conversation_length {
            self.turns.pop_front();
        }

        self.turns.push_back(turn);
        Ok(())
    }

    /// 获取相关的对话上下文
    pub async fn get_relevant_context(
        &self,
        query: &str,
        max_turns: usize,
    ) -> Vec<ConversationTurn> {
        let mut scored_turns: Vec<(f64, ConversationTurn)> = self
            .turns
            .iter()
            .map(|turn| (turn.relevance_score(query), turn.clone()))
            .collect();

        // 按相关性排序
        scored_turns.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // 返回最相关的对话
        scored_turns
            .into_iter()
            .take(max_turns)
            .map(|(_, turn)| turn)
            .collect()
    }

    /// 获取最近的对话
    pub fn get_recent_turns(&self, count: usize) -> Vec<ConversationTurn> {
        self.turns
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// 检查是否需要摘要
    pub fn should_summarize(&self, config: &MemoryConfig) -> bool {
        self.turns.len() >= config.summary_threshold
    }

    /// 摘要对话历史
    pub async fn summarize_if_needed(
        &mut self,
        config: &MemoryConfig,
    ) -> Result<(), super::MemoryError> {
        if !self.should_summarize(config) {
            return Ok(());
        }

        // 简化实现：创建对话摘要
        let summary_turns = self
            .turns
            .iter()
            .take(config.summary_threshold / 2)
            .collect::<Vec<_>>();
        let summary = self.create_summary(&summary_turns);

        // 保存摘要
        self.summaries.push(summary);

        // 移除已摘要的对话
        for _ in 0..(config.summary_threshold / 2) {
            self.turns.pop_front();
        }

        Ok(())
    }

    /// 创建对话摘要
    fn create_summary(&self, turns: &[&ConversationTurn]) -> String {
        let mut topics = HashMap::new();
        let mut summary_parts = Vec::new();

        for turn in turns {
            // 简单的主题提取
            let words: Vec<&str> = turn.user_input.split_whitespace().collect();
            for word in words {
                if word.len() > 3 {
                    // 忽略短词
                    *topics.entry(word.to_lowercase()).or_insert(0) += 1;
                }
            }

            summary_parts.push(turn.get_summary());
        }

        // 找出最常见的主题
        let mut topic_vec: Vec<_> = topics.into_iter().collect();
        topic_vec.sort_by(|a, b| b.1.cmp(&a.1));

        let main_topics: Vec<String> = topic_vec
            .into_iter()
            .take(3)
            .map(|(topic, _)| topic)
            .collect();

        format!(
            "对话摘要 ({}轮): 主要讨论了{}等话题。包含: {}",
            turns.len(),
            main_topics.join("、"),
            summary_parts.join(" | ")
        )
    }

    /// 清理旧的对话
    pub async fn cleanup_old_turns(
        &mut self,
        config: &MemoryConfig,
    ) -> Result<(), super::MemoryError> {
        let cutoff_date = Utc::now() - Duration::days(config.retention_days);

        // 移除过期的对话
        self.turns.retain(|turn| turn.timestamp > cutoff_date);

        Ok(())
    }

    /// 获取对话统计信息
    pub fn get_stats(&self) -> ConversationStats {
        let total_turns = self.turns.len();
        let total_summaries = self.summaries.len();

        let avg_user_length = if total_turns > 0 {
            self.turns.iter().map(|t| t.user_input.len()).sum::<usize>() / total_turns
        } else {
            0
        };

        let avg_response_length = if total_turns > 0 {
            self.turns
                .iter()
                .map(|t| t.assistant_response.len())
                .sum::<usize>()
                / total_turns
        } else {
            0
        };

        ConversationStats {
            total_turns,
            total_summaries,
            avg_user_input_length: avg_user_length,
            avg_response_length,
            oldest_turn: self.turns.front().map(|t| t.timestamp),
            newest_turn: self.turns.back().map(|t| t.timestamp),
        }
    }

    /// 导出对话历史
    pub fn export_history(&self) -> Result<String, super::MemoryError> {
        let export_data = ConversationExport {
            turns: self.turns.iter().cloned().collect(),
            summaries: self.summaries.clone(),
            exported_at: Utc::now(),
        };

        Ok(serde_json::to_string_pretty(&export_data)?)
    }

    /// 导入对话历史
    pub fn import_history(&mut self, data: &str) -> Result<(), super::MemoryError> {
        let import_data: ConversationExport = serde_json::from_str(data)?;

        // 清空现有数据
        self.turns.clear();
        self.summaries.clear();

        // 导入数据
        for turn in import_data.turns {
            self.turns.push_back(turn);
        }
        self.summaries = import_data.summaries;

        Ok(())
    }
}

/// 对话统计信息
#[derive(Debug, Serialize, Deserialize)]
pub struct ConversationStats {
    pub total_turns: usize,
    pub total_summaries: usize,
    pub avg_user_input_length: usize,
    pub avg_response_length: usize,
    pub oldest_turn: Option<DateTime<Utc>>,
    pub newest_turn: Option<DateTime<Utc>>,
}

/// 对话导出格式
#[derive(Debug, Serialize, Deserialize)]
struct ConversationExport {
    turns: Vec<ConversationTurn>,
    summaries: Vec<String>,
    exported_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_conversation_memory() {
        let config = MemoryConfig::default();
        let mut memory = ConversationMemory::new(config);

        // 添加对话
        memory
            .add_turn(
                "什么是RAG？".to_string(),
                "RAG是检索增强生成技术".to_string(),
                None,
            )
            .await
            .unwrap();

        // 获取相关上下文
        let context = memory.get_relevant_context("RAG", 5).await;
        assert_eq!(context.len(), 1);
        assert!(context[0].user_input.contains("RAG"));
    }

    #[test]
    fn test_relevance_score() {
        let turn = ConversationTurn::new(
            "什么是RAG技术？".to_string(),
            "RAG是检索增强生成技术".to_string(),
            None,
        );

        let score1 = turn.relevance_score("RAG");
        let score2 = turn.relevance_score("Python");

        assert!(score1 > score2);
    }

    #[test]
    fn test_conversation_summary() {
        let turn = ConversationTurn::new(
            "这是一个很长的用户输入，用来测试摘要功能是否正常工作，应该会被截断".to_string(),
            "这是一个很长的助手回应，用来测试摘要功能是否正常工作，应该会被截断，包含更多的内容来测试".to_string(),
            None,
        );

        let summary = turn.get_summary();
        assert!(summary.len() < turn.user_input.len() + turn.assistant_response.len());
        assert!(summary.contains("..."));
    }
}
