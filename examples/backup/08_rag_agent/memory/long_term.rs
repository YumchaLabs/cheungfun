//! 长期记忆管理
//!
//! 这个模块负责管理长期记忆，包括：
//! - 重要信息的持久化存储
//! - 知识点的关联和索引
//! - 用户偏好的学习和记录
//! - 记忆的检索和更新

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// 记忆类型
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    /// 知识点
    Knowledge,
    /// 用户偏好
    Preference,
    /// 重要事实
    Fact,
    /// 个人信息
    Personal,
    /// 技能或方法
    Skill,
}

/// 长期记忆条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// 唯一标识
    pub id: String,
    /// 记忆内容
    pub content: String,
    /// 记忆类型
    pub memory_type: MemoryType,
    /// 重要性分数 (0.0-1.0)
    pub importance: f64,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 最后访问时间
    pub last_accessed: DateTime<Utc>,
    /// 访问次数
    pub access_count: u32,
    /// 标签
    pub tags: Vec<String>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl MemoryEntry {
    /// 创建新的记忆条目
    pub fn new(
        content: String,
        memory_type: MemoryType,
        importance: f64,
        tags: Vec<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            content,
            memory_type,
            importance: importance.clamp(0.0, 1.0),
            created_at: now,
            last_accessed: now,
            access_count: 0,
            tags,
            metadata: HashMap::new(),
        }
    }

    /// 更新访问信息
    pub fn update_access(&mut self) {
        self.last_accessed = Utc::now();
        self.access_count += 1;
    }

    /// 计算与查询的相关性分数
    pub fn relevance_score(&self, query: &str) -> f64 {
        let query_lower = query.to_lowercase();
        let content_lower = self.content.to_lowercase();

        let mut score = 0.0;

        // 内容匹配
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let total_words = query_words.len() as f64;

        if total_words == 0.0 {
            return 0.0;
        }

        for word in query_words {
            if content_lower.contains(word) {
                score += 1.0;
            }
        }

        // 标签匹配
        for tag in &self.tags {
            if query_lower.contains(&tag.to_lowercase()) {
                score += 0.5;
            }
        }

        // 重要性权重
        let importance_weight = self.importance;

        // 访问频率权重
        let access_weight = (self.access_count as f64).ln().max(1.0) / 10.0;

        // 时间衰减（长期记忆衰减较慢）
        let days_ago = (Utc::now() - self.created_at).num_days() as f64;
        let time_decay = 1.0 / (1.0 + days_ago / 365.0); // 每年衰减一半

        (score / total_words) * importance_weight * (1.0 + access_weight) * time_decay
    }

    /// 获取记忆摘要
    pub fn get_summary(&self) -> String {
        let content_preview = if self.content.len() > 100 {
            format!("{}...", &self.content[..100])
        } else {
            self.content.clone()
        };

        format!(
            "[{}] {} (重要性: {:.2}, 访问: {}次)",
            self.memory_type_display(),
            content_preview,
            self.importance,
            self.access_count
        )
    }

    /// 获取记忆类型的显示名称
    fn memory_type_display(&self) -> &str {
        match self.memory_type {
            MemoryType::Knowledge => "知识",
            MemoryType::Preference => "偏好",
            MemoryType::Fact => "事实",
            MemoryType::Personal => "个人",
            MemoryType::Skill => "技能",
        }
    }
}

/// 长期记忆管理器
#[derive(Debug)]
pub struct LongTermMemory {
    /// 记忆条目
    entries: HashMap<String, MemoryEntry>,
    /// 标签索引
    tag_index: HashMap<String, Vec<String>>, // tag -> entry_ids
    /// 类型索引
    type_index: HashMap<MemoryType, Vec<String>>, // type -> entry_ids
}

impl LongTermMemory {
    /// 创建新的长期记忆管理器
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            tag_index: HashMap::new(),
            type_index: HashMap::new(),
        }
    }

    /// 添加记忆条目
    pub async fn add_entry(&mut self, entry: MemoryEntry) -> Result<(), super::MemoryError> {
        let entry_id = entry.id.clone();

        // 更新标签索引
        for tag in &entry.tags {
            self.tag_index
                .entry(tag.clone())
                .or_insert_with(Vec::new)
                .push(entry_id.clone());
        }

        // 更新类型索引
        self.type_index
            .entry(entry.memory_type.clone())
            .or_insert_with(Vec::new)
            .push(entry_id.clone());

        // 存储条目
        self.entries.insert(entry_id, entry);

        Ok(())
    }

    /// 搜索相关记忆
    pub async fn search_relevant(&self, query: &str, max_entries: usize) -> Vec<MemoryEntry> {
        let mut scored_entries: Vec<(f64, MemoryEntry)> = self
            .entries
            .values()
            .map(|entry| (entry.relevance_score(query), entry.clone()))
            .collect();

        // 按相关性排序
        scored_entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // 返回最相关的记忆
        let mut results: Vec<MemoryEntry> = scored_entries
            .into_iter()
            .take(max_entries)
            .map(|(_, mut entry)| {
                entry.update_access();
                entry
            })
            .collect();

        // 更新访问信息
        for result in &results {
            if let Some(entry) = self.entries.get(&result.id) {
                // 在实际实现中，这里应该更新存储中的访问信息
            }
        }

        results
    }

    /// 根据标签搜索
    pub async fn search_by_tag(&self, tag: &str) -> Vec<MemoryEntry> {
        if let Some(entry_ids) = self.tag_index.get(tag) {
            entry_ids
                .iter()
                .filter_map(|id| self.entries.get(id))
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// 根据类型搜索
    pub async fn search_by_type(&self, memory_type: &MemoryType) -> Vec<MemoryEntry> {
        if let Some(entry_ids) = self.type_index.get(memory_type) {
            entry_ids
                .iter()
                .filter_map(|id| self.entries.get(id))
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// 更新记忆条目
    pub async fn update_entry(
        &mut self,
        entry_id: &str,
        updated_entry: MemoryEntry,
    ) -> Result<(), super::MemoryError> {
        if !self.entries.contains_key(entry_id) {
            return Err(super::MemoryError::EntryNotFound(entry_id.to_string()));
        }

        // 移除旧的索引
        self.remove_from_indices(entry_id);

        // 添加新的索引
        for tag in &updated_entry.tags {
            self.tag_index
                .entry(tag.clone())
                .or_insert_with(Vec::new)
                .push(entry_id.to_string());
        }

        self.type_index
            .entry(updated_entry.memory_type.clone())
            .or_insert_with(Vec::new)
            .push(entry_id.to_string());

        // 更新条目
        self.entries.insert(entry_id.to_string(), updated_entry);

        Ok(())
    }

    /// 删除记忆条目
    pub async fn remove_entry(&mut self, entry_id: &str) -> Result<(), super::MemoryError> {
        if !self.entries.contains_key(entry_id) {
            return Err(super::MemoryError::EntryNotFound(entry_id.to_string()));
        }

        // 从索引中移除
        self.remove_from_indices(entry_id);

        // 删除条目
        self.entries.remove(entry_id);

        Ok(())
    }

    /// 清理旧的记忆条目
    pub async fn cleanup_old_entries(&mut self) -> Result<(), super::MemoryError> {
        let cutoff_date = Utc::now() - Duration::days(365); // 保留一年内的记忆

        let old_entry_ids: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry.last_accessed < cutoff_date
                    && entry.importance < 0.5
                    && entry.access_count < 5
            })
            .map(|(id, _)| id.clone())
            .collect();

        for entry_id in old_entry_ids {
            self.remove_entry(&entry_id).await?;
        }

        Ok(())
    }

    /// 获取记忆统计信息
    pub fn get_stats(&self) -> LongTermMemoryStats {
        let total_entries = self.entries.len();

        let mut type_counts = HashMap::new();
        let mut importance_sum = 0.0;
        let mut access_sum = 0;

        for entry in self.entries.values() {
            *type_counts.entry(entry.memory_type.clone()).or_insert(0) += 1;
            importance_sum += entry.importance;
            access_sum += entry.access_count;
        }

        let avg_importance = if total_entries > 0 {
            importance_sum / total_entries as f64
        } else {
            0.0
        };

        let avg_access_count = if total_entries > 0 {
            access_sum / total_entries as u32
        } else {
            0
        };

        LongTermMemoryStats {
            total_entries,
            type_counts,
            avg_importance,
            avg_access_count,
            total_tags: self.tag_index.len(),
        }
    }

    /// 从索引中移除条目
    fn remove_from_indices(&mut self, entry_id: &str) {
        if let Some(entry) = self.entries.get(entry_id) {
            // 从标签索引中移除
            for tag in &entry.tags {
                if let Some(ids) = self.tag_index.get_mut(tag) {
                    ids.retain(|id| id != entry_id);
                    if ids.is_empty() {
                        self.tag_index.remove(tag);
                    }
                }
            }

            // 从类型索引中移除
            if let Some(ids) = self.type_index.get_mut(&entry.memory_type) {
                ids.retain(|id| id != entry_id);
                if ids.is_empty() {
                    self.type_index.remove(&entry.memory_type);
                }
            }
        }
    }
}

/// 长期记忆统计信息
#[derive(Debug, Serialize, Deserialize)]
pub struct LongTermMemoryStats {
    pub total_entries: usize,
    pub type_counts: HashMap<MemoryType, usize>,
    pub avg_importance: f64,
    pub avg_access_count: u32,
    pub total_tags: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_long_term_memory() {
        let mut memory = LongTermMemory::new();

        let entry = MemoryEntry::new(
            "RAG是检索增强生成技术".to_string(),
            MemoryType::Knowledge,
            0.8,
            vec!["RAG".to_string(), "AI".to_string()],
        );

        memory.add_entry(entry).await.unwrap();

        let results = memory.search_relevant("RAG", 5).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("RAG"));
    }

    #[test]
    fn test_memory_entry_relevance() {
        let entry = MemoryEntry::new(
            "RAG是检索增强生成技术，结合了信息检索和文本生成".to_string(),
            MemoryType::Knowledge,
            0.9,
            vec!["RAG".to_string(), "AI".to_string()],
        );

        let score1 = entry.relevance_score("RAG技术");
        let score2 = entry.relevance_score("Python编程");

        assert!(score1 > score2);
    }

    #[tokio::test]
    async fn test_search_by_tag() {
        let mut memory = LongTermMemory::new();

        let entry = MemoryEntry::new(
            "测试内容".to_string(),
            MemoryType::Knowledge,
            0.5,
            vec!["测试".to_string()],
        );

        memory.add_entry(entry).await.unwrap();

        let results = memory.search_by_tag("测试").await;
        assert_eq!(results.len(), 1);
    }
}
