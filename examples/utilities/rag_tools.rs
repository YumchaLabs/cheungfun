//! RAG专用工具实现
//!
//! 这个模块实现了专门用于RAG系统的工具集合

use async_trait::async_trait;
use cheungfun_agents::{
    tool::{Tool, ToolError, ToolParams, ToolResult},
    types::*,
};
use cheungfun_core::{traits::VectorStore, Result};
use cheungfun_query::retriever::VectorRetriever;
use serde_json::{json, Value};
use std::sync::Arc;

/// RAG向量搜索工具
pub struct RagVectorSearchTool {
    retriever: Arc<VectorRetriever>,
    name: String,
    description: String,
}

impl RagVectorSearchTool {
    pub fn new(retriever: Arc<VectorRetriever>) -> Self {
        Self {
            retriever,
            name: "rag_vector_search".to_string(),
            description: "在知识库中进行向量语义搜索，找到与查询最相关的文档片段".to_string(),
        }
    }
}

#[async_trait]
impl Tool for RagVectorSearchTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询文本"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的结果数量",
                    "default": 5
                },
                "similarity_threshold": {
                    "type": "number",
                    "description": "相似度阈值，0-1之间",
                    "default": 0.0
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, params: ToolParams) -> ToolResult {
        let query = params
            .get_string("query")
            .map_err(|e| ToolError::InvalidParameter(format!("query parameter error: {}", e)))?;

        let top_k = params.get_usize("top_k").unwrap_or(5);
        let similarity_threshold = params.get_f64("similarity_threshold").unwrap_or(0.0);

        // 构建查询
        let mut search_query = cheungfun_core::types::Query::new(query);
        search_query.top_k = top_k;
        search_query.similarity_threshold = Some(similarity_threshold as f32);

        // 执行搜索
        match self.retriever.retrieve(&search_query).await {
            Ok(results) => {
                let search_results: Vec<Value> = results
                    .iter()
                    .map(|scored_node| {
                        json!({
                            "content": scored_node.node.content,
                            "score": scored_node.score,
                            "metadata": scored_node.node.metadata,
                            "id": scored_node.node.id
                        })
                    })
                    .collect();

                ToolResult::success(json!({
                    "results": search_results,
                    "total_found": results.len(),
                    "query": query
                }))
            }
            Err(e) => ToolResult::error(format!("搜索失败: {}", e)),
        }
    }
}

/// 文档摘要工具
pub struct DocumentSummarizerTool {
    llm_client: Arc<cheungfun_agents::llm::SiumaiLlmClient>,
    name: String,
    description: String,
}

impl DocumentSummarizerTool {
    pub fn new(llm_client: Arc<cheungfun_agents::llm::SiumaiLlmClient>) -> Self {
        Self {
            llm_client,
            name: "document_summarizer".to_string(),
            description: "对文档内容进行智能摘要，提取关键信息".to_string(),
        }
    }
}

#[async_trait]
impl Tool for DocumentSummarizerTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "需要摘要的文档内容"
                },
                "max_length": {
                    "type": "integer",
                    "description": "摘要的最大长度（字符数）",
                    "default": 200
                },
                "focus": {
                    "type": "string",
                    "description": "摘要的重点关注方向",
                    "default": "general"
                }
            },
            "required": ["content"]
        })
    }

    async fn execute(&self, params: ToolParams) -> ToolResult {
        let content = params
            .get_string("content")
            .map_err(|e| ToolError::InvalidParameter(format!("content parameter error: {}", e)))?;

        let max_length = params.get_usize("max_length").unwrap_or(200);
        let focus = params
            .get_string("focus")
            .unwrap_or_else(|_| "general".to_string());

        let prompt = format!(
            r#"请对以下内容进行摘要，要求：
1. 摘要长度不超过{}个字符
2. 重点关注：{}
3. 保留关键信息和要点
4. 使用简洁明了的语言

内容：
{}

摘要："#,
            max_length, focus, content
        );

        let messages = vec![AgentMessage::user(prompt)];

        match self.llm_client.chat(messages).await {
            Ok(summary) => ToolResult::success(json!({
                "summary": summary.trim(),
                "original_length": content.len(),
                "summary_length": summary.trim().len(),
                "compression_ratio": content.len() as f64 / summary.trim().len() as f64
            })),
            Err(e) => ToolResult::error(format!("摘要生成失败: {}", e)),
        }
    }
}

/// 事实核查工具
pub struct FactCheckerTool {
    retriever: Arc<VectorRetriever>,
    llm_client: Arc<cheungfun_agents::llm::SiumaiLlmClient>,
    name: String,
    description: String,
}

impl FactCheckerTool {
    pub fn new(
        retriever: Arc<VectorRetriever>,
        llm_client: Arc<cheungfun_agents::llm::SiumaiLlmClient>,
    ) -> Self {
        Self {
            retriever,
            llm_client,
            name: "fact_checker".to_string(),
            description: "核查声明或事实的准确性，基于知识库中的信息".to_string(),
        }
    }
}

#[async_trait]
impl Tool for FactCheckerTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "需要核查的声明或事实"
                },
                "context": {
                    "type": "string",
                    "description": "相关上下文信息",
                    "default": ""
                }
            },
            "required": ["claim"]
        })
    }

    async fn execute(&self, params: ToolParams) -> ToolResult {
        let claim = params
            .get_string("claim")
            .map_err(|e| ToolError::InvalidParameter(format!("claim parameter error: {}", e)))?;

        let context = params.get_string("context").unwrap_or_default();

        // 1. 搜索相关证据
        let mut search_query = cheungfun_core::types::Query::new(claim.clone());
        search_query.top_k = 5;

        let evidence = match self.retriever.retrieve(&search_query).await {
            Ok(results) => results,
            Err(e) => return ToolResult::error(format!("证据搜索失败: {}", e)),
        };

        // 2. 构建事实核查提示
        let evidence_text: Vec<String> = evidence
            .iter()
            .map(|scored_node| format!("- {}", scored_node.node.content))
            .collect();

        let prompt = format!(
            r#"请基于以下证据核查声明的准确性：

声明：{}

上下文：{}

证据：
{}

请分析：
1. 声明是否准确？
2. 证据支持程度如何？
3. 是否有矛盾信息？
4. 给出置信度评分（0-100）

请以JSON格式回复：
{{
    "accurate": true/false,
    "confidence": 0-100,
    "support_level": "strong/moderate/weak/none",
    "explanation": "详细解释",
    "contradictions": ["矛盾点1", "矛盾点2"]
}}"#,
            claim,
            context,
            evidence_text.join("\n")
        );

        let messages = vec![AgentMessage::user(prompt)];

        match self.llm_client.chat(messages).await {
            Ok(response) => {
                // 尝试解析JSON响应
                match serde_json::from_str::<Value>(&response) {
                    Ok(fact_check_result) => ToolResult::success(json!({
                        "claim": claim,
                        "fact_check": fact_check_result,
                        "evidence_count": evidence.len(),
                        "evidence_sources": evidence.iter().map(|e| &e.node.id).collect::<Vec<_>>()
                    })),
                    Err(_) => {
                        // 如果JSON解析失败，返回原始响应
                        ToolResult::success(json!({
                            "claim": claim,
                            "analysis": response,
                            "evidence_count": evidence.len()
                        }))
                    }
                }
            }
            Err(e) => ToolResult::error(format!("事实核查失败: {}", e)),
        }
    }
}

/// 引用生成工具
pub struct CitationGeneratorTool {
    name: String,
    description: String,
}

impl CitationGeneratorTool {
    pub fn new() -> Self {
        Self {
            name: "citation_generator".to_string(),
            description: "为检索到的内容生成标准格式的引用".to_string(),
        }
    }
}

#[async_trait]
impl Tool for CitationGeneratorTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "description": "源文档信息列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "author": {"type": "string"},
                            "url": {"type": "string"},
                            "date": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    }
                },
                "style": {
                    "type": "string",
                    "description": "引用格式风格",
                    "enum": ["APA", "MLA", "Chicago", "IEEE"],
                    "default": "APA"
                }
            },
            "required": ["sources"]
        })
    }

    async fn execute(&self, params: ToolParams) -> ToolResult {
        let sources = params
            .get_array("sources")
            .map_err(|e| ToolError::InvalidParameter(format!("sources parameter error: {}", e)))?;

        let style = params
            .get_string("style")
            .unwrap_or_else(|_| "APA".to_string());

        let mut citations = Vec::new();

        for (i, source) in sources.iter().enumerate() {
            if let Some(source_obj) = source.as_object() {
                let title = source_obj
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown Title");

                let author = source_obj
                    .get("author")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown Author");

                let url = source_obj.get("url").and_then(|v| v.as_str());

                let date = source_obj
                    .get("date")
                    .and_then(|v| v.as_str())
                    .unwrap_or("n.d.");

                // 根据风格生成引用
                let citation = match style.as_str() {
                    "APA" => {
                        if let Some(url) = url {
                            format!("{}. ({}). {}. Retrieved from {}", author, date, title, url)
                        } else {
                            format!("{}. ({}). {}.", author, date, title)
                        }
                    }
                    "MLA" => {
                        if let Some(url) = url {
                            format!("{}. \"{}.\" Web. {}.", author, title, date)
                        } else {
                            format!("{}. \"{}.\" {}.", author, title, date)
                        }
                    }
                    _ => format!("[{}] {}, \"{}\", {}", i + 1, author, title, date),
                };

                citations.push(citation);
            }
        }

        ToolResult::success(json!({
            "citations": citations,
            "style": style,
            "count": citations.len()
        }))
    }
}
