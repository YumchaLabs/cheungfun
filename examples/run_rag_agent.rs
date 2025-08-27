//! 运行RAG+Agent系统的完整示例
//!
//! 这个示例展示了一个完整的RAG+Agent智能问答系统，包括：
//! 1. 智能问题分类和路由
//! 2. 多种Agent策略（ReAct、多Agent协作）
//! 3. 专业化工具集成
//! 4. 交互式问答界面

use std::{
    collections::HashMap,
    io::{self, Write},
    sync::Arc,
    time::Instant,
};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use serde::{Deserialize, Serialize};

// Cheungfun核心组件
use cheungfun_core::{
    traits::{Embedder, Loader, NodeTransformer, Transformer, VectorStore},
    Result,
};
use cheungfun_indexing::{
    loaders::{DirectoryLoader, LoaderConfig},
    prelude::SplitterConfig,
    transformers::{MetadataExtractor, TextSplitter},
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    engine::{QueryEngine, QueryEngineBuilder},
    generator::SiumaiGenerator,
    retriever::VectorRetriever,
};

// Agent组件
use cheungfun_agents::{
    agent::{
        react::{ReActAgent, ReActConfig},
        base::{AgentContext, BaseAgent},
        builder::AgentBuilder,
    },
    tool::{ToolRegistry, builtin::*},
    types::*,
    llm::SiumaiLlmClient,
};

// Siumai LLM客户端
use siumai::prelude::*;

/// 问题类型分类
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuestionType {
    SimpleFactual,      // 简单事实查询："什么是RAG？"
    ComplexAnalysis,    // 复杂分析："比较RAG和传统搜索的优缺点"
    MultiDocument,      // 多文档对比："总结所有文档中关于AI的观点"
    Computational,      // 需要计算："计算这些数据的平均值"
    Conversational,     // 对话式："继续上一个话题"
}

/// 智能问题分类器
pub struct QuestionClassifier {
    llm_client: SiumaiLlmClient,
}

impl QuestionClassifier {
    pub fn new(llm_client: SiumaiLlmClient) -> Self {
        Self { llm_client }
    }
    
    pub async fn classify(&self, question: &str) -> Result<QuestionType> {
        let prompt = format!(
            r#"分析以下问题的类型，只返回对应的分类名称：

问题: "{}"

分类选项：
- SimpleFactual: 简单的事实查询，如"什么是RAG？"
- ComplexAnalysis: 需要复杂分析，如"比较RAG和传统搜索的优缺点"
- MultiDocument: 需要多文档对比，如"总结所有文档中关于AI的观点"
- Computational: 需要计算，如"计算平均值"
- Conversational: 对话式问题，如"继续上一个话题"

只返回分类名称："#,
            question
        );
        
        let messages = vec![AgentMessage::user(prompt)];
        let response = self.llm_client.chat(messages).await?;
        
        match response.trim() {
            "SimpleFactual" => Ok(QuestionType::SimpleFactual),
            "ComplexAnalysis" => Ok(QuestionType::ComplexAnalysis),
            "MultiDocument" => Ok(QuestionType::MultiDocument),
            "Computational" => Ok(QuestionType::Computational),
            "Conversational" => Ok(QuestionType::Conversational),
            _ => Ok(QuestionType::SimpleFactual), // 默认
        }
    }
}

/// RAG+Agent 智能问答系统
pub struct RagAgentSystem {
    // 核心RAG组件
    query_engine: QueryEngine,
    retriever: Arc<VectorRetriever>,
    
    // 智能分类器
    classifier: QuestionClassifier,
    
    // Agent组件
    react_agent: ReActAgent,
    
    // 工具注册表
    tool_registry: Arc<ToolRegistry>,
    
    // LLM客户端
    llm_client: SiumaiLlmClient,
}

impl RagAgentSystem {
    pub async fn new() -> Result<Self> {
        info!("🚀 初始化RAG+Agent智能问答系统...");
        
        // 1. 初始化基础组件
        info!("  📊 初始化嵌入器...");
        let embedder = Arc::new(FastEmbedder::new().await?);
        info!("    ✅ 嵌入器就绪 (维度: {})", embedder.dimension());
        
        info!("  🗄️ 初始化向量存储...");
        let vector_store = Arc::new(InMemoryVectorStore::new(embedder.dimension()));
        info!("    ✅ 向量存储就绪");
        
        info!("  🤖 初始化LLM客户端...");
        let llm_client = Self::create_llm_client().await?;
        info!("    ✅ LLM客户端就绪");
        
        // 2. 构建RAG索引
        info!("📚 构建文档索引...");
        let (query_engine, retriever) = Self::build_rag_index(
            embedder.clone(),
            vector_store.clone(),
            llm_client.clone(),
        ).await?;
        info!("  ✅ RAG索引构建完成");
        
        // 3. 初始化问题分类器
        let classifier = QuestionClassifier::new(llm_client.clone());
        
        // 4. 初始化工具注册表
        let mut tool_registry = ToolRegistry::new();
        
        // 注册内置工具
        tool_registry.register_tool(Arc::new(EchoTool::new()));
        tool_registry.register_tool(Arc::new(HttpTool::new()));
        tool_registry.register_tool(Arc::new(SearchTool::new()));
        tool_registry.register_tool(Arc::new(MathTool::new()));
        
        let tool_registry = Arc::new(tool_registry);
        
        // 5. 初始化ReAct Agent
        info!("  🧠 初始化ReAct Agent...");
        let react_config = ReActConfig::new("RAG-ReAct-Agent")
            .with_max_iterations(5)
            .with_include_trace(true);
        
        let mut react_agent = ReActAgent::new(react_config, tool_registry.clone());
        react_agent.set_llm_client(llm_client.clone());
        info!("    ✅ ReAct Agent就绪");
        
        Ok(Self {
            query_engine,
            retriever,
            classifier,
            react_agent,
            tool_registry,
            llm_client,
        })
    }
    
    /// 创建LLM客户端
    async fn create_llm_client() -> Result<SiumaiLlmClient> {
        // 尝试使用OpenAI，失败则使用Ollama
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            if !api_key.is_empty() {
                info!("    🌐 使用OpenAI GPT-4");
                let client = Siumai::builder()
                    .openai()
                    .api_key(&api_key)
                    .model("gpt-4")
                    .build()
                    .await?;
                return Ok(SiumaiLlmClient::new(client));
            }
        }
        
        info!("    🦙 使用本地Ollama");
        let client = Siumai::builder()
            .ollama()
            .base_url("http://localhost:11434")
            .model("llama3.2")
            .build()
            .await?;
        Ok(SiumaiLlmClient::new(client))
    }
    
    /// 构建RAG索引
    async fn build_rag_index(
        embedder: Arc<FastEmbedder>,
        vector_store: Arc<InMemoryVectorStore>,
        llm_client: SiumaiLlmClient,
    ) -> Result<(QueryEngine, Arc<VectorRetriever>)> {
        // 加载文档
        let loader_config = LoaderConfig::default();
        let loader = DirectoryLoader::new("./docs", loader_config);
        let documents = loader.load().await?;
        info!("  ✅ 加载了 {} 个文档", documents.len());
        
        // 文本分割
        let splitter_config = SplitterConfig {
            chunk_size: 500,
            chunk_overlap: 50,
            ..Default::default()
        };
        let text_splitter = TextSplitter::new(splitter_config);
        let metadata_extractor = MetadataExtractor::new();
        
        let mut all_nodes = Vec::new();
        for (i, document) in documents.iter().enumerate() {
            info!("  📄 处理文档 {}/{}: {}", 
                i + 1, 
                documents.len(),
                document.get_metadata_string("source")
                    .or_else(|| document.get_metadata_string("filename"))
                    .unwrap_or_else(|| format!("Document {}", i + 1))
            );
            
            let chunks = text_splitter.transform_document(document).await?;
            let nodes = metadata_extractor.transform_nodes(chunks).await?;
            all_nodes.extend(nodes);
        }
        
        info!("  📊 生成了 {} 个文本块", all_nodes.len());
        
        // 生成嵌入并存储
        for node in &all_nodes {
            let embedding = embedder.embed(&node.content).await?;
            vector_store.add_node(node.clone(), embedding).await?;
        }
        
        info!("  💾 存储了 {} 个节点", all_nodes.len());
        
        // 构建查询引擎
        let retriever = Arc::new(VectorRetriever::new(vector_store, embedder));
        let generator = SiumaiGenerator::new(llm_client);
        
        let query_engine = QueryEngineBuilder::new()
            .with_retriever(retriever.clone())
            .with_generator(Arc::new(generator))
            .build();
        
        Ok((query_engine, retriever))
    }
    
    /// 智能问答处理
    pub async fn intelligent_query(&self, question: &str) -> Result<String> {
        let start_time = Instant::now();
        
        info!("🤔 分析问题类型: {}", question);
        
        // 1. 问题分类
        let question_type = self.classifier.classify(question).await?;
        info!("  📋 问题类型: {:?}", question_type);
        
        // 2. 根据类型选择处理策略
        let response = match question_type {
            QuestionType::SimpleFactual => {
                info!("  🔍 使用直接RAG检索");
                self.simple_rag_query(question).await?
            }
            QuestionType::ComplexAnalysis => {
                info!("  🧠 使用ReAct推理Agent");
                self.complex_react_query(question).await?
            }
            QuestionType::MultiDocument => {
                info!("  📚 使用多文档分析");
                self.multi_document_query(question).await?
            }
            QuestionType::Computational => {
                info!("  🔧 使用工具增强Agent");
                self.tool_enhanced_query(question).await?
            }
            QuestionType::Conversational => {
                info!("  💬 使用对话式处理");
                self.conversational_query(question).await?
            }
        };
        
        let duration = start_time.elapsed();
        info!("⚡ 查询完成，耗时: {:?}", duration);
        
        Ok(response)
    }
    
    /// 简单RAG查询
    async fn simple_rag_query(&self, question: &str) -> Result<String> {
        let response = self.query_engine.query(question).await?;
        Ok(response.response)
    }
    
    /// 复杂ReAct查询
    async fn complex_react_query(&self, question: &str) -> Result<String> {
        let message = AgentMessage::user(question.to_string());
        let mut context = AgentContext::new();
        
        let response = self.react_agent.chat(message, Some(&mut context)).await?;
        Ok(response.content)
    }
    
    /// 多文档查询
    async fn multi_document_query(&self, question: &str) -> Result<String> {
        // 使用更大的top_k来获取更多文档
        let mut query = cheungfun_core::types::Query::new(question.to_string());
        query.top_k = 10;
        
        let results = self.retriever.retrieve(&query).await?;
        
        // 构建多文档分析提示
        let documents: Vec<String> = results.iter()
            .map(|scored_node| format!("文档片段 (相似度: {:.3}):\n{}", 
                scored_node.score, scored_node.node.content))
            .collect();
        
        let prompt = format!(
            r#"基于以下多个文档片段，回答问题："{}"

文档片段：
{}

请综合分析所有文档片段，提供全面的回答。如果不同文档有不同观点，请指出并进行对比。"#,
            question,
            documents.join("\n\n---\n\n")
        );
        
        let messages = vec![AgentMessage::user(prompt)];
        let response = self.llm_client.chat(messages).await?;
        
        Ok(response)
    }
    
    /// 工具增强查询
    async fn tool_enhanced_query(&self, question: &str) -> Result<String> {
        // 使用ReAct Agent处理，它会自动选择合适的工具
        self.complex_react_query(question).await
    }
    
    /// 对话式查询
    async fn conversational_query(&self, question: &str) -> Result<String> {
        // 简化实现，实际应该维护对话历史
        self.simple_rag_query(question).await
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    
    println!("🎯 RAG+Agent 智能问答系统");
    println!("=====================================");
    
    // 初始化系统
    let system = match RagAgentSystem::new().await {
        Ok(system) => system,
        Err(e) => {
            error!("❌ 系统初始化失败: {}", e);
            return Err(e);
        }
    };
    
    println!("\n💬 智能问答系统已就绪！");
    println!("特性：");
    println!("  🧠 智能问题分类 - 自动识别问题类型");
    println!("  🔍 多策略处理 - 根据问题选择最佳方案");
    println!("  🤖 Agent推理 - ReAct模式深度分析");
    println!("  🛠️ 工具增强 - 集成多种专业工具");
    println!("  📚 多文档分析 - 综合多个信息源");
    println!("\n提示：");
    println!("  - 输入问题开始智能对话");
    println!("  - 输入 'quit' 或 'exit' 退出");
    println!("  - 输入 'help' 查看示例问题");
    println!("==================================================\n");
    
    // 交互式问答循环
    loop {
        print!("🤔 您的问题: ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let question = input.trim();
        
        if question.is_empty() {
            continue;
        }
        
        if question == "quit" || question == "exit" {
            println!("👋 感谢使用RAG+Agent智能问答系统！");
            break;
        }
        
        if question == "help" {
            println!("\n📝 示例问题：");
            println!("  简单查询: \"什么是RAG？\"");
            println!("  复杂分析: \"比较RAG和传统搜索的优缺点\"");
            println!("  多文档: \"总结所有文档中关于AI的观点\"");
            println!("  计算类: \"计算2+3*4的结果\"");
            println!("  对话式: \"继续上一个话题\"\n");
            continue;
        }
        
        println!("🔍 正在智能分析和处理...");
        
        match system.intelligent_query(question).await {
            Ok(response) => {
                println!("\n🤖 AI回答:");
                println!("──────────────────────────────────────────────────");
                println!("{}", response);
                println!("──────────────────────────────────────────────────\n");
            }
            Err(e) => {
                error!("❌ 查询失败: {}", e);
                println!("抱歉，处理您的问题时出现了错误。请重试。\n");
            }
        }
    }
    
    Ok(())
}
