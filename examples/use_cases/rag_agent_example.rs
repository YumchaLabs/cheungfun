//! RAG+Agent 完善问答系统示例
//!
//! 这个示例展示如何结合RAG和Agent构建智能问答系统：
//! 1. 智能问题分类和路由
//! 2. ReAct推理Agent进行复杂分析
//! 3. 多Agent协作处理复杂任务
//! 4. 工具增强的RAG检索
//!
//! ## 使用方法
//!
//! ```bash
//! # 设置API密钥
//! export OPENAI_API_KEY="your-api-key-here"
//!
//! # 运行示例
//! cargo run --example rag_agent_example --features "fastembed,agents"
//! ```

use cheungfun_agents::{
    agent::{
        base::{AgentContext, BaseAgent},
        builder::AgentBuilder,
        multi_agent::{HandoffStrategy, MultiAgentConfig, MultiAgentSystem},
        react::{ReActAgent, ReActConfig},
    },
    llm::SiumaiLlmClient,
    tool::{builtin::*, ToolRegistry},
    types::*,
};
use cheungfun_core::{
    traits::{Embedder, Loader, Transform, TransformInput, VectorStore},
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
use serde::{Deserialize, Serialize};
use siumai::prelude::*;
use std::{
    collections::HashMap,
    io::{self, Write},
    sync::Arc,
    time::Instant,
};
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// 问题类型分类
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuestionType {
    SimpleFactual,   // 简单事实查询
    ComplexAnalysis, // 复杂分析
    MultiDocument,   // 多文档对比
    Computational,   // 需要计算
    Conversational,  // 对话式
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
            r#"分析以下问题的类型，返回对应的分类：

问题: "{}"

分类选项：
1. SimpleFactual - 简单的事实查询，如"什么是RAG？"
2. ComplexAnalysis - 需要复杂分析，如"比较RAG和传统搜索的优缺点"
3. MultiDocument - 需要多文档对比，如"总结所有文档中关于AI的观点"
4. Computational - 需要计算，如"计算平均值"
5. Conversational - 对话式问题，如"继续上一个话题"

只返回分类名称，不要其他内容。"#,
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

/// RAG专用工具
pub struct RagVectorSearchTool {
    retriever: Arc<VectorRetriever>,
}

impl RagVectorSearchTool {
    pub fn new(retriever: Arc<VectorRetriever>) -> Self {
        Self { retriever }
    }
}

/// RAG+Agent 完善问答系统
pub struct RagAgentSystem {
    // 核心组件
    query_engine: QueryEngine,
    classifier: QuestionClassifier,

    // Agent组件
    react_agent: ReActAgent,
    multi_agent_system: MultiAgentSystem,

    // 工具注册表
    tool_registry: Arc<ToolRegistry>,
}

impl RagAgentSystem {
    pub async fn new() -> Result<Self> {
        info!("🚀 初始化RAG+Agent系统...");

        // 1. 初始化基础RAG组件
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
        let (query_engine, retriever) =
            Self::build_rag_index(embedder.clone(), vector_store.clone(), llm_client.clone())
                .await?;
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

        // 注册RAG专用工具
        let rag_search_tool = Arc::new(RagVectorSearchTool::new(retriever));
        // tool_registry.register_tool(rag_search_tool); // 需要实现Tool trait

        let tool_registry = Arc::new(tool_registry);

        // 5. 初始化ReAct Agent
        info!("  🧠 初始化ReAct Agent...");
        let react_config = ReActConfig::new("RAG-ReAct-Agent")
            .with_max_iterations(5)
            .with_include_trace(true);

        let mut react_agent = ReActAgent::new(react_config, tool_registry.clone());
        react_agent.set_llm_client(llm_client.clone());
        info!("    ✅ ReAct Agent就绪");

        // 6. 初始化多Agent系统
        info!("  👥 初始化多Agent系统...");
        let multi_agent_config = MultiAgentConfig {
            max_handoffs: 3,
            handoff_strategy: HandoffStrategy::Sequential,
            enable_parallel_execution: false,
            coordination_timeout_ms: 30000,
        };

        let multi_agent_system = MultiAgentSystem::new(multi_agent_config);

        // 添加专业化Agent
        let doc_analyzer = AgentBuilder::new("DocumentAnalyzer")
            .with_description("专门分析单个文档内容的Agent")
            .with_tools(tool_registry.clone())
            .build();

        let summarizer = AgentBuilder::new("Summarizer")
            .with_description("专门进行内容总结的Agent")
            .with_tools(tool_registry.clone())
            .build();

        // multi_agent_system.add_agent(doc_analyzer, AgentRole::new("analyzer")).await?;
        // multi_agent_system.add_agent(summarizer, AgentRole::new("summarizer")).await?;

        info!("    ✅ 多Agent系统就绪");

        Ok(Self {
            query_engine,
            classifier,
            react_agent,
            multi_agent_system,
            tool_registry,
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

        // 文本分割 - 使用统一Transform接口
        let text_splitter = SentenceSplitter::from_defaults(500, 50)?;
        let metadata_extractor = MetadataExtractor::new();

        let mut all_nodes = Vec::new();
        for (i, document) in documents.iter().enumerate() {
            info!("  📄 处理文档 {}/{}", i + 1, documents.len());

            // 使用统一Transform接口
            let input = TransformInput::Document(document.clone());
            let chunks = text_splitter.transform(input).await?;
            let input = TransformInput::Nodes(chunks);
            let nodes = metadata_extractor.transform(input).await?;
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
                info!("  👥 使用多Agent协作");
                self.multi_agent_query(question).await?
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

    /// 多Agent协作查询
    async fn multi_agent_query(&self, question: &str) -> Result<String> {
        let message = AgentMessage::user(question.to_string());
        let response = self.multi_agent_system.execute(message).await?;

        // 整合多个Agent的响应
        let combined_response = response
            .iter()
            .map(|r| r.content.clone())
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        Ok(combined_response)
    }

    /// 工具增强查询
    async fn tool_enhanced_query(&self, question: &str) -> Result<String> {
        // 使用带工具的Agent处理
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

    // 初始化系统
    let system = RagAgentSystem::new().await?;

    println!("🎯 RAG+Agent 智能问答系统已就绪！");
    println!("提示：");
    println!("  - 输入问题开始智能对话");
    println!("  - 系统会自动分析问题类型并选择最佳处理策略");
    println!("  - 输入 'quit' 或 'exit' 退出");
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
            println!("👋 再见！");
            break;
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
