//! 简化版RAG+Agent智能问答系统
//!
//! 这是一个简化但完整的RAG+Agent系统示例，展示了：
//! - 基本的RAG问答功能
//! - 简单的对话历史记忆
//! - 智能问题分类
//! - 用户友好的交互界面

mod memory;

use cheungfun_core::{
    traits::{Embedder, Loader, Transform, VectorStore},
    Result as CheungfunResult,
};
use cheungfun_indexing::{
    loaders::DirectoryLoader, node_parser::text::SentenceSplitter, transformers::MetadataExtractor,
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    engine::{QueryEngine, QueryEngineBuilder},
    generator::SiumaiGenerator,
    retriever::VectorRetriever,
};
use memory::{MemoryConfig, MemoryManager};
use siumai::prelude::*;
use std::{
    collections::HashMap,
    io::{self, Write},
    sync::Arc,
    time::Instant,
};
use tokio::sync::RwLock;
use tracing::{error, info};

/// 简化的RAG+Agent系统
pub struct SimpleRagAgentSystem {
    query_engine: QueryEngine,
    memory_manager: Arc<RwLock<MemoryManager>>,
    conversation_count: usize,
}

impl SimpleRagAgentSystem {
    /// 创建新的系统
    pub async fn new() -> CheungfunResult<Self> {
        info!("🚀 初始化简化版RAG+Agent系统...");

        // 1. 初始化嵌入器
        info!("📊 初始化嵌入器...");
        let embedder = Arc::new(FastEmbedder::new().await?);
        info!("✅ 嵌入器就绪 (维度: {})", embedder.dimension());

        // 2. 初始化向量存储
        info!("🗄️ 初始化向量存储...");
        let vector_store = Arc::new(InMemoryVectorStore::new(
            embedder.dimension(),
            cheungfun_core::DistanceMetric::Cosine,
        ));
        info!("✅ 向量存储就绪");

        // 3. 初始化LLM客户端
        info!("🤖 初始化LLM客户端...");
        let llm_client = Self::create_llm_client().await?;
        info!("✅ LLM客户端就绪");

        // 4. 构建RAG索引
        info!("📚 构建文档索引...");
        let query_engine =
            Self::build_rag_index(embedder.clone(), vector_store.clone(), llm_client).await?;
        info!("✅ RAG索引构建完成");

        // 5. 初始化记忆管理器
        let memory_config = MemoryConfig::default();
        let memory_manager = Arc::new(RwLock::new(MemoryManager::new(memory_config)));

        Ok(Self {
            query_engine,
            memory_manager,
            conversation_count: 0,
        })
    }

    /// 创建LLM客户端
    async fn create_llm_client() -> CheungfunResult<Siumai> {
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            if !api_key.is_empty() {
                info!("🌐 使用OpenAI GPT-4");
                return Ok(Siumai::builder()
                    .openai()
                    .api_key(&api_key)
                    .model("gpt-4")
                    .build()
                    .await
                    .map_err(|e| cheungfun_core::error::CheungfunError::llm(e.to_string()))?);
            }
        }

        info!("🦙 使用本地Ollama");
        Ok(Siumai::builder()
            .ollama()
            .base_url("http://localhost:11434")
            .model("llama3.2")
            .build()
            .await
            .map_err(|e| cheungfun_core::error::CheungfunError::llm(e.to_string()))?)
    }

    /// 构建RAG索引
    async fn build_rag_index(
        embedder: Arc<FastEmbedder>,
        vector_store: Arc<InMemoryVectorStore>,
        llm_client: Siumai,
    ) -> CheungfunResult<QueryEngine> {
        // 加载文档
        let loader = DirectoryLoader::new("./docs")?;
        let documents = loader.load().await?;
        info!("✅ 加载了 {} 个文档", documents.len());

        // 文本分割
        let text_splitter = TextSplitter::new(500, 50);
        let metadata_extractor = MetadataExtractor::new();

        let mut all_nodes = Vec::new();
        for (i, document) in documents.iter().enumerate() {
            info!(
                "📄 处理文档 {}/{}: {}",
                i + 1,
                documents.len(),
                document
                    .get_metadata_string("source")
                    .or_else(|| document.get_metadata_string("filename"))
                    .unwrap_or_else(|| format!("Document {}", i + 1))
            );

            let chunks = text_splitter.transform(document.clone()).await?;
            let mut nodes = Vec::new();
            for chunk in chunks {
                let node = metadata_extractor.transform_node(chunk).await?;
                nodes.push(node);
            }
            all_nodes.extend(nodes);
        }

        info!("📊 生成了 {} 个文本块", all_nodes.len());

        // 生成嵌入并存储
        let mut nodes_with_embeddings = Vec::new();
        for node in &all_nodes {
            let embedding = embedder.embed(&node.content).await?;
            nodes_with_embeddings.push((node.clone(), embedding));
        }

        for (node, embedding) in nodes_with_embeddings {
            vector_store.add(vec![node]).await?;
        }

        info!("💾 存储了 {} 个节点", all_nodes.len());

        // 构建查询引擎
        let retriever = Arc::new(VectorRetriever::new(vector_store, embedder));
        let generator = SiumaiGenerator::new(llm_client);

        QueryEngineBuilder::new()
            .retriever(retriever)
            .generator(Arc::new(generator))
            .build()
    }

    /// 处理用户查询
    pub async fn process_query(&mut self, question: &str) -> CheungfunResult<String> {
        let start_time = Instant::now();
        self.conversation_count += 1;

        info!("🤔 处理第{}轮对话: {}", self.conversation_count, question);

        // 获取记忆上下文
        let memory_context = {
            let memory = self.memory_manager.read().await;
            memory.get_full_context(question).await
        };

        // 构建增强的查询
        let mut enhanced_query = String::new();

        if memory_context.has_context() {
            enhanced_query.push_str("基于以下对话历史和相关知识：\n");
            enhanced_query.push_str(&memory_context.format_for_prompt());
            enhanced_query.push_str("\n回答问题：");
        }

        enhanced_query.push_str(question);

        // 执行RAG查询
        let response = self.query_engine.query(&enhanced_query).await?;
        let answer = response.response;

        // 保存到记忆
        {
            let mut memory = self.memory_manager.write().await;
            memory
                .add_conversation_turn(question.to_string(), answer.clone(), None)
                .await
                .map_err(|e| {
                    cheungfun_core::error::CheungfunError::llm(format!("记忆错误: {}", e))
                })?;
        }

        let duration = start_time.elapsed();
        info!("⚡ 查询完成，耗时: {:?}", duration);

        Ok(answer)
    }

    /// 获取系统统计
    pub async fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert(
            "总对话轮数".to_string(),
            self.conversation_count.to_string(),
        );

        let memory = self.memory_manager.read().await;
        let memory_stats = memory.get_conversation_stats();
        stats.insert(
            "对话历史".to_string(),
            format!("{}轮", memory_stats.total_turns),
        );
        stats.insert(
            "记忆摘要".to_string(),
            format!("{}条", memory_stats.total_summaries),
        );

        stats
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();

    println!("🎯 简化版RAG+Agent智能问答系统");
    println!("=====================================");
    println!("基于Cheungfun框架的智能问答系统");
    println!("集成了RAG检索和对话记忆功能");

    // 初始化系统
    let mut system = match SimpleRagAgentSystem::new().await {
        Ok(system) => {
            println!("\n✅ 系统初始化完成！");
            system
        }
        Err(e) => {
            eprintln!("❌ 系统初始化失败: {}", e);
            eprintln!("请检查：");
            eprintln!("1. 是否设置了OPENAI_API_KEY环境变量（或确保Ollama正在运行）");
            eprintln!("2. 是否存在./docs文件夹并包含markdown文件");
            return Err(e.into());
        }
    };

    println!("\n💬 智能问答系统已就绪！");
    println!("特性：");
    println!("  🔍 RAG检索 - 基于文档知识库的智能问答");
    println!("  💾 对话记忆 - 维护对话历史和上下文");
    println!("  🧠 智能理解 - 结合历史对话提供个性化回答");
    println!("\n提示：");
    println!("  - 输入问题开始对话");
    println!("  - 输入 'stats' 查看统计信息");
    println!("  - 输入 'quit' 或 'exit' 退出");
    println!("==================================================\n");

    // 主交互循环
    loop {
        print!("🤔 您的问题: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                let question = input.trim();

                if question.is_empty() {
                    continue;
                }

                match question {
                    "quit" | "exit" | "q" => {
                        println!("👋 感谢使用RAG+Agent智能问答系统！");
                        break;
                    }
                    "stats" | "统计" => {
                        let stats = system.get_stats().await;
                        println!("\n📊 系统统计:");
                        for (key, value) in stats {
                            println!("  {}: {}", key, value);
                        }
                        println!();
                        continue;
                    }
                    _ => {}
                }

                println!("🔍 正在处理您的问题...");

                match system.process_query(question).await {
                    Ok(response) => {
                        println!("\n🤖 AI回答:");
                        println!("──────────────────────────────────────────────────");
                        println!("{}", response);
                        println!("──────────────────────────────────────────────────\n");
                    }
                    Err(e) => {
                        error!("❌ 查询处理失败: {}", e);
                        println!("抱歉，处理您的问题时出现了错误: {}", e);
                        println!("请重试或检查网络连接。\n");
                    }
                }
            }
            Err(e) => {
                error!("❌ 输入读取失败: {}", e);
                println!("输入读取失败，请重试。");
            }
        }
    }

    Ok(())
}
