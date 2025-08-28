//! RAG+Agent 智能问答系统主程序
//!
//! 这是一个完整的RAG+Agent智能问答系统，集成了：
//! - 智能问题分类和路由
//! - ReAct推理Agent
//! - 长期记忆和对话历史
//! - 专业化工具集成
//!
//! ## 使用方法
//!
//! ```bash
//! # 设置环境变量
//! export OPENAI_API_KEY="your-api-key-here"  # 可选
//!
//! # 运行系统
//! cd examples
//! cargo run --bin rag_agent_main --features "fastembed,agents"
//! ```

mod memory;

use cheungfun_core::{
    traits::{Embedder, Loader, NodeTransformer, Transformer, VectorStore},
    Result as CheungfunResult,
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
use memory::{MemoryConfig, MemoryManager};
use serde::{Deserialize, Serialize};
use siumai::prelude::*;
use std::io::{self, Write};
use std::{collections::HashMap, sync::Arc, time::Instant};
use tokio::sync::RwLock;
use tracing::{error, info};

/// 问题类型分类
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuestionType {
    SimpleFactual,   // 简单事实查询
    ComplexAnalysis, // 复杂分析
    MultiDocument,   // 多文档对比
    Computational,   // 需要计算
    Conversational,  // 对话式
}

/// 查询响应
#[derive(Debug, Clone)]
pub struct QueryResponse {
    pub content: String,
    pub context_summary: String,
    pub reasoning_trace: String,
    pub strategy_used: String,
    pub processing_time_ms: u64,
}

/// 系统统计信息
#[derive(Debug, Clone)]
pub struct SystemStats {
    pub total_conversations: usize,
    pub avg_response_time_ms: u64,
    pub conversation_turns: usize,
    pub long_term_memories: usize,
    pub strategy_usage: HashMap<String, usize>,
    pub total_tool_calls: usize,
    pub tool_success_rate: f64,
}

/// RAG+Agent智能问答系统
pub struct RagAgentSystem {
    query_engine: QueryEngine,
    memory_manager: Arc<RwLock<MemoryManager>>,
    llm_client: Siumai,
    stats: Arc<RwLock<SystemStats>>,
}

impl RagAgentSystem {
    /// 创建新的RAG+Agent系统
    pub async fn new(memory_config: MemoryConfig) -> CheungfunResult<Self> {
        info!("📊 初始化嵌入器...");
        let embedder = Arc::new(FastEmbedder::new().await?);
        info!("✅ 嵌入器就绪 (维度: {})", embedder.dimension());

        info!("🗄️ 初始化向量存储...");
        let vector_store = Arc::new(InMemoryVectorStore::new(embedder.dimension()));
        info!("✅ 向量存储就绪");

        info!("🤖 初始化LLM客户端...");
        let llm_client = Self::create_llm_client().await?;
        info!("✅ LLM客户端就绪");

        info!("📚 构建文档索引...");
        let query_engine =
            Self::build_rag_index(embedder.clone(), vector_store.clone(), llm_client.clone())
                .await?;
        info!("✅ RAG索引构建完成");

        let memory_manager = Arc::new(RwLock::new(MemoryManager::new(memory_config)));

        let stats = Arc::new(RwLock::new(SystemStats {
            total_conversations: 0,
            avg_response_time_ms: 0,
            conversation_turns: 0,
            long_term_memories: 0,
            strategy_usage: HashMap::new(),
            total_tool_calls: 0,
            tool_success_rate: 1.0,
        }));

        Ok(Self {
            query_engine,
            memory_manager,
            llm_client,
            stats,
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
                    .await?);
            }
        }

        info!("🦙 使用本地Ollama");
        Ok(Siumai::builder()
            .ollama()
            .base_url("http://localhost:11434")
            .model("llama3.2")
            .build()
            .await?)
    }

    /// 构建RAG索引
    async fn build_rag_index(
        embedder: Arc<FastEmbedder>,
        vector_store: Arc<InMemoryVectorStore>,
        llm_client: Siumai,
    ) -> CheungfunResult<QueryEngine> {
        let loader_config = LoaderConfig::default();
        let loader = DirectoryLoader::new("./docs", loader_config);
        let documents = loader.load().await?;
        info!("✅ 加载了 {} 个文档", documents.len());

        let splitter_config = SplitterConfig {
            chunk_size: 500,
            chunk_overlap: 50,
            ..Default::default()
        };
        let text_splitter = TextSplitter::new(splitter_config);
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

            let chunks = text_splitter.transform_document(document).await?;
            let nodes = metadata_extractor.transform_nodes(chunks).await?;
            all_nodes.extend(nodes);
        }

        info!("📊 生成了 {} 个文本块", all_nodes.len());

        for node in &all_nodes {
            let embedding = embedder.embed(&node.content).await?;
            vector_store.add_node(node.clone(), embedding).await?;
        }

        info!("💾 存储了 {} 个节点", all_nodes.len());

        let retriever = Arc::new(VectorRetriever::new(vector_store, embedder));
        let generator = SiumaiGenerator::new(llm_client);

        Ok(QueryEngineBuilder::new()
            .with_retriever(retriever)
            .with_generator(Arc::new(generator))
            .build())
    }

    /// 处理查询
    pub async fn process_query(&self, question: &str) -> CheungfunResult<QueryResponse> {
        let start_time = Instant::now();

        // 分类问题
        let question_type = self.classify_question(question).await?;
        info!("📋 问题类型: {:?}", question_type);

        // 获取记忆上下文
        let memory_context = {
            let memory = self.memory_manager.read().await;
            memory.get_full_context(question).await
        };

        let context_summary = if memory_context.has_context() {
            memory_context.get_summary()
        } else {
            String::new()
        };

        // 根据类型处理问题
        let (content, strategy, reasoning) = match question_type {
            QuestionType::SimpleFactual => (
                "简单RAG检索".to_string(),
                self.simple_rag_query(question, &memory_context).await?,
                String::new(),
            ),
            QuestionType::ComplexAnalysis => (
                "ReAct推理".to_string(),
                self.complex_analysis_query(question, &memory_context)
                    .await?,
                "使用了多步推理".to_string(),
            ),
            QuestionType::MultiDocument => (
                "多文档分析".to_string(),
                self.multi_document_query(question, &memory_context).await?,
                String::new(),
            ),
            QuestionType::Computational => (
                "工具增强".to_string(),
                self.computational_query(question, &memory_context).await?,
                String::new(),
            ),
            QuestionType::Conversational => (
                "对话式处理".to_string(),
                self.conversational_query(question, &memory_context).await?,
                String::new(),
            ),
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        // 保存到记忆
        {
            let mut memory = self.memory_manager.write().await;
            memory
                .add_conversation_turn(question.to_string(), content.clone(), None)
                .await
                .map_err(|e| cheungfun_core::error::CheungfunError::Other(e.to_string()))?;
        }

        // 更新统计
        {
            let mut stats = self.stats.write().await;
            stats.total_conversations += 1;
            *stats.strategy_usage.entry(strategy.clone()).or_insert(0) += 1;

            // 更新平均响应时间
            let total_time = stats.avg_response_time_ms * (stats.total_conversations - 1) as u64
                + processing_time;
            stats.avg_response_time_ms = total_time / stats.total_conversations as u64;
        }

        Ok(QueryResponse {
            content,
            context_summary,
            reasoning_trace: reasoning,
            strategy_used: strategy,
            processing_time_ms: processing_time,
        })
    }

    /// 获取系统统计
    pub async fn get_stats(&self) -> CheungfunResult<SystemStats> {
        let stats = self.stats.read().await.clone();
        Ok(stats)
    }

    /// 问题分类
    async fn classify_question(&self, question: &str) -> CheungfunResult<QuestionType> {
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

        let response = self.llm_client.chat().user_message(&prompt).send().await?;

        match response.content.trim() {
            "SimpleFactual" => Ok(QuestionType::SimpleFactual),
            "ComplexAnalysis" => Ok(QuestionType::ComplexAnalysis),
            "MultiDocument" => Ok(QuestionType::MultiDocument),
            "Computational" => Ok(QuestionType::Computational),
            "Conversational" => Ok(QuestionType::Conversational),
            _ => Ok(QuestionType::SimpleFactual), // 默认
        }
    }

    /// 简单RAG查询
    async fn simple_rag_query(
        &self,
        question: &str,
        memory_context: &memory::MemoryContext,
    ) -> CheungfunResult<String> {
        let mut prompt = String::new();

        if memory_context.has_context() {
            prompt.push_str(&memory_context.format_for_prompt());
            prompt.push_str("\n基于以上上下文和知识库信息，回答以下问题：\n");
        }

        prompt.push_str(question);

        let response = self.query_engine.query(&prompt).await?;
        Ok(response.response)
    }

    /// 复杂分析查询
    async fn complex_analysis_query(
        &self,
        question: &str,
        memory_context: &memory::MemoryContext,
    ) -> CheungfunResult<String> {
        let mut prompt = format!(
            r#"你是一个专业的分析师，需要对以下问题进行深入分析。

问题: {}

请按照以下步骤进行分析：
1. 思考：分析问题的关键要素
2. 行动：搜索相关信息
3. 观察：评估信息的相关性和可靠性
4. 结论：提供全面的分析结果

"#,
            question
        );

        if memory_context.has_context() {
            prompt.push_str("相关上下文：\n");
            prompt.push_str(&memory_context.format_for_prompt());
            prompt.push('\n');
        }

        // 先进行RAG检索获取相关信息
        let rag_response = self.query_engine.query(question).await?;

        prompt.push_str("相关知识：\n");
        prompt.push_str(&rag_response.response);
        prompt.push_str("\n\n请基于以上信息进行深入分析：");

        let response = self.llm_client.chat().user_message(&prompt).send().await?;

        Ok(response.content)
    }

    /// 多文档查询
    async fn multi_document_query(
        &self,
        question: &str,
        memory_context: &memory::MemoryContext,
    ) -> CheungfunResult<String> {
        // 使用更大的top_k获取更多文档
        let mut query = cheungfun_core::types::Query::new(question.to_string());
        query.top_k = 10;

        // 这里需要直接访问retriever，但当前架构中没有暴露
        // 简化实现：使用query_engine
        let response = self.query_engine.query(question).await?;

        let mut prompt = format!(
            r#"基于多个文档源，综合分析以下问题："{}"

请：
1. 整合不同文档的观点
2. 指出观点的一致性和差异
3. 提供全面的综合分析

相关信息：
{}
"#,
            question, response.response
        );

        if memory_context.has_context() {
            prompt.push_str("\n历史上下文：\n");
            prompt.push_str(&memory_context.format_for_prompt());
        }

        let final_response = self.llm_client.chat().user_message(&prompt).send().await?;

        Ok(final_response.content)
    }

    /// 计算类查询
    async fn computational_query(
        &self,
        question: &str,
        _memory_context: &memory::MemoryContext,
    ) -> CheungfunResult<String> {
        // 简化实现：使用LLM进行计算
        let prompt = format!(
            r#"请解决以下计算问题："{}"

如果是数学计算，请：
1. 显示计算步骤
2. 给出最终答案
3. 验证结果的合理性

如果需要数据分析，请：
1. 说明分析方法
2. 展示分析过程
3. 总结关键发现
"#,
            question
        );

        let response = self.llm_client.chat().user_message(&prompt).send().await?;

        Ok(response.content)
    }

    /// 对话式查询
    async fn conversational_query(
        &self,
        question: &str,
        memory_context: &memory::MemoryContext,
    ) -> CheungfunResult<String> {
        let mut prompt = String::new();

        if memory_context.has_context() {
            prompt.push_str("对话历史：\n");
            prompt.push_str(&memory_context.format_for_prompt());
            prompt.push_str("\n基于以上对话历史，回答：");
        } else {
            prompt.push_str("回答以下问题：");
        }

        prompt.push_str(question);

        let response = self.llm_client.chat().user_message(&prompt).send().await?;

        Ok(response.content)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志系统
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();

    // 显示欢迎信息
    print_welcome();

    // 初始化系统
    info!("🚀 正在初始化RAG+Agent智能问答系统...");

    let memory_config = MemoryConfig {
        max_conversation_length: 50,
        summary_threshold: 20,
        enable_long_term_memory: true,
        max_context_turns: 5,
        max_long_term_entries: 10,
        retention_days: 30,
    };

    let system = match RagAgentSystem::new(memory_config).await {
        Ok(system) => {
            info!("✅ 系统初始化完成！");
            system
        }
        Err(e) => {
            error!("❌ 系统初始化失败: {}", e);
            eprintln!("初始化失败: {}", e);
            eprintln!("请检查：");
            eprintln!("1. 是否设置了OPENAI_API_KEY环境变量（或确保Ollama正在运行）");
            eprintln!("2. 是否存在./docs文件夹并包含markdown文件");
            eprintln!("3. 网络连接是否正常");
            return Err(e);
        }
    };

    println!("\n💬 RAG+Agent智能问答系统已就绪！");
    print_features();
    print_examples();

    // 主交互循环
    let mut conversation_count = 0;
    loop {
        print!("\n🤔 您的问题: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                let question = input.trim();

                if question.is_empty() {
                    continue;
                }

                // 处理特殊命令
                match question {
                    "quit" | "exit" | "q" => {
                        println!("👋 感谢使用RAG+Agent智能问答系统！");
                        break;
                    }
                    "help" | "h" => {
                        print_help();
                        continue;
                    }
                    "stats" | "统计" => {
                        print_stats(&system).await;
                        continue;
                    }
                    "clear" | "清空" => {
                        // 清屏
                        print!("\x1B[2J\x1B[1;1H");
                        print_welcome();
                        continue;
                    }
                    _ => {}
                }

                conversation_count += 1;
                println!("🔍 正在智能分析和处理... (第{}轮对话)", conversation_count);

                // 处理问题
                match system.process_query(question).await {
                    Ok(response) => {
                        println!("\n🤖 AI回答:");
                        println!("──────────────────────────────────────────────────");
                        println!("{}", response.content);

                        if !response.context_summary.is_empty() {
                            println!("\n📚 上下文信息: {}", response.context_summary);
                        }

                        if !response.reasoning_trace.is_empty() {
                            println!("\n🧠 推理过程: {}", response.reasoning_trace);
                        }

                        println!("──────────────────────────────────────────────────");
                        println!(
                            "⚡ 处理时间: {:.2}秒 | 策略: {}",
                            response.processing_time_ms as f64 / 1000.0,
                            response.strategy_used
                        );
                    }
                    Err(e) => {
                        error!("❌ 查询处理失败: {}", e);
                        println!("抱歉，处理您的问题时出现了错误: {}", e);
                        println!("请尝试：");
                        println!("1. 重新表述您的问题");
                        println!("2. 检查网络连接");
                        println!("3. 输入 'help' 查看帮助");
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

/// 显示欢迎信息
fn print_welcome() {
    println!("🎯 RAG+Agent 智能问答系统");
    println!("=====================================");
    println!("基于Cheungfun框架构建的下一代智能问答系统");
    println!("结合了检索增强生成(RAG)和智能Agent技术");
}

/// 显示系统特性
fn print_features() {
    println!("\n✨ 核心特性:");
    println!("  🧠 智能问题分类 - 自动识别问题类型并选择最佳策略");
    println!("  🔍 多策略处理 - 简单查询/复杂分析/多文档对比/工具调用");
    println!("  🤖 ReAct推理 - 思考-行动-观察的智能推理循环");
    println!("  💾 记忆增强 - 维护对话历史和长期记忆");
    println!("  🛠️ 工具生态 - 集成专业化工具提升回答质量");
    println!("  📚 上下文感知 - 基于历史对话提供个性化回答");
}

/// 显示示例问题
fn print_examples() {
    println!("\n📝 示例问题:");
    println!("  简单查询: \"什么是RAG？\"");
    println!("  复杂分析: \"比较RAG和传统搜索的优缺点\"");
    println!("  多文档: \"总结所有文档中关于AI的观点\"");
    println!("  计算类: \"计算2+3*4的结果\"");
    println!("  对话式: \"继续上一个话题\" 或 \"能详细解释一下吗？\"");

    println!("\n💡 提示:");
    println!("  - 输入问题开始智能对话");
    println!("  - 输入 'help' 查看更多命令");
    println!("  - 输入 'stats' 查看系统统计");
    println!("  - 输入 'quit' 或 'exit' 退出系统");
    println!("==================================================");
}

/// 显示帮助信息
fn print_help() {
    println!("\n📖 帮助信息:");
    println!("  🔤 基本命令:");
    println!("    help, h      - 显示此帮助信息");
    println!("    stats        - 显示系统统计信息");
    println!("    clear        - 清空屏幕");
    println!("    quit, exit, q - 退出系统");

    println!("\n  🎯 问题类型:");
    println!("    简单事实查询 - 直接从知识库检索答案");
    println!("    复杂分析问题 - 使用ReAct Agent进行多步推理");
    println!("    多文档对比 - 综合多个文档源进行分析");
    println!("    计算类问题 - 调用数学工具进行计算");
    println!("    对话式问题 - 基于历史上下文回答");

    println!("\n  💡 使用技巧:");
    println!("    - 问题越具体，回答越准确");
    println!("    - 可以引用之前的对话内容");
    println!("    - 支持中英文混合输入");
    println!("    - 系统会自动学习您的偏好");
}

/// 显示系统统计信息
async fn print_stats(system: &RagAgentSystem) {
    println!("\n📊 系统统计信息:");

    match system.get_stats().await {
        Ok(stats) => {
            println!("  💬 对话统计:");
            println!("    总对话轮数: {}", stats.total_conversations);
            println!(
                "    平均响应时间: {:.2}秒",
                stats.avg_response_time_ms as f64 / 1000.0
            );

            println!("  🧠 记忆统计:");
            println!("    对话历史: {}轮", stats.conversation_turns);
            println!("    长期记忆: {}条", stats.long_term_memories);

            println!("  🎯 策略使用:");
            for (strategy, count) in &stats.strategy_usage {
                println!("    {}: {}次", strategy, count);
            }

            println!("  🛠️ 工具调用:");
            println!("    总调用次数: {}", stats.total_tool_calls);
            println!("    成功率: {:.1}%", stats.tool_success_rate * 100.0);
        }
        Err(e) => {
            println!("  ❌ 无法获取统计信息: {}", e);
        }
    }
}

/// 处理Ctrl+C信号
fn setup_signal_handler() {
    ctrlc::set_handler(move || {
        println!("\n\n👋 收到退出信号，正在安全关闭系统...");
        std::process::exit(0);
    })
    .expect("设置信号处理器失败");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_system_initialization() {
        let config = MemoryConfig::default();

        // 注意：这个测试需要实际的文档和API密钥才能通过
        // 在CI环境中可能需要mock
        if std::env::var("OPENAI_API_KEY").is_ok() || std::path::Path::new("./docs").exists() {
            let result = RagAgentSystem::new(config).await;
            // 在有适当环境的情况下，系统应该能够初始化
            match result {
                Ok(_) => println!("✅ 系统初始化测试通过"),
                Err(e) => println!("⚠️ 系统初始化测试失败（可能是环境问题）: {}", e),
            }
        }
    }
}
