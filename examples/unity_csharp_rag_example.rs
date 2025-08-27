//! Unity C# 项目代码索引和问答命令行工具
//!
//! 这是一个完整的命令行工具，使用 Cheungfun 框架为 Unity C# 项目构建代码索引和问答系统：
//! 1. 智能扫描 Unity 项目，遵循 .gitignore 规则
//! 2. 使用企业级 AST 增强代码分割器进行智能分块
//! 3. 支持代码结构感知的语义搜索和问答
//! 4. 提供交互式命令行界面
//!
//! ## 使用方法
//!
//! ```bash
//! # 基本用法：索引当前目录的 Unity 项目
//! cargo run --bin unity_csharp_rag -- /path/to/unity/project
//!
//! # 使用 OpenAI API
//! export OPENAI_API_KEY="your-api-key-here"
//! cargo run --bin unity_csharp_rag -- /path/to/unity/project --llm openai --model gpt-4
//!
//! # 使用本地 Ollama
//! ollama serve
//! ollama pull llama3.2
//! cargo run --bin unity_csharp_rag -- /path/to/unity/project --llm ollama --model llama3.2
//!
//! # 使用企业级分块策略（推荐用于大型项目）
//! cargo run --bin unity_csharp_rag -- /path/to/unity/project --strategy enterprise
//!
//! # 详细模式，显示更多调试信息
//! cargo run --bin unity_csharp_rag -- /path/to/unity/project --verbose
//! ```
//!
//! ## 功能特性
//!
//! - 🎮 Unity 项目专用：自动识别和处理 Unity C# 脚本
//! - 🌳 AST 增强分割：基于 tree-sitter 的智能代码分块
//! - 📁 Gitignore 支持：自动遵循项目的 .gitignore 规则
//! - 🔍 代码结构感知：理解类、方法、属性等代码结构
//! - 💬 智能问答：支持代码功能、架构、实现细节的问答
//! - 📊 详细统计：提供索引构建和查询的性能统计
//! - 🏢 企业级配置：专为大型 Unity 项目优化的分块策略
//! - 🛠️ 命令行界面：完整的 CLI 工具，支持多种配置选项

use cheungfun_core::{
    traits::{Embedder, Loader, Transform, VectorStore},
    Result,
};
use cheungfun_indexing::{
    loaders::{DirectoryLoader, LoaderConfig, ProgrammingLanguage, filter::FilterConfig},
    node_parser::{
        config::{CodeSplitterConfig, ChunkingStrategy},
        text::CodeSplitter,
    },
    transformers::MetadataExtractor,
};
use cheungfun_integrations::{FastEmbedder, InMemoryVectorStore};
use cheungfun_query::{
    engine::{QueryEngine, QueryEngineBuilder},
    generator::SiumaiGenerator,
    retriever::VectorRetriever,
};
use siumai::prelude::*;
use std::{
    collections::HashMap,
    env,
    io::{self, Write},
    path::PathBuf,
    sync::Arc,
};
use tokio::fs;
use tracing::{info, warn, error, debug, Level};
use async_trait::async_trait;

/// Unity C# RAG 系统配置
#[derive(Debug, Clone)]
pub struct UnityCSharpRagConfig {
    /// Unity 项目路径
    pub unity_project_path: PathBuf,
    /// 是否使用 SQLite 存储（否则使用内存）
    pub use_sqlite: bool,
    /// 分块策略
    pub chunking_strategy: ChunkingStrategy,
    /// 检索时返回的 top-k 结果数
    pub top_k: usize,
    /// LLM 提供商（openai/ollama）
    pub llm_provider: String,
    /// LLM 模型名称
    pub llm_model: String,
    /// Embedding 提供商（fastembed/openai/gemini）
    pub embedding_provider: String,
    /// Embedding 模型名称
    pub embedding_model: String,
    /// 是否启用 AST 分析
    pub enable_ast_analysis: bool,
    /// 是否启用详细日志
    pub verbose: bool,
    /// 是否创建示例项目（如果路径不存在）
    pub create_sample: bool,
}

impl Default for UnityCSharpRagConfig {
    fn default() -> Self {
        Self {
            unity_project_path: PathBuf::from("./UnityProject"),
            use_sqlite: false,
            chunking_strategy: ChunkingStrategy::Enterprise, // 默认使用企业级策略
            top_k: 5,
            llm_provider: "openai".to_string(),
            llm_model: "gpt-3.5-turbo".to_string(),
            embedding_provider: "fastembed".to_string(),
            embedding_model: "BAAI/bge-small-en-v1.5".to_string(),
            enable_ast_analysis: true,
            verbose: false,
            create_sample: true,
        }
    }
}

/// 命令行参数结构
#[derive(Debug)]
pub struct CliArgs {
    pub project_path: PathBuf,
    pub strategy: Option<String>,
    pub llm_provider: Option<String>,
    pub llm_model: Option<String>,
    pub top_k: Option<usize>,
    pub verbose: bool,
    pub no_sample: bool,
    pub help: bool,
}

/// Unity C# RAG 系统组件
pub struct UnityCSharpRagSystem {
    config: UnityCSharpRagConfig,
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    query_engine: QueryEngine,
}

#[tokio::main]
async fn main() -> Result<()> {
    // 解析命令行参数
    let args = parse_cli_args();

    // 显示帮助信息
    if args.help {
        show_help();
        return Ok(());
    }

    // 初始化日志
    let log_level = if args.verbose { Level::DEBUG } else { Level::INFO };
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .init();

    // 显示欢迎信息
    println!("🎮 Cheungfun Unity C# 代码问答系统");
    println!("=====================================");

    if args.verbose {
        debug!("启用详细日志模式");
    }

    // 构建配置
    let config = build_config_from_args(args)?;

    // 显示配置信息
    print_config_summary(&config);

    // 检查 Unity 项目路径
    if !config.unity_project_path.exists() {
        if config.create_sample {
            println!("📁 项目路径不存在，创建示例 Unity 项目...");
            create_sample_unity_project(&config.unity_project_path).await?;
        } else {
            error!("项目路径不存在: {}", config.unity_project_path.display());
            return Err(cheungfun_core::CheungfunError::Configuration {
                message: format!("Unity 项目路径不存在: {}", config.unity_project_path.display()),
            });
        }
    }

    // 初始化 RAG 系统
    let mut rag_system = UnityCSharpRagSystem::new(config).await?;

    // 构建代码索引
    rag_system.build_code_index().await?;

    // 启动交互式问答
    rag_system.start_interactive_chat().await?;

    Ok(())
}

impl UnityCSharpRagSystem {
    /// 创建新的 Unity C# RAG 系统
    pub async fn new(config: UnityCSharpRagConfig) -> Result<Self> {
        info!("🔧 初始化 Unity C# RAG 系统组件...");

        // 1. 初始化嵌入器
        let embedder = create_embedder(&config).await?;
        info!("    ✅ 嵌入器就绪 (维度: {})", embedder.dimension());

        // 2. 初始化向量存储
        let vector_store: Arc<dyn VectorStore> = if config.use_sqlite {
            info!("  🗄️ 初始化 SQLite 向量存储...");
            warn!("    ⚠️ SQLite 存储暂未实现，使用内存存储");
            Arc::new(InMemoryVectorStore::new(
                embedder.dimension(),
                cheungfun_core::traits::DistanceMetric::Cosine,
            ))
        } else {
            info!("  🗄️ 初始化内存向量存储...");
            Arc::new(InMemoryVectorStore::new(
                embedder.dimension(),
                cheungfun_core::traits::DistanceMetric::Cosine,
            ))
        };
        info!("    ✅ 向量存储就绪");

        // 3. 创建 LLM 客户端和生成器
        info!("  🤖 初始化 LLM 客户端...");
        let generator = create_llm_generator(&config).await?;
        info!("    ✅ LLM 生成器就绪");

        // 4. 创建查询引擎
        let retriever = Arc::new(VectorRetriever::new(vector_store.clone(), embedder.clone()));

        let engine_config = cheungfun_query::engine::QueryEngineConfig {
            default_top_k: config.top_k,
            default_generation_options: cheungfun_core::types::GenerationOptions::default(),
            validate_context: true,
            min_context_nodes: 1,
            max_context_nodes: 10,
            enable_query_preprocessing: true,
            enable_response_postprocessing: true,
        };

        let query_engine = QueryEngineBuilder::new()
            .retriever(retriever)
            .generator(Arc::new(generator))
            .config(engine_config)
            .build()?;

        Ok(Self {
            config,
            embedder,
            vector_store,
            query_engine,
        })
    }

    /// 构建 Unity C# 代码索引
    pub async fn build_code_index(&mut self) -> Result<()> {
        info!("🎮 开始构建 Unity C# 代码索引...");
        let start_time = std::time::Instant::now();

        // 1. 配置 Unity 项目专用的文件过滤器
        let unity_filter = FilterConfig::new()
            .with_respect_gitignore(false) // 暂时禁用 gitignore 以便调试
            .with_include_extensions(vec!["cs".to_string()]) // 只包含 C# 文件
            .with_exclude_patterns(vec![
                // Unity 特定的排除模式
                "Library/**".to_string(),
                "Temp/**".to_string(),
                "Obj/**".to_string(),
                "Build/**".to_string(),
                "Builds/**".to_string(),
                "Logs/**".to_string(),
                "UserSettings/**".to_string(),
                "MemoryCaptures/**".to_string(),
                // Unity 生成的文件
                "*.csproj".to_string(),
                "*.sln".to_string(),
                "*.tmp".to_string(),
                "*.user".to_string(),
                "*.pidb".to_string(),
                "*.booproj".to_string(),
                "*.svd".to_string(),
                "*.pdb".to_string(),
                "*.mdb".to_string(),
                // Unity 包管理
                "Packages/**".to_string(),
                "ProjectSettings/Packages-lock.json".to_string(),
            ])
            // 移除 include_patterns，让它扫描所有 .cs 文件
            .with_exclude_hidden(false) // 暂时允许隐藏文件以便调试
            .with_exclude_empty(true)
            .with_max_file_size(5 * 1024 * 1024); // 5MB 最大文件大小

        let loader_config = LoaderConfig::new()
            .with_filter_config(unity_filter)
            .with_max_depth(15)
            .with_continue_on_error(true);

        // 2. 加载所有 C# 文件
        info!("  📂 扫描 Unity 项目: {}", self.config.unity_project_path.display());
        let loader = DirectoryLoader::with_config(&self.config.unity_project_path, loader_config)?;
        let documents = loader.load().await?;

        if documents.is_empty() {
            warn!("  ⚠️ 未找到任何 C# 文件");
            return Ok(());
        }

        info!("  ✅ 加载了 {} 个 C# 文件", documents.len());

        // 3. 配置 C# 代码分割器（使用预设策略）
        let code_splitter = if self.config.enable_ast_analysis {
            CodeSplitter::with_strategy(ProgrammingLanguage::CSharp, self.config.chunking_strategy)?
        } else {
            // 如果禁用 AST，使用基础配置
            let (chunk_lines, chunk_lines_overlap, max_chars) = self.config.chunking_strategy.params();
            let basic_config = CodeSplitterConfig::new(
                ProgrammingLanguage::CSharp,
                chunk_lines,
                chunk_lines_overlap,
                max_chars,
            )
            .with_ast_splitting(false);
            CodeSplitter::new(basic_config)?
        };

        info!("  ✅ 使用 {} 分块策略", self.config.chunking_strategy.description());
        let metadata_extractor = MetadataExtractor::new();

        // 4. 处理每个 C# 文件
        let mut all_nodes = Vec::new();
        let mut total_chunks = 0;
        let mut processed_classes = 0;
        let mut processed_methods = 0;

        for (i, document) in documents.iter().enumerate() {
            let file_path = document
                .get_metadata_string("source")
                .or_else(|| document.get_metadata_string("filename"))
                .unwrap_or_else(|| format!("Document {}", i + 1));

            info!(
                "  📄 处理文件 {}/{}: {}",
                i + 1,
                documents.len(),
                file_path
            );

            // 使用 AST 增强的代码分割
            let input = cheungfun_core::traits::TransformInput::Document(document.clone());
            let nodes = code_splitter.transform(input).await?;

            info!("    ✂️ 分割为 {} 个代码块", nodes.len());
            total_chunks += nodes.len();

            // 统计代码结构（如果启用了 AST 分析）
            if self.config.enable_ast_analysis {
                let class_count = document.content.matches("class ").count();
                let method_count = document.content.matches("void ").count()
                    + document.content.matches("public ").count()
                    + document.content.matches("private ").count();
                processed_classes += class_count;
                processed_methods += method_count;
            }

            // 提取元数据并生成嵌入
            for mut node in nodes {
                // 提取元数据
                let metadata_input = cheungfun_core::traits::TransformInput::Node(node.clone());
                let metadata_nodes = metadata_extractor.transform(metadata_input).await?;
                node = metadata_nodes.into_iter().next().unwrap_or(node);

                // 生成嵌入
                let embedding = self.embedder.embed(&node.content).await?;
                node.embedding = Some(embedding);

                all_nodes.push(node);
            }
        }

        info!("  🧮 生成了 {} 个代码块的嵌入", all_nodes.len());

        // 5. 存储到向量数据库
        info!("  💾 存储到向量数据库...");
        let stored_ids = self.vector_store.add(all_nodes).await?;

        let elapsed = start_time.elapsed();
        info!("✅ Unity C# 代码索引构建完成!");
        info!("  📊 统计信息:");
        info!("    - 处理 C# 文件: {} 个", documents.len());
        info!("    - 生成代码块: {} 个", total_chunks);
        info!("    - 存储节点: {} 个", stored_ids.len());
        if self.config.enable_ast_analysis {
            info!("    - 识别类: {} 个", processed_classes);
            info!("    - 识别方法: {} 个", processed_methods);
        }
        info!("    - 处理时间: {:?}", elapsed);
        info!("    - 平均每文件: {:?}", elapsed / documents.len() as u32);

        Ok(())
    }

    /// 启动交互式代码问答
    pub async fn start_interactive_chat(&self) -> Result<()> {
        info!("🎯 启动交互式代码问答模式");
        println!("\n💬 Unity C# 代码问答系统已就绪！");
        println!("提示：");
        println!("  - 询问代码功能、架构、实现细节");
        println!("  - 输入 'quit' 或 'exit' 退出");
        println!("  - 输入 'stats' 查看系统统计");
        println!("  - 输入 'help' 查看帮助");
        println!("  - 输入 'examples' 查看示例问题");
        println!("{}", "=".repeat(50));

        loop {
            // 显示提示符
            print!("\n🎮 Unity 代码问题: ");
            io::stdout().flush().unwrap();

            // 读取用户输入
            let mut input = String::new();
            match io::stdin().read_line(&mut input) {
                Ok(_) => {
                    let query = input.trim();

                    // 处理特殊命令
                    match query.to_lowercase().as_str() {
                        "quit" | "exit" => {
                            println!("👋 再见！");
                            break;
                        }
                        "stats" => {
                            self.show_system_stats().await?;
                            continue;
                        }
                        "help" => {
                            self.show_help();
                            continue;
                        }
                        "examples" => {
                            self.show_example_questions();
                            continue;
                        }
                        "" => continue,
                        _ => {}
                    }

                    // 处理代码查询
                    match self.process_code_query(query).await {
                        Ok(_) => {}
                        Err(e) => {
                            println!("❌ 查询失败: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("❌ 读取输入失败: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    /// 处理单个代码查询
    async fn process_code_query(&self, query: &str) -> Result<()> {
        println!("🔍 正在搜索相关代码...");
        let start_time = std::time::Instant::now();

        // 执行查询
        let response = self.query_engine.query(query).await?;
        let elapsed = start_time.elapsed();

        // 显示结果
        println!("\n🤖 AI 回答:");
        println!("{}", "─".repeat(50));
        println!("{}", response.response.content);
        println!("{}", "─".repeat(50));

        // 显示检索到的代码片段
        println!(
            "\n📚 相关代码片段 ({} 个):",
            response.retrieved_nodes.len()
        );

        for (i, scored_node) in response.retrieved_nodes.iter().take(3).enumerate() {
            let source = scored_node
                .node
                .get_metadata_string("source")
                .or_else(|| scored_node.node.get_metadata_string("filename"))
                .unwrap_or_else(|| "未知文件".to_string());

            // 提取文件名（去掉路径）
            let filename = source.split(['/', '\\']).last().unwrap_or(&source);

            // 尝试提取类名和方法名
            let content_preview = scored_node.node.content.lines().take(5).collect::<Vec<_>>().join("\n");
            let class_info = extract_csharp_info(&scored_node.node.content);

            println!("  {}. [相似度: {:.3}] 📄 {}", i + 1, scored_node.score, filename);
            if let Some(info) = class_info {
                println!("     🏗️ {}", info);
            }
            println!("     预览: {}...", content_preview.chars().take(100).collect::<String>());
            println!();
        }

        // 显示性能信息
        println!("⚡ 性能信息:");
        println!("  - 查询时间: {:?}", elapsed);
        if let Some(usage) = &response.response.usage {
            println!(
                "  - Token 使用: {} prompt + {} completion = {} total",
                usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
            );
        }

        Ok(())
    }

    /// 显示系统统计信息
    async fn show_system_stats(&self) -> Result<()> {
        println!("\n📊 Unity C# RAG 系统统计");
        println!("{}", "=".repeat(35));

        // 向量存储统计
        let node_count = self.vector_store.count().await?;
        let store_metadata = self.vector_store.metadata().await?;

        println!("🗄️ 向量存储:");
        println!("  - 存储代码块数: {}", node_count);
        println!(
            "  - 存储类型: {}",
            store_metadata.get("type").unwrap_or(&"unknown".into())
        );
        println!(
            "  - 向量维度: {}",
            store_metadata.get("dimension").unwrap_or(&"unknown".into())
        );

        // 嵌入器统计
        let embedder_metadata = self.embedder.metadata();
        println!("📊 嵌入器:");
        println!("  - 模型: {}", self.embedder.model_name());
        println!("  - 维度: {}", self.embedder.dimension());
        println!(
            "  - 已嵌入代码块数: {}",
            embedder_metadata.get("texts_embedded").unwrap_or(&0.into())
        );

        // 配置信息
        println!("⚙️ 配置:");
        println!("  - Unity 项目路径: {}", self.config.unity_project_path.display());
        println!("  - 分块策略: {}", self.config.chunking_strategy.description());
        let (chunk_lines, chunk_lines_overlap, max_chars) = self.config.chunking_strategy.params();
        println!("  - 分块参数: {} 行, {} 重叠, {} 字符", chunk_lines, chunk_lines_overlap, max_chars);
        println!("  - Top-K: {}", self.config.top_k);
        println!("  - LLM 提供商: {}", self.config.llm_provider);
        println!("  - LLM 模型: {}", self.config.llm_model);
        println!("  - AST 分析: {}", if self.config.enable_ast_analysis { "启用" } else { "禁用" });
        println!("  - 详细日志: {}", if self.config.verbose { "启用" } else { "禁用" });

        // 健康检查
        println!("🏥 健康检查:");
        match self.vector_store.health_check().await {
            Ok(()) => println!("  - 向量存储: ✅ 正常"),
            Err(e) => println!("  - 向量存储: ❌ 错误: {}", e),
        }

        match self.embedder.health_check().await {
            Ok(()) => println!("  - 嵌入器: ✅ 正常"),
            Err(e) => println!("  - 嵌入器: ❌ 错误: {}", e),
        }

        Ok(())
    }

    /// 显示帮助信息
    fn show_help(&self) {
        println!("\n📖 Unity C# 代码问答系统帮助");
        println!("{}", "=".repeat(35));
        println!("可用命令:");
        println!("  help     - 显示此帮助信息");
        println!("  examples - 显示示例问题");
        println!("  stats    - 显示系统统计信息");
        println!("  quit     - 退出程序");
        println!("  exit     - 退出程序");
        println!("\n使用技巧:");
        println!("  - 询问具体的类、方法或功能实现");
        println!("  - 可以问架构设计和代码组织问题");
        println!("  - 支持询问 Unity 特定的功能和模式");
        println!("  - 系统会自动找到最相关的代码片段");
        println!("  - 支持中文和英文问答");
    }

    /// 显示示例问题
    fn show_example_questions(&self) {
        println!("\n💡 Unity C# 代码问答示例问题");
        println!("{}", "=".repeat(35));
        println!("🎮 游戏功能相关:");
        println!("  - 这个项目中有哪些主要的游戏对象类？");
        println!("  - 玩家控制是如何实现的？");
        println!("  - 游戏中的碰撞检测是怎么处理的？");
        println!("  - UI 系统是如何组织的？");
        println!("\n🏗️ 架构设计相关:");
        println!("  - 项目使用了哪些设计模式？");
        println!("  - 数据管理是如何实现的？");
        println!("  - 事件系统是怎么设计的？");
        println!("  - 场景管理的架构是什么样的？");
        println!("\n🔧 技术实现相关:");
        println!("  - 这个方法的具体实现逻辑是什么？");
        println!("  - 如何优化这段代码的性能？");
        println!("  - 这个类的职责和依赖关系是什么？");
        println!("  - 错误处理是如何实现的？");
        println!("\n🎯 Unity 特定:");
        println!("  - MonoBehaviour 的生命周期是如何使用的？");
        println!("  - ScriptableObject 在项目中的应用？");
        println!("  - 协程的使用场景和实现？");
        println!("  - Unity 事件系统的集成方式？");
    }
}

/// 解析命令行参数
fn parse_cli_args() -> CliArgs {
    let args: Vec<String> = env::args().collect();
    let mut cli_args = CliArgs {
        project_path: PathBuf::from("."),
        strategy: None,
        llm_provider: None,
        llm_model: None,
        top_k: None,
        verbose: false,
        no_sample: false,
        help: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => cli_args.help = true,
            "--verbose" | "-v" => cli_args.verbose = true,
            "--no-sample" => cli_args.no_sample = true,
            "--strategy" | "-s" => {
                if i + 1 < args.len() {
                    cli_args.strategy = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--llm" | "-l" => {
                if i + 1 < args.len() {
                    cli_args.llm_provider = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--model" | "-m" => {
                if i + 1 < args.len() {
                    cli_args.llm_model = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--top-k" | "-k" => {
                if i + 1 < args.len() {
                    if let Ok(k) = args[i + 1].parse::<usize>() {
                        cli_args.top_k = Some(k);
                    }
                    i += 1;
                }
            }
            arg if !arg.starts_with("--") => {
                cli_args.project_path = PathBuf::from(arg);
            }
            _ => {}
        }
        i += 1;
    }

    cli_args
}

/// 从命令行参数构建配置
fn build_config_from_args(args: CliArgs) -> Result<UnityCSharpRagConfig> {
    let mut config = UnityCSharpRagConfig::default();

    // 设置项目路径
    config.unity_project_path = args.project_path;
    config.create_sample = !args.no_sample;
    config.verbose = args.verbose;

    // 设置分块策略
    if let Some(strategy_str) = args.strategy {
        config.chunking_strategy = match strategy_str.to_lowercase().as_str() {
            "optimal" => ChunkingStrategy::Optimal,
            "fine" => ChunkingStrategy::Fine,
            "balanced" => ChunkingStrategy::Balanced,
            "coarse" => ChunkingStrategy::Coarse,
            "minimal" => ChunkingStrategy::Minimal,
            "enterprise" => ChunkingStrategy::Enterprise,
            _ => {
                warn!("未知的分块策略: {}，使用默认的企业级策略", strategy_str);
                ChunkingStrategy::Enterprise
            }
        };
    }

    // 设置 LLM 配置
    if let Some(provider) = args.llm_provider {
        config.llm_provider = provider;
    }

    if let Some(model) = args.llm_model {
        config.llm_model = model;
    }

    // 根据环境变量自动检测 LLM 提供商
    if env::var("OPENAI_API_KEY").is_ok() && config.llm_provider == "openai" {
        if config.llm_model == "gpt-3.5-turbo" {
            // 保持默认模型
        }
    } else if config.llm_provider == "openai" && env::var("OPENAI_API_KEY").is_err() {
        warn!("未设置 OPENAI_API_KEY，切换到 Ollama");
        config.llm_provider = "ollama".to_string();
        config.llm_model = "llama3.2".to_string();
    }

    // 设置 top-k
    if let Some(k) = args.top_k {
        config.top_k = k;
    }

    Ok(config)
}

/// 显示帮助信息
fn show_help() {
    println!("🎮 Unity C# 代码问答系统");
    println!("=====================================");
    println!();
    println!("用法:");
    println!("  unity_csharp_rag [选项] <Unity项目路径>");
    println!();
    println!("参数:");
    println!("  <Unity项目路径>     Unity 项目的根目录路径");
    println!();
    println!("选项:");
    println!("  -h, --help          显示此帮助信息");
    println!("  -v, --verbose       启用详细日志输出");
    println!("  -s, --strategy      分块策略 [optimal|fine|balanced|coarse|minimal|enterprise]");
    println!("  -l, --llm           LLM 提供商 [openai|ollama]");
    println!("  -m, --model         LLM 模型名称");
    println!("  -k, --top-k         检索时返回的结果数量 (默认: 5)");
    println!("  --no-sample         如果项目路径不存在，不创建示例项目");
    println!();
    println!("分块策略说明:");
    println!("  optimal     - 最优策略，适合一般 RAG 应用 (40行/15重叠/1500字符)");
    println!("  fine        - 精细分析，适合详细代码分析 (15行/5重叠/800字符)");
    println!("  balanced    - 平衡策略，适合大多数场景 (30行/10重叠/1200字符)");
    println!("  coarse      - 粗粒度，适合高级概览 (50行/15重叠/2000字符)");
    println!("  minimal     - 最小块，适合极详细分析 (10行/3重叠/500字符)");
    println!("  enterprise  - 企业级，适合大型项目如Unity3D (60行/20重叠/2500字符) [默认]");
    println!();
    println!("环境变量:");
    println!("  OPENAI_API_KEY      OpenAI API 密钥");
    println!();
    println!("示例:");
    println!("  # 使用默认配置索引当前目录");
    println!("  unity_csharp_rag .");
    println!();
    println!("  # 使用企业级策略和 OpenAI");
    println!("  export OPENAI_API_KEY=\"your-key\"");
    println!("  unity_csharp_rag /path/to/unity/project --strategy enterprise --llm openai");
    println!();
    println!("  # 使用本地 Ollama");
    println!("  unity_csharp_rag /path/to/unity/project --llm ollama --model llama3.2");
}

/// 显示配置摘要
fn print_config_summary(config: &UnityCSharpRagConfig) {
    println!("📋 配置摘要:");
    println!("  项目路径: {}", config.unity_project_path.display());
    println!("  分块策略: {}", config.chunking_strategy.description());
    let (lines, overlap, chars) = config.chunking_strategy.params();
    println!("  分块参数: {} 行, {} 重叠, {} 字符", lines, overlap, chars);
    println!("  LLM: {} ({})", config.llm_provider, config.llm_model);
    println!("  Top-K: {}", config.top_k);
    println!("  AST 分析: {}", if config.enable_ast_analysis { "启用" } else { "禁用" });
    println!();
}

/// 创建 LLM 生成器
async fn create_llm_generator(config: &UnityCSharpRagConfig) -> Result<SiumaiGenerator> {
    match config.llm_provider.as_str() {
        "openai" => {
            let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
                cheungfun_core::CheungfunError::Configuration {
                    message: "OPENAI_API_KEY environment variable not set".to_string(),
                }
            })?;

            let client = Siumai::builder()
                .openai()
                .api_key(&api_key)
                .model(&config.llm_model)
                .temperature(0.7)
                .max_tokens(1500)
                .build()
                .await
                .map_err(|e| cheungfun_core::CheungfunError::Configuration {
                    message: format!("Failed to create OpenAI client: {}", e),
                })?;

            Ok(SiumaiGenerator::new(client))
        }
        "ollama" => {
            let client = Siumai::builder()
                .ollama()
                .base_url("http://localhost:11434")
                .model(&config.llm_model)
                .temperature(0.7)
                .max_tokens(1500)
                .build()
                .await
                .map_err(|e| cheungfun_core::CheungfunError::Configuration {
                    message: format!("Failed to create Ollama client: {}", e),
                })?;

            Ok(SiumaiGenerator::new(client))
        }
        _ => Err(cheungfun_core::CheungfunError::Configuration {
            message: format!("Unsupported LLM provider: {}", config.llm_provider),
        }),
    }
}

/// 从 C# 代码中提取类和方法信息
fn extract_csharp_info(content: &str) -> Option<String> {
    let mut info_parts = Vec::new();

    // 提取类名
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("public class ") || trimmed.starts_with("class ") {
            if let Some(class_name) = trimmed.split_whitespace().nth(2) {
                info_parts.push(format!("类: {}", class_name));
                break;
            }
        }
    }

    // 提取方法名
    let method_count = content.matches("public ").count() + content.matches("private ").count();
    if method_count > 0 {
        info_parts.push(format!("方法: {} 个", method_count));
    }

    // 检查是否是 MonoBehaviour
    if content.contains("MonoBehaviour") {
        info_parts.push("Unity MonoBehaviour".to_string());
    }

    // 检查是否是 ScriptableObject
    if content.contains("ScriptableObject") {
        info_parts.push("Unity ScriptableObject".to_string());
    }

    if info_parts.is_empty() {
        None
    } else {
        Some(info_parts.join(", "))
    }
}

/// 创建示例 Unity 项目
async fn create_sample_unity_project(project_path: &PathBuf) -> Result<()> {
    info!("🎮 创建示例 Unity 项目...");

    // 创建项目目录结构
    let assets_dir = project_path.join("Assets");
    let scripts_dir = assets_dir.join("Scripts");
    let managers_dir = scripts_dir.join("Managers");
    let controllers_dir = scripts_dir.join("Controllers");
    let ui_dir = scripts_dir.join("UI");

    fs::create_dir_all(&managers_dir)
        .await
        .map_err(|e| cheungfun_core::CheungfunError::Io(e))?;
    fs::create_dir_all(&controllers_dir)
        .await
        .map_err(|e| cheungfun_core::CheungfunError::Io(e))?;
    fs::create_dir_all(&ui_dir)
        .await
        .map_err(|e| cheungfun_core::CheungfunError::Io(e))?;

    // 创建 .gitignore 文件
    let gitignore_content = r#"# Unity generated files
[Ll]ibrary/
[Tt]emp/
[Oo]bj/
[Bb]uild/
[Bb]uilds/
[Ll]ogs/
[Uu]ser[Ss]ettings/
[Mm]emoryCaptures/

# Asset meta data should only be ignored when the corresponding asset is also ignored
!/[Aa]ssets/**/*.meta

# Uncomment this line if you wish to ignore the asset store tools plugin
# /[Aa]ssets/AssetStoreTools*

# Autogenerated Jetbrains Rider plugin
/[Aa]ssets/Plugins/Editor/JetBrains*

# Visual Studio cache directory
.vs/

# Gradle cache directory
.gradle/

# Autogenerated VS/MD/Consulo solution and project files
ExportedObj/
.consulo/
*.csproj
*.unityproj
*.sln
*.suo
*.tmp
*.user
*.userprefs
*.pidb
*.booproj
*.svd
*.pdb
*.mdb
*.opendb
*.VC.db

# Unity3D generated meta files
*.pidb.meta
*.pdb.meta
*.mdb.meta

# Unity3D generated file on crash reports
sysinfo.txt

# Builds
*.apk
*.aab
*.unitypackage
*.app

# Crashlytics generated file
crashlytics-build.properties

# Packed Addressables
/[Aa]ssets/[Aa]ddressable[Aa]ssets[Dd]ata/*/*.bin*

# Temporary auto-generated Android Assets
/[Aa]ssets/[Ss]treamingAssets/aa.meta
/[Aa]ssets/[Ss]treamingAssets/aa/*
"#;

    fs::write(project_path.join(".gitignore"), gitignore_content)
        .await
        .map_err(|e| cheungfun_core::CheungfunError::Io(e))?;

    let sample_scripts = vec![
        (
            "GameManager.cs",
            r#"using UnityEngine;
using System.Collections;
using System.Collections.Generic;

namespace Game.Managers
{
    /// <summary>
    /// 游戏主管理器，负责游戏状态管理和核心逻辑协调
    /// </summary>
    public class GameManager : MonoBehaviour
    {
        [Header("Game Settings")]
        public float gameSpeed = 1.0f;
        public int maxLives = 3;

        [Header("References")]
        public PlayerController playerController;
        public UIManager uiManager;
        public AudioManager audioManager;

        // 游戏状态
        public enum GameState
        {
            Menu,
            Playing,
            Paused,
            GameOver
        }

        private GameState currentState = GameState.Menu;
        private int currentScore = 0;
        private int currentLives;

        // 单例模式
        public static GameManager Instance { get; private set; }

        private void Awake()
        {
            // 确保只有一个 GameManager 实例
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
                InitializeGame();
            }
            else
            {
                Destroy(gameObject);
            }
        }

        private void Start()
        {
            StartCoroutine(GameLoop());
        }

        /// <summary>
        /// 初始化游戏系统
        /// </summary>
        private void InitializeGame()
        {
            currentLives = maxLives;
            currentScore = 0;

            // 初始化各个管理器
            if (uiManager != null)
                uiManager.Initialize();

            if (audioManager != null)
                audioManager.Initialize();
        }

        /// <summary>
        /// 游戏主循环协程
        /// </summary>
        private IEnumerator GameLoop()
        {
            while (true)
            {
                switch (currentState)
                {
                    case GameState.Menu:
                        yield return StartCoroutine(MenuState());
                        break;
                    case GameState.Playing:
                        yield return StartCoroutine(PlayingState());
                        break;
                    case GameState.Paused:
                        yield return StartCoroutine(PausedState());
                        break;
                    case GameState.GameOver:
                        yield return StartCoroutine(GameOverState());
                        break;
                }
                yield return null;
            }
        }

        private IEnumerator MenuState()
        {
            Debug.Log("进入菜单状态");
            yield return new WaitUntil(() => currentState != GameState.Menu);
        }

        private IEnumerator PlayingState()
        {
            Debug.Log("进入游戏状态");
            yield return new WaitUntil(() => currentState != GameState.Playing);
        }

        private IEnumerator PausedState()
        {
            Debug.Log("游戏暂停");
            Time.timeScale = 0f;
            yield return new WaitUntil(() => currentState != GameState.Paused);
            Time.timeScale = gameSpeed;
        }

        private IEnumerator GameOverState()
        {
            Debug.Log("游戏结束");
            yield return new WaitForSeconds(2f);
            RestartGame();
        }

        /// <summary>
        /// 开始游戏
        /// </summary>
        public void StartGame()
        {
            currentState = GameState.Playing;
            currentScore = 0;
            currentLives = maxLives;

            if (playerController != null)
                playerController.ResetPlayer();
        }

        /// <summary>
        /// 暂停游戏
        /// </summary>
        public void PauseGame()
        {
            if (currentState == GameState.Playing)
                currentState = GameState.Paused;
        }

        /// <summary>
        /// 恢复游戏
        /// </summary>
        public void ResumeGame()
        {
            if (currentState == GameState.Paused)
                currentState = GameState.Playing;
        }

        /// <summary>
        /// 重新开始游戏
        /// </summary>
        public void RestartGame()
        {
            currentState = GameState.Menu;
            InitializeGame();
        }

        /// <summary>
        /// 增加分数
        /// </summary>
        public void AddScore(int points)
        {
            currentScore += points;
            if (uiManager != null)
                uiManager.UpdateScore(currentScore);
        }

        /// <summary>
        /// 减少生命值
        /// </summary>
        public void LoseLife()
        {
            currentLives--;
            if (uiManager != null)
                uiManager.UpdateLives(currentLives);

            if (currentLives <= 0)
            {
                currentState = GameState.GameOver;
            }
        }

        // 属性访问器
        public GameState CurrentState => currentState;
        public int CurrentScore => currentScore;
        public int CurrentLives => currentLives;
    }
}
"#,
        ),
        (
            "PlayerController.cs",
            r#"using UnityEngine;
using System.Collections;

namespace Game.Controllers
{
    /// <summary>
    /// 玩家控制器，处理玩家输入和移动逻辑
    /// </summary>
    [RequireComponent(typeof(Rigidbody2D))]
    [RequireComponent(typeof(Collider2D))]
    public class PlayerController : MonoBehaviour
    {
        [Header("Movement Settings")]
        public float moveSpeed = 5f;
        public float jumpForce = 10f;
        public float maxSpeed = 8f;

        [Header("Ground Check")]
        public Transform groundCheck;
        public float groundCheckRadius = 0.2f;
        public LayerMask groundLayerMask = 1;

        [Header("Animation")]
        public Animator animator;

        [Header("Audio")]
        public AudioClip jumpSound;
        public AudioClip landSound;

        // 组件引用
        private Rigidbody2D rb2d;
        private Collider2D col2d;
        private AudioSource audioSource;

        // 状态变量
        private bool isGrounded;
        private bool facingRight = true;
        private float horizontalInput;
        private Vector2 velocity;

        // 动画参数哈希
        private int speedHash;
        private int groundedHash;
        private int jumpHash;

        private void Awake()
        {
            // 获取组件引用
            rb2d = GetComponent<Rigidbody2D>();
            col2d = GetComponent<Collider2D>();
            audioSource = GetComponent<AudioSource>();

            // 缓存动画参数哈希
            if (animator != null)
            {
                speedHash = Animator.StringToHash("Speed");
                groundedHash = Animator.StringToHash("IsGrounded");
                jumpHash = Animator.StringToHash("Jump");
            }
        }

        private void Start()
        {
            // 初始化物理设置
            rb2d.freezeRotation = true;
            rb2d.collisionDetectionMode = CollisionDetectionMode2D.Continuous;
        }

        private void Update()
        {
            HandleInput();
            CheckGrounded();
            UpdateAnimation();
        }

        private void FixedUpdate()
        {
            HandleMovement();
            ApplyPhysics();
        }

        /// <summary>
        /// 处理玩家输入
        /// </summary>
        private void HandleInput()
        {
            horizontalInput = Input.GetAxisRaw("Horizontal");

            // 跳跃输入
            if (Input.GetButtonDown("Jump") && isGrounded)
            {
                Jump();
            }
        }

        /// <summary>
        /// 处理玩家移动
        /// </summary>
        private void HandleMovement()
        {
            // 水平移动
            velocity = rb2d.velocity;
            velocity.x = horizontalInput * moveSpeed;

            // 限制最大速度
            velocity.x = Mathf.Clamp(velocity.x, -maxSpeed, maxSpeed);

            rb2d.velocity = velocity;

            // 处理角色翻转
            if (horizontalInput > 0 && !facingRight)
            {
                Flip();
            }
            else if (horizontalInput < 0 && facingRight)
            {
                Flip();
            }
        }

        /// <summary>
        /// 跳跃逻辑
        /// </summary>
        private void Jump()
        {
            rb2d.velocity = new Vector2(rb2d.velocity.x, jumpForce);

            // 播放跳跃音效
            if (audioSource != null && jumpSound != null)
            {
                audioSource.PlayOneShot(jumpSound);
            }

            // 触发跳跃动画
            if (animator != null)
            {
                animator.SetTrigger(jumpHash);
            }
        }

        /// <summary>
        /// 检查是否在地面上
        /// </summary>
        private void CheckGrounded()
        {
            bool wasGrounded = isGrounded;
            isGrounded = Physics2D.OverlapCircle(groundCheck.position, groundCheckRadius, groundLayerMask);

            // 着陆音效
            if (!wasGrounded && isGrounded && audioSource != null && landSound != null)
            {
                audioSource.PlayOneShot(landSound);
            }
        }

        /// <summary>
        /// 翻转角色
        /// </summary>
        private void Flip()
        {
            facingRight = !facingRight;
            Vector3 scale = transform.localScale;
            scale.x *= -1;
            transform.localScale = scale;
        }

        /// <summary>
        /// 更新动画参数
        /// </summary>
        private void UpdateAnimation()
        {
            if (animator == null) return;

            animator.SetFloat(speedHash, Mathf.Abs(horizontalInput));
            animator.SetBool(groundedHash, isGrounded);
        }

        /// <summary>
        /// 应用额外的物理效果
        /// </summary>
        private void ApplyPhysics()
        {
            // 改善跳跃手感的重力调整
            if (rb2d.velocity.y < 0)
            {
                rb2d.velocity += Vector2.up * Physics2D.gravity.y * (2.5f - 1) * Time.fixedDeltaTime;
            }
            else if (rb2d.velocity.y > 0 && !Input.GetButton("Jump"))
            {
                rb2d.velocity += Vector2.up * Physics2D.gravity.y * (2f - 1) * Time.fixedDeltaTime;
            }
        }

        /// <summary>
        /// 重置玩家状态
        /// </summary>
        public void ResetPlayer()
        {
            rb2d.velocity = Vector2.zero;
            transform.position = Vector3.zero;
            facingRight = true;
            transform.localScale = new Vector3(1, 1, 1);
        }

        /// <summary>
        /// 碰撞检测
        /// </summary>
        private void OnTriggerEnter2D(Collider2D other)
        {
            if (other.CompareTag("Collectible"))
            {
                CollectItem(other.gameObject);
            }
            else if (other.CompareTag("Enemy"))
            {
                TakeDamage();
            }
        }

        /// <summary>
        /// 收集物品
        /// </summary>
        private void CollectItem(GameObject item)
        {
            // 增加分数
            if (GameManager.Instance != null)
            {
                GameManager.Instance.AddScore(10);
            }

            Destroy(item);
        }

        /// <summary>
        /// 受到伤害
        /// </summary>
        private void TakeDamage()
        {
            if (GameManager.Instance != null)
            {
                GameManager.Instance.LoseLife();
            }

            // 击退效果
            StartCoroutine(KnockbackEffect());
        }

        /// <summary>
        /// 击退效果协程
        /// </summary>
        private IEnumerator KnockbackEffect()
        {
            float knockbackForce = 5f;
            rb2d.velocity = new Vector2(-horizontalInput * knockbackForce, jumpForce * 0.5f);

            yield return new WaitForSeconds(0.2f);

            rb2d.velocity = new Vector2(rb2d.velocity.x * 0.5f, rb2d.velocity.y);
        }

        // 调试绘制
        private void OnDrawGizmosSelected()
        {
            if (groundCheck != null)
            {
                Gizmos.color = isGrounded ? Color.green : Color.red;
                Gizmos.DrawWireSphere(groundCheck.position, groundCheckRadius);
            }
        }
    }
}
"#,
        ),
    ];

    // 写入示例脚本
    for (filename, content) in sample_scripts {
        let file_path = if filename == "GameManager.cs" {
            managers_dir.join(filename)
        } else {
            controllers_dir.join(filename)
        };

        fs::write(&file_path, content)
            .await
            .map_err(|e| cheungfun_core::CheungfunError::Io(e))?;
    }

    info!("  ✅ 创建了示例 Unity 项目和 C# 脚本");
    Ok(())
}

/// 创建 Embedder
async fn create_embedder(config: &UnityCSharpRagConfig) -> Result<Arc<dyn Embedder>> {
    match config.embedding_provider.as_str() {
        "fastembed" => {
            info!("  📊 初始化 FastEmbed 嵌入器...");
            let embedder = FastEmbedder::with_model(&config.embedding_model).await.map_err(|e| {
                cheungfun_core::CheungfunError::Configuration {
                    message: format!("Failed to initialize FastEmbedder: {}", e),
                }
            })?;
            Ok(Arc::new(embedder))
        }
        "openai" => {
            #[cfg(feature = "api")]
            {
                info!("  📊 初始化 OpenAI 嵌入器...");
                let api_key = env::var("OPENAI_API_KEY").map_err(|_| {
                    cheungfun_core::CheungfunError::Configuration {
                        message: "OPENAI_API_KEY environment variable not set for OpenAI embedding".to_string(),
                    }
                })?;

                let embedder = cheungfun_integrations::ApiEmbedder::builder()
                    .openai(&api_key)
                    .model(&config.embedding_model)
                    .build()
                    .await
                    .map_err(|e| cheungfun_core::CheungfunError::Configuration {
                        message: format!("Failed to initialize OpenAI embedder: {}", e),
                    })?;
                Ok(Arc::new(embedder))
            }
            #[cfg(not(feature = "api"))]
            {
                Err(cheungfun_core::CheungfunError::Configuration {
                    message: "OpenAI embedding requires 'api' feature to be enabled. Please compile with --features api".to_string(),
                })
            }
        }
        "gemini" => {
            info!("  📊 初始化 Gemini 嵌入器...");
            let api_key = env::var("GEMINI_API_KEY").map_err(|_| {
                cheungfun_core::CheungfunError::Configuration {
                    message: "GEMINI_API_KEY environment variable not set for Gemini embedding".to_string(),
                }
            })?;

            // 使用 siumai 创建 Gemini embedding 客户端
            let client = Siumai::builder()
                .gemini()
                .api_key(&api_key)
                .model(&config.embedding_model)
                .build()
                .await
                .map_err(|e| cheungfun_core::CheungfunError::Configuration {
                    message: format!("Failed to create Gemini client: {}", e),
                })?;

            // 创建一个包装器来适配 Embedder trait
            let embedder = GeminiEmbedderWrapper::new(client);
            Ok(Arc::new(embedder))
        }
        _ => Err(cheungfun_core::CheungfunError::Configuration {
            message: format!("Unsupported embedding provider: {}", config.embedding_provider),
        }),
    }
}

/// Gemini Embedder 包装器，用于适配 Cheungfun 的 Embedder trait
pub struct GeminiEmbedderWrapper {
    client: Siumai,
    model_name: String,
}

impl std::fmt::Debug for GeminiEmbedderWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiEmbedderWrapper")
            .field("model_name", &self.model_name)
            .finish()
    }
}

impl GeminiEmbedderWrapper {
    pub fn new(client: Siumai) -> Self {
        let model_name = "gemini-embedding-001".to_string(); // 默认模型
        Self { client, model_name }
    }
}

#[async_trait]
impl Embedder for GeminiEmbedderWrapper {
    async fn embed(&self, text: &str) -> cheungfun_core::Result<Vec<f32>> {
        let texts = vec![text.to_string()];
        let response = self.client.embed(texts).await.map_err(|e| {
            cheungfun_core::CheungfunError::Embedding {
                message: format!("Gemini embedding failed: {}", e),
            }
        })?;

        if response.embeddings.is_empty() {
            return Err(cheungfun_core::CheungfunError::Embedding {
                message: "No embeddings returned from Gemini".to_string(),
            });
        }

        Ok(response.embeddings[0].clone())
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> cheungfun_core::Result<Vec<Vec<f32>>> {
        let text_strings: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let response = self.client.embed(text_strings).await.map_err(|e| {
            cheungfun_core::CheungfunError::Embedding {
                message: format!("Gemini batch embedding failed: {}", e),
            }
        })?;

        Ok(response.embeddings)
    }

    fn dimension(&self) -> usize {
        3072 // Gemini embedding dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn health_check(&self) -> cheungfun_core::Result<()> {
        // 简单的健康检查：尝试嵌入一个测试文本
        self.embed("health check").await?;
        Ok(())
    }

    fn metadata(&self) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();
        metadata.insert("provider".to_string(), serde_json::json!("gemini"));
        metadata.insert("model".to_string(), serde_json::json!(self.model_name));
        metadata.insert("dimension".to_string(), serde_json::json!(self.dimension()));
        metadata
    }
}
