//! Markdown文件夹RAG问答示例
//!
//! 这个示例展示如何使用Cheungfun框架构建一个完整的RAG系统：
//! 1. 批量加载指定文件夹下的所有markdown文件
//! 2. 使用真实LLM API进行嵌入和问答
//! 3. 支持内存或SQLite存储
//! 4. 提供交互式问答功能
//!
//! ## 使用方法
//!
//! ```bash
//! # 设置OpenAI API密钥（可选，会回退到Ollama）
//! export OPENAI_API_KEY="your-api-key-here"
//!
//! # 运行示例
//! cargo run --example markdown_rag_example --features "fastembed,sqlite"
//!
//! # 或使用本地Ollama
//! ollama serve
//! ollama pull llama3.2
//! cargo run --example markdown_rag_example --features "fastembed"
//! ```
//!
//! ## 功能特性
//!
//! - 🗂️ 批量处理markdown文件
//! - 🤖 真实LLM集成（OpenAI/Ollama）
//! - 💾 灵活存储选择（内存/SQLite）
//! - 🔍 语义搜索和问答
//! - 📊 详细的处理统计
//! - 🎯 交互式查询界面

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
use siumai::prelude::*;
use std::{
    io::{self, Write},
    path::PathBuf,
    sync::Arc,
};
use tokio::fs;
use tracing::{info, warn, Level};

/// 配置结构体
#[derive(Debug, Clone)]
pub struct RagConfig {
    /// markdown文件夹路径
    pub docs_folder: PathBuf,
    /// 是否使用SQLite存储（否则使用内存）
    pub use_sqlite: bool,
    /// 文本分块大小
    pub chunk_size: usize,
    /// 分块重叠大小
    pub chunk_overlap: usize,
    /// 检索时返回的top-k结果数
    pub top_k: usize,
    /// LLM提供商（openai/ollama）
    pub llm_provider: String,
    /// LLM模型名称
    pub llm_model: String,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            docs_folder: PathBuf::from("./docs"),
            use_sqlite: false,
            chunk_size: 500,
            chunk_overlap: 50,
            top_k: 5,
            llm_provider: "openai".to_string(),
            llm_model: "gpt-3.5-turbo".to_string(),
        }
    }
}

/// RAG系统组件
pub struct MarkdownRagSystem {
    config: RagConfig,
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    query_engine: QueryEngine,
}

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    println!("🚀 Cheungfun Markdown RAG 问答系统");
    println!("=====================================");

    // 解析配置
    let config = parse_config_from_args();

    // 检查文档文件夹
    if !config.docs_folder.exists() {
        create_sample_markdown_docs(&config.docs_folder).await?;
    }

    // 初始化RAG系统
    let mut rag_system = MarkdownRagSystem::new(config).await?;

    // 构建索引
    rag_system.build_index().await?;

    // 启动交互式问答
    rag_system.start_interactive_chat().await?;

    Ok(())
}

impl MarkdownRagSystem {
    /// 创建新的RAG系统
    pub async fn new(config: RagConfig) -> Result<Self> {
        info!("🔧 初始化RAG系统组件...");

        // 1. 初始化嵌入器
        info!("  📊 初始化FastEmbed嵌入器...");
        let embedder = Arc::new(FastEmbedder::new().await.map_err(|e| {
            cheungfun_core::CheungfunError::Configuration {
                message: format!("Failed to initialize FastEmbedder: {}", e),
            }
        })?);
        info!("    ✅ 嵌入器就绪 (维度: {})", embedder.dimension());

        // 2. 初始化向量存储
        let vector_store: Arc<dyn VectorStore> = if config.use_sqlite {
            info!("  🗄️ 初始化SQLite向量存储...");
            // TODO: 实现SQLite存储
            warn!("    ⚠️ SQLite存储暂未实现，使用内存存储");
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

        // 3. 创建LLM客户端和生成器
        info!("  🤖 初始化LLM客户端...");
        let generator = create_llm_generator(&config).await?;
        info!("    ✅ LLM生成器就绪");

        // 4. 创建查询引擎
        let retriever = Arc::new(VectorRetriever::new(vector_store.clone(), embedder.clone()));

        // 创建查询引擎配置
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

    /// 构建文档索引
    pub async fn build_index(&mut self) -> Result<()> {
        info!("📚 开始构建文档索引...");
        let start_time = std::time::Instant::now();

        // 1. 配置文件加载器，只处理markdown文件
        let loader_config = LoaderConfig::new()
            .with_include_extensions(vec!["md".to_string()])
            .with_max_depth(10)
            .with_continue_on_error(true);

        // 2. 加载所有markdown文件
        info!("  📂 扫描文件夹: {}", self.config.docs_folder.display());
        let loader = DirectoryLoader::with_config(&self.config.docs_folder, loader_config)?;
        let documents = loader.load().await?;

        if documents.is_empty() {
            warn!("  ⚠️ 未找到任何markdown文件");
            return Ok(());
        }

        info!("  ✅ 加载了 {} 个markdown文件", documents.len());

        // 3. 配置文本分割器
        let splitter_config = SplitterConfig {
            chunk_size: self.config.chunk_size,
            chunk_overlap: self.config.chunk_overlap,
            separators: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                "。".to_string(),
            ],
            keep_separators: true,
            ..Default::default()
        };
        let text_splitter = TextSplitter::with_config(splitter_config);
        let metadata_extractor = MetadataExtractor::new();

        // 4. 处理每个文档
        let mut all_nodes = Vec::new();
        let mut total_chunks = 0;

        for (i, document) in documents.iter().enumerate() {
            info!(
                "  📄 处理文档 {}/{}: {}",
                i + 1,
                documents.len(),
                document
                    .get_metadata_string("file_path")
                    .unwrap_or_else(|| format!("Document {}", i + 1))
            );

            // 分割文档
            let nodes = text_splitter.transform(document.clone()).await?;
            info!("    ✂️ 分割为 {} 个块", nodes.len());
            total_chunks += nodes.len();

            // 提取元数据并生成嵌入
            for mut node in nodes {
                // 提取元数据
                node = metadata_extractor.transform_node(node).await?;

                // 生成嵌入
                let embedding = self.embedder.embed(&node.content).await?;
                node.embedding = Some(embedding);

                all_nodes.push(node);
            }
        }

        info!("  🧮 生成了 {} 个文本块的嵌入", all_nodes.len());

        // 5. 存储到向量数据库
        info!("  💾 存储到向量数据库...");
        let stored_ids = self.vector_store.add(all_nodes).await?;

        let elapsed = start_time.elapsed();
        info!("✅ 索引构建完成!");
        info!("  📊 统计信息:");
        info!("    - 处理文档: {} 个", documents.len());
        info!("    - 生成块: {} 个", total_chunks);
        info!("    - 存储节点: {} 个", stored_ids.len());
        info!("    - 处理时间: {:?}", elapsed);
        info!("    - 平均每文档: {:?}", elapsed / documents.len() as u32);

        Ok(())
    }

    /// 启动交互式问答
    pub async fn start_interactive_chat(&self) -> Result<()> {
        info!("🎯 启动交互式问答模式");
        println!("\n💬 RAG问答系统已就绪！");
        println!("提示：");
        println!("  - 输入问题开始对话");
        println!("  - 输入 'quit' 或 'exit' 退出");
        println!("  - 输入 'stats' 查看系统统计");
        println!("  - 输入 'help' 查看帮助");
        println!("{}", "=".repeat(50));

        loop {
            // 显示提示符
            print!("\n🤔 您的问题: ");
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
                        "" => continue,
                        _ => {}
                    }

                    // 处理查询
                    match self.process_query(query).await {
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

    /// 处理单个查询
    async fn process_query(&self, query: &str) -> Result<()> {
        println!("🔍 正在搜索相关内容...");
        let start_time = std::time::Instant::now();

        // 执行查询
        let response = self.query_engine.query(query).await?;
        let elapsed = start_time.elapsed();

        // 显示结果
        println!("\n🤖 AI回答:");
        println!("{}", "─".repeat(50));
        println!("{}", response.response.content);
        println!("{}", "─".repeat(50));

        // 显示检索信息
        println!(
            "\n📚 参考来源 ({} 个相关片段):",
            response.retrieved_nodes.len()
        );
        for (i, scored_node) in response.retrieved_nodes.iter().take(3).enumerate() {
            let source = scored_node
                .node
                .get_metadata_string("file_path")
                .unwrap_or_else(|| "未知来源".to_string());
            let preview = scored_node
                .node
                .content
                .chars()
                .take(100)
                .collect::<String>();

            println!("  {}. [相似度: {:.3}] {}", i + 1, scored_node.score, source);
            println!("     预览: {}...", preview);
        }

        // 显示性能信息
        println!("\n⚡ 性能信息:");
        println!("  - 查询时间: {:?}", elapsed);
        if let Some(usage) = &response.response.usage {
            println!(
                "  - Token使用: {} prompt + {} completion = {} total",
                usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
            );
        }

        Ok(())
    }

    /// 显示系统统计信息
    async fn show_system_stats(&self) -> Result<()> {
        println!("\n📊 系统统计信息");
        println!("{}", "=".repeat(30));

        // 向量存储统计
        let node_count = self.vector_store.count().await?;
        let store_metadata = self.vector_store.metadata().await?;

        println!("🗄️ 向量存储:");
        println!("  - 存储节点数: {}", node_count);
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
            "  - 已嵌入文本数: {}",
            embedder_metadata.get("texts_embedded").unwrap_or(&0.into())
        );

        // 配置信息
        println!("⚙️ 配置:");
        println!("  - 文档文件夹: {}", self.config.docs_folder.display());
        println!("  - 分块大小: {}", self.config.chunk_size);
        println!("  - 分块重叠: {}", self.config.chunk_overlap);
        println!("  - Top-K: {}", self.config.top_k);
        println!("  - LLM提供商: {}", self.config.llm_provider);
        println!("  - LLM模型: {}", self.config.llm_model);

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
        println!("\n📖 帮助信息");
        println!("{}", "=".repeat(30));
        println!("可用命令:");
        println!("  help  - 显示此帮助信息");
        println!("  stats - 显示系统统计信息");
        println!("  quit  - 退出程序");
        println!("  exit  - 退出程序");
        println!("\n使用技巧:");
        println!("  - 尽量使用具体、清晰的问题");
        println!("  - 可以询问文档中的具体内容");
        println!("  - 支持中文和英文问答");
        println!("  - 系统会自动找到最相关的文档片段");
    }
}

/// 解析命令行参数配置
fn parse_config_from_args() -> RagConfig {
    let mut config = RagConfig::default();

    // 从环境变量读取配置
    if let Ok(docs_path) = std::env::var("DOCS_FOLDER") {
        config.docs_folder = PathBuf::from(docs_path);
    }

    if let Ok(chunk_size) = std::env::var("CHUNK_SIZE") {
        if let Ok(size) = chunk_size.parse::<usize>() {
            config.chunk_size = size;
        }
    }

    if let Ok(top_k) = std::env::var("TOP_K") {
        if let Ok(k) = top_k.parse::<usize>() {
            config.top_k = k;
        }
    }

    // 检测LLM提供商
    if std::env::var("OPENAI_API_KEY").is_ok() {
        config.llm_provider = "openai".to_string();
        config.llm_model = "gpt-3.5-turbo".to_string();
    } else {
        config.llm_provider = "ollama".to_string();
        config.llm_model = "llama3.2".to_string();
    }

    config
}

/// 创建LLM生成器
async fn create_llm_generator(config: &RagConfig) -> Result<SiumaiGenerator> {
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
                .max_tokens(1000)
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
                .max_tokens(1000)
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

/// 创建示例markdown文档
async fn create_sample_markdown_docs(docs_folder: &PathBuf) -> Result<()> {
    info!("📝 创建示例markdown文档...");

    fs::create_dir_all(docs_folder)
        .await
        .map_err(|e| cheungfun_core::CheungfunError::Io(e))?;

    let sample_docs = vec![
        (
            "rust_basics.md",
            r#"# Rust编程语言基础

## 什么是Rust？

Rust是一种系统编程语言，专注于安全性、速度和并发性。它由Mozilla开发，旨在解决C和C++中常见的内存安全问题，同时保持高性能。

## Rust的核心特性

### 1. 内存安全
Rust通过所有权系统（Ownership System）在编译时防止内存泄漏、悬空指针和数据竞争等问题。

### 2. 零成本抽象
Rust提供高级抽象，但不会产生运行时开销。编译器会优化代码，使抽象的成本为零。

### 3. 并发安全
Rust的类型系统防止数据竞争，使并发编程更加安全。

## Rust的应用领域

- 系统编程：操作系统内核、设备驱动程序
- Web开发：Web服务器、API服务
- 区块链：以太坊客户端、加密货币钱包
- 游戏开发：游戏引擎、高性能游戏逻辑
- 机器学习：高性能计算库、数据处理工具
"#,
        ),
        (
            "rag_introduction.md",
            r#"# RAG系统介绍

## 什么是RAG？

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索和文本生成的AI技术。它通过从外部知识库检索相关信息来增强大语言模型的回答能力。

## RAG的工作原理

### 1. 知识库构建阶段
- 文档处理：加载、提取、分割文档
- 向量化存储：文本嵌入、向量存储、索引构建

### 2. 查询处理阶段
- 检索过程：查询嵌入、相似性搜索、结果排序
- 生成过程：上下文组装、提示构建、文本生成

## RAG的优势

- 知识更新：实时性、准确性、可追溯性
- 成本效益：无需重训练、资源节约、灵活性
- 可控性：内容控制、质量保证、隐私保护

## RAG系统的组件

1. 文档加载器：从各种数据源加载文档
2. 文本处理器：处理和准备文本数据
3. 嵌入模型：将文本转换为向量表示
4. 向量数据库：存储和检索向量
5. 检索器：执行相似性搜索
6. 生成器：生成最终回答
"#,
        ),
        (
            "ai_development.md",
            r#"# AI开发实践指南

## AI开发概述

人工智能开发是一个涉及多个学科的复杂过程，包括机器学习、深度学习、自然语言处理、计算机视觉等多个领域。

## AI开发生命周期

### 1. 问题定义阶段
- 业务理解：需求分析、可行性评估、成功指标、风险评估
- 问题建模：问题类型、输入输出、约束条件、评估标准

### 2. 数据准备阶段
- 数据收集：数据源识别、质量评估、合规性、数据存储
- 数据预处理：数据清理、特征工程、数据变换

### 3. 模型开发阶段
- 模型选择：传统机器学习、深度学习、集成方法、预训练模型
- 模型训练：数据分割、模型训练、交叉验证、模型评估

### 4. 模型评估阶段
- 评估指标：分类、回归、生成、检索任务的不同指标
- 模型解释性：特征重要性、SHAP值、LIME、注意力机制

### 5. 部署和监控阶段
- 模型部署：API服务、容器化、云部署
- 模型监控：性能监控、数据漂移检测、A/B测试

## AI开发最佳实践

1. 数据管理：版本控制、数据血缘、质量监控、隐私保护
2. 实验管理：实验跟踪、超参数优化、模型版本管理、可重现性
3. 代码质量：模块化设计、单元测试、代码审查、文档编写
4. 团队协作：角色分工、沟通机制、知识共享、技能培训
"#,
        ),
    ];

    for (filename, content) in sample_docs {
        let file_path = docs_folder.join(filename);
        fs::write(&file_path, content)
            .await
            .map_err(|e| cheungfun_core::CheungfunError::Io(e))?;
    }

    info!("  ✅ 创建了 {} 个示例文档", 3);
    Ok(())
}
