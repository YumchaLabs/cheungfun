//! Parser Showcase Example
//!
//! This example demonstrates all the different parsers available in cheungfun-indexing
//! and shows how to use them effectively for different types of content.

use cheungfun_core::{Document, traits::{Transform, TransformInput}};
use cheungfun_indexing::node_parser::{
    text::{
        SentenceSplitter, TokenTextSplitter, MarkdownNodeParser, 
        SentenceWindowNodeParser, CodeSplitter
    },
    config::MarkdownConfig,
    NodeParser,
};
use cheungfun_indexing::loaders::ProgrammingLanguage;
use std::time::Instant;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    println!("🚀 Cheungfun Indexing Parser Showcase");
    println!("=====================================");

    info!("🚀 Cheungfun Indexing Parser Showcase");
    info!("=====================================");

    // Sample content for different parsers
    let text_content = r#"
        人工智能正在改变我们的世界。机器学习算法能够从大量数据中学习模式。
        深度学习网络可以识别图像、理解语言、生成内容。自然语言处理技术让计算机能够理解人类语言。
        
        RAG（检索增强生成）是一种新兴的AI架构。它结合了信息检索和文本生成的优势。
        通过检索相关文档，RAG系统能够生成更准确、更有根据的回答。
    "#;

    let markdown_content = r#"# Cheungfun RAG Framework

Cheungfun 是一个高性能的 RAG 框架，使用 Rust 编写。

## 核心特性

### 高性能
- 零拷贝字符串处理
- 并行处理支持
- 内存安全保证

### 模块化设计
- 统一的 Transform 接口
- 可插拔的组件架构
- 丰富的配置选项

## 支持的功能

### 文档加载
支持多种文档格式的加载和处理。

### 文本分割
提供多种分割策略以适应不同场景。
"#;

    let code_content = r#"
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// 用户配置结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserConfig {
    pub name: String,
    pub email: String,
    pub preferences: HashMap<String, String>,
}

impl UserConfig {
    /// 创建新的用户配置
    pub fn new(name: String, email: String) -> Self {
        Self {
            name,
            email,
            preferences: HashMap::new(),
        }
    }
    
    /// 添加偏好设置
    pub fn add_preference(&mut self, key: String, value: String) {
        self.preferences.insert(key, value);
    }
    
    /// 获取偏好设置
    pub fn get_preference(&self, key: &str) -> Option<&String> {
        self.preferences.get(key)
    }
}

/// 配置管理器
pub struct ConfigManager {
    configs: HashMap<String, UserConfig>,
}

impl ConfigManager {
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
        }
    }
    
    pub fn add_user(&mut self, id: String, config: UserConfig) {
        self.configs.insert(id, config);
    }
}
"#;

    // 1. Sentence Splitter Demo
    info!("\n📝 1. Sentence Splitter Demo");
    info!("============================");
    
    let sentence_splitter = SentenceSplitter::from_defaults(150, 30)?;
    let start = Instant::now();
    
    let input = TransformInput::Document(Document::new(text_content));
    let sentence_nodes = sentence_splitter.transform(input).await?;
    let duration = start.elapsed();
    
    info!("生成节点数: {}", sentence_nodes.len());
    info!("处理时间: {:?}", duration);
    for (i, node) in sentence_nodes.iter().take(2).enumerate() {
        info!("节点 {}: {}", i + 1, node.content.trim().chars().take(50).collect::<String>() + "...");
    }

    // 2. Token Text Splitter Demo
    info!("\n🔢 2. Token Text Splitter Demo");
    info!("==============================");
    
    let token_splitter = TokenTextSplitter::from_defaults(100, 20)?;
    let start = Instant::now();
    
    let input = TransformInput::Document(Document::new(text_content));
    let token_nodes = token_splitter.transform(input).await?;
    let duration = start.elapsed();
    
    info!("生成节点数: {}", token_nodes.len());
    info!("处理时间: {:?}", duration);
    for (i, node) in token_nodes.iter().take(2).enumerate() {
        info!("节点 {}: {}", i + 1, node.content.trim().chars().take(50).collect::<String>() + "...");
    }

    // 3. Markdown Parser Demo
    info!("\n📄 3. Markdown Parser Demo");
    info!("===========================");
    
    let markdown_parser = MarkdownNodeParser::new()
        .with_max_header_depth(3)
        .with_preserve_header_hierarchy(true);
    let start = Instant::now();
    
    let markdown_nodes = <MarkdownNodeParser as NodeParser>::parse_nodes(&markdown_parser, &[Document::new(markdown_content)], false).await?;
    let duration = start.elapsed();
    
    info!("生成节点数: {}", markdown_nodes.len());
    info!("处理时间: {:?}", duration);
    for (i, node) in markdown_nodes.iter().take(3).enumerate() {
        let header = node.metadata.get("header")
            .map(|h| h.as_str().unwrap_or(""))
            .unwrap_or("(无标题)");
        let path = node.metadata.get("header_path")
            .map(|p| p.as_str().unwrap_or(""))
            .unwrap_or("");
        info!("节点 {}: 标题='{}', 路径='{}'", i + 1, header, path);
    }

    // 4. Sentence Window Parser Demo
    info!("\n🪟 4. Sentence Window Parser Demo");
    info!("==================================");
    
    let window_parser = SentenceWindowNodeParser::new()
        .with_window_size(1)
        .with_window_metadata_key("context");
    let start = Instant::now();
    
    let window_nodes = <SentenceWindowNodeParser as NodeParser>::parse_nodes(&window_parser, &[Document::new(text_content)], false).await?;
    let duration = start.elapsed();
    
    info!("生成节点数: {}", window_nodes.len());
    info!("处理时间: {:?}", duration);
    for (i, node) in window_nodes.iter().take(2).enumerate() {
        let context = node.metadata.get("context")
            .map(|c| c.as_str().unwrap_or(""))
            .unwrap_or("");
        info!("节点 {}: 内容='{}', 上下文长度={}", 
              i + 1, 
              node.content.trim().chars().take(30).collect::<String>() + "...",
              context.len());
    }

    // 5. Code Splitter Demo
    info!("\n💻 5. Code Splitter Demo");
    info!("========================");
    
    let code_splitter = CodeSplitter::from_defaults(
        ProgrammingLanguage::Rust,
        20,   // chunk_lines
        5,    // chunk_lines_overlap
        1000  // max_chars
    )?;
    let start = Instant::now();
    
    let code_nodes = <CodeSplitter as NodeParser>::parse_nodes(&code_splitter, &[Document::new(code_content)], false).await?;
    let duration = start.elapsed();
    
    info!("生成节点数: {}", code_nodes.len());
    info!("处理时间: {:?}", duration);
    for (i, node) in code_nodes.iter().take(3).enumerate() {
        let first_line = node.content.lines().next().unwrap_or("").trim();
        info!("节点 {}: {}", i + 1, first_line);
    }

    // 6. Performance Comparison
    info!("\n⚡ 6. Performance Comparison");
    info!("============================");

    let test_doc = Document::new(text_content.repeat(10)); // 10x larger content

    // Test SentenceSplitter performance
    let start = Instant::now();
    let sentence_splitter = SentenceSplitter::from_defaults(200, 40)?;
    let input = TransformInput::Document(test_doc.clone());
    let sentence_nodes = sentence_splitter.transform(input).await?;
    let sentence_duration = start.elapsed();
    info!("SentenceSplitter: {} 节点, {:?}", sentence_nodes.len(), sentence_duration);

    // Test TokenTextSplitter performance
    let start = Instant::now();
    let token_splitter = TokenTextSplitter::from_defaults(150, 30)?;
    let input = TransformInput::Document(test_doc.clone());
    let token_nodes = token_splitter.transform(input).await?;
    let token_duration = start.elapsed();
    info!("TokenTextSplitter: {} 节点, {:?}", token_nodes.len(), token_duration);

    // Test MarkdownNodeParser performance
    let start = Instant::now();
    let md_parser = MarkdownNodeParser::new();
    let md_nodes = <MarkdownNodeParser as NodeParser>::parse_nodes(&md_parser, &[test_doc], false).await?;
    let md_duration = start.elapsed();
    info!("MarkdownNodeParser: {} 节点, {:?}", md_nodes.len(), md_duration);

    // 7. Configuration Examples
    info!("\n⚙️  7. Configuration Examples");
    info!("=============================");
    
    // Different configurations for different use cases
    let _qa_splitter = SentenceSplitter::from_defaults(300, 50)?; // Q&A systems
    let _summary_splitter = SentenceSplitter::from_defaults(800, 100)?; // Summarization
    let _search_splitter = SentenceSplitter::from_defaults(500, 75)?; // Semantic search
    
    info!("Q&A 分割器: chunk_size=300, overlap=50");
    info!("摘要分割器: chunk_size=800, overlap=100");
    info!("搜索分割器: chunk_size=500, overlap=75");

    // Markdown configurations
    let doc_config = MarkdownConfig::for_documentation();
    let blog_config = MarkdownConfig::for_blog_posts();
    let readme_config = MarkdownConfig::for_readme();
    
    info!("文档配置: max_depth={}, min_length={}", 
          doc_config.max_header_depth, doc_config.min_section_length);
    info!("博客配置: max_depth={}, separator='{}'", 
          blog_config.max_header_depth, blog_config.header_path_separator);
    info!("README配置: max_depth={}, min_length={}", 
          readme_config.max_header_depth, readme_config.min_section_length);

    info!("\n✅ Parser Showcase 完成!");
    info!("\n💡 使用建议:");
    info!("   • 通用文档处理: 使用 SentenceSplitter");
    info!("   • LLM 应用: 使用 TokenTextSplitter 控制 token 数量");
    info!("   • Markdown 文档: 使用 MarkdownNodeParser 保持结构");
    info!("   • 问答系统: 使用 SentenceWindowNodeParser 提供上下文");
    info!("   • 代码分析: 使用 CodeSplitter 保持代码结构");

    Ok(())
}
