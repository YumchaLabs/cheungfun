# Markdown文件夹RAG问答示例

这个示例展示了如何使用Cheungfun框架构建一个完整的RAG（检索增强生成）系统，能够：

1. 🗂️ 批量加载指定文件夹下的所有markdown文件
2. 🤖 使用真实LLM API进行嵌入和问答
3. 💾 支持内存或SQLite存储
4. 🔍 提供语义搜索和智能问答
5. 🎯 提供交互式查询界面

## 功能特性

- ✅ **完整的RAG流程**：文档加载 → 文本分割 → 向量嵌入 → 存储 → 检索 → 生成回答
- ✅ **真实LLM集成**：支持OpenAI GPT和本地Ollama模型
- ✅ **高性能嵌入**：使用FastEmbed进行快速文本嵌入
- ✅ **灵活存储**：支持内存存储和SQLite持久化存储
- ✅ **智能检索**：基于语义相似度的文档片段检索
- ✅ **交互式界面**：命令行交互式问答体验
- ✅ **详细统计**：提供处理统计和系统状态信息

## 快速开始

### 1. 环境准备

确保你已经安装了Rust和Cargo：

```bash
# 检查Rust版本
rustc --version
cargo --version
```

### 2. 设置API密钥（可选）

如果要使用OpenAI API：

```bash
# Linux/macOS
export OPENAI_API_KEY="your-api-key-here"

# Windows
set OPENAI_API_KEY=your-api-key-here
```

如果没有设置API密钥，系统会自动使用本地Ollama。

### 3. 安装Ollama（如果不使用OpenAI）

```bash
# 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 启动Ollama服务
ollama serve

# 下载模型（在另一个终端）
ollama pull llama3.2
```

### 4. 运行示例

```bash
# 进入项目目录
cd cheungfun

# 运行示例（使用FastEmbed嵌入）
cargo run --example markdown_rag_example --features "fastembed"

# 或者使用SQLite存储（暂未实现）
cargo run --example markdown_rag_example --features "fastembed,sqlite"
```

## 使用方法

### 1. 准备文档

将你的markdown文件放在一个文件夹中，例如：

```
docs/
├── rust_basics.md
├── rag_introduction.md
├── ai_development.md
└── your_custom_docs.md
```

### 2. 配置环境变量

```bash
# 设置文档文件夹路径
export DOCS_FOLDER="./your-docs-folder"

# 设置文本分块大小
export CHUNK_SIZE=500

# 设置检索返回的文档数量
export TOP_K=5
```

### 3. 启动系统

运行示例后，系统会：

1. **初始化组件**：嵌入器、向量存储、LLM客户端
2. **构建索引**：扫描文档、分割文本、生成嵌入、存储向量
3. **启动问答**：进入交互式问答模式

### 4. 交互式问答

```
🤔 您的问题: 什么是Rust？

🔍 正在搜索相关内容...

🤖 AI回答:
──────────────────────────────────────────────────
Rust是一种系统编程语言，专注于安全性、速度和并发性。它由Mozilla开发，旨在解决C和C++中常见的内存安全问题，同时保持高性能。Rust通过所有权系统在编译时防止内存泄漏、悬空指针和数据竞争等问题。
──────────────────────────────────────────────────

📚 参考来源 (3 个相关片段):
  1. [相似度: 0.892] rust_basics.md
     预览: # Rust编程语言基础\n\n## 什么是Rust？\n\nRust是一种系统编程语言，专注于安全性、速度和并发性...
  2. [相似度: 0.756] rust_basics.md  
     预览: ## Rust的核心特性\n\n### 1. 内存安全\nRust通过所有权系统（Ownership System）...
  3. [相似度: 0.689] rust_basics.md
     预览: ### 3. 并发安全\nRust的类型系统防止数据竞争，使并发编程更加安全...

⚡ 性能信息:
  - 查询时间: 1.234s
  - Token使用: 156 prompt + 89 completion = 245 total
```

### 5. 可用命令

在问答模式中，你可以使用以下命令：

- `help` - 显示帮助信息
- `stats` - 显示系统统计信息
- `quit` 或 `exit` - 退出程序

## 配置选项

### 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DOCS_FOLDER` | `./docs` | markdown文档文件夹路径 |
| `CHUNK_SIZE` | `500` | 文本分块大小 |
| `CHUNK_OVERLAP` | `50` | 分块重叠大小 |
| `TOP_K` | `5` | 检索返回的文档数量 |
| `OPENAI_API_KEY` | - | OpenAI API密钥 |

### 代码配置

你也可以直接修改代码中的`RagConfig`结构体：

```rust
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
```

## 示例查询

以下是一些示例查询，你可以尝试：

### 关于Rust的问题
- "Rust有哪些核心特性？"
- "Rust适用于哪些应用领域？"
- "Rust如何保证内存安全？"

### 关于RAG的问题
- "什么是RAG系统？"
- "RAG系统有哪些组件？"
- "RAG的工作原理是什么？"

### 关于AI开发的问题
- "AI开发的生命周期包括哪些阶段？"
- "有哪些常用的AI开发工具？"
- "AI开发的最佳实践是什么？"

## 故障排除

### 常见问题

1. **FastEmbed初始化失败**
   ```
   解决方案：确保网络连接正常，FastEmbed需要下载预训练模型
   ```

2. **Ollama连接失败**
   ```bash
   # 检查Ollama是否运行
   curl http://localhost:11434/api/tags
   
   # 重启Ollama服务
   ollama serve
   ```

3. **OpenAI API调用失败**
   ```
   解决方案：检查API密钥是否正确设置，账户是否有足够余额
   ```

4. **文档加载失败**
   ```
   解决方案：检查文档文件夹路径是否正确，确保有.md文件
   ```

### 调试模式

启用详细日志：

```bash
RUST_LOG=debug cargo run --example markdown_rag_example --features "fastembed"
```

## 扩展功能

### 添加新的文档类型

修改`DirectoryLoader`配置以支持更多文件类型：

```rust
let loader_config = LoaderConfig::new()
    .with_include_extensions(vec![
        "md".to_string(),
        "txt".to_string(),
        "rst".to_string(),
    ])
    .with_max_depth(Some(10))
    .with_continue_on_error(true);
```

### 自定义嵌入模型

替换FastEmbed为其他嵌入器：

```rust
// 使用OpenAI嵌入
let embedder = Arc::new(ApiEmbedder::openai("your-api-key").await?);

// 使用Candle本地嵌入
let embedder = Arc::new(CandleEmbedder::from_pretrained("model-name").await?);
```

### 添加SQLite存储

实现SQLite向量存储（当前为TODO）：

```rust
// TODO: 实现SQLite向量存储
let vector_store = Arc::new(SqliteVectorStore::new("rag.db").await?);
```

## 性能优化

1. **调整分块大小**：根据文档类型调整`chunk_size`和`chunk_overlap`
2. **优化检索数量**：调整`top_k`值平衡准确性和性能
3. **使用GPU加速**：如果有GPU，可以使用Candle GPU嵌入器
4. **缓存嵌入**：对于大量文档，考虑缓存嵌入结果

## 贡献

欢迎提交Issue和Pull Request来改进这个示例！

## 许可证

本示例遵循Cheungfun项目的许可证。
