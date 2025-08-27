# Unity C# RAG CLI - Embedding 配置示例

本文档展示如何在 Unity C# RAG CLI 中配置不同的 embedding 提供商。

## 支持的 Embedding 提供商

### 1. FastEmbed (默认)
使用本地 FastEmbed 模型，无需 API 密钥：

```bash
# 使用默认 FastEmbed
cargo run --bin unity_csharp_rag -- /path/to/unity/project

# 指定 FastEmbed 模型
cargo run --bin unity_csharp_rag -- /path/to/unity/project \
  --embedding-provider fastembed \
  --embedding-model "BAAI/bge-large-en-v1.5"
```

**支持的 FastEmbed 模型：**
- `BAAI/bge-small-en-v1.5` (384维，默认)
- `BAAI/bge-large-en-v1.5` (1024维，高质量)
- `intfloat/multilingual-e5-base` (768维，多语言)
- `sentence-transformers/all-MiniLM-L6-v2` (384维，快速)
- `jinaai/jina-embeddings-v2-base-code` (768维，代码专用)

### 2. OpenAI Embedding
使用 OpenAI 的 embedding API：

```bash
# 设置 API 密钥
export OPENAI_API_KEY="your-openai-api-key"

# 使用 OpenAI embedding
cargo run --bin unity_csharp_rag -- /path/to/unity/project \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small
```

**支持的 OpenAI 模型：**
- `text-embedding-3-small` (1536维，推荐)
- `text-embedding-3-large` (3072维，高质量)
- `text-embedding-ada-002` (1536维，经典)

### 3. Gemini Embedding
使用 Google Gemini 的 embedding API：

```bash
# 设置 API 密钥
export GEMINI_API_KEY="your-gemini-api-key"

# 使用 Gemini embedding
cargo run --bin unity_csharp_rag -- /path/to/unity/project \
  --embedding-provider gemini \
  --embedding-model gemini-embedding-001
```

**支持的 Gemini 模型：**
- `gemini-embedding-001` (3072维)

## 完整示例

### 使用 Gemini embedding + OpenAI LLM
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"

cargo run --bin unity_csharp_rag -- /path/to/unity/project \
  --embedding-provider gemini \
  --embedding-model gemini-embedding-001 \
  --llm openai \
  --model gpt-4 \
  --strategy enterprise \
  --verbose
```

### 使用 OpenAI embedding + Ollama LLM
```bash
export OPENAI_API_KEY="your-openai-api-key"

# 确保 Ollama 正在运行
ollama serve
ollama pull llama3.2

cargo run --bin unity_csharp_rag -- /path/to/unity/project \
  --embedding-provider openai \
  --embedding-model text-embedding-3-large \
  --llm ollama \
  --model llama3.2 \
  --top-k 10
```

### 本地运行（无需 API 密钥）
```bash
# 使用 FastEmbed + Ollama，完全本地运行
ollama serve
ollama pull llama3.2

cargo run --bin unity_csharp_rag -- /path/to/unity/project \
  --embedding-provider fastembed \
  --embedding-model "BAAI/bge-large-en-v1.5" \
  --llm ollama \
  --model llama3.2
```

## 性能对比

| Provider | 模型 | 维度 | 速度 | 质量 | 成本 |
|----------|------|------|------|------|------|
| FastEmbed | bge-small-en-v1.5 | 384 | 很快 | 好 | 免费 |
| FastEmbed | bge-large-en-v1.5 | 1024 | 快 | 很好 | 免费 |
| OpenAI | text-embedding-3-small | 1536 | 中等 | 很好 | 低 |
| OpenAI | text-embedding-3-large | 3072 | 慢 | 优秀 | 中等 |
| Gemini | gemini-embedding-001 | 3072 | 中等 | 优秀 | 低 |

## 选择建议

- **开发/测试**: 使用 FastEmbed (免费，快速)
- **生产环境**: 使用 OpenAI 或 Gemini (质量更高)
- **多语言项目**: 使用 multilingual-e5-base 或 Gemini
- **代码专用**: 使用 jina-embeddings-v2-base-code
- **成本敏感**: 使用 FastEmbed 或 Gemini
- **最高质量**: 使用 text-embedding-3-large 或 gemini-embedding-001

## 编译要求

### Feature 标志

- **默认 (FastEmbed only)**: `cargo run --bin unity_csharp_rag --features fastembed`
- **包含 API embedders**: `cargo run --bin unity_csharp_rag --features "fastembed,api"`

### 注意事项

1. **OpenAI 和 Gemini embedding** 需要启用 `api` feature
2. **FastEmbed** 只需要 `fastembed` feature（默认）
3. 如果不启用 `api` feature，使用 OpenAI 或 Gemini 会显示错误提示

## 故障排除

### 常见错误

1. **API 密钥未设置**
   ```
   Error: OPENAI_API_KEY environment variable not set
   ```
   解决：设置对应的环境变量

2. **模型不支持**
   ```
   Error: Unsupported embedding provider: xyz
   ```
   解决：检查提供商名称拼写，支持的有：fastembed, openai, gemini

3. **Feature 未启用**
   ```
   Error: OpenAI embedding requires 'api' feature to be enabled
   ```
   解决：使用 `--features "fastembed,api"` 编译

4. **网络连接问题**
   ```
   Error: Failed to create Gemini client: network error
   ```
   解决：检查网络连接和 API 密钥有效性

### 调试技巧

使用 `--verbose` 参数查看详细日志：
```bash
cargo run --bin unity_csharp_rag --features "fastembed,api" -- /path/to/unity/project \
  --embedding-provider gemini \
  --verbose
```
