# QdrantVectorStore 使用指南

QdrantVectorStore 是 Cheungfun 项目中的生产级向量存储实现，使用 Qdrant 作为后端数据库。它提供了高性能、可扩展的向量存储和检索功能。

## 🚀 快速开始

### 1. 启动 Qdrant 服务器

使用 Docker 启动 Qdrant：

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. 基本使用

```rust
use cheungfun_integrations::{QdrantVectorStore, QdrantConfig};
use cheungfun_core::traits::{VectorStore, DistanceMetric};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建配置
    let config = QdrantConfig::new("http://localhost:6334", "my_collection", 384)
        .with_distance_metric(DistanceMetric::Cosine)
        .with_create_collection_if_missing(true);

    // 创建向量存储
    let store = QdrantVectorStore::new(config).await?;

    // 健康检查
    store.health_check().await?;
    println!("✅ Qdrant 连接成功！");

    Ok(())
}
```

## 📋 配置选项

### QdrantConfig 构建器

```rust
let config = QdrantConfig::new("http://localhost:6334", "collection_name", 384)
    .with_api_key("your_api_key")                    // 可选：API密钥
    .with_distance_metric(DistanceMetric::Cosine)    // 距离度量
    .with_timeout(Duration::from_secs(30))           // 请求超时
    .with_max_retries(3)                             // 最大重试次数
    .with_create_collection_if_missing(true);        // 自动创建集合
```

### 支持的距离度量

- `DistanceMetric::Cosine` - 余弦相似度（推荐）
- `DistanceMetric::Euclidean` - 欧几里得距离
- `DistanceMetric::DotProduct` - 点积
- `DistanceMetric::Manhattan` - 曼哈顿距离

## 🔧 核心功能

### 添加向量

```rust
use cheungfun_core::types::{Node, ChunkInfo};
use uuid::Uuid;

// 创建节点
let source_doc_id = Uuid::new_v4();
let chunk_info = ChunkInfo::new(0, 100, 0);
let mut node = Node::new("这是文档内容", source_doc_id, chunk_info);

// 设置嵌入向量（通常由嵌入模型生成）
node.embedding = Some(vec![0.1, 0.2, 0.3, /* ... 384维向量 */]);

// 添加到存储
let ids = store.add(vec![node]).await?;
println!("✅ 添加了 {} 个节点", ids.len());
```

### 向量搜索

```rust
use cheungfun_core::types::{Query, SearchMode};

let query = Query::builder()
    .text("搜索查询")
    .embedding(query_vector)  // 查询向量
    .top_k(5)                 // 返回前5个结果
    .similarity_threshold(0.7) // 相似度阈值
    .search_mode(SearchMode::Vector)
    .build();

let results = store.search(&query).await?;

for result in results {
    println!("相似度: {:.4}, 内容: {}", result.score, result.node.content);
}
```

### 批量操作

```rust
// 批量添加（自动分批处理）
let nodes = create_many_nodes(); // 创建大量节点
let ids = store.batch_upsert(nodes, 100).await?; // 每批100个

println!("✅ 批量添加了 {} 个节点", ids.len());
```

### 高级搜索

```rust
// 使用过滤器的高级搜索
let results = store.search_with_filter(
    query_vector,
    10,                    // top_k
    None,                  // 过滤器（暂未实现）
    Some(0.8),            // 分数阈值
).await?;
```

## 📊 监控和统计

### 获取统计信息

```rust
// 基本统计
let count = store.count().await?;
let stats = store.stats().await?;

println!("节点总数: {}", count);
println!("插入操作: {}", stats.insert_operations);
println!("搜索操作: {}", stats.search_operations);

// 详细的集合统计
let collection_stats = store.collection_stats().await?;
println!("集合统计: {:?}", collection_stats);
```

### 健康检查

```rust
match store.health_check().await {
    Ok(_) => println!("✅ Qdrant 服务正常"),
    Err(e) => println!("❌ Qdrant 服务异常: {}", e),
}
```

## 🛠️ 高级功能

### 集合管理

```rust
// 获取集合信息
let info = store.collection_info().await?;
println!("集合状态: {:?}", info.status());

// 创建字段索引（提升搜索性能）
store.create_index("category").await?;

// 优化集合
store.optimize_collection().await?;
```

### 元数据管理

```rust
// 添加带元数据的节点
let mut node = Node::new("内容", source_doc_id, chunk_info);
node.metadata.insert("category".to_string(), "技术".into());
node.metadata.insert("priority".to_string(), 5.into());

store.add(vec![node]).await?;
```

## 🔍 最佳实践

### 1. 向量维度选择

```rust
// 常用的嵌入模型维度
let config = QdrantConfig::new("http://localhost:6334", "collection", 384);  // sentence-transformers
// let config = QdrantConfig::new("http://localhost:6334", "collection", 768);  // BERT
// let config = QdrantConfig::new("http://localhost:6334", "collection", 1536); // OpenAI text-embedding-ada-002
```

### 2. 批量操作优化

```rust
// 大量数据时使用批量操作
if nodes.len() > 100 {
    let ids = store.batch_upsert(nodes, 50).await?; // 每批50个
} else {
    let ids = store.add(nodes).await?;
}
```

### 3. 错误处理

```rust
use cheungfun_core::CheungfunError;

match store.search(&query).await {
    Ok(results) => {
        // 处理结果
    }
    Err(CheungfunError::VectorStore { message }) => {
        eprintln!("向量存储错误: {}", message);
    }
    Err(CheungfunError::NotFound { resource }) => {
        eprintln!("资源未找到: {}", resource);
    }
    Err(e) => {
        eprintln!("其他错误: {}", e);
    }
}
```

## 🚀 性能优化

### 1. 连接配置

```rust
let config = QdrantConfig::new("http://localhost:6334", "collection", 384)
    .with_timeout(Duration::from_secs(60))  // 增加超时时间
    .with_max_retries(5);                   // 增加重试次数
```

### 2. 批量大小调优

```rust
// 根据数据大小和网络条件调整批量大小
let batch_size = if vector_dimension > 1000 { 20 } else { 100 };
store.batch_upsert(nodes, batch_size).await?;
```

## 🔧 故障排除

### 常见问题

1. **连接失败**
   ```
   确保 Qdrant 服务器正在运行：
   docker ps | grep qdrant
   ```

2. **维度不匹配**
   ```
   检查向量维度是否与配置一致
   ```

3. **内存不足**
   ```
   使用批量操作减少内存使用
   ```

## 📚 示例代码

完整的使用示例请参考：
- `examples/src/qdrant_vector_store_demo.rs` - 完整演示
- `cheungfun-integrations/src/vector_stores/qdrant.rs` - 单元测试

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进 QdrantVectorStore！

## 📄 许可证

本项目采用 MIT 许可证。
