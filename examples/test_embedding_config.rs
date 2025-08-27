#!/usr/bin/env cargo
//! 测试不同 embedding 配置的简单脚本
//! 
//! 使用方法：
//! ```bash
//! # 测试 FastEmbed (默认)
//! cargo run --bin test_embedding_config --features fastembed
//! 
//! # 测试 OpenAI embedding (需要 API 密钥)
//! export OPENAI_API_KEY="your-key"
//! cargo run --bin test_embedding_config --features "fastembed,api" -- --provider openai
//! 
//! # 测试 Gemini embedding (需要 API 密钥)
//! export GEMINI_API_KEY="your-key"
//! cargo run --bin test_embedding_config --features "fastembed,api" -- --provider gemini
//! ```

use std::env;
use std::sync::Arc;
use cheungfun_core::traits::Embedder;
use cheungfun_integrations::FastEmbedder;
use siumai::prelude::*;
use async_trait::async_trait;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args: Vec<String> = env::args().collect();
    let provider = if args.len() > 2 && args[1] == "--provider" {
        args[2].clone()
    } else {
        "fastembed".to_string()
    };

    println!("🧪 测试 Embedding 配置");
    println!("📊 提供商: {}", provider);
    println!();

    // 创建 embedder
    let embedder = create_test_embedder(&provider).await?;
    
    println!("✅ Embedder 创建成功!");
    println!("📏 维度: {}", embedder.dimension());
    println!("🏷️  模型: {}", embedder.model_name());
    println!();

    // 测试单个文本嵌入
    let test_text = "Unity is a powerful game development engine.";
    println!("🔍 测试文本: \"{}\"", test_text);
    
    let start = std::time::Instant::now();
    let embedding = embedder.embed(test_text).await?;
    let duration = start.elapsed();
    
    println!("⚡ 嵌入完成! 耗时: {:?}", duration);
    println!("📊 嵌入向量长度: {}", embedding.len());
    println!("🔢 前5个值: {:?}", &embedding[..5.min(embedding.len())]);
    println!();

    // 测试批量嵌入
    let test_texts = vec![
        "Unity C# scripting tutorial",
        "Game development with Unity",
        "3D graphics programming",
    ];
    
    println!("📦 测试批量嵌入 ({} 个文本):", test_texts.len());
    let start = std::time::Instant::now();
    let embeddings = embedder.embed_batch(test_texts.iter().map(|s| s.as_ref()).collect()).await?;
    let duration = start.elapsed();
    
    println!("⚡ 批量嵌入完成! 耗时: {:?}", duration);
    println!("📊 结果数量: {}", embeddings.len());
    for (i, emb) in embeddings.iter().enumerate() {
        println!("  文本 {}: 维度 {}, 前3个值: {:?}", 
                 i + 1, emb.len(), &emb[..3.min(emb.len())]);
    }
    println!();

    // 健康检查
    println!("🏥 执行健康检查...");
    embedder.health_check().await?;
    println!("✅ 健康检查通过!");
    println!();

    // 显示元数据
    let metadata = embedder.metadata();
    println!("📋 Embedder 元数据:");
    for (key, value) in metadata {
        println!("  {}: {}", key, value);
    }

    println!();
    println!("🎉 所有测试完成!");

    Ok(())
}

/// 创建测试用的 embedder
async fn create_test_embedder(provider: &str) -> Result<Arc<dyn Embedder>, Box<dyn std::error::Error>> {
    match provider {
        "fastembed" => {
            println!("🚀 初始化 FastEmbed embedder...");
            let embedder = FastEmbedder::new().await?;
            Ok(Arc::new(embedder))
        }
        "openai" => {
            #[cfg(feature = "api")]
            {
                println!("🚀 初始化 OpenAI embedder...");
                let api_key = env::var("OPENAI_API_KEY")
                    .map_err(|_| "OPENAI_API_KEY environment variable not set")?;

                let embedder = cheungfun_integrations::ApiEmbedder::builder()
                    .openai(&api_key)
                    .model("text-embedding-3-small")
                    .build()
                    .await?;
                Ok(Arc::new(embedder))
            }
            #[cfg(not(feature = "api"))]
            {
                Err("OpenAI embedding requires 'api' feature. Compile with --features \"fastembed,api\"".into())
            }
        }
        "gemini" => {
            println!("🚀 初始化 Gemini embedder...");
            let api_key = env::var("GEMINI_API_KEY")
                .map_err(|_| "GEMINI_API_KEY environment variable not set")?;

            let client = Siumai::builder()
                .gemini()
                .api_key(&api_key)
                .model("gemini-embedding-001")
                .build()
                .await?;

            let embedder = GeminiEmbedderWrapper::new(client);
            Ok(Arc::new(embedder))
        }
        _ => Err(format!("Unsupported provider: {}. Use: fastembed, openai, or gemini", provider).into()),
    }
}

/// Gemini Embedder 包装器
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
        let model_name = "gemini-embedding-001".to_string();
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
