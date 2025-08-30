//! Contextual Compression Example
//!
//! This example demonstrates contextual compression techniques that reduce
//! the amount of irrelevant information in retrieved contexts while preserving
//! the most relevant parts for the query. This is especially useful for
//! long documents where only small portions are relevant.
//!
//! # Features Demonstrated
//!
//! - **LLM-based Compression**: Use LLM to extract relevant portions
//! - **Keyword-based Compression**: Extract sentences containing query keywords
//! - **Similarity-based Compression**: Keep only high-similarity chunks
//! - **Hybrid Compression**: Combine multiple compression strategies
//! - **Context Window Management**: Optimize for LLM context limits
//!
//! # Usage
//!
//! ```bash
//! # Basic advanced chain with all 4 postprocessors
//! cargo run --bin contextual_compression --features fastembed -- \
//!     --query "machine learning algorithms" \
//!     --compression-method "advanced_chain" \
//!     --use-keyword-filter \
//!     --use-metadata-filter
//!
//! # Compare different strategies
//! cargo run --bin contextual_compression --features fastembed -- \
//!     --query "deep learning neural networks" \
//!     --compare-strategies
//!
//! # Traditional compression methods
//! cargo run --bin contextual_compression --features fastembed -- \
//!     --query "artificial intelligence" \
//!     --compression-method "llm_based" \
//!     --compression-ratio 0.3
//! ```

use cheungfun::prelude::*;
use cheungfun_integrations::FastEmbedder;
use cheungfun_query::{
    advanced::fusion::DistributionBasedFusion,
    postprocessor::{
        DocumentCompressor, KeywordFilter, KeywordFilterConfig, MetadataFilter,
        MetadataFilterConfig, NodePostprocessor, SentenceEmbeddingConfig,
        SentenceEmbeddingOptimizer, SimilarityFilter, SimilarityFilterConfig,
    },
};
use clap::Parser;
use siumai::{prelude::*, ChatMessage as SiumaiChatMessage};
use std::{collections::HashSet, sync::Arc, time::Instant};
use tracing::{debug, info};

#[derive(Parser, Debug)]
#[command(name = "contextual-compression")]
#[command(about = "Demonstrates contextual compression for efficient retrieval")]
struct Args {
    /// Query to search for
    #[arg(long, default_value = "machine learning algorithms")]
    query: String,

    /// Compression method: llm_based, keyword_based, similarity_based, hybrid, sentence_embedding, postprocessor_chain, advanced_chain
    #[arg(long, default_value = "advanced_chain")]
    compression_method: String,

    /// Target compression ratio (0.0-1.0, lower = more compression)
    #[arg(long, default_value = "0.3")]
    compression_ratio: f32,

    /// Number of results to retrieve before compression
    #[arg(long, default_value = "10")]
    top_k: usize,

    /// Enable keyword filtering in the processing chain
    #[arg(long)]
    use_keyword_filter: bool,

    /// Enable metadata filtering in the processing chain
    #[arg(long)]
    use_metadata_filter: bool,

    /// Compare different postprocessor strategies
    #[arg(long)]
    compare_strategies: bool,

    /// Enable verbose output
    #[arg(long)]
    verbose: bool,
}

/// Contextual compression strategies.
#[derive(Clone)]
pub enum CompressionStrategy {
    /// Use LLM to extract relevant portions
    LlmBased {
        /// LLM client for compression
        client: Siumai,
        /// Target compression ratio
        target_ratio: f32,
    },
    /// Extract sentences containing query keywords
    KeywordBased {
        /// Keywords to look for
        keywords: Vec<String>,
        /// Minimum keyword matches per sentence
        min_matches: usize,
    },
    /// Keep only high-similarity chunks
    SimilarityBased {
        /// Minimum similarity threshold
        threshold: f32,
        /// Maximum chunks to keep
        max_chunks: usize,
    },
    /// Combine multiple strategies
    Hybrid {
        /// Primary strategy
        primary: Box<CompressionStrategy>,
        /// Secondary strategy
        secondary: Box<CompressionStrategy>,
        /// Weight for primary strategy (0.0-1.0)
        primary_weight: f32,
    },
}

impl std::fmt::Debug for CompressionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LlmBased { target_ratio, .. } => f
                .debug_struct("LlmBased")
                .field("target_ratio", target_ratio)
                .finish(),
            Self::KeywordBased {
                keywords,
                min_matches,
            } => f
                .debug_struct("KeywordBased")
                .field("keywords", keywords)
                .field("min_matches", min_matches)
                .finish(),
            Self::SimilarityBased {
                threshold,
                max_chunks,
            } => f
                .debug_struct("SimilarityBased")
                .field("threshold", threshold)
                .field("max_chunks", max_chunks)
                .finish(),
            Self::Hybrid { primary_weight, .. } => f
                .debug_struct("Hybrid")
                .field("primary_weight", primary_weight)
                .finish(),
        }
    }
}

/// Contextual compressor that reduces context size while preserving relevance.
#[derive(Debug)]
pub struct ContextualCompressor {
    /// Compression strategy to use
    strategy: CompressionStrategy,
}

impl ContextualCompressor {
    /// Create a new contextual compressor.
    pub fn new(strategy: CompressionStrategy) -> Self {
        Self { strategy }
    }

    /// Compress a list of scored nodes based on the query.
    pub async fn compress(
        &self,
        nodes: Vec<ScoredNode>,
        query: &str,
    ) -> std::result::Result<Vec<ScoredNode>, ExampleError> {
        match &self.strategy {
            CompressionStrategy::LlmBased {
                client,
                target_ratio,
            } => {
                self.llm_based_compression(nodes, query, client, *target_ratio)
                    .await
            }
            CompressionStrategy::KeywordBased {
                keywords,
                min_matches,
            } => Ok(self.keyword_based_compression(nodes, keywords, *min_matches)),
            CompressionStrategy::SimilarityBased {
                threshold,
                max_chunks,
            } => Ok(self.similarity_based_compression(nodes, *threshold, *max_chunks)),
            CompressionStrategy::Hybrid {
                primary,
                secondary,
                primary_weight,
            } => {
                self.hybrid_compression(nodes, query, primary, secondary, *primary_weight)
                    .await
            }
        }
    }

    /// LLM-based compression using intelligent extraction.
    async fn llm_based_compression(
        &self,
        nodes: Vec<ScoredNode>,
        query: &str,
        client: &Siumai,
        target_ratio: f32,
    ) -> std::result::Result<Vec<ScoredNode>, ExampleError> {
        info!(
            "Applying LLM-based compression with target ratio: {:.2}",
            target_ratio
        );

        let mut compressed_nodes = Vec::new();

        for node in nodes {
            let compression_prompt = format!(
                r#"Extract the most relevant portions of the following text for answering this query: "{}"

Text to compress:
{}

Instructions:
1. Extract only the portions directly relevant to the query
2. Maintain context and coherence
3. Target compression ratio: {:.0}% of original length
4. Preserve key facts and relationships

Compressed text:"#,
                query,
                node.node.content,
                target_ratio * 100.0
            );

            let messages = vec![SiumaiChatMessage::user(&compression_prompt).build()];

            match client.chat(messages).await {
                Ok(response) => {
                    let compressed_content = match &response.content {
                        MessageContent::Text(text) => text.clone(),
                        _ => node.node.content.clone(), // Fallback to original
                    };

                    // Create compressed node
                    let mut compressed_node = node.clone();
                    compressed_node.node.content = compressed_content;

                    // Adjust score based on compression quality
                    let compression_quality = self.calculate_compression_quality(
                        &node.node.content,
                        &compressed_node.node.content,
                        query,
                    );
                    compressed_node.score *= compression_quality;

                    compressed_nodes.push(compressed_node);
                }
                Err(e) => {
                    debug!("LLM compression failed for node: {}", e);
                    // Keep original node if compression fails
                    compressed_nodes.push(node);
                }
            }
        }

        Ok(compressed_nodes)
    }

    /// Keyword-based compression by extracting relevant sentences.
    fn keyword_based_compression(
        &self,
        nodes: Vec<ScoredNode>,
        keywords: &[String],
        min_matches: usize,
    ) -> Vec<ScoredNode> {
        info!(
            "Applying keyword-based compression with {} keywords",
            keywords.len()
        );

        let mut compressed_nodes = Vec::new();

        for node in nodes {
            let sentences: Vec<&str> = node
                .node
                .content
                .split(&['.', '!', '?'])
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            let mut relevant_sentences = Vec::new();

            for sentence in &sentences {
                let sentence_lower = sentence.to_lowercase();
                let matches = keywords
                    .iter()
                    .filter(|keyword| sentence_lower.contains(&keyword.to_lowercase()))
                    .count();

                if matches >= min_matches {
                    relevant_sentences.push(*sentence);
                }
            }

            if !relevant_sentences.is_empty() {
                let compressed_content = relevant_sentences.join(". ");
                let mut compressed_node = node.clone();
                compressed_node.node.content = compressed_content;

                // Adjust score based on keyword density
                let total_sentences = sentences.len();
                let keyword_density = relevant_sentences.len() as f32 / total_sentences as f32;
                compressed_node.score *= keyword_density;

                compressed_nodes.push(compressed_node);
            }
        }

        compressed_nodes
    }

    /// Similarity-based compression keeping only high-similarity chunks.
    fn similarity_based_compression(
        &self,
        mut nodes: Vec<ScoredNode>,
        threshold: f32,
        max_chunks: usize,
    ) -> Vec<ScoredNode> {
        info!(
            "Applying similarity-based compression with threshold: {:.2}",
            threshold
        );

        // Filter by similarity threshold
        nodes.retain(|node| node.score >= threshold);

        // Sort by score and take top chunks
        nodes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        nodes.truncate(max_chunks);

        nodes
    }

    /// Hybrid compression combining multiple strategies.
    fn hybrid_compression<'a>(
        &'a self,
        nodes: Vec<ScoredNode>,
        query: &'a str,
        _primary: &CompressionStrategy,
        _secondary: &CompressionStrategy,
        primary_weight: f32,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = std::result::Result<Vec<ScoredNode>, ExampleError>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            info!(
                "Applying hybrid compression with primary weight: {:.2}",
                primary_weight
            );

            // For hybrid compression, we'll use a simpler approach to avoid recursion
            // Apply keyword-based compression as fallback
            let keywords = query.split_whitespace().map(|s| s.to_lowercase()).collect();
            let keyword_strategy = CompressionStrategy::KeywordBased {
                keywords,
                min_matches: 1,
            };

            let keyword_compressor = ContextualCompressor::new(keyword_strategy);
            let keyword_results = keyword_compressor.keyword_based_compression(
                nodes.clone(),
                &query
                    .split_whitespace()
                    .map(|s| s.to_lowercase())
                    .collect::<Vec<_>>(),
                1,
            );

            // Apply similarity-based compression as secondary
            let similarity_results = self.similarity_based_compression(nodes, 0.5, 5);

            // Combine results using weighted fusion
            let fusion =
                DistributionBasedFusion::with_weights(vec![primary_weight, 1.0 - primary_weight]);
            let fused_results = fusion.fuse_results(vec![keyword_results, similarity_results]);

            Ok(fused_results)
        })
    }

    /// Calculate compression quality score.
    fn calculate_compression_quality(&self, original: &str, compressed: &str, query: &str) -> f32 {
        // Factor 1: Compression ratio
        let compression_ratio = compressed.len() as f32 / original.len() as f32;
        let ratio_score = if compression_ratio > 0.8 {
            0.5 // Too little compression
        } else if compression_ratio < 0.1 {
            0.5 // Too much compression
        } else {
            1.0 // Good compression
        };

        // Factor 2: Query keyword preservation
        let query_words: HashSet<&str> = query.split_whitespace().collect();
        let compressed_words: HashSet<&str> = compressed.split_whitespace().collect();
        let keyword_preservation =
            query_words.intersection(&compressed_words).count() as f32 / query_words.len() as f32;

        // Factor 3: Content coherence
        let coherence_score = if compressed.split_whitespace().count() < 5 {
            0.5 // Too short to be coherent
        } else {
            1.0 // Assume coherent if reasonable length
        };

        (ratio_score * 0.3 + keyword_preservation * 0.5 + coherence_score * 0.2).clamp(0.1, 1.0)
    }
}

#[derive(Debug)]
pub enum ExampleError {
    Siumai(siumai::LlmError),
    Other(String),
}

impl std::fmt::Display for ExampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExampleError::Siumai(e) => write!(f, "Siumai error: {}", e),
            ExampleError::Other(e) => write!(f, "Error: {}", e),
        }
    }
}

impl std::error::Error for ExampleError {}

impl From<siumai::LlmError> for ExampleError {
    fn from(err: siumai::LlmError) -> Self {
        ExampleError::Siumai(err)
    }
}

impl From<cheungfun_core::CheungfunError> for ExampleError {
    fn from(err: cheungfun_core::CheungfunError) -> Self {
        ExampleError::Other(format!("Cheungfun error: {}", err))
    }
}

/// Create metadata for sample nodes.
fn create_metadata(
    category: &str,
    topic: &str,
    keywords: Vec<&str>,
    priority: &str,
) -> std::collections::HashMap<String, serde_json::Value> {
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "category".to_string(),
        serde_json::Value::String(category.to_string()),
    );
    metadata.insert(
        "topic".to_string(),
        serde_json::Value::String(topic.to_string()),
    );
    metadata.insert(
        "keywords".to_string(),
        serde_json::Value::Array(
            keywords
                .into_iter()
                .map(|k| serde_json::Value::String(k.to_string()))
                .collect(),
        ),
    );
    metadata.insert(
        "priority".to_string(),
        serde_json::Value::String(priority.to_string()),
    );
    metadata.insert(
        "source".to_string(),
        serde_json::Value::String("sample_document".to_string()),
    );
    metadata
}

/// Create sample long documents for compression testing with rich metadata.
fn create_sample_long_documents() -> Vec<ScoredNode> {
    use cheungfun_core::ChunkInfo;
    use uuid::Uuid;

    let doc_id = Uuid::new_v4();
    let chunk_info = ChunkInfo::new(Some(0), Some(1000), 0);

    vec![
        {
            let mut node = Node::new(
                r#"Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience. The field encompasses various approaches including supervised learning, unsupervised learning, and reinforcement learning. Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data. Common supervised learning algorithms include linear regression, decision trees, random forests, support vector machines, and neural networks. These algorithms are widely used in applications such as image recognition, natural language processing, and predictive analytics. The training process involves feeding the algorithm examples of input-output pairs so it can learn the underlying patterns and relationships."#.to_string(),
                doc_id,
                chunk_info.clone(),
            );
            let metadata = create_metadata(
                "technology",
                "machine learning",
                vec!["algorithms", "artificial intelligence", "data science"],
                "high",
            );
            for (key, value) in metadata {
                node = node.with_metadata(key, value);
            }
            ScoredNode::new(node, 0.9)
        },
        {
            let mut node = Node::new(
                r#"Deep learning represents a revolutionary approach within machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. The architecture of deep neural networks allows them to automatically learn hierarchical representations of data, making them particularly effective for tasks involving high-dimensional data such as images, text, and audio. Convolutional Neural Networks (CNNs) have transformed computer vision by enabling automatic feature extraction from images. Recurrent Neural Networks (RNNs) and their variants like LSTM and GRU have been instrumental in natural language processing and sequence modeling. The recent development of Transformer architectures has led to breakthrough models like BERT, GPT, and T5 that have achieved state-of-the-art performance across numerous NLP tasks."#.to_string(),
                doc_id,
                chunk_info.clone(),
            );
            let metadata = create_metadata(
                "technology",
                "deep learning",
                vec![
                    "neural networks",
                    "artificial intelligence",
                    "computer vision",
                ],
                "high",
            );
            for (key, value) in metadata {
                node = node.with_metadata(key, value);
            }
            ScoredNode::new(node, 0.8)
        },
        {
            let mut node = Node::new(
                r#"The history of artificial intelligence dates back to ancient times with myths and stories of artificial beings endowed with intelligence or consciousness. However, the modern field of AI began in the 1950s with the work of pioneers like Alan Turing, John McCarthy, and Marvin Minsky. The term 'artificial intelligence' was coined by John McCarthy in 1956 at the Dartmouth Conference. Early AI research focused on symbolic reasoning and expert systems. The field experienced several 'AI winters' - periods of reduced funding and interest - but has seen remarkable resurgence in recent decades due to advances in computing power, data availability, and algorithmic improvements. Today, AI applications are ubiquitous, from search engines and recommendation systems to autonomous vehicles and medical diagnosis tools."#.to_string(),
                doc_id,
                chunk_info.clone(),
            );
            let metadata = create_metadata(
                "technology",
                "artificial intelligence history",
                vec!["AI history", "computer science", "technology evolution"],
                "medium",
            );
            for (key, value) in metadata {
                node = node.with_metadata(key, value);
            }
            ScoredNode::new(node, 0.6)
        },
        {
            let mut node = Node::new(
                r#"Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations occur naturally, scientific evidence shows that human activities, particularly the emission of greenhouse gases like carbon dioxide from burning fossil fuels, have been the dominant driver of climate change since the mid-20th century. The effects include rising sea levels, more frequent extreme weather events, changes in precipitation patterns, and impacts on ecosystems and biodiversity. Renewable energy sources such as solar, wind, and hydroelectric power are becoming increasingly important in the global transition away from fossil fuels."#.to_string(),
                doc_id,
                chunk_info.clone(),
            );
            let metadata = create_metadata(
                "environment",
                "climate change",
                vec![
                    "global warming",
                    "greenhouse gases",
                    "environmental science",
                ],
                "high",
            );
            for (key, value) in metadata {
                node = node.with_metadata(key, value);
            }
            ScoredNode::new(node, 0.65)
        },
        {
            let mut node = Node::new(
                r#"Natural language processing (NLP) is a branch of artificial intelligence that focuses on the interaction between computers and human language. It involves developing algorithms and models that can understand, interpret, and generate human language in a valuable way. NLP applications include machine translation, sentiment analysis, chatbots, text summarization, and information extraction. Modern NLP relies heavily on deep learning techniques and large language models like GPT, BERT, and T5. These models have achieved remarkable performance on various language understanding and generation tasks."#.to_string(),
                doc_id,
                chunk_info.clone(),
            );
            let metadata = create_metadata(
                "technology",
                "natural language processing",
                vec!["NLP", "language models", "text analysis"],
                "high",
            );
            for (key, value) in metadata {
                node = node.with_metadata(key, value);
            }
            ScoredNode::new(node, 0.75)
        },
    ]
}

/// Demonstrate sentence embedding-based compression using our new components.
async fn demonstrate_sentence_embedding_compression(
    nodes: Vec<ScoredNode>,
    query: &str,
    target_ratio: f32,
) -> std::result::Result<Vec<ScoredNode>, ExampleError> {
    println!("🧠 Using Sentence Embedding Optimization (LlamaIndex-style)");

    // Create FastEmbed embedder for real embeddings

    let embedder = Arc::new(
        FastEmbedder::new()
            .await
            .map_err(|e| ExampleError::Other(format!("FastEmbed initialization failed: {}", e)))?,
    );

    println!("✅ FastEmbed embedder initialized: {}", embedder.name());

    let config = SentenceEmbeddingConfig {
        percentile_cutoff: Some(target_ratio),
        threshold_cutoff: Some(0.3),
        context_before: Some(1),
        context_after: Some(1),
        max_sentences_per_node: Some(20),
    };

    let optimizer = SentenceEmbeddingOptimizer::new(embedder.clone(), config);

    // Step 1: Apply similarity filtering first to remove low-quality nodes
    println!("🔍 Step 1: Applying similarity filtering...");
    let similarity_filter = SimilarityFilter::new(SimilarityFilterConfig {
        similarity_cutoff: 0.3,
        max_nodes: Some(20), // Keep top 20 nodes before compression
        use_query_embedding: true,
    });

    let original_count = nodes.len();
    let filtered_nodes = similarity_filter
        .postprocess(nodes, query)
        .await
        .map_err(|e| ExampleError::Other(format!("Similarity filtering failed: {}", e)))?;

    println!(
        "✅ Similarity filtering: {} → {} nodes",
        original_count,
        filtered_nodes.len()
    );

    // Step 2: Apply sentence embedding compression
    println!("🧠 Step 2: Applying sentence embedding compression...");
    let compressed_nodes = optimizer
        .compress(filtered_nodes, query)
        .await
        .map_err(|e| ExampleError::Other(format!("Compression failed: {}", e)))?;

    println!("✅ Combined postprocessing completed: Filtering + Compression");
    println!(
        "📊 Final result: {} compressed nodes",
        compressed_nodes.len()
    );
    Ok(compressed_nodes)
}

/// Demonstrate advanced postprocessor chaining (LlamaIndex-style)
async fn demonstrate_postprocessor_chain(
    nodes: Vec<ScoredNode>,
    query: &str,
    embedder: Arc<dyn Embedder>,
) -> std::result::Result<Vec<ScoredNode>, ExampleError> {
    println!("🔗 Advanced Postprocessor Chain (LlamaIndex-style)");
    println!("✨ Demonstrating sequential postprocessor application");

    let original_count = nodes.len();
    let mut current_nodes = nodes;

    // Step 1: Similarity filtering (remove low-relevance nodes)
    println!("\n🎯 Step 1: Similarity Filtering");
    let similarity_filter = SimilarityFilter::new(SimilarityFilterConfig {
        similarity_cutoff: 0.4,
        max_nodes: Some(15),
        use_query_embedding: true,
    });

    current_nodes = similarity_filter
        .postprocess(current_nodes, query)
        .await
        .map_err(|e| ExampleError::Other(format!("Similarity filtering failed: {}", e)))?;

    println!(
        "   ✅ Filtered: {} → {} nodes",
        original_count,
        current_nodes.len()
    );

    // Step 2: Sentence embedding optimization (compress content)
    println!("\n🧠 Step 2: Sentence Embedding Optimization");
    let config = SentenceEmbeddingConfig {
        percentile_cutoff: Some(0.6), // Keep top 60% of sentences
        threshold_cutoff: Some(0.3),
        context_before: Some(1),
        context_after: Some(1),
        max_sentences_per_node: Some(15),
    };

    let optimizer = SentenceEmbeddingOptimizer::new(embedder, config);
    current_nodes = optimizer
        .postprocess(current_nodes, query)
        .await
        .map_err(|e| ExampleError::Other(format!("Sentence optimization failed: {}", e)))?;

    println!("   ✅ Optimized: Content compressed while preserving relevance");

    // Step 3: Final quality check (could add more postprocessors here)
    println!("\n✨ Step 3: Final Quality Assessment");
    let final_count = current_nodes.len();
    let avg_score = if !current_nodes.is_empty() {
        current_nodes.iter().map(|n| n.score).sum::<f32>() / current_nodes.len() as f32
    } else {
        0.0
    };

    println!(
        "   📊 Final results: {} nodes with avg score: {:.3}",
        final_count, avg_score
    );
    println!(
        "   🎯 Total processing: {} → {} nodes ({:.1}% reduction)",
        original_count,
        final_count,
        (1.0 - final_count as f32 / original_count as f32) * 100.0
    );

    println!("\n✅ Postprocessor chain completed successfully!");
    println!("💡 This demonstrates LlamaIndex-style node postprocessing pipeline");

    Ok(current_nodes)
}

/// Demonstrate advanced postprocessor chain with all four components
async fn demonstrate_advanced_postprocessor_chain(
    nodes: Vec<ScoredNode>,
    query: &str,
    embedder: Arc<dyn Embedder>,
    use_keyword_filter: bool,
    use_metadata_filter: bool,
) -> std::result::Result<Vec<ScoredNode>, ExampleError> {
    println!("🚀 Advanced 4-Component Postprocessor Chain");
    println!("✨ Demonstrating comprehensive node processing pipeline");
    println!("🔧 Components: Keyword → Metadata → Similarity → Sentence Embedding");

    let original_count = nodes.len();
    let mut current_nodes = nodes;
    let mut step_counter = 1;

    // Step 1: Keyword filtering (if enabled)
    if use_keyword_filter {
        println!("\n🔍 Step {}: Keyword Filtering", step_counter);
        let keywords = query
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect::<Vec<_>>();

        let keyword_config = KeywordFilterConfig {
            required_keywords: keywords,
            exclude_keywords: vec!["spam".to_string(), "irrelevant".to_string()],
            case_sensitive: false,
            min_required_matches: 1,
        };

        let keyword_filter = KeywordFilter::new(keyword_config)
            .map_err(|e| ExampleError::Other(format!("KeywordFilter creation failed: {}", e)))?;

        let before_count = current_nodes.len();
        current_nodes = keyword_filter
            .postprocess(current_nodes, query)
            .await
            .map_err(|e| ExampleError::Other(format!("Keyword filtering failed: {}", e)))?;

        println!(
            "   ✅ Keyword filtered: {} → {} nodes",
            before_count,
            current_nodes.len()
        );
        step_counter += 1;
    }

    // Step 2: Metadata filtering (if enabled)
    if use_metadata_filter {
        println!("\n📋 Step {}: Metadata Filtering", step_counter);
        let mut required_metadata = std::collections::HashMap::new();
        required_metadata.insert("category".to_string(), "technology".to_string());

        let metadata_config = MetadataFilterConfig {
            required_metadata,
            excluded_metadata: std::collections::HashMap::new(),
            require_all: false,
        };

        let metadata_filter = MetadataFilter::new(metadata_config);

        let before_count = current_nodes.len();
        current_nodes = metadata_filter
            .postprocess(current_nodes, query)
            .await
            .map_err(|e| ExampleError::Other(format!("Metadata filtering failed: {}", e)))?;

        println!(
            "   ✅ Metadata filtered: {} → {} nodes",
            before_count,
            current_nodes.len()
        );
        step_counter += 1;
    }

    // Step 3: Similarity filtering (always applied)
    println!("\n🎯 Step {}: Similarity Filtering", step_counter);
    let similarity_filter = SimilarityFilter::new(SimilarityFilterConfig {
        similarity_cutoff: 0.3,
        max_nodes: Some(10),
        use_query_embedding: true,
    });

    let before_count = current_nodes.len();
    current_nodes = similarity_filter
        .postprocess(current_nodes, query)
        .await
        .map_err(|e| ExampleError::Other(format!("Similarity filtering failed: {}", e)))?;

    println!(
        "   ✅ Similarity filtered: {} → {} nodes",
        before_count,
        current_nodes.len()
    );
    step_counter += 1;

    // Step 4: Sentence embedding optimization (always applied)
    println!(
        "\n🧠 Step {}: Sentence Embedding Optimization",
        step_counter
    );
    let config = SentenceEmbeddingConfig {
        percentile_cutoff: Some(0.7), // Keep top 70% of sentences
        threshold_cutoff: Some(0.4),
        context_before: Some(1),
        context_after: Some(1),
        max_sentences_per_node: Some(12),
    };

    let optimizer = SentenceEmbeddingOptimizer::new(embedder, config);
    let before_count = current_nodes.len();
    current_nodes = optimizer
        .postprocess(current_nodes, query)
        .await
        .map_err(|e| ExampleError::Other(format!("Sentence optimization failed: {}", e)))?;

    println!("   ✅ Content optimized: {} nodes processed", before_count);

    // Final summary
    println!("\n🎯 Advanced Chain Summary");
    println!("═══════════════════════════");
    let final_count = current_nodes.len();
    let avg_score = if !current_nodes.is_empty() {
        current_nodes.iter().map(|n| n.score).sum::<f32>() / current_nodes.len() as f32
    } else {
        0.0
    };

    println!("📊 Processing Results:");
    println!("   • Original nodes: {}", original_count);
    println!("   • Final nodes: {}", final_count);
    println!(
        "   • Reduction: {:.1}%",
        (1.0 - final_count as f32 / original_count as f32) * 100.0
    );
    println!("   • Average score: {:.3}", avg_score);

    let components_used = vec![
        if use_keyword_filter {
            "KeywordFilter"
        } else {
            ""
        },
        if use_metadata_filter {
            "MetadataFilter"
        } else {
            ""
        },
        "SimilarityFilter",
        "SentenceEmbeddingOptimizer",
    ]
    .into_iter()
    .filter(|s| !s.is_empty())
    .collect::<Vec<_>>();

    println!("   • Components used: {}", components_used.join(" → "));

    println!("\n✅ Advanced postprocessor chain completed successfully!");
    println!("💡 This demonstrates the power of combining multiple postprocessors");

    Ok(current_nodes)
}

/// Compare different postprocessor strategies
async fn compare_postprocessor_strategies(
    nodes: Vec<ScoredNode>,
    query: &str,
    embedder: Arc<dyn Embedder>,
) -> std::result::Result<(), ExampleError> {
    println!("📊 Postprocessor Strategy Comparison");
    println!("═══════════════════════════════════");

    let strategies = vec![
        ("Basic Chain", false, false),
        ("With Keyword Filter", true, false),
        ("With Metadata Filter", false, true),
        ("Full Advanced Chain", true, true),
    ];

    for (strategy_name, use_keyword, use_metadata) in strategies {
        println!("\n🔬 Testing Strategy: {}", strategy_name);
        println!("{}", "─".repeat(50));

        let start_time = std::time::Instant::now();
        let result = demonstrate_advanced_postprocessor_chain(
            nodes.clone(),
            query,
            embedder.clone(),
            use_keyword,
            use_metadata,
        )
        .await?;
        let duration = start_time.elapsed();

        let original_chars: usize = nodes.iter().map(|n| n.node.content.len()).sum();
        let final_chars: usize = result.iter().map(|n| n.node.content.len()).sum();
        let compression_ratio = final_chars as f32 / original_chars as f32;

        println!("⏱️  Processing time: {:.2}s", duration.as_secs_f64());
        println!(
            "📝 Content compression: {:.1}%",
            (1.0 - compression_ratio) * 100.0
        );
        println!("🎯 Final node count: {}", result.len());

        if !result.is_empty() {
            let avg_score = result.iter().map(|n| n.score).sum::<f32>() / result.len() as f32;
            println!("⭐ Average relevance score: {:.3}", avg_score);
        }

        println!();
    }

    println!("✅ Strategy comparison completed!");
    println!(
        "💡 Choose the strategy that best balances performance and quality for your use case."
    );

    Ok(())
}

#[tokio::main]
async fn main() -> std::result::Result<(), ExampleError> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();

    println!("🗜️  Contextual Compression Demo");
    println!("===============================");
    println!("Query: {}", args.query);
    println!("Compression Method: {}", args.compression_method);
    println!(
        "Target Compression Ratio: {:.1}%",
        args.compression_ratio * 100.0
    );
    println!();

    // Create sample documents
    let original_nodes = create_sample_long_documents();

    println!("📚 Original Documents:");
    for (i, node) in original_nodes.iter().enumerate() {
        println!(
            "Document {}: {} chars (Score: {:.2})",
            i + 1,
            node.node.content.len(),
            node.score
        );
        if args.verbose {
            println!(
                "Content preview: {}...",
                node.node.content.chars().take(100).collect::<String>()
            );
        }
    }
    println!();

    // Create compression strategy
    let strategy = match args.compression_method.as_str() {
        "llm_based" => {
            let client = Siumai::builder().openai().build().await?;
            CompressionStrategy::LlmBased {
                client,
                target_ratio: args.compression_ratio,
            }
        }
        "keyword_based" => {
            let keywords = args
                .query
                .split_whitespace()
                .map(|s| s.to_lowercase())
                .collect();
            CompressionStrategy::KeywordBased {
                keywords,
                min_matches: 1,
            }
        }
        "similarity_based" => CompressionStrategy::SimilarityBased {
            threshold: 0.5,
            max_chunks: (original_nodes.len() as f32 * args.compression_ratio) as usize,
        },
        "sentence_embedding" => {
            // This will be handled separately using our new components
            CompressionStrategy::SimilarityBased {
                threshold: 0.5,
                max_chunks: original_nodes.len(),
            }
        }
        "hybrid" => {
            let client = Siumai::builder().openai().build().await?;
            let primary = Box::new(CompressionStrategy::LlmBased {
                client,
                target_ratio: args.compression_ratio,
            });
            let keywords = args
                .query
                .split_whitespace()
                .map(|s| s.to_lowercase())
                .collect();
            let secondary = Box::new(CompressionStrategy::KeywordBased {
                keywords,
                min_matches: 1,
            });
            CompressionStrategy::Hybrid {
                primary,
                secondary,
                primary_weight: 0.7,
            }
        }
        _ => {
            return Err(ExampleError::Other(format!(
                "Unknown compression method: {}",
                args.compression_method
            )));
        }
    };

    // Apply compression
    let compression_timer = Instant::now();
    let compressed_nodes = match args.compression_method.as_str() {
        "sentence_embedding" => {
            // Use our sentence embedding optimizer with filtering
            demonstrate_sentence_embedding_compression(
                original_nodes.clone(),
                &args.query,
                args.compression_ratio,
            )
            .await?
        }
        "postprocessor_chain" => {
            // Use our advanced postprocessor chain
            let embedder = Arc::new(FastEmbedder::new().await.map_err(|e| {
                ExampleError::Other(format!("FastEmbed initialization failed: {}", e))
            })?);
            demonstrate_postprocessor_chain(original_nodes.clone(), &args.query, embedder).await?
        }
        "advanced_chain" => {
            // Use our new 4-component advanced chain
            let embedder = Arc::new(FastEmbedder::new().await.map_err(|e| {
                ExampleError::Other(format!("FastEmbed initialization failed: {}", e))
            })?);
            demonstrate_advanced_postprocessor_chain(
                original_nodes.clone(),
                &args.query,
                embedder,
                args.use_keyword_filter,
                args.use_metadata_filter,
            )
            .await?
        }
        _ => {
            // Use the existing compressor
            let compressor = ContextualCompressor::new(strategy);
            compressor
                .compress(original_nodes.clone(), &args.query)
                .await?
        }
    };
    let compression_time = compression_timer.elapsed();

    // Handle strategy comparison
    if args.compare_strategies {
        let embedder =
            Arc::new(FastEmbedder::new().await.map_err(|e| {
                ExampleError::Other(format!("FastEmbed initialization failed: {}", e))
            })?);
        compare_postprocessor_strategies(original_nodes, &args.query, embedder).await?;
        return Ok(());
    }

    println!("🗜️  Compression Results:");
    println!("========================");
    println!("Compression time: {:.2}s", compression_time.as_secs_f64());
    println!();

    // Calculate compression statistics
    let original_total_chars: usize = original_nodes.iter().map(|n| n.node.content.len()).sum();
    let compressed_total_chars: usize = compressed_nodes.iter().map(|n| n.node.content.len()).sum();
    let actual_compression_ratio = compressed_total_chars as f32 / original_total_chars as f32;

    println!("📊 Compression Statistics:");
    println!("Original total chars: {}", original_total_chars);
    println!("Compressed total chars: {}", compressed_total_chars);
    println!(
        "Actual compression ratio: {:.1}%",
        actual_compression_ratio * 100.0
    );
    println!(
        "Space saved: {:.1}%",
        (1.0 - actual_compression_ratio) * 100.0
    );
    println!();

    // Display compressed results
    println!("📝 Compressed Documents:");
    for (i, node) in compressed_nodes.iter().enumerate() {
        println!(
            "Document {}: {} chars (Score: {:.2})",
            i + 1,
            node.node.content.len(),
            node.score
        );
        if args.verbose {
            println!("Compressed content: {}", node.node.content);
        } else {
            println!(
                "Content preview: {}...",
                node.node.content.chars().take(150).collect::<String>()
            );
        }
        println!("{}", "-".repeat(50));
    }

    println!();
    println!("✅ Contextual compression demonstration completed!");
    println!("🎯 This technique helps optimize context window usage while preserving relevance.");
    println!("📈 Compression can significantly reduce token costs for LLM generation.");

    Ok(())
}
