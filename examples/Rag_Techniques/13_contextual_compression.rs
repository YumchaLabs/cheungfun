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
//! cargo run --bin contextual_compression --features fastembed -- \
//!     --query "machine learning algorithms" \
//!     --compression-method "llm_based" \
//!     --compression-ratio 0.3 \
//!     --top-k 5
//! ```

use cheungfun::prelude::*;
use cheungfun_integrations::FastEmbedder;
use cheungfun_query::{
    advanced::fusion::DistributionBasedFusion,
    postprocessor::{DocumentCompressor, SentenceEmbeddingConfig, SentenceEmbeddingOptimizer},
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

    /// Compression method: llm_based, keyword_based, similarity_based, hybrid, sentence_embedding
    #[arg(long, default_value = "sentence_embedding")]
    compression_method: String,

    /// Target compression ratio (0.0-1.0, lower = more compression)
    #[arg(long, default_value = "0.3")]
    compression_ratio: f32,

    /// Number of results to retrieve before compression
    #[arg(long, default_value = "10")]
    top_k: usize,

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

/// Create sample long documents for compression testing.
fn create_sample_long_documents() -> Vec<ScoredNode> {
    use cheungfun_core::ChunkInfo;
    use uuid::Uuid;

    let doc_id = Uuid::new_v4();
    let chunk_info = ChunkInfo::new(Some(0), Some(1000), 0);

    vec![
        ScoredNode::new(
            Node::new(
                r#"Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience. The field encompasses various approaches including supervised learning, unsupervised learning, and reinforcement learning. Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data. Common supervised learning algorithms include linear regression, decision trees, random forests, support vector machines, and neural networks. These algorithms are widely used in applications such as image recognition, natural language processing, and predictive analytics. The training process involves feeding the algorithm examples of input-output pairs so it can learn the underlying patterns and relationships."#.to_string(),
                doc_id,
                chunk_info.clone(),
            ),
            0.9,
        ),
        ScoredNode::new(
            Node::new(
                r#"Deep learning represents a revolutionary approach within machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. The architecture of deep neural networks allows them to automatically learn hierarchical representations of data, making them particularly effective for tasks involving high-dimensional data such as images, text, and audio. Convolutional Neural Networks (CNNs) have transformed computer vision by enabling automatic feature extraction from images. Recurrent Neural Networks (RNNs) and their variants like LSTM and GRU have been instrumental in natural language processing and sequence modeling. The recent development of Transformer architectures has led to breakthrough models like BERT, GPT, and T5 that have achieved state-of-the-art performance across numerous NLP tasks."#.to_string(),
                doc_id,
                chunk_info.clone(),
            ),
            0.8,
        ),
        ScoredNode::new(
            Node::new(
                r#"The history of artificial intelligence dates back to ancient times with myths and stories of artificial beings endowed with intelligence or consciousness. However, the modern field of AI began in the 1950s with the work of pioneers like Alan Turing, John McCarthy, and Marvin Minsky. The term 'artificial intelligence' was coined by John McCarthy in 1956 at the Dartmouth Conference. Early AI research focused on symbolic reasoning and expert systems. The field experienced several 'AI winters' - periods of reduced funding and interest - but has seen remarkable resurgence in recent decades due to advances in computing power, data availability, and algorithmic improvements. Today, AI applications are ubiquitous, from search engines and recommendation systems to autonomous vehicles and medical diagnosis tools."#.to_string(),
                doc_id,
                chunk_info.clone(),
            ),
            0.6,
        ),
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

    let optimizer = SentenceEmbeddingOptimizer::new(embedder, config);

    // Apply compression
    let compressed_nodes = optimizer
        .compress(nodes, query)
        .await
        .map_err(|e| ExampleError::Other(format!("Compression failed: {}", e)))?;

    println!("✅ Sentence embedding compression completed");
    Ok(compressed_nodes)
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
    let compressed_nodes = if args.compression_method == "sentence_embedding" {
        // Use our new sentence embedding optimizer
        demonstrate_sentence_embedding_compression(
            original_nodes.clone(),
            &args.query,
            args.compression_ratio,
        )
        .await?
    } else {
        // Use the existing compressor
        let compressor = ContextualCompressor::new(strategy);
        compressor
            .compress(original_nodes.clone(), &args.query)
            .await?
    };
    let compression_time = compression_timer.elapsed();

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
