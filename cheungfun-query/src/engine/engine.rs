//! Query engine implementations for high-level query processing.
//!
//! This module provides the main query engine that combines retrievers
//! and generators to provide a unified interface for RAG operations.

use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

use cheungfun_core::{
    traits::{BaseMemory, ResponseGenerator, Retriever},
    types::{GenerationOptions, Query, QueryResponse, ScoredNode},
    ChatMessage, MessageRole, Result,
};

/// Strategies for query rewriting to improve retrieval effectiveness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryRewriteStrategy {
    /// Make the query more specific and clear.
    Clarification,
    /// Expand the query with related terms and synonyms.
    Expansion,
    /// Break down complex queries into simpler sub-questions.
    Decomposition,
    /// Generate a hypothetical document that would answer the query (HyDE).
    HyDE,
}

/// A high-level query engine that combines retrieval and generation.
///
/// The query engine orchestrates the complete RAG process:
/// 1. Takes a user query
/// 2. Uses a retriever to find relevant context
/// 3. Uses a generator to create a response based on the context
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_query::engine::QueryEngine;
/// use cheungfun_core::prelude::*;
///
/// # async fn example() -> Result<()> {
/// let engine = QueryEngine::builder()
///     .retriever(retriever)
///     .generator(generator)
///     .build()?;
///
/// let response = engine.query("What is machine learning?").await?;
/// println!("Answer: {}", response.content);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct QueryEngine {
    /// Retriever for finding relevant context.
    retriever: Arc<dyn Retriever>,

    /// Generator for creating responses.
    generator: Arc<dyn ResponseGenerator>,

    /// Optional memory for conversation history.
    memory: Option<Arc<tokio::sync::Mutex<dyn BaseMemory>>>,

    /// Configuration for query processing.
    config: QueryEngineConfig,
}

/// Configuration for query engine operations.
#[derive(Debug, Clone)]
pub struct QueryEngineConfig {
    /// Default number of context nodes to retrieve.
    pub default_top_k: usize,

    /// Default generation options.
    pub default_generation_options: GenerationOptions,

    /// Whether to validate retrieved context before generation.
    pub validate_context: bool,

    /// Minimum number of context nodes required for generation.
    pub min_context_nodes: usize,

    /// Maximum number of context nodes to use for generation.
    pub max_context_nodes: usize,

    /// Whether to enable query preprocessing.
    pub enable_query_preprocessing: bool,

    /// Whether to enable response postprocessing.
    pub enable_response_postprocessing: bool,
}

impl Default for QueryEngineConfig {
    fn default() -> Self {
        Self {
            default_top_k: 5,
            default_generation_options: GenerationOptions::default(),
            validate_context: true,
            min_context_nodes: 1,
            max_context_nodes: 10,
            enable_query_preprocessing: true,
            enable_response_postprocessing: true,
        }
    }
}

impl QueryEngine {
    /// Create a new query engine.
    pub fn new(retriever: Arc<dyn Retriever>, generator: Arc<dyn ResponseGenerator>) -> Self {
        Self {
            retriever,
            generator,
            memory: None,
            config: QueryEngineConfig::default(),
        }
    }

    /// Create a new query engine with custom configuration.
    pub fn with_config(
        retriever: Arc<dyn Retriever>,
        generator: Arc<dyn ResponseGenerator>,
        config: QueryEngineConfig,
    ) -> Self {
        Self {
            retriever,
            generator,
            memory: None,
            config,
        }
    }

    /// Create a new query engine with memory support.
    pub fn with_memory(
        retriever: Arc<dyn Retriever>,
        generator: Arc<dyn ResponseGenerator>,
        memory: Arc<tokio::sync::Mutex<dyn BaseMemory>>,
    ) -> Self {
        Self {
            retriever,
            generator,
            memory: Some(memory),
            config: QueryEngineConfig::default(),
        }
    }

    /// Create a new query engine with memory and custom configuration.
    pub fn with_memory_and_config(
        retriever: Arc<dyn Retriever>,
        generator: Arc<dyn ResponseGenerator>,
        memory: Arc<tokio::sync::Mutex<dyn BaseMemory>>,
        config: QueryEngineConfig,
    ) -> Self {
        Self {
            retriever,
            generator,
            memory: Some(memory),
            config,
        }
    }

    /// Create a builder for constructing query engines.
    #[must_use]
    pub fn builder() -> QueryEngineBuilder {
        QueryEngineBuilder::new()
    }

    /// Execute a query and return a response.
    ///
    /// This is the main method that orchestrates the complete RAG process.
    #[instrument(skip(self), fields(engine = "QueryEngine"))]
    pub async fn query(&self, query_text: &str) -> Result<QueryResponse> {
        self.query_with_options(query_text, &QueryEngineOptions::default())
            .await
    }

    /// Execute a query with custom options.
    #[instrument(skip(self), fields(engine = "QueryEngine"))]
    pub async fn query_with_options(
        &self,
        query_text: &str,
        options: &QueryEngineOptions,
    ) -> Result<QueryResponse> {
        info!("Processing query: {}", query_text);

        // Build query object
        let mut query = Query::new(query_text);
        query.top_k = options.top_k.unwrap_or(self.config.default_top_k);

        // Apply any additional query options
        if let Some(search_mode) = &options.search_mode {
            query.search_mode = search_mode.clone();
        }

        for (key, value) in &options.filters {
            query.filters.insert(key.clone(), value.clone());
        }

        // Preprocess query if enabled
        if self.config.enable_query_preprocessing {
            // TODO: Implement query preprocessing (query expansion, spell correction, etc.)
            debug!("Query preprocessing enabled but not yet implemented");
        }

        // Retrieve relevant context
        debug!("Retrieving context for query");
        let mut retrieved_nodes = self.retriever.retrieve(&query).await?;
        info!("Retrieved {} context nodes", retrieved_nodes.len());

        // Validate context if enabled
        if self.config.validate_context {
            self.validate_retrieved_context(&retrieved_nodes)?;
        }

        // Limit context nodes
        if retrieved_nodes.len() > self.config.max_context_nodes {
            retrieved_nodes.truncate(self.config.max_context_nodes);
            debug!(
                "Truncated context to {} nodes",
                self.config.max_context_nodes
            );
        }

        // Prepare generation options
        let generation_options = options
            .generation_options
            .as_ref()
            .unwrap_or(&self.config.default_generation_options);

        // Generate response
        debug!("Generating response");
        let generated_response = self
            .generator
            .generate_response(query_text, retrieved_nodes.clone(), generation_options)
            .await?;

        // Postprocess response if enabled
        let final_response = if self.config.enable_response_postprocessing {
            // TODO: Implement response postprocessing (fact checking, formatting, etc.)
            debug!("Response postprocessing enabled but not yet implemented");
            generated_response
        } else {
            generated_response
        };

        // Build query metadata
        let mut query_metadata = HashMap::new();
        query_metadata.insert(
            "retriever".to_string(),
            serde_json::Value::String(self.retriever.name().to_string()),
        );
        query_metadata.insert(
            "generator".to_string(),
            serde_json::Value::String(self.generator.name().to_string()),
        );
        query_metadata.insert(
            "context_nodes_used".to_string(),
            serde_json::Value::Number(retrieved_nodes.len().into()),
        );

        info!("Query processing completed successfully");

        Ok(QueryResponse {
            response: final_response,
            retrieved_nodes,
            query_metadata,
        })
    }

    /// Process a chat query with conversation memory.
    ///
    /// This method integrates conversation history from memory into the query
    /// processing pipeline, enabling multi-turn conversations with context awareness.
    ///
    /// # Arguments
    ///
    /// * `user_message` - The user's message content
    /// * `options` - Optional query processing options
    ///
    /// # Returns
    ///
    /// A query response that includes the assistant's reply and retrieved context.
    ///
    /// # Errors
    ///
    /// Returns an error if memory is not configured, retrieval fails, or generation fails.
    #[instrument(skip(self, options))]
    pub async fn chat(
        &self,
        user_message: &str,
        options: Option<&QueryEngineOptions>,
    ) -> Result<QueryResponse> {
        let memory =
            self.memory
                .as_ref()
                .ok_or_else(|| cheungfun_core::CheungfunError::Configuration {
                    message: "Memory not configured for chat functionality".to_string(),
                })?;

        info!("Processing chat query: {}", user_message);

        // Add user message to memory
        {
            let mut memory_guard = memory.lock().await;
            let user_chat_message = ChatMessage {
                role: MessageRole::User,
                content: user_message.to_string(),
                timestamp: chrono::Utc::now(),
                metadata: None,
            };
            memory_guard.add_message(user_chat_message).await?;
        }

        // Get conversation history for context
        let conversation_history = {
            let memory_guard = memory.lock().await;
            memory_guard.get_messages().await?
        };

        // Build enhanced query with conversation context
        let enhanced_query = self.build_contextual_query(user_message, &conversation_history)?;

        // Process the enhanced query
        let mut response = if let Some(opts) = options {
            self.query_with_options(&enhanced_query, opts).await?
        } else {
            self.query(&enhanced_query).await?
        };

        // Add assistant response to memory
        {
            let mut memory_guard = memory.lock().await;
            let assistant_chat_message = ChatMessage {
                role: MessageRole::Assistant,
                content: response.response.content.clone(),
                timestamp: chrono::Utc::now(),
                metadata: None,
            };
            memory_guard.add_message(assistant_chat_message).await?;
        }

        // Add conversation metadata
        response.query_metadata.insert(
            "conversation_turns".to_string(),
            serde_json::Value::Number((conversation_history.len() / 2).into()),
        );
        response.query_metadata.insert(
            "has_conversation_context".to_string(),
            serde_json::Value::Bool(conversation_history.len() > 1),
        );

        info!("Chat query processing completed successfully");
        Ok(response)
    }

    /// Build a contextual query that includes conversation history.
    fn build_contextual_query(
        &self,
        current_query: &str,
        conversation_history: &[ChatMessage],
    ) -> Result<String> {
        if conversation_history.len() <= 1 {
            // No previous context, use query as-is
            return Ok(current_query.to_string());
        }

        // Build context from recent conversation
        let mut context_parts = Vec::new();

        // Add recent conversation turns (last 3 pairs to avoid token overflow)
        let recent_messages: Vec<_> = conversation_history
            .iter()
            .rev()
            .take(6) // Last 3 user-assistant pairs
            .collect();

        for message in recent_messages.iter().rev() {
            match message.role {
                MessageRole::User => {
                    context_parts.push(format!("User: {}", message.content));
                }
                MessageRole::Assistant => {
                    context_parts.push(format!("Assistant: {}", message.content));
                }
                MessageRole::System => {
                    context_parts.push(format!("System: {}", message.content));
                }
                MessageRole::Tool => {
                    // Skip tool messages in context for now
                    continue;
                }
            }
        }

        // Combine context with current query
        if context_parts.is_empty() {
            Ok(current_query.to_string())
        } else {
            let context = context_parts.join("\n");
            Ok(format!(
                "Previous conversation:\n{}\n\nCurrent question: {}",
                context, current_query
            ))
        }
    }

    /// Get memory statistics if memory is configured.
    pub async fn get_memory_stats(&self) -> Result<Option<crate::memory::MemoryStats>> {
        if let Some(memory) = &self.memory {
            let memory_guard = memory.lock().await;
            let vars = memory_guard.get_memory_variables().await?;

            // Convert memory variables to stats (simplified)
            let stats = crate::memory::MemoryStats {
                message_count: vars
                    .get("message_count")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
                estimated_tokens: vars
                    .get("estimated_tokens")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
                system_message_count: vars
                    .get("system_messages")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
                user_message_count: vars
                    .get("user_messages")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
                assistant_message_count: vars
                    .get("assistant_messages")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
                tool_message_count: vars
                    .get("tool_messages")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
                last_truncated_count: vars
                    .get("last_truncated")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
                at_capacity: vars
                    .get("at_capacity")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(false),
            };

            Ok(Some(stats))
        } else {
            Ok(None)
        }
    }

    /// Clear conversation memory if configured.
    pub async fn clear_memory(&self) -> Result<()> {
        if let Some(memory) = &self.memory {
            let mut memory_guard = memory.lock().await;
            memory_guard.clear().await?;
            info!("Cleared conversation memory");
        }
        Ok(())
    }

    /// Perform deep research with multiple rounds of query refinement.
    ///
    /// This method implements a multi-stage research process similar to DeepWiki's
    /// deep research functionality. It iteratively refines queries based on previous
    /// results to provide comprehensive answers to complex questions.
    ///
    /// # Arguments
    ///
    /// * `initial_query` - The initial research question
    /// * `depth` - Number of research iterations (default: 3, max: 5)
    /// * `options` - Optional query processing options
    ///
    /// # Returns
    ///
    /// A comprehensive query response that synthesizes information from multiple rounds.
    ///
    /// # Errors
    ///
    /// Returns an error if any stage of the research process fails.
    #[instrument(skip(self, options))]
    pub async fn deep_research(
        &self,
        initial_query: &str,
        depth: Option<usize>,
        options: Option<&QueryEngineOptions>,
    ) -> Result<QueryResponse> {
        let research_depth = depth.unwrap_or(3).min(5); // Cap at 5 iterations
        info!(
            "Starting deep research with {} iterations: {}",
            research_depth, initial_query
        );

        let mut research_context = Vec::new();
        let mut current_query = initial_query.to_string();
        let mut all_retrieved_nodes = Vec::new();
        let mut research_metadata = HashMap::new();

        for iteration in 0..research_depth {
            info!(
                "Deep research iteration {}/{}: {}",
                iteration + 1,
                research_depth,
                current_query
            );

            // Execute current query
            let response = if let Some(opts) = options {
                self.query_with_options(&current_query, opts).await?
            } else {
                self.query(&current_query).await?
            };

            // Collect retrieved nodes
            all_retrieved_nodes.extend(response.retrieved_nodes.clone());

            // Add response to research context
            research_context.push(format!(
                "Research Question {}: {}\nAnswer: {}",
                iteration + 1,
                current_query,
                response.response.content
            ));

            // Generate follow-up query for next iteration (except for last iteration)
            if iteration < research_depth - 1 {
                current_query = self
                    .generate_follow_up_query(
                        initial_query,
                        &research_context,
                        &response.response.content,
                    )
                    .await?;
            }
        }

        // Synthesize final comprehensive answer
        let final_answer = self
            .synthesize_research_findings(initial_query, &research_context)
            .await?;

        // Build comprehensive metadata
        research_metadata.insert(
            "research_depth".to_string(),
            serde_json::Value::Number(research_depth.into()),
        );
        research_metadata.insert(
            "total_nodes_retrieved".to_string(),
            serde_json::Value::Number(all_retrieved_nodes.len().into()),
        );
        research_metadata.insert(
            "research_iterations".to_string(),
            serde_json::Value::Array(
                research_context
                    .iter()
                    .enumerate()
                    .map(|(i, context)| {
                        serde_json::Value::String(format!(
                            "Iteration {}: {}",
                            i + 1,
                            context.lines().next().unwrap_or("")
                        ))
                    })
                    .collect(),
            ),
        );

        // Deduplicate retrieved nodes by ID
        let mut seen_ids = std::collections::HashSet::new();
        let unique_nodes: Vec<ScoredNode> = all_retrieved_nodes
            .into_iter()
            .filter(|node| seen_ids.insert(node.node.id))
            .collect();

        info!(
            "Deep research completed with {} unique nodes",
            unique_nodes.len()
        );

        Ok(QueryResponse {
            response: final_answer,
            retrieved_nodes: unique_nodes,
            query_metadata: research_metadata,
        })
    }

    /// Generate a follow-up query based on previous research.
    async fn generate_follow_up_query(
        &self,
        original_query: &str,
        research_context: &[String],
        last_answer: &str,
    ) -> Result<String> {
        let context_summary = research_context.join("\n\n");

        let follow_up_prompt = format!(
            r#"Based on the original question and previous research, generate a specific follow-up question that would help provide a more comprehensive answer.

Original Question: {}

Previous Research:
{}

Latest Answer: {}

Generate a specific, focused follow-up question that explores a different aspect or goes deeper into the topic. The question should be directly related but not repetitive.

Follow-up Question:"#,
            original_query, context_summary, last_answer
        );

        // Use the generator to create follow-up query
        let generation_options = &self.config.default_generation_options;
        let follow_up_response = self
            .generator
            .generate_response(&follow_up_prompt, vec![], generation_options)
            .await?;

        // Extract the follow-up question (take first line, clean up)
        let follow_up_query = follow_up_response
            .content
            .lines()
            .next()
            .unwrap_or(original_query)
            .trim()
            .trim_end_matches('?')
            .to_string()
            + "?";

        debug!("Generated follow-up query: {}", follow_up_query);
        Ok(follow_up_query)
    }

    /// Synthesize research findings into a comprehensive answer.
    async fn synthesize_research_findings(
        &self,
        original_query: &str,
        research_context: &[String],
    ) -> Result<cheungfun_core::types::GeneratedResponse> {
        let context_summary = research_context.join("\n\n");

        let synthesis_prompt = format!(
            r#"Based on the comprehensive research conducted, provide a detailed and well-structured answer to the original question.

Original Question: {}

Research Findings:
{}

Please synthesize this information into a comprehensive, well-organized answer that:
1. Directly addresses the original question
2. Incorporates insights from all research iterations
3. Provides specific details and examples where available
4. Maintains a logical flow and structure
5. Acknowledges any limitations or areas where more research might be needed

Comprehensive Answer:"#,
            original_query, context_summary
        );

        // Use the generator to synthesize the final answer
        let generation_options = &self.config.default_generation_options;
        let synthesis_response = self
            .generator
            .generate_response(&synthesis_prompt, vec![], generation_options)
            .await?;

        info!("Synthesized comprehensive research answer");
        Ok(synthesis_response)
    }

    /// Rewrite a query to improve retrieval effectiveness.
    ///
    /// This method uses LLM-based query rewriting to transform user queries
    /// into more effective search queries that are likely to retrieve better results.
    ///
    /// # Arguments
    ///
    /// * `original_query` - The original user query
    /// * `rewrite_strategy` - The strategy to use for rewriting
    ///
    /// # Returns
    ///
    /// A rewritten query that should be more effective for retrieval.
    pub async fn rewrite_query(
        &self,
        original_query: &str,
        rewrite_strategy: QueryRewriteStrategy,
    ) -> Result<String> {
        let rewrite_prompt = match rewrite_strategy {
            QueryRewriteStrategy::Clarification => {
                format!(
                    r#"Rewrite the following query to be more specific and clear for information retrieval:

Original Query: {}

Rewritten Query (be more specific and use better keywords):"#,
                    original_query
                )
            }
            QueryRewriteStrategy::Expansion => {
                format!(
                    r#"Expand the following query with related terms and synonyms to improve search coverage:

Original Query: {}

Expanded Query (include related terms and synonyms):"#,
                    original_query
                )
            }
            QueryRewriteStrategy::Decomposition => {
                format!(
                    r#"Break down the following complex query into simpler, more focused sub-questions:

Original Query: {}

Decomposed Queries (list 2-3 focused sub-questions):"#,
                    original_query
                )
            }
            QueryRewriteStrategy::HyDE => {
                format!(
                    r#"Generate a hypothetical document passage that would perfectly answer this query:

Query: {}

Hypothetical Answer Passage:"#,
                    original_query
                )
            }
        };

        let generation_options = &self.config.default_generation_options;
        let rewrite_response = self
            .generator
            .generate_response(&rewrite_prompt, vec![], generation_options)
            .await?;

        let rewritten_query = rewrite_response.content.trim().to_string();
        debug!(
            "Rewrote query using {:?}: {} -> {}",
            rewrite_strategy, original_query, rewritten_query
        );

        Ok(rewritten_query)
    }

    /// Perform query with automatic rewriting for better results.
    ///
    /// This method automatically rewrites the query using the specified strategy
    /// and then performs retrieval and generation.
    ///
    /// # Arguments
    ///
    /// * `query_text` - The original query text
    /// * `rewrite_strategy` - The strategy to use for rewriting
    /// * `options` - Optional query processing options
    ///
    /// # Returns
    ///
    /// A query response based on the rewritten query.
    pub async fn query_with_rewrite(
        &self,
        query_text: &str,
        rewrite_strategy: QueryRewriteStrategy,
        options: Option<&QueryEngineOptions>,
    ) -> Result<QueryResponse> {
        info!(
            "Processing query with rewrite strategy: {:?}",
            rewrite_strategy
        );

        let rewritten_query = self.rewrite_query(query_text, rewrite_strategy).await?;

        let mut response = if let Some(opts) = options {
            self.query_with_options(&rewritten_query, opts).await?
        } else {
            self.query(&rewritten_query).await?
        };

        // Add rewrite metadata
        response.query_metadata.insert(
            "original_query".to_string(),
            serde_json::Value::String(query_text.to_string()),
        );
        response.query_metadata.insert(
            "rewritten_query".to_string(),
            serde_json::Value::String(rewritten_query),
        );
        response.query_metadata.insert(
            "rewrite_strategy".to_string(),
            serde_json::Value::String(format!("{:?}", rewrite_strategy)),
        );

        Ok(response)
    }

    /// Rerank retrieved nodes using LLM-based reranking.
    ///
    /// This method uses the LLM to rerank retrieved nodes based on their
    /// relevance to the original query, potentially improving result quality.
    ///
    /// # Arguments
    ///
    /// * `query_text` - The original query text
    /// * `nodes` - The nodes to rerank
    /// * `top_k` - Number of top nodes to return after reranking
    ///
    /// # Returns
    ///
    /// Reranked nodes with updated scores.
    pub async fn rerank_nodes(
        &self,
        query_text: &str,
        nodes: Vec<ScoredNode>,
        top_k: Option<usize>,
    ) -> Result<Vec<ScoredNode>> {
        if nodes.is_empty() {
            return Ok(nodes);
        }

        let rerank_limit = top_k.unwrap_or(nodes.len()).min(nodes.len());

        // Prepare reranking prompt
        let node_texts: Vec<String> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| format!("Document {}: {}", i + 1, node.node.content))
            .collect();

        let rerank_prompt = format!(
            r#"Given the query and the following documents, rank them by relevance to the query.
Return only the document numbers in order of relevance (most relevant first).

Query: {}

Documents:
{}

Ranking (comma-separated document numbers, most relevant first):"#,
            query_text,
            node_texts.join("\n\n")
        );

        let generation_options = &self.config.default_generation_options;
        let rerank_response = self
            .generator
            .generate_response(&rerank_prompt, vec![], generation_options)
            .await?;

        // Parse the ranking response
        let ranking_text = rerank_response.content.trim();
        let mut reranked_nodes = Vec::new();

        // Try to parse the ranking
        for (new_rank, doc_num_str) in ranking_text.split(',').enumerate() {
            if let Ok(doc_num) = doc_num_str.trim().parse::<usize>() {
                if doc_num > 0 && doc_num <= nodes.len() {
                    let mut node = nodes[doc_num - 1].clone();
                    // Update score based on new ranking (higher rank = higher score)
                    node.score = 1.0 - (new_rank as f32 / nodes.len() as f32);
                    reranked_nodes.push(node);

                    if reranked_nodes.len() >= rerank_limit {
                        break;
                    }
                }
            }
        }

        // If parsing failed, return original nodes
        if reranked_nodes.is_empty() {
            warn!("Failed to parse reranking response, returning original nodes");
            return Ok(nodes.into_iter().take(rerank_limit).collect());
        }

        info!("Reranked {} nodes using LLM", reranked_nodes.len());
        Ok(reranked_nodes)
    }

    /// Validate that retrieved context meets minimum requirements.
    fn validate_retrieved_context(&self, nodes: &[ScoredNode]) -> Result<()> {
        if nodes.len() < self.config.min_context_nodes {
            return Err(cheungfun_core::CheungfunError::Validation {
                message: format!(
                    "Insufficient context: got {} nodes, minimum required: {}",
                    nodes.len(),
                    self.config.min_context_nodes
                ),
            });
        }
        Ok(())
    }

    /// Get the retriever used by this engine.
    #[must_use]
    pub fn retriever(&self) -> &Arc<dyn Retriever> {
        &self.retriever
    }

    /// Get the generator used by this engine.
    #[must_use]
    pub fn generator(&self) -> &Arc<dyn ResponseGenerator> {
        &self.generator
    }

    /// Get the configuration of this engine.
    #[must_use]
    pub fn config(&self) -> &QueryEngineConfig {
        &self.config
    }

    /// Perform a health check on all components.
    pub async fn health_check(&self) -> Result<()> {
        self.retriever.health_check().await?;
        self.generator.health_check().await?;
        Ok(())
    }
}

/// Options for query execution.
#[derive(Debug, Clone, Default)]
pub struct QueryEngineOptions {
    /// Number of context nodes to retrieve.
    pub top_k: Option<usize>,

    /// Search mode to use.
    pub search_mode: Option<cheungfun_core::types::SearchMode>,

    /// Metadata filters to apply.
    pub filters: HashMap<String, serde_json::Value>,

    /// Generation options.
    pub generation_options: Option<GenerationOptions>,
}

impl QueryEngineOptions {
    /// Create new query engine options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of context nodes to retrieve.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set the search mode.
    #[must_use]
    pub fn with_search_mode(mut self, search_mode: cheungfun_core::types::SearchMode) -> Self {
        self.search_mode = Some(search_mode);
        self
    }

    /// Add a metadata filter.
    pub fn with_filter<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.filters.insert(key.into(), value.into());
        self
    }

    /// Set generation options.
    #[must_use]
    pub fn with_generation_options(mut self, options: GenerationOptions) -> Self {
        self.generation_options = Some(options);
        self
    }
}

/// Builder for creating query engines.
#[derive(Debug, Default)]
pub struct QueryEngineBuilder {
    retriever: Option<Arc<dyn Retriever>>,
    generator: Option<Arc<dyn ResponseGenerator>>,
    memory: Option<Arc<tokio::sync::Mutex<dyn BaseMemory>>>,
    config: Option<QueryEngineConfig>,
}

impl QueryEngineBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the retriever.
    pub fn retriever(mut self, retriever: Arc<dyn Retriever>) -> Self {
        self.retriever = Some(retriever);
        self
    }

    /// Set the generator.
    pub fn generator(mut self, generator: Arc<dyn ResponseGenerator>) -> Self {
        self.generator = Some(generator);
        self
    }

    /// Set the memory for conversation history.
    #[must_use]
    pub fn memory(mut self, memory: Arc<tokio::sync::Mutex<dyn BaseMemory>>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Set the configuration.
    #[must_use]
    pub fn config(mut self, config: QueryEngineConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the query engine.
    pub fn build(self) -> Result<QueryEngine> {
        let retriever =
            self.retriever
                .ok_or_else(|| cheungfun_core::CheungfunError::Configuration {
                    message: "Retriever is required".to_string(),
                })?;

        let generator =
            self.generator
                .ok_or_else(|| cheungfun_core::CheungfunError::Configuration {
                    message: "Generator is required".to_string(),
                })?;

        let config = self.config.unwrap_or_default();

        if let Some(memory) = self.memory {
            Ok(QueryEngine::with_memory_and_config(
                retriever, generator, memory, config,
            ))
        } else {
            Ok(QueryEngine::with_config(retriever, generator, config))
        }
    }
}
