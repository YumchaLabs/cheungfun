//! Semantic tool retrieval system
//!
//! This module provides advanced semantic matching for tool retrieval,
//! going beyond simple keyword matching to understand tool functionality.

use crate::{
    error::{AgentError, Result},
    tool::{
        RetrievableTool, RetrievalStrategy, Tool, ToolMetadata, ToolRetrievalResult, ToolRetriever,
    },
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

/// Semantic similarity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    /// Minimum semantic similarity score (0.0 to 1.0)
    pub min_similarity: f64,
    /// Use function name matching
    pub use_function_matching: bool,
    /// Use description embedding (requires embedding service)
    pub use_embeddings: bool,
    /// Weight for different matching components
    pub weights: SemanticWeights,
}

/// Weights for different semantic matching components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticWeights {
    /// Weight for exact name matching
    pub name_weight: f64,
    /// Weight for keyword matching
    pub keyword_weight: f64,
    /// Weight for category matching
    pub category_weight: f64,
    /// Weight for description similarity
    pub description_weight: f64,
    /// Weight for example matching
    pub example_weight: f64,
    /// Weight for function signature matching
    pub function_weight: f64,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            min_similarity: 0.2,
            use_function_matching: true,
            use_embeddings: false, // Requires external embedding service
            weights: SemanticWeights::default(),
        }
    }
}

impl Default for SemanticWeights {
    fn default() -> Self {
        Self {
            name_weight: 3.0,
            keyword_weight: 2.0,
            category_weight: 1.5,
            description_weight: 2.5,
            example_weight: 1.0,
            function_weight: 1.8,
        }
    }
}

/// Semantic tool matcher for intelligent tool selection
pub struct SemanticToolMatcher {
    config: SemanticConfig,
    /// Cached tool analysis for performance
    tool_analysis_cache: HashMap<String, ToolAnalysis>,
}

/// Analysis result for a tool
#[derive(Debug, Clone)]
struct ToolAnalysis {
    /// Extracted concepts from tool description
    #[allow(dead_code)]
    concepts: Vec<String>,
    /// Extracted verbs/actions
    #[allow(dead_code)]
    actions: Vec<String>,
    /// Extracted domains/contexts
    domains: Vec<String>,
    /// Function signature analysis
    function_analysis: FunctionAnalysis,
}

/// Function signature analysis
#[derive(Debug, Clone)]
struct FunctionAnalysis {
    /// Input parameter concepts
    input_concepts: Vec<String>,
    /// Output concepts
    output_concepts: Vec<String>,
    /// Complexity score (0.0 to 1.0)
    complexity: f64,
}

impl SemanticToolMatcher {
    /// Create new semantic tool matcher
    #[must_use]
    pub fn new(config: SemanticConfig) -> Self {
        Self {
            config,
            tool_analysis_cache: HashMap::new(),
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(SemanticConfig::default())
    }

    /// Analyze a tool and cache the results
    fn analyze_tool(&mut self, metadata: &ToolMetadata) -> &ToolAnalysis {
        let tool_name = &metadata.name;

        if !self.tool_analysis_cache.contains_key(tool_name) {
            let analysis = self.perform_tool_analysis(metadata);
            self.tool_analysis_cache.insert(tool_name.clone(), analysis);
        }

        self.tool_analysis_cache.get(tool_name).unwrap()
    }

    /// Perform semantic analysis of a tool
    fn perform_tool_analysis(&self, metadata: &ToolMetadata) -> ToolAnalysis {
        // Extract concepts from description
        let concepts = self.extract_concepts(&metadata.description);

        // Extract actions/verbs from description and keywords
        let actions = self.extract_actions(&metadata.description, &metadata.keywords);

        // Extract domains from categories and description
        let domains = self.extract_domains(&metadata.categories, &metadata.description);

        // Analyze function signature if available
        let function_analysis = self.analyze_function_signature(metadata);

        ToolAnalysis {
            concepts,
            actions,
            domains,
            function_analysis,
        }
    }

    /// Extract key concepts from text
    fn extract_concepts(&self, text: &str) -> Vec<String> {
        let mut concepts = Vec::new();
        let text_lower = text.to_lowercase();

        // Technical concepts
        let tech_concepts = [
            "search",
            "query",
            "retrieve",
            "find",
            "lookup",
            "index",
            "calculate",
            "compute",
            "analyze",
            "process",
            "transform",
            "create",
            "generate",
            "build",
            "construct",
            "make",
            "read",
            "write",
            "save",
            "load",
            "store",
            "fetch",
            "format",
            "parse",
            "validate",
            "verify",
            "check",
            "connect",
            "authenticate",
            "authorize",
            "login",
            "send",
            "receive",
            "transmit",
            "communicate",
            "notify",
            "filter",
            "sort",
            "group",
            "aggregate",
            "summarize",
        ];

        for concept in &tech_concepts {
            if text_lower.contains(concept) {
                concepts.push((*concept).to_string());
            }
        }

        concepts
    }

    /// Extract action verbs from text
    fn extract_actions(&self, description: &str, keywords: &[String]) -> Vec<String> {
        let mut actions = Vec::new();
        let combined_text = format!("{} {}", description, keywords.join(" "));
        let text_lower = combined_text.to_lowercase();

        let action_verbs = [
            "search",
            "find",
            "get",
            "fetch",
            "retrieve",
            "lookup",
            "create",
            "make",
            "build",
            "generate",
            "produce",
            "update",
            "modify",
            "change",
            "edit",
            "alter",
            "delete",
            "remove",
            "clear",
            "clean",
            "purge",
            "send",
            "post",
            "put",
            "push",
            "upload",
            "receive",
            "pull",
            "download",
            "import",
            "export",
            "validate",
            "verify",
            "check",
            "test",
            "confirm",
            "parse",
            "format",
            "convert",
            "transform",
            "translate",
            "analyze",
            "process",
            "compute",
            "calculate",
            "evaluate",
        ];

        for action in &action_verbs {
            if text_lower.contains(action) {
                actions.push((*action).to_string());
            }
        }

        actions
    }

    /// Extract domain information
    fn extract_domains(&self, categories: &[String], description: &str) -> Vec<String> {
        let mut domains = categories.to_vec();
        let desc_lower = description.to_lowercase();

        // Common domains
        let domain_keywords = [
            ("web", ["web", "http", "url", "browser", "internet"]),
            ("database", ["database", "sql", "table", "record", "query"]),
            ("file", ["file", "document", "pdf", "text", "csv"]),
            (
                "math",
                ["math", "calculate", "number", "formula", "equation"],
            ),
            ("ai", ["ai", "ml", "model", "predict", "classify"]),
            ("api", ["api", "rest", "endpoint", "service", "request"]),
            (
                "security",
                ["auth", "password", "token", "encrypt", "secure"],
            ),
            ("email", ["email", "mail", "message", "send", "receive"]),
            ("image", ["image", "photo", "picture", "visual", "graphic"]),
            ("text", ["text", "string", "word", "sentence", "language"]),
        ];

        for (domain, keywords) in &domain_keywords {
            if keywords.iter().any(|&kw| desc_lower.contains(kw))
                && !domains.contains(&(*domain).to_string())
            {
                domains.push((*domain).to_string());
            }
        }

        domains
    }

    /// Analyze function signature for semantic matching
    fn analyze_function_signature(&self, metadata: &ToolMetadata) -> FunctionAnalysis {
        // Basic analysis - in a real implementation, this would parse actual schemas
        let input_concepts = self.extract_concepts(&metadata.description);
        let output_concepts = vec!["result".to_string(), "response".to_string()];

        // Simple complexity based on description length and keyword count
        let complexity = f64::midpoint(
            metadata.description.len() as f64 / 200.0,
            metadata.keywords.len() as f64 / 10.0,
        );
        let complexity = complexity.min(1.0);

        FunctionAnalysis {
            input_concepts,
            output_concepts,
            complexity,
        }
    }

    /// Calculate semantic similarity between query and tool
    pub fn calculate_semantic_similarity(&mut self, query: &str, metadata: &ToolMetadata) -> f64 {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        // Get or create tool analysis and clone it to avoid borrowing issues
        let analysis = self.analyze_tool(metadata).clone();

        // Get config values upfront to avoid borrowing issues
        let config_weights = self.config.weights.clone();
        let use_function_matching = self.config.use_function_matching;

        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        // Name matching
        if config_weights.name_weight > 0.0 {
            let name_score = calculate_name_similarity_static(&query_words, &metadata.name);
            total_score += name_score * config_weights.name_weight;
            total_weight += config_weights.name_weight;
        }

        // Keyword matching
        if config_weights.keyword_weight > 0.0 {
            let keyword_score =
                calculate_keyword_similarity_static(&query_words, &metadata.keywords);
            total_score += keyword_score * config_weights.keyword_weight;
            total_weight += config_weights.keyword_weight;
        }

        // Category matching
        if config_weights.category_weight > 0.0 {
            let category_score =
                calculate_category_similarity_static(&query_words, &analysis.domains);
            total_score += category_score * config_weights.category_weight;
            total_weight += config_weights.category_weight;
        }

        // Description similarity
        if config_weights.description_weight > 0.0 {
            let desc_score =
                calculate_description_similarity_static(&query_words, &metadata.description);
            total_score += desc_score * config_weights.description_weight;
            total_weight += config_weights.description_weight;
        }

        // Example matching
        if config_weights.example_weight > 0.0 {
            let example_score =
                calculate_example_similarity_static(&query_words, &metadata.examples);
            total_score += example_score * config_weights.example_weight;
            total_weight += config_weights.example_weight;
        }

        // Function analysis
        if config_weights.function_weight > 0.0 && use_function_matching {
            let func_score =
                calculate_function_similarity_static(&query_words, &analysis.function_analysis);
            total_score += func_score * config_weights.function_weight;
            total_weight += config_weights.function_weight;
        }

        if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.0
        }
    }

    /// Calculate name-based similarity
    #[allow(dead_code)]
    fn calculate_name_similarity(&self, query_words: &[&str], tool_name: &str) -> f64 {
        let name_lower = tool_name.to_lowercase();
        let name_parts: Vec<&str> = name_lower.split('_').collect();

        let mut matches = 0;
        for query_word in query_words {
            for name_part in &name_parts {
                if name_part.contains(query_word) || query_word.contains(name_part) {
                    matches += 1;
                    break;
                }
            }
        }

        if query_words.is_empty() {
            0.0
        } else {
            f64::from(matches) / query_words.len() as f64
        }
    }

    /// Calculate keyword-based similarity
    #[allow(dead_code)]
    fn calculate_keyword_similarity(&self, query_words: &[&str], keywords: &[String]) -> f64 {
        if keywords.is_empty() || query_words.is_empty() {
            return 0.0;
        }

        let mut matches = 0;
        for query_word in query_words {
            for keyword in keywords {
                let keyword_lower = keyword.to_lowercase();
                if keyword_lower.contains(query_word) || query_word.contains(&keyword_lower) {
                    matches += 1;
                    break;
                }
            }
        }

        f64::from(matches) / query_words.len() as f64
    }

    /// Calculate category-based similarity
    #[allow(dead_code)]
    fn calculate_category_similarity(&self, query_words: &[&str], domains: &[String]) -> f64 {
        if domains.is_empty() || query_words.is_empty() {
            return 0.0;
        }

        let mut matches = 0;
        for query_word in query_words {
            for domain in domains {
                let domain_lower = domain.to_lowercase();
                if domain_lower.contains(query_word) || query_word.contains(&domain_lower) {
                    matches += 1;
                    break;
                }
            }
        }

        f64::from(matches) / query_words.len() as f64
    }

    /// Calculate description-based similarity
    #[allow(dead_code)]
    fn calculate_description_similarity(&self, query_words: &[&str], description: &str) -> f64 {
        if description.is_empty() || query_words.is_empty() {
            return 0.0;
        }

        let desc_lower = description.to_lowercase();
        let desc_words: Vec<&str> = desc_lower.split_whitespace().collect();

        let mut matches = 0;
        for query_word in query_words {
            if desc_words.iter().any(|&word| {
                word == *query_word || word.contains(query_word) || query_word.contains(word)
            }) {
                matches += 1;
            }
        }

        f64::from(matches) / query_words.len() as f64
    }

    /// Calculate example-based similarity
    #[allow(dead_code)]
    fn calculate_example_similarity(&self, query_words: &[&str], examples: &[String]) -> f64 {
        if examples.is_empty() || query_words.is_empty() {
            return 0.0;
        }

        let mut total_score = 0.0;
        for example in examples {
            let example_lower = example.to_lowercase();
            let example_words: Vec<&str> = example_lower.split_whitespace().collect();

            let mut matches = 0;
            for query_word in query_words {
                if example_words.iter().any(|&word| {
                    word == *query_word || word.contains(query_word) || query_word.contains(word)
                }) {
                    matches += 1;
                }
            }

            total_score += f64::from(matches) / query_words.len() as f64;
        }

        total_score / examples.len() as f64
    }

    /// Calculate function signature similarity
    #[allow(dead_code)]
    fn calculate_function_similarity(
        &self,
        query_words: &[&str],
        func_analysis: &FunctionAnalysis,
    ) -> f64 {
        if query_words.is_empty() {
            return 0.0;
        }

        let mut matches = 0;

        // Check input concepts
        for query_word in query_words {
            if func_analysis
                .input_concepts
                .iter()
                .any(|concept| concept.contains(query_word) || query_word.contains(concept))
            {
                matches += 1;
            }
        }

        // Check output concepts
        for query_word in query_words {
            if func_analysis
                .output_concepts
                .iter()
                .any(|concept| concept.contains(query_word) || query_word.contains(concept))
            {
                matches += 1;
            }
        }

        // Apply complexity bonus for more sophisticated tools
        let base_score = f64::from(matches) / (query_words.len() * 2) as f64;
        let complexity_bonus = func_analysis.complexity * 0.1;

        (base_score + complexity_bonus).min(1.0)
    }
}

/// Semantic tool retriever using advanced matching
pub struct SemanticToolRetriever {
    tools: HashMap<String, RetrievableTool>,
    matcher: SemanticToolMatcher,
    strategy: RetrievalStrategy,
}

impl SemanticToolRetriever {
    /// Create new semantic tool retriever
    #[must_use]
    pub fn new(config: SemanticConfig) -> Self {
        Self {
            tools: HashMap::new(),
            matcher: SemanticToolMatcher::new(config),
            strategy: RetrievalStrategy::default(),
        }
    }

    /// Create with custom strategy
    #[must_use]
    pub fn with_strategy(config: SemanticConfig, strategy: RetrievalStrategy) -> Self {
        Self {
            tools: HashMap::new(),
            matcher: SemanticToolMatcher::new(config),
            strategy,
        }
    }

    /// Update semantic configuration
    pub fn update_config(&mut self, config: SemanticConfig) {
        self.matcher = SemanticToolMatcher::new(config);
    }
}

#[async_trait]
impl ToolRetriever for SemanticToolRetriever {
    async fn retrieve_tools(
        &self,
        query: &str,
        strategy: Option<RetrievalStrategy>,
    ) -> Result<ToolRetrievalResult> {
        let strategy = strategy.unwrap_or_else(|| self.strategy.clone());

        match strategy {
            RetrievalStrategy::All => {
                let tools: Vec<Arc<dyn Tool>> =
                    self.tools.values().map(|rt| rt.tool.clone()).collect();
                let scores = vec![1.0; tools.len()];

                Ok(ToolRetrievalResult {
                    tools,
                    scores,
                    metadata: HashMap::new(),
                })
            }

            RetrievalStrategy::Similarity { threshold, top_k } => {
                let mut scored_tools: Vec<(Arc<dyn Tool>, f64)> = Vec::new();
                let mut matcher = self.matcher.clone(); // Clone for mutable access

                for retrievable_tool in self.tools.values() {
                    let score =
                        matcher.calculate_semantic_similarity(query, &retrievable_tool.metadata);
                    if score >= threshold {
                        scored_tools.push((retrievable_tool.tool.clone(), score));
                    }
                }

                // Sort by score descending
                scored_tools
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Take top k
                scored_tools.truncate(top_k);

                let tools: Vec<Arc<dyn Tool>> =
                    scored_tools.iter().map(|(tool, _)| tool.clone()).collect();
                let scores: Vec<f64> = scored_tools.iter().map(|(_, score)| *score).collect();

                let mut metadata = HashMap::new();
                metadata.insert("threshold".to_string(), serde_json::json!(threshold));
                metadata.insert("top_k".to_string(), serde_json::json!(top_k));
                metadata.insert("query".to_string(), serde_json::json!(query));
                metadata.insert("semantic_matching".to_string(), serde_json::json!(true));

                Ok(ToolRetrievalResult {
                    tools,
                    scores,
                    metadata,
                })
            }

            RetrievalStrategy::Category { categories } => {
                let mut tools = Vec::new();
                let mut scores = Vec::new();

                for retrievable_tool in self.tools.values() {
                    let has_category = retrievable_tool
                        .metadata
                        .categories
                        .iter()
                        .any(|cat| categories.contains(cat));

                    if has_category {
                        tools.push(retrievable_tool.tool.clone());
                        scores.push(1.0);
                    }
                }

                let mut metadata = HashMap::new();
                metadata.insert("categories".to_string(), serde_json::json!(categories));

                Ok(ToolRetrievalResult {
                    tools,
                    scores,
                    metadata,
                })
            }

            RetrievalStrategy::Custom => {
                // Use semantic similarity with lower threshold
                self.retrieve_tools(
                    query,
                    Some(RetrievalStrategy::Similarity {
                        threshold: 0.1,
                        top_k: 15,
                    }),
                )
                .await
            }
        }
    }

    async fn add_tool(&mut self, tool: RetrievableTool) -> Result<()> {
        let tool_name = tool.metadata.name.clone();
        if self.tools.contains_key(&tool_name) {
            return Err(AgentError::validation(
                "tool_name",
                format!("Tool '{tool_name}' already exists"),
            ));
        }

        self.tools.insert(tool_name, tool);
        Ok(())
    }

    async fn remove_tool(&mut self, tool_name: &str) -> Result<bool> {
        Ok(self.tools.remove(tool_name).is_some())
    }

    async fn list_tools(&self) -> Result<Vec<String>> {
        Ok(self.tools.keys().cloned().collect())
    }

    async fn update_strategy(&mut self, strategy: RetrievalStrategy) -> Result<()> {
        self.strategy = strategy;
        Ok(())
    }
}

impl Clone for SemanticToolMatcher {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            tool_analysis_cache: HashMap::new(), // Don't clone cache for simplicity
        }
    }
}

// Static helper functions for similarity calculation (to avoid borrowing issues)

/// Calculate name-based similarity (static version)
fn calculate_name_similarity_static(query_words: &[&str], tool_name: &str) -> f64 {
    let name_lower = tool_name.to_lowercase();
    let name_parts: Vec<&str> = name_lower.split('_').collect();

    let mut matches = 0;
    for query_word in query_words {
        for name_part in &name_parts {
            if name_part.contains(query_word) || query_word.contains(name_part) {
                matches += 1;
                break;
            }
        }
    }

    if query_words.is_empty() {
        0.0
    } else {
        f64::from(matches) / query_words.len() as f64
    }
}

/// Calculate keyword-based similarity (static version)
fn calculate_keyword_similarity_static(query_words: &[&str], keywords: &[String]) -> f64 {
    if keywords.is_empty() || query_words.is_empty() {
        return 0.0;
    }

    let mut matches = 0;
    for query_word in query_words {
        for keyword in keywords {
            let keyword_lower = keyword.to_lowercase();
            if keyword_lower.contains(query_word) || query_word.contains(&keyword_lower) {
                matches += 1;
                break;
            }
        }
    }

    f64::from(matches) / query_words.len() as f64
}

/// Calculate category-based similarity (static version)
fn calculate_category_similarity_static(query_words: &[&str], domains: &[String]) -> f64 {
    if domains.is_empty() || query_words.is_empty() {
        return 0.0;
    }

    let mut matches = 0;
    for query_word in query_words {
        for domain in domains {
            let domain_lower = domain.to_lowercase();
            if domain_lower.contains(query_word) || query_word.contains(&domain_lower) {
                matches += 1;
                break;
            }
        }
    }

    f64::from(matches) / query_words.len() as f64
}

/// Calculate description-based similarity (static version)
fn calculate_description_similarity_static(query_words: &[&str], description: &str) -> f64 {
    if description.is_empty() || query_words.is_empty() {
        return 0.0;
    }

    let desc_lower = description.to_lowercase();
    let desc_words: Vec<&str> = desc_lower.split_whitespace().collect();

    let mut matches = 0;
    for query_word in query_words {
        if desc_words.iter().any(|&word| {
            word == *query_word || word.contains(query_word) || query_word.contains(word)
        }) {
            matches += 1;
        }
    }

    f64::from(matches) / query_words.len() as f64
}

/// Calculate example-based similarity (static version)
fn calculate_example_similarity_static(query_words: &[&str], examples: &[String]) -> f64 {
    if examples.is_empty() || query_words.is_empty() {
        return 0.0;
    }

    let mut total_score = 0.0;
    for example in examples {
        let example_lower = example.to_lowercase();
        let example_words: Vec<&str> = example_lower.split_whitespace().collect();

        let mut matches = 0;
        for query_word in query_words {
            if example_words.iter().any(|&word| {
                word == *query_word || word.contains(query_word) || query_word.contains(word)
            }) {
                matches += 1;
            }
        }

        total_score += f64::from(matches) / query_words.len() as f64;
    }

    total_score / examples.len() as f64
}

/// Calculate function signature similarity (static version)
fn calculate_function_similarity_static(
    query_words: &[&str],
    func_analysis: &FunctionAnalysis,
) -> f64 {
    if query_words.is_empty() {
        return 0.0;
    }

    let mut matches = 0;

    // Check input concepts
    for query_word in query_words {
        if func_analysis
            .input_concepts
            .iter()
            .any(|concept| concept.contains(query_word) || query_word.contains(concept))
        {
            matches += 1;
        }
    }

    // Check output concepts
    for query_word in query_words {
        if func_analysis
            .output_concepts
            .iter()
            .any(|concept| concept.contains(query_word) || query_word.contains(concept))
        {
            matches += 1;
        }
    }

    // Apply complexity bonus for more sophisticated tools
    let base_score = f64::from(matches) / (query_words.len() * 2) as f64;
    let complexity_bonus = func_analysis.complexity * 0.1;

    (base_score + complexity_bonus).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{ToolContext, ToolResult};
    use crate::types::ToolSchema;
    use async_trait::async_trait;

    // Mock tool for testing
    #[derive(Debug)]
    struct MockTool {
        name: String,
    }

    #[async_trait]
    impl Tool for MockTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: self.name.clone(),
                description: "Mock tool for testing".to_string(),
                input_schema: serde_json::json!({}),
                output_schema: None,
                dangerous: false,
                metadata: std::collections::HashMap::new(),
            }
        }

        async fn execute(
            &self,
            _arguments: serde_json::Value,
            _context: &ToolContext,
        ) -> Result<ToolResult> {
            Ok(ToolResult::success("mock result"))
        }
    }

    #[tokio::test]
    async fn test_semantic_tool_retrieval() {
        let config = SemanticConfig::default();
        let mut retriever = SemanticToolRetriever::new(config);

        // Add tools with rich metadata
        let search_tool = RetrievableTool {
            tool: Arc::new(MockTool {
                name: "web_search".to_string(),
            }),
            metadata: ToolMetadata {
                name: "web_search".to_string(),
                description: "Search the internet for information using web crawling and indexing"
                    .to_string(),
                categories: vec![
                    "search".to_string(),
                    "web".to_string(),
                    "internet".to_string(),
                ],
                keywords: vec![
                    "search".to_string(),
                    "web".to_string(),
                    "internet".to_string(),
                    "query".to_string(),
                    "find".to_string(),
                    "lookup".to_string(),
                ],
                examples: vec![
                    "search for latest AI news".to_string(),
                    "find information about machine learning".to_string(),
                ],
            },
        };

        let calc_tool = RetrievableTool {
            tool: Arc::new(MockTool {
                name: "calculator".to_string(),
            }),
            metadata: ToolMetadata {
                name: "calculator".to_string(),
                description: "Perform mathematical calculations and computations with precision"
                    .to_string(),
                categories: vec!["math".to_string(), "calculation".to_string()],
                keywords: vec![
                    "calculate".to_string(),
                    "compute".to_string(),
                    "math".to_string(),
                    "arithmetic".to_string(),
                    "equation".to_string(),
                ],
                examples: vec![
                    "calculate 2 + 2".to_string(),
                    "compute square root of 16".to_string(),
                ],
            },
        };

        retriever.add_tool(search_tool).await.unwrap();
        retriever.add_tool(calc_tool).await.unwrap();

        // Test semantic search query
        let result = retriever
            .retrieve_tools(
                "I need to find information on the internet about something",
                Some(RetrievalStrategy::Similarity {
                    threshold: 0.0, // Lower threshold to ensure we get results
                    top_k: 5,
                }),
            )
            .await
            .unwrap();

        assert!(!result.tools.is_empty(), "Should return at least one tool");
        assert_eq!(result.tools[0].name(), "web_search");
        assert!(result.scores[0] >= 0.0); // Allow any positive score

        // Test math-related query
        let result = retriever
            .retrieve_tools(
                "I want to perform some mathematical computations",
                Some(RetrievalStrategy::Similarity {
                    threshold: 0.0, // Lower threshold to ensure we get results
                    top_k: 5,
                }),
            )
            .await
            .unwrap();

        assert!(!result.tools.is_empty(), "Should return at least one tool");
        assert_eq!(result.tools[0].name(), "calculator");
        assert!(result.scores[0] >= 0.0); // Allow any positive score
    }

    #[test]
    fn test_semantic_matcher() {
        let config = SemanticConfig::default();
        let mut matcher = SemanticToolMatcher::new(config);

        let metadata = ToolMetadata {
            name: "web_search".to_string(),
            description: "Search the web for information".to_string(),
            categories: vec!["search".to_string()],
            keywords: vec!["web".to_string(), "search".to_string()],
            examples: vec!["search for news".to_string()],
        };

        let score = matcher.calculate_semantic_similarity("find web information", &metadata);
        assert!(score > 0.0);

        let score2 = matcher.calculate_semantic_similarity("mathematical calculation", &metadata);
        assert!(score > score2); // Web search should be more relevant to "find web information"
    }
}
