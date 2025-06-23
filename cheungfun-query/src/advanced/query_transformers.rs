// Query Transformers Implementation

use super::*;
use anyhow::{Context, Result};
use async_trait::async_trait;
use cheungfun_core::ResponseGenerator;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// HyDE (Hypothetical Document Embeddings) query transformer.
#[derive(Debug)]
pub struct HyDETransformer {
    /// LLM client.
    pub llm_client: Arc<dyn ResponseGenerator>,
    /// Prompt template for generating hypothetical documents.
    pub prompt_template: String,
    /// Whether to include the original query.
    pub include_original: bool,
    /// The number of hypothetical documents to generate.
    pub num_hypothetical_docs: usize,
    /// Maximum document length.
    pub max_doc_length: usize,
}

impl HyDETransformer {
    /// Creates a new HyDE transformer.
    pub fn new(llm_client: Arc<dyn ResponseGenerator>) -> Self {
        Self {
            llm_client,
            prompt_template: DEFAULT_HYDE_PROMPT.to_string(),
            include_original: true,
            num_hypothetical_docs: 1,
            max_doc_length: 500,
        }
    }

    /// Sets the prompt template.
    pub fn with_prompt_template(mut self, template: String) -> Self {
        self.prompt_template = template;
        self
    }

    /// Sets whether to include the original query.
    pub fn with_include_original(mut self, include: bool) -> Self {
        self.include_original = include;
        self
    }

    /// Sets the number of hypothetical documents.
    pub fn with_num_hypothetical_docs(mut self, num: usize) -> Self {
        self.num_hypothetical_docs = num;
        self
    }

    /// Generates hypothetical documents.
    async fn generate_hypothetical_documents(&self, query: &str) -> Result<Vec<String>> {
        debug!(
            "Generating {} hypothetical documents for query: {}",
            self.num_hypothetical_docs, query
        );

        let mut documents = Vec::new();

        for i in 0..self.num_hypothetical_docs {
            let prompt = self
                .prompt_template
                .replace("{query}", query)
                .replace("{doc_index}", &(i + 1).to_string());

            // Generate hypothetical document using the LLM.
            let response = self
                .llm_client
                .generate_response(
                    &prompt,
                    vec![], // empty context nodes
                    &Default::default(),
                )
                .await
                .context("Failed to generate hypothetical document")?;

            let mut doc = response.content;

            // Limit the document length.
            if doc.len() > self.max_doc_length {
                doc.truncate(self.max_doc_length);
                // Ensure truncation happens at a word boundary.
                if let Some(last_space) = doc.rfind(' ') {
                    doc.truncate(last_space);
                }
            }

            documents.push(doc);
        }

        info!("Generated {} hypothetical documents", documents.len());
        Ok(documents)
    }
}

#[async_trait]
impl QueryTransformer for HyDETransformer {
    async fn transform(&self, query: &mut AdvancedQuery) -> Result<()> {
        debug!(
            "Applying HyDE transformation to query: {}",
            query.original_text
        );

        // Generate hypothetical documents.
        let hypothetical_docs = self
            .generate_hypothetical_documents(&query.original_text)
            .await?;

        // Add hypothetical documents to the transformed queries.
        query.transformed_queries.extend(hypothetical_docs);

        // If including the original query, ensure it is in the list.
        if self.include_original && !query.transformed_queries.contains(&query.original_text) {
            query
                .transformed_queries
                .insert(0, query.original_text.clone());
        }

        // Add metadata.
        query
            .metadata
            .insert("hyde_applied".to_string(), serde_json::Value::Bool(true));
        query.metadata.insert(
            "hyde_docs_count".to_string(),
            serde_json::Value::Number(self.num_hypothetical_docs.into()),
        );

        info!(
            "HyDE transformation completed. Total queries: {}",
            query.transformed_queries.len()
        );
        Ok(())
    }

    fn name(&self) -> &'static str {
        "HyDETransformer"
    }

    fn validate_query(&self, query: &AdvancedQuery) -> Result<()> {
        if query.original_text.trim().is_empty() {
            anyhow::bail!("Query text cannot be empty for HyDE transformation");
        }
        if query.original_text.len() > 1000 {
            warn!(
                "Query text is very long ({}), HyDE may not work well",
                query.original_text.len()
            );
        }
        Ok(())
    }
}

/// Subquestion generation transformer.
#[derive(Debug)]
pub struct SubquestionTransformer {
    /// LLM client.
    pub llm_client: Arc<dyn ResponseGenerator>,
    /// Prompt template for subquestion generation.
    pub prompt_template: String,
    /// The number of subquestions to generate.
    pub num_subquestions: usize,
    /// Whether to include the original query.
    pub include_original: bool,
    /// Maximum subquestion length.
    pub max_subquestion_length: usize,
}

impl SubquestionTransformer {
    /// Creates a new subquestion transformer.
    pub fn new(llm_client: Arc<dyn ResponseGenerator>) -> Self {
        Self {
            llm_client,
            prompt_template: DEFAULT_SUBQUESTION_PROMPT.to_string(),
            num_subquestions: 3,
            include_original: true,
            max_subquestion_length: 200,
        }
    }

    /// Sets the number of subquestions.
    pub fn with_num_subquestions(mut self, num: usize) -> Self {
        self.num_subquestions = num;
        self
    }

    /// Sets the prompt template.
    pub fn with_prompt_template(mut self, template: String) -> Self {
        self.prompt_template = template;
        self
    }

    /// Generates subquestions.
    async fn generate_subquestions(&self, query: &str) -> Result<Vec<String>> {
        debug!(
            "Generating {} subquestions for query: {}",
            self.num_subquestions, query
        );

        let prompt = self
            .prompt_template
            .replace("{query}", query)
            .replace("{num_questions}", &self.num_subquestions.to_string());

        let response = self
            .llm_client
            .generate_response(&prompt, vec![], &Default::default())
            .await
            .context("Failed to generate subquestions")?;

        // Parse the generated subquestions.
        let subquestions = self.parse_subquestions(&response.content)?;

        info!("Generated {} subquestions", subquestions.len());
        Ok(subquestions)
    }

    /// Parses the subquestion text generated by the LLM.
    fn parse_subquestions(&self, text: &str) -> Result<Vec<String>> {
        let mut subquestions = Vec::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Remove number prefixes (e.g., "1. ", "- ", "• " etc.).
            let cleaned = line
                .trim_start_matches(|c: char| {
                    c.is_ascii_digit() || c == '.' || c == '-' || c == '•' || c.is_whitespace()
                })
                .trim();

            if !cleaned.is_empty() && cleaned.len() <= self.max_subquestion_length {
                subquestions.push(cleaned.to_string());
            }

            // Limit the number of subquestions.
            if subquestions.len() >= self.num_subquestions {
                break;
            }
        }

        if subquestions.is_empty() {
            warn!("No valid subquestions parsed from LLM response");
        }

        Ok(subquestions)
    }
}

#[async_trait]
impl QueryTransformer for SubquestionTransformer {
    async fn transform(&self, query: &mut AdvancedQuery) -> Result<()> {
        debug!(
            "Applying subquestion transformation to query: {}",
            query.original_text
        );

        let subquestions = self.generate_subquestions(&query.original_text).await?;

        if self.include_original {
            query.transformed_queries.push(query.original_text.clone());
        }
        query.transformed_queries.extend(subquestions);

        // Add metadata.
        query.metadata.insert(
            "subquestions_applied".to_string(),
            serde_json::Value::Bool(true),
        );
        query.metadata.insert(
            "subquestions_count".to_string(),
            serde_json::Value::Number(self.num_subquestions.into()),
        );

        info!(
            "Subquestion transformation completed. Total queries: {}",
            query.transformed_queries.len()
        );
        Ok(())
    }

    fn name(&self) -> &'static str {
        "SubquestionTransformer"
    }

    fn validate_query(&self, query: &AdvancedQuery) -> Result<()> {
        if query.original_text.trim().is_empty() {
            anyhow::bail!("Query text cannot be empty for subquestion generation");
        }
        if query.original_text.len() < 10 {
            warn!("Query text is very short, subquestion generation may not be effective");
        }
        Ok(())
    }
}

/// Query expansion transformer.
#[derive(Debug)]
pub struct QueryExpansionTransformer {
    /// Expansion dictionary.
    pub expansion_dict: HashMap<String, Vec<String>>,
    /// Synonym weight.
    pub synonym_weight: f32,
    /// Maximum number of expansions.
    pub max_expansions: usize,
}

impl QueryExpansionTransformer {
    /// Creates a new query expansion transformer.
    pub fn new() -> Self {
        Self {
            expansion_dict: HashMap::new(),
            synonym_weight: 0.8,
            max_expansions: 5,
        }
    }

    /// Adds synonyms.
    pub fn add_synonyms(&mut self, word: String, synonyms: Vec<String>) {
        self.expansion_dict.insert(word, synonyms);
    }

    /// Loads the synonym dictionary from a file.
    pub async fn load_synonyms_from_file(&mut self, file_path: &str) -> Result<()> {
        // TODO: Implement synonym loading from file
        todo!("Implement synonym loading from file")
    }
}

#[async_trait]
impl QueryTransformer for QueryExpansionTransformer {
    async fn transform(&self, query: &mut AdvancedQuery) -> Result<()> {
        debug!("Applying query expansion to: {}", query.original_text);

        let words: Vec<&str> = query.original_text.split_whitespace().collect();
        let mut expanded_terms = Vec::new();

        for word in words {
            let word_lower = word.to_lowercase();
            if let Some(synonyms) = self.expansion_dict.get(&word_lower) {
                expanded_terms.extend(synonyms.iter().take(self.max_expansions).cloned());
            }
        }

        if !expanded_terms.is_empty() {
            let expanded_query = format!("{} {}", query.original_text, expanded_terms.join(" "));
            query.transformed_queries.push(expanded_query);

            query.metadata.insert(
                "expansion_applied".to_string(),
                serde_json::Value::Bool(true),
            );
            query.metadata.insert(
                "expanded_terms_count".to_string(),
                serde_json::Value::Number(expanded_terms.len().into()),
            );
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "QueryExpansionTransformer"
    }
}

// Default prompt templates
const DEFAULT_HYDE_PROMPT: &str = r#"
Please write a passage to answer the question: {query}

The passage should be informative and directly address the question. 
Write as if you are providing a comprehensive answer based on reliable sources.

Passage:
"#;

const DEFAULT_SUBQUESTION_PROMPT: &str = r#"
Given the following complex question, break it down into {num_questions} simpler, more specific sub-questions that would help answer the original question comprehensively.

Original question: {query}

Please provide {num_questions} sub-questions, each on a new line:
"#;
