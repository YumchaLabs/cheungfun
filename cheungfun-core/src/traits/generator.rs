//! Response generation traits for LLM integration.
//!
//! This module defines traits for generating responses using Large Language Models.
//! These traits provide a unified interface for different LLM providers and
//! response generation strategies.

use async_trait::async_trait;
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;

use crate::{GeneratedResponse, GenerationOptions, Result, ScoredNode};

/// Generates responses using Large Language Models.
///
/// This trait provides the interface for generating text responses based on
/// retrieved context and user queries. Implementations can use different
/// LLM providers like OpenAI, Anthropic, local models, etc.
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_core::traits::ResponseGenerator;
/// use cheungfun_core::{ScoredNode, GeneratedResponse, GenerationOptions, Result};
/// use async_trait::async_trait;
/// use futures::Stream;
/// use std::pin::Pin;
///
/// struct SimpleGenerator;
///
/// #[async_trait]
/// impl ResponseGenerator for SimpleGenerator {
///     async fn generate_response(
///         &self,
///         query: &str,
///         context_nodes: Vec<ScoredNode>,
///         options: &GenerationOptions,
///     ) -> Result<GeneratedResponse> {
///         // Implementation would call LLM and return response
///         Ok(GeneratedResponse::new("This is a generated response."))
///     }
///
///     async fn generate_response_stream(
///         &self,
///         query: &str,
///         context_nodes: Vec<ScoredNode>,
///         options: &GenerationOptions,
///     ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
///         // Implementation would return streaming response
///         Ok(Box::pin(futures::stream::empty()))
///     }
/// }
/// ```
#[async_trait]
pub trait ResponseGenerator: Send + Sync + std::fmt::Debug {
    /// Generate a complete response from retrieved context.
    ///
    /// This method takes a user query and relevant context nodes to generate
    /// a comprehensive response using an LLM.
    ///
    /// # Arguments
    ///
    /// * `query` - The user's question or query
    /// * `context_nodes` - Relevant nodes retrieved for the query
    /// * `options` - Generation options and parameters
    ///
    /// # Returns
    ///
    /// A generated response containing the answer, source references,
    /// and metadata about the generation process.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails due to LLM issues,
    /// network problems, or invalid input.
    async fn generate_response(
        &self,
        query: &str,
        context_nodes: Vec<ScoredNode>,
        options: &GenerationOptions,
    ) -> Result<GeneratedResponse>;

    /// Generate a streaming response for real-time output.
    ///
    /// This method provides the same functionality as `generate_response`
    /// but returns a stream of text chunks for real-time display.
    ///
    /// # Arguments
    ///
    /// * `query` - The user's question or query
    /// * `context_nodes` - Relevant nodes retrieved for the query
    /// * `options` - Generation options and parameters
    ///
    /// # Returns
    ///
    /// A stream that yields text chunks as they are generated.
    async fn generate_response_stream(
        &self,
        query: &str,
        context_nodes: Vec<ScoredNode>,
        options: &GenerationOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>>;

    /// Get a human-readable name for this generator.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Check if the generator is healthy and ready to generate responses.
    async fn health_check(&self) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Get information about the underlying model.
    fn model_info(&self) -> ModelInfo {
        ModelInfo::default()
    }

    /// Get configuration information about this generator.
    fn config(&self) -> HashMap<String, serde_json::Value> {
        // Default implementation returns empty config
        HashMap::new()
    }

    /// Get statistics about generation operations.
    async fn stats(&self) -> Result<GenerationStats> {
        Ok(GenerationStats::default())
    }

    /// Validate that the generator can process the given input.
    async fn can_generate(&self, _query: &str, _context_nodes: &[ScoredNode]) -> bool {
        // Default implementation accepts all inputs
        true
    }

    /// Estimate the cost of generating a response.
    async fn estimate_cost(
        &self,
        _query: &str,
        _context_nodes: &[ScoredNode],
        _options: &GenerationOptions,
    ) -> Result<Option<GenerationCost>> {
        // Default implementation returns no cost estimate
        Ok(None)
    }
}

/// Information about the underlying language model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Name of the model.
    pub name: String,

    /// Version of the model.
    pub version: Option<String>,

    /// Provider of the model (e.g., "openai", "anthropic", "local").
    pub provider: String,

    /// Maximum context length in tokens.
    pub max_context_length: Option<usize>,

    /// Maximum output length in tokens.
    pub max_output_length: Option<usize>,

    /// Supported features.
    pub features: Vec<ModelFeature>,

    /// Additional model metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            name: "unknown".to_string(),
            version: None,
            provider: "unknown".to_string(),
            max_context_length: None,
            max_output_length: None,
            features: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Features supported by a language model.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelFeature {
    /// Supports streaming responses.
    Streaming,

    /// Supports function calling.
    FunctionCalling,

    /// Supports vision/image inputs.
    Vision,

    /// Supports JSON mode.
    JsonMode,

    /// Supports system messages.
    SystemMessages,

    /// Custom feature.
    Custom(String),
}

/// Statistics about response generation operations.
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Total number of responses generated.
    pub responses_generated: usize,

    /// Number of generation operations that failed.
    pub generations_failed: usize,

    /// Total tokens consumed in prompts.
    pub total_prompt_tokens: usize,

    /// Total tokens generated in completions.
    pub total_completion_tokens: usize,

    /// Average generation time per response.
    pub avg_generation_time: std::time::Duration,

    /// Total generation time across all operations.
    pub total_generation_time: std::time::Duration,

    /// Total cost of all operations.
    pub total_cost: Option<f64>,

    /// Additional generator-specific statistics.
    pub additional_stats: HashMap<String, serde_json::Value>,
}

impl GenerationStats {
    /// Create new generation statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate the success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        let total = self.responses_generated + self.generations_failed;
        if total == 0 {
            0.0
        } else {
            (self.responses_generated as f64 / total as f64) * 100.0
        }
    }

    /// Calculate total tokens used.
    pub fn total_tokens(&self) -> usize {
        self.total_prompt_tokens + self.total_completion_tokens
    }

    /// Calculate tokens per second.
    pub fn tokens_per_second(&self) -> f64 {
        if self.total_generation_time.is_zero() {
            0.0
        } else {
            self.total_tokens() as f64 / self.total_generation_time.as_secs_f64()
        }
    }

    /// Calculate average tokens per response.
    pub fn avg_tokens_per_response(&self) -> f64 {
        if self.responses_generated == 0 {
            0.0
        } else {
            self.total_tokens() as f64 / self.responses_generated as f64
        }
    }

    /// Update average generation time.
    pub fn update_avg_time(&mut self) {
        let total = self.responses_generated + self.generations_failed;
        if total > 0 {
            self.avg_generation_time = self.total_generation_time / total as u32;
        }
    }
}

/// Cost information for response generation.
#[derive(Debug, Clone)]
pub struct GenerationCost {
    /// Cost in the provider's currency (usually USD).
    pub amount: f64,

    /// Currency code (e.g., "USD").
    pub currency: String,

    /// Breakdown of costs by component.
    pub breakdown: HashMap<String, f64>,

    /// Additional cost metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl GenerationCost {
    /// Create a new generation cost.
    pub fn new(amount: f64, currency: &str) -> Self {
        Self {
            amount,
            currency: currency.to_string(),
            breakdown: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a cost component to the breakdown.
    pub fn with_component<S: Into<String>>(mut self, component: S, cost: f64) -> Self {
        self.breakdown.insert(component.into(), cost);
        self
    }
}

/// Configuration for response generation.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Default model to use.
    pub default_model: Option<String>,

    /// Default temperature for generation.
    pub default_temperature: Option<f32>,

    /// Default maximum tokens for responses.
    pub default_max_tokens: Option<usize>,

    /// Timeout for generation operations in seconds.
    pub timeout_seconds: Option<u64>,

    /// Maximum number of retry attempts.
    pub max_retries: Option<usize>,

    /// Whether to enable response caching.
    pub enable_caching: bool,

    /// Additional provider-specific configuration.
    pub provider_config: HashMap<String, serde_json::Value>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            default_model: None,
            default_temperature: Some(0.7),
            default_max_tokens: Some(1000),
            timeout_seconds: Some(60),
            max_retries: Some(3),
            enable_caching: false,
            provider_config: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_stats() {
        let mut stats = GenerationStats::new();
        stats.responses_generated = 95;
        stats.generations_failed = 5;
        stats.total_prompt_tokens = 10000;
        stats.total_completion_tokens = 5000;
        stats.total_generation_time = std::time::Duration::from_secs(30);
        stats.update_avg_time();

        assert_eq!(stats.success_rate(), 95.0);
        assert_eq!(stats.total_tokens(), 15000);
        assert_eq!(stats.tokens_per_second(), 500.0);
        assert_eq!(stats.avg_tokens_per_response(), 157.89473684210526);
        assert_eq!(
            stats.avg_generation_time,
            std::time::Duration::from_millis(300)
        );
    }

    #[test]
    fn test_model_info() {
        let info = ModelInfo {
            name: "gpt-4".to_string(),
            provider: "openai".to_string(),
            features: vec![ModelFeature::Streaming, ModelFeature::FunctionCalling],
            ..Default::default()
        };

        assert_eq!(info.name, "gpt-4");
        assert!(info.features.contains(&ModelFeature::Streaming));
    }

    #[test]
    fn test_generation_cost() {
        let cost = GenerationCost::new(0.05, "USD")
            .with_component("prompt", 0.02)
            .with_component("completion", 0.03);

        assert_eq!(cost.amount, 0.05);
        assert_eq!(cost.currency, "USD");
        assert_eq!(cost.breakdown.get("prompt"), Some(&0.02));
        assert_eq!(cost.breakdown.get("completion"), Some(&0.03));
    }
}
