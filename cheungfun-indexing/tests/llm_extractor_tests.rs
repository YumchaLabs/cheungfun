//! Comprehensive tests for LLM-driven entity extraction.
//!
//! These tests cover both unit tests with mocks and integration tests with real LLM calls.

use async_trait::async_trait;
use cheungfun_core::{
    traits::{Transform, TransformInput},
    types::{Document, Triplet},
};
use cheungfun_indexing::transformers::llm_extractor::{ExtractionFormat, JsonTriplet};
use cheungfun_indexing::transformers::{LlmExtractionConfig, LlmExtractor};
use siumai::{prelude::*, LlmClient, LlmError};
use std::{collections::HashMap, sync::Arc};

/// Mock LLM client for testing without API calls.
#[derive(Debug, Clone)]
pub struct MockLlmClient {
    pub responses: HashMap<String, String>,
    pub default_response: String,
}

impl MockLlmClient {
    pub fn new() -> Self {
        let mut responses = HashMap::new();

        // Mock response for JSON format
        responses.insert(
            "json_extraction".to_string(),
            r#"[
                {"subject": "Alice", "predicate": "works at", "object": "Microsoft"},
                {"subject": "Alice", "predicate": "lives in", "object": "Seattle"},
                {"subject": "Microsoft", "predicate": "located in", "object": "Seattle"}
            ]"#
            .to_string(),
        );

        // Mock response for parentheses format
        responses.insert(
            "parentheses_extraction".to_string(),
            "(Alice, works at, Microsoft)\n(Alice, lives in, Seattle)\n(Microsoft, located in, Seattle)".to_string(),
        );

        Self {
            responses,
            default_response: r#"[{"subject": "test", "predicate": "is", "object": "example"}]"#
                .to_string(),
        }
    }

    pub fn with_response(mut self, key: &str, response: &str) -> Self {
        self.responses.insert(key.to_string(), response.to_string());
        self
    }
}

#[async_trait]
impl ChatCapability for MockLlmClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Determine response based on message content
        let prompt = messages
            .iter()
            .map(|m| m.content.all_text())
            .collect::<Vec<_>>()
            .join(" ");

        let response_text = if prompt.contains("JSON") {
            self.responses
                .get("json_extraction")
                .unwrap_or(&self.default_response)
                .clone()
        } else if prompt.contains("(") {
            self.responses
                .get("parentheses_extraction")
                .unwrap_or(&self.default_response)
                .clone()
        } else {
            self.default_response.clone()
        };

        Ok(ChatResponse {
            id: Some("mock-response".to_string()),
            content: MessageContent::Text(response_text),
            model: Some("mock-model".to_string()),
            usage: None,
            finish_reason: Some(FinishReason::Stop),
            tool_calls: None,
            thinking: None,
            metadata: HashMap::new(),
        })
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Streaming not supported in mock".to_string(),
        ))
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    fn provider_name(&self) -> &'static str {
        "mock"
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["mock-model".to_string()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
}

// Unit tests for JSON parsing functionality
#[tokio::test]
async fn test_json_triplet_parsing() {
    let json_response = r#"[
        {"subject": "Alice", "predicate": "works at", "object": "Microsoft"},
        {"subject": "Alice", "predicate": "lives in", "object": "Seattle"},
        {"subject": "Microsoft", "predicate": "located in", "object": "Seattle"}
    ]"#;

    let triplets: Vec<JsonTriplet> = serde_json::from_str(json_response).unwrap();
    assert_eq!(triplets.len(), 3);
    assert_eq!(triplets[0].subject, "Alice");
    assert_eq!(triplets[0].predicate, "works at");
    assert_eq!(triplets[0].object, "Microsoft");
}

#[tokio::test]
async fn test_parentheses_parsing() {
    // This would test the parentheses parsing logic
    // For now, we'll just test the basic structure
    let parentheses_response = "(Alice, works at, Microsoft)\n(Alice, lives in, Seattle)\n(Microsoft, located in, Seattle)";
    let lines: Vec<&str> = parentheses_response.lines().collect();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].contains("Alice"));
    assert!(lines[0].contains("works at"));
    assert!(lines[0].contains("Microsoft"));
}

#[tokio::test]
async fn test_llm_extraction_config_creation() {
    let config = LlmExtractionConfig {
        format: ExtractionFormat::Json,
        max_triplets_per_chunk: 5,
        enable_validation: true,
        enable_deduplication: true,
        ..Default::default()
    };

    assert_eq!(config.format, ExtractionFormat::Json);
    assert_eq!(config.max_triplets_per_chunk, 5);
    assert!(config.enable_validation);
    assert!(config.enable_deduplication);
}

#[tokio::test]
async fn test_extraction_format_enum() {
    // Test that ExtractionFormat enum works correctly
    let json_format = ExtractionFormat::Json;
    let parentheses_format = ExtractionFormat::Parentheses;

    assert_ne!(json_format, parentheses_format);
}

// Mock tests with controlled responses
#[tokio::test]
async fn test_llm_extractor_json_format() {
    let llm_client = MockLlmClient::new();

    let config = LlmExtractionConfig {
        format: ExtractionFormat::Json,
        max_triplets_per_chunk: 10,
        temperature: 0.1,
        enable_validation: true,
        enable_deduplication: true,
        ..Default::default()
    };

    let extractor = LlmExtractor::new(Arc::new(llm_client), config).unwrap();

    let documents = vec![Document::new(
        "Alice works at Microsoft and lives in Seattle.",
    )];
    let input = TransformInput::Documents(documents);
    let result = extractor.transform(input).await.unwrap();

    assert_eq!(result.len(), 1);

    let node = &result[0];
    let triplets_value = node.metadata.get("extracted_triplets").unwrap();
    let triplets: Vec<Triplet> = serde_json::from_value(triplets_value.clone()).unwrap();

    // Should extract 3 triplets from the mock response
    assert_eq!(triplets.len(), 3);

    // Verify specific triplets
    let subjects: Vec<&str> = triplets.iter().map(|t| t.source.name.as_str()).collect();
    assert!(subjects.contains(&"Alice"));
    assert!(subjects.contains(&"Microsoft"));
}

#[tokio::test]
async fn test_llm_extractor_parentheses_format() {
    let llm_client = MockLlmClient::new();

    let config = LlmExtractionConfig {
        format: ExtractionFormat::Parentheses,
        max_triplets_per_chunk: 10,
        temperature: 0.1,
        enable_validation: true,
        enable_deduplication: true,
        ..Default::default()
    };

    let extractor = LlmExtractor::new(Arc::new(llm_client), config).unwrap();

    let documents = vec![Document::new(
        "Alice works at Microsoft and lives in Seattle.",
    )];
    let input = TransformInput::Documents(documents);
    let result = extractor.transform(input).await.unwrap();

    assert_eq!(result.len(), 1);

    let node = &result[0];
    let triplets_value = node.metadata.get("extracted_triplets").unwrap();
    let triplets: Vec<Triplet> = serde_json::from_value(triplets_value.clone()).unwrap();

    // Should extract 3 triplets from the mock response
    assert_eq!(triplets.len(), 3);
}

#[tokio::test]
async fn test_llm_extraction_config_defaults() {
    let config = LlmExtractionConfig::default();
    assert_eq!(config.format, ExtractionFormat::Json);
    assert_eq!(config.max_triplets_per_chunk, 10);
    assert_eq!(config.temperature, 0.1);
    assert!(config.enable_validation);
    assert!(config.enable_deduplication);
}

#[tokio::test]
async fn test_json_triplet_deserialization() {
    let json_str = r#"[
        {"subject": "Alice", "predicate": "works at", "object": "Microsoft"},
        {"subject": "Bob", "predicate": "lives in", "object": "Seattle"}
    ]"#;

    let triplets: Vec<JsonTriplet> = serde_json::from_str(json_str).unwrap();
    assert_eq!(triplets.len(), 2);
    assert_eq!(triplets[0].subject, "Alice");
    assert_eq!(triplets[0].predicate, "works at");
    assert_eq!(triplets[0].object, "Microsoft");
}

#[tokio::test]
async fn test_validation_filters_self_loops() {
    let mock_client = Arc::new(MockLlmClient::new().with_response(
        "json_extraction",
        r#"[
            {"subject": "Alice", "predicate": "works at", "object": "Microsoft"},
            {"subject": "Alice", "predicate": "is", "object": "Alice"}
        ]"#,
    ));

    let config = LlmExtractionConfig {
        format: ExtractionFormat::Json,
        enable_validation: true,
        ..Default::default()
    };

    let extractor = LlmExtractor::new(mock_client, config).unwrap();

    let documents = vec![Document::new("Test document")];
    let input = TransformInput::Documents(documents);
    let result = extractor.transform(input).await.unwrap();

    let node = &result[0];
    let triplets_value = node.metadata.get("extracted_triplets").unwrap();
    let triplets: Vec<Triplet> = serde_json::from_value(triplets_value.clone()).unwrap();

    // Should filter out the self-loop (Alice -> is -> Alice)
    assert_eq!(triplets.len(), 1);
    assert_eq!(triplets[0].source.name, "Alice");
    assert_eq!(triplets[0].target.name, "Microsoft");
}

#[tokio::test]
async fn test_deduplication() {
    let mock_client = Arc::new(MockLlmClient::new().with_response(
        "json_extraction",
        r#"[
            {"subject": "Alice", "predicate": "works at", "object": "Microsoft"},
            {"subject": "alice", "predicate": "works at", "object": "microsoft"}
        ]"#,
    ));

    let config = LlmExtractionConfig {
        format: ExtractionFormat::Json,
        enable_deduplication: true,
        ..Default::default()
    };

    let extractor = LlmExtractor::new(mock_client, config).unwrap();

    let documents = vec![Document::new("Test document")];
    let input = TransformInput::Documents(documents);
    let result = extractor.transform(input).await.unwrap();

    let node = &result[0];
    let triplets_value = node.metadata.get("extracted_triplets").unwrap();
    let triplets: Vec<Triplet> = serde_json::from_value(triplets_value.clone()).unwrap();

    // Should deduplicate case-insensitive duplicates
    assert_eq!(triplets.len(), 1);
}

#[tokio::test]
async fn test_fallback_parsing() {
    let mock_client = Arc::new(MockLlmClient::new().with_response(
        "json_extraction",
        "Invalid JSON that should trigger fallback: (Alice, works at, Microsoft)",
    ));

    let config = LlmExtractionConfig {
        format: ExtractionFormat::Json,
        enable_fallback: true,
        ..Default::default()
    };

    let extractor = LlmExtractor::new(mock_client, config).unwrap();

    let documents = vec![Document::new("Test document")];
    let input = TransformInput::Documents(documents);
    let result = extractor.transform(input).await.unwrap();

    let node = &result[0];
    let triplets_value = node.metadata.get("extracted_triplets").unwrap();
    let triplets: Vec<Triplet> = serde_json::from_value(triplets_value.clone()).unwrap();

    // Should fall back to parentheses parsing
    assert_eq!(triplets.len(), 1);
    assert_eq!(triplets[0].source.name, "Alice");
}

// Integration tests with real LLM (requires API key)
#[tokio::test]
#[ignore] // Ignored by default, run with --ignored to test with real API
async fn test_real_llm_extraction() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let llm_client = Siumai::builder()
        .openai()
        .api_key(&api_key)
        .model("gpt-4o-mini")
        .build()
        .await
        .unwrap();

    let config = LlmExtractionConfig {
        format: ExtractionFormat::Json,
        max_triplets_per_chunk: 10,
        temperature: 0.1,
        enable_validation: true,
        enable_deduplication: true,
        ..Default::default()
    };

    let extractor = LlmExtractor::new(Arc::new(llm_client), config).unwrap();

    let documents = vec![
        Document::new("Alice is a software engineer at Microsoft. She works on Azure cloud services and lives in Seattle, Washington."),
        Document::new("Bob is Alice's colleague at Microsoft. He leads the Azure AI team and previously worked at Google in Mountain View."),
    ];

    let input = TransformInput::Documents(documents);
    let result = extractor.transform(input).await.unwrap();

    assert_eq!(result.len(), 2);

    // Check that both documents have extracted triplets
    for node in &result {
        assert!(node.metadata.contains_key("extracted_triplets"));
        let triplets_value = node.metadata.get("extracted_triplets").unwrap();
        let triplets: Vec<Triplet> = serde_json::from_value(triplets_value.clone()).unwrap();
        assert!(!triplets.is_empty(), "Should extract at least one triplet");

        // Print extracted triplets for manual verification
        println!("Extracted triplets from: {}", &node.content[..50]);
        for triplet in &triplets {
            println!(
                "  ({}, {}, {})",
                triplet.source.name, triplet.relation.label, triplet.target.name
            );
        }
    }
}

#[tokio::test]
#[ignore] // Ignored by default
async fn test_real_llm_with_complex_text() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let llm_client = Siumai::builder()
        .openai()
        .api_key(&api_key)
        .model("gpt-4o-mini")
        .build()
        .await
        .unwrap();

    let config = LlmExtractionConfig {
        format: ExtractionFormat::Json,
        max_triplets_per_chunk: 15,
        temperature: 0.1,
        enable_validation: true,
        enable_deduplication: true,
        show_progress: true,
        ..Default::default()
    };

    let extractor = LlmExtractor::new(Arc::new(llm_client), config).unwrap();

    let documents = vec![
        Document::new("Microsoft Corporation is an American multinational technology corporation headquartered in Redmond, Washington. It was founded by Bill Gates and Paul Allen on April 4, 1975. Microsoft develops, manufactures, licenses, supports, and sells computer software, consumer electronics, personal computers, and related services."),
    ];

    let input = TransformInput::Documents(documents);
    let result = extractor.transform(input).await.unwrap();

    assert_eq!(result.len(), 1);

    let node = &result[0];
    let triplets_value = node.metadata.get("extracted_triplets").unwrap();
    let triplets: Vec<Triplet> = serde_json::from_value(triplets_value.clone()).unwrap();

    println!("Extracted {} triplets from complex text:", triplets.len());
    for (i, triplet) in triplets.iter().enumerate() {
        println!(
            "  {}. ({}, {}, {})",
            i + 1,
            triplet.source.name,
            triplet.relation.label,
            triplet.target.name
        );
    }

    // Should extract multiple meaningful relationships
    assert!(
        triplets.len() >= 5,
        "Should extract at least 5 triplets from complex text"
    );

    // Check for expected entities
    let subjects: Vec<&str> = triplets.iter().map(|t| t.source.name.as_str()).collect();
    assert!(
        subjects.iter().any(|&s| s.contains("Microsoft")),
        "Should extract Microsoft as subject"
    );
    assert!(
        subjects
            .iter()
            .any(|&s| s.contains("Bill Gates") || s.contains("Gates")),
        "Should extract Bill Gates as subject"
    );
}
