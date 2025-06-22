//! Unit tests for individual components in cheungfun-query.

use std::collections::HashMap;
use std::time::Duration;

use cheungfun_core::types::{GeneratedResponse, Query, SearchMode};
use cheungfun_query::utils::{
    CitationFormat, QueryCache, QueryOptimizer, QueryOptimizerConfig, ResponsePostProcessor,
    ResponsePostProcessorConfig, query_utils, response_utils,
};

#[test]
fn test_query_optimizer_default() {
    let optimizer = QueryOptimizer::new();
    let query = Query::new("test query");

    // With default config, should return unchanged query
    let optimized = optimizer.optimize_query(&query).unwrap();
    assert_eq!(optimized.text, query.text);
}

#[test]
fn test_query_optimizer_with_config() {
    let config = QueryOptimizerConfig {
        enable_spell_correction: true,
        enable_query_expansion: true,
        enable_stop_word_removal: true,
        enable_stemming: true,
        max_expanded_terms: 5,
    };

    let optimizer = QueryOptimizer::with_config(config);
    let query = Query::new("test query");

    // Should not fail even though implementations are placeholders
    let optimized = optimizer.optimize_query(&query).unwrap();
    assert_eq!(optimized.text, query.text); // Placeholder returns original
}

#[test]
fn test_response_post_processor_default() {
    let processor = ResponsePostProcessor::new();
    let response = GeneratedResponse {
        content: "Test response content".to_string(),
        source_nodes: vec![],
        metadata: HashMap::new(),
        usage: None,
    };

    let processed = processor.process_response(&response, &[]).unwrap();
    assert!(!processed.content.is_empty());
}

#[test]
fn test_response_post_processor_with_citations() {
    let config = ResponsePostProcessorConfig {
        enable_citation_formatting: true,
        citation_format: CitationFormat::Numbered,
        ..Default::default()
    };

    let processor = ResponsePostProcessor::with_config(config);
    let response = GeneratedResponse {
        content: "Test response".to_string(),
        source_nodes: vec![],
        metadata: HashMap::new(),
        usage: None,
    };

    let processed = processor.process_response(&response, &[]).unwrap();
    assert!(processed.content.contains("Test response"));
}

#[test]
fn test_query_cache_basic_operations() {
    let cache = QueryCache::new(Duration::from_secs(60));

    let response = GeneratedResponse {
        content: "Cached response".to_string(),
        source_nodes: vec![],
        metadata: HashMap::new(),
        usage: None,
    };

    // Test cache miss
    assert!(cache.get("test").is_none());

    // Test cache put and hit
    cache.put("test", response.clone());
    let cached = cache.get("test");
    assert!(cached.is_some());
    assert_eq!(cached.unwrap().content, "Cached response");

    // Test cache stats
    let stats = cache.stats();
    assert_eq!(stats.active_entries, 1);
    assert_eq!(stats.total_entries, 1);
    assert_eq!(stats.expired_entries, 0);
}

#[test]
fn test_query_cache_ttl() {
    let cache = QueryCache::new(Duration::from_millis(1));

    let response = GeneratedResponse {
        content: "Short-lived response".to_string(),
        source_nodes: vec![],
        metadata: HashMap::new(),
        usage: None,
    };

    cache.put("test", response);

    // Should be available immediately
    assert!(cache.get("test").is_some());

    // Wait for expiration
    std::thread::sleep(Duration::from_millis(10));

    // Should be expired now
    assert!(cache.get("test").is_none());
}

#[test]
fn test_query_cache_cleanup() {
    let cache = QueryCache::new(Duration::from_millis(1));

    let response = GeneratedResponse {
        content: "Test".to_string(),
        source_nodes: vec![],
        metadata: HashMap::new(),
        usage: None,
    };

    cache.put("test1", response.clone());
    cache.put("test2", response.clone());

    // Wait for expiration
    std::thread::sleep(Duration::from_millis(10));

    // Before cleanup
    let stats_before = cache.stats();
    assert_eq!(stats_before.total_entries, 2);
    assert_eq!(stats_before.expired_entries, 2);

    // After cleanup
    cache.cleanup_expired();
    let stats_after = cache.stats();
    assert_eq!(stats_after.total_entries, 0);
    assert_eq!(stats_after.expired_entries, 0);
}

#[test]
fn test_query_utils_extract_keywords() {
    let keywords = query_utils::extract_keywords("What is machine learning and AI?");

    assert!(keywords.contains(&"machine".to_string()));
    assert!(keywords.contains(&"learning".to_string()));
    assert!(!keywords.contains(&"is".to_string())); // Too short
    // Note: "and" might be included if it's 3+ characters, which it is
    // So we test for a definitely short word instead
    assert!(!keywords.contains(&"is".to_string())); // Too short
}

#[test]
fn test_query_utils_text_similarity() {
    // Identical texts
    let sim1 = query_utils::calculate_text_similarity("hello world", "hello world");
    assert_eq!(sim1, 1.0);

    // Completely different texts
    let sim2 = query_utils::calculate_text_similarity("hello world", "foo bar");
    assert_eq!(sim2, 0.0);

    // Partial overlap
    let sim3 = query_utils::calculate_text_similarity("hello world", "hello universe");
    assert!(sim3 > 0.0 && sim3 < 1.0);
}

#[test]
fn test_query_utils_truncate_text() {
    let text = "This is a very long sentence that should be truncated properly at word boundaries";

    // Test normal truncation
    let truncated = query_utils::truncate_text(text, 20);
    assert!(truncated.len() <= 23); // 20 + "..."
    assert!(truncated.ends_with("..."));

    // Test no truncation needed
    let short_text = "Short";
    let not_truncated = query_utils::truncate_text(short_text, 20);
    assert_eq!(not_truncated, "Short");
}

#[test]
fn test_response_utils_extract_main_points() {
    let content = "This is the first point. This is a very detailed second point that explains something important. Short. This is the third point with good length.";

    let points = response_utils::extract_main_points(content);

    // Should filter out very short and very long sentences
    assert!(points.len() >= 1);
    for point in &points {
        assert!(point.len() >= 20);
        assert!(point.len() <= 200);
    }
}

#[test]
fn test_response_utils_quality_score() {
    let response = GeneratedResponse {
        content: "This is a good quality response with appropriate length and content.".to_string(),
        source_nodes: vec![],
        metadata: HashMap::new(),
        usage: None,
    };

    let score = response_utils::calculate_response_quality(&response, &[]);
    assert!(score >= 0.0 && score <= 1.0);
}

#[test]
fn test_search_mode_helpers() {
    let vector_mode = SearchMode::Vector;
    assert!(vector_mode.is_vector());
    assert!(!vector_mode.is_keyword());
    assert!(!vector_mode.is_hybrid());

    let keyword_mode = SearchMode::Keyword;
    assert!(!keyword_mode.is_vector());
    assert!(keyword_mode.is_keyword());
    assert!(!keyword_mode.is_hybrid());

    let hybrid_mode = SearchMode::hybrid(0.7);
    assert!(!hybrid_mode.is_vector());
    assert!(!hybrid_mode.is_keyword());
    assert!(hybrid_mode.is_hybrid());
    if let SearchMode::Hybrid { alpha } = hybrid_mode {
        assert_eq!(alpha, 0.7);
    } else {
        panic!("Expected hybrid mode");
    }
}

#[test]
fn test_query_builder_pattern() {
    let query = Query::new("test query")
        .with_top_k(15)
        .with_search_mode(SearchMode::hybrid(0.8))
        .with_similarity_threshold(0.75)
        .with_filter("category", "science");

    assert_eq!(query.text, "test query");
    assert_eq!(query.top_k, 15);
    assert_eq!(query.similarity_threshold, Some(0.75));
    if let SearchMode::Hybrid { alpha } = query.search_mode {
        assert_eq!(alpha, 0.8);
    } else {
        panic!("Expected hybrid mode");
    }
    assert_eq!(
        query.filters.get("category"),
        Some(&serde_json::Value::String("science".to_string()))
    );
}
