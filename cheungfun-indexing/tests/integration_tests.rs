//! Integration tests for the cheungfun-indexing crate.

use cheungfun_core::{ChunkInfo, Document, Node};
use cheungfun_indexing::prelude::*;
use std::fs;
use tempfile::TempDir;

/// Create a temporary directory with test files.
fn create_test_directory() -> (TempDir, Vec<String>) {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let mut file_paths = Vec::new();

    // Create test files
    let test_files = vec![
        (
            "test1.txt",
            "This is the first test document.\nIt contains multiple lines.",
        ),
        (
            "test2.md",
            "# Test Document\n\nThis is a **markdown** document with some content.",
        ),
        (
            "test3.txt",
            "Another test document with different content.\nThis one is also multi-line.",
        ),
    ];

    for (filename, content) in test_files {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).expect("Failed to write test file");
        file_paths.push(file_path.display().to_string());
    }

    (temp_dir, file_paths)
}

#[tokio::test]
async fn test_file_loader() {
    let (_temp_dir, file_paths) = create_test_directory();

    // Test loading a single file
    let loader = FileLoader::new(&file_paths[0]).expect("Failed to create file loader");
    let documents = loader.load().await.expect("Failed to load documents");

    assert_eq!(documents.len(), 1);
    assert!(documents[0].content.contains("first test document"));
    assert!(documents[0].metadata.contains_key("source"));
    assert!(documents[0].metadata.contains_key("filename"));
}

#[tokio::test]
async fn test_directory_loader() {
    let (temp_dir, _file_paths) = create_test_directory();

    // Test loading all files from directory
    let loader = DirectoryLoader::new(temp_dir.path()).expect("Failed to create directory loader");
    let documents = loader.load().await.expect("Failed to load documents");

    assert_eq!(documents.len(), 3);

    // Check that all documents have proper metadata
    for doc in &documents {
        assert!(doc.metadata.contains_key("source"));
        assert!(doc.metadata.contains_key("filename"));
        assert!(!doc.content.is_empty());
    }
}

#[tokio::test]
async fn test_text_splitter() {
    let long_text =
        "This is a very long document that needs to be split into smaller chunks. ".repeat(50);
    let document = Document::new(long_text);

    let splitter = TextSplitter::new(500, 100);
    let nodes = splitter
        .transform(document)
        .await
        .expect("Failed to split text");

    assert!(nodes.len() > 1);

    // Check that nodes have proper chunk information
    for (i, node) in nodes.iter().enumerate() {
        assert_eq!(node.chunk_info.chunk_index, i);
        assert!(node.content.len() <= 600); // Should be around chunk_size + some tolerance
        assert!(node.metadata.contains_key("chunk_index"));
        assert!(node.metadata.contains_key("chunk_size"));
    }
}

#[tokio::test]
async fn test_metadata_extractor() {
    let content = "# Test Document\n\nThis is a test document with some content.\nIt has multiple paragraphs and sentences.";
    let node = Node::new(
        content.to_string(),
        uuid::Uuid::new_v4(),
        ChunkInfo {
            start_offset: 0,
            end_offset: content.len(),
            chunk_index: 0,
        },
    );

    let extractor = MetadataExtractor::new();
    let enriched_node = extractor
        .transform_node(node)
        .await
        .expect("Failed to extract metadata");

    // Check that metadata was extracted
    assert!(enriched_node.metadata.contains_key("character_count"));
    assert!(enriched_node.metadata.contains_key("word_count"));
    assert!(enriched_node.metadata.contains_key("line_count"));
    assert!(enriched_node.metadata.contains_key("metadata_extracted_at"));
}

#[tokio::test]
async fn test_pipeline_builder() {
    let (temp_dir, _file_paths) = create_test_directory();

    // Create a simple pipeline
    let loader = std::sync::Arc::new(
        DirectoryLoader::new(temp_dir.path()).expect("Failed to create directory loader"),
    );
    let transformer = std::sync::Arc::new(TextSplitter::new(200, 50));
    let node_transformer = std::sync::Arc::new(MetadataExtractor::new());

    let pipeline = DefaultIndexingPipeline::builder()
        .with_loader(loader)
        .with_transformer(transformer)
        .with_node_transformer(node_transformer)
        .build()
        .expect("Failed to build pipeline");

    // Validate the pipeline
    pipeline.validate().expect("Pipeline validation failed");
}

#[tokio::test]
async fn test_loader_config() {
    let config = cheungfun_indexing::loaders::LoaderConfig::new()
        .with_max_file_size(1024)
        .with_include_extensions(vec!["txt".to_string(), "md".to_string()])
        .with_exclude_extensions(vec!["tmp".to_string()])
        .with_continue_on_error(true);

    assert_eq!(config.max_file_size, Some(1024));
    assert_eq!(
        config.include_extensions,
        Some(vec!["txt".to_string(), "md".to_string()])
    );
    assert!(config.exclude_extensions.contains(&"tmp".to_string()));
    assert!(config.continue_on_error);
}

#[tokio::test]
async fn test_splitter_config() {
    let config = SplitterConfig::new(1000, 200)
        .with_min_chunk_size(100)
        .with_max_chunk_size(2000)
        .with_respect_sentence_boundaries(true)
        .with_keep_separators(false);

    assert_eq!(config.chunk_size, 1000);
    assert_eq!(config.chunk_overlap, 200);
    assert_eq!(config.min_chunk_size, Some(100));
    assert_eq!(config.max_chunk_size, Some(2000));
    assert!(config.respect_sentence_boundaries);
    assert!(!config.keep_separators);
}

#[tokio::test]
async fn test_metadata_config() {
    let config = MetadataConfig::new()
        .with_title_extraction(true)
        .with_statistics(true)
        .with_keyword_extraction(true, 5);

    assert!(config.extract_title);
    assert!(config.extract_statistics);
    assert!(config.extract_keywords);
    assert_eq!(config.max_keywords, 5);
}

#[tokio::test]
async fn test_error_handling() {
    // Test file not found error
    let result = FileLoader::new("nonexistent_file.txt");
    assert!(result.is_err());

    // Test directory not found error
    let result = DirectoryLoader::new("nonexistent_directory");
    assert!(result.is_err());
}

#[tokio::test]
async fn test_utils_functions() {
    use cheungfun_indexing::loaders::utils;
    use std::path::Path;

    // Test content type detection
    assert_eq!(
        utils::detect_content_type(Path::new("test.txt")),
        Some("text/plain".to_string())
    );
    assert_eq!(
        utils::detect_content_type(Path::new("test.md")),
        Some("text/markdown".to_string())
    );
    assert_eq!(
        utils::detect_content_type(Path::new("test.pdf")),
        Some("application/pdf".to_string())
    );
    assert_eq!(utils::detect_content_type(Path::new("test.unknown")), None);

    // Test file type support check
    assert!(utils::is_supported_file_type(Path::new("test.txt")));
    assert!(utils::is_supported_file_type(Path::new("test.pdf")));
    assert!(!utils::is_supported_file_type(Path::new("test.unknown")));
}

#[tokio::test]
async fn test_transformer_utils() {
    use cheungfun_indexing::transformers::utils;

    // Test text cleaning
    let dirty_text = "  This   is   messy    text  \n\n  with   extra   spaces  ";
    let clean_text = utils::clean_text(dirty_text);
    assert_eq!(clean_text, "This is messy text with extra spaces");

    // Test title extraction
    let markdown_text = "# Main Title\n\nThis is the content.";
    let title = utils::extract_title(markdown_text);
    assert_eq!(title, Some("Main Title".to_string()));

    // Test statistics calculation
    let text = "This is a test document with multiple words and sentences.";
    let stats = utils::calculate_statistics(text);
    assert!(stats.contains_key("character_count"));
    assert!(stats.contains_key("word_count"));
    assert!(stats.contains_key("line_count"));
}
