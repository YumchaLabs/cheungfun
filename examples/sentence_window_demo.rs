//! Sentence Window Node Parser Demo
//!
//! This example demonstrates the SentenceWindowNodeParser functionality,
//! showing how it creates nodes for individual sentences while preserving
//! surrounding context in metadata.

use cheungfun_core::{Document, Result as CoreResult};
use cheungfun_indexing::node_parser::{text::SentenceWindowNodeParser, NodeParser, TextSplitter};
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> CoreResult<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("🪟 Sentence Window Node Parser Demo");
    info!("===================================");

    // Create sentence window parsers with different configurations
    info!("Creating sentence window parsers with different configurations...");

    let parser_small_window = SentenceWindowNodeParser::new()
        .with_window_size(1);
    
    let parser_large_window = SentenceWindowNodeParser::new()
        .with_window_size(3);

    let parser_custom_keys = SentenceWindowNodeParser::new()
        .with_window_size(2)
        .with_window_metadata_key("context")
        .with_original_text_metadata_key("sentence");

    // Test text with clear sentence boundaries
    let test_text = r#"
        Machine learning is transforming industries worldwide. Companies are investing heavily in AI research and development.
        Deep learning models require massive amounts of data to train effectively. Neural networks can recognize complex patterns that traditional algorithms miss.
        
        The weather forecast predicts sunny skies for the weekend. Many families are planning outdoor activities and picnics.
        Local parks expect increased visitor numbers during the pleasant weather. Rangers are preparing for busy hiking trails.
        
        Quantum computing represents the next frontier in computational power. These systems use quantum mechanical principles to process information.
        Current quantum computers are still experimental and require extremely cold temperatures. However, they show promise for solving complex optimization problems.
    "#;

    info!("\n📝 Test Text:");
    info!("{}", test_text.trim());

    // Test small window parser
    info!("\n🔍 Small Window Parser (window_size=1):");
    let document = Document::new(test_text);
    let nodes_small = <SentenceWindowNodeParser as NodeParser>::parse_nodes(&parser_small_window, &[document.clone()], false).await?;
    
    info!("Generated {} nodes:", nodes_small.len());
    for (i, node) in nodes_small.iter().take(3).enumerate() {
        info!("Node {}: '{}'", i + 1, node.content.trim());
        if let Some(window) = node.metadata.get("window") {
            info!("  Window: '{}'", window.as_str().unwrap().trim());
        }
        if let Some(index) = node.metadata.get("sentence_index") {
            info!("  Sentence Index: {}", index.as_u64().unwrap());
        }
        info!("");
    }
    if nodes_small.len() > 3 {
        info!("... and {} more nodes", nodes_small.len() - 3);
    }

    // Test large window parser
    info!("\n🔍 Large Window Parser (window_size=3):");
    let nodes_large = <SentenceWindowNodeParser as NodeParser>::parse_nodes(&parser_large_window, &[document.clone()], false).await?;
    
    info!("Generated {} nodes:", nodes_large.len());
    for (i, node) in nodes_large.iter().take(2).enumerate() {
        info!("Node {}: '{}'", i + 1, node.content.trim());
        if let Some(window) = node.metadata.get("window") {
            let window_text = window.as_str().unwrap();
            let truncated = if window_text.len() > 100 {
                format!("{}...", &window_text[..100])
            } else {
                window_text.to_string()
            };
            info!("  Window: '{}'", truncated.trim());
        }
        info!("");
    }
    if nodes_large.len() > 2 {
        info!("... and {} more nodes", nodes_large.len() - 2);
    }

    // Test custom metadata keys
    info!("\n🔍 Custom Metadata Keys Parser (window_size=2, custom keys):");
    let nodes_custom = <SentenceWindowNodeParser as NodeParser>::parse_nodes(&parser_custom_keys, &[document.clone()], false).await?;
    
    info!("Generated {} nodes:", nodes_custom.len());
    for (i, node) in nodes_custom.iter().take(2).enumerate() {
        info!("Node {}: '{}'", i + 1, node.content.trim());
        if let Some(context) = node.metadata.get("context") {
            let context_text = context.as_str().unwrap();
            let truncated = if context_text.len() > 80 {
                format!("{}...", &context_text[..80])
            } else {
                context_text.to_string()
            };
            info!("  Context: '{}'", truncated.trim());
        }
        if let Some(sentence) = node.metadata.get("sentence") {
            info!("  Original Sentence: '{}'", sentence.as_str().unwrap().trim());
        }
        info!("");
    }

    // Test TextSplitter interface
    info!("\n📄 Testing TextSplitter Interface:");
    let sentences = parser_small_window.split_text(test_text)?;
    info!("Split into {} sentences:", sentences.len());
    for (i, sentence) in sentences.iter().take(5).enumerate() {
        info!("  {}: '{}'", i + 1, sentence.trim());
    }
    if sentences.len() > 5 {
        info!("  ... and {} more sentences", sentences.len() - 5);
    }

    // Compare window sizes
    info!("\n📊 Window Size Comparison:");
    info!("Small window (size=1): {} nodes", nodes_small.len());
    info!("Large window (size=3): {} nodes", nodes_large.len());
    info!("Custom keys (size=2): {} nodes", nodes_custom.len());
    info!("All parsers generate the same number of nodes (one per sentence)");

    // Demonstrate metadata differences
    info!("\n🔍 Metadata Comparison for First Node:");
    if let Some(node_small) = nodes_small.first() {
        info!("Small window metadata keys: {:?}", node_small.metadata.keys().collect::<Vec<_>>());
    }
    if let Some(node_custom) = nodes_custom.first() {
        info!("Custom keys metadata keys: {:?}", node_custom.metadata.keys().collect::<Vec<_>>());
    }

    // Performance comparison
    info!("\n⏱️  Performance Comparison:");
    let start = std::time::Instant::now();
    let _ = <SentenceWindowNodeParser as NodeParser>::parse_nodes(&parser_small_window, &[document.clone()], false).await?;
    let window_time = start.elapsed();

    let sentence_splitter = cheungfun_indexing::node_parser::text::SentenceSplitter::from_defaults(1000, 200)?;
    let start = std::time::Instant::now();
    let _ = <cheungfun_indexing::node_parser::text::SentenceSplitter as NodeParser>::parse_nodes(&sentence_splitter, &[document], false).await?;
    let sentence_time = start.elapsed();

    info!("Sentence Window Parser: {:?}", window_time);
    info!("Regular Sentence Parser: {:?}", sentence_time);
    info!("Window parser is {:.2}x slower (due to metadata processing)", 
          window_time.as_secs_f64() / sentence_time.as_secs_f64());

    // Test edge cases
    info!("\n🧪 Testing Edge Cases:");

    // Empty text
    let empty_doc = Document::new("");
    let empty_nodes = <SentenceWindowNodeParser as NodeParser>::parse_nodes(&parser_small_window, &[empty_doc], false).await?;
    info!("Empty text: {} nodes", empty_nodes.len());

    // Single sentence
    let single_doc = Document::new("This is a single sentence.");
    let single_nodes = <SentenceWindowNodeParser as NodeParser>::parse_nodes(&parser_small_window, &[single_doc], false).await?;
    info!("Single sentence: {} nodes", single_nodes.len());
    if let Some(node) = single_nodes.first() {
        if let Some(window) = node.metadata.get("window") {
            info!("  Window content: '{}'", window.as_str().unwrap());
        }
    }

    // Very short sentences
    let short_doc = Document::new("One. Two. Three. Four.");
    let short_nodes = <SentenceWindowNodeParser as NodeParser>::parse_nodes(&parser_large_window, &[short_doc], false).await?;
    info!("Short sentences with large window: {} nodes", short_nodes.len());

    info!("\n✅ Sentence Window Node Parser Demo completed successfully!");
    info!("\n💡 Key Features Demonstrated:");
    info!("   • Individual sentence nodes with surrounding context");
    info!("   • Configurable window sizes for different context needs");
    info!("   • Custom metadata keys for flexible integration");
    info!("   • Precise sentence-level retrieval with rich context");
    info!("   • Seamless integration with existing TextSplitter interface");

    info!("\n🎯 Use Cases:");
    info!("   • Question-answering systems requiring exact sentence matching");
    info!("   • Fine-grained document analysis and annotation");
    info!("   • Context-aware text processing pipelines");
    info!("   • Precise citation and reference systems");

    Ok(())
}

/// Demonstrate advanced sentence window features
#[allow(dead_code)]
async fn demonstrate_advanced_features() -> CoreResult<()> {
    info!("\n🚀 Advanced Sentence Window Features:");

    // This would demonstrate:
    // - Integration with different sentence splitting algorithms
    // - Custom window content formatting
    // - Relationship building between adjacent nodes
    // - Batch processing optimizations

    Ok(())
}

/// Compare different parsing strategies
#[allow(dead_code)]
async fn compare_parsing_strategies(_text: &str) -> CoreResult<()> {
    info!("\n📈 Parsing Strategy Comparison:");

    // This would compare:
    // - Sentence window parsing
    // - Regular sentence parsing
    // - Token-based parsing
    // - Semantic parsing

    // Metrics to compare:
    // - Number of nodes generated
    // - Context preservation quality
    // - Processing time
    // - Memory usage
    // - Retrieval precision

    Ok(())
}
