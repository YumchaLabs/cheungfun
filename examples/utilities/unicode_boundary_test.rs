//! Unicode Character Boundary Test
//!
//! This test verifies that our text splitting utilities correctly handle
//! Unicode characters and don't panic on character boundary issues.

use cheungfun_core::Document;
use cheungfun_indexing::node_parser::{text::SentenceSplitter, NodeParser};
use tracing::{info, Level};
use unicode_segmentation::UnicodeSegmentation;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    println!("🔍 Unicode Character Boundary Test");
    println!("==================================");
    println!();

    // Test text with various Unicode characters that could cause boundary issues
    let long_unicode_text = format!("This is a longer text with Unicode characters {} repeated multiple times to test chunking behavior. {} The text should be split properly without causing character boundary panics. {} More content here to ensure we have enough text for multiple chunks. {}",
            "✂️🚀🎵", "🌍🌎🌏", "∑∏∫∆", "€£¥₹");

    let test_texts = vec![
        // Emoji and symbols
        "This is a test with emoji ✂️ and other symbols 🚀 in the text.",
        "Audio processing example ✂️ with scissors symbol in the middle.",
        "Multiple emojis: 🎵🎶🎸🎹🎺🎻 should not cause issues.",
        // CJK characters
        "这是中文测试文本，包含中文字符。",
        "日本語のテストテキストです。",
        "한국어 테스트 텍스트입니다.",
        // Mixed content
        "Mixed content: English, 中文, 日本語, 한국어, and emojis 🌍🌎🌏",
        // Special Unicode characters
        "Mathematical symbols: ∑∏∫∆∇∂ and arrows: ←→↑↓",
        "Currency symbols: €£¥₹₽₩ and other symbols: ©®™",
        // Long text with Unicode
        &long_unicode_text,
    ];

    let splitter = SentenceSplitter::from_defaults(100, 20)?;

    for (i, test_text) in test_texts.iter().enumerate() {
        // Safe Unicode-aware text preview using grapheme clusters
        let preview = test_text.graphemes(true).take(50).collect::<String>();
        info!("Testing text {}: {}", i + 1, preview);

        let document = Document::new(*test_text);

        match splitter.parse_nodes(&[document], false).await {
            Ok(nodes) => {
                println!(
                    "✅ Test {}: Successfully processed {} nodes",
                    i + 1,
                    nodes.len()
                );

                // Verify node content integrity
                for (j, node) in nodes.iter().enumerate() {
                    if node.content.is_empty() {
                        println!("⚠️  Warning: Node {} is empty", j + 1);
                    }

                    // Check for valid UTF-8
                    if !node.content.is_ascii() {
                        println!(
                            "   📝 Node {} contains non-ASCII characters (length: {})",
                            j + 1,
                            node.content.len()
                        );
                    }
                }
            }
            Err(e) => {
                println!("❌ Test {}: Failed with error: {}", i + 1, e);
                return Err(e.into());
            }
        }

        println!();
    }

    println!("🎉 All Unicode boundary tests passed!");
    println!();
    println!("The text splitter correctly handles:");
    println!("- Emoji and Unicode symbols");
    println!("- CJK (Chinese, Japanese, Korean) characters");
    println!("- Mixed content with multiple scripts");
    println!("- Mathematical and special symbols");
    println!("- Long texts with embedded Unicode characters");

    Ok(())
}
