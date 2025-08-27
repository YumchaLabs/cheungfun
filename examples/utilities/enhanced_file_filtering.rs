//! Enhanced file filtering example for cheungfun-indexing.
//!
//! This example demonstrates the new enhanced file filtering capabilities,
//! including gitignore support, glob patterns, and various filtering options.

use cheungfun_core::traits::Loader;
use cheungfun_indexing::loaders::{DirectoryLoader, FilterConfig, LoaderConfig};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🔍 Enhanced File Filtering Example");
    println!("==================================\n");

    // Example 1: Basic enhanced filtering with gitignore support
    println!("📁 Example 1: Basic enhanced filtering with gitignore");
    let basic_filter = FilterConfig::new()
        .with_respect_gitignore(true)
        .with_exclude_hidden(true);

    let basic_config = LoaderConfig::new()
        .with_filter_config(basic_filter)
        .with_max_depth(3);

    let loader = DirectoryLoader::with_config(".", basic_config)?;
    let documents = loader.load().await?;
    println!(
        "✅ Loaded {} documents with basic filtering\n",
        documents.len()
    );

    // Example 2: Source code only filtering
    println!("💻 Example 2: Source code only filtering");
    let source_config = LoaderConfig::new().with_source_code_filtering();

    let loader = DirectoryLoader::with_config(".", source_config)?;
    let documents = loader.load().await?;
    println!("✅ Loaded {} source code documents\n", documents.len());

    // Example 3: Custom glob patterns
    println!("🎯 Example 3: Custom glob patterns");
    let custom_filter = FilterConfig::new()
        .with_respect_gitignore(true)
        .with_exclude_patterns(vec![
            "target/**".to_string(),
            "*.log".to_string(),
            "**/.cache/**".to_string(),
            "node_modules/**".to_string(),
        ])
        .with_include_patterns(vec![
            "src/**/*.rs".to_string(),
            "examples/**/*.rs".to_string(),
            "*.md".to_string(),
            "*.toml".to_string(),
        ]);

    let custom_config = LoaderConfig::new().with_filter_config(custom_filter);

    let loader = DirectoryLoader::with_config(".", custom_config)?;
    let documents = loader.load().await?;
    println!(
        "✅ Loaded {} documents with custom patterns\n",
        documents.len()
    );

    // Example 4: Size-based filtering
    println!("📏 Example 4: Size-based filtering");
    let size_filter = FilterConfig::new()
        .with_max_file_size(1024 * 1024) // 1MB max
        .with_min_file_size(10) // 10 bytes min
        .with_exclude_empty(true);

    let size_config = LoaderConfig::new().with_filter_config(size_filter);

    let loader = DirectoryLoader::with_config(".", size_config)?;
    let documents = loader.load().await?;
    println!(
        "✅ Loaded {} documents with size filtering\n",
        documents.len()
    );

    // Example 5: Text files only with custom ignore files
    println!("📝 Example 5: Text files with custom ignore files");
    let text_filter = FilterConfig::text_files_only().with_ignore_files(vec![
        PathBuf::from(".dockerignore"),
        PathBuf::from(".eslintignore"),
    ]);

    let text_config = LoaderConfig::new().with_filter_config(text_filter);

    let loader = DirectoryLoader::with_config(".", text_config)?;
    let documents = loader.load().await?;
    println!("✅ Loaded {} text documents\n", documents.len());

    // Example 6: Case-insensitive filtering
    println!("🔤 Example 6: Case-insensitive filtering");
    let case_filter = FilterConfig::new()
        .with_exclude_patterns(vec!["*.LOG".to_string(), "*.TMP".to_string()])
        .with_case_sensitive(false); // Case-insensitive matching

    let case_config = LoaderConfig::new().with_filter_config(case_filter);

    let loader = DirectoryLoader::with_config(".", case_config)?;
    let documents = loader.load().await?;
    println!(
        "✅ Loaded {} documents with case-insensitive filtering\n",
        documents.len()
    );

    // Example 7: Complex filtering scenario
    println!("🎛️ Example 7: Complex filtering scenario");
    let complex_filter = FilterConfig::new()
        .with_respect_gitignore(true)
        .with_exclude_patterns(vec![
            // Build artifacts
            "target/**".to_string(),
            "build/**".to_string(),
            "dist/**".to_string(),
            // Dependencies
            "node_modules/**".to_string(),
            "vendor/**".to_string(),
            // Temporary files
            "**/*.tmp".to_string(),
            "**/*.temp".to_string(),
            "**/*~".to_string(),
            // IDE files
            ".vscode/**".to_string(),
            ".idea/**".to_string(),
            // OS files
            "**/.DS_Store".to_string(),
            "**/Thumbs.db".to_string(),
        ])
        .with_include_patterns(vec![
            // Source code
            "**/*.rs".to_string(),
            "**/*.py".to_string(),
            "**/*.js".to_string(),
            "**/*.ts".to_string(),
            // Documentation
            "**/*.md".to_string(),
            "**/*.rst".to_string(),
            // Configuration
            "**/*.toml".to_string(),
            "**/*.yaml".to_string(),
            "**/*.yml".to_string(),
            "**/*.json".to_string(),
        ])
        .with_exclude_hidden(true)
        .with_exclude_empty(true)
        .with_max_file_size(10 * 1024 * 1024) // 10MB max
        .with_case_sensitive(false);

    let complex_config = LoaderConfig::new()
        .with_filter_config(complex_filter)
        .with_max_depth(5)
        .with_continue_on_error(true);

    let loader = DirectoryLoader::with_config(".", complex_config)?;
    let documents = loader.load().await?;
    println!(
        "✅ Loaded {} documents with complex filtering\n",
        documents.len()
    );

    println!("🎉 All examples completed successfully!");
    println!("\n📊 Summary:");
    println!("- Enhanced filtering supports gitignore patterns");
    println!("- Custom glob patterns for include/exclude");
    println!("- File size and content-based filtering");
    println!("- Case-sensitive/insensitive matching");
    println!("- Multiple ignore file support");
    println!("- Predefined configurations for common use cases");

    Ok(())
}
