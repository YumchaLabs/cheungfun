//! 调试代码分块实现
//!
//! 这个工具用于分析和对比我们的代码分块策略与 LlamaIndex 的差异

use cheungfun_indexing::{
    loaders::ProgrammingLanguage,
    node_parser::{
        config::{ChunkingStrategy, CodeSplitterConfig},
        text::CodeSplitter,
    },
    parsers::{AstParser, AstParserConfig},
    TextSplitter,
};
use std::fs;
use tracing::Level;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_target(false)
        .init();

    println!("🔍 代码分块策略分析工具");
    println!("========================");

    // 读取示例 C# 文件
    let csharp_file = "UnityProject/Assets/Scripts/Controllers/PlayerController.cs";
    let content = fs::read_to_string(csharp_file)?;

    println!("📄 分析文件: {}", csharp_file);
    println!(
        "📏 文件大小: {} 字符, {} 行",
        content.len(),
        content.lines().count()
    );
    println!();

    // 1. 分析 AST 结构
    analyze_ast_structure(&content).await?;

    // 2. 测试不同的分块策略
    test_chunking_strategies(&content).await?;

    // 3. 对比分块质量
    compare_chunking_quality(&content).await?;

    Ok(())
}

async fn analyze_ast_structure(content: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("🌳 AST 结构分析");
    println!("---------------");

    let ast_config = AstParserConfig {
        extract_functions: true,
        extract_classes: true,
        extract_imports: true,
        extract_comments: true,
        include_function_bodies: true,
        max_depth: Some(10),
    };

    let ast_parser = AstParser::new()?;
    let analysis = ast_parser.parse(content, ProgrammingLanguage::CSharp)?;

    println!("📊 AST 分析结果:");
    println!("  - 类数量: {}", analysis.classes.len());
    println!("  - 函数数量: {}", analysis.functions.len());
    println!("  - 导入数量: {}", analysis.imports.len());
    println!("  - 注释数量: {}", analysis.comments.len());

    println!("\n🏗️ 类结构:");
    for (i, class) in analysis.classes.iter().enumerate() {
        println!(
            "  {}. {} (行 {}-{})",
            i + 1,
            class.name,
            class.start_line,
            class.end_line
        );
        println!("     类型: {:?}", class.kind);
        println!("     可见性: {:?}", class.visibility);
        if let Some(doc) = &class.documentation {
            println!("     文档: {}", doc.lines().next().unwrap_or(""));
        }
    }

    println!("\n🔧 函数结构:");
    for (i, func) in analysis.functions.iter().take(10).enumerate() {
        println!(
            "  {}. {} (行 {}-{})",
            i + 1,
            func.name,
            func.start_line,
            func.end_line
        );
        println!("     签名: {}", func.signature);
        println!("     返回类型: {:?}", func.return_type);
        println!("     可见性: {:?}", func.visibility);
        println!("     异步: {}", func.is_async);
    }

    if analysis.functions.len() > 10 {
        println!("  ... 还有 {} 个函数", analysis.functions.len() - 10);
    }

    println!();
    Ok(())
}

async fn test_chunking_strategies(content: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 测试不同分块策略");
    println!("------------------");

    // 使用新的预设配置系统
    let strategies = vec![
        ("🚀 最优策略", ChunkingStrategy::Optimal),
        ("🔍 精细分析", ChunkingStrategy::Fine),
        ("⚖️ 平衡策略", ChunkingStrategy::Balanced),
        ("📊 粗粒度", ChunkingStrategy::Coarse),
        ("🔬 最小块", ChunkingStrategy::Minimal),
        ("🏢 企业级", ChunkingStrategy::Enterprise),
    ];

    for (name, strategy) in strategies {
        println!("\n🧪 测试 {} - {}", name, strategy.description());

        let (chunk_lines, overlap, max_chars) = strategy.params();
        println!(
            "   参数: {} 行/块, {} 行重叠, {} 字符上限",
            chunk_lines, overlap, max_chars
        );

        // 使用预设配置创建 AST 增强分块器
        let ast_splitter = CodeSplitter::with_strategy(ProgrammingLanguage::CSharp, strategy)?;
        let ast_chunks = ast_splitter.split_text(content)?;

        // 基础行分块（禁用 AST）
        let basic_config = CodeSplitterConfig::with_strategy(ProgrammingLanguage::CSharp, strategy)
            .with_ast_splitting(false)
            .with_respect_function_boundaries(false)
            .with_respect_class_boundaries(false);

        let basic_splitter = CodeSplitter::new(basic_config)?;
        let basic_chunks = basic_splitter.split_text(content)?;

        println!("   AST 增强: {} 块", ast_chunks.len());
        println!("   基础分块: {} 块", basic_chunks.len());

        // 显示最优策略的详细分块内容（仅对第一个策略）
        if name == "🚀 最优策略" {
            println!("\n📋 最优策略详细分块内容:");
            println!("================================");
            for (i, chunk) in ast_chunks.iter().enumerate().take(10) {
                // 限制最多显示10个块
                let lines = chunk.lines().collect::<Vec<_>>();

                println!("\n🔸 块 {} :", i + 1);
                println!("   长度: {} 字符, {} 行", chunk.len(), lines.len());
                println!("   内容预览:");

                // 只显示前3行
                for (j, line) in lines.iter().take(3).enumerate() {
                    println!(
                        "   {:3}: {}",
                        j + 1,
                        line.chars().take(80).collect::<String>()
                    );
                }

                if lines.len() > 3 {
                    println!("   ... ({} 行省略)", lines.len() - 3);
                }

                println!("   ----------------------------------------");
            }

            if ast_chunks.len() > 10 {
                println!("\n   ... 还有 {} 个块未显示", ast_chunks.len() - 10);
            }
        }

        // 分析块的质量
        analyze_chunk_quality(&ast_chunks, "AST 增强");
        analyze_chunk_quality(&basic_chunks, "基础分块");
    }

    Ok(())
}

fn analyze_chunk_quality(chunks: &[String], strategy_name: &str) {
    let total_chars: usize = chunks.iter().map(|c| c.len()).sum();
    let avg_chars = if chunks.is_empty() {
        0
    } else {
        total_chars / chunks.len()
    };
    let avg_lines: f64 = if chunks.is_empty() {
        0.0
    } else {
        chunks.iter().map(|c| c.lines().count()).sum::<usize>() as f64 / chunks.len() as f64
    };

    // 分析代码结构完整性
    let complete_functions = chunks
        .iter()
        .filter(|chunk| {
            let open_braces = chunk.matches('{').count();
            let close_braces = chunk.matches('}').count();
            open_braces == close_braces && chunk.contains("(") && chunk.contains(")")
        })
        .count();

    let has_class_definition = chunks
        .iter()
        .filter(|chunk| chunk.contains("class ") || chunk.contains("public class"))
        .count();

    println!("     {} 质量分析:", strategy_name);
    println!("       - 平均字符数: {}", avg_chars);
    println!("       - 平均行数: {:.1}", avg_lines);
    println!(
        "       - 完整函数块: {}/{}",
        complete_functions,
        chunks.len()
    );
    println!("       - 包含类定义: {}", has_class_definition);
}

async fn compare_chunking_quality(content: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 分块质量详细对比");
    println!("------------------");

    // 使用当前配置
    let config = CodeSplitterConfig::new(ProgrammingLanguage::CSharp, 30, 10, 1200)
        .with_ast_splitting(true)
        .with_respect_function_boundaries(true)
        .with_respect_class_boundaries(true);

    let splitter = CodeSplitter::new(config)?;
    let chunks = splitter.split_text(content)?;

    println!("📊 详细分块分析 ({} 块):", chunks.len());

    for (i, chunk) in chunks.iter().take(5).enumerate() {
        println!("\n--- 块 {} ---", i + 1);
        println!("长度: {} 字符, {} 行", chunk.len(), chunk.lines().count());

        // 分析代码结构
        let has_class = chunk.contains("class ");
        let has_function =
            chunk.contains("public ") || chunk.contains("private ") || chunk.contains("void ");
        let has_comments = chunk.contains("//") || chunk.contains("/*");
        let open_braces = chunk.matches('{').count();
        let close_braces = chunk.matches('}').count();
        let brace_balanced = open_braces == close_braces;

        println!(
            "结构: 类={}, 函数={}, 注释={}, 括号平衡={}",
            has_class, has_function, has_comments, brace_balanced
        );

        // 显示前几行作为预览
        let preview_lines: Vec<&str> = chunk.lines().take(3).collect();
        println!("预览:");
        for line in preview_lines {
            println!("  {}", line.trim());
        }
        if chunk.lines().count() > 3 {
            println!("  ... ({} 行)", chunk.lines().count() - 3);
        }
    }

    if chunks.len() > 5 {
        println!("\n... 还有 {} 个块", chunks.len() - 5);
    }

    // 分析与 LlamaIndex 的差异
    println!("\n🆚 与 LlamaIndex 的差异分析:");
    println!("我们的实现:");
    println!("  ✅ 基于行数的分块 (类似 LlamaIndex)");
    println!("  ✅ AST 结构感知 (类似 LlamaIndex)");
    println!("  ✅ 函数/类边界尊重");
    println!("  ✅ 重叠机制");
    println!("  ⚠️  可能的差异:");
    println!("     - LlamaIndex 基于字节位置递归分块");
    println!("     - 我们基于行数和结构边界分块");
    println!("     - LlamaIndex 更严格的字符限制");
    println!("     - 我们有更多的回退机制");

    Ok(())
}
