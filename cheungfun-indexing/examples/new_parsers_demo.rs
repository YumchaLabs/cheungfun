//! Demo of new node parsers: HierarchicalNodeParser, HTMLNodeParser, and JSONNodeParser.

use cheungfun_core::Document;
use cheungfun_indexing::node_parser::{
    file::{HTMLNodeParser, JSONNodeParser},
    relational::HierarchicalNodeParser,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Cheungfun New Node Parsers Demo\n");

    // Demo 1: HierarchicalNodeParser
    demo_hierarchical_parser().await?;

    // Demo 2: HTMLNodeParser
    demo_html_parser().await?;

    // Demo 3: JSONNodeParser
    demo_json_parser().await?;

    Ok(())
}

async fn demo_hierarchical_parser() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 HierarchicalNodeParser Demo");
    println!("================================");

    let parser = HierarchicalNodeParser::from_defaults(vec![1024, 256, 64])?;

    let long_text =
        "This is a very long document that will be split into multiple hierarchical levels. "
            .repeat(50);
    let document = Document::new(long_text);

    let nodes = parser.parse_nodes(&[document], true).await?;

    println!("Generated {} hierarchical nodes", nodes.len());

    // Show hierarchy structure
    for (i, node) in nodes.iter().enumerate() {
        let parent_id = node
            .metadata
            .get("parent_id")
            .and_then(|v| v.as_str())
            .unwrap_or("root");
        println!(
            "  Node {}: {} chars (parent: {})",
            i + 1,
            node.content.len(),
            parent_id
        );
    }

    println!();
    Ok(())
}

async fn demo_html_parser() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 HTMLNodeParser Demo");
    println!("======================");

    let parser = HTMLNodeParser::new();

    let html_content = r#"
        <html>
            <body>
                <h1>Introduction to RAG</h1>
                <p>Retrieval-Augmented Generation (RAG) is a powerful technique that combines retrieval and generation.</p>
                
                <h2>Key Components</h2>
                <p>RAG systems typically consist of three main components:</p>
                <ul>
                    <li>Document indexing and storage</li>
                    <li>Retrieval mechanism</li>
                    <li>Generation model</li>
                </ul>
                
                <h2>Benefits</h2>
                <p>RAG provides several advantages over traditional language models:</p>
                <p>It can access up-to-date information and provide more accurate responses.</p>
            </body>
        </html>
    "#;

    let document = Document::new(html_content);
    let nodes = parser.parse_nodes(&[document], false).await?;

    println!("Generated {} nodes from HTML", nodes.len());

    for (i, node) in nodes.iter().enumerate() {
        let tag = node
            .metadata
            .get("tag")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        println!("  Node {}: <{}> - {} chars", i + 1, tag, node.content.len());
        println!(
            "    Content: {}",
            &node.content[..node.content.len().min(50)]
        );
    }

    println!();
    Ok(())
}

async fn demo_json_parser() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 JSONNodeParser Demo");
    println!("======================");

    let parser = JSONNodeParser::new();

    let json_content = r#"
        {
            "company": "Cheungfun AI",
            "products": [
                {
                    "name": "RAG Framework",
                    "description": "High-performance RAG framework in Rust",
                    "features": ["Fast", "Scalable", "Modular"]
                },
                {
                    "name": "Vector Store",
                    "description": "Optimized vector storage solution",
                    "features": ["SIMD", "HNSW", "Parallel"]
                }
            ],
            "team": {
                "size": 10,
                "locations": ["San Francisco", "Beijing", "London"],
                "founded": "2024"
            },
            "metrics": {
                "performance": {
                    "qps": 378,
                    "latency_p95": "90.98ms"
                },
                "scalability": {
                    "max_documents": 1000000,
                    "max_concurrent_queries": 100
                }
            }
        }
    "#;

    let document = Document::new(json_content);
    let nodes = parser.parse_nodes(&[document], false).await?;

    println!("Generated {} nodes from JSON", nodes.len());

    for (i, node) in nodes.iter().enumerate() {
        println!("  Node {}: {} chars", i + 1, node.content.len());
        println!(
            "    Content preview: {}",
            &node.content[..node.content.len().min(100)]
        );
        if node.content.len() > 100 {
            println!("    ...");
        }
    }

    println!();
    Ok(())
}
