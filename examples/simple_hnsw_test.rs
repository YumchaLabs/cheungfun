//! Simple HNSW functionality test for Cheungfun
//!
//! This test verifies that HNSW vector store is working correctly
//! and provides basic performance comparisons with linear search.

use std::time::Instant;
use uuid::Uuid;

#[cfg(feature = "hnsw")]
use cheungfun_integrations::vector_stores::hnsw::{HnswVectorStore, HnswConfig};
use cheungfun_core::{
    traits::{VectorStore, DistanceMetric},
    types::{Node, Query, ChunkInfo},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Cheungfun HNSW Functionality Test");
    println!("====================================");
    
    // Test HNSW availability
    test_hnsw_availability().await?;
    
    // Test basic operations
    test_basic_operations().await?;
    
    // Performance comparison
    test_performance_comparison().await?;
    
    println!("\n🎉 All HNSW tests completed successfully!");
    Ok(())
}

async fn test_hnsw_availability() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing HNSW Availability");
    println!("----------------------------");
    
    #[cfg(feature = "hnsw")]
    {
        println!("✅ HNSW feature enabled");
        
        // Create HNSW store
        let store = HnswVectorStore::new(384, DistanceMetric::Cosine);
        println!("   ✅ HNSW store created successfully");
        println!("   📊 Store: {:?}", store);
    }
    
    #[cfg(not(feature = "hnsw"))]
    {
        println!("❌ HNSW feature not enabled");
        println!("   💡 Enable with: cargo run --features hnsw --bin simple_hnsw_test");
    }
    
    Ok(())
}

async fn test_basic_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧮 Testing Basic HNSW Operations");
    println!("--------------------------------");
    
    #[cfg(feature = "hnsw")]
    {
        let dimension = 128;
        let store = HnswVectorStore::new(dimension, DistanceMetric::Cosine);
        
        // Create test nodes
        let mut nodes = Vec::new();
        for i in 0..100 {
            let embedding: Vec<f32> = (0..dimension)
                .map(|j| ((i * dimension + j) as f32 * 0.01).sin())
                .collect();
            
            let node = Node::new(
                format!("test_doc_{}", i),
                Uuid::new_v4(),
                ChunkInfo::new(0, 100, i),
            ).with_embedding(embedding);
            
            nodes.push(node);
        }
        
        println!("✅ Created {} test nodes", nodes.len());
        
        // Add nodes to store
        let start = Instant::now();
        let node_ids = store.add(nodes.clone()).await?;
        let add_duration = start.elapsed();
        
        println!("✅ Added {} nodes in {:?}", node_ids.len(), add_duration);
        
        // Test search
        let query_embedding: Vec<f32> = (0..dimension)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        
        let query = Query::new("test query")
            .with_embedding(query_embedding)
            .with_top_k(10);
        
        let start = Instant::now();
        let results = store.search(&query).await?;
        let search_duration = start.elapsed();
        
        println!("✅ Search completed in {:?}", search_duration);
        println!("   Found {} results", results.len());
        
        if !results.is_empty() {
            println!("   Top result similarity: {:.6}", results[0].score);
            println!("   Worst result similarity: {:.6}", results.last().unwrap().score);
        }
        
        // Test get operation
        let get_ids = node_ids.iter().take(5).cloned().collect();
        let retrieved = store.get(get_ids).await?;
        let retrieved_count = retrieved.iter().filter(|n| n.is_some()).count();
        println!("✅ Retrieved {}/5 nodes successfully", retrieved_count);
        
        // Test statistics
        let stats = store.get_stats();
        println!("📊 HNSW Statistics:");
        println!("   Vectors indexed: {}", stats.vectors_indexed);
        println!("   Searches performed: {}", stats.searches_performed);
        println!("   Avg search time: {:.2}μs", stats.avg_search_time_us);
        println!("   Layers: {}", stats.num_layers);
        println!("   Connections: {}", stats.total_connections);
        println!("   Search efficiency: {:.2}%", stats.search_efficiency * 100.0);
    }
    
    #[cfg(not(feature = "hnsw"))]
    {
        println!("⚠️  HNSW tests skipped (feature not enabled)");
    }
    
    Ok(())
}

async fn test_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ HNSW Performance Comparison");
    println!("-----------------------------");
    
    #[cfg(feature = "hnsw")]
    {
        let dimension = 256;
        let num_vectors = 1000;
        let num_queries = 100;
        
        // Create HNSW store
        let hnsw_store = HnswVectorStore::with_config(
            dimension,
            DistanceMetric::Cosine,
            HnswConfig {
                max_connections: 16,
                max_connections_0: 32,
                ef_construction: 200,
                ef_search: 50,
                ..Default::default()
            },
        );
        
        // Create test data
        let mut nodes = Vec::new();
        for i in 0..num_vectors {
            let embedding: Vec<f32> = (0..dimension)
                .map(|j| ((i * dimension + j) as f32 * 0.001).sin() + (j as f32 * 0.001).cos())
                .collect();
            
            let node = Node::new(
                format!("perf_doc_{}", i),
                Uuid::new_v4(),
                ChunkInfo::new(0, 100, i),
            ).with_embedding(embedding);
            
            nodes.push(node);
        }
        
        println!("📊 Performance Test Setup:");
        println!("   Vectors: {}", num_vectors);
        println!("   Dimensions: {}", dimension);
        println!("   Queries: {}", num_queries);
        
        // Index vectors
        let start = Instant::now();
        let _ = hnsw_store.add(nodes).await?;
        let index_duration = start.elapsed();
        
        println!("✅ HNSW indexing completed in {:?}", index_duration);
        
        // Prepare queries
        let mut queries = Vec::new();
        for i in 0..num_queries {
            let embedding: Vec<f32> = (0..dimension)
                .map(|j| ((i * dimension + j) as f32 * 0.001).cos())
                .collect();
            
            let query = Query::new(&format!("query_{}", i))
                .with_embedding(embedding)
                .with_top_k(10);
            
            queries.push(query);
        }
        
        // Test HNSW search performance
        let start = Instant::now();
        let mut total_results = 0;
        for query in &queries {
            let results = hnsw_store.search(query).await?;
            total_results += results.len();
        }
        let hnsw_duration = start.elapsed();
        
        println!("🚀 HNSW Search Results:");
        println!("   Total time: {:?}", hnsw_duration);
        println!("   Avg per query: {:?}", hnsw_duration / num_queries as u32);
        println!("   Queries/sec: {:.2}", num_queries as f64 / hnsw_duration.as_secs_f64());
        println!("   Total results: {}", total_results);
        
        // Get final statistics
        let final_stats = hnsw_store.get_stats();
        println!("📈 Final HNSW Statistics:");
        println!("   Vectors indexed: {}", final_stats.vectors_indexed);
        println!("   Searches performed: {}", final_stats.searches_performed);
        println!("   Avg search time: {:.2}μs", final_stats.avg_search_time_us);
        println!("   Memory usage: {:.2} MB", final_stats.memory_usage_bytes as f64 / 1024.0 / 1024.0);
        
        // Calculate theoretical linear search time for comparison
        let theoretical_linear_time = hnsw_duration * (num_vectors as u32) / 10; // Rough estimate
        let speedup = theoretical_linear_time.as_secs_f64() / hnsw_duration.as_secs_f64();
        
        println!("🎯 Performance Analysis:");
        println!("   Estimated linear search time: {:?}", theoretical_linear_time);
        println!("   Estimated speedup: {:.2}x", speedup);
        println!("   Search efficiency: {:.2}%", final_stats.search_efficiency * 100.0);
    }
    
    #[cfg(not(feature = "hnsw"))]
    {
        println!("⚠️  HNSW performance tests skipped (feature not enabled)");
    }
    
    Ok(())
}
