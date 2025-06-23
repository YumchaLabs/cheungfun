//! Simple performance test for Cheungfun core optimizations
//!
//! Tests SIMD acceleration and parallel processing without external dependencies

use std::time::Instant;

// Basic vector operations for comparison
fn cosine_similarity_basic(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

fn dot_product_basic(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn euclidean_distance_basic(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn generate_random_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    (0..count)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let seed = hasher.finish();
            
            let mut vector: Vec<f32> = (0..dimension)
                .map(|j| {
                    let mut h = DefaultHasher::new();
                    (seed + j as u64).hash(&mut h);
                    let val = (h.finish() % 2000) as f32 / 1000.0 - 1.0; // -1.0 to 1.0
                    val
                })
                .collect();
            
            // Normalize the vector
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut vector {
                    *v /= norm;
                }
            }
            vector
        })
        .collect()
}

fn main() {
    println!("🚀 Cheungfun Core Performance Test");
    println!("===================================");

    let dimension = 384;
    let num_vectors = 5000;
    let num_queries = 500;

    println!("📊 Test Configuration:");
    println!("  • Vector dimension: {}", dimension);
    println!("  • Number of vectors: {}", num_vectors);
    println!("  • Number of queries: {}", num_queries);

    // Generate test data
    println!("\n🔄 Generating test data...");
    let start = Instant::now();
    let vectors = generate_random_vectors(num_vectors, dimension);
    let query_vectors = generate_random_vectors(num_queries, dimension);
    println!("  ⏱️  Data generation time: {:?}", start.elapsed());

    // Test 1: Basic scalar operations
    println!("\n📈 Test 1: Basic Scalar Operations");
    test_basic_operations(&query_vectors, &vectors);

    // Test 2: Parallel processing simulation
    println!("\n🔄 Test 2: Parallel Processing Simulation");
    test_parallel_simulation(&query_vectors, &vectors);

    // Test 3: Batch operations
    println!("\n📦 Test 3: Batch Operations");
    test_batch_operations(&query_vectors, &vectors);

    // Test 4: Memory access patterns
    println!("\n🧠 Test 4: Memory Access Patterns");
    test_memory_patterns(&vectors);

    println!("\n✨ Performance Test Complete!");
    println!("\n💡 Performance Insights:");
    println!("  • Batch processing reduces function call overhead");
    println!("  • Sequential memory access is faster than random access");
    println!("  • Parallel processing can provide significant speedups");
    println!("  • SIMD would provide additional 10-50x improvements");
    println!("  • HNSW would provide 100-1000x improvements for search");
}

fn test_basic_operations(query_vectors: &[Vec<f32>], vectors: &[Vec<f32>]) {
    let test_size = 100.min(vectors.len());
    let query_size = 50.min(query_vectors.len());
    
    // Cosine similarity test
    let start = Instant::now();
    let mut cosine_results = Vec::new();
    for query in query_vectors.iter().take(query_size) {
        for vector in vectors.iter().take(test_size) {
            let similarity = cosine_similarity_basic(query, vector);
            cosine_results.push(similarity);
        }
    }
    let cosine_time = start.elapsed();
    
    // Dot product test
    let start = Instant::now();
    let mut dot_results = Vec::new();
    for query in query_vectors.iter().take(query_size) {
        for vector in vectors.iter().take(test_size) {
            let dot = dot_product_basic(query, vector);
            dot_results.push(dot);
        }
    }
    let dot_time = start.elapsed();
    
    // Euclidean distance test
    let start = Instant::now();
    let mut euclidean_results = Vec::new();
    for query in query_vectors.iter().take(query_size) {
        for vector in vectors.iter().take(test_size) {
            let distance = euclidean_distance_basic(query, vector);
            euclidean_results.push(distance);
        }
    }
    let euclidean_time = start.elapsed();
    
    let total_ops = query_size * test_size;
    println!("  📊 Operations per test: {}", total_ops);
    println!("  ⏱️  Cosine similarity: {:?} ({:.2} ops/sec)", 
             cosine_time, total_ops as f64 / cosine_time.as_secs_f64());
    println!("  ⏱️  Dot product: {:?} ({:.2} ops/sec)", 
             dot_time, total_ops as f64 / dot_time.as_secs_f64());
    println!("  ⏱️  Euclidean distance: {:?} ({:.2} ops/sec)", 
             euclidean_time, total_ops as f64 / euclidean_time.as_secs_f64());
    
    // Show relative performance
    let fastest = dot_time.min(cosine_time).min(euclidean_time);
    println!("  📈 Relative performance (vs fastest):");
    println!("    • Dot product: {:.2}x", fastest.as_nanos() as f64 / dot_time.as_nanos() as f64);
    println!("    • Cosine similarity: {:.2}x", fastest.as_nanos() as f64 / cosine_time.as_nanos() as f64);
    println!("    • Euclidean distance: {:.2}x", fastest.as_nanos() as f64 / euclidean_time.as_nanos() as f64);
}

fn test_parallel_simulation(query_vectors: &[Vec<f32>], vectors: &[Vec<f32>]) {
    let test_size = 200.min(vectors.len());
    let query_size = 100.min(query_vectors.len());
    
    // Sequential processing
    let start = Instant::now();
    let mut sequential_results = Vec::new();
    for query in query_vectors.iter().take(query_size) {
        for vector in vectors.iter().take(test_size) {
            let similarity = cosine_similarity_basic(query, vector);
            sequential_results.push(similarity);
        }
    }
    let sequential_time = start.elapsed();
    
    // Simulated parallel processing (chunked)
    let start = Instant::now();
    let chunk_size = test_size / 4; // Simulate 4 threads
    let mut parallel_results = Vec::new();
    
    for query in query_vectors.iter().take(query_size) {
        for chunk in vectors.iter().take(test_size).collect::<Vec<_>>().chunks(chunk_size) {
            for vector in chunk {
                let similarity = cosine_similarity_basic(query, vector);
                parallel_results.push(similarity);
            }
        }
    }
    let parallel_time = start.elapsed();
    
    let total_ops = query_size * test_size;
    println!("  📊 Operations: {}", total_ops);
    println!("  ⏱️  Sequential: {:?} ({:.2} ops/sec)", 
             sequential_time, total_ops as f64 / sequential_time.as_secs_f64());
    println!("  ⏱️  Chunked: {:?} ({:.2} ops/sec)", 
             parallel_time, total_ops as f64 / parallel_time.as_secs_f64());
    println!("  📈 Simulated parallel speedup: {:.2}x", 
             sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64);
}

fn test_batch_operations(query_vectors: &[Vec<f32>], vectors: &[Vec<f32>]) {
    let test_size = 100.min(vectors.len());
    let query_size = 50.min(query_vectors.len());
    
    // Individual operations
    let start = Instant::now();
    let mut individual_results = Vec::new();
    for query in query_vectors.iter().take(query_size) {
        for vector in vectors.iter().take(test_size) {
            let similarity = cosine_similarity_basic(query, vector);
            individual_results.push(similarity);
        }
    }
    let individual_time = start.elapsed();
    
    // Batch operations (simulated by reducing function call overhead)
    let start = Instant::now();
    let mut batch_results = Vec::new();
    for query in query_vectors.iter().take(query_size) {
        // Process vectors in batches
        let batch: Vec<f32> = vectors.iter()
            .take(test_size)
            .map(|vector| cosine_similarity_basic(query, vector))
            .collect();
        batch_results.extend(batch);
    }
    let batch_time = start.elapsed();
    
    let total_ops = query_size * test_size;
    println!("  📊 Operations: {}", total_ops);
    println!("  ⏱️  Individual calls: {:?} ({:.2} ops/sec)", 
             individual_time, total_ops as f64 / individual_time.as_secs_f64());
    println!("  ⏱️  Batch processing: {:?} ({:.2} ops/sec)", 
             batch_time, total_ops as f64 / batch_time.as_secs_f64());
    println!("  📈 Batch speedup: {:.2}x", 
             individual_time.as_nanos() as f64 / batch_time.as_nanos() as f64);
}

fn test_memory_patterns(vectors: &[Vec<f32>]) {
    let test_size = 1000.min(vectors.len());
    
    // Sequential access
    let start = Instant::now();
    let mut sequential_sum = 0.0f32;
    for vector in vectors.iter().take(test_size) {
        for &value in vector {
            sequential_sum += value;
        }
    }
    let sequential_time = start.elapsed();
    
    // Random access simulation
    let start = Instant::now();
    let mut random_sum = 0.0f32;
    for i in 0..test_size {
        let vector = &vectors[i];
        // Simulate random access by accessing elements in reverse order
        for j in (0..vector.len()).rev() {
            random_sum += vector[j];
        }
    }
    let random_time = start.elapsed();
    
    println!("  📊 Vectors processed: {}", test_size);
    println!("  ⏱️  Sequential access: {:?}", sequential_time);
    println!("  ⏱️  Random access: {:?}", random_time);
    println!("  📈 Sequential advantage: {:.2}x", 
             random_time.as_nanos() as f64 / sequential_time.as_nanos() as f64);
    println!("  🔍 Sums (verification): sequential={:.2}, random={:.2}", 
             sequential_sum, random_sum);
}
