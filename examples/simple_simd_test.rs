//! Simple SIMD functionality test for Cheungfun
//!
//! This test verifies that SIMD vector operations are working correctly
//! and provides basic performance comparisons.

use std::time::Instant;

#[cfg(feature = "simd")]
use cheungfun_integrations::simd::SimdVectorOps;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Cheungfun SIMD Functionality Test");
    println!("====================================");
    
    // Test basic SIMD availability
    test_simd_availability()?;
    
    // Test vector operations
    test_vector_operations()?;
    
    // Performance comparison
    test_performance_comparison()?;
    
    println!("\n🎉 All SIMD tests completed successfully!");
    Ok(())
}

fn test_simd_availability() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing SIMD Availability");
    println!("----------------------------");
    
    #[cfg(feature = "simd")]
    {
        let simd_ops = SimdVectorOps::new();
        println!("✅ SIMD feature enabled");
        println!("   Available: {}", simd_ops.is_simd_available());
        println!("   Capabilities: {}", simd_ops.get_capabilities());
    }
    
    #[cfg(not(feature = "simd"))]
    {
        println!("❌ SIMD feature not enabled");
        println!("   💡 Enable with: cargo run --features simd --bin simple_simd_test");
    }
    
    Ok(())
}

fn test_vector_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧮 Testing Vector Operations");
    println!("----------------------------");
    
    #[cfg(feature = "simd")]
    {
        let simd_ops = SimdVectorOps::new();
        
        // Test vectors
        let vec1 = vec![1.0, 2.0, 3.0, 4.0];
        let vec2 = vec![5.0, 6.0, 7.0, 8.0];
        let vec3 = vec![1.0, 2.0, 3.0, 4.0]; // Same as vec1
        
        // Test cosine similarity
        let cosine_sim = simd_ops.cosine_similarity_f32(&vec1, &vec2)?;
        let cosine_identical = simd_ops.cosine_similarity_f32(&vec1, &vec3)?;
        
        println!("✅ Cosine Similarity Tests:");
        println!("   vec1 vs vec2: {:.6}", cosine_sim);
        println!("   vec1 vs vec3 (identical): {:.6}", cosine_identical);
        
        // Test dot product
        let dot_product = simd_ops.dot_product_f32(&vec1, &vec2)?;
        println!("✅ Dot Product Test:");
        println!("   vec1 · vec2: {:.6}", dot_product);
        
        // Test Euclidean distance
        let euclidean_dist = simd_ops.euclidean_distance_squared_f32(&vec1, &vec2)?;
        println!("✅ Euclidean Distance Test:");
        println!("   ||vec1 - vec2||²: {:.6}", euclidean_dist);
        
        // Test batch operations
        let pairs = vec![(&vec1[..], &vec2[..]), (&vec1[..], &vec3[..])];
        let batch_results = simd_ops.batch_cosine_similarity_f32(&pairs)?;
        println!("✅ Batch Operations Test:");
        println!("   Batch results: {:?}", batch_results);
        
        // Verify results make sense
        assert!((cosine_identical - 1.0).abs() < 0.001, "Identical vectors should have similarity ~1.0");
        assert!(cosine_sim > 0.0 && cosine_sim < 1.0, "Different vectors should have similarity between 0 and 1");
        assert!((dot_product - 70.0).abs() < 0.001, "Expected dot product: 1*5 + 2*6 + 3*7 + 4*8 = 70");
        
        println!("✅ All vector operation tests passed!");
    }
    
    #[cfg(not(feature = "simd"))]
    {
        println!("⚠️  SIMD tests skipped (feature not enabled)");
    }
    
    Ok(())
}

fn test_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Performance Comparison");
    println!("------------------------");
    
    #[cfg(feature = "simd")]
    {
        let simd_ops = SimdVectorOps::new();
        
        // Create larger test vectors
        let size = 1000;
        let vec1: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let vec2: Vec<f32> = (0..size).map(|i| (i + 100) as f32 * 0.1).collect();
        
        let iterations = 1000;
        
        // Test SIMD performance
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = simd_ops.cosine_similarity_f32(&vec1, &vec2)?;
        }
        let simd_duration = start.elapsed();
        
        // Test scalar performance (fallback)
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = cosine_similarity_scalar(&vec1, &vec2);
        }
        let scalar_duration = start.elapsed();
        
        println!("📊 Performance Results ({} iterations, {} dimensions):", iterations, size);
        println!("   SIMD:   {:?} ({:.2} ops/sec)", simd_duration, iterations as f64 / simd_duration.as_secs_f64());
        println!("   Scalar: {:?} ({:.2} ops/sec)", scalar_duration, iterations as f64 / scalar_duration.as_secs_f64());
        
        if simd_duration < scalar_duration {
            let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();
            println!("   🚀 SIMD is {:.2}x faster!", speedup);
        } else {
            println!("   ⚠️  Scalar implementation was faster (overhead for small vectors)");
        }
        
        // Test batch operations performance
        let pairs: Vec<(&[f32], &[f32])> = (0..100).map(|_| (vec1.as_slice(), vec2.as_slice())).collect();
        
        let start = Instant::now();
        let _ = simd_ops.batch_cosine_similarity_f32(&pairs)?;
        let batch_duration = start.elapsed();
        
        println!("   Batch:  {:?} (100 pairs)", batch_duration);
    }
    
    #[cfg(not(feature = "simd"))]
    {
        println!("⚠️  Performance tests skipped (SIMD feature not enabled)");
    }
    
    Ok(())
}

// Scalar implementation for comparison
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
