//! High-performance SIMD vector operations for Cheungfun
//!
//! This module provides optimized vector similarity calculations using SIMD instructions
//! through the `SimSIMD` library, offering 10-200x performance improvements over naive implementations.

use cheungfun_core::{CheungfunError, Result};

/// High-performance vector similarity calculator using SIMD instructions
#[derive(Debug, Clone)]
pub struct SimdVectorOps {
    /// Whether SIMD operations are available
    simd_available: bool,
}

impl SimdVectorOps {
    /// Create a new SIMD vector operations instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "simd")]
            simd_available: Self::check_simd_support(),
            #[cfg(not(feature = "simd"))]
            simd_available: false,
        }
    }

    /// Check if SIMD operations are available on this CPU
    #[cfg(feature = "simd")]
    fn check_simd_support() -> bool {
        // SimSIMD automatically detects and uses the best available SIMD instructions
        true
    }

    /// Check if SIMD operations are available
    pub fn is_simd_available(&self) -> bool {
        self.simd_available
    }

    /// Get available CPU capabilities for SIMD operations
    #[cfg(feature = "simd")]
    pub fn get_capabilities(&self) -> String {
        format!(
            "SIMD enabled with SimSIMD - CPU features: {}",
            std::env::consts::ARCH
        )
    }

    #[cfg(not(feature = "simd"))]
    pub fn get_capabilities(&self) -> String {
        "SIMD not available (feature not enabled)".to_string()
    }

    /// Calculate cosine similarity between two f32 vectors using SIMD
    pub fn cosine_similarity_f32(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(CheungfunError::Validation {
                message: format!("Vector dimensions must match: {} vs {}", a.len(), b.len()),
            });
        }

        #[cfg(feature = "simd")]
        {
            // Use SimSIMD for high-performance cosine similarity
            if let Some(distance) = simsimd::SpatialSimilarity::cosine(a, b) {
                // SimSIMD returns cosine distance (1 - similarity), convert to similarity
                Ok(1.0 - distance as f32)
            } else {
                // Fallback to scalar implementation if SIMD fails
                Ok(self.cosine_similarity_f32_scalar(a, b))
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            // Fallback to optimized scalar implementation
            Ok(self.cosine_similarity_f32_scalar(a, b))
        }
    }

    /// Calculate cosine similarity between two f16 vectors using SIMD
    #[cfg(feature = "simd")]
    pub fn cosine_similarity_f16(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(CheungfunError::Validation {
                message: format!("Vector dimensions must match: {} vs {}", a.len(), b.len()),
            });
        }

        // TODO: Implement proper f16 SIMD support
        // For now, use f32 scalar implementation
        Ok(self.cosine_similarity_f32_scalar(a, b))
    }

    /// Calculate dot product between two f32 vectors using SIMD
    pub fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(CheungfunError::Validation {
                message: format!("Vector dimensions must match: {} vs {}", a.len(), b.len()),
            });
        }

        #[cfg(feature = "simd")]
        {
            // Use SimSIMD for high-performance dot product
            if let Some(result) = simsimd::SpatialSimilarity::dot(a, b) {
                Ok(result as f32)
            } else {
                // Fallback to scalar implementation if SIMD fails
                Ok(self.dot_product_f32_scalar(a, b))
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            // Fallback to scalar implementation
            Ok(self.dot_product_f32_scalar(a, b))
        }
    }

    /// Calculate squared Euclidean distance between two f32 vectors using SIMD
    pub fn euclidean_distance_squared_f32(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(CheungfunError::Validation {
                message: format!("Vector dimensions must match: {} vs {}", a.len(), b.len()),
            });
        }

        #[cfg(feature = "simd")]
        {
            // Use SimSIMD for high-performance euclidean distance
            if let Some(distance) = simsimd::SpatialSimilarity::sqeuclidean(a, b) {
                Ok(distance as f32)
            } else {
                // Fallback to scalar implementation if SIMD fails
                Ok(self.euclidean_distance_squared_f32_scalar(a, b))
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            // Fallback to scalar implementation
            Ok(self.euclidean_distance_squared_f32_scalar(a, b))
        }
    }

    /// Batch cosine similarity calculation for multiple vector pairs
    pub fn batch_cosine_similarity_f32(&self, pairs: &[(&[f32], &[f32])]) -> Result<Vec<f32>> {
        #[cfg(feature = "simd")]
        {
            // Use parallel processing with SIMD for large batches
            if pairs.len() > 100 {
                use rayon::prelude::*;
                pairs
                    .par_iter()
                    .map(|(a, b)| self.cosine_similarity_f32(a, b))
                    .collect()
            } else {
                pairs
                    .iter()
                    .map(|(a, b)| self.cosine_similarity_f32(a, b))
                    .collect()
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            pairs
                .iter()
                .map(|(a, b)| self.cosine_similarity_f32(a, b))
                .collect()
        }
    }

    /// One-to-many cosine similarity calculation (query vector vs multiple vectors)
    ///
    /// # Errors
    ///
    /// Returns an error if vector dimensions don't match or SIMD operations fail.
    pub fn one_to_many_cosine_similarity_f32(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "simd")]
        {
            // Use parallel processing for large vector sets
            if vectors.len() > 50 {
                use rayon::prelude::*;
                vectors
                    .par_iter()
                    .map(|vector| self.cosine_similarity_f32(query, vector))
                    .collect()
            } else {
                vectors
                    .iter()
                    .map(|vector| self.cosine_similarity_f32(query, vector))
                    .collect()
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            vectors
                .iter()
                .map(|vector| self.cosine_similarity_f32(query, vector))
                .collect()
        }
    }

    // Scalar fallback implementations for when SIMD is not available

    fn cosine_similarity_f32_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product = self.dot_product_f32_scalar(a, b);
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    fn dot_product_f32_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn euclidean_distance_squared_f32_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }
}

impl Default for SimdVectorOps {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_vector_ops_creation() {
        let ops = SimdVectorOps::new();
        println!("SIMD available: {}", ops.is_simd_available());
        println!("Capabilities: {}", ops.get_capabilities());
    }

    #[test]
    fn test_cosine_similarity() {
        let ops = SimdVectorOps::new();
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        let vec3 = vec![1.0, 0.0, 0.0];

        let sim_orthogonal = ops.cosine_similarity_f32(&vec1, &vec2).unwrap();
        let sim_identical = ops.cosine_similarity_f32(&vec1, &vec3).unwrap();

        assert!((sim_orthogonal - 0.0).abs() < 0.001);
        assert!((sim_identical - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dot_product() {
        let ops = SimdVectorOps::new();
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];

        let result = ops.dot_product_f32(&vec1, &vec2).unwrap();
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // 32.0

        assert!((result - expected).abs() < 0.001);
    }

    #[test]
    fn test_batch_operations() {
        let ops = SimdVectorOps::new();
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        let vec3 = vec![1.0, 0.0, 0.0];

        let pairs = vec![(&vec1[..], &vec2[..]), (&vec1[..], &vec3[..])];
        let results = ops.batch_cosine_similarity_f32(&pairs).unwrap();

        assert_eq!(results.len(), 2);
        assert!((results[0] - 0.0).abs() < 0.001); // orthogonal
        assert!((results[1] - 1.0).abs() < 0.001); // identical
    }
}
