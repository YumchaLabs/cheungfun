//! GPU-accelerated vector operations using Candle framework
//!
//! This module provides high-performance GPU-accelerated vector similarity calculations
//! using the Candle machine learning framework. It supports both CUDA and Metal backends
//! for maximum hardware compatibility.

use cheungfun_core::{CheungfunError, Result};

#[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
use candle_core::{DType, Device, Tensor};

/// GPU-accelerated vector operations
#[derive(Debug, Clone)]
pub struct GpuVectorOps {
    /// Compute device (CPU, CUDA, or Metal)
    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
    device: Device,
    /// Whether GPU acceleration is available
    gpu_available: bool,
    /// Device information
    device_info: String,
}

impl GpuVectorOps {
    /// Create a new GPU vector operations instance
    pub fn new() -> Self {
        #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
        {
            let (device, gpu_available, device_info) = Self::initialize_device();
            Self {
                device,
                gpu_available,
                device_info,
            }
        }

        #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
        {
            Self {
                gpu_available: false,
                device_info: "GPU features not enabled".to_string(),
            }
        }
    }

    /// Initialize the best available compute device
    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
    fn initialize_device() -> (Device, bool, String) {
        // Try CUDA first
        #[cfg(feature = "gpu-cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                return (device, true, "CUDA GPU".to_string());
            }
        }

        // Try Metal on macOS
        #[cfg(feature = "gpu-metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                return (device, true, "Metal GPU".to_string());
            }
        }

        // Fallback to CPU
        (Device::Cpu, false, "CPU (GPU not available)".to_string())
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// Get device information
    pub fn device_info(&self) -> &str {
        &self.device_info
    }

    /// Calculate cosine similarity between two f32 vectors using GPU
    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn cosine_similarity_f32(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(CheungfunError::Validation {
                message: format!("Vector dimensions must match: {} vs {}", a.len(), b.len()),
            });
        }

        if !self.gpu_available {
            return Ok(self.cosine_similarity_f32_cpu(a, b));
        }

        // Convert to tensors on GPU
        let tensor_a = Tensor::from_slice(a, (a.len(),), &self.device).map_err(|e| {
            CheungfunError::Computation {
                message: format!("Failed to create tensor A: {}", e),
            }
        })?;

        let tensor_b = Tensor::from_slice(b, (b.len(),), &self.device).map_err(|e| {
            CheungfunError::Computation {
                message: format!("Failed to create tensor B: {}", e),
            }
        })?;

        // Calculate dot product
        let dot_product = tensor_a
            .mul(&tensor_b)?
            .sum_all()?
            .to_scalar::<f32>()
            .map_err(|e| CheungfunError::Computation {
                message: format!("Failed to compute dot product: {}", e),
            })?;

        // Calculate norms
        let norm_a = tensor_a
            .sqr()?
            .sum_all()?
            .sqrt()?
            .to_scalar::<f32>()
            .map_err(|e| CheungfunError::Computation {
                message: format!("Failed to compute norm A: {}", e),
            })?;

        let norm_b = tensor_b
            .sqr()?
            .sum_all()?
            .sqrt()?
            .to_scalar::<f32>()
            .map_err(|e| CheungfunError::Computation {
                message: format!("Failed to compute norm B: {}", e),
            })?;

        // Calculate cosine similarity
        if norm_a == 0.0 || norm_b == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm_a * norm_b))
        }
    }

    /// CPU fallback for cosine similarity
    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
    fn cosine_similarity_f32_cpu(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// CPU-only cosine similarity when GPU features are disabled
    #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn cosine_similarity_f32(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(CheungfunError::Validation {
                message: format!("Vector dimensions must match: {} vs {}", a.len(), b.len()),
            });
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm_a * norm_b))
        }
    }

    /// Batch cosine similarity calculation using GPU
    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn batch_cosine_similarity_f32(
        &self,
        vectors_a: &[Vec<f32>],
        vectors_b: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        if vectors_a.len() != vectors_b.len() {
            return Err(CheungfunError::Validation {
                message: format!(
                    "Batch sizes must match: {} vs {}",
                    vectors_a.len(),
                    vectors_b.len()
                ),
            });
        }

        if vectors_a.is_empty() {
            return Ok(Vec::new());
        }

        let dim = vectors_a[0].len();
        if !vectors_a.iter().all(|v| v.len() == dim) || !vectors_b.iter().all(|v| v.len() == dim) {
            return Err(CheungfunError::Validation {
                message: "All vectors must have the same dimension".to_string(),
            });
        }

        if !self.gpu_available {
            // Fallback to CPU processing
            return vectors_a
                .iter()
                .zip(vectors_b.iter())
                .map(|(a, b)| self.cosine_similarity_f32_cpu(a, b))
                .map(Ok)
                .collect();
        }

        // Flatten vectors for batch processing
        let flat_a: Vec<f32> = vectors_a.iter().flatten().copied().collect();
        let flat_b: Vec<f32> = vectors_b.iter().flatten().copied().collect();

        // Create batch tensors
        let batch_size = vectors_a.len();
        let tensor_a =
            Tensor::from_slice(&flat_a, (batch_size, dim), &self.device).map_err(|e| {
                CheungfunError::Computation {
                    message: format!("Failed to create batch tensor A: {}", e),
                }
            })?;

        let tensor_b =
            Tensor::from_slice(&flat_b, (batch_size, dim), &self.device).map_err(|e| {
                CheungfunError::Computation {
                    message: format!("Failed to create batch tensor B: {}", e),
                }
            })?;

        // Calculate batch dot products
        let dot_products = tensor_a.mul(&tensor_b)?.sum(1)?; // Sum along dimension 1 (vector dimension)

        // Calculate batch norms
        let norms_a = tensor_a.sqr()?.sum(1)?.sqrt()?;
        let norms_b = tensor_b.sqr()?.sum(1)?.sqrt()?;

        // Calculate cosine similarities
        let similarities = dot_products.div(&norms_a)?.div(&norms_b)?;

        // Convert back to Vec<f32>
        let result: Vec<f32> = similarities
            .to_vec1()
            .map_err(|e| CheungfunError::Computation {
                message: format!("Failed to convert similarities to vector: {}", e),
            })?;

        Ok(result)
    }

    /// CPU-only batch cosine similarity when GPU features are disabled
    #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn batch_cosine_similarity_f32(
        &self,
        vectors_a: &[Vec<f32>],
        vectors_b: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        if vectors_a.len() != vectors_b.len() {
            return Err(CheungfunError::Validation {
                message: format!(
                    "Batch sizes must match: {} vs {}",
                    vectors_a.len(),
                    vectors_b.len()
                ),
            });
        }

        vectors_a
            .iter()
            .zip(vectors_b.iter())
            .map(|(a, b)| self.cosine_similarity_f32(a, b))
            .collect()
    }

    /// One-to-many cosine similarity calculation using GPU
    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn one_to_many_cosine_similarity_f32(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let dim = query.len();
        if !vectors.iter().all(|v| v.len() == dim) {
            return Err(CheungfunError::Validation {
                message: "All vectors must have the same dimension as query".to_string(),
            });
        }

        if !self.gpu_available {
            // Fallback to CPU processing
            return vectors
                .iter()
                .map(|v| self.cosine_similarity_f32_cpu(query, v))
                .map(Ok)
                .collect();
        }

        // Create query tensor
        let query_tensor = Tensor::from_slice(query, (1, dim), &self.device).map_err(|e| {
            CheungfunError::Computation {
                message: format!("Failed to create query tensor: {}", e),
            }
        })?;

        // Create batch tensor for all vectors
        let flat_vectors: Vec<f32> = vectors.iter().flatten().copied().collect();
        let batch_size = vectors.len();
        let vectors_tensor = Tensor::from_slice(&flat_vectors, (batch_size, dim), &self.device)
            .map_err(|e| CheungfunError::Computation {
                message: format!("Failed to create vectors tensor: {}", e),
            })?;

        // Calculate dot products (query @ vectors.T)
        let dot_products = query_tensor.matmul(&vectors_tensor.t()?)?.squeeze(0)?; // Remove batch dimension from query

        // Calculate norms
        let query_norm = query_tensor.sqr()?.sum(1)?.sqrt()?;
        let vector_norms = vectors_tensor.sqr()?.sum(1)?.sqrt()?;

        // Calculate cosine similarities
        let similarities = dot_products.div(&query_norm)?.div(&vector_norms)?;

        // Convert back to Vec<f32>
        let result: Vec<f32> = similarities
            .to_vec1()
            .map_err(|e| CheungfunError::Computation {
                message: format!("Failed to convert similarities to vector: {}", e),
            })?;

        Ok(result)
    }

    /// CPU-only one-to-many cosine similarity when GPU features are disabled
    #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn one_to_many_cosine_similarity_f32(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        vectors
            .iter()
            .map(|v| self.cosine_similarity_f32(query, v))
            .collect()
    }
}

impl Default for GpuVectorOps {
    fn default() -> Self {
        Self::new()
    }
}
