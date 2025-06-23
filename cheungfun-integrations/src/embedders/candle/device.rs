//! Device management for Candle embedder.
//!
//! This module handles device detection, selection, and management for
//! Candle-based embedding models. It supports automatic device selection
//! based on availability and performance characteristics.

use candle_core::Device;
use tracing::{debug, info, warn};
use super::error::CandleError;

/// Device manager for Candle operations.
#[derive(Debug, Clone)]
pub struct DeviceManager {
    device: Device,
    device_type: DeviceType,
}

/// Supported device types.
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    /// CPU device
    Cpu,
    /// CUDA GPU device
    Cuda(usize), // GPU index
    /// Metal GPU device (macOS)
    Metal,
}

impl DeviceManager {
    /// Create a new device manager with automatic device selection.
    pub fn new() -> Result<Self, CandleError> {
        Self::with_preference("auto")
    }
    
    /// Create a device manager with the specified device preference.
    ///
    /// # Arguments
    ///
    /// * `preference` - Device preference: "auto", "cpu", "cuda", "cuda:0", "metal"
    pub fn with_preference(preference: &str) -> Result<Self, CandleError> {
        let (device, device_type) = match preference.to_lowercase().as_str() {
            "auto" => Self::select_best_device()?,
            "cpu" => (Device::Cpu, DeviceType::Cpu),
            "cuda" => Self::select_cuda_device(None)?,
            device_str if device_str.starts_with("cuda:") => {
                let gpu_index = device_str
                    .strip_prefix("cuda:")
                    .and_then(|s| s.parse::<usize>().ok())
                    .ok_or_else(|| CandleError::Configuration {
                        message: format!("Invalid CUDA device specification: {}", device_str),
                    })?;
                Self::select_cuda_device(Some(gpu_index))?
            }
            "metal" => Self::select_metal_device()?,
            _ => {
                return Err(CandleError::Configuration {
                    message: format!("Unsupported device preference: {}", preference),
                });
            }
        };
        
        info!("Selected device: {:?}", device_type);
        
        Ok(Self {
            device,
            device_type,
        })
    }
    
    /// Get the Candle device.
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get the device type.
    pub fn device_type(&self) -> &DeviceType {
        &self.device_type
    }
    
    /// Check if the device is CUDA.
    pub fn is_cuda(&self) -> bool {
        matches!(self.device_type, DeviceType::Cuda(_))
    }
    
    /// Check if the device is CPU.
    pub fn is_cpu(&self) -> bool {
        matches!(self.device_type, DeviceType::Cpu)
    }
    
    /// Check if the device is Metal.
    pub fn is_metal(&self) -> bool {
        matches!(self.device_type, DeviceType::Metal)
    }
    
    /// Get device information as a string.
    pub fn device_info(&self) -> String {
        match &self.device_type {
            DeviceType::Cpu => "CPU".to_string(),
            DeviceType::Cuda(index) => format!("CUDA GPU {}", index),
            DeviceType::Metal => "Metal GPU".to_string(),
        }
    }
    
    /// Select the best available device automatically.
    fn select_best_device() -> Result<(Device, DeviceType), CandleError> {
        debug!("Auto-selecting best available device");
        
        // Try CUDA first (if available)
        if let Ok((device, device_type)) = Self::select_cuda_device(None) {
            debug!("Selected CUDA device");
            return Ok((device, device_type));
        }
        
        // Try Metal on macOS
        if cfg!(target_os = "macos") {
            if let Ok((device, device_type)) = Self::select_metal_device() {
                debug!("Selected Metal device");
                return Ok((device, device_type));
            }
        }
        
        // Fall back to CPU
        debug!("Falling back to CPU device");
        Ok((Device::Cpu, DeviceType::Cpu))
    }
    
    /// Select a CUDA device.
    fn select_cuda_device(gpu_index: Option<usize>) -> Result<(Device, DeviceType), CandleError> {
        let index = gpu_index.unwrap_or(0);
        
        match Device::new_cuda(index) {
            Ok(device) => {
                info!("Successfully initialized CUDA device {}", index);
                Ok((device, DeviceType::Cuda(index)))
            }
            Err(e) => {
                warn!("Failed to initialize CUDA device {}: {}", index, e);
                Err(CandleError::Device {
                    message: format!("CUDA device {} not available: {}", index, e),
                })
            }
        }
    }
    
    /// Select a Metal device.
    fn select_metal_device() -> Result<(Device, DeviceType), CandleError> {
        match Device::new_metal(0) {
            Ok(device) => {
                info!("Successfully initialized Metal device");
                Ok((device, DeviceType::Metal))
            }
            Err(e) => {
                warn!("Failed to initialize Metal device: {}", e);
                Err(CandleError::Device {
                    message: format!("Metal device not available: {}", e),
                })
            }
        }
    }
    
    /// Get memory usage information (if supported).
    pub fn memory_info(&self) -> Option<MemoryInfo> {
        match &self.device_type {
            DeviceType::Cuda(_) => {
                // TODO: Implement CUDA memory info when available in Candle
                None
            }
            DeviceType::Metal => {
                // TODO: Implement Metal memory info when available in Candle
                None
            }
            DeviceType::Cpu => None, // CPU memory is managed by the OS
        }
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            device: Device::Cpu,
            device_type: DeviceType::Cpu,
        })
    }
}

/// Memory usage information for GPU devices.
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total memory in bytes
    pub total: u64,
    /// Used memory in bytes
    pub used: u64,
    /// Free memory in bytes
    pub free: u64,
}

impl MemoryInfo {
    /// Get memory usage as a percentage.
    pub fn usage_percentage(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.used as f32 / self.total as f32) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_manager_creation() {
        let manager = DeviceManager::new();
        assert!(manager.is_ok());
    }
    
    #[test]
    fn test_cpu_device_selection() {
        let manager = DeviceManager::with_preference("cpu").unwrap();
        assert!(manager.is_cpu());
        assert_eq!(manager.device_info(), "CPU");
    }
    
    #[test]
    fn test_invalid_device_preference() {
        let result = DeviceManager::with_preference("invalid");
        assert!(result.is_err());
    }
}
