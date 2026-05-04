//! Load tensors from GGUF files into candle Tensors.

use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use voxui_gguf::{GgufFile, MetadataValue, TensorInfo};

/// Loads tensors from a GGUF file into candle Tensors.
pub struct GgufModelLoader {
    gguf: GgufFile,
    device: Device,
}

impl GgufModelLoader {
    /// Open a GGUF file and prepare for tensor loading.
    pub fn new(path: &Path, device: Device) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Ok(Self { gguf, device })
    }

    /// Load a tensor by name, dequantizing to f32, and place it on the configured device.
    pub fn load_tensor(&self, name: &str) -> Result<Tensor> {
        let info = self
            .gguf
            .tensor_info(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor '{}' not found in GGUF file", name))?;
        let data = self.gguf.tensor_f32(name)?;
        let shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
        let tensor = Tensor::from_vec(data, shape.as_slice(), &self.device)?;
        Ok(tensor)
    }

    /// Load a tensor and cast to f16 (useful for saving memory on GPU).
    pub fn load_tensor_f16(&self, name: &str) -> Result<Tensor> {
        self.load_tensor(name)?
            .to_dtype(DType::F16)
            .map_err(Into::into)
    }

    /// Load a tensor in the optimal dtype for the target device.
    /// CUDA: f16 (faster, lower VRAM). CPU: f32 (no precision issues).
    pub fn load_tensor_optimal(&self, name: &str) -> Result<Tensor> {
        let tensor = self.load_tensor(name)?;
        if self.device.is_cuda() {
            tensor.to_dtype(DType::F16).map_err(Into::into)
        } else {
            Ok(tensor)
        }
    }

    /// Get a reference to the device this loader targets.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if a tensor exists in the file.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.gguf.tensor_info(name).is_some()
    }

    /// Access file metadata.
    pub fn metadata(&self) -> &std::collections::HashMap<String, MetadataValue> {
        &self.gguf.metadata
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.gguf.tensor_names()
    }

    /// Get tensor info (shape, dtype, etc.) without loading data.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.gguf.tensor_info(name)
    }
}
