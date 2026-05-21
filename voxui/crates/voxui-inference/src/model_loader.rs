//! Load tensors from GGUF files into candle Tensors.

use std::{
    collections::HashMap,
    fmt,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use voxui_gguf::{GgufFile, MetadataValue, TensorInfo};

use crate::{LinearWeight, RuntimeTensor};

struct GgufTensorStore {
    gguf: GgufFile,
    cache: Mutex<HashMap<String, Tensor>>,
    path: PathBuf,
}

/// Loads tensors from a GGUF file into candle Tensors.
#[derive(Clone)]
pub struct GgufModelLoader {
    store: Arc<GgufTensorStore>,
    device: Device,
}

impl fmt::Debug for GgufModelLoader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GgufModelLoader")
            .field("path", &self.store.path)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl GgufModelLoader {
    /// Open a GGUF file and prepare for tensor loading.
    pub fn new(path: &Path, device: Device) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        let store = GgufTensorStore {
            gguf,
            cache: Mutex::new(HashMap::new()),
            path: path.to_path_buf(),
        };
        Ok(Self {
            store: Arc::new(store),
            device,
        })
    }

    /// Open the canonical single-file GGUF export from a model directory.
    pub fn from_model_dir(model_dir: &Path, device: Device) -> Result<Self> {
        let path = model_dir.join("model.gguf");
        if !path.is_file() {
            anyhow::bail!("model.gguf not found in model directory '{}'", model_dir.display());
        }
        Self::new(&path, device)
    }

    /// Load a tensor by name, dequantizing to f32, and place it on the configured device.
    pub fn load_tensor(&self, name: &str) -> Result<Tensor> {
        if let Some(tensor) = self
            .store
            .cache
            .lock()
            .map_err(|_| anyhow::anyhow!("GGUF tensor cache lock poisoned"))?
            .get(name)
            .cloned()
        {
            return Ok(tensor);
        }

        let tensor = self.load_tensor_uncached(name)?;

        self.store
            .cache
            .lock()
            .map_err(|_| anyhow::anyhow!("GGUF tensor cache lock poisoned"))?
            .insert(name.to_string(), tensor.clone());

        Ok(tensor)
    }

    /// Load a tensor and cast to f16 (useful for saving memory on GPU).
    pub fn load_tensor_f16(&self, name: &str) -> Result<Tensor> {
        self.load_tensor_uncached(name)?
            .to_dtype(DType::F16)
            .map_err(Into::into)
    }

    /// Load a tensor in the optimal dtype for the target device.
    /// CUDA: f16 (faster, lower VRAM). CPU: f32 (no precision issues).
    pub fn load_tensor_optimal(&self, name: &str) -> Result<Tensor> {
        if self.device.is_cuda() {
            let tensor = self.load_tensor_uncached(name)?;
            tensor.to_dtype(DType::F16).map_err(Into::into)
        } else {
            self.load_tensor(name)
        }
    }

    /// Get a reference to the device this loader targets.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if a tensor exists in the file.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.store.gguf.tensor_info(name).is_some()
    }

    /// Access file metadata.
    pub fn metadata(&self) -> &HashMap<String, MetadataValue> {
        &self.store.gguf.metadata
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.store.gguf.tensor_names()
    }

    /// Get tensor info (shape, dtype, etc.) without loading data.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.store.gguf.tensor_info(name)
    }

    pub(crate) fn load_runtime_tensor(&self, name: &str) -> Result<RuntimeTensor> {
        let info = self.tensor_info(name).ok_or_else(|| {
            anyhow::anyhow!(
                "Tensor '{}' not found in GGUF file '{}'",
                name,
                self.store.path.display()
            )
        })?;
        if info.is_quantized() {
            let raw = self.store.gguf.tensor_raw(name)?;
            RuntimeTensor::from_raw_quantized(raw, &self.device)
        } else {
            self.load_tensor_optimal(name).map(RuntimeTensor::Dense)
        }
    }

    pub(crate) fn load_linear_weight(&self, name: &str) -> Result<LinearWeight> {
        let info = self.tensor_info(name).ok_or_else(|| {
            anyhow::anyhow!(
                "Tensor '{}' not found in GGUF file '{}'",
                name,
                self.store.path.display()
            )
        })?;
        if info.is_quantized() {
            let raw = self.store.gguf.tensor_raw(name)?;
            LinearWeight::from_raw_quantized(raw, &self.device)
        } else {
            self.load_tensor_optimal(name).map(LinearWeight::Dense)
        }
    }

    pub(crate) fn ensure_dense_supported(&self, name: &str, operator: &str) -> Result<()> {
        if let Some(info) = self.tensor_info(name) {
            if info.is_quantized() {
                anyhow::bail!(
                    "unsupported quantized tensor {name}: dtype {} cannot be used by {operator}; re-export this tensor as fp16/f32 or add quantized {operator} support",
                    info.dtype
                );
            }
        }
        Ok(())
    }

    fn load_tensor_uncached(&self, name: &str) -> Result<Tensor> {
        let info = self
            .store
            .gguf
            .tensor_info(name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Tensor '{}' not found in GGUF file '{}'",
                    name,
                    self.store.path.display()
                )
            })?;
        if info.is_quantized() {
            anyhow::bail!(
                "quantized tensor {name} has dtype {}; use load_linear_weight or load_runtime_tensor so it is not cached as a dense resident tensor",
                info.dtype
            );
        }
        let data = self.store.gguf.tensor_f32(name).with_context(|| {
            format!(
                "Failed to load tensor '{}' from GGUF file '{}'",
                name,
                self.store.path.display()
            )
        })?;
        let shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
        Tensor::from_vec(data, shape.as_slice(), &self.device).with_context(|| {
            format!(
                "Failed to create tensor '{}' from GGUF file '{}'",
                name,
                self.store.path.display()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use byteorder::{LittleEndian, WriteBytesExt};
    use candle_core::{Device, Tensor};
    use std::io::Write;

    fn write_string(mut out: impl Write, value: &str) -> Result<()> {
        out.write_u64::<LittleEndian>(value.len() as u64)?;
        out.write_all(value.as_bytes())?;
        Ok(())
    }

    fn align_32(len: usize) -> usize {
        (len + 31) & !31
    }

    fn write_loader_test_gguf(dir: &std::path::Path) -> Result<std::path::PathBuf> {
        let path = dir.join("model.gguf");
        let mut bytes = Vec::new();
        bytes.write_u32::<LittleEndian>(0x46554747)?;
        bytes.write_u32::<LittleEndian>(3)?;
        bytes.write_u64::<LittleEndian>(2)?;
        bytes.write_u64::<LittleEndian>(0)?;

        write_string(&mut bytes, "dense.weight")?;
        bytes.write_u32::<LittleEndian>(2)?;
        bytes.write_u64::<LittleEndian>(1)?;
        bytes.write_u64::<LittleEndian>(32)?;
        bytes.write_u32::<LittleEndian>(0)?;
        bytes.write_u64::<LittleEndian>(0)?;

        write_string(&mut bytes, "linear.weight")?;
        bytes.write_u32::<LittleEndian>(2)?;
        bytes.write_u64::<LittleEndian>(1)?;
        bytes.write_u64::<LittleEndian>(32)?;
        bytes.write_u32::<LittleEndian>(2)?;
        bytes.write_u64::<LittleEndian>(128)?;

        bytes.resize(align_32(bytes.len()), 0);
        for _ in 0..32 {
            bytes.extend_from_slice(&1.0f32.to_le_bytes());
        }
        bytes.extend_from_slice(&0.25f32.to_le_bytes()[0..2]);
        bytes.extend_from_slice(&[0x88; 16]);
        std::fs::write(&path, bytes)?;
        Ok(path)
    }

    #[test]
    fn dense_loader_rejects_quantized_tensor_names() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = write_loader_test_gguf(dir.path())?;
        let loader = GgufModelLoader::new(&path, Device::Cpu)?;

        let err = loader.load_tensor("linear.weight").unwrap_err().to_string();

        assert!(err.contains("quantized tensor linear.weight"));
        assert!(err.contains("load_linear_weight"));
        Ok(())
    }

    #[test]
    fn linear_loader_accepts_quantized_tensor_names() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = write_loader_test_gguf(dir.path())?;
        let loader = GgufModelLoader::new(&path, Device::Cpu)?;

        let weight = loader.load_linear_weight("linear.weight")?;
        let input = Tensor::from_vec(vec![1.0f32; 32], (1, 32), &Device::Cpu)?;
        let out = weight.forward(&input)?;

        assert_eq!(out.dims(), &[1, 1]);
        Ok(())
    }
}
