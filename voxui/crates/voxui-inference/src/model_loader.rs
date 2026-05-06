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
