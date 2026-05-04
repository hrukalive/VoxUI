//! Test trace helpers for comparing native tensors against Python goldens.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TraceManifest {
    #[serde(default)]
    lists: HashMap<String, Vec<u32>>,
    tensors: Vec<TensorRecord>,
}

#[derive(Debug, Clone, Deserialize)]
struct TensorRecord {
    name: String,
    file: String,
    dtype: String,
    shape: Vec<usize>,
}

pub struct TraceCase {
    root: PathBuf,
    manifest: TraceManifest,
}

impl TraceCase {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        let manifest_path = root.join("trace.json");
        let text = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?;
        let manifest = serde_json::from_str(&text)
            .with_context(|| format!("parse {}", manifest_path.display()))?;
        Ok(Self { root, manifest })
    }

    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let record = self
            .manifest
            .tensors
            .iter()
            .find(|record| record.name == name)
            .ok_or_else(|| anyhow::anyhow!("trace tensor `{name}` not found"))?;
        if record.dtype != "f32" {
            bail!("unsupported trace tensor dtype `{}`", record.dtype);
        }
        let path = self.root.join(&record.file);
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        if bytes.len() % 4 != 0 {
            bail!("trace tensor {} byte length is not divisible by 4", path.display());
        }
        let mut data = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Tensor::from_vec(data, record.shape.as_slice(), &Device::Cpu).map_err(Into::into)
    }

    pub fn u32_list(&self, name: &str) -> Result<Vec<u32>> {
        self.manifest
            .lists
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("trace list `{name}` not found"))
    }
}

pub fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
    if actual.dims() != expected.dims() {
        bail!("shape mismatch: actual {:?}, expected {:?}", actual.dims(), expected.dims());
    }
    let actual = flat_f32(actual)?;
    let expected = flat_f32(expected)?;
    assert_close_slices(&actual, &expected, tolerance)
}

pub fn assert_close_prefix(actual: &Tensor, expected_prefix: &Tensor, tolerance: f32) -> Result<()> {
    let actual = flat_f32(actual)?;
    let expected = flat_f32(expected_prefix)?;
    if actual.len() < expected.len() {
        bail!("actual tensor has {} values, expected prefix has {}", actual.len(), expected.len());
    }
    assert_close_slices(&actual[..expected.len()], &expected, tolerance)
}

fn flat_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    let len = tensor.dims().iter().product::<usize>();
    tensor
        .to_dtype(DType::F32)?
        .reshape((len,))?
        .to_vec1::<f32>()
        .map_err(Into::into)
}

fn assert_close_slices(actual: &[f32], expected: &[f32], tolerance: f32) -> Result<()> {
    if actual.len() != expected.len() {
        bail!("length mismatch: actual {}, expected {}", actual.len(), expected.len());
    }
    let mut max_abs = 0.0f32;
    let mut max_idx = 0usize;
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let abs = (a - e).abs();
        if abs > max_abs {
            max_abs = abs;
            max_idx = idx;
        }
    }
    if max_abs > tolerance {
        bail!("max abs diff {max_abs} at flat index {max_idx} exceeds tolerance {tolerance}");
    }
    Ok(())
}
