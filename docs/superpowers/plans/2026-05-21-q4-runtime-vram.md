# Q4 Runtime VRAM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make quantized GGUF tensors stay quantized during VoxCPM inference so q4/q8 exports reduce resident CUDA VRAM instead of only reducing disk size.

**Architecture:** Add raw GGUF tensor access, then add a quantized-aware runtime weight abstraction around Candle `QTensor` and `QMatMul`. Refactor model components to request linear, embedding, dense, or unsupported-op bindings explicitly, and make exporter low-VRAM profiles emit only quantization the runtime can actually honor.

**Tech Stack:** Rust 2021, Candle 0.8 quantized APIs, `voxui-gguf`, `voxui-inference`, Python exporter tests under the UV/PowerShell environment, CUDA/MSVC setup from `README.txt`.

---

## File Structure

- Modify `voxui/crates/voxui-gguf/src/types.rs`: add quantization helpers and a borrowed raw tensor view.
- Modify `voxui/crates/voxui-gguf/src/parser.rs`: expose raw tensor data without dequantization and add GGUF parser tests.
- Modify `voxui/crates/voxui-gguf/Cargo.toml`: add `tempfile` as a dev dependency for GGUF parser tests.
- Create `voxui/crates/voxui-inference/src/weights.rs`: define `RuntimeTensor`, `LinearWeight`, quantized dequant helpers, and unit tests.
- Modify `voxui/crates/voxui-inference/src/lib.rs`: register the new weight module and keep dense `linear` for dense-only call sites.
- Modify `voxui/crates/voxui-inference/src/model_loader.rs`: add quantized-aware load methods and make dense loaders reject q4/q8 residency.
- Modify `voxui/crates/voxui-inference/src/base_lm.rs`: use quantized linear weights, runtime norm tensors, and selected-row quantized embeddings.
- Modify `voxui/crates/voxui-inference/src/encoder.rs`: use quantized linear weights and runtime dense materialization for small non-linear tensors.
- Modify `voxui/crates/voxui-inference/src/dit.rs`: use quantized linear weights and runtime norm tensors.
- Modify `voxui/crates/voxui-inference/src/fsq.rs`: use quantized linear weights.
- Modify `voxui/crates/voxui-inference/src/engine.rs`: use quantized linear projection weights.
- Modify `voxui/crates/voxui-inference/src/audiovae.rs`: reject quantized VAE conv/weight-norm tensors explicitly.
- Modify `exporter/export_voxcpm.py`: add role-based low-VRAM quantization and reject q4/q8 VAE export.
- Modify `exporter/verify_gguf.py`: print dtype counts for mixed-dtype exports.
- Modify `exporter/tests/test_export_manifest.py`: cover role-based q4 profiles and unsupported VAE quantization.
- Modify `voxui/crates/voxui-inference/tests/cuda_vram_report.rs`: write JSON/Markdown artifacts and measure each model in an isolated child test process.

---

### Task 1: Add Raw GGUF Tensor Access

**Files:**
- Modify: `voxui/crates/voxui-gguf/Cargo.toml`
- Modify: `voxui/crates/voxui-gguf/src/types.rs`
- Modify: `voxui/crates/voxui-gguf/src/parser.rs`

- [ ] **Step 1: Add the failing parser test**

Add this dev dependency to `voxui/crates/voxui-gguf/Cargo.toml`:

```toml
[dev-dependencies]
tempfile = "3"
```

Add this test module at the bottom of `voxui/crates/voxui-gguf/src/parser.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;

    fn write_string(mut out: impl Write, value: &str) -> anyhow::Result<()> {
        out.write_u64::<LittleEndian>(value.len() as u64)?;
        out.write_all(value.as_bytes())?;
        Ok(())
    }

    fn align_32(len: usize) -> usize {
        (len + 31) & !31
    }

    fn write_test_gguf(path: &std::path::Path) -> anyhow::Result<()> {
        let mut bytes = Vec::new();
        bytes.write_u32::<LittleEndian>(GGUF_MAGIC)?;
        bytes.write_u32::<LittleEndian>(3)?;
        bytes.write_u64::<LittleEndian>(2)?;
        bytes.write_u64::<LittleEndian>(0)?;

        write_string(&mut bytes, "dense")?;
        bytes.write_u32::<LittleEndian>(1)?;
        bytes.write_u64::<LittleEndian>(4)?;
        bytes.write_u32::<LittleEndian>(0)?;
        bytes.write_u64::<LittleEndian>(0)?;

        write_string(&mut bytes, "q4")?;
        bytes.write_u32::<LittleEndian>(2)?;
        bytes.write_u64::<LittleEndian>(1)?;
        bytes.write_u64::<LittleEndian>(32)?;
        bytes.write_u32::<LittleEndian>(2)?;
        bytes.write_u64::<LittleEndian>(16)?;

        bytes.resize(align_32(bytes.len()), 0);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes.extend_from_slice(&3.0f32.to_le_bytes());
        bytes.extend_from_slice(&4.0f32.to_le_bytes());
        bytes.extend_from_slice(&[0x00; 18]);
        std::fs::write(path, bytes)?;
        Ok(())
    }

    #[test]
    fn tensor_raw_exposes_dtype_shape_and_bytes_without_dequantizing() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("model.gguf");
        write_test_gguf(&path)?;

        let gguf = GgufFile::open(&path)?;
        let raw = gguf.tensor_raw("q4")?;

        assert_eq!(raw.info.name, "q4");
        assert_eq!(raw.info.shape, vec![1, 32]);
        assert_eq!(raw.info.dtype, GgmlType::Q4_0);
        assert_eq!(raw.info.element_count(), 32);
        assert!(raw.info.is_quantized());
        assert_eq!(raw.data.len(), 18);
        Ok(())
    }

    #[test]
    fn dense_tensor_info_reports_non_quantized_element_count() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("model.gguf");
        write_test_gguf(&path)?;

        let gguf = GgufFile::open(&path)?;
        let raw = gguf.tensor_raw("dense")?;

        assert_eq!(raw.info.dtype, GgmlType::F32);
        assert_eq!(raw.info.element_count(), 4);
        assert!(!raw.info.is_quantized());
        assert_eq!(raw.data.len(), 16);
        Ok(())
    }
}
```

- [ ] **Step 2: Run the parser tests and verify they fail**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
cargo test -p voxui-gguf tensor_raw_exposes_dtype_shape_and_bytes_without_dequantizing
```

Expected: compile fails because `tensor_raw`, `element_count`, and `is_quantized` do not exist.

- [ ] **Step 3: Add the raw tensor API**

In `voxui/crates/voxui-gguf/src/types.rs`, add:

```rust
#[derive(Debug, Clone, Copy)]
pub struct RawTensor<'a> {
    pub info: &'a TensorInfo,
    pub data: &'a [u8],
}

impl GgmlType {
    pub fn is_quantized(self) -> bool {
        matches!(self, GgmlType::Q4_0 | GgmlType::Q8_0)
    }
}

impl TensorInfo {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    pub fn is_quantized(&self) -> bool {
        self.dtype.is_quantized()
    }
}
```

In `voxui/crates/voxui-gguf/src/parser.rs`, add this method inside `impl GgufFile`:

```rust
pub fn tensor_raw(&self, name: &str) -> anyhow::Result<RawTensor<'_>> {
    let info = self
        .tensor_info(name)
        .with_context(|| format!("tensor '{}' not found", name))?;
    let data = self
        .tensor_data(name)
        .with_context(|| format!("tensor '{}' data out of bounds", name))?;
    Ok(RawTensor { info, data })
}
```

- [ ] **Step 4: Run the parser tests and commit**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
cargo test -p voxui-gguf tensor_raw
cargo test -p voxui-gguf dense_tensor_info
```

Expected: both tests pass.

Commit:

```powershell
git add voxui/crates/voxui-gguf/Cargo.toml voxui/crates/voxui-gguf/src/types.rs voxui/crates/voxui-gguf/src/parser.rs
git commit -m "feat(gguf): expose raw quantized tensor data"
```

---

### Task 2: Add Runtime Weight Abstraction

**Files:**
- Create: `voxui/crates/voxui-inference/src/weights.rs`
- Modify: `voxui/crates/voxui-inference/src/lib.rs`

- [ ] **Step 1: Write the failing runtime weight tests**

Create `voxui/crates/voxui-inference/src/weights.rs` with this test scaffold:

```rust
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

#[cfg(test)]
mod tests {
    use super::*;

    fn test_weight(device: &Device) -> Result<Tensor> {
        let data = (0..64)
            .map(|v| (v as f32 - 31.0) / 16.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(data, (2, 32), device)
    }

    #[test]
    fn q4_linear_forward_matches_dense_shape_and_reasonable_values() -> Result<()> {
        let device = Device::Cpu;
        let weight = test_weight(&device)?;
        let input = Tensor::from_vec(vec![0.25f32; 64], (2, 32), &device)?;

        let dense = LinearWeight::Dense(weight.clone());
        let q4 = LinearWeight::quantize_for_test(&weight, candle_core::quantized::GgmlDType::Q4_0)?;

        let dense_out = dense.forward(&input)?;
        let q4_out = q4.forward(&input)?;

        assert_eq!(dense_out.dims(), &[2, 2]);
        assert_eq!(q4_out.dims(), &[2, 2]);
        let dense_values = dense_out.to_vec2::<f32>()?;
        let q4_values = q4_out.to_vec2::<f32>()?;
        for (dense_row, q4_row) in dense_values.iter().zip(q4_values.iter()) {
            for (dense_value, q4_value) in dense_row.iter().zip(q4_row.iter()) {
                assert!(
                    (dense_value - q4_value).abs() < 0.50,
                    "dense={dense_value} q4={q4_value}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn dense_runtime_tensor_can_materialize_matching_dtype() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
        let runtime = RuntimeTensor::Dense(tensor);

        let materialized = runtime.to_dense_dtype(DType::F32)?;

        assert_eq!(materialized.dims(), &[2, 2]);
        assert_eq!(materialized.dtype(), DType::F32);
        Ok(())
    }
}
```

Modify `voxui/crates/voxui-inference/src/lib.rs` to include the module:

```rust
mod weights;
pub(crate) use weights::{LinearWeight, RuntimeTensor};
```

- [ ] **Step 2: Run the weight tests and verify they fail**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
cargo test -p voxui-inference weights::tests::q4_linear_forward_matches_dense_shape_and_reasonable_values
```

Expected: compile fails because `LinearWeight`, `RuntimeTensor`, and `quantize_for_test` are not implemented.

- [ ] **Step 3: Implement the minimal runtime weight layer**

Replace the non-test portion of `weights.rs` with:

```rust
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{
    quantized::{self, GgmlDType, QMatMul},
    DType, Device, Module, Tensor,
};
use voxui_gguf::{GgmlType, RawTensor};

#[derive(Clone, Debug)]
pub(crate) enum RuntimeTensor {
    Dense(Tensor),
    Quantized(Arc<QuantizedTensor>),
}

#[derive(Debug)]
pub(crate) struct QuantizedTensor {
    name: String,
    shape: Vec<usize>,
    dtype: GgmlDType,
    raw: Arc<[u8]>,
    device: Device,
}

#[derive(Clone, Debug)]
pub(crate) enum LinearWeight {
    Dense(Tensor),
    Quantized(QMatMul),
}

pub(crate) fn map_ggml_dtype(dtype: GgmlType) -> Result<GgmlDType> {
    match dtype {
        GgmlType::F32 => Ok(GgmlDType::F32),
        GgmlType::F16 => Ok(GgmlDType::F16),
        GgmlType::Q4_0 => Ok(GgmlDType::Q4_0),
        GgmlType::Q8_0 => Ok(GgmlDType::Q8_0),
    }
}

impl RuntimeTensor {
    pub(crate) fn from_raw_quantized(raw: RawTensor<'_>, device: &Device) -> Result<Self> {
        let dtype = map_ggml_dtype(raw.info.dtype)?;
        let shape = raw
            .info
            .shape
            .iter()
            .map(|&dim| dim as usize)
            .collect::<Vec<_>>();
        let raw_bytes: Arc<[u8]> = Arc::from(raw.data.to_vec().into_boxed_slice());
        let tensor = QuantizedTensor {
            name: raw.info.name.clone(),
            shape,
            dtype,
            raw: raw_bytes,
            device: device.clone(),
        };
        Ok(RuntimeTensor::Quantized(Arc::new(tensor)))
    }

    pub(crate) fn to_dense_dtype(&self, dtype: DType) -> Result<Tensor> {
        match self {
            RuntimeTensor::Dense(tensor) => tensor.to_dtype(dtype).map_err(Into::into),
            RuntimeTensor::Quantized(tensor) => tensor.dequantize()?.to_dtype(dtype).map_err(Into::into),
        }
    }

    pub(crate) fn embedding_rows(&self, token_ids: &[u32], dtype: DType) -> Result<Tensor> {
        match self {
            RuntimeTensor::Dense(tensor) => {
                let ids = Tensor::new(token_ids, tensor.device())?;
                tensor.index_select(&ids, 0)?.to_dtype(dtype).map_err(Into::into)
            }
            RuntimeTensor::Quantized(tensor) => tensor.embedding_rows(token_ids, dtype),
        }
    }
}

impl QuantizedTensor {
    fn dequantize(&self) -> Result<Tensor> {
        let qtensor = quantized::ggml_file::qtensor_from_ggml(
            self.dtype,
            self.raw.as_ref(),
            self.shape.clone(),
            &self.device,
        )
        .with_context(|| format!("dequantize quantized tensor {}", self.name))?;
        qtensor.dequantize(&self.device).map_err(Into::into)
    }

    fn embedding_rows(&self, token_ids: &[u32], dtype: DType) -> Result<Tensor> {
        let dense = self.dequantize()?;
        let ids = Tensor::new(token_ids, &self.device)?;
        dense.index_select(&ids, 0)?.to_dtype(dtype).map_err(Into::into)
    }
}

impl LinearWeight {
    pub(crate) fn from_raw_quantized(raw: RawTensor<'_>, device: &Device) -> Result<Self> {
        let dtype = map_ggml_dtype(raw.info.dtype)?;
        let shape = raw
            .info
            .shape
            .iter()
            .map(|&dim| dim as usize)
            .collect::<Vec<_>>();
        let qtensor = quantized::ggml_file::qtensor_from_ggml(dtype, raw.data, shape, device)
            .with_context(|| format!("load quantized linear tensor {}", raw.info.name))?;
        Ok(LinearWeight::Quantized(QMatMul::from_qtensor(qtensor)?))
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            LinearWeight::Dense(weight) => crate::linear(input, weight),
            LinearWeight::Quantized(weight) => {
                let input_dtype = input.dtype();
                let out = weight.forward(&input.to_dtype(DType::F32)?)?;
                out.to_dtype(input_dtype).map_err(Into::into)
            }
        }
    }

    #[cfg(test)]
    fn quantize_for_test(weight: &Tensor, dtype: GgmlDType) -> Result<Self> {
        let qtensor = quantized::QTensor::quantize(weight, dtype)?;
        Ok(Self::Quantized(QMatMul::from_qtensor(qtensor)?))
    }
}
```

- [ ] **Step 4: Run the weight tests and commit**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
cargo test -p voxui-inference weights::tests
```

Expected: both tests pass.

Commit:

```powershell
git add voxui/crates/voxui-inference/src/lib.rs voxui/crates/voxui-inference/src/weights.rs
git commit -m "feat(inference): add quantized runtime weights"
```

---

### Task 3: Add Quantized-Aware Loader Methods

**Files:**
- Modify: `voxui/crates/voxui-inference/src/model_loader.rs`

- [ ] **Step 1: Write loader tests for q4 residency rules**

Add a `#[cfg(test)] mod tests` to `model_loader.rs` with this helper code and tests:

```rust
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
```

Then add these tests before the closing brace of the module:

```rust
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
```

- [ ] **Step 2: Run the loader tests and verify they fail**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
cargo test -p voxui-inference model_loader::tests::dense_loader_rejects_quantized_tensor_names
```

Expected: compile fails because `load_linear_weight` does not exist and `load_tensor` still accepts quantized tensors.

- [ ] **Step 3: Implement loader methods**

In `model_loader.rs`, import runtime weights:

```rust
use crate::{LinearWeight, RuntimeTensor};
```

Add methods:

```rust
pub(crate) fn load_runtime_tensor(&self, name: &str) -> Result<RuntimeTensor> {
    let info = self.tensor_info(name).ok_or_else(|| {
        anyhow::anyhow!("Tensor '{}' not found in GGUF file '{}'", name, self.store.path.display())
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
        anyhow::anyhow!("Tensor '{}' not found in GGUF file '{}'", name, self.store.path.display())
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
```

At the start of `load_tensor_uncached`, add:

```rust
if info.is_quantized() {
    anyhow::bail!(
        "quantized tensor {name} has dtype {}; use load_linear_weight or load_runtime_tensor so it is not cached as a dense resident tensor",
        info.dtype
    );
}
```

- [ ] **Step 4: Run loader tests and commit**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
cargo test -p voxui-inference model_loader::tests
```

Expected: loader tests pass, and no existing dense tensor tests regress.

Commit:

```powershell
git add voxui/crates/voxui-inference/src/model_loader.rs
git commit -m "feat(inference): preserve quantized GGUF tensors in loader"
```

---

### Task 4: Refactor LM And Encoder Paths

**Files:**
- Modify: `voxui/crates/voxui-inference/src/base_lm.rs`
- Modify: `voxui/crates/voxui-inference/src/encoder.rs`

- [ ] **Step 1: Add failing compile-focused tests**

Run the existing tests before editing:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
cargo test -p voxui-inference --test inference_suite matrix_text_inputs_are_sentence_length
```

Expected before editing: test passes on current code. After Task 3 and before this refactor, q4 model paths that hit quantized LM tensors will fail with the new dense-loader error.

- [ ] **Step 2: Update BaseLM field types**

In `base_lm.rs`, change imports:

```rust
use crate::{LinearWeight, RuntimeTensor};
```

Change `TransformerLayer` and `BaseLM` fields:

```rust
struct TransformerLayer {
    q_proj: LinearWeight,
    k_proj: LinearWeight,
    v_proj: LinearWeight,
    o_proj: LinearWeight,
    gate_proj: LinearWeight,
    up_proj: LinearWeight,
    down_proj: LinearWeight,
    input_layernorm: RuntimeTensor,
    post_attention_layernorm: RuntimeTensor,
}

pub struct BaseLM {
    config: BaseLMConfig,
    embed_tokens: Option<RuntimeTensor>,
    layers: Vec<TransformerLayer>,
    norm: RuntimeTensor,
    cos_cache: Tensor,
    sin_cache: Tensor,
    k_cache: Vec<Tensor>,
    v_cache: Vec<Tensor>,
    cache_len: usize,
    device: Device,
}
```

- [ ] **Step 3: Update BaseLM loading**

Replace BaseLM tensor loads:

```rust
let embed_tokens = if loader.has_tensor(&embed_name) {
    Some(loader.load_runtime_tensor(&embed_name)?)
} else {
    None
};
let norm = loader.load_runtime_tensor(&format!("{}.norm.weight", config.prefix))?;
```

For every projection field use:

```rust
q_proj: loader.load_linear_weight(&format!("{prefix}.self_attn.q_proj.weight"))?,
```

For every layer norm field use:

```rust
input_layernorm: loader.load_runtime_tensor(&format!("{prefix}.input_layernorm.weight"))?,
post_attention_layernorm: loader.load_runtime_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
```

- [ ] **Step 4: Update BaseLM math helpers**

Change `rms_norm` signature and materialize the small norm vector per call:

```rust
fn rms_norm(x: &Tensor, weight: &RuntimeTensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let sq = x.sqr()?;
    let mean_sq = sq.mean_keepdim(D::Minus1)?;
    let eps_t = mean_sq
        .zeros_like()?
        .broadcast_add(&Tensor::new(&[eps as f32], mean_sq.device())?)?;
    let norm = (mean_sq + eps_t)?.sqrt()?.recip()?;
    let out = x.broadcast_mul(&norm)?;
    let weight = weight.to_dense_dtype(DType::F32)?;
    let out = out.broadcast_mul(&weight)?;
    out.to_dtype(dtype).map_err(Into::into)
}
```

Change the BaseLM linear helper:

```rust
fn linear(x: &Tensor, weight: &LinearWeight) -> Result<Tensor> {
    weight.forward(x)
}
```

Change embedding:

```rust
let hidden = embed.embedding_rows(token_ids, DType::F32)?;
let hidden = hidden.reshape((1, seq_len, self.config.hidden_size))?;
```

- [ ] **Step 5: Refactor LocalEncoder**

In `encoder.rs`, change fields:

```rust
in_proj: LinearWeight,
in_proj_bias: RuntimeTensor,
special_token: RuntimeTensor,
```

Load them with:

```rust
let in_proj = loader.load_linear_weight("feat_encoder.in_proj.weight")?;
let in_proj_bias = loader.load_runtime_tensor("feat_encoder.in_proj.bias")?;
let special_token = loader.load_runtime_tensor("feat_encoder.special_token")?;
```

In `encode_patches`, remove `self.in_proj.dtype()` usage and use:

```rust
let feat = feat.to_dtype(DType::F32)?;
let flat = feat.reshape((b * t, p, d))?;
let projected = self.in_proj.forward(&flat)?;
let bias = self.in_proj_bias.to_dense_dtype(projected.dtype())?.reshape((1, 1, self.hidden_size))?;
let projected = projected.broadcast_add(&bias)?;
let cls = self
    .special_token
    .to_dense_dtype(projected.dtype())?
    .reshape(&[1, 1, self.hidden_size])?
    .broadcast_as((b * t, 1, self.hidden_size))?;
```

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
cargo test -p voxui-inference --test inference_suite matrix_text_inputs_are_sentence_length
cargo test -p voxui-inference weights::tests
```

Expected: tests pass.

Commit:

```powershell
git add voxui/crates/voxui-inference/src/base_lm.rs voxui/crates/voxui-inference/src/encoder.rs
git commit -m "feat(inference): use quantized weights in LM and encoder"
```

---

### Task 5: Refactor DiT, FSQ, And Engine Projections

**Files:**
- Modify: `voxui/crates/voxui-inference/src/dit.rs`
- Modify: `voxui/crates/voxui-inference/src/fsq.rs`
- Modify: `voxui/crates/voxui-inference/src/engine.rs`

- [ ] **Step 1: Update DiT linear and norm types**

In `dit.rs`, change imports:

```rust
use crate::{LinearWeight as RuntimeLinearWeight, RuntimeTensor};
```

Change the local `LinearWeight` struct and `DiTLayer`:

```rust
struct LinearWeight {
    weight: RuntimeLinearWeight,
    bias: Option<RuntimeTensor>,
}

struct DiTLayer {
    q_proj: RuntimeLinearWeight,
    k_proj: RuntimeLinearWeight,
    v_proj: RuntimeLinearWeight,
    o_proj: RuntimeLinearWeight,
    gate_proj: RuntimeLinearWeight,
    up_proj: RuntimeLinearWeight,
    down_proj: RuntimeLinearWeight,
    input_layernorm: RuntimeTensor,
    post_attention_layernorm: RuntimeTensor,
}
```

Change `final_norm` to `RuntimeTensor`.

- [ ] **Step 2: Update DiT helper functions**

Replace local linear helpers with:

```rust
fn linear(x: &Tensor, w: &LinearWeight) -> Result<Tensor> {
    let out = w.weight.forward(x)?;
    if let Some(ref bias) = w.bias {
        Ok(out.broadcast_add(&bias.to_dense_dtype(out.dtype())?)?)
    } else {
        Ok(out)
    }
}

fn linear_no_bias(x: &Tensor, weight: &RuntimeLinearWeight) -> Result<Tensor> {
    weight.forward(x)
}
```

Change DiT `rms_norm` to accept `&RuntimeTensor` and materialize with `weight.to_dense_dtype(DType::F32)?`, matching the BaseLM helper.

- [ ] **Step 3: Update DiT loading**

Use `loader.load_linear_weight` for all `.weight` fields used by `linear`/`linear_no_bias`, `loader.load_runtime_tensor` for biases and norms:

```rust
let weight = loader.load_linear_weight(&format!("{p}.{name}.weight"))?;
let bias = if loader.has_tensor(&bias_name) {
    Some(loader.load_runtime_tensor(&bias_name)?)
} else {
    None
};
```

For layer projections use:

```rust
q_proj: loader.load_linear_weight(&format!("{lp}.self_attn.q_proj.weight"))?,
```

For norms use:

```rust
input_layernorm: loader.load_runtime_tensor(&format!("{lp}.input_layernorm.weight"))?,
```

- [ ] **Step 4: Refactor FSQ**

In `fsq.rs`, change fields:

```rust
in_proj: LinearWeight,
out_proj: LinearWeight,
```

Load with:

```rust
let in_proj = loader.load_linear_weight("fsq_layer.in_proj.weight")?;
let out_proj = loader.load_linear_weight("fsq_layer.out_proj.weight")?;
```

Forward with:

```rust
let x = self.in_proj.forward(hidden)?;
...
let out = self.out_proj.forward(&quantized)?;
```

Keep the scalar quantization in f32 exactly as it is.

- [ ] **Step 5: Refactor engine projections**

In `engine.rs`, change `LinearProjection`:

```rust
struct LinearProjection {
    weight: LinearWeight,
    bias: Option<RuntimeTensor>,
}
```

Load with:

```rust
weight: loader
    .load_linear_weight(&weight_name)
    .with_context(|| format!("load projection tensor {weight_name}"))?,
bias: if loader.has_tensor(&bias_name) {
    Some(loader.load_runtime_tensor(&bias_name)?)
} else {
    None
},
```

Forward with:

```rust
let out = projection.weight.forward(x)?;
if let Some(bias) = projection.bias.as_ref() {
    out.broadcast_add(&bias.to_dense_dtype(out.dtype())?).map_err(Into::into)
} else {
    Ok(out)
}
```

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
cargo test -p voxui-inference dit_conditioning
cargo test -p voxui-inference --test inference_suite matrix_text_inputs_are_sentence_length
```

Expected: tests pass.

Commit:

```powershell
git add voxui/crates/voxui-inference/src/dit.rs voxui/crates/voxui-inference/src/fsq.rs voxui/crates/voxui-inference/src/engine.rs
git commit -m "feat(inference): use quantized weights in DiT projections"
```

---

### Task 6: Reject Unsupported Quantized VAE Conv Paths

**Files:**
- Modify: `voxui/crates/voxui-inference/src/audiovae.rs`

- [ ] **Step 1: Add an unsupported-op test**

Add a focused unit test in `audiovae.rs` that builds a loader with a q4 `audio_vae.decoder.model.0.weight_v` tensor and asserts the error message contains:

```text
unsupported quantized tensor audio_vae.decoder.model.0.weight_v
```

Add a small test GGUF helper in the `audiovae.rs` test module that writes three tensors:

```text
audio_vae.decoder.model.0.weight_g  F32  [1,1,1]
audio_vae.decoder.model.0.weight_v  Q4_0 [1,1,32]
audio_vae.decoder.model.0.bias      F32  [1]
```

The helper should use the same `write_string`, `align_32`, and little-endian tensor header code shown in Task 3, but with the three tensor names above. The q4 `weight_v` raw data is 18 bytes: two bytes for an f16 scale followed by sixteen quant bytes.

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
cargo test -p voxui-inference audiovae::tests::quantized_conv_weight_is_rejected
```

Expected: test fails because `load_conv` still calls `load_tensor` directly and emits the generic dense-loader message.

- [ ] **Step 3: Add explicit VAE guards**

At the start of `load_conv`, add:

```rust
let g_name = format!("{prefix}.weight_g");
let v_name = format!("{prefix}.weight_v");
let bias_name = format!("{prefix}.bias");
loader.ensure_dense_supported(&g_name, "audio_vae weight_norm")?;
loader.ensure_dense_supported(&v_name, "conv1d")?;
loader.ensure_dense_supported(&bias_name, "conv1d bias")?;
```

Then load using `g_name`, `v_name`, and `bias_name`.

In `infer_decoder_rates`, before loading `weight_v`, add:

```rust
loader.ensure_dense_supported(&name, "audio_vae decoder rate inference")?;
```

- [ ] **Step 4: Run tests and commit**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
cargo test -p voxui-inference audiovae::tests::quantized_conv_weight_is_rejected
cargo test -p voxui-inference --test inference_suite matrix_text_inputs_are_sentence_length
```

Expected: unsupported-op test passes and existing inference test still passes.

Commit:

```powershell
git add voxui/crates/voxui-inference/src/audiovae.rs
git commit -m "feat(inference): reject unsupported quantized VAE convs"
```

---

### Task 7: Add Runtime-Supported Exporter Quantization Profiles

**Files:**
- Modify: `exporter/export_voxcpm.py`
- Modify: `exporter/verify_gguf.py`
- Modify: `exporter/tests/test_export_manifest.py`

- [ ] **Step 1: Add failing exporter tests**

Add these tests to `ExportManifestTests`:

```python
def test_q4_linear_profile_quantizes_supported_linear_roles_only(self):
    quant_args = resolve_quant_args(
        variant="2.0",
        profile="q4-linear",
        quant_lm=None,
        quant_encoder=None,
        quant_dit=None,
        quant_vae=None,
    )
    self.assertEqual(
        quant_args,
        {
            "quant_lm": "q4",
            "quant_encoder": "q4",
            "quant_dit": "q4",
            "quant_vae": "f32",
        },
    )

def test_export_rejects_q4_audio_vae_until_quantized_conv_exists(self):
    main_weights = {"base_lm.norm.weight": np.zeros(2, dtype=np.float32)}
    vae_weights = {"decoder.model.0.weight_v": np.zeros((1, 1, 32), dtype=np.float32)}
    config = {"architecture": "voxcpm2"}

    with TemporaryDirectory() as model_tmp, TemporaryDirectory() as output_tmp:
        model_dir = Path(model_tmp)
        output_dir = Path(output_tmp)
        (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        RecordingWriter.instances = []
        with (
            patch("exporter.export_voxcpm.GGUFWriter", RecordingWriter),
            patch("exporter.export_voxcpm.load_weights", return_value=(main_weights, vae_weights, "safetensors")),
        ):
            with self.assertRaisesRegex(ValueError, "audio_vae q4/q8 export is unsupported"):
                export(
                    model_dir,
                    output_dir,
                    {
                        "quant_lm": "fp16",
                        "quant_encoder": "fp16",
                        "quant_dit": "fp16",
                        "quant_vae": "q4",
                    },
                    "2.0",
                )
```

Add a test that calls the new tensor policy directly:

```python
def test_runtime_supported_policy_keeps_norms_dense_in_q4_lm(self):
    self.assertEqual(resolve_tensor_quantization("base_lm", "base_lm.norm.weight", "q4-lm", "q4"), "fp16")
    self.assertEqual(resolve_tensor_quantization("base_lm", "base_lm.embed_tokens.weight", "q4-lm", "q4"), "q4")
    self.assertEqual(resolve_tensor_quantization("base_lm", "base_lm.layers.0.self_attn.q_proj.weight", "q4-lm", "q4"), "q4")
```

- [ ] **Step 2: Run exporter tests and verify they fail**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI
& C:\Users\Reon\py_env\voxcpm\Scripts\activate.ps1
python -m unittest exporter.tests.test_export_manifest.ExportManifestTests.test_q4_linear_profile_quantizes_supported_linear_roles_only
```

Expected: failure because `q4-linear` and `resolve_tensor_quantization` do not exist.

- [ ] **Step 3: Implement role-based tensor quantization**

In `export_voxcpm.py`, include `"q4-linear"` in `QUANT_PROFILES`.

Add:

```python
LINEAR_WEIGHT_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
    ".in_proj.weight",
    ".cond_proj.weight",
    ".out_proj.weight",
    ".linear_1.weight",
    ".linear_2.weight",
)

EXACT_LINEAR_WEIGHTS = {
    "feat_encoder.in_proj.weight",
    "fsq_layer.in_proj.weight",
    "fsq_layer.out_proj.weight",
    "lm_to_dit_proj.weight",
    "res_to_dit_proj.weight",
    "enc_to_lm_proj.weight",
    "fusion_concat_proj.weight",
    "stop_proj.weight",
    "stop_head.weight",
}

def is_runtime_supported_quantized_tensor(component: str, tensor_name: str) -> bool:
    if component in {BASE_LM, RESIDUAL_LM}:
        return tensor_name.endswith(".embed_tokens.weight") or tensor_name.endswith(LINEAR_WEIGHT_SUFFIXES)
    if component in {FEAT_ENCODER, FEAT_DECODER, PROJECTIONS}:
        return tensor_name in EXACT_LINEAR_WEIGHTS or tensor_name.endswith(LINEAR_WEIGHT_SUFFIXES)
    return False

def resolve_tensor_quantization(component: str, tensor_name: str, quant_profile: str, component_quant: str) -> str:
    if component_quant not in {"q4", "q8"}:
        return component_quant
    if quant_profile in {"q4-lm", "q4-linear"}:
        return component_quant if is_runtime_supported_quantized_tensor(component, tensor_name) else "fp16"
    return component_quant
```

Update `profile_default_quant_args`:

```python
if profile == "q4-linear":
    return {
        "quant_lm": "q4",
        "quant_encoder": "q4",
        "quant_dit": "q4",
        "quant_vae": "f32" if variant == "2.0" else "fp16",
    }
```

At the start of `write_base_gguf`, reject unsupported VAE quantization:

```python
if quant_args.get("quant_vae") in {"q4", "q8"}:
    raise ValueError("audio_vae q4/q8 export is unsupported until quantized conv inference exists")
```

In the tensor write loop, replace component-wide quant selection with:

```python
tensor_quant_name = resolve_tensor_quantization(component, tensor_name, quant_profile, quant_name)
quant_fn, ggml_dtype = QUANT_MAP[tensor_quant_name]
writer.add_tensor(tensor_name, quant_fn(arr), list(arr.shape), ggml_dtype)
```

- [ ] **Step 4: Add dtype counts to verify_gguf**

In `verify_gguf.py`, collect dtype counts in `verify_file`:

```python
dtype_counts = {}
...
dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
...
print("\nTensor dtype counts:")
for dtype_name, count in sorted(dtype_counts.items()):
    print(f"  {dtype_name}: {count}")
```

- [ ] **Step 5: Run exporter tests and commit**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI
& C:\Users\Reon\py_env\voxcpm\Scripts\activate.ps1
python -m unittest exporter.tests.test_export_manifest
```

Expected: all exporter tests pass.

Commit:

```powershell
git add exporter/export_voxcpm.py exporter/verify_gguf.py exporter/tests/test_export_manifest.py
git commit -m "feat(exporter): add runtime-supported q4 profiles"
```

---

### Task 8: Improve CUDA VRAM Report Artifacts

**Files:**
- Modify: `voxui/crates/voxui-inference/tests/cuda_vram_report.rs`

- [ ] **Step 1: Refactor report result data**

Add:

```rust
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct ModelVramReport {
    model_name: String,
    baseline: MemorySample,
    after_load: MemorySample,
    after_synth: MemorySample,
    peak_process_delta_mib: Option<i64>,
}
```

Change `run_model_report` to return `Result<Option<ModelVramReport>>`. Return `Ok(None)` when `model.gguf` is missing. Return `Ok(Some(report))` after synthesis.

- [ ] **Step 2: Add child-process isolation**

Add constants and helpers:

```rust
#[cfg(feature = "cuda")]
const CHILD_MODEL_ENV: &str = "VOXUI_VRAM_CHILD_MODEL";

#[cfg(feature = "cuda")]
const CHILD_JSON_PREFIX: &str = "VOXUI_VRAM_JSON=";
```

Add a hidden child test:

```rust
#[test]
#[cfg(feature = "cuda")]
fn vram_report_child() -> Result<()> {
    let Some(model_name) = std::env::var(CHILD_MODEL_ENV).ok() else {
        eprintln!("[SKIP] child env not set");
        return Ok(());
    };
    let Some(report) = run_model_report(&model_name)? else {
        println!("{CHILD_JSON_PREFIX}{{\"model_name\":\"{model_name}\",\"skipped\":true}}");
        return Ok(());
    };
    println!(
        "{CHILD_JSON_PREFIX}{}",
        serde_json::json!({
            "model_name": report.model_name,
            "peak_process_delta_mib": report.peak_process_delta_mib,
            "baseline_process_mib": report.baseline.process.map(|v| v.used_mib),
            "after_load_process_mib": report.after_load.process.map(|v| v.used_mib),
            "after_synth_process_mib": report.after_synth.process.map(|v| v.used_mib),
        })
    );
    Ok(())
}
```

In the parent test, run each model through the current test executable:

```rust
#[cfg(feature = "cuda")]
fn run_child_model_report(model_name: &str) -> Result<serde_json::Value> {
    let output = Command::new(std::env::current_exe()?)
        .args(["--exact", "vram_report_child", "--nocapture", "--test-threads=1"])
        .env(CHILD_MODEL_ENV, model_name)
        .output()
        .with_context(|| format!("run child VRAM report for {model_name}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        anyhow::bail!("child VRAM report failed for {model_name}\nstdout:\n{stdout}\nstderr:\n{stderr}");
    }
    let json_line = stdout
        .lines()
        .find_map(|line| line.strip_prefix(CHILD_JSON_PREFIX))
        .ok_or_else(|| anyhow::anyhow!("child VRAM report did not emit JSON for {model_name}"))?;
    serde_json::from_str(json_line).map_err(Into::into)
}
```

- [ ] **Step 3: Write JSON and Markdown artifacts**

Add:

```rust
#[cfg(feature = "cuda")]
fn artifact_dir() -> PathBuf {
    repo_root().join("voxui").join("target").join("cuda-vram-report")
}

#[cfg(feature = "cuda")]
fn write_report_artifacts(reports: &[serde_json::Value]) -> Result<()> {
    let dir = artifact_dir();
    std::fs::create_dir_all(&dir)?;
    let json_path = dir.join("voxcpm-vram-report.json");
    let md_path = dir.join("voxcpm-vram-report.md");
    std::fs::write(&json_path, serde_json::to_string_pretty(reports)?)?;

    let mut markdown = String::from("# VoxCPM CUDA VRAM Report\n\n");
    markdown.push_str("| model | peak process delta MiB | skipped |\n");
    markdown.push_str("| --- | ---: | --- |\n");
    for report in reports {
        markdown.push_str(&format!(
            "| {} | {} | {} |\n",
            report["model_name"].as_str().unwrap_or("<unknown>"),
            report["peak_process_delta_mib"].as_i64().map(|v| v.to_string()).unwrap_or_else(|| "n/a".to_string()),
            report["skipped"].as_bool().unwrap_or(false),
        ));
    }
    markdown.push_str("\n`cargo test` captures stdout by default. Use `-- --nocapture --test-threads=1` for console output.\n");
    std::fs::write(&md_path, markdown)?;
    println!("wrote VRAM artifacts: {} and {}", json_path.display(), md_path.display());
    Ok(())
}
```

- [ ] **Step 4: Assert q4 uses less process VRAM when both measurements are available**

Change the parent test body to:

```rust
#[test]
#[cfg(feature = "cuda")]
fn reports_voxcpm2_cuda_vram_for_fp16_and_q4_lm() -> Result<()> {
    assert_eq!(CHINESE_20.chars().count(), 20);
    let fp16 = run_child_model_report("voxcpm2-fp16")?;
    let q4 = run_child_model_report("voxcpm2-q4-lm")?;
    let reports = vec![fp16.clone(), q4.clone()];
    write_report_artifacts(&reports)?;

    if fp16["skipped"].as_bool().unwrap_or(false) || q4["skipped"].as_bool().unwrap_or(false) {
        eprintln!("[SKIP] one or more model bundles are missing");
        return Ok(());
    }
    if let (Some(fp16_peak), Some(q4_peak)) = (
        fp16["peak_process_delta_mib"].as_i64(),
        q4["peak_process_delta_mib"].as_i64(),
    ) {
        assert!(
            q4_peak < fp16_peak,
            "expected q4 peak process VRAM ({q4_peak} MiB) to be below fp16 ({fp16_peak} MiB)"
        );
    }
    Ok(())
}
```

- [ ] **Step 5: Run report unit tests and commit**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
cargo test -p voxui-inference --test cuda_vram_report memory_formatting
```

Expected: unit tests pass without requiring CUDA.

Commit:

```powershell
git add voxui/crates/voxui-inference/tests/cuda_vram_report.rs
git commit -m "test(inference): write isolated CUDA VRAM artifacts"
```

---

### Task 9: Full Verification

**Files:**
- No planned file changes.

- [ ] **Step 1: Run Rust CPU-safe tests**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
cargo test -p voxui-gguf
cargo test -p voxui-inference weights::tests
cargo test -p voxui-inference model_loader::tests
cargo test -p voxui-inference --test inference_suite matrix_text_inputs_are_sentence_length
```

Expected: all commands pass.

- [ ] **Step 2: Run exporter tests**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI
& C:\Users\Reon\py_env\voxcpm\Scripts\activate.ps1
python -m unittest exporter.tests.test_export_manifest
```

Expected: all exporter tests pass.

- [ ] **Step 3: Run CUDA q4 inference suite**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "$env:CUDA_PATH\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
$env:CUDA_COMPUTE_CAP = "89"
$env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"
cargo test -p voxui-inference --features cuda --test inference_suite q4_lm_cuda -- --nocapture --test-threads=1
```

Expected output includes:

```text
test voxcpm05_q4_lm_cuda ... ok
test voxcpm15_q4_lm_cuda ... ok
test voxcpm2_q4_lm_cuda ... ok
```

- [ ] **Step 4: Run CUDA VRAM report**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "$env:CUDA_PATH\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
$env:CUDA_COMPUTE_CAP = "89"
$env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"
cargo test -p voxui-inference --features cuda --test cuda_vram_report reports_voxcpm2_cuda_vram_for_fp16_and_q4_lm -- --nocapture --test-threads=1
```

Expected:

```text
test reports_voxcpm2_cuda_vram_for_fp16_and_q4_lm ... ok
wrote VRAM artifacts:
```

Then inspect:

```powershell
Get-Content D:\Sandbox_Share\VoxUI\voxui\target\cuda-vram-report\voxcpm-vram-report.md
```

Expected: `voxcpm2-q4-lm` peak process delta is lower than `voxcpm2-fp16`.

- [ ] **Step 5: Confirm final status**

Run:

```powershell
cd D:\Sandbox_Share\VoxUI
git status --short --branch
git log --oneline -5
```

Expected: clean working tree after all task commits. If this command reports changes, stop and inspect them before reporting completion.

## Self-Review Notes

- Spec coverage: raw GGUF access is Task 1; quantized runtime residency is Tasks 2-6; exporter role policy is Task 7; VRAM artifact and child process measurement is Task 8; full CUDA verification is Task 9.
- Placeholder scan: this plan uses exact files, commands, expected results, and code snippets for each edit.
- Type consistency: `RuntimeTensor`, `LinearWeight`, `load_runtime_tensor`, `load_linear_weight`, and `ensure_dense_supported` are introduced before downstream model refactors use them.
