# VoxUI Implementation Plan

**Spec**: `docs/superpowers/specs/2026-05-03-voxui-tts-design.md`
**Scope**: Python GGUF exporter + Rust TUI TTS application

---

## Phase 1: Python GGUF Exporter

### Task 1.1: GGUF Binary Writer (`exporter/gguf_writer.py`)

Write a standalone GGUF v3 binary format writer.

**Implementation**:
- GGUF header: magic `GGUF` (0x46475547), version=3, tensor_count, metadata_kv_count
- Metadata KV: write key-value pairs (string, uint32, float32, etc.)
- Tensor info: name, n_dims, shape, type enum, offset
- Tensor data: aligned to 32 bytes, written after all tensor infos
- Support tensor types: F16 (1), F32 (0), Q4_0 (2), Q8_0 (8)
  - Note: FP8 E4M3 is not a standard GGUF type. We'll use Q8_0 as the closest 8-bit format, or define a custom type. Decision: use Q8_0 block format for "FP8" since ggml supports it natively.

**GGUF type enums** (from ggml):
```
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
```

**Verification**: Write a test that creates a small GGUF file, reads it back, and validates header/metadata/tensors.

**Files**: `exporter/gguf_writer.py`

---

### Task 1.2: Quantization Module (`exporter/quantize.py`)

Implement quantization functions.

**Implementation**:
- `quantize_fp16(tensor: np.ndarray) -> bytes`: Cast float32 to float16, return raw bytes
- `quantize_q8_0(tensor: np.ndarray) -> bytes`: Block quantize to Q8_0 format
  - Block size = 32
  - Per-block: compute scale = max(abs(block)) / 127, store as f16
  - Quantize: round(x / scale), clamp to [-128, 127], store as int8
  - Layout per block: 2 bytes (f16 scale) + 32 bytes (int8 data) = 34 bytes
- `quantize_q4_0(tensor: np.ndarray) -> bytes`: Block quantize to Q4_0 format
  - Block size = 32
  - Per-block: compute scale = max(abs(block)) / 7 (Q4 = 4-bit signed, range [-8, 7])
  - Quantize to 4-bit, pack two values per byte (low nibble first)
  - Layout per block: 2 bytes (f16 scale) + 16 bytes (packed 4-bit) = 18 bytes

**Verification**: Quantize a known tensor, dequantize, check error is within expected bounds (Q4: ~5% max error, Q8: ~0.5%).

**Files**: `exporter/quantize.py`

---

### Task 1.3: Main Export Script (`exporter/export_voxcpm.py`)

The CLI entry point that loads VoxCPM weights and exports to multi-file GGUF.

**Weight key prefix mapping** (from VoxCPM2Model source):

| GGUF File | Source Key Prefix | Notes |
|-----------|-------------------|-------|
| `base_lm.gguf` | `base_lm.` | From `model.safetensors` |
| `residual_lm.gguf` | `residual_lm.` | From `model.safetensors` |
| `encoder.gguf` | `feat_encoder.` | From `model.safetensors` |
| `dit.gguf` | `feat_decoder.` | From `model.safetensors`, includes CFM components |
| `audiovae.gguf` | `audio_vae.` → loaded from `audiovae.pth` separately | Keys in `audiovae.pth` have NO `audio_vae.` prefix |
| `projections.gguf` | `fsq_layer.`, `enc_to_lm_proj.`, `lm_to_dit_proj.`, `res_to_dit_proj.`, `fusion_concat_proj.`, `stop_proj.`, `stop_head.` | Small tensors, bundle together |

Note: The spec design has 5 GGUF files but the actual model has additional small modules (projections, FSQ, stop predictor). These are bundled into a `projections.gguf` file since they're all small.

**Implementation**:
1. Parse CLI args (argparse): `--model-dir`, `--output-dir`, `--lora-dir`, `--quant-lm`, `--quant-encoder`, `--quant-dit`, `--quant-vae`
2. Load weights:
   - `model.safetensors` via `safetensors.torch.load_file()`
   - `audiovae.pth` via `torch.load()`
   - `config.json` for architecture parameters
3. For VoxCPM-0.5B: load from `pytorch_model.bin` instead
4. Partition keys by prefix into component groups
5. For each component group:
   - Create GGUF writer
   - Write metadata (architecture, component, quantization, model params from config.json)
   - For each tensor: rename key (strip source prefix, add GGUF prefix), quantize, write
6. Key renaming: `base_lm.model.layers.{i}.self_attn.q_proj.weight` → `base_lm.layers.{i}.self_attn.q_proj.weight` (strip `.model.` from MiniCPM submodules)

**Config.json fields to extract**:
- `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`
- `intermediate_size`, `rms_norm_eps`, `rope_theta`, `vocab_size`
- `max_position_embeddings`, `scale_emb`, `scale_depth`
- `residual_lm_no_rope`, `dit_hidden_size`, `dit_num_layers`, `dit_num_heads`
- `latent_dim`, `fsq_scale`, `patch_size`, `sample_rate`
- `rope_scaling` (for LongRope short_factor/long_factor arrays)

**Verification**: Run export on VoxCPM2, check output files exist, have correct metadata, tensor count matches expected.

**Files**: `exporter/export_voxcpm.py`

---

### Task 1.4: LoRA Export

Export LoRA adapter weights as a separate GGUF file.

**Implementation**:
1. Load `lora_weights.safetensors` from `--lora-dir`
2. Load `lora_config.json` for rank, alpha, target_modules
3. Keys in lora safetensors: `{module_path}.lora_A`, `{module_path}.lora_B`
   - e.g. `base_lm.model.layers.0.self_attn.q_proj.lora_A`
4. Rename keys: strip `.model.`, keep component prefix
5. Write to `lora_{name}.gguf` with metadata: rank, alpha, target_modules, component
6. Always FP16 (LoRA matrices are small)

**Verification**: Check exported LoRA file has correct tensor count (2 per target layer per target module).

**Files**: `exporter/export_voxcpm.py` (add `--lora-dir` handling)

---

### Task 1.5: Export Verification Script

A small script that loads exported GGUF files and prints metadata + tensor info for manual verification.

**Files**: `exporter/verify_gguf.py`

---

### Phase 1 Checkpoint

At this point, run the exporter on all three model variants (VoxCPM-0.5B, VoxCPM1.5, VoxCPM2) with LoRA, verify outputs. **STOP for review before continuing to Phase 2.**

---

## Phase 2: Rust Project Setup + GGUF Parser

### Task 2.1: Cargo Workspace Setup

**Implementation**:
- Create `voxui/Cargo.toml` workspace with 4 crates
- Create minimal `src/lib.rs` or `src/main.rs` for each crate
- Add shared dependencies to workspace `Cargo.toml`
- Verify `cargo check` passes

**Crate dependency graph**:
```
voxui-app → voxui-inference → voxui-gguf
voxui-app → voxui-audio
```

**Files**:
```
voxui/Cargo.toml
voxui/crates/voxui-gguf/Cargo.toml + src/lib.rs
voxui/crates/voxui-inference/Cargo.toml + src/lib.rs
voxui/crates/voxui-audio/Cargo.toml + src/lib.rs
voxui/crates/voxui-app/Cargo.toml + src/main.rs
```

---

### Task 2.2: GGUF Parser (`voxui-gguf`)

Parse GGUF v3 files and provide tensor access.

**Implementation**:
```rust
pub struct GgufFile {
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: HashMap<String, TensorInfo>,
    mmap: Mmap,
}

pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64,  // offset into mmap data section
}

pub enum GgmlType { F32, F16, Q4_0, Q8_0 }
pub enum MetadataValue { String(String), Uint32(u32), Float32(f32), ... }

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self>;
    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>>;  // dequantize to f32
    pub fn get_tensor_f16(&self, name: &str) -> Result<Vec<half::f16>>;
}
```

- Memory-map the file with `memmap2`
- Parse header: magic, version, tensor_count, kv_count
- Parse metadata KV pairs
- Parse tensor infos
- Dequantize functions: `dequant_q4_0`, `dequant_q8_0`, `f16_to_f32`

**Dependencies**: `memmap2`, `half`, `byteorder`

**Verification**: Load an exported GGUF file, print metadata, read a tensor, verify values match Python source.

**Files**: `voxui/crates/voxui-gguf/src/lib.rs` (may split into `parser.rs`, `dequant.rs`)

---

### Phase 2 Checkpoint

Verify GGUF parser can load all exported files correctly. **STOP for review.**

---

## Phase 3: Inference Engine (`voxui-inference`)

This is the largest phase. Each task is a module within the crate.

### Task 3.1: Tokenizer Integration

Load HuggingFace tokenizer and provide encode/decode.

**Implementation**:
```rust
pub struct VoxTokenizer {
    tokenizer: tokenizers::Tokenizer,
}
impl VoxTokenizer {
    pub fn from_dir(model_dir: &Path) -> Result<Self>;
    pub fn encode(&self, text: &str) -> Vec<u32>;
    pub fn decode(&self, ids: &[u32]) -> String;
}
```

**Dependencies**: `tokenizers`

**Files**: `voxui/crates/voxui-inference/src/tokenizer.rs`

---

### Task 3.2: ggml FFI Layer

Set up ggml C library compilation and create safe Rust wrappers.

**Implementation**:
- `build.rs`: Download/compile ggml from source (or use `ggml-sys` crate)
- Feature flag `cuda`: link `ggml-cuda`
- Safe wrappers for:
  - `ggml_context` creation/destruction
  - Tensor creation (1D, 2D, 3D, 4D)
  - Operations: `matmul`, `add`, `mul`, `rms_norm`, `silu`, `rope`, `softmax`
  - Graph computation: `ggml_build_forward`, `ggml_graph_compute`
  - KV cache: pre-allocated tensor pair per layer

**Decision**: Use `ggml` crate from crates.io if adequate, else write minimal FFI bindings. Check crate quality first.

**Files**: `voxui/crates/voxui-inference/src/ggml_ffi.rs`, `voxui/build.rs`

---

### Task 3.3: Base LM Forward Pass

Implement MiniCPM-4 transformer forward pass using ggml.

**Implementation**:
- Load weights from `base_lm.gguf` into ggml tensors
- Build computation graph for single-token forward:
  1. Embed token: `embed_tokens[token_id]` × `scale_emb`
  2. For each layer:
     - RMSNorm (input_layernorm)
     - GQA attention:
       - Q = x @ q_proj, K = x @ k_proj, V = x @ v_proj
       - Reshape for multi-head (GQA: num_heads=16, num_kv_heads=2)
       - RoPE positional encoding (with LongRope scaling if applicable)
       - Update KV cache at current position
       - Attention: softmax(Q @ K^T / sqrt(d_k)) @ V
       - Output: attn @ o_proj
     - Residual connection (with scale_depth scaling)
     - RMSNorm (post_attention_layernorm)
     - FFN: gate = silu(x @ gate_proj) * (x @ up_proj); out = gate @ down_proj
     - Residual connection (with scale_depth scaling)
  3. Final RMSNorm
  4. LM head (tied to embed_tokens or separate)
- Autoregressive loop: generate tokens until stop_predictor fires or max_length

**LongRope implementation**:
- `rope_scaling.type = "longrope"`
- `short_factor` array: per-dimension frequency scaling for positions < original_max_position
- `long_factor` array: for positions >= original_max_position
- `freq_i = base_freq / (factor[i] ** (2i/d))` where factor comes from short or long array

**Stop predictor**:
```rust
fn should_stop(hidden: &Tensor) -> bool {
    let x = silu(hidden @ stop_proj);
    let logits = x @ stop_head; // shape [2]
    logits[1] > logits[0]  // argmax == 1 means stop
}
```

**Verification**: Load VoxCPM2 base_lm.gguf, run forward on a test token sequence, compare hidden states with Python reference (within tolerance for quantized weights).

**Files**: `voxui/crates/voxui-inference/src/base_lm.rs`

---

### Task 3.4: Scalar Quantization Layer (FSQ)

**Implementation**:
```rust
fn scalar_quantize(hidden: &Tensor, in_proj_weight: &Tensor, scale: f32) -> Tensor {
    let x = hidden.matmul(&in_proj_weight.transpose());
    let x = x.tanh();
    let x = (x * scale).round() / scale;
    x
}
```
Scale is typically 9.0 (from config: `fsq_scale`).

**Files**: `voxui/crates/voxui-inference/src/fsq.rs`

---

### Task 3.5: Residual LM Forward Pass

Same architecture as Base LM but:
- Fewer layers (from config)
- Optionally no RoPE (`residual_lm_no_rope` flag)
- LoRA applied dynamically

**Implementation**: Reuse Base LM code with configuration flags.

**Files**: `voxui/crates/voxui-inference/src/residual_lm.rs` (or parameterize `base_lm.rs`)

---

### Task 3.6: LoRA Runtime Loading

**Implementation**:
```rust
pub struct LoraAdapter {
    layers: HashMap<String, (Tensor, Tensor)>,  // name -> (lora_A, lora_B)
    alpha: f32,
    rank: u32,
}

impl LoraAdapter {
    pub fn load(path: &Path) -> Result<Self>;
    pub fn apply(&self, name: &str, base_output: &Tensor, input: &Tensor) -> Tensor {
        if let Some((a, b)) = self.layers.get(name) {
            let scaling = self.alpha / self.rank as f32;
            let lora_out = input.matmul(&a.transpose()).matmul(&b.transpose());
            base_output + lora_out * scaling
        } else {
            base_output.clone()
        }
    }
}
```

**Verification**: Compare LoRA-applied output with Python reference.

**Files**: `voxui/crates/voxui-inference/src/lora.rs`

---

### Task 3.7: DiT + CFM Solver (candle)

Implement the diffusion transformer and Euler ODE solver using candle.

**Implementation**:
- Load weights from `dit.gguf` into candle tensors
- Time embedding: sinusoidal → MLP (2 layers)
- DiT transformer layers with time conditioning
- CFM Euler solver:
  ```rust
  fn euler_solve(dit: &DiT, cond: &Tensor, steps: usize, cfg_value: f32) -> Tensor {
      let dt = 1.0 / steps as f32;
      let mut x = Tensor::randn(...); // initial noise
      for i in 0..steps {
          let t = i as f32 * dt;
          // Classifier-free guidance
          let v_cond = dit.forward(x, cond, t);
          let v_uncond = dit.forward(x, null_cond, t);
          let v = v_uncond + cfg_value * (v_cond - v_uncond);
          x = x + v * dt;
      }
      x
  }
  ```
- Projection layers: `lm_to_dit_proj`, `res_to_dit_proj`

**Dependencies**: `candle-core`, `candle-nn`

**Files**: `voxui/crates/voxui-inference/src/dit.rs`

---

### Task 3.8: AudioVAE Decoder (candle)

Decode latent patches to PCM waveform.

**Implementation**:
- Load weights from `audiovae.gguf` into candle tensors
- Decoder architecture: transposed conv blocks with Snake activation
- Snake activation: `x + (1/alpha) * sin(alpha * x)^2` where alpha is learnable
- Weight normalization: `weight = g * v / ||v||`
- Output: raw PCM f32 samples at model's sample rate

**Files**: `voxui/crates/voxui-inference/src/audiovae.rs`

---

### Task 3.9: Full Pipeline Integration

Wire all components into a single inference function.

**Implementation**:
```rust
pub struct VoxCPMEngine {
    base_lm: BaseLM,
    residual_lm: ResidualLM,
    fsq: FSQLayer,
    dit: DiT,
    vae: AudioVAE,
    tokenizer: VoxTokenizer,
    lora: Option<LoraAdapter>,
    projections: Projections,
}

impl VoxCPMEngine {
    pub fn load(model_dir: &Path, device: Device) -> Result<Self>;
    pub fn load_lora(&mut self, path: &Path) -> Result<()>;
    pub fn unload_lora(&mut self);
    pub fn synthesize(&self, text: &str, progress: impl Fn(f32)) -> Result<Vec<f32>>;
}
```

`synthesize` follows the pipeline: tokenize → base_lm (autoregressive) → FSQ → residual_lm (with LoRA) → projections → DiT+CFM → VAE → PCM.

Progress callback fires after each autoregressive step.

**Verification**: Generate audio from text, save as WAV, listen. Compare spectrograms with Python reference.

**Files**: `voxui/crates/voxui-inference/src/lib.rs` (or `engine.rs`)

---

### Phase 3 Checkpoint

Full inference pipeline works end-to-end on CPU. Audio output is intelligible. **STOP for review.**

---

## Phase 4: Audio Playback (`voxui-audio`)

### Task 4.1: Device Enumeration

**Implementation**:
```rust
pub struct AudioSystem {
    hosts: Vec<HostInfo>,
}
pub struct HostInfo {
    pub name: String,
    pub host_id: HostId,
    pub devices: Vec<DeviceInfo>,
}
pub struct DeviceInfo {
    pub name: String,
    pub device: Device,
    pub sample_rates: Vec<u32>,
}

impl AudioSystem {
    pub fn new() -> Self;  // enumerate all hosts/devices
    pub fn host_names(&self) -> Vec<&str>;
    pub fn device_names(&self, host: &str) -> Vec<&str>;
}
```

**Files**: `voxui/crates/voxui-audio/src/lib.rs`

---

### Task 4.2: PCM Playback

**Implementation**:
```rust
pub struct AudioPlayer {
    stream: Option<Stream>,
}
impl AudioPlayer {
    pub fn new(host: &str, device: &str, sample_rate: u32) -> Result<Self>;
    pub fn play(&mut self, samples: Vec<f32>, on_complete: impl Fn()) -> Result<()>;
    pub fn stop(&mut self);
}
```

- Write samples to ring buffer
- cpal stream callback reads from buffer
- Signal completion when buffer exhausted

**Dependencies**: `cpal`

**Files**: `voxui/crates/voxui-audio/src/lib.rs`

---

### Phase 4 Checkpoint

Can play generated PCM through selected audio device. **STOP for review.**

---

## Phase 5: TUI Application (`voxui-app`)

### Task 5.1: Basic TUI Shell

Set up ratatui + crossterm, render the main layout (title, empty history, progress bar, input box, status bar). Handle basic keyboard events (typing, Enter, Esc, F2).

**Files**: `voxui/crates/voxui-app/src/main.rs`, `src/ui.rs`, `src/app.rs`

---

### Task 5.2: Input Box + TTS Queue

Implement text input with cursor, Enter to submit, max character limit. TTS requests go into a VecDeque.

**Files**: `voxui/crates/voxui-app/src/input.rs`, `src/app.rs`

---

### Task 5.3: History List

Scrollable list widget showing TTS history entries with timestamp, text, status icon.

**Files**: `voxui/crates/voxui-app/src/history.rs`

---

### Task 5.4: Progress Bar

Gauge widget showing generation progress, updated via channel from inference thread.

**Files**: `voxui/crates/voxui-app/src/ui.rs`

---

### Task 5.5: Settings Popup

Modal popup with dropdown selections for model path, LoRA, backend, audio host, audio device, max chars. Tab to navigate, Space to open dropdown, Enter to confirm.

**Files**: `voxui/crates/voxui-app/src/settings.rs`

---

### Task 5.6: Async Integration

Wire inference engine and audio player into the TUI event loop via tokio channels.

**Implementation**:
- Main thread: ratatui event loop, processes crossterm events + channel messages
- Spawn inference on `spawn_blocking` when queue has items
- Inference sends `Progress(f32)` and `Complete(Vec<f32>)` via channel
- On `Complete`: send PCM to audio player, update history status
- Ctrl+C: cancel inference (via cancellation token/flag)

**Files**: `voxui/crates/voxui-app/src/main.rs`, `src/app.rs`

---

### Task 5.7: Status Bar

Display model info (read from GGUF metadata), backend, audio device. Update when settings change.

**Files**: `voxui/crates/voxui-app/src/ui.rs`

---

### Phase 5 Checkpoint

Full application works end-to-end: type text → generate → play audio. Settings popup works. **STOP for review.**

---

## Phase 6: Polish & Multi-Variant Support

### Task 6.1: VoxCPM-0.5B and VoxCPM1.5 Support

Ensure exporter and inference engine handle all three variants:
- VoxCPM-0.5B: `pytorch_model.bin`, patch_size=2, 16kHz, older architecture
- VoxCPM1.5: `model.safetensors`, patch_size=4, 44.1kHz
- VoxCPM2: `model.safetensors`, 48kHz, 2B params

Differences to handle in inference:
- Sample rate affects cpal output stream config
- `patch_size` affects DiT processing
- Architecture differences in config.json

**Files**: Exporter + inference engine

---

### Task 6.2: Error Handling & Edge Cases

- Model loading failures → status bar error
- CUDA unavailable → CPU fallback with warning
- Audio device errors → allow re-selection
- Generation NaN/divergence → mark as error, continue queue
- LoRA mismatch → reject with error message

---

### Task 6.3: Config Persistence

Save/load user settings (last model, device, etc.) to a JSON config file.

**Files**: `voxui/crates/voxui-app/src/config.rs`

---

## Execution Notes

- **Phase 1** is fully independent and should be completed first (prerequisite for all Rust work)
- **Phases 2-3** are sequential (GGUF parser → inference engine)
- **Phase 4** (audio) can be developed in parallel with Phase 3 tasks 3.7-3.9
- **Phase 5** (TUI) depends on Phases 3+4 for integration, but UI shell (5.1-5.5) can start in parallel
- **Phase 6** is polish after core functionality works

Total estimated tasks: 22
