# VoxUI: TUI TTS Application for Streamers

## Overview

VoxUI is a terminal-based TTS application for streamers, powered by VoxCPM models. It consists of two subsystems:

1. **Python GGUF Exporter** — Converts VoxCPM model weights (safetensors/PyTorch) to multi-file GGUF format with per-component quantization
2. **Rust TUI Application** — ratatui-based interface with ggml+candle hybrid inference, cpal audio playback

## Project Structure

```
VoxUI/
├── VoxCPM/                    # Existing: model weights + VoxCPM Python package
├── exporter/                  # Python GGUF exporter
│   ├── export_voxcpm.py       # Main export script (CLI entry point)
│   ├── gguf_writer.py         # GGUF binary format writer
│   ├── quantize.py            # FP16/FP8/Q4 quantization implementations
│   └── lora_merge.py          # LoRA weight loading (no merge — export as separate file)
├── voxui/                     # Rust TUI application (cargo workspace)
│   ├── Cargo.toml             # Workspace root
│   ├── crates/
│   │   ├── voxui-gguf/        # GGUF parser + tensor loading
│   │   ├── voxui-inference/   # Inference engine (ggml LM + candle DiT/VAE)
│   │   ├── voxui-audio/       # cpal audio device enumeration + playback
│   │   └── voxui-app/         # ratatui TUI application
│   └── build.rs               # ggml C library compilation
└── models/                    # Exported GGUF files (output directory)
```

---

## Part 1: GGUF Exporter

### Multi-File GGUF Scheme

Each VoxCPM component is exported to its own GGUF file, allowing independent quantization levels:

| File | Content | Recommended Precision |
|------|---------|----------------------|
| `base_lm.gguf` | Base LM (MiniCPM-4 backbone) — all transformer layers | Q4 / FP8 / FP16 |
| `residual_lm.gguf` | Residual LM — acoustic refinement layers | Q4 / FP8 / FP16 |
| `encoder.gguf` | Local Encoder — non-causal transformer | FP8 / FP16 |
| `dit.gguf` | Local DiT + CFM time embeddings | FP16 (diffusion is quantization-sensitive) |
| `audiovae.gguf` | AudioVAE decoder (encoder optional) | FP16 (waveform reconstruction needs precision) |
| `lora_<name>.gguf` | LoRA adapter weights (optional) | FP16 |

### Supported Quantization Levels

- **FP16**: 16-bit float, no precision loss, largest file size
- **FP8** (E4M3): 8-bit float, near-lossless for most tensors
- **Q4** (4-bit integer with block scaling): Aggressive compression, suitable for LM components

### Export CLI

```bash
python exporter/export_voxcpm.py \
    --model-dir VoxCPM/models/VoxCPM2 \
    --output-dir models/voxcpm2-mixed \
    --lora-dir VoxCPM/ft2/latest \        # Optional: export LoRA as separate file
    --quant-lm q4 \                        # Base LM + Residual LM quantization
    --quant-encoder fp8 \                  # Encoder quantization
    --quant-dit fp16 \                     # DiT quantization
    --quant-vae fp16                       # AudioVAE quantization
```

### GGUF Metadata Per File

Each GGUF file is self-describing with metadata keys:

```
voxcpm.architecture    = "voxcpm2"           # or "voxcpm"
voxcpm.component       = "base_lm"           # base_lm|residual_lm|encoder|dit|audiovae|lora
voxcpm.version         = 2                   # Model version (0.5, 1.5, 2)
voxcpm.quantization    = "q4"                # fp16|fp8|q4

# Architecture-specific (varies by component):
voxcpm.hidden_size     = 2048
voxcpm.num_layers      = 28
voxcpm.num_heads       = 16
voxcpm.num_kv_heads    = 2
voxcpm.intermediate_size = 6144
voxcpm.rms_norm_eps    = 1e-05
voxcpm.rope_theta      = 10000
voxcpm.vocab_size      = 73448
voxcpm.max_position_embeddings = 32768

# LoRA-specific metadata:
voxcpm.lora.rank       = 8
voxcpm.lora.alpha      = 16
voxcpm.lora.target_modules = "q_proj,v_proj,k_proj,o_proj"
voxcpm.lora.component  = "base_lm"          # Which component this LoRA applies to
```

### LoRA Export (Separate File, Runtime Loading)

LoRA adapters are exported as independent GGUF files containing only `lora_A` and `lora_B` matrices. The Rust inference engine loads them at runtime and applies dynamically.

**VoxCPM LoRA forward (reference implementation)**:
```python
# From VoxCPM/src/voxcpm/modules/layers/lora.py
output = F.linear(x, weight, bias) + dropout(F.linear(F.linear(x, lora_A), lora_B)) * (alpha / r)
# lora_A shape: [r, in_features]
# lora_B shape: [out_features, r]
# scaling = alpha / r (stored as buffer, not parameter)
```

The Rust implementation MUST replicate this exact computation:
```rust
let lora_out = x.matmul(&lora_a.transpose()) // [batch, r]
               .matmul(&lora_b.transpose()); // [batch, out_features]
let output = base_output + lora_out * (alpha / rank);
```

### Tensor Naming Convention in GGUF

```
# Base LM tensors:
base_lm.layers.{i}.self_attn.q_proj.weight
base_lm.layers.{i}.self_attn.k_proj.weight
base_lm.layers.{i}.self_attn.v_proj.weight
base_lm.layers.{i}.self_attn.o_proj.weight
base_lm.layers.{i}.mlp.gate_proj.weight
base_lm.layers.{i}.mlp.up_proj.weight
base_lm.layers.{i}.mlp.down_proj.weight
base_lm.layers.{i}.input_layernorm.weight
base_lm.layers.{i}.post_attention_layernorm.weight
base_lm.embed_tokens.weight
base_lm.norm.weight

# Residual LM (same structure, fewer layers):
residual_lm.layers.{i}.self_attn.q_proj.weight
...

# Encoder:
encoder.layers.{i}.self_attn.{q,k,v,o}_proj.weight
encoder.layers.{i}.mlp.{gate,up,down}_proj.weight
encoder.input_proj.weight
encoder.special_token

# DiT:
dit.layers.{i}.self_attn.{q,k,v,o}_proj.weight
dit.layers.{i}.mlp.{gate,up,down}_proj.weight
dit.time_embed.{layers}.weight
dit.in_proj.weight
dit.cond_proj.weight
dit.out_proj.weight

# AudioVAE:
audiovae.decoder.blocks.{i}.conv.weight
audiovae.decoder.blocks.{i}.residual.{j}.conv.weight
...

# LoRA:
lora.layers.{i}.self_attn.q_proj.lora_A
lora.layers.{i}.self_attn.q_proj.lora_B
...
```

### Supported Model Variants

| Variant | Source Weights | Notes |
|---------|---------------|-------|
| VoxCPM-0.5B | `VoxCPM/models/VoxCPM-0.5B/pytorch_model.bin` | Older format, patch_size=2 |
| VoxCPM1.5 | `VoxCPM/models/VoxCPM1.5/model.safetensors` | patch_size=4, 44.1kHz output |
| VoxCPM2 | `VoxCPM/models/VoxCPM2/model.safetensors` | 2B params, 48kHz output |

LoRA checkpoints:
- `VoxCPM/ft0.5/latest/lora_weights.safetensors`
- `VoxCPM/ft1.5/latest/lora_weights.safetensors`
- `VoxCPM/ft2/latest/lora_weights.safetensors`

---

## Part 2: Rust Inference Engine

### Inference Pipeline

```
Text Input
    ↓
[Tokenizer] (HuggingFace tokenizers crate, loads tokenizer.json)
    ↓
[Base LM - Autoregressive] (ggml, with KV cache)
  - MiniCPM-4 architecture
  - GQA (Grouped Query Attention)
  - RoPE with LongRope scaling (short_factor/long_factor)
  - SiLU-gated FFN
  - RMSNorm
  - scale_emb=12, scale_depth=1.4
  - Generates until stop_predictor fires or max_length reached
    ↓
[Scalar Quantization] (FSQ layer)
  - in_proj: hidden_size → latent_dim
  - tanh activation
  - quantize: round(x * scale) / scale  (scale=9)
    ↓
[Residual LM] (ggml, with LoRA applied at runtime)
  - Same MiniCPM-4 architecture, fewer layers
  - Optional: no RoPE (voxcpm2 residual_lm_no_rope=true)
  - LoRA: dynamically loaded, applied per-layer
    ↓
[Projection Layers]
  - lm_to_dit_proj: lm_hidden → dit_hidden
  - res_to_dit_proj: lm_hidden → dit_hidden
    ↓
[Local DiT + CFM Solver] (candle)
  - Conditional Flow Matching with Euler ODE solver
  - 10 inference steps (configurable)
  - Classifier-free guidance: cfg_value=2.0
  - Time scheduler: log-norm
  - sigma_min: 1e-6
  - Processes patch_size frames at a time
    ↓
[AudioVAE Decoder] (candle)
  - Transposed convolution decoder
  - Snake activation (learnable)
  - Weight-normalized convolutions
  - Output: 48kHz PCM (v2) or 44.1kHz (v1.5) or 16kHz (v0.5B)
    ↓
PCM Waveform (f32 samples)
```

### Crate Details

#### `voxui-gguf`
- Parse GGUF binary format (header, metadata KV, tensor info, tensor data)
- Memory-map tensor data for efficient access
- Dequantize on-the-fly: Q4 → f32, FP8 → f32, FP16 → f32
- Tensor lookup by name

#### `voxui-inference`
- **Feature flags**: `cuda` (enables ggml-cuda + candle-cuda)
- **LM module** (ggml FFI):
  - Build ggml computation graph for transformer forward pass
  - StaticKVCache: pre-allocated, position-indexed
  - RoPE with dynamic LongRope frequency scaling
  - Stop predictor: linear → SiLU → linear(2) → argmax
- **LoRA module**:
  - Load LoRA GGUF file
  - Patch linear layers at runtime: `output += (x @ A^T @ B^T) * (alpha/r)`
  - Support hot-swapping: unload current LoRA, load new one
- **DiT module** (candle):
  - MiniCPM-style transformer with time conditioning
  - CFM Euler solver loop
  - CFG: run forward twice (conditioned + unconditioned), interpolate
- **VAE module** (candle):
  - Causal decoder with Snake activations
  - Super-resolution built into V2 decoder rates
- **Tokenizer**: `tokenizers` crate, load from `tokenizer.json`

#### `voxui-audio`
- Enumerate hosts: `cpal::available_hosts()` → WASAPI, DirectSound
- Enumerate devices per host: output devices with names
- Create output stream with matching sample rate (48kHz/44.1kHz/16kHz)
- Play PCM buffer: write f32 samples to stream callback
- Volume control (simple gain multiply)

#### `voxui-app`
- ratatui event loop with crossterm backend
- Async architecture using tokio channels
- TTS request queue (FIFO)
- Progress updates via channel from inference thread

### CPU vs CUDA

- Compile-time: `cargo build --features cuda` enables GPU support
- Runtime: detect CUDA availability, user selects in settings
- ggml: uses cuBLAS for GEMM when CUDA enabled
- candle: `Device::Cuda(0)` for GPU tensors

### Text Length Limiting

- Default max: 80 characters per TTS request
- Longer text is rejected at input with UI feedback
- Configurable via settings (Max Chars field)

---

## Part 3: TUI Interface

### Main Layout (Single Panel)

```
┌─ VoxUI ────────────────────────────────────────────┐
│ TTS History                                         │
│ ┌──────────────────────────────────────────────────┐│
│ │ [12:03:01] 大家好，欢迎来到直播间！           ✓ ││
│ │ [12:03:15] 感谢xxx的关注                      ✓ ││
│ │ [12:03:28] 谢谢打赏！                         ✓ ││
│ │ [12:03:45] 我们现在开始今天的内容              ▶ ││
│ │                                                  ││
│ └──────────────────────────────────────────────────┘│
│ Progress ████████████░░░░░░░░░ 60% (generating...)  │
│ ┌──────────────────────────────────────────────────┐│
│ │ > 输入文字按 Enter 发送...                       ││
│ └──────────────────────────────────────────────────┘│
│ Model: VoxCPM2 (LM:Q4 DiT:FP16 VAE:FP16) | CUDA   │
│ Audio: WASAPI / Speakers (Realtek)                  │
└─────────────────────────────────────────────────────┘
```

### Components

- **Title bar**: "VoxUI"
- **History list**: Scrollable list of completed/in-progress TTS items. Each entry shows timestamp + text + status icon (✓ done, ▶ playing, ⏳ queued, ⚠ error)
- **Progress bar**: Generation progress based on autoregressive steps completed vs estimated total
- **Input box**: Text input, Enter sends to TTS queue
- **Status bar**: Two-line footer showing current model (with quantization info read from GGUF metadata), backend, audio host/device

### Settings Popup (F2)

```
┌─ Settings ─────────────────────────────────────┐
│                                                 │
│  Model:    [▾ models/voxcpm2-mixed  ]          │
│  LoRA:     [▾ ft2/latest            ]          │
│  Backend:  [▾ CUDA                  ]          │
│  Audio:    [▾ WASAPI                ]          │
│  Device:   [▾ Speakers (Realtek)    ]          │
│  Max Chars:[  80                    ]          │
│                                                 │
│  [Apply]  [Cancel]                              │
└─────────────────────────────────────────────────┘
```

- **Model**: dropdown listing directories under `models/` that contain valid GGUF files. Quantization info is displayed in status bar, not selectable here.
- **LoRA**: dropdown listing available LoRA GGUF files (or "None"). Hot-swappable at runtime.
- **Backend**: CPU or CUDA
- **Audio Host**: WASAPI / DirectSound (enumerated from cpal)
- **Device**: Output devices for selected host
- **Max Chars**: Maximum characters per TTS request

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Enter | Send input text to TTS queue |
| F2 | Toggle settings popup |
| Esc | Close popup / quit application |
| ↑↓ | Scroll history list |
| Ctrl+C | Cancel current TTS generation |
| Tab | Navigate between dropdown fields in settings |
| Space | Open/confirm dropdown selection |

### Async Architecture

```
┌─────────────────────┐
│  Main Thread        │  ratatui event loop + render
│  (crossterm events) │
└────────┬────────────┘
         │ mpsc channels
         ▼
┌─────────────────────┐
│  Inference Thread   │  tokio spawn_blocking
│  (ggml + candle)    │  → progress updates via channel
└────────┬────────────┘
         │ completed PCM buffer
         ▼
┌─────────────────────┐
│  Audio Thread       │  cpal output stream callback
│  (cpal playback)    │  reads from buffer, signals completion
└─────────────────────┘
```

- TTS requests enter a bounded queue (backpressure if queue full)
- Inference thread processes one request at a time
- Each autoregressive step sends progress update to main thread
- Completed PCM buffer is sent to audio thread
- Audio thread signals playback completion → main thread updates history status

---

## Part 4: Error Handling

- **Model loading failure**: Show error in status bar, disable TTS input until valid model selected
- **CUDA unavailable**: Fall back to CPU, show warning
- **Audio device error**: Show error, allow re-selection of device
- **Generation failure** (NaN, divergence): Mark item as ⚠ error, log details, continue queue
- **LoRA mismatch** (wrong model variant): Reject with clear error message in settings

---

## Part 5: Dependencies

### Python (exporter)
- `torch` — load PyTorch/safetensors weights
- `safetensors` — efficient weight loading
- `numpy` — tensor manipulation during quantization
- `struct` — GGUF binary format writing

### Rust (voxui)
- `ggml-sys` — ggml C library FFI bindings
- `candle-core`, `candle-nn` — DiT/VAE inference
- `candle-cuda` (optional) — CUDA backend for candle
- `tokenizers` — HuggingFace tokenizer loading
- `ratatui` + `crossterm` — TUI framework
- `cpal` — cross-platform audio
- `tokio` — async runtime
- `memmap2` — memory-mapped GGUF file access
- `serde`, `serde_json` — config parsing

---

## Part 6: Scope & Constraints

- **Platform**: Windows 11 (primary target), may work on Linux/macOS via cpal abstraction
- **Max text length**: 80 characters default (configurable)
- **Generation mode**: Full generation then playback (no streaming)
- **Audio output**: PCM playback only (no WAV file export in v1)
- **Model variants**: VoxCPM-0.5B, VoxCPM1.5, VoxCPM2 all supported
- **LoRA**: Runtime hot-swap, one LoRA active at a time per component
