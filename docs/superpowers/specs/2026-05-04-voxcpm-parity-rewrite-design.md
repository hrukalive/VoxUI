# VoxCPM Exporter And Inference Parity Rewrite

## Context

The current VoxUI exporter and Rust inference engine can load generated GGUF files, but generated WAVs are random noise with only a weak hint of voice. The observed structure shows this is not a simple tuning bug. The Rust engine implements an approximate pipeline that diverges from the Python implementation in `VoxCPM/src/voxcpm/model/voxcpm.py` and `VoxCPM/src/voxcpm/model/voxcpm2.py`.

The repair may discard the current exported model layout and current inference structure. The required outcome is compatibility with VoxCPM 0.5, VoxCPM 1.5, and VoxCPM 2.0, with and without LoRA, on CPU and CUDA. VoxCPM 2.0 must support reference-audio synthesis. Prompt/reference WAVs from `for_test_wav/` will be used for testing.

The chosen approach is a Python-parity exporter plus a native Rust inference engine rewrite.

## Goals

- Regenerate model bundles for VoxCPM 0.5, 1.5, and 2.0 from the local `VoxCPM/models` sources.
- Preserve enough Python model structure in exported metadata and tensor names for Rust to reconstruct the same graph.
- Implement the Rust inference path to match Python zero-shot, continuation, and reference-audio modes.
- Support LoRA loading for LM, residual LM, DiT, and optional projection targets.
- Support CPU and CUDA execution through Candle.
- Add parity tests that compare Rust intermediate tensors to Python golden traces before relying on audible checks.
- Produce intelligible audio in end-to-end tests.

## Non-Goals

- Backward compatibility with the current `models/*` GGUF layout.
- Streaming audio playback as part of this repair.
- UI redesign.
- Denoiser integration.
- Multi-LoRA composition. One active LoRA adapter bundle is enough.

## Model Bundle Format

Each exported model is a directory containing a manifest, copied tokenizer files, GGUF component files, and optional LoRA component files.

Required files:

- `manifest.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `config.json`
- `base_lm.gguf`
- `residual_lm.gguf`
- `feat_encoder.gguf`
- `feat_decoder.gguf`
- `audio_vae.gguf`
- `projections.gguf`

Optional LoRA files:

- `lora_base_lm.gguf`
- `lora_residual_lm.gguf`
- `lora_feat_decoder.gguf`
- `lora_projections.gguf`

The manifest records:

- Bundle schema version.
- VoxCPM architecture: `voxcpm` or `voxcpm2`.
- Model variant: `0.5`, `1.5`, or `2.0`.
- Source model path and source weight format.
- Special token IDs: `audio_start = 101`, `audio_end = 102`, and for VoxCPM2 `ref_audio_start = 103`, `ref_audio_end = 104`.
- `patch_size`, `feat_dim`, scalar quantization dim and scale.
- AudioVAE `sample_rate`, `out_sample_rate` when present, `chunk_size`, `decode_chunk_size`, `latent_dim`, encoder rates, and decoder rates.
- MiniCPM config fields, including `use_mup`, `scale_emb`, `scale_depth`, `rope_scaling`, `kv_channels`, and `no_rope`.
- Component file map and quantization choices.
- LoRA metadata: rank, alpha, enabled targets, and target modules.

GGUF tensor names should follow Python module paths closely:

- `base_lm.*`
- `residual_lm.*`
- `feat_encoder.*`
- `feat_decoder.*`
- `audio_vae.*`
- projection layer names exactly as Python attributes: `fsq_layer.*`, `enc_to_lm_proj.*`, `lm_to_dit_proj.*`, `res_to_dit_proj.*`, `fusion_concat_proj.*`, `stop_proj.*`, `stop_head.*`

The exporter must validate required tensor coverage. Missing, duplicate, unmapped, or shape-mismatched tensors are hard errors.

## Exporter Design

The exporter will be rewritten around model introspection and explicit component manifests rather than ad hoc prefix splitting.

Source handling:

- VoxCPM 0.5 loads `pytorch_model.bin`.
- VoxCPM 1.5 and 2.0 load `model.safetensors`.
- AudioVAE loads from `audiovae.pth` or `audiovae.safetensors` if present.
- LoRA loads from `lora_weights.safetensors` and `lora_config.json`.

The exporter will instantiate or inspect the matching Python config class enough to determine required component structure. It will then partition tensors according to source-faithful module prefixes and write GGUF files plus manifest metadata.

Quantization support:

- FP16 for all components.
- Q8/Q4 for LM and projection components where requested.
- DiT and AudioVAE default to FP16 because they are sensitive to quantization.
- LoRA weights are FP16.

Export commands should be able to regenerate all local variants:

- `VoxCPM/models/VoxCPM-0.5B`
- `VoxCPM/models/VoxCPM1.5`
- `VoxCPM/models/VoxCPM2`

## Rust Inference API

The Rust engine exposes one canonical synthesis request modeled after `VoxCPM.generate()`. The Rust API uses Python-facing argument names where practical so UI, tests, and future bindings do not drift from the reference implementation.

```rust
pub struct SynthesisRequest {
    pub text: String,
    pub prompt_wav_path: Option<PathBuf>,
    pub prompt_text: Option<String>,
    pub reference_wav_path: Option<PathBuf>,
    pub cfg_value: f32,
    pub inference_timesteps: usize,
    pub min_len: usize,
    pub max_len: usize,
    pub normalize: bool,
    pub retry_badcase: bool,
    pub retry_badcase_max_times: usize,
    pub retry_badcase_ratio_threshold: f32,
}
```

The Rust API intentionally omits `denoise` because denoising uses the separate ZipEnhancer pipeline and is not part of this repair. Streaming remains out of scope for this rewrite; it can be added later as a separate API matching `generate_streaming()`.

Supported modes:

- VoxCPM 0.5 and 1.5: zero-shot and continuation with `prompt_text` plus `prompt_wav_path`.
- VoxCPM 2.0: zero-shot, continuation, reference-only, and reference plus continuation.
- Reference audio on non-VoxCPM2 models is rejected with a clear error.
- `reference_wav_path` does not require transcript text. It is encoded as isolated reference audio using VoxCPM2 reference-audio tokens.
- `prompt_text` is required only when `prompt_wav_path` is present, because prompt audio is continuation context.
- `text` is trimmed, newlines are normalized to spaces, repeated whitespace is collapsed, and empty text is rejected.
- `normalize = true` applies VoxCPM text normalization before tokenization. If Rust parity for the normalizer is not yet implemented, the engine must reject `normalize = true` with a clear error rather than silently ignoring it.
- `retry_badcase` follows Python behavior: generation retries when generated audio-feature length exceeds `target_text_token_count * retry_badcase_ratio_threshold`, capped by `retry_badcase_max_times`.

## Rust Inference Flow

The implementation follows Python `_generate` and `_inference`.

1. Tokenize text exactly like Python and append `audio_start_token`.
2. Load prompt/reference WAV when provided, convert to mono, resample to the AudioVAE encoder sample rate, pad to `patch_size * chunk_size`, and encode with AudioVAE encoder. Reference WAVs are used without transcript text; prompt WAVs require paired `prompt_text`.
3. Build `text_token`, `text_mask`, `audio_feat`, and `audio_mask` for the selected mode.
4. Run local encoder over prefill audio features as `[B, T, P, D]`.
5. Project local encoder output with `enc_to_lm_proj`.
6. Build combined embeddings from text embeddings and audio embeddings using the masks.
7. Prefill `base_lm` with the combined embeddings.
8. Apply FSQ only on audio-mask positions during prefill, then on generated LM steps.
9. Prefill `residual_lm`.
   - VoxCPM 0.5 and 1.5 use the Python add path.
   - VoxCPM 2.0 uses `fusion_concat_proj`.
10. Generate patches autoregressively.
11. For each patch, project LM and residual hidden states into DiT conditioning.
12. Run `feat_decoder` CFM Euler generation with previous patch as `cond`, using `inference_timesteps` and `cfg_value`.
13. Re-encode the predicted patch with local encoder and feed it back to base LM and residual LM.
14. Stop after `min_len` only when `stop_head` predicts class 1.
15. Decode generated latent patches with AudioVAE decoder.
16. Trim continuation context exactly like Python.

## Component Parity Requirements

MiniCPM:

- Implement `scale_emb` when `use_mup` requires it.
- Implement `scale_depth / sqrt(num_hidden_layers)` residual scaling when `use_mup` is true.
- Implement LongRoPE exactly, including short/long factors, original max position, and scaling factor.
- Use Python `rotate_half` behavior for RoPE.
- Support causal prefill and single-step KV-cache inference.
- Support non-causal mode for local encoder and DiT.

Local encoder:

- Accept `[B, T, P, D]`.
- Apply `in_proj`, prepend `special_token`, flatten `(B, T)` for the MiniCPM encoder, then restore `[B, T, hidden]`.

DiT / CFM:

- Implement the Python `UnifiedCFM.forward` and `solve_euler` flow.
- Preserve `inference_cfg_rate`, `cfg_value`, `sway_sampling_coef`, and CFG-Zero* warmup behavior.
- Support both VoxCPM 0.5/1.5 `VoxCPMLocDiT` and VoxCPM2 `VoxCPMLocDiT` behavior as represented by the bundle manifest.

AudioVAE:

- Implement encoder and decoder.
- Support V1 and V2 AudioVAE configs.
- Use exact causal convolution, transposed convolution, weight norm, Snake activation, and SR conditioning behavior.
- Use encode rate for prompt/reference WAVs and output sample rate for decoded WAVs.

LoRA:

- Load component LoRA files from an adapter bundle.
- Apply `base + (x @ A^T @ B^T) * (alpha / rank)`.
- Support target modules for base LM, residual LM, and DiT.
- Support projection LoRA if the adapter contains projection targets.
- Reject adapters whose manifest does not match model architecture, variant, rank/shape, or target modules.

## CPU And CUDA

CPU is always supported through Candle CPU tensors. CUDA is enabled with the workspace `cuda` feature.

CUDA verification commands must set:

```powershell
$env:PATH = "$env:USERPROFILE\scoop\apps\rustup\current\.cargo\bin;$env:PATH"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "$env:CUDA_PATH\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
$env:CUDA_COMPUTE_CAP = "89"
$env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"
```

Python verification must activate:

```powershell
& ~\py_env\voxcpm\Scripts\activate.ps1
```

## Testing Strategy

Testing is parity-first.

Python golden trace generation:

- Use local Python VoxCPM implementation and source models.
- Generate small artifacts for each variant.
- Store token IDs, masks, encoded prompt/reference features, prefill hidden states, residual prefill states, first DiT patch, stop logits, decoded waveform metadata, and final WAV.
- Include VoxCPM2 reference-audio cases using `for_test_wav/`.

Exporter tests:

- GGUF writer roundtrip.
- Quantization roundtrip.
- Required tensor coverage by component.
- Manifest validation.
- Export all three model variants.
- Export LoRA for all three local fine-tuned checkpoints.

Rust parity tests:

- Tokenization parity.
- AudioVAE encode/decode shape and numeric parity.
- MiniCPM layer parity on selected small inputs.
- Local encoder parity.
- DiT first-step parity with fixed RNG/noise.
- Full first autoregressive patch parity.
- LoRA application parity.

End-to-end tests:

- VoxCPM 0.5, 1.5, and 2.0 without LoRA on CPU.
- VoxCPM 0.5, 1.5, and 2.0 with LoRA on CPU.
- The same matrix on CUDA when available.
- VoxCPM2 reference-only and reference-plus-continuation tests using `for_test_wav/`.
- Save WAVs to `test_output/`.
- Validate non-empty finite samples, reasonable RMS, and audible/manual inspection.

## Migration Plan

1. Add the new bundle manifest schema and exporter validation.
2. Generate Python golden traces.
3. Rewrite exporter and regenerate model bundles.
4. Rewrite Rust model loader to use `manifest.json`.
5. Fix MiniCPM parity.
6. Implement AudioVAE encoder and decoder parity.
7. Implement local encoder parity.
8. Implement DiT/CFM parity.
9. Implement Python-parity synthesis request flow.
10. Implement component-aware LoRA.
11. Run CPU parity and end-to-end tests.
12. Run CUDA parity and end-to-end tests with the provided environment variables.

## Risks

- AudioVAE encoder/decoder parity is the largest Rust implementation risk because prompt/reference support depends on encode, and audible output depends on decode.
- Candle convolution and transposed convolution behavior must match PyTorch padding and causal trimming precisely.
- CUDA numeric differences may require tolerances, but should not produce random noise if the graph is correct.
- Q4 quantization may be too aggressive for some components; the exporter should allow FP16 fallback per component.

## Acceptance Criteria

- Freshly exported bundles load for VoxCPM 0.5, 1.5, and 2.0.
- CPU synthesis produces intelligible WAVs for all three variants.
- CUDA synthesis produces intelligible WAVs for all three variants when built with the provided environment.
- LoRA changes the generated voice/style without breaking synthesis for all three variants.
- VoxCPM2 reference-audio synthesis works with WAVs in `for_test_wav/`.
- Rust parity tests pass within documented tolerances for core intermediate tensors.
- Exporter rejects incomplete or mismatched bundles instead of silently skipping tensors.
