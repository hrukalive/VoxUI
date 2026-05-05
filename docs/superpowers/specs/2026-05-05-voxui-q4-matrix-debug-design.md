# VoxUI Q4 Matrix Debug Design

## Goal

Add a mixed q4 export/test path, regenerate full inference WAV outputs, document concise build/test commands, keep source path JSON fields informational, and add debug-mode logging for desktop model loading.

## Scope

This change covers the native GGUF exporter, Rust inference manifest loading, the end-to-end inference suite, root README command lines, and desktop model-load diagnostics. It does not change VoxCPM model architecture, tokenizer behavior, audio playback behavior, or the golden parity trace format.

## Q4 Export Profile

The q4 profile is a mixed-precision profile named `q4-lm`. It applies q4 quantization to the less precision-sensitive and largest memory consumers:

- `base_lm.gguf`
- `residual_lm.gguf`
- `projections.gguf`

The audio-sensitive path stays higher precision:

- `feat_encoder.gguf`: fp16
- `feat_decoder.gguf`: fp16
- `audio_vae.gguf`: fp16 for VoxCPM 0.5/1.5 and f32 for VoxCPM2, matching the current fp16 bundle convention

The exporter already supports `q4` at the low-level GGUF writer and quantizer layers. The implementation should add a named convenience profile or documented command path so q4-lm bundles can be produced consistently as `models/voxcpm05-q4-lm`, `models/voxcpm15-q4-lm`, and `models/voxcpm2-q4-lm`.

## Manifest Semantics

Fields such as `source_model_dir` and LoRA `source_lora_dir` are provenance metadata only. Runtime loading must not require them to exist or point to readable local paths. Rust manifest deserialization should tolerate these fields being absent by using optional/default values. Validation should continue to require runtime-critical fields such as schema version, architecture, variant, special tokens, audio VAE settings, component filenames, and quantization metadata.

## Inference Matrix

The full matrix should discover every `models/*/manifest.json` bundle, including q4-lm bundles when present. For every model/device combination it should run:

- One sentence-length Chinese zero-shot synthesis.
- One sentence-length English zero-shot synthesis.
- VoxCPM2 reference and reference-continuation cases when a test WAV exists.
- LoRA cases when model-local LoRA manifests exist.

The Chinese and English test strings should be real sentence-length inputs, not short token smoke tests. Output WAVs remain under `test_output/`, and the test should create the directory as needed. Since the user cleaned the old WAVs, running the full matrix should regenerate a fresh complete set.

## README Commands

`README.txt` should remain concise and command-oriented. It should include copyable PowerShell command lines for:

- Setting the CUDA/MSVC environment for local builds.
- Building the inference crate and desktop app.
- Exporting fp16 bundles.
- Exporting q4-lm bundles.
- Verifying exported GGUF bundles.
- Running focused Rust tests.
- Running the full inference matrix, including CUDA when built with the CUDA feature.
- Running the desktop app in debug mode with logs enabled.

## Desktop Debug Logging

In debug builds, model loading should emit useful diagnostics to the console/log stream. Logging should cover:

- Frontend initialization milestones: config load, model list, selected model, audio device list, LoRA list, model load invoke start, model load success/failure.
- Tauri backend command milestones: requested model path, requested backend, selected actual backend, component load start/end/failure, total load duration.
- Busy-state rejections and errors from `load_model`, `apply_lora`, and `synthesize`.

Logging must not alter release behavior or require a UI redesign. The immediate goal is to make the current loading hang diagnosable from the debug console.

## Testing

Verification should include:

- Python exporter unit tests, including q4-lm quantization metadata expectations.
- Rust manifest loader tests proving missing `source_model_dir` is accepted.
- Rust GGUF/inference tests for q4 dequantization coverage if not already exercised by q4-lm model loading.
- The full inference matrix after q4-lm bundles are generated.
- A desktop debug run, or at minimum a successful desktop crate test/build if the GUI cannot be interactively inspected.

## Risks

Q4 on every component could degrade generated audio quality, so q4 is limited to LM/projection components. The full inference matrix is expensive and may require CUDA setup; the plan should provide CPU-safe commands and CUDA commands separately. Desktop hanging may have multiple causes, so logging is a diagnostic improvement first and may expose a deeper loading bug during verification.
