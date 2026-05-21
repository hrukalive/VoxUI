# Q4 Runtime VRAM Design

Date: 2026-05-21
Branch: codex/q4-runtime-vram
Status: approved for spec, pending implementation plan

## Problem

The current GGUF runtime treats q4/q8 tensors as storage compression only. The Rust GGUF loader reads tensor data through `tensor_f32`, expands quantized blocks into `Vec<f32>`, creates a Candle `Tensor`, and then casts CUDA tensors to f16 through `load_tensor_optimal`. As a result, a q4 VoxCPM GGUF can use essentially the same VRAM as an fp16 GGUF after model load and synthesis.

The required behavior is that GGUF dtype controls inference residency. If a tensor is quantized in GGUF, VoxUI must not silently keep a long-lived dense GPU copy of that tensor. Quantized model weights should remain quantized at inference time, and unsupported quantized operator paths must be explicit rather than hidden behind dense fallback.

## Goals

- Preserve q4/q8 GGUF tensors as quantized runtime storage instead of expanding them into cached dense tensors.
- Use quantized matmul for q4/q8 2D linear weights in CUDA inference.
- Apply the same rule to VoxCPM 0.5, VoxCPM 1.5, and VoxCPM 2.0 through their shared Rust model components.
- Make exporter profiles describe real runtime behavior: profiles that emit q4/q8 must only emit quantized tensors for operator paths the runtime can execute as quantized, or fail clearly.
- Extend the CUDA VRAM report so it proves q4 reduces peak process VRAM for model load plus one 20-Chinese-character synthesis, and writes an artifact instead of relying only on test stdout.

## Non-Goals

- Do not implement custom q4 convolution kernels in the first implementation.
- Do not make VAE q4 exports look supported until quantized conv and weight-norm handling exist.
- Do not keep the current behavior where quantized GGUF tensors are dequantized and cached as model-resident f16/f32 tensors.

## Runtime Architecture

Add a quantized-aware runtime weight layer in `voxui-inference`.

The loader should expose separate concepts:

- Dense tensor loading for tensors that are dense in GGUF.
- Runtime tensor loading for tensors whose GGUF dtype may be dense or quantized.
- Linear weight loading that returns a dispatchable linear operator.

The central representation will distinguish:

- Dense Candle `Tensor`
- Quantized Candle `QTensor`
- Quantized Candle `QMatMul` for 2D weights used by linear layers

The loader cache must be dtype-aware. Dense GGUF tensors may still use the existing dense tensor cache. Quantized GGUF tensors should be cached as quantized storage or quantized matmul wrappers only. There must be no cached dense tensor created from q4/q8 GGUF data.

`voxui-gguf` should expose enough raw tensor information for this path: tensor dtype, shape, and raw data slice. `voxui-inference` can then map `voxui_gguf::GgmlType` to Candle `quantized::GgmlDType` and construct Candle quantized tensors from the raw GGUF bytes.

## Operator Coverage

Linear-heavy paths are the first required target because they dominate model weight memory and Candle already provides quantized CUDA matmul.

The following should use the new quantized-aware linear abstraction:

- `BaseLM` and `residual_lm`: attention q/k/v/o projections and MLP gate/up/down projections.
- `DiT`: input, conditioning, output, time MLP, delta-time MLP, attention projections, and MLP projections.
- `LocalEncoder`: input projection.
- `FSQLayer`: input and output projections.
- Engine projections: LM-to-DiT, residual-to-DiT, encoder-to-LM, fusion concat, stop projection, and stop head.

LoRA remains additive on top of the base linear output. The base q4/q8 projection should execute through quantized matmul; LoRA A/B weights can stay dense because they are separate adapter tensors.

Embedding weights need special handling. If `embed_tokens.weight` is quantized in GGUF, the runtime must not dequantize the full embedding table into a resident dense tensor. It will gather/dequantize only the selected token rows into activation-sized output tensors because q4 LM profiles may include the embedding table.

VAE convolution and weight-norm tensors are not covered by the first quantized operator pass. If a GGUF contains q4/q8 VAE conv tensors, inference should fail with an explicit unsupported quantized operator error rather than materializing and storing dense conv weights. Exporter profiles should avoid q4/q8 VAE output by default.

## Data Flow

For a q4 linear tensor:

1. GGUF parser returns tensor info and raw q4 block bytes.
2. `GgufModelLoader` constructs a Candle `QTensor` on the target device.
3. The linear loader wraps it in Candle `QMatMul`.
4. Forward calls route through the quantized matmul path.
5. The output activation is a normal Candle tensor.

For a dense linear tensor:

1. GGUF parser returns tensor info and dense bytes.
2. `GgufModelLoader` creates a normal Candle tensor.
3. Forward calls route through the existing dense matmul path.

For unsupported quantized operators:

1. The loader still recognizes the tensor as quantized.
2. The model component reports a named error when it tries to bind that tensor to an unsupported operator.
3. The error includes the tensor name, GGUF dtype, operator kind, and suggested export setting.

## Exporter Changes

The exporter should stop relying only on coarse component quantization when building low-VRAM profiles. Component-level q4 can accidentally quantize tensors whose runtime operator is not supported, such as VAE conv weights or non-linear small tensors.

Add a tensor-role quantization policy for runtime-supported profiles:

- `q4-lm`: q4 for LM/residual LM linear weights and embedding tables where supported; dense for norms and unsupported tensors.
- `q4-linear`: q4 for all supported 2D linear weights across LM, residual LM, encoder, DiT, FSQ, and engine projections; dense for VAE convs, norms, biases unless explicitly supported.
- Manual component flags remain available, but unsupported q4/q8 combinations should fail during export unless a developer-only override is provided for test fixtures.

Manifest metadata should continue to record the profile and component intent. If tensor-role policy means a component contains mixed dtypes, the manifest or GGUF metadata should make that visible through the profile name, and `verify_gguf.py` should report per-tensor dtype counts for exported files.

## Error Handling

Unsupported quantized runtime paths must be clear and actionable. A failure should say, for example:

`unsupported quantized tensor audio_vae.decoder.model.2.block.1.weight_v: dtype Q4_0 cannot be used by conv1d; re-export audio_vae as fp16/f32 or add quantized conv support`

Silent fallback from q4/q8 GGUF tensor to resident dense GPU tensor is not allowed. Temporary activation tensors produced by quantized operations are expected and do not violate the residency rule.

## Testing

Add focused tests before or alongside implementation:

- GGUF parser test: raw tensor data, dtype, shape, and byte size are available without calling `tensor_f32`.
- Loader test: q4/q8 tensors are cached as quantized runtime storage, not dense tensors.
- Linear parity test: q4/q8 quantized matmul output is close to dense dequantized output for 2D and 3D inputs.
- Embedding test: quantized embedding does not load the full embedding table as a dense resident tensor.
- Unsupported op test: q4/q8 VAE conv tensor fails with a named unsupported-operator error.
- Exporter tests: `q4-linear` quantizes only runtime-supported tensor roles, and q4/q8 VAE export is rejected unless support is added.
- CUDA VRAM report: compare fp16 and q4 GGUFs using model load plus one 20-Chinese-character synthesis. The report should use a dedicated process or per-process GPU memory accounting, write JSON/Markdown artifacts under `target/`, and explain that `cargo test` hides prints unless `-- --nocapture` is used.

## Acceptance Criteria

- Existing q4 VoxCPM GGUF no longer reports the same peak CUDA VRAM as the matching fp16 GGUF when the quantized tensors cover model-resident linear weights.
- q4/q8 GGUF tensors are never cached as dense model-resident tensors by default.
- All three VoxCPM variants route supported linear weights through the same quantized-aware abstraction.
- Unsupported q4/q8 tensors fail with explicit operator errors rather than silently expanding into dense residency.
- Exported low-VRAM q4 profiles only claim runtime-supported quantization.
- The VRAM report artifact is usable without relying on test stdout.
