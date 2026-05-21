# VoxCPM2 CUDA VRAM Report Design

## Goal

Add a focused CUDA VRAM usage report test for the Rust `voxui-inference` engine. The test should report how much VRAM VoxCPM2 uses when loading the model and running one short synthesis, without failing solely because the measured usage is above the initial 7 GiB budget.

## Context

The main Rust project lives in `voxui/`. Python VoxCPM references and source models live under `VoxCPM/`. Exported GGUF bundles live under `models/`.

The current exported VoxCPM2 bundles include:

- `models/voxcpm2-fp16`
- `models/voxcpm2-q4-lm`

The q4 bundle is useful to measure now because it already exists and is much smaller on disk. However, the current Rust GGUF loader dequantizes tensors to f32 and then casts to f16 on CUDA, so GGUF q4 may not reduce steady-state CUDA VRAM unless the runtime keeps weights quantized or changes its loading strategy. The test should make that behavior visible before changing exporter or inference internals.

## Approach

Create a dedicated CUDA integration test:

```text
voxui/crates/voxui-inference/tests/cuda_vram_report.rs
```

The test will measure both VoxCPM2 bundles:

1. `voxcpm2-fp16`
2. `voxcpm2-q4-lm`

Each model runs the same sequence:

1. Create `Device::new_cuda(0)`.
2. Capture a baseline VRAM snapshot for the current process.
3. Load `VoxCPMEngine`.
4. Capture a post-load VRAM snapshot.
5. Run one synthesis request with a fixed 20-Chinese-character sentence.
6. Capture a post-synthesis VRAM snapshot.
7. Print a compact report with baseline, post-load, post-synthesis, load delta, synthesis delta, peak delta, and a soft warning if the peak delta is above 7 GiB.

The test is report-first: it fails only when CUDA is available but model load or synthesis fails. Exceeding 7 GiB is reported as a warning, not an assertion.

## Test Input

Use one deterministic 20-Chinese-character sentence. The exact sentence should be stored as a constant in the test, with a test assertion that `chars().count() == 20` so future edits do not silently change the workload.

The synthesis request should keep runtime bounded:

- `inference_timesteps`: use the same short value as the existing inference suite, currently 10.
- `max_len`: set `SynthesisRequest::max_len` to 6 generated patches, matching the existing focused inference-suite limit.
- `retry_badcase`: false.

The test should write no WAV output by default. It exists to report memory, not audio artifacts.

## VRAM Measurement

Use a test-local memory snapshot helper.

Primary measurement should be process-scoped dedicated GPU memory:

```powershell
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
```

The helper filters rows to `std::process::id()` and treats the resulting value as the current process VRAM. This avoids confusing the report with unrelated processes on the same GPU.

Secondary measurement should use the CUDA driver memory API exposed through the existing dependency path:

```rust
candle_core::cuda::cudarc::driver::result::mem_get_info()
```

The CUDA free/total numbers are global device context, so they are diagnostic only. They should be printed for context but not used for the per-process delta or 7 GiB warning.

If `nvidia-smi` is unavailable or does not report the current process, the test should still run synthesis and print that process-scoped VRAM was unavailable. It should not fall back to asserting against global GPU usage.

## Command

CUDA builds on this Windows workspace need the environment variables from `README.txt`:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "$env:CUDA_PATH\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
$env:CUDA_COMPUTE_CAP = "89"
$env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"
cargo test -p voxui-inference --features cuda --test cuda_vram_report -- --nocapture --test-threads=1
```

Run from the `voxui/` workspace directory.

## Report Format

The output should be easy to compare between models:

```text
=== CUDA VRAM report: voxcpm2-fp16 ===
baseline process:       512 MiB
after load process:    6520 MiB  delta +6008 MiB
after synth process:   6900 MiB  delta +6388 MiB
peak process delta:    6388 MiB  threshold 7168 MiB  OK
cuda global free/total: ...

=== CUDA VRAM report: voxcpm2-q4-lm ===
...
```

If a peak is above 7 GiB, print:

```text
WARNING: peak process VRAM delta exceeded 7 GiB; investigate runtime quantization or more aggressive export/inference strategy.
```

## Exporter Follow-Up

Do not change `exporter/` in this report-first pass.

If `voxcpm2-fp16` or `voxcpm2-q4-lm` reports peak process VRAM above 7 GiB, use the measured data to design the next step. Likely options are:

- Keep GGUF q4 on GPU as quantized weights and dequantize inside compute paths.
- Add more aggressive exporter profiles, such as q4 for additional components, only if runtime can preserve the memory benefit.
- Explore component offload or staged loading for parts of the model that are not active at the same time.

## Acceptance Criteria

- A focused CUDA test exists at `voxui/crates/voxui-inference/tests/cuda_vram_report.rs`.
- The test reports both `voxcpm2-fp16` and `voxcpm2-q4-lm` if their `model.gguf` files exist.
- The test uses a fixed 20-Chinese-character synthesis input.
- The primary VRAM report is process-scoped and filtered to the current PID.
- Other GPU processes do not affect the reported process delta.
- The 7 GiB budget is a warning only.
- CUDA setup instructions are documented with the README environment variables.
- No exporter changes are included in this pass.

## Out of Scope

- New quantization formats.
- Exporter profile changes.
- Runtime quantized matmul or quantized weight residency.
- Desktop UI changes.
- Hard CI gating on VRAM.
