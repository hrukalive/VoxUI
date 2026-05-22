# VoxCPM Stop Parity Fix Design

## Goal

Fix native Rust generation for VoxCPM 0.5, with VoxCPM 1.5 and VoxCPM2 checked for regressions, by matching the Python generation loop's stop behavior before addressing broader audio-quality drift.

The immediate symptom is that Python can stop early while Rust is reported to run to the bounded maximum step count. That can produce speech that keeps rising in tone, becomes increasingly excited, and eventually degrades into noise.

## Context

The Rust project lives in `voxui/`. The Python references live in `VoxCPM/src/voxcpm/model/`.

Current Rust code already has variant-aware DiT conditioning and Chinese tokenizer parity changes. A direct comparison of the high-level loops shows the same intended order in Rust and Python:

1. Prefill base and residual language model state.
2. Build DiT conditioning from current `lm_hidden` and `residual_hidden`.
3. Generate one latent patch.
4. Encode the generated patch and append it to the generated feature sequence.
5. Evaluate stop logits from the current pre-step `lm_hidden`.
6. Stop if the stop class wins after `min_len`.
7. Advance base and residual language model states using the generated patch embedding.

The stop predicate itself is equivalent in intent: Python uses `argmax == 1`, while Rust compares `stop > keep`. If Rust does not stop, the likely mismatch is upstream of the predicate: hidden-state drift, cache position handling, projection or LoRA application, quantization/runtime linear behavior, tokenizer length effects, or stop-head input timing.

## Approach

Use stop-loop parity as the first repair path. Add matching Python and Rust trace points for per-step stop and hidden-state data, compare VoxCPM 0.5 first, patch the first confirmed mismatch, then re-run the same checks for VoxCPM 1.5 and VoxCPM2.

This avoids adding heuristic stop guards that may mask the real parity issue.

## Alternatives Considered

### Full Generation Trace First

Tracing every tensor through prefill, DiT, encoder, base LM, residual LM, and VAE would be comprehensive. It is slower to build and harder to inspect. It should be reserved for cases where stop-focused traces show divergence before the stop/logit path can isolate it.

### Stop Heuristic or Audio Guard

Adding a forced max ratio, stop smoothing, or a noise/energy guard could reduce bad output. This is not the first fix because it would not explain why Rust differs from Python and could hide a real model-state bug.

### Stop-Loop Parity First

This is the selected approach. It is narrow, testable, and targets the reported failure directly. It still leaves a clear escalation path into deeper tensor tracing if stop-loop data proves the mismatch is earlier than expected.

## Components

### Python Reference Trace

Extend or add a trace script under `tools/golden_trace/` for VoxCPM 0.5. The trace should run the Python reference through the same prompt used by Rust verification and record:

- the exact target text and optional prompt/reference inputs used for the trace
- text token ids and target text token count
- bounded `max_len`, `min_len`, `cfg_value`, and `inference_timesteps`
- per-step `stop_logits`
- per-step stop decision
- compact stats for `lm_hidden`, `residual_hidden`, generated latent patch, and generated audio feature patch

The trace should write structured JSON and small binary float arrays using the existing golden trace conventions where practical. The Rust comparison must read the prompt and generation settings from the trace metadata rather than hard-coding a second copy.

### Rust Debug Trace

Add an internal debug path or test helper in `voxui/crates/voxui-inference` that emits matching per-step data from `VoxCPMEngine` without changing the normal public synthesis API.

The Rust trace must record values at the same semantic points as Python:

- after prefill, before the first patch
- after each DiT patch generation
- before the LM state is advanced, for stop logits
- after base and residual LM state advancement

### Comparison Test

Add a focused Rust test that loads the VoxCPM 0.5 reference trace and compares:

- the first generated step's stop logits and hidden summaries
- the sequence of stop decisions up to the Python stop step
- the final Rust stop step against the Python stop step for the traced prompt

Use the existing numeric tolerance style from current parity tests. If exact equality is not reasonable for quantized or CUDA-specific paths, the test should compare logits within tolerance and separately assert the stop decision sequence.

### Fix Point

Patch the first confirmed mismatch only. The expected investigation order is:

1. If step 0 stop logits diverge, inspect prefill hidden states, stop projections, and tokenizer/input construction.
2. If step 0 matches but later steps diverge, inspect base/residual LM cache positions and per-step state update order.
3. If hidden states match but stop logits diverge, inspect stop projection tensor loading and runtime linear behavior.
4. If stop logits match but Rust still does not stop, inspect stop class ordering and threshold logic.

## Testing

Run focused CPU or CUDA-independent tests first where possible:

```powershell
cargo test -p voxui-inference --test generate_flow_parity
cargo test -p voxui-inference --test tokenizer_parity
```

Run the new stop parity test for VoxCPM 0.5.

Then run CUDA generation verification for 0.5, 1.5, and 2.0 with the same q4 matrix used by existing inference tests:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "$env:CUDA_PATH\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
$env:CUDA_COMPUTE_CAP = "89"
$env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"
cargo test -p voxui-inference --features cuda --test inference_suite q4_lm_cuda -- --nocapture --test-threads=1
```

## Acceptance Criteria

- VoxCPM 0.5 Rust stop logits match the Python trace within the chosen tolerance until the Python stop decision.
- VoxCPM 0.5 Rust stops at the same step as Python for the trace-owned prompt, or the remaining difference is explained by a documented tolerance/quantization boundary.
- VoxCPM 0.5 no longer runs to bounded max length for the trace-owned prompt when Python stops early.
- VoxCPM 1.5 and VoxCPM2 existing parity and generation tests continue to pass.
- No heuristic stop guard is added unless parity shows Python and Rust agree but the shared model behavior is still bad.

## Out of Scope

- Re-exporting model bundles.
- Changing GGUF layout.
- Rewriting the engine into separate top-level model structs.
- UI changes.
- General audio post-processing, silence trimming, or pitch/noise heuristics.
- Full VAE waveform parity unless stop-loop parity passes and the audio issue remains.
