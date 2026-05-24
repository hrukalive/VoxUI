# VoxUI Inference Streaming Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clean metadata-rich streaming synthesis API to `voxui-inference` while preserving the current batch synthesis API.

**Architecture:** Introduce `SynthesisChunk` as the public stream item, add callback-based streaming methods, and share the existing per-patch generation loop between streaming and batch-oriented paths. Streaming decodes a bounded rolling latent window and emits only the newest PCM region per generated patch.

**Tech Stack:** Rust 2021, Candle tensors, `anyhow`, existing `voxui-inference` model/golden tests.

---

## File Structure

- Modify `voxui/crates/voxui-inference/src/engine.rs`: add chunk type, streaming methods, shared attempt helper, rolling-window chunk decode helpers, and retry validation.
- Modify `voxui/crates/voxui-inference/src/lib.rs`: re-export `SynthesisChunk`.
- Modify `voxui/crates/voxui-inference/tests/request_validation.rs`: add lightweight API/request behavior tests that do not require model weights.
- Modify `voxui/crates/voxui-inference/tests/generate_flow_parity.rs`: add model-backed streaming invariants using short requests and existing test infrastructure.

## Task 1: Public Type And Retry Validation Tests

**Files:**
- Modify: `voxui/crates/voxui-inference/tests/request_validation.rs`
- Modify: `voxui/crates/voxui-inference/src/engine.rs`
- Modify: `voxui/crates/voxui-inference/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Add tests proving the public chunk type can be named from the crate root and documenting that streaming rejects `retry_badcase=true`.

```rust
use voxui_inference::SynthesisChunk;

#[test]
fn synthesis_chunk_is_public_api() {
    let chunk = SynthesisChunk {
        samples: vec![0.0, 0.25],
        sample_rate: 48_000,
        patch_index: 0,
        max_patches: 8,
        generated_patch_count: 1,
        is_final: false,
    };

    assert_eq!(chunk.samples.len(), 2);
    assert_eq!(chunk.sample_rate, 48_000);
    assert!(!chunk.is_final);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p voxui-inference synthesis_chunk_is_public_api`

Expected: compile failure because `SynthesisChunk` is not exported yet.

- [ ] **Step 3: Add public type and export**

In `engine.rs`, add `SynthesisChunk`. In `lib.rs`, change the engine re-export to include it:

```rust
pub use engine::{SynthesisChunk, VoxCPMEngine};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p voxui-inference synthesis_chunk_is_public_api`

Expected: pass.

## Task 2: Streaming API Shape

**Files:**
- Modify: `voxui/crates/voxui-inference/src/engine.rs`
- Modify: `voxui/crates/voxui-inference/tests/generate_flow_parity.rs`

- [ ] **Step 1: Write failing model-backed streaming test**

Add a short generation test that calls `generate_streaming`, collects chunks, and asserts stream invariants:

```rust
#[test]
fn voxcpm2_streaming_yields_finite_ordered_chunks() {
    let root = repo_root();
    let model_dir = root.join("models/voxcpm2-fp16");
    let mut engine = VoxCPMEngine::load(&model_dir, Device::Cpu).unwrap();
    let request = SynthesisRequest {
        text: "Hello, welcome to the stream!".to_string(),
        inference_timesteps: 4,
        min_len: 1,
        max_len: 3,
        retry_badcase: false,
        ..SynthesisRequest::default()
    };

    let mut chunks = Vec::new();
    engine.generate_streaming(request, |chunk| {
        chunks.push(chunk);
        Ok(())
    }).unwrap();

    assert!(!chunks.is_empty());
    assert!(chunks.iter().all(|chunk| chunk.sample_rate == engine.sample_rate()));
    assert!(chunks.iter().all(|chunk| !chunk.samples.is_empty()));
    assert!(chunks.iter().flat_map(|chunk| chunk.samples.iter()).all(|v| v.is_finite()));
    assert!(chunks.iter().enumerate().all(|(idx, chunk)| chunk.patch_index == idx));
    assert!(chunks.last().unwrap().is_final);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p voxui-inference voxcpm2_streaming_yields_finite_ordered_chunks -- --ignored`

Expected: compile failure because `generate_streaming` does not exist.

- [ ] **Step 3: Add streaming methods**

Add `generate_streaming` and `generate_streaming_cancellable` wrappers that validate the request and run one streaming attempt. Return an error if `request.retry_badcase` is true.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p voxui-inference voxcpm2_streaming_yields_finite_ordered_chunks`

Expected: pass if model weights are present.

## Task 3: Shared Generation Attempt

**Files:**
- Modify: `voxui/crates/voxui-inference/src/engine.rs`

- [ ] **Step 1: Write failing equivalence test**

Add a test that collects streaming chunks and compares basic aggregate behavior with `generate` under the same short deterministic configuration. Because random DiT noise can diverge between separate runs, assert structural equivalence rather than sample equality: non-empty output, finite samples, valid final chunk, and patch count within `max_len`.

- [ ] **Step 2: Refactor minimal implementation**

Extract the existing loop body from `run_generation_once` into a helper that accepts an optional per-patch callback:

```rust
fn run_generation_once_with_patches<F>(
    &mut self,
    prepared: &PreparedInputs,
    request: &SynthesisRequest,
    max_len: usize,
    progress: &impl Fn(usize, usize),
    cancel: Option<&AtomicBool>,
    on_patch: Option<F>,
) -> Result<GenerationOutput>
where
    F: FnMut(&GenerationState, usize, usize, bool) -> Result<()>;
```

Keep `run_generation_once` behavior unchanged by calling the helper with no patch callback.

- [ ] **Step 3: Run existing tests**

Run: `cargo test -p voxui-inference generate_flow_parity`

Expected: existing parity tests still pass.

## Task 4: Rolling-Window Chunk Decode

**Files:**
- Modify: `voxui/crates/voxui-inference/src/engine.rs`

- [ ] **Step 1: Write focused helper tests**

Add unit tests for pure helper math where possible:

- variant streaming prefix length returns 3 for VoxCPM 0.5/1.5 and 4 for VoxCPM 2.0.
- newest sample offset is `window_generated_patch_count - 1` patches times `patch_size * decode_chunk_size`.

- [ ] **Step 2: Implement chunk decode helper**

Add a helper that takes the current patch window, decodes it with `AudioVAE::decode`, and returns only the newest PCM region.

- [ ] **Step 3: Wire streaming callback**

Inside `generate_streaming_cancellable`, keep a bounded decode window, build each `SynthesisChunk`, and call `on_chunk`.

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
cargo test -p voxui-inference synthesis_chunk_is_public_api
cargo test -p voxui-inference voxcpm2_streaming_yields_finite_ordered_chunks
cargo test -p voxui-inference generate_flow_parity
```

Expected: all pass.

## Task 5: Final Verification

**Files:**
- All touched files.

- [ ] **Step 1: Format**

Run: `cargo fmt -p voxui-inference`

Expected: no formatting diff beyond touched files.

- [ ] **Step 2: Test**

Run: `cargo test -p voxui-inference`

Expected: pass, or document any model-weight-dependent skipped/failing tests separately.

- [ ] **Step 3: Inspect diff**

Run: `git diff -- voxui/crates/voxui-inference docs/superpowers/specs/2026-05-24-voxui-inference-streaming-synthesis-design.md docs/superpowers/plans/2026-05-24-voxui-inference-streaming-synthesis.md`

Expected: only streaming synthesis API, tests, and docs are changed.

