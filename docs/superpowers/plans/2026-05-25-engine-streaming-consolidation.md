# Engine Streaming Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add engine-level `consolidate_n` support so streaming synthesis can decode and emit larger stateful chunks.

**Architecture:** Add `consolidate_n` to `SynthesisRequest` with default `1` and validation. In `VoxCPMEngine::generate_streaming_cancellable`, buffer generated patch tensors and flush the buffer through `StreamingAudioVaeDecoder::decode_chunk()` when it reaches `consolidate_n` or the final patch arrives. Expose the knob through the CLI as `--stream-consolidate-n` while preserving existing default behavior.

**Tech Stack:** Rust 2021, Candle tensors, Clap, existing `voxui-inference` and `voxui-cli` crates.

---

## File Structure

- Modify `voxui/crates/voxui-inference/src/request.rs`: add request field, default, validation.
- Modify `voxui/crates/voxui-inference/tests/request_validation.rs`: add request validation tests.
- Modify `voxui/crates/voxui-inference/src/engine.rs`: buffer streaming patches before decode, add tensor-order unit test.
- Modify `voxui/crates/voxui-cli/src/args.rs`: add CLI option and validation.
- Modify `voxui/crates/voxui-cli/src/main.rs`: pass CLI consolidation value into runner.
- Modify `voxui/crates/voxui-cli/src/runner.rs`: carry consolidation value into streaming request.
- Modify `voxui/crates/voxui-desktop/src-tauri/src/app_core.rs`: populate `consolidate_n: 1` for desktop batch requests.

## Task 1: Add Request Field And Validation

**Files:**
- Modify: `voxui/crates/voxui-inference/src/request.rs`
- Modify: `voxui/crates/voxui-inference/tests/request_validation.rs`

- [ ] **Step 1: Write failing request validation tests**

Add these tests to `voxui/crates/voxui-inference/tests/request_validation.rs`:

```rust
#[test]
fn request_default_consolidate_n_is_one() {
    assert_eq!(SynthesisRequest::default().consolidate_n, 1);
}

#[test]
fn request_rejects_zero_consolidate_n() {
    let err = SynthesisRequest {
        text: "hello".to_string(),
        consolidate_n: 0,
        ..SynthesisRequest::default()
    }
    .validated(ModelVariant::VoxCpm2)
    .unwrap_err();

    assert!(err.to_string().contains("consolidate_n"));
}
```

- [ ] **Step 2: Run tests and verify they fail**

Run from `voxui/`:

```powershell
cargo test -p voxui-inference --test request_validation request_default_consolidate_n_is_one request_rejects_zero_consolidate_n
```

Expected result: compile failure because `SynthesisRequest` has no `consolidate_n` field.

- [ ] **Step 3: Add the request field, default, and validation**

Update `voxui/crates/voxui-inference/src/request.rs`:

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
    pub consolidate_n: usize,
}
```

Add the default value:

```rust
consolidate_n: 1,
```

Add validation near the other numeric checks:

```rust
if self.consolidate_n == 0 {
    bail!("consolidate_n must be greater than zero");
}
```

- [ ] **Step 4: Run request validation tests and verify they pass**

Run from `voxui/`:

```powershell
cargo test -p voxui-inference --test request_validation
```

Expected result: all `request_validation` tests pass.

## Task 2: Implement Engine Streaming Consolidation

**Files:**
- Modify: `voxui/crates/voxui-inference/src/engine.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/app_core.rs`

- [ ] **Step 1: Write a tensor-order unit test for consolidated patch latent layout**

Add this test to the `#[cfg(test)] mod tests` block in `voxui/crates/voxui-inference/src/engine.rs`:

```rust
#[test]
fn patches_to_latent_preserves_patch_time_order_for_streaming_consolidation() {
    let device = candle_core::Device::Cpu;
    let patch_size = 2;
    let latent_dim = 3;
    let patch0 = Tensor::from_vec(
        vec![1f32, 2., 3., 4., 5., 6.],
        (1, patch_size, latent_dim),
        &device,
    )
    .unwrap();
    let patch1 = Tensor::from_vec(
        vec![7f32, 8., 9., 10., 11., 12.],
        (1, patch_size, latent_dim),
        &device,
    )
    .unwrap();

    let latent = patches_to_latent(&[patch0, patch1], latent_dim, patch_size).unwrap();

    assert_eq!(latent.dims3().unwrap(), (1, latent_dim, patch_size * 2));
    assert_eq!(
        latent.to_vec3::<f32>().unwrap(),
        vec![vec![
            vec![1., 4., 7., 10.],
            vec![2., 5., 8., 11.],
            vec![3., 6., 9., 12.],
        ]]
    );
}
```

- [ ] **Step 2: Run the new unit test and verify it passes before behavior changes**

Run from `voxui/`:

```powershell
cargo test -p voxui-inference patches_to_latent_preserves_patch_time_order_for_streaming_consolidation
```

Expected result: pass. This confirms the existing `patches_to_latent()` helper can produce the consolidated VAE latent layout.

- [ ] **Step 3: Buffer patches in `generate_streaming_cancellable`**

Replace the existing single-patch decode closure in `voxui/crates/voxui-inference/src/engine.rs` with buffered flushing logic:

```rust
let progress = |_, _| {};
let consolidate_n = request.consolidate_n;
let mut decoder = self.vae.streaming_decoder();
let mut pending_patches = Vec::with_capacity(consolidate_n);
let mut emit_chunk = |engine: &VoxCPMEngine,
                      state: &GenerationState,
                      step: usize,
                      max_patches: usize,
                      is_final: bool|
 -> Result<()> {
    let latest_patch = state
        .generated_patches
        .last()
        .context("streaming generation produced no patch")?;
    pending_patches.push(latest_patch.clone());

    if pending_patches.len() < consolidate_n && !is_final {
        return Ok(());
    }

    let latent_chunk = patches_to_latent(
        &pending_patches,
        engine.config.latent_dim,
        engine.config.patch_size,
    )?;
    let audio = decoder.decode_chunk(&latent_chunk.to_dtype(DType::F32)?)?;
    pending_patches.clear();

    on_chunk(SynthesisChunk {
        samples: audio.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?,
        sample_rate: engine.config.sample_rate,
        patch_index: step,
        max_patches,
        generated_patch_count: step + 1,
        is_final,
    })
};
```

- [ ] **Step 4: Update desktop request construction**

In `voxui/crates/voxui-desktop/src-tauri/src/app_core.rs`, add the new field to `synthesis_request()`:

```rust
consolidate_n: 1,
```

Place it after `retry_badcase_ratio_threshold` in the existing struct literal.

- [ ] **Step 5: Run inference tests affected by engine/request changes**

Run from `voxui/`:

```powershell
cargo test -p voxui-inference request_validation patches_to_latent_preserves_patch_time_order_for_streaming_consolidation streaming_request_rejects_retry_badcase
```

Expected result: all selected tests pass.

## Task 3: Expose Consolidation Through CLI

**Files:**
- Modify: `voxui/crates/voxui-cli/src/args.rs`
- Modify: `voxui/crates/voxui-cli/src/main.rs`
- Modify: `voxui/crates/voxui-cli/src/runner.rs`

- [ ] **Step 1: Write failing CLI validation test**

Update every `Args` struct literal in `voxui/crates/voxui-cli/src/args.rs` tests to include:

```rust
stream_consolidate_n: 1,
```

Then add this test to the same test module:

```rust
#[test]
fn validate_rejects_zero_stream_consolidate_n() {
    let tmp = std::env::temp_dir().join("voxui_cli_test_stream_consolidate_zero");
    let _ = std::fs::create_dir_all(&tmp);
    std::fs::write(tmp.join("model.gguf"), b"").unwrap();
    std::fs::write(tmp.join("config.json"), b"").unwrap();
    std::fs::write(tmp.join("tokenizer.json"), b"").unwrap();

    let args = Args {
        model: tmp.clone(),
        lora: None,
        cpu: true,
        stream: true,
        stream_consolidate_n: 0,
    };

    let err = args.validate().unwrap_err();
    assert!(err.to_string().contains("stream-consolidate-n"));

    let _ = std::fs::remove_dir_all(&tmp);
}
```

- [ ] **Step 2: Run CLI tests and verify they fail**

Run from `voxui/`:

```powershell
cargo test -p voxui-cli validate_rejects_zero_stream_consolidate_n
```

Expected result: compile failure because `Args` has no `stream_consolidate_n` field.

- [ ] **Step 3: Add CLI argument and validation**

Add this field to `Args` in `voxui/crates/voxui-cli/src/args.rs`:

```rust
/// Number of generated patches to decode and emit per streaming chunk
#[arg(long, default_value_t = 1, value_name = "N")]
pub stream_consolidate_n: usize,
```

Add validation before `Ok(())`:

```rust
if self.stream_consolidate_n == 0 {
    bail!("stream-consolidate-n must be greater than zero");
}
```

- [ ] **Step 4: Pass the value from main to runner**

In `voxui/crates/voxui-cli/src/main.rs`, capture the value near `stream`:

```rust
let stream = args.stream;
let stream_consolidate_n = args.stream_consolidate_n;
```

Update the call:

```rust
match runner.synthesize_and_play(&line, stream, stream_consolidate_n, Some(&cancel)) {
```

- [ ] **Step 5: Store the value in streaming requests**

In `voxui/crates/voxui-cli/src/runner.rs`, update the signature:

```rust
pub fn synthesize_and_play(
    &mut self,
    text: &str,
    stream: bool,
    stream_consolidate_n: usize,
    cancel: Option<&AtomicBool>,
) -> Result<()> {
```

Update dispatch:

```rust
if stream {
    self.synthesize_streaming(text, stream_consolidate_n, cancel)
} else {
    self.synthesize_batch(text, cancel)
}
```

Update `synthesize_streaming` signature:

```rust
fn synthesize_streaming(
    &mut self,
    text: &str,
    stream_consolidate_n: usize,
    cancel: Option<&AtomicBool>,
) -> Result<()> {
```

Add the request field:

```rust
let request = SynthesisRequest {
    text: text.to_string(),
    retry_badcase: false,
    consolidate_n: stream_consolidate_n,
    ..Default::default()
};
```

- [ ] **Step 6: Run CLI tests and verify they pass**

Run from `voxui/`:

```powershell
cargo test -p voxui-cli
```

Expected result: all `voxui-cli` tests pass.

## Task 4: Workspace Verification

**Files:**
- No additional source edits expected.

- [ ] **Step 1: Format Rust code**

Run from `voxui/`:

```powershell
cargo fmt
```

Expected result: command exits successfully and formats touched Rust files.

- [ ] **Step 2: Run targeted tests**

Run from `voxui/`:

```powershell
cargo test -p voxui-inference --test request_validation
cargo test -p voxui-inference patches_to_latent_preserves_patch_time_order_for_streaming_consolidation streaming_request_rejects_retry_badcase
cargo test -p voxui-cli
```

Expected result: all targeted tests pass.

- [ ] **Step 3: Run workspace check**

Run from `voxui/`:

```powershell
cargo check --workspace
```

Expected result: workspace compiles. If this exposes unrelated pre-existing failures, capture the exact errors and do not mask them with unrelated changes.

- [ ] **Step 4: Inspect final diff**

Run from repository root:

```powershell
git diff -- docs/superpowers/specs/2026-05-25-engine-streaming-consolidation-design.md docs/superpowers/plans/2026-05-25-engine-streaming-consolidation.md voxui/crates/voxui-inference/src/request.rs voxui/crates/voxui-inference/tests/request_validation.rs voxui/crates/voxui-inference/src/engine.rs voxui/crates/voxui-cli/src/args.rs voxui/crates/voxui-cli/src/main.rs voxui/crates/voxui-cli/src/runner.rs voxui/crates/voxui-desktop/src-tauri/src/app_core.rs
```

Expected result: diff only contains the spec, plan, consolidation request field, streaming buffer logic, CLI plumbing, and tests described above.

## Self-Review Notes

- Spec coverage: the plan implements engine-level `consolidate_n`, keeps `StreamingAudioVaeDecoder::decode_chunk()`, validates zero, flushes final partial buffers, preserves defaults, and exposes CLI as a wrapper.
- Placeholder scan: no `TBD`, `TODO`, or vague edge-case steps are intentionally present.
- Type consistency: the field name is consistently `consolidate_n` on `SynthesisRequest` and `stream_consolidate_n` on CLI args.
