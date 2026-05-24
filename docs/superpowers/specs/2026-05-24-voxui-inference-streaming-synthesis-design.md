# VoxUI Inference Streaming Synthesis Design

## Goal

Add a clean streaming synthesis API to `voxui-inference` without changing the existing `generate`, `generate_cancellable`, or `synthesize` public APIs.

The new API should expose synthesis as incremental audio chunks with engine-level metadata. Batch synthesis and streaming synthesis should share the same generation machinery where possible, while preserving current retry behavior for batch synthesis.

## Scope

This work is limited to the Rust inference crate:

- `voxui/crates/voxui-inference/src/engine.rs`
- `voxui/crates/voxui-inference/src/lib.rs`
- focused tests under `voxui/crates/voxui-inference/tests`

GUI integration, desktop commands, playback, and Tauri/Leptos files are out of scope.

## Reference Behavior

The Python references expose two useful streaming patterns:

- VoxCPM 0.5 and 1.5 stream by generating one audio-feature patch at a time, decoding a recent latent window, and emitting the newest PCM region.
- VoxCPM 2.0 streams by yielding one newest latent patch into `audio_vae.streaming_decode()`, whose stateful decoder keeps causal-convolution state between chunks.

The Rust engine already has the important generation state:

- request validation and input preparation
- base and residual LM cache setup
- per-patch DiT generation through `generate_one_patch`
- stop-logit evaluation
- final full-buffer AudioVAE decode

The first Rust streaming implementation should preserve this behavior while introducing a stable engine API. A stateful Rust AudioVAE streaming decoder can be added later behind the same public API.

## Public API

Add a metadata-rich public chunk type:

```rust
#[derive(Debug, Clone)]
pub struct SynthesisChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub patch_index: usize,
    pub max_patches: usize,
    pub generated_patch_count: usize,
    pub is_final: bool,
}
```

Add streaming methods to `VoxCPMEngine`:

```rust
pub fn generate_streaming<F>(
    &mut self,
    request: SynthesisRequest,
    on_chunk: F,
) -> Result<()>
where
    F: FnMut(SynthesisChunk) -> Result<()>;

pub fn generate_streaming_cancellable<F>(
    &mut self,
    request: SynthesisRequest,
    on_chunk: F,
    cancel: Option<&AtomicBool>,
) -> Result<()>
where
    F: FnMut(SynthesisChunk) -> Result<()>;
```

Re-export `SynthesisChunk` from `lib.rs`.

The existing APIs remain source-compatible:

- `generate(request, progress) -> Result<Vec<f32>>`
- `generate_cancellable(request, progress, cancel) -> Result<Vec<f32>>`
- `synthesize(text, dit_steps, progress) -> Result<Vec<f32>>`

## Retry Semantics

Batch generation keeps the current retry-on-badcase behavior. Bad attempts are not externally visible because batch generation only returns after an attempt is accepted.

Streaming generation must not retry after emitting audio. If `retry_badcase` is enabled in a streaming request, the streaming API should return a clear validation error. This matches the Python VoxCPM 2.0 reference, which disables retry in streaming mode because emitted audio cannot be withdrawn.

## Internal Architecture

Split the engine flow into reusable units:

1. Prepare inputs and compute `max_len`.
2. Run one generation attempt over `GenerationState`.
3. Emit each generated patch to a decode/output path.
4. Stop on cancellation, max length, or stop logits.

Batch synthesis should call a shared attempt function and decode only the accepted final latent sequence.

Streaming synthesis should call the same attempt loop with a per-patch callback. Each generated patch is converted into a PCM `SynthesisChunk` and passed to the caller immediately.

## Streaming Decode Strategy

The first implementation should use bounded rolling-window decode:

- Keep the same variant-specific prefix lengths already used by the engine: 3 for VoxCPM 0.5/1.5 and 4 for VoxCPM 2.0.
- Include continuation prompt context patches when present, using the existing `continuation_context_len` and `initial_context_patches` behavior.
- Decode the recent latent window with the current `AudioVAE::decode`.
- Emit only the newest audio region for each generated patch.

This avoids expanding `audiovae.rs` with stateful causal-convolution cache logic in the same change. The public API remains compatible with a later `StreamingAudioVaeDecoder` optimization.

## Error Handling

Streaming should return errors for:

- invalid `SynthesisRequest`
- cancellation
- callback errors
- model generation failures
- `retry_badcase=true`

The callback returns `Result<()>` so callers can stop streaming with their own error.

## Testing

Add focused tests for:

- `SynthesisChunk` public export and metadata expectations where practical.
- streaming rejects `retry_badcase=true`.
- collected streaming chunks match the existing full synthesis result for a short deterministic request, using existing model/golden infrastructure where feasible.
- existing generate-flow parity tests still pass.

Where deterministic sample equality is not practical because DiT noise is random, test invariants such as non-empty finite chunks, sample rate, monotonic patch index, and final chunk marker.

## Out of Scope

- GUI or desktop integration.
- Playback buffering.
- Prompt-cache APIs.
- A stateful Rust `AudioVAE` streaming decoder.
- Changing `SynthesisRequest` fields.
- Changing current batch synthesis signatures.

