# Engine Streaming Consolidation Design

## Goal

Add an engine-level streaming consolidation knob so callers can trade first-audio latency for better streaming throughput. The engine should still use stateful streaming VAE decode, but decode and emit multiple generated patches at a time.

## Motivation

Current streaming synthesis decodes and emits one generated patch at a time. Each patch forces a VAE decode over a very small latent tensor, followed by GPU-to-CPU transfer through `to_vec1::<f32>()`. This creates many small kernel launches, stateful convolution buffer updates, tensor concatenations, and synchronization points.

Consolidating generated patches before VAE decode makes each streaming decode chunk larger. This should reduce GPU/CPU sync frequency and improve VAE decoder efficiency while preserving stateful streaming behavior.

## Public API

Add a new field to `SynthesisRequest`:

```rust
pub consolidate_n: usize,
```

Semantics:

- Default value is `1`.
- `1` preserves current streaming behavior exactly.
- Values greater than `1` buffer that many generated patches before decoding and emitting audio.
- `0` is invalid and should be rejected during request validation.

This is engine-level configuration, not just CLI behavior. Any caller using `VoxCPMEngine::generate_streaming()` or `generate_streaming_cancellable()` gets the same behavior.

## Streaming Data Flow

Current behavior:

```text
generate patch -> decode patch -> copy samples to CPU -> emit chunk
```

New behavior with `consolidate_n = N`:

```text
generate patch -> buffer
generate patch -> buffer
...
buffer reaches N or final patch -> concatenate buffered patches -> streaming decode once -> copy samples to CPU -> emit chunk
```

The engine should continue using `StreamingAudioVaeDecoder::decode_chunk()`. It must not switch to `AudioVAE::decode()` for streaming consolidation. The decoder state must persist across consolidated chunks.

## Patch Buffering

Generated patches are currently stored in `GenerationState::generated_patches` in patch format `[1, patch_size, latent_dim]`. For VAE decode, buffered patches should be converted to latent format and concatenated along latent time:

```text
[1, patch_size, latent_dim] -> transpose -> [1, latent_dim, patch_size]
concat dim 2 -> [1, latent_dim, patch_size * buffered_count]
```

The buffer should flush when either condition is true:

- `buffered_count == request.consolidate_n`
- `is_final == true`

The final flush may contain fewer than `consolidate_n` patches.

## SynthesisChunk Metadata

`SynthesisChunk` should remain minimal unless implementation reveals a concrete need for extra metadata.

Use existing fields as follows:

- `samples`: merged audio samples for all patches in the flushed buffer.
- `sample_rate`: unchanged.
- `patch_index`: index of the latest generated patch included in this emitted chunk.
- `max_patches`: unchanged.
- `generated_patch_count`: total generated patch count so far.
- `is_final`: true only for the final emitted chunk.

Do not add `patches_in_chunk` unless a caller requires it.

## Validation

Validation should reject `consolidate_n == 0` for all synthesis requests. Batch generation does not need to use the field, but validation should remain consistent so invalid requests fail early.

Streaming still rejects `retry_badcase = true` as it does today.

## CLI Exposure

The core design is engine-level. CLI exposure can be added as a thin wrapper:

```text
--stream-consolidate-n <N>
```

Default is `1`. The CLI should pass the value into `SynthesisRequest::consolidate_n` only for streaming requests. Batch behavior remains unchanged.

## Performance Expectations

Increasing `consolidate_n` should reduce:

- GPU-to-CPU transfer frequency.
- Host synchronization points caused by sample extraction.
- Number of VAE decoder forwards.
- Number of small-tensor kernel launches.
- Number of stateful convolution buffer updates and temporary tensor concatenations.

The tradeoff is higher first-audio latency. Very large values approach batch behavior but remain implemented through stateful streaming decode.

## Testing

Add or update tests to cover:

- `SynthesisRequest::default()` sets `consolidate_n = 1`.
- Validation rejects `consolidate_n = 0`.
- Streaming with `consolidate_n = 1` preserves existing chunk count behavior.
- Streaming with `consolidate_n > 1` emits fewer chunks and flushes the final partial buffer.
- Consolidated streaming decode remains numerically close to full decode for the same latent patches.

Benchmarking should compare streaming total synthesis time for representative values such as `1`, `2`, `4`, and `8`.

## Non-Goals

- Do not replace streaming decode with full batch decode.
- Do not change non-streaming generation behavior.
- Do not add a second public streaming API unless the existing `SynthesisRequest` field proves insufficient.
- Do not add extra `SynthesisChunk` metadata without a caller requirement.
