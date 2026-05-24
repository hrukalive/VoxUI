# AudioVAE Stateful Streaming Decoder Design

## Goal

Add a real stateful `AudioVAE` streaming decoder matching the Python VoxCPM 2.0 `StreamingVAEDecoder` design. The existing `VoxCPMEngine::generate_streaming*` public API remains unchanged, but it should use the stateful decoder instead of rolling-window full decodes.

## Reference Behavior

Python `AudioVAE.streaming_decode()` wraps the decoder so each `decode_chunk()` call processes only the newest latent chunk. It carries per-layer state for:

- `CausalConv1d`: previous left-padding samples are reused instead of zero padding every chunk.
- `CausalTransposeConv1d`: one previous input frame is reused and the left-overlap/right-trim region is removed.

This is not merely a speed optimization if the engine decodes one newest latent patch at a time. Without state, causal convolutions would reset history to zero at every chunk boundary.

## Rust Design

Add `StreamingAudioVaeDecoder` in `audiovae.rs`:

```rust
pub struct StreamingAudioVaeDecoder {
    vae: AudioVAE,
    states: StreamingConvStates,
}

impl AudioVAE {
    pub fn streaming_decoder(&self) -> StreamingAudioVaeDecoder;
}

impl StreamingAudioVaeDecoder {
    pub fn decode_chunk(&mut self, latent_chunk: &Tensor) -> Result<Tensor>;
    pub fn reset(&mut self);
}
```

The decoder owns a cheap clone of the `AudioVAE` weights, so `VoxCPMEngine` can hold a streaming decoder while mutably generating patches.

## Stateful Operations

Add explicit stateful versions of decoder operations instead of monkey-patching modules:

- `causal_conv1d_streaming`
- `causal_conv_transpose1d_streaming`
- `ResidualUnit::forward_streaming`
- `DecoderBlock::forward_streaming`
- `StreamingAudioVaeDecoder::decode_chunk`

Only causal convolutions with positive left padding need state. Kernel-1 pointwise convolutions keep using the normal stateless path.

## Engine Integration

In `VoxCPMEngine::generate_streaming_cancellable`, create one `StreamingAudioVaeDecoder` per synthesis call. For each generated patch, convert the newest `[B, patch_size, latent_dim]` feature patch to `[B, latent_dim, patch_size]`, call `decode_chunk`, and emit the returned PCM samples.

The rolling-window helper becomes unnecessary for engine streaming.

## Tests

Use existing VoxCPM2 golden latent data:

- Full decode: `vae.decode(generated_latent)`.
- Streaming decode: split `generated_latent` into patch-sized chunks, call `decode_chunk` for each, concatenate on time dimension.
- Assert the concatenated streaming output matches full decode within existing AudioVAE tolerance.

This directly proves the stateful decoder preserves full-decode behavior at chunk boundaries.

