# AudioVAE Stateful Streaming Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Python-style stateful `AudioVAE` streaming decoder and wire inference streaming to it.

**Architecture:** `AudioVAE` gains an owned `StreamingAudioVaeDecoder` with per-layer causal-conv state. `VoxCPMEngine::generate_streaming_cancellable` creates one decoder per synthesis call and decodes each newest generated latent patch directly.

**Tech Stack:** Rust 2021, Candle tensors, existing VoxCPM2 golden latent/audio parity tests.

---

## File Structure

- Modify `voxui/crates/voxui-inference/src/audiovae.rs`: add clone support, streaming state, stateful conv helpers, stateful decoder block/unit paths, and `StreamingAudioVaeDecoder`.
- Modify `voxui/crates/voxui-inference/src/engine.rs`: replace rolling-window chunk decode with stateful VAE chunk decode.
- Modify `voxui/crates/voxui-inference/tests/audiovae_parity.rs`: add full-vs-streaming decode parity test.

## Task 1: Failing Streaming Decoder Parity Test

- [ ] **Step 1: Add test**

Add `audiovae_streaming_decode_matches_full_decode_for_patch_chunks` in `audiovae_parity.rs`. It should load VoxCPM2 VAE, split `generated_latent` by patch size 4, call `vae.streaming_decoder().decode_chunk(...)`, concatenate outputs, and compare to `vae.decode(...)`.

- [ ] **Step 2: Run red**

Run: `cargo test -p voxui-inference audiovae_streaming_decode_matches_full_decode_for_patch_chunks`

Expected: compile failure because `AudioVAE::streaming_decoder` does not exist.

## Task 2: AudioVAE Stateful Decode

- [ ] **Step 1: Add state structs and clone derives**

Add `Clone` derives for decoder structs and `StreamingConvStates`.

- [ ] **Step 2: Add stateful helper functions**

Implement `causal_conv1d_streaming` and `causal_conv_transpose1d_streaming` following Python `StreamingVAEDecoder`.

- [ ] **Step 3: Add streaming forward methods**

Add `ResidualUnit::forward_streaming` and `DecoderBlock::forward_streaming`.

- [ ] **Step 4: Add public decoder**

Add `AudioVAE::streaming_decoder`, `StreamingAudioVaeDecoder::decode_chunk`, and `reset`.

- [ ] **Step 5: Run green**

Run: `cargo test -p voxui-inference audiovae_streaming_decode_matches_full_decode_for_patch_chunks`

Expected: pass.

## Task 3: Engine Integration

- [ ] **Step 1: Update engine streaming decode**

Create a stateful decoder inside `generate_streaming_cancellable` and decode only the newest generated patch.

- [ ] **Step 2: Run existing streaming API test**

Run: `cargo test -p voxui-inference voxcpm2_streaming_yields_finite_ordered_chunks -- --nocapture`

Expected: pass.

## Task 4: Final Checks

- [ ] **Step 1: Compile all inference tests**

Run: `cargo test -p voxui-inference --no-run`

- [ ] **Step 2: Run fast targeted tests**

Run:

```powershell
cargo test -p voxui-inference synthesis_chunk_is_public_api
cargo test -p voxui-inference streaming_request_rejects_retry_badcase
cargo test -p voxui-inference streaming_prefix_len_matches_python_reference_defaults
```

- [ ] **Step 3: Inspect scoped diff**

Run: `git diff -- voxui/crates/voxui-inference/src/audiovae.rs voxui/crates/voxui-inference/src/engine.rs voxui/crates/voxui-inference/tests/audiovae_parity.rs`

