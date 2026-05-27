# Sidecar Inference and Streaming Audio Design

Date: 2026-05-27

## Summary

VoxUI should move model inference out of the Tauri process and into a long-lived inference sidecar process. The Tauri backend remains the control plane: it owns queueing, cancellation, history state, playback state, and frontend events. The sidecar owns `VoxCPMEngine`, model loading, and synthesis execution.

Playback should remain in the Rust backend through `voxui-audio`, but its streaming playback path must be upgraded into a first-class, sample-rate-independent PCM streaming engine. Streaming and non-streaming synthesis should share the same playback path: streaming pushes many PCM chunks, while non-streaming pushes one final PCM buffer.

## Goals

- Isolate model loading and synthesis from the Tauri UI process.
- Preserve one-at-a-time synthesis execution.
- Keep request scheduling and queue state in the Tauri backend.
- Support both streaming and non-streaming synthesis.
- Allow users to queue requests and cancel queued or active generation.
- Preserve model loading progress and synthesis progress.
- Start playback during streaming synthesis when chunks arrive.
- Continue using configured host/device output through `voxui-audio`.
- Make playback independent of model output sample rate by using a stateful resampler per playback session.

## Non-Goals

- Browser-native audio playback is not part of this design.
- Opus, MP3, WebM, or other encoded streaming formats are not part of the internal playback path.
- Multiple concurrent synthesis jobs are not supported.
- A separate audio process is not required unless future evidence shows CPAL/audio failures are destabilizing the app process.
- Persistent encoded audio export is out of scope for the first implementation.

## Current State

The current app already has most control-plane concepts in place:

- `crates/voxui-desktop/src-tauri/src/app_core.rs` owns configuration, model state, queue state, active generation, active playback, cancellation flags, and audio cache.
- `crates/voxui-desktop/src-tauri/src/generation_queue.rs` stores history items and queue transitions.
- `crates/voxui-desktop/src-tauri/src/commands.rs` starts background threads, emits Tauri events, and kicks the next queued generation.
- `crates/voxui-inference/src/engine.rs` supports cancellable model loading, batch generation, and streaming generation.
- `crates/voxui-audio/src/lib.rs` contains a `StreamingPlayer` prototype with a persistent `r8brain_rs::Resampler`, but it is not ready to be the main streaming playback path.

The main architectural limitation is that `VoxCPMEngine` is currently stored in `AppCore` and runs inside the Tauri process. Streaming generation also accumulates audio into a final `Vec<f32>` before the item becomes ready, so the existing UI path does not fully treat streaming as live audio playback.

## Architecture

### Process Layout

```text
Leptos frontend
  <-> Tauri commands/events

Tauri backend process
  - queue and history state
  - sidecar lifecycle
  - active generation state
  - playback state
  - audio cache
  - frontend event emission

Inference sidecar process
  - VoxCPMEngine
  - model loading
  - LoRA loading
  - streaming and non-streaming synthesis

voxui-audio
  - CPAL output stream
  - selected host/device playback
  - stateful resampling
  - streaming ring buffer
```

The sidecar is a worker. It does not own queueing, history, playback decisions, or UI state.

### Tauri Backend Responsibilities

The Tauri backend remains authoritative for:

- accepting frontend commands;
- validating user-visible state transitions;
- adding items to `GenerationQueue`;
- selecting the next queued item;
- ensuring only one active synthesis request is sent to the sidecar;
- canceling queued items directly;
- forwarding active generation cancellation to the sidecar;
- receiving sidecar progress, audio, done, and error messages;
- updating `AppCore` snapshots;
- emitting frontend events;
- starting, stopping, and draining audio playback sessions;
- preserving completed audio in the existing audio cache.

### Sidecar Responsibilities

The inference sidecar owns one optional loaded engine and accepts commands from Tauri:

- `load_model`
- `cancel_load`
- `synthesize`
- `cancel_synthesis`
- `shutdown`

The sidecar processes commands serially. It may reject a new synthesis command while model loading or synthesis is active. In normal operation, Tauri prevents that condition.

### Communication Protocol

The sidecar protocol will use length-prefixed frames over piped stdio. Each frame contains a JSON header and an optional binary payload. Control-only frames have no payload. Audio frames carry PCM `f32` samples as little-endian bytes in the payload. This avoids base64 overhead while keeping message parsing deterministic.

The Tauri backend will launch the packaged sidecar executable and own its stdin/stdout pipes. Stderr is reserved for logs and diagnostics, not protocol messages.

Message families:

- Commands from Tauri to sidecar:
  - `LoadModel`
  - `CancelLoad`
  - `Synthesize`
  - `CancelSynthesis`
  - `Shutdown`
- Events from sidecar to Tauri:
  - `Ready`
  - `ModelLoadProgress`
  - `ModelLoadDone`
  - `GenerationProgress`
  - `AudioChunk`
  - `AudioFinal`
  - `GenerationDone`
  - `Error`

Every synthesis-related event includes `item_id`. Stale events for a no-longer-active item are ignored by Tauri.

## Synthesis Modes

### Streaming Mode

For streaming requests, the sidecar calls `VoxCPMEngine::generate_streaming_cancellable`. Each generated `SynthesisChunk` becomes:

- a `GenerationProgress` event;
- an `AudioChunk` event containing mono PCM `f32` samples, source sample rate, generated patch count, max patch count, and final-chunk flag.

Tauri starts or reuses the active playback session for that item and pushes each chunk into `voxui-audio`. It also accumulates PCM samples in memory so the completed history item can be replayed after synthesis finishes.

### Non-Streaming Mode

For non-streaming requests, the sidecar calls `VoxCPMEngine::generate_cancellable`. It emits progress events during generation and sends one `AudioFinal` message when generation succeeds.

Tauri handles `AudioFinal` through the same playback abstraction by pushing the full buffer once and finishing the playback session. The item is marked ready only after final audio is received.

### Retry Behavior

The existing rule remains:

- streaming disables `retry_badcase` at request construction;
- non-streaming may use `retry_badcase` according to user configuration.

## Queueing and Cancellation

Queueing stays in `GenerationQueue`.

Queued item cancellation is local to Tauri:

- if an item is still queued, mark it canceled and never send it to the sidecar.

Active item cancellation is cooperative:

- Tauri marks the item canceled or ready-with-existing-audio according to current queue rules;
- Tauri sends `CancelSynthesis` to the sidecar;
- Tauri stops the active audio playback session for that item;
- Tauri ignores later stale chunks for that item.

If cooperative cancellation does not complete within a bounded timeout, Tauri may terminate and restart the sidecar. This is a fallback, not the normal path.

## Model Loading

Model loading moves from Tauri background threads into the sidecar. Tauri still owns UI state:

- `load_state`;
- selected model;
- loaded model id;
- load progress modal state through frontend events.

The sidecar emits component progress equivalent to the current `model_load_progress` event. On success, Tauri records the loaded model id. On failure or cancellation, Tauri clears the active load state without losing the previous loaded model id unless the sidecar process was restarted and no longer has that previous engine loaded.

If the sidecar crashes, Tauri must treat the loaded engine as gone and clear `loaded_model_id`.

## Audio Playback Engine

`voxui-audio` should expose a production streaming playback abstraction. The current `StreamingPlayer` prototype already owns one `r8brain_rs::Resampler` per player instance, so it is stateful across chunks. It needs these corrections before use:

- split incoming chunks into blocks no larger than `Resampler::max_input_len()`;
- size the ring buffer by device sample rate, not source sample rate;
- support configured host/device instead of only the default output device;
- use a `VolumeHandle` for live volume changes;
- support explicit `stop`;
- support explicit `finish`, which flushes the resampler tail and drains the ring buffer;
- report completion back to Tauri;
- handle underruns predictably;
- expose enough status for tests.

The intended session lifecycle is:

```text
start session
  open selected CPAL output device
  create one resampler for source_sample_rate -> device_sample_rate
  create ring buffer sized at device sample rate
  begin output stream

push chunk
  split input into valid r8brain blocks
  process through the same resampler
  push resampled output into ring buffer

finish
  flush resampler tail
  push flushed output
  drain ring buffer
  complete session

stop
  stop stream
  clear pending audio
  complete as stopped
```

For non-streaming playback, the same session receives one large push followed by `finish`.

## Frontend Behavior

The Leptos frontend continues to use Tauri commands and events:

- `load_model`
- `cancel_model_load`
- `enqueue_generation`
- `cancel_generation`
- `play_audio`
- `stop_audio`
- `regenerate`

Frontend event listeners remain long-lived at the root app level. The frontend does not receive raw audio chunks in this design. It receives progress, done, load, and playback state events and refreshes the snapshot as it does today.

## Error Handling

Sidecar command errors:

- Tauri marks the active operation failed and emits the corresponding frontend event.

Sidecar crash:

- active generation is marked failed unless it had already been canceled;
- active load is marked failed;
- active playback is stopped if it depends on the failed generation;
- loaded model state is cleared;
- queued items remain queued;
- sidecar restart is attempted lazily on the next load or generation command.

Audio playback errors:

- Tauri emits `playback_state` with an error state;
- the history item remains ready if generated audio exists;
- generation continues unless the error is tied to active streaming playback and the user cancels.

Stale events:

- Tauri ignores sidecar events whose `item_id` or load id no longer matches the active operation.

## Testing Strategy

### Unit Tests

- Queue transitions for queued, generating, canceled, failed, ready, and playing states.
- Tauri-side stale event rejection.
- Sidecar protocol encode/decode.
- Active generation cancellation state transitions.
- Non-streaming final audio handling.
- Streaming chunk accumulation and ready-state transition.

### Audio Tests

- Chunked resampling should closely match whole-buffer resampling for the same input.
- `push` must split input larger than `max_input_len`.
- `finish` must flush the resampler tail.
- Ring buffer sizing must account for device sample rate.
- `stop` must clear pending playback and return promptly.
- Volume changes must affect later output reads.

### Integration Tests

- Load model progress reaches done.
- Enqueue two items and verify sidecar receives only one synthesis at a time.
- Cancel active streaming generation and verify playback stops and queued next item can start.
- Non-streaming generation emits no audio chunks before final audio.
- Sidecar crash clears loaded model state and fails active work.

## Implementation Notes

- Prefer a separate workspace binary crate for the sidecar, for example `crates/voxui-inference-sidecar`.
- Keep shared protocol types in a small crate or module that both Tauri and the sidecar can use.
- Keep `voxui-inference` independent of Tauri.
- Keep `voxui-audio` independent of Tauri.
- Do not move queueing into the sidecar.
- Do not make frontend audio responsible for raw playback in this design.
- Keep completed audio cache memory-only for the first implementation.
- Use a 250 ms default streaming prebuffer before starting playback; make this an internal constant first, not a user setting.
- Restart the sidecar lazily on the next load or generation command after a crash.
