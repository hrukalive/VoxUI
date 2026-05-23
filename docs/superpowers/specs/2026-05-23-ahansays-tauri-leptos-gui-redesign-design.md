# AhanSays Tauri/Leptos GUI Redesign Design

## Context

VoxUI needs a fresh desktop GUI for a TTS workflow using Tauri and Leptos. The new desktop app is named `焓言焓语` in Chinese and `AhanSays` in English. The design starts from the active library crates only:

- `voxui-inference` provides VoxCPM model loading, optional LoRA loading, cancellable generation, generation progress, and `SynthesisRequest`.
- `voxui-audio` provides audio host/device discovery, r8brain sample-rate conversion, and playback through CPAL.
- `voxui-gguf` provides GGUF model parsing support used by inference.

The redesigned desktop app is a new `voxui/crates/voxui-desktop` crate built from these active crates and the requirements in this spec.

## Goals

- Build a clean Tauri 2 + Leptos CSR desktop app.
- Support Chinese and English UI, with language detected from the system on first launch and manually overrideable in settings.
- Discover models from a configurable model root directory, defaulting to `models` next to the executable.
- Flatten model and LoRA choices into one dropdown.
- Keep one loaded model engine at a time.
- Make model loading explicit, cancellable, and visible through a progress modal.
- Let users enqueue text generation jobs sequentially.
- Show generation history with per-item progress, cancel, regenerate, play, and stop controls.
- Play generated audio automatically on successful generation through the configured audio driver/device and volume.
- Preserve previous successful audio when a regeneration attempt is canceled or fails.
- Expose important VoxCPM generation parameters and maximum input character count in settings.

## Non-Goals

- No streaming playback during generation; playback begins after a generation succeeds.
- No parallel generation.
- No remote server mode.
- No legacy GUI compatibility layer.
- No WAV export workflow in the first redesign unless added later as a separate feature.

## Recommended Approach

Use a fresh `voxui-desktop` crate with a typed backend app-core layer.

The Leptos frontend owns visible state and user interaction. The Tauri backend owns consistency: loaded engine state, cancellation tokens, queue order, generated audio cache, playback, config persistence, and filesystem/device discovery. Long-running model loading and generation work run off the UI thread and communicate through typed Tauri events.

This avoids splitting critical state between WebView and Rust while staying simpler than a full actor system.

## Project Structure

```text
voxui/
  crates/
    voxui-desktop/
      Cargo.toml
      Trunk.toml
      index.html
      src/
        main.rs
        app.rs
        i18n.rs
        tauri_api.rs
        components/
          header.rs
          history.rs
          input_box.rs
          settings_modal.rs
          load_progress_modal.rs
      src-tauri/
        Cargo.toml
        tauri.conf.json
        build.rs
        src/
          main.rs
          lib.rs
          app_core.rs
          audio.rs
          commands.rs
          config.rs
          generation_queue.rs
          model_discovery.rs
          playback.rs
          types.rs
```

The Rust backend modules are intentionally small:

- `app_core.rs`: shared app state and high-level operations.
- `model_discovery.rs`: model root scanning and stable choice ids.
- `generation_queue.rs`: sequential queue and generation cancellation.
- `playback.rs`: generated audio cache, play/stop, volume scaling.
- `audio.rs`: host/device listing and sine-wave test playback.
- `config.rs`: persisted config load/save/migration.
- `commands.rs`: Tauri command wrappers.
- `types.rs`: serializable DTOs shared with the frontend.

## UI Layout

The main window uses a single workbench layout.

### Header

The header contains:

- Localized title: `焓言焓语` in Chinese, `AhanSays` in English.
- Model choice dropdown.
- `Load` button.
- Settings icon button that opens a modal.

The model dropdown displays a flattened model list. Selecting an option changes only `selected_model_choice`; it does not unload or reload the engine. The `Load` button is enabled when a choice is selected, no model load is active, no generation is running, and the selected choice differs from the loaded choice.

### Generation History

The center of the app is a chronological generation history. Each item shows:

- Input text.
- Status: queued, generating, canceled, failed, ready, playing.
- Progress for queued/running work.
- Model choice used for the attempt.
- Controls appropriate for the state.

Queued and running items show a progress bar and `Cancel`. Finished items show `Play` or `Stop` and `Regenerate`. Failed and canceled items remain visible with their error/cancel state. If an item has previous successful audio and a regeneration attempt is canceled or fails, the old audio remains playable.

### Composer

The bottom area contains a multiline text box, character counter, and `Generate` button. `Generate` is enabled only when:

- A model is loaded.
- No model load is active.
- Text is non-empty after trimming.
- Text length is at or below the configured maximum input character count.

Pressing `Generate` appends a history item and enqueues it. The queue is processed sequentially.

### Model Load Progress Modal

Model loading uses a modal progress surface. It has a cancel button and two progress phases:

1. `reading`: determinate byte progress across `model.gguf` and the optional LoRA `.gguf`.
2. `device_loading`: component progress from the inference engine, covering major components such as `base_lm`, `residual_lm`, `feat_encoder`, `feat_decoder`, `audio_vae`, and projections.

Canceling load leaves the previous loaded engine available.

### Settings Modal

Settings are grouped into sections:

- Models: model discovery directory with a browse button and rescan.
- Interface: language selection (`System`, `中文`, `English`).
- Inference: CPU/CUDA backend.
- Audio: driver/host, output device, test button, and volume.
- Generation: important VoxCPM parameters.
- Input: maximum input character count.
- Advanced prompt/reference: prompt WAV, prompt text, and reference WAV.

Settings are locked while model loading is active. Settings that would affect queued generation are captured at enqueue time so each history item is reproducible.

## Model Discovery

The model root defaults to `models` next to the desktop executable. In development, the app may fall back to the repository `models` folder only if executable-relative discovery does not exist.

A model directory is valid when it contains `model.gguf`.

For each valid model directory:

- Emit one base choice displayed as the directory name.
- Emit one LoRA choice per direct `.gguf` file in the same directory except `model.gguf`.

Example:

```text
models/
  voxcpm2-fp16/
    model.gguf
    lora_a1.gguf
    lora_a2.gguf
```

Produces:

```text
voxcpm2-fp16
voxcpm2-fp16 | lora_a1
voxcpm2-fp16 | lora_a2
```

Each choice includes:

- Stable id.
- Display name.
- Model directory.
- `model.gguf` path.
- Optional LoRA path.
- Known file sizes for progress display.

Stable ids are derived from the model directory path relative to the model root plus the optional LoRA file name.

## Backend State

The backend app core owns:

- Current config.
- Discovered model choices.
- Selected model choice id.
- Loaded model choice id.
- Optional `VoxCPMEngine`.
- Optional active model-load cancellation token.
- Generation queue.
- Optional active generation cancellation token.
- History item metadata.
- Generated audio cache by history item id.
- Optional active `AudioPlayer`.

There is only one loaded engine slot. A load operation builds a temporary engine and swaps it into the loaded slot only after the full base model and optional LoRA load succeeds.

## Tauri Commands

The backend exposes typed Tauri commands:

- `get_app_state() -> AppSnapshot`
- `set_config_patch(patch: ConfigPatch) -> AppSnapshot`
- `browse_model_dir() -> Option<String>`
- `browse_prompt_wav() -> Option<String>`
- `browse_reference_wav() -> Option<String>`
- `discover_models() -> Vec<ModelChoice>`
- `load_model(choice_id: String) -> LoadStartResult`
- `cancel_model_load() -> CommandResult`
- `enqueue_generation(text: String) -> HistoryItem`
- `cancel_generation(item_id: String) -> CommandResult`
- `regenerate(item_id: String) -> CommandResult`
- `play_audio(item_id: String) -> CommandResult`
- `stop_audio() -> CommandResult`
- `test_audio() -> CommandResult`

`load_model` cancels any active load before starting the new one. It does not evict the current loaded engine unless the new load succeeds.

`enqueue_generation` captures the current loaded model id and generation settings for that item. Queue processing starts automatically if idle.

`regenerate` creates a new attempt for an existing history item. It preserves existing audio until a new generation succeeds.

## Tauri Events

Backend events are the source of truth for progress and terminal state:

- `model_load_progress`: `{ phase, loaded_bytes, total_bytes, component, component_index, component_total }`
- `model_load_done`: `{ status, selected_model_id, loaded_model_id, error }`
- `generation_progress`: `{ item_id, current, total }`
- `generation_done`: `{ item_id, status, error, sample_rate, duration_seconds }`
- `playback_state`: `{ item_id, state }`
- `settings_changed`: `{ snapshot }`
- `models_changed`: `{ models, selected_model_id }`
- `app_error`: `{ message, context }`

Status values should be explicit strings such as `success`, `canceled`, and `error`.

## Loading Semantics

When the user presses `Load`:

1. Cancel any active model load.
2. Create a fresh cancellation token.
3. Emit `model_load_progress` in `reading` phase while reading `model.gguf` and optional LoRA bytes.
4. Load the base engine with CPU or CUDA according to settings.
5. Emit component progress while the engine loads major components.
6. Load optional LoRA into the temporary engine.
7. If successful, replace the loaded engine slot and loaded model id.
8. If canceled or failed, keep the previous loaded engine and loaded model id.

If the user changes the dropdown while loading, the in-flight load target is unchanged. Pressing `Load` again after the active load resolves starts a load for the then-selected choice.

## Generation Semantics

Generation is sequential. The queue processes one item at a time.

For each item:

1. Validate text and request settings.
2. Build a `SynthesisRequest` using the captured settings.
3. Run `VoxCPMEngine::generate_cancellable`.
4. Emit progress for the item.
5. On success, cache PCM samples and metadata under the item id.
6. Automatically start playback using selected audio settings and volume.

Canceling a queued item marks it canceled before it starts. Canceling the active item signals its cancellation token. Canceling an active regeneration attempt restores the visible item to its previous playable output if one exists.

## Playback Semantics

Generated audio may have a sample rate different from the selected output device. Playback goes through `voxui-audio`, which uses r8brain conversion before CPAL output.

Before playback, the backend applies the configured volume to samples. Starting playback for one item stops any existing playback first. `Stop` stops playback without changing generation history.

The audio test button plays a sine wave through the selected host/device. The generated sine wave has fade-in and fade-out to avoid clicks and uses the same volume path as generated audio.

## Configuration

Persisted config includes:

- Model root.
- Last selected model choice id.
- Language mode: system, Chinese, or English.
- Backend: CPU or CUDA.
- Audio host/driver.
- Audio output device.
- Volume.
- Prompt WAV path.
- Prompt text.
- Reference WAV path.
- VoxCPM generation parameters.
- Maximum input character count.

On first launch, language is inferred from system locale. If the locale starts with `zh`, use Chinese; otherwise use English.

## VoxCPM Parameters

Expose these important generation parameters initially:

- `cfg_value`
- `inference_timesteps`
- `min_len`
- `max_len`
- `retry_badcase`
- `retry_badcase_max_times`
- `retry_badcase_ratio_threshold`

Keep `normalize` disabled until Rust-side normalization is implemented by `voxui-inference`.

## Error Handling

- No models found: disable dropdown/load/generate, show a clear message, keep settings available.
- Model load error: keep selected choice, keep previous loaded engine, close or update progress modal with error.
- Model load cancellation: restore previous loaded state.
- CUDA unavailable: show an explicit load error unless the user chose CPU.
- Invalid generation request: mark the item failed with the validation message.
- Generation cancellation: mark the item canceled while preserving old audio if applicable.
- Audio device failure: mark playback failed but keep generated audio available.
- Device list failure: show the error in settings and allow retry/rescan.

## Testing Strategy

Use test-first changes where practical.

Backend unit tests:

- Default model root resolution.
- Model discovery with base-only and base-plus-LoRA directories.
- Stable model choice ids.
- Config round trip.
- System language detection fallback.
- Load button state derivation.
- Queue sequential behavior.
- Regeneration preserves old audio on cancellation/failure.
- Settings patch persistence and rescan triggers.
- Sine wave fade-in/fade-out sample generation.

Integration or command tests:

- Failed/canceled load does not replace an existing loaded engine.
- Generation command rejects when no model is loaded.
- Generation captures settings at enqueue time.
- Playback command rejects unknown item ids and stops current playback before starting another.

Compile/build verification:

- `cd voxui; cargo test -p voxui-desktop`
- `cd voxui; cargo check -p voxui-desktop`
- `cd voxui; cargo check -p voxui-desktop --features cuda`
- `cd voxui; cargo test -p voxui-inference`
- `cd voxui/crates/voxui-desktop; trunk build`

## Acceptance Criteria

- The app launches as `焓言焓语` or `AhanSays` according to language.
- The UI detects system language on first launch and allows manual language selection.
- Model discovery defaults to executable-adjacent `models` and can be changed in settings.
- The dropdown shows base and LoRA-expanded model choices.
- `Load` is explicit, cancellable, and shows byte and component progress.
- Loading a new model never leaves two engines active.
- Failed or canceled loading does not evict the previous loaded model.
- Users can generate text only after a model is loaded.
- Generation history shows per-item progress and supports cancellation.
- Completed items support automatic playback, manual play/stop, and regeneration.
- Sequential queue order is preserved.
- Regeneration cancellation does not delete prior playable audio.
- Audio output respects selected driver/device and volume.
- r8brain conversion is used when generated sample rate differs from device rate.
- Settings include model directory, language, backend, audio, volume, key VoxCPM parameters, and max input characters.
