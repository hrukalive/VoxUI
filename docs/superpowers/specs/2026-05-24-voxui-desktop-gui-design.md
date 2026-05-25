# VoxUI Desktop GUI Rewrite Design

Date: 2026-05-24

## Goal

Create a new end-to-end desktop GUI crate at `voxui/crates/voxui-desktop` using Tauri, Svelte, Tailwind, and DaisyUI. The app is a desktop front end for VoxCPM text-to-speech generation with model discovery, single-model lifecycle management, sequential generation, browser/WebAudio playback, Chinese and English UI, and a readable history log written next to the executable on exit.

The desktop app title is `焓言焓语` in Chinese and `AhanSays` in English. The app supports Chinese and English initially, detects the system language on first launch, and allows the user to override the language in settings.

## Existing Context

The current repository contains Rust workspace crates for `voxui-gguf`, `voxui-inference`, `voxui-audio`, and `voxui-cli`. The previous desktop crate has been removed from the workspace, but `README.txt` still documents old desktop build commands. The CLI already demonstrates model loading, optional LoRA loading, cancellable generation, and streaming generation. The desktop app should reuse `voxui-inference` for engine behavior rather than duplicating inference logic.

## Recommended Approach

Use a single end-to-end Tauri/Svelte desktop app.

Rust owns model discovery, model load/unload, inference queue, cancellation, settings persistence, readable exit log, and event emission. Svelte owns the visual shell, localized UI, settings modal, generation history, WebAudio playback, test tone, volume, output device selection when supported, and playback fallback behavior.

Generated audio is not played through `voxui-audio` in this design. Rust sends mono PCM `f32` audio chunks and sample-rate metadata to the frontend. The Svelte/WebAudio layer plays both streaming chunks and completed batch audio, handles the sine-wave test, applies volume, and routes audio to the selected browser output device when the WebView supports output-device APIs. If output-device selection is unsupported or permission is denied, playback falls back to the default output device.

## Architecture

`voxui-desktop` will be a new workspace crate at `voxui/crates/voxui-desktop` with a Tauri Rust backend and a Svelte/Tailwind/DaisyUI frontend.

Backend responsibilities:

- Own exactly one `VoxCPMEngine` instance behind desktop app state.
- Discover model folders under the configured model directory.
- Expand each model folder into dropdown entries: one base entry plus one entry per `*.gguf` LoRA file except `model.gguf`.
- Cancel any in-flight model load when a new load starts.
- Unload the previous model before loading the selected model.
- Run a single sequential generation queue with at most one active inference.
- Emit model-load progress, generation progress, audio chunks, completion, cancellation, and errors as Tauri events.
- Persist settings.
- Write a readable generation history log next to the executable on normal app exit.

Frontend responsibilities:

- Render the fixed top navbar, scrollable generation history, fixed bottom input bar, and tabbed settings modal.
- Detect system language on first launch through backend-provided locale information and then allow explicit Chinese or English override.
- Manage WebAudio playback, including test sine wave, fade-in/fade-out, volume, output-device selection when supported, and default-device fallback.
- Ensure only one history item is playing at a time.
- Start playback immediately for streaming generations after enough audio is buffered.
- Auto-play non-streaming generations after the final audio buffer arrives.
- Keep UI controls synchronized with backend state.

Event protocol boundary:

- Rust never plays generated audio.
- Rust sends PCM `f32` mono chunks plus sample-rate metadata.
- Svelte converts chunks into WebAudio playback buffers.
- Cancellation flows both ways: the UI sends cancel commands, Rust sets cancellation flags, and the frontend stops playback for the affected item.

## UI Design

The app shell has three fixed functional zones.

Top navbar:

- Left side shows the localized app title: `焓言焓语` for Chinese UI and `AhanSays` for English UI.
- Right side contains the model dropdown, `Load` button, and settings button.
- The navbar remains fixed at the top.
- Model dropdown entries show base and LoRA variants, for example `voxcpm2-fp16` and `voxcpm2-fp16 | lora_ft2`.
- The model dropdown is disabled when no models are detected.
- The `Load` button is disabled when no model is selected, when no models are detected, or while a load is active.

Middle generation history:

- Scrollable list of generation cards.
- Empty state explains that a model must be loaded before synthesis.
- Each card records text, model/LoRA, params snapshot, status, progress, duration, and errors if any.
- A streaming item shows an audio status area immediately, buffer/progress indication, and a `Cancel` button.
- A non-streaming item shows an audio area immediately but disables playback controls until audio is complete, then auto-plays.
- Completed items show replay and regenerate controls.
- Canceled and failed items show status and regenerate.

Bottom input bar:

- Fixed at the bottom.
- Multiline text box with max-character counter.
- `Push to generate` button.
- Disabled when no model is loaded, when input is empty, or when input exceeds the configured character limit.
- The generation queue is sequential: users may enqueue jobs while one is active, but only one inference and one playback are active at a time.

Settings modal:

- Uses left-side tabs.
- `General` contains model directory, browse/rescan, and language setting.
- `Inference` contains CPU/CUDA backend, VoxCPM generation parameters, max input characters, and streaming toggle.
- `Audio` contains browser output-device selection, permission/refresh button, sine-wave test with fade-in/fade-out, volume slider, and fallback warning if device selection is unsupported.
- `About` contains app title/version, engine/backend summary, attribution text `Coded by 久嘉 & OpenCode & Codex`, GPLv3 license notice, and a note that the project uses the VoxCPM Python implementation as reference/upstream.
- The readable history log path is not shown in the UI.

## Model Discovery

The default model directory is the parent directory of the executable. During development, the user can configure `D:\Sandbox_Share\VoxUI\models`.

A valid model folder contains `model.gguf`. Additional `*.gguf` files inside the same folder are treated as LoRA adapters, excluding `model.gguf` itself. Each valid model folder produces one base dropdown entry and one additional dropdown entry per LoRA file. For example, if `voxcpm2-fp16` contains `model.gguf`, `lora_a1.gguf`, and `lora_a2.gguf`, then the dropdown entries are:

- `voxcpm2-fp16`
- `voxcpm2-fp16 | lora_a1`
- `voxcpm2-fp16 | lora_a2`

Invalid folders are ignored for dropdown purposes and may be surfaced in a compact discovery warning area.

## Model Lifecycle

Only one model can be loaded at a time.

When the user clicks `Load`:

- If a model load is already in progress, the backend cancels it.
- If a model is loaded, the backend unloads it.
- The backend loads the selected base model and optional LoRA.
- Loading failure leaves no model loaded.
- LoRA load failure fails the selected variant because the dropdown entry explicitly means base model plus that LoRA.
- CUDA selection failure reports a clear UI error and leaves no model loaded.

The loading progress modal supports cancellation. It first shows bytes-read progress when available from the loader. It then shows component progress from `VoxCPMEngine::load_with_progress`. If bytes-read progress is not available without invasive changes, component progress is implemented first and byte progress is added at the smallest loader boundary that already reads GGUF data.

Load cancellation is treated as a normal user outcome rather than as an error toast.

## Generation Queue

The generation queue is sequential and does not multitask. At every moment there is at most one active inference and one active playback.

When a user pushes text:

- The frontend creates a generation history card immediately.
- The frontend sends a generation job to the backend with the selected model snapshot and current generation parameters.
- The backend validates that a model is loaded, validates request parameters, and queues the job.
- Jobs run one at a time.

Streaming mode:

- Uses streaming generation and sends PCM chunks to the frontend as they are generated.
- Disables badcase retry to match current CLI behavior.
- The frontend shows an audio component immediately and starts playback after enough buffered audio exists.
- The card shows a `Cancel` button while active.
- When synthesis finishes, `Cancel` changes to `Regenerate`.

Batch mode:

- Uses non-streaming generation and sends generation progress during inference.
- The frontend shows an audio component immediately but does not allow playback until the final PCM buffer arrives.
- Playback starts automatically after completion.
- The card shows a `Cancel` button while active.
- When synthesis finishes, `Cancel` changes to `Regenerate`.

If a loaded model is replaced while jobs are queued, queued jobs tied to the previous model snapshot are canceled.

## WebAudio Design

The frontend owns all generated-audio playback and audio testing.

Capabilities:

- Sine-wave test uses `OscillatorNode` and `GainNode`.
- Fade-in and fade-out are implemented by scheduled gain ramps.
- Volume uses a shared gain stage for generated playback and test tone.
- Different model sample rates are accepted; WebAudio handles output-device resampling.
- Streaming playback uses a queued PCM playback service, implemented with either scheduled `AudioBufferSourceNode`s or an `AudioWorklet` if needed for smoother buffering.
- Batch playback converts the final PCM buffer into an `AudioBuffer` and plays it through the same output path.
- Output-device selection uses browser APIs such as `AudioContext.setSinkId`, `HTMLMediaElement.setSinkId`, or related audio-output selection APIs when available in the WebView.
- Unsupported output-device APIs fall back to default output and show an explanatory warning in the `Audio` tab.

The app should not depend on native `voxui-audio` for generated playback in the desktop GUI.

## Settings

Settings persisted by the desktop app include:

- Model directory.
- Language mode: system/default, English, or Chinese.
- Inference backend: CPU or CUDA.
- VoxCPM generation parameters exposed by the UI.
- Max input characters.
- Streaming enabled/disabled.
- Browser audio volume.
- Last selected browser output device identifier only when the browser API returns a stable non-empty identifier. If the API does not expose a stable identifier, the app persists no output-device selection and uses the default output device on next launch.

The first-run language is derived from the system locale. If the system language is Chinese, the UI starts in Chinese. Otherwise, it starts in English.

## History Log

The app writes a readable history log next to the executable on normal exit. The log filename uses `ahan-says-history-YYYYMMDD-HHMMSS.log`. The log is not managed through the UI and no log path is shown.

Each generation attempt records:

- Timestamp.
- Input text.
- Selected model and optional LoRA.
- Inference backend.
- Streaming flag.
- Generation parameters.
- Result status: completed, canceled, or failed.
- Timing information.
- Error message if failed.

The log does not store raw audio. If writing the log fails, the app reports it to backend logs and does not block shutdown.

## Error Handling

Model discovery:

- Missing model directory shows an empty catalog with a clear message and browse action.
- Valid model folders require `model.gguf`.
- Additional LoRA files are discovered case-insensitively by `.gguf` extension.

Model loading:

- New load cancels the old load.
- Previous model is unloaded before a replacement model is loaded.
- Cancellation is non-error UI state.
- Load, CUDA, and LoRA failures leave no model loaded and show actionable errors.

Generation:

- Generation is disabled until a model is loaded.
- Empty and over-limit input are blocked in the frontend and revalidated in Rust.
- Replacing the loaded model cancels queued jobs tied to the old model snapshot.
- Playback failure does not discard generated audio; the card remains completed and shows playback error controls when possible.

Shutdown:

- Normal app exit writes the readable history log next to the executable.
- Log-write failure does not block shutdown.

## Testing And Verification Scope

Testing focuses only on code directly added or changed for `voxui-desktop`. Existing `voxui-inference` correctness tests, golden parity tests, and full generation matrices are out of scope for this task.

Rust/Tauri desktop backend checks:

- Model discovery for base-only folders, LoRA expansion, ignored invalid folders, and case-insensitive `.gguf` handling.
- Desktop settings defaults and serialization.
- UI-to-request parameter mapping.
- Queue state transitions and cancellation logic with a fake or mock generation worker where practical.
- `cargo check` for the desktop crate and workspace integration.

Svelte frontend checks:

- TypeScript/Svelte checks.
- Pure logic coverage or equivalent structure for i18n selection, model dropdown labels, settings validation, history card state transitions, and WebAudio fallback detection.
- Frontend build to catch bundling, Tailwind, and DaisyUI integration issues.

Manual verification:

- Launch the Tauri desktop app in dev mode.
- Confirm model discovery and LoRA dropdown entries.
- Confirm load, cancel load, and replacement load behavior.
- Confirm streaming and batch UI flows with a small manual sample.
- Confirm WebAudio sine test, volume, and default-device fallback.
- Confirm no overlapping inference or playback.
- Confirm readable history log is written next to the executable on normal exit.

## Implementation Notes

The previous README desktop commands should be updated during implementation after the new crate structure is known.

The exact WebAudio streaming implementation should start with the smallest reliable service. If scheduled `AudioBufferSourceNode` playback is smooth enough, use it. If not, move only the streaming buffer internals to `AudioWorklet` while preserving the same frontend service API.

Byte-level load progress should be added only where the loader already has a natural byte-read boundary. The minimum acceptable first version is component progress from the existing `VoxCPMEngine::load_with_progress` callback.

Frontend code should prefer `@tauri-apps/api` imports for Tauri access. If implementation uses the global `window.__TAURI__` object instead, `withGlobalTauri` must be enabled in Tauri configuration.
