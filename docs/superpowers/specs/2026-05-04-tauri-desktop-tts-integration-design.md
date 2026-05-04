# VoxUI Tauri Desktop TTS Integration

## Overview

VoxUI will become a Tauri desktop text-to-speech app backed directly by the native Rust/Candle VoxCPM inference engine. The app will load exported VoxCPM 0.5, 1.5, or 2.0 model bundles from `models`, optionally apply one LoRA adapter, synthesize from text, optionally use prompt/reference audio where the selected model supports it, and play the generated PCM through the selected system audio host and output device.

The ratatui TUI crate is no longer an active product target. Its code can be used as reference during migration, but the runnable app focus is `voxui-desktop`.

## Goals

- Make the Tauri desktop app the primary user-facing app.
- Connect Tauri commands to `voxui-inference::VoxCPMEngine::generate` using the native `SynthesisRequest` shape.
- Support CPU and CUDA backends, including the existing `cuda` feature wiring.
- Support VoxCPM 0.5, 1.5, and 2.0 exported bundle manifests.
- Support LoRA loading/unloading from model-local LoRA directories.
- Support optional prompt audio plus required prompt text.
- Support optional reference audio for VoxCPM 2 without requiring reference text.
- Play synthesized audio through the selected `voxui-audio` host/device.
- Keep generated WAV export out of the desktop app flow.
- Preserve configuration across launches.

## Non-Goals

- No TUI maintenance beyond removing it from active workspace builds.
- No Python bridge or Python-backed inference.
- No WAV export button or automatic output-file workflow.
- No streaming audio playback during generation; synthesis completes before playback begins.
- No text normalization flag until the Rust normalizer exists.

## Architecture

The desktop app keeps three boundaries:

- `voxui-inference`: owns model loading, LoRA application, request validation, and waveform generation.
- `voxui-audio`: owns output host/device enumeration and PCM playback.
- `voxui-desktop`: owns app state, config, Tauri commands/events, and the Leptos UI.

`voxui-desktop/src-tauri` will hold the loaded `VoxCPMEngine` inside application state. Commands mutate that state through a narrow API: load model, apply LoRA, synthesize, list available resources, and save/load config. Long-running model load and synthesis work runs on blocking tasks so the WebView remains responsive.

The Leptos frontend does not perform inference or audio playback. It sends typed command payloads to Tauri and listens for progress/completion/error events.

## Backend API

The Tauri backend exposes these command-level behaviors:

- `list_models() -> Vec<ModelEntry>`: scans `models` for directories containing `manifest.json`. Each entry includes a display name and load path.
- `list_lora_dirs(model_dir) -> Vec<LoraEntry>`: scans the selected model directory for valid LoRA adapter directories. Each entry includes a display name and load path, plus a `None` option.
- `list_audio_hosts() / list_audio_devices(host)`: provides selectable output hosts and devices.
- `load_model(model_dir, backend) -> ModelInfo`: loads a VoxCPM bundle on CPU or CUDA.
- `apply_lora(lora_dir: Option<String>)`: unloads when `None`, otherwise loads the chosen LoRA directory.
- `synthesize(SynthesisArgs) -> Result<()>`: builds a `SynthesisRequest`, runs generation, emits progress, plays the resulting PCM through the selected device, and emits completion.
- `get_config() -> AppConfig` and `save_config(AppConfig)`.

`SynthesisArgs` includes:

- `text: String`
- `dit_steps: usize`
- `prompt_wav_path: Option<String>`
- `prompt_text: Option<String>`
- `reference_wav_path: Option<String>`
- `index: u32`

The backend converts these fields to `SynthesisRequest`. It rejects invalid combinations using the inference crate's validator. In particular, prompt audio requires prompt text, and reference audio is only valid for VoxCPM 2.

## Frontend UX

The existing Leptos dark desktop layout stays, but becomes a complete TTS tool:

- Header with app title and settings button.
- Status/history list with generated, playing, completed, and error states.
- Progress bar driven by `tts-progress` events.
- Text input and generate button.
- Settings modal with:
  - model selector
  - backend selector
  - LoRA selector
  - audio host selector
  - audio output device selector
  - diffusion steps
  - max text characters
  - optional prompt WAV path
  - prompt text
  - optional reference WAV path
  - language selector

The app disables synthesis while no model is loaded, while a model is loading, or while a synthesis request is active. Errors are shown in the status/history area rather than only in the browser console.

## Config

`voxui_config.json` remains the persisted config file. The desktop config stores:

- model directory
- selected LoRA directory or none
- backend
- audio host and device
- prompt WAV path
- prompt text
- reference WAV path
- max text characters
- diffusion steps
- UI language

LoRA config stores either `None` or a path that can be resolved from the selected model directory. Applying settings reloads the model only when the model directory or backend changes. LoRA changes hot-swap on the currently loaded engine.

## Audio Playback

The synthesis command plays returned `Vec<f32>` PCM samples through `voxui_audio::AudioPlayer` using the selected host/device and the loaded model's sample rate. If no host or device is selected, the backend resolves the default host/device.

Playback is blocking inside the synthesis task so the frontend receives `tts-complete` only after audio playback finishes. Playback errors mark the history item as error and return a command error.

## TUI Deprecation

`voxui-app` is removed from the workspace members so regular workspace checks/builds focus on inference, audio, GGUF, and desktop. The source may remain temporarily as reference, but it is not part of the desktop app verification path.

## Error Handling

- Model load failure: emit `engine-error`, set desktop status to error, keep input disabled.
- CUDA unavailable or not compiled: fall back to CPU only if the backend selector clearly reports that fallback; otherwise show a clear load error.
- LoRA load failure: keep the base model loaded and show the adapter error.
- Invalid synthesis request: mark the history entry as error.
- Audio device failure: mark the history entry as error and keep the engine ready for later retries.

## Testing

Implementation will use test-first changes where practical:

- Backend unit tests for config serialization and request construction.
- Backend tests for LoRA path resolution and model/list scanning using temp directories.
- Existing inference tests continue covering VoxCPM 0.5, 1.5, 2.0, LoRA, CPU, CUDA, and reference audio behavior.
- Desktop compile checks cover command signatures and feature wiring.

Verification commands:

- `cargo check -p voxui-inference`
- `cargo check -p voxui-desktop`
- `cargo check -p voxui-desktop --features cuda` with the provided CUDA environment
- `cargo test -p voxui-desktop`
- `cargo test -p voxui-inference --test native_runtime_purity`

If frontend tooling is available:

- build the Leptos frontend with Trunk
- run the Tauri desktop check/build path

## Acceptance Criteria

- The Tauri app can load an exported VoxCPM 0.5, 1.5, or 2.0 bundle from `models`.
- The user can choose CPU or CUDA at model load time.
- The user can select, apply, and clear a model-local LoRA adapter.
- The user can synthesize text and hear audio through the selected output device.
- The user can provide prompt audio with prompt text.
- The user can provide reference audio for VoxCPM 2 without text.
- Errors are visible in the UI.
- `voxui-app` is no longer part of active workspace builds.
