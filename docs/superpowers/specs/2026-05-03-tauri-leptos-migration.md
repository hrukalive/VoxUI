# VoxUI Tauri + Leptos Migration

## Overview

Migrate VoxUI from ratatui TUI to a Tauri desktop app with Leptos CSR frontend. Fix 4 backend bugs during migration.

## Architecture

- **Frontend**: Leptos (CSR, compiled to WASM), runs in Tauri WebView
- **Backend**: Tauri Rust backend, reuses existing `voxui-inference`, `voxui-audio`, `voxui-gguf` crates
- **Communication**: Tauri commands (invoke) for request/response, Tauri events for streaming progress
- **Styling**: Tailwind CSS, modern/clean aesthetic, dark theme

## Project Structure

```
voxui/
├── Cargo.toml                    # workspace (add voxui-desktop members)
├── crates/
│   ├── voxui-gguf/              # unchanged
│   ├── voxui-inference/         # bug fixes only
│   ├── voxui-audio/            # unchanged
│   ├── voxui-app/              # DEPRECATED (keep for reference, remove from workspace)
│   └── voxui-desktop/          # NEW: Tauri + Leptos app
│       ├── src-tauri/
│       │   ├── Cargo.toml      # Tauri backend deps
│       │   ├── tauri.conf.json
│       │   ├── src/
│       │   │   ├── main.rs     # Tauri entry, setup
│       │   │   ├── commands.rs # Tauri command handlers
│       │   │   └── state.rs    # AppState (engine, audio, config)
│       │   └── icons/
│       ├── src/                 # Leptos frontend (WASM)
│       │   ├── main.rs         # Mount Leptos app
│       │   ├── app.rs          # Root component
│       │   ├── components/
│       │   │   ├── header.rs
│       │   │   ├── history.rs
│       │   │   ├── input.rs
│       │   │   ├── progress.rs
│       │   │   ├── status_bar.rs
│       │   │   ├── settings_modal.rs
│       │   │   └── model_select_modal.rs
│       │   └── i18n.rs
│       ├── Cargo.toml          # Frontend crate (wasm32 target)
│       ├── index.html
│       ├── Trunk.toml          # Trunk build config
│       └── tailwind.config.js
```

## Tauri Commands (Backend API)

```rust
#[tauri::command]
async fn list_models(state: State<AppState>) -> Result<Vec<String>, String>;

#[tauri::command]
async fn list_lora_dirs(state: State<AppState>, model_dir: String) -> Result<Vec<String>, String>;

#[tauri::command]
async fn list_audio_devices(state: State<AppState>) -> Result<AudioDeviceList, String>;

#[tauri::command]
async fn load_model(state: State<AppState>, model_dir: String, backend: String) -> Result<ModelInfo, String>;

#[tauri::command]
async fn load_lora(state: State<AppState>, lora_dir: String) -> Result<(), String>;

#[tauri::command]
async fn unload_lora(state: State<AppState>) -> Result<(), String>;

#[tauri::command]
async fn synthesize(window: Window, state: State<AppState>, text: String, dit_steps: u32) -> Result<(), String>;
// Progress sent via window.emit("tts-progress", { step, total })
// Completion sent via window.emit("tts-complete", { index })
// Audio playback handled server-side (not in WebView)

#[tauri::command]
async fn stop_synthesis(state: State<AppState>) -> Result<(), String>;

#[tauri::command]
async fn get_config(state: State<AppState>) -> Result<AppConfig, String>;

#[tauri::command]
async fn save_config(state: State<AppState>, config: AppConfig) -> Result<(), String>;
```

## Tauri Events (Backend → Frontend)

```
"tts-progress" → { step: u32, total: u32, index: u32 }
"tts-complete" → { index: u32 }
"tts-error" → { index: u32, message: String }
"engine-ready" → {}
"engine-error" → { message: String }
```

## Bug Fixes (in backend crates)

### Bug 1: LoRA never applied during inference
**Location**: `voxui-inference/src/engine.rs` synthesize loop
**Fix**: After each linear projection in the forward pass (q/k/v/o_proj for base_lm and residual_lm), call `self.lora.apply(layer_name, output, input)` if LoRA is loaded.

### Bug 2: No sound playback
**Location**: Tauri `synthesize` command handler
**Fix**: After synthesis returns PCM samples, play them via `AudioPlayer::play()`. The Tauri backend manages audio (not the WebView).

### Bug 3: LoRA not saved properly
**Location**: Config save logic
**Fix**: Ensure LoRA path is correctly saved relative to model dir. After model reload, also send LoadLora command.

### Bug 4: Wrong progress bar
**Location**: `voxui-inference/src/engine.rs` progress callback
**Fix**: Use estimated total (based on text length) instead of hard max_steps=200. After synthesis completes, send a final 100% event.

## Frontend Layout

```
┌────────────────────────────────────────────────────────────┐
│ VoxUI                                    [⚙] Settings      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  [12:03:01] 大家好，欢迎来到直播间！                  ✓  │
│  [12:03:15] 感谢xxx的关注                            ✓  │
│  [12:03:28] 正在生成...                              ⏳  │
│                                                            │
├────────────────────────────────────────────────────────────┤
│  ████████████░░░░░░░░░ 60% 生成中...                      │
├────────────────────────────────────────────────────────────┤
│  [输入文字...]                              [发送]         │
├────────────────────────────────────────────────────────────┤
│  VoxCPM2 (Q4) | CUDA | WASAPI / Speakers | LoRA: Akit     │
└────────────────────────────────────────────────────────────┘
```

## i18n

Same Chinese/English strings as current, carried over to Leptos components. Default: Chinese.

## Config

Same `voxui_config.json` format, managed by Tauri backend (not WebView filesystem).

## Dependencies

### Tauri Backend (src-tauri/Cargo.toml)
- `tauri = "2"`
- `serde`, `serde_json`
- `tokio`
- `voxui-inference` (path dep)
- `voxui-audio` (path dep)

### Leptos Frontend (Cargo.toml)
- `leptos = "0.7"`
- `wasm-bindgen`
- `serde`, `serde_json`, `serde-wasm-bindgen`
- `js-sys`, `web-sys`

### Build
- `trunk` for WASM build
- `tailwindcss` for styles
- `tauri-cli` for packaging
