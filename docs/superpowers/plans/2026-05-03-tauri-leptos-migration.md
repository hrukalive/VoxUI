# Tauri + Leptos Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ratatui TUI with a Tauri v2 desktop app using Leptos CSR frontend, and fix 4 backend bugs.

**Architecture:** Tauri v2 backend exposes commands wrapping voxui-inference/voxui-audio. Leptos WASM frontend renders in WebView, invokes Tauri commands via wasm-bindgen JS interop. Events stream progress from backend to frontend.

**Tech Stack:** Tauri 2, Leptos 0.7 (CSR), Trunk, Tailwind CSS, candle, cpal

---

## Phase 1: Bug Fixes (Backend Crates)

### Task 1: Fix LoRA never applied during inference

**Files:**
- Modify: `crates/voxui-inference/src/engine.rs`
- Modify: `crates/voxui-inference/src/base_lm.rs`

The `LoraAdapter` is loaded but never called during the forward pass. The base_lm and residual_lm forward methods need to accept an optional LoRA and apply it to each linear projection.

- [ ] **Step 1: Add LoRA application to BaseLM forward pass**

In `crates/voxui-inference/src/base_lm.rs`, modify the `forward_embed` method. After each Q/K/V/O projection, apply LoRA if provided.

Add a method to BaseLM:
```rust
/// Run forward with optional LoRA applied to attention projections
pub fn forward_embed_with_lora(&mut self, input: &Tensor, lora: Option<&LoraAdapter>) -> Result<Tensor> {
    // ... same as forward_embed but after each linear:
    // let q = Self::linear(hidden, &layer.q_proj)?;
    // becomes:
    // let mut q = Self::linear(hidden, &layer.q_proj)?;
    // if let Some(lora) = lora {
    //     q = lora.apply(&format!("{}.layers.{i}.self_attn.q_proj", self.config.prefix), &q, hidden)?;
    // }
}
```

- [ ] **Step 2: Wire LoRA into engine.rs synthesize loop**

In `crates/voxui-inference/src/engine.rs`, in the `synthesize` method, pass `self.lora.as_ref()` to base_lm and residual_lm forward calls:
```rust
// Before:
let lm_out = self.base_lm.forward_step_embed(&curr_embed)?;
// After:
let lm_out = self.base_lm.forward_embed_with_lora(&curr_embed, self.lora.as_ref())?;
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p voxui-inference`

---

### Task 2: Fix no sound playback after synthesis

**Files:**
- Modify: `crates/voxui-inference/src/engine.rs` (synthesize return value)

The issue: the app receives PCM samples but never plays them. In the Tauri migration, audio playback will be handled by the Tauri backend command handler directly after synthesis. For now, fix the existing app too.

- [ ] **Step 1: Ensure synthesize returns valid PCM**

Verify `synthesize()` returns `Vec<f32>` correctly. Add a debug assertion:
```rust
// At end of synthesize(), before returning:
assert!(!samples.is_empty(), "synthesize produced empty audio");
log::info!("Synthesize produced {} samples ({:.1}s at {}Hz)", 
    samples.len(), samples.len() as f32 / self.config.sample_rate as f32, self.config.sample_rate);
```

- [ ] **Step 2: In the Tauri command (Task 7), play audio after synthesis**

This will be handled in the Tauri command handler — after `engine.synthesize()` returns samples, call `AudioPlayer::play()`.

---

### Task 3: Fix progress bar showing wrong total

**Files:**
- Modify: `crates/voxui-inference/src/engine.rs`

The progress callback uses `max_steps` (200) as total, but synthesis typically stops at 20-50 steps. Fix: estimate total from text length.

- [ ] **Step 1: Estimate total steps from text length**

```rust
// In synthesize(), before the loop:
// Rough estimate: ~1 step per 2 characters for Chinese text
let estimated_steps = (text.chars().count() / 2).max(5).min(max_steps);
```

- [ ] **Step 2: Use estimated_steps in progress callback**

```rust
// Change from:
progress(step, max_steps);
// To:
progress(step, estimated_steps);
```

- [ ] **Step 3: Send final 100% progress after loop**

```rust
// After the loop completes:
progress(patches.len(), patches.len());
```

---

### Task 4: Fix LoRA config save and reload

**Files:**
- Modify: `crates/voxui-inference/src/engine.rs` (ensure load_lora path resolution works)

The LoRA path needs to be stored relative to the model directory. When model is reloaded, LoRA should also be reloaded if configured.

- [ ] **Step 1: In engine.rs, make load_lora accept absolute or relative path**

```rust
pub fn load_lora(&mut self, lora_dir: &Path) -> Result<()> {
    if !lora_dir.exists() {
        anyhow::bail!("LoRA directory not found: {}", lora_dir.display());
    }
    let adapter = LoraAdapter::load_from_dir(lora_dir, &self.device)?;
    log::info!("Loaded LoRA from {:?} ({} layer pairs)", lora_dir, adapter.layers.len());
    self.lora = Some(adapter);
    Ok(())
}
```

---

## Phase 2: Tauri Project Setup

### Task 5: Create Tauri project structure

**Files:**
- Create: `crates/voxui-desktop/src-tauri/Cargo.toml`
- Create: `crates/voxui-desktop/src-tauri/tauri.conf.json`
- Create: `crates/voxui-desktop/src-tauri/src/main.rs`
- Create: `crates/voxui-desktop/src-tauri/src/lib.rs`
- Create: `crates/voxui-desktop/src-tauri/build.rs`
- Create: `crates/voxui-desktop/src-tauri/capabilities/default.json`
- Modify: `Cargo.toml` (workspace — add new member)

- [ ] **Step 1: Create Tauri backend Cargo.toml**

```toml
[package]
name = "voxui-desktop"
version = "0.1.0"
edition = "2021"

[dependencies]
tauri = { version = "2", features = ["tray-icon"] }
tauri-plugin-shell = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
log = "0.4"
env_logger = "0.11"
anyhow = "1"

voxui-inference = { path = "../../voxui-inference" }
voxui-audio = { path = "../../voxui-audio" }

[features]
default = ["custom-protocol"]
custom-protocol = ["tauri/custom-protocol"]
cuda = ["voxui-inference/cuda"]

[build-dependencies]
tauri-build = { version = "2", features = [] }
```

- [ ] **Step 2: Create tauri.conf.json**

```json
{
  "productName": "VoxUI",
  "version": "0.1.0",
  "identifier": "com.voxui.app",
  "build": {
    "frontendDist": "../dist",
    "devUrl": "http://localhost:8080",
    "beforeDevCommand": "trunk serve --port 8080",
    "beforeBuildCommand": "trunk build --release"
  },
  "app": {
    "windows": [
      {
        "title": "VoxUI",
        "width": 800,
        "height": 600,
        "resizable": true,
        "minWidth": 600,
        "minHeight": 400
      }
    ],
    "security": {
      "csp": null
    }
  }
}
```

- [ ] **Step 3: Create src/main.rs and src/lib.rs**

`src/main.rs`:
```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    voxui_desktop::run();
}
```

`src/lib.rs`:
```rust
mod commands;
mod state;

use state::AppState;

pub fn run() {
    env_logger::init();

    tauri::Builder::default()
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            commands::list_models,
            commands::list_lora_dirs,
            commands::list_audio_devices,
            commands::load_model,
            commands::load_lora,
            commands::unload_lora,
            commands::synthesize,
            commands::get_config,
            commands::save_config,
        ])
        .run(tauri::generate_context!())
        .expect("error while running VoxUI");
}
```

- [ ] **Step 4: Create build.rs**

```rust
fn main() {
    tauri_build::build();
}
```

- [ ] **Step 5: Create capabilities/default.json**

```json
{
  "identifier": "default",
  "description": "Default capabilities for VoxUI",
  "windows": ["main"],
  "permissions": ["core:default", "shell:allow-open"]
}
```

- [ ] **Step 6: Update workspace Cargo.toml**

Add `"crates/voxui-desktop/src-tauri"` to workspace members.

- [ ] **Step 7: Verify compilation**

Run: `cargo check -p voxui-desktop`

---

### Task 6: Create Tauri backend state

**Files:**
- Create: `crates/voxui-desktop/src-tauri/src/state.rs`

- [ ] **Step 1: Implement AppState**

```rust
use std::sync::Mutex;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use voxui_inference::VoxCPMEngine;
use voxui_audio::{AudioSystem, AudioPlayer};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub model_dir: String,
    pub lora_dir: Option<String>,
    pub backend: String,
    pub audio_host: String,
    pub audio_device: String,
    pub max_chars: usize,
    pub dit_steps: usize,
    pub language: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model_dir: "models".into(),
            lora_dir: None,
            backend: "CUDA".into(),
            audio_host: String::new(),
            audio_device: String::new(),
            max_chars: 80,
            dit_steps: 10,
            language: "Chinese".into(),
        }
    }
}

impl AppConfig {
    pub fn load() -> Self {
        let path = PathBuf::from("voxui_config.json");
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write("voxui_config.json", json)?;
        Ok(())
    }
}

pub struct AppState {
    pub engine: Mutex<Option<VoxCPMEngine>>,
    pub audio_system: AudioSystem,
    pub config: Mutex<AppConfig>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            engine: Mutex::new(None),
            audio_system: AudioSystem::new(),
            config: Mutex::new(AppConfig::load()),
        }
    }
}
```

---

### Task 7: Create Tauri commands

**Files:**
- Create: `crates/voxui-desktop/src-tauri/src/commands.rs`

- [ ] **Step 1: Implement all Tauri commands**

```rust
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, State};
use serde::Serialize;
use crate::state::{AppConfig, AppState};
use voxui_inference::VoxCPMEngine;
use voxui_audio::AudioPlayer;

#[derive(Serialize)]
pub struct ModelInfo {
    pub architecture: String,
    pub sample_rate: u32,
}

#[derive(Serialize)]
pub struct AudioDeviceList {
    pub hosts: Vec<String>,
    pub devices: Vec<String>,
}

#[derive(Clone, Serialize)]
struct ProgressPayload {
    step: u32,
    total: u32,
    index: u32,
}

#[tauri::command]
pub fn list_models() -> Vec<String> {
    // Scan models/ directory for subdirs containing base_lm.gguf
    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir("models") {
        for entry in entries.flatten() {
            if entry.path().join("base_lm.gguf").exists() {
                if let Some(name) = entry.path().to_str() {
                    models.push(name.replace('\\', "/"));
                }
            }
        }
    }
    models.sort();
    models
}

#[tauri::command]
pub fn list_lora_dirs(model_dir: String) -> Vec<String> {
    let mut dirs = vec!["None".to_string()];
    let path = PathBuf::from(&model_dir);
    if let Ok(entries) = std::fs::read_dir(&path) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("lora_") && entry.path().is_dir() {
                if entry.path().join("lora_base_lm.gguf").exists() {
                    dirs.push(name);
                }
            }
        }
    }
    dirs
}

#[tauri::command]
pub fn list_audio_devices(state: State<AppState>) -> AudioDeviceList {
    let hosts: Vec<String> = state.audio_system.hosts().iter().map(|h| h.name.clone()).collect();
    let default_host = state.audio_system.default_host_name();
    let devices = state.audio_system.devices(&default_host)
        .map(|devs| devs.into_iter().map(|d| d.name).collect())
        .unwrap_or_default();
    AudioDeviceList { hosts, devices }
}

#[tauri::command]
pub async fn load_model(app: AppHandle, state: State<'_, AppState>, model_dir: String, backend: String) -> Result<ModelInfo, String> {
    let model_path = PathBuf::from(&model_dir);
    let device = select_device(&backend);
    
    let engine = tokio::task::spawn_blocking(move || {
        VoxCPMEngine::load(&model_path, &model_path, device)
    }).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
    
    let info = ModelInfo {
        architecture: engine.architecture().to_string(),
        sample_rate: engine.sample_rate(),
    };
    
    *state.engine.lock().unwrap() = Some(engine);
    let _ = app.emit("engine-ready", ());
    Ok(info)
}

#[tauri::command]
pub fn load_lora(state: State<AppState>, lora_dir: String) -> Result<(), String> {
    let mut engine_guard = state.engine.lock().unwrap();
    let engine = engine_guard.as_mut().ok_or("Engine not loaded")?;
    let path = PathBuf::from(&lora_dir);
    engine.load_lora(&path).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn unload_lora(state: State<AppState>) -> Result<(), String> {
    let mut engine_guard = state.engine.lock().unwrap();
    let engine = engine_guard.as_mut().ok_or("Engine not loaded")?;
    engine.unload_lora();
    Ok(())
}

#[tauri::command]
pub async fn synthesize(app: AppHandle, state: State<'_, AppState>, text: String, dit_steps: u32, index: u32) -> Result<(), String> {
    // Extract what we need from state before spawning blocking task
    let config = state.config.lock().unwrap().clone();
    
    let mut engine_guard = state.engine.lock().unwrap();
    let engine = engine_guard.as_mut().ok_or("Engine not loaded")?.clone();
    // Note: engine needs to be Send. If not, use a channel-based approach.
    drop(engine_guard);
    
    // For now, simplified: run synthesis synchronously (will block)
    // In production, use a dedicated inference thread with channels
    let app_clone = app.clone();
    let samples = {
        let mut eng = state.engine.lock().unwrap();
        let engine = eng.as_mut().ok_or("Engine not loaded")?;
        engine.synthesize(&text, dit_steps as usize, |step, total| {
            let _ = app_clone.emit("tts-progress", ProgressPayload { step: step as u32, total: total as u32, index });
        }).map_err(|e| e.to_string())?
    };
    
    // Play audio
    let sample_rate = {
        let eng = state.engine.lock().unwrap();
        eng.as_ref().map(|e| e.sample_rate()).unwrap_or(48000)
    };
    let host = &config.audio_host;
    let device = &config.audio_device;
    let player = AudioPlayer::new(host, device, sample_rate).map_err(|e| e.to_string())?;
    player.play_blocking(samples).map_err(|e| e.to_string())?;
    
    let _ = app.emit("tts-complete", serde_json::json!({ "index": index }));
    Ok(())
}

#[tauri::command]
pub fn get_config(state: State<AppState>) -> AppConfig {
    state.config.lock().unwrap().clone()
}

#[tauri::command]
pub fn save_config(state: State<AppState>, config: AppConfig) -> Result<(), String> {
    config.save().map_err(|e| e.to_string())?;
    *state.config.lock().unwrap() = config;
    Ok(())
}

fn select_device(backend: &str) -> candle_core::Device {
    match backend {
        "CUDA" => {
            #[cfg(feature = "cuda")]
            {
                candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu)
            }
            #[cfg(not(feature = "cuda"))]
            candle_core::Device::Cpu
        }
        _ => candle_core::Device::Cpu,
    }
}
```

---

## Phase 3: Leptos Frontend

### Task 8: Create Leptos project structure

**Files:**
- Create: `crates/voxui-desktop/Cargo.toml` (frontend crate)
- Create: `crates/voxui-desktop/src/main.rs`
- Create: `crates/voxui-desktop/index.html`
- Create: `crates/voxui-desktop/Trunk.toml`
- Create: `crates/voxui-desktop/tailwind.config.js`
- Create: `crates/voxui-desktop/input.css`

- [ ] **Step 1: Create frontend Cargo.toml**

```toml
[package]
name = "voxui-frontend"
version = "0.1.0"
edition = "2021"

[dependencies]
leptos = { version = "0.7", features = ["csr"] }
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
serde-wasm-bindgen = "0.6"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["console"] }
chrono = { version = "0.4", features = ["wasmbind"] }

[lib]
crate-type = ["cdylib"]
```

- [ ] **Step 2: Create index.html**

```html
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoxUI</title>
    <link data-trunk rel="css" href="style.css">
</head>
<body class="bg-gray-900 text-white h-screen overflow-hidden">
</body>
</html>
```

- [ ] **Step 3: Create Trunk.toml**

```toml
[build]
target = "index.html"
dist = "dist"

[[hooks]]
stage = "pre_build"
command = "sh"
command_arguments = ["-c", "npx tailwindcss -i input.css -o style.css --minify"]
```

- [ ] **Step 4: Create tailwind.config.js and input.css**

`tailwind.config.js`:
```js
module.exports = {
  content: ["./src/**/*.rs", "./index.html"],
  theme: { extend: {} },
  plugins: [],
}
```

`input.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

---

### Task 9: Implement Leptos Tauri bridge

**Files:**
- Create: `crates/voxui-desktop/src/tauri_api.rs`

- [ ] **Step 1: Create Tauri invoke/listen wrappers**

```rust
use wasm_bindgen::prelude::*;
use serde::{de::DeserializeOwned, Serialize};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"])]
    async fn invoke(cmd: &str, args: JsValue) -> JsValue;

    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "event"])]
    async fn listen(event: &str, handler: &Closure<dyn FnMut(JsValue)>) -> JsValue;
}

pub async fn tauri_invoke<T: DeserializeOwned>(cmd: &str, args: impl Serialize) -> Result<T, String> {
    let args_js = serde_wasm_bindgen::to_value(&args).map_err(|e| e.to_string())?;
    let result = invoke(cmd, args_js).await;
    serde_wasm_bindgen::from_value(result).map_err(|e| e.to_string())
}
```

---

### Task 10: Implement main Leptos app and components

**Files:**
- Create: `crates/voxui-desktop/src/main.rs`
- Create: `crates/voxui-desktop/src/app.rs`
- Create: `crates/voxui-desktop/src/components/header.rs`
- Create: `crates/voxui-desktop/src/components/history.rs`
- Create: `crates/voxui-desktop/src/components/input.rs`
- Create: `crates/voxui-desktop/src/components/progress.rs`
- Create: `crates/voxui-desktop/src/components/status_bar.rs`
- Create: `crates/voxui-desktop/src/components/settings_modal.rs`
- Create: `crates/voxui-desktop/src/components/model_select_modal.rs`
- Create: `crates/voxui-desktop/src/components/mod.rs`
- Create: `crates/voxui-desktop/src/i18n.rs`

This is the largest task. Each component is a Leptos `#[component]` fn that uses reactive signals.

- [ ] **Step 1: Create main.rs (mount point)**
- [ ] **Step 2: Create i18n.rs (Chinese/English strings)**
- [ ] **Step 3: Create app.rs (root component with signals for state)**
- [ ] **Step 4: Create header component (title + settings gear icon)**
- [ ] **Step 5: Create history component (scrollable TTS entry list)**
- [ ] **Step 6: Create input component (text input + send button)**
- [ ] **Step 7: Create progress component (progress bar with percentage)**
- [ ] **Step 8: Create status_bar component (model/backend/audio info)**
- [ ] **Step 9: Create settings_modal component (dropdowns for all settings)**
- [ ] **Step 10: Create model_select_modal (path input when no model found)**

---

## Phase 4: Integration & Testing

### Task 11: Wire Tauri events to Leptos signals

Connect backend events (tts-progress, tts-complete, engine-ready) to Leptos reactive signals so the UI updates in real-time.

### Task 12: End-to-end testing

- Build with `cargo tauri build` (or `cargo tauri dev` for development)
- Test: model loading, text input, synthesis with progress, audio playback, LoRA loading, settings persistence

---

## Build Commands

```powershell
# Development
cd crates/voxui-desktop
cargo tauri dev

# Production (CPU)
cargo tauri build

# Production (CUDA)
cargo tauri build --features cuda

# CUDA env setup (Windows)
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "$env:CUDA_PATH\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
$env:CUDA_COMPUTE_CAP = "89"
$env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"
```

## Prerequisites

Install before starting:
```powershell
# Tauri CLI
cargo install tauri-cli

# Trunk (WASM bundler)
cargo install trunk

# wasm32 target
rustup target add wasm32-unknown-unknown

# Node.js (for Tailwind)
npm install -D tailwindcss
```
