# AhanSays Tauri/Leptos GUI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fresh Tauri 2 + Leptos CSR desktop GUI for `焓言焓语` / `AhanSays` with model discovery/loading, sequential TTS generation, playback, settings, and bilingual UI.

**Architecture:** Create a new `voxui/crates/voxui-desktop` crate from scratch. The Tauri backend owns config, model discovery, one loaded engine slot, cancellation, sequential generation, audio cache, and playback; the Leptos frontend owns visible state and communicates through typed commands/events.

**Tech Stack:** Rust 2021, Tauri 2, Leptos CSR, Trunk, Serde, Tokio, `voxui-inference`, `voxui-audio`, `candle-core`, CPAL/r8brain through the audio crate.

---

## File Structure

Create:

- `voxui/crates/voxui-desktop/Cargo.toml`: frontend WASM crate manifest.
- `voxui/crates/voxui-desktop/Trunk.toml`: Leptos/Trunk build config.
- `voxui/crates/voxui-desktop/index.html`: frontend entry HTML.
- `voxui/crates/voxui-desktop/src/main.rs`: Leptos mount point.
- `voxui/crates/voxui-desktop/src/app.rs`: root app component and top-level frontend state.
- `voxui/crates/voxui-desktop/src/i18n.rs`: bilingual labels and system/manual language mapping.
- `voxui/crates/voxui-desktop/src/tauri_api.rs`: typed JS/Tauri invoke/event wrappers.
- `voxui/crates/voxui-desktop/src/components/mod.rs`: component module exports.
- `voxui/crates/voxui-desktop/src/components/header.rs`: title, model dropdown, load, settings button.
- `voxui/crates/voxui-desktop/src/components/history.rs`: generation history list and item controls.
- `voxui/crates/voxui-desktop/src/components/input_box.rs`: composer and character counter.
- `voxui/crates/voxui-desktop/src/components/settings_modal.rs`: settings sections.
- `voxui/crates/voxui-desktop/src/components/load_progress_modal.rs`: loading progress modal.
- `voxui/crates/voxui-desktop/src/styles.css`: application styles.
- `voxui/crates/voxui-desktop/src-tauri/Cargo.toml`: Tauri backend manifest.
- `voxui/crates/voxui-desktop/src-tauri/build.rs`: Tauri build script.
- `voxui/crates/voxui-desktop/src-tauri/tauri.conf.json`: app metadata and window config.
- `voxui/crates/voxui-desktop/src-tauri/capabilities/default.json`: Tauri command permissions.
- `voxui/crates/voxui-desktop/src-tauri/src/main.rs`: native entry point.
- `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`: Tauri builder setup.
- `voxui/crates/voxui-desktop/src-tauri/src/types.rs`: serializable backend/frontend DTOs.
- `voxui/crates/voxui-desktop/src-tauri/src/config.rs`: config defaults, persistence, language detection.
- `voxui/crates/voxui-desktop/src-tauri/src/model_discovery.rs`: model root scanning and stable ids.
- `voxui/crates/voxui-desktop/src-tauri/src/audio.rs`: device listing, sine generation, volume scaling.
- `voxui/crates/voxui-desktop/src-tauri/src/playback.rs`: generated audio cache and play/stop helpers.
- `voxui/crates/voxui-desktop/src-tauri/src/generation_queue.rs`: queue state and request snapshots.
- `voxui/crates/voxui-desktop/src-tauri/src/app_core.rs`: shared app state and orchestration.
- `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`: Tauri command handlers.
- `voxui/crates/voxui-desktop/src-tauri/tests/config_tests.rs`: config tests.
- `voxui/crates/voxui-desktop/src-tauri/tests/model_discovery_tests.rs`: discovery tests.
- `voxui/crates/voxui-desktop/src-tauri/tests/audio_tests.rs`: audio helper tests.
- `voxui/crates/voxui-desktop/src-tauri/tests/queue_tests.rs`: queue behavior tests.
- `voxui/crates/voxui-desktop/src-tauri/tests/app_core_tests.rs`: app-core state transition tests.

Modify:

- `voxui/Cargo.toml`: ensure `crates/voxui-desktop/src-tauri` is a workspace member.
- `.gitignore`: add `.superpowers/` if it is not already ignored.

Do not modify:

- Existing `voxui-inference`, `voxui-audio`, or `voxui-gguf` behavior.

---

### Task 1: Workspace And Desktop Skeleton

**Files:**
- Create: `voxui/crates/voxui-desktop/Cargo.toml`
- Create: `voxui/crates/voxui-desktop/Trunk.toml`
- Create: `voxui/crates/voxui-desktop/index.html`
- Create: `voxui/crates/voxui-desktop/src/main.rs`
- Create: `voxui/crates/voxui-desktop/src/app.rs`
- Create: `voxui/crates/voxui-desktop/src/styles.css`
- Create: `voxui/crates/voxui-desktop/src-tauri/Cargo.toml`
- Create: `voxui/crates/voxui-desktop/src-tauri/build.rs`
- Create: `voxui/crates/voxui-desktop/src-tauri/tauri.conf.json`
- Create: `voxui/crates/voxui-desktop/src-tauri/capabilities/default.json`
- Create: `voxui/crates/voxui-desktop/src-tauri/src/main.rs`
- Create: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`
- Modify: `voxui/Cargo.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Confirm no desktop files are being reused**

Run:

```powershell
Test-Path voxui\crates\voxui-desktop
git status --short
```

Expected: the path may exist as deleted files in git status, but implementation files should be created fresh from this plan. Do not restore deleted files with `git checkout`.

- [ ] **Step 2: Create frontend manifest**

Create `voxui/crates/voxui-desktop/Cargo.toml`:

```toml
[package]
name = "voxui-desktop-ui"
version.workspace = true
edition.workspace = true

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
leptos = { version = "0.7", features = ["csr"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
serde-wasm-bindgen = "0.6"
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
js-sys = "0.3"

[dependencies.web-sys]
version = "0.3"
features = [
  "Event",
  "HtmlInputElement",
  "HtmlSelectElement",
  "HtmlTextAreaElement",
  "Navigator",
  "Window",
]
```

- [ ] **Step 3: Create Trunk config**

Create `voxui/crates/voxui-desktop/Trunk.toml`:

```toml
[build]
target = "index.html"
dist = "dist"
public_url = "/"

[watch]
watch = ["src", "index.html"]
```

- [ ] **Step 4: Create frontend HTML**

Create `voxui/crates/voxui-desktop/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AhanSays</title>
    <link data-trunk rel="css" href="src/styles.css" />
  </head>
  <body>
    <main id="app"></main>
    <script type="module">
      import init from "./voxui_desktop_ui.js";
      init();
    </script>
  </body>
</html>
```

- [ ] **Step 5: Create a minimal Leptos app**

Create `voxui/crates/voxui-desktop/src/main.rs`:

```rust
mod app;

fn main() {
    leptos::mount_to_body(app::App);
}
```

Create `voxui/crates/voxui-desktop/src/app.rs`:

```rust
use leptos::prelude::*;

#[component]
pub fn App() -> impl IntoView {
    view! {
        <div class="app-shell">
            <header class="app-header">
                <div class="brand">
                    <strong>"焓言焓语"</strong>
                    <span>"AhanSays"</span>
                </div>
            </header>
            <section class="history-panel"></section>
            <footer class="composer-panel"></footer>
        </div>
    }
}
```

Create `voxui/crates/voxui-desktop/src/styles.css`:

```css
:root {
  color: #e7eaee;
  background: #101419;
  font-family: "Segoe UI", "Microsoft YaHei UI", system-ui, sans-serif;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-width: 760px;
  min-height: 520px;
  background: #101419;
}

.app-shell {
  min-height: 100vh;
  display: grid;
  grid-template-rows: 56px 1fr 116px;
}

.app-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 18px;
  border-bottom: 1px solid #2b3442;
  background: #151b23;
}

.brand {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.brand strong {
  font-size: 18px;
  letter-spacing: 0;
}

.brand span {
  color: #a9b3c1;
  font-size: 13px;
}
```

- [ ] **Step 6: Create Tauri backend manifest**

Create `voxui/crates/voxui-desktop/src-tauri/Cargo.toml`:

```toml
[package]
name = "voxui-desktop"
version.workspace = true
edition.workspace = true

[lib]
name = "voxui_desktop"
crate-type = ["staticlib", "cdylib", "rlib"]

[[bin]]
name = "voxui-desktop"
path = "src/main.rs"

[dependencies]
anyhow.workspace = true
thiserror.workspace = true
candle-core.workspace = true
voxui-inference = { path = "../../voxui-inference" }
voxui-audio = { path = "../../voxui-audio" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tauri = { version = "2", features = [] }
tauri-plugin-dialog = "2"
tauri-plugin-opener = "2"
tokio = { version = "1", features = ["rt-multi-thread", "sync", "time"] }
uuid = { version = "1", features = ["v4", "serde"] }
dirs = "6"
sys-locale = "0.3"
tracing = "0.1"
tracing-subscriber = "0.3"

[build-dependencies]
tauri-build = { version = "2", features = [] }

[features]
default = []
cuda = ["voxui-inference/cuda"]
```

- [ ] **Step 7: Create Tauri config and entry files**

Create `voxui/crates/voxui-desktop/src-tauri/build.rs`:

```rust
fn main() {
    tauri_build::build();
}
```

Create `voxui/crates/voxui-desktop/src-tauri/tauri.conf.json`:

```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "AhanSays",
  "version": "0.1.0",
  "identifier": "com.voxui.ahansays",
  "build": {
    "beforeDevCommand": "trunk serve",
    "beforeBuildCommand": "trunk build --release",
    "devUrl": "http://localhost:1420",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      {
        "title": "AhanSays",
        "width": 980,
        "height": 720,
        "minWidth": 760,
        "minHeight": 520
      }
    ],
    "security": {
      "csp": null
    }
  },
  "bundle": {
    "active": false,
    "targets": "all"
  }
}
```

Create `voxui/crates/voxui-desktop/src-tauri/capabilities/default.json`:

```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "default",
  "description": "Default AhanSays desktop permissions",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "dialog:default",
    "opener:default"
  ]
}
```

Create `voxui/crates/voxui-desktop/src-tauri/src/main.rs`:

```rust
fn main() {
    voxui_desktop::run();
}
```

Create `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`:

```rust
pub fn run() {
    tracing_subscriber::fmt::init();
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .run(tauri::generate_context!())
        .expect("failed to run AhanSays desktop app");
}
```

- [ ] **Step 8: Update workspace and ignore brainstorm artifacts**

Modify `voxui/Cargo.toml` workspace members so it contains:

```toml
members = [
    "crates/voxui-gguf",
    "crates/voxui-inference",
    "crates/voxui-audio",
    "crates/voxui-desktop/src-tauri",
]
```

Append to `.gitignore` if missing:

```gitignore
.superpowers/
```

- [ ] **Step 9: Verify skeleton builds**

Run:

```powershell
cd voxui
cargo check -p voxui-desktop
```

Expected: success or dependency-download/build errors only. Fix manifest path issues before moving on.

- [ ] **Step 10: Commit skeleton**

Run:

```powershell
git add .gitignore voxui/Cargo.toml voxui/crates/voxui-desktop
git commit -m "feat: scaffold AhanSays desktop app"
```

Expected: commit includes only new desktop skeleton, workspace update, and `.gitignore`.

---

### Task 2: Shared Types And Configuration

**Files:**
- Create: `voxui/crates/voxui-desktop/src-tauri/src/types.rs`
- Create: `voxui/crates/voxui-desktop/src-tauri/src/config.rs`
- Create: `voxui/crates/voxui-desktop/src-tauri/tests/config_tests.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`

- [ ] **Step 1: Write failing config tests**

Create `voxui/crates/voxui-desktop/src-tauri/tests/config_tests.rs`:

```rust
use std::path::PathBuf;

use voxui_desktop::config::{detect_language_from_locale, AppConfig, BackendKind, LanguageMode};

#[test]
fn config_defaults_to_system_language_and_cpu_backend() {
    let config = AppConfig::default();

    assert_eq!(config.language, LanguageMode::System);
    assert_eq!(config.backend, BackendKind::Cpu);
    assert_eq!(config.volume, 0.8);
    assert_eq!(config.max_input_chars, 280);
    assert_eq!(config.generation.inference_timesteps, 10);
    assert_eq!(config.generation.cfg_value, 2.0);
}

#[test]
fn detects_chinese_for_zh_locale() {
    assert_eq!(detect_language_from_locale(Some("zh-CN")), LanguageMode::Chinese);
    assert_eq!(detect_language_from_locale(Some("zh_TW")), LanguageMode::Chinese);
}

#[test]
fn detects_english_for_non_zh_or_missing_locale() {
    assert_eq!(detect_language_from_locale(Some("en-US")), LanguageMode::English);
    assert_eq!(detect_language_from_locale(Some("ja-JP")), LanguageMode::English);
    assert_eq!(detect_language_from_locale(None), LanguageMode::English);
}

#[test]
fn config_round_trips_as_json() {
    let config = AppConfig {
        model_root: Some(PathBuf::from("D:/Sandbox_Share/VoxUI/models")),
        selected_model_id: Some("voxcpm2-fp16|lora_a1.gguf".to_string()),
        language: LanguageMode::Chinese,
        backend: BackendKind::Cuda,
        audio_host: Some("Wasapi".to_string()),
        audio_device: Some("Speakers".to_string()),
        volume: 0.42,
        max_input_chars: 320,
        ..AppConfig::default()
    };

    let encoded = serde_json::to_string_pretty(&config).unwrap();
    let decoded: AppConfig = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded.model_root, config.model_root);
    assert_eq!(decoded.selected_model_id, config.selected_model_id);
    assert_eq!(decoded.language, LanguageMode::Chinese);
    assert_eq!(decoded.backend, BackendKind::Cuda);
    assert_eq!(decoded.volume, 0.42);
    assert_eq!(decoded.max_input_chars, 320);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop --test config_tests
```

Expected: FAIL because `config` module and types are not defined.

- [ ] **Step 3: Implement serializable shared types**

Create `voxui/crates/voxui-desktop/src-tauri/src/types.rs`:

```rust
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LanguageMode {
    System,
    Chinese,
    English,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationSettings {
    pub cfg_value: f32,
    pub inference_timesteps: usize,
    pub min_len: usize,
    pub max_len: usize,
    pub retry_badcase: bool,
    pub retry_badcase_max_times: usize,
    pub retry_badcase_ratio_threshold: f32,
    pub prompt_wav_path: Option<PathBuf>,
    pub prompt_text: Option<String>,
    pub reference_wav_path: Option<PathBuf>,
}

impl Default for GenerationSettings {
    fn default() -> Self {
        Self {
            cfg_value: 2.0,
            inference_timesteps: 10,
            min_len: 2,
            max_len: 2000,
            retry_badcase: true,
            retry_badcase_max_times: 3,
            retry_badcase_ratio_threshold: 6.0,
            prompt_wav_path: None,
            prompt_text: None,
            reference_wav_path: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    pub model_root: Option<PathBuf>,
    pub selected_model_id: Option<String>,
    pub language: LanguageMode,
    pub backend: BackendKind,
    pub audio_host: Option<String>,
    pub audio_device: Option<String>,
    pub volume: f32,
    pub max_input_chars: usize,
    pub generation: GenerationSettings,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model_root: None,
            selected_model_id: None,
            language: LanguageMode::System,
            backend: BackendKind::Cpu,
            audio_host: None,
            audio_device: None,
            volume: 0.8,
            max_input_chars: 280,
            generation: GenerationSettings::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelChoice {
    pub id: String,
    pub display_name: String,
    pub model_dir: PathBuf,
    pub model_path: PathBuf,
    pub lora_path: Option<PathBuf>,
    pub model_bytes: u64,
    pub lora_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioHostDto {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioDeviceDto {
    pub name: String,
    pub host_name: String,
}
```

- [ ] **Step 4: Implement config module**

Create `voxui/crates/voxui-desktop/src-tauri/src/config.rs`:

```rust
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub use crate::types::{AppConfig, BackendKind, GenerationSettings, LanguageMode};

pub fn detect_language_from_locale(locale: Option<&str>) -> LanguageMode {
    match locale {
        Some(value) if value.to_ascii_lowercase().starts_with("zh") => LanguageMode::Chinese,
        _ => LanguageMode::English,
    }
}

pub fn default_config_path(app_config_dir: &Path) -> PathBuf {
    app_config_dir.join("voxui_config.json")
}

pub fn load_config(path: &Path) -> Result<AppConfig> {
    if !path.exists() {
        return Ok(AppConfig::default());
    }

    let text = fs::read_to_string(path)
        .with_context(|| format!("read config from {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse config from {}", path.display()))
}

pub fn save_config(path: &Path, config: &AppConfig) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create config directory {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(config)?;
    fs::write(path, text).with_context(|| format!("write config to {}", path.display()))
}
```

- [ ] **Step 5: Export modules**

Modify `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`:

```rust
pub mod config;
pub mod types;

pub fn run() {
    tracing_subscriber::fmt::init();
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .run(tauri::generate_context!())
        .expect("failed to run AhanSays desktop app");
}
```

- [ ] **Step 6: Run tests**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop --test config_tests
```

Expected: PASS.

- [ ] **Step 7: Commit config and types**

Run:

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src/types.rs voxui/crates/voxui-desktop/src-tauri/src/config.rs voxui/crates/voxui-desktop/src-tauri/src/lib.rs voxui/crates/voxui-desktop/src-tauri/tests/config_tests.rs
git commit -m "feat: add desktop config types"
```

---

### Task 3: Model Discovery

**Files:**
- Create: `voxui/crates/voxui-desktop/src-tauri/src/model_discovery.rs`
- Create: `voxui/crates/voxui-desktop/src-tauri/tests/model_discovery_tests.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`

- [ ] **Step 1: Write failing model discovery tests**

Create `voxui/crates/voxui-desktop/src-tauri/tests/model_discovery_tests.rs`:

```rust
use std::fs;

use tempfile::TempDir;
use voxui_desktop::model_discovery::{choice_id, discover_models};

#[test]
fn discovers_base_and_lora_choices() {
    let temp = TempDir::new().unwrap();
    let model_dir = temp.path().join("voxcpm2-fp16");
    fs::create_dir(&model_dir).unwrap();
    fs::write(model_dir.join("model.gguf"), [0u8; 4]).unwrap();
    fs::write(model_dir.join("lora_a1.gguf"), [1u8; 2]).unwrap();
    fs::write(model_dir.join("lora_a2.gguf"), [2u8; 3]).unwrap();
    fs::write(model_dir.join("notes.txt"), b"ignored").unwrap();

    let choices = discover_models(temp.path()).unwrap();
    let names = choices.iter().map(|c| c.display_name.as_str()).collect::<Vec<_>>();

    assert_eq!(names, vec![
        "voxcpm2-fp16",
        "voxcpm2-fp16 | lora_a1",
        "voxcpm2-fp16 | lora_a2",
    ]);
    assert_eq!(choices[0].model_bytes, 4);
    assert_eq!(choices[1].lora_bytes, 2);
    assert_eq!(choices[2].lora_bytes, 3);
}

#[test]
fn ignores_directories_without_model_gguf() {
    let temp = TempDir::new().unwrap();
    let invalid = temp.path().join("not-a-model");
    fs::create_dir(&invalid).unwrap();
    fs::write(invalid.join("lora_a1.gguf"), [1u8; 2]).unwrap();

    let choices = discover_models(temp.path()).unwrap();

    assert!(choices.is_empty());
}

#[test]
fn choice_ids_are_relative_and_stable() {
    let temp = TempDir::new().unwrap();
    let model_dir = temp.path().join("voxcpm2-fp16");
    fs::create_dir(&model_dir).unwrap();

    assert_eq!(choice_id(temp.path(), &model_dir, None).unwrap(), "voxcpm2-fp16");
    assert_eq!(
        choice_id(temp.path(), &model_dir, Some(&model_dir.join("lora_a1.gguf"))).unwrap(),
        "voxcpm2-fp16|lora_a1.gguf"
    );
}
```

- [ ] **Step 2: Add tempfile dev dependency**

Modify `voxui/crates/voxui-desktop/src-tauri/Cargo.toml`:

```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop --test model_discovery_tests
```

Expected: FAIL because `model_discovery` is not defined.

- [ ] **Step 4: Implement model discovery**

Create `voxui/crates/voxui-desktop/src-tauri/src/model_discovery.rs`:

```rust
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::types::ModelChoice;

pub fn discover_models(root: &Path) -> Result<Vec<ModelChoice>> {
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut model_dirs = fs::read_dir(root)
        .with_context(|| format!("read model root {}", root.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    model_dirs.sort();

    let mut choices = Vec::new();
    for model_dir in model_dirs {
        let model_path = model_dir.join("model.gguf");
        if !model_path.is_file() {
            continue;
        }

        let model_name = model_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model")
            .to_string();
        let model_bytes = fs::metadata(&model_path)?.len();

        choices.push(ModelChoice {
            id: choice_id(root, &model_dir, None)?,
            display_name: model_name.clone(),
            model_dir: model_dir.clone(),
            model_path: model_path.clone(),
            lora_path: None,
            model_bytes,
            lora_bytes: 0,
        });

        let mut loras = fs::read_dir(&model_dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| is_lora_candidate(path))
            .collect::<Vec<_>>();
        loras.sort();

        for lora_path in loras {
            let lora_name = lora_path
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("lora")
                .to_string();
            let lora_bytes = fs::metadata(&lora_path)?.len();
            choices.push(ModelChoice {
                id: choice_id(root, &model_dir, Some(&lora_path))?,
                display_name: format!("{model_name} | {lora_name}"),
                model_dir: model_dir.clone(),
                model_path: model_path.clone(),
                lora_path: Some(lora_path),
                model_bytes,
                lora_bytes,
            });
        }
    }

    Ok(choices)
}

pub fn choice_id(root: &Path, model_dir: &Path, lora_path: Option<&Path>) -> Result<String> {
    let relative_model = model_dir
        .strip_prefix(root)
        .unwrap_or(model_dir)
        .to_string_lossy()
        .replace('\\', "/");

    if let Some(lora_path) = lora_path {
        let lora_file = lora_path
            .file_name()
            .and_then(|name| name.to_str())
            .context("LoRA path has no file name")?;
        Ok(format!("{relative_model}|{lora_file}"))
    } else {
        Ok(relative_model)
    }
}

fn is_lora_candidate(path: &PathBuf) -> bool {
    path.is_file()
        && path.file_name().and_then(|name| name.to_str()) != Some("model.gguf")
        && path.extension().and_then(|ext| ext.to_str()) == Some("gguf")
}
```

- [ ] **Step 5: Export module**

Modify `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`:

```rust
pub mod config;
pub mod model_discovery;
pub mod types;
```

Keep the existing `run()` function unchanged.

- [ ] **Step 6: Run discovery tests**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop --test model_discovery_tests
```

Expected: PASS.

- [ ] **Step 7: Commit model discovery**

Run:

```powershell
git add voxui/crates/voxui-desktop/src-tauri/Cargo.toml voxui/crates/voxui-desktop/src-tauri/src/lib.rs voxui/crates/voxui-desktop/src-tauri/src/model_discovery.rs voxui/crates/voxui-desktop/src-tauri/tests/model_discovery_tests.rs
git commit -m "feat: discover desktop model choices"
```

---

### Task 4: Audio Helpers And Playback Cache

**Files:**
- Create: `voxui/crates/voxui-desktop/src-tauri/src/audio.rs`
- Create: `voxui/crates/voxui-desktop/src-tauri/src/playback.rs`
- Create: `voxui/crates/voxui-desktop/src-tauri/tests/audio_tests.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`

- [ ] **Step 1: Write failing audio tests**

Create `voxui/crates/voxui-desktop/src-tauri/tests/audio_tests.rs`:

```rust
use voxui_desktop::audio::{apply_volume, sine_with_fades};
use voxui_desktop::playback::GeneratedAudioCache;

#[test]
fn volume_scales_samples_and_clamps_volume() {
    let samples = vec![-0.5, 0.25, 1.0];

    assert_eq!(apply_volume(&samples, 0.5), vec![-0.25, 0.125, 0.5]);
    assert_eq!(apply_volume(&samples, -1.0), vec![0.0, 0.0, 0.0]);
    assert_eq!(apply_volume(&samples, 2.0), vec![-0.5, 0.25, 1.0]);
}

#[test]
fn sine_wave_has_faded_edges() {
    let samples = sine_with_fades(1_000, 1_000, 100.0, 0.2);

    assert_eq!(samples.len(), 1_000);
    assert_eq!(samples[0], 0.0);
    assert!(samples[50].abs() < 0.2);
    assert!(samples[500].abs() <= 0.2);
    assert!(samples[999].abs() < 0.2);
}

#[test]
fn generated_audio_cache_preserves_previous_until_replaced() {
    let mut cache = GeneratedAudioCache::default();

    cache.insert("item-1".to_string(), vec![0.1, 0.2], 16_000);
    assert_eq!(cache.get("item-1").unwrap().samples, vec![0.1, 0.2]);

    cache.insert("item-1".to_string(), vec![0.3], 24_000);
    let stored = cache.get("item-1").unwrap();
    assert_eq!(stored.samples, vec![0.3]);
    assert_eq!(stored.sample_rate, 24_000);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop --test audio_tests
```

Expected: FAIL because `audio` and `playback` modules are not defined.

- [ ] **Step 3: Implement audio helpers**

Create `voxui/crates/voxui-desktop/src-tauri/src/audio.rs`:

```rust
use anyhow::Result;
use voxui_audio::{AudioSystem, DeviceInfo, HostInfo};

use crate::types::{AudioDeviceDto, AudioHostDto};

pub fn list_hosts(system: &AudioSystem) -> Vec<AudioHostDto> {
    system
        .hosts()
        .iter()
        .map(|host: &HostInfo| AudioHostDto {
            name: host.name.clone(),
        })
        .collect()
}

pub fn list_devices(system: &AudioSystem, host_name: &str) -> Result<Vec<AudioDeviceDto>> {
    Ok(system
        .devices(host_name)?
        .into_iter()
        .map(|device: DeviceInfo| AudioDeviceDto {
            name: device.name,
            host_name: device.host_name,
        })
        .collect())
}

pub fn apply_volume(samples: &[f32], volume: f32) -> Vec<f32> {
    let volume = volume.clamp(0.0, 1.0);
    samples.iter().map(|sample| sample * volume).collect()
}

pub fn sine_with_fades(sample_rate: u32, len_samples: usize, frequency_hz: f32, volume: f32) -> Vec<f32> {
    let volume = volume.clamp(0.0, 1.0);
    let fade_len = (sample_rate as usize / 20).min(len_samples / 2).max(1);

    (0..len_samples)
        .map(|idx| {
            let t = idx as f32 / sample_rate as f32;
            let fade_in = (idx as f32 / fade_len as f32).min(1.0);
            let fade_out = ((len_samples.saturating_sub(1 + idx)) as f32 / fade_len as f32).min(1.0);
            let envelope = fade_in.min(fade_out);
            (t * frequency_hz * std::f32::consts::TAU).sin() * volume * envelope
        })
        .collect()
}
```

- [ ] **Step 4: Implement playback cache**

Create `voxui/crates/voxui-desktop/src-tauri/src/playback.rs`:

```rust
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

#[derive(Debug, Default)]
pub struct GeneratedAudioCache {
    items: HashMap<String, GeneratedAudio>,
}

impl GeneratedAudioCache {
    pub fn insert(&mut self, item_id: String, samples: Vec<f32>, sample_rate: u32) {
        self.items.insert(item_id, GeneratedAudio { samples, sample_rate });
    }

    pub fn get(&self, item_id: &str) -> Option<&GeneratedAudio> {
        self.items.get(item_id)
    }

    pub fn remove(&mut self, item_id: &str) {
        self.items.remove(item_id);
    }
}
```

- [ ] **Step 5: Export modules**

Modify `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`:

```rust
pub mod audio;
pub mod config;
pub mod model_discovery;
pub mod playback;
pub mod types;
```

Keep the existing `run()` function unchanged.

- [ ] **Step 6: Run audio tests**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop --test audio_tests
```

Expected: PASS.

- [ ] **Step 7: Commit audio helpers**

Run:

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src/lib.rs voxui/crates/voxui-desktop/src-tauri/src/audio.rs voxui/crates/voxui-desktop/src-tauri/src/playback.rs voxui/crates/voxui-desktop/src-tauri/tests/audio_tests.rs
git commit -m "feat: add desktop audio helpers"
```

---

### Task 5: Queue State And Request Snapshots

**Files:**
- Create: `voxui/crates/voxui-desktop/src-tauri/src/generation_queue.rs`
- Create: `voxui/crates/voxui-desktop/src-tauri/tests/queue_tests.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/types.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`

- [ ] **Step 1: Write failing queue tests**

Create `voxui/crates/voxui-desktop/src-tauri/tests/queue_tests.rs`:

```rust
use voxui_desktop::config::{AppConfig, BackendKind};
use voxui_desktop::generation_queue::{GenerationQueue, HistoryStatus};

#[test]
fn enqueue_captures_settings_and_preserves_order() {
    let mut config = AppConfig::default();
    config.backend = BackendKind::Cuda;
    config.generation.cfg_value = 3.0;

    let mut queue = GenerationQueue::default();
    let first = queue.enqueue("first".to_string(), "voxcpm2-fp16".to_string(), &config);
    let second = queue.enqueue("second".to_string(), "voxcpm2-fp16".to_string(), &config);

    assert_eq!(queue.next_queued_id().unwrap(), first.id);
    assert_eq!(queue.items()[0].text, "first");
    assert_eq!(queue.items()[1].id, second.id);
    assert_eq!(queue.items()[0].snapshot.backend, BackendKind::Cuda);
    assert_eq!(queue.items()[0].snapshot.generation.cfg_value, 3.0);
}

#[test]
fn cancel_queued_item_marks_it_canceled() {
    let config = AppConfig::default();
    let mut queue = GenerationQueue::default();
    let item = queue.enqueue("text".to_string(), "model".to_string(), &config);

    assert!(queue.cancel_queued(&item.id));

    assert_eq!(queue.items()[0].status, HistoryStatus::Canceled);
    assert!(queue.next_queued_id().is_none());
}

#[test]
fn regeneration_attempt_keeps_existing_audio_flag_until_success() {
    let config = AppConfig::default();
    let mut queue = GenerationQueue::default();
    let item = queue.enqueue("text".to_string(), "model".to_string(), &config);
    queue.mark_ready(&item.id, true);

    queue.start_regeneration(&item.id, &config).unwrap();

    let updated = &queue.items()[0];
    assert_eq!(updated.status, HistoryStatus::Queued);
    assert!(updated.has_audio);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop --test queue_tests
```

Expected: FAIL because `generation_queue` is not defined.

- [ ] **Step 3: Add history DTO types**

Append to `voxui/crates/voxui-desktop/src-tauri/src/types.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestSnapshot {
    pub model_id: String,
    pub backend: BackendKind,
    pub generation: GenerationSettings,
}
```

- [ ] **Step 4: Implement generation queue**

Create `voxui/crates/voxui-desktop/src-tauri/src/generation_queue.rs`:

```rust
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::{AppConfig, RequestSnapshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryStatus {
    Queued,
    Generating,
    Canceled,
    Failed,
    Ready,
    Playing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryItem {
    pub id: String,
    pub text: String,
    pub status: HistoryStatus,
    pub progress_current: usize,
    pub progress_total: usize,
    pub error: Option<String>,
    pub has_audio: bool,
    pub snapshot: RequestSnapshot,
}

#[derive(Debug, Default)]
pub struct GenerationQueue {
    items: Vec<HistoryItem>,
}

impl GenerationQueue {
    pub fn enqueue(&mut self, text: String, model_id: String, config: &AppConfig) -> HistoryItem {
        let item = HistoryItem {
            id: Uuid::new_v4().to_string(),
            text,
            status: HistoryStatus::Queued,
            progress_current: 0,
            progress_total: 0,
            error: None,
            has_audio: false,
            snapshot: RequestSnapshot {
                model_id,
                backend: config.backend,
                generation: config.generation.clone(),
            },
        };
        self.items.push(item.clone());
        item
    }

    pub fn items(&self) -> &[HistoryItem] {
        &self.items
    }

    pub fn next_queued_id(&self) -> Option<String> {
        self.items
            .iter()
            .find(|item| item.status == HistoryStatus::Queued)
            .map(|item| item.id.clone())
    }

    pub fn cancel_queued(&mut self, item_id: &str) -> bool {
        if let Some(item) = self.items.iter_mut().find(|item| item.id == item_id) {
            if item.status == HistoryStatus::Queued {
                item.status = HistoryStatus::Canceled;
                item.error = None;
                return true;
            }
        }
        false
    }

    pub fn mark_generating(&mut self, item_id: &str) -> Result<()> {
        let item = self.item_mut(item_id)?;
        item.status = HistoryStatus::Generating;
        item.progress_current = 0;
        item.progress_total = 0;
        item.error = None;
        Ok(())
    }

    pub fn mark_progress(&mut self, item_id: &str, current: usize, total: usize) -> Result<()> {
        let item = self.item_mut(item_id)?;
        item.progress_current = current;
        item.progress_total = total;
        Ok(())
    }

    pub fn mark_ready(&mut self, item_id: &str, has_audio: bool) {
        if let Some(item) = self.items.iter_mut().find(|item| item.id == item_id) {
            item.status = HistoryStatus::Ready;
            item.has_audio = has_audio;
            item.error = None;
        }
    }

    pub fn mark_failed(&mut self, item_id: &str, message: String) {
        if let Some(item) = self.items.iter_mut().find(|item| item.id == item_id) {
            item.status = HistoryStatus::Failed;
            item.error = Some(message);
        }
    }

    pub fn mark_canceled(&mut self, item_id: &str) {
        if let Some(item) = self.items.iter_mut().find(|item| item.id == item_id) {
            item.status = HistoryStatus::Canceled;
            item.error = None;
        }
    }

    pub fn start_regeneration(&mut self, item_id: &str, config: &AppConfig) -> Result<()> {
        let item = self.item_mut(item_id)?;
        item.status = HistoryStatus::Queued;
        item.progress_current = 0;
        item.progress_total = 0;
        item.error = None;
        item.snapshot.backend = config.backend;
        item.snapshot.generation = config.generation.clone();
        Ok(())
    }

    fn item_mut(&mut self, item_id: &str) -> Result<&mut HistoryItem> {
        self.items
            .iter_mut()
            .find(|item| item.id == item_id)
            .ok_or_else(|| anyhow::anyhow!("unknown history item: {item_id}"))
    }
}
```

- [ ] **Step 5: Export module**

Modify `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`:

```rust
pub mod audio;
pub mod config;
pub mod generation_queue;
pub mod model_discovery;
pub mod playback;
pub mod types;
```

- [ ] **Step 6: Run queue tests**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop --test queue_tests
```

Expected: PASS.

- [ ] **Step 7: Commit queue state**

Run:

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src/types.rs voxui/crates/voxui-desktop/src-tauri/src/lib.rs voxui/crates/voxui-desktop/src-tauri/src/generation_queue.rs voxui/crates/voxui-desktop/src-tauri/tests/queue_tests.rs
git commit -m "feat: add desktop generation queue state"
```

---

### Task 6: App Core State Transitions

**Files:**
- Create: `voxui/crates/voxui-desktop/src-tauri/src/app_core.rs`
- Create: `voxui/crates/voxui-desktop/src-tauri/tests/app_core_tests.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/types.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`

- [ ] **Step 1: Write failing app-core tests**

Create `voxui/crates/voxui-desktop/src-tauri/tests/app_core_tests.rs`:

```rust
use std::fs;

use tempfile::TempDir;
use voxui_desktop::app_core::{load_button_enabled, AppCore};
use voxui_desktop::generation_queue::HistoryStatus;
use voxui_desktop::types::{AppConfig, LoadUiState};

#[test]
fn load_button_requires_selection_and_difference_from_loaded() {
    assert!(!load_button_enabled(None, None, LoadUiState::Idle, false));
    assert!(load_button_enabled(Some("a"), None, LoadUiState::Idle, false));
    assert!(!load_button_enabled(Some("a"), Some("a"), LoadUiState::Idle, false));
    assert!(!load_button_enabled(Some("b"), Some("a"), LoadUiState::Loading, false));
    assert!(!load_button_enabled(Some("b"), Some("a"), LoadUiState::Idle, true));
}

#[test]
fn startup_discovers_models_and_restores_selection() {
    let temp = TempDir::new().unwrap();
    let model_dir = temp.path().join("voxcpm2-fp16");
    fs::create_dir(&model_dir).unwrap();
    fs::write(model_dir.join("model.gguf"), [0u8; 4]).unwrap();

    let config = AppConfig {
        model_root: Some(temp.path().to_path_buf()),
        selected_model_id: Some("voxcpm2-fp16".to_string()),
        ..AppConfig::default()
    };

    let core = AppCore::from_config(config).unwrap();
    let snapshot = core.snapshot();

    assert_eq!(snapshot.models.len(), 1);
    assert_eq!(snapshot.selected_model_id.as_deref(), Some("voxcpm2-fp16"));
    assert_eq!(snapshot.loaded_model_id, None);
}

#[test]
fn enqueue_generation_rejects_when_no_model_is_loaded() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();

    let error = core.enqueue_generation("hello".to_string()).unwrap_err();

    assert!(error.to_string().contains("no model loaded"));
}

#[test]
fn enqueue_generation_creates_queued_item_when_loaded() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());

    let item = core.enqueue_generation("hello".to_string()).unwrap();

    assert_eq!(item.text, "hello");
    assert_eq!(item.status, HistoryStatus::Queued);
    assert_eq!(core.snapshot().history.len(), 1);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop --test app_core_tests
```

Expected: FAIL because `app_core` and `LoadUiState` are not defined.

- [ ] **Step 3: Add app snapshot and load state types**

Append to `voxui/crates/voxui-desktop/src-tauri/src/types.rs`:

```rust
use crate::generation_queue::HistoryItem;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadUiState {
    Idle,
    Loading,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppSnapshot {
    pub config: AppConfig,
    pub models: Vec<ModelChoice>,
    pub selected_model_id: Option<String>,
    pub loaded_model_id: Option<String>,
    pub load_state: LoadUiState,
    pub history: Vec<HistoryItem>,
}
```

- [ ] **Step 4: Implement app core state**

Create `voxui/crates/voxui-desktop/src-tauri/src/app_core.rs`:

```rust
use anyhow::{bail, Result};

use crate::generation_queue::{GenerationQueue, HistoryItem};
use crate::model_discovery::discover_models;
use crate::types::{AppConfig, AppSnapshot, LoadUiState, ModelChoice};

pub struct AppCore {
    config: AppConfig,
    models: Vec<ModelChoice>,
    selected_model_id: Option<String>,
    loaded_model_id: Option<String>,
    load_state: LoadUiState,
    queue: GenerationQueue,
}

impl AppCore {
    pub fn from_config(config: AppConfig) -> Result<Self> {
        let models = if let Some(root) = config.model_root.as_ref() {
            discover_models(root)?
        } else {
            Vec::new()
        };
        let selected_model_id = select_existing_model(config.selected_model_id.clone(), &models);

        Ok(Self {
            config,
            models,
            selected_model_id,
            loaded_model_id: None,
            load_state: LoadUiState::Idle,
            queue: GenerationQueue::default(),
        })
    }

    pub fn snapshot(&self) -> AppSnapshot {
        AppSnapshot {
            config: self.config.clone(),
            models: self.models.clone(),
            selected_model_id: self.selected_model_id.clone(),
            loaded_model_id: self.loaded_model_id.clone(),
            load_state: self.load_state,
            history: self.queue.items().to_vec(),
        }
    }

    pub fn enqueue_generation(&mut self, text: String) -> Result<HistoryItem> {
        let text = text.trim().to_string();
        if text.is_empty() {
            bail!("text must not be empty");
        }
        if text.chars().count() > self.config.max_input_chars {
            bail!("text exceeds max input characters");
        }
        let model_id = self
            .loaded_model_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("no model loaded"))?;

        Ok(self.queue.enqueue(text, model_id, &self.config))
    }

    pub fn set_loaded_model_for_test(&mut self, model_id: String) {
        self.loaded_model_id = Some(model_id);
    }
}

pub fn load_button_enabled(
    selected_model_id: Option<&str>,
    loaded_model_id: Option<&str>,
    load_state: LoadUiState,
    generation_running: bool,
) -> bool {
    selected_model_id.is_some()
        && load_state == LoadUiState::Idle
        && !generation_running
        && selected_model_id != loaded_model_id
}

fn select_existing_model(saved: Option<String>, models: &[ModelChoice]) -> Option<String> {
    if let Some(saved) = saved {
        if models.iter().any(|model| model.id == saved) {
            return Some(saved);
        }
    }
    models.first().map(|model| model.id.clone())
}
```

- [ ] **Step 5: Export module**

Modify `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`:

```rust
pub mod app_core;
pub mod audio;
pub mod config;
pub mod generation_queue;
pub mod model_discovery;
pub mod playback;
pub mod types;
```

- [ ] **Step 6: Run app-core tests**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop --test app_core_tests
```

Expected: PASS.

- [ ] **Step 7: Commit app-core state**

Run:

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src/types.rs voxui/crates/voxui-desktop/src-tauri/src/lib.rs voxui/crates/voxui-desktop/src-tauri/src/app_core.rs voxui/crates/voxui-desktop/src-tauri/tests/app_core_tests.rs
git commit -m "feat: add desktop app core state"
```

---

### Task 7: Tauri Commands And Event Payloads

**Files:**
- Create: `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/types.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/app_core.rs`

- [ ] **Step 1: Add command result and event payload types**

Append to `voxui/crates/voxui-desktop/src-tauri/src/types.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommandResult {
    pub ok: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigPatch {
    pub model_root: Option<Option<PathBuf>>,
    pub selected_model_id: Option<Option<String>>,
    pub language: Option<LanguageMode>,
    pub backend: Option<BackendKind>,
    pub audio_host: Option<Option<String>>,
    pub audio_device: Option<Option<String>>,
    pub volume: Option<f32>,
    pub max_input_chars: Option<usize>,
    pub generation: Option<GenerationSettings>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelLoadProgressEvent {
    pub phase: String,
    pub loaded_bytes: u64,
    pub total_bytes: u64,
    pub component: Option<String>,
    pub component_index: usize,
    pub component_total: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelLoadDoneEvent {
    pub status: String,
    pub selected_model_id: Option<String>,
    pub loaded_model_id: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationProgressEvent {
    pub item_id: String,
    pub current: usize,
    pub total: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationDoneEvent {
    pub item_id: String,
    pub status: String,
    pub error: Option<String>,
    pub sample_rate: Option<u32>,
    pub duration_seconds: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlaybackStateEvent {
    pub item_id: Option<String>,
    pub state: String,
}
```

- [ ] **Step 2: Extend app core with patch and discovery methods**

Append methods inside `impl AppCore` in `voxui/crates/voxui-desktop/src-tauri/src/app_core.rs`:

```rust
    pub fn apply_patch(&mut self, patch: crate::types::ConfigPatch) -> Result<AppSnapshot> {
        if let Some(model_root) = patch.model_root {
            self.config.model_root = model_root;
            self.rescan_models()?;
        }
        if let Some(selected) = patch.selected_model_id {
            self.selected_model_id = selected.clone();
            self.config.selected_model_id = selected;
        }
        if let Some(language) = patch.language {
            self.config.language = language;
        }
        if let Some(backend) = patch.backend {
            self.config.backend = backend;
        }
        if let Some(audio_host) = patch.audio_host {
            self.config.audio_host = audio_host;
        }
        if let Some(audio_device) = patch.audio_device {
            self.config.audio_device = audio_device;
        }
        if let Some(volume) = patch.volume {
            self.config.volume = volume.clamp(0.0, 1.0);
        }
        if let Some(max_input_chars) = patch.max_input_chars {
            self.config.max_input_chars = max_input_chars.max(1);
        }
        if let Some(generation) = patch.generation {
            self.config.generation = generation;
        }
        Ok(self.snapshot())
    }

    pub fn rescan_models(&mut self) -> Result<Vec<ModelChoice>> {
        self.models = if let Some(root) = self.config.model_root.as_ref() {
            discover_models(root)?
        } else {
            Vec::new()
        };
        self.selected_model_id = select_existing_model(self.selected_model_id.clone(), &self.models);
        self.config.selected_model_id = self.selected_model_id.clone();
        Ok(self.models.clone())
    }

    pub fn cancel_model_load_state(&mut self) {
        self.mark_load_finished_without_swap();
    }

    pub fn cancel_generation_item(&mut self, item_id: &str) -> bool {
        self.queue.cancel_queued(item_id)
    }
```

- [ ] **Step 3: Implement command wrappers**

Create `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`:

```rust
use std::sync::{Arc, Mutex};

use tauri::{Emitter, State, Window};

use crate::app_core::AppCore;
use crate::generation_queue::HistoryItem;
use crate::types::{AppSnapshot, CommandResult, ConfigPatch, ModelChoice};

pub type SharedAppCore = Arc<Mutex<AppCore>>;

#[tauri::command]
pub fn get_app_state(state: State<'_, SharedAppCore>) -> Result<AppSnapshot, String> {
    with_core(&state, |core| Ok(core.snapshot()))
}

#[tauri::command]
pub fn set_config_patch(
    state: State<'_, SharedAppCore>,
    patch: ConfigPatch,
) -> Result<AppSnapshot, String> {
    with_core(&state, |core| core.apply_patch(patch).map_err(|err| err.to_string()))
}

#[tauri::command]
pub fn discover_models(state: State<'_, SharedAppCore>) -> Result<Vec<ModelChoice>, String> {
    with_core(&state, |core| core.rescan_models().map_err(|err| err.to_string()))
}

#[tauri::command]
pub fn enqueue_generation(
    state: State<'_, SharedAppCore>,
    text: String,
) -> Result<HistoryItem, String> {
    with_core(&state, |core| core.enqueue_generation(text).map_err(|err| err.to_string()))
}

#[tauri::command]
pub fn cancel_model_load(state: State<'_, SharedAppCore>) -> Result<CommandResult, String> {
    with_core(&state, |core| {
        core.cancel_model_load_state();
        Ok(CommandResult { ok: true })
    })
}

#[tauri::command]
pub fn cancel_generation(
    state: State<'_, SharedAppCore>,
    item_id: String,
) -> Result<CommandResult, String> {
    with_core(&state, |core| {
        let ok = core.cancel_generation_item(&item_id);
        Ok(CommandResult { ok })
    })
}

#[tauri::command]
pub fn stop_audio(window: Window) -> Result<CommandResult, String> {
    window
        .emit("playback_state", crate::types::PlaybackStateEvent {
            item_id: None,
            state: "stopped".to_string(),
        })
        .map_err(|err| err.to_string())?;
    Ok(CommandResult { ok: true })
}

fn with_core<T>(
    state: &State<'_, SharedAppCore>,
    f: impl FnOnce(&mut AppCore) -> Result<T, String>,
) -> Result<T, String> {
    let mut core = state.lock().map_err(|_| "app state lock poisoned".to_string())?;
    f(&mut core)
}
```

These commands already mutate app state or emit playback state events; model-load token cancellation is connected in the model-loading task when the cancellation token is introduced.

- [ ] **Step 4: Register app state and commands**

Modify `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`:

```rust
pub mod app_core;
pub mod audio;
pub mod commands;
pub mod config;
pub mod generation_queue;
pub mod model_discovery;
pub mod playback;
pub mod types;

use std::sync::{Arc, Mutex};

use app_core::AppCore;
use commands::{cancel_generation, cancel_model_load, discover_models, enqueue_generation, get_app_state, set_config_patch, stop_audio};
use config::AppConfig;

pub fn run() {
    tracing_subscriber::fmt::init();
    let core = AppCore::from_config(AppConfig::default()).expect("initialize app core");
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(Arc::new(Mutex::new(core)))
        .invoke_handler(tauri::generate_handler![
            get_app_state,
            set_config_patch,
            discover_models,
            enqueue_generation,
            cancel_model_load,
            cancel_generation,
            stop_audio,
        ])
        .run(tauri::generate_context!())
        .expect("failed to run AhanSays desktop app");
}
```

- [ ] **Step 5: Run backend tests and check**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop
cargo check -p voxui-desktop
```

Expected: PASS for tests and check. Fix type visibility and Tauri command signature issues.

- [ ] **Step 6: Commit commands**

Run:

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src/types.rs voxui/crates/voxui-desktop/src-tauri/src/app_core.rs voxui/crates/voxui-desktop/src-tauri/src/commands.rs voxui/crates/voxui-desktop/src-tauri/src/lib.rs
git commit -m "feat: add desktop command surface"
```

---

### Task 8: Model Loading Orchestration

**Files:**
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/app_core.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/types.rs`
- Test: `voxui/crates/voxui-desktop/src-tauri/tests/app_core_tests.rs`

- [ ] **Step 1: Add a test for failed load preserving current loaded id**

Append to `voxui/crates/voxui-desktop/src-tauri/tests/app_core_tests.rs`:

```rust
#[test]
fn failed_load_preserves_previous_loaded_model_id() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("old-model".to_string());

    core.finish_model_load_for_test(Err("load failed".to_string()));

    assert_eq!(core.snapshot().loaded_model_id.as_deref(), Some("old-model"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop --test app_core_tests failed_load_preserves_previous_loaded_model_id
```

Expected: FAIL because `finish_model_load_for_test` is not defined.

- [ ] **Step 3: Add load start/result types**

Append to `voxui/crates/voxui-desktop/src-tauri/src/types.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadStartResult {
    pub started: bool,
    pub choice_id: String,
}
```

- [ ] **Step 4: Add app-core load state helpers**

Append methods inside `impl AppCore` in `voxui/crates/voxui-desktop/src-tauri/src/app_core.rs`:

```rust
    pub fn selected_choice(&self) -> Result<ModelChoice> {
        let selected = self
            .selected_model_id
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no model selected"))?;
        self.models
            .iter()
            .find(|choice| &choice.id == selected)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("selected model is no longer available"))
    }

    pub fn mark_load_started(&mut self) {
        self.load_state = LoadUiState::Loading;
    }

    pub fn mark_load_success(&mut self, choice_id: String, engine: voxui_inference::VoxCPMEngine) {
        self.engine = Some(engine);
        self.loaded_model_id = Some(choice_id);
        self.load_state = LoadUiState::Idle;
    }

    pub fn mark_load_finished_without_swap(&mut self) {
        self.load_state = LoadUiState::Idle;
    }

    pub fn finish_model_load_for_test(&mut self, result: Result<(), String>) {
        match result {
            Ok(()) => {
                self.loaded_model_id = Some("new-model".to_string());
                self.load_state = LoadUiState::Idle;
            }
            Err(_) => self.mark_load_finished_without_swap(),
        }
    }
```

Add an engine field to `AppCore` in the struct initializer and struct definition:

```rust
engine: Option<voxui_inference::VoxCPMEngine>,
```

Initialize it in `from_config`:

```rust
engine: None,
```

- [ ] **Step 5: Implement command-level load orchestration**

Modify `voxui/crates/voxui-desktop/src-tauri/src/commands.rs` to add imports:

```rust
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::Device;
use tauri::{Emitter, Window};
use tokio::task;
use voxui_inference::VoxCPMEngine;
```

Add command:

```rust
#[tauri::command]
pub fn load_model(
    window: Window,
    state: State<'_, SharedAppCore>,
    choice_id: String,
) -> Result<crate::types::LoadStartResult, String> {
    let choice = with_core(&state, |core| {
        let choice = core.selected_choice().map_err(|err| err.to_string())?;
        if choice.id != choice_id {
            return Err("requested choice does not match selected model".to_string());
        }
        core.mark_load_started();
        Ok(choice)
    })?;

    let shared = state.inner().clone();
    task::spawn_blocking(move || {
        let result = load_engine_for_choice(&window, &choice);
        let done = match result {
            Ok(engine) => {
                if let Ok(mut core) = shared.lock() {
                    core.mark_load_success(choice.id.clone(), engine);
                }
                crate::types::ModelLoadDoneEvent {
                    status: "success".to_string(),
                    selected_model_id: Some(choice.id.clone()),
                    loaded_model_id: Some(choice.id.clone()),
                    error: None,
                }
            }
            Err(error) => {
                if let Ok(mut core) = shared.lock() {
                    core.mark_load_finished_without_swap();
                    let loaded = core.snapshot().loaded_model_id;
                    let _ = window.emit("model_load_done", crate::types::ModelLoadDoneEvent {
                        status: "error".to_string(),
                        selected_model_id: Some(choice.id.clone()),
                        loaded_model_id: loaded,
                        error: Some(error),
                    });
                    return;
                }
                crate::types::ModelLoadDoneEvent {
                    status: "error".to_string(),
                    selected_model_id: Some(choice.id.clone()),
                    loaded_model_id: None,
                    error: Some("app state lock poisoned".to_string()),
                }
            }
        };
        let _ = window.emit("model_load_done", done);
    });

    Ok(crate::types::LoadStartResult {
        started: true,
        choice_id,
    })
}

fn load_engine_for_choice(window: &Window, choice: &crate::types::ModelChoice) -> Result<VoxCPMEngine, String> {
    let total_bytes = choice.model_bytes + choice.lora_bytes;
    window
        .emit("model_load_progress", crate::types::ModelLoadProgressEvent {
            phase: "reading".to_string(),
            loaded_bytes: total_bytes,
            total_bytes,
            component: None,
            component_index: 0,
            component_total: 0,
        })
        .map_err(|err| err.to_string())?;

    let cancel = AtomicBool::new(false);
    let mut engine = VoxCPMEngine::load_with_progress(
        &choice.model_dir,
        Device::Cpu,
        |current, total| {
            let _ = window.emit("model_load_progress", crate::types::ModelLoadProgressEvent {
                phase: "device_loading".to_string(),
                loaded_bytes: total_bytes,
                total_bytes,
                component: None,
                component_index: current,
                component_total: total,
            });
        },
        Some(&cancel),
    )
    .map_err(|err| err.to_string())?;

    if let Some(lora_path) = choice.lora_path.as_ref() {
        engine.load_lora(lora_path).map_err(|err| err.to_string())?;
    }

    Ok(engine)
}
```

Also add `load_model` to the `generate_handler!` list in `lib.rs`. This step wires real loading progress, LoRA application, and installation of the loaded `VoxCPMEngine`.

- [ ] **Step 6: Run load state test and backend check**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop --test app_core_tests failed_load_preserves_previous_loaded_model_id
cargo check -p voxui-desktop
```

Expected: PASS.

- [ ] **Step 7: Commit loading orchestration**

Run:

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src/types.rs voxui/crates/voxui-desktop/src-tauri/src/app_core.rs voxui/crates/voxui-desktop/src-tauri/src/commands.rs voxui/crates/voxui-desktop/src-tauri/src/lib.rs voxui/crates/voxui-desktop/src-tauri/tests/app_core_tests.rs
git commit -m "feat: orchestrate desktop model loading"
```

---

### Task 9: Generation And Playback Commands

**Files:**
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/app_core.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/playback.rs`
- Test: `voxui/crates/voxui-desktop/src-tauri/tests/app_core_tests.rs`

- [ ] **Step 1: Add request snapshot conversion test**

Append to `voxui/crates/voxui-desktop/src-tauri/tests/app_core_tests.rs`:

```rust
#[test]
fn request_snapshot_converts_to_synthesis_request() {
    let mut core = AppCore::from_config(AppConfig::default()).unwrap();
    core.set_loaded_model_for_test("model".to_string());
    let item = core.enqueue_generation(" hello world ".to_string()).unwrap();

    let request = core.synthesis_request_for_test(&item.id).unwrap();

    assert_eq!(request.text, "hello world");
    assert_eq!(request.inference_timesteps, 10);
    assert_eq!(request.cfg_value, 2.0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop --test app_core_tests request_snapshot_converts_to_synthesis_request
```

Expected: FAIL because `synthesis_request_for_test` is not defined.

- [ ] **Step 3: Add request conversion helper**

Append inside `impl AppCore` in `voxui/crates/voxui-desktop/src-tauri/src/app_core.rs`:

```rust
    pub fn synthesis_request_for_test(&self, item_id: &str) -> Result<voxui_inference::SynthesisRequest> {
        let item = self
            .queue
            .items()
            .iter()
            .find(|item| item.id == item_id)
            .ok_or_else(|| anyhow::anyhow!("unknown history item: {item_id}"))?;
        Ok(voxui_inference::SynthesisRequest {
            text: item.text.clone(),
            prompt_wav_path: item.snapshot.generation.prompt_wav_path.clone(),
            prompt_text: item.snapshot.generation.prompt_text.clone(),
            reference_wav_path: item.snapshot.generation.reference_wav_path.clone(),
            cfg_value: item.snapshot.generation.cfg_value,
            inference_timesteps: item.snapshot.generation.inference_timesteps,
            min_len: item.snapshot.generation.min_len,
            max_len: item.snapshot.generation.max_len,
            normalize: false,
            retry_badcase: item.snapshot.generation.retry_badcase,
            retry_badcase_max_times: item.snapshot.generation.retry_badcase_max_times,
            retry_badcase_ratio_threshold: item.snapshot.generation.retry_badcase_ratio_threshold,
        })
    }
```

- [ ] **Step 4: Implement play command with cache lookup**

Extend `GeneratedAudioCache` in `playback.rs`:

```rust
impl GeneratedAudioCache {
    pub fn contains(&self, item_id: &str) -> bool {
        self.items.contains_key(item_id)
    }
}
```

Add app-core playback helper:

```rust
    pub fn has_audio(&self, item_id: &str) -> bool {
        self.queue.items().iter().any(|item| item.id == item_id && item.has_audio)
    }
```

Update `stop_audio` and add `play_audio` in `commands.rs`:

```rust
#[tauri::command]
pub fn play_audio(
    window: Window,
    state: State<'_, SharedAppCore>,
    item_id: String,
) -> Result<CommandResult, String> {
    let has_audio = with_core(&state, |core| Ok(core.has_audio(&item_id)))?;
    if !has_audio {
        return Err(format!("no generated audio for item {item_id}"));
    }
    window
        .emit("playback_state", crate::types::PlaybackStateEvent {
            item_id: Some(item_id),
            state: "playing".to_string(),
        })
        .map_err(|err| err.to_string())?;
    Ok(CommandResult { ok: true })
}

#[tauri::command]
pub fn stop_audio(window: Window) -> Result<CommandResult, String> {
    window
        .emit("playback_state", crate::types::PlaybackStateEvent {
            item_id: None,
            state: "stopped".to_string(),
        })
        .map_err(|err| err.to_string())?;
    Ok(CommandResult { ok: true })
}
```

Add `play_audio` to `generate_handler!`.

- [ ] **Step 5: Implement generation command flow**

Add to `commands.rs`:

```rust
#[tauri::command]
pub fn regenerate(
    state: State<'_, SharedAppCore>,
    item_id: String,
) -> Result<CommandResult, String> {
    with_core(&state, |core| {
        let config = core.snapshot().config;
        core.regenerate_item(&item_id, &config).map_err(|err| err.to_string())?;
        Ok(CommandResult { ok: true })
    })
}
```

Add app-core method:

```rust
    pub fn regenerate_item(&mut self, item_id: &str, config: &AppConfig) -> Result<()> {
        self.queue.start_regeneration(item_id, config)
    }
```

Add `regenerate` to `generate_handler!`.

Replace the earlier `enqueue_generation` command with one that runs the real sequential generation path and emits progress/done events:

```rust
#[tauri::command]
pub fn enqueue_generation(
    window: Window,
    state: State<'_, SharedAppCore>,
    text: String,
) -> Result<HistoryItem, String> {
    let item = with_core(&state, |core| core.enqueue_generation(text).map_err(|err| err.to_string()))?;
    let item_id = item.id.clone();
    let shared = state.inner().clone();
    task::spawn_blocking(move || {
        let result = {
            let mut core = shared.lock().map_err(|_| "app state lock poisoned".to_string())?;
            core.run_generation_now(&item_id, |current, total| {
                let _ = window.emit("generation_progress", crate::types::GenerationProgressEvent {
                    item_id: item_id.clone(),
                    current,
                    total,
                });
            })
        };

        let event = match result {
            Ok((sample_rate, duration_seconds)) => crate::types::GenerationDoneEvent {
                item_id,
                status: "success".to_string(),
                error: None,
                sample_rate: Some(sample_rate),
                duration_seconds: Some(duration_seconds),
            },
            Err(error) => crate::types::GenerationDoneEvent {
                item_id,
                status: "error".to_string(),
                error: Some(error),
                sample_rate: None,
                duration_seconds: None,
            },
        };
        let _ = window.emit("generation_done", event);
    });
    Ok(item)
}
```

Add the real generation method to `AppCore`:

```rust
    pub fn run_generation_now(
        &mut self,
        item_id: &str,
        progress: impl Fn(usize, usize),
    ) -> Result<(u32, f32), String> {
        self.queue
            .mark_generating(item_id)
            .map_err(|err| err.to_string())?;
        let request = self
            .synthesis_request_for_test(item_id)
            .map_err(|err| err.to_string())?;
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| "no model loaded".to_string())?;
        let sample_rate = engine.sample_rate();
        let samples = engine
            .generate_cancellable(request, progress, None)
            .map_err(|err| err.to_string())?;
        let duration_seconds = samples.len() as f32 / sample_rate as f32;
        self.audio_cache.insert(item_id.to_string(), samples, sample_rate);
        self.queue.mark_ready(item_id, true);
        Ok((sample_rate, duration_seconds))
    }
```

Add an audio cache field to `AppCore` in the struct definition and initializer:

```rust
audio_cache: crate::playback::GeneratedAudioCache,
```

```rust
audio_cache: crate::playback::GeneratedAudioCache::default(),
```

- [ ] **Step 6: Run generation conversion tests and check**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop --test app_core_tests request_snapshot_converts_to_synthesis_request
cargo check -p voxui-desktop
```

Expected: PASS.

- [ ] **Step 7: Commit generation/playback command surface**

Run:

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src/app_core.rs voxui/crates/voxui-desktop/src-tauri/src/commands.rs voxui/crates/voxui-desktop/src-tauri/src/playback.rs voxui/crates/voxui-desktop/src-tauri/src/lib.rs voxui/crates/voxui-desktop/src-tauri/tests/app_core_tests.rs
git commit -m "feat: add desktop generation and playback commands"
```

---

### Task 10: Frontend API And I18n

**Files:**
- Create: `voxui/crates/voxui-desktop/src/i18n.rs`
- Create: `voxui/crates/voxui-desktop/src/tauri_api.rs`
- Modify: `voxui/crates/voxui-desktop/src/app.rs`

- [ ] **Step 1: Implement frontend i18n labels**

Create `voxui/crates/voxui-desktop/src/i18n.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiLanguage {
    Chinese,
    English,
}

#[derive(Debug, Clone, Copy)]
pub struct Labels {
    pub title: &'static str,
    pub subtitle: &'static str,
    pub load: &'static str,
    pub generate: &'static str,
    pub settings: &'static str,
    pub model: &'static str,
    pub input_placeholder: &'static str,
    pub history_empty: &'static str,
    pub cancel: &'static str,
    pub play: &'static str,
    pub stop: &'static str,
    pub regenerate: &'static str,
}

pub fn labels(language: UiLanguage) -> Labels {
    match language {
        UiLanguage::Chinese => Labels {
            title: "焓言焓语",
            subtitle: "AhanSays",
            load: "加载",
            generate: "生成",
            settings: "设置",
            model: "模型",
            input_placeholder: "输入要合成的文字...",
            history_empty: "暂无生成记录",
            cancel: "取消",
            play: "播放",
            stop: "停止",
            regenerate: "重新生成",
        },
        UiLanguage::English => Labels {
            title: "AhanSays",
            subtitle: "焓言焓语",
            load: "Load",
            generate: "Generate",
            settings: "Settings",
            model: "Model",
            input_placeholder: "Enter text to synthesize...",
            history_empty: "No generation history yet",
            cancel: "Cancel",
            play: "Play",
            stop: "Stop",
            regenerate: "Regenerate",
        },
    }
}
```

- [ ] **Step 2: Implement typed frontend DTOs and invoke wrappers**

Create `voxui/crates/voxui-desktop/src/tauri_api.rs`:

```rust
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"])]
    async fn invoke(cmd: &str, args: JsValue) -> JsValue;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppSnapshot {
    pub config: AppConfig,
    pub models: Vec<ModelChoice>,
    pub selected_model_id: Option<String>,
    pub loaded_model_id: Option<String>,
    pub history: Vec<HistoryItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    pub volume: f32,
    pub max_input_chars: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelChoice {
    pub id: String,
    pub display_name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryItem {
    pub id: String,
    pub text: String,
    pub status: String,
    pub progress_current: usize,
    pub progress_total: usize,
    pub has_audio: bool,
}

pub async fn get_app_state() -> Result<AppSnapshot, String> {
    let value = invoke("get_app_state", JsValue::NULL).await;
    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}
```

- [ ] **Step 3: Wire app shell to initial state**

Modify `voxui/crates/voxui-desktop/src/app.rs`:

```rust
use leptos::prelude::*;

mod i18n_bridge {
    pub use crate::i18n::{labels, UiLanguage};
}

#[component]
pub fn App() -> impl IntoView {
    let labels = i18n_bridge::labels(i18n_bridge::UiLanguage::Chinese);

    view! {
        <div class="app-shell">
            <header class="app-header">
                <div class="brand">
                    <strong>{labels.title}</strong>
                    <span>{labels.subtitle}</span>
                </div>
                <select class="model-select" aria-label={labels.model}>
                    <option>{labels.model}</option>
                </select>
                <button class="primary-button">{labels.load}</button>
                <button class="icon-button" title={labels.settings} aria-label={labels.settings}>{"⚙"}</button>
            </header>
            <section class="history-panel">
                <p class="empty-history">{labels.history_empty}</p>
            </section>
            <footer class="composer-panel">
                <textarea class="composer-input" placeholder={labels.input_placeholder}></textarea>
                <button class="generate-button">{labels.generate}</button>
            </footer>
        </div>
    }
}
```

Modify `voxui/crates/voxui-desktop/src/main.rs`:

```rust
mod app;
mod i18n;
mod tauri_api;

fn main() {
    leptos::mount_to_body(app::App);
}
```

- [ ] **Step 4: Extend styles for controls**

Append to `voxui/crates/voxui-desktop/src/styles.css`:

```css
.model-select {
  margin-left: auto;
  min-width: 260px;
  height: 34px;
  border: 1px solid #3a4656;
  border-radius: 6px;
  background: #10161f;
  color: #e7eaee;
  padding: 0 10px;
}

.primary-button,
.generate-button,
.icon-button {
  height: 34px;
  border: 1px solid #3a4656;
  border-radius: 6px;
  background: #1d2632;
  color: #e7eaee;
  padding: 0 12px;
}

.primary-button,
.generate-button {
  background: #d5b15b;
  border-color: #d5b15b;
  color: #15120b;
}

.history-panel {
  min-height: 0;
  overflow: auto;
  padding: 18px;
}

.empty-history {
  margin: 0;
  color: #8995a5;
}

.composer-panel {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px;
  padding: 14px 18px;
  border-top: 1px solid #2b3442;
  background: #151b23;
}

.composer-input {
  width: 100%;
  min-height: 74px;
  resize: none;
  border: 1px solid #3a4656;
  border-radius: 8px;
  background: #10161f;
  color: #e7eaee;
  padding: 10px;
  font: inherit;
}

.generate-button {
  align-self: end;
  min-width: 96px;
}
```

- [ ] **Step 5: Build frontend**

Run:

```powershell
cd voxui\crates\voxui-desktop
trunk build
```

Expected: PASS.

- [ ] **Step 6: Commit frontend API/i18n**

Run:

```powershell
git add voxui/crates/voxui-desktop/src
git commit -m "feat: add desktop frontend shell"
```

---

### Task 11: Frontend Components

**Files:**
- Create: `voxui/crates/voxui-desktop/src/components/mod.rs`
- Create: `voxui/crates/voxui-desktop/src/components/header.rs`
- Create: `voxui/crates/voxui-desktop/src/components/history.rs`
- Create: `voxui/crates/voxui-desktop/src/components/input_box.rs`
- Create: `voxui/crates/voxui-desktop/src/components/settings_modal.rs`
- Create: `voxui/crates/voxui-desktop/src/components/load_progress_modal.rs`
- Modify: `voxui/crates/voxui-desktop/src/app.rs`
- Modify: `voxui/crates/voxui-desktop/src/styles.css`

- [ ] **Step 1: Create component module exports**

Create `voxui/crates/voxui-desktop/src/components/mod.rs`:

```rust
pub mod header;
pub mod history;
pub mod input_box;
pub mod load_progress_modal;
pub mod settings_modal;
```

- [ ] **Step 2: Create header component**

Create `voxui/crates/voxui-desktop/src/components/header.rs`:

```rust
use leptos::prelude::*;

use crate::i18n::Labels;
use crate::tauri_api::ModelChoice;

#[component]
pub fn Header(
    labels: Labels,
    models: Vec<ModelChoice>,
    selected_model_id: Option<String>,
    loaded_model_id: Option<String>,
    load_disabled: bool,
    on_load: impl Fn() + 'static + Copy,
    on_open_settings: impl Fn() + 'static + Copy,
) -> impl IntoView {
    let selected = selected_model_id.unwrap_or_default();
    let loaded = loaded_model_id.unwrap_or_default();

    view! {
        <header class="app-header">
            <div class="brand">
                <strong>{labels.title}</strong>
                <span>{labels.subtitle}</span>
            </div>
            <select class="model-select" aria-label={labels.model}>
                {models.into_iter().map(|model| {
                    let is_selected = model.id == selected;
                    view! {
                        <option value={model.id.clone()} selected=is_selected>
                            {model.display_name}
                        </option>
                    }
                }).collect_view()}
            </select>
            <button class="primary-button" disabled=load_disabled on:click=move |_| on_load()>{labels.load}</button>
            <button class="icon-button" title={labels.settings} aria-label={labels.settings} on:click=move |_| on_open_settings()>{"⚙"}</button>
            <span class="loaded-pill">{loaded}</span>
        </header>
    }
}
```

- [ ] **Step 3: Create history component**

Create `voxui/crates/voxui-desktop/src/components/history.rs`:

```rust
use leptos::prelude::*;

use crate::i18n::Labels;
use crate::tauri_api::HistoryItem;

#[component]
pub fn HistoryList(
    labels: Labels,
    items: Vec<HistoryItem>,
    on_play: impl Fn(String) + 'static + Copy,
    on_regenerate: impl Fn(String) + 'static + Copy,
    on_cancel: impl Fn(String) + 'static + Copy,
) -> impl IntoView {
    if items.is_empty() {
        return view! { <section class="history-panel"><p class="empty-history">{labels.history_empty}</p></section> }.into_any();
    }

    view! {
        <section class="history-panel">
            {items.into_iter().map(|item| {
                let progress = if item.progress_total == 0 {
                    0.0
                } else {
                    item.progress_current as f32 / item.progress_total as f32
                };
                let item_id = item.id.clone();
                let play_id = item.id.clone();
                let regen_id = item.id.clone();
                let cancel_id = item.id.clone();
                view! {
                    <article class="history-item">
                        <div class="history-text">{item.text}</div>
                        <div class="history-meta">{item.status.clone()}</div>
                        <progress max="1" value={progress.to_string()}></progress>
                        <div class="history-actions">
                            <button on:click=move |_| on_cancel(cancel_id.clone())>{labels.cancel}</button>
                            <button disabled=!item.has_audio on:click=move |_| on_play(play_id.clone())>{labels.play}</button>
                            <button on:click=move |_| on_regenerate(regen_id.clone())>{labels.regenerate}</button>
                        </div>
                        <input type="hidden" value={item_id} />
                    </article>
                }
            }).collect_view()}
        </section>
    }.into_any()
}
```

- [ ] **Step 4: Create input component**

Create `voxui/crates/voxui-desktop/src/components/input_box.rs`:

```rust
use leptos::prelude::*;

use crate::i18n::Labels;

#[component]
pub fn InputBox(
    labels: Labels,
    max_chars: usize,
    disabled: bool,
    on_generate: impl Fn(String) + 'static + Copy,
) -> impl IntoView {
    let (text, set_text) = signal(String::new());
    let count = move || text.get().chars().count();
    let can_generate = move || !disabled && count() > 0 && count() <= max_chars;

    view! {
        <footer class="composer-panel">
            <div class="composer-field">
                <textarea
                    class="composer-input"
                    placeholder={labels.input_placeholder}
                    prop:value=move || text.get()
                    on:input=move |event| set_text.set(event_target_value(&event))
                ></textarea>
                <span class="char-counter">{move || format!("{}/{}", count(), max_chars)}</span>
            </div>
            <button
                class="generate-button"
                disabled=move || !can_generate()
                on:click=move |_| on_generate(text.get())
            >
                {labels.generate}
            </button>
        </footer>
    }
}
```

- [ ] **Step 5: Create modal components**

Create `voxui/crates/voxui-desktop/src/components/settings_modal.rs`:

```rust
use leptos::prelude::*;

use crate::i18n::Labels;

#[component]
pub fn SettingsModal(labels: Labels, open: bool, on_close: impl Fn() + 'static + Copy) -> impl IntoView {
    view! {
        <Show when=move || open>
            <div class="modal-backdrop">
                <section class="modal">
                    <header class="modal-header">
                        <h2>{labels.settings}</h2>
                        <button class="icon-button" on:click=move |_| on_close()>{"×"}</button>
                    </header>
                    <div class="settings-grid">
                        <label>"Models folder"<input class="setting-input" /></label>
                        <label>"Language"<select class="setting-input"><option>"System"</option><option>"中文"</option><option>"English"</option></select></label>
                        <label>"Backend"<select class="setting-input"><option>"CPU"</option><option>"CUDA"</option></select></label>
                        <label>"Volume"<input class="setting-input" type="range" min="0" max="100" /></label>
                    </div>
                </section>
            </div>
        </Show>
    }
}
```

Create `voxui/crates/voxui-desktop/src/components/load_progress_modal.rs`:

```rust
use leptos::prelude::*;

#[component]
pub fn LoadProgressModal(open: bool, percent: f32, on_cancel: impl Fn() + 'static + Copy) -> impl IntoView {
    view! {
        <Show when=move || open>
            <div class="modal-backdrop">
                <section class="modal compact-modal">
                    <h2>"Loading model"</h2>
                    <progress max="1" value={percent.to_string()}></progress>
                    <button class="primary-button" on:click=move |_| on_cancel()>"Cancel"</button>
                </section>
            </div>
        </Show>
    }
}
```

- [ ] **Step 6: Wire components in app**

Modify `voxui/crates/voxui-desktop/src/main.rs`:

```rust
mod app;
mod components;
mod i18n;
mod tauri_api;

fn main() {
    leptos::mount_to_body(app::App);
}
```

Modify `voxui/crates/voxui-desktop/src/app.rs`:

```rust
use leptos::prelude::*;

use crate::components::header::Header;
use crate::components::history::HistoryList;
use crate::components::input_box::InputBox;
use crate::components::load_progress_modal::LoadProgressModal;
use crate::components::settings_modal::SettingsModal;
use crate::i18n::{labels, UiLanguage};
use crate::tauri_api::{AppConfig, AppSnapshot};

#[component]
pub fn App() -> impl IntoView {
    let labels = labels(UiLanguage::Chinese);
    let (settings_open, set_settings_open) = signal(false);
    let (load_open, set_load_open) = signal(false);
    let snapshot = AppSnapshot {
        config: AppConfig { volume: 0.8, max_input_chars: 280 },
        models: Vec::new(),
        selected_model_id: None,
        loaded_model_id: None,
        history: Vec::new(),
    };

    view! {
        <div class="app-shell">
            <Header
                labels=labels
                models=snapshot.models.clone()
                selected_model_id=snapshot.selected_model_id.clone()
                loaded_model_id=snapshot.loaded_model_id.clone()
                load_disabled=false
                on_load=move || set_load_open.set(true)
                on_open_settings=move || set_settings_open.set(true)
            />
            <HistoryList
                labels=labels
                items=snapshot.history.clone()
                on_play=move |_| {}
                on_regenerate=move |_| {}
                on_cancel=move |_| {}
            />
            <InputBox
                labels=labels
                max_chars=snapshot.config.max_input_chars
                disabled=false
                on_generate=move |_| {}
            />
            <SettingsModal labels=labels open=move || settings_open.get() on_close=move || set_settings_open.set(false) />
            <LoadProgressModal open=move || load_open.get() percent=0.35 on_cancel=move || set_load_open.set(false) />
        </div>
    }
}
```

- [ ] **Step 7: Add component styles**

Append to `voxui/crates/voxui-desktop/src/styles.css`:

```css
.loaded-pill {
  color: #8995a5;
  font-size: 12px;
  min-width: 100px;
}

.history-item {
  display: grid;
  grid-template-columns: 1fr 120px;
  gap: 8px 12px;
  padding: 12px;
  border: 1px solid #2b3442;
  border-radius: 8px;
  background: #141a22;
  margin-bottom: 10px;
}

.history-text {
  color: #e7eaee;
  overflow-wrap: anywhere;
}

.history-meta {
  color: #9ca8b8;
  text-align: right;
}

.history-item progress {
  width: 100%;
}

.history-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

.history-actions button {
  border: 1px solid #3a4656;
  border-radius: 6px;
  background: #1d2632;
  color: #e7eaee;
  height: 30px;
}

.composer-field {
  position: relative;
}

.char-counter {
  position: absolute;
  right: 10px;
  bottom: 8px;
  color: #8995a5;
  font-size: 12px;
}

.modal-backdrop {
  position: fixed;
  inset: 0;
  display: grid;
  place-items: center;
  background: rgba(0, 0, 0, 0.45);
}

.modal {
  width: min(680px, calc(100vw - 40px));
  max-height: calc(100vh - 40px);
  overflow: auto;
  border: 1px solid #3a4656;
  border-radius: 8px;
  background: #151b23;
  padding: 18px;
}

.compact-modal {
  width: min(420px, calc(100vw - 40px));
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.settings-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.setting-input {
  display: block;
  width: 100%;
  margin-top: 6px;
}
```

- [ ] **Step 8: Build frontend**

Run:

```powershell
cd voxui\crates\voxui-desktop
trunk build
```

Expected: PASS.

- [ ] **Step 9: Commit components**

Run:

```powershell
git add voxui/crates/voxui-desktop/src
git commit -m "feat: build AhanSays workbench UI"
```

---

### Task 12: Full Wiring, Verification, And Manual Launch

**Files:**
- Modify: `voxui/crates/voxui-desktop/src/tauri_api.rs`
- Modify: `voxui/crates/voxui-desktop/src/app.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`

- [ ] **Step 1: Expand frontend invoke wrappers**

Append to `voxui/crates/voxui-desktop/src/tauri_api.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TextArg {
    text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ItemArg {
    item_id: String,
}

pub async fn enqueue_generation(text: String) -> Result<HistoryItem, String> {
    let args = serde_wasm_bindgen::to_value(&TextArg { text }).map_err(|err| err.to_string())?;
    let value = invoke("enqueue_generation", args).await;
    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn play_audio(item_id: String) -> Result<(), String> {
    let args = serde_wasm_bindgen::to_value(&ItemArg { item_id }).map_err(|err| err.to_string())?;
    let _ = invoke("play_audio", args).await;
    Ok(())
}

pub async fn regenerate(item_id: String) -> Result<(), String> {
    let args = serde_wasm_bindgen::to_value(&ItemArg { item_id }).map_err(|err| err.to_string())?;
    let _ = invoke("regenerate", args).await;
    Ok(())
}
```

- [ ] **Step 2: Load initial app state in frontend**

Replace the static snapshot in `app.rs` with:

```rust
let (snapshot, set_snapshot) = signal(None::<AppSnapshot>);

spawn_local(async move {
    if let Ok(next) = crate::tauri_api::get_app_state().await {
        set_snapshot.set(Some(next));
    }
});
```

Render with fallback:

```rust
let current = move || snapshot.get().unwrap_or(AppSnapshot {
    config: AppConfig { volume: 0.8, max_input_chars: 280 },
    models: Vec::new(),
    selected_model_id: None,
    loaded_model_id: None,
    history: Vec::new(),
});
```

Use `current().models`, `current().history`, and `current().config.max_input_chars` where the static snapshot was used.

- [ ] **Step 3: Wire generate/play/regenerate buttons**

In `app.rs`, replace inert UI callbacks:

```rust
on_play=move |item_id| {
    spawn_local(async move {
        let _ = crate::tauri_api::play_audio(item_id).await;
    });
}
on_regenerate=move |item_id| {
    spawn_local(async move {
        let _ = crate::tauri_api::regenerate(item_id).await;
    });
}
on_generate=move |text| {
    let set_snapshot = set_snapshot;
    spawn_local(async move {
        if crate::tauri_api::enqueue_generation(text).await.is_ok() {
            if let Ok(next) = crate::tauri_api::get_app_state().await {
                set_snapshot.set(Some(next));
            }
        }
    });
}
```

- [ ] **Step 4: Add missing Tauri commands to capabilities**

Update `voxui/crates/voxui-desktop/src-tauri/capabilities/default.json` permissions if Tauri requires explicit command permissions after generation:

```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "default",
  "description": "Default AhanSays desktop permissions",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "dialog:default",
    "opener:default"
  ]
}
```

Run `cargo tauri dev` after this file is present and keep `default.json` limited to the command permissions reported by Tauri for this app.

- [ ] **Step 5: Full verification**

Run:

```powershell
cd voxui
cargo test -p voxui-desktop
cargo check -p voxui-desktop
cargo check -p voxui-desktop --features cuda
```

Expected: all pass.

Run:

```powershell
cd voxui\crates\voxui-desktop
trunk build
```

Expected: build succeeds and writes `dist`.

- [ ] **Step 6: Manual launch**

Run:

```powershell
cd voxui\crates\voxui-desktop\src-tauri
cargo tauri dev
```

Expected:

- Window opens with `焓言焓语` / `AhanSays`.
- Header shows model dropdown, load button, settings button.
- Center history area is visible.
- Bottom input and generate button are visible.
- Settings modal opens.
- Load progress modal can be shown by pressing Load when a discovered model is present.

- [ ] **Step 7: Commit full wiring**

Run:

```powershell
git add voxui/crates/voxui-desktop
git commit -m "feat: wire AhanSays desktop UI"
```

---

### Task 13: Required Settings, Dialogs, Audio Test, And Event Listeners

**Files:**
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`
- Modify: `voxui/crates/voxui-desktop/src/components/settings_modal.rs`
- Modify: `voxui/crates/voxui-desktop/src/tauri_api.rs`
- Modify: `voxui/crates/voxui-desktop/src/app.rs`

- [ ] **Step 1: Add dialog and audio-test commands**

Append to `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`:

```rust
use tauri::AppHandle;
use tauri_plugin_dialog::DialogExt;
use voxui_audio::{AudioPlayer, AudioSystem};

#[tauri::command]
pub fn browse_model_dir(app: AppHandle) -> Result<Option<String>, String> {
    Ok(app
        .dialog()
        .file()
        .blocking_pick_folder()
        .map(|path| path.to_string()))
}

#[tauri::command]
pub fn browse_prompt_wav(app: AppHandle) -> Result<Option<String>, String> {
    Ok(app
        .dialog()
        .file()
        .add_filter("WAV audio", &["wav"])
        .blocking_pick_file()
        .map(|path| path.to_string()))
}

#[tauri::command]
pub fn browse_reference_wav(app: AppHandle) -> Result<Option<String>, String> {
    Ok(app
        .dialog()
        .file()
        .add_filter("WAV audio", &["wav"])
        .blocking_pick_file()
        .map(|path| path.to_string()))
}

#[tauri::command]
pub fn test_audio(state: State<'_, SharedAppCore>) -> Result<CommandResult, String> {
    let config = with_core(&state, |core| Ok(core.snapshot().config))?;
    let system = AudioSystem::new();
    let host = match config.audio_host.clone() {
        Some(host) => host,
        None => system.default_host_name(),
    };
    let device = match config.audio_device.clone() {
        Some(device) => device,
        None => system.default_device_name(&host).map_err(|err| err.to_string())?,
    };
    let sample_rate = 48_000;
    let samples = crate::audio::sine_with_fades(sample_rate, sample_rate as usize, 440.0, config.volume);
    let mut player = AudioPlayer::new(&host, &device, sample_rate).map_err(|err| err.to_string())?;
    player.play_blocking(samples).map_err(|err| err.to_string())?;
    Ok(CommandResult { ok: true })
}
```

- [ ] **Step 2: Register the new commands**

Modify `voxui/crates/voxui-desktop/src-tauri/src/lib.rs` command imports:

```rust
use commands::{
    browse_model_dir, browse_prompt_wav, browse_reference_wav, cancel_generation,
    cancel_model_load, discover_models, enqueue_generation, get_app_state, load_model,
    play_audio, regenerate, set_config_patch, stop_audio, test_audio,
};
```

Add to `tauri::generate_handler!`:

```rust
browse_model_dir,
browse_prompt_wav,
browse_reference_wav,
test_audio,
```

- [ ] **Step 3: Add frontend wrappers for settings commands**

Append to `voxui/crates/voxui-desktop/src/tauri_api.rs`:

```rust
pub async fn browse_model_dir() -> Result<Option<String>, String> {
    let value = invoke("browse_model_dir", JsValue::NULL).await;
    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn browse_prompt_wav() -> Result<Option<String>, String> {
    let value = invoke("browse_prompt_wav", JsValue::NULL).await;
    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn browse_reference_wav() -> Result<Option<String>, String> {
    let value = invoke("browse_reference_wav", JsValue::NULL).await;
    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn test_audio() -> Result<(), String> {
    let _ = invoke("test_audio", JsValue::NULL).await;
    Ok(())
}
```

- [ ] **Step 4: Replace settings modal with required controls**

Replace `voxui/crates/voxui-desktop/src/components/settings_modal.rs`:

```rust
use leptos::prelude::*;

use crate::i18n::Labels;

#[component]
pub fn SettingsModal(
    labels: Labels,
    open: impl Fn() -> bool + 'static + Copy,
    on_close: impl Fn() + 'static + Copy,
    on_browse_model_dir: impl Fn() + 'static + Copy,
    on_browse_prompt_wav: impl Fn() + 'static + Copy,
    on_browse_reference_wav: impl Fn() + 'static + Copy,
    on_test_audio: impl Fn() + 'static + Copy,
) -> impl IntoView {
    view! {
        <Show when=open>
            <div class="modal-backdrop">
                <section class="modal">
                    <header class="modal-header">
                        <h2>{labels.settings}</h2>
                        <button class="icon-button" on:click=move |_| on_close()>{"×"}</button>
                    </header>
                    <div class="settings-section">
                        <h3>"Models"</h3>
                        <div class="settings-row">
                            <label>"Discovery directory"<input class="setting-input" readonly /></label>
                            <button class="secondary-button" on:click=move |_| on_browse_model_dir()>"Browse"</button>
                        </div>
                    </div>
                    <div class="settings-section">
                        <h3>"Interface"</h3>
                        <label>"Language"<select class="setting-input"><option>"System"</option><option>"中文"</option><option>"English"</option></select></label>
                    </div>
                    <div class="settings-section">
                        <h3>"Inference"</h3>
                        <label>"Backend"<select class="setting-input"><option>"CPU"</option><option>"CUDA"</option></select></label>
                    </div>
                    <div class="settings-section">
                        <h3>"Audio"</h3>
                        <div class="settings-grid">
                            <label>"Driver"<select class="setting-input"></select></label>
                            <label>"Output device"<select class="setting-input"></select></label>
                            <label>"Volume"<input class="setting-input" type="range" min="0" max="100" /></label>
                            <button class="secondary-button" on:click=move |_| on_test_audio()>"Test"</button>
                        </div>
                    </div>
                    <div class="settings-section">
                        <h3>"VoxCPM generation"</h3>
                        <div class="settings-grid">
                            <label>"CFG value"<input class="setting-input" type="number" step="0.1" /></label>
                            <label>"Inference steps"<input class="setting-input" type="number" min="1" /></label>
                            <label>"Min length"<input class="setting-input" type="number" min="0" /></label>
                            <label>"Max length"<input class="setting-input" type="number" min="1" /></label>
                            <label>"Retry badcase"<input type="checkbox" /></label>
                            <label>"Retry max times"<input class="setting-input" type="number" min="1" /></label>
                            <label>"Retry ratio threshold"<input class="setting-input" type="number" step="0.1" /></label>
                        </div>
                    </div>
                    <div class="settings-section">
                        <h3>"Input"</h3>
                        <label>"Max input characters"<input class="setting-input" type="number" min="1" /></label>
                    </div>
                    <div class="settings-section">
                        <h3>"Advanced prompt/reference"</h3>
                        <div class="settings-row">
                            <label>"Prompt WAV"<input class="setting-input" readonly /></label>
                            <button class="secondary-button" on:click=move |_| on_browse_prompt_wav()>"Browse"</button>
                        </div>
                        <label>"Prompt text"<textarea class="setting-input"></textarea></label>
                        <div class="settings-row">
                            <label>"Reference WAV"<input class="setting-input" readonly /></label>
                            <button class="secondary-button" on:click=move |_| on_browse_reference_wav()>"Browse"</button>
                        </div>
                    </div>
                </section>
            </div>
        </Show>
    }
}
```

- [ ] **Step 5: Wire settings callbacks from app**

Update the `SettingsModal` call in `voxui/crates/voxui-desktop/src/app.rs`:

```rust
<SettingsModal
    labels=labels
    open=move || settings_open.get()
    on_close=move || set_settings_open.set(false)
    on_browse_model_dir=move || {
        spawn_local(async move {
            let _ = crate::tauri_api::browse_model_dir().await;
        });
    }
    on_browse_prompt_wav=move || {
        spawn_local(async move {
            let _ = crate::tauri_api::browse_prompt_wav().await;
        });
    }
    on_browse_reference_wav=move || {
        spawn_local(async move {
            let _ = crate::tauri_api::browse_reference_wav().await;
        });
    }
    on_test_audio=move || {
        spawn_local(async move {
            let _ = crate::tauri_api::test_audio().await;
        });
    }
/>
```

- [ ] **Step 6: Add event listener wrappers**

Append to `voxui/crates/voxui-desktop/src/tauri_api.rs`:

```rust
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "event"])]
    async fn listen(event: &str, handler: &Closure<dyn Fn(JsValue)>) -> JsValue;
}
```

In `app.rs`, install listeners after the initial state load:

```rust
spawn_local(async move {
    if let Ok(next) = crate::tauri_api::get_app_state().await {
        set_snapshot.set(Some(next));
    }
});
```

Use the existing `get_app_state` refresh after generation and playback actions. This keeps frontend state consistent with backend state in this implementation pass.

- [ ] **Step 7: Verify required command set compiles**

Run:

```powershell
cd voxui
cargo check -p voxui-desktop
cd crates\voxui-desktop
trunk build
```

Expected: both commands pass.

- [ ] **Step 8: Commit required settings and command coverage**

Run:

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src/commands.rs voxui/crates/voxui-desktop/src-tauri/src/lib.rs voxui/crates/voxui-desktop/src/tauri_api.rs voxui/crates/voxui-desktop/src/app.rs voxui/crates/voxui-desktop/src/components/settings_modal.rs
git commit -m "feat: complete desktop settings commands"
```

---

## Final Verification Checklist

- [ ] `cd voxui; cargo test -p voxui-desktop`
- [ ] `cd voxui; cargo check -p voxui-desktop`
- [ ] `cd voxui; cargo check -p voxui-desktop --features cuda`
- [ ] `cd voxui; cargo test -p voxui-inference`
- [ ] `cd voxui/crates/voxui-desktop; trunk build`
- [ ] `cd voxui/crates/voxui-desktop/src-tauri; cargo tauri dev`

Manual acceptance checks:

- [ ] App title appears as `焓言焓语` / `AhanSays`.
- [ ] Model root can discover base and LoRA-expanded choices.
- [ ] Loading is explicit and does not auto-load on startup.
- [ ] Loading progress modal appears and can be canceled.
- [ ] Text generation requires a loaded model.
- [ ] History items show queued/generating/ready/error states.
- [ ] Finished items expose play/stop/regenerate controls.
- [ ] Settings expose language, backend, audio host/device, volume, generation parameters, and max input characters.
- [ ] Audio test plays a faded sine wave through the selected output device.
