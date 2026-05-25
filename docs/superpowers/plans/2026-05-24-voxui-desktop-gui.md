# VoxUI Desktop GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new end-to-end Tauri/Svelte desktop app at `voxui/crates/voxui-desktop` for VoxCPM model loading, sequential TTS generation, WebAudio playback, settings, localization, and readable exit history logging.

**Architecture:** The Tauri backend owns model discovery, settings persistence, one loaded `VoxCPMEngine`, cancellation flags, one sequential generation queue, and event emission. The Svelte frontend owns the fixed app shell, settings modal, history cards, i18n, WebAudio playback, sine test, volume, and browser output-device fallback behavior. Rust sends mono PCM `f32` chunks plus sample-rate metadata; frontend plays all generated audio through WebAudio.

**Tech Stack:** Rust 2021, Tauri 2, Svelte 5, TypeScript, Vite, Tailwind CSS, DaisyUI, `voxui-inference`, `candle-core`, `serde`, `tokio`, `anyhow`, WebAudio API

---

## File Structure

Create and modify these files.

Workspace and docs:

- Modify: `voxui/Cargo.toml` to add the desktop Tauri crate as a workspace member.
- Modify: `README.txt` to replace old desktop commands with the new Tauri/Svelte commands.

Desktop frontend root:

- Create: `voxui/crates/voxui-desktop/package.json` for Vite/Svelte/Tauri scripts.
- Create: `voxui/crates/voxui-desktop/package-lock.json` through `npm install`.
- Create: `voxui/crates/voxui-desktop/index.html` as Vite entry.
- Create: `voxui/crates/voxui-desktop/vite.config.ts` for Svelte and Tauri dev server settings.
- Create: `voxui/crates/voxui-desktop/tsconfig.json` and `tsconfig.node.json`.
- Create: `voxui/crates/voxui-desktop/svelte.config.js`.
- Create: `voxui/crates/voxui-desktop/src/app.css` for Tailwind and global layout.
- Create: `voxui/crates/voxui-desktop/src/main.ts`.
- Create: `voxui/crates/voxui-desktop/src/App.svelte` for the main app shell.

Frontend state, services, and types:

- Create: `voxui/crates/voxui-desktop/src/lib/types.ts` for backend payload and UI types.
- Create: `voxui/crates/voxui-desktop/src/lib/i18n.ts` for Chinese/English labels.
- Create: `voxui/crates/voxui-desktop/src/lib/backend.ts` for typed Tauri invoke/listen helpers.
- Create: `voxui/crates/voxui-desktop/src/lib/state.svelte.ts` for shared Svelte 5 rune state.
- Create: `voxui/crates/voxui-desktop/src/lib/audio.ts` for WebAudio playback, sine test, volume, device selection, and fallback detection.
- Create: `voxui/crates/voxui-desktop/src/lib/format.ts` for labels, durations, and dropdown formatting.

Frontend components:

- Create: `voxui/crates/voxui-desktop/src/components/NavBar.svelte`.
- Create: `voxui/crates/voxui-desktop/src/components/HistoryList.svelte`.
- Create: `voxui/crates/voxui-desktop/src/components/HistoryCard.svelte`.
- Create: `voxui/crates/voxui-desktop/src/components/InputBar.svelte`.
- Create: `voxui/crates/voxui-desktop/src/components/SettingsModal.svelte`.
- Create: `voxui/crates/voxui-desktop/src/components/LoadProgressModal.svelte`.

Tauri backend crate:

- Create: `voxui/crates/voxui-desktop/src-tauri/Cargo.toml`.
- Create: `voxui/crates/voxui-desktop/src-tauri/build.rs`.
- Create: `voxui/crates/voxui-desktop/src-tauri/tauri.conf.json`.
- Create: `voxui/crates/voxui-desktop/src-tauri/capabilities/default.json`.
- Create: `voxui/crates/voxui-desktop/src-tauri/src/main.rs`.
- Create: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`.
- Create: `voxui/crates/voxui-desktop/src-tauri/src/types.rs` for serializable request/event/settings types.
- Create: `voxui/crates/voxui-desktop/src-tauri/src/settings.rs` for defaults, load, save, and tests.
- Create: `voxui/crates/voxui-desktop/src-tauri/src/models.rs` for discovery and tests.
- Create: `voxui/crates/voxui-desktop/src-tauri/src/history.rs` for readable session log buffering and tests.
- Create: `voxui/crates/voxui-desktop/src-tauri/src/app_state.rs` for shared app state and queue state.
- Create: `voxui/crates/voxui-desktop/src-tauri/src/engine_runner.rs` for model loading and generation execution.
- Create: `voxui/crates/voxui-desktop/src-tauri/src/commands.rs` for Tauri commands.

Backend tests live next to Rust modules using `#[cfg(test)]`. Frontend verification for this plan is `npm run check` and `npm run build`.

---

### Task 1: Scaffold Tauri/Svelte Desktop Crate

**Files:**
- Create: `voxui/crates/voxui-desktop/**`
- Modify: `voxui/Cargo.toml`

- [ ] **Step 1: Scaffold the Svelte TypeScript Tauri app**

Run from repository root:

```powershell
Test-Path -LiteralPath "voxui\crates"
```

Expected: `True`.

Run:

```powershell
npm create tauri-app@latest "voxui/crates/voxui-desktop" -- --template svelte-ts --manager npm
```

Expected: `voxui/crates/voxui-desktop` exists with `src`, `src-tauri`, `package.json`, and Vite config files.

- [ ] **Step 2: Install frontend dependencies**

Run:

```powershell
npm install
```

Working directory: `voxui/crates/voxui-desktop`.

Expected: `node_modules` exists and `package-lock.json` is created.

- [ ] **Step 3: Add Tailwind and DaisyUI dependencies**

Run:

```powershell
npm install -D tailwindcss @tailwindcss/vite daisyui svelte-check typescript
```

Working directory: `voxui/crates/voxui-desktop`.

Expected: dependencies are added to `package.json`.

- [ ] **Step 4: Configure Vite for Tailwind**

Replace `voxui/crates/voxui-desktop/vite.config.ts` with:

```ts
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';

const host = process.env.TAURI_DEV_HOST;

export default defineConfig(async () => ({
  plugins: [svelte(), tailwindcss()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: 'ws',
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      ignored: ['**/src-tauri/**'],
    },
  },
}));
```

- [ ] **Step 5: Configure Tailwind and DaisyUI CSS**

Replace `voxui/crates/voxui-desktop/src/app.css` with:

```css
@import "tailwindcss";
@plugin "daisyui" {
  themes: light --default, dark --prefersdark;
}

:root {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color-scheme: light dark;
}

html,
body,
#app {
  height: 100%;
  margin: 0;
}

body {
  overflow: hidden;
}

button,
select,
textarea,
input {
  font: inherit;
}
```

- [ ] **Step 6: Add desktop workspace member**

Modify `voxui/Cargo.toml` so `[workspace].members` is:

```toml
[workspace]
members = [
    "crates/voxui-gguf",
    "crates/voxui-inference",
    "crates/voxui-audio",
    "crates/voxui-cli",
    "crates/voxui-desktop/src-tauri",
]
resolver = "2"
```

- [ ] **Step 7: Verify scaffold builds**

Run:

```powershell
npm run build
```

Working directory: `voxui/crates/voxui-desktop`.

Expected: Vite produces `dist` successfully.

- [ ] **Step 8: Commit scaffold**

```powershell
git add voxui/Cargo.toml voxui/crates/voxui-desktop
git commit -m "feat(desktop): scaffold tauri svelte app"
```

---

### Task 2: Add Backend Types And Settings

**Files:**
- Create: `voxui/crates/voxui-desktop/src-tauri/src/types.rs`
- Create: `voxui/crates/voxui-desktop/src-tauri/src/settings.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/Cargo.toml`

- [ ] **Step 1: Add backend dependencies**

Ensure `voxui/crates/voxui-desktop/src-tauri/Cargo.toml` contains these dependencies:

```toml
[dependencies]
tauri = { version = "2", features = [] }
tauri-plugin-dialog = "2"
tauri-plugin-opener = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow.workspace = true
thiserror.workspace = true
tokio = { version = "1", features = ["sync", "rt-multi-thread", "macros", "time"] }
tracing = "0.1"
dirs = "5"
candle-core.workspace = true
voxui-inference = { path = "../../voxui-inference" }
```

Keep the existing `tauri-build` build dependency generated by the scaffold.

- [ ] **Step 2: Create serializable types**

Create `voxui/crates/voxui-desktop/src-tauri/src/types.rs`:

```rust
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LanguageMode {
    System,
    English,
    Chinese,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum InferenceBackend {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationSettings {
    pub cfg_value: f32,
    pub inference_timesteps: usize,
    pub min_len: usize,
    pub max_len: usize,
    pub retry_badcase: bool,
    pub retry_badcase_max_times: usize,
    pub retry_badcase_ratio_threshold: f32,
    pub max_input_chars: usize,
    pub streaming: bool,
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
            max_input_chars: 500,
            streaming: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    pub model_dir: PathBuf,
    pub language: LanguageMode,
    pub backend: InferenceBackend,
    pub generation: GenerationSettings,
    pub volume: f32,
    pub output_device_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    pub id: String,
    pub label: String,
    pub model_name: String,
    pub model_dir: PathBuf,
    pub lora_path: Option<PathBuf>,
    pub lora_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCatalog {
    pub model_dir: PathBuf,
    pub entries: Vec<ModelEntry>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadModelRequest {
    pub entry_id: String,
    pub backend: InferenceBackend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateRequest {
    pub text: String,
    pub settings: GenerationSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSnapshot {
    pub settings: AppSettings,
    pub catalog: ModelCatalog,
    pub system_language: String,
}
```

- [ ] **Step 3: Create settings module with defaults and persistence**

Create `voxui/crates/voxui-desktop/src-tauri/src/settings.rs`:

```rust
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::types::{AppSettings, GenerationSettings, InferenceBackend, LanguageMode};

pub fn default_model_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn default_settings() -> AppSettings {
    AppSettings {
        model_dir: default_model_dir(),
        language: LanguageMode::System,
        backend: InferenceBackend::Cpu,
        generation: GenerationSettings::default(),
        volume: 0.8,
        output_device_id: None,
    }
}

pub fn settings_path(app_config_dir: &Path) -> PathBuf {
    app_config_dir.join("settings.json")
}

pub fn load_settings(app_config_dir: &Path) -> Result<AppSettings> {
    let path = settings_path(app_config_dir);
    if !path.exists() {
        return Ok(default_settings());
    }
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

pub fn save_settings(app_config_dir: &Path, settings: &AppSettings) -> Result<()> {
    fs::create_dir_all(app_config_dir)
        .with_context(|| format!("create {}", app_config_dir.display()))?;
    let path = settings_path(app_config_dir);
    let text = serde_json::to_string_pretty(settings)?;
    fs::write(&path, text).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_generation_settings_match_engine_defaults() {
        let generation = GenerationSettings::default();
        assert_eq!(generation.cfg_value, 2.0);
        assert_eq!(generation.inference_timesteps, 10);
        assert_eq!(generation.min_len, 2);
        assert_eq!(generation.max_len, 2000);
        assert!(generation.retry_badcase);
        assert_eq!(generation.retry_badcase_max_times, 3);
        assert_eq!(generation.retry_badcase_ratio_threshold, 6.0);
        assert_eq!(generation.max_input_chars, 500);
        assert!(generation.streaming);
    }

    #[test]
    fn save_then_load_round_trips_settings() {
        let dir = std::env::temp_dir().join(format!(
            "voxui_desktop_settings_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let mut settings = default_settings();
        settings.language = LanguageMode::Chinese;
        settings.backend = InferenceBackend::Cuda;
        settings.volume = 0.42;
        save_settings(&dir, &settings).unwrap();
        let loaded = load_settings(&dir).unwrap();
        assert_eq!(loaded.language, LanguageMode::Chinese);
        assert_eq!(loaded.backend, InferenceBackend::Cuda);
        assert_eq!(loaded.volume, 0.42);
        let _ = fs::remove_dir_all(&dir);
    }
}
```

- [ ] **Step 4: Wire modules in lib.rs**

Make `voxui/crates/voxui-desktop/src-tauri/src/lib.rs` begin with:

```rust
mod settings;
mod types;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .run(tauri::generate_context!())
        .expect("error while running VoxUI desktop application");
}
```

- [ ] **Step 5: Run backend tests for settings**

Run:

```powershell
cargo test -p voxui-desktop settings
```

Working directory: `voxui`.

Expected: two settings tests pass.

- [ ] **Step 6: Commit backend settings types**

```powershell
git add voxui/crates/voxui-desktop/src-tauri/Cargo.toml voxui/crates/voxui-desktop/src-tauri/src/lib.rs voxui/crates/voxui-desktop/src-tauri/src/types.rs voxui/crates/voxui-desktop/src-tauri/src/settings.rs
git commit -m "feat(desktop): add settings and shared backend types"
```

---

### Task 3: Add Model Discovery

**Files:**
- Create: `voxui/crates/voxui-desktop/src-tauri/src/models.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`

- [ ] **Step 1: Create model discovery implementation**

Create `voxui/crates/voxui-desktop/src-tauri/src/models.rs`:

```rust
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::types::{ModelCatalog, ModelEntry};

pub fn discover_models(model_dir: &Path) -> Result<ModelCatalog> {
    let mut entries = Vec::new();
    let mut warnings = Vec::new();

    if !model_dir.exists() {
        return Ok(ModelCatalog {
            model_dir: model_dir.to_path_buf(),
            entries,
            warnings: vec![format!("model directory does not exist: {}", model_dir.display())],
        });
    }

    for child in fs::read_dir(model_dir).with_context(|| format!("read {}", model_dir.display()))? {
        let child = child?;
        let path = child.path();
        if !path.is_dir() {
            continue;
        }
        let Some(model_name) = path.file_name().and_then(|name| name.to_str()).map(str::to_owned) else {
            warnings.push(format!("ignored model folder with non-UTF8 name: {}", path.display()));
            continue;
        };
        if !path.join("model.gguf").is_file() {
            warnings.push(format!("ignored {model_name}: missing model.gguf"));
            continue;
        }

        entries.push(ModelEntry {
            id: model_name.clone(),
            label: model_name.clone(),
            model_name: model_name.clone(),
            model_dir: path.clone(),
            lora_path: None,
            lora_name: None,
        });

        for lora in sorted_lora_files(&path)? {
            let lora_stem = lora
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("lora")
                .to_owned();
            entries.push(ModelEntry {
                id: format!("{model_name}|{lora_stem}"),
                label: format!("{model_name} | {lora_stem}"),
                model_name: model_name.clone(),
                model_dir: path.clone(),
                lora_path: Some(lora),
                lora_name: Some(lora_stem),
            });
        }
    }

    entries.sort_by(|a, b| a.label.to_lowercase().cmp(&b.label.to_lowercase()));
    Ok(ModelCatalog {
        model_dir: model_dir.to_path_buf(),
        entries,
        warnings,
    })
}

fn sorted_lora_files(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(model_dir).with_context(|| format!("read {}", model_dir.display()))? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.eq_ignore_ascii_case("model.gguf") {
            continue;
        }
        let is_gguf = path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
        if is_gguf {
            files.push(path);
        }
    }
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("voxui_desktop_models_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_model_dir_returns_empty_catalog_with_warning() {
        let dir = std::env::temp_dir().join("voxui_desktop_missing_models_dir");
        let _ = fs::remove_dir_all(&dir);
        let catalog = discover_models(&dir).unwrap();
        assert!(catalog.entries.is_empty());
        assert_eq!(catalog.warnings.len(), 1);
    }

    #[test]
    fn discovers_base_model_and_lora_entries() {
        let root = make_dir("with_lora");
        let model = root.join("voxcpm2-fp16");
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("model.gguf"), b"base").unwrap();
        fs::write(model.join("lora_a1.gguf"), b"lora").unwrap();
        fs::write(model.join("lora_a2.GGUF"), b"lora").unwrap();
        fs::write(model.join("notes.txt"), b"ignored").unwrap();

        let catalog = discover_models(&root).unwrap();
        let labels: Vec<_> = catalog.entries.iter().map(|entry| entry.label.as_str()).collect();
        assert_eq!(labels, vec!["voxcpm2-fp16", "voxcpm2-fp16 | lora_a1", "voxcpm2-fp16 | lora_a2"]);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn ignores_folder_without_model_gguf() {
        let root = make_dir("invalid");
        let invalid = root.join("not-a-model");
        fs::create_dir_all(&invalid).unwrap();
        fs::write(invalid.join("lora.gguf"), b"lora").unwrap();

        let catalog = discover_models(&root).unwrap();
        assert!(catalog.entries.is_empty());
        assert_eq!(catalog.warnings, vec!["ignored not-a-model: missing model.gguf"]);
        let _ = fs::remove_dir_all(&root);
    }
}
```

- [ ] **Step 2: Register models module**

Add the module to `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`:

```rust
mod models;
mod settings;
mod types;
```

- [ ] **Step 3: Run discovery tests**

```powershell
cargo test -p voxui-desktop models
```

Working directory: `voxui`.

Expected: three model discovery tests pass.

- [ ] **Step 4: Commit model discovery**

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src/lib.rs voxui/crates/voxui-desktop/src-tauri/src/models.rs
git commit -m "feat(desktop): discover models and lora variants"
```

---

### Task 4: Add History Log And App State

**Files:**
- Create: `voxui/crates/voxui-desktop/src-tauri/src/history.rs`
- Create: `voxui/crates/voxui-desktop/src-tauri/src/app_state.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`

- [ ] **Step 1: Create readable history log module**

Create `voxui/crates/voxui-desktop/src-tauri/src/history.rs`:

```rust
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use crate::types::{GenerationSettings, InferenceBackend, ModelEntry};

#[derive(Debug, Clone)]
pub struct HistoryRecord {
    pub timestamp: String,
    pub text: String,
    pub model: String,
    pub lora: Option<String>,
    pub backend: InferenceBackend,
    pub streaming: bool,
    pub settings: GenerationSettings,
    pub status: String,
    pub elapsed_ms: u128,
    pub error: Option<String>,
}

#[derive(Default)]
pub struct HistoryLog {
    records: Vec<HistoryRecord>,
}

impl HistoryLog {
    pub fn push(&mut self, record: HistoryRecord) {
        self.records.push(record);
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn write_next_to_exe(&self) -> Result<Option<PathBuf>> {
        if self.records.is_empty() {
            return Ok(None);
        }
        let exe = std::env::current_exe().context("resolve current executable")?;
        let dir = exe.parent().context("resolve executable directory")?;
        let file_name = format!("ahan-says-history-{}.log", compact_timestamp());
        let path = dir.join(file_name);
        self.write_to_path(&path)?;
        Ok(Some(path))
    }

    pub fn write_to_path(&self, path: &Path) -> Result<()> {
        let mut text = String::new();
        text.push_str("AhanSays / 焓言焓语 generation history\n");
        text.push_str("=====================================\n\n");
        for (index, record) in self.records.iter().enumerate() {
            text.push_str(&format!("#{} {}\n", index + 1, record.timestamp));
            text.push_str(&format!("Status: {}\n", record.status));
            text.push_str(&format!("Model: {}\n", record.model));
            text.push_str(&format!("LoRA: {}\n", record.lora.as_deref().unwrap_or("none")));
            text.push_str(&format!("Backend: {:?}\n", record.backend));
            text.push_str(&format!("Streaming: {}\n", record.streaming));
            text.push_str(&format!("Elapsed: {} ms\n", record.elapsed_ms));
            text.push_str(&format!("Params: cfg={} steps={} min_len={} max_len={} retry={} retry_max={} retry_ratio={}\n",
                record.settings.cfg_value,
                record.settings.inference_timesteps,
                record.settings.min_len,
                record.settings.max_len,
                record.settings.retry_badcase,
                record.settings.retry_badcase_max_times,
                record.settings.retry_badcase_ratio_threshold,
            ));
            if let Some(error) = &record.error {
                text.push_str(&format!("Error: {error}\n"));
            }
            text.push_str("Text:\n");
            text.push_str(&record.text);
            text.push_str("\n\n");
        }
        fs::write(path, text).with_context(|| format!("write {}", path.display()))
    }
}

pub fn record_from_result(
    text: String,
    model: &ModelEntry,
    backend: InferenceBackend,
    settings: GenerationSettings,
    status: impl Into<String>,
    elapsed_ms: u128,
    error: Option<String>,
) -> HistoryRecord {
    HistoryRecord {
        timestamp: readable_timestamp(),
        text,
        model: model.model_name.clone(),
        lora: model.lora_name.clone(),
        backend,
        streaming: settings.streaming,
        settings,
        status: status.into(),
        elapsed_ms,
        error,
    }
}

fn compact_timestamp() -> String {
    seconds_since_epoch().to_string()
}

fn readable_timestamp() -> String {
    format!("unix:{}", seconds_since_epoch())
}

fn seconds_since_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_readable_history_log() {
        let dir = std::env::temp_dir().join(format!("voxui_desktop_history_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.log");
        let model = ModelEntry {
            id: "m".into(),
            label: "model | lora".into(),
            model_name: "model".into(),
            model_dir: dir.clone(),
            lora_path: Some(dir.join("lora.gguf")),
            lora_name: Some("lora".into()),
        };
        let mut log = HistoryLog::default();
        log.push(record_from_result(
            "hello".into(),
            &model,
            InferenceBackend::Cpu,
            GenerationSettings::default(),
            "completed",
            123,
            None,
        ));
        log.write_to_path(&path).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("AhanSays"));
        assert!(text.contains("Model: model"));
        assert!(text.contains("LoRA: lora"));
        assert!(text.contains("hello"));
        let _ = fs::remove_dir_all(&dir);
    }
}
```

- [ ] **Step 2: Create shared app state shell**

Create `voxui/crates/voxui-desktop/src-tauri/src/app_state.rs`:

```rust
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use voxui_inference::VoxCPMEngine;

use crate::history::HistoryLog;
use crate::types::{AppSettings, ModelCatalog, ModelEntry};

pub struct LoadedModel {
    pub entry: ModelEntry,
    pub engine: VoxCPMEngine,
}

#[derive(Default)]
pub struct CancellationState {
    pub load: Option<Arc<AtomicBool>>,
    pub generation: Option<Arc<AtomicBool>>,
}

pub struct DesktopState {
    pub settings: Mutex<AppSettings>,
    pub catalog: Mutex<ModelCatalog>,
    pub loaded_model: Mutex<Option<LoadedModel>>,
    pub cancellation: Mutex<CancellationState>,
    pub history: Mutex<HistoryLog>,
}

impl DesktopState {
    pub fn new(settings: AppSettings, catalog: ModelCatalog) -> Self {
        Self {
            settings: Mutex::new(settings),
            catalog: Mutex::new(catalog),
            loaded_model: Mutex::new(None),
            cancellation: Mutex::new(CancellationState::default()),
            history: Mutex::new(HistoryLog::default()),
        }
    }
}
```

- [ ] **Step 3: Register modules**

Update `voxui/crates/voxui-desktop/src-tauri/src/lib.rs` module list:

```rust
mod app_state;
mod history;
mod models;
mod settings;
mod types;
```

- [ ] **Step 4: Run history tests**

```powershell
cargo test -p voxui-desktop history
```

Working directory: `voxui`.

Expected: readable history test passes.

- [ ] **Step 5: Commit history and app state**

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src/lib.rs voxui/crates/voxui-desktop/src-tauri/src/history.rs voxui/crates/voxui-desktop/src-tauri/src/app_state.rs
git commit -m "feat(desktop): add app state and history log"
```

---

### Task 5: Add Tauri Commands And Startup Wiring

**Files:**
- Create: `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/types.rs`

- [ ] **Step 1: Add command response type**

Append to `types.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandOk {
    pub ok: bool,
}
```

- [ ] **Step 2: Create commands module**

Create `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`:

```rust
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;

use crate::app_state::DesktopState;
use crate::models::discover_models;
use crate::settings::{save_settings, settings_path};
use crate::types::{AppSettings, AppSnapshot, CommandOk, ModelCatalog};

#[tauri::command]
pub fn get_snapshot(state: State<'_, DesktopState>) -> Result<AppSnapshot, String> {
    let settings = state.settings.lock().map_err(|_| "settings lock poisoned")?.clone();
    let catalog = state.catalog.lock().map_err(|_| "catalog lock poisoned")?.clone();
    Ok(AppSnapshot {
        settings,
        catalog,
        system_language: detect_system_language(),
    })
}

#[tauri::command]
pub fn save_app_settings(
    app: AppHandle,
    state: State<'_, DesktopState>,
    settings: AppSettings,
) -> Result<CommandOk, String> {
    let config_dir = app
        .path()
        .app_config_dir()
        .map_err(|err| format!("resolve config dir: {err}"))?;
    save_settings(&config_dir, &settings).map_err(|err| err.to_string())?;
    *state.settings.lock().map_err(|_| "settings lock poisoned")? = settings;
    Ok(CommandOk { ok: true })
}

#[tauri::command]
pub fn rescan_models(state: State<'_, DesktopState>) -> Result<ModelCatalog, String> {
    let settings = state.settings.lock().map_err(|_| "settings lock poisoned")?.clone();
    let catalog = discover_models(&settings.model_dir).map_err(|err| err.to_string())?;
    *state.catalog.lock().map_err(|_| "catalog lock poisoned")? = catalog.clone();
    Ok(catalog)
}

#[tauri::command]
pub async fn browse_model_dir(app: AppHandle) -> Result<Option<String>, String> {
    let selected = app.dialog().file().blocking_pick_folder();
    Ok(selected.map(|path| path.to_string()))
}

pub fn detect_system_language() -> String {
    std::env::var("LANG")
        .or_else(|_| std::env::var("LANGUAGE"))
        .or_else(|_| std::env::var("LC_ALL"))
        .unwrap_or_else(|_| "en".to_string())
}

pub fn log_settings_path(app: &AppHandle) {
    if let Ok(config_dir) = app.path().app_config_dir() {
        tracing::debug!("desktop settings path: {}", settings_path(&config_dir).display());
    }
}
```

- [ ] **Step 3: Wire startup state and commands**

Replace `voxui/crates/voxui-desktop/src-tauri/src/lib.rs` with:

```rust
mod app_state;
mod commands;
mod history;
mod models;
mod settings;
mod types;

use app_state::DesktopState;
use commands::{browse_model_dir, get_snapshot, rescan_models, save_app_settings};
use models::discover_models;
use settings::{default_settings, load_settings};
use tauri::{Manager, WindowEvent};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            commands::log_settings_path(&app.handle());
            let config_dir = app.path().app_config_dir()?;
            let settings = load_settings(&config_dir).unwrap_or_else(|err| {
                tracing::warn!("failed to load settings, using defaults: {err}");
                default_settings()
            });
            let catalog = discover_models(&settings.model_dir).unwrap_or_else(|err| {
                tracing::warn!("failed to discover models: {err}");
                crate::types::ModelCatalog {
                    model_dir: settings.model_dir.clone(),
                    entries: Vec::new(),
                    warnings: vec![err.to_string()],
                }
            });
            app.manage(DesktopState::new(settings, catalog));
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, WindowEvent::CloseRequested { .. }) {
                let state = window.state::<DesktopState>();
                if let Ok(history) = state.history.lock() {
                    if let Err(err) = history.write_next_to_exe() {
                        tracing::warn!("failed to write generation history: {err}");
                    }
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_snapshot,
            save_app_settings,
            rescan_models,
            browse_model_dir
        ])
        .run(tauri::generate_context!())
        .expect("error while running VoxUI desktop application");
}
```

- [ ] **Step 4: Check backend compiles**

```powershell
cargo check -p voxui-desktop
```

Working directory: `voxui`.

Expected: desktop backend compiles.

- [ ] **Step 5: Commit startup commands**

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src/lib.rs voxui/crates/voxui-desktop/src-tauri/src/commands.rs voxui/crates/voxui-desktop/src-tauri/src/types.rs
git commit -m "feat(desktop): add startup commands and settings persistence"
```

---

### Task 6: Add Engine Runner For Load And Generation Events

**Files:**
- Create: `voxui/crates/voxui-desktop/src-tauri/src/engine_runner.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/types.rs`

- [ ] **Step 1: Add event payload types**

Append to `types.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadProgressEvent {
    pub phase: String,
    pub current: usize,
    pub total: usize,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadedModelEvent {
    pub entry: ModelEntry,
    pub architecture: String,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationQueuedEvent {
    pub job_id: u64,
    pub text: String,
    pub entry: ModelEntry,
    pub settings: GenerationSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationProgressEvent {
    pub job_id: u64,
    pub current: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioChunkEvent {
    pub job_id: u64,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub is_final: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobFinishedEvent {
    pub job_id: u64,
    pub status: String,
    pub error: Option<String>,
}
```

- [ ] **Step 2: Create engine runner implementation**

Create `voxui/crates/voxui-desktop/src-tauri/src/engine_runner.rs`:

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::Device;
use tauri::{AppHandle, Emitter, State};
use voxui_inference::{SynthesisRequest, VoxCPMEngine};

use crate::app_state::{DesktopState, LoadedModel};
use crate::history::record_from_result;
use crate::types::{
    AudioChunkEvent, GenerateRequest, GenerationProgressEvent, GenerationQueuedEvent,
    InferenceBackend, JobFinishedEvent, LoadModelRequest, LoadProgressEvent, LoadedModelEvent,
};

pub fn cancel_load(state: &DesktopState) {
    if let Ok(cancellation) = state.cancellation.lock() {
        if let Some(flag) = &cancellation.load {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

pub fn cancel_generation(state: &DesktopState) {
    if let Ok(cancellation) = state.cancellation.lock() {
        if let Some(flag) = &cancellation.generation {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

pub fn spawn_load_model(
    app: AppHandle,
    state: State<'_, Arc<DesktopState>>,
    request: LoadModelRequest,
) -> Result<(), String> {
    let shared = state.inner().clone();
    let entry = {
        let catalog = shared.catalog.lock().map_err(|_| "catalog lock poisoned")?;
        catalog
            .entries
            .iter()
            .find(|entry| entry.id == request.entry_id)
            .cloned()
            .ok_or_else(|| format!("model entry not found: {}", request.entry_id))?
    };

    cancel_load(&shared);
    cancel_generation(&shared);
    *shared.loaded_model.lock().map_err(|_| "loaded model lock poisoned")? = None;

    let cancel = Arc::new(AtomicBool::new(false));
    shared.cancellation.lock().map_err(|_| "cancellation lock poisoned")?.load = Some(cancel.clone());
    std::thread::spawn(move || {
        if let Err(err) = load_model_inner(&app, &shared, entry, request.backend, cancel) {
            let _ = app.emit("model-load-error", err.to_string());
        }
    });
    Ok(())
}

fn load_model_inner(
    app: &AppHandle,
    state: &DesktopState,
    entry: crate::types::ModelEntry,
    backend: InferenceBackend,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    let device = match backend {
        InferenceBackend::Cpu => Device::Cpu,
        InferenceBackend::Cuda => Device::new_cuda(0).context("CUDA device not available")?,
    };
    app.emit(
        "model-load-progress",
        LoadProgressEvent {
            phase: "component".into(),
            current: 0,
            total: 6,
            message: "Starting model load".into(),
        },
    )?;
    let mut engine = VoxCPMEngine::load_with_progress(
        &entry.model_dir,
        device,
        |current, total| {
            let _ = app.emit(
                "model-load-progress",
                LoadProgressEvent {
                    phase: "component".into(),
                    current,
                    total,
                    message: format!("Loading component {current}/{total}"),
                },
            );
        },
        Some(cancel.as_ref()),
    )?;
    if cancel.load(Ordering::SeqCst) {
        bail!("model loading cancelled");
    }
    if let Some(lora_path) = &entry.lora_path {
        engine.load_lora(lora_path).context("failed to load LoRA adapter")?;
    }
    let event = LoadedModelEvent {
        entry: entry.clone(),
        architecture: engine.architecture().to_string(),
        sample_rate: engine.sample_rate(),
    };
    *state.loaded_model.lock().map_err(|_| anyhow::anyhow!("loaded model lock poisoned"))? = Some(LoadedModel { entry, engine });
    state.cancellation.lock().map_err(|_| anyhow::anyhow!("cancellation lock poisoned"))?.load = None;
    app.emit("model-loaded", event)?;
    Ok(())
}
```

- [ ] **Step 3: Register managed state as `Arc<DesktopState>`**

Replace `DesktopState::new` usage in `lib.rs` setup with:

```rust
app.manage(std::sync::Arc::new(DesktopState::new(settings, catalog)));
```

Change command signatures that currently use `State<'_, DesktopState>` to `State<'_, std::sync::Arc<DesktopState>>`. For example, `get_snapshot` becomes:

```rust
pub fn get_snapshot(
    state: State<'_, Arc<DesktopState>>,
) -> Result<AppSnapshot, String> {
    let settings = state.settings.lock().map_err(|_| "settings lock poisoned")?.clone();
    let catalog = state.catalog.lock().map_err(|_| "catalog lock poisoned")?.clone();
    Ok(AppSnapshot {
        settings,
        catalog,
        system_language: detect_system_language(),
    })
}
```

- [ ] **Step 4: Add generation command logic**

Append to `engine_runner.rs`:

```rust
pub fn spawn_generate(app: AppHandle, state: State<'_, Arc<DesktopState>>, request: GenerateRequest) -> Result<u64, String> {
    let shared = state.inner().clone();
    let job_id = next_job_id();
    let cancel = Arc::new(AtomicBool::new(false));
    shared.cancellation.lock().map_err(|_| "cancellation lock poisoned")?.generation = Some(cancel.clone());
    std::thread::spawn(move || {
        if let Err(err) = generate_inner(&app, &shared, job_id, request, cancel) {
            let _ = app.emit("generation-finished", JobFinishedEvent {
                job_id,
                status: "failed".into(),
                error: Some(err.to_string()),
            });
        }
    });
    Ok(job_id)
}

fn generate_inner(
    app: &AppHandle,
    state: &DesktopState,
    job_id: u64,
    request: GenerateRequest,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    if request.text.trim().is_empty() {
        bail!("text must not be empty");
    }
    if request.text.chars().count() > request.settings.max_input_chars {
        bail!("text exceeds max input characters");
    }
    let backend = state.settings.lock().map_err(|_| anyhow::anyhow!("settings lock poisoned"))?.backend;
    let started = Instant::now();
    let mut guard = state.loaded_model.lock().map_err(|_| anyhow::anyhow!("loaded model lock poisoned"))?;
    let loaded = guard.as_mut().context("no model loaded")?;
    let entry = loaded.entry.clone();
    app.emit("generation-queued", GenerationQueuedEvent {
        job_id,
        text: request.text.clone(),
        entry: entry.clone(),
        settings: request.settings.clone(),
    })?;
    let synthesis = SynthesisRequest {
        text: request.text.clone(),
        cfg_value: request.settings.cfg_value,
        inference_timesteps: request.settings.inference_timesteps,
        min_len: request.settings.min_len,
        max_len: request.settings.max_len,
        retry_badcase: if request.settings.streaming { false } else { request.settings.retry_badcase },
        retry_badcase_max_times: request.settings.retry_badcase_max_times,
        retry_badcase_ratio_threshold: request.settings.retry_badcase_ratio_threshold,
        ..Default::default()
    };
    let result = if request.settings.streaming {
        loaded.engine.generate_streaming_cancellable(
            synthesis,
            |chunk| {
                app.emit("audio-chunk", AudioChunkEvent {
                    job_id,
                    samples: chunk.samples,
                    sample_rate: chunk.sample_rate,
                    is_final: chunk.is_final,
                })?;
                Ok(())
            },
            Some(cancel.as_ref()),
        )
    } else {
        let sample_rate = loaded.engine.sample_rate();
        let samples = loaded.engine.generate_cancellable(
            synthesis,
            |current, total| {
                let _ = app.emit("generation-progress", GenerationProgressEvent { job_id, current, total });
            },
            Some(cancel.as_ref()),
        )?;
        app.emit("audio-chunk", AudioChunkEvent { job_id, samples, sample_rate, is_final: true })?;
        Ok(())
    };
    let status = if cancel.load(Ordering::SeqCst) { "canceled" } else if result.is_ok() { "completed" } else { "failed" };
    let error = result.as_ref().err().map(|err| err.to_string());
    state.history.lock().map_err(|_| anyhow::anyhow!("history lock poisoned"))?.push(record_from_result(
        request.text,
        &entry,
        backend,
        request.settings,
        status,
        started.elapsed().as_millis(),
        error.clone(),
    ));
    state.cancellation.lock().map_err(|_| anyhow::anyhow!("cancellation lock poisoned"))?.generation = None;
    app.emit("generation-finished", JobFinishedEvent { job_id, status: status.into(), error })?;
    result
}

fn next_job_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::SeqCst)
}
```

- [ ] **Step 5: Expose load, generation, and cancel commands**

Append to `commands.rs`:

```rust
use std::sync::Arc;

use crate::engine_runner;
use crate::types::{GenerateRequest, LoadModelRequest};

#[tauri::command]
pub fn load_model(app: AppHandle, state: State<'_, Arc<DesktopState>>, request: LoadModelRequest) -> Result<CommandOk, String> {
    engine_runner::spawn_load_model(app, state, request)?;
    Ok(CommandOk { ok: true })
}

#[tauri::command]
pub fn cancel_model_load(state: State<'_, Arc<DesktopState>>) -> Result<CommandOk, String> {
    engine_runner::cancel_load(&state);
    Ok(CommandOk { ok: true })
}

#[tauri::command]
pub fn generate(app: AppHandle, state: State<'_, Arc<DesktopState>>, request: GenerateRequest) -> Result<u64, String> {
    engine_runner::spawn_generate(app, state, request)
}

#[tauri::command]
pub fn cancel_generation(state: State<'_, Arc<DesktopState>>) -> Result<CommandOk, String> {
    engine_runner::cancel_generation(&state);
    Ok(CommandOk { ok: true })
}
```

- [ ] **Step 6: Register new commands**

Update the `generate_handler!` list in `lib.rs`:

```rust
.invoke_handler(tauri::generate_handler![
    get_snapshot,
    save_app_settings,
    rescan_models,
    browse_model_dir,
    commands::load_model,
    commands::cancel_model_load,
    commands::generate,
    commands::cancel_generation
])
```

Add `mod engine_runner;` to the module list.

- [ ] **Step 7: Check backend compiles**

```powershell
cargo check -p voxui-desktop
```

Working directory: `voxui`.

Expected: backend compiles before committing this task.

- [ ] **Step 8: Commit engine runner**

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src
git commit -m "feat(desktop): add model loading and generation events"
```

---

### Task 7: Add Frontend Types, Backend Client, I18n, And State

**Files:**
- Create: `voxui/crates/voxui-desktop/src/lib/types.ts`
- Create: `voxui/crates/voxui-desktop/src/lib/backend.ts`
- Create: `voxui/crates/voxui-desktop/src/lib/i18n.ts`
- Create: `voxui/crates/voxui-desktop/src/lib/state.svelte.ts`
- Create: `voxui/crates/voxui-desktop/src/lib/format.ts`

- [ ] **Step 1: Add frontend dependencies for Tauri APIs**

Run:

```powershell
npm install @tauri-apps/api
```

Working directory: `voxui/crates/voxui-desktop`.

Expected: `@tauri-apps/api` is in `dependencies`.

Use `@tauri-apps/api` imports throughout the Svelte code. Do not use the global `window.__TAURI__` object in this plan.

- [ ] **Step 2: Create frontend types**

Create `src/lib/types.ts`:

```ts
export type LanguageMode = 'system' | 'english' | 'chinese';
export type InferenceBackend = 'cpu' | 'cuda';
export type JobStatus = 'queued' | 'generating' | 'playing' | 'completed' | 'canceled' | 'failed';

export interface GenerationSettings {
  cfgValue: number;
  inferenceTimesteps: number;
  minLen: number;
  maxLen: number;
  retryBadcase: boolean;
  retryBadcaseMaxTimes: number;
  retryBadcaseRatioThreshold: number;
  maxInputChars: number;
  streaming: boolean;
}

export interface AppSettings {
  modelDir: string;
  language: LanguageMode;
  backend: InferenceBackend;
  generation: GenerationSettings;
  volume: number;
  outputDeviceId: string | null;
}

export interface ModelEntry {
  id: string;
  label: string;
  modelName: string;
  modelDir: string;
  loraPath: string | null;
  loraName: string | null;
}

export interface ModelCatalog {
  modelDir: string;
  entries: ModelEntry[];
  warnings: string[];
}

export interface AppSnapshot {
  settings: AppSettings;
  catalog: ModelCatalog;
  systemLanguage: string;
}

export interface AudioChunkEvent {
  jobId: number;
  samples: number[];
  sampleRate: number;
  isFinal: boolean;
}

export interface LoadProgressEvent {
  phase: string;
  current: number;
  total: number;
  message: string;
}

export interface JobFinishedEvent {
  jobId: number;
  status: 'completed' | 'canceled' | 'failed';
  error: string | null;
}

export interface HistoryItem {
  localId: number;
  jobId: number | null;
  text: string;
  modelLabel: string;
  settings: GenerationSettings;
  status: JobStatus;
  progressCurrent: number;
  progressTotal: number;
  sampleRate: number | null;
  chunks: Float32Array[];
  error: string | null;
  createdAt: number;
}
```

- [ ] **Step 3: Create typed backend client**

Create `src/lib/backend.ts`:

```ts
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import type { AppSettings, AppSnapshot, AudioChunkEvent, LoadProgressEvent, ModelCatalog, JobFinishedEvent, GenerationSettings } from './types';

export const backend = {
  snapshot: () => invoke<AppSnapshot>('get_snapshot'),
  saveSettings: (settings: AppSettings) => invoke<{ ok: boolean }>('save_app_settings', { settings }),
  rescanModels: () => invoke<ModelCatalog>('rescan_models'),
  browseModelDir: () => invoke<string | null>('browse_model_dir'),
  loadModel: (entryId: string, backendName: 'cpu' | 'cuda') => invoke<{ ok: boolean }>('load_model', { request: { entryId, backend: backendName } }),
  cancelModelLoad: () => invoke<{ ok: boolean }>('cancel_model_load'),
  generate: (text: string, settings: GenerationSettings) => invoke<number>('generate', { request: { text, settings } }),
  cancelGeneration: () => invoke<{ ok: boolean }>('cancel_generation'),
  onLoadProgress: (handler: (event: LoadProgressEvent) => void) => listen<LoadProgressEvent>('model-load-progress', (event) => handler(event.payload)),
  onLoadError: (handler: (message: string) => void) => listen<string>('model-load-error', (event) => handler(event.payload)),
  onAudioChunk: (handler: (event: AudioChunkEvent) => void) => listen<AudioChunkEvent>('audio-chunk', (event) => handler(event.payload)),
  onGenerationFinished: (handler: (event: JobFinishedEvent) => void) => listen<JobFinishedEvent>('generation-finished', (event) => handler(event.payload)),
};
```

- [ ] **Step 4: Create i18n labels**

Create `src/lib/i18n.ts`:

```ts
import type { LanguageMode } from './types';

export type Locale = 'en' | 'zh';

export function resolveLocale(mode: LanguageMode, systemLanguage: string): Locale {
  if (mode === 'english') return 'en';
  if (mode === 'chinese') return 'zh';
  return systemLanguage.toLowerCase().startsWith('zh') ? 'zh' : 'en';
}

export const labels = {
  en: {
    title: 'AhanSays',
    model: 'Model',
    load: 'Load',
    settings: 'Settings',
    inputPlaceholder: 'Type text to synthesize...',
    push: 'Push to generate',
    cancel: 'Cancel',
    regenerate: 'Regenerate',
    replay: 'Replay',
    general: 'General',
    inference: 'Inference',
    audio: 'Audio',
    about: 'About',
    modelDirectory: 'Model directory',
    browse: 'Browse',
    rescan: 'Rescan',
    language: 'Language',
    backend: 'Backend',
    streaming: 'Streaming',
    maxInputChars: 'Max input characters',
    volume: 'Volume',
    testTone: 'Test tone',
    outputDevice: 'Output device',
    aboutText: 'Coded by 久嘉 & OpenCode & Codex. Licensed under GPLv3. Uses the VoxCPM Python implementation as reference/upstream.',
  },
  zh: {
    title: '焓言焓语',
    model: '模型',
    load: '加载',
    settings: '设置',
    inputPlaceholder: '输入要合成的文本...',
    push: '加入生成队列',
    cancel: '取消',
    regenerate: '重新生成',
    replay: '重播',
    general: '通用',
    inference: '推理',
    audio: '音频',
    about: '关于',
    modelDirectory: '模型目录',
    browse: '浏览',
    rescan: '重新扫描',
    language: '语言',
    backend: '后端',
    streaming: '流式生成',
    maxInputChars: '最大输入字符数',
    volume: '音量',
    testTone: '测试音',
    outputDevice: '输出设备',
    aboutText: '由 久嘉 & OpenCode & Codex 编写。使用 GPLv3 许可证。以 VoxCPM Python 实现作为参考/上游。',
  },
} as const;
```

- [ ] **Step 5: Create state and formatting helpers**

Create `src/lib/state.svelte.ts`:

```ts
import type { AppSettings, HistoryItem, LoadProgressEvent, ModelCatalog, ModelEntry } from './types';

export const appState = $state({
  ready: false,
  systemLanguage: 'en',
  settings: null as AppSettings | null,
  catalog: null as ModelCatalog | null,
  selectedEntryId: '',
  loadedEntry: null as ModelEntry | null,
  loadProgress: null as LoadProgressEvent | null,
  loadingModel: false,
  loadError: null as string | null,
  history: [] as HistoryItem[],
  activePlaybackJobId: null as number | null,
  inputText: '',
});
```

Create `src/lib/format.ts`:

```ts
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function clampVolume(value: number): number {
  return Math.max(0, Math.min(1, value));
}
```

- [ ] **Step 6: Run frontend type check**

```powershell
npm run check
```

Working directory: `voxui/crates/voxui-desktop`.

Expected: type check passes.

- [ ] **Step 7: Commit frontend types and state**

```powershell
git add voxui/crates/voxui-desktop/package.json voxui/crates/voxui-desktop/package-lock.json voxui/crates/voxui-desktop/src/lib
git commit -m "feat(desktop): add frontend state and backend client"
```

---

### Task 8: Add WebAudio Service

**Files:**
- Create: `voxui/crates/voxui-desktop/src/lib/audio.ts`

- [ ] **Step 1: Create WebAudio service**

Create `src/lib/audio.ts`:

```ts
export interface AudioOutputDevice {
  deviceId: string;
  label: string;
}

type SinkAudioContext = AudioContext & {
  setSinkId?: (sinkId: string | { type: 'none' }) => Promise<void>;
};

export class WebAudioService {
  private context: SinkAudioContext | null = null;
  private gain: GainNode | null = null;
  private nextStartTime = 0;
  private sources: AudioBufferSourceNode[] = [];
  private volume = 0.8;

  get supportsOutputSelection(): boolean {
    return typeof AudioContext !== 'undefined' && 'setSinkId' in AudioContext.prototype;
  }

  async ensureContext(): Promise<SinkAudioContext> {
    if (!this.context) {
      this.context = new AudioContext() as SinkAudioContext;
      this.gain = this.context.createGain();
      this.gain.gain.value = this.volume;
      this.gain.connect(this.context.destination);
    }
    if (this.context.state === 'suspended') {
      await this.context.resume();
    }
    return this.context;
  }

  setVolume(value: number): void {
    this.volume = Math.max(0, Math.min(1, value));
    if (this.gain && this.context) {
      this.gain.gain.setValueAtTime(this.volume, this.context.currentTime);
    }
  }

  async setOutputDevice(deviceId: string | null): Promise<boolean> {
    const context = await this.ensureContext();
    if (!deviceId || !context.setSinkId) {
      return false;
    }
    await context.setSinkId(deviceId);
    return true;
  }

  async listOutputDevices(): Promise<AudioOutputDevice[]> {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((device) => device.kind === 'audiooutput')
      .map((device) => ({ deviceId: device.deviceId, label: device.label || 'Audio output' }));
  }

  async requestAudioPermission(): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) return;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    for (const track of stream.getTracks()) track.stop();
  }

  async playSineTest(): Promise<void> {
    const context = await this.ensureContext();
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(440, context.currentTime);
    gain.gain.setValueAtTime(0.0001, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(Math.max(0.0001, this.volume), context.currentTime + 0.08);
    gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + 0.75);
    oscillator.connect(gain).connect(context.destination);
    oscillator.start(context.currentTime);
    oscillator.stop(context.currentTime + 0.8);
  }

  stopAll(): void {
    for (const source of this.sources) {
      try {
        source.stop();
      } catch {
        // Already stopped sources are harmless.
      }
    }
    this.sources = [];
    this.nextStartTime = this.context?.currentTime ?? 0;
  }

  async enqueue(samples: Float32Array, sampleRate: number): Promise<void> {
    const context = await this.ensureContext();
    const buffer = context.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.gain ?? context.destination);
    const startAt = Math.max(context.currentTime + 0.05, this.nextStartTime);
    source.start(startAt);
    this.nextStartTime = startAt + buffer.duration;
    this.sources.push(source);
    source.onended = () => {
      this.sources = this.sources.filter((item) => item !== source);
    };
  }
}

export const webAudio = new WebAudioService();
```

- [ ] **Step 2: Run frontend type check**

```powershell
npm run check
```

Working directory: `voxui/crates/voxui-desktop`.

Expected: `audio.ts` type checks.

- [ ] **Step 3: Commit WebAudio service**

```powershell
git add voxui/crates/voxui-desktop/src/lib/audio.ts
git commit -m "feat(desktop): add webaudio playback service"
```

---

### Task 9: Add Main UI Components

**Files:**
- Create: `voxui/crates/voxui-desktop/src/components/NavBar.svelte`
- Create: `voxui/crates/voxui-desktop/src/components/HistoryList.svelte`
- Create: `voxui/crates/voxui-desktop/src/components/HistoryCard.svelte`
- Create: `voxui/crates/voxui-desktop/src/components/InputBar.svelte`
- Create: `voxui/crates/voxui-desktop/src/components/LoadProgressModal.svelte`
- Create: `voxui/crates/voxui-desktop/src/components/SettingsModal.svelte`
- Modify: `voxui/crates/voxui-desktop/src/App.svelte`
- Modify: `voxui/crates/voxui-desktop/src/main.ts`

- [ ] **Step 1: Create NavBar component**

Create `src/components/NavBar.svelte`:

```svelte
<script lang="ts">
  import type { ModelEntry } from '../lib/types';

  interface Props {
    title: string;
    modelLabel: string;
    loadLabel: string;
    settingsLabel: string;
    entries: ModelEntry[];
    selectedEntryId: string;
    loading: boolean;
    onSelect: (id: string) => void;
    onLoad: () => void;
    onSettings: () => void;
  }

  let { title, modelLabel, loadLabel, settingsLabel, entries, selectedEntryId, loading, onSelect, onLoad, onSettings }: Props = $props();
  const hasEntries = $derived(entries.length > 0);
</script>

<div class="navbar border-b border-base-300 bg-base-100 px-4 shadow-sm">
  <div class="navbar-start">
    <div class="text-xl font-semibold tracking-tight">{title}</div>
  </div>
  <div class="navbar-end gap-2">
    <label class="sr-only" for="model-select">{modelLabel}</label>
    <select id="model-select" class="select select-bordered select-sm w-56" value={selectedEntryId} disabled={!hasEntries} onchange={(event) => onSelect(event.currentTarget.value)}>
      <option value="">{hasEntries ? modelLabel : 'No models detected'}</option>
      {#each entries as entry}
        <option value={entry.id}>{entry.label}</option>
      {/each}
    </select>
    <button class="btn btn-primary btn-sm" disabled={!hasEntries || !selectedEntryId || loading} onclick={onLoad}>
      {loading ? '...' : loadLabel}
    </button>
    <button class="btn btn-ghost btn-sm" aria-label={settingsLabel} onclick={onSettings}>⚙</button>
  </div>
</div>
```

- [ ] **Step 2: Create HistoryCard component**

Create `src/components/HistoryCard.svelte`:

```svelte
<script lang="ts">
  import type { HistoryItem } from '../lib/types';

  interface Props {
    item: HistoryItem;
    cancelLabel: string;
    regenerateLabel: string;
    replayLabel: string;
    onCancel: (item: HistoryItem) => void;
    onRegenerate: (item: HistoryItem) => void;
    onReplay: (item: HistoryItem) => void;
  }

  let { item, cancelLabel, regenerateLabel, replayLabel, onCancel, onRegenerate, onReplay }: Props = $props();
  const canCancel = $derived(item.status === 'queued' || item.status === 'generating' || item.status === 'playing');
  const canReplay = $derived(item.status === 'completed' && item.chunks.length > 0);
  const progressPercent = $derived(item.progressTotal > 0 ? Math.round((item.progressCurrent / item.progressTotal) * 100) : 0);
</script>

<article class="card border border-base-300 bg-base-100 shadow-sm">
  <div class="card-body gap-3">
    <div class="flex items-start justify-between gap-4">
      <div>
        <div class="font-medium">{item.modelLabel}</div>
        <div class="text-xs opacity-70">{new Date(item.createdAt).toLocaleString()}</div>
      </div>
      <div class="badge badge-outline">{item.status}</div>
    </div>
    <p class="whitespace-pre-wrap rounded-box bg-base-200 p-3 text-sm">{item.text}</p>
    {#if item.progressTotal > 0}
      <progress class="progress progress-primary w-full" value={progressPercent} max="100"></progress>
    {/if}
    {#if item.error}
      <div class="alert alert-error py-2 text-sm">{item.error}</div>
    {/if}
    <div class="card-actions justify-end">
      {#if canCancel}
        <button class="btn btn-warning btn-sm" onclick={() => onCancel(item)}>{cancelLabel}</button>
      {:else}
        {#if canReplay}
          <button class="btn btn-ghost btn-sm" onclick={() => onReplay(item)}>{replayLabel}</button>
        {/if}
        <button class="btn btn-primary btn-sm" onclick={() => onRegenerate(item)}>{regenerateLabel}</button>
      {/if}
    </div>
  </div>
</article>
```

- [ ] **Step 3: Create HistoryList and InputBar components**

Create `src/components/HistoryList.svelte`:

```svelte
<script lang="ts">
  import type { HistoryItem } from '../lib/types';
  import HistoryCard from './HistoryCard.svelte';

  interface Props {
    items: HistoryItem[];
    emptyText: string;
    cancelLabel: string;
    regenerateLabel: string;
    replayLabel: string;
    onCancel: (item: HistoryItem) => void;
    onRegenerate: (item: HistoryItem) => void;
    onReplay: (item: HistoryItem) => void;
  }

  let props: Props = $props();
</script>

<div class="h-full overflow-y-auto p-4">
  {#if props.items.length === 0}
    <div class="flex h-full items-center justify-center text-center opacity-70">{props.emptyText}</div>
  {:else}
    <div class="mx-auto flex max-w-4xl flex-col gap-3">
      {#each props.items as item (item.localId)}
        <HistoryCard item={item} cancelLabel={props.cancelLabel} regenerateLabel={props.regenerateLabel} replayLabel={props.replayLabel} onCancel={props.onCancel} onRegenerate={props.onRegenerate} onReplay={props.onReplay} />
      {/each}
    </div>
  {/if}
</div>
```

Create `src/components/InputBar.svelte`:

```svelte
<script lang="ts">
  interface Props {
    value: string;
    placeholder: string;
    pushLabel: string;
    maxChars: number;
    disabled: boolean;
    onInput: (value: string) => void;
    onPush: () => void;
  }

  let { value, placeholder, pushLabel, maxChars, disabled, onInput, onPush }: Props = $props();
  const count = $derived(value.length);
</script>

<div class="border-t border-base-300 bg-base-100 p-3">
  <div class="mx-auto flex max-w-4xl gap-3">
    <div class="flex-1">
      <textarea class="textarea textarea-bordered h-24 w-full resize-none" {placeholder} value={value} oninput={(event) => onInput(event.currentTarget.value)}></textarea>
      <div class="mt-1 text-right text-xs opacity-70">{count}/{maxChars}</div>
    </div>
    <button class="btn btn-primary h-24" disabled={disabled || count === 0 || count > maxChars} onclick={onPush}>{pushLabel}</button>
  </div>
</div>
```

- [ ] **Step 4: Create LoadProgressModal component**

Create `src/components/LoadProgressModal.svelte`:

```svelte
<script lang="ts">
  import type { LoadProgressEvent } from '../lib/types';

  interface Props {
    open: boolean;
    progress: LoadProgressEvent | null;
    cancelLabel: string;
    onCancel: () => void;
  }

  let { open, progress, cancelLabel, onCancel }: Props = $props();
  const percent = $derived(progress && progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0);
</script>

{#if open}
  <div class="modal modal-open">
    <div class="modal-box">
      <h3 class="text-lg font-bold">Loading model</h3>
      <p class="py-3">{progress?.message ?? 'Starting...'}</p>
      <progress class="progress progress-primary w-full" value={percent} max="100"></progress>
      <div class="modal-action">
        <button class="btn btn-warning" onclick={onCancel}>{cancelLabel}</button>
      </div>
    </div>
  </div>
{/if}
```

- [ ] **Step 5: Create SettingsModal component**

Create `src/components/SettingsModal.svelte`:

```svelte
<script lang="ts">
  import type { AppSettings } from '../lib/types';
  import type { AudioOutputDevice } from '../lib/audio';

  interface TextLabels {
    general: string;
    inference: string;
    audio: string;
    about: string;
    modelDirectory: string;
    browse: string;
    rescan: string;
    language: string;
    backend: string;
    streaming: string;
    maxInputChars: string;
    volume: string;
    testTone: string;
    outputDevice: string;
    aboutText: string;
  }

  interface Props {
    open: boolean;
    labels: TextLabels;
    settings: AppSettings;
    devices: AudioOutputDevice[];
    supportsOutputSelection: boolean;
    onClose: () => void;
    onSave: (settings: AppSettings) => void;
    onBrowse: () => void;
    onRescan: () => void;
    onRequestPermission: () => void;
    onTestTone: () => void;
  }

  let { open, labels, settings, devices, supportsOutputSelection, onClose, onSave, onBrowse, onRescan, onRequestPermission, onTestTone }: Props = $props();
  let activeTab = $state<'general' | 'inference' | 'audio' | 'about'>('general');

  function saveAndClose() {
    onSave(settings);
    onClose();
  }
</script>

{#if open}
  <div class="modal modal-open">
    <div class="modal-box grid max-w-4xl grid-cols-[12rem_1fr] gap-4">
      <div class="menu rounded-box bg-base-200">
        <button class:active={activeTab === 'general'} onclick={() => (activeTab = 'general')}>{labels.general}</button>
        <button class:active={activeTab === 'inference'} onclick={() => (activeTab = 'inference')}>{labels.inference}</button>
        <button class:active={activeTab === 'audio'} onclick={() => (activeTab = 'audio')}>{labels.audio}</button>
        <button class:active={activeTab === 'about'} onclick={() => (activeTab = 'about')}>{labels.about}</button>
      </div>

      <div class="space-y-4">
        {#if activeTab === 'general'}
          <label class="form-control">
            <div class="label"><span class="label-text">{labels.modelDirectory}</span></div>
            <input class="input input-bordered" bind:value={settings.modelDir} />
          </label>
          <div class="flex gap-2">
            <button class="btn" onclick={onBrowse}>{labels.browse}</button>
            <button class="btn" onclick={onRescan}>{labels.rescan}</button>
          </div>
          <label class="form-control">
            <div class="label"><span class="label-text">{labels.language}</span></div>
            <select class="select select-bordered" bind:value={settings.language}>
              <option value="system">System</option>
              <option value="english">English</option>
              <option value="chinese">中文</option>
            </select>
          </label>
        {:else if activeTab === 'inference'}
          <label class="form-control">
            <div class="label"><span class="label-text">{labels.backend}</span></div>
            <select class="select select-bordered" bind:value={settings.backend}>
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA</option>
            </select>
          </label>
          <label class="label cursor-pointer justify-start gap-3">
            <input class="toggle toggle-primary" type="checkbox" bind:checked={settings.generation.streaming} />
            <span>{labels.streaming}</span>
          </label>
          <label class="form-control">
            <div class="label"><span class="label-text">CFG</span></div>
            <input class="input input-bordered" type="number" step="0.1" bind:value={settings.generation.cfgValue} />
          </label>
          <label class="form-control">
            <div class="label"><span class="label-text">Timesteps</span></div>
            <input class="input input-bordered" type="number" min="1" bind:value={settings.generation.inferenceTimesteps} />
          </label>
          <label class="form-control">
            <div class="label"><span class="label-text">{labels.maxInputChars}</span></div>
            <input class="input input-bordered" type="number" min="1" bind:value={settings.generation.maxInputChars} />
          </label>
        {:else if activeTab === 'audio'}
          <div class="alert" class:alert-warning={!supportsOutputSelection}>
            {supportsOutputSelection ? 'Browser output selection is available.' : 'Output device selection is unavailable; default output will be used.'}
          </div>
          <label class="form-control">
            <div class="label"><span class="label-text">{labels.outputDevice}</span></div>
            <select class="select select-bordered" bind:value={settings.outputDeviceId} disabled={!supportsOutputSelection}>
              <option value={null}>Default</option>
              {#each devices as device}
                <option value={device.deviceId}>{device.label}</option>
              {/each}
            </select>
          </label>
          <button class="btn" onclick={onRequestPermission}>Refresh / Permission</button>
          <label class="form-control">
            <div class="label"><span class="label-text">{labels.volume}</span></div>
            <input class="range range-primary" type="range" min="0" max="1" step="0.01" bind:value={settings.volume} />
          </label>
          <button class="btn btn-primary" onclick={onTestTone}>{labels.testTone}</button>
        {:else}
          <h3 class="text-xl font-semibold">AhanSays / 焓言焓语</h3>
          <p class="leading-relaxed">{labels.aboutText}</p>
        {/if}

        <div class="modal-action">
          <button class="btn btn-ghost" onclick={onClose}>Close</button>
          <button class="btn btn-primary" onclick={saveAndClose}>Save</button>
        </div>
      </div>
    </div>
  </div>
{/if}
```

- [ ] **Step 6: Replace App.svelte with integrated shell**

Replace `src/App.svelte` with this complete app shell:

```svelte
<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import NavBar from './components/NavBar.svelte';
  import HistoryList from './components/HistoryList.svelte';
  import InputBar from './components/InputBar.svelte';
  import LoadProgressModal from './components/LoadProgressModal.svelte';
  import SettingsModal from './components/SettingsModal.svelte';
  import { webAudio, type AudioOutputDevice } from './lib/audio';
  import { backend } from './lib/backend';
  import { labels, resolveLocale } from './lib/i18n';
  import { appState } from './lib/state.svelte';
  import type { AppSettings, HistoryItem } from './lib/types';

  let settingsOpen = $state(false);
  let audioDevices = $state<AudioOutputDevice[]>([]);
  let unlisteners: Array<() => void> = [];
  let nextLocalId = 1;
  const locale = $derived(appState.settings ? resolveLocale(appState.settings.language, appState.systemLanguage) : 'en');
  const t = $derived(labels[locale]);
  const selectedEntry = $derived(appState.catalog?.entries.find((entry) => entry.id === appState.selectedEntryId) ?? null);
  const canGenerate = $derived(Boolean(appState.loadedEntry && appState.inputText.trim() && appState.settings && appState.inputText.length <= appState.settings.generation.maxInputChars));

  onMount(async () => {
    const snapshot = await backend.snapshot();
    appState.settings = snapshot.settings;
    appState.catalog = snapshot.catalog;
    appState.systemLanguage = snapshot.systemLanguage;
    appState.selectedEntryId = snapshot.catalog.entries[0]?.id ?? '';
    webAudio.setVolume(snapshot.settings.volume);
    audioDevices = await webAudio.listOutputDevices();
    unlisteners = [
      await backend.onLoadProgress((event) => {
        appState.loadProgress = event;
        appState.loadingModel = true;
      }),
      await backend.onLoadError((message) => {
        appState.loadError = message;
        appState.loadingModel = false;
      }),
      await backend.onAudioChunk((event) => appendChunk(event.jobId, event.samples, event.sampleRate, event.isFinal)),
      await backend.onGenerationFinished((event) => finishJob(event.jobId, event.status, event.error)),
    ];
    appState.ready = true;
  });

  onDestroy(() => {
    for (const unlisten of unlisteners) unlisten();
    webAudio.stopAll();
  });

  async function loadSelectedModel() {
    if (!appState.settings || !appState.selectedEntryId) return;
    appState.loadError = null;
    appState.loadingModel = true;
    await backend.loadModel(appState.selectedEntryId, appState.settings.backend);
    appState.loadedEntry = selectedEntry;
  }

  async function cancelLoad() {
    await backend.cancelModelLoad();
    appState.loadingModel = false;
    appState.loadProgress = null;
  }

  async function pushGeneration(text = appState.inputText) {
    if (!appState.settings || !appState.loadedEntry || !text.trim()) return;
    const item: HistoryItem = {
      localId: nextLocalId++,
      jobId: null,
      text: text.trim(),
      modelLabel: appState.loadedEntry.label,
      settings: structuredClone(appState.settings.generation),
      status: 'queued',
      progressCurrent: 0,
      progressTotal: 0,
      sampleRate: null,
      chunks: [],
      error: null,
      createdAt: Date.now(),
    };
    appState.history = [...appState.history, item];
    const jobId = await backend.generate(item.text, item.settings);
    item.jobId = jobId;
    item.status = 'generating';
    appState.inputText = '';
  }

  async function cancelGeneration() {
    await backend.cancelGeneration();
    webAudio.stopAll();
  }

  function appendChunk(jobId: number, samples: number[], sampleRate: number, isFinal: boolean) {
    const item = appState.history.find((entry) => entry.jobId === jobId);
    if (!item) return;
    const chunk = new Float32Array(samples);
    item.sampleRate = sampleRate;
    item.chunks.push(chunk);
    item.status = isFinal ? 'completed' : 'playing';
    appState.activePlaybackJobId = jobId;
    void webAudio.enqueue(chunk, sampleRate);
  }

  function finishJob(jobId: number, status: 'completed' | 'canceled' | 'failed', error: string | null) {
    const item = appState.history.find((entry) => entry.jobId === jobId);
    if (!item) return;
    item.status = status;
    item.error = error;
    if (status !== 'completed') webAudio.stopAll();
  }

  function replay(item: HistoryItem) {
    if (!item.sampleRate) return;
    webAudio.stopAll();
    appState.activePlaybackJobId = item.jobId;
    for (const chunk of item.chunks) void webAudio.enqueue(chunk, item.sampleRate);
  }

  function regenerate(item: HistoryItem) {
    void pushGeneration(item.text);
  }

  async function saveSettings(settings: AppSettings) {
    appState.settings = settings;
    webAudio.setVolume(settings.volume);
    await webAudio.setOutputDevice(settings.outputDeviceId);
    await backend.saveSettings(settings);
  }

  async function browseModelDir() {
    if (!appState.settings) return;
    const selected = await backend.browseModelDir();
    if (selected) appState.settings.modelDir = selected;
  }

  async function rescanModels() {
    appState.catalog = await backend.rescanModels();
    appState.selectedEntryId = appState.catalog.entries[0]?.id ?? '';
  }

  async function refreshAudioDevices() {
    await webAudio.requestAudioPermission();
    audioDevices = await webAudio.listOutputDevices();
  }
</script>

{#if appState.ready && appState.settings && appState.catalog}
  <div class="grid h-full grid-rows-[auto_1fr_auto] bg-base-200">
    <NavBar title={t.title} modelLabel={t.model} loadLabel={t.load} settingsLabel={t.settings} entries={appState.catalog.entries} selectedEntryId={appState.selectedEntryId} loading={appState.loadingModel} onSelect={(id) => (appState.selectedEntryId = id)} onLoad={loadSelectedModel} onSettings={() => (settingsOpen = true)} />
    <HistoryList items={appState.history} emptyText="Load a model, then push text to generate speech." cancelLabel={t.cancel} regenerateLabel={t.regenerate} replayLabel={t.replay} onCancel={cancelGeneration} onRegenerate={regenerate} onReplay={replay} />
    <InputBar value={appState.inputText} placeholder={t.inputPlaceholder} pushLabel={t.push} maxChars={appState.settings.generation.maxInputChars} disabled={!canGenerate} onInput={(value) => (appState.inputText = value)} onPush={() => pushGeneration()} />
  </div>
  <SettingsModal open={settingsOpen} labels={t} settings={appState.settings} devices={audioDevices} supportsOutputSelection={webAudio.supportsOutputSelection} onClose={() => (settingsOpen = false)} onSave={saveSettings} onBrowse={browseModelDir} onRescan={rescanModels} onRequestPermission={refreshAudioDevices} onTestTone={() => webAudio.playSineTest()} />
  <LoadProgressModal open={appState.loadingModel} progress={appState.loadProgress} cancelLabel={t.cancel} onCancel={cancelLoad} />
{:else}
  <div class="flex h-full items-center justify-center bg-base-200">Loading...</div>
{/if}
```

- [ ] **Step 7: Ensure main.ts imports CSS and mounts app**

`src/main.ts` should be:

```ts
import './app.css';
import App from './App.svelte';

const app = new App({
  target: document.getElementById('app')!,
});

export default app;
```

- [ ] **Step 8: Type check and build frontend**

```powershell
npm run check; if ($?) { npm run build }
```

Working directory: `voxui/crates/voxui-desktop`.

Expected: Svelte check and Vite build pass.

- [ ] **Step 9: Commit UI components**

```powershell
git add voxui/crates/voxui-desktop/src
git commit -m "feat(desktop): add main svelte interface"
```

---

### Task 10: Final Backend/Frontend Integration And Tauri Config

**Files:**
- Modify: `voxui/crates/voxui-desktop/src-tauri/tauri.conf.json`
- Modify: `voxui/crates/voxui-desktop/src-tauri/capabilities/default.json`
- Modify: `voxui/crates/voxui-desktop/package.json`

- [ ] **Step 1: Configure Tauri product metadata**

Ensure `src-tauri/tauri.conf.json` includes:

```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "AhanSays",
  "version": "0.1.0",
  "identifier": "com.voxui.ahansays",
  "build": {
    "beforeDevCommand": "npm run dev",
    "devUrl": "http://localhost:1420",
    "beforeBuildCommand": "npm run build",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      {
        "title": "AhanSays",
        "width": 1100,
        "height": 760,
        "minWidth": 820,
        "minHeight": 600
      }
    ],
    "security": {
      "csp": null
    }
  },
  "bundle": {
    "active": true,
    "targets": "all",
    "icon": []
  }
}
```

This plan uses `@tauri-apps/api` imports, so `withGlobalTauri` is not required. If an implementation chooses to call the global `window.__TAURI__` object instead of using imports, add `"withGlobalTauri": true` inside the `app` object in `tauri.conf.json`.

- [ ] **Step 2: Configure capabilities for commands and dialog**

Ensure `src-tauri/capabilities/default.json` permits core events, dialog, and app commands:

```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "default",
  "description": "Default desktop capability",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "dialog:default",
    "opener:default"
  ]
}
```

- [ ] **Step 3: Ensure package scripts exist**

Ensure `package.json` scripts include:

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "check": "svelte-check --tsconfig ./tsconfig.json",
    "tauri": "tauri"
  }
}
```

- [ ] **Step 4: Run full desktop checks**

```powershell
npm run check; if ($?) { npm run build }; if ($?) { cargo check -p voxui-desktop }
```

Run first two commands in `voxui/crates/voxui-desktop`; run `cargo check` in `voxui`.

Expected: frontend and backend checks pass.

- [ ] **Step 5: Commit Tauri integration config**

```powershell
git add voxui/crates/voxui-desktop/package.json voxui/crates/voxui-desktop/src-tauri/tauri.conf.json voxui/crates/voxui-desktop/src-tauri/capabilities/default.json
git commit -m "chore(desktop): configure tauri desktop app"
```

---

### Task 11: Update README Desktop Commands

**Files:**
- Modify: `README.txt`

- [ ] **Step 1: Replace desktop section**

Update lines 10-15 of `README.txt` to:

```text
Build desktop:
cd voxui\crates\voxui-desktop; npm install
cd voxui\crates\voxui-desktop; npm run build
cd voxui; cargo build -p voxui-desktop --release

Run desktop with debug logs:
cd voxui\crates\voxui-desktop; $env:RUST_LOG = "voxui_desktop=debug,voxui_inference=debug"; npm run tauri dev -- --features cuda
```

- [ ] **Step 2: Verify README commands reference existing paths**

```powershell
Test-Path -LiteralPath "voxui\crates\voxui-desktop\package.json"; Test-Path -LiteralPath "voxui\crates\voxui-desktop\src-tauri\Cargo.toml"
```

Expected: both outputs are `True`.

- [ ] **Step 3: Commit README update**

```powershell
git add README.txt
git commit -m "docs: update desktop build commands"
```

---

### Task 12: Manual Verification Pass

**Files:**
- No required code changes unless verification reveals desktop GUI defects.

- [ ] **Step 1: Run desktop-only automated checks**

```powershell
cargo test -p voxui-desktop settings models history
```

Working directory: `voxui`.

Expected: desktop settings, model discovery, and history tests pass.

- [ ] **Step 2: Run frontend checks**

```powershell
npm run check; if ($?) { npm run build }
```

Working directory: `voxui/crates/voxui-desktop`.

Expected: Svelte check and build pass.

- [ ] **Step 3: Run backend check without expensive inference tests**

```powershell
cargo check -p voxui-desktop
```

Working directory: `voxui`.

Expected: desktop backend compiles. This does not run `voxui-inference` golden or matrix tests.

- [ ] **Step 4: Run Tauri dev app manually**

```powershell
$env:RUST_LOG = "voxui_desktop=debug,voxui_inference=debug"; npm run tauri dev -- --features cuda
```

Working directory: `voxui/crates/voxui-desktop`.

Expected: app opens, shows fixed navbar, model dropdown, settings button, empty history, and fixed input bar.

- [ ] **Step 5: Manual UI checklist**

Verify each item and record failures in the next implementation task:

```text
[ ] Model directory can be set to D:\Sandbox_Share\VoxUI\models
[ ] Dropdown shows voxcpm2-fp16 and voxcpm2-fp16 | lora_ft2 when those files exist
[ ] Load modal appears and can be canceled
[ ] CPU/CUDA selection affects the next load attempt
[ ] English and Chinese labels switch in settings
[ ] Sine test plays with fade-in and fade-out
[ ] Volume slider changes sine test and generated playback volume
[ ] Streaming generation creates a history item and plays chunks
[ ] Batch generation creates a history item and auto-plays after completion
[ ] Cancel stops active generation playback
[ ] Regenerate queues a new item with the same text
[ ] No two items play at the same time
[ ] Exiting writes ahan-says-history-*.log next to the executable when generation history exists
```

- [ ] **Step 6: Commit verification fixes if any were needed**

If verification required code changes, commit only those fixes:

```powershell
git status --short
git add voxui/crates/voxui-desktop README.txt voxui/Cargo.toml
git commit -m "fix(desktop): address manual verification issues"
```

If no changes were needed, do not create a commit.
