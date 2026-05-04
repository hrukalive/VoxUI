# Tauri Desktop TTS Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `voxui-desktop` the primary usable Tauri app for native VoxCPM text-to-speech playback through the selected audio device.

**Architecture:** Keep inference and GGUF loading behind `voxui-inference`; keep audio host/device playback behind `voxui-audio`; make `voxui-desktop` own app state, config, Tauri commands/events, and the Leptos UI. Add a small pure Rust backend helper module so scanning, request construction, and config behavior can be tested without a running Tauri WebView.

**Tech Stack:** Rust 2021, Tauri 2, Leptos 0.7 CSR, Trunk, Candle, cpal, serde, tokio.

---

## File Structure

- Modify `voxui/Cargo.toml`: remove `crates/voxui-app` from active workspace members.
- Modify `voxui/crates/voxui-desktop/src-tauri/Cargo.toml`: add test dependency `tempfile`, keep `cuda` feature forwarding to inference.
- Create `voxui/crates/voxui-desktop/src-tauri/src/desktop_core.rs`: serializable DTOs and pure helpers for model scanning, LoRA scanning, model root discovery, synthesis request construction, and string/path normalization.
- Modify `voxui/crates/voxui-desktop/src-tauri/src/state.rs`: desktop config defaults, testable load/save helpers, shared engine/config state, and a synthesis busy guard.
- Modify `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`: Tauri command handlers that call `desktop_core`, manage engine/LoRA state, emit events, and play generated PCM.
- Modify `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`: register the new command set.
- Modify `voxui/crates/voxui-desktop/src/app.rs`: typed frontend state and Tauri invoke payloads for model, LoRA, prompt audio, prompt text, reference audio, synthesis, progress, completion, and errors.
- Modify `voxui/crates/voxui-desktop/src/components/settings_modal.rs`: add prompt/reference fields and language setting, and return complete `SettingsValues`.
- Modify `voxui/crates/voxui-desktop/src/components/input_box.rs`: use textarea input so longer TTS text is practical.
- Modify `voxui/crates/voxui-desktop/src/components/status_bar.rs`: show actual loaded backend, model, LoRA, and selected audio device.
- Modify `voxui/crates/voxui-desktop/src/components/history.rs`: render explicit error messages.
- Modify `voxui/crates/voxui-desktop/src/i18n.rs`: replace mojibake Chinese strings with UTF-8 strings and add labels for prompt/reference/audio path fields.
- Modify `voxui/crates/voxui-desktop/src-tauri/tauri.conf.json`: run Trunk for dev/build.
- Modify `voxui/crates/voxui-desktop/Trunk.toml`: keep the frontend build self-contained.

---

## Task 1: Add Testable Desktop Backend Core

**Files:**
- Create: `voxui/crates/voxui-desktop/src-tauri/src/desktop_core.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/Cargo.toml`

- [ ] **Step 1: Add the test dependency**

In `voxui/crates/voxui-desktop/src-tauri/Cargo.toml`, add:

```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 2: Write failing tests for model scanning, LoRA scanning, and request construction**

Create `voxui/crates/voxui-desktop/src-tauri/src/desktop_core.rs` with these tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn create_manifest_dir(root: &std::path::Path, name: &str) -> std::path::PathBuf {
        let path = root.join(name);
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("manifest.json"), "{}").unwrap();
        path
    }

    #[test]
    fn scan_model_entries_returns_only_manifest_dirs_sorted() {
        let tmp = tempdir().unwrap();
        create_manifest_dir(tmp.path(), "voxcpm2-fp16");
        create_manifest_dir(tmp.path(), "voxcpm05-fp16");
        fs::create_dir_all(tmp.path().join("not-a-model")).unwrap();

        let entries = scan_model_entries(tmp.path());

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "voxcpm05-fp16");
        assert_eq!(entries[1].name, "voxcpm2-fp16");
        assert!(entries.iter().all(|entry| entry.path.contains("voxcpm")));
    }

    #[test]
    fn scan_lora_entries_includes_none_and_manifest_dirs_sorted() {
        let tmp = tempdir().unwrap();
        let model = create_manifest_dir(tmp.path(), "voxcpm2-fp16");
        let lora_b = model.join("lora_b");
        let lora_a = model.join("lora_a");
        fs::create_dir_all(&lora_b).unwrap();
        fs::create_dir_all(&lora_a).unwrap();
        fs::write(lora_b.join("lora_manifest.json"), "{}").unwrap();
        fs::write(lora_a.join("lora_manifest.json"), "{}").unwrap();
        fs::create_dir_all(model.join("lora_without_manifest")).unwrap();

        let entries = scan_lora_entries(&model);

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], LoraEntry::none());
        assert_eq!(entries[1].name, "lora_a");
        assert_eq!(entries[2].name, "lora_b");
        assert!(entries[1].path.as_ref().unwrap().ends_with("lora_a"));
    }

    #[test]
    fn synthesis_args_builds_native_request_with_prompt_and_reference_paths() {
        let args = SynthesisArgs {
            index: 4,
            text: " hello   world ".to_string(),
            dit_steps: 7,
            prompt_wav_path: Some("for_test_wav/prompt.wav".to_string()),
            prompt_text: Some("prompt text".to_string()),
            reference_wav_path: Some("for_test_wav/reference.wav".to_string()),
        };

        let request = args.into_request();

        assert_eq!(request.text, " hello   world ");
        assert_eq!(request.inference_timesteps, 7);
        assert_eq!(request.prompt_text.as_deref(), Some("prompt text"));
        assert_eq!(
            request.prompt_wav_path.as_ref().unwrap(),
            &std::path::PathBuf::from("for_test_wav/prompt.wav")
        );
        assert_eq!(
            request.reference_wav_path.as_ref().unwrap(),
            &std::path::PathBuf::from("for_test_wav/reference.wav")
        );
    }

    #[test]
    fn empty_optional_strings_do_not_create_paths() {
        let args = SynthesisArgs {
            index: 0,
            text: "hello".to_string(),
            dit_steps: 10,
            prompt_wav_path: Some("   ".to_string()),
            prompt_text: Some("   ".to_string()),
            reference_wav_path: Some(String::new()),
        };

        let request = args.into_request();

        assert!(request.prompt_wav_path.is_none());
        assert!(request.prompt_text.is_none());
        assert!(request.reference_wav_path.is_none());
    }
}
```

- [ ] **Step 3: Run tests to verify they fail because the module API is missing**

Run:

```powershell
cargo test -p voxui-desktop desktop_core -- --nocapture
```

Expected: compile failure mentioning missing `scan_model_entries`, `scan_lora_entries`, `LoraEntry`, or `SynthesisArgs`.

- [ ] **Step 4: Implement `desktop_core.rs`**

Replace `voxui/crates/voxui-desktop/src-tauri/src/desktop_core.rs` with:

```rust
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use voxui_inference::SynthesisRequest;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelEntry {
    pub name: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoraEntry {
    pub name: String,
    pub path: Option<String>,
}

impl LoraEntry {
    pub fn none() -> Self {
        Self {
            name: "None".to_string(),
            path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisArgs {
    pub index: u32,
    pub text: String,
    pub dit_steps: usize,
    pub prompt_wav_path: Option<String>,
    pub prompt_text: Option<String>,
    pub reference_wav_path: Option<String>,
}

impl SynthesisArgs {
    pub fn into_request(self) -> SynthesisRequest {
        SynthesisRequest {
            text: self.text,
            prompt_wav_path: optional_path(self.prompt_wav_path),
            prompt_text: optional_string(self.prompt_text),
            reference_wav_path: optional_path(self.reference_wav_path),
            inference_timesteps: self.dit_steps,
            ..SynthesisRequest::default()
        }
    }
}

pub fn scan_model_entries(models_root: &Path) -> Vec<ModelEntry> {
    let mut entries = std::fs::read_dir(models_root)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() || !path.join("manifest.json").exists() {
                return None;
            }
            Some(ModelEntry {
                name: entry.file_name().to_string_lossy().to_string(),
                path: display_path(&path),
            })
        })
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

pub fn scan_lora_entries(model_dir: &Path) -> Vec<LoraEntry> {
    let mut entries = std::fs::read_dir(model_dir)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .filter_map(|entry| {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if !path.is_dir() || !name.starts_with("lora_") || !path.join("lora_manifest.json").exists() {
                return None;
            }
            Some(LoraEntry {
                name,
                path: Some(display_path(&path)),
            })
        })
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    let mut with_none = vec![LoraEntry::none()];
    with_none.extend(entries);
    with_none
}

pub fn discover_models_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for candidate_base in cwd.ancestors().take(6) {
        let candidate = candidate_base.join("models");
        if candidate.is_dir() {
            return candidate;
        }
    }
    PathBuf::from("models")
}

pub fn display_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn optional_string(value: Option<String>) -> Option<String> {
    value.and_then(|text| {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn optional_path(value: Option<String>) -> Option<PathBuf> {
    optional_string(value).map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn create_manifest_dir(root: &std::path::Path, name: &str) -> std::path::PathBuf {
        let path = root.join(name);
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("manifest.json"), "{}").unwrap();
        path
    }

    #[test]
    fn scan_model_entries_returns_only_manifest_dirs_sorted() {
        let tmp = tempdir().unwrap();
        create_manifest_dir(tmp.path(), "voxcpm2-fp16");
        create_manifest_dir(tmp.path(), "voxcpm05-fp16");
        fs::create_dir_all(tmp.path().join("not-a-model")).unwrap();

        let entries = scan_model_entries(tmp.path());

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "voxcpm05-fp16");
        assert_eq!(entries[1].name, "voxcpm2-fp16");
        assert!(entries.iter().all(|entry| entry.path.contains("voxcpm")));
    }

    #[test]
    fn scan_lora_entries_includes_none_and_manifest_dirs_sorted() {
        let tmp = tempdir().unwrap();
        let model = create_manifest_dir(tmp.path(), "voxcpm2-fp16");
        let lora_b = model.join("lora_b");
        let lora_a = model.join("lora_a");
        fs::create_dir_all(&lora_b).unwrap();
        fs::create_dir_all(&lora_a).unwrap();
        fs::write(lora_b.join("lora_manifest.json"), "{}").unwrap();
        fs::write(lora_a.join("lora_manifest.json"), "{}").unwrap();
        fs::create_dir_all(model.join("lora_without_manifest")).unwrap();

        let entries = scan_lora_entries(&model);

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], LoraEntry::none());
        assert_eq!(entries[1].name, "lora_a");
        assert_eq!(entries[2].name, "lora_b");
        assert!(entries[1].path.as_ref().unwrap().ends_with("lora_a"));
    }

    #[test]
    fn synthesis_args_builds_native_request_with_prompt_and_reference_paths() {
        let args = SynthesisArgs {
            index: 4,
            text: " hello   world ".to_string(),
            dit_steps: 7,
            prompt_wav_path: Some("for_test_wav/prompt.wav".to_string()),
            prompt_text: Some("prompt text".to_string()),
            reference_wav_path: Some("for_test_wav/reference.wav".to_string()),
        };

        let request = args.into_request();

        assert_eq!(request.text, " hello   world ");
        assert_eq!(request.inference_timesteps, 7);
        assert_eq!(request.prompt_text.as_deref(), Some("prompt text"));
        assert_eq!(
            request.prompt_wav_path.as_ref().unwrap(),
            &std::path::PathBuf::from("for_test_wav/prompt.wav")
        );
        assert_eq!(
            request.reference_wav_path.as_ref().unwrap(),
            &std::path::PathBuf::from("for_test_wav/reference.wav")
        );
    }

    #[test]
    fn empty_optional_strings_do_not_create_paths() {
        let args = SynthesisArgs {
            index: 0,
            text: "hello".to_string(),
            dit_steps: 10,
            prompt_wav_path: Some("   ".to_string()),
            prompt_text: Some("   ".to_string()),
            reference_wav_path: Some(String::new()),
        };

        let request = args.into_request();

        assert!(request.prompt_wav_path.is_none());
        assert!(request.prompt_text.is_none());
        assert!(request.reference_wav_path.is_none());
    }
}
```

- [ ] **Step 5: Register the module**

In `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`, add:

```rust
mod desktop_core;
```

above the existing module declarations.

- [ ] **Step 6: Verify tests pass**

Run:

```powershell
cargo test -p voxui-desktop desktop_core -- --nocapture
```

Expected: four tests pass.

- [ ] **Step 7: Commit**

```powershell
git add voxui/Cargo.lock voxui/crates/voxui-desktop/src-tauri/Cargo.toml voxui/crates/voxui-desktop/src-tauri/src/desktop_core.rs voxui/crates/voxui-desktop/src-tauri/src/lib.rs
git commit -m "test(desktop): cover TTS command helper behavior"
```

---

## Task 2: Make Desktop State Config and Busy Guard Testable

**Files:**
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/state.rs`

- [ ] **Step 1: Write failing tests for config persistence and synthesis busy guard**

Add this test module to the bottom of `state.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn config_round_trips_desktop_tts_fields() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("voxui_config.json");
        let config = AppConfig {
            model_dir: "models/voxcpm2-fp16".to_string(),
            lora_dir: Some("models/voxcpm2-fp16/lora_ft2".to_string()),
            prompt_wav_path: Some("for_test_wav/prompt.wav".to_string()),
            prompt_text: Some("prompt text".to_string()),
            reference_wav_path: Some("for_test_wav/reference.wav".to_string()),
            backend: "CUDA".to_string(),
            audio_host: "Wasapi".to_string(),
            audio_device: "Speakers".to_string(),
            max_chars: 120,
            dit_steps: 12,
            language: "English".to_string(),
        };

        config.save_to_path(&path).unwrap();
        let loaded = AppConfig::load_from_path(&path);

        assert_eq!(loaded.model_dir, config.model_dir);
        assert_eq!(loaded.lora_dir, config.lora_dir);
        assert_eq!(loaded.prompt_wav_path, config.prompt_wav_path);
        assert_eq!(loaded.prompt_text, config.prompt_text);
        assert_eq!(loaded.reference_wav_path, config.reference_wav_path);
        assert_eq!(loaded.backend, "CUDA");
        assert_eq!(loaded.dit_steps, 12);
    }

    #[test]
    fn busy_guard_rejects_second_synthesis_until_dropped() {
        let state = AppState::new();

        let first = state.try_begin_synthesis();
        assert!(first.is_ok());
        assert!(state.try_begin_synthesis().is_err());

        drop(first);
        assert!(state.try_begin_synthesis().is_ok());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cargo test -p voxui-desktop state::tests -- --nocapture
```

Expected: compile failure for missing `save_to_path`, `load_from_path`, or `try_begin_synthesis`.

- [ ] **Step 3: Replace `state.rs` with shared state, config helpers, and busy guard**

Use this implementation:

```rust
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use voxui_audio::AudioSystem;
use voxui_inference::VoxCPMEngine;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AppConfig {
    #[serde(default = "default_model_dir")]
    pub model_dir: String,
    #[serde(default)]
    pub lora_dir: Option<String>,
    #[serde(default)]
    pub prompt_wav_path: Option<String>,
    #[serde(default)]
    pub prompt_text: Option<String>,
    #[serde(default)]
    pub reference_wav_path: Option<String>,
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default)]
    pub audio_host: String,
    #[serde(default)]
    pub audio_device: String,
    #[serde(default = "default_max_chars")]
    pub max_chars: usize,
    #[serde(default = "default_dit_steps")]
    pub dit_steps: usize,
    #[serde(default = "default_language")]
    pub language: String,
}

fn default_model_dir() -> String {
    "models".to_string()
}

fn default_backend() -> String {
    "CUDA".to_string()
}

fn default_max_chars() -> usize {
    120
}

fn default_dit_steps() -> usize {
    10
}

fn default_language() -> String {
    "Chinese".to_string()
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model_dir: default_model_dir(),
            lora_dir: None,
            prompt_wav_path: None,
            prompt_text: None,
            reference_wav_path: None,
            backend: default_backend(),
            audio_host: String::new(),
            audio_device: String::new(),
            max_chars: default_max_chars(),
            dit_steps: default_dit_steps(),
            language: default_language(),
        }
    }
}

impl AppConfig {
    pub fn config_path() -> PathBuf {
        PathBuf::from("voxui_config.json")
    }

    pub fn load() -> Self {
        Self::load_from_path(&Self::config_path())
    }

    pub fn load_from_path(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|contents| serde_json::from_str(&contents).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) -> Result<()> {
        self.save_to_path(&Self::config_path())
    }

    pub fn save_to_path(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

pub struct AppState {
    pub engine: Arc<Mutex<Option<VoxCPMEngine>>>,
    pub audio_system: AudioSystem,
    pub config: Arc<Mutex<AppConfig>>,
    synthesis_busy: Arc<AtomicBool>,
}

pub struct SynthesisBusyGuard {
    busy: Arc<AtomicBool>,
}

impl Drop for SynthesisBusyGuard {
    fn drop(&mut self) {
        self.busy.store(false, Ordering::Release);
    }
}

impl AppState {
    pub fn new() -> Self {
        Self {
            engine: Arc::new(Mutex::new(None)),
            audio_system: AudioSystem::new(),
            config: Arc::new(Mutex::new(AppConfig::load())),
            synthesis_busy: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn try_begin_synthesis(&self) -> Result<SynthesisBusyGuard, String> {
        self.synthesis_busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| SynthesisBusyGuard {
                busy: Arc::clone(&self.synthesis_busy),
            })
            .map_err(|_| "synthesis is already running".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn config_round_trips_desktop_tts_fields() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("voxui_config.json");
        let config = AppConfig {
            model_dir: "models/voxcpm2-fp16".to_string(),
            lora_dir: Some("models/voxcpm2-fp16/lora_ft2".to_string()),
            prompt_wav_path: Some("for_test_wav/prompt.wav".to_string()),
            prompt_text: Some("prompt text".to_string()),
            reference_wav_path: Some("for_test_wav/reference.wav".to_string()),
            backend: "CUDA".to_string(),
            audio_host: "Wasapi".to_string(),
            audio_device: "Speakers".to_string(),
            max_chars: 120,
            dit_steps: 12,
            language: "English".to_string(),
        };

        config.save_to_path(&path).unwrap();
        let loaded = AppConfig::load_from_path(&path);

        assert_eq!(loaded.model_dir, config.model_dir);
        assert_eq!(loaded.lora_dir, config.lora_dir);
        assert_eq!(loaded.prompt_wav_path, config.prompt_wav_path);
        assert_eq!(loaded.prompt_text, config.prompt_text);
        assert_eq!(loaded.reference_wav_path, config.reference_wav_path);
        assert_eq!(loaded.backend, "CUDA");
        assert_eq!(loaded.dit_steps, 12);
    }

    #[test]
    fn busy_guard_rejects_second_synthesis_until_dropped() {
        let state = AppState::new();

        let first = state.try_begin_synthesis();
        assert!(first.is_ok());
        assert!(state.try_begin_synthesis().is_err());

        drop(first);
        assert!(state.try_begin_synthesis().is_ok());
    }
}
```

- [ ] **Step 4: Verify tests pass**

Run:

```powershell
cargo test -p voxui-desktop state::tests -- --nocapture
```

Expected: two tests pass.

- [ ] **Step 5: Commit**

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src/state.rs
git commit -m "feat(desktop): persist TTS config and guard synthesis"
```

---

## Task 3: Refactor Tauri Commands Around Native Inference

**Files:**
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/commands.rs`
- Modify: `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`

- [ ] **Step 1: Replace command DTOs and command names**

At the top of `commands.rs`, import the helper types:

```rust
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};
use voxui_audio::{AudioPlayer, AudioSystem};
use voxui_inference::VoxCPMEngine;

use crate::desktop_core::{
    discover_models_root, scan_lora_entries, scan_model_entries, LoraEntry, ModelEntry,
    SynthesisArgs,
};
use crate::state::{AppConfig, AppState};
```

Replace `ModelInfo`, `AudioDeviceList`, and `ProgressPayload` with:

```rust
#[derive(Serialize, Clone)]
pub struct ModelInfo {
    pub architecture: String,
    pub sample_rate: u32,
    pub backend: String,
    pub warning: Option<String>,
}

#[derive(Serialize, Clone)]
pub struct AudioDeviceList {
    pub hosts: Vec<String>,
    pub selected_host: String,
    pub devices: Vec<String>,
    pub selected_device: String,
}

#[derive(Clone, Serialize)]
struct ProgressPayload {
    step: u32,
    total: u32,
    index: u32,
}

#[derive(Clone, Serialize)]
struct ErrorPayload {
    index: u32,
    message: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApplyLoraArgs {
    pub lora_dir: Option<String>,
}
```

- [ ] **Step 2: Replace model and LoRA listing commands**

Use:

```rust
#[tauri::command]
pub fn list_models() -> Vec<ModelEntry> {
    scan_model_entries(&discover_models_root())
}

#[tauri::command]
pub fn list_lora_dirs(model_dir: String) -> Vec<LoraEntry> {
    scan_lora_entries(&PathBuf::from(model_dir))
}
```

- [ ] **Step 3: Replace audio device listing**

Use:

```rust
#[tauri::command]
pub fn list_audio_devices(state: State<AppState>, host: Option<String>) -> AudioDeviceList {
    let hosts: Vec<String> = state.audio_system.hosts().iter().map(|h| h.name.clone()).collect();
    let selected_host = host
        .filter(|name| hosts.iter().any(|known| known == name))
        .unwrap_or_else(|| state.audio_system.default_host_name());
    let devices = state
        .audio_system
        .devices(&selected_host)
        .map(|devs| devs.into_iter().map(|device| device.name).collect::<Vec<_>>())
        .unwrap_or_default();
    let selected_device = state
        .audio_system
        .default_device_name(&selected_host)
        .ok()
        .filter(|device| devices.iter().any(|known| known == device))
        .or_else(|| devices.first().cloned())
        .unwrap_or_default();

    AudioDeviceList {
        hosts,
        selected_host,
        devices,
        selected_device,
    }
}
```

- [ ] **Step 4: Replace `load_model` with an async blocking task**

Use:

```rust
#[tauri::command]
pub async fn load_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model_dir: String,
    backend: String,
) -> Result<ModelInfo, String> {
    let model_path = PathBuf::from(&model_dir);
    let (device, actual_backend, warning) = select_device(&backend);
    let engine_slot = Arc::clone(&state.engine);

    let engine = tokio::task::spawn_blocking(move || VoxCPMEngine::load(&model_path, device))
        .await
        .map_err(|e| format!("model load task failed: {e}"))?
        .map_err(|e| format!("model load failed: {e}"))?;

    let info = ModelInfo {
        architecture: engine.architecture().to_string(),
        sample_rate: engine.sample_rate(),
        backend: actual_backend,
        warning,
    };

    *engine_slot.lock().map_err(|_| "engine lock poisoned".to_string())? = Some(engine);
    let _ = app.emit("engine-ready", info.clone());
    Ok(info)
}
```

- [ ] **Step 5: Replace separate LoRA commands with `apply_lora`**

Use:

```rust
#[tauri::command]
pub fn apply_lora(state: State<AppState>, args: ApplyLoraArgs) -> Result<(), String> {
    let mut guard = state
        .engine
        .lock()
        .map_err(|_| "engine lock poisoned".to_string())?;
    let engine = guard.as_mut().ok_or("Engine not loaded")?;

    match args.lora_dir {
        Some(path) if !path.trim().is_empty() => engine
            .load_lora(&PathBuf::from(path.trim()))
            .map_err(|e| format!("LoRA load failed: {e}")),
        _ => {
            engine.unload_lora();
            Ok(())
        }
    }
}
```

- [ ] **Step 6: Replace `synthesize` with native `SynthesisArgs` and playback-only completion**

Use:

```rust
#[tauri::command]
pub async fn synthesize(
    app: AppHandle,
    state: State<'_, AppState>,
    args: SynthesisArgs,
) -> Result<(), String> {
    let busy = state.try_begin_synthesis()?;
    let index = args.index;
    let request = args.into_request();
    let config = state
        .config
        .lock()
        .map_err(|_| "config lock poisoned".to_string())?
        .clone();
    let engine_slot = Arc::clone(&state.engine);
    let app_for_task = app.clone();

    let result = tokio::task::spawn_blocking(move || {
        let (samples, sample_rate) = {
            let mut guard = engine_slot.lock().map_err(|_| "engine lock poisoned".to_string())?;
            let engine = guard.as_mut().ok_or_else(|| "Engine not loaded".to_string())?;
            let sample_rate = engine.sample_rate();
            let app_for_progress = app_for_task.clone();
            let samples = engine
                .generate(request, |step, total| {
                    let _ = app_for_progress.emit(
                        "tts-progress",
                        ProgressPayload {
                            step: step as u32,
                            total: total as u32,
                            index,
                        },
                    );
                })
                .map_err(|e| format!("synthesis failed: {e}"))?;
            (samples, sample_rate)
        };

        let (host, device) = resolve_audio_output(&config)?;
        let mut player = AudioPlayer::new(&host, &device, sample_rate)
            .map_err(|e| format!("audio init failed: {e}"))?;
        player
            .play_blocking(samples)
            .map_err(|e| format!("playback failed: {e}"))?;
        Ok::<(), String>(())
    })
    .await
    .map_err(|e| format!("synthesis task failed: {e}"))?;

    drop(busy);

    match result {
        Ok(()) => {
            let _ = app.emit("tts-complete", serde_json::json!({ "index": index }));
            Ok(())
        }
        Err(message) => {
            let _ = app.emit(
                "tts-error",
                ErrorPayload {
                    index,
                    message: message.clone(),
                },
            );
            Err(message)
        }
    }
}
```

- [ ] **Step 7: Add command helper functions**

Add to the bottom of `commands.rs`:

```rust
fn resolve_audio_output(config: &AppConfig) -> Result<(String, String), String> {
    let audio_system = AudioSystem::new();
    let host = if config.audio_host.trim().is_empty() {
        audio_system.default_host_name()
    } else {
        config.audio_host.clone()
    };
    let device = if config.audio_device.trim().is_empty() {
        audio_system
            .default_device_name(&host)
            .map_err(|e| format!("default audio device lookup failed: {e}"))?
    } else {
        config.audio_device.clone()
    };
    Ok((host, device))
}

fn select_device(requested: &str) -> (candle_core::Device, String, Option<String>) {
    match requested {
        "CUDA" => select_cuda_device(),
        _ => (candle_core::Device::Cpu, "CPU".to_string(), None),
    }
}

#[cfg(feature = "cuda")]
fn select_cuda_device() -> (candle_core::Device, String, Option<String>) {
    match candle_core::Device::new_cuda(0) {
        Ok(device) => (device, "CUDA".to_string(), None),
        Err(err) => (
            candle_core::Device::Cpu,
            "CPU".to_string(),
            Some(format!("CUDA unavailable, using CPU: {err}")),
        ),
    }
}

#[cfg(not(feature = "cuda"))]
fn select_cuda_device() -> (candle_core::Device, String, Option<String>) {
    (
        candle_core::Device::Cpu,
        "CPU".to_string(),
        Some("CUDA was requested, but this build was compiled without CUDA support".to_string()),
    )
}
```

- [ ] **Step 8: Update Tauri command registration**

In `voxui/crates/voxui-desktop/src-tauri/src/lib.rs`, replace `load_lora` and `unload_lora` registration with `apply_lora`:

```rust
.invoke_handler(tauri::generate_handler![
    commands::list_models,
    commands::list_lora_dirs,
    commands::list_audio_devices,
    commands::load_model,
    commands::apply_lora,
    commands::synthesize,
    commands::get_config,
    commands::save_config,
])
```

- [ ] **Step 9: Verify backend compile**

Run:

```powershell
cargo check -p voxui-desktop
```

Expected: command backend compiles.

- [ ] **Step 10: Commit**

```powershell
git add voxui/crates/voxui-desktop/src-tauri/src/commands.rs voxui/crates/voxui-desktop/src-tauri/src/lib.rs
git commit -m "feat(desktop): route Tauri commands to native TTS"
```

---

## Task 4: Wire Frontend Types and Tauri Invokes

**Files:**
- Modify: `voxui/crates/voxui-desktop/src/app.rs`
- Modify: `voxui/crates/voxui-desktop/src/tauri_api.rs`

- [ ] **Step 1: Add frontend DTOs matching the backend**

In `app.rs`, replace the old argument and response structs with:

```rust
#[derive(Clone, Debug, Deserialize)]
struct ModelEntry {
    name: String,
    path: String,
}

#[derive(Clone, Debug, Deserialize)]
struct LoraEntry {
    name: String,
    path: Option<String>,
}

#[derive(Serialize)]
struct SynthesizeArgs {
    args: SynthesisPayload,
}

#[derive(Serialize)]
struct SynthesisPayload {
    index: u32,
    text: String,
    dit_steps: usize,
    prompt_wav_path: Option<String>,
    prompt_text: Option<String>,
    reference_wav_path: Option<String>,
}

#[derive(Serialize)]
struct LoadModelArgs {
    model_dir: String,
    backend: String,
}

#[derive(Serialize)]
struct ListLoraArgs {
    model_dir: String,
}

#[derive(Serialize)]
struct ListAudioDevicesArgs {
    host: Option<String>,
}

#[derive(Serialize)]
struct ApplyLoraArgs {
    args: ApplyLoraPayload,
}

#[derive(Serialize)]
struct ApplyLoraPayload {
    lora_dir: Option<String>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct AppConfig {
    pub model_dir: String,
    pub lora_dir: Option<String>,
    pub prompt_wav_path: Option<String>,
    pub prompt_text: Option<String>,
    pub reference_wav_path: Option<String>,
    pub backend: String,
    pub audio_host: String,
    pub audio_device: String,
    pub max_chars: usize,
    pub dit_steps: usize,
    pub language: String,
}

#[derive(Deserialize, Debug)]
struct ModelInfo {
    architecture: String,
    sample_rate: u32,
    backend: String,
    warning: Option<String>,
}

#[derive(Deserialize, Debug)]
struct AudioDeviceList {
    hosts: Vec<String>,
    selected_host: String,
    devices: Vec<String>,
    selected_device: String,
}

#[derive(Deserialize, Debug)]
struct ProgressPayload {
    step: u32,
    total: u32,
    index: u32,
}

#[derive(Deserialize, Debug)]
struct ErrorPayload {
    index: u32,
    message: String,
}
```

- [ ] **Step 2: Add signals for prompt/reference config and richer status**

In `App()`, add signals beside existing config signals:

```rust
let (prompt_wav_path, set_prompt_wav_path) = signal(String::new());
let (prompt_text, set_prompt_text) = signal(String::new());
let (reference_wav_path, set_reference_wav_path) = signal(String::new());
let (actual_backend, set_actual_backend) = signal(String::new());
let (status_message, set_status_message) = signal(String::new());
```

Change model and LoRA option signals:

```rust
let (models, set_models) = signal(Vec::<ModelEntry>::new());
let (loras, set_loras) = signal(Vec::<LoraEntry>::new());
```

- [ ] **Step 3: Add string option helpers**

Add this helper near the DTOs in `app.rs`:

```rust
fn non_empty_option(value: String) -> Option<String> {
    let trimmed = value.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}
```

- [ ] **Step 4: Update initialization config loading**

In the config load block, after reading `config`, set the new fields:

```rust
set_prompt_wav_path.set(config.prompt_wav_path.unwrap_or_default());
set_prompt_text.set(config.prompt_text.unwrap_or_default());
set_reference_wav_path.set(config.reference_wav_path.unwrap_or_default());
set_actual_backend.set(config.backend.clone());
```

- [ ] **Step 5: Update model listing and initial selection**

Replace the old `Vec<String>` model selection logic with:

```rust
if let Ok(model_list) = tauri_api::invoke_no_args::<Vec<ModelEntry>>("list_models").await {
    if model_list.is_empty() {
        set_no_model.set(true);
        set_status.set("idle".into());
        return;
    }
    let configured = model_dir.get_untracked();
    let selected = model_list
        .iter()
        .find(|entry| entry.path == configured)
        .cloned()
        .unwrap_or_else(|| model_list[0].clone());
    set_model_dir.set(selected.path.clone());
    set_model_name.set(selected.name.clone());
    set_models.set(model_list);
}
```

- [ ] **Step 6: Update LoRA listing**

Replace the old LoRA list invocation with:

```rust
let md = model_dir.get_untracked();
if let Ok(lora_list) = tauri_api::invoke::<_, Vec<LoraEntry>>(
    "list_lora_dirs",
    &ListLoraArgs { model_dir: md.clone() },
).await {
    set_loras.set(lora_list);
}
```

- [ ] **Step 7: Update audio device loading**

Replace the old audio device load with:

```rust
if let Ok(audio) = tauri_api::invoke::<_, AudioDeviceList>(
    "list_audio_devices",
    &ListAudioDevicesArgs {
        host: non_empty_option(audio_host.get_untracked()),
    },
).await {
    set_hosts.set(audio.hosts);
    set_audio_host.set(audio.selected_host);
    set_devices.set(audio.devices);
    set_audio_device.set(audio.selected_device);
}
```

- [ ] **Step 8: Update model loading and apply LoRA after load**

When loading the model, update `actual_backend` from `ModelInfo` and call `apply_lora`:

```rust
match tauri_api::invoke::<_, ModelInfo>(
    "load_model",
    &LoadModelArgs { model_dir: md, backend: be },
).await {
    Ok(info) => {
        set_engine_ready.set(true);
        set_actual_backend.set(info.backend.clone());
        set_status.set("ready".into());
        set_status_message.set(info.warning.unwrap_or_default());
        let lora_path = non_empty_option(lora_dir.get_untracked());
        let _ = tauri_api::invoke_unit(
            "apply_lora",
            &ApplyLoraArgs {
                args: ApplyLoraPayload { lora_dir: lora_path },
            },
        )
        .await;
    }
    Err(e) => {
        set_engine_ready.set(false);
        set_status.set(format!("Error: {}", e));
        set_status_message.set(e);
    }
}
```

- [ ] **Step 9: Update synthesis invocation**

Replace the old `invoke_unit("synthesize", &SynthesizeArgs { ... })` call with:

```rust
let payload = SynthesisPayload {
    index: idx,
    text: trimmed,
    dit_steps: steps as usize,
    prompt_wav_path: non_empty_option(prompt_wav_path.get_untracked()),
    prompt_text: non_empty_option(prompt_text.get_untracked()),
    reference_wav_path: non_empty_option(reference_wav_path.get_untracked()),
};

if let Err(e) = tauri_api::invoke_unit("synthesize", &SynthesizeArgs { args: payload }).await {
    web_sys::console::error_1(&format!("Synthesize error: {}", e).into());
}
```

- [ ] **Step 10: Add `tts-error` listener**

Add this event listener next to the completion listener:

```rust
{
    let set_status = set_status.clone();
    let set_progress = set_progress.clone();
    let set_history = set_history.clone();
    spawn_local(async move {
        let error_cb = Closure::new(move |val: JsValue| {
            if let Ok(payload) = serde_wasm_bindgen::from_value::<ErrorPayload>(val) {
                set_status.set("ready".into());
                set_progress.set(0.0);
                set_history.update(|history| {
                    if let Some(entry) = history.get_mut(payload.index as usize) {
                        entry.status = format!("error: {}", payload.message);
                        entry.progress = 0.0;
                    }
                });
            }
        });
        let _ = tauri_api::tauri_listen("tts-error", &error_cb).await;
        error_cb.forget();
    });
}
```

- [ ] **Step 11: Verify frontend type check**

Run from `voxui/crates/voxui-desktop`:

```powershell
rustup target add wasm32-unknown-unknown
cargo check --target wasm32-unknown-unknown
```

Expected: frontend crate compiles for WASM.

- [ ] **Step 12: Commit**

```powershell
git add voxui/crates/voxui-desktop/src/app.rs voxui/crates/voxui-desktop/src/tauri_api.rs
git commit -m "feat(frontend): send native synthesis requests"
```

---

## Task 5: Complete Desktop Settings and TTS UI

**Files:**
- Modify: `voxui/crates/voxui-desktop/src/components/settings_modal.rs`
- Modify: `voxui/crates/voxui-desktop/src/components/input_box.rs`
- Modify: `voxui/crates/voxui-desktop/src/components/history.rs`
- Modify: `voxui/crates/voxui-desktop/src/components/status_bar.rs`
- Modify: `voxui/crates/voxui-desktop/src/i18n.rs`
- Modify: `voxui/crates/voxui-desktop/src/app.rs`

- [ ] **Step 1: Update settings value type**

In `settings_modal.rs`, replace `SettingsValues` with:

```rust
#[derive(Clone, Debug)]
pub struct SettingsValues {
    pub model_dir: String,
    pub lora_dir: String,
    pub backend: String,
    pub audio_host: String,
    pub audio_device: String,
    pub max_chars: usize,
    pub dit_steps: usize,
    pub prompt_wav_path: String,
    pub prompt_text: String,
    pub reference_wav_path: String,
    pub language: String,
}
```

- [ ] **Step 2: Change settings props to use model/LoRA entries**

At the top of `settings_modal.rs`, import frontend entry types from `app.rs`:

```rust
use crate::app::{LoraEntry, ModelEntry};
```

Change prop types:

```rust
prompt_wav_path: ReadSignal<String>,
prompt_text: ReadSignal<String>,
reference_wav_path: ReadSignal<String>,
models: ReadSignal<Vec<ModelEntry>>,
loras: ReadSignal<Vec<LoraEntry>>,
```

Make `ModelEntry` and `LoraEntry` public in `app.rs`:

```rust
pub struct ModelEntry {
    pub name: String,
    pub path: String,
}

pub struct LoraEntry {
    pub name: String,
    pub path: Option<String>,
}
```

- [ ] **Step 3: Add settings modal signals for prompt/reference/language**

Inside `SettingsModal`, add:

```rust
let (sel_prompt_wav, set_sel_prompt_wav) = signal(prompt_wav_path.get_untracked());
let (sel_prompt_text, set_sel_prompt_text) = signal(prompt_text.get_untracked());
let (sel_reference_wav, set_sel_reference_wav) = signal(reference_wav_path.get_untracked());
let (sel_language, set_sel_language) = signal(match lang.get_untracked() {
    Language::Chinese => "Chinese".to_string(),
    Language::English => "English".to_string(),
});
```

- [ ] **Step 4: Update model and LoRA option rendering**

For models:

```rust
<For
    each=move || models.get()
    key=|model| model.path.clone()
    children=move |model| {
        let selected = model.path == sel_model.get();
        view! { <option value={model.path.clone()} selected=selected>{model.name}</option> }
    }
/>
```

For LoRA:

```rust
<For
    each=move || loras.get()
    key=|lora| lora.path.clone().unwrap_or_else(|| "None".to_string())
    children=move |lora| {
        let value = lora.path.clone().unwrap_or_default();
        let selected = value == sel_lora.get() || (value.is_empty() && sel_lora.get() == "None");
        view! { <option value={value} selected=selected>{lora.name}</option> }
    }
/>
```

- [ ] **Step 5: Add prompt/reference fields to settings modal**

Below diffusion steps, add:

```rust
<SettingsField label=move || lang.get().t("prompt_wav")>
    <input
        type="text"
        class="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm"
        prop:value=move || sel_prompt_wav.get()
        on:input=move |ev| set_sel_prompt_wav.set(event_target_value(&ev))
    />
</SettingsField>

<SettingsField label=move || lang.get().t("prompt_text")>
    <textarea
        class="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm min-h-16 resize-y"
        prop:value=move || sel_prompt_text.get()
        on:input=move |ev| set_sel_prompt_text.set(event_target_value(&ev))
    />
</SettingsField>

<SettingsField label=move || lang.get().t("reference_wav")>
    <input
        type="text"
        class="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm"
        prop:value=move || sel_reference_wav.get()
        on:input=move |ev| set_sel_reference_wav.set(event_target_value(&ev))
    />
</SettingsField>

<SettingsField label=move || lang.get().t("language")>
    <select
        class="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm"
        on:change=move |ev| set_sel_language.set(event_target_value(&ev))
    >
        <option value="Chinese" selected=move || sel_language.get() == "Chinese">"中文"</option>
        <option value="English" selected=move || sel_language.get() == "English">"English"</option>
    </select>
</SettingsField>
```

- [ ] **Step 6: Include new settings fields in apply callback**

Replace the `on_apply(SettingsValues { ... })` construction with:

```rust
on_apply(SettingsValues {
    model_dir: sel_model.get(),
    lora_dir: sel_lora.get(),
    backend: sel_backend.get(),
    audio_host: sel_host.get(),
    audio_device: sel_device.get(),
    max_chars: sel_max_chars.get(),
    dit_steps: sel_dit_steps.get(),
    prompt_wav_path: sel_prompt_wav.get(),
    prompt_text: sel_prompt_text.get(),
    reference_wav_path: sel_reference_wav.get(),
    language: sel_language.get(),
});
```

- [ ] **Step 7: Update `App` settings modal props and apply handler**

Pass the new signals to `<SettingsModal ... />`:

```rust
prompt_wav_path=prompt_wav_path
prompt_text=prompt_text
reference_wav_path=reference_wav_path
```

In `on_apply_settings`, set the new signals:

```rust
set_prompt_wav_path.set(vals.prompt_wav_path.clone());
set_prompt_text.set(vals.prompt_text.clone());
set_reference_wav_path.set(vals.reference_wav_path.clone());
set_lang.set(if vals.language == "English" { Language::English } else { Language::Chinese });
```

In the config JSON, add:

```rust
"prompt_wav_path": non_empty_option(vals.prompt_wav_path.clone()),
"prompt_text": non_empty_option(vals.prompt_text.clone()),
"reference_wav_path": non_empty_option(vals.reference_wav_path.clone()),
"language": vals.language,
```

- [ ] **Step 8: Update input box to textarea**

In `input_box.rs`, replace the `<input type="text"... />` with:

```rust
<textarea
    class="flex-1 px-3 py-2 bg-gray-900 border border-gray-600 rounded text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 disabled:opacity-50 min-h-12 max-h-32 resize-y"
    placeholder=move || lang.get().t("input_placeholder")
    disabled=move || !engine_ready.get()
    prop:value=move || text.get()
    on:input=move |ev| {
        set_text.set(event_target_value(&ev));
    }
    on:keydown=handle_keydown
/>
```

- [ ] **Step 9: Show history errors explicitly**

In `history.rs`, change status matching to treat `error:` as error:

```rust
let is_error = entry.status.starts_with("error:");
let status_color = if is_error {
    "text-red-400"
} else {
    match entry.status.as_str() {
        "generating" => "text-yellow-400",
        "playing" => "text-green-400",
        "done" => "text-gray-500",
        _ => "text-gray-400",
    }
};
let status_icon = if is_error {
    "!"
} else {
    match entry.status.as_str() {
        "queued" => "...",
        "generating" => ">>",
        "playing" => ">|",
        "done" => "ok",
        _ => "-",
    }
};
```

Below the timestamp line, render the error message:

```rust
<Show when=move || is_error>
    <p class="text-xs text-red-300 whitespace-normal">{entry.status.trim_start_matches("error: ").to_string()}</p>
</Show>
```

- [ ] **Step 10: Update status bar props**

In `status_bar.rs`, add props:

```rust
actual_backend: ReadSignal<String>,
lora_dir: ReadSignal<String>,
audio_host: ReadSignal<String>,
audio_device: ReadSignal<String>,
status_message: ReadSignal<String>,
```

Render:

```rust
<span>{move || {
    let msg = status_message.get();
    if msg.is_empty() { status_text() } else { format!("{} - {}", status_text(), msg) }
}}</span>
<span>{move || {
    let m = model_name.get();
    if m.is_empty() {
        String::new()
    } else {
        let lora = lora_dir.get();
        let lora_text = if lora.is_empty() || lora == "None" { "LoRA: None".to_string() } else { format!("LoRA: {}", lora.rsplit(['/', '\\']).next().unwrap_or(&lora)) };
        format!("{} | {} | {} / {} | {}", m, actual_backend.get(), audio_host.get(), audio_device.get(), lora_text)
    }
}}</span>
```

- [ ] **Step 11: Replace mojibake i18n strings**

In `i18n.rs`, replace the Chinese match arms with valid UTF-8 strings:

```rust
(Language::Chinese, "history") => "语音合成历史",
(Language::Chinese, "input_placeholder") => "输入文字，按 Enter 生成语音...",
(Language::Chinese, "settings") => "设置",
(Language::Chinese, "model") => "模型",
(Language::Chinese, "lora") => "LoRA",
(Language::Chinese, "backend") => "推理后端",
(Language::Chinese, "audio_host") => "音频驱动",
(Language::Chinese, "audio_device") => "音频设备",
(Language::Chinese, "max_chars") => "最大字数",
(Language::Chinese, "dit_steps") => "扩散步数",
(Language::Chinese, "language") => "语言",
(Language::Chinese, "prompt_wav") => "Prompt 音频",
(Language::Chinese, "prompt_text") => "Prompt 文本",
(Language::Chinese, "reference_wav") => "参考音频",
(Language::Chinese, "apply") => "应用",
(Language::Chinese, "cancel") => "取消",
(Language::Chinese, "loading") => "正在加载模型...",
(Language::Chinese, "ready") => "就绪",
(Language::Chinese, "generating") => "生成中...",
(Language::Chinese, "send") => "生成",
(Language::Chinese, "no_model") => "未找到模型",
(Language::Chinese, "no_model_msg") => "请选择模型目录:",
(Language::Chinese, "model_dir") => "模型目录",
(Language::Chinese, "lora_dir") => "LoRA 目录",
(Language::Chinese, "none") => "无",
```

Add English keys:

```rust
(Language::English, "prompt_wav") => "Prompt WAV",
(Language::English, "prompt_text") => "Prompt Text",
(Language::English, "reference_wav") => "Reference WAV",
```

- [ ] **Step 12: Verify frontend and backend compile**

Run:

```powershell
cargo check -p voxui-desktop
```

Run from `voxui/crates/voxui-desktop`:

```powershell
cargo check --target wasm32-unknown-unknown
```

Expected: both compile.

- [ ] **Step 13: Commit**

```powershell
git add voxui/crates/voxui-desktop/src/app.rs voxui/crates/voxui-desktop/src/components/settings_modal.rs voxui/crates/voxui-desktop/src/components/input_box.rs voxui/crates/voxui-desktop/src/components/history.rs voxui/crates/voxui-desktop/src/components/status_bar.rs voxui/crates/voxui-desktop/src/i18n.rs
git commit -m "feat(desktop): expose usable TTS settings UI"
```

---

## Task 6: Make Desktop the Active App and Verify Build Paths

**Files:**
- Modify: `voxui/Cargo.toml`
- Modify: `voxui/crates/voxui-desktop/src-tauri/tauri.conf.json`
- Modify: `voxui/crates/voxui-desktop/Trunk.toml`

- [ ] **Step 1: Remove the TUI crate from workspace members**

In `voxui/Cargo.toml`, remove this line from `members`:

```toml
"crates/voxui-app",
```

Keep these members:

```toml
members = [
    "crates/voxui-gguf",
    "crates/voxui-inference",
    "crates/voxui-audio",
    "crates/voxui-desktop/src-tauri",
]
```

- [ ] **Step 2: Configure Tauri to run Trunk**

In `voxui/crates/voxui-desktop/src-tauri/tauri.conf.json`, set:

```json
"build": {
  "frontendDist": "../dist",
  "devUrl": "http://localhost:8080",
  "beforeDevCommand": "trunk serve --port 8080 --open=false",
  "beforeBuildCommand": "trunk build --release"
}
```

- [ ] **Step 3: Keep Trunk output stable**

In `voxui/crates/voxui-desktop/Trunk.toml`, keep:

```toml
[build]
target = "index.html"
dist = "dist"
```

- [ ] **Step 4: Verify workspace compile excludes the TUI**

Run from `voxui`:

```powershell
cargo check --workspace
```

Expected: checks `voxui-gguf`, `voxui-inference`, `voxui-audio`, and `voxui-desktop`; it does not build package `voxui-app`.

- [ ] **Step 5: Verify desktop CUDA feature compile**

Run from `voxui` with the user-provided CUDA environment:

```powershell
$env:PATH = "$env:USERPROFILE\scoop\apps\rustup\current\.cargo\bin;$env:PATH"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "$env:CUDA_PATH\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
$env:CUDA_COMPUTE_CAP = "89"
$env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"
cargo check -p voxui-desktop --features cuda
```

Expected: CUDA-enabled desktop backend compiles.

- [ ] **Step 6: Verify frontend build**

Run from `voxui/crates/voxui-desktop`:

```powershell
trunk build --release
```

Expected: `voxui/crates/voxui-desktop/dist/index.html` and WASM assets are generated.

- [ ] **Step 7: Verify Tauri backend check with generated frontend dist**

Run from `voxui`:

```powershell
cargo check -p voxui-desktop
```

Expected: desktop backend compiles with `../dist` present.

- [ ] **Step 8: Commit**

```powershell
git add voxui/Cargo.toml voxui/crates/voxui-desktop/src-tauri/tauri.conf.json voxui/crates/voxui-desktop/Trunk.toml
git commit -m "chore(app): make Tauri desktop the active app"
```

---

## Task 7: End-to-End Desktop Verification

**Files:**
- Verification task. Source edits occur only when a command reports a concrete defect.

- [ ] **Step 1: Run desktop backend unit tests**

Run from `voxui`:

```powershell
cargo test -p voxui-desktop -- --nocapture
```

Expected: all desktop backend unit tests pass.

- [ ] **Step 2: Run inference purity test**

Run from `voxui`:

```powershell
cargo test -p voxui-inference --test native_runtime_purity
```

Expected: native runtime purity test passes.

- [ ] **Step 3: Run core inference request tests**

Run from `voxui`:

```powershell
cargo test -p voxui-inference --test manifest_loader --test request_validation
```

Expected: manifest and request validation tests pass.

- [ ] **Step 4: Run release synthesis coverage on CPU**

Run from `voxui`:

```powershell
cargo test -p voxui-inference --release --test generate_flow_parity --test lora_parity -- --nocapture
cargo test -p voxui-inference --release --test inference_suite -- --nocapture --test-threads=1
```

Expected: VoxCPM 0.5, 1.5, and 2.0 CPU cases pass, including LoRA and VoxCPM 2 reference audio.

- [ ] **Step 5: Run CUDA synthesis coverage**

Run from `voxui` with the CUDA environment:

```powershell
$env:PATH = "$env:USERPROFILE\scoop\apps\rustup\current\.cargo\bin;$env:PATH"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "$env:CUDA_PATH\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
$env:CUDA_COMPUTE_CAP = "89"
$env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"
cargo test -p voxui-inference --release --features cuda --test inference_suite -- --nocapture --test-threads=1
```

Expected: CPU and CUDA matrix passes.

- [ ] **Step 6: Launch the desktop app for manual audio verification**

Run from `voxui/crates/voxui-desktop`:

```powershell
cargo tauri dev
```

Manual verification:

- Select a VoxCPM model from `models`.
- Select CPU or CUDA.
- Select an audio host and output device.
- Enter text and click generate.
- Confirm audio plays through the selected device.
- Apply a LoRA and synthesize again.
- For VoxCPM 2, set a reference WAV path from `for_test_wav` and synthesize again without reference text.
- Set prompt WAV and prompt text together and synthesize.

- [ ] **Step 7: Final status check**

Run from repository root:

```powershell
git status --short --branch
```

Expected: only intended generated artifacts or the pre-existing untracked `New Text Document.txt` remain uncommitted. Do not add `New Text Document.txt`.
