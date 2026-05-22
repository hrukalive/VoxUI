# VoxUI Desktop Manual Model Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit model selection/loading to `voxui-desktop`, with flattened model+LoRA choices, cancellable load progress, remembered selection, and no startup auto-load.

**Architecture:** Keep model discovery/config/load orchestration in the Tauri backend, and keep selected-vs-loaded state in the Leptos frontend. Backend model choice scanning returns flattened entries; loading builds a new engine first and only replaces the active engine after success. Frontend header owns model choice selection and the `Load`/`Cancel` control while Settings owns model root, backend, audio, prompt, and language settings.

**Tech Stack:** Rust workspace, Tauri 2 backend, Leptos 0.7 CSR frontend, WASM bindings through `window.__TAURI__`, Candle/voxui-inference, Tokio, existing Rust unit tests.

---

## File Structure

- Modify `crates/voxui-desktop/src-tauri/src/desktop_core.rs`: model root helper, flattened `ModelChoice`, choice scanning, UI load state helper tests.
- Modify `crates/voxui-desktop/src-tauri/src/state.rs`: config fields for `model_root` and `selected_model_choice_id`, default program-folder models path, config migration tests.
- Modify `crates/voxui-desktop/src-tauri/src/commands.rs`: replace model/LoRA listing with model choices, add load-choice args/progress payload, add folder-picker command, update load flow so old engine survives failed/cancelled loads.
- Modify `crates/voxui-desktop/src-tauri/src/lib.rs`: register new commands and the Tauri dialog plugin.
- Modify `crates/voxui-desktop/src-tauri/Cargo.toml`: add `tauri-plugin-dialog`.
- Modify `crates/voxui-desktop/src/app.rs`: remove auto-load startup path, add selected/loaded choice state, wire header model selector/load/cancel, consume new load progress events, persist config.
- Modify `crates/voxui-desktop/src/components/header.rs`: title, dropdown, Load/Cancel, disabled state.
- Modify `crates/voxui-desktop/src/components/settings_modal.rs`: add model root field and folder browse button, remove LoRA selector.
- Modify `crates/voxui-desktop/src/components/progress_bar.rs`: add a separate model-load progress component.
- Modify `crates/voxui-desktop/src/components/status_bar.rs`: display selected-vs-loaded choice state.
- Modify `crates/voxui-desktop/src/i18n.rs`: add `焓言焓语`, `AhanSays`, model folder/load/cancel/progress/no-model labels.
- Modify `crates/voxui-desktop/index.html` and `crates/voxui-desktop/src-tauri/tauri.conf.json`: update app/window title.

## Task 1: Backend Model Choice Discovery

**Files:**
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src-tauri\src\desktop_core.rs`

- [ ] **Step 1: Write failing tests for flattened model choices**

Add these tests inside the existing `#[cfg(test)] mod tests` in `desktop_core.rs`:

```rust
#[test]
fn scan_model_choices_flattens_base_and_lora_files_sorted() {
    let tmp = tempdir().unwrap();
    let model_b = create_model_dir(tmp.path(), "voxcpm2-q4-lm");
    fs::write(model_b.join("lora_ft2.gguf"), b"placeholder").unwrap();
    fs::write(model_b.join("lora_alpha.gguf"), b"placeholder").unwrap();
    create_model_dir(tmp.path(), "voxcpm05-fp16");
    fs::write(model_b.join("not_lora.gguf"), b"placeholder").unwrap();
    fs::create_dir_all(model_b.join("lora_old_dir.gguf")).unwrap();

    let choices = super::scan_model_choices(tmp.path());
    let names = choices
        .iter()
        .map(|choice| choice.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        names,
        vec![
            "voxcpm05-fp16",
            "voxcpm2-q4-lm",
            "voxcpm2-q4-lm | lora_alpha",
            "voxcpm2-q4-lm | lora_ft2",
        ]
    );
    assert!(choices[0].lora_path.is_none());
    assert!(choices[2].lora_path.as_ref().unwrap().ends_with("lora_alpha.gguf"));
}

#[test]
fn model_choice_id_is_relative_to_model_root_and_lora_file() {
    let tmp = tempdir().unwrap();
    let model = create_model_dir(tmp.path(), "voxcpm2-q4-lm");
    fs::write(model.join("lora_ft2.gguf"), b"placeholder").unwrap();

    let choices = super::scan_model_choices(tmp.path());

    assert!(choices.iter().any(|choice| choice.id == "voxcpm2-q4-lm"));
    assert!(choices
        .iter()
        .any(|choice| choice.id == "voxcpm2-q4-lm::lora_ft2.gguf"));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cargo test -p voxui-desktop desktop_core::tests::scan_model_choices_flattens_base_and_lora_files_sorted desktop_core::tests::model_choice_id_is_relative_to_model_root_and_lora_file
```

Expected: both tests fail because `scan_model_choices` and `ModelChoice` do not exist.

- [ ] **Step 3: Add `ModelChoice` and scanner**

In `desktop_core.rs`, keep `ModelEntry` and `LoraEntry` for compatibility while migrating the frontend, and add:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelChoice {
    pub id: String,
    pub name: String,
    pub model_dir: String,
    pub model_path: String,
    pub model_size_bytes: u64,
    pub lora_path: Option<String>,
    pub lora_size_bytes: Option<u64>,
}
```

Then add:

```rust
pub fn scan_model_choices(models_root: &Path) -> Vec<ModelChoice> {
    let mut choices = Vec::new();
    let mut model_dirs = fs::read_dir(models_root)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            let model_path = path.join("model.gguf");
            if !model_path.is_file() {
                return None;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            Some((name, path, model_path))
        })
        .collect::<Vec<_>>();

    model_dirs.sort_by(|left, right| left.0.cmp(&right.0));

    for (model_name, model_dir, model_path) in model_dirs {
        let model_size_bytes = file_size(&model_path);
        choices.push(ModelChoice {
            id: model_name.clone(),
            name: model_name.clone(),
            model_dir: display_path(&model_dir),
            model_path: display_path(&model_path),
            model_size_bytes,
            lora_path: None,
            lora_size_bytes: None,
        });

        let mut loras = fs::read_dir(&model_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let file_name = entry.file_name().to_string_lossy().into_owned();
                let is_lora_file = path.is_file()
                    && path.extension().and_then(|value| value.to_str()) == Some("gguf")
                    && file_name.starts_with("lora_");
                if !is_lora_file {
                    return None;
                }
                let stem = path.file_stem()?.to_string_lossy().into_owned();
                Some((file_name, stem, path))
            })
            .collect::<Vec<_>>();
        loras.sort_by(|left, right| left.0.cmp(&right.0));

        for (file_name, lora_name, lora_path) in loras {
            choices.push(ModelChoice {
                id: format!("{model_name}::{file_name}"),
                name: format!("{model_name} | {lora_name}"),
                model_dir: display_path(&model_dir),
                model_path: display_path(&model_path),
                model_size_bytes,
                lora_path: Some(display_path(&lora_path)),
                lora_size_bytes: Some(file_size(&lora_path)),
            });
        }
    }

    choices
}

fn file_size(path: &Path) -> u64 {
    fs::metadata(path).map(|metadata| metadata.len()).unwrap_or(0)
}
```

- [ ] **Step 4: Run tests to verify green**

Run:

```powershell
cargo test -p voxui-desktop desktop_core::tests::scan_model_choices_flattens_base_and_lora_files_sorted desktop_core::tests::model_choice_id_is_relative_to_model_root_and_lora_file
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- crates/voxui-desktop/src-tauri/src/desktop_core.rs
git commit -m "feat(desktop): scan flattened model choices"
```

## Task 2: Config Model Root And Selection Persistence

**Files:**
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src-tauri\src\state.rs`

- [ ] **Step 1: Write failing config tests**

Add these tests to `state.rs`:

```rust
#[test]
fn config_round_trips_model_root_and_selected_choice() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("voxui_config.json");
    let config = AppConfig {
        model_root: "D:/Models".to_string(),
        selected_model_choice_id: "voxcpm2-q4-lm::lora_ft2.gguf".to_string(),
        model_dir: "models/voxcpm2-q4-lm".to_string(),
        lora_dir: Some("models/voxcpm2-q4-lm/lora_ft2.gguf".to_string()),
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

    assert_eq!(loaded.model_root, "D:/Models");
    assert_eq!(loaded.selected_model_choice_id, "voxcpm2-q4-lm::lora_ft2.gguf");
}

#[test]
fn missing_model_root_uses_non_empty_default() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("voxui_config.json");
    fs::write(&path, r#"{"backend":"CPU"}"#).unwrap();

    let loaded = AppConfig::load_from_path(&path);

    assert!(!loaded.model_root.trim().is_empty());
    assert_eq!(loaded.selected_model_choice_id, "");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cargo test -p voxui-desktop state::tests::config_round_trips_model_root_and_selected_choice state::tests::missing_model_root_uses_non_empty_default
```

Expected: compile failure because `model_root` and `selected_model_choice_id` do not exist.

- [ ] **Step 3: Add config fields and defaults**

In `AppConfig`, add fields before legacy `model_dir`:

```rust
#[serde(default = "default_model_root")]
pub model_root: String,
#[serde(default)]
pub selected_model_choice_id: String,
```

Update `Default for AppConfig`:

```rust
model_root: default_model_root(),
selected_model_choice_id: String::new(),
```

Add:

```rust
fn default_model_root() -> String {
    default_program_models_dir()
        .unwrap_or_else(|| PathBuf::from("models"))
        .to_string_lossy()
        .replace('\\', "/")
}

pub fn default_program_models_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("models")))
}
```

Keep `default_model_dir()` for legacy config compatibility.

Update the existing `config_round_trips_desktop_tts_fields` test fixture with:

```rust
model_root: "models".to_string(),
selected_model_choice_id: "voxcpm2-fp16".to_string(),
```

- [ ] **Step 4: Run tests**

Run:

```powershell
cargo test -p voxui-desktop state::tests
```

Expected: all `state::tests` pass.

- [ ] **Step 5: Commit**

```powershell
git add -- crates/voxui-desktop/src-tauri/src/state.rs
git commit -m "feat(desktop): persist model root selection"
```

## Task 3: Backend Commands For Choices, Folder Browsing, And Safe Load Replacement

**Files:**
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src-tauri\src\commands.rs`
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src-tauri\src\lib.rs`
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src-tauri\Cargo.toml`

- [ ] **Step 1: Write failing tests for load button state helper**

Add this test to `desktop_core.rs` rather than `commands.rs`, so the button-state rule stays unit-testable:

```rust
#[test]
fn load_button_state_requires_selected_different_idle_choice() {
    assert!(super::load_button_enabled(
        Some("model-a"),
        None,
        super::ActivityState::Idle
    ));
    assert!(super::load_button_enabled(
        Some("model-b"),
        Some("model-a"),
        super::ActivityState::Idle
    ));
    assert!(!super::load_button_enabled(
        Some("model-a"),
        Some("model-a"),
        super::ActivityState::Idle
    ));
    assert!(!super::load_button_enabled(None, None, super::ActivityState::Idle));
    assert!(!super::load_button_enabled(
        Some("model-a"),
        None,
        super::ActivityState::Loading
    ));
    assert!(!super::load_button_enabled(
        Some("model-a"),
        None,
        super::ActivityState::Generating
    ));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cargo test -p voxui-desktop desktop_core::tests::load_button_state_requires_selected_different_idle_choice
```

Expected: compile failure until the helper and enum are added.

- [ ] **Step 3: Implement helper and run green**

Add this enum and function above the test module in `desktop_core.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityState {
    Idle,
    Loading,
    Generating,
}

pub fn load_button_enabled(
    selected_choice_id: Option<&str>,
    loaded_choice_id: Option<&str>,
    activity: ActivityState,
) -> bool {
    if activity != ActivityState::Idle {
        return false;
    }
    let Some(selected) = selected_choice_id.filter(|value| !value.trim().is_empty()) else {
        return false;
    };
    Some(selected) != loaded_choice_id
}
```

Run:

```powershell
cargo test -p voxui-desktop desktop_core::tests::load_button_state_requires_selected_different_idle_choice
```

Expected: pass.

- [ ] **Step 4: Add command data types and model-choice command**

In `commands.rs`, update imports:

```rust
use crate::desktop_core::{
    scan_model_choices, ModelChoice, SynthesisArgs,
};
```

Add:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct ListModelChoicesArgs {
    pub model_root: String,
}

#[tauri::command(rename_all = "snake_case")]
pub fn list_model_choices(model_root: String) -> Vec<ModelChoice> {
    let root = if model_root.trim().is_empty() {
        default_program_models_dir().unwrap_or_else(|| PathBuf::from("models"))
    } else {
        PathBuf::from(model_root)
    };
    tracing::debug!("list_model_choices model_root={}", root.display());
    scan_model_choices(&root)
}
```

Import the helper directly:

```rust
use crate::state::{default_program_models_dir, AppConfig, AppState};
```

and call `default_program_models_dir()` in `list_model_choices`.

- [ ] **Step 5: Add load-choice args and progress payload**

In `commands.rs`, add:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct LoadModelChoiceArgs {
    pub choice_id: String,
    pub model_dir: String,
    pub model_path: String,
    pub lora_path: Option<String>,
    pub backend: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct LoadProgressPayload {
    pub phase: String,
    pub file_label: Option<String>,
    pub bytes_read: u64,
    pub total_bytes: u64,
    pub backend: Option<String>,
}
```

- [ ] **Step 6: Add byte-read progress helper**

In `commands.rs`, add:

```rust
fn emit_read_progress(
    app: &AppHandle,
    file_label: &str,
    bytes_read: u64,
    total_bytes: u64,
) {
    let _ = app.emit(
        "load-progress",
        LoadProgressPayload {
            phase: "reading".to_string(),
            file_label: Some(file_label.to_string()),
            bytes_read,
            total_bytes,
            backend: None,
        },
    );
}

fn read_file_for_progress(
    app: &AppHandle,
    path: &PathBuf,
    file_label: &str,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<(), String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let total = file
        .metadata()
        .map_err(|err| format!("failed to read metadata for {}: {err}", path.display()))?
        .len();
    let mut read = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    emit_read_progress(app, file_label, 0, total);
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("model loading cancelled".to_string());
        }
        let count = file
            .read(&mut buffer)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        if count == 0 {
            break;
        }
        read += count as u64;
        emit_read_progress(app, file_label, read.min(total), total);
    }
    Ok(())
}

fn emit_device_loading(app: &AppHandle, backend: &str) {
    let _ = app.emit(
        "load-progress",
        LoadProgressPayload {
            phase: "device_loading".to_string(),
            file_label: None,
            bytes_read: 0,
            total_bytes: 0,
            backend: Some(backend.to_string()),
        },
    );
}
```

- [ ] **Step 7: Replace load command with load-choice flow**

Keep the existing `load_model` command for compatibility during this migration, and add the new command:

```rust
#[tauri::command(rename_all = "snake_case")]
pub async fn load_model_choice(
    app: AppHandle,
    state: State<'_, AppState>,
    args: LoadModelChoiceArgs,
) -> Result<ModelInfo, String> {
    tracing::debug!(
        "load_model_choice requested choice_id={} model_dir={} lora_path={:?} backend={}",
        args.choice_id,
        args.model_dir,
        args.lora_path,
        args.backend
    );
    let _busy = match state.try_begin_synthesis() {
        Ok(guard) => guard,
        Err(_) => return Err(engine_busy_message()),
    };
    state.cancel_load.store(false, Ordering::Release);
    let cancel_token = Arc::clone(&state.cancel_load);

    let started_at = Instant::now();
    let model_dir = PathBuf::from(&args.model_dir);
    let model_path = PathBuf::from(&args.model_path);
    let lora_path = args.lora_path.clone().map(PathBuf::from);
    let requested_backend = args.backend.clone();
    let choice_id = args.choice_id.clone();
    let (device, actual_backend, warning) = select_device(&requested_backend);
    let actual_backend_for_task = actual_backend.clone();
    let engine_slot = Arc::clone(&state.engine);
    let app_for_task = app.clone();

    let engine = match tokio::task::spawn_blocking(move || {
        read_file_for_progress(&app_for_task, &model_path, "model.gguf", &cancel_token)?;
        if let Some(path) = lora_path.as_ref() {
            let label = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("LoRA GGUF");
            read_file_for_progress(&app_for_task, path, label, &cancel_token)?;
        }
        emit_device_loading(&app_for_task, &actual_backend_for_task);
        if cancel_token.load(Ordering::Relaxed) {
            return Err("model loading cancelled".to_string());
        }

        let mut engine = VoxCPMEngine::load_with_progress(
            &model_dir,
            device,
            |_, _| {},
            Some(&cancel_token),
        )
        .map_err(|err| format!("{err:#}"))?;
        if let Some(path) = lora_path.as_ref() {
            if cancel_token.load(Ordering::Relaxed) {
                return Err("model loading cancelled".to_string());
            }
            engine
                .load_lora(path)
                .map_err(|err| format!("LoRA load failed: {err:#}"))?;
        }
        Ok::<_, String>(engine)
    })
    .await
    {
        Ok(Ok(engine)) => engine,
        Ok(Err(message)) => return Err(format!("model load failed: {message}")),
        Err(err) => return Err(format!("model load task failed: {err}")),
    };

    let info = ModelInfo {
        architecture: engine.architecture().to_string(),
        sample_rate: engine.sample_rate(),
        backend: actual_backend,
        warning,
    };

    *engine_slot
        .lock()
        .map_err(|_| "engine lock poisoned".to_string())? = Some(engine);
    tracing::debug!(
        "load_model_choice complete choice_id={} elapsed_seconds={:.3}",
        choice_id,
        started_at.elapsed().as_secs_f64()
    );
    let _ = app.emit("engine-ready", info.clone());
    Ok(info)
}
```

This command intentionally installs the engine only after load and LoRA application succeed.

- [ ] **Step 8: Add folder browser command**

Use a backend command so the frontend stays consistent with the current `window.__TAURI__.core.invoke` pattern.

Add dependency in `src-tauri/Cargo.toml`:

```toml
tauri-plugin-dialog = "2"
```

Register plugin in `lib.rs`:

```rust
.plugin(tauri_plugin_dialog::init())
```

Add command in `commands.rs`:

```rust
use tauri_plugin_dialog::DialogExt;

#[tauri::command]
pub async fn browse_model_root(app: AppHandle) -> Result<Option<String>, String> {
    tokio::task::spawn_blocking(move || {
        app.dialog()
            .file()
            .set_title("Select models folder")
            .blocking_pick_folder()
            .map(|path| {
                path.into_path()
                    .map(|path| path.to_string_lossy().replace('\\', "/"))
                    .map_err(|err| format!("selected folder is not a filesystem path: {err}"))
            })
            .transpose()
    })
    .await
    .map_err(|err| format!("folder browser task failed: {err}"))?
}
```

- [ ] **Step 9: Register commands**

In `lib.rs`, add to `generate_handler!`:

```rust
commands::list_model_choices,
commands::load_model_choice,
commands::browse_model_root,
```

Keep these legacy commands registered during migration:

```rust
commands::list_models,
commands::list_lora_dirs,
commands::load_model,
commands::apply_lora,
```

- [ ] **Step 10: Build backend tests**

Run:

```powershell
cargo test -p voxui-desktop desktop_core::tests
cargo test -p voxui-desktop state::tests
```

Expected: pass.

- [ ] **Step 11: Commit**

```powershell
git add -- crates/voxui-desktop/src-tauri/src/desktop_core.rs crates/voxui-desktop/src-tauri/src/commands.rs crates/voxui-desktop/src-tauri/src/lib.rs crates/voxui-desktop/src-tauri/Cargo.toml Cargo.lock
git commit -m "feat(desktop): load selected model choices"
```

## Task 4: Frontend Types, Startup, And Selection State

**Files:**
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src\app.rs`

- [ ] **Step 1: Add frontend model-choice types**

Replace or supplement `ModelEntry` and `LoraEntry` with:

```rust
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct ModelChoice {
    pub id: String,
    pub name: String,
    pub model_dir: String,
    pub model_path: String,
    pub model_size_bytes: u64,
    pub lora_path: Option<String>,
    pub lora_size_bytes: Option<u64>,
}
```

Update `AppConfig`:

```rust
pub model_root: String,
pub selected_model_choice_id: String,
```

Keep legacy fields during migration:

```rust
pub model_dir: String,
pub lora_dir: Option<String>,
```

Add arg structs:

```rust
#[derive(Serialize)]
struct ListModelChoicesArgs {
    model_root: String,
}

#[derive(Serialize)]
struct LoadModelChoiceArgs {
    args: LoadModelChoicePayload,
}

#[derive(Serialize)]
struct LoadModelChoicePayload {
    choice_id: String,
    model_dir: String,
    model_path: String,
    lora_path: Option<String>,
    backend: String,
}
```

- [ ] **Step 2: Replace load progress payload**

Replace the old step/total load progress type with:

```rust
#[derive(Deserialize, Debug)]
struct LoadProgressPayload {
    phase: String,
    file_label: Option<String>,
    bytes_read: u64,
    total_bytes: u64,
    backend: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LoadProgress {
    Hidden,
    Reading {
        label: String,
        bytes_read: u64,
        total_bytes: u64,
    },
    DeviceLoading {
        backend: String,
    },
}
```

- [ ] **Step 3: Add choice helpers**

Add pure helpers near existing helper functions:

```rust
fn selected_choice(
    choices: &[ModelChoice],
    selected_id: &str,
) -> Option<ModelChoice> {
    choices
        .iter()
        .find(|choice| choice.id == selected_id)
        .cloned()
        .or_else(|| choices.first().cloned())
}

fn choice_lora_label(choice: Option<&ModelChoice>) -> String {
    choice
        .and_then(|choice| choice.lora_path.as_ref())
        .and_then(|path| path.replace('\\', "/").rsplit('/').next().map(str::to_string))
        .and_then(|file| file.strip_suffix(".gguf").map(str::to_string).or(Some(file)))
        .unwrap_or_else(|| "None".to_string())
}
```

- [ ] **Step 4: Replace state variables**

Replace:

```rust
let (model_dir, set_model_dir) = signal(String::new());
let (lora_dir, set_lora_dir) = signal("None".to_string());
let (model_name, set_model_name) = signal(String::new());
let (models, set_models) = signal(Vec::<ModelEntry>::new());
let (loras, set_loras) = signal(Vec::<LoraEntry>::new());
```

with:

```rust
let (model_root, set_model_root) = signal(String::new());
let (selected_choice_id, set_selected_choice_id) = signal(String::new());
let (loaded_choice_id, set_loaded_choice_id) = signal(None::<String>);
let (model_choices, set_model_choices) = signal(Vec::<ModelChoice>::new());
let (load_progress, set_load_progress) = signal(LoadProgress::Hidden);
let (load_in_progress, set_load_in_progress) = signal(false);
```

Keep `engine_ready`, `backend`, `actual_backend`, and audio/prompt signals.

- [ ] **Step 5: Rewrite startup to scan but not load**

In the mount `spawn_local`, after config load:

```rust
set_model_root.set(config.model_root.clone());
set_selected_choice_id.set(config.selected_model_choice_id.clone());
```

Replace the old `list_models`, `list_loras_for_model`, and automatic `load_model` sections with:

```rust
debug_log("startup: model choices list start");
match tauri_api::invoke::<_, Vec<ModelChoice>>(
    "list_model_choices",
    &ListModelChoicesArgs {
        model_root: model_root.get_untracked(),
    },
)
.await
{
    Ok(choices) => {
        debug_log(&format!("startup: model choices count={}", choices.len()));
        if choices.is_empty() {
            set_no_model.set(true);
        }
        let selected = selected_choice(&choices, &selected_choice_id.get_untracked());
        if let Some(choice) = selected {
            set_selected_choice_id.set(choice.id.clone());
        }
        set_model_choices.set(choices);
        set_status.set("idle".into());
    }
    Err(e) => {
        debug_log(&format!("startup: model choices error {e}"));
        set_status_message.set(e);
        set_status.set("idle".into());
    }
}
```

Do not call `load_model` or `apply_lora_selection` on startup.

- [ ] **Step 6: Update load-progress event listener**

Replace the old `load_step` listener body with:

```rust
if let Ok(payload) = serde_wasm_bindgen::from_value::<LoadProgressPayload>(payload_value) {
    match payload.phase.as_str() {
        "reading" => set_load_progress.set(LoadProgress::Reading {
            label: payload.file_label.unwrap_or_else(|| "GGUF".to_string()),
            bytes_read: payload.bytes_read,
            total_bytes: payload.total_bytes,
        }),
        "device_loading" => set_load_progress.set(LoadProgress::DeviceLoading {
            backend: payload.backend.unwrap_or_else(|| "device".to_string()),
        }),
        _ => {}
    }
}
```

- [ ] **Step 7: Add load/cancel callbacks**

Add:

```rust
let on_choice_selected = move |choice_id: String| {
    set_selected_choice_id.set(choice_id);
    set_status_message.set(String::new());
};

let on_load_or_cancel = move |_| {
    if load_in_progress.get_untracked() {
        spawn_local(async move {
            let _ = tauri_api::invoke_no_args::<()>("cancel_load").await;
        });
        return;
    }

    let selected = selected_choice(&model_choices.get_untracked(), &selected_choice_id.get_untracked());
    let Some(choice) = selected else {
        set_status_message.set("No model selected".to_string());
        return;
    };

    set_load_in_progress.set(true);
    set_engine_ready.set(false);
    set_status.set("loading".into());
    set_status_message.set(String::new());
    set_load_progress.set(LoadProgress::Reading {
        label: "model.gguf".to_string(),
        bytes_read: 0,
        total_bytes: choice.model_size_bytes,
    });

    spawn_local(async move {
        let result = tauri_api::invoke::<_, ModelInfo>(
            "load_model_choice",
            &LoadModelChoiceArgs {
                args: LoadModelChoicePayload {
                    choice_id: choice.id.clone(),
                    model_dir: choice.model_dir.clone(),
                    model_path: choice.model_path.clone(),
                    lora_path: choice.lora_path.clone(),
                    backend: backend.get_untracked(),
                },
            },
        )
        .await;

        set_load_in_progress.set(false);
        set_load_progress.set(LoadProgress::Hidden);
        match result {
            Ok(info) => {
                set_engine_ready.set(true);
                set_loaded_choice_id.set(Some(choice.id.clone()));
                set_actual_backend.set(info.backend.clone());
                set_status.set("ready".into());
                set_status_message.set(info.warning.unwrap_or_default());
            }
            Err(e) => {
                let had_loaded = loaded_choice_id.get_untracked().is_some();
                set_engine_ready.set(had_loaded);
                set_status.set(if had_loaded { "ready".into() } else { "idle".into() });
                set_status_message.set(e);
            }
        }
    });
};
```

If Rust reports moved-value errors for signal handles in this closure, clone each signal handle before `spawn_local` and use the cloned handle inside the async block.

- [ ] **Step 8: Persist selected choice in save config**

When saving config, replace `model_dir` and `lora_dir` source values with the selected choice:

```rust
let selected = selected_choice(&model_choices.get_untracked(), &selected_choice_id.get_untracked());
let selected_model_dir = selected
    .as_ref()
    .map(|choice| choice.model_dir.clone())
    .unwrap_or_default();
let selected_lora_path = selected.as_ref().and_then(|choice| choice.lora_path.clone());
```

Save JSON fields:

```rust
"model_root": requested_model_root.clone(),
"selected_model_choice_id": selected_choice_id.get_untracked(),
"model_dir": selected_model_dir,
"lora_dir": selected_lora_path,
```

- [ ] **Step 9: Build frontend**

Run:

```powershell
cd crates/voxui-desktop
cargo check --target wasm32-unknown-unknown
```

Expected: backend command changes are complete. Frontend component prop errors are addressed in Tasks 5-7 before the UI commit.

- [ ] **Step 10: Commit after component tasks compile**

Do not commit this task until Tasks 5-7 compile together.

## Task 5: Header Dropdown And Load Button

**Files:**
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src\components\header.rs`
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src\app.rs`
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src\i18n.rs`

- [ ] **Step 1: Update i18n title and labels**

In `i18n.rs`, change:

```rust
(Language::Chinese, "title") => "焓言焓语",
(Language::English, "title") => "AhanSays",
```

Add keys:

```rust
(Language::Chinese, "load") => "加载",
(Language::English, "load") => "Load",
(Language::Chinese, "select_model") => "选择模型",
(Language::English, "select_model") => "Select model",
(Language::Chinese, "no_models_found") => "未找到模型",
(Language::English, "no_models_found") => "No models found",
```

- [ ] **Step 2: Replace Header props and markup**

Update `header.rs` imports:

```rust
use crate::app::ModelChoice;
use crate::i18n::Language;
use leptos::prelude::*;
```

Replace `Header` signature with:

```rust
#[component]
pub fn Header(
    lang: ReadSignal<Language>,
    choices: ReadSignal<Vec<ModelChoice>>,
    selected_choice_id: ReadSignal<String>,
    loaded_choice_id: ReadSignal<Option<String>>,
    load_in_progress: ReadSignal<bool>,
    generating: Signal<bool>,
    on_choice_selected: impl Fn(String) + 'static + Clone,
    on_load_or_cancel: impl Fn(()) + 'static + Clone,
    on_settings: impl Fn(()) + 'static,
) -> impl IntoView {
```

Add helper closure:

```rust
let can_load = move || {
    if load_in_progress.get() || generating.get() {
        return false;
    }
    let selected = selected_choice_id.get();
    !selected.is_empty() && Some(selected) != loaded_choice_id.get()
};
```

Replace view with:

```rust
view! {
    <header class="flex items-center gap-3 px-4 py-2 bg-gray-800 border-b border-gray-700 shrink-0">
        <h1 class="text-xl font-bold text-blue-400 whitespace-nowrap">{move || lang.get().t("title")}</h1>
        <select
            class="min-w-0 flex-1 max-w-md bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm disabled:opacity-50"
            title=move || lang.get().t("select_model")
            disabled=move || load_in_progress.get() || generating.get() || choices.get().is_empty()
            on:change=move |ev| on_choice_selected(event_target_value(&ev))
        >
            <For
                each=move || choices.get()
                key=|choice| choice.id.clone()
                children=move |choice| {
                    let selected = choice.id == selected_choice_id.get();
                    view! {
                        <option value={choice.id.clone()} selected=selected>{choice.name}</option>
                    }
                }
            />
        </select>
        <button
            class=move || {
                if load_in_progress.get() {
                    "px-3 py-1.5 rounded bg-red-600 hover:bg-red-700 text-sm font-medium"
                } else {
                    "px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-sm font-medium"
                }
            }
            disabled=move || !load_in_progress.get() && !can_load()
            on:click=move |_| on_load_or_cancel(())
        >
            {move || {
                let l = lang.get();
                if load_in_progress.get() { l.t("cancel") } else { l.t("load") }
            }}
        </button>
        <button
            class="p-2 rounded hover:bg-gray-700 transition-colors text-gray-300 hover:text-white disabled:opacity-50"
            title=move || lang.get().t("settings")
            disabled=move || load_in_progress.get() || generating.get()
            on:click=move |_| on_settings(())
        >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd"/>
            </svg>
        </button>
    </header>
}
```

- [ ] **Step 3: Wire Header from App**

In `app.rs` view, replace current `Header` call:

```rust
<Header
    lang=lang
    choices=model_choices
    selected_choice_id=selected_choice_id
    loaded_choice_id=loaded_choice_id
    load_in_progress=load_in_progress
    generating=Signal::derive(move || active_index.get().is_some() || status.get() == "generating")
    on_choice_selected=on_choice_selected
    on_load_or_cancel=on_load_or_cancel
    on_settings=move |_| { ... }
/>
```

The settings callback should keep the existing busy guard, but also check `load_in_progress`.

- [ ] **Step 4: Check frontend compile**

Run:

```powershell
cd crates/voxui-desktop
cargo check --target wasm32-unknown-unknown
```

Expected: Settings model-root changes are complete. Progress/status prop errors are addressed in Task 7 before the UI commit.

## Task 6: Settings Model Root And Folder Browse

**Files:**
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src\components\settings_modal.rs`
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src\app.rs`
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src\i18n.rs`

- [ ] **Step 1: Add labels**

In `i18n.rs`, add:

```rust
(Language::Chinese, "models_folder") => "模型文件夹",
(Language::English, "models_folder") => "Models folder",
(Language::Chinese, "browse") => "浏览",
(Language::English, "browse") => "Browse",
```

- [ ] **Step 2: Update Settings values**

In `settings_modal.rs`, remove `model_dir`, `lora_dir`, `models`, and `loras` props. Add:

```rust
model_root: ReadSignal<String>,
loading_or_generating: Signal<bool>,
```

In `SettingsValues`, replace model/lora fields with:

```rust
pub model_root: String,
```

Add local signal:

```rust
let (sel_model_root, set_sel_model_root) = signal(model_root.get_untracked());
```

- [ ] **Step 3: Add model root field**

At the top of modal fields, add:

```rust
<SettingsField label=move || lang.get().t("models_folder")>
    <div class="flex gap-2">
        <input
            type="text"
            class="flex-1 min-w-0 bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm"
            prop:value=move || sel_model_root.get()
            disabled=move || loading_or_generating.get()
            on:input=move |ev| set_sel_model_root.set(event_target_value(&ev))
        />
        <button
            class="px-3 py-1 rounded bg-gray-600 hover:bg-gray-500 text-sm whitespace-nowrap disabled:opacity-50"
            disabled=move || loading_or_generating.get()
            on:click=move |_| {
                spawn_local(async move {
                    match tauri_api::invoke_no_args::<Option<String>>("browse_model_root").await {
                        Ok(Some(path)) => set_sel_model_root.set(path),
                        Ok(None) => {}
                        Err(e) => web_sys::console::error_1(&format!("Browse error: {e}").into()),
                    }
                });
            }
        >
            {move || lang.get().t("browse")}
        </button>
    </div>
</SettingsField>
```

- [ ] **Step 4: Remove model and LoRA selectors**

Delete the `SettingsField` blocks for `model` and `lora`. Backend choice selection now lives in the header.

- [ ] **Step 5: Update apply payload**

In the Apply button callback, build:

```rust
on_apply(SettingsValues {
    model_root: sel_model_root.get(),
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

- [ ] **Step 6: Update `on_apply_settings` in App**

Remove model reload and LoRA application from settings. Settings should rescan choices if `requested_model_root != model_root.get_untracked()`.

Use:

```rust
let requested_model_root = vals.model_root.clone();
let model_root_changed = requested_model_root != model_root.get_untracked();
```

If changed:

```rust
let choices = match tauri_api::invoke::<_, Vec<ModelChoice>>(
    "list_model_choices",
    &ListModelChoicesArgs {
        model_root: requested_model_root.clone(),
    },
)
.await
{
    Ok(choices) => choices,
    Err(e) => {
        restore_after_error(e);
        return;
    }
};
let previous_selected = selected_choice_id.get_untracked();
let next_selected = selected_choice(&choices, &previous_selected)
    .map(|choice| choice.id)
    .unwrap_or_default();
set_model_choices.set(choices);
set_selected_choice_id.set(next_selected);
set_model_root.set(requested_model_root.clone());
set_no_model.set(model_choices.get_untracked().is_empty());
```

Do not call `load_model_choice`; root changes do not auto-load.

- [ ] **Step 7: Save config**

Update config JSON in `on_apply_settings`:

```rust
"model_root": requested_model_root.clone(),
"selected_model_choice_id": selected_choice_id.get_untracked(),
```

Also keep legacy `model_dir`/`lora_dir` from the current selected choice as described in Task 4.

- [ ] **Step 8: Compile**

Run:

```powershell
cd crates/voxui-desktop
cargo check --target wasm32-unknown-unknown
```

Expected: Settings changes are complete. Progress/status prop errors are addressed in Task 7 before the UI commit.

## Task 7: Load Progress Bar And Status Bar

**Files:**
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src\components\progress_bar.rs`
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src\components\status_bar.rs`
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src\app.rs`
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src\i18n.rs`

- [ ] **Step 1: Add load progress labels**

In `i18n.rs`, add:

```rust
(Language::Chinese, "reading") => "读取中",
(Language::English, "reading") => "Reading",
(Language::Chinese, "loading_to_device") => "加载到设备",
(Language::English, "loading_to_device") => "Loading to device",
(Language::Chinese, "loaded") => "已加载",
(Language::English, "loaded") => "Loaded",
(Language::Chinese, "selected") => "已选择",
(Language::English, "selected") => "Selected",
```

- [ ] **Step 2: Add model load progress component**

In `progress_bar.rs`, import:

```rust
use crate::app::LoadProgress;
```

Add:

```rust
#[component]
pub fn ModelLoadProgressBar(
    progress: ReadSignal<LoadProgress>,
    lang: ReadSignal<Language>,
) -> impl IntoView {
    let visible = move || progress.get() != LoadProgress::Hidden;
    let label = move || match progress.get() {
        LoadProgress::Hidden => String::new(),
        LoadProgress::Reading { label, .. } => format!("{} {}", lang.get().t("reading"), label),
        LoadProgress::DeviceLoading { backend } => {
            format!("{} {}", lang.get().t("loading_to_device"), backend)
        }
    };
    let percent = move || match progress.get() {
        LoadProgress::Reading {
            bytes_read,
            total_bytes,
            ..
        } if total_bytes > 0 => (bytes_read as f64 / total_bytes as f64).clamp(0.0, 1.0),
        _ => 0.0,
    };

    view! {
        <div class="shrink-0 px-4 py-2 bg-gray-850" class:hidden=move || !visible()>
            <div class="flex items-center gap-3">
                <span class="text-xs text-gray-400 w-40 truncate">{label}</span>
                <div class="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                        class=move || match progress.get() {
                            LoadProgress::DeviceLoading { .. } => "h-full w-1/3 bg-blue-500 rounded-full animate-pulse",
                            _ => "h-full bg-blue-500 rounded-full transition-all duration-200",
                        }
                        style=move || match progress.get() {
                            LoadProgress::Reading { .. } => format!("width: {}%", percent() * 100.0),
                            LoadProgress::DeviceLoading { .. } => "width: 33%".to_string(),
                            LoadProgress::Hidden => "width: 0%".to_string(),
                        }
                    />
                </div>
                <span class="text-xs text-gray-400 w-12 text-right">
                    {move || match progress.get() {
                        LoadProgress::Reading { total_bytes, .. } if total_bytes > 0 => {
                            format!("{:.0}%", percent() * 100.0)
                        }
                        LoadProgress::DeviceLoading { .. } => "...".to_string(),
                        _ => String::new(),
                    }}
                </span>
            </div>
        </div>
    }
}
```

- [ ] **Step 3: Update status bar props**

In `status_bar.rs`, replace model/lora props:

```rust
selected_choice_name: Signal<String>,
loaded_choice_name: Signal<String>,
```

Build right text:

```rust
let right_text = move || {
    let selected = selected_choice_name.get();
    let loaded = loaded_choice_name.get();
    let backend = actual_backend.get();
    let host = audio_host.get();
    let device = audio_device.get();
    let mut parts = Vec::new();
    if !loaded.is_empty() {
        parts.push(format!("{}: {}", lang.get().t("loaded"), loaded));
    }
    if !selected.is_empty() && selected != loaded {
        parts.push(format!("{}: {}", lang.get().t("selected"), selected));
    }
    if !backend.is_empty() {
        parts.push(backend);
    }
    if !host.is_empty() || !device.is_empty() {
        parts.push(if host.is_empty() {
            device
        } else if device.is_empty() {
            host
        } else {
            format!("{host}/{device}")
        });
    }
    parts.join(" | ")
};
```

- [ ] **Step 4: Wire progress and status from App**

In `app.rs`, insert:

```rust
<ModelLoadProgressBar progress=load_progress lang=lang />
```

near the end of main content, before `InputBox` or immediately above the status bar.

Pass status names:

```rust
selected_choice_name=Signal::derive(move || {
    selected_choice(&model_choices.get(), &selected_choice_id.get())
        .map(|choice| choice.name)
        .unwrap_or_default()
})
loaded_choice_name=Signal::derive(move || {
    let loaded = loaded_choice_id.get();
    loaded
        .and_then(|id| model_choices.get().into_iter().find(|choice| choice.id == id))
        .map(|choice| choice.name)
        .unwrap_or_default()
})
```

- [ ] **Step 5: Compile frontend**

Run:

```powershell
cd crates/voxui-desktop
cargo check --target wasm32-unknown-unknown
```

Expected: pass.

- [ ] **Step 6: Commit Tasks 4-7 together**

```powershell
git add -- crates/voxui-desktop/src/app.rs crates/voxui-desktop/src/components/header.rs crates/voxui-desktop/src/components/settings_modal.rs crates/voxui-desktop/src/components/progress_bar.rs crates/voxui-desktop/src/components/status_bar.rs crates/voxui-desktop/src/i18n.rs
git commit -m "feat(desktop): add manual model loading UI"
```

## Task 8: App Titles And Cleanup

**Files:**
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\index.html`
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src-tauri\tauri.conf.json`
- Modify: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src\components\mod.rs`
- Delete: `D:\Sandbox_Share\VoxUI\voxui\crates\voxui-desktop\src\components\model_select.rs`

- [ ] **Step 1: Update HTML title**

In `index.html`, change:

```html
<title>AhanSays</title>
```

- [ ] **Step 2: Update Tauri title**

In `tauri.conf.json`, change:

```json
"productName": "AhanSays"
```

and window title:

```json
"title": "AhanSays"
```

- [ ] **Step 3: Remove obsolete modal usage**

Remove from `components/mod.rs`:

```rust
mod model_select;
pub use model_select::*;
```

Delete the obsolete component file:

```powershell
git rm crates/voxui-desktop/src/components/model_select.rs
```

- [ ] **Step 4: Check workspace formatting**

Run:

```powershell
cargo fmt --all
```

Expected: completes without errors.

- [ ] **Step 5: Run checks**

Run:

```powershell
cargo test -p voxui-desktop
cd crates/voxui-desktop
cargo check --target wasm32-unknown-unknown
```

Expected: both pass.

- [ ] **Step 6: Commit**

```powershell
git add -- crates/voxui-desktop/index.html crates/voxui-desktop/src-tauri/tauri.conf.json crates/voxui-desktop/src/components/mod.rs crates/voxui-desktop/src/components/model_select.rs
git commit -m "chore(desktop): update app title"
```

## Task 9: End-To-End Verification

**Files:**
- No production edits expected.

- [ ] **Step 1: Run backend tests**

```powershell
cargo test -p voxui-desktop
```

Expected: all desktop backend tests pass.

- [ ] **Step 2: Run frontend build check**

```powershell
cd crates/voxui-desktop
cargo check --target wasm32-unknown-unknown
```

Expected: pass.

- [ ] **Step 3: Run full workspace tests**

```powershell
cd D:\Sandbox_Share\VoxUI\voxui
cargo test --workspace
```

Expected: pass. If CUDA/model-dependent inference tests skip or require local assets, record exact skipped/failing tests and why.

- [ ] **Step 4: Run desktop app manually**

```powershell
cd crates/voxui-desktop\src-tauri
cargo tauri dev
```

Expected manual observations:

- App opens with title `焓言焓语` in Chinese mode or `AhanSays` in English mode.
- It scans the configured model root but does not load a model automatically.
- Dropdown shows base entries and duplicated LoRA entries such as `voxcpm2-q4-lm | lora_ft2`.
- `Load` is enabled for an unloaded selected model.
- During load, button reads `Cancel` and progress shows byte-reading then indeterminate device loading.
- Cancelling load unlocks controls and does not remove a previously loaded engine.
- Switching dropdown away from a loaded model keeps generation available.
- Switching back to the loaded entry disables `Load`.
- Settings can change the models folder and rescans without auto-loading.

- [ ] **Step 5: Final status**

Run:

```powershell
git status --short
```

Expected: only intentional uncommitted changes remain. Do not revert unrelated pre-existing changes such as old `crates/voxui-app` deletions.
