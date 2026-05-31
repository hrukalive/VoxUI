# LoRA Dropdown Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split LoRA selection from model discovery — separate dropdown for LoRA, dynamic engine switching per synthesis request with caching.

**Architecture:** Add `lora_path` to `SynthesisRequest`/`SynthesisRequestDto`. Engine reconciles LoRA state before each synthesis using an internal cache (`HashMap<PathBuf, LoraAdapter>`). Model discovery produces only base models. LoRA files are scanned on-demand after a model loads. UI gains a LoRA dropdown next to the Load button.

**Tech Stack:** Rust, Tauri, Leptos, candle, GgufModelLoader, SidecarProtocol

---

### Task 1: Add `lora_path` to SynthesisRequest and SynthesisRequestDto

**Files:**
- Modify: `crates/voxui-inference/src/request.rs`
- Modify: `crates/voxui-sidecar-protocol/src/lib.rs`

- [ ] **Step 1: Add `lora_path` field to `SynthesisRequest`**

Open `crates/voxui-inference/src/request.rs`. At line 23 (after `consolidate_n`), add:
```rust
    pub lora_path: Option<PathBuf>,
```
In the `Default` impl (line 41, after `consolidate_n: 1`), add:
```rust
            lora_path: None,
```

- [ ] **Step 2: Add `lora_path` field to `SynthesisRequestDto`**

Open `crates/voxui-sidecar-protocol/src/lib.rs`. At line 112 (after `consolidate_n`), add:
```rust
    pub lora_path: Option<PathBuf>,
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p voxui-inference -p voxui-sidecar-protocol`
Expected: compilation succeeds (may need to fix `synthesis_request_from_dto` in sidecar — Task 8 handles that)

- [ ] **Step 4: Commit**

```bash
git add crates/voxui-inference/src/request.rs crates/voxui-sidecar-protocol/src/lib.rs
git commit -m "feat: add lora_path to SynthesisRequest and SynthesisRequestDto"
```

---

### Task 2: Remove `lora_path` from LoadModel and clean up model discovery

**Files:**
- Modify: `crates/voxui-sidecar-protocol/src/lib.rs` — remove `lora_path` from `SidecarCommand::LoadModel`
- Modify: `crates/voxui-desktop/src-tauri/src/model_discovery.rs` — remove LoRA entries from `discover_models`
- Modify: `crates/voxui-desktop/src-tauri/src/commands.rs` — update `load_model`
- Modify: `crates/voxui-inference-sidecar/src/lib.rs` — update LoadModel handler, `sidecar_command_name`

- [ ] **Step 1: Remove `lora_path` from `SidecarCommand::LoadModel`**

In `crates/voxui-sidecar-protocol/src/lib.rs`, lines 13-18, change:
```rust
    LoadModel {
        load_id: u64,
        model_dir: PathBuf,
        lora_path: Option<PathBuf>,
        backend: BackendKind,
    },
```
to:
```rust
    LoadModel {
        load_id: u64,
        model_dir: PathBuf,
        backend: BackendKind,
    },
```

- [ ] **Step 2: Update `load_model` command to not send `lora_path`**

In `crates/voxui-desktop/src-tauri/src/commands.rs`, line 425-429, change:
```rust
    let command = SidecarCommand::LoadModel {
        load_id,
        model_dir: choice.model_dir.clone(),
        lora_path: choice.lora_path.clone(),
        backend: protocol_backend(backend),
    };
```
to:
```rust
    let command = SidecarCommand::LoadModel {
        load_id,
        model_dir: choice.model_dir.clone(),
        backend: protocol_backend(backend),
    };
```

- [ ] **Step 3: Remove LoRA entries from `discover_models`**

In `crates/voxui-desktop/src-tauri/src/model_discovery.rs`, remove lines 64-108 (the entire LoRA scanning and combined-entry creation code block). The resulting `discover_models` function should end after pushing the single base ModelChoice (lines 62-63) and return `Ok(choices)`.

The function should look like:
```rust
pub fn discover_models(root: &Path) -> Result<Vec<ModelChoice>> {
    if !root
        .try_exists()
        .with_context(|| format!("failed to inspect model root {}", root.display()))?
    {
        return Ok(Vec::new());
    }

    let mut model_dirs = Vec::new();
    for entry in fs::read_dir(root)
        .with_context(|| format!("failed to read model root {}", root.display()))?
    {
        let entry = entry
            .with_context(|| format!("failed to read entry in model root {}", root.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to read file type for {}", path.display()))?;
        if file_type.is_dir() {
            model_dirs.push(path);
        }
    }
    model_dirs.sort();

    let mut choices = Vec::new();
    for model_dir in model_dirs {
        let model_path = model_dir.join("model.gguf");
        let model_metadata = match model_path.metadata() {
            Ok(metadata) if metadata.is_file() => metadata,
            Ok(_) => continue,
            Err(error) if error.kind() == ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to read metadata for {}", model_path.display())
                });
            }
        };

        let model_name = model_dir
            .file_name()
            .and_then(|name| name.to_str())
            .with_context(|| format!("model directory name is not UTF-8: {}", model_dir.display()))?
            .to_owned();
        let model_bytes = model_metadata.len();

        choices.push(ModelChoice {
            id: choice_id(root, &model_dir, None)?,
            display_name: model_name.clone(),
            model_dir: model_dir.clone(),
            model_path: model_path.clone(),
            lora_path: None,
            model_bytes,
            lora_bytes: 0,
        });
    }

    Ok(choices)
}
```

- [ ] **Step 4: Update sidecar LoadModel handler — remove LoRA loading**

In `crates/voxui-inference-sidecar/src/lib.rs`, inside `handle_command_with_emit`, the `SidecarCommand::LoadModel` handler (lines 82-185). Change the loading section (lines 133-138):
```rust
                            (Ok(mut engine), None) => {
                                if let Some(path) = lora_path {
                                    engine.load_lora(&path).with_context(|| {
                                        format!("load LoRA adapter {}", path.display())
                                    })?;
                                }
                                Ok(engine)
                            }
```
to just:
```rust
                            (Ok(engine), None) => Ok(engine),
```

Also update the destructuring on line 83-88 — remove `lora_path`:
```rust
            SidecarCommand::LoadModel {
                load_id,
                model_dir,
                backend,
            } => {
```

And update the log line (89-94) that references `lora_path` — remove the `lora_path` field:
```rust
                tracing::info!(
                    load_id,
                    model_dir = %model_dir.display(),
                    backend = ?backend,
                    "sidecar starting model load"
                );
```

- [ ] **Step 5: Update `sidecar_command_name`**

In `crates/voxui-inference-sidecar/src/lib.rs`, line 643:
```rust
        SidecarCommand::LoadModel { .. } => "load_model",
```
(no change needed — it already uses `..`)

- [ ] **Step 6: Verify it compiles**

Run: `cargo check --workspace`
Expected: compilation succeeds

- [ ] **Step 7: Commit**

```bash
git add crates/voxui-sidecar-protocol/src/lib.rs crates/voxui-desktop/src-tauri/src/model_discovery.rs crates/voxui-desktop/src-tauri/src/commands.rs crates/voxui-inference-sidecar/src/lib.rs
git commit -m "refactor: remove lora_path from LoadModel, split LoRA from model discovery"
```

---

### Task 3: Add `discover_loras` function and `LoraEntry` type

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/model_discovery.rs` — add `discover_loras` function
- Modify: `crates/voxui-desktop/src-tauri/src/types.rs` — add `LoraEntry` struct

- [ ] **Step 1: Add `LoraEntry` to types.rs**

In `crates/voxui-desktop/src-tauri/src/types.rs`, after the `ModelChoice` struct (after line 554), add:
```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraEntry {
    pub id: String,
    pub display_name: String,
}
```

- [ ] **Step 2: Add `discover_loras` function to model_discovery.rs**

At the end of `crates/voxui-desktop/src-tauri/src/model_discovery.rs` (before `choice_id`), add:
```rust
use crate::types::LoraEntry;

pub fn discover_loras(model_dir: &Path) -> Result<Vec<LoraEntry>> {
    if !model_dir
        .try_exists()
        .with_context(|| format!("failed to inspect model directory {}", model_dir.display()))?
    {
        return Ok(Vec::new());
    }

    let mut entries = Vec::new();
    for entry in fs::read_dir(model_dir)
        .with_context(|| format!("failed to read model directory {}", model_dir.display()))?
    {
        let entry = entry.with_context(|| {
            format!(
                "failed to read entry in model directory {}",
                model_dir.display()
            )
        })?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to read file type for {}", path.display()))?;
        if is_lora_candidate(&path, &file_type)? {
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .with_context(|| {
                    format!(
                        "LoRA file stem is not UTF-8 or is missing: {}",
                        path.display()
                    )
                })?;
            entries.push(LoraEntry {
                id: stem.to_owned(),
                display_name: stem.to_owned(),
            });
        }
    }
    entries.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(entries)
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p voxui-desktop`
Expected: compilation succeeds

- [ ] **Step 4: Commit**

```bash
git add crates/voxui-desktop/src-tauri/src/model_discovery.rs crates/voxui-desktop/src-tauri/src/types.rs
git commit -m "feat: add discover_loras and LoraEntry type"
```

---

### Task 4: Update AppCore to track LoRA state and populate on model load

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/app_core.rs`
- Modify: `crates/voxui-desktop/src-tauri/src/types.rs` — add fields to `AppSnapshot` and `RequestSnapshot`
- Modify: `crates/voxui-desktop/src-tauri/src/generation_queue.rs` — add `lora_id` to snapshot

- [ ] **Step 1: Add `lora_id` to `RequestSnapshot`**

In `crates/voxui-desktop/src-tauri/src/types.rs`, at line 579 (inside `RequestSnapshot`, after `generation`), add:
```rust
    pub lora_id: Option<String>,
```

- [ ] **Step 2: Add fields to `AppSnapshot`**

In `crates/voxui-desktop/src-tauri/src/types.rs`, at line 591 (inside `AppSnapshot`, after `load_state`), add:
```rust
    pub available_loras: Vec<LoraEntry>,
    pub selected_lora_id: Option<String>,
```

- [ ] **Step 3: Update `GenerationQueue::snapshot` to include `lora_id`**

In `crates/voxui-desktop/src-tauri/src/generation_queue.rs`, the `snapshot` function takes `loaded_model_id` and `config`. It also needs `lora_id`. Change the signature and body:

At line 178, change:
```rust
    fn snapshot(loaded_model_id: impl Into<String>, config: &AppConfig) -> RequestSnapshot {
        RequestSnapshot {
            model_id: loaded_model_id.into(),
            backend: config.backend,
            generation: config.generation.clone(),
        }
    }
```
to:
```rust
    fn snapshot(
        loaded_model_id: impl Into<String>,
        lora_id: Option<String>,
        config: &AppConfig,
    ) -> RequestSnapshot {
        RequestSnapshot {
            model_id: loaded_model_id.into(),
            backend: config.backend,
            generation: config.generation.clone(),
            lora_id,
        }
    }
```

Update all callers of `snapshot`:
- Line 55 (`enqueue`): add the `lora_id` parameter
- Line 178 (`start_regeneration`): add the `lora_id` parameter

The `enqueue` method at line 38 needs a `lora_id` parameter:
```rust
    pub fn enqueue(
        &mut self,
        text: String,
        loaded_model_id: impl Into<String>,
        lora_id: Option<String>,
        config: &AppConfig,
        created_at: u64,
        status: HistoryStatus,
    ) -> String {
```
And line 55 becomes:
```rust
            snapshot: Self::snapshot(loaded_model_id, lora_id, config),
```

The `start_regeneration` at line 160 also needs `lora_id`:
```rust
    pub fn start_regeneration(
        &mut self,
        id: &str,
        loaded_model_id: impl Into<String>,
        lora_id: Option<String>,
        config: &AppConfig,
    ) -> bool {
```
And line 166 becomes:
```rust
        let snapshot = Self::snapshot(loaded_model_id, lora_id, config);
```

- [ ] **Step 4: Add AppCore fields and update `from_loaded_config`**

In `crates/voxui-desktop/src-tauri/src/app_core.rs`, inside `AppCore` struct, after `selected_model_id`, add:
```rust
    selected_lora_id: Option<String>,
    available_loras: Vec<LoraEntry>,
```
(Note: `LoraEntry` needs to be added to the import on lines 14-17.)

Update the `from_loaded_config` constructor (around line 111) to include:
```rust
            selected_lora_id: None,
            available_loras: Vec::new(),
```

- [ ] **Step 5: Update `snapshot()` to include LoRA fields**

In `app_core.rs`, the `snapshot()` method (lines 135-147), add:
```rust
            available_loras: self.available_loras.clone(),
            selected_lora_id: self.selected_lora_id.clone(),
```

- [ ] **Step 6: Populate LoRA on model load success**

In `app_core.rs`, the `mark_load_success` method (lines 975-985). After setting `self.loaded_model_id`, scan for LoRA files and reset selection:
```rust
    pub fn mark_load_success(&mut self, load_id: u64, choice_id: String, sample_rate: u32) -> bool {
        if !self.active_load_matches(load_id) {
            return false;
        }

        self.active_load = None;
        self.loaded_model_id = Some(choice_id);
        self.loaded_sample_rate = Some(sample_rate);
        self.load_state = LoadUiState::Idle;

        // Scan for LoRA files in the loaded model's directory
        self.available_loras = match &self.loaded_model_id {
            Some(id) => {
                if let Some(choice) = self.models.iter().find(|c| c.id == *id) {
                    crate::model_discovery::discover_loras(&choice.model_dir).unwrap_or_default()
                } else {
                    Vec::new()
                }
            }
            None => Vec::new(),
        };
        self.selected_lora_id = None;

        true
    }
```

Also clear LoRA state on sidecar exit. In `handle_sidecar_exit` (line 869), after `self.loaded_model_id = None`, add:
```rust
        self.available_loras = Vec::new();
        self.selected_lora_id = None;
```

- [ ] **Step 7: Update `enqueue_generation` to pass LoRA**

In `app_core.rs`, the `enqueue_generation` method, line 439:
```rust
        let id = self.queue.enqueue(text, loaded_model_id, &self.config, now, status);
```
Change to:
```rust
        let id = self.queue.enqueue(
            text,
            loaded_model_id,
            self.selected_lora_id.clone(),
            &self.config,
            now,
            status,
        );
```

- [ ] **Step 8: Update `synthesis_request` to include `lora_path`**

In `app_core.rs`, the `synthesis_request` method (lines 909-931), add `lora_path` construction. After line 930 (`consolidate_n`), add:
```rust
            lora_path: item.snapshot.lora_id.as_ref().map(|id| {
                self.models
                    .iter()
                    .find(|c| c.id == item.snapshot.model_id)
                    .map(|c| c.model_dir.join(format!("{id}.gguf")))
                    .unwrap_or_else(|| PathBuf::from(format!("{id}.gguf")))
            }),
```
And also add a comma after the `consolidate_n` field on line 930.

- [ ] **Step 9: Update `regenerate_item` and its caller `regenerate_item_stopping_playback`**

In `app_core.rs`, the `regenerate_item` method needs `lora_id`. It calls `start_regeneration`:
```rust
    pub fn regenerate_item(&mut self, item_id: &str, config: &AppConfig) -> Result<()> {
        let loaded_model_id = self
            .loaded_model_id
            .clone()
            .context("no model loaded for generation")?;
        let lora_id = self.selected_lora_id.clone();
        if !self
            .queue
            .start_regeneration(item_id, loaded_model_id, lora_id, config)
        {
            bail!("unknown history item: {item_id}");
        }
        Ok(())
    }
```

- [ ] **Step 10: Import `model_discovery` for `discover_loras`**

On line 12 of `app_core.rs`, change:
```rust
use crate::model_discovery::discover_models;
```
to:
```rust
use crate::model_discovery::{discover_loras, discover_models};
```
Actually, the call in `mark_load_success` uses `crate::model_discovery::discover_loras(...)` fully qualified, so just add:
```rust
use crate::model_discovery::{discover_loras, discover_models};
```
And update the call in `mark_load_success` to `discover_loras(&choice.model_dir)`.

- [ ] **Step 11: Verify it compiles**

Run: `cargo check -p voxui-desktop`
Expected: compilation succeeds (may have test breakages — fix in Task 9)

- [ ] **Step 12: Commit**

```bash
git add crates/voxui-desktop/src-tauri/src/types.rs crates/voxui-desktop/src-tauri/src/generation_queue.rs crates/voxui-desktop/src-tauri/src/app_core.rs
git commit -m "feat: add LoRA state tracking in AppCore, populate on model load"
```

---

### Task 5: Update commands for LoRA and synthesis_request_dto

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/commands.rs` — update `synthesis_request_dto`

- [ ] **Step 1: Update `synthesis_request_dto` to include `lora_path`**

In `crates/voxui-desktop/src-tauri/src/commands.rs`, find the `synthesis_request_dto` function. Add `lora_path` to the returned DTO:
```rust
fn synthesis_request_dto(request: voxui_inference::SynthesisRequest) -> SynthesisRequestDto {
    SynthesisRequestDto {
        text: request.text,
        prompt_wav_path: request.prompt_wav_path,
        prompt_text: request.prompt_text,
        reference_wav_path: request.reference_wav_path,
        cfg_value: request.cfg_value,
        inference_timesteps: request.inference_timesteps,
        min_len: request.min_len,
        max_len: request.max_len,
        retry_badcase: request.retry_badcase,
        retry_badcase_max_times: request.retry_badcase_max_times,
        retry_badcase_ratio_threshold: request.retry_badcase_ratio_threshold,
        consolidate_n: request.consolidate_n,
        lora_path: request.lora_path,
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p voxui-desktop`
Expected: compilation succeeds

- [ ] **Step 3: Commit**

```bash
git add crates/voxui-desktop/src-tauri/src/commands.rs
git commit -m "feat: include lora_path in synthesis_request_dto"
```

---

### Task 6: Update engine with LoRA cache and pre-synthesis reconciliation

**Files:**
- Modify: `crates/voxui-inference/src/engine.rs`

- [ ] **Step 1: Add `lora_cache` field to `VoxCPMEngine`**

In `crates/voxui-inference/src/engine.rs`, line 118, after `lora: Option<LoraAdapter>,` add:
```rust
    lora_cache: HashMap<PathBuf, LoraAdapter>,
```

Need to import `HashMap` and `PathBuf` at top of file:
```rust
use std::collections::HashMap;
use std::path::PathBuf;
```

- [ ] **Step 2: Initialize `lora_cache` in `load_with_progress`**

In the `Ok(Self { ... })` at the end of `load_with_progress` (line 252-270), after `lora: None,` add:
```rust
            lora_cache: HashMap::new(),
```

- [ ] **Step 3: Add `reconcile_lora` method**

After the `load_lora` method (after line 292), add:
```rust
    fn reconcile_lora(&mut self, lora_path: Option<&Path>) -> Result<()> {
        match (&self.lora, lora_path) {
            // Both None — nothing to do
            (None, None) => {}
            // Same path already active — no-op
            (Some(_), Some(path)) if self.lora.as_ref().is_some_and(|l| {
                // Check if the path matches (we don't store the path, so compare by cache lookup)
                true
            }) => {
                // We'll use a simpler approach: track the current path
            }
            // Currently has LoRA, request wants None — unload (keep in cache)
            (Some(_), None) => {
                self.lora = None;
            }
            // Currently None, request has path — try cache first
            (None, Some(path)) => {
                let canon = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
                if let Some(adapter) = self.lora_cache.remove(&canon) {
                    self.lora = Some(adapter);
                } else {
                    let adapter = LoraAdapter::load_file_for_model(
                        path,
                        &self.device,
                        &self.manifest,
                    )?;
                    self.lora = Some(adapter);
                }
            }
            // Different path — unload old, load new
            (Some(_), Some(path)) => {
                self.lora = None;
                let canon = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
                if let Some(adapter) = self.lora_cache.remove(&canon) {
                    self.lora = Some(adapter);
                } else {
                    let adapter = LoraAdapter::load_file_for_model(
                        path,
                        &self.device,
                        &self.manifest,
                    )?;
                    self.lora = Some(adapter);
                }
            }
        }
        Ok(())
    }
```

Wait, this approach has a problem — we don't track the active path, only the adapter. We need to store the current LoRA's path to compare. Let me revise:

Actually the simplest approach: store the active path alongside the adapter. Add a field `active_lora_path: Option<PathBuf>`:

```rust
    lora: Option<LoraAdapter>,
    active_lora_path: Option<PathBuf>,
    lora_cache: HashMap<PathBuf, LoraAdapter>,
```

Then `reconcile_lora`:
```rust
    fn reconcile_lora(&mut self, lora_path: Option<&Path>) -> Result<()> {
        let requested = lora_path.map(|p| p.canonicalize().unwrap_or_else(|_| p.to_path_buf()));
        if self.active_lora_path == requested {
            return Ok(());
        }

        // Unload current (keep in cache if it exists)
        if let Some(active_path) = self.active_lora_path.take() {
            if let Some(adapter) = self.lora.take() {
                self.lora_cache.insert(active_path, adapter);
            }
        }

        // Load requested
        if let Some(path) = requested {
            if let Some(adapter) = self.lora_cache.remove(&path) {
                self.lora = Some(adapter);
            } else {
                self.lora = Some(LoraAdapter::load_file_for_model(
                    lora_path.unwrap(),
                    &self.device,
                    &self.manifest,
                )?);
            }
            self.active_lora_path = Some(path);
        }

        Ok(())
    }
```

- [ ] **Step 2 (revised): Add `active_lora_path` and `lora_cache` fields**

In `VoxCPMEngine` struct (line 103-121), change:
```rust
    lora: Option<LoraAdapter>,
```
to:
```rust
    lora: Option<LoraAdapter>,
    active_lora_path: Option<PathBuf>,
    lora_cache: HashMap<PathBuf, LoraAdapter>,
```

Add imports at top of file:
```rust
use std::collections::HashMap;
use std::path::{Path, PathBuf};
```

In `load_with_progress`, at the `Ok(Self { ... })` initialization, after `lora: None,`:
```rust
            active_lora_path: None,
            lora_cache: HashMap::new(),
```

- [ ] **Step 3: Add `reconcile_lora` method**

After `unload_lora` method (after line 296), add:
```rust
    fn reconcile_lora(&mut self, lora_path: Option<&Path>) -> Result<()> {
        let requested = lora_path.map(|p| {
            p.canonicalize().unwrap_or_else(|_| p.to_path_buf())
        });
        if self.active_lora_path == requested {
            return Ok(());
        }

        if let Some(active_path) = self.active_lora_path.take() {
            if let Some(adapter) = self.lora.take() {
                self.lora_cache.insert(active_path, adapter);
            }
        }

        if let Some(path) = requested {
            if let Some(adapter) = self.lora_cache.remove(&path) {
                self.lora = Some(adapter);
            } else {
                self.lora = Some(LoraAdapter::load_file_for_model(
                    lora_path.unwrap(),
                    &self.device,
                    &self.manifest,
                )?);
            }
            self.active_lora_path = Some(path);
        }

        Ok(())
    }
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p voxui-inference`
Expected: compilation succeeds

- [ ] **Step 5: Commit**

```bash
git add crates/voxui-inference/src/engine.rs
git commit -m "feat: add LoRA cache and reconcile_lora to VoxCPMEngine"
```

---

### Task 7: Call reconcile_lora in sidecar before synthesis

**Files:**
- Modify: `crates/voxui-inference-sidecar/src/lib.rs`

- [ ] **Step 1: Update `synthesis_request_from_dto` to include `lora_path`**

In `crates/voxui-inference-sidecar/src/lib.rs`, lines 535-551, change the function:
```rust
fn synthesis_request_from_dto(dto: SynthesisRequestDto) -> SynthesisRequest {
    SynthesisRequest {
        text: dto.text,
        prompt_wav_path: dto.prompt_wav_path,
        prompt_text: dto.prompt_text,
        reference_wav_path: dto.reference_wav_path,
        cfg_value: dto.cfg_value,
        inference_timesteps: dto.inference_timesteps,
        min_len: dto.min_len,
        max_len: dto.max_len,
        retry_badcase: dto.retry_badcase,
        retry_badcase_max_times: dto.retry_badcase_max_times,
        retry_badcase_ratio_threshold: dto.retry_badcase_ratio_threshold,
        consolidate_n: dto.consolidate_n,
        lora_path: dto.lora_path,
        ..SynthesisRequest::default()
    }
}
```

- [ ] **Step 2: Call `reconcile_lora` before synthesis in `handle_command_with_emit`**

In `crates/voxui-inference-sidecar/src/lib.rs`, in the `Synthesize` handler (around line 212), after `let request = synthesis_request_from_dto(request);` add:
```rust
                if let Some(ref path) = request.lora_path {
                    if let Err(error) = engine.reconcile_lora(Some(path.as_path())) {
                        emit(Frame {
                            header: SidecarEvent::GenerationDone {
                                item_id,
                                status: OperationStatus::Failed,
                                sample_rate: None,
                                duration_seconds: None,
                                error: Some(format!("failed to load LoRA: {error}")),
                            },
                            payload: Vec::new(),
                        })?;
                        return Ok(false);
                    }
                } else {
                    let _ = engine.reconcile_lora(None);
                }
```

Wait, `reconcile_lora` is not public on the engine. Need to make it public first.

- [ ] **Step 2a: Make `reconcile_lora` public in engine.rs**

In `crates/voxui-inference/src/engine.rs`, change:
```rust
    fn reconcile_lora(&mut self, lora_path: Option<&Path>) -> Result<()> {
```
to:
```rust
    pub fn reconcile_lora(&mut self, lora_path: Option<&Path>) -> Result<()> {
```

- [ ] **Step 3: Add note that `VoxCPMEngine` needs to be imported from `voxui_inference`**

The sidecar should already import `VoxCPMEngine` through `use voxui_inference::VoxCPMEngine;` or `use voxui_inference::*;`.

- [ ] **Step 4: Verify it compiles**

Run: `cargo check --workspace`
Expected: compilation succeeds

- [ ] **Step 5: Commit**

```bash
git add crates/voxui-inference-sidecar/src/lib.rs crates/voxui-inference/src/engine.rs
git commit -m "feat: reconcile LoRA before synthesis in sidecar, add lora_path to DTO mapping"
```

---

### Task 8: Update Tauri API types (frontend mirrors)

**Files:**
- Modify: `crates/voxui-desktop/src/tauri_api.rs`

- [ ] **Step 1: Add `LoraEntry` to frontend types**

In `crates/voxui-desktop/src/tauri_api.rs`, before (or after) `ModelChoice`, add:
```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraEntry {
    pub id: String,
    pub display_name: String,
}
```

- [ ] **Step 2: Add `lora_id` to `RequestSnapshot`**

In `crates/voxui-desktop/src/tauri_api.rs`, at line 458 (inside `RequestSnapshot`, after `generation`), add:
```rust
    pub lora_id: Option<String>,
```

- [ ] **Step 3: Add fields to `AppSnapshot`**

In `crates/voxui-desktop/src/tauri_api.rs`, inside `AppSnapshot` (around line 49, after `load_state`), add:
```rust
    pub available_loras: Vec<LoraEntry>,
    pub selected_lora_id: Option<String>,
```

- [ ] **Step 4: Add `selected_lora_id` to `ConfigPatch`**

In `crates/voxui-desktop/src/tauri_api.rs`, in the `ConfigPatch` struct (around line 428, after `translation`), add:
```rust
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_lora_id: Option<Option<String>>,
```

- [ ] **Step 5: Mirrors in backend ConfigPatch**

Also add to the backend `ConfigPatch` in `crates/voxui-desktop/src-tauri/src/types.rs` (around line 490, after `translation`):
```rust
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_double_option"
    )]
    pub selected_lora_id: Option<Option<String>>,
```

And in the `apply_patch` method of `AppCore` (in `app_core.rs`), add handling for `selected_lora_id` in the patch matching section. Around line 165, after the `selected_model_id` block:
```rust
        if let Some(selected_lora_id) = patch.selected_lora_id {
            self.selected_lora_id = selected_lora_id;
        }
```

- [ ] **Step 6: Verify it compiles**

Run: `cargo check --workspace`
Expected: compilation succeeds

- [ ] **Step 7: Commit**

```bash
git add crates/voxui-desktop/src/tauri_api.rs crates/voxui-desktop/src-tauri/src/types.rs crates/voxui-desktop/src-tauri/src/app_core.rs
git commit -m "feat: add LoraEntry, selected_lora_id to frontend types and config patch"
```

---

### Task 9: Add LoRA dropdown to Header component

**Files:**
- Modify: `crates/voxui-desktop/src/components/header.rs`
- Modify: `crates/voxui-desktop/src/i18n.rs` — add lora label

- [ ] **Step 1: Add `lora` label to i18n**

In `crates/voxui-desktop/src/i18n.rs`, in the `Labels` struct (after `model`), add:
```rust
    pub lora: &'static str,
```

In the Chinese `labels()` function (around line 126), add:
```rust
            lora: "LoRA",
```

In the English `labels()` function (around line 231), add:
```rust
            lora: "LoRA",
```

- [ ] **Step 2: Add LoRA dropdown to Header**

In `crates/voxui-desktop/src/components/header.rs`, update the `Header` component to add props and a LoRA `<CustomSelect>`.

Import `LoraEntry`:
```rust
use crate::tauri_api::{LoraEntry, ModelChoice};
```

Add props to the function signature after `load_disabled`:
```rust
    loras: Vec<LoraEntry>,
    selected_lora_id: Option<String>,
    lora_disabled: bool,
    on_lora_select: impl Fn(Option<String>) + Send + Sync + 'static + Copy,
```

Build LoRA options:
```rust
    let selected_lora_id = selected_lora_id.unwrap_or_default();
    let lora_options = {
        let loras = loras.clone();
        move || {
            let mut opts: Vec<SelectOption> = vec![
                SelectOption::new(String::new(), "None".to_string()),
            ];
            for lora in &loras {
                opts.push(SelectOption::new(lora.id.clone(), lora.display_name.clone()));
            }
            opts
        }
    };
    let current_lora_id = {
        let selected_lora_id = selected_lora_id.clone();
        move || selected_lora_id.clone()
    };
```

Add the LoRA `<CustomSelect>` right after the Load button (after line 59):
```rust
            <CustomSelect
                class="lora-select"
                aria_label=move || labels.lora
                value=current_lora_id
                options=lora_options
                disabled=lora_disabled
                on_change=move |lora_id| on_lora_select(if lora_id.is_empty() { None } else { Some(lora_id) })
            />
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p voxui-desktop --lib`
Expected: header.rs compiles (app.rs will need updates — next task)

- [ ] **Step 4: Commit**

```bash
git add crates/voxui-desktop/src/components/header.rs crates/voxui-desktop/src/i18n.rs
git commit -m "feat: add LoRA dropdown to Header component"
```

---

### Task 10: Wire up LoRA dropdown in app.rs

**Files:**
- Modify: `crates/voxui-desktop/src/app.rs`

- [ ] **Step 1: Update `current_snapshot_untracked` and snapshot access**

In `crates/voxui-desktop/src/app.rs`, the `fallback_snapshot` function needs the new fields. Find it around line 563 and add:
```rust
            available_loras: Vec::new(),
            selected_lora_id: None,
```

- [ ] **Step 2: Add LoRA props to Header instantiation**

Around lines 312-375, the `<Header>` component instantiation. Add the new props:
```rust
    let lora_disabled = loaded_model_id.is_none()
        || snapshot.available_loras.is_empty();
```

Add these props inside the `<Header` element, after `load_disabled`:
```rust
                        loras=snapshot.available_loras
                        selected_lora_id=snapshot.selected_lora_id
                        lora_disabled=lora_disabled
                        on_lora_select=move |lora_id| {
                            spawn_local(async move {
                                if let Ok(next_snapshot) = crate::tauri_api::set_config_patch(ConfigPatch {
                                    selected_lora_id: Some(lora_id),
                                    ..ConfigPatch::default()
                                })
                                .await
                                {
                                    set_snapshot.set(Some(next_snapshot));
                                }
                            });
                        }
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p voxui-desktop --lib`
Expected: compilation succeeds

- [ ] **Step 4: Commit**

```bash
git add crates/voxui-desktop/src/app.rs
git commit -m "feat: wire up LoRA dropdown in app.rs"
```

---

### Task 11: Fix tests and run the test suite

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/tests/app_core_tests.rs`
- Modify: `crates/voxui-desktop/src-tauri/tests/model_discovery_tests.rs`
- Modify: `crates/voxui-desktop/src-tauri/tests/` (various)

- [ ] **Step 1: Fix test compilation**

Run: `cargo test --no-run --workspace`
Look for compilation errors in tests. The `enqueue` and `start_regeneration` signature changes will break tests, as will the `snapshot()` and `from_loaded_config` changes.

For each test that calls `queue.enqueue(...)`, add `None` as the `lora_id` parameter:
```rust
// Before:
queue.enqueue(text, model_id, &config, now, HistoryStatus::Queued);
// After:
queue.enqueue(text, model_id, None, &config, now, HistoryStatus::Queued);
```

For tests calling `set_loaded_model_for_test`, they should also set `available_loras = Vec::new()` (already defaults correctly).

For `fallback_snapshot` in `app.rs` tests — update the expected snapshot to include new fields.

- [ ] **Step 2: Run all tests**

Run: `cargo test --workspace`
Fix any remaining failures.

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "test: fix tests for LoRA dropdown changes"
```

---

### Task 12: Final integration check

**Files:** None (verification only)

- [ ] **Step 1: Full build**

Run: `cargo build --workspace`
Expected: builds successfully

- [ ] **Step 2: Run tests again**

Run: `cargo test --workspace`
Expected: all tests pass

- [ ] **Step 3: Manual check**

```bash
git diff --stat main
```

Verify all changed files are intentional.

- [ ] **Step 4: Commit if any format changes needed**

```bash
cargo fmt --all -- --check
```
If format issues, fix:
```bash
cargo fmt --all
git add -u
git commit -m "style: cargo fmt"
```

---

### Implementation Notes

**For the engine `reconcile_lora`**: Since the engine is consumed by `load_with_progress` and stored in `self.engine: Option<VoxCPMEngine>`, calling `engine.reconcile_lora(...)` in the `Synthesize` handler works because `self.engine` is `&mut Option<VoxCPMEngine>`. The sidecar already does `let Some(engine) = self.engine.as_mut()` before the synthesis block.

**Quick-dedup check**: The `reconcile_lora` method already checks `self.active_lora_path == requested` before doing any work, so consecutive identical requests are no-ops.

**Cache lifetime**: The LoRA cache lives as long as the engine. When a new model is loaded (which produces a new engine), the cache is cleared. This is correct — different base models have incompatible LoRA adapters anyway.
