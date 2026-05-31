# Generation Request Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deduplication guard to the TTS generation enqueue path that compares new requests against recent history items and enqueues duplicates with a `Dedupped` status instead of `Queued`.

**Architecture:** Add a `Dedupped` variant to `HistoryStatus`, a `created_at` timestamp to `HistoryItem`, and two config fields (`dedup_window_secs`, `dedup_edit_threshold`) to `AppConfig`. The dedup check runs in `AppCore::enqueue_generation` by iterating the queue in reverse, normalizing text (lowercase + collapse whitespace), computing Levenshtein distance, and breaking when outside the time window.

**Tech Stack:** Rust, Tauri 2, Leptos, `levenshtein` crate

---

### Task 1: Add `levenshtein` dependency

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/Cargo.toml`

- [ ] **Step 1: Add `levenshtein` to Cargo.toml dependencies**

```toml
levenshtein = "1"
```

Add it after the `uuid` line:

```toml
uuid = { version = "1", features = ["v4", "serde"] }
levenshtein = "1"
```

- [ ] **Step 2: Build to verify dependency resolves**

Run: `cargo check -p voxui-desktop 2>&1`
Expected: No errors related to the new dependency.

- [ ] **Step 3: Commit**

```bash
git add crates/voxui-desktop/src-tauri/Cargo.toml
git commit -m "chore: add levenshtein dependency for dedup"
```

---

### Task 2: Update `HistoryStatus` and `HistoryItem` in `generation_queue.rs`

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/generation_queue.rs`

- [ ] **Step 1: Add `Dedupped` variant to `HistoryStatus`**

Add `Dedupped` before the closing `}` of the enum (line 14):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryStatus {
    Queued,
    Generating,
    Canceled,
    Failed,
    Ready,
    Playing,
    Dedupped,
}
```

- [ ] **Step 2: Add `created_at` field to `HistoryItem`**

Add `#[serde(default)]` on the field for backward compatibility:

```rust
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
    #[serde(default)]
    pub created_at: u64,
}
```

- [ ] **Step 3: Update `enqueue` to accept `created_at` and `status` parameters**

Replace the `enqueue` method body:

```rust
pub fn enqueue(
    &mut self,
    text: String,
    loaded_model_id: impl Into<String>,
    config: &AppConfig,
    created_at: u64,
    status: HistoryStatus,
) -> String {
    let id = Uuid::new_v4().to_string();
    self.items.push(HistoryItem {
        id: id.clone(),
        text,
        status,
        progress_current: 0,
        progress_total: 0,
        error: None,
        has_audio: false,
        snapshot: Self::snapshot(loaded_model_id, config),
        created_at,
    });
    id
}
```

- [ ] **Step 4: Build to verify compilation**

Run: `cargo build -p voxui-desktop 2>&1`
Expected: Compilation errors in callers (tests, app_core) — these will be fixed in subsequent tasks.

- [ ] **Step 5: Commit**

```bash
git add crates/voxui-desktop/src-tauri/src/generation_queue.rs
git commit -m "feat: add Dedupped status and created_at timestamp to queue model"
```

---

### Task 3: Add dedup config fields to `AppConfig`

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/types.rs`

- [ ] **Step 1: Add `dedup_window_secs` and `dedup_edit_threshold` to `AppConfig`**

Add after `pub auto_period: bool,` (line 381):

```rust
pub dedup_window_secs: u64,
pub dedup_edit_threshold: usize,
```

- [ ] **Step 2: Set defaults in `AppConfig::default()`**

Add after `auto_period: true,` (line 399):

```rust
dedup_window_secs: 10,
dedup_edit_threshold: 1,
```

- [ ] **Step 3: Build to verify**

Run: `cargo build -p voxui-desktop 2>&1`
Expected: Still compilation errors from old `enqueue` callers. No new errors related to AppConfig.

- [ ] **Step 4: Commit**

```bash
git add crates/voxui-desktop/src-tauri/src/types.rs
git commit -m "feat: add dedup config fields to AppConfig"
```

---

### Task 4: Add dedup logic to `AppCore::enqueue_generation`

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/app_core.rs`

- [ ] **Step 1: Add `use std::time::SystemTime;` import**

Add to the imports at the top (after `use std::sync::{mpsc, Arc};` on line 4):

```rust
use std::time::SystemTime;
```

- [ ] **Step 2: Add `normalize_for_compare` helper function**

Add at the bottom of the file, after `fn lang_to_code` (line ~1170):

```rust
fn normalize_for_compare(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn levenshtein_distance(a: &str, b: &str) -> usize {
    levenshtein::levenshtein(a, b)
}
```

- [ ] **Step 3: Replace `enqueue_generation` with dedup logic**

Replace the existing `enqueue_generation` method (lines 391-415). Cloning items avoids mutable-borrow-while-iterating conflicts:

```rust
pub fn enqueue_generation(&mut self, text: String) -> Result<HistoryItem> {
    let text = text.trim().to_string();
    if text.is_empty() {
        bail!("input text is empty");
    }
    if text.chars().count() > self.config.max_input_chars {
        bail!(
            "input text exceeds maximum length of {} characters",
            self.config.max_input_chars
        );
    }

    let loaded_model_id = self
        .loaded_model_id
        .clone()
        .context("no model loaded for generation")?;

    let now = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let deduped = self.config.dedup_window_secs > 0 && {
        let normalized = normalize_for_compare(&text);
        let items: Vec<_> = self.queue.items().iter().rev().cloned().collect();
        let mut found = false;
        for item in &items {
            if now.saturating_sub(item.created_at) > self.config.dedup_window_secs {
                break;
            }
            let item_normalized = normalize_for_compare(&item.text);
            if levenshtein_distance(&normalized, &item_normalized)
                <= self.config.dedup_edit_threshold
            {
                found = true;
                break;
            }
        }
        found
    };

    let status = if deduped {
        HistoryStatus::Dedupped
    } else {
        HistoryStatus::Queued
    };

    let id = self.queue.enqueue(text, loaded_model_id, &self.config, now, status);
    self.queue
        .items()
        .iter()
        .find(|item| item.id == id)
        .cloned()
        .context("queued item was not found after enqueue")
}
```

- [ ] **Step 4: Build to verify**

Run: `cargo build -p voxui-desktop 2>&1`
Expected: Compilation errors only in tests that call `enqueue()` directly. `enqueue_generation` should compile.

- [ ] **Step 5: Commit**

```bash
git add crates/voxui-desktop/src-tauri/src/app_core.rs
git commit -m "feat: add dedup guard to enqueue_generation"
```

---

### Task 5: Update queue tests for new `enqueue` signature

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/tests/queue_tests.rs`

- [ ] **Step 1: Update all `enqueue` calls to pass `created_at` and `status`**

Every `queue.enqueue(text, model, &config)` call needs to become `queue.enqueue(text, model, &config, 0, HistoryStatus::Queued)`. Here is the complete updated file:

```rust
use voxui_desktop::generation_queue::{GenerationQueue, HistoryStatus};
use voxui_desktop::types::{AppConfig, BackendKind};

fn configured_model(model_id: &str) -> AppConfig {
    AppConfig {
        selected_model_id: Some(model_id.to_string()),
        backend: BackendKind::Cuda,
        ..AppConfig::default()
    }
}

#[test]
fn enqueue_captures_settings_and_preserves_order() {
    let mut config = configured_model("selected-model-a");
    config.generation.cfg_value = 3.25;
    config.generation.inference_timesteps = 12;
    let mut queue = GenerationQueue::default();

    let first_id = queue.enqueue("first text".to_string(), "loaded-model-a", &config, 1, HistoryStatus::Queued);
    config.selected_model_id = Some("selected-model-b".to_string());
    config.backend = BackendKind::Cpu;
    config.generation.cfg_value = 1.5;
    let second_id = queue.enqueue("second text".to_string(), "loaded-model-b", &config, 2, HistoryStatus::Queued);

    let items = queue.items();

    assert_eq!(items.len(), 2);
    assert_eq!(items[0].id, first_id);
    assert_eq!(items[0].text, "first text");
    assert_eq!(items[0].status, HistoryStatus::Queued);
    assert_eq!(items[0].created_at, 1);
    assert_eq!(items[0].snapshot.model_id, "loaded-model-a");
    assert_eq!(items[0].snapshot.backend, BackendKind::Cuda);
    assert_eq!(items[0].snapshot.generation.cfg_value, 3.25);
    assert_eq!(items[0].snapshot.generation.inference_timesteps, 12);
    assert_eq!(items[1].id, second_id);
    assert_eq!(items[1].text, "second text");
    assert_eq!(items[1].created_at, 2);
    assert_eq!(items[1].snapshot.model_id, "loaded-model-b");
    assert_eq!(items[1].snapshot.backend, BackendKind::Cpu);
    assert_eq!(queue.next_queued_id(), Some(first_id.as_str()));
}

#[test]
fn enqueue_with_dedupped_status_is_not_picked_up_by_next_queued() {
    let config = configured_model("model-a");
    let mut queue = GenerationQueue::default();

    let dedupped_id = queue.enqueue("text".to_string(), "model-a", &config, 1, HistoryStatus::Dedupped);
    let queued_id = queue.enqueue("text".to_string(), "model-a", &config, 2, HistoryStatus::Queued);

    assert_eq!(queue.next_queued_id(), Some(queued_id.as_str()));

    let items = queue.items();
    assert_eq!(items[0].id, dedupped_id);
    assert_eq!(items[0].status, HistoryStatus::Dedupped);
    assert_eq!(items[1].id, queued_id);
    assert_eq!(items[1].status, HistoryStatus::Queued);
}

#[test]
fn cancel_queued_item_marks_it_canceled() {
    let config = configured_model("model-a");
    let mut queue = GenerationQueue::default();
    let first_id = queue.enqueue("first text".to_string(), "model-a", &config, 1, HistoryStatus::Queued);
    let second_id = queue.enqueue("second text".to_string(), "model-a", &config, 2, HistoryStatus::Queued);

    assert!(queue.cancel_queued(&first_id));

    let items = queue.items();
    assert_eq!(items[0].status, HistoryStatus::Canceled);
    assert_eq!(items[0].error, None);
    assert_eq!(queue.next_queued_id(), Some(second_id.as_str()));
}

#[test]
fn regeneration_attempt_keeps_existing_audio_flag_until_success() {
    let mut config = configured_model("model-a");
    let mut queue = GenerationQueue::default();
    let id = queue.enqueue("text".to_string(), "loaded-model-a", &config, 1, HistoryStatus::Queued);
    queue.mark_ready(&id);

    config.selected_model_id = Some("selected-but-not-loaded-model-b".to_string());
    config.backend = BackendKind::Cpu;
    config.generation.cfg_value = 4.0;

    assert!(queue.start_regeneration(&id, "loaded-model-a", &config));

    let item = &queue.items()[0];
    assert_eq!(item.status, HistoryStatus::Queued);
    assert_eq!(item.progress_current, 0);
    assert_eq!(item.progress_total, 0);
    assert_eq!(item.error, None);
    assert!(item.has_audio);
    assert_eq!(item.snapshot.model_id, "loaded-model-a");
    assert_eq!(item.snapshot.backend, BackendKind::Cpu);
    assert_eq!(item.snapshot.generation.cfg_value, 4.0);
}

#[test]
fn canceling_regeneration_with_existing_audio_returns_item_to_ready() {
    let config = configured_model("model-a");
    let mut queue = GenerationQueue::default();
    let id = queue.enqueue("text".to_string(), "loaded-model-a", &config, 1, HistoryStatus::Queued);
    queue.mark_ready(&id);

    assert!(queue.start_regeneration(&id, "loaded-model-a", &config));
    assert!(queue.mark_canceled(&id));

    let item = &queue.items()[0];
    assert_eq!(item.status, HistoryStatus::Ready);
    assert!(item.has_audio);
    assert_eq!(item.error, None);
}

#[test]
fn playback_marks_ready_audio_as_playing_and_stops_it() {
    let config = configured_model("model-a");
    let mut queue = GenerationQueue::default();
    let ready_id = queue.enqueue("ready text".to_string(), "model-a", &config, 1, HistoryStatus::Queued);
    let queued_id = queue.enqueue("queued text".to_string(), "model-a", &config, 2, HistoryStatus::Queued);

    assert!(!queue.mark_playing(&ready_id));

    queue.mark_ready(&ready_id);
    assert!(queue.mark_playing(&ready_id));
    assert!(!queue.mark_playing(&queued_id));

    let items = queue.items();
    assert_eq!(items[0].status, HistoryStatus::Playing);
    assert_eq!(items[1].status, HistoryStatus::Queued);

    assert_eq!(queue.mark_all_stopped(), Some(ready_id.clone()));

    let items = queue.items();
    assert_eq!(items[0].status, HistoryStatus::Ready);
    assert_eq!(items[1].status, HistoryStatus::Queued);
}
```

- [ ] **Step 2: Run queue tests to verify**

Run: `cargo test -p voxui-desktop --test queue_tests 2>&1`
Expected: All 7 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/voxui-desktop/src-tauri/tests/queue_tests.rs
git commit -m "test: update queue tests for new enqueue signature, add Dedupped test"
```

---

### Task 6: Add dedup tests in app_core tests

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/app_core.rs` (test module at bottom)

- [ ] **Step 1: Add dedup test functions**

Add after `cancel_generation_item_only_cancels_queued_items` test (before line ~1450):

```rust
#[test]
fn dedup_drops_similar_text_within_window() {
    let mut config = AppConfig::default();
    config.dedup_window_secs = 60;
    config.dedup_edit_threshold = 1;
    let mut core = AppCore::from_config(config).unwrap();
    core.set_loaded_model_for_test("alpha".to_string());

    let first = core.enqueue_generation("hello world".to_string()).unwrap();
    assert_eq!(first.status, HistoryStatus::Queued);

    let dup = core.enqueue_generation("hello world".to_string()).unwrap();
    assert_eq!(dup.status, HistoryStatus::Dedupped);
}

#[test]
fn dedup_allows_different_text() {
    let mut config = AppConfig::default();
    config.dedup_window_secs = 60;
    config.dedup_edit_threshold = 1;
    let mut core = AppCore::from_config(config).unwrap();
    core.set_loaded_model_for_test("alpha".to_string());

    let first = core.enqueue_generation("hello".to_string()).unwrap();
    assert_eq!(first.status, HistoryStatus::Queued);

    let second = core.enqueue_generation("world".to_string()).unwrap();
    assert_eq!(second.status, HistoryStatus::Queued);
}

#[test]
fn dedup_window_zero_disables_dedup() {
    let mut config = AppConfig::default();
    config.dedup_window_secs = 0;
    let mut core = AppCore::from_config(config).unwrap();
    core.set_loaded_model_for_test("alpha".to_string());

    let first = core.enqueue_generation("hello".to_string()).unwrap();
    assert_eq!(first.status, HistoryStatus::Queued);

    let second = core.enqueue_generation("hello".to_string()).unwrap();
    assert_eq!(second.status, HistoryStatus::Queued);
}

#[test]
fn dedup_normalizes_whitespace_and_case() {
    let mut config = AppConfig::default();
    config.dedup_window_secs = 60;
    config.dedup_edit_threshold = 1;
    let mut core = AppCore::from_config(config).unwrap();
    core.set_loaded_model_for_test("alpha".to_string());

    let first = core.enqueue_generation("  Hello   World  ".to_string()).unwrap();
    assert_eq!(first.status, HistoryStatus::Queued);

    let dup = core.enqueue_generation("hello world".to_string()).unwrap();
    assert_eq!(dup.status, HistoryStatus::Dedupped);
}

#[test]
fn dedup_edit_threshold_zero_requires_exact_normalized_match() {
    let mut config = AppConfig::default();
    config.dedup_window_secs = 60;
    config.dedup_edit_threshold = 0;
    let mut core = AppCore::from_config(config).unwrap();
    core.set_loaded_model_for_test("alpha".to_string());

    let first = core.enqueue_generation("hello world".to_string()).unwrap();
    assert_eq!(first.status, HistoryStatus::Queued);

    let allowed = core.enqueue_generation("hello world!".to_string()).unwrap();
    assert_eq!(allowed.status, HistoryStatus::Queued);
}
```

- [ ] **Step 2: Run the new dedup tests**

Run: `cargo test -p voxui-desktop -- dedup 2>&1`
Expected: All 5 dedup tests PASS.

- [ ] **Step 3: Run ALL tests to verify no regressions**

Run: `cargo test -p voxui-desktop 2>&1`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/voxui-desktop/src-tauri/src/app_core.rs
git commit -m "test: add dedup guard unit tests"
```

---

### Task 7: Update frontend `HistoryStatus` and `HistoryItem` in `tauri_api.rs`

**Files:**
- Modify: `crates/voxui-desktop/src/tauri_api.rs`

- [ ] **Step 1: Add `Dedupped` to frontend `HistoryStatus`**

Add after `Playing,` (line 498):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryStatus {
    Queued,
    Generating,
    Canceled,
    Failed,
    Ready,
    Playing,
    Dedupped,
}
```

- [ ] **Step 2: Add `created_at` to frontend `HistoryItem`**

Add after `pub snapshot: RequestSnapshot,` (line 451):

```rust
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
    #[serde(default)]
    pub created_at: u64,
}
```

- [ ] **Step 3: Build frontend to verify**

Run: `cargo build -p voxui-desktop 2>&1`
Expected: Compilation succeeds. The `Dedupped` variant match in `history.rs` will cause a non-exhaustive match warning/error in a later task.

- [ ] **Step 4: Commit**

```bash
git add crates/voxui-desktop/src/tauri_api.rs
git commit -m "feat: add Dedupped status and created_at to frontend API types"
```

---

### Task 8: Update history component for `Dedupped` status

**Files:**
- Modify: `crates/voxui-desktop/src/components/history.rs`
- Modify: `crates/voxui-desktop/src/i18n.rs`

- [ ] **Step 1: Add `history_status_dedupped` to `Labels` struct**

Add after `pub history_status_canceled: &'static str,` (line 61 in i18n.rs):

```rust
pub history_status_dedupped: &'static str,
```

- [ ] **Step 2: Add Chinese translation**

Add after `history_status_canceled: "已取消",` (line 169 in i18n.rs):

```rust
history_status_dedupped: "已去重",
```

- [ ] **Step 3: Add English translation**

Add after `history_status_canceled: "Canceled",` (line 274 in i18n.rs):

```rust
history_status_dedupped: "Dedupped",
```

- [ ] **Step 4: Add `Dedupped` arm to `status_label` in `history.rs`**

Add after `HistoryStatus::Canceled => labels.history_status_canceled,` (line 134):

```rust
HistoryStatus::Dedupped => labels.history_status_dedupped,
```

- [ ] **Step 5: Build to verify**

Run: `cargo build -p voxui-desktop 2>&1`
Expected: Full compilation succeeds with no warnings.

- [ ] **Step 6: Run all tests**

Run: `cargo test -p voxui-desktop 2>&1`
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/voxui-desktop/src/components/history.rs crates/voxui-desktop/src/i18n.rs
git commit -m "feat: add Dedupped status display in history and i18n"
```

---

### Task 9: Final integration verification

- [ ] **Step 1: Run full build**

Run: `cargo build -p voxui-desktop 2>&1`
Expected: Build succeeds with zero errors and zero warnings.

- [ ] **Step 2: Run full test suite**

Run: `cargo test -p voxui-desktop 2>&1`
Expected: All tests PASS.

- [ ] **Step 3: Check for unused warnings**

Run: `cargo clippy -p voxui-desktop -- -D warnings 2>&1`
Expected: No clippy warnings.

- [ ] **Step 4: Commit**

```bash
git add -A
git diff --cached --stat
git commit -m "chore: final verification after dedup implementation"
```
