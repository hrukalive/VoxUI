# Generation Request Deduplication

## Problem

The streamer can accidentally send the same TTS generation request multiple times, producing duplicate tasks in the generation queue. They must manually cancel duplicates.

## Solution

Add a deduplication guard to the enqueue path that compares each incoming request against recent items in the generation history. Similar items within a configurable time window are enqueued with a `Dedupped` status instead of `Queued`, so they appear in history but do not auto-generate. The streamer can still manually regenerate a dedupped item.

## Data Model Changes

### `HistoryStatus` enum (generation_queue.rs)

Add new variant:

```rust
Dedupped,
```

### `HistoryItem` struct (generation_queue.rs)

Add field:

```rust
pub created_at: u64,  // Unix timestamp in seconds
```

Serialization: `#[serde(default)]` for backward compatibility with existing stored/emitted data.

### `AppConfig` (types.rs)

Add fields:

```rust
pub dedup_window_secs: u64,      // default 10
pub dedup_edit_threshold: usize,  // default 1
```

Defaults applied in `AppConfig::default()`. Window of 0 disables dedup.

## Algorithm

### Text Normalization

`normalize_for_compare(text: &str) -> String`:
- Convert to lowercase
- Collapse all whitespace runs to single spaces
- Trim leading/trailing whitespace

### Dedup Check (AppCore::enqueue_generation)

After existing validation (empty, max chars, model loaded), before enqueuing:

1. If `dedup_window_secs == 0` → skip dedup entirely, enqueue normally
2. Compute `now` = current Unix timestamp (seconds)
3. Normalize incoming text
4. Iterate `self.queue.items()` in **reverse** (newest to oldest)
5. For each item:
   - If `now - item.created_at > dedup_window_secs` → **break** (remaining items are older)
   - Normalize item text
   - Compute Levenshtein distance between normalized strings
   - If distance `<= dedup_edit_threshold` → enqueue with `HistoryStatus::Dedupped`, return `Ok(item)`
6. If no match → enqueue with `HistoryStatus::Queued` as before

### GenerationQueue::enqueue signature

```rust
pub fn enqueue(
    &mut self,
    text: String,
    loaded_model_id: impl Into<String>,
    config: &AppConfig,
    created_at: u64,
    status: HistoryStatus,
) -> String
```

Existing callers (tests, regeneration path) pass `HistoryStatus::Queued`.

## Behavior

### Dedupped items

- Visible in generation history with status "Dedupped"
- **Not** picked up by `next_queued_id()` → skipped by `kick_generation_queue` → never auto-generate
- Regenerate button remains available → `regenerate_item` sets status to `Queued` unconditionally, bypassing dedup

### Regeneration path

Unchanged. `regenerate_item` calls `start_regeneration()` which sets status to `Queued` directly. Dedup guard only applies to `enqueue_generation`.

### Levenshtein implementation

Add the `levenshtein` crate (zero-dependency, ~500 LOC) to Cargo.toml. Input texts are short (typically under 200 chars), so the O(n*m) algorithm is fine.

## Frontend Changes

### tauri_api.rs

- `HistoryItem` struct gains `created_at: u64`
- `#[serde(default)]` for backward compatibility

### app.rs / history component

- Display dedupped items in the history list with their status
- Regenerate button enabled for dedupped items (as with Ready/Failed/Canceled statuses)
- No other behavioral changes required

## Files Affected

| File | Change |
|------|--------|
| `crates/voxui-desktop/src-tauri/src/generation_queue.rs` | Add `Dedupped` variant, `created_at` field, `status` param to `enqueue` |
| `crates/voxui-desktop/src-tauri/src/types.rs` | Add `dedup_window_secs`, `dedup_edit_threshold` to `AppConfig` |
| `crates/voxui-desktop/src-tauri/src/app_core.rs` | Add dedup check + Levenshtein in `enqueue_generation` |
| `crates/voxui-desktop/src/tauri_api.rs` | Add `created_at` field to `HistoryItem` |
| `crates/voxui-desktop/src-tauri/tests/queue_tests.rs` | Update tests for new `enqueue` signature |
| `crates/voxui-desktop/src-tauri/Cargo.toml` | Add `levenshtein` dependency |
| `crates/voxui-desktop/src/components/history.rs` | Handle `Dedupped` status display, Regenerate button logic |
