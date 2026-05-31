# LoRA Dropdown Split: Dynamic LoRA Selection and Switching

## Summary

Split LoRA selection from model discovery and loading. Instead of listing combined "model | lora" entries in the model dropdown, show only base models. Add a separate LoRA dropdown next to the Load button, populated on demand after a base model loads. The inference engine dynamically loads/unloads LoRA checkpoints per synthesis request, with caching to avoid redundant I/O.

---

## 1. Model & LoRA Discovery

### Model discovery (model_discovery.rs)

`discover_models` is simplified to produce only base model entries. For each directory containing `model.gguf`, one `ModelChoice` is created with `lora_path: None`. The scanning loop that discovered LoRA `.gguf` files inside the model directory and created combined entries is removed.

### LoRA discovery

New function `discover_loras(model_dir: &Path) -> Result<Vec<LoraEntry>>` scans the model directory for `.gguf` files that are not `model.gguf`. Each entry provides:

```rust
pub struct LoraEntry {
    pub id: String,          // file stem, e.g. "style-v1"
    pub display_name: String,
}
```

The `id` is used to reconstruct the full path (`model_dir.join("{id}.gguf")`) when building synthesis requests. Called on demand after a model loads, not at app startup.

### ModelChoice

Fields unchanged, but `lora_path` is always `None`. The existing `lora_bytes` field remains but is always 0.

---

## 2. AppSnapshot & AppCore State

### AppSnapshot additions

```rust
pub struct AppSnapshot {
    // ... existing fields ...
    pub available_loras: Vec<LoraEntry>,
    pub selected_lora_id: Option<String>,
}
```

- `available_loras` — empty when no model is loaded; populated after a successful model load
- `selected_lora_id` — the user's current LoRA dropdown selection; resets to `None` when a new model loads

### AppCore internal state

- Tracks `selected_lora_id: Option<String>`
- On successful model load: calls `discover_loras(choice.model_dir)`, populates `available_loras`, sets `selected_lora_id = None`
- On `enqueue_generation`: snapshots `selected_lora_id` into the queue item's `RequestSnapshot`
- `synthesis_request()` includes `lora_path` reconstructed from the snapshot: `snapshot.lora_id.map(|id| model_dir.join(format!("{id}.gguf")))`

### RequestSnapshot

Gains `lora_id: Option<String>` for per-item LoRA snapshot.

---

## 3. Protocol Changes

### SynthesisRequest (voxui-inference)

```rust
pub struct SynthesisRequest {
    // ... existing fields ...
    pub lora_path: Option<PathBuf>,
}
```

### SynthesisRequestDto (voxui-sidecar-protocol)

```rust
pub struct SynthesisRequestDto {
    // ... existing fields ...
    pub lora_path: Option<PathBuf>,
}
```

### SidecarCommand::LoadModel

The `lora_path` field is removed. Model load only loads the base model; LoRA is handled per-synthesis.

### Mapping

- `synthesis_request_dto()` includes `lora_path` from the request
- `synthesis_request_from_dto()` includes `lora_path` from the DTO (default `None` for the `..SynthesisRequest::default()` spread)

---

## 4. UI — LoRA Dropdown

### Header component

A new `<CustomSelect>` for LoRA is added to the right of the Load button.

New props:
- `loras: Vec<LoraEntry>` — available LoRA entries
- `selected_lora_id: Option<String>` — current selection
- `lora_disabled: bool` — disabled when no model is loaded or the LoRA list is empty
- `on_lora_select: impl Fn(Option<String>)` — callback on selection change

The dropdown prepends a "None" entry as the first option. Selecting "None" sends `None` to the callback; selecting a LoRA sends `Some(id)`.

### app.rs integration

- `lora_disabled` is `true` when `loaded_model_id.is_none()` or `available_loras.is_empty()`
- On LoRA select, an `AppCore` config patch sets `selected_lora_id`, triggering a snapshot refresh
- The dropdown visually resets to "None" when `loaded_model_id` changes (new model loaded)

---

## 5. Engine — LoRA Switching with Caching

### VoxCPMEngine additions

```rust
pub struct VoxCPMEngine {
    // ... existing fields ...
    lora: Option<LoraAdapter>,           // currently active
    lora_cache: HashMap<PathBuf, LoraAdapter>,  // cached for re-apply
}
```

### Pre-synthesis logic

In `handle_command_with_emit` for `Synthesize`, before executing synthesis, the engine reconciles the request's `lora_path` with its current LoRA state:

| Current active `lora` | Request `lora_path` | Action |
|---|---|---|
| `None` | `None` | No-op |
| `Some(adapter)` | `Some(path)` same path | No-op |
| `Some(adapter)` | `None` | `self.lora = None` (keep adapter in cache) |
| `None` | `Some(path)` | Lookup cache → restore; on miss, load from disk + cache |
| `Some(old)` | `Some(new)` different path | `self.lora = None`, then load new (cache or disk) |

Cache key is the canonicalized path. Two requests with the same path do not re-read the file.

### Model load

`VoxCPMEngine::load_with_progress` no longer calls `load_lora`. `self.lora` and `self.lora_cache` start empty after a model load. The existing `load_lora` method is kept but only called internally from the pre-synthesis reconciliation step.

### Error handling

If a LoRA file fails to load (missing, corrupt, incompatible architecture), the synthesis fails with an error. The engine clears `self.lora` on load failure to avoid a stale state.

---

## 6. Impact Summary

### Removed
- Combined "model | lora" entries in model discovery
- `lora_path` from `SidecarCommand::LoadModel`
- LoRA loading during initial model load

### Added
- `discover_loras()` function
- `LoraEntry` type
- `available_loras` and `selected_lora_id` to `AppSnapshot`
- LoRA dropdown in `Header`
- `lora_path` to `SynthesisRequest` and `SynthesisRequestDto`
- `lora_id` to `RequestSnapshot`
- `lora_cache: HashMap<PathBuf, LoraAdapter>` to `VoxCPMEngine`
- Pre-synthesis LoRA reconciliation in engine

### Existing data preserved
- `ModelChoice.lora_path` field retained (always `None`)
- `LoraAdapter` and all its methods unchanged
- `BaseLM::forward_embed_with_lora` unchanged
- `DiT::forward` lora-aware forwarding unchanged
