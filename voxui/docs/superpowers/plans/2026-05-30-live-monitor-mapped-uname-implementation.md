# Live Monitor Mapped Username Editing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-item live monitor button that opens a modal for editing and immediately saving a viewer's `mapped_uname`.

**Architecture:** Implement the feature inside the existing Leptos `LiveMonitor` component. The modal writes through the existing `LiveConfigPatch.mapped_unames` pipeline so persistence and feed row re-evaluation stay centralized.

**Tech Stack:** Rust, Leptos 0.7 CSR, Tauri frontend bindings, existing CSS in `crates/voxui-desktop/src/styles.css`.

---

## File Structure

- Modify `crates/voxui-desktop/src/components/live_monitor.rs`
  - Add small helper functions for mapped-name initial value and attention-state styling.
  - Add local modal state and save behavior to `LiveMonitor`.
  - Add a per-row SVG icon button that is always visible, independent of send/auto-generation controls.
  - Add focused unit/source tests in the existing `#[cfg(test)]` module.
- Modify `crates/voxui-desktop/src/styles.css`
  - Add icon-button sizing for the primary mapped-name state.
  - Add layout for the mapped-name modal fields.
  - Add a send-action wrapper so send buttons can still hide during auto-generation while the mapped-name button remains visible.

No backend files need changes. `AppCore::apply_live_patch` already persists `mapped_unames`, and `LiveState::snapshot` already recomputes `mapped_uname` and suggestions from config.

---

### Task 1: Add Mapped Name Helper Tests And Helpers

**Files:**
- Modify: `crates/voxui-desktop/src/components/live_monitor.rs`

- [ ] **Step 1: Write failing helper tests**

In `crates/voxui-desktop/src/components/live_monitor.rs`, inside the existing `#[cfg(test)] mod tests`, add these tests after `live_item_render_key_changes_with_rendered_text`:

```rust
    #[test]
    fn mapped_uname_input_prefers_existing_mapping() {
        let mut mapped_unames = BTreeMap::new();
        mapped_unames.insert("u1".to_string(), "A-chan".to_string());

        assert_eq!(
            mapped_uname_initial_value(&mapped_unames, "u1", "Alice"),
            "A-chan"
        );
        assert_eq!(
            mapped_uname_initial_value(&mapped_unames, "u2", "Bob"),
            "Bob"
        );
    }

    #[test]
    fn mapped_uname_attention_tracks_missing_or_changed_names() {
        let mut mapped_unames = BTreeMap::new();
        mapped_unames.insert("u1".to_string(), "A-chan".to_string());

        let mut original_unames = BTreeMap::new();
        original_unames.insert("u1".to_string(), "Alice".to_string());

        assert!(!mapped_uname_needs_attention(
            &mapped_unames,
            &original_unames,
            "u1",
            "Alice"
        ));
        assert!(mapped_uname_needs_attention(
            &mapped_unames,
            &original_unames,
            "u2",
            "Bob"
        ));
        assert!(mapped_uname_needs_attention(
            &mapped_unames,
            &original_unames,
            "u1",
            "AliceNew"
        ));
    }

    #[test]
    fn mapped_uname_button_class_matches_attention_state() {
        let mut mapped_unames = BTreeMap::new();
        mapped_unames.insert("u1".to_string(), "A-chan".to_string());

        let mut original_unames = BTreeMap::new();
        original_unames.insert("u1".to_string(), "Alice".to_string());

        assert_eq!(
            mapped_uname_button_class(&mapped_unames, &original_unames, "u1", "Alice"),
            "live-monitor-button live-map-button"
        );
        assert_eq!(
            mapped_uname_button_class(&mapped_unames, &original_unames, "u1", "AliceNew"),
            "primary-button live-map-button"
        );
    }
```

- [ ] **Step 2: Run the helper tests and verify they fail**

Run:

```bash
cargo test -p voxui-desktop-ui mapped_uname
```

Expected: FAIL with unresolved function errors for `mapped_uname_initial_value`, `mapped_uname_needs_attention`, and `mapped_uname_button_class`.

- [ ] **Step 3: Implement the minimal helpers**

At the top of `crates/voxui-desktop/src/components/live_monitor.rs`, change the imports from:

```rust
use std::collections::HashSet;
```

to:

```rust
use std::collections::{BTreeMap, HashSet};
```

Then add these helpers after `live_item_render_key`:

```rust
fn mapped_uname_initial_value(
    mapped_unames: &BTreeMap<String, String>,
    open_id: &str,
    uname: &str,
) -> String {
    mapped_unames
        .get(open_id)
        .cloned()
        .unwrap_or_else(|| uname.to_string())
}

fn mapped_uname_needs_attention(
    mapped_unames: &BTreeMap<String, String>,
    original_unames: &BTreeMap<String, String>,
    open_id: &str,
    uname: &str,
) -> bool {
    !mapped_unames.contains_key(open_id)
        || original_unames
            .get(open_id)
            .map(|original| original != uname)
            .unwrap_or(true)
}

fn mapped_uname_button_class(
    mapped_unames: &BTreeMap<String, String>,
    original_unames: &BTreeMap<String, String>,
    open_id: &str,
    uname: &str,
) -> &'static str {
    if mapped_uname_needs_attention(mapped_unames, original_unames, open_id, uname) {
        "primary-button live-map-button"
    } else {
        "live-monitor-button live-map-button"
    }
}
```

- [ ] **Step 4: Run the helper tests and verify they pass**

Run:

```bash
cargo test -p voxui-desktop-ui mapped_uname
```

Expected: PASS for the three mapped username helper tests.

- [ ] **Step 5: Commit the helper work**

```bash
git add crates/voxui-desktop/src/components/live_monitor.rs
git commit -m "test: cover live monitor mapped username state"
```

---

### Task 2: Add The Per-Item Button And Modal

**Files:**
- Modify: `crates/voxui-desktop/src/components/live_monitor.rs`

- [ ] **Step 1: Write a failing source-level wiring test**

In `crates/voxui-desktop/src/components/live_monitor.rs`, inside the existing test module, add this test after `monitor_buttons_use_svg_icons`:

```rust
    #[test]
    fn monitor_renders_mapped_uname_modal_and_button() {
        let source = include_str!("live_monitor.rs");

        assert!(
            source.contains("mapped-uname-modal"),
            "Monitor should render the focused mapped username modal"
        );
        assert!(
            source.contains("live-map-button"),
            "Monitor should render a mapped username button for live items"
        );
        assert!(
            source.contains("mapped_uname_button_class"),
            "Monitor should derive mapped username button styling from mapping state"
        );
        assert!(
            source.contains("mapped_uname_initial_value"),
            "Monitor should initialize modal input from mapping config or current uname"
        );
    }
```

- [ ] **Step 2: Run the wiring test and verify it fails**

Run:

```bash
cargo test -p voxui-desktop-ui monitor_renders_mapped_uname_modal_and_button
```

Expected: FAIL with the message `Monitor should render the focused mapped username modal`.

- [ ] **Step 3: Add modal draft state**

In `crates/voxui-desktop/src/components/live_monitor.rs`, add this struct above `#[component] pub fn LiveMonitor`:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
struct MappedUnameDraft {
    open_id: String,
    uname: String,
    value: String,
}
```

Inside `LiveMonitor`, after the existing signal declarations, add:

```rust
    let (mapped_uname_draft, set_mapped_uname_draft) = signal(None::<MappedUnameDraft>);

    let open_mapped_uname_modal = move |item: LiveMonitorItem| {
        let initial_value =
            mapped_uname_initial_value(&snapshot().config.mapped_unames, &item.open_id, &item.uname);
        set_mapped_uname_draft.set(Some(MappedUnameDraft {
            open_id: item.open_id,
            uname: item.uname,
            value: initial_value,
        }));
    };

    let save_mapped_uname = move || {
        let Some(draft) = mapped_uname_draft.get_untracked() else {
            return;
        };

        let mut mapped_unames = snapshot().config.mapped_unames.clone();
        mapped_unames.insert(draft.open_id, draft.value);
        on_live_patch(LiveConfigPatch {
            mapped_unames: Some(mapped_unames),
            ..LiveConfigPatch::default()
        });
        set_mapped_uname_draft.set(None);
    };
```

- [ ] **Step 4: Add the per-row mapped-name button without hiding it during auto-generation**

In the `<For ... children=move |item| { ... }>` body, after existing item-local values such as `paid`, add:

```rust
                        let item_for_name_edit = item.clone();
                        let open_id_for_name_class = item.open_id.clone();
                        let uname_for_name_class = item.uname.clone();
                        let mapped_uname_button = view! {
                            <button
                                class=move || {
                                    let snapshot = snapshot();
                                    mapped_uname_button_class(
                                        &snapshot.config.mapped_unames,
                                        &snapshot.config.original_unames,
                                        &open_id_for_name_class,
                                        &uname_for_name_class,
                                    )
                                }
                                type="button"
                                title=labels.uname_map
                                aria-label=labels.uname_map
                                on:click=move |_| open_mapped_uname_modal(item_for_name_edit.clone())
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M20 21a8 8 0 0 0-16 0"></path>
                                    <circle cx="12" cy="7" r="4"></circle>
                                    <path d="M19 8v6"></path>
                                    <path d="M22 11h-6"></path>
                                </svg>
                            </button>
                        };
```

Then change the live item actions block from one hideable container:

```rust
                                <div
                                    class="live-item-actions"
                                    class:live-item-actions-hidden=move || {
                                        auto_generation_switch(
                                            snapshot().config.auto_gen_mode,
                                            kind,
                                            &snapshot().config,
                                        )
                                        .is_some()
                                    }
                                >
                                    <button
```

to an always-visible container with only send actions hidden:

```rust
                                <div class="live-item-actions">
                                    {mapped_uname_button}
                                    <div
                                        class="live-send-actions"
                                        class:live-item-actions-hidden=move || {
                                            auto_generation_switch(
                                                snapshot().config.auto_gen_mode,
                                                kind,
                                                &snapshot().config,
                                            )
                                            .is_some()
                                        }
                                    >
                                        <button
```

Close the new nested `<div class="live-send-actions">` immediately after `{switch_button}`:

```rust
                                    {switch_button}
                                    </div>
                                </div>
```

- [ ] **Step 5: Add the modal markup**

In the `view!` returned by `LiveMonitor`, add this `<Show>` after the live feed `</div>` and before the closing `</section>`:

```rust
            <Show when=move || mapped_uname_draft.get().is_some()>
                {move || {
                    let draft = mapped_uname_draft.get().unwrap_or_else(|| MappedUnameDraft {
                        open_id: String::new(),
                        uname: String::new(),
                        value: String::new(),
                    });

                    view! {
                        <div class="modal-backdrop" role="presentation">
                            <section class="modal mapped-uname-modal" role="dialog" aria-modal="true" aria-label=move || labels().uname_map>
                                <header class="modal-header">
                                    <h2>{move || labels().uname_map}</h2>
                                    <button
                                        class="secondary-button"
                                        type="button"
                                        aria-label=move || labels().close
                                        on:click=move |_| set_mapped_uname_draft.set(None)
                                    >
                                        {move || labels().close}
                                    </button>
                                </header>
                                <div class="mapped-uname-form">
                                    <label class="mapped-uname-field">
                                        <span>"open_id"</span>
                                        <code>{draft.open_id.clone()}</code>
                                    </label>
                                    <label class="mapped-uname-field">
                                        <span>"uname"</span>
                                        <strong>{draft.uname.clone()}</strong>
                                    </label>
                                    <label class="mapped-uname-field">
                                        <span>{move || labels().uname_map}</span>
                                        <input
                                            type="text"
                                            aria-label=move || labels().uname_map
                                            prop:value=move || {
                                                mapped_uname_draft
                                                    .get()
                                                    .map(|draft| draft.value)
                                                    .unwrap_or_default()
                                            }
                                            on:input=move |event| {
                                                let value = event_target_value(&event);
                                                set_mapped_uname_draft.update(|draft| {
                                                    if let Some(draft) = draft {
                                                        draft.value = value;
                                                    }
                                                });
                                            }
                                        />
                                    </label>
                                </div>
                                <footer class="mapped-uname-actions">
                                    <button
                                        class="secondary-button"
                                        type="button"
                                        on:click=move |_| set_mapped_uname_draft.set(None)
                                    >
                                        {move || labels().cancel}
                                    </button>
                                    <button class="primary-button" type="button" on:click=move |_| save_mapped_uname()>
                                        {move || labels().save}
                                    </button>
                                </footer>
                            </section>
                        </div>
                    }
                }}
            </Show>
```

- [ ] **Step 6: Run the wiring test and verify it passes**

Run:

```bash
cargo test -p voxui-desktop-ui monitor_renders_mapped_uname_modal_and_button
```

Expected: PASS.

- [ ] **Step 7: Commit the component work**

```bash
git add crates/voxui-desktop/src/components/live_monitor.rs
git commit -m "feat: add live monitor mapped username modal"
```

---

### Task 3: Add Focused Styles For The Button And Modal

**Files:**
- Modify: `crates/voxui-desktop/src/components/live_monitor.rs`
- Modify: `crates/voxui-desktop/src/styles.css`

- [ ] **Step 1: Write a failing style coverage test**

In `crates/voxui-desktop/src/components/live_monitor.rs`, inside the existing test module, add this test after `monitor_renders_mapped_uname_modal_and_button`:

```rust
    #[test]
    fn mapped_uname_modal_styles_are_present() {
        let styles = include_str!("../styles.css");

        assert!(styles.contains(".live-map-button"));
        assert!(styles.contains(".live-send-actions"));
        assert!(styles.contains(".mapped-uname-modal"));
        assert!(styles.contains(".mapped-uname-field"));
        assert!(styles.contains(".mapped-uname-actions"));
    }
```

- [ ] **Step 2: Run the style test and verify it fails**

Run:

```bash
cargo test -p voxui-desktop-ui mapped_uname_modal_styles_are_present
```

Expected: FAIL because the new CSS selectors are not present.

- [ ] **Step 3: Add CSS for the new controls**

In `crates/voxui-desktop/src/styles.css`, after the existing `.live-monitor-button` and `.live-monitor-button:hover:not(:disabled)` block, add:

```css
.live-map-button {
  width: 30px;
  height: 30px;
  min-width: 0;
  padding: 0;
}

.live-map-button.primary-button {
  border-color: var(--primary-bg);
  background: var(--primary-bg);
  color: var(--primary-text);
}
```

After the existing `.live-item-actions` block, add:

```css
.live-send-actions {
  display: grid;
  grid-auto-flow: column;
  gap: 6px;
  align-items: center;
}
```

After the existing `.modal-header h2` block, add:

```css
.mapped-uname-modal {
  width: min(440px, calc(100vw - 48px));
}

.mapped-uname-form {
  display: grid;
  gap: 12px;
  padding: 18px 20px;
}

.mapped-uname-field {
  display: grid;
  gap: 6px;
  min-width: 0;
  color: var(--text-muted);
  font-size: 12px;
}

.mapped-uname-field code,
.mapped-uname-field strong {
  min-width: 0;
  overflow-wrap: anywhere;
  color: var(--text);
  font-size: 13px;
  font-weight: 500;
}

.mapped-uname-field input {
  width: 100%;
  min-width: 0;
  height: 34px;
  border: 1px solid var(--control-border);
  border-radius: 5px;
  background: var(--control-bg);
  color: var(--text);
  padding: 0 10px;
  font: inherit;
}

.mapped-uname-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  padding: 0 20px 18px;
}
```

- [ ] **Step 4: Run the style test and verify it passes**

Run:

```bash
cargo test -p voxui-desktop-ui mapped_uname_modal_styles_are_present
```

Expected: PASS.

- [ ] **Step 5: Commit the style work**

```bash
git add crates/voxui-desktop/src/components/live_monitor.rs crates/voxui-desktop/src/styles.css
git commit -m "style: add live monitor mapped username controls"
```

---

### Task 4: Final Verification

**Files:**
- Verify: `crates/voxui-desktop/src/components/live_monitor.rs`
- Verify: `crates/voxui-desktop/src/styles.css`

- [ ] **Step 1: Run the mapped username tests**

Run:

```bash
cargo test -p voxui-desktop-ui mapped_uname
```

Expected: PASS for all mapped username helper, modal, and style tests.

- [ ] **Step 2: Run the existing live monitor tests**

Run:

```bash
cargo test -p voxui-desktop-ui live_monitor
```

Expected: PASS for the live monitor test module.

- [ ] **Step 3: Compile the frontend for the Tauri webview target**

Run:

```bash
cargo check -p voxui-desktop-ui --target wasm32-unknown-unknown
```

Expected: PASS with no Rust errors.

- [ ] **Step 4: Inspect the final diff**

Run:

```bash
git diff --stat
git diff -- crates/voxui-desktop/src/components/live_monitor.rs crates/voxui-desktop/src/styles.css
```

Expected: Diff is limited to the live monitor component and CSS, with no backend or settings changes.

- [ ] **Step 5: Commit final verification fixes if any were needed**

If verification required a fix, commit only those final fixes:

```bash
git add crates/voxui-desktop/src/components/live_monitor.rs crates/voxui-desktop/src/styles.css
git commit -m "fix: finalize live monitor mapped username editor"
```

If no verification fixes were needed, do not create an empty commit.
