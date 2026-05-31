# Translation Feature Design

## Overview

Add translation capabilities to the voxui-desktop app using DeepL's free oneshot API, called directly from the Rust backend. Translation is available in two contexts: outbound (main window input box → TTS) and inbound (live monitor received messages → streamer reads). Language settings are persisted in `AppConfig`, synced across all translation UIs via the existing config patch pipeline.

## Data Model

### TranslationLang

A validated string type representing a DeepL language code. Supported values (from DeepL's 36 target-capable codes plus `auto`):

- `"auto"` — auto-detect (valid only as source_lang)
- `"ZH"`, `"EN"`, `"JA"` — top 3, placed first in select menus
- Remaining 33 codes: `AR`, `BG`, `CS`, `DA`, `DE`, `EL`, `EN-GB`, `EN-US`, `ES`, `ES-419`, `ET`, `FI`, `FR`, `HE`, `HU`, `ID`, `IT`, `KO`, `LT`, `LV`, `NB`, `NL`, `PL`, `PT-BR`, `PT-PT`, `RO`, `RU`, `SK`, `SL`, `SV`, `TR`, `UK`, `VI`, `ZH-HANT`

### TranslationPair

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TranslationPair {
    pub source_lang: String, // "auto", "ZH", "EN", etc.
    pub target_lang: String, // "ZH", "EN", "JA", etc. (never "auto")
}
```

### TranslationSettings

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TranslationSettings {
    pub outbound: TranslationPair,   // Main window input → TTS
    pub inbound: TranslationPair,    // Live monitor received → reader
    pub translate_enqueue: bool,     // true = enqueue to TTS, false = replace input text
}

impl Default for TranslationSettings {
    fn default() -> Self {
        Self {
            outbound: TranslationPair {
                source_lang: "ZH".to_string(),   // Overridden at first load
                target_lang: "EN".to_string(),   // Overridden at first load
            },
            inbound: TranslationPair {
                source_lang: "auto".to_string(), // Auto-detect
                target_lang: "ZH".to_string(),   // Overridden at first load
            },
            translate_enqueue: false,             // Default: replace input text
        }
    }
}
```

### Config Integration

`TranslationSettings` is added as a field on `AppConfig`:

```rust
pub struct AppConfig {
    // ... existing fields ...
    pub translation: TranslationSettings,
}
```

`ConfigPatch` adds:

```rust
pub struct ConfigPatch {
    // ... existing fields ...
    pub translation: Option<TranslationSettings>,
}
```

### Default Override on Global Language Change

When the global `language` setting changes, the following dependent translation fields are automatically updated (the user can always override them in Settings → Translation afterwards):

- `outbound.source_lang` ← mirrors `config.language` (the global UI language setting)
- `outbound.target_lang` ← opposite language: if global is Chinese → `"EN"`, if English → `"ZH"`, if Japanese → `"ZH"`
- `inbound.source_lang` ← `"auto"` (unchanged)
- `inbound.target_lang` ← mirrors `config.language`

This recomputation happens in `app_core.rs` when processing a `ConfigPatch` with a `language` change alongside a default `translation` value. On first startup (fresh config), the `TranslationSettings::default()` values are also overridden by this same logic.

### Optimistic Update

`apply_optimistic_patch` in `app.rs` handles the new `translation` field like all other config fields — applies locally before the server round-trip.

## Backend — DeepL API Integration

### Tauri Command: `translate_text`

**Signature:** `tauri::command` in `commands.rs`

```
translate_text(text: String, source_lang: String, target_lang: String) -> Result<String, String>
```

**Behavior:**
1. If `source_lang` is `"auto"`, omit it from the request (DeepL auto-detects)
2. Make HTTP POST to DeepL's oneshot endpoint using the same protocol that DeepLX reverse-engineers (calling DeepL's Chrome extension backend directly — no API key required)
3. Use `reqwest` for the HTTP call (already available in the Tauri backend)
4. Parse the JSON response, extract the translated `data` field
5. Return the translated string on success, or a user-facing error message on failure
6. Handle error cases: empty text (400), rate limiting (429), upstream failure (503)

**Dependencies:** Add `reqwest` with `json` feature to `src-tauri/Cargo.toml` if not already present.

**Request format (mirrors DeepLX's free endpoint):**
- POST to DeepL oneshot URL
- Content-Type: `application/json`
- Body: `{"text": "...", "target_lang": "ZH", "source_lang": "auto"}` (source_lang optional when "auto")

### Frontend API

In `tauri_api.rs`:

```rust
pub async fn translate_text(text: String, source_lang: String, target_lang: String) -> Result<String, String>
```

Calls `invoke("translate_text", args)`, deserializes the result string.

## Main Window — TranslationBar Component

### Layout

A new `TranslationBar` component placed in the main window's composer panel, to the **left** of the text input area. Layout (existing Generate/Clear on the right unchanged):

```
┌──────────────────────────────────────────────────────────────┐
│  ┌──────────────┐  ┌──────────────────────────────────┐  ┌──┐│
│  │   [ZH ▼]     │  │  Input text...                    │  │G ││
│  │  [Translate] │  │                                   │  │e ││
│  │  [☐ Enqueue] │  │                                   │  │n ││
│  └──────────────┘  └──────────────────────────────────┘  │  ││
│                   42/200                                  │C ││
│                                                          │l ││
│                                                          │r ││
│                                                          └──┘│
└──────────────────────────────────────────────────────────────┘
```

### Component Signature

```rust
#[component]
pub fn TranslationBar(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    config: impl Fn() -> AppConfig + Send + Sync + 'static + Copy,
    input_text: impl Fn() -> String + Send + Sync + 'static + Copy,
    disabled: impl Fn() -> bool + Send + Sync + 'static + Copy,
    on_replace_text: impl Fn(String) + 'static + Copy,  // Replace input content
    on_enqueue: impl Fn(String) + 'static + Copy,        // Enqueue to TTS
    on_config_patch: impl Fn(ConfigPatch) + Send + Sync + 'static + Copy,
) -> impl IntoView
```

### Behavior

- **Target select (top):** `CustomSelect` showing language codes, initialized from `config().translation.outbound.target_lang`. Changes fire `on_config_patch(ConfigPatch { translation: Some(updated), ..default() })`.
- **Translate button:** On click, calls `translate_text(input_text(), outbound.source_lang, selected_target)`. Disabled while translating (loading state). On success:
  - If checkbox is **unchecked** → calls `on_replace_text(translated)` (replaces textarea content)
  - If checkbox is **checked** → calls `on_enqueue(translated)` (TTS generation queue)
  - On error → show a brief status message (or use existing error modal pattern)
- **Checkbox:** Reads from `config().translation.translate_enqueue`, writes back via `on_config_patch`. When checked, translate enqueues the result for TTS; when unchecked, translate replaces the input text.
- **Source language** is NOT shown inline — only in Settings → Translate tab, where it defaults to the global language.

### Integration

The `TranslationBar` is rendered in `app.rs` as part of the main view, passing:
- `input_text` from the `InputBox`'s text signal (or a shared signal)
- `on_replace_text` that sets `input_replacement` via the existing mechanism
- `on_enqueue` that calls `crate::tauri_api::enqueue_generation(text)`
- `on_config_patch` that delegates to the existing `commit_config_patch`

The `InputBox` component needs minor changes:
- Accept an optional `left_column` slot or render the `TranslationBar` alongside the textarea
- The textarea width adjusts to accommodate the left column

## Live Monitor — Inline Translation

### Translate Button Placement

A translate button (🔄 icon) is added **to the left** of the existing mapped-username edit button, visible only for `Danmu` and `Superchat` message kinds.

### Inline Translation Panel

When the translate button is clicked, it **toggles** an inline panel appended below the message item:

```
┌────────────────────────────────────────────────────────────┐
│  [🎤 UserName]  Danmu: ようこそ                              │
│  [🔄] [✏️]                                                 │
│  ───────────────────────────────────────────────────────── │
│  From: [auto ▼]   To: [ZH ▼]   [Translate]                │
│  Translated: 欢迎                                            │
└────────────────────────────────────────────────────────────┘
```

### Behavior

- **Toggle:** Each item independently tracks whether its translation panel is open (local state, not persisted). Clicking the translate button toggles it.
- **Source select:** Initialized from `inbound.source_lang` (defaults to `"auto"`). Changes propagate to config.
- **Target select:** Initialized from `inbound.target_lang` (defaults to global language). Changes propagate to config.
- **Translate button (inline):** Calls `translate_text` with the raw message content (`LiveEvent.msg` for danmu, `LiveEvent.superchat_message` for superchat), NOT the template-rendered suggestion. Displays result inline.
- **Syncing:** Select changes go through `on_live_patch` or a separate config patch to update `TranslationSettings.inbound`.
- **Language applies per-item:** The source/target selectors in each item's panel read from the shared config, so changing them in one item's panel updates all open panels and the settings tab.

### Data Access

The raw message content is available in `LiveMonitorItemDto`. The `LiveMonitorItemDto` struct needs an additional field `raw_message: Option<String>` to carry the untemplated message text from the backend (`LiveEvent.msg` or `LiveEvent.superchat_message`). Backend `dto_for_item` is updated to populate this field.

## Settings Modal — Translation Tab

### New Tab

Add `SettingsPage::Translation` variant to the existing enum. The tab appears in the sidebar between Live and About.

### Layout

```
┌─ General ─── Inference ─── Audio ─── Live ─── [Translation] ─── About ─┐
│
│  Outbound Translation
│  ─────────────────────
│  Used to translate text typed in the main input box before TTS
│
│  Source Language:   [Chinese ▼]
│  Target Language:   [English ▼]
│  Enqueue to TTS:    [☐ Auto-enqueue]
│
│  Inbound Translation
│  ────────────────────
│  Used to translate received livestream messages
│
│  Source Language:  [Auto Detect ▼]
│  Target Language:  [Chinese ▼]
│
│                                               [Close]
└─────────────────────────────────────────────────────────────────────────┘
```

### Behavior

- All four selects are `CustomSelect` components; the checkbox controls the enqueue-vs-replace behavior
- Values read from `config().translation.{outbound,inbound}.{source_lang,target_lang}` and `config().translation.translate_enqueue`
- Changes fire `on_config_patch(ConfigPatch { translation: Some(updated), ..default() })` — same pipeline as all other settings
- Language options list: `"auto"` (source only) + `"ZH"`, `"EN"`, `"JA"` at top, then remaining codes alphabetically
- Disabled during active TTS generation (matches existing settings behavior)

## i18n / Labels

New label keys needed:

| Key | English | Chinese |
|-----|---------|---------|
| `translate` | Translate | 翻译 |
| `enqueue_translation` | Auto-enqueue | 自动入队列 |
| `source_language` | Source Language | 源语言 |
| `target_language` | Target Language | 目标语言 |
| `auto_detect` | Auto Detect | 自动检测 |
| `translation_tab` | Translation | 翻译 |
| `outbound_translation` | Outbound Translation | 输出翻译 |
| `inbound_translation` | Inbound Translation | 输入翻译 |
| `outbound_description` | Used to translate text typed in the main input box before TTS | 用于在语音合成前翻译输入框中输入的文字 |
| `inbound_description` | Used to translate received livestream messages | 用于翻译收到的直播消息 |
| `translating` | Translating... | 翻译中... |
| `translation_failed` | Translation failed | 翻译失败 |
| `no_text_to_translate` | No text to translate | 没有可翻译的文字 |

## File Changes Summary

| File | Change |
|------|--------|
| `src-tauri/Cargo.toml` | Add `reqwest` dependency |
| `src-tauri/src/types.rs` | Add `TranslationPair`, `TranslationSettings`, add `translation` to `AppConfig` and `ConfigPatch`. Add `raw_message` to `LiveMonitorItemDto` |
| `src-tauri/src/config.rs` | Default override logic for translation settings on first load |
| `src-tauri/src/app_core.rs` | Wire translation default override, apply translation patch |
| `src-tauri/src/commands.rs` | Add `translate_text` command |
| `src-tauri/src/live.rs` | Populate `raw_message` in `dto_for_item` |
| `src/tauri_api.rs` | Add `TranslationPair`, `TranslationSettings`, `translate_text()` function, add `raw_message` to `LiveMonitorItem` |
| `src/app.rs` | Wire `TranslationBar` into main view, handle translation callbacks, apply translation config patch optimistically |
| `src/components/input_box.rs` | Add slot for left column (translation bar), adjust layout |
| `src/components/translation_bar.rs` | **New** — TranslationBar component |
| `src/components/live_monitor.rs` | Add translate button to danmu/superchat items, inline translation panel |
| `src/components/settings_modal.rs` | Add `SettingsPage::Translation`, render Translation tab content |
| `src/i18n.rs` | Add translation-related label fields |
| `src/components/mod.rs` | Register new `translation_bar` module |
| `src/styles.css` | Add styles for translation bar, inline translation panel, settings tab |

## Error Handling

- **Empty input text:** Translate button disabled when text is empty
- **Network failure:** Catch reqwest errors, return user-facing message
- **DeepL rate limit (429):** Return "Translation service busy, please try again later"
- **DeepL error responses (400, 404, 503):** Map to user-facing messages
- **Loading state:** Translate buttons show spinner/disabled during in-flight requests
- **Frontend errors:** Use existing error modal pattern or transient status messages (matching existing `status_notice` pattern in live monitor)

## Testing

- **Unit tests:** Language code validation, default override logic, config serialization round-trip
- **Component tests:** Existing inline test patterns in `live_monitor.rs` — verify translate button presence, inline panel toggling
- **Integration tests:** Backend `translate_text` command with a mock DeepL endpoint or integration test fixture
