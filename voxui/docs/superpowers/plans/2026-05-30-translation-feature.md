# Translation Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DeepL-powered text translation to the desktop app in both main window (outbound: input → TTS) and live monitor (inbound: received messages → read), with persisted language settings synced across all UIs.

**Architecture:** A `TranslationSettings` struct (two `TranslationPair`s: outbound + inbound, plus `translate_enqueue` bool) is added to `AppConfig` and persisted through the existing ConfigPatch pipeline. A new Tauri command `translate_text` makes direct HTTP calls to DeepL's oneshot endpoint from Rust. A new `TranslationBar` component sits left of the main textarea. The live monitor gains per-item inline translation panels with shared config-driven defaults.

**Tech Stack:** Rust (Tauri backend, reqwest), Leptos (frontend), existing CustomSelect/ConfigPatch patterns

---

## File Responsibilities

| File | Role |
|------|------|
| `src-tauri/src/types.rs` | `TranslationPair`, `TranslationSettings` structs; add to `AppConfig`, `ConfigPatch`; add `raw_message` to `LiveMonitorItemDto` |
| `src-tauri/src/app_core.rs` | Apply translation config patch; override defaults on global language change |
| `src-tauri/src/commands.rs` | `translate_text` Tauri command |
| `src-tauri/src/live.rs` | Populate `raw_message` in `dto_for_item` |
| `src-tauri/Cargo.toml` | Add `reqwest` dependency |
| `src/tauri_api.rs` | Frontend types for translation; `translate_text()` async function |
| `src/i18n.rs` | Add translation-related label fields to `Labels` struct and `labels()` constructor |
| `src/components/controls.rs` | Add `translation_lang_options()` helper (shared language code list) |
| `src/components/translation_bar.rs` | **New** — TranslationBar component |
| `src/components/input_box.rs` | Accept left-column slot; adjust layout |
| `src/components/live_monitor.rs` | Translate button + inline translation panel for danmu/superchat items |
| `src/components/settings_modal.rs` | `SettingsPage::Translation` variant; Translation tab content |
| `src/components/mod.rs` | Register `translation_bar` module |
| `src/app.rs` | Wire TranslationBar; handle `translation` in `apply_optimistic_patch`; translate_text proxying |
| `src/styles.css` | Styles for translation bar, inline panel, settings |

---

### Task 1: Backend Types — Add TranslationPair and TranslationSettings

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/types.rs`

- [ ] **Step 1: Add `TranslationPair` and `TranslationSettings` structs**

Add to `crates/voxui-desktop/src-tauri/src/types.rs`, after the existing `GenerationSettings` Default impl (~line 72):

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationPair {
    #[serde(default = "default_translation_source")]
    pub source_lang: String,
    #[serde(default = "default_translation_target")]
    pub target_lang: String,
}

fn default_translation_source() -> String {
    "auto".to_string()
}

fn default_translation_target() -> String {
    "EN".to_string()
}

impl Default for TranslationPair {
    fn default() -> Self {
        Self {
            source_lang: default_translation_source(),
            target_lang: default_translation_target(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TranslationSettings {
    pub outbound: TranslationPair,
    pub inbound: TranslationPair,
    pub translate_enqueue: bool,
}

impl Default for TranslationSettings {
    fn default() -> Self {
        Self {
            outbound: TranslationPair::default(),
            inbound: TranslationPair::default(),
            translate_enqueue: false,
        }
    }
}
```

- [ ] **Step 2: Add `translation` field to `AppConfig`**

In `AppConfig` (~line 327), add after the `generation` field:

```rust
pub translation: TranslationSettings,
```

And in `AppConfig`'s `Default` impl (~line 342), add after the `generation` default:

```rust
translation: TranslationSettings::default(),
```

- [ ] **Step 3: Add `translation` field to `ConfigPatch`**

In `ConfigPatch` (~line 400), add after `generation`:

```rust
#[serde(default, skip_serializing_if = "Option::is_none")]
pub translation: Option<TranslationSettings>,
```

- [ ] **Step 4: Add `raw_message` to `LiveMonitorItemDto`**

In `LiveMonitorItemDto` (~line 257), add after `suggestion`:

```rust
pub raw_message: Option<String>,
```

- [ ] **Step 5: Build to verify types compile**

```bash
cargo build -p voxui-desktop 2>&1 | Select-Object -Last 20
```

Expected: compiles (may warn about unused fields).

---

### Task 2: Backend — Apply Translation Config Patch + Default Override

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/app_core.rs`

- [ ] **Step 1: Apply translation in the config patch handler**

In `apply_patch` method (~line 154), add after the generation patch block:

```rust
if let Some(translation) = patch.translation {
    self.config.translation = translation;
}
```

- [ ] **Step 2: Override translation defaults when global language changes**

In `apply_patch`, find the existing `if let Some(language) = patch.language` block (~line 167). Modify it to also update translation defaults:

```rust
if let Some(language) = patch.language {
    self.config.language = language;
    self.config.translation.outbound.source_lang = lang_to_code(language);
    self.config.translation.inbound.target_lang = lang_to_code(language);
    self.config.translation.outbound.target_lang = opposite_lang_code(language);
}
```

Add these helper functions in `app_core.rs`:

```rust
fn lang_to_code(language: LanguageMode) -> String {
    match language {
        LanguageMode::Chinese => "ZH".to_string(),
        LanguageMode::English => "EN".to_string(),
        LanguageMode::System => "ZH".to_string(), // default for system
    }
}

fn opposite_lang_code(language: LanguageMode) -> String {
    match language {
        LanguageMode::Chinese => "EN".to_string(),
        LanguageMode::English => "ZH".to_string(),
        LanguageMode::System => "EN".to_string(), // default opposite for system
    }
}
```

- [ ] **Step 3: Build to verify**

```bash
cargo build -p voxui-desktop 2>&1 | Select-Object -Last 20
```

---

### Task 3: Backend — translate_text Tauri Command

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/Cargo.toml`
- Modify: `crates/voxui-desktop/src-tauri/src/commands.rs`

- [ ] **Step 1: Add reqwest dependency**

In `crates/voxui-desktop/src-tauri/Cargo.toml`, add under `[dependencies]`:

```toml
reqwest = { version = "0.12", default-features = false, features = ["json", "rustls-tls"] }
```

- [ ] **Step 2: Add `translate_text` command**

In `crates/voxui-desktop/src-tauri/src/commands.rs`, add after the last existing command:

```rust
#[tauri::command]
pub async fn translate_text(
    text: String,
    source_lang: String,
    target_lang: String,
) -> Result<String, String> {
    if text.trim().is_empty() {
        return Err("No text to translate".to_string());
    }

    let client = reqwest::Client::new();
    let mut body = serde_json::json!({
        "text": text,
        "target_lang": target_lang,
    });
    if source_lang != "auto" && !source_lang.is_empty() {
        body["source_lang"] = serde_json::Value::String(source_lang);
    }

    let response = client
        .post("https://www2.deepl.com/jsonrpc")
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "method": "LMT_handle_jobs",
            "params": {
                "jobs": [{
                    "kind": "default",
                    "raw_en_sentence": text,
                    "raw_en_context_before": [],
                    "raw_en_context_after": [],
                    "preferred_num_beams": 4,
                }],
                "lang": {
                    "user_preferred_langs": [target_lang],
                    "source_lang_user_selected": source_lang,
                },
                "priority": -1,
                "commonJobParams": {},
            },
            "id": 1,
        }))
        .send()
        .await
        .map_err(|e| format!("Translation request failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Translation service returned status {}", response.status()));
    }

    let json: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse translation response: {}", e))?;

    let translated = json["result"]["translations"][0]["beams"][0]["postprocessed_sentence"]
        .as_str()
        .ok_or_else(|| "Unexpected translation response format".to_string())?
        .to_string();

    Ok(translated)
}
```

**Note:** The exact DeepL JSON-RPC request format mirrors the one DeepLX reverse-engineers. If DeepL's Chrome extension changes its protocol, this request body may need updating. The JSON-RPC endpoint is `https://www2.deepl.com/jsonrpc` as used by the extension.

- [ ] **Step 3: Register the command in lib.rs**

In `crates/voxui-desktop/src-tauri/src/lib.rs`, add `translate_text` to the `invoke_handler`:

```rust
.invoke_handler(tauri::generate_handler![
    // ... existing commands ...
    crate::commands::translate_text,
])
```

- [ ] **Step 4: Build to verify**

```bash
cargo build -p voxui-desktop 2>&1 | Select-Object -Last 20
```

Expected: compiles cleanly.

---

### Task 4: Frontend API Types + translate_text

**Files:**
- Modify: `crates/voxui-desktop/src/tauri_api.rs`

- [ ] **Step 1: Add frontend TranslationPair and TranslationSettings**

Add after `AppConfig` (~line 67):

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationPair {
    pub source_lang: String,
    pub target_lang: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationSettings {
    pub outbound: TranslationPair,
    pub inbound: TranslationPair,
    pub translate_enqueue: bool,
}
```

- [ ] **Step 2: Add `translation` to frontend AppConfig**

In the frontend `AppConfig` struct (~line 54), add after `generation`:

```rust
pub translation: TranslationSettings,
```

- [ ] **Step 3: Add `translation` to frontend ConfigPatch**

In the frontend `ConfigPatch` struct (~line 388), add after `generation`:

```rust
#[serde(skip_serializing_if = "Option::is_none")]
pub translation: Option<TranslationSettings>,
```

- [ ] **Step 4: Add `raw_message` to frontend LiveMonitorItem**

In `LiveMonitorItem` (~line 364), add after `suggestion`:

```rust
pub raw_message: Option<String>,
```

- [ ] **Step 5: Add `translate_text` async function**

After the last existing API function (~line 640), add:

```rust
pub async fn translate_text(
    text: String,
    source_lang: String,
    target_lang: String,
) -> Result<String, String> {
    let args = serde_wasm_bindgen::to_value(&serde_json::json!({
        "text": text,
        "sourceLang": source_lang,
        "targetLang": target_lang,
    }))
    .map_err(|err| err.to_string())?;
    let value = invoke("translate_text", args)
        .await
        .map_err(stringify_js_error)?;
    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}
```

Tauri converts camelCase arg names to snake_case on the Rust side, so `sourceLang` → `source_lang`.

- [ ] **Step 6: Build frontend to verify**

```bash
cargo build -p voxui-desktop 2>&1 | Select-Object -Last 20
```

---

### Task 5: i18n Labels

**Files:**
- Modify: `crates/voxui-desktop/src/i18n.rs`

- [ ] **Step 1: Add new fields to `Labels` struct**

Add after `about_text` (~line 99):

```rust
pub translate: &'static str,
pub enqueue_translation: &'static str,
pub source_language: &'static str,
pub target_language: &'static str,
pub auto_detect: &'static str,
pub translation_tab: &'static str,
pub outbound_translation: &'static str,
pub inbound_translation: &'static str,
pub outbound_description: &'static str,
pub inbound_description: &'static str,
pub translating: &'static str,
pub translation_failed: &'static str,
pub no_text_to_translate: &'static str,
```

- [ ] **Step 2: Add Chinese labels in `labels()` function**

In the `Chinese` match arm, add before the closing `}` of the struct:

```rust
translate: "翻译",
enqueue_translation: "自动入队列",
source_language: "源语言",
target_language: "目标语言",
auto_detect: "自动检测",
translation_tab: "翻译",
outbound_translation: "输出翻译",
inbound_translation: "输入翻译",
outbound_description: "用于在语音合成前翻译输入框中输入的文字",
inbound_description: "用于翻译收到的直播消息",
translating: "翻译中...",
translation_failed: "翻译失败",
no_text_to_translate: "没有可翻译的文字",
```

- [ ] **Step 3: Add English labels in `labels()` function**

In the `English` match arm, add before the closing `}` of the struct:

```rust
translate: "Translate",
enqueue_translation: "Auto-enqueue",
source_language: "Source Language",
target_language: "Target Language",
auto_detect: "Auto Detect",
translation_tab: "Translation",
outbound_translation: "Outbound Translation",
inbound_translation: "Inbound Translation",
outbound_description: "Used to translate text typed in the main input box before TTS",
inbound_description: "Used to translate received livestream messages",
translating: "Translating...",
translation_failed: "Translation failed",
no_text_to_translate: "No text to translate",
```

---

### Task 6: Shared Language Options Helper

**Files:**
- Modify: `crates/voxui-desktop/src/components/controls.rs`

- [ ] **Step 1: Add `translation_lang_options` function**

Add before the closing `}` of the file (~line 237):

```rust
pub fn translation_lang_options(include_auto: bool, labels: &crate::i18n::Labels) -> Vec<SelectOption> {
    let mut options = Vec::new();

    if include_auto {
        options.push(SelectOption::new("auto", labels.auto_detect.to_string()));
    }

    let top: &[(&str, &str)] = &[
        ("ZH", "中文"),
        ("EN", "English"),
        ("JA", "日本語"),
    ];

    for (code, label) in top {
        options.push(SelectOption::new(*code, label.to_string()));
    }

    let rest: &[(&str, &str)] = &[
        ("AR", "العربية"), ("BG", "Български"), ("CS", "Čeština"), ("DA", "Dansk"),
        ("DE", "Deutsch"), ("EL", "Ελληνικά"), ("EN-GB", "English (UK)"), ("EN-US", "English (US)"),
        ("ES", "Español"), ("ES-419", "Español (Latinoamérica)"), ("ET", "Eesti"),
        ("FI", "Suomi"), ("FR", "Français"), ("HE", "עברית"), ("HU", "Magyar"),
        ("ID", "Bahasa Indonesia"), ("IT", "Italiano"), ("KO", "한국어"), ("LT", "Lietuvių"),
        ("LV", "Latviešu"), ("NB", "Norsk"), ("NL", "Nederlands"), ("PL", "Polski"),
        ("PT-BR", "Português (Brasil)"), ("PT-PT", "Português (Portugal)"),
        ("RO", "Română"), ("RU", "Русский"), ("SK", "Slovenčina"), ("SL", "Slovenščina"),
        ("SV", "Svenska"), ("TR", "Türkçe"), ("UK", "Українська"), ("VI", "Tiếng Việt"),
        ("ZH-HANT", "繁體中文"),
    ];

    for (code, label) in rest {
        options.push(SelectOption::new(*code, label.to_string()));
    }

    options
}
```

---

### Task 7: TranslationBar Component

**Files:**
- Create: `crates/voxui-desktop/src/components/translation_bar.rs`
- Modify: `crates/voxui-desktop/src/components/mod.rs`

- [ ] **Step 1: Create `translation_bar.rs`**

```rust
use leptos::prelude::*;

use crate::components::controls::{translation_lang_options, CustomSelect};
use crate::i18n::Labels;
use crate::tauri_api::{AppConfig, ConfigPatch, TranslationSettings};

#[component]
pub fn TranslationBar(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    config: impl Fn() -> AppConfig + Send + Sync + 'static + Copy,
    input_text: impl Fn() -> String + Send + Sync + 'static + Copy,
    disabled: impl Fn() -> bool + Send + Sync + 'static + Copy,
    on_replace_text: impl Fn(String) + 'static + Copy,
    on_enqueue: impl Fn(String) + 'static + Copy,
    on_config_patch: impl Fn(ConfigPatch) + Send + Sync + 'static + Copy,
) -> impl IntoView {
    let (translating, set_translating) = signal(false);

    let target_value = move || config().translation.outbound.target_lang.clone();
    let target_disabled = move || disabled() || translating.get();

    let translate_action = move || {
        let text = input_text();
        if text.trim().is_empty() || translating.get() {
            return;
        }
        set_translating.set(true);
        let source_lang = config().translation.outbound.source_lang.clone();
        let target_lang = config().translation.outbound.target_lang.clone();
        let enqueue = config().translation.translate_enqueue;

        spawn_local(async move {
            match crate::tauri_api::translate_text(text, source_lang, target_lang).await {
                Ok(translated) => {
                    if enqueue {
                        on_enqueue(translated);
                    } else {
                        on_replace_text(translated);
                    }
                }
                Err(_) => {
                    // Translation failed silently — UI state resets below
                }
            }
            set_translating.set(false);
        });
    };

    view! {
        <div class="translation-bar">
            <label class="translation-bar-select" for="translation-target-select">
                <CustomSelect
                    class="translation-target-select"
                    aria_label=move || labels().target_language
                    value=target_value
                    options=move || translation_lang_options(false, &labels())
                    disabled=target_disabled
                    on_change=move |value| {
                        let mut translation = config().translation.clone();
                        translation.outbound.target_lang = value;
                        on_config_patch(ConfigPatch {
                            translation: Some(translation),
                            ..ConfigPatch::default()
                        });
                    }
                />
            </label>
            <button
                class="primary-button translation-button"
                type="button"
                disabled=move || disabled() || input_text().trim().is_empty() || translating.get()
                on:click=move |_| translate_action()
            >
                {move || if translating.get() { labels().translating } else { labels().translate }}
            </button>
            <label class="translation-checkbox" for="translation-enqueue">
                <input
                    id="translation-enqueue"
                    type="checkbox"
                    prop:checked=move || config().translation.translate_enqueue
                    disabled=target_disabled
                    on:change=move |event| {
                        let mut translation = config().translation.clone();
                        translation.translate_enqueue = event_target_checked(&event);
                        on_config_patch(ConfigPatch {
                            translation: Some(translation),
                            ..ConfigPatch::default()
                        });
                    }
                />
                <span>{move || labels().enqueue_translation}</span>
            </label>
        </div>
    }
}
```

- [ ] **Step 2: Register the module**

In `crates/voxui-desktop/src/components/mod.rs`, add:

```rust
pub mod translation_bar;
```

- [ ] **Step 3: Build to verify**

```bash
cargo build -p voxui-desktop 2>&1 | Select-Object -Last 20
```

---

### Task 8: Modify InputBox for Left Column

**Files:**
- Modify: `crates/voxui-desktop/src/components/input_box.rs`

- [ ] **Step 1: Restructure InputBox layout**

Replace the current `InputBox` view (starting at the `form` element, ~line 57) to wrap textarea in a horizontal flex container that accepts a left slot:

```rust
#[component]
pub fn InputBox(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    language: impl Fn() -> UiLanguage + Send + Sync + 'static + Copy,
    max_chars: impl Fn() -> usize + Send + Sync + 'static + Copy,
    auto_period: impl Fn() -> bool + Send + Sync + 'static + Copy,
    disabled: impl Fn() -> bool + Send + Sync + 'static + Copy,
    replacement_text: impl Fn() -> Option<String> + Send + Sync + 'static + Copy,
    on_replacement_consumed: impl Fn() + Send + Sync + 'static + Copy,
    on_generate: impl Fn(String) + 'static + Copy,
    translation_bar: Option<AnyView>,
) -> impl IntoView {
    // ... keep existing signal/Effect/submit logic exactly as-is ...

    view! {
        <form class="composer-panel" on:submit=submit>
            <div class="composer-row">
                {if let Some(bar) = translation_bar {
                    view! { <div class="composer-translation-column">{bar}</div> }.into_any()
                } else {
                    ().into_any()
                }}
                <div class="composer-field">
                    <textarea
                        class="composer-input"
                        prop:value=move || text.get()
                        placeholder=move || labels().input_placeholder
                        disabled=move || disabled()
                        on:input=move |event| set_text.set(event_target_value(&event))
                    ></textarea>
                    <span class:over-limit=is_over_limit class="char-counter">
                        {move || format!("{}/{}", char_count(), max_chars())}
                    </span>
                </div>
            </div>
            <div class="composer-actions">
                <button class="generate-button" type="submit" disabled=generate_disabled>
                    {move || labels().generate}
                </button>
                <button
                    class="secondary-button composer-clear-button"
                    type="button"
                    disabled=move || text.get().is_empty()
                    on:click=move |_| set_text.set(String::new())
                >
                    {move || labels().clear}
                </button>
            </div>
        </form>
    }
}
```

Key changes:
- Add `use leptos::prelude::AnyView;` at top
- Add `translation_bar: Option<AnyView>` parameter
- Wrap textarea in `div.composer-row` with the optional `div.composer-translation-column` before `composer-field`
- Char counter stays inside `composer-field`
- Generate + Clear stay in `composer-actions`, unchanged

---

### Task 9: Wire TranslationBar in App

**Files:**
- Modify: `crates/voxui-desktop/src/app.rs`

- [ ] **Step 1: Add translation to `apply_optimistic_patch`**

In `apply_optimistic_patch` (~line 713), add after the generation patch block:

```rust
if let Some(translation) = patch.translation.as_ref() {
    snapshot.config.translation = translation.clone();
}
```

- [ ] **Step 2: Add a shared text signal**

After the `input_replacement` signal (~line 33), add:

```rust
let (input_text, set_input_text) = signal(String::new());
```

- [ ] **Step 3: Modify the InputBox invocation**

Replace the existing `<InputBox>` invocation (~line 377) with:

```rust
<InputBox
    labels=current_labels
    language=current_ui_language
    max_chars=move || current_snapshot().config.max_input_chars
    auto_period=move || current_snapshot().config.auto_period
    disabled=move || {
        let snapshot = current_snapshot();
        snapshot.loaded_model_id.is_none() || matches!(snapshot.load_state, LoadUiState::Loading)
    }
    replacement_text=move || input_replacement.get()
    on_replacement_consumed=move || set_input_replacement.set(None)
    on_generate=move |text| {
        spawn_local(async move {
            if crate::tauri_api::enqueue_generation(text).await.is_ok() {
                refresh_snapshot();
            }
        });
    }
    translation_bar=Some(
        view! {
            <TranslationBar
                labels=current_labels
                config=move || current_snapshot().config.clone()
                input_text=move || input_text.get()
                disabled=move || {
                    let snapshot = current_snapshot();
                    snapshot.loaded_model_id.is_none() || matches!(snapshot.load_state, LoadUiState::Loading)
                }
                on_replace_text=move |text| set_input_replacement.set(Some(text))
                on_enqueue=move |text| {
                    spawn_local(async move {
                        if crate::tauri_api::enqueue_generation(text).await.is_ok() {
                            refresh_snapshot();
                        }
                    });
                }
                on_config_patch=commit_config_patch
            />
        }.into_any()
    )
/>
```

The `input_text` signal needs to be synced with the InputBox's internal text. This requires modifying `InputBox` to expose its text as a signal. See Task 8's step — add an `on_text_change` callback or accept an `input_text` write signal.

- [ ] **Step 4: Add a text change callback to InputBox**

In InputBox, add parameter:

```rust
on_text_change: Option<impl Fn(String) + 'static + Copy>,
```

And in the `on:input` handler, after `set_text.set(...)`, add:

```rust
if let Some(ref callback) = on_text_change {
    callback(event_target_value(&event));
}
```

In `app.rs`, add:

```rust
on_text_change=Some(move |text| set_input_text.set(text))
```

to the `<InputBox>` invocation.

- [ ] **Step 5: Import TranslationBar**

Add at top of `app.rs`:

```rust
use crate::components::translation_bar::TranslationBar;
```

- [ ] **Step 6: Build to verify**

```bash
cargo build -p voxui-desktop 2>&1 | Select-Object -Last 30
```

Expected: compiles. Fix any type mismatches (particularly around `ConfigPatch::default()` needing all optional fields as `None`).

---

### Task 10: Live Monitor — Translate Button + Inline Panel

**Files:**
- Modify: `crates/voxui-desktop/src/components/live_monitor.rs`

- [ ] **Step 1: Add translation state signals**

In the `LiveMonitor` component signals section (~line 37), add:

```rust
let (expanded_translations, set_expanded_translations) = signal(HashSet::<String>::new());
let (translation_results, set_translation_results) = signal(HashMap::<String, (String, bool)>::new());
```

- [ ] **Step 2: Add translate button to danmu/superchat items**

In the `live-item-actions` div (~line 271), add a translate button **left** of the `mapped_uname_button`:

```rust
{
    let item_for_translate = item.clone();
    let supports_translation = matches!(kind, LiveMessageKind::Danmu | LiveMessageKind::Superchat);
    let has_raw_message = item.raw_message.is_some();
    let show_translate = supports_translation && has_raw_message;

    if show_translate {
        let translate_item_id = item.id.clone();
        view! {
            <button
                class="live-monitor-button"
                type="button"
                title=labels.translate
                aria-label=labels.translate
                on:click=move |_| {
                    let id = translate_item_id.clone();
                    set_expanded_translations.update(|set| {
                        if set.contains(&id) {
                            set.remove(&id);
                        } else {
                            set.insert(id);
                        }
                    });
                }
            >
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M5 8l6-6 6 6"></path>
                    <path d="M5 16l6 6 6-6"></path>
                    <line x1="12" y1="2" x2="12" y2="22"></line>
                </svg>
            </button>
        }.into_any()
    } else {
        ().into_any()
    }
}
```

- [ ] **Step 3: Add inline translation panel below each item**

After the `live-item-actions` closing div but before the closing `</article>` tag, add:

```rust
{
    let item_id_for_panel = item.id.clone();
    let raw_msg = item.raw_message.clone().unwrap_or_default();
    let trans_config = translation_config.clone();
    let trans_patch = on_translation_patch.clone();

    view! {
        <Show when=move || expanded_translations.get().contains(&item_id_for_panel)>
            <div class="live-translation-panel">
                <div class="live-translation-controls">
                    <CustomSelect
                        class="live-translation-source-select"
                        aria_label=move || labels().source_language
                        value=move || trans_config().inbound.source_lang.clone()
                        options=move || translation_lang_options(true, &labels())
                        disabled=move || false
                        on_change=move |value| {
                            let mut cfg = trans_config();
                            cfg.inbound.source_lang = value;
                            trans_patch(cfg);
                        }
                    />
                    <CustomSelect
                        class="live-translation-target-select"
                        aria_label=move || labels().target_language
                        value=move || trans_config().inbound.target_lang.clone()
                        options=move || translation_lang_options(false, &labels())
                        disabled=move || false
                        on_change=move |value| {
                            let mut cfg = trans_config();
                            cfg.inbound.target_lang = value;
                            trans_patch(cfg);
                        }
                    />
                    <button
                        class="primary-button live-translation-do-button"
                        type="button"
                        disabled=move || {
                            translation_results.get()
                                .get(&item_id_for_panel)
                                .map(|(_, loading)| *loading)
                                .unwrap_or(false)
                        }
                        on:click=move |_| {
                            let item_id = item_id_for_panel.clone();
                            let msg = raw_msg.clone();
                            let source = trans_config().inbound.source_lang.clone();
                            let target = trans_config().inbound.target_lang.clone();
                            set_translation_results.update(|results| {
                                results.insert(item_id.clone(), (String::new(), true));
                            });
                            spawn_local(async move {
                                match crate::tauri_api::translate_text(msg, source, target).await {
                                    Ok(translated) => {
                                        set_translation_results.update(|results| {
                                            results.insert(item_id, (translated, false));
                                        });
                                    }
                                    Err(_) => {
                                        set_translation_results.update(|results| {
                                            results.remove(&item_id);
                                        });
                                    }
                                }
                            });
                        }
                    >
                        {move || labels().translate}
                    </button>
                </div>
                {move || {
                    translation_results.get()
                        .get(&item_id_for_panel)
                        .map(|(result, loading)| {
                            if *loading {
                                view! { <p class="live-translation-result live-translation-loading">{labels().translating}</p> }.into_any()
                            } else {
                                view! { <p class="live-translation-result">{result.clone()}</p> }.into_any()
                            }
                        })
                }}
            </div>
        </Show>
    }.into_any()
}
```

The `translation_config` and `on_translation_patch` are passed as component props to `LiveMonitor` (see Step 4). They provide the translation settings and a callback to persist changes, respectively.

- [ ] **Step 4: Update LiveMonitor signature to accept translation props**

```rust
#[component]
pub fn LiveMonitor(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    snapshot: impl Fn() -> LiveSnapshot + Send + Sync + 'static + Copy,
    on_live_patch: impl Fn(LiveConfigPatch) + Send + Sync + 'static + Copy,
    on_send: impl Fn(String, bool, bool) + Send + Sync + 'static + Copy,
    on_clear: impl Fn() + Send + Sync + 'static + Copy,
    translation_config: impl Fn() -> TranslationSettings + Send + Sync + 'static + Copy,
    on_translation_patch: impl Fn(TranslationSettings) + Send + Sync + 'static + Copy,
) -> impl IntoView {
```

---

### Task 11: Settings Modal — Translation Tab

**Files:**
- Modify: `crates/voxui-desktop/src/components/settings_modal.rs`

- [ ] **Step 1: Add `SettingsPage::Translation` variant**

In the `SettingsPage` enum (~line 12), add `Translation`:

```rust
pub enum SettingsPage {
    General,
    Inference,
    Audio,
    Live,
    Translation,
    About,
}
```

- [ ] **Step 2: Add Translation tab button**

In the settings tabs nav (~line 98), add before the About button:

```rust
<button type="button" class:active=move || active_page() == SettingsPage::Translation on:click=move |_| on_page_select(SettingsPage::Translation)>{move || labels().translation_tab}</button>
```

- [ ] **Step 3: Add Translation tab content**

After the Live section `Show` block but before the closing of `settings-content`, add:

```rust
<Show when=move || active_page() == SettingsPage::Translation>
    <section class="settings-section">
        <h3>{move || labels().outbound_translation}</h3>
        <p class="settings-section-desc">{move || labels().outbound_description}</p>
        <div class="settings-grid">
            <label class="settings-field" for="settings-outbound-source">
                <span>{move || labels().source_language}</span>
                <CustomSelect
                    class="settings-select-control"
                    aria_label=move || labels().source_language
                    value=move || config().translation.outbound.source_lang.clone()
                    options=move || translation_lang_options(true, &labels())
                    disabled=move || false
                    on_change=move |value| {
                        let mut translation = config().translation.clone();
                        translation.outbound.source_lang = value;
                        on_config_patch(ConfigPatch {
                            translation: Some(translation),
                            ..ConfigPatch::default()
                        });
                    }
                />
            </label>
            <label class="settings-field" for="settings-outbound-target">
                <span>{move || labels().target_language}</span>
                <CustomSelect
                    class="settings-select-control"
                    aria_label=move || labels().target_language
                    value=move || config().translation.outbound.target_lang.clone()
                    options=move || translation_lang_options(false, &labels())
                    disabled=move || false
                    on_change=move |value| {
                        let mut translation = config().translation.clone();
                        translation.outbound.target_lang = value;
                        on_config_patch(ConfigPatch {
                            translation: Some(translation),
                            ..ConfigPatch::default()
                        });
                    }
                />
            </label>
            <label class="settings-checkbox settings-switch" for="settings-translate-enqueue">
                <input
                    id="settings-translate-enqueue"
                    type="checkbox"
                    prop:checked=move || config().translation.translate_enqueue
                    on:change=move |event| {
                        let mut translation = config().translation.clone();
                        translation.translate_enqueue = event_target_checked(&event);
                        on_config_patch(ConfigPatch {
                            translation: Some(translation),
                            ..ConfigPatch::default()
                        });
                    }
                />
                <span>{move || labels().enqueue_translation}</span>
            </label>
        </div>

        <h3>{move || labels().inbound_translation}</h3>
        <p class="settings-section-desc">{move || labels().inbound_description}</p>
        <div class="settings-grid">
            <label class="settings-field" for="settings-inbound-source">
                <span>{move || labels().source_language}</span>
                <CustomSelect
                    class="settings-select-control"
                    aria_label=move || labels().source_language
                    value=move || config().translation.inbound.source_lang.clone()
                    options=move || translation_lang_options(true, &labels())
                    disabled=move || false
                    on_change=move |value| {
                        let mut translation = config().translation.clone();
                        translation.inbound.source_lang = value;
                        on_config_patch(ConfigPatch {
                            translation: Some(translation),
                            ..ConfigPatch::default()
                        });
                    }
                />
            </label>
            <label class="settings-field" for="settings-inbound-target">
                <span>{move || labels().target_language}</span>
                <CustomSelect
                    class="settings-select-control"
                    aria_label=move || labels().target_language
                    value=move || config().translation.inbound.target_lang.clone()
                    options=move || translation_lang_options(false, &labels())
                    disabled=move || false
                    on_change=move |value| {
                        let mut translation = config().translation.clone();
                        translation.inbound.target_lang = value;
                        on_config_patch(ConfigPatch {
                            translation: Some(translation),
                            ..ConfigPatch::default()
                        });
                    }
                />
            </label>
        </div>
    </section>
</Show>
```

- [ ] **Step 4: Import `translation_lang_options`**

Add at top of `settings_modal.rs`:

```rust
use crate::components::controls::translation_lang_options;
```

- [ ] **Step 5: Build to verify**

```bash
cargo build -p voxui-desktop 2>&1 | Select-Object -Last 30
```

---

### Task 12: Populate `raw_message` in Backend Live DTO

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/live.rs`

- [ ] **Step 1: Add `raw_message` to `dto_for_item` output**

In the `dto_for_item` method (~line 124), find where `LiveMonitorItemDto` is constructed and add `raw_message`. The exact field depends on the message kind:

```rust
let raw_message = match item.event.kind {
    LiveMessageKind::Danmu => item.event.msg.clone(),
    LiveMessageKind::Superchat => item.event.superchat_message.clone(),
    _ => None,
};
```

Then include it in the `LiveMonitorItemDto` constructor:

```rust
LiveMonitorItemDto {
    // ... existing fields ...
    raw_message,
}
```

- [ ] **Step 2: Build to verify**

```bash
cargo build -p voxui-desktop 2>&1 | Select-Object -Last 20
```

---

### Task 13: CSS Styles

**Files:**
- Modify: `crates/voxui-desktop/src/styles.css`

- [ ] **Step 1: Add styles for translation bar, inline panel, settings tab**

Append to `styles.css`:

```css
/* Translation Bar (main window, left of input) */
.composer-row {
    display: flex;
    gap: 10px;
    align-items: stretch;
}

.composer-translation-column {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 120px;
    flex-shrink: 0;
}

.composer-translation-column .translation-bar {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.translation-bar .custom-select {
    width: 100%;
}

.translation-button {
    width: 100%;
}

.translation-checkbox {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    cursor: pointer;
}

.translation-checkbox input[type="checkbox"] {
    margin: 0;
}

/* Live Monitor Translation Panel */
.live-translation-panel {
    margin-top: 8px;
    padding: 8px;
    border-top: 1px solid var(--border-color);
    background: var(--bg-hover);
    border-radius: 4px;
}

.live-translation-controls {
    display: flex;
    gap: 6px;
    align-items: center;
    flex-wrap: wrap;
}

.live-translation-source-select,
.live-translation-target-select {
    min-width: 100px;
}

.live-translation-do-button {
    flex-shrink: 0;
}

.live-translation-result {
    margin-top: 8px;
    font-size: 13px;
    color: var(--text-secondary);
    word-break: break-word;
}

.live-translation-loading {
    font-style: italic;
    opacity: 0.7;
}

/* Settings Translation section */
.settings-section-desc {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 0 0 10px 0;
}
```

- [ ] **Step 2: Build CSS (trunk build)**

```bash
cd crates/voxui-desktop; trunk build 2>&1 | Select-Object -Last 10
```

---

### Task 14: Live Monitor Window — Wire Translation Config

**Files:**
- Modify: `crates/voxui-desktop/src/app.rs`

- [ ] **Step 1: Update the live-monitor window branch**

In the `is_live_monitor_window` branch (~line 208), update the `<LiveMonitor>` call:

```rust
<LiveMonitor
    labels=current_labels
    snapshot=move || live_snapshot.get()
    on_live_patch=commit_live_patch
    on_send=move |item_id, switch, enqueue_direct| {
        // ... existing send logic unchanged ...
    }
    on_clear=move || {
        // ... existing clear logic unchanged ...
    }
    translation_config=move || current_snapshot().config.translation.clone()
    on_translation_patch=move |translation| {
        commit_config_patch(ConfigPatch {
            translation: Some(translation),
            ..ConfigPatch::default()
        });
    }
/>
```

- [ ] **Step 2: Also update main window LiveMonitor usage**

In the main window branch (if any LiveMonitor is rendered there), similarly pass the translation props.

---

### Task 15: Integration Test — translate_text Command

**Files:**
- Create: `crates/voxui-desktop/src-tauri/tests/translation_tests.rs`

- [ ] **Step 1: Write unit test for the translate command handler (offline)**

```rust
#[cfg(test)]
mod tests {
    use voxui_desktop::commands::translate_text;

    #[tokio::test]
    async fn translate_text_rejects_empty_input() {
        let result = translate_text("".to_string(), "auto".to_string(), "ZH".to_string()).await;
        assert!(result.is_err());
    }
}
```

Note: This test only validates the empty-text guard. Full integration testing requires a running DeepL endpoint or mock server.

- [ ] **Step 2: Write component tests for translation UI elements**

In `live_monitor.rs` tests, add:

```rust
#[test]
fn monitor_renders_translate_button_for_danmu_and_superchat() {
    let source = include_str!("live_monitor.rs");
    assert!(source.contains("LiveMessageKind::Danmu | LiveMessageKind::Superchat"),
        "Translate button should be shown for danmu and superchat items");
}
```

- [ ] **Step 3: Run existing tests**

```bash
cargo test -p voxui-desktop 2>&1 | Select-Object -Last 20
```

---

### Task 16: Full Build + Verify

- [ ] **Step 1: Build the full workspace**

```bash
cargo build -p voxui-desktop 2>&1 | Select-Object -Last 30
```

Expected: compiles with no errors.

- [ ] **Step 2: Run all tests**

```bash
cargo test -p voxui-desktop 2>&1 | Select-Object -Last 30
```

Expected: all tests pass.

- [ ] **Step 3: Verify frontend builds**

```bash
Set-Location -LiteralPath "crates/voxui-desktop"; if ($?) { trunk build } 2>&1 | Select-Object -Last 15
```

Expected: trunk bundles successfully.

- [ ] **Step 4: Lint**

```bash
cargo clippy -p voxui-desktop -- -D warnings 2>&1 | Select-Object -Last 20
```

Expected: no warnings.

---

### Task 17: Commit

- [ ] **Step 1: Stage and commit**

```bash
git add crates/voxui-desktop/
git commit -m "feat: add translation feature with DeepL integration"
```
