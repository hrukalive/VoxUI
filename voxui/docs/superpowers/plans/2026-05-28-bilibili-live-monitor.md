# Bilibili Live Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Bilibili OpenLive connectivity, live suggestion settings, and a separate native live monitor window to `voxui-desktop`.

**Architecture:** The Tauri backend owns OpenLive connection lifecycle, raw event storage, parsing, suggestion rendering, and cross-window events. The Leptos frontend adds Live settings, a monitor-window route, main-input replacement, and compact feed controls. The OpenLive client runs in a backend worker thread using blocking HTTP/websocket calls and emits state changes through existing Tauri event patterns.

**Tech Stack:** Rust 2021, Tauri 2, Leptos 0.7 CSR, serde/serde_json, reqwest blocking client, tungstenite blocking websocket, existing Tauri global API bridge.

---

## File Structure

- Create `crates/voxui-desktop/src-tauri/src/live.rs`: live config, live state, raw item storage, message parsing, suggestion rendering, replacement rules, unit tests.
- Create `crates/voxui-desktop/src-tauri/src/openblive.rs`: OpenLive constants, compact JSON signing flow, binary packet protocol, blocking client lifecycle, worker command channel.
- Modify `crates/voxui-desktop/src-tauri/src/types.rs`: add `LiveConfig`, `LiveConfigPatch`, `LiveSnapshot`, DTOs, and include `live` in `AppConfig`.
- Modify `crates/voxui-desktop/src-tauri/src/config.rs`: keep serde defaults compatible through the new `AppConfig.live` default.
- Modify `crates/voxui-desktop/src-tauri/src/app_core.rs`: own live state, expose live snapshot and patch methods, initialize mappings from raw events.
- Modify `crates/voxui-desktop/src-tauri/src/commands.rs`: add live commands, worker lifecycle statics, window creation/closing, `main_input_replace` event.
- Modify `crates/voxui-desktop/src-tauri/src/lib.rs`: register new modules, commands, setup/window event handling, and app-exit cleanup.
- Modify `crates/voxui-desktop/src-tauri/Cargo.toml`: add backend dependencies after preserving existing user edits.
- Modify `crates/voxui-desktop/src-tauri/tauri.conf.json`: define `live-monitor` window with `create: false`.
- Modify `crates/voxui-desktop/src-tauri/capabilities/default.json`: permit both `main` and `live-monitor` windows.
- Modify `crates/voxui-desktop/src/tauri_api.rs`: add live DTOs, commands, and event payload structs.
- Modify `crates/voxui-desktop/src/i18n.rs`: add Live labels in English and Chinese.
- Modify `crates/voxui-desktop/src/app.rs`: route by window label, listen for input replacement, pass live props.
- Modify `crates/voxui-desktop/src/main.rs`: keep a single mount point; route selection happens in `App`.
- Modify `crates/voxui-desktop/src/components/input_box.rs`: support external replacement and add clear button.
- Modify `crates/voxui-desktop/src/components/settings_modal.rs`: add `SettingsPage::Live` and Live settings UI.
- Create `crates/voxui-desktop/src/components/live_monitor.rs`: monitor feed UI, auto-scroll, send buttons.
- Modify `crates/voxui-desktop/src/components/mod.rs`: export `live_monitor`.
- Modify `crates/voxui-desktop/src/styles.css`: Live tab, composer clear button, monitor window styling.
- Create `crates/voxui-desktop/src-tauri/tests/live_tests.rs`: backend live parsing/config/state tests.
- Create `crates/voxui-desktop/src-tauri/tests/openblive_protocol_tests.rs`: packet encoding/decoding and lifecycle unit tests that avoid network.

## Task 1: Backend Live Types and Defaults

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/types.rs`
- Modify: `crates/voxui-desktop/src-tauri/src/config.rs`
- Create: `crates/voxui-desktop/src-tauri/tests/live_tests.rs`

- [ ] **Step 1: Write failing config-default tests**

Add this to `crates/voxui-desktop/src-tauri/tests/live_tests.rs`:

```rust
use voxui_desktop::types::{
    AppConfig, LiveMessageKind, ReplacementRule, TemplateConfig,
};

#[test]
fn live_config_defaults_match_bilibili_monitor_spec() {
    let live = AppConfig::default().live;

    assert_eq!(live.identity_code, "");
    assert!(!live.enable_ceve_server_heartbeat);
    assert!(live.show_danmu);
    assert!(live.show_gifts);
    assert!(live.show_superchats);
    assert!(live.show_guards);
    assert!(!live.show_likes);
    assert!(live.show_enters);
    assert_eq!(live.templates.danmu, "{msg}");
    assert_eq!(
        live.templates.gift_zh,
        "感谢{mapped_uname}送出的{gift_num}个{gift_name}"
    );
    assert!(live.replacement_rules.contains(&ReplacementRule {
        enabled: true,
        from: "我的".to_string(),
        to: "你的".to_string(),
    }));
    assert_eq!(LiveMessageKind::Gift.is_paid(), true);
    assert_eq!(LiveMessageKind::Danmu.is_paid(), false);
}

#[test]
fn old_config_json_deserializes_with_live_defaults() {
    let decoded: AppConfig = serde_json::from_str(r#"{ "max_input_chars": 123 }"#).unwrap();

    assert_eq!(decoded.max_input_chars, 123);
    assert!(decoded.live.show_danmu);
    assert!(!decoded.live.enable_ceve_server_heartbeat);
}

#[test]
fn live_template_config_contains_all_message_templates() {
    let templates = TemplateConfig::default();

    assert_eq!(templates.danmu, "{msg}");
    assert!(templates.gift_en.contains("{gift_num}"));
    assert!(templates.superchat_en.contains("{message}"));
    assert!(templates.guard_zh.contains("{guard_label}"));
    assert!(templates.like_zh.contains("{mapped_uname}"));
    assert!(templates.enter_en.contains("welcome"));
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
cargo test -p voxui-desktop --test live_tests live_config_defaults_match_bilibili_monitor_spec
```

Expected: compile failure naming missing `LiveMessageKind`, `ReplacementRule`, `TemplateConfig`, or `AppConfig.live`.

- [ ] **Step 3: Add live config and DTO types**

In `crates/voxui-desktop/src-tauri/src/types.rs`, add these definitions near existing config types:

```rust
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveMessageKind {
    Danmu,
    Gift,
    Superchat,
    Guard,
    Like,
    Enter,
}

impl LiveMessageKind {
    pub fn is_paid(self) -> bool {
        matches!(self, Self::Gift | Self::Superchat | Self::Guard)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplacementRule {
    pub enabled: bool,
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TemplateConfig {
    pub danmu: String,
    pub gift_zh: String,
    pub gift_en: String,
    pub superchat_zh: String,
    pub superchat_en: String,
    pub guard_zh: String,
    pub guard_en: String,
    pub like_zh: String,
    pub like_en: String,
    pub enter_zh: String,
    pub enter_en: String,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            danmu: "{msg}".to_string(),
            gift_zh: "感谢{mapped_uname}送出的{gift_num}个{gift_name}".to_string(),
            gift_en: "Thank you {mapped_uname} for {gift_num} {gift_name}".to_string(),
            superchat_zh: "感谢{mapped_uname}的醒目留言：{message}".to_string(),
            superchat_en: "Thank you {mapped_uname} for the superchat saying {message}".to_string(),
            guard_zh: "感谢{mapped_uname}开通的{guard_label}".to_string(),
            guard_en: "Thank you {mapped_uname} for joining as {guard_label}".to_string(),
            like_zh: "感谢{mapped_uname}给直播间点赞".to_string(),
            like_en: "Thank you {mapped_uname} for liking the stream".to_string(),
            enter_zh: "欢迎{mapped_uname}进入直播间".to_string(),
            enter_en: "Hi {mapped_uname}, welcome to the stream".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LiveConfig {
    pub identity_code: String,
    pub enable_ceve_server_heartbeat: bool,
    pub show_danmu: bool,
    pub show_gifts: bool,
    pub show_superchats: bool,
    pub show_guards: bool,
    pub show_likes: bool,
    pub show_enters: bool,
    pub templates: TemplateConfig,
    pub replacement_rules: Vec<ReplacementRule>,
    pub mapped_unames: BTreeMap<String, String>,
    pub original_unames: BTreeMap<String, String>,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            identity_code: String::new(),
            enable_ceve_server_heartbeat: false,
            show_danmu: true,
            show_gifts: true,
            show_superchats: true,
            show_guards: true,
            show_likes: false,
            show_enters: true,
            templates: TemplateConfig::default(),
            replacement_rules: vec![
                ReplacementRule { enabled: true, from: "我的".to_string(), to: "你的".to_string() },
                ReplacementRule { enabled: true, from: "我".to_string(), to: "你".to_string() },
                ReplacementRule { enabled: true, from: "I".to_string(), to: "you".to_string() },
                ReplacementRule { enabled: true, from: "me".to_string(), to: "you".to_string() },
                ReplacementRule { enabled: true, from: "my".to_string(), to: "your".to_string() },
            ],
            mapped_unames: BTreeMap::new(),
            original_unames: BTreeMap::new(),
        }
    }
}
```

Then add `pub live: LiveConfig,` to `AppConfig`, initialize it in `Default`, and derive serde defaults through the existing `#[serde(default)]`.

- [ ] **Step 4: Run config tests**

Run:

```powershell
cargo test -p voxui-desktop --test live_tests
cargo test -p voxui-desktop --test config_tests old_config_json_deserializes_to_defaults
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add crates/voxui-desktop/src-tauri/src/types.rs crates/voxui-desktop/src-tauri/tests/live_tests.rs
git commit -m "Add live configuration defaults"
```

## Task 2: Live Message Parsing and Suggestion Rendering

**Files:**
- Create: `crates/voxui-desktop/src-tauri/src/live.rs`
- Modify: `crates/voxui-desktop/src-tauri/src/lib.rs`
- Modify: `crates/voxui-desktop/src-tauri/tests/live_tests.rs`

- [ ] **Step 1: Write failing parser and renderer tests**

Append these tests to `crates/voxui-desktop/src-tauri/tests/live_tests.rs`:

```rust
use serde_json::json;
use voxui_desktop::live::{
    parse_live_event, render_suggestion, switch_text, LiveLanguage, SuggestionMode,
};

#[test]
fn danmu_skips_emote_only_and_cleans_text() {
    let emote_only = json!({
        "cmd": "LIVE_OPEN_PLATFORM_DM",
        "data": { "dm_type": 1, "msg": "[dog]", "open_id": "u1", "uname": "Alice" }
    });
    assert!(parse_live_event(emote_only).unwrap().is_none());

    let normal = json!({
        "cmd": "LIVE_OPEN_PLATFORM_DM",
        "data": { "dm_type": 0, "msg": "hello [dog]  world", "open_id": "u1", "uname": "Alice" }
    });
    let event = parse_live_event(normal).unwrap().unwrap();
    let config = AppConfig::default().live;

    assert_eq!(
        render_suggestion(&event, &config, LiveLanguage::English, SuggestionMode::Normal),
        Some("hello.world".to_string())
    );
}

#[test]
fn paid_gift_requires_paid_true() {
    let unpaid = json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": { "paid": false, "gift_name": "花", "gift_num": 2, "open_id": "u1", "uname": "Alice" }
    });
    assert!(parse_live_event(unpaid).unwrap().is_none());

    let paid = json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": { "paid": true, "gift_name": "花", "gift_num": 2, "open_id": "u1", "uname": "Alice" }
    });
    let event = parse_live_event(paid).unwrap().unwrap();
    let mut config = AppConfig::default().live;
    config.mapped_unames.insert("u1".to_string(), "A酱".to_string());

    assert_eq!(
        render_suggestion(&event, &config, LiveLanguage::Chinese, SuggestionMode::Normal),
        Some("感谢A酱送出的2个花".to_string())
    );
}

#[test]
fn superchat_guard_like_and_enter_render_expected_text() {
    let mut config = AppConfig::default().live;
    config.mapped_unames.insert("u2".to_string(), "Bob".to_string());

    let superchat = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SUPER_CHAT",
        "data": { "message": "加油", "rmb": 30, "open_id": "u2", "uname": "Bob" }
    })).unwrap().unwrap();
    assert_eq!(
        render_suggestion(&superchat, &config, LiveLanguage::English, SuggestionMode::Normal),
        Some("Thank you Bob for the superchat saying 加油".to_string())
    );

    let guard = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_GUARD",
        "data": { "guard_level": 3, "guard_num": 1, "price": 198000, "user_info": { "open_id": "u2", "uname": "Bob", "uface": "" } }
    })).unwrap().unwrap();
    assert_eq!(
        render_suggestion(&guard, &config, LiveLanguage::Chinese, SuggestionMode::Normal),
        Some("感谢Bob开通的舰长".to_string())
    );

    let like = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_LIKE",
        "data": { "open_id": "u2", "uname": "Bob", "like_count": 4 }
    })).unwrap().unwrap();
    assert_eq!(
        render_suggestion(&like, &config, LiveLanguage::Chinese, SuggestionMode::Normal),
        Some("感谢Bob给直播间点赞".to_string())
    );

    let enter = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_LIVE_ROOM_ENTER",
        "data": { "open_id": "u2", "uname": "Bob", "uface": "" }
    })).unwrap().unwrap();
    assert_eq!(
        render_suggestion(&enter, &config, LiveLanguage::English, SuggestionMode::Normal),
        Some("Hi Bob, welcome to the stream".to_string())
    );
}

#[test]
fn switch_mode_applies_enabled_replacement_rules_in_order() {
    let mut config = AppConfig::default().live;
    config.replacement_rules[2].enabled = false;

    assert_eq!(switch_text("我的猫 and I like my chair", &config), "你的猫 and I like your chair");
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
cargo test -p voxui-desktop --test live_tests danmu_skips_emote_only_and_cleans_text
```

Expected: compile failure because `voxui_desktop::live` is missing.

- [ ] **Step 3: Implement parser and renderer**

Create `crates/voxui-desktop/src-tauri/src/live.rs` with this core shape:

```rust
use serde_json::Value;

use crate::types::{LiveConfig, LiveMessageKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveLanguage {
    Chinese,
    English,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionMode {
    Normal,
    Switch,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LiveEvent {
    pub kind: LiveMessageKind,
    pub raw: Value,
    pub open_id: String,
    pub uname: String,
    pub msg: Option<String>,
    pub gift_name: Option<String>,
    pub gift_num: Option<u64>,
    pub superchat_message: Option<String>,
    pub guard_label: Option<String>,
}

pub fn parse_live_event(raw: Value) -> anyhow::Result<Option<LiveEvent>> {
    let cmd = raw.get("cmd").and_then(Value::as_str).unwrap_or_default();
    let data = raw.get("data").unwrap_or(&Value::Null);
    let event = match cmd {
        "LIVE_OPEN_PLATFORM_DM" => {
            if data.get("dm_type").and_then(Value::as_u64) == Some(1) {
                return Ok(None);
            }
            LiveEvent {
                kind: LiveMessageKind::Danmu,
                raw,
                open_id: string_at(data, "open_id"),
                uname: string_at(data, "uname"),
                msg: Some(string_at(data, "msg")),
                gift_name: None,
                gift_num: None,
                superchat_message: None,
                guard_label: None,
            }
        }
        "LIVE_OPEN_PLATFORM_SEND_GIFT" => {
            if data.get("paid").and_then(Value::as_bool) != Some(true) {
                return Ok(None);
            }
            LiveEvent {
                kind: LiveMessageKind::Gift,
                raw,
                open_id: string_at(data, "open_id"),
                uname: string_at(data, "uname"),
                msg: None,
                gift_name: Some(string_at(data, "gift_name")),
                gift_num: Some(data.get("gift_num").and_then(Value::as_u64).unwrap_or(0)),
                superchat_message: None,
                guard_label: None,
            }
        }
        "LIVE_OPEN_PLATFORM_SUPER_CHAT" => LiveEvent {
            kind: LiveMessageKind::Superchat,
            raw,
            open_id: string_at(data, "open_id"),
            uname: string_at(data, "uname"),
            msg: None,
            gift_name: None,
            gift_num: None,
            superchat_message: Some(string_at(data, "message")),
            guard_label: None,
        },
        "LIVE_OPEN_PLATFORM_GUARD" => {
            let user = data.get("user_info").unwrap_or(&Value::Null);
            LiveEvent {
                kind: LiveMessageKind::Guard,
                raw,
                open_id: string_at(user, "open_id"),
                uname: string_at(user, "uname"),
                msg: None,
                gift_name: None,
                gift_num: None,
                superchat_message: None,
                guard_label: Some(guard_label(data.get("guard_level").and_then(Value::as_u64))),
            }
        }
        "LIVE_OPEN_PLATFORM_LIKE" => LiveEvent {
            kind: LiveMessageKind::Like,
            raw,
            open_id: string_at(data, "open_id"),
            uname: string_at(data, "uname"),
            msg: None,
            gift_name: None,
            gift_num: None,
            superchat_message: None,
            guard_label: None,
        },
        "LIVE_OPEN_PLATFORM_LIVE_ROOM_ENTER" => LiveEvent {
            kind: LiveMessageKind::Enter,
            raw,
            open_id: string_at(data, "open_id"),
            uname: string_at(data, "uname"),
            msg: None,
            gift_name: None,
            gift_num: None,
            superchat_message: None,
            guard_label: None,
        },
        _ => return Ok(None),
    };
    Ok(Some(event))
}

pub fn render_suggestion(
    event: &LiveEvent,
    config: &LiveConfig,
    language: LiveLanguage,
    mode: SuggestionMode,
) -> Option<String> {
    if !kind_enabled(event.kind, config) {
        return None;
    }
    let mapped = config
        .mapped_unames
        .get(&event.open_id)
        .cloned()
        .unwrap_or_else(|| event.uname.clone());
    let period = match language {
        LiveLanguage::Chinese => "。",
        LiveLanguage::English => ".",
    };
    let text = match event.kind {
        LiveMessageKind::Danmu => clean_danmu(event.msg.as_deref().unwrap_or_default(), period),
        LiveMessageKind::Gift => render_template(
            choose(&config.templates.gift_zh, &config.templates.gift_en, language),
            &[
                ("mapped_uname", mapped.as_str()),
                ("gift_num", &event.gift_num.unwrap_or(0).to_string()),
                ("gift_name", event.gift_name.as_deref().unwrap_or_default()),
            ],
        ),
        LiveMessageKind::Superchat => render_template(
            choose(&config.templates.superchat_zh, &config.templates.superchat_en, language),
            &[
                ("mapped_uname", mapped.as_str()),
                ("message", event.superchat_message.as_deref().unwrap_or_default()),
            ],
        ),
        LiveMessageKind::Guard => render_template(
            choose(&config.templates.guard_zh, &config.templates.guard_en, language),
            &[
                ("mapped_uname", mapped.as_str()),
                ("guard_label", event.guard_label.as_deref().unwrap_or("航海")),
            ],
        ),
        LiveMessageKind::Like => render_template(
            choose(&config.templates.like_zh, &config.templates.like_en, language),
            &[("mapped_uname", mapped.as_str())],
        ),
        LiveMessageKind::Enter => render_template(
            choose(&config.templates.enter_zh, &config.templates.enter_en, language),
            &[("mapped_uname", mapped.as_str())],
        ),
    };
    Some(match mode {
        SuggestionMode::Normal => text,
        SuggestionMode::Switch => switch_text(&text, config),
    })
}

pub fn switch_text(text: &str, config: &LiveConfig) -> String {
    config
        .replacement_rules
        .iter()
        .filter(|rule| rule.enabled)
        .fold(text.to_string(), |current, rule| current.replace(&rule.from, &rule.to))
}
```

Add helper functions in the same file: `string_at`, `guard_label`, `kind_enabled`, `choose`, `render_template`, and `clean_danmu`. Implement `clean_danmu` with a small state machine that skips characters between `[` and `]`, collapses whitespace runs into the requested period, and trims periods at both ends.

In `crates/voxui-desktop/src-tauri/src/lib.rs`, add:

```rust
pub mod live;
```

- [ ] **Step 4: Run parser tests**

Run:

```powershell
cargo test -p voxui-desktop --test live_tests
```

Expected: all live parser/config tests pass.

- [ ] **Step 5: Commit**

```powershell
git add crates/voxui-desktop/src-tauri/src/live.rs crates/voxui-desktop/src-tauri/src/lib.rs crates/voxui-desktop/src-tauri/tests/live_tests.rs
git commit -m "Add Bilibili live event parsing"
```

## Task 3: Live State, Patches, and Recomputed Items

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/types.rs`
- Modify: `crates/voxui-desktop/src-tauri/src/live.rs`
- Modify: `crates/voxui-desktop/src-tauri/src/app_core.rs`
- Modify: `crates/voxui-desktop/src-tauri/tests/live_tests.rs`

- [ ] **Step 1: Write failing live-state tests**

Append to `crates/voxui-desktop/src-tauri/tests/live_tests.rs`:

```rust
use voxui_desktop::types::{LiveConfigPatch, LiveStatus};

#[test]
fn adding_live_event_initializes_name_mapping_and_recomputes_after_patch() {
    let mut core = voxui_desktop::app_core::AppCore::from_config(AppConfig::default()).unwrap();
    let event = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": { "paid": true, "gift_name": "花", "gift_num": 2, "open_id": "u1", "uname": "Alice" }
    })).unwrap().unwrap();

    let item_id = core.add_live_event_for_test(event).unwrap();
    let first = core.live_snapshot_for_test(LiveLanguage::Chinese);
    assert_eq!(first.config.original_unames.get("u1").map(String::as_str), Some("Alice"));
    assert_eq!(first.items[0].suggestion, "感谢Alice送出的2个花");

    core.apply_live_patch(LiveConfigPatch {
        mapped_unames: Some([("u1".to_string(), "A酱".to_string())].into_iter().collect()),
        ..LiveConfigPatch::default()
    }).unwrap();
    let second = core.live_snapshot_for_test(LiveLanguage::Chinese);

    assert_eq!(second.items[0].id, item_id);
    assert_eq!(second.items[0].suggestion, "感谢A酱送出的2个花");
}

#[test]
fn live_snapshot_filters_likes_by_default_until_enabled() {
    let mut core = voxui_desktop::app_core::AppCore::from_config(AppConfig::default()).unwrap();
    let like = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_LIKE",
        "data": { "open_id": "u2", "uname": "Bob", "like_count": 1 }
    })).unwrap().unwrap();

    core.add_live_event_for_test(like).unwrap();
    assert!(core.live_snapshot_for_test(LiveLanguage::Chinese).items.is_empty());

    core.apply_live_patch(LiveConfigPatch {
        show_likes: Some(true),
        ..LiveConfigPatch::default()
    }).unwrap();

    assert_eq!(core.live_snapshot_for_test(LiveLanguage::Chinese).items.len(), 1);
}

#[test]
fn live_status_defaults_to_disconnected() {
    let core = voxui_desktop::app_core::AppCore::from_config(AppConfig::default()).unwrap();

    assert_eq!(core.live_status_for_test(), LiveStatus::Disconnected);
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
cargo test -p voxui-desktop --test live_tests adding_live_event_initializes_name_mapping_and_recomputes_after_patch
```

Expected: compile failure naming missing live state APIs and DTOs.

- [ ] **Step 3: Add DTOs and patch type**

In `crates/voxui-desktop/src-tauri/src/types.rs`, add:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveStatus {
    Disconnected,
    Connecting,
    Connected,
    Disconnecting,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveMonitorItemDto {
    pub id: String,
    pub kind: LiveMessageKind,
    pub paid: bool,
    pub open_id: String,
    pub uname: String,
    pub mapped_uname: String,
    pub suggestion: String,
    pub raw_json: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveSnapshot {
    pub status: LiveStatus,
    pub status_message: Option<String>,
    pub config: LiveConfig,
    pub items: Vec<LiveMonitorItemDto>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LiveConfigPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_ceve_server_heartbeat: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_danmu: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_gifts: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_superchats: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_guards: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_likes: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_enters: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub templates: Option<TemplateConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replacement_rules: Option<Vec<ReplacementRule>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapped_unames: Option<BTreeMap<String, String>>,
}
```

- [ ] **Step 4: Add live state implementation**

In `crates/voxui-desktop/src-tauri/src/live.rs`, add `LiveState`:

```rust
#[derive(Debug, Clone)]
struct StoredLiveEvent {
    id: String,
    event: LiveEvent,
}

#[derive(Debug, Default)]
pub struct LiveState {
    status: crate::types::LiveStatus,
    status_message: Option<String>,
    next_item_id: u64,
    events: Vec<StoredLiveEvent>,
}

impl LiveState {
    pub fn snapshot(
        &self,
        config: &LiveConfig,
        language: LiveLanguage,
    ) -> crate::types::LiveSnapshot {
        let items = self
            .events
            .iter()
            .filter_map(|stored| {
                let event = &stored.event;
                let suggestion = render_suggestion(event, config, language, SuggestionMode::Normal)?;
                let mapped = config
                    .mapped_unames
                    .get(&event.open_id)
                    .cloned()
                    .unwrap_or_else(|| event.uname.clone());
                Some(crate::types::LiveMonitorItemDto {
                    id: stored.id.clone(),
                    kind: event.kind,
                    paid: event.kind.is_paid(),
                    open_id: event.open_id.clone(),
                    uname: event.uname.clone(),
                    mapped_uname: mapped,
                    suggestion,
                    raw_json: event.raw.clone(),
                })
            })
            .collect();
        crate::types::LiveSnapshot {
            status: self.status,
            status_message: self.status_message.clone(),
            config: config.clone(),
            items,
        }
    }

    pub fn push(&mut self, mut event: LiveEvent) -> String {
        self.next_item_id = self.next_item_id.saturating_add(1);
        let id = format!("live-{}", self.next_item_id);
        self.events.push(StoredLiveEvent { id: id.clone(), event });
        id
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }
}
```

Add this lookup method so sending a monitor item always recomputes from the stored event and current settings instead of caching stale suggestion text:

```rust
pub fn suggestion_for_item(
    &self,
    item_id: &str,
    config: &LiveConfig,
    language: LiveLanguage,
    mode: SuggestionMode,
) -> Option<String> {
    self.events
        .iter()
        .find(|stored| stored.id == item_id)
        .and_then(|stored| render_suggestion(&stored.event, config, language, mode))
}
```

- [ ] **Step 5: Wire live state into AppCore**

In `crates/voxui-desktop/src-tauri/src/app_core.rs`, add a `live: crate::live::LiveState` field, initialize it in `from_config`, and add methods:

```rust
pub fn add_live_event(&mut self, event: crate::live::LiveEvent) -> Result<String> {
    if !event.open_id.is_empty() {
        self.config
            .live
            .original_unames
            .entry(event.open_id.clone())
            .or_insert_with(|| event.uname.clone());
        self.config
            .live
            .mapped_unames
            .entry(event.open_id.clone())
            .or_insert_with(|| event.uname.clone());
    }
    let id = self.live.push(event);
    self.persist_config()?;
    Ok(id)
}

pub fn apply_live_patch(&mut self, patch: crate::types::LiveConfigPatch) -> Result<crate::types::LiveSnapshot> {
    if let Some(value) = patch.identity_code {
        self.config.live.identity_code = value;
    }
    if let Some(value) = patch.enable_ceve_server_heartbeat {
        self.config.live.enable_ceve_server_heartbeat = value;
    }
    if let Some(value) = patch.show_danmu {
        self.config.live.show_danmu = value;
    }
    if let Some(value) = patch.show_gifts {
        self.config.live.show_gifts = value;
    }
    if let Some(value) = patch.show_superchats {
        self.config.live.show_superchats = value;
    }
    if let Some(value) = patch.show_guards {
        self.config.live.show_guards = value;
    }
    if let Some(value) = patch.show_likes {
        self.config.live.show_likes = value;
    }
    if let Some(value) = patch.show_enters {
        self.config.live.show_enters = value;
    }
    if let Some(value) = patch.templates {
        self.config.live.templates = value;
    }
    if let Some(value) = patch.replacement_rules {
        self.config.live.replacement_rules = value;
    }
    if let Some(value) = patch.mapped_unames {
        self.config.live.mapped_unames = value;
    }
    self.persist_config()?;
    Ok(self.live_snapshot(crate::live::LiveLanguage::English))
}
```

Add `live_snapshot`, `clear_live_items`, and test-only wrappers named exactly as used by the tests.

Also add:

```rust
pub fn live_suggestion_for_item(
    &self,
    item_id: &str,
    language: crate::live::LiveLanguage,
    mode: crate::live::SuggestionMode,
) -> Option<String> {
    self.live
        .suggestion_for_item(item_id, &self.config.live, language, mode)
}
```

- [ ] **Step 6: Run tests**

Run:

```powershell
cargo test -p voxui-desktop --test live_tests
cargo test -p voxui-desktop --test app_core_tests applying_config_patch_persists_the_saved_config_to_disk
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```powershell
git add crates/voxui-desktop/src-tauri/src/types.rs crates/voxui-desktop/src-tauri/src/live.rs crates/voxui-desktop/src-tauri/src/app_core.rs crates/voxui-desktop/src-tauri/tests/live_tests.rs
git commit -m "Store and recompute live monitor items"
```

## Task 4: OpenLive Packet Protocol and Client Skeleton

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/Cargo.toml`
- Create: `crates/voxui-desktop/src-tauri/src/openblive.rs`
- Modify: `crates/voxui-desktop/src-tauri/src/lib.rs`
- Create: `crates/voxui-desktop/src-tauri/tests/openblive_protocol_tests.rs`

- [ ] **Step 1: Write failing protocol tests**

Create `crates/voxui-desktop/src-tauri/tests/openblive_protocol_tests.rs`:

```rust
use voxui_desktop::openblive::{
    compact_json_body, unpack_packet, OpenBlivePacket, APP_ID, CEVE_HEARTBEAT_URL,
    HEARTBEAT_INTERVAL_SECS, HOST, SIGN_URLS,
};

#[test]
fn openblive_constants_match_proven_provider_values() {
    assert_eq!(APP_ID, 1651388990835);
    assert_eq!(HOST, "https://live-open.biliapi.com");
    assert_eq!(SIGN_URLS[0], "https://soft.ceve-market.org/bopen/sign");
    assert_eq!(SIGN_URLS[1], "https://bopen.ceve-market.org/sign");
    assert_eq!(CEVE_HEARTBEAT_URL, "http://localhost.ceve-market.org:5218/heartbeat");
    assert_eq!(HEARTBEAT_INTERVAL_SECS, 20);
}

#[test]
fn compact_json_body_matches_signing_requirement() {
    let body = compact_json_body(&serde_json::json!({
        "code": "ABC",
        "app_id": APP_ID
    })).unwrap();

    assert_eq!(body, r#"{"app_id":1651388990835,"code":"ABC"}"#);
}

#[test]
fn packet_pack_round_trips_auth_body() {
    let packet = OpenBlivePacket {
        op: 7,
        body: br#"{"roomid":1}"#.to_vec(),
    };
    let packed = packet.pack();
    let decoded = unpack_packet(&packed).unwrap();

    assert_eq!(decoded.op, 7);
    assert_eq!(decoded.body, br#"{"roomid":1}"#);
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
cargo test -p voxui-desktop --test openblive_protocol_tests
```

Expected: compile failure because `openblive` module is missing.

- [ ] **Step 3: Add dependencies without overwriting user edits**

First inspect the pre-existing Cargo change:

```powershell
git diff -- crates/voxui-desktop/src-tauri/Cargo.toml
```

Then add these dependencies to `crates/voxui-desktop/src-tauri/Cargo.toml` while preserving the existing diff:

```toml
reqwest = { version = "0.12", default-features = false, features = ["blocking", "json", "rustls-tls"] }
tungstenite = { version = "0.26", default-features = false, features = ["rustls-tls-webpki-roots"] }
url = "2"
```

- [ ] **Step 4: Implement packet helpers and constants**

Create `crates/voxui-desktop/src-tauri/src/openblive.rs` with:

```rust
use anyhow::{bail, Context, Result};
use serde_json::Value;

pub const APP_ID: u64 = 1_651_388_990_835;
pub const HOST: &str = "https://live-open.biliapi.com";
pub const SIGN_URLS: [&str; 2] = [
    "https://soft.ceve-market.org/bopen/sign",
    "https://bopen.ceve-market.org/sign",
];
pub const CEVE_HEARTBEAT_URL: &str = "http://localhost.ceve-market.org:5218/heartbeat";
pub const HEARTBEAT_INTERVAL_SECS: u64 = 20;

const HEADER_LEN: u32 = 16;
const PROTOCOL_VERSION: u16 = 0;
const SEQUENCE_ID: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenBlivePacket {
    pub op: u32,
    pub body: Vec<u8>,
}

impl OpenBlivePacket {
    pub fn pack(&self) -> Vec<u8> {
        let packet_len = HEADER_LEN as usize + self.body.len();
        let mut out = Vec::with_capacity(packet_len);
        out.extend_from_slice(&(packet_len as u32).to_be_bytes());
        out.extend_from_slice(&(HEADER_LEN as u16).to_be_bytes());
        out.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
        out.extend_from_slice(&self.op.to_be_bytes());
        out.extend_from_slice(&SEQUENCE_ID.to_be_bytes());
        out.extend_from_slice(&self.body);
        out
    }
}

pub fn unpack_packet(bytes: &[u8]) -> Result<OpenBlivePacket> {
    if bytes.len() < HEADER_LEN as usize {
        bail!("OpenLive packet shorter than header");
    }
    let packet_len = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let header_len = u16::from_be_bytes(bytes[4..6].try_into().unwrap()) as usize;
    if packet_len > bytes.len() || header_len > packet_len || header_len < HEADER_LEN as usize {
        bail!("invalid OpenLive packet lengths");
    }
    let op = u32::from_be_bytes(bytes[8..12].try_into().unwrap());
    Ok(OpenBlivePacket {
        op,
        body: bytes[header_len..packet_len].to_vec(),
    })
}

pub fn compact_json_body(value: &Value) -> Result<String> {
    match value {
        Value::Object(map) => {
            let sorted = map
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<serde_json::Map<_, _>>();
            serde_json::to_string(&Value::Object(sorted)).context("serialize compact JSON body")
        }
        _ => serde_json::to_string(value).context("serialize compact JSON body"),
    }
}
```

Add `pub mod openblive;` in `crates/voxui-desktop/src-tauri/src/lib.rs`.

- [ ] **Step 5: Run protocol tests**

Run:

```powershell
cargo test -p voxui-desktop --test openblive_protocol_tests
```

Expected: protocol tests pass.

- [ ] **Step 6: Commit**

```powershell
git add crates/voxui-desktop/src-tauri/Cargo.toml Cargo.lock crates/voxui-desktop/src-tauri/src/openblive.rs crates/voxui-desktop/src-tauri/src/lib.rs crates/voxui-desktop/src-tauri/tests/openblive_protocol_tests.rs
git commit -m "Add OpenLive protocol helpers"
```

## Task 5: OpenLive Worker Lifecycle and Tauri Commands

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/src/openblive.rs`
- Modify: `crates/voxui-desktop/src-tauri/src/commands.rs`
- Modify: `crates/voxui-desktop/src-tauri/src/lib.rs`
- Modify: `crates/voxui-desktop/src-tauri/src/types.rs`
- Modify: `crates/voxui-desktop/src-tauri/tests/openblive_protocol_tests.rs`

- [ ] **Step 1: Add lifecycle test for cleanup idempotence**

Append to `crates/voxui-desktop/src-tauri/tests/openblive_protocol_tests.rs`:

```rust
use voxui_desktop::openblive::{LiveWorkerState, WorkerCommand};

#[test]
fn worker_state_disconnect_is_idempotent() {
    let mut state = LiveWorkerState::for_test(Some("game-1".to_string()));

    assert_eq!(state.mark_disconnect_requested(), Some(WorkerCommand::EndApp { game_id: "game-1".to_string() }));
    assert_eq!(state.mark_disconnect_requested(), None);
}
```

- [ ] **Step 2: Run test and verify failure**

Run:

```powershell
cargo test -p voxui-desktop --test openblive_protocol_tests worker_state_disconnect_is_idempotent
```

Expected: compile failure for missing `LiveWorkerState` and `WorkerCommand`.

- [ ] **Step 3: Implement worker state and command model**

In `crates/voxui-desktop/src-tauri/src/openblive.rs`, add:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerCommand {
    EndApp { game_id: String },
    Stop,
}

#[derive(Debug, Default)]
pub struct LiveWorkerState {
    game_id: Option<String>,
    end_sent: bool,
}

impl LiveWorkerState {
    pub fn for_test(game_id: Option<String>) -> Self {
        Self { game_id, end_sent: false }
    }

    pub fn set_game_id(&mut self, game_id: String) {
        self.game_id = Some(game_id);
    }

    pub fn mark_disconnect_requested(&mut self) -> Option<WorkerCommand> {
        if self.end_sent {
            return None;
        }
        self.end_sent = true;
        self.game_id
            .clone()
            .map(|game_id| WorkerCommand::EndApp { game_id })
    }
}
```

Then implement the blocking connection flow around `reqwest::blocking::Client` and `tungstenite::connect`. Keep the public entrypoint small:

```rust
pub fn run_openblive_worker(
    identity_code: String,
    enable_ceve_server_heartbeat: bool,
    mut on_event: impl FnMut(serde_json::Value) + Send + 'static,
    mut on_status: impl FnMut(crate::types::LiveStatus, Option<String>) + Send + 'static,
    mut should_stop: impl FnMut() -> bool + Send + 'static,
) {
    on_status(crate::types::LiveStatus::Connecting, None);
    let result = run_openblive_worker_inner(
        &identity_code,
        enable_ceve_server_heartbeat,
        &mut on_event,
        &mut should_stop,
    );
    match result {
        Ok(()) => on_status(crate::types::LiveStatus::Disconnected, None),
        Err(error) => on_status(crate::types::LiveStatus::Error, Some(error.to_string())),
    }
}
```

Keep HTTP calls in helper functions named `sign_body`, `post_openlive`, `start_app`, `heartbeat_app_once`, `heartbeat_ceve_once`, and `end_app_once`. Each helper should accept a `reqwest::blocking::Client` reference so tests can later isolate serialization without network.

- [ ] **Step 4: Add Tauri live commands**

In `crates/voxui-desktop/src-tauri/src/commands.rs`, add command handlers:

```rust
#[tauri::command]
pub fn get_live_state(state: State<'_, SharedAppCore>) -> Result<crate::types::LiveSnapshot, String> {
    with_core(state, |core| Ok(core.live_snapshot(crate::live::LiveLanguage::English)))
}

#[tauri::command]
pub fn set_live_config_patch(
    state: State<'_, SharedAppCore>,
    patch: crate::types::LiveConfigPatch,
) -> Result<crate::types::LiveSnapshot, String> {
    with_core(state, |core| core.apply_live_patch(patch))
}

#[tauri::command]
pub fn clear_live_items(state: State<'_, SharedAppCore>) -> Result<crate::types::LiveSnapshot, String> {
    with_core(state, |core| {
        core.clear_live_items();
        Ok(core.live_snapshot(crate::live::LiveLanguage::English))
    })
}
```

Add `connect_openblive`, `disconnect_openblive`, and `send_live_suggestion` in the same file. Use `OnceLock<Mutex<Option<LiveWorkerHandle>>>` for the worker, matching the sidecar process pattern.

`connect_openblive` must not run signing, HTTP calls, websocket auth, websocket receives, or heartbeat loops inline in the Tauri command handler. The command should:

1. set live status to `Connecting`;
2. create a stop channel or atomic stop flag;
3. spawn `run_openblive_worker` with `thread::Builder::new().name("voxui-openblive".to_string()).spawn(...)`;
4. store the worker handle in the `OnceLock` slot;
5. return to the frontend immediately after the worker is spawned.

The worker thread emits `live_status_changed` and `live_items_changed` events as connection progress and messages arrive. On successful websocket auth, the worker asks the Tauri app handle to create or show `live-monitor`. On unexpected disconnect, it emits error/disconnected status and asks the app handle to close `live-monitor`.

`disconnect_openblive` should signal the worker to stop and return promptly. The worker owns `/v2/app/end` cleanup because it owns `game_id` and the websocket.

`send_live_suggestion` should recompute the selected item text from stored raw-backed event state and current settings, then emit:

```rust
window.emit("main_input_replace", crate::types::MainInputReplaceEvent { text })
```

Define `MainInputReplaceEvent` in `types.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MainInputReplaceEvent {
    pub text: String,
}
```

- [ ] **Step 5: Register commands and shutdown cleanup**

In `crates/voxui-desktop/src-tauri/src/lib.rs`, add commands to `invoke_handler`:

```rust
commands::get_live_state,
commands::set_live_config_patch,
commands::clear_live_items,
commands::connect_openblive,
commands::disconnect_openblive,
commands::send_live_suggestion,
```

Add `.on_window_event` handling so closing `live-monitor` calls the same disconnect path. Add `.setup` if needed to prepare the hidden monitor window config. Use an app-exit hook or final cleanup path available in Tauri 2 to call `commands::shutdown_live_worker_for_app_exit()`.

- [ ] **Step 6: Run backend checks**

Run:

```powershell
cargo test -p voxui-desktop --test openblive_protocol_tests
cargo test -p voxui-desktop --test live_tests
cargo check -p voxui-desktop
```

Expected: tests pass and `cargo check` finishes without errors.

- [ ] **Step 7: Commit**

```powershell
git add crates/voxui-desktop/src-tauri/src/openblive.rs crates/voxui-desktop/src-tauri/src/commands.rs crates/voxui-desktop/src-tauri/src/lib.rs crates/voxui-desktop/src-tauri/src/types.rs crates/voxui-desktop/src-tauri/tests/openblive_protocol_tests.rs
git commit -m "Wire OpenLive worker commands"
```

## Task 6: Tauri Window Configuration

**Files:**
- Modify: `crates/voxui-desktop/src-tauri/tauri.conf.json`
- Modify: `crates/voxui-desktop/src-tauri/capabilities/default.json`
- Modify: `crates/voxui-desktop/src-tauri/tests/config_tests.rs`

- [ ] **Step 1: Write failing config tests**

Append to `crates/voxui-desktop/src-tauri/tests/config_tests.rs`:

```rust
#[test]
fn tauri_config_defines_hidden_live_monitor_window() {
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tauri.conf.json");
    let config_text = fs::read_to_string(config_path).unwrap();
    let config: serde_json::Value = serde_json::from_str(&config_text).unwrap();
    let windows = config["app"]["windows"].as_array().unwrap();
    let live = windows
        .iter()
        .find(|window| window["label"] == "live-monitor")
        .expect("live-monitor window config");

    assert_eq!(live["title"], "Bilibili Live Monitor");
    assert_eq!(live["create"], false);
    assert_eq!(live["width"], 420);
    assert_eq!(live["height"], 640);
}

#[test]
fn default_capability_allows_main_and_live_monitor_windows() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("capabilities/default.json");
    let text = fs::read_to_string(path).unwrap();
    let capability: serde_json::Value = serde_json::from_str(&text).unwrap();

    assert_eq!(capability["windows"], serde_json::json!(["main", "live-monitor"]));
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
cargo test -p voxui-desktop --test config_tests tauri_config_defines_hidden_live_monitor_window
```

Expected: failure because the second window is not configured.

- [ ] **Step 3: Update Tauri config**

In `crates/voxui-desktop/src-tauri/tauri.conf.json`, add this second window object to `app.windows`:

```json
{
  "label": "live-monitor",
  "title": "Bilibili Live Monitor",
  "width": 420,
  "height": 640,
  "minWidth": 360,
  "minHeight": 420,
  "create": false
}
```

In `crates/voxui-desktop/src-tauri/capabilities/default.json`, change:

```json
"windows": ["main"]
```

to:

```json
"windows": ["main", "live-monitor"]
```

- [ ] **Step 4: Run config tests**

Run:

```powershell
cargo test -p voxui-desktop --test config_tests
```

Expected: config tests pass.

- [ ] **Step 5: Commit**

```powershell
git add crates/voxui-desktop/src-tauri/tauri.conf.json crates/voxui-desktop/src-tauri/capabilities/default.json crates/voxui-desktop/src-tauri/tests/config_tests.rs
git commit -m "Configure live monitor window"
```

## Task 7: Frontend API, Labels, Input Replacement, and Clear Button

**Files:**
- Modify: `crates/voxui-desktop/src/tauri_api.rs`
- Modify: `crates/voxui-desktop/src/i18n.rs`
- Modify: `crates/voxui-desktop/src/app.rs`
- Modify: `crates/voxui-desktop/src/components/input_box.rs`
- Modify: `crates/voxui-desktop/src/styles.css`

- [ ] **Step 1: Add frontend DTOs and commands**

In `crates/voxui-desktop/src/tauri_api.rs`, mirror backend DTOs:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveStatus {
    Disconnected,
    Connecting,
    Connected,
    Disconnecting,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveMessageKind {
    Danmu,
    Gift,
    Superchat,
    Guard,
    Like,
    Enter,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveSnapshot {
    pub status: LiveStatus,
    pub status_message: Option<String>,
    pub config: LiveConfig,
    pub items: Vec<LiveMonitorItem>,
}
```

Add matching `LiveConfig`, `TemplateConfig`, `ReplacementRule`, `LiveConfigPatch`, `LiveMonitorItem`, and async wrappers for `get_live_state`, `set_live_config_patch`, `connect_openblive`, `disconnect_openblive`, `send_live_suggestion`, and `clear_live_items`.

Add this JS helper to the existing `#[wasm_bindgen(inline_js = ...)]` block:

```javascript
export function currentWindowLabel() {
  const current = globalThis.__TAURI__?.webviewWindow?.getCurrentWebviewWindow?.();
  return typeof current?.label === "string" ? current.label : "main";
}
```

Add the Rust extern and wrapper:

```rust
#[wasm_bindgen(js_name = currentWindowLabel)]
fn current_window_label_js() -> String;

pub fn current_window_label() -> String {
    current_window_label_js()
}
```

- [ ] **Step 2: Add labels**

In `crates/voxui-desktop/src/i18n.rs`, add fields:

```rust
pub live: &'static str,
pub identity_code: &'static str,
pub connect: &'static str,
pub disconnect: &'static str,
pub clear: &'static str,
pub ceve_heartbeat: &'static str,
pub danmu: &'static str,
pub gift: &'static str,
pub superchat: &'static str,
pub guard: &'static str,
pub like: &'static str,
pub enter: &'static str,
pub paid: &'static str,
pub send: &'static str,
pub switch_send: &'static str,
```

Use clear English labels and Chinese labels matching the spec. For example English `Live`, `Identity code`, `Connect`, `Disconnect`, `Clear`, `ceve heartbeat`, `Danmu`, `Gift`, `Superchat`, `Guard`, `Like`, `Enter`, `Paid`, `Send`, `Switch pronouns`. Chinese `直播`, `身份码`, `连接`, `断开`, `清空`, `ceve 心跳`, `弹幕`, `礼物`, `醒目留言`, `大航海`, `点赞`, `进场`, `付费`, `发送`, `人称替换`.

- [ ] **Step 3: Modify InputBox props and event handling**

Change `InputBox` signature in `crates/voxui-desktop/src/components/input_box.rs`:

```rust
pub fn InputBox(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    max_chars: impl Fn() -> usize + Send + Sync + 'static + Copy,
    disabled: impl Fn() -> bool + Send + Sync + 'static + Copy,
    replacement_text: impl Fn() -> Option<String> + Send + Sync + 'static + Copy,
    on_replacement_consumed: impl Fn() + Send + Sync + 'static + Copy,
    on_generate: impl Fn(String) + 'static + Copy,
) -> impl IntoView
```

Inside the component, add:

```rust
Effect::new(move |_| {
    if let Some(next_text) = replacement_text() {
        set_text.set(next_text);
        on_replacement_consumed();
    }
});
```

Change the composer button area to:

```rust
<div class="composer-actions">
    <button class="generate-button" type="submit" disabled=generate_disabled>
        {move || labels().generate}
    </button>
    <button
        class="secondary-button composer-clear-button"
        type="button"
        disabled=move || disabled() || text.get().is_empty()
        on:click=move |_| set_text.set(String::new())
    >
        {move || labels().clear}
    </button>
</div>
```

- [ ] **Step 4: Listen for main input replacement in App**

In `crates/voxui-desktop/src/app.rs`, add:

```rust
let (input_replacement, set_input_replacement) = signal(None::<String>);

spawn_local(async move {
    let _ = crate::tauri_api::listen_app_event("main_input_replace", move |event| {
        if let Ok(payload) = crate::tauri_api::decode_app_event::<crate::tauri_api::MainInputReplaceEvent>(event) {
            set_input_replacement.set(Some(payload.text));
        }
    }).await;
});
```

Pass `replacement_text=move || input_replacement.get()` and `on_replacement_consumed=move || set_input_replacement.set(None)` into `InputBox`.

- [ ] **Step 5: Add CSS for composer actions**

In `crates/voxui-desktop/src/styles.css`, update composer grid:

```css
.composer-panel {
  grid-template-columns: minmax(0, 1fr) 116px;
}

.composer-actions {
  display: grid;
  grid-template-rows: minmax(0, 1fr) 34px;
  gap: 8px;
  min-width: 0;
}

.composer-clear-button {
  width: 100%;
  min-width: 0;
}
```

- [ ] **Step 6: Run frontend checks**

Run:

```powershell
cargo check -p voxui-desktop-ui --target wasm32-unknown-unknown
cargo check -p voxui-desktop
```

Expected: both checks pass.

- [ ] **Step 7: Commit**

```powershell
git add crates/voxui-desktop/src/tauri_api.rs crates/voxui-desktop/src/i18n.rs crates/voxui-desktop/src/app.rs crates/voxui-desktop/src/components/input_box.rs crates/voxui-desktop/src/styles.css
git commit -m "Add live input replacement UI"
```

## Task 8: Live Settings Tab UI

**Files:**
- Modify: `crates/voxui-desktop/src/components/settings_modal.rs`
- Modify: `crates/voxui-desktop/src/app.rs`
- Modify: `crates/voxui-desktop/src/styles.css`

- [ ] **Step 1: Add Live page enum and tab**

In `settings_modal.rs`, add `Live`:

```rust
pub enum SettingsPage {
    General,
    Inference,
    Audio,
    Live,
    About,
}
```

Add a tab button:

```rust
<button type="button" class:active=move || active_page() == SettingsPage::Live on:click=move |_| on_page_select(SettingsPage::Live)>{move || labels().live}</button>
```

- [ ] **Step 2: Extend SettingsModal props**

Add props:

```rust
live_snapshot: impl Fn() -> crate::tauri_api::LiveSnapshot + Send + Sync + 'static + Copy,
on_live_patch: impl Fn(crate::tauri_api::LiveConfigPatch) + Send + Sync + 'static + Copy,
on_live_connect: impl Fn() + Send + Sync + 'static + Copy,
on_live_disconnect: impl Fn() + Send + Sync + 'static + Copy,
```

- [ ] **Step 3: Add Live settings section**

Add this section in `settings_modal.rs`:

```rust
<Show when=move || active_page() == SettingsPage::Live>
    <section class="settings-section live-settings-section">
        <h3>{move || labels().live}</h3>
        <div class="settings-grid">
            <label class="settings-field settings-span-2" for="settings-live-code">
                <span>{move || labels().identity_code}</span>
                <input
                    id="settings-live-code"
                    type="text"
                    prop:value=move || live_snapshot().config.identity_code
                    on:change=move |event| on_live_patch(crate::tauri_api::LiveConfigPatch {
                        identity_code: Some(event_target_value(&event)),
                        ..crate::tauri_api::LiveConfigPatch::default()
                    })
                />
            </label>
            <label class="settings-checkbox settings-switch" for="settings-ceve-heartbeat">
                <input
                    id="settings-ceve-heartbeat"
                    type="checkbox"
                    prop:checked=move || live_snapshot().config.enable_ceve_server_heartbeat
                    on:change=move |event| on_live_patch(crate::tauri_api::LiveConfigPatch {
                        enable_ceve_server_heartbeat: Some(event_target_checked(&event)),
                        ..crate::tauri_api::LiveConfigPatch::default()
                    })
                />
                <span>{move || labels().ceve_heartbeat}</span>
            </label>
            <div class="settings-field settings-action-field">
                <span>{move || live_status_label(live_snapshot().status, labels())}</span>
                <button class="primary-button" type="button" on:click=move |_| {
                    if live_snapshot().status == crate::tauri_api::LiveStatus::Connected {
                        on_live_disconnect();
                    } else {
                        on_live_connect();
                    }
                }>
                    {move || if live_snapshot().status == crate::tauri_api::LiveStatus::Connected { labels().disconnect } else { labels().connect }}
                </button>
            </div>
        </div>
    </section>
</Show>
```

Then add compact checkbox rows for `show_danmu`, `show_gifts`, `show_superchats`, `show_guards`, `show_likes`, and `show_enters`. Add textareas for templates and a simple editable list for replacement rules and mapped names.

- [ ] **Step 4: Wire Live settings in App**

In `app.rs`, add `live_snapshot` signal, load it on startup with `get_live_state`, listen for `live_status_changed` and `live_items_changed`, and pass callbacks:

```rust
let commit_live_patch = move |patch: crate::tauri_api::LiveConfigPatch| {
    spawn_local(async move {
        if let Ok(next) = crate::tauri_api::set_live_config_patch(patch).await {
            set_live_snapshot.set(next);
        }
    });
};
```

For connect:

```rust
spawn_local(async move {
    let code = live_snapshot.get().config.identity_code;
    let _ = crate::tauri_api::connect_openblive(code).await;
});
```

- [ ] **Step 5: Run checks**

Run:

```powershell
cargo check -p voxui-desktop-ui --target wasm32-unknown-unknown
cargo check -p voxui-desktop
```

Expected: both checks pass.

- [ ] **Step 6: Commit**

```powershell
git add crates/voxui-desktop/src/components/settings_modal.rs crates/voxui-desktop/src/app.rs crates/voxui-desktop/src/styles.css
git commit -m "Add Bilibili live settings tab"
```

## Task 9: Live Monitor Window UI

**Files:**
- Create: `crates/voxui-desktop/src/components/live_monitor.rs`
- Modify: `crates/voxui-desktop/src/components/mod.rs`
- Modify: `crates/voxui-desktop/src/app.rs`
- Modify: `crates/voxui-desktop/src/styles.css`

- [ ] **Step 1: Add live monitor component**

Create `crates/voxui-desktop/src/components/live_monitor.rs`:

```rust
use leptos::prelude::*;
use wasm_bindgen::JsCast;

use crate::i18n::Labels;
use crate::tauri_api::{LiveMessageKind, LiveMonitorItem, LiveSnapshot};

#[component]
pub fn LiveMonitor(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    snapshot: impl Fn() -> LiveSnapshot + Send + Sync + 'static + Copy,
    on_send: impl Fn(String, bool) + Send + Sync + 'static + Copy,
    on_clear: impl Fn() + Send + Sync + 'static + Copy,
) -> impl IntoView {
    let feed_ref = NodeRef::<leptos::html::Div>::new();
    let near_bottom = move || {
        feed_ref.get().is_none_or(|node| {
            let scroll_gap = node.scroll_height() - node.scroll_top() - node.client_height();
            scroll_gap < 48
        })
    };

    Effect::new(move |_| {
        let count = snapshot().items.len();
        let _ = count;
        if near_bottom() {
            if let Some(node) = feed_ref.get() {
                node.scroll_to_with_x_and_y(0.0, node.scroll_height() as f64);
            }
        }
    });

    view! {
        <main class="live-monitor-shell">
            <header class="live-monitor-header">
                <strong>{move || labels().live}</strong>
                <span>{move || format!("{:?}", snapshot().status)}</span>
                <button class="icon-button" type="button" title=move || labels().clear on:click=move |_| on_clear()>"×"</button>
            </header>
            <div class="live-feed" node_ref=feed_ref>
                {move || snapshot().items.into_iter().map(|item| view! {
                    <LiveMonitorRow labels=labels item=item on_send=on_send />
                }).collect_view()}
            </div>
        </main>
    }
}

#[component]
fn LiveMonitorRow(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    item: LiveMonitorItem,
    on_send: impl Fn(String, bool) + Send + Sync + 'static + Copy,
) -> impl IntoView {
    let item_id = item.id.clone();
    let switch_id = item.id.clone();
    let is_danmu = item.kind == LiveMessageKind::Danmu;
    view! {
        <article class="live-item" class:paid=item.paid>
            <div class="live-item-meta">
                <span>{kind_label(item.kind, labels())}</span>
                <Show when=move || item.paid>
                    <span class="live-paid">{move || labels().paid}</span>
                </Show>
                <strong>{item.mapped_uname.clone()}</strong>
            </div>
            <p>{item.suggestion.clone()}</p>
            <div class="live-item-actions">
                <button class="icon-button" type="button" title=move || labels().send on:click=move |_| on_send(item_id.clone(), false)>"▶"</button>
                <Show when=move || is_danmu>
                    <button class="icon-button" type="button" title=move || labels().switch_send on:click=move |_| on_send(switch_id.clone(), true)>"⇄"</button>
                </Show>
            </div>
        </article>
    }
}
```

Replace text glyphs with CSS-stable icon text if needed; keep buttons square and labelled with `title`.

- [ ] **Step 2: Export component and route by window label**

In `components/mod.rs`:

```rust
pub mod live_monitor;
```

In `app.rs`, determine window label with the helper from `tauri_api.rs`:

```rust
let is_live_monitor_window = crate::tauri_api::current_window_label() == "live-monitor";
```

If `is_live_monitor_window` is true, render only `LiveMonitor`. Otherwise render the existing main app shell.

- [ ] **Step 3: Wire monitor actions**

In the `LiveMonitor` route in `app.rs`, pass:

```rust
on_send=move |item_id, switch| {
    spawn_local(async move {
        let mode = if switch { "switch" } else { "normal" }.to_string();
        let _ = crate::tauri_api::send_live_suggestion(item_id, mode).await;
    });
}
on_clear=move || {
    spawn_local(async move {
        let _ = crate::tauri_api::clear_live_items().await;
    });
}
```

- [ ] **Step 4: Add monitor CSS**

In `styles.css`, add:

```css
.live-monitor-shell {
  height: 100vh;
  display: grid;
  grid-template-rows: 44px 1fr;
  background: var(--app-bg);
  color: var(--text);
  overflow: hidden;
}

.live-monitor-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto 34px;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-bottom: 1px solid var(--panel-border);
  background: var(--panel-bg);
}

.live-feed {
  min-height: 0;
  overflow: auto;
  scroll-behavior: smooth;
  padding: 10px;
  display: grid;
  align-content: start;
  gap: 8px;
}

.live-item {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
  border: 1px solid var(--history-border);
  border-left: 3px solid var(--control-border);
  border-radius: 6px;
  background: var(--history-bg);
  padding: 8px;
}

.live-item.paid {
  border-left-color: var(--primary-bg);
}

.live-item-meta {
  grid-column: 1 / -1;
  display: flex;
  min-width: 0;
  gap: 6px;
  align-items: center;
  color: var(--text-muted);
  font-size: 12px;
}

.live-paid {
  color: var(--primary-text);
  background: var(--primary-bg);
  border-radius: 4px;
  padding: 1px 5px;
}

.live-item p {
  margin: 0;
  overflow-wrap: anywhere;
}

.live-item-actions {
  display: flex;
  gap: 4px;
}
```

- [ ] **Step 5: Run frontend checks**

Run:

```powershell
cargo check -p voxui-desktop-ui --target wasm32-unknown-unknown
```

Expected: check passes.

- [ ] **Step 6: Commit**

```powershell
git add crates/voxui-desktop/src/components/live_monitor.rs crates/voxui-desktop/src/components/mod.rs crates/voxui-desktop/src/app.rs crates/voxui-desktop/src/styles.css
git commit -m "Add live monitor window UI"
```

## Task 10: Final Integration Verification

**Files:**
- Modify any files from earlier tasks only to fix integration issues found by commands below.

- [ ] **Step 1: Run backend tests**

Run:

```powershell
cargo test -p voxui-desktop --test live_tests
cargo test -p voxui-desktop --test openblive_protocol_tests
cargo test -p voxui-desktop --test config_tests
cargo test -p voxui-desktop --test app_core_tests
```

Expected: all pass.

- [ ] **Step 2: Run workspace checks**

Run:

```powershell
cargo check -p voxui-desktop
cargo check -p voxui-desktop-ui --target wasm32-unknown-unknown
```

Expected: both pass.

- [ ] **Step 3: Run formatting**

Run:

```powershell
cargo fmt --all --check
```

Expected: pass. If it fails, run:

```powershell
cargo fmt --all
cargo fmt --all --check
```

Then stage formatting changes with the files from the task that caused them.

- [ ] **Step 4: Manual Tauri smoke test**

Run the desktop app:

```powershell
cargo tauri dev
```

Verify:

- Settings contains a `Live` tab.
- `身份码` / `Identity code` is editable and persists after closing settings.
- ceve heartbeat checkbox is visible and disabled by default.
- Connect with an invalid identity code reports an error and does not open the monitor.
- Clear button clears the main composer draft.
- A generated `main_input_replace` event replaces the composer draft.

- [ ] **Step 5: Real OpenLive smoke test when identity code is available**

With a valid Bilibili identity code:

- click Connect;
- confirm the monitor opens only after auth succeeds;
- confirm danmu appends to the monitor and scrolls near the bottom;
- click normal send and confirm the main input is replaced;
- click danmu switch-send and confirm replacement rules are applied;
- close the monitor and confirm backend logs show `/v2/app/end`;
- disconnect network or stop websocket and confirm the monitor closes automatically.

- [ ] **Step 6: Final commit**

If integration fixes were needed:

```powershell
git add crates/voxui-desktop
git commit -m "Finish Bilibili live monitor integration"
```

If no integration fixes were needed, do not create an empty commit.

## Self-Review Notes

- Spec coverage: tasks cover fixed signing providers/app id, ceve heartbeat setting, raw JSON-backed items, filters/templates, username mapping, danmu cleanup, replacement rules, separate monitor window, connect/open lifecycle, close/end cleanup, unexpected disconnect closure, main input replacement, and composer clear.
- Scope: all changes stay in `crates/voxui-desktop` except workspace `Cargo.lock` updates caused by backend dependencies.
- Known risk: Tauri 2 app-exit cleanup APIs and window label access may need exact API adjustment during implementation. The plan localizes those changes to `lib.rs`, `commands.rs`, and `app.rs`, and requires checks after each UI/backend task.
