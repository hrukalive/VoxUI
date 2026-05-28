use serde_json::Value;

use crate::types::{LiveConfig, LiveMessageKind, LiveMonitorItemDto, LiveSnapshot, LiveStatus};

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

#[derive(Debug, Clone, PartialEq)]
struct LiveItem {
    id: String,
    event: LiveEvent,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LiveState {
    status: LiveStatus,
    status_message: Option<String>,
    items: Vec<LiveItem>,
    next_item_id: u64,
}

impl Default for LiveState {
    fn default() -> Self {
        Self {
            status: LiveStatus::Disconnected,
            status_message: None,
            items: Vec::new(),
            next_item_id: 1,
        }
    }
}

impl LiveState {
    pub fn add_event(&mut self, event: LiveEvent) -> String {
        let id = format!("live-{}", self.next_item_id);
        self.next_item_id += 1;
        self.items.push(LiveItem {
            id: id.clone(),
            event,
        });
        id
    }

    pub fn clear_items(&mut self) {
        self.items.clear();
    }

    pub fn status(&self) -> LiveStatus {
        self.status
    }

    pub fn set_status(&mut self, status: LiveStatus, status_message: Option<String>) {
        self.status = status;
        self.status_message = status_message;
    }

    pub fn snapshot(&self, config: &LiveConfig, language: LiveLanguage) -> LiveSnapshot {
        LiveSnapshot {
            status: self.status,
            status_message: self.status_message.clone(),
            config: config.clone(),
            items: self
                .items
                .iter()
                .filter_map(|item| {
                    self.dto_for_item(item, config, language, SuggestionMode::Normal)
                })
                .collect(),
        }
    }

    pub fn suggestion_for_item(
        &self,
        item_id: &str,
        config: &LiveConfig,
        language: LiveLanguage,
        mode: SuggestionMode,
    ) -> Option<String> {
        self.items
            .iter()
            .find(|item| item.id == item_id)
            .and_then(|item| render_suggestion(&item.event, config, language, mode))
    }

    fn dto_for_item(
        &self,
        item: &LiveItem,
        config: &LiveConfig,
        language: LiveLanguage,
        mode: SuggestionMode,
    ) -> Option<LiveMonitorItemDto> {
        let suggestion = render_suggestion(&item.event, config, language, mode)?;
        let mapped_uname = config
            .mapped_unames
            .get(&item.event.open_id)
            .cloned()
            .unwrap_or_else(|| item.event.uname.clone());

        Some(LiveMonitorItemDto {
            id: item.id.clone(),
            kind: item.event.kind,
            paid: item.event.kind.is_paid(),
            open_id: item.event.open_id.clone(),
            uname: item.event.uname.clone(),
            mapped_uname,
            suggestion,
            raw_json: item.event.raw.clone(),
        })
    }
}

pub fn parse_live_event(raw: Value) -> anyhow::Result<Option<LiveEvent>> {
    let Some(command) = raw.get("cmd").and_then(Value::as_str) else {
        return Ok(None);
    };
    let data = raw.get("data").unwrap_or(&Value::Null);

    let event = match command {
        "LIVE_OPEN_PLATFORM_DM" => {
            if data.get("dm_type").and_then(Value::as_u64) == Some(1) {
                return Ok(None);
            }

            LiveEvent {
                kind: LiveMessageKind::Danmu,
                raw: raw.clone(),
                open_id: string_at(data, &["open_id"]),
                uname: string_at(data, &["uname"]),
                msg: Some(string_at(data, &["msg"])),
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
                raw: raw.clone(),
                open_id: string_at(data, &["open_id"]),
                uname: string_at(data, &["uname"]),
                msg: None,
                gift_name: Some(string_at(data, &["gift_name"])),
                gift_num: data.get("gift_num").and_then(Value::as_u64),
                superchat_message: None,
                guard_label: None,
            }
        }
        "LIVE_OPEN_PLATFORM_SUPER_CHAT" => LiveEvent {
            kind: LiveMessageKind::Superchat,
            raw: raw.clone(),
            open_id: string_at(data, &["open_id"]),
            uname: string_at(data, &["uname"]),
            msg: None,
            gift_name: None,
            gift_num: None,
            superchat_message: Some(string_at(data, &["message"])),
            guard_label: None,
        },
        "LIVE_OPEN_PLATFORM_GUARD" => {
            let user_info = data.get("user_info").unwrap_or(&Value::Null);

            LiveEvent {
                kind: LiveMessageKind::Guard,
                raw: raw.clone(),
                open_id: string_at(user_info, &["open_id"]),
                uname: string_at(user_info, &["uname"]),
                msg: None,
                gift_name: None,
                gift_num: None,
                superchat_message: None,
                guard_label: Some(guard_label(data.get("guard_level").and_then(Value::as_u64))),
            }
        }
        "LIVE_OPEN_PLATFORM_LIKE" => LiveEvent {
            kind: LiveMessageKind::Like,
            raw: raw.clone(),
            open_id: string_at(data, &["open_id"]),
            uname: string_at(data, &["uname"]),
            msg: None,
            gift_name: None,
            gift_num: None,
            superchat_message: None,
            guard_label: None,
        },
        "LIVE_OPEN_PLATFORM_LIVE_ROOM_ENTER" => LiveEvent {
            kind: LiveMessageKind::Enter,
            raw: raw.clone(),
            open_id: string_at(data, &["open_id"]),
            uname: string_at(data, &["uname"]),
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

    let mapped_uname = config
        .mapped_unames
        .get(&event.open_id)
        .map(String::as_str)
        .unwrap_or(&event.uname);

    let rendered = if event.kind == LiveMessageKind::Danmu {
        let period = match language {
            LiveLanguage::Chinese => '。',
            LiveLanguage::English => '.',
        };
        let cleaned = clean_danmu(event.msg.as_deref().unwrap_or_default(), period);
        if cleaned.is_empty() {
            return None;
        }
        cleaned
    } else {
        render_template(choose(event.kind, language, config), event, mapped_uname)
    };

    match mode {
        SuggestionMode::Normal => Some(rendered),
        SuggestionMode::Switch => Some(switch_text(&rendered, config)),
    }
}

pub fn switch_text(text: &str, config: &LiveConfig) -> String {
    config
        .replacement_rules
        .iter()
        .filter(|rule| rule.enabled && !rule.from.is_empty())
        .fold(text.to_string(), |current, rule| {
            current.replace(&rule.from, &rule.to)
        })
}

fn string_at(value: &Value, path: &[&str]) -> String {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn guard_label(level: Option<u64>) -> String {
    match level {
        Some(1) => "总督",
        Some(2) => "提督",
        Some(3) => "舰长",
        _ => "舰长",
    }
    .to_string()
}

fn kind_enabled(kind: LiveMessageKind, config: &LiveConfig) -> bool {
    match kind {
        LiveMessageKind::Danmu => config.show_danmu,
        LiveMessageKind::Gift => config.show_gifts,
        LiveMessageKind::Superchat => config.show_superchats,
        LiveMessageKind::Guard => config.show_guards,
        LiveMessageKind::Like => config.show_likes,
        LiveMessageKind::Enter => config.show_enters,
    }
}

fn choose(kind: LiveMessageKind, language: LiveLanguage, config: &LiveConfig) -> &str {
    match (kind, language) {
        (LiveMessageKind::Danmu, _) => &config.templates.danmu,
        (LiveMessageKind::Gift, LiveLanguage::Chinese) => &config.templates.gift_zh,
        (LiveMessageKind::Gift, LiveLanguage::English) => &config.templates.gift_en,
        (LiveMessageKind::Superchat, LiveLanguage::Chinese) => &config.templates.superchat_zh,
        (LiveMessageKind::Superchat, LiveLanguage::English) => &config.templates.superchat_en,
        (LiveMessageKind::Guard, LiveLanguage::Chinese) => &config.templates.guard_zh,
        (LiveMessageKind::Guard, LiveLanguage::English) => &config.templates.guard_en,
        (LiveMessageKind::Like, LiveLanguage::Chinese) => &config.templates.like_zh,
        (LiveMessageKind::Like, LiveLanguage::English) => &config.templates.like_en,
        (LiveMessageKind::Enter, LiveLanguage::Chinese) => &config.templates.enter_zh,
        (LiveMessageKind::Enter, LiveLanguage::English) => &config.templates.enter_en,
    }
}

fn render_template(template: &str, event: &LiveEvent, mapped_uname: &str) -> String {
    let mut rendered = String::new();
    let mut remaining = template;

    while let Some(open) = remaining.find('{') {
        rendered.push_str(&remaining[..open]);
        let after_open = &remaining[open + 1..];

        let Some(close) = after_open.find('}') else {
            rendered.push_str(&remaining[open..]);
            return rendered;
        };

        let placeholder = &after_open[..close];
        match placeholder {
            "mapped_uname" => rendered.push_str(mapped_uname),
            "uname" => rendered.push_str(&event.uname),
            "msg" => rendered.push_str(event.msg.as_deref().unwrap_or_default()),
            "gift_name" => rendered.push_str(event.gift_name.as_deref().unwrap_or_default()),
            "gift_num" => {
                if let Some(gift_num) = event.gift_num {
                    rendered.push_str(&gift_num.to_string());
                }
            }
            "message" => rendered.push_str(event.superchat_message.as_deref().unwrap_or_default()),
            "guard_label" => rendered.push_str(event.guard_label.as_deref().unwrap_or_default()),
            _ => {
                rendered.push('{');
                rendered.push_str(placeholder);
                rendered.push('}');
            }
        }

        remaining = &after_open[close + 1..];
    }

    rendered.push_str(remaining);
    rendered
}

fn clean_danmu(text: &str, period: char) -> String {
    let mut cleaned = String::new();
    let mut in_brackets = false;
    let mut last_was_period = false;

    for ch in text.chars() {
        match ch {
            '[' => in_brackets = true,
            ']' if in_brackets => in_brackets = false,
            _ if in_brackets => {}
            _ if ch.is_whitespace() => {
                if !cleaned.is_empty() && !last_was_period {
                    cleaned.push(period);
                    last_was_period = true;
                }
            }
            _ => {
                cleaned.push(ch);
                last_was_period = false;
            }
        }
    }

    cleaned.trim_matches(period).to_string()
}
