use std::collections::HashSet;

use leptos::html::Div;
use leptos::prelude::*;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

use crate::components::controls::{CustomSelect, SelectOption};
use crate::i18n::Labels;
use crate::tauri_api::{LiveMessageKind, LiveMonitorItem, LiveSnapshot, LiveStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiveSendMode {
    Manual,
    AutoEnqueue,
    AutoGen,
    AutoGenSwap,
}

impl LiveSendMode {
    fn value(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::AutoEnqueue => "auto_enqueue",
            Self::AutoGen => "auto_gen",
            Self::AutoGenSwap => "auto_gen_swap",
        }
    }

    fn from_value(value: &str) -> Self {
        match value {
            "auto_enqueue" => Self::AutoEnqueue,
            "auto_gen" => Self::AutoGen,
            "auto_gen_swap" => Self::AutoGenSwap,
            _ => Self::Manual,
        }
    }

    fn label(self, labels: Labels) -> &'static str {
        match self {
            Self::Manual => labels.send_mode_manual,
            Self::AutoEnqueue => labels.send_mode_auto_enqueue,
            Self::AutoGen => labels.send_mode_auto_gen,
            Self::AutoGenSwap => labels.send_mode_auto_gen_swap,
        }
    }

    fn direct_enqueue_on_click(self) -> bool {
        matches!(self, Self::AutoEnqueue)
    }

    fn hides_item_actions(self) -> bool {
        matches!(self, Self::AutoGen | Self::AutoGenSwap)
    }
}

#[component]
pub fn LiveMonitor(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    snapshot: impl Fn() -> LiveSnapshot + Send + Sync + 'static + Copy,
    on_send: impl Fn(String, bool, bool) + Send + Sync + 'static + Copy,
    on_clear: impl Fn() + Send + Sync + 'static + Copy,
) -> impl IntoView {
    let feed_ref = NodeRef::<Div>::new();
    let (was_near_bottom, set_was_near_bottom) = signal(true);
    let (previous_item_count, set_previous_item_count) = signal(0_usize);
    let (send_mode, set_send_mode) = signal(LiveSendMode::Manual);
    let (_seen_item_ids, set_seen_item_ids) = signal(HashSet::<String>::new());
    let (status_notice, set_status_notice) = signal(None::<String>);
    let (status_notice_generation, set_status_notice_generation) = signal(0_u64);
    let (last_status_key, set_last_status_key) = signal(None::<(LiveStatus, Option<String>)>);

    Effect::new(move |_| {
        let item_count = snapshot().items.len();
        let previous_item_count = previous_item_count.get_untracked();
        if item_count == 0 {
            set_previous_item_count.set(0);
            return;
        }
        if item_count == previous_item_count {
            return;
        }

        if let Some(feed) = feed_ref.get() {
            let feed = feed.into();
            if should_scroll_after_item_change(previous_item_count, was_near_bottom.get_untracked())
            {
                scroll_to_bottom(&feed);
                set_was_near_bottom.set(is_near_bottom(&feed));
            }
        }
        set_previous_item_count.set(item_count);
    });

    Effect::new(move |_| {
        let snapshot = snapshot();
        let current_key = (snapshot.status, snapshot.status_message.clone());
        let previous_key = last_status_key.get_untracked();
        if previous_key == Some(current_key.clone()) {
            return;
        }

        set_last_status_key.set(Some(current_key));
        if previous_key.is_none() {
            return;
        }
        set_status_notice.set(Some(status_text(&snapshot, labels())));
        let generation = status_notice_generation.get_untracked().saturating_add(1);
        set_status_notice_generation.set(generation);
        schedule_status_notice_clear(set_status_notice, status_notice_generation, generation);
    });

    Effect::new(move |_| {
        let mode = send_mode.get();
        let snapshot = snapshot();
        let mut pending = Vec::new();

        set_seen_item_ids.update(|seen| {
            for item in snapshot.items.iter() {
                if seen.insert(item.id.clone()) {
                    if let Some(use_switch) = auto_generation_switch(mode, item.kind) {
                        pending.push((item.id.clone(), use_switch));
                    }
                }
            }
        });

        for (item_id, use_switch) in pending {
            on_send(item_id, use_switch, true);
        }
    });

    view! {
        <section class="live-monitor-shell" aria-label=move || labels().live>
            <header class="live-monitor-header">
                <h1>{move || labels().live}</h1>
                <span class="live-monitor-status" title=move || status_title(&snapshot(), labels())>
                    {move || status_text(&snapshot(), labels())}
                </span>
                <div class="live-monitor-header-actions">
                    <CustomSelect
                        class="live-send-mode-select"
                        aria_label=labels().send_mode
                        value=move || send_mode.get().value().to_string()
                        options=move || live_send_mode_options(labels())
                        disabled=move || false
                        on_change=move |value| {
                            set_send_mode.set(LiveSendMode::from_value(&value));
                        }
                    />
                    <button
                    class="live-monitor-button"
                    type="button"
                    title=move || labels().clear
                    aria-label=move || labels().clear
                    on:click=move |_| on_clear()
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                </button>
                </div>
            </header>
            <Show when=move || status_notice.get().is_some()>
                <div class="live-status-notice" role="status" aria-live="polite">
                    {move || status_notice.get().unwrap_or_default()}
                </div>
            </Show>

            <div
                class="live-feed"
                node_ref=feed_ref
                on:scroll=move |_| {
                    if let Some(feed) = feed_ref.get() {
                        set_was_near_bottom.set(is_near_bottom(&feed.into()));
                    }
                }
            >
                <For
                    each=move || snapshot().items
                    key=live_item_render_key
                    children=move |item| {
                        let labels = labels();
                        let kind = item.kind;
                        let item_id_for_send = item.id.clone();
                        let item_id_for_switch = item.id.clone();
                        let mapped_uname = item.mapped_uname.clone();
                        let suggestion = item.suggestion.clone();
                        let paid = item.paid;
                        let show_switch = live_item_supports_switch(kind);
                        let switch_button = show_switch.then(|| {
                            view! {
                                <button
                                    class="live-monitor-button"
                                    type="button"
                                    title=labels.swap_send
                                    aria-label=labels.swap_send
                                        on:click=move |_| {
                                            on_send(
                                                item_id_for_switch.clone(),
                                                true,
                                                send_mode.get_untracked().direct_enqueue_on_click(),
                                            )
                                        }
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                            <polyline points="17 1 21 5 17 9"></polyline>
                                            <path d="M3 11V9a4 4 0 0 1 4-4h14"></path>
                                            <polyline points="7 23 3 19 7 15"></polyline>
                                            <path d="M21 13v2a4 4 0 0 1-4 4H3"></path>
                                        </svg>
                                    </button>
                            }
                        });

                        view! {
                            <article class="live-item" class:live-item-paid=paid>
                                <div class="live-item-main">
                                    <div class="live-item-meta">
                                        <span>{kind_label(kind, labels)}</span>
                                        {paid.then(|| view! { <strong class="live-paid">{labels.paid}</strong> })}
                                        <span class="live-uname">{mapped_uname}</span>
                                    </div>
                                    <p>{suggestion}</p>
                                </div>
                                <div
                                    class="live-item-actions"
                                    class:live-item-actions-hidden=move || send_mode.get().hides_item_actions()
                                >
                                    <button
                                        class="live-monitor-button"
                                        type="button"
                                        title=labels.send
                                        aria-label=labels.send
                                        on:click=move |_| {
                                            on_send(
                                                item_id_for_send.clone(),
                                                false,
                                                send_mode.get_untracked().direct_enqueue_on_click(),
                                            )
                                        }
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                            <line x1="22" y1="2" x2="11" y2="13"></line>
                                            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                                        </svg>
                                    </button>
                                    {switch_button}
                                </div>
                            </article>
                        }
                    }
                />
            </div>
        </section>
    }
}

fn kind_label(kind: LiveMessageKind, labels: Labels) -> &'static str {
    match kind {
        LiveMessageKind::Danmu => labels.danmu,
        LiveMessageKind::Gift => labels.gift,
        LiveMessageKind::Superchat => labels.superchat,
        LiveMessageKind::Guard => labels.guard,
        LiveMessageKind::Like => labels.like,
        LiveMessageKind::Enter => labels.enter,
    }
}

fn auto_generation_switch(mode: LiveSendMode, kind: LiveMessageKind) -> Option<bool> {
    match mode {
        LiveSendMode::Manual | LiveSendMode::AutoEnqueue => None,
        LiveSendMode::AutoGen => Some(false),
        LiveSendMode::AutoGenSwap => Some(live_item_supports_switch(kind)),
    }
}

fn live_item_supports_switch(kind: LiveMessageKind) -> bool {
    matches!(kind, LiveMessageKind::Danmu | LiveMessageKind::Superchat)
}

fn live_send_mode_options(labels: Labels) -> Vec<SelectOption> {
    [
        LiveSendMode::Manual,
        LiveSendMode::AutoEnqueue,
        LiveSendMode::AutoGen,
        LiveSendMode::AutoGenSwap,
    ]
    .into_iter()
    .map(|mode| SelectOption::new(mode.value(), mode.label(labels)))
    .collect()
}

fn live_item_render_key(item: &LiveMonitorItem) -> String {
    format!(
        "{}\x1f{}\x1f{}",
        item.id, item.mapped_uname, item.suggestion
    )
}

fn status_text(snapshot: &LiveSnapshot, labels: Labels) -> String {
    match snapshot.status_message.as_deref() {
        Some(message) if !message.is_empty() => {
            format!("{}: {message}", status_label(snapshot.status, labels))
        }
        _ => status_label(snapshot.status, labels).to_string(),
    }
}

fn status_title(snapshot: &LiveSnapshot, labels: Labels) -> String {
    status_text(snapshot, labels)
}

fn status_label(status: LiveStatus, labels: Labels) -> &'static str {
    match status {
        LiveStatus::Disconnected => labels.status_disconnected,
        LiveStatus::Connecting => labels.status_connecting,
        LiveStatus::Connected => labels.status_connected,
        LiveStatus::Disconnecting => labels.status_disconnecting,
        LiveStatus::Error => labels.history_status_failed,
    }
}

fn should_scroll_after_item_change(previous_item_count: usize, was_near_bottom: bool) -> bool {
    previous_item_count == 0 || was_near_bottom
}

fn is_near_bottom(feed: &web_sys::HtmlElement) -> bool {
    distance_from_bottom(feed) <= 160
}

fn distance_from_bottom(feed: &web_sys::HtmlElement) -> i32 {
    feed.scroll_height() - feed.client_height() - feed.scroll_top()
}

fn scroll_to_bottom(element: &web_sys::HtmlElement) {
    let element = element.clone();
    if let Some(window) = web_sys::window() {
        let callback = Closure::wrap(Box::new(move || {
            element.set_scroll_top(element.scroll_height());
        }) as Box<dyn FnMut()>);
        let _ = window.request_animation_frame(callback.as_ref().unchecked_ref());
        callback.forget();
    } else {
        element.set_scroll_top(element.scroll_height());
    }
}

fn schedule_status_notice_clear(
    set_status_notice: WriteSignal<Option<String>>,
    status_notice_generation: ReadSignal<u64>,
    generation: u64,
) {
    let Some(window) = web_sys::window() else {
        set_status_notice.set(None);
        return;
    };

    let callback = Closure::wrap(Box::new(move || {
        if status_notice_generation.get_untracked() == generation {
            set_status_notice.set(None);
        }
    }) as Box<dyn FnMut()>);

    let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
        callback.as_ref().unchecked_ref(),
        1800,
    );
    callback.forget();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::i18n::{labels, UiLanguage};
    use crate::tauri_api::{LiveConfig, TemplateConfig};
    use std::collections::BTreeMap;

    #[test]
    fn maps_live_message_kinds_to_labels() {
        let labels = labels(UiLanguage::English);

        assert_eq!(kind_label(LiveMessageKind::Danmu, labels), labels.danmu);
        assert_eq!(kind_label(LiveMessageKind::Gift, labels), labels.gift);
        assert_eq!(
            kind_label(LiveMessageKind::Superchat, labels),
            labels.superchat
        );
        assert_eq!(kind_label(LiveMessageKind::Guard, labels), labels.guard);
        assert_eq!(kind_label(LiveMessageKind::Like, labels), labels.like);
        assert_eq!(kind_label(LiveMessageKind::Enter, labels), labels.enter);
    }

    #[test]
    fn renders_status_with_optional_message() {
        let labels = labels(UiLanguage::English);
        let snapshot = LiveSnapshot {
            status: LiveStatus::Connected,
            status_message: None,
            config: test_live_config(),
            items: Vec::new(),
        };
        assert_eq!(status_text(&snapshot, labels), "Connected");
        let snapshot = LiveSnapshot {
            status: LiveStatus::Error,
            status_message: Some("bad token".to_string()),
            config: test_live_config(),
            items: Vec::new(),
        };
        assert_eq!(status_text(&snapshot, labels), "Failed: bad token");
    }

    #[test]
    fn scrolls_on_first_load_or_when_feed_was_near_bottom() {
        assert!(should_scroll_after_item_change(0, false));
        assert!(should_scroll_after_item_change(4, true));
        assert!(!should_scroll_after_item_change(4, false));
    }

    #[test]
    fn does_not_read_snapshot_prop_in_component_body() {
        let source = include_str!("live_monitor.rs");
        let body_snapshot_read = ["signal(Some((snapshot", "().status"].concat();

        assert!(
            !source.contains(&body_snapshot_read),
            "LiveMonitor should read snapshot inside effects/view closures, not during component setup"
        );
    }

    #[test]
    fn live_item_key_tracks_rendered_content() {
        let source = include_str!("live_monitor.rs");
        let render_key_helper = ["live", "_item", "_render", "_key"].concat();
        let id_only_key = ["key=|item| item", ".id.clone()"].concat();

        assert!(
            source.contains(&render_key_helper),
            "live item keys should include rendered content that can change after settings patches"
        );
        assert!(
            !source.contains(&id_only_key),
            "live item keys should not use only id, because templates/name mappings can change row content"
        );
    }

    #[test]
    fn live_item_render_key_changes_with_rendered_text() {
        let mut item = LiveMonitorItem {
            id: "1".to_string(),
            kind: LiveMessageKind::Gift,
            paid: true,
            open_id: "u1".to_string(),
            uname: "Alice".to_string(),
            mapped_uname: "Alice".to_string(),
            suggestion: "old template".to_string(),
            raw_json: serde_json::Value::Null,
        };
        let first_key = live_item_render_key(&item);

        item.suggestion = "new template".to_string();
        assert_ne!(first_key, live_item_render_key(&item));

        let second_key = live_item_render_key(&item);
        item.mapped_uname = "A-chan".to_string();
        assert_ne!(second_key, live_item_render_key(&item));
    }

    #[test]
    fn live_send_mode_values_round_trip() {
        for mode in [
            LiveSendMode::Manual,
            LiveSendMode::AutoEnqueue,
            LiveSendMode::AutoGen,
            LiveSendMode::AutoGenSwap,
        ] {
            assert_eq!(LiveSendMode::from_value(mode.value()), mode);
        }
        assert_eq!(LiveSendMode::from_value("unknown"), LiveSendMode::Manual);
    }

    #[test]
    fn auto_generation_modes_choose_expected_send_action() {
        assert_eq!(
            auto_generation_switch(LiveSendMode::Manual, LiveMessageKind::Danmu),
            None
        );
        assert_eq!(
            auto_generation_switch(LiveSendMode::AutoEnqueue, LiveMessageKind::Danmu),
            None
        );
        assert_eq!(
            auto_generation_switch(LiveSendMode::AutoGen, LiveMessageKind::Danmu),
            Some(false)
        );
        assert_eq!(
            auto_generation_switch(LiveSendMode::AutoGenSwap, LiveMessageKind::Danmu),
            Some(true)
        );
        assert_eq!(
            auto_generation_switch(LiveSendMode::AutoGenSwap, LiveMessageKind::Gift),
            Some(false)
        );
    }

    #[test]
    fn monitor_uses_send_mode_dropdown_instead_of_auto_send_checkbox() {
        let source = include_str!("live_monitor.rs");
        assert!(
            source.contains("<CustomSelect"),
            "Monitor should expose send modes through the shared CustomSelect"
        );
        assert!(
            !source.contains("type=\"checkbox\""),
            "Monitor send mode should not render the old auto-send checkbox"
        );
    }

    #[test]
    fn monitor_buttons_use_svg_icons() {
        let source = include_str!("live_monitor.rs");
        assert!(
            source.contains("<svg"),
            "Monitor buttons should use SVG icons"
        );
    }

    fn test_live_config() -> LiveConfig {
        LiveConfig {
            identity_code: String::new(),
            enable_ceve_server_heartbeat: false,
            show_danmu: true,
            show_gifts: true,
            show_superchats: true,
            show_guards: true,
            show_likes: true,
            show_enters: true,
            templates: TemplateConfig {
                danmu: String::new(),
                gift_zh: String::new(),
                gift_en: String::new(),
                superchat_zh: String::new(),
                superchat_en: String::new(),
                guard_zh: String::new(),
                guard_en: String::new(),
                like_zh: String::new(),
                like_en: String::new(),
                enter_zh: String::new(),
                enter_en: String::new(),
            },
            replacement_rules: Vec::new(),
            mapped_unames: BTreeMap::new(),
            original_unames: BTreeMap::new(),
        }
    }
}
