use std::collections::{BTreeMap, HashSet};

use leptos::html::Div;
use leptos::prelude::*;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

use crate::components::controls::{CustomSelect, SelectOption};
use crate::i18n::Labels;
use crate::tauri_api::{
    AutoGenMode, LiveConfig, LiveConfigPatch, LiveMessageKind, LiveMonitorItem, LiveSnapshot,
    LiveStatus, SendMode,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct MappedUnameDraft {
    open_id: String,
    uname: String,
    value: String,
}

#[component]
pub fn LiveMonitor(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    snapshot: impl Fn() -> LiveSnapshot + Send + Sync + 'static + Copy,
    on_live_patch: impl Fn(LiveConfigPatch) + Send + Sync + 'static + Copy,
    on_send: impl Fn(String, bool, bool) + Send + Sync + 'static + Copy,
    on_clear: impl Fn() + Send + Sync + 'static + Copy,
) -> impl IntoView {
    let feed_ref = NodeRef::<Div>::new();
    let (was_near_bottom, set_was_near_bottom) = signal(true);
    let (previous_item_count, set_previous_item_count) = signal(0_usize);
    let (_seen_item_ids, set_seen_item_ids) = signal(HashSet::<String>::new());
    let (status_notice, set_status_notice) = signal(None::<String>);
    let (status_notice_generation, set_status_notice_generation) = signal(0_u64);
    let (last_status_key, set_last_status_key) = signal(None::<(LiveStatus, Option<String>)>);
    let (mapped_uname_draft, set_mapped_uname_draft) = signal(None::<MappedUnameDraft>);

    let open_mapped_uname_modal = move |item: LiveMonitorItem| {
        let initial_value = mapped_uname_initial_value(
            &snapshot().config.mapped_unames,
            &item.open_id,
            &item.uname,
        );
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
        let mut original_unames = snapshot().config.original_unames.clone();
        mapped_unames.insert(draft.open_id.clone(), draft.value);
        original_unames.insert(draft.open_id, draft.uname);
        on_live_patch(LiveConfigPatch {
            mapped_unames: Some(mapped_unames),
            original_unames: Some(original_unames),
            ..LiveConfigPatch::default()
        });
        set_mapped_uname_draft.set(None);
    };

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
        let snapshot = snapshot();
        let mode = snapshot.config.auto_gen_mode;
        let mut pending = Vec::new();

        set_seen_item_ids.update(|seen| {
            for item in snapshot.items.iter() {
                if seen.insert(item.id.clone()) {
                    if let Some(use_switch) =
                        auto_generation_switch(mode, item.kind, &snapshot.config)
                    {
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
                        value=move || snapshot().config.send_mode.value().to_string()
                        options=move || live_send_mode_options(labels())
                        disabled=move || false
                        on_change=move |value| {
                            on_live_patch(LiveConfigPatch {
                                send_mode: Some(SendMode::from_value(&value)),
                                ..LiveConfigPatch::default()
                            });
                        }
                    />
                    <CustomSelect
                        class="live-auto-gen-mode-select"
                        aria_label=labels().auto_gen_mode
                        value=move || snapshot().config.auto_gen_mode.value().to_string()
                        options=move || live_auto_gen_mode_options(labels())
                        disabled=move || false
                        on_change=move |value| {
                            on_live_patch(LiveConfigPatch {
                                auto_gen_mode: Some(AutoGenMode::from_value(&value)),
                                ..LiveConfigPatch::default()
                            });
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
                                                snapshot()
                                                    .config
                                                    .send_mode
                                                    .direct_enqueue_on_click(),
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
                                            class="live-monitor-button"
                                            type="button"
                                            title=labels.send
                                            aria-label=labels.send
                                            on:click=move |_| {
                                                on_send(
                                                    item_id_for_send.clone(),
                                                    false,
                                                    snapshot()
                                                        .config
                                                        .send_mode
                                                        .direct_enqueue_on_click(),
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
                                </div>
                            </article>
                        }
                    }
                />
            </div>
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

fn auto_generation_switch(
    mode: AutoGenMode,
    kind: LiveMessageKind,
    config: &LiveConfig,
) -> Option<bool> {
    if !live_item_auto_gen_enabled(kind, config) {
        return None;
    }

    match mode {
        AutoGenMode::None => None,
        AutoGenMode::Normal => Some(false),
        AutoGenMode::Replacement => Some(live_item_supports_switch(kind)),
    }
}

fn live_item_supports_switch(kind: LiveMessageKind) -> bool {
    matches!(kind, LiveMessageKind::Danmu | LiveMessageKind::Superchat)
}

fn live_item_auto_gen_enabled(kind: LiveMessageKind, config: &LiveConfig) -> bool {
    match kind {
        LiveMessageKind::Danmu => config.auto_gen_danmu,
        LiveMessageKind::Gift => config.auto_gen_gifts,
        LiveMessageKind::Superchat => config.auto_gen_superchats,
        LiveMessageKind::Guard => config.auto_gen_guards,
        LiveMessageKind::Like => config.auto_gen_likes,
        LiveMessageKind::Enter => config.auto_gen_enters,
    }
}

fn live_send_mode_options(labels: Labels) -> Vec<SelectOption> {
    [SendMode::Manual, SendMode::AutoEnqueue]
        .into_iter()
        .map(|mode| SelectOption::new(mode.value(), mode.label(labels)))
        .collect()
}

fn live_auto_gen_mode_options(labels: Labels) -> Vec<SelectOption> {
    [
        AutoGenMode::None,
        AutoGenMode::Normal,
        AutoGenMode::Replacement,
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
        mapped_unames.insert("u3".to_string(), "C-chan".to_string());

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
        assert!(mapped_uname_needs_attention(
            &mapped_unames,
            &original_unames,
            "u3",
            "Charlie"
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

        original_unames.insert("u1".to_string(), "AliceNew".to_string());
        assert_eq!(
            mapped_uname_button_class(&mapped_unames, &original_unames, "u1", "AliceNew"),
            "live-monitor-button live-map-button"
        );
    }

    #[test]
    fn live_send_mode_values_round_trip() {
        for mode in [SendMode::Manual, SendMode::AutoEnqueue] {
            assert_eq!(SendMode::from_value(mode.value()), mode);
        }
        assert_eq!(SendMode::from_value("unknown"), SendMode::Manual);
        assert!(!SendMode::Manual.direct_enqueue_on_click());
        assert!(SendMode::AutoEnqueue.direct_enqueue_on_click());
    }

    #[test]
    fn live_auto_gen_mode_values_round_trip() {
        for mode in [
            AutoGenMode::None,
            AutoGenMode::Normal,
            AutoGenMode::Replacement,
        ] {
            assert_eq!(AutoGenMode::from_value(mode.value()), mode);
        }
        assert_eq!(AutoGenMode::from_value("unknown"), AutoGenMode::None);
    }

    #[test]
    fn auto_generation_modes_choose_expected_send_action() {
        let mut config = test_live_config();
        config.auto_gen_danmu = true;

        assert_eq!(
            auto_generation_switch(AutoGenMode::None, LiveMessageKind::Danmu, &config),
            None
        );
        assert_eq!(
            auto_generation_switch(AutoGenMode::Normal, LiveMessageKind::Danmu, &config),
            Some(false)
        );
        assert_eq!(
            auto_generation_switch(AutoGenMode::Replacement, LiveMessageKind::Danmu, &config),
            Some(true)
        );
        assert_eq!(
            auto_generation_switch(AutoGenMode::Replacement, LiveMessageKind::Gift, &config),
            Some(false)
        );

        config.auto_gen_danmu = false;
        assert_eq!(
            auto_generation_switch(AutoGenMode::Normal, LiveMessageKind::Danmu, &config),
            None
        );

        config.auto_gen_gifts = false;
        assert_eq!(
            auto_generation_switch(AutoGenMode::Replacement, LiveMessageKind::Gift, &config),
            None
        );
    }

    #[test]
    fn monitor_uses_mode_dropdowns_instead_of_auto_send_checkbox() {
        let source = include_str!("live_monitor.rs");
        assert!(
            source.contains("<CustomSelect"),
            "Monitor should expose modes through the shared CustomSelect"
        );
        assert!(
            source.contains("live-send-mode-select"),
            "Monitor should expose the click send mode select"
        );
        assert!(
            source.contains("live-auto-gen-mode-select"),
            "Monitor should expose the auto generation mode select"
        );
        assert!(
            !source.contains("type=\"checkbox\""),
            "Monitor modes should not render the old auto-send checkbox"
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

    #[test]
    fn monitor_renders_mapped_uname_modal_and_button() {
        let source = include_str!("live_monitor.rs").replace("\r\n", "\n");
        let row_button_click = [
            "on:click=move |_| ",
            "open",
            "_mapped",
            "_uname",
            "_modal(",
            "item",
            "_for",
            "_name",
            "_edit.clone()",
            ")",
        ]
        .concat();
        let modal_initial_value = [
            "mapped",
            "_uname",
            "_initial",
            "_value(\n",
            "            &snapshot().config.mapped_unames,\n",
            "            &item.open_id,\n",
            "            &item.uname,\n",
            "        )",
        ]
        .concat();
        let modal_save_click = ["on:click=move |_| ", "save", "_mapped", "_uname()"].concat();
        let live_patch_write = ["mapped", "_unames: Some(", "mapped", "_unames)"].concat();
        let original_name_patch_write =
            ["original", "_unames: Some(", "original", "_unames)"].concat();
        let row_button_class_helper = ["mapped", "_uname", "_button", "_class", "("].concat();

        assert!(
            source.contains(&row_button_click),
            "Monitor should wire the row mapped username button to the modal opener"
        );
        assert!(
            source.contains(&modal_initial_value),
            "Monitor should initialize modal input from mapping config or current uname"
        );
        assert!(
            source.contains(&modal_save_click),
            "Monitor should wire the modal save button to the save closure"
        );
        assert!(
            source.contains(&live_patch_write),
            "Monitor should save mapped usernames through the live patch pipeline"
        );
        assert!(
            source.contains(&original_name_patch_write),
            "Monitor should acknowledge the current uname so row button styling can update"
        );
        assert!(
            source.contains(&row_button_class_helper),
            "Monitor should derive mapped username button styling from mapping state"
        );
    }

    #[test]
    fn mapped_uname_modal_styles_are_present() {
        let styles = include_str!("../styles.css");
        let class_selector = |name: &[&str]| [".", &name.concat()].concat();
        let selectors = [
            class_selector(&["live", "-map", "-button"]),
            [
                class_selector(&["live", "-map", "-button"]),
                class_selector(&["primary", "-button"]),
            ]
            .concat(),
            class_selector(&["live", "-send", "-actions"]),
            class_selector(&["mapped", "-uname", "-modal"]),
            class_selector(&["mapped", "-uname", "-form"]),
            class_selector(&["mapped", "-uname", "-field"]),
            [
                class_selector(&["mapped", "-uname", "-field"]),
                " input".to_string(),
            ]
            .concat(),
            class_selector(&["mapped", "-uname", "-actions"]),
        ];

        for selector in selectors {
            assert!(
                styles.contains(&selector),
                "Expected styles to contain selector {selector}"
            );
        }
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
            send_mode: SendMode::Manual,
            auto_gen_mode: AutoGenMode::None,
            auto_gen_danmu: false,
            auto_gen_gifts: true,
            auto_gen_superchats: true,
            auto_gen_guards: true,
            auto_gen_likes: false,
            auto_gen_enters: true,
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
