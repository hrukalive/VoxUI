use leptos::html::Div;
use leptos::prelude::*;

use crate::i18n::Labels;
use crate::tauri_api::{LiveMessageKind, LiveSnapshot, LiveStatus};

#[component]
pub fn LiveMonitor(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    snapshot: impl Fn() -> LiveSnapshot + Send + Sync + 'static + Copy,
    on_send: impl Fn(String, bool) + Send + Sync + 'static + Copy,
    on_clear: impl Fn() + Send + Sync + 'static + Copy,
) -> impl IntoView {
    let feed_ref = NodeRef::<Div>::new();
    let (was_near_bottom, set_was_near_bottom) = signal(true);
    let (previous_item_count, set_previous_item_count) = signal(0_usize);

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

    view! {
        <section class="live-monitor-shell" aria-label=move || labels().live>
            <header class="live-monitor-header">
                <h1>{move || labels().live}</h1>
                <span class="live-monitor-status" title=move || status_title(snapshot())>
                    {move || status_text(snapshot())}
                </span>
                <button
                    class="live-monitor-button"
                    type="button"
                    title=move || labels().clear
                    aria-label=move || labels().clear
                    on:click=move |_| on_clear()
                >
                    {move || labels().clear}
                </button>
            </header>

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
                    key=|item| item.id.clone()
                    children=move |item| {
                        let labels = labels();
                        let kind = item.kind;
                        let item_id_for_send = item.id.clone();
                        let item_id_for_switch = item.id.clone();
                        let mapped_uname = item.mapped_uname.clone();
                        let suggestion = item.suggestion.clone();
                        let paid = item.paid;
                        let show_switch = matches!(kind, LiveMessageKind::Danmu);
                        let switch_button = show_switch.then(|| {
                            view! {
                                <button
                                    class="live-monitor-button"
                                    type="button"
                                    title=labels.switch_send
                                    aria-label=labels.switch_send
                                    on:click=move |_| on_send(item_id_for_switch.clone(), true)
                                >
                                    {compact_switch_label(labels)}
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
                                    <button
                                        class="live-monitor-button"
                                        type="button"
                                        title=labels.send
                                        aria-label=labels.send
                                        on:click=move |_| on_send(item_id_for_send.clone(), false)
                                    >
                                        {compact_send_label(labels)}
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

fn compact_send_label(labels: Labels) -> &'static str {
    if labels.send == "发送" {
        "发送"
    } else {
        "Send"
    }
}

fn compact_switch_label(labels: Labels) -> &'static str {
    if labels.switch_send == "人称替换" {
        "替换"
    } else {
        "Swap"
    }
}

fn status_text(snapshot: LiveSnapshot) -> String {
    status_text_parts(snapshot.status, snapshot.status_message.as_deref())
}

fn status_title(snapshot: LiveSnapshot) -> String {
    status_text(snapshot)
}

fn status_text_parts(status: LiveStatus, message: Option<&str>) -> String {
    match message {
        Some(message) if !message.is_empty() => format!("{}: {message}", status_label(status)),
        _ => status_label(status).to_string(),
    }
}

fn status_label(status: LiveStatus) -> &'static str {
    match status {
        LiveStatus::Disconnected => "Disconnected",
        LiveStatus::Connecting => "Connecting",
        LiveStatus::Connected => "Connected",
        LiveStatus::Disconnecting => "Disconnecting",
        LiveStatus::Error => "Error",
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

fn scroll_to_bottom(feed: &web_sys::HtmlElement) {
    feed.set_scroll_top(feed.scroll_height());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::i18n::{labels, UiLanguage};

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
        assert_eq!(status_text_parts(LiveStatus::Connected, None), "Connected");
        assert_eq!(
            status_text_parts(LiveStatus::Error, Some("bad token")),
            "Error: bad token"
        );
    }

    #[test]
    fn scrolls_on_first_load_or_when_feed_was_near_bottom() {
        assert!(should_scroll_after_item_change(0, false));
        assert!(should_scroll_after_item_change(4, true));
        assert!(!should_scroll_after_item_change(4, false));
    }

    #[test]
    fn compact_action_labels_fit_fixed_monitor_buttons() {
        let english = labels(UiLanguage::English);
        assert_eq!(compact_send_label(english), "Send");
        assert_eq!(compact_switch_label(english), "Swap");

        let chinese = labels(UiLanguage::Chinese);
        assert_eq!(compact_send_label(chinese), "发送");
        assert_eq!(compact_switch_label(chinese), "替换");
    }
}
