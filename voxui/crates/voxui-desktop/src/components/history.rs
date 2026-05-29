use std::cell::{Cell, RefCell};
use std::rc::Rc;

use leptos::html::Section;
use leptos::prelude::*;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

use crate::i18n::Labels;
use crate::tauri_api::{HistoryItem, HistoryStatus};

#[component]
pub fn HistoryList(
    labels: impl Fn() -> Labels + Send + Sync + 'static + Copy,
    items: impl Fn() -> Vec<HistoryItem> + Send + Sync + 'static + Copy,
    on_play: impl Fn(String) + Send + Sync + 'static + Copy,
    on_regenerate: impl Fn(String) + Send + Sync + 'static + Copy,
    on_cancel: impl Fn(String) + Send + Sync + 'static + Copy,
) -> impl IntoView {
    let list_ref = NodeRef::<Section>::new();
    let (previous_item_ids, set_previous_item_ids) = signal(Vec::<String>::new());

    Effect::new(move |_| {
        let current_items = items();
        let current_item_ids = history_item_ids(&current_items);
        let should_scroll =
            has_appended_history_item(&previous_item_ids.get_untracked(), &current_item_ids);
        set_previous_item_ids.set(current_item_ids);

        if should_scroll {
            if let Some(element) = list_ref.get() {
                scroll_to_bottom(element.into());
            }
        }
    });

    let items_for_view = items.clone();
    view! {
        <section class="history-panel" node_ref=list_ref>
            <div class="history-list">
                {move || {
                    let current = items();
                    if current.is_empty() {
                        view! {
                            <p class="empty-history">{move || labels().history_empty}</p>
                        }.into_any()
                    } else {
                        view! {
                            <For
                                each=move || items_for_view()
                                key=|item| item.id.clone()
                                children=move |item| {
                                    let id_for_play = item.id.clone();
                                    let id_for_regenerate = item.id.clone();
                                    let id_for_cancel = item.id.clone();
                                    let row_labels = labels();
                                    let progress_label = progress_text(row_labels, &item);
                                    let can_cancel = matches!(item.status, HistoryStatus::Queued | HistoryStatus::Generating);
                                    let can_play = item.has_audio && matches!(item.status, HistoryStatus::Ready | HistoryStatus::Playing);
                                    let play_label = if matches!(item.status, HistoryStatus::Playing) { row_labels.stop } else { row_labels.play };

                                    view! {
                                        <article class="history-item">
                                            <div class="history-main">
                                                <p class="history-text">{item.text}</p>
                                                <div class="history-meta">
                                                    <span>{status_label(row_labels, item.status)}</span>
                                                    {progress_label
                                                        .map(|label| view! { <span>{label}</span> })}
                                                </div>
                                                {item.error
                                                    .map(|error| view! { <p class="history-error">{error}</p> })}
                                            </div>
                                            <div class="history-actions">
                                                <button
                                                    class="history-action-button"
                                                    title={row_labels.cancel}
                                                    aria-label={row_labels.cancel}
                                                    disabled={!can_cancel}
                                                    on:click=move |_| on_cancel(id_for_cancel.clone())
                                                >
                                                    {cancel_icon()}
                                                </button>
                                                <button
                                                    class="history-action-button"
                                                    title={play_label}
                                                    aria-label={play_label}
                                                    disabled={!can_play}
                                                    on:click=move |_| on_play(id_for_play.clone())
                                                >
                                                    {if matches!(item.status, HistoryStatus::Playing) {
                                                        stop_icon().into_any()
                                                    } else {
                                                        play_icon().into_any()
                                                    }}
                                                </button>
                                                <button
                                                    class="history-action-button"
                                                    title={row_labels.regenerate}
                                                    aria-label={row_labels.regenerate}
                                                    on:click=move |_| on_regenerate(id_for_regenerate.clone())
                                                >
                                                    {regenerate_icon()}
                                                </button>
                                            </div>
                                        </article>
                                    }
                                }
                            />
                        }.into_any()
                    }
                }}
            </div>
        </section>
    }
    .into_any()
}

fn history_item_ids(items: &[HistoryItem]) -> Vec<String> {
    items.iter().map(|item| item.id.clone()).collect()
}

fn has_appended_history_item(previous_ids: &[String], current_ids: &[String]) -> bool {
    current_ids.len() > previous_ids.len() && current_ids.starts_with(previous_ids)
}

fn status_label(labels: Labels, status: HistoryStatus) -> &'static str {
    match status {
        HistoryStatus::Queued => labels.history_status_queued,
        HistoryStatus::Generating => labels.history_status_generating,
        HistoryStatus::Ready => labels.history_status_ready,
        HistoryStatus::Playing => labels.history_status_playing,
        HistoryStatus::Failed => labels.history_status_failed,
        HistoryStatus::Canceled => labels.history_status_canceled,
    }
}

fn progress_text(labels: Labels, item: &HistoryItem) -> Option<String> {
    (item.progress_total > 0).then(|| {
        format!(
            "{} {}/{}",
            labels.progress, item.progress_current, item.progress_total
        )
    })
}

fn cancel_icon() -> impl IntoView {
    view! {
        <svg class="history-action-icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M18 6 6 18M6 6l12 12" />
        </svg>
    }
}

fn play_icon() -> impl IntoView {
    view! {
        <svg class="history-action-icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="m8 5 11 7-11 7z" />
        </svg>
    }
}

fn stop_icon() -> impl IntoView {
    view! {
        <svg class="history-action-icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M7 7h10v10H7z" />
        </svg>
    }
}

fn regenerate_icon() -> impl IntoView {
    view! {
        <svg class="history-action-icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M20 12a8 8 0 1 1-2.34-5.66" />
            <path d="M20 4v6h-6" />
        </svg>
    }
}

fn scroll_to_bottom(element: web_sys::HtmlElement) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let element = Rc::new(element);
    let callback = Closure::wrap(Box::new(move || {
        let target = element.scroll_height();
        if prefers_reduced_motion() {
            element.set_scroll_top(target);
            return;
        }
        let start = element.scroll_top();
        let distance = target - start;
        if distance <= 0 {
            return;
        }
        animate_scroll(Rc::clone(&element), start, distance);
    }) as Box<dyn FnMut()>);
    let _ = window.request_animation_frame(callback.as_ref().unchecked_ref());
    callback.forget();
}

fn prefers_reduced_motion() -> bool {
    web_sys::window()
        .and_then(|window| window.match_media("(prefers-reduced-motion: reduce)").ok())
        .flatten()
        .is_some_and(|query| query.matches())
}

fn animate_scroll(element: Rc<web_sys::HtmlElement>, start: i32, distance: i32) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let start_time = Rc::new(Cell::new(None::<f64>));
    let frame = Rc::new(RefCell::new(None::<Closure<dyn FnMut(f64)>>));
    let frame_for_step = Rc::clone(&frame);
    let window_for_step = window.clone();

    *frame.borrow_mut() = Some(Closure::wrap(Box::new(move |timestamp: f64| {
        let first_timestamp = start_time.get().unwrap_or(timestamp);
        start_time.set(Some(first_timestamp));
        let progress = ((timestamp - first_timestamp) / 180.0).clamp(0.0, 1.0);
        let eased = 1.0 - (1.0 - progress).powi(3);
        element.set_scroll_top(start + ((distance as f64) * eased).round() as i32);
        if progress < 1.0 {
            if let Some(callback) = frame_for_step.borrow().as_ref() {
                let _ = window_for_step.request_animation_frame(callback.as_ref().unchecked_ref());
            }
        } else {
            frame_for_step.borrow_mut().take();
        }
    }) as Box<dyn FnMut(f64)>));

    if let Some(callback) = frame.borrow().as_ref() {
        let _ = window.request_animation_frame(callback.as_ref().unchecked_ref());
    };
}

#[cfg(test)]
mod tests {
    use super::has_appended_history_item;

    fn css_block<'a>(css: &'a str, selector: &str) -> &'a str {
        let Some(start) = css.find(selector) else {
            return "";
        };
        let after_selector = &css[start..];
        let Some(open_brace) = after_selector.find('{') else {
            return "";
        };
        let after_open = &after_selector[open_brace + 1..];
        let Some(close_brace) = after_open.find('}') else {
            return "";
        };
        &after_open[..close_brace]
    }

    #[test]
    fn scroll_ref_targets_scrollable_history_panel() {
        let css = include_str!("../styles.css");
        assert!(css_block(css, ".history-panel").contains("overflow: auto;"));
        assert!(!css_block(css, ".history-list").contains("overflow: auto;"));

        let history_source = include_str!("history.rs");
        let node_ref_line = history_source
            .lines()
            .find(|line| line.contains("node_ref=list_ref"))
            .expect("HistoryList should attach list_ref");

        assert!(
            node_ref_line.contains("class=\"history-panel\""),
            "list_ref must be attached to the CSS scroll container, got: {node_ref_line}"
        );
    }

    #[test]
    fn only_scrolls_when_history_ids_are_appended() {
        let previous = vec!["one".to_string(), "two".to_string()];
        assert!(has_appended_history_item(
            &previous,
            &["one".to_string(), "two".to_string(), "three".to_string()]
        ));
        assert!(!has_appended_history_item(
            &previous,
            &["one".to_string(), "two".to_string()]
        ));
        assert!(!has_appended_history_item(
            &previous,
            &["one".to_string(), "changed".to_string()]
        ));
        assert!(!has_appended_history_item(&previous, &["one".to_string()]));
    }

    #[test]
    fn scroll_trigger_tracks_item_identity_not_count() {
        let history_source = include_str!("history.rs");
        let id_state = ["previous", "_item", "_ids"].concat();
        let count_state = ["previous", "_count"].concat();

        assert!(
            history_source.contains(&id_state),
            "history scroll state should track item ids so progress/status changes do not look new"
        );
        assert!(
            !history_source.contains(&count_state),
            "history scroll state should not be based only on item count"
        );
    }
}
