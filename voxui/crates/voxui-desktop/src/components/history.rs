use std::cell::{Cell, RefCell};
use std::rc::Rc;

use leptos::html::Div;
use leptos::prelude::*;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

use crate::i18n::Labels;
use crate::tauri_api::{HistoryItem, HistoryStatus};

#[component]
pub fn HistoryList(
    labels: Labels,
    items: impl Fn() -> Vec<HistoryItem> + Send + Sync + 'static + Copy,
    on_play: impl Fn(String) + Send + Sync + 'static + Copy,
    on_regenerate: impl Fn(String) + Send + Sync + 'static + Copy,
    on_cancel: impl Fn(String) + Send + Sync + 'static + Copy,
) -> impl IntoView {
    let list_ref = NodeRef::<Div>::new();
    let (previous_count, set_previous_count) = signal(0_usize);

    Effect::new(move |_| {
        let item_count = items().len();
        if item_count == 0 {
            return;
        }
        if previous_count.get() != item_count {
            set_previous_count.set(item_count);
            if let Some(element) = list_ref.get() {
                scroll_to_bottom(element.into());
            }
        }
    });

    let initial = items();
    if initial.is_empty() {
        return view! {
            <section class="history-panel">
                <p class="empty-history">{labels.history_empty}</p>
            </section>
        }
        .into_any();
    }

    let items_for_view = items.clone();
    view! {
        <section class="history-panel">
            <div class="history-list" node_ref=list_ref>
                <For
                    each=move || items_for_view()
                    key=|item| item.id.clone()
                    children=move |item| {
                        let id_for_play = item.id.clone();
                        let id_for_regenerate = item.id.clone();
                        let id_for_cancel = item.id.clone();
                        let progress_label = progress_text(labels, &item);
                        let can_cancel = matches!(item.status, HistoryStatus::Queued | HistoryStatus::Generating);
                        let can_play = item.has_audio && matches!(item.status, HistoryStatus::Ready | HistoryStatus::Playing);
                        let play_label = if matches!(item.status, HistoryStatus::Playing) { labels.stop } else { labels.play };

                        view! {
                            <article class="history-item">
                                <div class="history-main">
                                    <p class="history-text">{item.text}</p>
                                    <div class="history-meta">
                                        <span>{status_label(labels, item.status)}</span>
                                        {progress_label
                                            .map(|label| view! { <span>{label}</span> })}
                                    </div>
                                    {item.error
                                        .map(|error| view! { <p class="history-error">{error}</p> })}
                                </div>
                                <div class="history-actions">
                                    <button
                                        class="history-action-button"
                                        title={labels.cancel}
                                        aria-label={labels.cancel}
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
                                        title={labels.regenerate}
                                        aria-label={labels.regenerate}
                                        on:click=move |_| on_regenerate(id_for_regenerate.clone())
                                    >
                                        {regenerate_icon()}
                                    </button>
                                </div>
                            </article>
                        }
                    }
                />
            </div>
        </section>
    }
    .into_any()
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

    let Some(window) = web_sys::window() else {
        element.set_scroll_top(target);
        return;
    };

    let element = Rc::new(element);
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

fn prefers_reduced_motion() -> bool {
    web_sys::window()
        .and_then(|window| window.match_media("(prefers-reduced-motion: reduce)").ok())
        .flatten()
        .is_some_and(|query| query.matches())
}
