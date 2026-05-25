use leptos::prelude::*;

use crate::i18n::Labels;
use crate::tauri_api::{HistoryItem, HistoryStatus};

#[component]
pub fn HistoryList(
    labels: Labels,
    items: Vec<HistoryItem>,
    on_play: impl Fn(String) + 'static + Copy,
    on_regenerate: impl Fn(String) + 'static + Copy,
    on_cancel: impl Fn(String) + 'static + Copy,
) -> impl IntoView {
    if items.is_empty() {
        return view! {
            <section class="history-panel">
                <p class="empty-history">{labels.history_empty}</p>
            </section>
        }
        .into_any();
    }

    view! {
        <section class="history-panel">
            <div class="history-list">
                {items
                    .into_iter()
                    .map(|item| {
                        let id_for_play = item.id.clone();
                        let id_for_regenerate = item.id.clone();
                        let id_for_cancel = item.id.clone();
                        let progress_label = progress_text(&item);
                        let can_cancel = matches!(item.status, HistoryStatus::Queued | HistoryStatus::Generating);
                        let can_play = item.has_audio && matches!(item.status, HistoryStatus::Ready | HistoryStatus::Playing);

                        view! {
                            <article class="history-item">
                                <div class="history-main">
                                    <p class="history-text">{item.text}</p>
                                    <div class="history-meta">
                                        <span>{status_label(item.status)}</span>
                                        {progress_label
                                            .map(|label| view! { <span>{label}</span> })}
                                    </div>
                                    {item.error
                                        .map(|error| view! { <p class="history-error">{error}</p> })}
                                </div>
                                <div class="history-actions">
                                    <button
                                        class="secondary-button"
                                        disabled={!can_cancel}
                                        on:click=move |_| on_cancel(id_for_cancel.clone())
                                    >
                                        {labels.cancel}
                                    </button>
                                    <button
                                        class="secondary-button"
                                        disabled={!can_play}
                                        on:click=move |_| on_play(id_for_play.clone())
                                    >
                                        {if matches!(item.status, HistoryStatus::Playing) { labels.stop } else { labels.play }}
                                    </button>
                                    <button
                                        class="secondary-button"
                                        on:click=move |_| on_regenerate(id_for_regenerate.clone())
                                    >
                                        {labels.regenerate}
                                    </button>
                                </div>
                            </article>
                        }
                    })
                    .collect_view()}
            </div>
        </section>
    }
    .into_any()
}

fn status_label(status: HistoryStatus) -> String {
    format!("{status:?}")
}

fn progress_text(item: &HistoryItem) -> Option<String> {
    (item.progress_total > 0).then(|| format!("{}/{}", item.progress_current, item.progress_total))
}
