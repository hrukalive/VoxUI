use leptos::prelude::*;

use crate::i18n::Labels;

#[component]
pub fn LoadProgressModal(
    labels: Labels,
    open: impl Fn() -> bool + Send + Sync + 'static + Copy,
    percent: impl Fn() -> f32 + Send + Sync + 'static + Copy,
    on_close: impl Fn() + Send + Sync + 'static + Copy,
) -> impl IntoView {
    let normalized_percent = move || percent().clamp(0.0, 100.0);

    view! {
        <Show when=open>
            <div class="modal-backdrop" role="presentation">
                <section class="modal progress-modal" role="dialog" aria-modal="true" aria-label={labels.load}>
                    <header class="modal-header">
                        <h2>{labels.load}</h2>
                        <button class="icon-button" aria-label={labels.cancel} on:click=move |_| { on_close() }>
                            {"×"}
                        </button>
                    </header>
                    <div class="progress-track" aria-valuemin="0" aria-valuemax="100" aria-valuenow=move || normalized_percent() as i32>
                        <div
                            class="progress-bar"
                            style:width=move || format!("{}%", normalized_percent())
                        ></div>
                    </div>
                    <p class="progress-label">{move || format!("{:.0}%", normalized_percent())}</p>
                </section>
            </div>
        </Show>
    }
}
