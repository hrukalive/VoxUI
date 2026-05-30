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
                <section class="modal progress-modal" role="dialog" aria-modal="true" aria-label={labels.loading}>
                    <header class="modal-header">
                        <h2>{labels.loading}</h2>
                    </header>
                    <div class="progress-body">
                        <progress
                            class="progress-track"
                            role="progressbar"
                            max="100"
                            value=move || normalized_percent()
                            aria-valuemin="0"
                            aria-valuemax="100"
                            aria-valuenow=move || normalized_percent() as i32
                        >
                            {move || format!("{:.0}%", normalized_percent())}
                        </progress>
                        <p class="progress-label">{move || format!("{:.0}%", normalized_percent())}</p>
                    </div>
                    <footer class="modal-footer">
                        <button class="danger-button progress-cancel-button" type="button" aria-label={labels.cancel} on:click=move |_| { on_close() }>
                            {labels.cancel}
                        </button>
                    </footer>
                </section>
            </div>
        </Show>
    }
}
