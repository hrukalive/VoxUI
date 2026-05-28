use leptos::prelude::*;

use crate::i18n::Labels;

#[component]
pub fn ErrorModal(
    labels: Labels,
    open: impl Fn() -> bool + Send + Sync + 'static + Copy,
    title: impl Fn() -> String + Send + Sync + 'static + Copy,
    message: impl Fn() -> String + Send + Sync + 'static + Copy,
    on_close: impl Fn() + Send + Sync + 'static + Copy,
) -> impl IntoView {
    view! {
        <Show when=open>
            <div class="modal-backdrop" role="presentation">
                <section
                    class="modal error-modal"
                    role="alertdialog"
                    aria-modal="true"
                    aria-label=move || title()
                >
                    <header class="modal-header">
                        <h2>{move || title()}</h2>
                        <button
                            class="primary-button"
                            type="button"
                            aria-label={labels.close}
                            on:click=move |_| { on_close() }
                        >
                            {labels.close}
                        </button>
                    </header>
                    <p class="error-dialog-message">{move || message()}</p>
                </section>
            </div>
        </Show>
    }
}
