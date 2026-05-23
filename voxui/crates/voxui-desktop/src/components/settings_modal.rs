use leptos::prelude::*;

use crate::i18n::Labels;

#[component]
pub fn SettingsModal(
    labels: Labels,
    open: impl Fn() -> bool + Send + Sync + 'static + Copy,
    on_close: impl Fn() + Send + Sync + 'static + Copy,
) -> impl IntoView {
    view! {
        <Show when=open>
            <div class="modal-backdrop" role="presentation">
                <section class="modal" role="dialog" aria-modal="true" aria-label={labels.settings}>
                    <header class="modal-header">
                        <h2>{labels.settings}</h2>
                        <button class="icon-button" aria-label={labels.cancel} on:click=move |_| { on_close() }>
                            {"×"}
                        </button>
                    </header>
                    <div class="settings-grid">
                        <label>
                            <span>{labels.model}</span>
                            <input type="text" value="" placeholder={labels.model} disabled=true />
                        </label>
                        <label>
                            <span>{"Language"}</span>
                            <select>
                                <option>{"中文"}</option>
                                <option>{"English"}</option>
                            </select>
                        </label>
                        <label>
                            <span>{"Backend"}</span>
                            <select>
                                <option>{"CPU"}</option>
                                <option>{"CUDA"}</option>
                            </select>
                        </label>
                        <label>
                            <span>{"Volume"}</span>
                            <input type="range" min="0" max="100" value="80" />
                        </label>
                    </div>
                </section>
            </div>
        </Show>
    }
}
