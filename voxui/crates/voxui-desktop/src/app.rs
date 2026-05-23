use leptos::prelude::*;

mod i18n_bridge {
    pub use crate::i18n::{labels, UiLanguage};
}

#[component]
pub fn App() -> impl IntoView {
    let labels = i18n_bridge::labels(i18n_bridge::UiLanguage::Chinese);

    view! {
        <div class="app-shell">
            <header class="app-header">
                <div class="brand">
                    <strong>{labels.title}</strong>
                    <span>{labels.subtitle}</span>
                </div>
                <select class="model-select" aria-label={labels.model}>
                    <option>{labels.model}</option>
                </select>
                <button class="primary-button">{labels.load}</button>
                <button class="icon-button" title={labels.settings} aria-label={labels.settings}>{"⚙"}</button>
            </header>
            <section class="history-panel">
                <p class="empty-history">{labels.history_empty}</p>
            </section>
            <footer class="composer-panel">
                <textarea class="composer-input" placeholder={labels.input_placeholder}></textarea>
                <button class="generate-button">{labels.generate}</button>
            </footer>
        </div>
    }
}
