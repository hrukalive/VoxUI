use leptos::prelude::*;
use crate::i18n::Language;

#[component]
pub fn StatusBar(
    lang: ReadSignal<Language>,
    status: ReadSignal<String>,
    model_name: ReadSignal<String>,
    backend: ReadSignal<String>,
) -> impl IntoView {
    let status_text = move || {
        let l = lang.get();
        match status.get().as_str() {
            "loading" => l.t("loading").to_string(),
            "ready" => l.t("ready").to_string(),
            "generating" => l.t("generating").to_string(),
            other => other.to_string(),
        }
    };

    view! {
        <footer class="shrink-0 flex items-center justify-between px-4 py-1 bg-gray-800 border-t border-gray-700 text-xs text-gray-500">
            <span>{status_text}</span>
            <span>{move || {
                let m = model_name.get();
                let b = backend.get();
                if m.is_empty() { String::new() } else { format!("{} | {}", m, b) }
            }}</span>
        </footer>
    }
}
