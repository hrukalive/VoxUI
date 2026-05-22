use crate::i18n::Language;
use leptos::prelude::*;

#[component]
pub fn StatusBar(
    lang: ReadSignal<Language>,
    status: ReadSignal<String>,
    selected_choice_name: Signal<String>,
    loaded_choice_name: Signal<String>,
    selected_matches_loaded: Signal<bool>,
    actual_backend: ReadSignal<String>,
    audio_host: ReadSignal<String>,
    audio_device: ReadSignal<String>,
    status_message: ReadSignal<String>,
) -> impl IntoView {
    let status_text = move || {
        let l = lang.get();
        let status = match status.get().as_str() {
            "loading" => l.t("loading").to_string(),
            "idle" => l.t("idle").to_string(),
            "ready" => l.t("ready").to_string(),
            "generating" => l.t("generating").to_string(),
            other => other.to_string(),
        };
        let message = status_message.get();
        if message.trim().is_empty() {
            status
        } else {
            format!("{} - {}", status, message)
        }
    };

    let right_text = move || {
        let selected = selected_choice_name.get();
        let loaded = loaded_choice_name.get();
        let selected_matches_loaded = selected_matches_loaded.get();
        let backend = actual_backend.get();
        let host = audio_host.get();
        let device = audio_device.get();
        let mut parts = Vec::new();
        if !loaded.is_empty() {
            parts.push(format!("{}: {}", lang.get().t("loaded"), loaded));
        }
        if !selected.is_empty() && !selected_matches_loaded {
            parts.push(format!("{}: {}", lang.get().t("selected"), selected));
        }
        if !backend.is_empty() {
            parts.push(backend);
        }
        if !host.is_empty() || !device.is_empty() {
            parts.push(if host.is_empty() {
                device
            } else if device.is_empty() {
                host
            } else {
                format!("{host}/{device}")
            });
        }
        parts.join(" | ")
    };

    view! {
        <footer class="shrink-0 flex items-center justify-between gap-4 px-4 py-1 bg-gray-800 border-t border-gray-700 text-xs text-gray-500">
            <span class="truncate">{status_text}</span>
            <span class="truncate text-right">{right_text}</span>
        </footer>
    }
}
