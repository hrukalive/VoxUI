use leptos::prelude::*;
use crate::i18n::Language;

#[component]
pub fn StatusBar(
    lang: ReadSignal<Language>,
    status: ReadSignal<String>,
    model_name: ReadSignal<String>,
    actual_backend: ReadSignal<String>,
    lora_dir: ReadSignal<String>,
    audio_host: ReadSignal<String>,
    audio_device: ReadSignal<String>,
    status_message: ReadSignal<String>,
) -> impl IntoView {
    let status_text = move || {
        let l = lang.get();
        let status = match status.get().as_str() {
            "loading" => l.t("loading").to_string(),
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
        let model = model_name.get();
        if model.is_empty() {
            return String::new();
        }

        let backend = actual_backend.get();
        let host = audio_host.get();
        let device = audio_device.get();
        let lora = lora_dir.get();
        let lora_label = if lora.trim().is_empty() || lora == "None" {
            "LoRA: None".to_string()
        } else {
            let normalized = lora.replace('\\', "/");
            let basename = normalized.rsplit('/').next().unwrap_or(lora.as_str());
            format!("LoRA: {}", basename)
        };

        let audio = if host.is_empty() && device.is_empty() {
            String::new()
        } else if host.is_empty() {
            device
        } else if device.is_empty() {
            host
        } else {
            format!("{}/{}", host, device)
        };

        [model, backend, audio, lora_label]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(" | ")
    };

    view! {
        <footer class="shrink-0 flex items-center justify-between gap-4 px-4 py-1 bg-gray-800 border-t border-gray-700 text-xs text-gray-500">
            <span class="truncate">{status_text}</span>
            <span class="truncate text-right">{right_text}</span>
        </footer>
    }
}
