use leptos::prelude::*;
use crate::i18n::Language;
use crate::app::TtsEntry;

#[component]
pub fn History(
    lang: ReadSignal<Language>,
    entries: ReadSignal<Vec<TtsEntry>>,
) -> impl IntoView {
    view! {
        <div class="flex-1 overflow-y-auto p-4 space-y-2">
            <h2 class="text-sm font-semibold text-gray-400 mb-2">{move || lang.get().t("history")}</h2>
            {move || {
                entries.get().into_iter().enumerate().map(|(_, entry)| {
                    let is_error = entry.status.starts_with("error:");
                    let status_color = if is_error {
                        "text-red-400"
                    } else {
                        match entry.status.as_str() {
                        "generating" => "text-yellow-400",
                        "playing" => "text-green-400",
                        "done" => "text-gray-500",
                        "error" => "text-red-400",
                        _ => "text-gray-400",
                        }
                    };
                    let status_icon = if is_error {
                        "!"
                    } else {
                        match entry.status.as_str() {
                        "queued" => "...",
                        "generating" => ">>",
                        "playing" => ">|",
                        "done" => "ok",
                        "error" => "!",
                        _ => "-",
                        }
                    };
                    let error_message = entry
                        .status
                        .strip_prefix("error:")
                        .map(|message| message.trim().to_string())
                        .unwrap_or_default();
                    let error_message_view = error_message.clone();
                    view! {
                        <div class="flex items-start gap-2 p-2 rounded bg-gray-800 border border-gray-700">
                            <span class={format!("mt-0.5 {}", status_color)}>{status_icon}</span>
                            <div class="flex-1 min-w-0">
                                <p class="text-sm truncate">{entry.text.clone()}</p>
                                <p class="text-xs text-gray-500">{entry.timestamp.clone()}</p>
                                <Show when=move || !error_message_view.is_empty()>
                                    <p class="text-xs text-red-300 whitespace-normal">{error_message.clone()}</p>
                                </Show>
                            </div>
                        </div>
                    }
                }).collect_view()
            }}
        </div>
    }
}
