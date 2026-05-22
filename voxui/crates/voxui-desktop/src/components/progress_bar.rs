use crate::app::LoadProgress;
use crate::i18n::Language;
use leptos::prelude::*;

#[component]
pub fn ProgressBar(
    progress: ReadSignal<f64>,
    status: ReadSignal<String>,
    lang: ReadSignal<Language>,
) -> impl IntoView {
    view! {
        <div class="shrink-0 px-4 py-2 bg-gray-850" class:hidden=move || status.get() != "generating">
            <div class="flex items-center gap-3">
                <span class="text-xs text-gray-400 w-16">{move || lang.get().t("generating")}</span>
                <div class="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                        class="h-full bg-blue-500 rounded-full transition-all duration-200"
                        style=move || format!("width: {}%", (progress.get() * 100.0).min(100.0))
                    />
                </div>
                <span class="text-xs text-gray-400 w-10 text-right">
                    {move || format!("{:.0}%", (progress.get() * 100.0).min(100.0))}
                </span>
            </div>
        </div>
    }
}

#[component]
pub fn ModelLoadProgressBar(
    progress: ReadSignal<LoadProgress>,
    lang: ReadSignal<Language>,
) -> impl IntoView {
    let visible = move || progress.get() != LoadProgress::Hidden;
    let label = move || match progress.get() {
        LoadProgress::Hidden => String::new(),
        LoadProgress::Reading { label, .. } => format!("{} {}", lang.get().t("reading"), label),
        LoadProgress::DeviceLoading { backend, .. } => {
            format!("{} {}", lang.get().t("loading_to_device"), backend)
        }
    };
    let percent = move || match progress.get() {
        LoadProgress::Reading {
            bytes_read,
            total_bytes,
            ..
        } if total_bytes > 0 => (bytes_read as f64 / total_bytes as f64).clamp(0.0, 1.0),
        LoadProgress::DeviceLoading {
            step: Some(step),
            total: Some(total),
            ..
        } if total > 0 => (step as f64 / total as f64).clamp(0.0, 1.0),
        _ => 0.0,
    };

    view! {
        <div class="shrink-0 px-4 py-2 bg-gray-850" class:hidden=move || !visible()>
            <div class="flex items-center gap-3">
                <span class="text-xs text-gray-400 w-40 truncate">{label}</span>
                <div class="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                        class=move || match progress.get() {
                            LoadProgress::DeviceLoading { step: None, .. } => "h-full w-1/3 bg-blue-500 rounded-full animate-pulse",
                            _ => "h-full bg-blue-500 rounded-full transition-all duration-200",
                        }
                        style=move || match progress.get() {
                            LoadProgress::Reading { .. } => format!("width: {}%", percent() * 100.0),
                            LoadProgress::DeviceLoading { step: Some(_), .. } => format!("width: {}%", percent() * 100.0),
                            LoadProgress::DeviceLoading { step: None, .. } => "width: 33%".to_string(),
                            LoadProgress::Hidden => "width: 0%".to_string(),
                        }
                    />
                </div>
                <span class="text-xs text-gray-400 w-12 text-right">
                    {move || match progress.get() {
                        LoadProgress::Reading { total_bytes, .. } if total_bytes > 0 => {
                            format!("{:.0}%", percent() * 100.0)
                        }
                        LoadProgress::DeviceLoading { total: Some(total), .. } if total > 0 => {
                            format!("{:.0}%", percent() * 100.0)
                        }
                        LoadProgress::DeviceLoading { .. } => "...".to_string(),
                        _ => String::new(),
                    }}
                </span>
            </div>
        </div>
    }
}
