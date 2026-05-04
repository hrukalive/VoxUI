use leptos::prelude::*;
use crate::i18n::Language;

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
