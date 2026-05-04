use leptos::prelude::*;
use crate::i18n::Language;

#[component]
pub fn ModelSelect(
    lang: ReadSignal<Language>,
    on_select: impl Fn(String) + 'static,
) -> impl IntoView {
    let (path, set_path) = signal(String::new());

    view! {
        <div class="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
            <div class="bg-gray-800 rounded-lg shadow-xl w-[400px] border border-gray-600 p-6">
                <h2 class="text-lg font-semibold mb-2">{move || lang.get().t("no_model")}</h2>
                <p class="text-sm text-gray-400 mb-4">{move || lang.get().t("no_model_msg")}</p>
                <input
                    type="text"
                    class="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm mb-4"
                    placeholder="models/VoxCPM-v1"
                    prop:value=move || path.get()
                    on:input=move |ev| set_path.set(event_target_value(&ev))
                />
                <div class="flex justify-end">
                    <button
                        class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium disabled:opacity-50"
                        disabled=move || path.get().trim().is_empty()
                        on:click=move |_| {
                            let p = path.get();
                            if !p.trim().is_empty() {
                                on_select(p);
                            }
                        }
                    >
                        {move || lang.get().t("apply")}
                    </button>
                </div>
            </div>
        </div>
    }
}
