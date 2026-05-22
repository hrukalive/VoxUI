use crate::app::ModelChoice;
use crate::i18n::Language;
use leptos::prelude::*;

#[component]
pub fn Header(
    lang: ReadSignal<Language>,
    choices: ReadSignal<Vec<ModelChoice>>,
    selected_choice_id: ReadSignal<String>,
    loaded_choice_id: ReadSignal<Option<String>>,
    load_in_progress: ReadSignal<bool>,
    generating: Signal<bool>,
    on_choice_selected: impl Fn(String) + 'static + Clone,
    on_load_or_cancel: impl Fn(()) + 'static + Clone,
    on_settings: impl Fn(()) + 'static,
) -> impl IntoView {
    let can_load = move || {
        if load_in_progress.get() || generating.get() {
            return false;
        }
        let selected = selected_choice_id.get();
        !selected.is_empty() && Some(selected) != loaded_choice_id.get()
    };

    view! {
        <header class="flex items-center gap-3 px-4 py-2 bg-gray-800 border-b border-gray-700 shrink-0">
            <h1 class="text-xl font-bold text-blue-400 whitespace-nowrap">{move || lang.get().t("title")}</h1>
            <select
                class="min-w-0 flex-1 max-w-md bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm disabled:opacity-50"
                title=move || lang.get().t("select_model")
                disabled=move || load_in_progress.get() || generating.get() || choices.get().is_empty()
                on:change=move |ev| on_choice_selected(event_target_value(&ev))
            >
                <For
                    each=move || choices.get()
                    key=|choice| choice.id.clone()
                    children=move |choice| {
                        let selected = choice.id == selected_choice_id.get();
                        view! {
                            <option value={choice.id.clone()} selected=selected>{choice.name}</option>
                        }
                    }
                />
            </select>
            <button
                class=move || {
                    if load_in_progress.get() {
                        "px-3 py-1.5 rounded bg-red-600 hover:bg-red-700 text-sm font-medium"
                    } else {
                        "px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-sm font-medium"
                    }
                }
                disabled=move || !load_in_progress.get() && !can_load()
                on:click=move |_| on_load_or_cancel(())
            >
                {move || {
                    let l = lang.get();
                    if load_in_progress.get() { l.t("cancel") } else { l.t("load") }
                }}
            </button>
            <button
                class="p-2 rounded hover:bg-gray-700 transition-colors text-gray-300 hover:text-white disabled:opacity-50"
                title=move || lang.get().t("settings")
                disabled=move || load_in_progress.get() || generating.get()
                on:click=move |_| on_settings(())
            >
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd"/>
                </svg>
            </button>
        </header>
    }
}
