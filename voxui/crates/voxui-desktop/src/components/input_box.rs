use leptos::prelude::*;
use crate::i18n::Language;
use web_sys::KeyboardEvent;

#[component]
pub fn InputBox(
    lang: ReadSignal<Language>,
    engine_ready: ReadSignal<bool>,
    on_submit: impl Fn(String) + 'static + Clone,
) -> impl IntoView {
    let (text, set_text) = signal(String::new());
    let on_submit_clone = on_submit.clone();

    let handle_keydown = move |ev: KeyboardEvent| {
        if ev.key() == "Enter" && !ev.shift_key() {
            ev.prevent_default();
            let val = text.get();
            if !val.trim().is_empty() {
                on_submit_clone(val);
                set_text.set(String::new());
            }
        }
    };

    view! {
        <div class="shrink-0 p-3 bg-gray-800 border-t border-gray-700">
            <div class="flex gap-2">
                <input
                    type="text"
                    class="flex-1 px-3 py-2 bg-gray-900 border border-gray-600 rounded text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 disabled:opacity-50"
                    placeholder=move || lang.get().t("input_placeholder")
                    disabled=move || !engine_ready.get()
                    prop:value=move || text.get()
                    on:input=move |ev| {
                        set_text.set(event_target_value(&ev));
                    }
                    on:keydown=handle_keydown
                />
                <button
                    class="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-sm font-medium transition-colors"
                    disabled=move || !engine_ready.get() || text.get().trim().is_empty()
                    on:click=move |_| {
                        let val = text.get();
                        if !val.trim().is_empty() {
                            on_submit(val);
                            set_text.set(String::new());
                        }
                    }
                >
                    {move || lang.get().t("send")}
                </button>
            </div>
        </div>
    }
}
