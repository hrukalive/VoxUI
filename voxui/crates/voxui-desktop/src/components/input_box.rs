use crate::i18n::Language;
use leptos::prelude::*;
use web_sys::KeyboardEvent;

#[component]
pub fn InputBox(
    lang: ReadSignal<Language>,
    engine_ready: ReadSignal<bool>,
    status: ReadSignal<String>,
    on_submit: impl Fn(String) -> bool + 'static + Clone,
    on_cancel: impl Fn(()) + 'static + Clone,
) -> impl IntoView {
    let (text, set_text) = signal(String::new());
    let on_submit_clone = on_submit.clone();

    let handle_keydown = move |ev: KeyboardEvent| {
        if ev.key() == "Enter" && !ev.shift_key() {
            ev.prevent_default();
            let val = text.get();
            if !val.trim().is_empty() && on_submit_clone(val) {
                set_text.set(String::new());
            }
        }
    };

    let on_cancel_clone = on_cancel.clone();

    view! {
        <div class="shrink-0 p-3 bg-gray-800 border-t border-gray-700">
            <div class="flex gap-2">
                <textarea
                    class="flex-1 px-3 py-2 bg-gray-900 border border-gray-600 rounded text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 disabled:opacity-50 min-h-12 max-h-32 resize-y"
                    placeholder=move || lang.get().t("input_placeholder")
                    disabled=move || !engine_ready.get() || status.get() == "generating"
                    prop:value=move || text.get()
                    on:input=move |ev| {
                        set_text.set(event_target_value(&ev));
                    }
                    on:keydown=handle_keydown
                />
                <button
                    class=move || {
                        if status.get() == "generating" {
                            "px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm font-medium transition-colors"
                        } else {
                            "px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-sm font-medium transition-colors"
                        }
                    }
                    disabled=move || {
                        let s = status.get();
                        if s == "generating" {
                            false
                        } else {
                            !engine_ready.get() || text.get().trim().is_empty()
                        }
                    }
                    on:click=move |_| {
                        if status.get_untracked() == "generating" {
                            on_cancel_clone(());
                        } else {
                            let val = text.get();
                            if !val.trim().is_empty() && on_submit(val) {
                                set_text.set(String::new());
                            }
                        }
                    }
                >
                    {move || {
                        let l = lang.get();
                        if status.get() == "generating" { l.t("stop") } else { l.t("send") }
                    }}
                </button>
            </div>
        </div>
    }
}
